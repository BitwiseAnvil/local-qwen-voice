"""Idle unload: parking policy, waking under load and host round trips; no ML imports."""

import contextlib
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from local_voice.cli import wake_and_wait
from local_voice.common import DEFAULT_VOICE
from local_voice.engine import QwenEngine, move_tensors, tensor_slots
from local_voice.http_server import dispatch
from test_streaming import PCM
from test_voice import PROJECT, sample_wav


class FakeTensor:
    def __init__(self, device, payload):
        self.device = SimpleNamespace(type=device)
        self.payload = payload
        self.storage = object()
        self.shape = (2,)
        self.dtype = "bf16"
        self.nbytes = 4

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: id(self.storage))

    def storage_offset(self):
        return 0

    def stride(self):
        return (1,)

    def to(self, device, non_blocking=False):
        return FakeTensor(device.split(":")[0], self.payload)


class FakeParameter:
    """Keeps its identity while .data moves, like torch.nn.Parameter."""

    def __init__(self, tensor):
        self.data = tensor

    def __getattr__(self, name):
        return getattr(self.data, name)


class FakeModule:
    def __init__(self, parameters=None, buffers=None, children=()):
        self._parameters = dict(parameters or {})
        self._buffers = dict(buffers or {})
        self.children = list(children)

    def modules(self):
        yield self
        for child in self.children:
            yield from child.modules()

    def named_parameters(self):
        return list(self._parameters.items())


class MoveTensorTests(unittest.TestCase):
    def test_parameters_keep_identity_and_tied_tensors_share_one_copy(self):
        tied = FakeParameter(FakeTensor("cuda", "embedding"))
        shared_buffer = FakeTensor("cuda", "inv_freq")
        one = FakeModule({"weight": tied}, {"inv_freq": shared_buffer})
        two = FakeModule({"weight": tied, "bias": FakeParameter(FakeTensor("cuda", "bias"))},
                         {"inv_freq": shared_buffer, "host_only": FakeTensor("cpu", "table")})
        root = FakeModule(children=[one, two])
        slots = tensor_slots([root], "cuda")
        copies = []

        def to_host(tensor):
            copies.append(tensor.payload)
            return FakeTensor("cpu", tensor.payload)

        self.assertEqual(3, move_tensors(slots, to_host))
        self.assertEqual(["embedding", "inv_freq", "bias"], copies)
        self.assertIs(tied, one._parameters["weight"])
        self.assertIs(tied, two._parameters["weight"])
        self.assertEqual("cpu", tied.device.type)
        self.assertIs(one._buffers["inv_freq"], two._buffers["inv_freq"])
        self.assertEqual("cpu", two._parameters["bias"].device.type)
        # Tensors that were not on the GPU are not part of a park.
        self.assertEqual("table", two._buffers["host_only"].payload)
        move_tensors(slots, lambda tensor: tensor.to("cuda:0"))
        self.assertEqual("cuda", tied.device.type)
        self.assertIs(one._buffers["inv_freq"], two._buffers["inv_freq"])
        self.assertEqual("inv_freq", one._buffers["inv_freq"].payload)


def fake_torch(cleared=None):
    return SimpleNamespace(
        bfloat16="bf16", inference_mode=contextlib.nullcontext,
        _C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: cleared is not None and cleared.append(True)),
        cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
                             max_memory_allocated=lambda: 1024**3, memory_reserved=lambda: 5 * 1024**3,
                             empty_cache=lambda: None))


class EngineCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        with patch.dict("os.environ", {"TTS_IDLE_UNLOAD_SECONDS": "60"}):
            self.engine = QwenEngine(PROJECT / "config" / "voice.json", directory.name)
        self.engine.store.save(DEFAULT_VOICE, sample_wav(), {"reference_text": "Saved voice transcript."})
        self.engine.state = "ready"
        self.engine.gpu = "NVIDIA GeForce RTX 5090"
        self.engine.model_kind = "base"
        self.engine.model_snapshot = "test-revision"
        self.engine.residency = "resident"
        self.engine.torch = fake_torch()
        self.engine._pcm = lambda waveform: (waveform, 0.25)
        self.generated = []

        def generate(**kwargs):
            self.generated.append(kwargs["text"])
            yield PCM, 24000, {"steps": 4}

        self.engine.model = SimpleNamespace(_warmed_up=True, generate_voice_clone_streaming=generate)
        verified = patch("local_voice.engine.verify_cuda_parameters", return_value=42)
        verified.start()
        self.addCleanup(verified.stop)

    def speak(self, text="Hello."):
        return list(self.engine.stream_synthesize(text, DEFAULT_VOICE, uuid.uuid4().hex))

    def make_idle(self):
        self.engine.last_activity = time.monotonic() - 61


class IdlePolicyTests(EngineCase):
    def setUp(self):
        super().setUp()
        self.parks = []
        self.wakes = []
        self.engine._park = self.fake_park
        self.engine._wake = self.fake_wake

    def fake_park(self, reason):
        self.parks.append(reason)
        self.engine.residency = "parked"
        self.engine.park_count += 1

    def fake_wake(self, reason):
        self.wakes.append(reason)
        self.engine.residency = "resident"
        self.engine.wake_count += 1
        self.engine.last_wake = {"reason": reason, "seconds": 1.1}

    def test_disabled_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            engine = QwenEngine(PROJECT / "config" / "voice.json", self.engine.store.directory)
        self.assertEqual(0, engine.idle_unload_seconds)
        with patch.dict("os.environ", {"TTS_IDLE_UNLOAD_SECONDS": "-1"}), self.assertRaises(ValueError):
            QwenEngine(PROJECT / "config" / "voice.json", self.engine.store.directory)

    def test_parks_only_after_the_idle_period_with_nothing_pending(self):
        self.assertFalse(self.engine.park_if_idle())
        self.make_idle()
        self.engine.pending = 1
        self.assertFalse(self.engine.park_if_idle())
        self.engine.pending = 0
        self.engine.keep_resident_until = time.monotonic() + 30
        self.assertFalse(self.engine.park_if_idle())
        self.engine.keep_resident_until = 0
        self.assertTrue(self.engine.park_if_idle())
        self.assertEqual(["idle"], self.parks)
        self.assertTrue(self.engine.health()["parked"])
        # Already parked: nothing more to release.
        self.assertFalse(self.engine.park_if_idle())

    def test_never_parks_while_a_stream_holds_the_gpu(self):
        stream = self.engine.stream_synthesize("Hello.", DEFAULT_VOICE, uuid.uuid4().hex)
        self.assertEqual("H", next(stream)[0])
        self.make_idle()
        self.assertFalse(self.engine.park_if_idle())
        list(stream)
        # Finishing the request restarts the idle timer.
        self.assertFalse(self.engine.park_if_idle())
        self.make_idle()
        self.assertTrue(self.engine.park_if_idle())

    def test_request_on_a_parked_model_wakes_it_and_reports_the_wake(self):
        self.engine.residency = "parked"
        metrics = self.speak()[-1][1]
        self.assertEqual(["request"], self.wakes)
        self.assertTrue(metrics["woke_model"])
        self.assertEqual(1.1, metrics["wake_seconds"])
        self.assertFalse(self.speak()[-1][1]["woke_model"])
        self.assertEqual(1, len(self.wakes))

    def test_wake_request_returns_at_once_and_queued_requests_wait_then_succeed(self):
        self.engine.residency = "parked"
        entered, release = threading.Event(), threading.Event()

        def slow_wake(reason):
            self.engine.residency = "waking"
            entered.set()
            release.wait(5)
            self.fake_wake(reason)

        self.engine._wake = slow_wake
        warmed = []
        self.engine._warm_voice = lambda name, force=False, first_chunk_only=False: warmed.append((name, force))
        started = time.monotonic()
        status, body, _ = dispatch(self.engine, "POST", "/wake", {})
        self.assertEqual(202, status)
        self.assertTrue(body["wake_started"])
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertTrue(entered.wait(2))
        self.assertEqual("waking", self.engine.health()["residency"])
        # A second hint while waking does not start another wake.
        self.assertFalse(self.engine.request_wake()["wake_started"])
        results, errors = [], []

        def request(index):
            try:
                results.append(self.speak(f"Request {index}.")[-1][1])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=request, args=(index,)) for index in range(3)]
        for thread in threads:
            thread.start()
        time.sleep(0.2)
        self.assertEqual([], results)
        # The three requests and the background wake all hold off idle unload.
        self.assertEqual(4, self.engine.pending)
        release.set()
        for thread in threads:
            thread.join(5)
        self.engine.wake_thread.join(5)
        self.assertEqual([], errors)
        self.assertEqual(3, len(results))
        self.assertEqual(["wake request"], self.wakes)
        # Requests were already waiting, so the wake skipped its warm-up generation.
        self.assertEqual([], warmed)
        self.assertEqual("resident", self.engine.health()["residency"])
        self.assertEqual(0, self.engine.pending)

    def test_wake_hint_warms_up_when_nothing_is_waiting(self):
        self.engine.residency = "parked"
        warmed = []
        self.engine._warm_voice = lambda name, **kwargs: warmed.append((name, kwargs))
        self.assertTrue(self.engine.request_wake()["wake_started"])
        self.engine.wake_thread.join(5)
        self.assertEqual([(self.engine.config["name"], {"force": True, "first_chunk_only": True})], warmed)
        self.assertIn("warm_seconds", self.engine.health()["last_wake"])
        self.assertEqual(0, self.engine.pending)

    def test_failed_wake_is_reported_and_the_next_request_retries(self):
        self.engine.residency = "parked"
        attempts = []

        def flaky_wake(reason):
            attempts.append(reason)
            if len(attempts) == 1:
                raise RuntimeError("CUDA out of memory")
            self.fake_wake(reason)

        self.engine._wake = flaky_wake
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.speak()
        self.assertFalse(self.engine.gpu_lock.locked())
        self.assertEqual(0, self.engine.pending)
        self.assertTrue(self.speak()[-1][1]["woke_model"])
        self.assertEqual(2, len(attempts))

    def test_wake_endpoint_validates_and_can_hold_the_model_resident(self):
        for value in (-1, 3601, "10", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                dispatch(self.engine, "POST", "/wake", {"keep_resident_seconds": value})
        status, body, _ = dispatch(self.engine, "POST", "/wake", {"keep_resident_seconds": 600})
        self.assertEqual(202, status)
        self.assertFalse(body["wake_started"])
        self.assertGreater(body["keep_resident_seconds"], 590)
        self.make_idle()
        self.assertFalse(self.engine.park_if_idle())

    def test_wake_is_not_attempted_before_startup_finishes(self):
        self.engine.state = "loading"
        self.engine.residency = None
        self.assertFalse(self.engine.request_wake()["wake_started"])
        self.assertIsNone(self.engine.wake_thread)

    def test_health_reports_idle_fields(self):
        health = self.engine.health()
        for key in ("residency", "parked", "idle_unload_seconds", "idle_seconds", "active_requests", "keep_resident_seconds",
                    "torch_reserved_mb", "wake_count", "park_count", "last_wake_seconds", "last_wake", "last_park"):
            self.assertIn(key, health)
        self.assertEqual(5120.0, health["torch_reserved_mb"])
        self.assertEqual(60.0, health["idle_unload_seconds"])
        self.assertEqual("ready", health["status"])

    def test_voice_design_drops_a_parked_model_instead_of_waking_it(self):
        self.engine.residency = "parked"
        loaded = []

        class DesignModel:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                loaded.append(path)
                return cls()

        self.engine.torch = SimpleNamespace(bfloat16="bf16", cuda=fake_torch().cuda)
        with patch.dict(sys.modules, {
            "huggingface_hub": SimpleNamespace(snapshot_download=lambda **kwargs: "/models/design"),
            "qwen_tts": SimpleNamespace(Qwen3TTSModel=DesignModel),
        }):
            self.engine._load_model("design")
        self.assertEqual([], self.wakes)
        self.assertEqual(["/models/design"], loaded)
        self.assertEqual("design", self.engine.model_kind)
        self.assertEqual("resident", self.engine.residency)
        self.assertIsNone(self.engine.parked_slots)
        # The idle monitor only parks the Base model.
        self.make_idle()
        self.assertFalse(self.engine.park_if_idle())


class CliWakeTests(unittest.TestCase):
    def test_benchmark_wakes_a_parked_model_and_waits_for_its_warmup(self):
        states = iter([
            {"status": "ready", "residency": "parked"},
            {"status": "ready", "residency": "waking", "active_requests": 1},
            {"status": "ready", "residency": "resident", "active_requests": 1},
            {"status": "ready", "residency": "resident", "active_requests": 0},
        ])
        woke = []
        client = SimpleNamespace(health=lambda: next(states), wake=lambda: woke.append(True))
        with patch("local_voice.cli.time.sleep"):
            self.assertEqual(0, wake_and_wait(client)["active_requests"])
        self.assertEqual([True], woke)

    def test_resident_or_older_engines_are_not_woken(self):
        for state in ({"status": "ready", "residency": "resident"}, {"status": "ready"}):
            client = SimpleNamespace(health=lambda: state, wake=lambda: self.fail("woke a resident model"))
            self.assertIs(state, wake_and_wait(client))


class ParkRoundTripTests(EngineCase):
    """The real _park/_wake bookkeeping, with fake tensors and CUDA graphs."""

    def setUp(self):
        super().setUp()
        self.tied = FakeParameter(FakeTensor("cuda", "codec embedding"))
        self.talker = FakeModule({"weight": FakeParameter(FakeTensor("cuda", "talker"))},
                                 {"inv_freq": FakeTensor("cuda", "rope")})
        self.talker.code_predictor = SimpleNamespace(model=SimpleNamespace(config="predictor config"))
        self.talker.model = "talker backbone"
        self.tokenizer = FakeModule({"weight": self.tied, "decoder": FakeParameter(FakeTensor("cuda", "decoder"))})
        qwen = FakeModule({"embedding": self.tied}, children=[self.talker])
        qwen.talker = self.talker
        qwen.config = SimpleNamespace(talker_config=SimpleNamespace(hidden_size=2048))
        qwen.speech_tokenizer = SimpleNamespace(model=self.tokenizer)
        self.warmups = []
        engine = self.engine

        class Wrapper:
            device = "cuda:0"
            dtype = "bf16"
            model = SimpleNamespace(model=qwen)
            _warmed_up = True
            predictor_graph = SimpleNamespace(do_sample=True, top_k=50, top_p=1.0, temperature=0.9)
            talker_graph = SimpleNamespace(max_seq_len=2048)

            def warmup(wrapper, prefill_len):
                engine.warmups.append(prefill_len)
                if engine.fail_warmup:
                    raise RuntimeError("CUDA out of memory")
                wrapper._warmed_up = True

        self.cleared = []
        self.engine.torch = fake_torch(self.cleared)
        self.engine.warmups = self.warmups
        self.engine.fail_warmup = False
        self.engine.model = Wrapper()
        self.engine._reserve_host = lambda nbytes: None
        self.engine.host_buffer = SimpleNamespace(numel=lambda: 4 * 1024**2)
        self.engine._to_host = lambda tensor: FakeTensor("cpu", tensor.payload)
        graphs = {
            "faster_qwen3_tts": SimpleNamespace(),
            "faster_qwen3_tts.predictor_graph": SimpleNamespace(PredictorGraph=lambda *args, **kwargs: ("predictor", args, kwargs)),
            "faster_qwen3_tts.talker_graph": SimpleNamespace(TalkerGraph=lambda *args, **kwargs: ("talker", args, kwargs)),
        }
        modules = patch.dict(sys.modules, graphs)
        modules.start()
        self.addCleanup(modules.stop)

    def devices(self):
        return {self.tied.device.type, self.talker._parameters["weight"].device.type,
                self.talker._buffers["inv_freq"].device.type, self.tokenizer._parameters["decoder"].device.type}

    def test_park_and_failed_wake_return_to_host_then_a_retry_restores_everything(self):
        model = self.engine.model
        self.engine._park("idle")
        self.assertEqual({"cpu"}, self.devices())
        self.assertIsNone(model.predictor_graph)
        self.assertFalse(model._warmed_up)
        self.assertEqual("parked", self.engine.residency)
        self.assertEqual(4, self.engine.last_park["tensors"])
        # Per-stream cuBLAS workspaces from the old captures are released.
        self.assertEqual([True], self.cleared)

        self.engine.fail_warmup = True
        with self.assertRaisesRegex(RuntimeError, "out of memory"), self.assertLogs("local_voice.engine", "ERROR"):
            self.engine._wake("request")
        self.assertEqual({"cpu"}, self.devices())
        self.assertEqual("parked", self.engine.residency)
        self.assertEqual("ready", self.engine.state)
        self.assertIsNone(model.talker_graph)

        self.engine.fail_warmup = False
        self.engine._wake("request")
        self.assertEqual({"cuda"}, self.devices())
        self.assertIs(self.tied, self.tokenizer._parameters["weight"])
        self.assertEqual("codec embedding", self.tied.payload)
        self.assertEqual("resident", self.engine.residency)
        self.assertEqual([100, 100], self.warmups)
        self.assertEqual("predictor", model.predictor_graph[0])
        self.assertEqual({"do_sample": True, "top_k": 50, "top_p": 1.0, "temperature": 0.9,
                          "device": "cuda:0", "dtype": "bf16"}, model.predictor_graph[2])
        self.assertEqual(2048, model.talker_graph[2]["max_seq_len"])
        self.assertEqual(1, self.engine.wake_count)
        self.assertIsNone(self.engine.parked_slots)


if __name__ == "__main__":
    unittest.main()
