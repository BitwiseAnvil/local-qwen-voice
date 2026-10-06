import contextlib
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import wave

from local_voice.cli import verify_sample, wait_ready
from local_voice.client import EngineClient
from local_voice.common import Cancelled, DEFAULT_VOICE, NotReady, sentence_chunks, voice_name, wav_info
from local_voice.controller import SpeechController
from local_voice.engine import CancellationRegistry, QwenEngine, cancellable_generation, cancellable_lock, verify_cuda_parameters
from local_voice.http_server import dispatch
from local_voice.mcp_server import RpcServer
from local_voice.settings import backend_url
from local_voice.voices import VoiceStore
from local_voice.streaming import FORMAT

PROJECT = Path(__file__).resolve().parents[1]


def sample_wav(silent=False):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        frames = [0 if silent else int(8000 * math.sin(index * 0.1)) for index in range(240)]
        stream.writeframes(struct.pack("<" + "h" * len(frames), *frames))
    return output.getvalue()


def metrics():
    return {"gpu": "NVIDIA GeForce RTX 5090", "device": "cuda:0", "cuda_parameters_verified": 12, "peak_cuda_memory_mb": 1024}


class FakeTalker:
    def register_forward_pre_hook(self, callback):
        self.callback = callback
        return SimpleNamespace(remove=lambda: setattr(self, "callback", None))


class FakeClient:
    def __init__(self):
        self.cancelled = []
        self.texts = []

    def health(self):
        return {"status": "ready", "gpu": "RTX 5090", "device": "cuda:0", "streaming_protocol": 1}

    def synthesize(self, text, voice, identifier):
        self.texts.append((text, voice))
        return sample_wav(), metrics()

    def cancel(self, identifier):
        self.cancelled.append(identifier)

    def close_stream(self, identifier):
        pass

    def stream_synthesize(self, text, voice, identifier, cancel):
        yield "H", dict(FORMAT)
        audio, result = self.synthesize(text, voice, identifier)
        with wave.open(io.BytesIO(audio), "rb") as stream:
            pcm = stream.readframes(stream.getnframes())
        yield "A", pcm
        yield "M", dict(result, audio_bytes=len(pcm), audio_seconds=len(pcm)/48000)

    def design(self, name, description, identifier):
        return {"name": name, "description": description}

    def voices(self):
        return {"voices": [{"name": DEFAULT_VOICE}]}


class FakePlayer:
    def __init__(self, duration=0.001):
        self.started = []
        self.stopped = 0
        self.duration = duration
        self.started_event = threading.Event()
        self.underruns = 0

    def open_stream(self, sample_rate):
        pass

    def write(self, pcm):
        self.start(pcm)

    def wait_capacity(self, cancel):
        if cancel.is_set():
            raise Cancelled()

    def finish(self, cancel):
        if cancel.wait(self.duration):
            raise Cancelled()

    def start(self, audio):
        self.started.append(audio)
        self.started_event.set()
        return self.duration

    def stop(self):
        self.stopped += 1


def await_job(controller, identifier, states=("completed", "failed", "cancelled")):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = controller.status(identifier)
        if state["status"] in states:
            return state
        time.sleep(0.005)
    raise AssertionError(f"Job did not reach {states}: {controller.status(identifier)}")


class TextAudioTests(unittest.TestCase):
    def test_splitting_preserves_every_character_except_whitespace(self):
        text = "A clear explanation. " + "longword" * 150 + "\n" + "Another sentence! " * 40
        chunks = sentence_chunks(text)
        self.assertTrue(all(0 < len(chunk) <= 280 for chunk in chunks))
        self.assertEqual("".join(text.split()), "".join("".join(chunks).split()))

    def test_rejects_voice_path_traversal(self):
        for name in ("../secret", "a/b", "C:\\temp", "", ".hidden", "CON", "x" * 49):
            with self.subTest(name=name), self.assertRaises(ValueError):
                voice_name(name)

    def test_rejects_truncated_audio(self):
        with self.assertRaises(ValueError):
            wav_info(sample_wav()[:-8])

    def test_real_smoke_rejects_silence_and_cpu_evidence(self):
        self.assertGreater(verify_sample(sample_wav(), metrics())["rms"], 0)
        with self.assertRaises(RuntimeError):
            verify_sample(sample_wav(True), metrics())
        with self.assertRaises(RuntimeError):
            verify_sample(sample_wav(), dict(metrics(), device="cpu"))
        self.assertGreater(verify_sample(sample_wav(), dict(metrics(), gpu="NVIDIA GeForce RTX 4070"))["rms"], 0)
        with self.assertRaises(RuntimeError):
            verify_sample(sample_wav(), dict(metrics(), cuda_parameters_verified=0))


class VoiceStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = VoiceStore(self.directory.name)

    def test_voice_survives_a_new_store_and_cannot_be_overwritten(self):
        saved = self.store.save("warm", sample_wav(), {"reference_text": "Hello.", "description": "Warm voice"})
        restored, audio = VoiceStore(self.directory.name).load("warm")
        self.assertEqual(saved, restored)
        self.assertEqual(sample_wav(), audio.read_bytes())
        with self.assertRaises(ValueError):
            self.store.save("warm", sample_wav(), {"reference_text": "Changed."})
        self.assertEqual("Hello.", self.store.load("warm")[0]["reference_text"])

    def test_tampered_audio_path_is_rejected(self):
        metadata = {"name": "warm", "reference_text": "Hello.", "audio_file": "../other.wav"}
        (Path(self.directory.name) / "warm.json").write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            self.store.load("warm")

    def test_failed_commit_does_not_leave_a_partial_voice(self):
        with patch("local_voice.voices.os.replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.save("warm", sample_wav(), {"reference_text": "Hello."})
        self.assertEqual([], list(Path(self.directory.name).iterdir()))


class CancellationTests(unittest.TestCase):
    def test_decoder_hook_stops_inference_and_is_removed(self):
        talker = FakeTalker()
        wrapper = SimpleNamespace(model=SimpleNamespace(talker=talker))
        event = threading.Event()
        with self.assertRaises(Cancelled):
            with cancellable_generation(wrapper, event):
                talker.callback(talker, ())
                event.set()
                talker.callback(talker, ())
        self.assertIsNone(talker.callback)

    def test_cancel_before_request_arrives_is_preserved(self):
        registry = CancellationRegistry()
        identifier = uuid.uuid4().hex
        self.assertFalse(registry.cancel(identifier))
        with registry.track(identifier) as event:
            self.assertTrue(event.is_set())
        self.assertEqual({}, registry.active)

    def test_active_cancel_and_duplicate_id(self):
        registry = CancellationRegistry()
        identifier = uuid.uuid4().hex
        with registry.track(identifier) as event:
            with self.assertRaises(ValueError):
                with registry.track(identifier):
                    pass
            self.assertTrue(registry.cancel(identifier))
            self.assertTrue(event.is_set())

    def test_cancelled_request_does_not_wait_for_gpu(self):
        lock = threading.Lock()
        lock.acquire()
        event = threading.Event()
        event.set()
        try:
            with self.assertRaises(Cancelled):
                with cancellable_lock(lock, event):
                    self.fail("Cancelled request entered GPU work")
        finally:
            lock.release()


class ControllerTests(unittest.TestCase):
    def make_controller(self, client=None, player=None):
        controller = SpeechController(client or FakeClient(), player or FakePlayer())
        self.addCleanup(controller.close)
        return controller

    def test_speech_finishes_after_audio_and_keeps_order(self):
        client, player = FakeClient(), FakePlayer()
        controller = self.make_controller(client, player)
        one = controller.speak("First sentence.")
        two = controller.speak("Second sentence.")
        self.assertEqual("completed", await_job(controller, two["id"])["status"])
        self.assertEqual("completed", controller.status(one["id"])["status"])
        self.assertEqual(["First sentence.", "Second sentence."], [item[0] for item in client.texts])
        self.assertEqual(2, len(player.started))

    def test_stop_during_generation_prevents_late_playback(self):
        class BlockingClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.entered = threading.Event()
                self.release = threading.Event()

            def synthesize(self, *args):
                self.entered.set()
                self.release.wait(2)
                return super().synthesize(*args)

            def cancel(self, identifier):
                super().cancel(identifier)
                self.release.set()

        client, player = BlockingClient(), FakePlayer()
        controller = self.make_controller(client, player)
        job = controller.speak("This should never reach the speakers.")
        self.assertTrue(client.entered.wait(1))
        queued = controller.speak("Neither should this.")
        controller.stop()
        self.assertEqual("cancelled", await_job(controller, job["id"])["status"])
        self.assertEqual("cancelled", controller.status(queued["id"])["status"])
        time.sleep(0.05)
        self.assertEqual([], player.started)
        self.assertEqual(1, len(client.cancelled))
        # Queue capacity is released, and a new requested utterance can play.
        new = controller.speak("New speech.")
        self.assertEqual("completed", await_job(controller, new["id"])["status"])

    def test_stop_during_playback_wakes_the_worker(self):
        player = FakePlayer(duration=60)
        controller = self.make_controller(player=player)
        job = controller.speak("A long recording.")
        self.assertTrue(player.started_event.wait(1))
        started = time.monotonic()
        controller.stop()
        await_job(controller, job["id"])
        self.assertLess(time.monotonic() - started, 1)
        self.assertGreater(player.stopped, 0)

    def test_generation_failure_is_visible(self):
        client = FakeClient()
        client.synthesize = lambda *args: (_ for _ in ()).throw(RuntimeError("GPU out of memory"))
        controller = self.make_controller(client)
        job = controller.speak("Hello.")
        state = await_job(controller, job["id"])
        self.assertEqual("failed", state["status"])
        self.assertIn("out of memory", state["error"])

    def test_voice_design_job_is_not_played(self):
        player = FakePlayer()
        controller = self.make_controller(player=player)
        job = controller.design("new_voice", "A warm American female voice.")
        result = await_job(controller, job["id"])
        self.assertEqual("new_voice", result["voice"]["name"])
        self.assertEqual([], player.started)

    def test_unready_engine_is_not_reported_as_queued(self):
        client = FakeClient()
        client.health = lambda: {"status": "loading"}
        controller = self.make_controller(client)
        with self.assertRaises(RuntimeError):
            controller.speak("Hello.")
        self.assertEqual([], controller.status()["jobs"])


class EngineLogicTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.engine = QwenEngine(PROJECT / "config" / "voice.json", self.directory.name)

    def test_cpu_is_rejected_at_startup(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"torch": fake_torch}), self.assertLogs("local_voice.engine", level="ERROR"):
            self.engine.initialize()
        self.assertEqual("failed", self.engine.health()["status"])
        self.assertIn("CUDA is unavailable", self.engine.health()["error"])

    def test_model_and_tokenizer_parameters_are_checked(self):
        class Model:
            def __init__(self, kind):
                self.kind = kind

            def named_parameters(self):
                return [("weight", SimpleNamespace(device=SimpleNamespace(type=self.kind)))]

        wrapper = SimpleNamespace(model=Model("cuda"))
        self.assertEqual(1, verify_cuda_parameters(wrapper))
        wrapper.model.speech_tokenizer = SimpleNamespace(model=Model("cpu"))
        with self.assertRaises(RuntimeError):
            verify_cuda_parameters(wrapper)

    def test_ready_gate_does_not_attempt_inference(self):
        with self.assertRaises(NotReady):
            self.engine.synthesize("Hello", DEFAULT_VOICE, uuid.uuid4().hex)

    def test_saved_reference_is_passed_to_base_generation(self):
        engine = self.engine
        engine.store.save(DEFAULT_VOICE, sample_wav(), {"reference_text": "Original reference transcript."})
        called = {}

        class Model:
            _warmed_up = True

            def generate_voice_clone_streaming(self, **kwargs):
                called["generation"] = kwargs
                yield [0.1, -0.1], 24000, {"steps": 4}

        engine.model = Model()
        engine.model_kind = "base"
        engine.model_snapshot = "test-snapshot"
        engine.gpu = "NVIDIA GeForce RTX 5090"
        engine.state = "ready"
        engine.torch = SimpleNamespace(
            inference_mode=contextlib.nullcontext,
            cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None, max_memory_allocated=lambda: 1024**3),
        )
        with wave.open(io.BytesIO(sample_wav()), "rb") as stream:
            pcm = stream.readframes(stream.getnframes())
        engine._pcm = lambda waveform: (pcm, 0.25)
        with patch("local_voice.engine.verify_cuda_parameters", return_value=42):
            audio, result = engine.synthesize("New speech.", DEFAULT_VOICE, uuid.uuid4().hex)
        self.assertEqual(sample_wav(), audio)
        self.assertEqual("Original reference transcript.", called["generation"]["ref_text"])
        self.assertFalse(called["generation"]["xvec_only"])
        self.assertTrue(Path(called["generation"]["ref_audio"]).is_file())
        self.assertEqual("New speech.", called["generation"]["text"])
        self.assertEqual(42, result["cuda_parameters_verified"])


class RpcTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.controller = SpeechController(self.client, FakePlayer())
        self.addCleanup(self.controller.close)
        self.server = RpcServer(self.controller, self.client)

    def initialize(self):
        result = self.server.receive({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"},
        }})
        self.assertEqual("2025-06-18", result["result"]["protocolVersion"])
        instructions = result["result"]["instructions"]
        self.assertIn(f'"{self.controller.default_voice}"', instructions)
        self.assertLess(len(instructions.encode()), 2048)
        self.assertIsNone(self.server.receive({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def call(self, name, arguments):
        return self.server.receive({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": name, "arguments": arguments}})

    def test_handshake_tools_and_speech_end_to_end_with_fake_audio(self):
        self.initialize()
        listed = self.server.receive({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertEqual({"speak", "stop_speaking", "speech_status", "list_voices", "design_voice"}, {item["name"] for item in listed["result"]["tools"]})
        response = self.call("speak", {"text": "Test the complete tool path."})
        self.assertFalse(response["result"]["isError"])
        job = json.loads(response["result"]["content"][0]["text"])
        self.assertEqual("completed", await_job(self.controller, job["id"])["status"])

    def test_invalid_arguments_and_unknown_method_are_protocol_errors(self):
        self.initialize()
        for arguments in ({}, {"text": None}, {"text": "Hello", "unknown": "value"}):
            self.assertEqual(-32602, self.call("speak", arguments)["error"]["code"])
        response = self.server.receive({"jsonrpc": "2.0", "id": 8, "method": "unknown"})
        self.assertEqual(-32601, response["error"]["code"])

    def test_failed_tool_is_visible_as_mcp_error(self):
        self.initialize()
        response = self.call("speak", {"text": " "})
        self.assertTrue(response["result"]["isError"])

    def test_tools_before_initialized_notification_are_rejected(self):
        response = self.server.receive({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertIn("error", response)

    def test_real_stdio_process_from_another_directory(self):
        messages = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"},
            }},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "ping"},
        ]
        result = subprocess.run(
            [sys.executable, str(PROJECT / "mcp_server.py")],
            input="\n".join(json.dumps(item) for item in messages) + "\n",
            text=True, capture_output=True, cwd=tempfile.gettempdir(), timeout=10,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        output = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([1, 2, 3], [item["id"] for item in output])
        self.assertEqual(5, len(output[1]["result"]["tools"]))


class TransportConfigurationTests(unittest.TestCase):
    def test_remote_backend_and_credential_urls_are_rejected(self):
        for url in ("https://example.com", "http://127.0.0.1.evil", "http://user:secret@localhost", "http://localhost/path"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                EngineClient(url)

    def test_bridge_reads_compose_port_from_env_file(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            (Path(directory) / ".env").write_text('TTS_PORT="8877" # local port\n')
            self.assertEqual("http://127.0.0.1:8877", backend_url(directory))

    def test_http_health_and_not_ready_have_different_status_codes(self):
        fake = SimpleNamespace(health=lambda: {"status": "failed", "error": "CUDA missing"})
        self.assertEqual(200, dispatch(fake, "GET", "/health")[0])
        self.assertEqual(503, dispatch(fake, "GET", "/readyz")[0])

    def test_wait_reports_startup_failure_immediately(self):
        fake = SimpleNamespace(health=lambda: {"status": "failed", "error": "Wrong GPU"})
        with self.assertRaisesRegex(RuntimeError, "Wrong GPU"):
            wait_ready(fake, 30)


if __name__ == "__main__":
    unittest.main()
