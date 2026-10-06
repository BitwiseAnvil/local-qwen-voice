"""Streaming, cancellation and native-buffer lifecycle regressions; no ML imports."""

import contextlib
import ctypes as ct
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from local_voice.cli import stream_sample, verify_interactive
from local_voice.client import EngineClient
from local_voice.common import Cancelled, DEFAULT_VOICE
from local_voice.controller import SpeechController
from local_voice.engine import QwenEngine
from local_voice.http_server import Handler
from local_voice.playback import WaveFormat, WaveHeader, WindowsPlayer
from local_voice.streaming import FORMAT, FRAME, MAX_FRAME, encode_frame, read_frames
from test_voice import FakeClient, FakePlayer, PROJECT, await_job, sample_wav

PCM = struct.pack("<240h", *([8000, -8000] * 120))


class FrameTests(unittest.TestCase):
    def test_fragmented_transport_preserves_every_sample(self):
        class Fragmented(io.BytesIO):
            def read(self, size=-1):
                return super().read(min(size, 3))

        data = (encode_frame("H", FORMAT) + encode_frame("A", PCM) + encode_frame("A", PCM)
                + encode_frame("M", {"audio_bytes": len(PCM) * 2}))
        frames = list(read_frames(Fragmented(data)))
        self.assertEqual(["H", "A", "A", "M"], [kind for kind, _ in frames])
        self.assertEqual(PCM * 2, b"".join(value for kind, value in frames if kind == "A"))

    def test_truncation_and_false_completion_are_rejected(self):
        header = encode_frame("H", FORMAT)
        audio = encode_frame("A", PCM)
        bad = [b"", header[:-1], header + audio[:-1], header + audio,
               header + encode_frame("M", {"audio_bytes": 0}),
               header + audio + encode_frame("M", {"audio_bytes": 2}),
               audio, header + header, FRAME.pack(b"A", MAX_FRAME + 1)]
        for data in bad:
            with self.subTest(data=data[:30]), self.assertRaises(RuntimeError):
                list(read_frames(io.BytesIO(data)))

    def test_model_failure_after_partial_audio_is_not_success(self):
        stream = read_frames(io.BytesIO(encode_frame("H", FORMAT) + encode_frame("A", PCM)
                            + encode_frame("E", {"error": "GPU failure"})))
        self.assertEqual("H", next(stream)[0])
        self.assertEqual(PCM, next(stream)[1])
        with self.assertRaisesRegex(RuntimeError, "GPU failure"):
            next(stream)


class FakeWinMM:
    """Driver surrogate that owns each prepared header until unprepare."""
    def __init__(self):
        self.calls = []
        self.owned = []
        self.written = []
        self.fail_write = False
        self.fail_unprepare = False

    def call(self, name, *args):
        self.calls.append(name)
        if name == "waveOutOpen":
            args[0]._obj.value = 123
            self.format = args[2]._obj
        elif name == "waveOutPrepareHeader":
            header = args[1]._obj
            header.flags = 2
            self.owned.append(header)
        elif name == "waveOutWrite":
            if self.fail_write:
                raise RuntimeError("Audio driver write failed")
            header = args[1]._obj
            self.written.append(ct.string_at(header.data, header.length))
            header.flags |= 16
        elif name == "waveOutReset":
            for header in self.owned:
                header.flags |= 1
        elif name == "waveOutUnprepareHeader":
            if self.fail_unprepare:
                raise RuntimeError("Driver still owns buffer")
            self.owned.remove(args[1]._obj)
        elif name == "waveOutClose":
            if self.owned:
                raise AssertionError("Device closed with live driver buffers")


class PlayerTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeWinMM()
        self.player = WindowsPlayer(api=self.api)
        self.addCleanup(self.player.stop)

    def test_many_chunks_use_one_device_and_keep_memory_until_played(self):
        self.player.open_stream(24000)
        for _ in range(3):
            self.player.write(PCM)
        self.assertEqual(1, self.api.calls.count("waveOutOpen"))
        self.assertEqual([PCM] * 3, self.api.written)
        self.assertEqual(3, len(self.player.buffers))
        self.assertEqual(0, self.player.underruns)
        for header in self.api.owned:
            header.flags |= 1
        self.player.finish(threading.Event())
        self.assertEqual(0, len(self.player.buffers))
        self.assertEqual(3, self.api.calls.count("waveOutUnprepareHeader"))

    def test_stop_flushes_and_releases_native_buffers_in_order(self):
        self.player.open_stream()
        self.player.write(PCM)
        self.player.stop()
        self.assertEqual(["waveOutReset", "waveOutUnprepareHeader", "waveOutClose"], self.api.calls[-3:])
        self.assertIsNone(self.player.handle)
        self.assertFalse(self.api.owned)
        with self.assertRaises(Cancelled):
            self.player.write(PCM)

    def test_backpressure_wait_can_be_interrupted_without_device_lock(self):
        self.player.max_buffer_seconds = 0.001
        self.player.open_stream()
        self.player.write(PCM)
        cancel = threading.Event()
        exited = threading.Event()

        def wait():
            try:
                self.player.wait_capacity(cancel)
            except Cancelled:
                exited.set()

        worker = threading.Thread(target=wait)
        worker.start()
        cancel.set()
        self.player.stop()
        worker.join(1)
        self.assertTrue(exited.is_set())

    def test_failed_unprepare_keeps_buffer_allocations_alive(self):
        self.player.open_stream()
        self.player.write(PCM)
        self.api.fail_unprepare = True
        with self.assertRaisesRegex(RuntimeError, "still owns"):
            self.player.stop()
        self.assertEqual(1, len(self.player.buffers))
        self.assertEqual(PCM, ct.string_at(self.api.owned[0].data, len(PCM)))
        self.api.fail_unprepare = False
        self.player.stop()

    def test_write_failure_unprepares_the_rejected_buffer(self):
        self.player.open_stream()
        self.api.fail_write = True
        with self.assertRaisesRegex(RuntimeError, "write failed"):
            self.player.write(PCM)
        self.assertFalse(self.player.buffers)
        self.assertFalse(self.api.owned)

    def test_empty_device_queue_between_chunks_counts_an_underrun(self):
        self.player.open_stream()
        self.player.write(PCM)
        self.api.owned[0].flags |= 1
        self.player.write(PCM)
        self.assertEqual(1, self.player.underruns)


class StreamingControllerTests(unittest.TestCase):
    def make_controller(self, client, player):
        controller = SpeechController(client, player)
        self.addCleanup(controller.close)
        return controller

    def test_first_audio_plays_before_generator_finishes(self):
        player = FakePlayer()
        release = threading.Event()
        entered_second_chunk = threading.Event()
        self.addCleanup(release.set)

        class Client(FakeClient):
            def stream_synthesize(self, *args):
                yield "H", FORMAT
                yield "A", PCM
                entered_second_chunk.set()
                release.wait(2)
                yield "A", PCM
                yield "M", {"audio_bytes": len(PCM) * 2, "text_segments": 1}

        controller = self.make_controller(Client(), player)
        job = controller.speak("First words. The remainder is still generating.")
        self.assertTrue(entered_second_chunk.wait(1))
        self.assertTrue(player.started_event.is_set())
        self.assertEqual("playing", controller.status(job["id"])["status"])
        self.assertEqual(1, controller.status(job["id"])["audio_chunks_received"])
        release.set()
        done = await_job(controller, job["id"])
        self.assertEqual("completed", done["status"])
        self.assertEqual(2, done["audio_chunks_received"])
        self.assertIn("first_playback_ms", done["metrics"])

    def test_long_reply_is_one_http_stream_not_one_request_per_sentence(self):
        client = FakeClient()
        controller = self.make_controller(client, FakePlayer())
        text = "This is another sentence in a longer response. " * 30
        job = controller.speak(text)
        self.assertEqual("completed", await_job(controller, job["id"])["status"])
        self.assertEqual([(text.strip(), DEFAULT_VOICE)], client.texts)

    def test_incomplete_stream_after_audio_is_failed_and_flushed(self):
        class Client(FakeClient):
            def stream_synthesize(self, *args):
                yield "H", FORMAT
                yield "A", PCM

        player = FakePlayer()
        controller = self.make_controller(Client(), player)
        job = controller.speak("Incomplete audio must not count as success.")
        state = await_job(controller, job["id"])
        self.assertEqual("failed", state["status"])
        self.assertIn("without completed", state["error"])
        self.assertGreater(player.stopped, 0)

    def test_stop_interrupts_blocked_stream_before_sending_remote_cancel(self):
        entered = threading.Event()
        release = threading.Event()
        order = []

        class Client(FakeClient):
            def stream_synthesize(self, *args):
                yield "H", FORMAT
                yield "A", PCM
                entered.set()
                release.wait(2)
                yield "A", PCM

            def close_stream(self, identifier):
                order.append("close")
                release.set()

            def cancel(self, identifier):
                order.append("cancel")

        player = FakePlayer()
        controller = self.make_controller(Client(), player)
        job = controller.speak("Stop while the next audio frame is pending.")
        self.assertTrue(entered.wait(1))
        controller.stop()
        controller.queue.join()
        self.assertEqual(["close", "cancel"], order)
        self.assertEqual("cancelled", controller.status(job["id"])["status"])
        self.assertEqual(1, len(player.started))

    def test_old_container_is_rejected_instead_of_silently_using_slow_path(self):
        client = FakeClient()
        client.health = lambda: {"status": "ready"}
        controller = self.make_controller(client, FakePlayer())
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            controller.speak("Old container.")


class StreamingEngineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.engine = QwenEngine(PROJECT / "config" / "voice.json", directory.name)
        self.engine.store.save(DEFAULT_VOICE, sample_wav(), {"reference_text": "Saved voice transcript."})
        self.engine.state = "ready"
        self.engine.gpu = "NVIDIA GeForce RTX 5090"
        self.engine.model_kind = "base"
        self.engine.model_snapshot = "test-revision"
        self.engine.torch = SimpleNamespace(
            bfloat16="bf16", inference_mode=contextlib.nullcontext,
            cuda=SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
                                 max_memory_allocated=lambda: 1024**3))
        self.engine._pcm = lambda waveform: (waveform, 0.25)
        verified = patch("local_voice.engine.verify_cuda_parameters", return_value=42)
        verified.start()
        self.addCleanup(verified.stop)

    def model(self, generate):
        self.engine.model = SimpleNamespace(_warmed_up=True, generate_voice_clone_streaming=generate)

    def test_cancel_at_chunk_boundary_closes_decoder_and_releases_gpu(self):
        closed = []

        def generate(**kwargs):
            try:
                yield PCM, 24000, {"steps": 4}
                self.fail("Decoded another chunk after cancellation")
            finally:
                closed.append(True)

        self.model(generate)
        identifier = uuid.uuid4().hex
        stream = self.engine.stream_synthesize("Hello.", DEFAULT_VOICE, identifier)
        self.assertEqual("H", next(stream)[0])
        self.assertEqual("A", next(stream)[0])
        self.engine.cancellations.cancel(identifier)
        with self.assertRaises(Cancelled):
            next(stream)
        self.assertEqual([True], closed)
        self.assertFalse(self.engine.gpu_lock.locked())
        self.assertFalse(self.engine.cancellations.active)

    def test_disconnect_closes_generator_without_finishing_the_utterance(self):
        closed = []

        def generate(**kwargs):
            try:
                yield PCM, 24000, {}
                self.fail("Generation continued after client disconnected")
            finally:
                closed.append(True)

        self.model(generate)
        stream = self.engine.stream_synthesize("Hello.", DEFAULT_VOICE, uuid.uuid4().hex)
        next(stream)
        next(stream)
        stream.close()
        self.assertEqual([True], closed)
        self.assertFalse(self.engine.gpu_lock.locked())

    def test_bounded_context_segments_preserve_the_whole_long_reply(self):
        segments = []

        def generate(**kwargs):
            segments.append(kwargs["text"])
            yield PCM, 24000, {"steps": 4}

        self.model(generate)
        text = "A complete sentence with a natural ending. " * 65
        frames = list(self.engine.stream_synthesize(text, DEFAULT_VOICE, uuid.uuid4().hex))
        self.assertGreater(len(segments), 1)
        self.assertTrue(all(len(segment) <= 800 for segment in segments))
        self.assertEqual("".join(text.split()), "".join("".join(segments).split()))
        self.assertEqual(1, sum(kind == "H" for kind, _ in frames))
        self.assertEqual(len(segments), frames[-1][1]["text_segments"])
        self.assertEqual(len(PCM) * len(segments), frames[-1][1]["audio_bytes"])

    def test_generation_limit_is_reported_instead_of_silent_truncation(self):
        def generate(**kwargs):
            yield PCM, 24000, {"total_steps_so_far": 1536}

        self.model(generate)
        with self.assertRaisesRegex(RuntimeError, "generation limit"):
            list(self.engine.stream_synthesize("Hello.", DEFAULT_VOICE, uuid.uuid4().hex))

    def test_static_context_limit_is_not_reported_as_complete(self):
        def generate(**kwargs):
            yield PCM, 24000, {"total_steps_so_far": 1500}

        self.model(generate)
        self.engine.model.talker_graph = SimpleNamespace(
            max_seq_len=2048, cache_position=SimpleNamespace(item=lambda: 2046))
        with self.assertRaisesRegex(RuntimeError, "context limit"):
            list(self.engine.stream_synthesize("Hello.", DEFAULT_VOICE, uuid.uuid4().hex))

    def test_warmup_caches_reference_and_runs_only_once_per_loaded_voice(self):
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            yield PCM, 24000, {}

        self.model(generate)
        self.engine._warm_voice(DEFAULT_VOICE)
        self.engine._warm_voice(DEFAULT_VOICE)
        self.assertEqual(1, len(calls))
        self.assertEqual("Saved voice transcript.", calls[0]["ref_text"])
        self.assertTrue(Path(calls[0]["ref_audio"]).is_file())
        self.assertFalse(calls[0]["xvec_only"])

    def test_load_selects_cuda_graph_backend_and_captures_before_ready(self):
        called = {}

        class FasterModel:
            _warmed_up = False

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                called.update(kwargs)
                return cls()

            def warmup(self, prefill_len):
                self._warmed_up = True
                called["warmed"] = True

        self.engine.model = None
        with patch.dict(sys.modules, {
            "huggingface_hub": SimpleNamespace(snapshot_download=lambda **kwargs: "/models/revision"),
            "qwen_tts": SimpleNamespace(Qwen3TTSModel=None),
            "faster_qwen3_tts": SimpleNamespace(FasterQwen3TTS=FasterModel),
        }):
            self.engine._load_model("base")
        self.assertEqual("torch", called["backend"])
        self.assertEqual("cuda:0", called["device"])
        self.assertTrue(called["warmed"])


class FakeSocket:
    def __init__(self, request):
        self.request = io.BytesIO(request)
        self.sent = bytearray()

    def makefile(self, *args):
        return self.request

    def sendall(self, data):
        self.sent.extend(data)

    def settimeout(self, value):
        pass

    def setsockopt(self, *args):
        pass


class HttpStreamTests(unittest.TestCase):
    def request(self, generate, origin=False):
        body = json.dumps({"text": "Hello.", "request_id": uuid.uuid4().hex}).encode()
        origin_header = b"Origin: https://example.com\r\n" if origin else b""
        request = (b"POST /speech/stream HTTP/1.0\r\nHost: 127.0.0.1:8765\r\n"
                   b"Content-Type: application/json\r\n" + origin_header
                   + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
        sock = FakeSocket(request)
        engine = SimpleNamespace(config={"name": DEFAULT_VOICE},
                                 stream_synthesize=lambda *args: generate(sock))
        Handler(sock, ("127.0.0.1", 1234), SimpleNamespace(engine=engine))
        return bytes(sock.sent)

    def test_http_flushes_first_pcm_before_resuming_generation(self):
        def generate(sock):
            yield "H", FORMAT
            yield "A", PCM
            self.assertIn(encode_frame("A", PCM), sock.sent)
            yield "M", {"audio_bytes": len(PCM)}

        response = self.request(generate)
        headers, _, body = response.partition(b"\r\n\r\n")
        self.assertIn(b"200 OK", headers)
        self.assertNotIn(b"Content-Length:", headers)
        self.assertEqual(["H", "A", "M"], [kind for kind, _ in read_frames(io.BytesIO(body))])

    def test_http_emits_error_frame_after_partial_generation_failure(self):
        def generate(sock):
            yield "H", FORMAT
            yield "A", PCM
            raise RuntimeError("GPU failed midway")

        with self.assertLogs("local_voice.http_server", level="ERROR"):
            response = self.request(generate)
        with self.assertRaisesRegex(RuntimeError, "GPU failed midway"):
            list(read_frames(io.BytesIO(response.partition(b"\r\n\r\n")[2])))

    def test_stream_endpoint_keeps_browser_origin_protection(self):
        response = self.request(lambda sock: self.fail("Browser started inference"), origin=True)
        self.assertIn(b"403 Forbidden", response)


class ClientStreamTests(unittest.TestCase):
    def make_transport(self, payload, *, block_at_end=False):
        release = threading.Event()
        waiting = threading.Event()
        calls = []

        class Socket:
            def shutdown(self, how):
                calls.append("shutdown")
                release.set()

        sock = Socket()

        class Response(io.BytesIO):
            status = 200
            fp = SimpleNamespace(raw=SimpleNamespace(_sock=sock))

            def getheader(self, key, default=None):
                return "application/x-local-qwen-pcm" if key == "Content-Type" else default

            def read(self, length):
                if block_at_end and self.tell() == len(payload):
                    waiting.set()
                    if not release.wait(2):
                        raise AssertionError("Cancellation failed to interrupt the HTTP read")
                return super().read(length)

        response = Response(payload)

        class Connection:
            # Simulate HTTP/1.0: its socket is held only by HTTPResponse.fp.
            sock = None

            def __init__(self, *args, **kwargs):
                pass

            def request(self, method, path, **kwargs):
                calls.append(path)

            def getresponse(self):
                return response

            def close(self):
                calls.append("close")

        self.addCleanup(release.set)
        return Connection, response, waiting, calls

    def test_http_client_yields_pcm_without_reading_the_complete_response(self):
        payload = encode_frame("H", FORMAT) + encode_frame("A", PCM) + encode_frame("M", {"audio_bytes": len(PCM)})
        connection, response, _, calls = self.make_transport(payload)
        client = EngineClient()
        with patch("local_voice.client.http.client.HTTPConnection", connection):
            stream = client.stream_synthesize("Hello.", DEFAULT_VOICE, uuid.uuid4().hex, threading.Event())
            self.assertEqual("H", next(stream)[0])
            self.assertLess(response.tell(), len(payload))
            self.assertEqual(("A", PCM), next(stream))
            self.assertLess(response.tell(), len(payload))
            self.assertEqual("M", next(stream)[0])
            with self.assertRaises(StopIteration):
                next(stream)
        self.assertEqual("/speech/stream", calls[0])
        self.assertTrue(response.closed)
        self.assertFalse(client.streams)

    def test_cancel_shuts_down_detached_socket_and_unblocks_reader(self):
        connection, response, waiting, calls = self.make_transport(encode_frame("H", FORMAT), block_at_end=True)
        client = EngineClient()
        cancel = threading.Event()
        identifier = uuid.uuid4().hex
        errors = []
        with patch("local_voice.client.http.client.HTTPConnection", connection):
            stream = client.stream_synthesize("Hello.", DEFAULT_VOICE, identifier, cancel)
            next(stream)

            def consume():
                try:
                    next(stream)
                except Exception as exc:
                    errors.append(exc)

            worker = threading.Thread(target=consume)
            worker.start()
            self.assertTrue(waiting.wait(1))
            cancel.set()
            client.close_stream(identifier)
            worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(1, len(errors))
        self.assertIsInstance(errors[0], Cancelled)
        self.assertIn("shutdown", calls)
        self.assertTrue(response.closed)
        self.assertFalse(client.streams)

    def test_cancel_before_connect_does_not_start_a_request(self):
        cancel = threading.Event()
        cancel.set()
        client = EngineClient()
        with patch("local_voice.client.http.client.HTTPConnection") as connection:
            with self.assertRaises(Cancelled):
                list(client.stream_synthesize("Hello.", DEFAULT_VOICE, uuid.uuid4().hex, cancel))
            connection.return_value.request.assert_not_called()


class BenchmarkTests(unittest.TestCase):
    def test_benchmark_plays_before_the_second_chunk_is_produced(self):
        player = FakePlayer()

        class Client(FakeClient):
            def stream_synthesize(self, *args):
                yield "H", FORMAT
                yield "A", PCM
                if not player.started_event.is_set():
                    raise AssertionError("Playback waited for the entire response")
                yield "A", PCM
                yield "M", {"audio_bytes": 2 * len(PCM), "audio_seconds": 0.02}

        audio, metrics = stream_sample(Client(), "Hello.", DEFAULT_VOICE, player)
        self.assertTrue(audio.startswith(b"RIFF"))
        self.assertIn("first_playback_ms", metrics)
        self.assertEqual(2, len(player.started))

    def test_interactive_check_rejects_slow_and_buffered_results(self):
        good = {"backend": "faster-qwen3-tts", "cuda_graphs": True, "streaming": True,
                "audio_chunks": 20, "real_time_factor": 0.25,
                "first_audio_received_ms": 250, "playback_underruns": 0}
        verify_interactive(good)
        for changes in ({"streaming": False}, {"cuda_graphs": False}, {"audio_chunks": 1},
                        {"real_time_factor": 3}, {"first_audio_received_ms": 1500},
                        {"delivery_real_time_factor": 2}, {"playback_underruns": 1}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                verify_interactive(dict(good, **changes))


if __name__ == "__main__":
    unittest.main()
