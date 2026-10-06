"""Ordered speech/design jobs with prompt cancellation and bounded history."""

from collections import OrderedDict
from contextlib import closing
import queue
import threading
import time
import uuid

from .common import DEFAULT_VOICE, Cancelled, clean_text, voice_name


class SpeechController:
    def __init__(self, client, player, default_voice=DEFAULT_VOICE):
        self.client = client
        self.player = player
        self.default_voice = voice_name(default_voice)
        self.jobs = OrderedDict()
        self.queue = queue.Queue(maxsize=8)
        self.lock = threading.RLock()
        self.active = None
        self.closed = False
        self.worker = threading.Thread(target=self._work, name="speech-playback", daemon=True)
        self.worker.start()

    def _enqueue(self, kind, **arguments):
        state = self.client.health()
        if state.get("status") != "ready":
            raise RuntimeError(state.get("error") or "The TTS service is still loading")
        if kind == "speech" and state.get("streaming_protocol") != 1:
            raise RuntimeError("The running Docker image does not support streaming. Run scripts/start.ps1 to rebuild it, then reload the MCP server.")
        with self.lock:
            if self.closed:
                raise RuntimeError("Speech controller is closed")
            identifier = uuid.uuid4().hex
            job = {
                "id": identifier, "kind": kind, "status": "queued", "error": None,
                "created_at": time.time(), "cancel": threading.Event(), "request_id": None,
                "arguments": arguments, "chunks_completed": 0,
                "queued_at": time.perf_counter(), "audio_chunks_received": 0,
            }
            self.queue.put_nowait(job)
            self.jobs[identifier] = job
            while len(self.jobs) > 32:
                self.jobs.popitem(last=False)
            return self._public(job)

    def speak(self, text, voice=None):
        return self._enqueue("speech", text=clean_text(text), voice=voice_name(voice or self.default_voice))

    def design(self, name, description):
        return self._enqueue("design", name=voice_name(name), description=clean_text(description, field="description", limit=2000))

    @staticmethod
    def _public(job):
        keys = ("id", "kind", "status", "error", "created_at", "chunks_completed", "metrics", "voice",
                "audio_chunks_received", "first_audio_received_ms", "first_playback_ms")
        return {key: job[key] for key in keys if key in job}

    def status(self, identifier=None):
        with self.lock:
            if identifier:
                if identifier not in self.jobs:
                    raise ValueError("Unknown job id or job has expired from the recent history")
                return self._public(self.jobs[identifier])
            return {"jobs": [self._public(job) for job in self.jobs.values()], "default_voice": self.default_voice}

    def stop(self):
        identifiers = []
        count = 0
        errors = []
        with self.lock:
            for job in self.jobs.values():
                if job["status"] not in ("completed", "cancelled", "failed"):
                    job["cancel"].set()
                    job["status"] = "cancelled"
                    count += 1
                    if job["request_id"]:
                        identifiers.append(job["request_id"])
            try:
                self.player.stop()
            except Exception as exc:
                errors.append(str(exc))
            # Free bounded queue capacity immediately, even if the GPU is still
            # acknowledging cancellation of the active HTTP request.
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self.queue.task_done()
        for identifier in identifiers:
            # Interrupt a blocked HTTP read immediately, before the HTTP cancel.
            try:
                self.client.close_stream(identifier)
                self.client.cancel(identifier)
            except Exception as exc:
                errors.append(str(exc))
        return {"stopped": count, "cancellation_errors": errors}

    def close(self):
        with self.lock:
            self.closed = True
        self.stop()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.worker.join(timeout=3)

    def _check(self, job):
        if job["cancel"].is_set() or self.closed:
            raise Cancelled()

    def _request(self, job, function, *args):
        with self.lock:
            self._check(job)
            identifier = uuid.uuid4().hex
            job["request_id"] = identifier
            job["status"] = "generating"
        try:
            return function(*args, identifier)
        finally:
            with self.lock:
                job["request_id"] = None

    def _perform(self, job):
        self._check(job)
        arguments = job["arguments"]
        if job["kind"] == "design":
            voice = self._request(job, self.client.design, arguments["name"], arguments["description"])
            with self.lock:
                job["voice"] = voice
            return
        with self.lock:
            self._check(job)
            identifier = uuid.uuid4().hex
            job["request_id"] = identifier
            job["status"] = "generating"
        completed_stream = False
        try:
            with closing(self.client.stream_synthesize(arguments["text"], arguments["voice"],
                         identifier, job["cancel"])) as frames:
                for kind, value in frames:
                    self._check(job)
                    if kind == "H":
                        with self.lock:
                            self._check(job)
                            self.player.open_stream(value["sample_rate"])
                    elif kind == "A":
                        if not job["audio_chunks_received"]:
                            with self.lock:
                                job["first_audio_received_ms"] = round((time.perf_counter() - job["queued_at"]) * 1000, 1)
                        # Allow generation, network reads and playback to overlap.
                        # Bounded buffering also keeps memory and stop latency low.
                        self.player.wait_capacity(job["cancel"])
                        with self.lock:
                            self._check(job)
                            self.player.write(value)
                            if not job["audio_chunks_received"]:
                                job["first_playback_ms"] = round((time.perf_counter() - job["queued_at"]) * 1000, 1)
                            job["audio_chunks_received"] += 1
                            job["status"] = "playing"
                    elif kind == "M":
                        with self.lock:
                            job["metrics"] = dict(value,
                                first_audio_received_ms=job.get("first_audio_received_ms"),
                                first_playback_ms=job.get("first_playback_ms"))
                        completed_stream = True
            if not completed_stream or not job["audio_chunks_received"]:
                raise RuntimeError("Speech stream ended without completed audio")
        finally:
            with self.lock:
                job["request_id"] = None
        self.player.finish(job["cancel"])
        with self.lock:
            self._check(job)
            job["chunks_completed"] = job["metrics"].get("text_segments", 1)
            job["metrics"]["playback_underruns"] = getattr(self.player, "underruns", 0)

    def _work(self):
        while True:
            job = self.queue.get()
            if job is None:
                self.queue.task_done()
                return
            try:
                with self.lock:
                    self.active = job
                self._perform(job)
                with self.lock:
                    self._check(job)
                    job["status"] = "completed"
            except Exception as exc:
                with self.lock:
                    if job["cancel"].is_set() or isinstance(exc, Cancelled):
                        job["status"] = "cancelled"
                    else:
                        job["status"] = "failed"
                        job["error"] = str(exc)
            finally:
                with self.lock:
                    try:
                        self.player.stop()
                    except Exception as exc:
                        job["status"] = "failed"
                        job["error"] = f"Playback cleanup failed: {exc}"
                    self.active = None
                self.queue.task_done()
            if self.closed:
                return
