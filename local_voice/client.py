"""Standard-library HTTP client, restricted to the user's local speech service."""

import http.client
import json
import socket
import threading
from urllib.parse import urlsplit

from .common import Cancelled, wav_info
from .streaming import CONTENT_TYPE, read_frames


class EngineClient:
    def __init__(self, url="http://127.0.0.1:8765"):
        parts = urlsplit(url)
        if (parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost")
                or parts.username or parts.password or parts.query or parts.fragment
                or parts.path not in ("", "/")):
            raise ValueError("TTS_URL must be an HTTP localhost URL without credentials or a path")
        self.host = parts.hostname
        self.port = parts.port or 80
        self.stream_lock = threading.Lock()
        self.streams = {}

    def request(self, method, path, body=None, timeout=600):
        # http.client does not honor proxy environment variables.
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        try:
            connection.request(method, path, body=payload, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            data = response.read(64 * 1024 * 1024 + 1)
            if len(data) > 64 * 1024 * 1024:
                raise RuntimeError("Speech service returned an oversized response")
            if response.status >= 400:
                try:
                    detail = json.loads(data).get("error", data.decode("utf-8", errors="replace"))
                except (ValueError, AttributeError):
                    detail = data.decode("utf-8", errors="replace")
                raise RuntimeError(f"Speech service HTTP {response.status}: {detail}")
            if response.getheader("Content-Type", "").split(";")[0] == "audio/wav":
                wav_info(data)
                metrics = json.loads(response.getheader("X-TTS-Metrics", "{}"))
                return data, metrics
            return json.loads(data)
        except OSError as exc:
            raise RuntimeError(f"Cannot reach local TTS: {exc}. Run scripts/start.ps1 and check its health.") from exc
        finally:
            connection.close()

    def health(self):
        return self.request("GET", "/health", timeout=2)

    def voices(self):
        return self.request("GET", "/voices", timeout=5)

    def synthesize(self, text, voice, identifier):
        return self.request("POST", "/speech", {"text": text, "voice": voice, "request_id": identifier})

    def stream_synthesize(self, text, voice, identifier, cancel):
        connection = http.client.HTTPConnection(self.host, self.port, timeout=120)
        with self.stream_lock:
            if cancel.is_set():
                raise Cancelled("Speech cancelled before connecting")
            self.streams[identifier] = connection
        try:
            payload = json.dumps({"text": text, "voice": voice, "request_id": identifier}).encode("utf-8")
            connection.request("POST", "/speech/stream", body=payload,
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                detail = response.read(32768).decode("utf-8", errors="replace")
                raise RuntimeError(f"Streaming TTS HTTP {response.status}: {detail}. Rebuild the Compose service if it is out of date.")
            if response.getheader("Content-Type", "").split(";")[0] != CONTENT_TYPE:
                raise RuntimeError("The speech engine returned an unsupported streaming format")
            # HTTP/1.0 responses can detach their socket from HTTPConnection.
            # Track the response too, so close_stream can interrupt a blocked read.
            with self.stream_lock:
                self.streams[identifier] = (connection, response)
            if cancel.is_set():
                raise Cancelled("Speech cancelled before audio arrived")
            for frame in read_frames(response):
                if cancel.is_set():
                    raise Cancelled("Speech cancelled")
                yield frame
        except (OSError, http.client.HTTPException, RuntimeError) as exc:
            if cancel.is_set():
                raise Cancelled("Speech cancelled") from exc
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"Local TTS stream failed: {exc}") from exc
        finally:
            with self.stream_lock:
                self.streams.pop(identifier, None)
            if 'response' in locals():
                response.close()
            connection.close()

    def close_stream(self, identifier):
        with self.stream_lock:
            active = self.streams.get(identifier)
        if active is None:
            return
        connection, response = active if isinstance(active, tuple) else (active, None)
        sock = connection.sock
        if sock is None and response is not None and response.fp is not None:
            sock = getattr(getattr(response.fp, "raw", None), "_sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        # The reading thread owns response.close(); closing its buffered reader
        # from here could wait on its internal lock before shutdown takes effect.
        connection.close()

    def design(self, name, description, identifier):
        return self.request("POST", "/voices", {
            "name": name, "description": description, "request_id": identifier,
        }, timeout=900)

    def cancel(self, identifier):
        return self.request("POST", "/cancel", {"request_id": identifier}, timeout=2)
