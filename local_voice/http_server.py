"""Local HTTP synthesis service; model work is serialized by QwenEngine."""

import json
from contextlib import closing
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import socket

from .common import Cancelled, NotReady, REFERENCE_TEXT
from .engine import QwenEngine
from .streaming import CONTENT_TYPE, encode_frame

MAX_BODY = 32 * 1024
log = logging.getLogger(__name__)


def dispatch(engine, method, path, body=None):
    """A transport-independent handler, also used by the integration tests."""
    body = body or {}
    if method == "GET" and path in ("/health", "/readyz"):
        state = engine.health()
        return (200 if path == "/health" or state["status"] == "ready" else 503), state, {}
    if method == "GET" and path == "/voices":
        return 200, {"voices": engine.store.list(), "default_voice": engine.config["name"]}, {}
    if method == "POST" and path == "/cancel":
        active = engine.cancellations.cancel(body.get("request_id"))
        return 200, {"cancelled": True, "was_active": active}, {}
    if method == "POST" and path == "/voices":
        voice = engine.design(
            body.get("name"), body.get("description"),
            body.get("reference_text", REFERENCE_TEXT), body.get("request_id"),
        )
        return 201, voice, {}
    if method == "POST" and path == "/speech":
        audio, metrics = engine.synthesize(
            body.get("text"), body.get("voice", engine.config["name"]), body.get("request_id"),
        )
        return 200, audio, {"X-TTS-Metrics": json.dumps(metrics, separators=(",", ":"))}
    return 404, {"error": "Unknown endpoint"}, {}


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalQwenVoice/0.2"

    def setup(self):
        super().setup()
        self.connection.settimeout(30)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def _handle(self, method):
        try:
            # Browser pages are not clients of this private localhost service.
            if self.headers.get("Origin") or self.headers.get("Sec-Fetch-Site"):
                self._send(403, {"error": "Browser-origin requests are not supported"})
                return
            host = self.headers.get("Host", "").split(":")[0].lower()
            if host not in ("localhost", "127.0.0.1", "tts"):
                self._send(403, {"error": "Unrecognized Host header"})
                return
            body = None
            if method == "POST":
                if self.headers.get_content_type() != "application/json":
                    self._send(415, {"error": "application/json required"})
                    return
                if self.headers.get("Transfer-Encoding"):
                    self._send(400, {"error": "Content-Length required"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    self._send(413, {"error": "Request body must be 1-32768 bytes"})
                    return
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise ValueError("Incomplete request body")
                body = json.loads(payload)
                if not isinstance(body, dict):
                    raise ValueError("JSON body must be an object")
            if method == "POST" and self.path == "/speech/stream":
                self._stream(body)
                return
            status, result, headers = dispatch(self.server.engine, method, self.path, body)
            self._send(status, result, headers)
        except Cancelled as exc:
            self._send(409, {"error": str(exc), "cancelled": True})
        except NotReady as exc:
            self._send(503, {"error": str(exc)})
        except (ValueError, TypeError, KeyError) as exc:
            self._send(400, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            log.exception("Request failed")
            self._send(500, {"error": str(exc)})

    def _send(self, status, result, headers=None):
        binary = isinstance(result, bytes)
        payload = result if binary else json.dumps(result).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "audio/wav" if binary else "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def _stream(self, body):
        engine = self.server.engine
        with closing(engine.stream_synthesize(body.get("text"),
                     body.get("voice", engine.config["name"]), body.get("request_id"))) as frames:
            # Validate readiness, voice and request before committing HTTP 200.
            first = next(frames)
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                self.wfile.write(encode_frame(*first))
                self.wfile.flush()
                for kind, value in frames:
                    self.wfile.write(encode_frame(kind, value))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                # Closing the generator releases its GPU lock and decoding state.
                pass
            except Exception as exc:
                log.exception("Streaming synthesis failed")
                try:
                    self.wfile.write(encode_frame("E", {"error": str(exc)}))
                    self.wfile.flush()
                except OSError:
                    pass

    def log_message(self, format, *args):
        log.info(format, *args)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = QwenEngine()
    server = ThreadingHTTPServer(("0.0.0.0", 8765), Handler)
    server.daemon_threads = True
    server.engine = engine
    threading.Thread(target=engine.initialize, daemon=True, name="model-loader").start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
