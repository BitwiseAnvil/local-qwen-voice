"""MCP stdio tools using the 2025-06-18 protocol and Python's standard library.

Only the tools capability is advertised. Inference stays in Docker; this process
owns Windows speaker playback. stdout contains JSON-RPC messages exclusively.
"""

import json
from pathlib import Path
import queue
import sys

from . import __version__
from .client import EngineClient
from .controller import SpeechController
from .playback import WindowsPlayer
from .settings import backend_url, default_voice

PROTOCOL_VERSION = "2025-06-18"
MAX_MESSAGE = 64 * 1024
# Sent to every client on initialize with {voice} set to the configured default.
# Claude Code truncates this to about 2 KB. Codex ignores it; see README.
INSTRUCTIONS = (
    "Local streaming speech on the user's computer. "
    "When to speak: (1) after every final answer, give a 1-3 sentence spoken summary of the key outcome; "
    "(2) when you are blocked waiting for the user's answer, approval, or decision, briefly say what you need; "
    "(3) when a long-running task such as a build, test run, or install finishes or fails, announce the result. "
    "Speak plain natural language only. Keep code, file paths, commands, tables, and long lists in the written reply. "
    'Always pass voice "{voice}" unless the user asks for a different voice. '
    "How to speak: send each spoken message in ONE speak call; the server streams it continuously. "
    "Do not split a message into sentence-by-sentence calls or wait for each sentence. "
    "After queueing, check speech_status once; playing means audio has started, so do not keep polling. "
    "If the job failed, say so instead of claiming speech played. "
    "Silence: when the user says stop or quiet, or interrupts, call stop_speaking immediately and stay silent "
    "for the rest of that reply; resume normal speech on the next answer. "
    "If the user asks for no speech at all, stay silent until they ask for it again."
)


def tool(name, description, properties, required=(), read_only=False):
    return {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": read_only, "destructiveHint": False, "openWorldHint": False},
    }


TOOLS = [
    tool("speak", "Stream local Qwen speech to the user's Windows speakers while it generates. Send the complete spoken response in ONE call, up to 4000 characters; do not split sentences into separate calls or wait between chunks. Returns a job id immediately. Keep spoken replies concise; omit code and URLs. speech_status reports first-audio timing and errors.", {
        "text": {"type": "string", "minLength": 1, "maxLength": 4000},
        "voice": {"type": "string", "description": "Saved voice name; omitted uses the configured default."},
    }, ["text"]),
    tool("stop_speaking", "Immediately stop playback, cancel generation, and discard queued speech and voice-design jobs.", {}),
    tool("speech_status", "Get a queued job's state or recent jobs. completed means playback/design succeeded; failed contains an error. Also reports Docker engine health and GPU information.", {
        "job_id": {"type": "string"},
    }, read_only=True),
    tool("list_voices", "List saved locally designed voices, descriptions, and the default voice.", {}, read_only=True),
    tool("design_voice", "Create a reusable local voice from a description, without a user recording. This queues a GPU job; check speech_status for completion. Existing names are preserved. Use the new name in speak's voice argument.", {
        "name": {"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,47}$"},
        "description": {"type": "string", "minLength": 1, "maxLength": 2000},
    }, ["name", "description"]),
]


class RpcError(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


class RpcServer:
    def __init__(self, controller, client):
        self.controller = controller
        self.client = client
        self.initialized = False
        self.ready = False

    def receive(self, message):
        identifier = None
        try:
            if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                raise RpcError(-32600, "Invalid JSON-RPC request")
            identifier = message.get("id")
            if "id" in message and (identifier is None or isinstance(identifier, bool) or not isinstance(identifier, (str, int))):
                identifier = None
                raise RpcError(-32600, "Request id must be a string or integer")
            method = message.get("method")
            if not isinstance(method, str):
                raise RpcError(-32600, "Request method must be a string")
            params = message.get("params", {})
            if not isinstance(params, dict):
                raise RpcError(-32602, "params must be an object")
            if "id" not in message:
                if method == "notifications/initialized" and self.initialized:
                    self.ready = True
                # No responses to notifications, including cancellation of completed
                # tool requests. Audio jobs are separate and use stop_speaking.
                return None
            result = self._request(method, params)
            return {"jsonrpc": "2.0", "id": identifier, "result": result}
        except RpcError as exc:
            if isinstance(message, dict) and "id" not in message and isinstance(message.get("method"), str):
                return None
            return {"jsonrpc": "2.0", "id": identifier, "error": {"code": exc.code, "message": str(exc)}}

    def _request(self, method, params):
        if method == "ping":
            return {}
        if method == "initialize":
            if self.initialized:
                raise RpcError(-32600, "Session is already initialized")
            if not isinstance(params.get("protocolVersion"), str):
                raise RpcError(-32602, "protocolVersion is required")
            if not isinstance(params.get("capabilities"), dict) or not isinstance(params.get("clientInfo"), dict):
                raise RpcError(-32602, "clientInfo and capabilities are required")
            self.initialized = True
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "local-qwen-voice", "version": __version__},
                "instructions": INSTRUCTIONS.format(voice=self.controller.default_voice),
            }
        if not self.ready:
            raise RpcError(-32000, "Initialize the session and send notifications/initialized first")
        if method == "tools/list":
            return {"tools": TOOLS}
        if method != "tools/call":
            raise RpcError(-32601, f"Unknown method: {method}")
        name = params.get("name")
        descriptor = next((item for item in TOOLS if item["name"] == name), None)
        if descriptor is None:
            raise RpcError(-32602, "Unknown tool")
        arguments = params.get("arguments", {})
        schema = descriptor["inputSchema"]
        if not isinstance(arguments, dict) or set(arguments) - set(schema["properties"]):
            raise RpcError(-32602, "Invalid tool arguments")
        if set(schema["required"]) - set(arguments):
            raise RpcError(-32602, "Missing required tool arguments")
        if any(not isinstance(value, str) for value in arguments.values()):
            raise RpcError(-32602, "Tool arguments must be strings")
        try:
            value = self._call(name, arguments)
            return {"content": [{"type": "text", "text": json.dumps(value)}], "isError": False}
        except Exception as exc:
            message = "Speech queue is full; wait or call stop_speaking" if isinstance(exc, queue.Full) else str(exc)
            return {"content": [{"type": "text", "text": message}], "isError": True}

    def _call(self, name, arguments):
        if name == "speak":
            return self.controller.speak(**arguments)
        if name == "design_voice":
            return self.controller.design(**arguments)
        if name == "stop_speaking":
            return self.controller.stop()
        if name == "list_voices":
            return self.client.voices()
        result = self.controller.status(arguments.get("job_id"))
        result["bridge_version"] = __version__
        try:
            result["engine"] = self.client.health()
        except Exception as exc:
            result["engine"] = {"status": "unreachable", "error": str(exc)}
        return result


def main():
    project = Path(__file__).resolve().parents[1]
    client = EngineClient(backend_url(project))
    player = WindowsPlayer(project / ".runtime" / "audio")
    controller = SpeechController(client, player, default_voice(project))
    server = RpcServer(controller, client)
    try:
        while True:
            line = sys.stdin.buffer.readline(MAX_MESSAGE + 1)
            if not line:
                break
            if len(line) > MAX_MESSAGE:
                print("MCP message exceeded 64 KiB", file=sys.stderr)
                break
            try:
                response = server.receive(json.loads(line))
            except (ValueError, UnicodeDecodeError):
                response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Invalid JSON"}}
            if response is not None:
                sys.stdout.buffer.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
                sys.stdout.buffer.flush()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    finally:
        controller.close()


if __name__ == "__main__":
    main()
