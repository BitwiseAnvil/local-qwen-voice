"""Resolve the bridge's port and voice from the same local project configuration."""

import json
import os
from pathlib import Path

from .common import DEFAULT_VOICE, voice_name


def backend_url(project):
    if os.environ.get("TTS_URL"):
        return os.environ["TTS_URL"]
    port = os.environ.get("TTS_PORT")
    env_file = Path(project) / ".env"
    if port is None and env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "TTS_PORT":
                port = value.split("#", 1)[0].strip().strip("\"'")
    port = int(port or "8765")
    if not 1 <= port <= 65535:
        raise ValueError("TTS_PORT must be between 1 and 65535")
    return f"http://127.0.0.1:{port}"


def default_voice(project):
    value = os.environ.get("TTS_DEFAULT_VOICE")
    if value is None:
        config_path = Path(project) / "config" / "voice.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        value = config.get("name", DEFAULT_VOICE)
    return voice_name(value)

