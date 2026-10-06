"""Bounded PCM stream framing shared by the container and Windows client."""

import io
import json
import struct
import wave

CONTENT_TYPE = "application/x-local-qwen-pcm"
FORMAT = {"protocol": 1, "sample_rate": 24000, "channels": 1, "sample_width": 2}
FRAME = struct.Struct("!cI")
MAX_FRAME = 1024 * 1024
MAX_AUDIO = 64 * 1024 * 1024


def encode_frame(kind, value):
    if kind not in ("H", "A", "M", "E"):
        raise ValueError("Unknown audio frame type")
    payload = value if kind == "A" else json.dumps(value, separators=(",", ":")).encode("utf-8")
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_FRAME:
        raise ValueError("Invalid audio frame size")
    if kind == "A" and len(payload) % 2:
        raise ValueError("PCM frames must contain complete 16-bit samples")
    return FRAME.pack(kind.encode("ascii"), len(payload)) + payload


def _read_exact(stream, length):
    parts = []
    remaining = length
    while remaining:
        part = stream.read(remaining)
        if not part:
            raise RuntimeError("TTS audio stream ended before its completion frame")
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


def read_frames(stream):
    """Require format -> PCM -> metrics; truncation is never a successful job."""
    opened = False
    audio_bytes = 0
    while True:
        kind, size = FRAME.unpack(_read_exact(stream, FRAME.size))
        if kind not in (b"H", b"A", b"M", b"E") or not 0 < size <= MAX_FRAME:
            raise RuntimeError("Invalid TTS stream frame")
        payload = _read_exact(stream, size)
        if kind == b"A":
            if not opened or size % 2:
                raise RuntimeError("PCM arrived before its format or contains an incomplete sample")
            audio_bytes += size
            if audio_bytes > MAX_AUDIO:
                raise RuntimeError("TTS stream exceeds the audio size limit")
            yield "A", payload
            continue
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise RuntimeError("Invalid TTS stream metadata")
        if kind == b"E":
            raise RuntimeError(value.get("error", "Streaming synthesis failed"))
        if kind == b"H":
            if opened or value != FORMAT:
                raise RuntimeError("Unsupported or repeated PCM stream format")
            opened = True
            yield "H", value
        else:
            if not opened or not audio_bytes or value.get("audio_bytes") != audio_bytes:
                raise RuntimeError("TTS completion frame does not match the received audio")
            yield "M", value
            return


def pcm_to_wav(pcm, sample_rate=24000):
    if not pcm or len(pcm) % 2 or len(pcm) > MAX_AUDIO:
        raise ValueError("Invalid PCM audio")
    result = io.BytesIO()
    with wave.open(result, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm)
    return result.getvalue()
