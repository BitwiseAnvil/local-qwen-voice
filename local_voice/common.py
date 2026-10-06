"""Validation, PCM audio, and sentence boundaries shared without ML dependencies."""

import io
import re
import wave

DEFAULT_VOICE = "warm_american_female"
DEFAULT_DESCRIPTION = "Warm American female voice."
REFERENCE_TEXT = (
    "Hello, I'm here to help. We can work through your ideas together, one step at a time. "
    "Tell me what you're thinking, and I'll keep things clear, practical, and easy to follow."
)


class Cancelled(Exception):
    pass


class NotReady(Exception):
    pass


def clean_text(value, *, field="text", limit=4000):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    value = value.strip()
    if len(value) > limit:
        raise ValueError(f"{field} must be at most {limit} characters")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise ValueError(f"{field} contains control characters")
    return value


def voice_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", value):
        raise ValueError("Voice names must start with a-z and contain 1-48 lowercase letters, digits, _ or -")
    return value


def request_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9-]{16,64}", value):
        raise ValueError("request_id must be a UUID-style identifier")
    return value


def sentence_chunks(text, limit=280):
    """Bound synthesis latency without dropping text, including very long words."""
    text = clean_text(text)
    result = []
    # Keep punctuation attached; avoid cutting immediately after a one-letter initial.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n+", text)
    for sentence in sentences:
        sentence = sentence.strip()
        while len(sentence) > limit:
            boundary = sentence.rfind(" ", 0, limit + 1)
            if boundary < limit // 3:
                boundary = limit
            result.append(sentence[:boundary].strip())
            sentence = sentence[boundary:].strip()
        if sentence:
            if result and len(result[-1]) + len(sentence) + 1 <= limit:
                result[-1] += " " + sentence
            else:
                result.append(sentence)
    return result


def wav_info(data):
    if not isinstance(data, bytes) or len(data) > 64 * 1024 * 1024:
        raise ValueError("Invalid or oversized WAV response")
    try:
        with wave.open(io.BytesIO(data), "rb") as stream:
            if stream.getcomptype() != "NONE" or stream.getsampwidth() != 2:
                raise ValueError("Expected uncompressed 16-bit PCM WAV audio")
            frames = stream.getnframes()
            rate = stream.getframerate()
            channels = stream.getnchannels()
            if frames < 1 or rate < 8000 or channels != 1:
                raise ValueError("Expected nonempty mono audio at 8 kHz or above")
            if len(stream.readframes(frames)) != frames * channels * 2:
                raise ValueError("Truncated WAV audio")
            return {"sample_rate": rate, "duration_seconds": frames / rate, "frames": frames}
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Invalid WAV audio: {exc}") from exc

