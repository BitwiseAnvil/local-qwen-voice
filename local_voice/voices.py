"""Persist a designed reference and its transcript, committing metadata last."""

import json
import os
from pathlib import Path
import uuid

from .common import clean_text, voice_name, wav_info


class VoiceStore:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def load(self, name):
        name = voice_name(name)
        metadata_path = self.directory / f"{name}.json"
        if not metadata_path.exists():
            raise ValueError(f"Unknown voice: {name}. Use list_voices or design_voice.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        audio_path = (self.directory / metadata["audio_file"]).resolve()
        if audio_path.parent != self.directory or audio_path.suffix != ".wav":
            raise ValueError("Voice audio path must stay inside the voice data directory")
        if metadata.get("name") != name:
            raise ValueError("Voice metadata does not match its name")
        clean_text(metadata.get("reference_text"), field="reference_text", limit=1000)
        wav_info(audio_path.read_bytes())
        return metadata, audio_path

    def exists(self, name):
        return (self.directory / f"{voice_name(name)}.json").exists()

    def list(self):
        result = []
        for path in sorted(self.directory.glob("*.json")):
            metadata, _ = self.load(path.stem)
            result.append(metadata)
        return result

    def save(self, name, audio, metadata):
        name = voice_name(name)
        if self.exists(name):
            raise ValueError(f"Voice {name} already exists; choose a new name to preserve it")
        wav_info(audio)
        metadata = dict(metadata)
        clean_text(metadata.get("reference_text"), field="reference_text", limit=1000)
        audio_path = self.directory / f"{name}-{uuid.uuid4().hex}.wav"
        final_path = self.directory / f"{name}.json"
        temporary_path = self.directory / f".{name}-{uuid.uuid4().hex}.tmp"
        metadata.update(name=name, audio_file=audio_path.name)
        try:
            audio_path.write_bytes(audio)
            temporary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary_path, final_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            audio_path.unlink(missing_ok=True)
            raise
        return metadata

