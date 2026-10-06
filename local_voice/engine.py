"""GPU-only Qwen VoiceDesign -> persisted reference -> Base speech synthesis."""

from collections import OrderedDict
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import gc
import io
import json
import logging
import os
from pathlib import Path
import threading
import time

from .common import Cancelled, NotReady, clean_text, request_id, sentence_chunks, voice_name, wav_info
from .streaming import FORMAT, MAX_AUDIO, pcm_to_wav
from .voices import VoiceStore

log = logging.getLogger(__name__)


class CancellationRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = {}
        self.cancelled = OrderedDict()

    @contextmanager
    def track(self, identifier):
        identifier = request_id(identifier)
        with self.lock:
            if identifier in self.active:
                raise ValueError("Duplicate active request_id")
            event = threading.Event()
            if identifier in self.cancelled:
                event.set()
            self.active[identifier] = event
        try:
            yield event
        finally:
            with self.lock:
                self.active.pop(identifier, None)

    def cancel(self, identifier):
        identifier = request_id(identifier)
        with self.lock:
            # Remember early cancellations too: cancel can arrive before synthesis.
            self.cancelled[identifier] = True
            while len(self.cancelled) > 256:
                self.cancelled.popitem(last=False)
            event = self.active.get(identifier)
            if event:
                event.set()
            return event is not None


def check_cancelled(event):
    if event.is_set():
        raise Cancelled("Speech request was cancelled")


@contextmanager
def cancellable_lock(lock, event):
    while not lock.acquire(timeout=0.1):
        check_cancelled(event)
    try:
        check_cancelled(event)
        yield
    finally:
        lock.release()


def verify_cuda_parameters(wrapper):
    """Check actual model and audio-tokenizer tensors, not just a device label."""
    model = wrapper
    for _ in range(3):
        if hasattr(model, "named_parameters"):
            break
        model = model.model
    components = [model]
    tokenizer = getattr(model, "speech_tokenizer", None)
    tokenizer_model = getattr(tokenizer, "model", None)
    if tokenizer_model is not None:
        components.append(tokenizer_model)
    count = 0
    for component in components:
        for name, parameter in component.named_parameters():
            count += 1
            if parameter.device.type != "cuda":
                raise RuntimeError(f"Model tensor {name} is on {parameter.device}; CUDA is required")
    if not count:
        raise RuntimeError("Could not verify model parameters on CUDA")
    return count


@contextmanager
def cancellable_generation(wrapper, event):
    """Stop at a decoder forward boundary, even with Qwen 0.1.1's kwargs filter.

    Its outer generate method does not forward arbitrary stopping_criteria to
    the talker. A temporary PyTorch hook reaches the actual decoding loop.
    GPU work is serialized, and the hook is always removed after the request.
    """
    check_cancelled(event)
    talker = wrapper.model.talker

    def before_forward(module, inputs):
        check_cancelled(event)

    handle = talker.register_forward_pre_hook(before_forward)
    try:
        yield
        check_cancelled(event)
    finally:
        handle.remove()


class QwenEngine:
    def __init__(self, config_path=None, data_dir=None):
        config_path = config_path or os.environ.get("VOICE_CONFIG", "config/voice.json")
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        voice_name(self.config["name"])
        clean_text(self.config["description"], field="description", limit=2000)
        self.store = VoiceStore(data_dir or os.environ.get("VOICE_DATA_DIR", "/data/voices"))
        self.gpu_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.cancellations = CancellationRegistry()
        self.state = "loading"
        self.error = None
        self.gpu = None
        self.capability = None
        self.model = None
        self.model_kind = None
        self.model_snapshot = None
        self.prompt_cache = {}
        self.last_synthesis = None
        self.chunk_size = int(os.environ.get("TTS_CHUNK_FRAMES", "4"))
        if not 1 <= self.chunk_size <= 12:
            raise ValueError("TTS_CHUNK_FRAMES must be between 1 and 12")
        self.warmed_voices = set()

    def initialize(self):
        try:
            import torch

            self.torch = torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Start this service with Docker Compose GPU access.")
            self.gpu = torch.cuda.get_device_name(0)
            expected = os.environ.get("EXPECTED_GPU_NAME", "")
            if expected and expected.lower() not in self.gpu.lower():
                raise RuntimeError(f"Expected {expected}, but Docker exposed {self.gpu}")
            self.capability = list(torch.cuda.get_device_capability(0))
            # Exercise a real CUDA kernel so an incompatible wheel fails at startup.
            probe = torch.ones((64, 64), device="cuda:0", dtype=torch.bfloat16)
            if (probe @ probe)[0, 0].item() != 64:
                raise RuntimeError("CUDA matrix multiplication check failed")
            torch.cuda.synchronize()
            del probe
            with self.gpu_lock:
                event = threading.Event()
                name = self.config["name"]
                if not self.store.exists(name):
                    self._design(name, self.config["description"], self.config["reference_text"], event)
                else:
                    saved, _ = self.store.load(name)
                    if saved["description"] != self.config["description"]:
                        raise RuntimeError("Default voice description changed. Choose a new voice name to redesign it.")
                self._load_model("base")
                self._warm_voice(name)
            with self.state_lock:
                self.state = "ready"
            log.info("Ready on %s; default voice: %s", self.gpu, name)
        except Exception as exc:
            log.exception("Engine initialization failed")
            with self.state_lock:
                self.state = "failed"
                self.error = str(exc)

    def health(self):
        with self.state_lock:
            return {
                "status": self.state,
                "error": self.error,
                "gpu": self.gpu,
                "compute_capability": self.capability,
                "device": "cuda:0" if self.gpu else None,
                "default_voice": self.config["name"],
                "loaded_model": self.model_kind,
                "model_snapshot": self.model_snapshot,
                "design_model": self.config["design_model"],
                "base_model": self.config["base_model"],
                "last_synthesis": self.last_synthesis,
                "backend": "faster-qwen3-tts",
                "streaming_protocol": 1,
                "chunk_frames": self.chunk_size,
                "cuda_graphs": bool(self.model_kind == "base" and getattr(self.model, "_warmed_up", False)),
            }

    def _require_ready(self):
        if self.state != "ready":
            raise NotReady(self.error or "Models are loading or the default voice is being designed")

    def _load_model(self, kind):
        if self.model_kind == kind and self.model is not None:
            return
        from huggingface_hub import snapshot_download
        from qwen_tts import Qwen3TTSModel

        if self.model is not None:
            self.model = None
            self.prompt_cache.clear()
            self.warmed_voices.clear()
            gc.collect()
            self.torch.cuda.empty_cache()
        self.model_kind = None
        model_id = self.config[f"{kind}_model"]
        log.info("Loading %s on CUDA", model_id)
        # Resolve once to a complete local snapshot; record the revision in metadata.
        model_path = snapshot_download(
            repo_id=model_id,
            revision=self.config.get(f"{kind}_revision", "main"),
            local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
        )
        if kind == "base":
            from faster_qwen3_tts import FasterQwen3TTS

            model = FasterQwen3TTS.from_pretrained(
                model_path, device="cuda:0", dtype=self.torch.bfloat16,
                attn_implementation="sdpa", backend="torch", max_seq_len=2048,
            )
            model.warmup(prefill_len=100)
            if not model._warmed_up:
                raise RuntimeError("CUDA graph warmup did not complete")
        else:
            # Voice design is an occasional offline operation; preserve its API.
            model = Qwen3TTSModel.from_pretrained(
                model_path, device_map="cuda:0", torch_dtype=self.torch.bfloat16,
                attn_implementation="sdpa",
            )
        verify_cuda_parameters(model)
        self.model = model
        self.model_kind = kind
        self.model_snapshot = Path(model_path).name

    def _wav(self, waveform, sample_rate):
        import numpy as np
        import soundfile as sf

        data = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not data.size or not np.isfinite(data).all():
            raise RuntimeError("The model returned invalid audio")
        if float(np.max(np.abs(data))) < 0.0001:
            raise RuntimeError("The model returned silent audio")
        buffer = io.BytesIO()
        sf.write(buffer, data, sample_rate, format="WAV", subtype="PCM_16")
        result = buffer.getvalue()
        wav_info(result)
        return result

    def _design(self, name, description, reference_text, event):
        if self.store.exists(name):
            raise ValueError(f"Voice {name} already exists; choose a new name")
        self._load_model("design")
        check_cancelled(event)
        self.torch.manual_seed(int(self.config.get("seed", 42)))
        with self.torch.inference_mode(), cancellable_generation(self.model, event):
            waveforms, rate = self.model.generate_voice_design(
                text=reference_text,
                language=self.config.get("language", "English"),
                instruct=description,
                max_new_tokens=2048,
            )
        check_cancelled(event)
        self.torch.cuda.synchronize()
        return self.store.save(name, self._wav(waveforms[0], rate), {
            "description": description,
            "reference_text": reference_text,
            "language": self.config.get("language", "English"),
            "model": self.config["design_model"],
            "model_revision": self.model_snapshot,
            "seed": int(self.config.get("seed", 42)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    def design(self, name, description, reference_text, identifier):
        self._require_ready()
        name = voice_name(name)
        description = clean_text(description, field="description", limit=2000)
        reference_text = clean_text(reference_text, field="reference_text", limit=1000)
        with self.cancellations.track(identifier) as event:
            with cancellable_lock(self.gpu_lock, event):
                result = self._design(name, description, reference_text, event)
                self._load_model("base")
                self._warm_voice(name)
                self._warm_voice(self.config["name"])
                return result

    def _voice_prompt(self, name):
        if name not in self.prompt_cache:
            metadata, audio_path = self.store.load(name)
            # FasterQwen caches extracted embeddings and reference codec tokens.
            self.prompt_cache[name] = {
                "ref_audio": str(audio_path), "ref_text": metadata["reference_text"],
                "xvec_only": False, "append_silence": True,
            }
        return self.prompt_cache[name]

    def _warm_voice(self, name):
        if name in self.warmed_voices:
            return
        log.info("Warming streaming decode and reference voice %s", name)
        with self.torch.inference_mode(), closing(self.model.generate_voice_clone_streaming(
            text="Ready.", language=self.config.get("language", "English"),
            **self._voice_prompt(name), chunk_size=self.chunk_size, max_new_tokens=64,
        )) as chunks:
            for _ in chunks:
                pass
        self.torch.cuda.synchronize()
        self.warmed_voices.add(name)

    @staticmethod
    def _pcm(waveform):
        import numpy as np

        data = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not np.isfinite(data).all():
            raise RuntimeError("The model returned non-finite audio")
        if not data.size:
            return b"", 0.0
        peak = float(np.max(np.abs(data)))
        pcm = np.rint(np.clip(data, -1, 1) * 32767).astype("<i2").tobytes()
        return pcm, peak

    def synthesize(self, text, name, identifier):
        """Buffered compatibility endpoint, using the same accelerated decoder."""
        parts = []
        metrics = None
        for kind, value in self.stream_synthesize(text, name, identifier):
            if kind == "A":
                parts.append(value)
            elif kind == "M":
                metrics = value
        return pcm_to_wav(b"".join(parts)), metrics

    def stream_synthesize(self, text, name, identifier):
        self._require_ready()
        text = clean_text(text)
        name = voice_name(name)
        started = time.perf_counter()
        with self.cancellations.track(identifier) as event:
            with cancellable_lock(self.gpu_lock, event):
                self._load_model("base")
                prompt = self._voice_prompt(name)
                check_cancelled(event)
                self.torch.cuda.reset_peak_memory_stats()
                yield "H", dict(FORMAT)
                audio_bytes = 0
                audio_chunks = 0
                peak = 0.0
                compute_seconds = 0.0
                first_audio_ms = None
                # Split only to bound the model's static context. Playback keeps
                # draining one continuous buffer while the next segment generates.
                segments = sentence_chunks(text, limit=800)
                for segment in segments:
                    check_cancelled(event)
                    with self.torch.inference_mode(), closing(self.model.generate_voice_clone_streaming(
                        text=segment, language=self.config.get("language", "English"),
                        **prompt, chunk_size=self.chunk_size, max_new_tokens=1536,
                    )) as chunks:
                        last_timing = {}
                        while True:
                            check_cancelled(event)
                            tick = time.perf_counter()
                            try:
                                waveform, rate, last_timing = next(chunks)
                            except StopIteration:
                                compute_seconds += time.perf_counter() - tick
                                break
                            check_cancelled(event)
                            if rate != FORMAT["sample_rate"]:
                                raise RuntimeError(f"Unexpected model sample rate: {rate}")
                            pcm, chunk_peak = self._pcm(waveform)
                            compute_seconds += time.perf_counter() - tick
                            peak = max(peak, chunk_peak)
                            if not pcm:
                                continue
                            audio_bytes += len(pcm)
                            audio_chunks += 1
                            if audio_bytes > MAX_AUDIO:
                                raise RuntimeError("Generated audio exceeds the size limit")
                            if first_audio_ms is None:
                                first_audio_ms = (time.perf_counter() - started) * 1000
                            yield "A", pcm
                        if last_timing.get("total_steps_so_far", 0) >= 1536:
                            raise RuntimeError("Speech reached its generation limit; use a shorter request")
                        # The pinned fast decoder can also stop when its static
                        # cache fills. It does not expose a finish reason, so
                        # inspect its last replay position instead of declaring
                        # a potentially truncated utterance completed.
                        graph = getattr(self.model, "talker_graph", None)
                        if graph is not None and graph.cache_position.item() >= graph.max_seq_len - 2:
                            raise RuntimeError("Speech reached its model context limit; use shorter text or reference audio")
                check_cancelled(event)
                if not audio_bytes or peak < 0.0001:
                    raise RuntimeError("The model returned empty or silent audio")
                self.torch.cuda.synchronize()
                duration = audio_bytes / (FORMAT["sample_rate"] * 2)
                metrics = {
                    "gpu": self.gpu,
                    "device": "cuda:0",
                    "model": self.config["base_model"],
                    "model_revision": self.model_snapshot,
                    "voice": name,
                    "cuda_parameters_verified": verify_cuda_parameters(self.model),
                    "backend": "faster-qwen3-tts",
                    "cuda_graphs": bool(self.model._warmed_up),
                    "streaming": True,
                    "first_audio_ms": round(first_audio_ms, 1),
                    "synthesis_seconds": round(compute_seconds, 3),
                    "stream_elapsed_seconds": round(time.perf_counter() - started, 3),
                    "audio_seconds": round(duration, 3),
                    "audio_bytes": audio_bytes,
                    "audio_chunks": audio_chunks,
                    "text_segments": len(segments),
                    # Consistent with the original API: lower than 1 is faster
                    # than playback. Excludes time paused at yields by a reader.
                    "real_time_factor": round(compute_seconds / duration, 3),
                    "peak_cuda_memory_mb": round(self.torch.cuda.max_memory_allocated() / 1024**2, 1),
                }
                with self.state_lock:
                    self.last_synthesis = metrics
                yield "M", metrics
