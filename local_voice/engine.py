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

# The 12 Hz codec produces 12.5 frames per second of audio.
CODEC_FRAME_RATE = 12.5
# Static-context positions per voice prompt besides its reference audio frames.
PROMPT_OVERHEAD_FRAMES = 16
# Positions kept free at the end of the static context.
CONTEXT_MARGIN_FRAMES = 32
# Upper bound on codec frames per character of English text, from measurement.
FRAMES_PER_CHARACTER = 1.25
MAX_SEGMENT_CHARACTERS = 800
MIN_SEGMENT_CHARACTERS = 80
WEIGHT_QUANT_MODES = ("none", "int8")


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


def _aligned(nbytes, alignment=4096):
    return -(-nbytes // alignment) * alignment


def torch_components(wrapper):
    """The torch modules holding a wrapper's weights: the TTS model and its audio tokenizer."""
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
    return components


def host_resident_parameters(component):
    """Parameters of submodules deliberately kept in host RAM (see footprint.py)."""
    modules = getattr(component, "modules", None)
    if modules is None:
        return set()
    return {id(parameter) for module in modules() if getattr(module, "host_resident", False)
            for parameter in module.parameters()}


def verify_cuda_parameters(wrapper):
    """Check actual model and audio-tokenizer tensors, not just a device label."""
    components = torch_components(wrapper)
    count = 0
    for component in components:
        exempt = host_resident_parameters(component)
        for name, parameter in component.named_parameters():
            if id(parameter) in exempt:
                continue
            count += 1
            if parameter.device.type != "cuda":
                raise RuntimeError(f"Model tensor {name} is on {parameter.device}; CUDA is required")
    if not count:
        raise RuntimeError("Could not verify model parameters on CUDA")
    return count


def tensor_slots(components, device_type):
    """Parameter and buffer slots on one device type, as (table, name, is_parameter)."""
    slots = []
    seen = set()
    for component in components:
        for module in component.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            for table, is_parameter in ((module._parameters, True), (module._buffers, False)):
                for name, tensor in table.items():
                    if tensor is not None and tensor.device.type == device_type:
                        slots.append((table, name, is_parameter))
    return slots


def unique_tensors(slots):
    """Slots with the tensor each holds and a key shared by tied tensors."""
    entries = []
    seen = set()
    for table, name, is_parameter in slots:
        tensor = table[name]
        if is_parameter:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
        key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(),
               tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        entries.append((table, name, is_parameter, tensor, key))
    return entries


def move_tensors(slots, convert):
    """Re-home parameters and buffers in place; tied tensors keep sharing one copy.

    Parameters keep their identity (only .data changes), so modules, hooks and
    tied weights still reference the same objects. Buffers are replaced in
    their module's table.
    """
    converted = {}
    for table, name, is_parameter, tensor, key in unique_tensors(slots):
        if key not in converted:
            converted[key] = convert(tensor)
        if is_parameter:
            tensor.data = converted[key]
        else:
            table[name] = converted[key]
    return len(converted)


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
        # Weight storage for the talker's transformer linears; see footprint.py.
        self.weight_quant = os.environ.get("TTS_WEIGHT_QUANT", "none").strip().lower() or "none"
        if self.weight_quant not in WEIGHT_QUANT_MODES:
            raise ValueError(f"TTS_WEIGHT_QUANT must be one of {', '.join(WEIGHT_QUANT_MODES)}")
        # Static KV-cache positions: reference prompt plus one text segment's speech.
        self.max_seq_len = int(os.environ.get("TTS_MAX_SEQ_LEN", "1024"))
        if not 512 <= self.max_seq_len <= 4096:
            raise ValueError("TTS_MAX_SEQ_LEN must be between 512 and 4096")
        self.footprint = None
        self.prompt_frames = {}
        self.warmed_voices = set()
        # Idle unload: park the Base model's weights in pinned host RAM and drop
        # its CUDA graphs, then restore them when speech is next requested.
        self.idle_unload_seconds = float(os.environ.get("TTS_IDLE_UNLOAD_SECONDS", "0"))
        if not 0 <= self.idle_unload_seconds <= 86400:
            raise ValueError("TTS_IDLE_UNLOAD_SECONDS must be between 0 (never) and 86400")
        self.residency = None
        self.parked_slots = None
        self.parked_views = None
        self.graph_settings = None
        self.host_buffer = None
        self.host_pinned = False
        self.host_cursor = 0
        self.pending = 0
        self.last_activity = time.monotonic()
        self.keep_resident_until = 0.0
        self.wake_thread = None
        self.idle_thread = None
        self.wake_count = 0
        self.park_count = 0
        self.last_wake = None
        self.last_park = None

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
                if self.idle_unload_seconds:
                    # Page-locking host memory takes seconds; do it once at startup
                    # rather than in the first park, when a request might be waiting.
                    self._reserve_host(self._park_bytes(tensor_slots(torch_components(self.model), "cuda")))
            with self.state_lock:
                self.state = "ready"
                self.last_activity = time.monotonic()
            log.info("Ready on %s; default voice: %s", self.gpu, name)
            if self.idle_unload_seconds:
                log.info("Idle unload enabled: parking the model after %gs without requests", self.idle_unload_seconds)
                self.idle_thread = threading.Thread(target=self._idle_monitor, daemon=True, name="idle-unload")
                self.idle_thread.start()
        except Exception as exc:
            log.exception("Engine initialization failed")
            with self.state_lock:
                self.state = "failed"
                self.error = str(exc)

    def health(self):
        reserved = None
        if self.gpu:
            try:
                reserved = round(self.torch.cuda.memory_reserved() / 1024**2, 1)
            except Exception:
                pass
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
                "weight_quant": self.weight_quant,
                "max_seq_len": self.max_seq_len,
                # GPU bytes moved to host RAM or saved by quantization at load.
                "footprint": self.footprint and dict(self.footprint),
                "cuda_graphs": bool(self.model_kind == "base" and getattr(self.model, "_warmed_up", False)),
                # resident, parking, parked (weights in host RAM) or waking.
                # Requests made while parked wait for the wake and then succeed.
                "residency": self.residency,
                "parked": self.residency == "parked",
                "idle_unload_seconds": self.idle_unload_seconds,
                "idle_seconds": round(time.monotonic() - self.last_activity, 1),
                "active_requests": self.pending,
                "keep_resident_seconds": round(max(0.0, self.keep_resident_until - time.monotonic()), 1),
                "torch_reserved_mb": reserved,
                "wake_count": self.wake_count,
                "park_count": self.park_count,
                "last_wake_seconds": self.last_wake and self.last_wake["seconds"],
                # Copies: a background wake adds warm_seconds after publishing.
                "last_wake": self.last_wake and dict(self.last_wake),
                "last_park": self.last_park and dict(self.last_park),
            }

    @contextmanager
    def _gpu_session(self, event):
        """Hold the GPU for one request. Pending requests block idle unload;
        the idle timer restarts when the request releases the GPU."""
        with self.state_lock:
            self.pending += 1
            self.last_activity = time.monotonic()
        try:
            with cancellable_lock(self.gpu_lock, event):
                yield
        finally:
            with self.state_lock:
                self.pending -= 1
                self.last_activity = time.monotonic()

    def _idle_expired(self):
        with self.state_lock:
            now = time.monotonic()
            return (self.idle_unload_seconds > 0 and self.state == "ready"
                    and self.residency == "resident" and self.model_kind == "base"
                    and self.pending == 0 and now >= self.keep_resident_until
                    and now - self.last_activity >= self.idle_unload_seconds)

    def park_if_idle(self):
        """Park the model if it has been idle long enough; never waits for the GPU."""
        if not self._idle_expired() or not self.gpu_lock.acquire(blocking=False):
            return False
        try:
            # Re-check under the GPU lock: a request may have just arrived.
            if not self._idle_expired():
                return False
            self._park("idle")
            return True
        finally:
            self.gpu_lock.release()

    def _idle_monitor(self):
        interval = min(5.0, max(0.25, self.idle_unload_seconds / 10))
        while True:
            time.sleep(interval)
            try:
                self.park_if_idle()
            except Exception:
                log.exception("Idle unload failed")

    def request_wake(self, keep_resident_seconds=0):
        """Start restoring a parked model in the background and return at once.

        Clients call this when speech is likely soon (for example, as a prompt is
        submitted) so the restore overlaps their own work. It also restarts the
        idle timer, and can hold the model resident for up to an hour.
        """
        if isinstance(keep_resident_seconds, bool) or not isinstance(keep_resident_seconds, (int, float)):
            raise ValueError("keep_resident_seconds must be a number")
        if not 0 <= keep_resident_seconds <= 3600:
            raise ValueError("keep_resident_seconds must be between 0 and 3600")
        with self.state_lock:
            now = time.monotonic()
            self.last_activity = now
            self.keep_resident_until = max(self.keep_resident_until, now + keep_resident_seconds)
            started = (self.state == "ready" and self.residency in ("parked", "parking")
                       and not (self.wake_thread and self.wake_thread.is_alive()))
            if started:
                self.wake_thread = threading.Thread(target=self._background_wake, daemon=True, name="model-wake")
                self.wake_thread.start()
        return dict(self.health(), wake_started=started)

    def _background_wake(self):
        try:
            with self._gpu_session(threading.Event()):
                if self.state == "ready":
                    wakes = self.wake_count
                    self._load_model("base", reason="wake request")
                    with self.state_lock:
                        waiting = self.pending > 1
                    if self.wake_count != wakes and not waiting:
                        # Nobody is waiting yet, so also pay the first generation's
                        # one-time setup (about 0.2 s) now rather than in the next
                        # request. One audio chunk covers it.
                        started = time.perf_counter()
                        self._warm_voice(self.config["name"], force=True, first_chunk_only=True)
                        with self.state_lock:
                            self.last_wake["warm_seconds"] = round(time.perf_counter() - started, 3)
        except Exception:
            log.exception("Background wake failed; the next request will retry")

    def _park(self, reason):
        """Move Base weights to host RAM and free CUDA graphs. Caller holds gpu_lock."""
        torch = self.torch
        started = time.perf_counter()
        reserved_before = torch.cuda.memory_reserved()
        model = self.model
        graph = model.predictor_graph
        self.graph_settings = {
            "max_seq_len": model.talker_graph.max_seq_len,
            "predictor": {"do_sample": graph.do_sample, "top_k": graph.top_k,
                          "top_p": graph.top_p, "temperature": graph.temperature},
        }
        del graph
        slots = tensor_slots(torch_components(model), "cuda")
        self._reserve_host(self._park_bytes(slots))
        with self.state_lock:
            self.residency = "parking"
        self.parked_slots = slots
        try:
            # Graph replays read fixed addresses, so the graphs, their memory
            # pools and static KV caches go; wake captures fresh ones.
            model.predictor_graph = None
            model.talker_graph = None
            model._warmed_up = False
            self.host_cursor = 0
            count = move_tensors(slots, self._to_host)
            torch.cuda.synchronize()
            # Host copies stay valid through a wake, so a failed wake can return here.
            self.parked_views = [(table, name, is_parameter, table[name].data if is_parameter else table[name])
                                 for table, name, is_parameter in slots]
        except Exception:
            log.exception("Parking failed; restoring the model to the GPU")
            self._wake("park rollback")
            return
        gc.collect()
        # Each capture runs on a fresh side stream, and PyTorch keeps a cuBLAS
        # workspace per stream: about 17 MB more per wake unless released.
        # Safe here: the graphs that used them are gone and the GPU is idle.
        clear_workspaces = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        if clear_workspaces:
            clear_workspaces()
        torch.cuda.empty_cache()
        seconds = time.perf_counter() - started
        reserved_after = torch.cuda.memory_reserved()
        with self.state_lock:
            self.residency = "parked"
            self.park_count += 1
            self.last_park = {
                "reason": reason, "seconds": round(seconds, 3), "tensors": count,
                "pinned_host_memory": self.host_pinned,
                "host_buffer_mb": round(self.host_buffer.numel() / 1024**2, 1),
                "torch_reserved_mb_before": round(reserved_before / 1024**2, 1),
                "torch_reserved_mb_after": round(reserved_after / 1024**2, 1),
                "at": datetime.now(timezone.utc).isoformat(),
            }
        log.info("Parked %s model in host RAM (%s) in %.2fs: PyTorch reserved %.0f -> %.0f MB",
                 self.model_kind, reason, seconds, reserved_before / 1024**2, reserved_after / 1024**2)

    @staticmethod
    def _park_bytes(slots):
        return sum(_aligned(entry[3].nbytes) for entry in unique_tensors(slots))

    def _reserve_host(self, nbytes):
        """One page-locked host buffer for parked weights, kept for reuse.

        PyTorch's pinned allocator takes seconds per park on WSL2 (one
        cudaHostAlloc per tensor, each rounded up to a power of two), and every
        CUDA graph capture frees its cache again, adding most of a second to
        each wake. A single registered buffer avoids both.
        """
        torch = self.torch
        if self.host_buffer is not None and self.host_buffer.numel() >= nbytes:
            return
        self._release_host()
        buffer = torch.empty(nbytes, dtype=torch.uint8)
        result = []
        # CUDA's last-error state is per thread: a failed registration in a
        # helper thread cannot surface later as an unrelated kernel error.
        helper = threading.Thread(target=lambda: result.append(
            torch.cuda.cudart().cudaHostRegister(buffer.data_ptr(), nbytes, 0)), name="host-register")
        helper.start()
        helper.join()
        self.host_pinned = bool(result) and int(result[0]) == 0
        if not self.host_pinned:
            # Pageable memory still avoids a disk load; it only copies slower.
            log.warning("Could not page-lock %.0f MB of host memory (%s); parking in pageable memory",
                        nbytes / 1024**2, result[0] if result else "no result")
        self.host_buffer = buffer

    def _release_host(self):
        if self.host_buffer is not None and self.host_pinned:
            self.torch.cuda.cudart().cudaHostUnregister(self.host_buffer.data_ptr())
        self.host_buffer = None
        self.host_pinned = False

    def _to_host(self, tensor):
        """Copy one tensor into the next slice of the host buffer."""
        if not tensor.is_contiguous():
            return tensor.to("cpu")
        start = self.host_cursor
        self.host_cursor += _aligned(tensor.nbytes)
        host = self.host_buffer[start:start + tensor.nbytes].view(tensor.dtype).view(tensor.shape)
        host.copy_(tensor, non_blocking=self.host_pinned)
        return host

    def _wake(self, reason):
        """Restore parked weights and recapture CUDA graphs. Caller holds gpu_lock."""
        from faster_qwen3_tts.predictor_graph import PredictorGraph
        from faster_qwen3_tts.talker_graph import TalkerGraph

        torch = self.torch
        model = self.model
        started = time.perf_counter()
        with self.state_lock:
            self.residency = "waking"
        try:
            move_tensors(self.parked_slots, lambda tensor: tensor.to(model.device, non_blocking=True))
            torch.cuda.synchronize()
            weights_seconds = time.perf_counter() - started
            # Rebuild exactly as FasterQwen3TTS.from_pretrained does.
            qwen = model.model.model
            talker = qwen.talker
            talker_config = qwen.config.talker_config
            model.predictor_graph = PredictorGraph(
                talker.code_predictor, talker.code_predictor.model.config, talker_config.hidden_size,
                device=model.device, dtype=model.dtype, **self.graph_settings["predictor"],
            )
            model.talker_graph = TalkerGraph(
                talker.model, talker_config, device=model.device, dtype=model.dtype,
                max_seq_len=self.graph_settings["max_seq_len"],
            )
            model.warmup(prefill_len=100)
            if not model._warmed_up:
                raise RuntimeError("CUDA graph warmup did not complete")
            verify_cuda_parameters(model)
        except Exception:
            log.exception("Waking the model failed; returning it to host RAM so a later request can retry")
            try:
                model.predictor_graph = model.talker_graph = None
                model._warmed_up = False
                for table, name, is_parameter, host in self.parked_views:
                    if is_parameter:
                        table[name].data = host
                    else:
                        table[name] = host
                gc.collect()
                torch.cuda.empty_cache()
                with self.state_lock:
                    self.residency = "parked"
            except Exception:
                log.exception("Could not return the model to host RAM")
                with self.state_lock:
                    self.state = "failed"
                    self.error = "The model could not be restored to the GPU; restart the service"
            raise
        self.parked_slots = self.parked_views = None
        seconds = time.perf_counter() - started
        with self.state_lock:
            self.residency = "resident"
            self.wake_count += 1
            self.last_wake = {
                "reason": reason, "seconds": round(seconds, 3),
                "weights_seconds": round(weights_seconds, 3),
                "graph_capture_seconds": round(seconds - weights_seconds, 3),
                "at": datetime.now(timezone.utc).isoformat(),
            }
        log.info("Woke %s model (%s) in %.2fs: weights %.2fs, CUDA graphs %.2fs",
                 self.model_kind, reason, seconds, weights_seconds, seconds - weights_seconds)

    def _require_ready(self):
        if self.state != "ready":
            raise NotReady(self.error or "Models are loading or the default voice is being designed")

    def _load_model(self, kind, reason="request"):
        if self.model_kind == kind and self.model is not None:
            if self.residency == "parked":
                self._wake(reason)
            return
        from huggingface_hub import snapshot_download
        from qwen_tts import Qwen3TTSModel

        if self.model is not None:
            # The host buffer is kept: the Base model has the same size when reloaded.
            self.model = None
            self.parked_slots = self.parked_views = None
            self.prompt_cache.clear()
            self.warmed_voices.clear()
            gc.collect()
            self.torch.cuda.empty_cache()
        with self.state_lock:
            self.residency = None
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
                attn_implementation="sdpa", backend="torch", max_seq_len=self.max_seq_len,
            )
            # Before warmup: the CUDA graphs capture the layers as they will run.
            self._shrink(model)
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
        with self.state_lock:
            self.residency = "resident"

    def _shrink(self, model):
        """Move prompt-only weights to host RAM and optionally quantize the talker."""
        from .footprint import offload_encoder, offload_text_embedding, quantize_linears

        torch = self.torch
        started = time.perf_counter()
        qwen = model.model.model
        result = {
            "text_embedding_host_mb": offload_text_embedding(qwen.talker.model, model.device) / 1024**2,
            "tokenizer_encoder_host_mb": offload_encoder(qwen.speech_tokenizer.model, model.device) / 1024**2,
            "int8_linears": 0, "int8_saved_mb": 0.0, "int8_triton_gemv": False,
        }
        if self.weight_quant == "int8":
            count, saved, gemv = quantize_linears(qwen.talker.model.layers)
            result.update(int8_linears=count, int8_saved_mb=saved / 1024**2, int8_triton_gemv=gemv)
        # The freed blocks sit between the remaining weights, so the allocator
        # cannot return them. A round trip through host RAM repacks the weights.
        slots = tensor_slots(torch_components(model), "cuda")
        move_tensors(slots, lambda tensor: tensor.to("cpu"))
        gc.collect()
        torch.cuda.empty_cache()
        move_tensors(slots, lambda tensor: tensor.to(model.device))
        torch.cuda.synchronize()
        result["torch_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
        result = {key: round(value, 1) if isinstance(value, float) else value for key, value in result.items()}
        result["seconds"] = round(time.perf_counter() - started, 3)
        self.footprint = result
        log.info("Reduced the Base model's GPU footprint: %s", result)

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
            with self._gpu_session(event):
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
            # The reference codes (plus the appended 0.5 s of silence) occupy the
            # start of the static context in every generation with this voice.
            seconds = wav_info(audio_path.read_bytes())["duration_seconds"] + 0.5
            self.prompt_frames[name] = int(seconds * CODEC_FRAME_RATE) + 1 + PROMPT_OVERHEAD_FRAMES
        return self.prompt_cache[name]

    def _segment_limit(self, name):
        """Characters per text segment so one segment's speech fits the static context."""
        self._voice_prompt(name)
        budget = self.max_seq_len - self.prompt_frames[name] - CONTEXT_MARGIN_FRAMES
        limit = min(MAX_SEGMENT_CHARACTERS, int(budget / FRAMES_PER_CHARACTER))
        if limit < MIN_SEGMENT_CHARACTERS:
            raise RuntimeError(f"The reference audio of voice {name} is too long for TTS_MAX_SEQ_LEN={self.max_seq_len}; "
                               "raise TTS_MAX_SEQ_LEN")
        return limit

    def _warm_voice(self, name, force=False, first_chunk_only=False):
        if name in self.warmed_voices and not force:
            return
        log.info("Warming streaming decode and reference voice %s", name)
        with self.torch.inference_mode(), closing(self.model.generate_voice_clone_streaming(
            text="Ready.", language=self.config.get("language", "English"),
            **self._voice_prompt(name), chunk_size=self.chunk_size, max_new_tokens=64,
        )) as chunks:
            for _ in chunks:
                if first_chunk_only:
                    break
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
            with self._gpu_session(event):
                wakes = self.wake_count
                self._load_model("base")
                woke = self.wake_count != wakes
                prompt = self._voice_prompt(name)
                limit = self._segment_limit(name)
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
                segments = sentence_chunks(text, limit=limit)
                context_positions = 0
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
                        if graph is not None:
                            position = graph.cache_position.item()
                            context_positions = max(context_positions, position + 1)
                            if position >= graph.max_seq_len - 2:
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
                    # Static-context positions used by the longest segment, of max_seq_len.
                    "context_positions": context_positions,
                    "max_seq_len": self.max_seq_len,
                    "weight_quant": self.weight_quant,
                    # Consistent with the original API: lower than 1 is faster
                    # than playback. Excludes time paused at yields by a reader.
                    "real_time_factor": round(compute_seconds / duration, 3),
                    "peak_cuda_memory_mb": round(self.torch.cuda.max_memory_allocated() / 1024**2, 1),
                    # first_audio_ms includes this restore when the request found the model parked.
                    "woke_model": woke,
                    "wake_seconds": self.last_wake["seconds"] if woke else 0.0,
                }
                with self.state_lock:
                    self.last_synthesis = metrics
                yield "M", metrics
