"""Shrink the Base model's GPU footprint without touching its CUDA-graph decode path.

Three independent changes, all applied after loading and before graph capture:

- The talker's text embedding (151936 x 2048, about 0.6 GB) is only read while a
  prompt is built, so it stays in host RAM and only looked-up rows reach the GPU.
- The speech tokenizer's encoder is only needed to turn reference audio into a
  voice prompt, which the backend caches; it waits in host RAM between uses.
- Optionally, the talker's transformer linears keep int8 weights with one scale
  per output row (about 1.3 GB saved on the 1.7B model).

Host-resident tensors are plain attributes or live in modules flagged
host_resident, so idle parking (which moves CUDA parameters and buffers) and the
CUDA parameter check leave them alone.
"""

import logging

import torch
from torch import nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

log = logging.getLogger(__name__)


def module_bytes(module):
    return sum(tensor.nbytes for tensor in list(module.parameters()) + list(module.buffers()))


def _move(module, device):
    # Requests run under inference_mode; parameters must stay normal tensors.
    with torch.inference_mode(False), torch.no_grad():
        module.to(device)


class HostEmbedding(nn.Module):
    """An embedding table in host RAM; only the looked-up rows are copied to the GPU."""

    def __init__(self, embedding, device):
        super().__init__()
        self.num_embeddings, self.embedding_dim = embedding.weight.shape
        self.padding_idx = embedding.padding_idx
        self.target_device = torch.device(device)
        # A plain attribute, not a parameter or buffer: Module.to() and parking skip it.
        self.host_weight = embedding.weight.detach().to("cpu")

    def forward(self, ids):
        rows = F.embedding(ids.to("cpu"), self.host_weight, self.padding_idx)
        return rows.to(self.target_device)


def offload_text_embedding(talker_model, device):
    """Replace talker_model.text_embedding with a HostEmbedding; returns GPU bytes freed."""
    embedding = talker_model.text_embedding
    if isinstance(embedding, HostEmbedding):
        return 0
    freed = module_bytes(embedding)
    talker_model.text_embedding = HostEmbedding(embedding, device)
    return freed


def offload_encoder(tokenizer_model, device):
    """Keep the speech tokenizer's encoder in host RAM, on the GPU only while encoding.

    The backend caches each reference's voice prompt, so this runs once per voice.
    Returns GPU bytes freed.
    """
    encoder = tokenizer_model.encoder
    if getattr(encoder, "host_resident", False):
        return 0
    freed = module_bytes(encoder)
    _move(encoder, "cpu")
    encoder.host_resident = True
    encode = encoder.encode

    def encode_on_device(*args, **kwargs):
        _move(encoder, device)
        try:
            return encode(*args, **kwargs)
        finally:
            _move(encoder, "cpu")
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()

    encoder.encode = encode_on_device
    return freed


if triton is not None:
    @triton.jit
    def _int8_gemv(x_ptr, q_ptr, s_ptr, y_ptr, N, K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """y = (q @ x) * s for one activation row; each program owns BLOCK_N output rows."""
        rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for start in range(0, K, BLOCK_K):
            cols = start + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + cols).to(tl.float32)
            q = tl.load(q_ptr + rows[:, None] * K + cols[None, :], mask=rows[:, None] < N, other=0)
            acc += q.to(tl.float32) * x[None, :]
        scale = tl.load(s_ptr + rows, mask=rows < N, other=0).to(tl.float32)
        tl.store(y_ptr + rows, (tl.sum(acc, axis=1) * scale).to(y_ptr.dtype.element_ty), mask=rows < N)


# Fastest of a sweep on an RTX 5090 for the talker's shapes (K = 2048 or 6144).
GEMV_BLOCK_N, GEMV_BLOCK_K, GEMV_WARPS = 4, 512, 8


def int8_gemv(x, qweight, scale):
    n, k = qweight.shape
    out = torch.empty(n, device=x.device, dtype=x.dtype)
    _int8_gemv[(triton.cdiv(n, GEMV_BLOCK_N),)](x, qweight, scale, out, n, k, BLOCK_N=GEMV_BLOCK_N,
                                                 BLOCK_K=GEMV_BLOCK_K, num_warps=GEMV_WARPS)
    return out


class Int8Linear(nn.Module):
    """Weight-only int8 linear layer with a symmetric scale per output row.

    A single activation row (the talker's CUDA-graphed decode step) uses a
    Triton matrix-vector kernel that reads the int8 weight directly, about twice
    as fast as bf16 there. Longer inputs, such as a prefill, dequantize the
    weight for a regular GEMM.
    """

    def __init__(self, linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        weight = linear.weight.detach()
        amax = weight.float().abs().amax(dim=1).clamp_min(1e-12)
        # Round the scale to the activation dtype first, so int8 rounding is the only error.
        scale = (amax / 127).to(weight.dtype)
        qweight = torch.round(weight.float() / scale.float().unsqueeze(1)).clamp_(-127, 127).to(torch.int8)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.bias = linear.bias
        self.gemv = False

    def dequantized(self, dtype):
        return self.qweight.to(dtype) * self.scale.to(dtype).unsqueeze(1)

    def forward(self, x):
        rows = x.reshape(-1, self.in_features)
        if self.gemv and rows.shape[0] == 1:
            out = int8_gemv(rows.reshape(-1).contiguous(), self.qweight, self.scale).unsqueeze(0)
        else:
            out = F.linear(rows, self.qweight.to(rows.dtype)) * self.scale.to(rows.dtype)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x.shape[:-1], self.out_features)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, gemv={self.gemv}"


def _gemv_ok(layer):
    """Compile and check the Triton kernel for this layer's shape; False where it cannot run."""
    if triton is None or layer.qweight.device.type != "cuda" or layer.in_features % GEMV_BLOCK_K:
        return False
    try:
        with torch.no_grad():
            x = torch.randn(layer.in_features, device=layer.qweight.device, dtype=layer.scale.dtype)
            actual = int8_gemv(x, layer.qweight, layer.scale).float()
            reference = F.linear(x.float(), layer.dequantized(torch.float32))
        return (actual - reference).abs().max().item() <= 0.01 * reference.abs().max().item() + 1e-3
    except Exception:
        log.warning("Triton int8 kernel unavailable; using dequantize + GEMM", exc_info=True)
        return False


def quantize_linears(module):
    """Swap every nn.Linear under module for an Int8Linear, one layer at a time.

    Returns (layer count, GPU bytes freed, whether decode uses the Triton kernel).
    """
    # Hold names, not layers, so each bf16 weight is freed as soon as it is replaced.
    targets = [(parent, name) for parent in module.modules()
               for name, child in parent.named_children() if type(child) is nn.Linear]
    freed = 0
    checked = {}
    for parent, name in targets:
        linear = getattr(parent, name)
        layer = Int8Linear(linear)
        freed += module_bytes(linear) - module_bytes(layer)
        shape = tuple(layer.qweight.shape)
        if shape not in checked:
            # Also compiles the kernel now, never during CUDA graph capture.
            checked[shape] = _gemv_ok(layer)
        layer.gemv = checked[shape]
        setattr(parent, name, layer)
        del linear
    return len(targets), freed, bool(checked) and all(checked.values())
