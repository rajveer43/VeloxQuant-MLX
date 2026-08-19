"""Unprivileged run metrics: throughput, peak memory, and derived KV traffic.

What this module measures
-------------------------
* ``wall_s`` / ``prefill_s`` / ``decode_s`` -- real elapsed time, with
  ``mx.eval()`` forced before every timer stop. MLX is lazy: an unsynchronised
  timer measures graph *construction*, not execution, which is the easiest way
  to produce a wrong-but-plausible throughput number here.
* ``peak_memory_mb`` -- via ``mx.get_peak_memory()``, reset per arm.

What this module does NOT measure
----------------------------------
* ``kv_bytes_per_token`` is **DERIVED**, not measured. It is computed from
  cache geometry (layers x kv-heads x head-dim x bit-width, capped by budget).
  MLX exposes no bytes-moved counter, so no measured bandwidth figure exists;
  conflating this derived number with a measured one would be dishonest.
* Energy fields are filled in only when a privileged
  :class:`~veloxquant_mlx.profiling.power_sampler.PowerSampler` supplies them.
  Without root they stay ``None`` -- never ``0.0``, which would silently
  propagate as a fabricated measurement.

J/token here is whatever the sampler reported divided by decode tokens; it is a
sampled estimate, not exact.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx

# fp16/bf16 KV entries are 2 bytes; group scale+zero are stored fp16 (2B each),
# matching the accounting already used by KIVIKVCache._account_bytes.
_FP16_BYTES = 2
_GROUP_PARAM_BYTES = 4  # scale (fp16) + zero (fp16) per group


def _peak_mb() -> float:
    """Peak GPU memory in MiB. Prefers the non-deprecated MLX spelling."""
    try:
        return float(mx.get_peak_memory()) / (1024**2)
    except Exception:
        try:
            return float(mx.metal.get_peak_memory()) / (1024**2)
        except Exception:
            return float("nan")


def _reset_peak() -> None:
    try:
        mx.reset_peak_memory()
    except Exception:
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass


def _eviction_budget(config: Any) -> int | None:
    """Return the per-layer token budget for eviction methods, else ``None``.

    Each eviction method names its budget field separately in ``KVCacheConfig``
    (``qfilters_budget``, ``h2o_budget``, ...), so this maps method -> field
    rather than guessing a common attribute name.
    """
    budget_fields = {
        "qfilters": "qfilters_budget",
        "h2o": "h2o_budget",
        "tova": "tova_budget",
        "snapkv": "snap_budget",
        "pyramidkv": "pyramid_budget",
        "squeeze": "squeeze_budget",
        "chunkkv": "chunkkv_budget",
        "cam": "cam_budget",
        "knorm": "knorm_budget",
        "keyformer": "keyformer_budget",
        "morphkv": "morphkv_budget",
        "kvzip": "kvzip_budget",
        "curdkv": "curdkv_budget",
        "nestedkv": "nestedkv_budget",
    }
    field = budget_fields.get(getattr(config, "method", None))
    if field is None:
        return None
    value = getattr(config, field, None)
    return int(value) if value is not None else None


def kv_bytes_per_token(
    config: Any,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
) -> int:
    """Bytes of KV cache read per decode step. **DERIVED, NOT MEASURED.**

    This is an analytical model of cache geometry, not an observation of DRAM
    traffic -- MLX exposes no bytes-moved counter. It assumes each decode step
    reads the whole resident cache once, which is what dense attention does,
    and ignores tiling and cache-hierarchy effects. Use it to compare arms
    against each other, not as an absolute traffic figure.

    The two mechanisms it distinguishes are the point of the whole harness:

    * **Quantization** scales bytes/token by the bit ratio (``bits/16``) plus
      group scale/zero overhead. Traffic still grows linearly with sequence
      length -- it just grows more slowly.
    * **Eviction** *caps* the resident token count at the budget. Past the
      budget, bytes/token stops growing with sequence length entirely.

    Args:
        config: A ``KVCacheConfig``. ``method`` selects the mechanism;
            ``bit_width_inlier`` gives the quantized bit-width. ``None`` means
            an uncompressed fp16 baseline.
        n_layers: Attention-bearing layer count.
        n_kv_heads: KV heads per layer (post-GQA, not query heads).
        head_dim: Per-head dimension.
        seq_len: Tokens resident before eviction is applied.

    Returns:
        Bytes read per decode step across all layers, for K and V together.
    """
    # Baseline: no config at all means stock fp16 KV.
    if config is None:
        return n_layers * n_kv_heads * head_dim * seq_len * _FP16_BYTES * 2

    resident = seq_len
    budget = _eviction_budget(config)
    if budget is not None:
        # Eviction caps residency. This is the asymmetry the harness exists to
        # show: bytes/token plateaus instead of growing with seq_len.
        resident = min(seq_len, budget)

    per_layer_head = n_kv_heads * head_dim

    if budget is not None:
        # Eviction methods keep survivors in fp16 -- they drop tokens, they do
        # not narrow them.
        return n_layers * per_layer_head * resident * _FP16_BYTES * 2

    bits = getattr(config, "bit_width_inlier", None)
    if isinstance(bits, list):
        # Per-layer (RateQuant-style) allocation: sum each layer's own width.
        widths = [int(b) for b in bits]
    else:
        widths = [int(bits)] * n_layers if bits is not None else []

    if not widths:
        return n_layers * per_layer_head * resident * _FP16_BYTES * 2

    group_size = int(getattr(config, "kivi_group_size", 32) or 32)

    # KIVI-style methods keep the most recent `residual_length` tokens in fp16
    # and quantize them only once they age out. Charging every resident token
    # at the quantized width would understate their real traffic, so the
    # residual window is billed at fp16 -- matching KIVIKVCache._account_bytes,
    # which reports it separately as `residual_fp16_bytes`.
    residual = min(resident, int(getattr(config, "residual_length", 0) or 0))
    n_quantized = max(0, resident - residual)

    total = 0
    for b in widths:
        # Codes: b bits per element, for K and V.
        code_bytes = 2 * ((n_quantized * per_layer_head * b + 7) // 8)
        # Group params: scale+zero per group, for K and V.
        if n_quantized > 0:
            n_groups = max(1, (n_quantized * per_layer_head + group_size - 1) // group_size)
            param_bytes = 2 * n_groups * _GROUP_PARAM_BYTES
        else:
            param_bytes = 0
        # The still-fp16 residual tail, K and V.
        residual_bytes = residual * per_layer_head * _FP16_BYTES * 2
        total += code_bytes + param_bytes + residual_bytes
    return total


@dataclass(frozen=True)
class RunMetrics:
    """One measured arm. Energy fields are ``None`` when unprivileged."""

    tokens_generated: int
    wall_s: float
    tokens_per_s: float
    prefill_s: float
    decode_s: float
    peak_memory_mb: float
    kv_bytes_per_token: int  # DERIVED, not measured
    energy_j: float | None = None
    j_per_token: float | None = None
    mean_gpu_mw: float | None = None
    mean_cpu_mw: float | None = None
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def compute_j_per_token(energy_j: float | None, tokens: int) -> float | None:
    """J/token, or ``None`` when energy is unavailable.

    Deliberately returns ``None`` rather than ``0.0`` for missing energy: a
    silent zero would travel downstream as a real measurement showing free
    inference. Also returns ``None`` for a zero token count instead of
    dividing by zero.
    """
    if energy_j is None or tokens <= 0:
        return None
    return energy_j / tokens


def measure_generation(
    prefill: Callable[[], Any],
    decode_step: Callable[[int], Any],
    n_tokens: int,
    kv_bytes: int,
    sampler: Any = None,
    label: str = "",
) -> RunMetrics:
    """Time a prefill + decode loop, attaching power data when available.

    ``mx.eval()`` is called on whatever each callback returns before the timer
    stops, so the measured interval covers execution rather than lazy graph
    construction.

    Args:
        prefill: Runs the one-shot prompt pass; its return value is evaluated.
        decode_step: Called with the step index for each generated token.
        n_tokens: Number of decode steps to run.
        kv_bytes: Derived bytes/token from :func:`kv_bytes_per_token`.
        sampler: Optional ``PowerSampler``; ``None`` disables energy fields.
        label: Arm name for reporting.
    """
    _reset_peak()

    t0 = time.perf_counter()
    out = prefill()
    _sync(out)
    t_prefill_end = time.perf_counter()

    for i in range(n_tokens):
        out = decode_step(i)
    _sync(out)
    t_end = time.perf_counter()

    prefill_s = t_prefill_end - t0
    decode_s = t_end - t_prefill_end
    wall_s = t_end - t0

    energy_j = sampler.energy_joules() if sampler is not None else None
    power = sampler.mean_power_mw() if sampler is not None else {}

    return RunMetrics(
        tokens_generated=n_tokens,
        wall_s=wall_s,
        tokens_per_s=(n_tokens / decode_s) if decode_s > 0 else float("nan"),
        prefill_s=prefill_s,
        decode_s=decode_s,
        peak_memory_mb=_peak_mb(),
        kv_bytes_per_token=kv_bytes,
        energy_j=energy_j,
        j_per_token=compute_j_per_token(energy_j, n_tokens),
        mean_gpu_mw=power.get("gpu"),
        mean_cpu_mw=power.get("cpu"),
        label=label,
    )


def _sync(out: Any) -> None:
    """Force evaluation of pending MLX work; never raise from a timer path."""
    try:
        if isinstance(out, mx.array):
            mx.eval(out)
        else:
            mx.eval(*[o for o in _flatten(out) if isinstance(o, mx.array)])
    except Exception:
        try:
            mx.synchronize()
        except Exception:
            pass


def _flatten(obj: Any) -> list:
    if isinstance(obj, mx.array):
        return [obj]
    if isinstance(obj, (list, tuple)):
        out: list = []
        for item in obj:
            out.extend(_flatten(item))
        return out
    return []


__all__ = [
    "RunMetrics",
    "compute_j_per_token",
    "kv_bytes_per_token",
    "measure_generation",
]
