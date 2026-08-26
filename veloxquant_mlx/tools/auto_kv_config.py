"""Hardware-aware automatic KV-cache quantization configuration (#253).

A single compression configuration is unlikely to be optimal for every
model, head dimension, sequence length, and hardware configuration. This
module picks a quantization method, bit-width, group size, and packing
strategy directly from a description of the workload (expected context
length, head dimension) and the machine it runs on (unified memory, Metal
availability) — no manually-chosen "goal" required, unlike
:mod:`veloxquant_mlx.tools.mac_recommender`, which recommends a method
from a user-picked goal (max compression, best quality, ...) for a given
Mac chip/RAM combination.

Pure heuristics, no MLX required at import time (mirrors
``mac_recommender.py``) — only :func:`to_kv_cache_config` needs MLX
installed, since it constructs a real
:class:`~veloxquant_mlx.cache.base.KVCacheConfig`.

Deliberately restricted to methods that need no calibration pass
(``kivi``, ``turboquant_rvq``, ``streaming_llm``), so the recommendation
is usable immediately with no setup step the caller has to remember to
run first.

Example::

    from veloxquant_mlx.tools.auto_kv_config import (
        HardwareProfile,
        WorkloadProfile,
        select_kv_config,
    )

    result = select_kv_config(
        WorkloadProfile(seq_len=32768, head_dim=128, n_layers=32, n_kv_heads=8),
        HardwareProfile(ram_gb=16),
    )
    print(result.method, result.bit_width, result.knobs)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

ContextRegime = Literal["short", "medium", "long"]
PackingStrategy = Literal["metal_auto", "pure_mlx"]

# Context-length thresholds (tokens) separating the three regimes the issue
# calls out: "short context -> higher precision", "long context -> aggressive
# compression".
SHORT_CONTEXT_TOKENS: int = 4096
LONG_CONTEXT_TOKENS: int = 32768

# Base inlier bit-width per context regime, before any memory-pressure
# adjustment. Both candidate methods (kivi, turboquant_rvq) accept any
# bit_width_inlier >= 1.
_BASE_BITS: dict[ContextRegime, int] = {"short": 4, "medium": 2, "long": 1}
MIN_BIT_WIDTH: int = 1

# Preferred group size per regime (tokens per min/max quantization group),
# clamped down to a divisor of head_dim by _select_group_size. Large head
# dimensions can afford coarser (cheaper) groups; the issue's "large head
# dim -> specialized block size" is handled by deriving the actual group
# size from head_dim rather than using this value unclamped.
_PREFERRED_GROUP_SIZE: dict[ContextRegime, int] = {"short": 64, "medium": 32, "long": 32}

# Memory-pressure ratios (estimated fp16 KV size / memory budget) at which
# the selector escalates compression.
PRESSURE_LOWER_BITS: float = 1.0
PRESSURE_FORCE_MIN_BITS: float = 4.0
PRESSURE_FORCE_EVICTION: float = 16.0


@dataclass(frozen=True)
class WorkloadProfile:
    """Describes the inference workload the cache will serve.

    Attributes:
        seq_len: Expected/typical maximum context length in tokens.
        head_dim: Attention head dimension.
        n_layers: Number of transformer layers.
        n_kv_heads: Number of KV heads (post-GQA/MQA grouping).
    """

    seq_len: int = 4096
    head_dim: int = 128
    n_layers: int = 32
    n_kv_heads: int = 8


@dataclass(frozen=True)
class HardwareProfile:
    """Describes the machine the cache will run on.

    Attributes:
        ram_gb: Total unified memory in GB.
        memory_budget_gb: GB actually available for the KV cache once
            weights, activations, and OS overhead are accounted for.
            ``None`` derives a conservative estimate from ``ram_gb``.
        metal_available: Whether Metal kernel acceleration can be used.
            ``True``/``None`` are treated the same ("may be available,
            detect at runtime"); only an explicit ``False`` disables it,
            matching ``KVCacheConfig.use_metal_kernels``'s own
            auto-detect-with-silent-fallback default.
    """

    ram_gb: float = 16.0
    memory_budget_gb: Optional[float] = None
    metal_available: Optional[bool] = True

    def resolved_budget_gb(self) -> float:
        if self.memory_budget_gb is not None:
            return self.memory_budget_gb
        # Conservative: assume weights, activations, and OS overhead claim
        # the rest, leaving a fraction of unified memory for the KV cache.
        return max(self.ram_gb * 0.15, 1.0)


@dataclass(frozen=True)
class AutoConfigResult:
    """A selected configuration plus the reasoning behind it."""

    method: str
    bit_width: int
    group_size: Optional[int]
    packing_strategy: PackingStrategy
    context_regime: ContextRegime
    knobs: dict[str, Any]
    kv_fp16_gb: float
    memory_pressure_ratio: float
    rationale: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_kv_fp16_gb(workload: WorkloadProfile) -> float:
    """Full K+V fp16 cache size in GB at ``workload.seq_len``."""
    bytes_ = 2 * workload.n_layers * workload.n_kv_heads * workload.head_dim * workload.seq_len * 2
    return bytes_ / (1024**3)


def _context_regime(seq_len: int) -> ContextRegime:
    if seq_len <= SHORT_CONTEXT_TOKENS:
        return "short"
    if seq_len >= LONG_CONTEXT_TOKENS:
        return "long"
    return "medium"


def _select_group_size(head_dim: int, preferred: int) -> int:
    """Largest divisor of ``head_dim`` no larger than ``preferred``.

    Group-wise quantization (e.g. KIVI's min/max groups) needs group_size
    to divide head_dim evenly. Picking the largest such divisor keeps
    groups as coarse (cheap) as head_dim allows. Falls back to smaller
    divisors only for awkward head_dim values; group_size=1 (per-channel)
    always divides head_dim and so is always reachable.
    """
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    preferred = max(1, min(preferred, head_dim))
    for candidate in range(preferred, 0, -1):
        if head_dim % candidate == 0:
            return candidate
    return 1  # unreachable: candidate=1 always divides head_dim


def select_kv_config(
    workload: WorkloadProfile,
    hardware: Optional[HardwareProfile] = None,
) -> AutoConfigResult:
    """Pick a KV-cache quantization configuration for a workload + machine.

    Implements the issue's rules directly:

    * short context -> higher precision (higher bit-width)
    * long context -> aggressive compression (lower bit-width)
    * large head dim -> specialized (coarser) group size
    * memory pressure -> lower-bit KV, escalating to a bounded-memory
      eviction method when no bit-width fits the budget

    Args:
        workload: Expected context length, head dimension, layer/head counts.
        hardware: Target machine's memory and Metal availability. Defaults
            to a generic 16 GB Apple Silicon machine.

    Returns:
        The selected method, knobs, and the reasoning behind them.

    Raises:
        ValueError: If ``seq_len`` or ``head_dim`` is not positive.
    """
    hardware = hardware or HardwareProfile()
    if workload.seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {workload.seq_len}")
    if workload.head_dim < 1:
        raise ValueError(f"head_dim must be >= 1, got {workload.head_dim}")

    regime = _context_regime(workload.seq_len)
    kv_fp16_gb = estimate_kv_fp16_gb(workload)
    budget_gb = hardware.resolved_budget_gb()
    pressure = kv_fp16_gb / budget_gb if budget_gb > 0 else float("inf")

    warnings: list[str] = []
    packing: PackingStrategy = "metal_auto" if hardware.metal_available is not False else "pure_mlx"

    if pressure > PRESSURE_FORCE_EVICTION:
        method, bit_width, group_size, knobs, rationale = _select_eviction_fallback(
            workload, kv_fp16_gb, budget_gb, pressure, warnings
        )
    else:
        method, bit_width, group_size, knobs, rationale = _select_quantized(
            workload, regime, pressure, packing, warnings
        )

    return AutoConfigResult(
        method=method,
        bit_width=bit_width,
        group_size=group_size,
        packing_strategy=packing,
        context_regime=regime,
        knobs=knobs,
        kv_fp16_gb=round(kv_fp16_gb, 4),
        memory_pressure_ratio=round(pressure, 4),
        rationale=rationale,
        warnings=warnings,
    )


def _select_eviction_fallback(
    workload: WorkloadProfile,
    kv_fp16_gb: float,
    budget_gb: float,
    pressure: float,
    warnings: list[str],
) -> tuple[str, int, Optional[int], dict[str, Any], str]:
    """No bit-width survives this budget; bound memory outright instead.

    An eviction cache's memory footprint depends only on its window size,
    not on context length, so it is the only family that stops growing
    once ``seq_len`` outstrips even the most aggressive quantized method.
    """
    bytes_per_token = 2 * workload.n_layers * workload.n_kv_heads * workload.head_dim * 2
    budget_bytes = budget_gb * (1024**3)
    window = max(int(budget_bytes / bytes_per_token), 64) if bytes_per_token > 0 else 512

    warnings.append(
        f"Estimated fp16 KV cache ({kv_fp16_gb:.2f} GB at seq_len={workload.seq_len}) is "
        f"{pressure:.1f}x the available budget ({budget_gb:.2f} GB); no bit-width alone fits "
        f"this context. Falling back to streaming_llm, which bounds memory to a fixed window "
        f"({window} tokens) instead of growing with context length. Expect the model to lose "
        "track of anything outside the sink + window."
    )
    rationale = (
        f"Memory pressure ({pressure:.1f}x budget) forces a bounded-memory method regardless "
        "of context regime."
    )
    knobs = {"stream_n_sink": 4, "stream_window_size": window}
    # fp16-resident inside the window: no per-element quantization applies.
    return "streaming_llm", 16, None, knobs, rationale


def _select_quantized(
    workload: WorkloadProfile,
    regime: ContextRegime,
    pressure: float,
    packing: PackingStrategy,
    warnings: list[str],
) -> tuple[str, int, Optional[int], dict[str, Any], str]:
    bit_width = _BASE_BITS[regime]
    if pressure > PRESSURE_FORCE_MIN_BITS:
        bit_width = MIN_BIT_WIDTH
    elif pressure > PRESSURE_LOWER_BITS:
        bit_width = min(bit_width, 2)

    if bit_width < _BASE_BITS[regime]:
        warnings.append(
            f"Memory pressure ({pressure:.1f}x budget) lowered bit-width from "
            f"{_BASE_BITS[regime]} ({regime}-context default) to {bit_width}."
        )

    preferred_group = _PREFERRED_GROUP_SIZE[regime]
    group_size = _select_group_size(workload.head_dim, preferred_group)
    if workload.head_dim % preferred_group != 0:
        warnings.append(
            f"head_dim={workload.head_dim} does not divide evenly by the preferred group size "
            f"({preferred_group}); using {group_size} instead."
        )

    # KIVI's tuning-free min/max quantization is the accuracy-first pick
    # for short/medium contexts with no memory pressure. Long contexts (or
    # any memory pressure) move to turboquant_rvq: residual vector
    # quantization gives better inner-product fidelity at low bit-widths
    # and is the library's own serving default (see cache/registry.py's
    # DEFAULT_SERVE_METHOD).
    if regime == "long" or pressure > PRESSURE_LOWER_BITS:
        method = "turboquant_rvq"
        knobs: dict[str, Any] = {"bit_width_inlier": bit_width, "seed": 42}
    else:
        method = "kivi"
        knobs = {"bit_width_inlier": bit_width, "kivi_group_size": group_size}

    rationale = (
        f"{regime}-context regime ({workload.seq_len} tokens) at {pressure:.2f}x memory "
        f"pressure -> {method} at {bit_width}-bit"
        + (f", group_size={group_size}" if method == "kivi" else "")
        + f"; packing={packing}."
    )
    return method, bit_width, group_size, knobs, rationale


def to_kv_cache_config(result: AutoConfigResult, workload: WorkloadProfile) -> Any:
    """Build a real ``KVCacheConfig`` from a selection result.

    Imported lazily so this module stays importable without MLX installed
    (see module docstring) — only calling this function requires MLX.

    Args:
        result: Output of :func:`select_kv_config`.
        workload: The same profile passed to :func:`select_kv_config`.

    Returns:
        A ``veloxquant_mlx.cache.base.KVCacheConfig`` ready for
        ``KVCacheFactory.create()`` or ``KVCacheBuilder.for_model()``.
    """
    from veloxquant_mlx.cache.base import KVCacheConfig

    kwargs: dict[str, Any] = {"method": result.method, "head_dim": workload.head_dim}
    kwargs.update(result.knobs)
    kwargs["use_metal_kernels"] = None if result.packing_strategy == "metal_auto" else False
    return KVCacheConfig(**kwargs)
