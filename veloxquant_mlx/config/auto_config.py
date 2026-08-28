"""Hardware-aware automatic KV-cache configuration selector (issue #253).

A single compression configuration is unlikely to be optimal for every model,
head dimension, sequence length, and memory budget. :func:`select_kv_cache_config`
picks a method, bit-width, and group size for the caller from a small,
well-understood pool of quantization methods, given a :class:`WorkloadSpec`
describing the job and (optionally) a :class:`HardwareInfo` describing the
machine it will run on.

The pool is intentionally small — ``turboquant_rvq``, ``kivi``, ``kvquant``,
and ``gear`` — rather than the full 40-method registry. Every
member is servable (see :mod:`veloxquant_mlx.cache.registry`), reports full
key+value telemetry, and exposes a plain int bit-width / group-size knob, so
the rules below stay simple to state and to test. Eviction and hybrid methods
are out of scope: they change *which* tokens are kept, not how each kept
token is encoded, so they don't fit a bit-width/group-size selection axis.

Usage::

    from veloxquant_mlx.config import select_kv_cache_config, WorkloadSpec

    config = select_kv_cache_config(WorkloadSpec(head_dim=128, seq_len=32_000))
    cache = KVCacheFactory.create(config)
"""

from __future__ import annotations

from dataclasses import dataclass

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.core.exceptions import QuantizerConfigError

__all__ = [
    "WorkloadSpec",
    "HardwareInfo",
    "AutoConfigResult",
    "detect_hardware_info",
    "select_kv_cache_config",
]

# Sequence-length bands, in tokens. Below SHORT_CONTEXT_TOKENS the memory
# savings from aggressive compression are marginal but the accuracy cost is
# not, so short jobs default toward higher precision. Above
# LONG_CONTEXT_TOKENS the cache itself dominates memory, so aggressive
# compression starts paying for itself even without memory pressure.
SHORT_CONTEXT_TOKENS: int = 2_048
LONG_CONTEXT_TOKENS: int = 16_384

# Large head dims (e.g. some MQA/GQA configs, or wide non-Llama-style heads)
# get a bigger group size so each quantization group still has enough
# elements to amortize scale/zero-point overhead.
LARGE_HEAD_DIM: int = 256

# Fraction of total device memory considered "under pressure" once the
# workload's own estimated fp16 KV footprint is added to what's already
# active. Above this fraction we bias toward the lowest-bit-width method
# in the pool regardless of sequence length.
MEMORY_PRESSURE_FRACTION: float = 0.75

_BYTES_PER_FP16_ELEMENT: int = 2


@dataclass(frozen=True)
class WorkloadSpec:
    """Describes the job the KV cache needs to serve.

    Attributes:
        head_dim: Attention head dimension. Must be a positive power of 2,
            matching :class:`KVCacheConfig`'s own constraint.
        seq_len: Expected (or worst-case) sequence length in tokens. Drives
            the short-context / long-context precision tradeoff.
        n_layers: Number of attention layers whose caches will share this
            configuration. Used only to size the memory-pressure estimate;
            defaults to 1 (single-layer estimate).
        batch_size: Number of concurrent sequences. Used only to size the
            memory-pressure estimate; defaults to 1.
    """

    head_dim: int = 128
    seq_len: int = 4_096
    n_layers: int = 1
    batch_size: int = 1

    def __post_init__(self) -> None:
        d = self.head_dim
        if not (isinstance(d, int) and d >= 1 and (d & (d - 1)) == 0):
            raise QuantizerConfigError(f"WorkloadSpec: head_dim={d} must be a positive power of 2.")
        if self.seq_len < 1:
            raise QuantizerConfigError(f"WorkloadSpec: seq_len={self.seq_len} must be >= 1.")
        if self.n_layers < 1:
            raise QuantizerConfigError(f"WorkloadSpec: n_layers={self.n_layers} must be >= 1.")
        if self.batch_size < 1:
            raise QuantizerConfigError(f"WorkloadSpec: batch_size={self.batch_size} must be >= 1.")

    def fp16_kv_bytes(self) -> int:
        """Estimated fp16 footprint of the full K+V cache for this workload."""
        return (
            2  # K and V
            * self.batch_size
            * self.n_layers
            * self.seq_len
            * self.head_dim
            * _BYTES_PER_FP16_ELEMENT
        )


@dataclass(frozen=True)
class HardwareInfo:
    """Describes the machine the KV cache will run on.

    Attributes:
        total_memory_bytes: Total device memory (unified memory on Apple
            Silicon). None disables memory-pressure-based selection.
        active_memory_bytes: Memory already in use before this cache is
            allocated (e.g. model weights, other requests' caches).
    """

    total_memory_bytes: int | None = None
    active_memory_bytes: int = 0

    def pressure_fraction(self, additional_bytes: int) -> float | None:
        """Fraction of total memory used once ``additional_bytes`` is added.

        Returns None if ``total_memory_bytes`` is unknown.
        """
        if not self.total_memory_bytes:
            return None
        used = self.active_memory_bytes + additional_bytes
        return used / self.total_memory_bytes


def detect_hardware_info() -> HardwareInfo:
    """Auto-detect :class:`HardwareInfo` from the running MLX device.

    Falls back to an empty (unknown-memory) :class:`HardwareInfo` if MLX is
    unavailable or device introspection fails — callers then get
    sequence-length-only selection instead of an exception.
    """
    try:
        import mlx.core as mx

        info = mx.device_info()
        total = info.get("memory_size")
        active = mx.get_active_memory()
        return HardwareInfo(total_memory_bytes=total, active_memory_bytes=active)
    except Exception:
        return HardwareInfo()


@dataclass(frozen=True)
class AutoConfigResult:
    """The selected configuration plus the reasoning behind it.

    Attributes:
        config: The resulting KVCacheConfig, ready for KVCacheFactory.create()
            or KVCacheBuilder.for_model().
        reason: Short human-readable explanation of why this method/bit-width
            was chosen, for logging or a CLI's ``--explain`` flag.
    """

    config: KVCacheConfig
    reason: str


def select_kv_cache_config(
    workload: WorkloadSpec,
    hardware: HardwareInfo | None = None,
) -> AutoConfigResult:
    """Pick a KVCacheConfig for the given workload and hardware.

    Selection rules (in priority order):

    1. Memory pressure: if ``hardware`` reports the workload's estimated
       fp16 footprint would push total usage above
       :data:`MEMORY_PRESSURE_FRACTION` of device memory, pick the most
       aggressive method in the pool (``gear``, 2-bit) regardless of
       sequence length.
    2. Sequence length: short contexts (< :data:`SHORT_CONTEXT_TOKENS`)
       favor higher precision (``turboquant_rvq``, 4-bit); long contexts
       (>= :data:`LONG_CONTEXT_TOKENS`) favor aggressive compression
       (``kvquant``, 3-bit with outlier isolation); everything in between
       uses a balanced default (``kivi``, 2-bit).
    3. Head dimension: if ``head_dim >= LARGE_HEAD_DIM``, the group size is
       doubled (min(head_dim, 64)) so each group still holds enough
       elements to amortize per-group scale/zero-point overhead.

    Args:
        workload: Description of the job (head_dim, seq_len, n_layers, batch_size).
        hardware: Description of the target machine. If None, auto-detects
            via :func:`detect_hardware_info`.

    Returns:
        AutoConfigResult with the selected KVCacheConfig and a human-readable
        reason string.
    """
    if hardware is None:
        hardware = detect_hardware_info()

    group_size = 64 if workload.head_dim >= LARGE_HEAD_DIM else 32

    pressure = hardware.pressure_fraction(workload.fp16_kv_bytes())
    if pressure is not None and pressure >= MEMORY_PRESSURE_FRACTION:
        config = KVCacheConfig(
            method="gear",
            head_dim=workload.head_dim,
            gear_bits=2,
            gear_group_size=group_size,
        )
        reason = (
            f"memory pressure {pressure:.0%} >= {MEMORY_PRESSURE_FRACTION:.0%} threshold: "
            f"selected gear (2-bit, error-feedback low-rank residual) to maximize compression"
        )
        return AutoConfigResult(config=config, reason=reason)

    if workload.seq_len < SHORT_CONTEXT_TOKENS:
        config = KVCacheConfig(
            method="turboquant_rvq",
            head_dim=workload.head_dim,
            bit_width_inlier=4,
        )
        reason = (
            f"seq_len={workload.seq_len} < {SHORT_CONTEXT_TOKENS} (short context): "
            f"selected turboquant_rvq (4-bit) for higher precision"
        )
        return AutoConfigResult(config=config, reason=reason)

    if workload.seq_len >= LONG_CONTEXT_TOKENS:
        config = KVCacheConfig(
            method="kvquant",
            head_dim=workload.head_dim,
            kvquant_bits=3,
            kvquant_group_size=group_size,
            kvquant_outlier_fraction=0.01,
        )
        reason = (
            f"seq_len={workload.seq_len} >= {LONG_CONTEXT_TOKENS} (long context): "
            f"selected kvquant (3-bit NUQ + outlier isolation) for aggressive compression"
        )
        return AutoConfigResult(config=config, reason=reason)

    config = KVCacheConfig(
        method="kivi",
        head_dim=workload.head_dim,
        bit_width_inlier=2,
        kivi_group_size=group_size,
    )
    reason = (
        f"{SHORT_CONTEXT_TOKENS} <= seq_len={workload.seq_len} < {LONG_CONTEXT_TOKENS} "
        f"(mid-length context): selected kivi (2-bit asymmetric group quantization) as a "
        f"balanced default"
    )
    return AutoConfigResult(config=config, reason=reason)
