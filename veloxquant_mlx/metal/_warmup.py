"""Ahead-of-time Metal kernel warmup, keyed off ``KVCacheConfig``.

Every Metal kernel in this package specializes at compile time on config-fixed
constants (bit-width, head_dim, group_size, ...) via ``mx.fast.metal_kernel``
templates/headers, then memoizes the compiled pipeline in a module-level dict
(see ``_kivi_quant.py``, ``_bit_packing.py``, ``_scalar_quant.py``). That
compile is lazy: it happens on the first real call, which for a serving
workload means the first decode step of the first request pays a ~200-800ms
shader-compile stall (see ``docs-site/docs/guides/metal-kernels.md``).

``warmup_for_config`` runs that first call — with a throwaway, minimally
shaped input — once, synchronously, at cache-build time (when the config's
bit-width/head_dim/etc. are already fixed), so the compile cost lands during
setup instead of mid-generation.

This module only warms kernels with calling conventions fully derivable from
``KVCacheConfig`` alone. Kernels whose shapes also depend on runtime
artifacts (e.g. VecInfer's codebook-derived ``sub_dim``/``n_centroids``) are
left to compile lazily on first real use, same as today.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx

from veloxquant_mlx.metal import metal_available

# method -> warmup callable(config) -> None. Best-effort: a warmer raising
# never propagates past warmup_for_config (see below).
_WARMERS: dict[str, Callable[[Any], None]] = {}


def register_warmer(method: str, warmer: Callable[[Any], None]) -> None:
    """Register a warmup hook for ``config.method == method``."""
    _WARMERS[method] = warmer


def _warm_kivi(config: Any) -> None:
    from veloxquant_mlx.metal.kernels import kivi_group_quant_dequant

    d = int(config.head_dim)
    b = config.bit_width_inlier
    if isinstance(b, list):
        return
    levels = (1 << int(b)) - 1
    group_size = int(getattr(config, "kivi_group_size", 32))
    dtype = config.dtype or mx.float16

    # One threadgroup's worth of tokens is enough to compile both axis
    # variants (keys group along tokens=-2, values group along channels=-1);
    # the sequence length is never specialized on (see _kivi_quant.py), so
    # any small S compiles the same pipeline the real traffic will reuse.
    dummy = mx.zeros((1, 1, group_size, d), dtype=dtype)
    for axis in (-2, -1):
        kivi_group_quant_dequant(dummy, axis=axis, group_size=group_size, levels=levels)
    mx.eval(dummy)


register_warmer("kivi", _warm_kivi)
register_warmer("kivi_sink", _warm_kivi)


def warmup_for_config(config: Any) -> None:
    """Best-effort ahead-of-time compile of this config's Metal kernels.

    No-op if Metal is unavailable, the method has no registered warmer, or
    the warmer itself raises (warmup is a latency optimization, never a
    correctness requirement — a failed warmup just means the first real call
    falls back to the normal lazy-compile path).
    """
    if not metal_available():
        return
    warmer = _WARMERS.get(getattr(config, "method", None))
    if warmer is None:
        return
    try:
        warmer(config)
    except Exception:
        pass


__all__ = ["warmup_for_config", "register_warmer"]
