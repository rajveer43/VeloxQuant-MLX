"""Cross-model KV transfer — fused RoPE re-encode Metal kernel.

Replaces the ``strip_rope`` → ``apply_rope`` pair in
:mod:`veloxquant_mlx.transfer.rope` with a single GPU dispatch. The two
rotations act on the same ``(d, d + D/2)`` element pair at the same position,
so they compose into one rotation by the per-dimension angle *difference* —
one sincos and one 2×2 rotation per pair, with no intermediate ``[N, D]``
array and no cos/sin tables materialized.

This is a different composition from
:mod:`veloxquant_mlx.metal._h2o_evict`'s remap: there the RoPE base is shared
between the two rotations, so the angles collapse to
``(new_pos - old_pos) * inv_freq``. Here the *bases differ* (source and target
are different models with different ``rope_theta``), so the difference has to
be taken per dimension after exponentiation, and no position delta exists.

Public API:
  - :func:`crosskv_rope_recode`
  - :func:`is_available`
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

__all__ = ["crosskv_rope_recode", "is_available"]


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return (Path(__file__).parent / "src" / filename).read_text()


_cache: dict = {}

# ===========================================================================
# Metal source — one dispatch, one thread per rotated element pair
# ===========================================================================
# Grid:        (BH * N * D/2, 1, 1)
# Threadgroup: (256, 1, 1)
#
# Threads are independent (each owns one (d, d+half) pair of one token), so
# there are no barriers and no reduction phase — unlike the eviction kernels,
# which need a separate argmin dispatch before they can act.
_CROSSKV_ROPE_RECODE_SRC = _read_kernel_source("crosskv_rope_recode.metal")


def is_available() -> bool:
    """True when a Metal GPU is present to dispatch to."""
    try:
        return mx.metal.is_available()
    except Exception:
        return False


def _recode_kernel(dtype_name: str):
    key = ("crosskv_rope_recode", dtype_name)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"crosskv_rope_recode_{dtype_name}",
            input_names=["keys", "positions", "rope_bases"],
            output_names=["out"],
            source=_CROSSKV_ROPE_RECODE_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


def crosskv_rope_recode(
    keys: mx.array,
    positions: mx.array,
    source_base: float,
    target_base: float,
) -> mx.array:
    """Re-encode keys from the source model's RoPE to the target's, fused.

    Numerically equivalent to
    ``apply_rope(strip_rope(keys, positions, source_base), positions, target_base)``
    and to :func:`veloxquant_mlx.transfer.rope.recode_rope`, computed in one
    dispatch.

    Args:
        keys: ``[BH, N, D]`` fp16/fp32 keys rotated under ``source_base``.
            ``BH`` folds batch and head; ``D`` must be even.
        positions: ``[N]`` absolute positions, shared across the ``BH`` groups.
        source_base: Source model's ``rope_theta``.
        target_base: Target model's ``rope_theta``.

    Returns:
        ``[BH, N, D]`` keys rotated as though produced by the target model,
        same dtype as ``keys``.

    Raises:
        ValueError: If ``keys`` is not 3-D, ``D`` is odd, or ``positions``
            does not match the token count.
    """
    if keys.ndim != 3:
        raise ValueError(f"keys must be [BH, N, D]; got shape {keys.shape}.")
    bh, n, d = keys.shape
    if d % 2 != 0:
        raise ValueError(f"head_dim must be even for NeoX-style RoPE; got {d}.")
    if int(positions.shape[0]) != n:
        raise ValueError(
            f"positions has {int(positions.shape[0])} entries but keys hold {n} tokens."
        )
    if n == 0:
        return keys

    dtype_name = "float16" if keys.dtype == mx.float16 else "float32"
    kernel = _recode_kernel(dtype_name)

    total_threads = bh * n * (d // 2)
    tg = min(256, total_threads)

    (out,) = kernel(
        inputs=[
            keys,
            positions.astype(mx.float32),
            mx.array([source_base, target_base], dtype=mx.float32),
        ],
        template=[("T", keys.dtype)],
        grid=(total_threads, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[keys.shape],
        output_dtypes=[keys.dtype],
    )
    return out
