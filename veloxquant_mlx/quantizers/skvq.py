"""SKVQ — channel-reordered, clip-searched group quantization primitives.

Based on: "SKVQ: Sliding-window Key and Value Cache Quantization for Large
Language Models" (Duanmu, Yuan, Li, Duan, Zhang, Lin — COLM 2024;
arXiv:2405.06219). Reference implementation: https://github.com/cat538/SKVQ

**SKVQ-adapted (VeloxQuant-MLX implementation)** — not a faithful port.
The paper computes its channel permutation offline (KMeans over per-channel
distribution features on a WikiText-2 calibration set, fused into the
attention projection weights) and its clip factor offline (attention-output
MSE per block). A cache-side library sees no calibration corpus and cannot
touch projection weights, so this module adapts both to cache-observable
data (see ``paper/NEW_METHOD_SURVEY_V13.md``):

  * **Channel permutation** — computed from observed tokens by sorting
    channels on a scalar feature (per-channel dynamic range ``max − min``).
    Sorting a scalar feature and cutting into contiguous groups is the
    optimal 1-D grouping — the fixed point KMeans on a scalar feature
    converges to — so this is the paper's mechanism with a single-feature
    statistic instead of multi-feature KMeans. The permutation is applied
    explicitly at quantization time and inverted on dequantization (the
    round-trip is mathematically identical to weight fusion).
  * **Clipped dynamic quantization** — per-token, per-group asymmetric
    min/max quantization whose window is shrunk by a factor α: the clip
    window is centered on the group midpoint,

        lo    = mid − α · (gmax − gmin) / 2
        scale = α · (gmax − gmin) / (2**bits − 1)
        q     = clip(round((x − lo) / scale), 0, 2**bits − 1)

    trading saturation of a few extreme values for finer resolution
    everywhere else. α is chosen **per group** by grid search minimizing
    reconstruction MSE (the paper searches offline against attention-output
    MSE — documented deviation). α = 1.0 recovers plain asymmetric min/max
    quantization exactly, and the default grid contains it, so under the
    search metric clipping never loses to not clipping. The chosen α is
    folded into the stored ``(lo, scale)`` — nothing extra is stored.

Quantization groups run **along the channel axis** (per-token scales, the
KIVI *value* scheme) for both keys and values: reordering exists precisely
to make per-token channel groups viable for keys, whose heterogeneous
channels otherwise stretch the group range (the KIVI/KVQuant observation).

Everything here is deterministic — no RNG, no codebook training.

Public API:
  channel_permutation, invert_permutation, apply_permutation,
  clipped_group_quant, clipped_group_dequant, skvq_round_trip,
  skvq_compressed_bytes, skvq_fp16_bytes, DEFAULT_ALPHA_GRID
"""

from __future__ import annotations

import math
from typing import Sequence

import mlx.core as mx

# α = 1.0 (plain min/max) is always a candidate, so per-group search can
# only improve the reconstruction MSE relative to not clipping.
DEFAULT_ALPHA_GRID: tuple = (1.0, 0.97, 0.94, 0.90, 0.85)

_EPS = 1e-8


def _validate(bits: int, group_size: int, alphas: Sequence[float]) -> None:
    if not (1 <= int(bits) <= 8):
        raise ValueError(f"skvq: bits={bits} must be in [1, 8] (uint8 codes)")
    if int(group_size) < 1:
        raise ValueError(f"skvq: group_size={group_size} must be >= 1")
    if len(alphas) == 0:
        raise ValueError("skvq: alphas must be non-empty")
    for a in alphas:
        if not (0.0 < float(a) <= 1.0):
            raise ValueError(f"skvq: alpha={a} must be in (0, 1]")


def channel_permutation(x: mx.array) -> mx.array:
    """Sorted-by-dynamic-range channel permutation from observed tokens.

    Args:
        x: ``[N, D]`` observed rows (keys or values of one head).

    Returns:
        ``[D]`` int32 permutation ordering channels by ascending dynamic
        range (``max − min`` over the N rows), so contiguous quantization
        groups contain channels of similar range. Deterministic (argsort is
        stable).
    """
    x32 = x.astype(mx.float32)
    rng = mx.max(x32, axis=0) - mx.min(x32, axis=0)
    return mx.argsort(rng).astype(mx.int32)


def invert_permutation(perm: mx.array) -> mx.array:
    """Inverse permutation: ``invert_permutation(perm)[perm[i]] == i``."""
    return mx.argsort(perm).astype(mx.int32)


def apply_permutation(x: mx.array, perm: mx.array) -> mx.array:
    """Gather ``x[..., perm]`` along the last axis."""
    return mx.take(x, perm, axis=-1)


def clipped_group_quant(
    x: mx.array,
    bits: int,
    group_size: int,
    alphas: Sequence[float] = DEFAULT_ALPHA_GRID,
) -> tuple:
    """Per-token, per-group asymmetric quantization with clip-factor search.

    Groups partition the channel axis into blocks of ``group_size`` (a
    ragged final group is padded by repeating the last channel — the pad is
    discarded on dequantization). For every α in ``alphas`` the group's clip
    window is centered on its midpoint and shrunk by α; the α minimizing the
    group's reconstruction MSE wins. α is folded into the returned
    ``(lo, scale)`` — it is not stored separately.

    Args:
        x: ``[N, D]`` rows to quantize (already permuted if reordering).
        bits: code bit-width in [1, 8].
        group_size: channels per quantization group.
        alphas: candidate clip factors, each in (0, 1].

    Returns:
        ``(codes, lo, scale)`` — codes uint8 ``[N, G*group_size]`` (padded
        width), lo/scale float32 ``[N, G]``.
    """
    _validate(bits, group_size, alphas)
    n, d = x.shape
    gs = int(group_size)
    n_groups = (d + gs - 1) // gs
    pad = n_groups * gs - d

    x32 = x.astype(mx.float32)
    if pad:
        x32 = mx.concatenate([x32, mx.broadcast_to(x32[:, -1:], (n, pad))], axis=1)
    xg = x32.reshape(n, n_groups, gs)  # [N, G, gs]

    gmin = mx.min(xg, axis=-1, keepdims=True)  # [N, G, 1]
    gmax = mx.max(xg, axis=-1, keepdims=True)
    mid = (gmax + gmin) * 0.5
    rng = gmax - gmin
    levels = (1 << int(bits)) - 1

    a = mx.array(list(alphas), dtype=mx.float32).reshape(-1, 1, 1, 1)  # [A,1,1,1]
    lo_a = mid[None] - a * rng[None] * 0.5  # [A, N, G, 1]
    scale_a = mx.maximum(a * rng[None] / levels, _EPS)
    codes_a = mx.clip(mx.round((xg[None] - lo_a) / scale_a), 0, levels)
    recon_a = codes_a * scale_a + lo_a
    err_a = mx.mean(mx.square(recon_a - xg[None]), axis=-1)  # [A, N, G]

    best = mx.argmin(err_a, axis=0)  # [N, G]
    idx_c = mx.broadcast_to(best[None, :, :, None], (1, n, n_groups, gs))
    codes = mx.take_along_axis(codes_a, idx_c, axis=0)[0]  # [N, G, gs]
    idx_p = best[None, :, :, None]
    lo = mx.take_along_axis(lo_a, idx_p, axis=0)[0, :, :, 0]
    scale = mx.take_along_axis(scale_a, idx_p, axis=0)[0, :, :, 0]

    return (
        codes.reshape(n, n_groups * gs).astype(mx.uint8),
        lo.astype(mx.float32),
        scale.astype(mx.float32),
    )


def clipped_group_dequant(
    codes: mx.array,
    lo: mx.array,
    scale: mx.array,
    group_size: int,
    d: int,
) -> mx.array:
    """Reconstruct ``[N, d]`` float32 rows from clipped-group codes."""
    n = codes.shape[0]
    gs = int(group_size)
    n_groups = lo.shape[1]
    cg = codes.reshape(n, n_groups, gs).astype(mx.float32)
    recon = cg * scale[:, :, None] + lo[:, :, None]
    return recon.reshape(n, n_groups * gs)[:, :d]


def skvq_round_trip(
    x: mx.array,
    perm,
    bits: int,
    group_size: int,
    alphas: Sequence[float] = DEFAULT_ALPHA_GRID,
) -> mx.array:
    """Permute → clipped group quant → dequant → inverse permute.

    Args:
        x: ``[N, D]`` rows.
        perm: ``[D]`` channel permutation, or ``None`` for identity (the
            ``skvq_reorder=False`` ablation).

    Returns:
        ``[N, D]`` reconstruction in ``x``'s dtype.
    """
    n, d = x.shape
    xp = apply_permutation(x, perm) if perm is not None else x
    codes, lo, scale = clipped_group_quant(xp, bits, group_size, alphas)
    recon = clipped_group_dequant(codes, lo, scale, group_size, d)
    if perm is not None:
        recon = apply_permutation(recon, invert_permutation(perm))
    return recon.astype(x.dtype)


def skvq_compressed_bytes(n_tokens: int, d: int, bits: int, group_size: int) -> int:
    """Analytic storage for the quantized region: codes + fp16 (lo, scale)
    per (token, group). The searched α adds nothing — it is folded into
    (lo, scale)."""
    n_groups = (d + group_size - 1) // group_size
    code_bytes = math.ceil(n_tokens * d * bits / 8)
    param_bytes = n_tokens * n_groups * 2 * 2  # lo + scale, fp16
    return code_bytes + param_bytes


def skvq_fp16_bytes(n_tokens: int, d: int) -> int:
    """fp16 baseline bytes for one tensor (keys OR values)."""
    return n_tokens * d * 2


__all__ = [
    "DEFAULT_ALPHA_GRID",
    "channel_permutation",
    "invert_permutation",
    "apply_permutation",
    "clipped_group_quant",
    "clipped_group_dequant",
    "skvq_round_trip",
    "skvq_compressed_bytes",
    "skvq_fp16_bytes",
]
