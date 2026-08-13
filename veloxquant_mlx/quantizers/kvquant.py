"""KVQuant-NUQ quantizer — non-uniform quantization + dense/sparse outlier isolation.

Inspired by "KVQuant: Towards 10 Million Context Length LLM Inference with KV
Cache Quantization" (arXiv:2401.18079, NeurIPS 2024, Hooper et al.). Documented
as "KVQuant-adapted (VeloxQuant-MLX implementation)" — implements the two
cache-observable pillars and documents the third (pre-RoPE keys) as out of scope.

Two mechanisms, both fit online from the incoming tensor (zero calibration):

1. NUQ (Non-Uniform Quantization datatype). Every other quantizer in the suite
   snaps values to *uniformly* spaced levels (min/max). LLM K/V distributions are
   sharply non-uniform (bell-shaped, heavy-tailed). NUQ instead places ``2^bits``
   signpost levels where the data actually is, via a 1-D Lloyd-Max (k-means) fit.
   At the same bit-width this strictly reduces reconstruction error on
   non-uniform data. Quantize = nearest signpost; dequantize = table lookup.

2. Dense-and-Sparse outlier isolation. A small top-magnitude fraction of elements
   is stored separately in fp16 and excluded from the level fit, so a handful of
   outliers cannot stretch the level range and wreck precision for the bulk.

This module holds the pure, side-effect-free numerics; the cache wrapper in
``kvquant_cache.py`` applies them per-channel (keys) / per-token (values) and
manages the outlier side-channel.
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx


class DenseSparse(NamedTuple):
    """Split of a tensor into inliers (NUQ-coded) and fp16 outliers.

    Attributes:
        inliers: [N, D] fp32 with outlier positions replaced by the column mean
            (so they do not bias the level fit); the original outlier values are
            carried separately in ``outlier_vals``.
        outlier_mask: [N, D] bool — True at outlier positions.
        outlier_vals: [N, D] fp32 — original values at outlier positions, 0 else.
    """

    inliers: mx.array
    outlier_mask: mx.array
    outlier_vals: mx.array


def split_dense_sparse(x: mx.array, outlier_fraction: float) -> DenseSparse:
    """Carve out the top-magnitude ``outlier_fraction`` of elements per column.

    Args:
        x: [N, D] fp16 or fp32.
        outlier_fraction: Fraction in [0, 1). 0 → no outliers (pure NUQ).

    Returns:
        DenseSparse. With outlier_fraction == 0, mask is all-False and inliers == x.
    """
    x32 = x.astype(mx.float32)
    n, d = x32.shape
    if outlier_fraction <= 0.0 or n < 2:
        zeros = mx.zeros_like(x32)
        return DenseSparse(x32, zeros.astype(mx.bool_), zeros)

    k = max(1, int(round(n * outlier_fraction)))
    k = min(k, n - 1)  # always keep at least one inlier per column
    # Per-column top-k by magnitude, selected by *rank* rather than by comparing
    # against the k-th largest value. A value threshold (`mag >= thresh`) breaks
    # on ties: a constant column has every element equal to the threshold, so it
    # would flag all N elements as outliers instead of k — sending the whole
    # column to the fp16 side-channel while byte accounting charges only k.
    # argsort gives each element a distinct rank, so exactly k are selected.
    mag = mx.abs(x32)
    order = mx.argsort(-mag, axis=0)  # [N, D] indices, descending magnitude
    rank = mx.put_along_axis(
        mx.zeros_like(order),
        order,
        mx.broadcast_to(mx.arange(n, dtype=order.dtype)[:, None], order.shape),
        axis=0,
    )  # rank[i, d] = position of element i within column d's descending order
    mask = rank < k  # [N, D] bool — exactly k True per column
    col_mean = mx.mean(x32, axis=0, keepdims=True)  # [1, D]
    inliers = mx.where(mask, mx.broadcast_to(col_mean, x32.shape), x32)
    outlier_vals = mx.where(mask, x32, mx.zeros_like(x32))
    return DenseSparse(inliers, mask, outlier_vals)


def fit_nuq_levels(x: mx.array, bits: int, n_iters: int = 8) -> mx.array:
    """Fit ``2^bits`` non-uniform signpost levels per column via 1-D Lloyd-Max.

    Quantile initialization (deterministic) followed by ``n_iters`` assign/update
    sweeps. Distortion is monotone non-increasing across sweeps (Lloyd's lemma).
    Empty clusters retain their previous centroid (no NaNs).

    Args:
        x: [N, D] fp16 or fp32 (inliers only — outliers should be excluded first).
        bits: Bit-width; produces ``L = 2^bits`` levels.
        n_iters: Lloyd-Max iterations.

    Returns:
        levels: [L, D] fp32 ascending signpost levels per column.
    """
    x32 = x.astype(mx.float32)
    n, d = x32.shape
    L = 1 << bits

    # Quantile init: L evenly spaced quantiles per column → ascending, distinct-ish.
    xs = mx.sort(x32, axis=0)  # [N, D] ascending
    qpos = (mx.arange(L, dtype=mx.float32) + 0.5) / L  # [L] centers of L bins
    idx = mx.clip(mx.round(qpos * (n - 1)), 0, n - 1).astype(mx.int32)  # [L]
    levels = xs[idx, :]  # [L, D]

    for _ in range(max(1, n_iters)):
        # Assign: nearest level per element. dist[n, l, d] = |x - level|.
        # Vectorize over L with broadcasting: x[N,1,D] vs levels[1,L,D].
        diff = mx.abs(x32[:, None, :] - levels[None, :, :])  # [N, L, D]
        assign = mx.argmin(diff, axis=1)  # [N, D] in [0, L)

        # Update: per-(level, column) mean of assigned inliers.
        # one_hot[N, L, D] = (assign == l)
        lr = mx.arange(L)[None, :, None]  # [1, L, 1]
        one_hot = (assign[:, None, :] == lr).astype(mx.float32)  # [N, L, D]
        counts = mx.sum(one_hot, axis=0)  # [L, D]
        sums = mx.sum(one_hot * x32[:, None, :], axis=0)  # [L, D]
        new_levels = sums / mx.maximum(counts, 1.0)  # [L, D]
        # Empty clusters (count 0) keep the old level.
        levels = mx.where(counts > 0, new_levels, levels)
        mx.eval(levels)

    # Keep levels ascending per column for clean lookup semantics.
    return mx.sort(levels, axis=0)


def quantize_nuq(x: mx.array, levels: mx.array) -> mx.array:
    """Assign each element to its nearest signpost level index.

    Args:
        x: [N, D] fp16 or fp32.
        levels: [L, D] fp32 ascending levels per column.

    Returns:
        codes: [N, D] int32 indices into ``levels`` (column-wise).
    """
    x32 = x.astype(mx.float32)
    diff = mx.abs(x32[:, None, :] - levels[None, :, :])  # [N, L, D]
    return mx.argmin(diff, axis=1).astype(mx.int32)  # [N, D]


def dequant_nuq(codes: mx.array, levels: mx.array) -> mx.array:
    """Reconstruct fp16 [N, D] from NUQ codes via per-column table lookup.

    Args:
        codes: [N, D] int32 level indices.
        levels: [L, D] fp32 ascending levels per column.

    Returns:
        [N, D] fp16 reconstruction.
    """
    n, d = codes.shape
    L = levels.shape[0]
    # Gather levels[code[n,d], d] for each (n, d). take_along_axis over axis 0.
    lev = mx.take_along_axis(levels, codes.astype(mx.int32), axis=0)  # [N, D]
    return lev.astype(mx.float16)


def nuq_quant_dequant(
    x: mx.array,
    bits: int,
    outlier_fraction: float = 0.0,
    lloyd_iters: int = 8,
) -> mx.array:
    """Full NUQ + outlier-isolation reconstruct, signature-compatible with the
    suite's ``_group_quant_dequant`` (drop-in for existing wrappers).

    Args:
        x: [N, D] fp16 or fp32 (one quantization group — a head's keys/values).
        bits: NUQ bit-width.
        outlier_fraction: Top-magnitude fraction kept fp16 (0 = pure NUQ).
        lloyd_iters: Lloyd-Max iterations.

    Returns:
        [N, D] fp16 reconstruction.
    """
    ds = split_dense_sparse(x, outlier_fraction)
    levels = fit_nuq_levels(ds.inliers, bits, lloyd_iters)
    codes = quantize_nuq(ds.inliers, levels)
    recon = dequant_nuq(codes, levels).astype(mx.float32)
    # Scatter the fp16 outliers back over their positions.
    recon = mx.where(ds.outlier_mask, ds.outlier_vals, recon)
    return recon.astype(mx.float16)


def nuq_distortion(x: mx.array, levels: mx.array) -> float:
    """Mean squared NUQ reconstruction error (diagnostic for convergence tests)."""
    codes = quantize_nuq(x, levels)
    recon = dequant_nuq(codes, levels).astype(mx.float32)
    return float(mx.mean((x.astype(mx.float32) - recon) ** 2).item())


__all__ = [
    "DenseSparse",
    "split_dense_sparse",
    "fit_nuq_levels",
    "quantize_nuq",
    "dequant_nuq",
    "nuq_quant_dequant",
    "nuq_distortion",
]
