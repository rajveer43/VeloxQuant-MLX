"""VecInfer building blocks: smooth calibration, Walsh-Hadamard, product VQ.

Implements the algorithmic core of VecInfer (arxiv:2510.06175, Yao et al.
2025) for KV cache compression on MLX. The paper's CUDA kernel fusion is
not portable to Metal, so this module provides only the algorithmic
primitives; standard mlx_lm SDPA handles the math after dequantization.

Pipeline (key cache):
    1. Calibrate per-(head, channel) smooth factor lambda offline from a
       representative key sample (Eq. 4).
    2. Build a Walsh-Hadamard matrix H of size head_dim x head_dim (Eq. 5).
    3. Train a product VQ codebook on smooth+Hadamard-transformed keys.
    4. At inference: K_tilde = (K / lambda) @ H, quantize K_tilde via the
       codebook; queries get the inverse transform q_tilde = (q * lambda) @ H
       so q_tilde @ K_tilde.T == q @ K.T (Eq. 7).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# 1a. Smooth factor calibration
# ---------------------------------------------------------------------------
def calibrate_smooth_factors(keys_calib: mx.array, eps: float = 1e-4) -> mx.array:
    """Compute per-(head, channel) smooth scaling factor from a key sample.

    Args:
        keys_calib: Calibration keys with shape ``[n_tokens, n_heads, head_dim]``
            or ``[n_tokens, head_dim]`` (single-head).
        eps: Floor for the per-channel max to avoid divide-by-zero.

    Returns:
        Smooth factors shaped to match the input head layout
        (``[n_heads, head_dim]`` or ``[head_dim]``). Formula::

            lambda_i = sqrt(max_t |K[t, ..., i]|)
    """
    if keys_calib.ndim == 2:
        # [n_tokens, head_dim]
        max_abs = mx.max(mx.abs(keys_calib), axis=0)
    elif keys_calib.ndim == 3:
        # [n_tokens, n_heads, head_dim]
        max_abs = mx.max(mx.abs(keys_calib), axis=0)
    else:
        raise ValueError(
            f"calibrate_smooth_factors: keys_calib must be 2D or 3D, got shape {keys_calib.shape}"
        )
    max_abs = mx.maximum(max_abs, mx.array(eps, dtype=max_abs.dtype))
    return mx.sqrt(max_abs)


# ---------------------------------------------------------------------------
# 1b. Walsh-Hadamard matrix
# ---------------------------------------------------------------------------
def walsh_hadamard_matrix(d: int, dtype=mx.float32) -> mx.array:
    """Construct an orthonormal Walsh-Hadamard matrix.

    Recursive form from VecInfer Eq. 5::

        H_1 = [[1]]
        H_{2k} = (1/sqrt(2)) * [[H_k, H_k], [H_k, -H_k]]

    Args:
        d: Output dimension; must be a power of 2.
        dtype: MLX dtype for the returned matrix.

    Returns:
        Matrix of shape ``[d, d]`` with ``H @ H.T == I``.
    """
    if d < 1 or (d & (d - 1)) != 0:
        raise ValueError(f"walsh_hadamard_matrix: d={d} must be a power of 2.")

    # Build in numpy for stability, then move to MLX.
    H = np.array([[1.0]], dtype=np.float32)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    size = 1
    while size < d:
        top = np.concatenate([H, H], axis=1)
        bot = np.concatenate([H, -H], axis=1)
        H = np.concatenate([top, bot], axis=0) * inv_sqrt2
        size *= 2
    return mx.array(H).astype(dtype)


# ---------------------------------------------------------------------------
# 1c. Dual transformation
# ---------------------------------------------------------------------------
def apply_dual_transform_keys(K: mx.array, smooth: mx.array, H: mx.array) -> mx.array:
    """Apply ``K_tilde = (K / lambda) @ H`` (smooth then Hadamard).

    Args:
        K: Keys, shape ``[..., head_dim]``. Per-head smoothing assumes the
            second-to-last axis indexes heads when ``smooth`` is 2D.
        smooth: Either ``[head_dim]`` (broadcast across heads) or
            ``[n_heads, head_dim]`` (per-head); see
            :func:`calibrate_smooth_factors`.
        H: Walsh-Hadamard matrix ``[head_dim, head_dim]``.

    Returns:
        Transformed keys, same shape as input.
    """
    if smooth.ndim == 1:
        K_sm = K / smooth.astype(K.dtype)
    elif smooth.ndim == 2:
        # smooth: [n_heads, head_dim]
        # K: [..., head_dim] (last axis is always head_dim).
        # The head axis is always -3, whether K is 3D [H, S, D] (single
        # batch item) or 4D [B, H, S, D] -- checking K.ndim >= 4 used ndim
        # as a proxy for "has a head axis" and silently fell through to the
        # averaging branch for genuine 3D per-head input (#74). Require
        # only that a -3 axis exists (ndim >= 3) and that it matches
        # smooth's head count.
        if K.ndim >= 3 and K.shape[-3] == smooth.shape[0]:
            sm = smooth[:, None, :].astype(K.dtype)
            K_sm = K / sm
        elif K.shape[-1] == smooth.shape[-1]:
            # Head count mismatch (e.g. GQA: smooth calibrated on Q heads
            # but K has fewer KV heads). Average smooth across head axis to
            # get a single [head_dim] factor that still suppresses the
            # outlier channels.
            sm_1d = mx.mean(smooth, axis=0).astype(K.dtype)
            K_sm = K / sm_1d
        else:
            raise ValueError(
                f"apply_dual_transform_keys: cannot broadcast smooth {smooth.shape} to K {K.shape}"
            )
    else:
        raise ValueError(f"smooth must be 1D or 2D, got {smooth.shape}")
    return K_sm @ H.astype(K.dtype)


def apply_dual_transform_queries(q: mx.array, smooth: mx.array, H: mx.array) -> mx.array:
    """Apply ``q_tilde = (q * lambda) @ H`` so q_tilde @ K_tilde.T == q @ K.T.

    Args:
        q: Queries, shape ``[..., head_dim]``.
        smooth: Same convention as :func:`apply_dual_transform_keys`.
        H: Walsh-Hadamard matrix ``[head_dim, head_dim]``.

    Returns:
        Transformed queries, same shape as input.
    """
    if smooth.ndim == 1:
        q_sm = q * smooth.astype(q.dtype)
    elif smooth.ndim == 2:
        # See apply_dual_transform_keys for why this checks ndim >= 3, not
        # >= 4 (#74).
        if q.ndim >= 3 and q.shape[-3] == smooth.shape[0]:
            sm = smooth[:, None, :].astype(q.dtype)
            q_sm = q * sm
        elif q.shape[-1] == smooth.shape[-1]:
            sm_1d = mx.mean(smooth, axis=0).astype(q.dtype)
            q_sm = q * sm_1d
        else:
            raise ValueError(
                f"apply_dual_transform_queries: cannot broadcast smooth "
                f"{smooth.shape} to q {q.shape}"
            )
    else:
        raise ValueError(f"smooth must be 1D or 2D, got {smooth.shape}")
    return q_sm @ H.astype(q.dtype)


# ---------------------------------------------------------------------------
# 1d. Product vector quantization
# ---------------------------------------------------------------------------
def _kmeans_lloyd(
    x: np.ndarray, n_centroids: int, max_iter: int = 30, seed: int = 42
) -> np.ndarray:
    """Pure-numpy Lloyd's k-means.

    Args:
        x: ``[n_samples, sub_dim]`` float32.
        n_centroids: ``2**b`` codebook size.
        max_iter: Iteration cap.
        seed: RNG seed for initial centroid selection.

    Returns:
        ``[n_centroids, sub_dim]`` float32 centroids.
    """
    rng = np.random.default_rng(seed)
    n_samples, _ = x.shape
    if n_samples <= n_centroids:
        # Pad by replication so we always have enough samples for init.
        reps = (n_centroids // n_samples) + 2
        x = np.tile(x, (reps, 1))[: n_centroids * 2]
        n_samples = x.shape[0]

    # k-means++ light: random unique sample init
    init_idx = rng.choice(n_samples, size=n_centroids, replace=False)
    centroids = x[init_idx].copy()

    # Chunked assignment for memory safety
    chunk = max(1024, min(16384, n_samples))
    prev_inertia = np.inf
    for _ in range(max_iter):
        # Assign step
        labels = np.empty(n_samples, dtype=np.int32)
        for start in range(0, n_samples, chunk):
            stop = min(start + chunk, n_samples)
            # [chunk, n_centroids] distance
            diff = x[start:stop, None, :] - centroids[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diff, diff)
            labels[start:stop] = np.argmin(d2, axis=1)

        # Update step
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(n_centroids, dtype=np.int64)
        np.add.at(new_centroids, labels, x)
        np.add.at(counts, labels, 1)
        empty = counts == 0
        if empty.any():
            # Reseed empties from random samples
            reseed_idx = rng.choice(n_samples, size=int(empty.sum()), replace=False)
            new_centroids[empty] = x[reseed_idx]
            counts[empty] = 1
        new_centroids = new_centroids / counts[:, None].astype(new_centroids.dtype)

        # Inertia (for convergence check)
        inertia = 0.0
        for start in range(0, n_samples, chunk):
            stop = min(start + chunk, n_samples)
            diff = x[start:stop] - new_centroids[labels[start:stop]]
            inertia += float(np.sum(diff * diff))

        if abs(prev_inertia - inertia) < 1e-6 * max(prev_inertia, 1.0):
            centroids = new_centroids
            break
        centroids = new_centroids
        prev_inertia = inertia
    return centroids.astype(np.float32)


def train_codebook(x: mx.array, n_centroids: int, max_iter: int = 30, seed: int = 42) -> mx.array:
    """Train a VQ codebook on flat sub-vector samples.

    Args:
        x: Flat ``[n_samples, sub_dim]`` MLX array of training sub-vectors.
        n_centroids: Codebook size ``2**b``.
        max_iter: K-means iteration cap.
        seed: RNG seed.

    Returns:
        Codebook of shape ``[n_centroids, sub_dim]`` as an MLX float32 array.
    """
    if x.ndim != 2:
        raise ValueError(f"train_codebook: x must be 2D, got {x.shape}")
    x_np = np.asarray(x.astype(mx.float32))
    codebook_np = _kmeans_lloyd(x_np, n_centroids, max_iter=max_iter, seed=seed)
    return mx.array(codebook_np)


def quantize_vq(x: mx.array, codebook: mx.array, sub_dim: int) -> mx.array:
    """Encode ``x`` as nearest-centroid indices in a product VQ scheme.

    Args:
        x: ``[..., D]`` where ``D`` is divisible by ``sub_dim``.
        codebook: ``[n_centroids, sub_dim]``.
        sub_dim: Sub-vector dimension.

    Returns:
        ``[..., D // sub_dim]`` int32 indices into the codebook.
    """
    *leading, D = x.shape
    if D % sub_dim != 0:
        raise ValueError(f"quantize_vq: D={D} not divisible by sub_dim={sub_dim}")
    n_sub = D // sub_dim
    # Reshape to [..., n_sub, sub_dim] then flatten leading dims for batched
    # nearest-centroid search.
    x_sub = x.reshape(*leading, n_sub, sub_dim)
    flat = x_sub.reshape(-1, sub_dim)  # [N * n_sub, sub_dim]
    cb = codebook.astype(flat.dtype)

    n_centroids = cb.shape[0]
    # Chunked argmin to keep memory bounded for large codebooks
    chunk = max(1, 1_000_000 // max(n_centroids, 1))
    n_flat = flat.shape[0]
    out = mx.zeros((n_flat,), dtype=mx.int32)
    for start in range(0, n_flat, chunk):
        stop = min(start + chunk, n_flat)
        sub = flat[start:stop]  # [c, sub_dim]
        # [c, n_centroids]
        diff = sub[:, None, :] - cb[None, :, :]
        d2 = mx.sum(diff * diff, axis=-1)
        idx = mx.argmin(d2, axis=-1).astype(mx.int32)
        if start == 0 and stop == n_flat:
            out = idx
        else:
            # Build via concat — small number of chunks expected
            if start == 0:
                out = idx
            else:
                out = mx.concatenate([out, idx], axis=0)
    return out.reshape(*leading, n_sub)


def dequantize_vq(indices: mx.array, codebook: mx.array) -> mx.array:
    """Reconstruct vectors from codebook indices.

    Args:
        indices: ``[..., n_sub]`` int32 codebook indices.
        codebook: ``[n_centroids, sub_dim]``.

    Returns:
        ``[..., n_sub * sub_dim]`` reconstruction.
    """
    sub_dim = codebook.shape[-1]
    *leading, n_sub = indices.shape
    flat = indices.reshape(-1).astype(mx.int32)
    gathered = mx.take(codebook, flat, axis=0)  # [N*n_sub, sub_dim]
    return gathered.reshape(*leading, n_sub * sub_dim)


# ---------------------------------------------------------------------------
# 1e. LUT-based query precomputation (optional fast path)
# ---------------------------------------------------------------------------
def compute_query_lut(q_tilde: mx.array, codebook: mx.array, sub_dim: int) -> mx.array:
    """Precompute ``q_sub @ codebook.T`` so attention can be scored via lookup.

    Args:
        q_tilde: Transformed query ``[..., D]``.
        codebook: ``[n_centroids, sub_dim]``.
        sub_dim: Must match codebook width.

    Returns:
        LUT of shape ``[..., n_sub, n_centroids]``. Attention score for a
        token with indices ``[n_sub]`` is ``lut[token, range(n_sub), idx].sum()``.
    """
    *leading, D = q_tilde.shape
    if D % sub_dim != 0:
        raise ValueError(f"compute_query_lut: D={D} not divisible by sub_dim={sub_dim}")
    n_sub = D // sub_dim
    q_sub = q_tilde.reshape(*leading, n_sub, sub_dim)
    # [..., n_sub, n_centroids]
    return mx.matmul(q_sub, codebook.astype(q_sub.dtype).T)


__all__ = [
    "calibrate_smooth_factors",
    "walsh_hadamard_matrix",
    "apply_dual_transform_keys",
    "apply_dual_transform_queries",
    "train_codebook",
    "quantize_vq",
    "dequantize_vq",
    "compute_query_lut",
]
