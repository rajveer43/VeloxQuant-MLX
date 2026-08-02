"""PALU quantizer — low-rank K *and* V projection with group-head decomposition.

Inspired by "PALU: Compressing KV-Cache with Low-Rank Projection"
(arXiv:2407.21118, ICLR 2025, Chang et al.).  Documented as
"PALU-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

What makes this PALU and not SVDq (already in the repo):

  * Both **keys and values** are low-rank projected (SVDq is keys-only,
    values fp16).
  * The cache stores the latent codes ``[S, r]`` *directly* — it never holds
    full fp16 ``[S, D]`` keys/values for storage; the parent fp16 buffer is
    bypassed entirely (see :class:`~veloxquant_mlx.cache.palu_cache.PALUKVCache`).
    SVDq reconstructs full fp16 keys and hands them to the parent cache, so its
    win is bandwidth-accounting only.  PALU stores ``[S, r]`` and reconstructs
    to fp16 *only* at attend time, so it also wins peak storage.
  * **Group-head low-rank decomposition (G-LRD):** heads are partitioned into
    ``n_head_groups`` groups; all heads in a group share one projection matrix
    fit from their concatenated keys (resp. values).  This is PALU's middle
    ground between a single whole-matrix projection (max compression, worst
    fidelity) and per-head projections (best fidelity, projection overhead
    grows with head count).
  * The stored latents are further **mixed-bit quantized** (reusing the SVDq
    latent quantizer) so the compression is ``(D / r) * (16 / avg_bits)`` on
    both tensors.

Algorithm:

  Prefill (first call, S > 1):
    1. For keys and (optionally) values independently:
       a. Partition the H heads into ``n_head_groups`` contiguous groups.
       b. For each group g, stack the group's heads along the token axis and
          compute a truncated SVD → projection ``V_g [D, r]`` + mean ``mu_g [D]``.
       c. Project every head in the group into its group's latent space and
          mixed-bit quantize the resulting ``[S, r]`` latents.
    2. Store ``V_g``, ``mu_g``, singular values per group, and the quantized
       latents as cache state.

  Decode (S == 1):
    1. Project the new key/value into the already-stored group projections.
    2. Mixed-bit quantize and append to the latent buffers.

This module holds the pure projection/reconstruction maths.  The cache wrapper
(:class:`PALUKVCache`) owns the per-group state and the true-latent storage.

Adaptation notes:
  - Rank defaults to ``energy_threshold = 0.90`` (a touch more aggressive than
    SVDq's 0.95, since PALU pays the rank cost on both tensors).
  - Values use the same group-head SVD path; set ``quantize_values=False`` at
    the cache level for a low-rank-only (fp16 latent) value path.
  - PALU's fused low-rank-reconstruction attention CUDA kernel is *not* ported:
    we reconstruct fp16 then call MLX SDPA.  Documented as a known simplification.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from veloxquant_mlx.quantizers.svdq import quantize_latents_mixed


def head_group_bounds(n_heads: int, n_groups: int) -> list[tuple[int, int]]:
    """Partition ``n_heads`` into ``n_groups`` contiguous ``[lo, hi)`` ranges.

    Groups are as even as possible; the first ``n_heads % n_groups`` groups
    take one extra head.  ``n_groups`` is clamped to ``[1, n_heads]``.

    Returns:
        List of ``(lo, hi)`` half-open head-index ranges, length ``<= n_groups``.
    """
    n_groups = max(1, min(int(n_groups), int(n_heads)))
    base = n_heads // n_groups
    rem = n_heads % n_groups
    bounds: list[tuple[int, int]] = []
    lo = 0
    for g in range(n_groups):
        size = base + (1 if g < rem else 0)
        bounds.append((lo, lo + size))
        lo += size
    return bounds


def group_head_svd(
    x_group: mx.array,
    rank: Optional[int] = None,
    energy_threshold: float = 0.90,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fit a shared low-rank projection for a group of heads.

    The group's heads are stacked along the token axis so a single projection
    captures the group's dominant subspace (PALU's group-head decomposition).

    Args:
        x_group: Shape ``[G, S, D]`` (G = heads in this group), fp16 or fp32.
        rank: Explicit latent rank ``r``.  If None, chosen by ``energy_threshold``.
        energy_threshold: Fraction of singular-value energy to retain when
            ``rank`` is None.

    Returns:
        ``(V, mu, singular_values)`` where
          V               — right singular vectors ``[D, r]`` fp32
          mu              — mean vector ``[D]`` fp32 (shared across the group)
          singular_values — ``[r]`` fp32, descending (for mixed-bit ranking)
    """
    G, S, D = x_group.shape
    x = x_group.astype(mx.float32).reshape(G * S, D)  # stack heads along tokens
    mu = mx.mean(x, axis=0)  # [D]
    x_centered = x - mu[None, :]

    U, s_vals, Vt = mx.linalg.svd(x_centered, stream=mx.cpu)
    mx.eval(U, s_vals, Vt)

    if rank is None:
        total = float(mx.sum(s_vals).item())
        if total < 1e-12:
            rank = 1
        else:
            cumsum = 0.0
            rank = int(s_vals.shape[0])
            for i, sv in enumerate(s_vals.tolist()):
                cumsum += sv
                if cumsum / total >= energy_threshold:
                    rank = i + 1
                    break
    rank = max(1, min(int(rank), int(s_vals.shape[0]), D))

    V = Vt[:rank, :].T  # [D, r]
    s_r = s_vals[:rank]  # [r]
    return V, mu, s_r


def project_to_latent(x: mx.array, V: mx.array, mu: mx.array) -> mx.array:
    """Project ``[S, D]`` into the latent space ``[S, r]``: ``(x - mu) @ V``."""
    return (x.astype(mx.float32) - mu[None, :]) @ V


def reconstruct_from_latent(L: mx.array, V: mx.array, mu: mx.array) -> mx.array:
    """Reconstruct ``[S, D]`` fp16 from latents: ``L @ V.T + mu``."""
    out = L.astype(mx.float32) @ V.T + mu[None, :]
    return out.astype(mx.float16)


def quantize_latent(
    L: mx.array,
    singular_values: mx.array,
    hi_bit: int,
    lo_bit: int,
    hi_fraction: float,
    group_size: int,
) -> mx.array:
    """Mixed-bit quantize latents ``[S, r]`` → reconstructed fp16 latents ``[S, r]``.

    Thin wrapper over :func:`veloxquant_mlx.quantizers.svdq.quantize_latents_mixed`
    so PALU reuses the same, already-tested latent coder.  Top-``hi_fraction``
    channels by singular value get ``hi_bit``; the rest get ``lo_bit``.
    """
    return quantize_latents_mixed(
        L,
        singular_values,
        hi_bit=hi_bit,
        lo_bit=lo_bit,
        hi_fraction=hi_fraction,
        group_size=group_size,
    )


__all__ = [
    "head_group_bounds",
    "group_head_svd",
    "project_to_latent",
    "reconstruct_from_latent",
    "quantize_latent",
]
