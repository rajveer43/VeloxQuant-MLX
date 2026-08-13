"""xKV quantizer — cross-layer *shared-subspace* key compression via joint SVD.

Inspired by "xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector
Extraction" (Chang, Lin, Lin, Chiang, Akhauri, Dai, Jiang, Li, Ceze, Wu,
Abdelfattah — arXiv:2503.18893, preprint; code at
https://github.com/abdelfattah-lab/xKV). Documented as "xKV-adapted
(VeloxQuant-MLX implementation)" — not a faithful port.

How it differs from the repo's other two cross-layer methods:
  * XQuant **reuses quantized codes**: an anchor layer's integer codes are
    shared; each reuse layer refits its own scale/zero to those codes.
  * MiniCache **merges tensors pairwise**: two layers' KV directions are
    SLERP-blended into one shared direction per token.
  * xKV **jointly factorizes a *group* of layers into one shared low-rank
    basis**: rather than pairing tensors or codes, it stacks several layers'
    key matrices along the token axis and computes a *single* truncated SVD
    over the stack. The resulting basis (right singular vectors + mean) is
    amortized across every member of the group; each layer then stores only
    its own latent coordinates in that shared basis.

Core mechanism (per group, once, at prefill):
  1. Center and stack N layers' [S, D] key matrices into one [N*S, D] matrix.
  2. Truncated SVD of the stack → shared basis V_g [D, r] + shared mean
     K_mean_g [D] (the mean over *all* stacked rows, not per layer).
  3. Each layer projects its *own* keys into V_g: L_i = (K_i - K_mean_g) @ V_g.
  4. Latent codes are quantized at a single bit-width (default 4-bit) via the
     existing group-quant primitive — xKV's distinguishing feature is the
     shared basis, not a novel bit-allocation scheme (unlike SVDq's mixed-bit
     latent routing, which this module does not duplicate but can compose
     with via ``veloxquant_mlx.quantizers.svdq.quantize_latents_mixed``).

This module holds the pure numerics. Cross-layer coordination (which layers
form a group, who publishes first, broadcasting the shared basis back to every
member) is handled by
:class:`~veloxquant_mlx.cache.xkv_coordinator.XKVCoordinator`.

What we do NOT implement (see ``paper/NEW_METHOD_SURVEY_V10.md`` for the full
rationale):
  * CKA-based automatic layer grouping — groups are fixed-size and contiguous.
  * "Selective Reconstruction" — the paper's decode-time latency optimization
    (exactly reconstruct a subset of layers, derive the rest). We fully
    reconstruct every layer on every fetch, like every other wrapper here.
  * Values — keys only, mirroring SVDq's precedent in this repo.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant


def pair_layers_grouped(n_layers: int, group_size: int) -> list[tuple[int, int, int]]:
    """Assign each layer a position within a fixed-size contiguous group.

    Layers are chunked into contiguous groups of ``group_size``. A trailing
    partial group (fewer than ``group_size`` layers) is still a valid,
    smaller group — its ``group_size_actual`` reflects the true member count
    so byte-accounting and the coordinator's "wait for N members" logic stay
    correct.

    Args:
        n_layers: Number of attention-bearing layers.
        group_size: Layers per group (>=1). 1 → every layer is its own
            degenerate group of size 1 (standalone per-layer SVD, no sharing).

    Returns:
        Length-``n_layers`` list of ``(member_idx, group_id, group_size_actual)``
        where ``member_idx`` is this layer's 0-indexed position within its
        group (0 = leader, by convention the first layer of each group).
    """
    if group_size < 1:
        raise ValueError(f"pair_layers_grouped: group_size must be >= 1, got {group_size}.")
    roles: list[tuple[int, int, int]] = []
    i = 0
    group_id = 0
    while i < n_layers:
        remaining = n_layers - i
        actual = min(group_size, remaining)
        for member_idx in range(actual):
            roles.append((member_idx, group_id, actual))
        i += actual
        group_id += 1
    return roles


def joint_svd_compress(
    key_stack: list[mx.array],
    rank: Optional[int] = None,
    energy_threshold: float = 0.95,
) -> tuple[mx.array, mx.array, mx.array]:
    """Jointly factorize N layers' key matrices into one shared basis.

    Stacks the N ``[S, D]`` matrices along the token axis into one
    ``[N*S, D]`` matrix, centers by the *shared* mean, and computes a single
    truncated SVD. With ``N == 1`` this reduces to a plain single-matrix SVD
    (the group-of-1 degenerate case).

    Args:
        key_stack: N arrays, each shape ``[S, D]`` (same S and D across all;
            same token range, different layers), fp16 or fp32.
        rank: Explicit shared rank r. If None, determined by
            ``energy_threshold``.
        energy_threshold: Fraction of total singular value energy to retain
            when ``rank`` is None.

    Returns:
        ``(V_g, K_mean_g, singular_values)``:
          V_g              — shared right singular vectors ``[D, r]`` fp32
          K_mean_g         — shared mean key ``[D]`` fp32 (mean over all
                              stacked rows, i.e. over every layer's tokens)
          singular_values  — ``[r]`` fp32, descending
    """
    if not key_stack:
        raise ValueError("joint_svd_compress: key_stack must be non-empty.")
    D = key_stack[0].shape[-1]
    stacked = mx.concatenate([k.astype(mx.float32) for k in key_stack], axis=0)  # [N*S, D]

    K_mean_g = mx.mean(stacked, axis=0)  # [D]
    centered = stacked - K_mean_g[None, :]  # [N*S, D]

    U, S_vals, Vt = mx.linalg.svd(centered, stream=mx.cpu)
    mx.eval(U, S_vals, Vt)

    if rank is None:
        total_energy = float(mx.sum(S_vals).item())
        if total_energy < 1e-12:
            rank = 1
        else:
            cumsum = 0.0
            rank = len(S_vals)
            for i, sv in enumerate(S_vals.tolist()):
                cumsum += sv
                if cumsum / total_energy >= energy_threshold:
                    rank = i + 1
                    break
    rank = min(rank, int(S_vals.shape[0]), D)

    V_g = Vt[:rank, :].T  # [D, r]
    s_r = S_vals[:rank]  # [r]
    return V_g, K_mean_g, s_r


def project_into_shared_basis(
    keys: mx.array,
    V_g: mx.array,
    K_mean_g: mx.array,
) -> mx.array:
    """Project one layer's own keys into an already-computed shared basis.

    Args:
        keys: This layer's own keys, shape ``[S, D]``, fp16 or fp32.
        V_g: Shared right singular vectors ``[D, r]`` fp32.
        K_mean_g: Shared mean key ``[D]`` fp32.

    Returns:
        Latent codes ``[S, r]`` fp32.
    """
    x = keys.astype(mx.float32)
    centered = x - K_mean_g[None, :]
    return centered @ V_g


def reconstruct_from_shared_basis(
    L_q: mx.array,
    V_g: mx.array,
    K_mean_g: mx.array,
) -> mx.array:
    """Inverse of :func:`project_into_shared_basis`.

    Args:
        L_q: Quantized latent codes ``[S, r]``, fp16 or fp32.
        V_g: Shared right singular vectors ``[D, r]`` fp32.
        K_mean_g: Shared mean key ``[D]`` fp32.

    Returns:
        Reconstructed keys ``[S, D]`` fp16.
    """
    K_hat = L_q.astype(mx.float32) @ V_g.T + K_mean_g[None, :]
    return K_hat.astype(mx.float16)


def quantize_latents_uniform(
    L: mx.array,
    bits: int = 4,
    group_size: int = 32,
) -> mx.array:
    """Single-bit-width latent quantization (xKV's default path).

    A thin wrapper over the shared group-quant primitive. xKV's distinguishing
    feature is the *shared basis*, not a novel bit-allocation scheme — unlike
    SVDq's mixed-bit latent routing (top-25% channels at a higher bit-width).
    Callers who want mixed-bit latent coding on top of the shared basis should
    import and reuse ``veloxquant_mlx.quantizers.svdq.quantize_latents_mixed``
    directly rather than duplicating that logic here.

    Args:
        L: Latent codes ``[S, r]`` fp32.
        bits: Bit width for every latent channel.
        group_size: Group size for quantization along the token axis.

    Returns:
        Reconstructed latents ``[S, r]`` fp16.
    """
    return _group_quant_dequant(L, bits, group_size)


__all__ = [
    "pair_layers_grouped",
    "joint_svd_compress",
    "project_into_shared_basis",
    "reconstruct_from_shared_basis",
    "quantize_latents_uniform",
]
