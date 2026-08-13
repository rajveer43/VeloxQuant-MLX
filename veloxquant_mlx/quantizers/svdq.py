"""SVDq quantizer — sub-2-bit key compression via offline SVD + mixed-precision latent coding.

Inspired by "SVDq: Singular Value Decomposition-based KV Cache Quantization"
(arXiv:2502.15304, Feb 2025, unreviewed preprint).

Algorithm:
  Prefill phase (triggered once when a batch of keys arrives):
    1. Compute truncated SVD of the key matrix K ∈ R^{S×D}: K ≈ U·Σ_r·V^H + K̄
       where K̄ = mean(K, axis=0) is subtracted before decomposition.
    2. Store V (right singular vectors, shape [D, r]) and K̄ (mean key, [D])
       as layer attributes.
    3. Project keys into latent space: L = (K - K̄) @ V  →  shape [S, r]
    4. Apply mixed-precision quantization per latent channel:
       top-25% by singular value magnitude → 4-bit (TurboQuant group quant)
       remaining 75%                       → 2-bit (KIVI group quant)

  Decode phase (per new token):
    1. Project new key: l = (k - K̄) @ V  →  shape [1, r]
    2. Quantize l with the same mixed-bit scheme.
    3. On fetch, reconstruct: K_hat = L_dequant @ V^H + K̄

This module implements the per-token latent quantizer.  The cache wrapper
(SVDqKVCache) owns the SVD state and orchestrates prefill vs decode.

Adaptation notes:
  - Rank defaults to energy_threshold=0.95 (retain ≥95% singular value energy)
    rather than a fixed d//4 — this is more robust across models.
  - Mixed-bit routing uses singular value magnitudes computed at prefill.
  - Values are left at fp16 (the paper notes values have weak low-rank structure).
  - Documented as "SVDq-adapted" — not a faithful port; numbers come from
    committed results.json, not paper claims.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from veloxquant_mlx.core.abstractions import ArtifactStore
from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant


def svd_compress_keys(
    keys: mx.array,
    rank: Optional[int] = None,
    energy_threshold: float = 0.95,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Compute truncated SVD of a key matrix and return latents + projection.

    Args:
        keys: Shape [S, D], fp16 or fp32.
        rank: Explicit rank r. If None, determined by energy_threshold.
        energy_threshold: Fraction of total singular value energy to retain.

    Returns:
        (L, V, K_mean, singular_values) where:
          L          — latent codes [S, r] fp32
          V          — right singular vectors [D, r] fp32
          K_mean     — mean key [D] fp32
          singular_values — [r] fp32, descending
    """
    x = keys.astype(mx.float32)
    K_mean = mx.mean(x, axis=0)  # [D]
    x_centered = x - K_mean[None, :]  # [S, D]

    # MLX svd returns (U, S, Vt) with Vt shape [D, D] (economy=False)
    # or [min(S,D), D] — use economy form via mx.linalg.svd
    U, S_vals, Vt = mx.linalg.svd(x_centered, stream=mx.cpu)
    # Vt: [min(S,D), D] → V = Vt.T: [D, min(S,D)]
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
    rank = min(rank, int(S_vals.shape[0]), keys.shape[-1])

    V = Vt[:rank, :].T  # [D, r]
    s_r = S_vals[:rank]  # [r]
    L = x_centered @ V  # [S, r]
    return L, V, K_mean, s_r


def quantize_latents_mixed(
    L: mx.array,
    singular_values: mx.array,
    hi_bit: int = 4,
    lo_bit: int = 2,
    hi_fraction: float = 0.25,
    group_size: int = 32,
) -> mx.array:
    """Mixed-precision quantization of latent codes.

    Top-``hi_fraction`` latent channels (by singular value magnitude) use
    ``hi_bit``-bit quantization; the rest use ``lo_bit``-bit.

    Args:
        L: Latent codes [S, r] fp32.
        singular_values: [r] fp32, descending (used for channel ranking).
        hi_bit: Bits for high-importance channels.
        lo_bit: Bits for low-importance channels.
        hi_fraction: Fraction of channels routed to hi_bit.
        group_size: Group size for quantization along the token axis.

    Returns:
        Reconstructed latents [S, r] fp16.
    """
    S, r = L.shape
    n_hi = max(1, int(r * hi_fraction))
    # Channels with highest singular value magnitude → hi_bit
    sv_np = singular_values.tolist()
    sorted_idx = sorted(range(r), key=lambda i: -sv_np[i])
    hi_idx = sorted(sorted_idx[:n_hi])
    lo_idx = sorted(sorted_idx[n_hi:])

    recon = mx.zeros_like(L).astype(mx.float16)
    if hi_idx:
        L_hi = L[:, hi_idx]
        recon_hi = _group_quant_dequant(L_hi, hi_bit, group_size)
        recon = mx.scatter(recon, hi_idx, recon_hi, axis=1) if False else recon
        # Build via concatenation (scatter not directly available)
        parts = list(recon.T)
        for new_col_idx, orig_col_idx in enumerate(hi_idx):
            parts[orig_col_idx] = recon_hi[:, new_col_idx]
        if lo_idx:
            L_lo = L[:, lo_idx]
            recon_lo = _group_quant_dequant(L_lo, lo_bit, group_size)
            for new_col_idx, orig_col_idx in enumerate(lo_idx):
                parts[orig_col_idx] = recon_lo[:, new_col_idx]
        recon = mx.stack(parts, axis=1).astype(mx.float16)
    elif lo_idx:
        recon = _group_quant_dequant(L, lo_bit, group_size)
    return recon


def reconstruct_keys(
    L_q: mx.array,
    V: mx.array,
    K_mean: mx.array,
) -> mx.array:
    """Reconstruct full key matrix from quantized latents.

    Args:
        L_q: Quantized latents [S, r] fp16.
        V: Right singular vectors [D, r] fp32.
        K_mean: Mean key [D] fp32.

    Returns:
        Reconstructed keys [S, D] fp16.
    """
    K_hat = L_q.astype(mx.float32) @ V.T + K_mean[None, :]
    return K_hat.astype(mx.float16)


__all__ = [
    "svd_compress_keys",
    "quantize_latents_mixed",
    "reconstruct_keys",
    "_group_quant_dequant",
]
