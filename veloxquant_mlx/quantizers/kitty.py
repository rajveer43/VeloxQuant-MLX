"""Kitty quantizer — dynamic channel-wise mixed-precision key quantization.

Inspired by "Kitty: Plug-and-Play Continuous Batching with Dynamic Token
Selection" (arXiv:2511.18643, Nov 2025, unreviewed preprint). Documented as
"Kitty-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

Algorithm:
    For a given key tensor K ∈ R^{S×D}:
      1. Compute per-channel sensitivity as the variance across the sequence
         dimension: σ²_j = Var(K[:, j])  for j in 0..D-1.
      2. Rank channels by σ² descending. Top hi_fraction → hi_bit quantization.
         Remaining channels → lo_bit quantization.
      3. Each channel group is independently quantized with asymmetric min/max
         group quantization (reusing _group_quant_dequant).
      4. The full key tensor is reconstructed in fp16 for downstream SDPA.

    Operating in the original key space (no projection) makes Kitty zero-
    calibration: no SVD, no codebook training, sensitivity computed on-the-fly
    from the incoming K tensor itself.

Effective bit-width:
    avg_bits = hi_fraction × hi_bit + (1 − hi_fraction) × lo_bit
             = 0.25 × 4 + 0.75 × 2  =  2.5 bits/element  (default settings)

Adaptation notes:
    - Paper uses static offline sensitivity ranking; we compute it online from
      each incoming key batch (prefill uses full batch variance; decode uses a
      running accumulator updated per token).
    - Values are left at fp16 — Kitty is a key-only method.
    - Static calibration path is NOT implemented (deferred; no calibration
      dataset requirement keeps the cache-only contract intact).
"""

from __future__ import annotations

import mlx.core as mx

from veloxquant_mlx.quantizers._quant_utils import (
    _group_quant_dequant,
    _group_quant_dequant_batched,
)


def rank_channels_by_sensitivity(
    keys: mx.array,
    hi_fraction: float = 0.25,
) -> tuple[list[int], list[int]]:
    """Rank key channels by variance; return hi/lo channel index splits.

    Args:
        keys: Shape [S, D] fp16 or fp32.  S is sequence length, D is head dim.
        hi_fraction: Fraction of channels routed to high-bit quantization.

    Returns:
        (hi_indices, lo_indices) — sorted ascending lists of column indices.
        hi_indices: top hi_fraction channels by variance.
        lo_indices: remaining channels.
    """
    D = keys.shape[-1]
    variance = mx.var(keys.astype(mx.float32), axis=0)  # [D]
    mx.eval(variance)
    var_list = variance.tolist()
    sorted_idx = sorted(range(D), key=lambda i: -var_list[i])
    n_hi = max(1, int(D * hi_fraction))
    hi_idx = sorted(sorted_idx[:n_hi])
    lo_idx = sorted(sorted_idx[n_hi:])
    return hi_idx, lo_idx


def quantize_mixed_channels(
    keys: mx.array,
    hi_indices: list[int],
    lo_indices: list[int],
    hi_bit: int = 4,
    lo_bit: int = 2,
    group_size: int = 32,
) -> mx.array:
    """Apply mixed-precision quantization per channel group.

    Hi channels (high variance) get hi_bit, lo channels get lo_bit.
    Quantization is asymmetric min/max grouped along the sequence axis.

    Args:
        keys: [S, D] fp16 or fp32.
        hi_indices: Column indices to quantize at hi_bit.
        lo_indices: Column indices to quantize at lo_bit.
        hi_bit: Bit width for high-sensitivity channels.
        lo_bit: Bit width for low-sensitivity channels.
        group_size: Group size for quantization along axis 0.

    Returns:
        Reconstructed keys [S, D] fp16.
    """
    S, D = keys.shape
    parts: list[mx.array] = list(keys.astype(mx.float16).T)  # D × [S]

    if hi_indices:
        K_hi = keys[:, hi_indices]
        recon_hi = _group_quant_dequant(K_hi, hi_bit, group_size)  # [S, n_hi]
        for new_col, orig_col in enumerate(hi_indices):
            parts[orig_col] = recon_hi[:, new_col]

    if lo_indices:
        K_lo = keys[:, lo_indices]
        recon_lo = _group_quant_dequant(K_lo, lo_bit, group_size)  # [S, n_lo]
        for new_col, orig_col in enumerate(lo_indices):
            parts[orig_col] = recon_lo[:, new_col]

    return mx.stack(parts, axis=1).astype(mx.float16)  # [S, D]


def hi_mask_from_variance_batched(variance: mx.array, hi_fraction: float) -> mx.array:
    """Batched-leading-axis channel ranking: variance ``[N, D]`` -> hi-bit mask ``[N, D]`` bool.

    Replaces the per-row ``.tolist()`` + Python ``sorted()`` ranking in
    :func:`rank_channels_by_sensitivity` / the decode branch of
    ``KittyKVCache._quantize_keys`` with one host-sync-free ``mx.argsort``
    call across all ``N`` rows (e.g. flattened ``B*H``) at once. ``mx.argsort``
    breaks ties the same way Python's stable ``sorted()`` does (first index
    wins), so the selected channel set matches exactly.

    Args:
        variance: ``[N, D]`` fp32 per-row, per-channel variance.
        hi_fraction: Fraction of channels routed to high-bit quantization.

    Returns:
        ``[N, D]`` bool mask, True where the channel is in that row's
        top-``hi_fraction`` (by variance, descending; ties broken by lower
        channel index, matching Python's stable sort).
    """
    n, d = variance.shape
    n_hi = max(1, int(d * hi_fraction))
    order = mx.argsort(-variance, axis=-1)  # descending, stable on ties
    rank = mx.argsort(order, axis=-1)  # rank[i, j] = position of channel j
    return rank < n_hi


def quantize_mixed_channels_batched(
    keys: mx.array,
    hi_mask: mx.array,
    hi_bit: int = 4,
    lo_bit: int = 2,
    group_size: int = 32,
) -> mx.array:
    """Batched-leading-axis equivalent of :func:`quantize_mixed_channels`.

    Quantizes the full ``[N, S, D]`` tensor at ``hi_bit`` and at ``lo_bit``
    independently (each a single batched group-quant call, see
    :func:`_group_quant_dequant_batched`), then selects per-row, per-channel
    between the two reconstructions with ``hi_mask`` — avoiding a ragged
    gather/scatter over per-row index lists of different composition.
    Verified bit-for-bit equivalent to looping :func:`quantize_mixed_channels`
    over axis 0 with the corresponding ``hi_indices``/``lo_indices``.

    Args:
        keys: ``[N, S, D]`` fp16 or fp32.
        hi_mask: ``[N, D]`` bool, True where that row's channel is hi-bit
            (as returned by :func:`hi_mask_from_variance_batched`).
        hi_bit: Bit width for high-sensitivity channels.
        lo_bit: Bit width for low-sensitivity channels.
        group_size: Group size for quantization along the sequence axis.

    Returns:
        Reconstructed keys ``[N, S, D]`` fp16.
    """
    recon_hi = _group_quant_dequant_batched(keys, hi_bit, group_size)
    recon_lo = _group_quant_dequant_batched(keys, lo_bit, group_size)
    mask = hi_mask[:, None, :]  # [N, 1, D] broadcasts over S
    return mx.where(mask, recon_hi, recon_lo).astype(mx.float16)


def compute_running_variance(
    key_sum: mx.array,
    key_sq_sum: mx.array,
    n: int,
) -> mx.array:
    """Compute per-channel variance from running Welford-style accumulators.

    Args:
        key_sum:    [D] sum of all keys seen so far (fp32).
        key_sq_sum: [D] sum of squared keys seen so far (fp32).
        n:          Number of tokens accumulated.

    Returns:
        Per-channel variance [D] fp32.  Returns zeros if n < 2.
    """
    if n < 2:
        return mx.zeros_like(key_sum)
    mean = key_sum / n
    return mx.maximum(key_sq_sum / n - mean * mean, 0.0)


__all__ = [
    "rank_channels_by_sensitivity",
    "quantize_mixed_channels",
    "compute_running_variance",
    "hi_mask_from_variance_batched",
    "quantize_mixed_channels_batched",
]
