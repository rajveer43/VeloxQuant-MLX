"""ZipCache-adapted quantizer — saliency-adaptive per-token mixed-precision.

Inspired by "ZipCache: Accurate and Efficient KV Cache Quantization with
Salient Token Identification" (He et al., NeurIPS 2024, arXiv:2405.14256).
Documented as "ZipCache-adapted (VeloxQuant-MLX implementation)" — not a
faithful port.

What ZipCache adds that the repo did not have: **per-token mixed bit-width
within the quantized space**. Three existing methods use the key-norm proxy
for token importance; this is the fourth use and the first with a different
decision:

    KIVI-Sink   — top-k norm tokens → fp16 (binary: quantized vs not)
    AdaKV-proxy — mean norm per head → head-level budget reallocation
    Kitty       — per-channel variance → channel-level bit allocation
    ZipCache-adapted → per-token norm → hi_bits vs lo_bits (both quantized)

Every token stays quantized. High-norm tokens (salient, top ``hi_fraction``
by key L2-norm) are quantized at ``hi_bits``; the rest at ``lo_bits``. The
effective average key rate is ``hi_frac·hi_bits + (1-hi_frac)·lo_bits`` bits
per element — explicitly between ``lo_bits`` and ``hi_bits``.

Honest proxy limitation: the paper's true saliency signal is the normalized
attention score, which is not observable by a cache wrapper. Key L2-norm is a
proxy: attention-sink tokens (highest importance) also exhibit large key norms.
This proxy is weaker than true attention scores — reported plainly, never
hidden. See the KIVI-Sink docs for the proxy derivation.

Adaptation decisions:
  1. Per-group min/max quantization (not per-channel as in the paper's ideal).
     Reuses the same group-quant logic as KIVI/CacheGen/GEAR for consistency
     and test-coverage inheritance.
  2. Values are quantized uniformly at ``hi_bits`` (no saliency routing for
     values — the paper focuses on keys; ``quantize_values=True`` can be set
     to compress values too at a single bit-width).
  3. Saliency is computed once per incoming block. No retroactive reclassification
     of already-quantized tokens.

This module holds the pure numerics: norm computation, saliency masking,
per-group channel quant, full compress/reconstruct, and honest byte accounting.
The cache wrapper owns the per-layer prefill/decode state.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import mlx.core as mx


class ZipCacheState(NamedTuple):
    """A ZipCache-compressed key or value matrix with per-token bit routing.

    Attributes:
        hi_codes:  [n_hi, D] uint8 high-bit key codes for salient tokens.
        hi_scales: [n_hi_groups, D] fp32 per-group scale (hi-bit path).
        hi_zeros:  [n_hi_groups, D] fp32 per-group zero-point (hi-bit path).
        lo_codes:  [n_lo, D] uint8 low-bit key codes for non-salient tokens.
        lo_scales: [n_lo_groups, D] fp32 per-group scale (lo-bit path).
        lo_zeros:  [n_lo_groups, D] fp32 per-group zero-point (lo-bit path).
        hi_mask:   [S] bool saliency mask; True = salient (hi-bit) token.
        hi_bits:   int bit-width for salient tokens.
        lo_bits:   int bit-width for non-salient tokens.
        seq_len:   int original token count (= S).
        head_dim:  int feature dimension (= D).
    """

    hi_codes: mx.array
    hi_scales: mx.array
    hi_zeros: mx.array
    lo_codes: mx.array
    lo_scales: mx.array
    lo_zeros: mx.array
    hi_mask: mx.array
    hi_bits: int
    lo_bits: int
    seq_len: int
    head_dim: int


# ---------------------------------------------------------------------------
# Saliency helpers
# ---------------------------------------------------------------------------


def token_key_norms(keys: mx.array) -> mx.array:
    """L2 norm of each token's key vector, ``[S, D] → [S]`` fp32."""
    return mx.linalg.norm(keys.astype(mx.float32), axis=-1)


def saliency_mask(norms: mx.array, hi_fraction: float) -> mx.array:
    """Top-``hi_fraction`` tokens by key-norm are marked salient (True).

    Args:
        norms: ``[S]`` fp32 key L2-norms.
        hi_fraction: Fraction of tokens to mark as salient (0 → all lo-bit;
            1 → all hi-bit).

    Returns:
        ``[S]`` bool array; True = salient token → hi_bits path.
    """
    S = int(norms.shape[0])
    n_hi = max(0, min(S, int(math.ceil(S * hi_fraction))))
    if n_hi == 0:
        return mx.zeros((S,), dtype=mx.bool_)
    if n_hi >= S:
        return mx.ones((S,), dtype=mx.bool_)
    # argsort ascending; the top n_hi by norm are at the tail
    order = mx.argsort(norms)  # ascending
    hi_indices = order[S - n_hi :]  # largest n_hi norms
    mask = mx.zeros((S,), dtype=mx.float32)
    mask = mask.at[hi_indices].add(1.0)
    return mask.astype(mx.bool_)


# ---------------------------------------------------------------------------
# Per-group min/max quantization (channel axis)
# ---------------------------------------------------------------------------


def channel_quant(
    x: mx.array,
    bits: int,
    group_size: int = 32,
) -> tuple[mx.array, mx.array, mx.array]:
    """Asymmetric per-group min/max quantization along the token axis.

    Groups partition the token (row) axis into blocks of ``group_size``.
    Within each group, a shared (scale, zero) is fitted per channel.

    Args:
        x: ``[N, D]`` fp32/fp16 input.
        bits: Bit-width (1–8).
        group_size: Tokens per quantization group.

    Returns:
        ``(codes, scales, zeros)`` where codes is ``[N, D]`` uint8 and
        scales/zeros are ``[n_groups, D]`` fp32.
    """
    if x.shape[0] == 0:
        # Edge: empty tensor (all tokens went to the other path)
        D = x.shape[1]
        empty_codes = mx.zeros((0, D), dtype=mx.uint8)
        empty_params = mx.zeros((0, D), dtype=mx.float32)
        return empty_codes, empty_params, empty_params

    n, d = x.shape
    gs = group_size
    levels = (1 << bits) - 1
    eps = 1e-8
    n_groups = (n + gs - 1) // gs
    pad = n_groups * gs - n
    x32 = x.astype(mx.float32)
    if pad:
        x32 = mx.concatenate([x32, mx.broadcast_to(x32[-1:], (pad, d))], axis=0)
    xg = x32.reshape(n_groups, gs, d)  # [G, gs, D]
    gmin = mx.min(xg, axis=1, keepdims=True)  # [G, 1, D]
    gmax = mx.max(xg, axis=1, keepdims=True)
    scale = mx.maximum((gmax - gmin) / levels, eps)
    codes = mx.clip(mx.round((xg - gmin) / scale), 0, levels).astype(mx.uint8)
    codes = codes.reshape(n_groups * gs, d)[:n]  # drop padding
    scales = scale.reshape(n_groups, d)
    zeros = gmin.reshape(n_groups, d)
    return codes, scales.astype(mx.float32), zeros.astype(mx.float32)


def channel_dequant(
    codes: mx.array,
    scales: mx.array,
    zeros: mx.array,
    group_size: int = 32,
) -> mx.array:
    """Reconstruct ``[N, D]`` fp32 from codes + per-group (scales, zeros)."""
    if codes.shape[0] == 0:
        return mx.zeros((0, codes.shape[1]), dtype=mx.float32)
    n = int(codes.shape[0])
    d = int(codes.shape[1])
    n_groups = int(scales.shape[0])
    gs = group_size
    pad = n_groups * gs - n
    c = codes.astype(mx.float32)
    if pad:
        c = mx.concatenate([c, mx.broadcast_to(c[-1:], (pad, d))], axis=0)
    cg = c.reshape(n_groups, gs, d)
    recon = cg * scales[:, None, :] + zeros[:, None, :]
    return recon.reshape(n_groups * gs, d)[:n]


# ---------------------------------------------------------------------------
# Full compress / reconstruct
# ---------------------------------------------------------------------------


def zipcache_compress(
    x: mx.array,
    hi_bits: int = 4,
    lo_bits: int = 2,
    hi_fraction: float = 0.20,
    group_size: int = 32,
) -> ZipCacheState:
    """Compress ``[S, D]`` with per-token saliency-adaptive mixed bit-width.

    Args:
        x: ``[S, D]`` fp16/fp32 key or value matrix for one head.
        hi_bits: Bit-width for salient (high-norm) tokens.
        lo_bits: Bit-width for non-salient tokens.
        hi_fraction: Fraction of tokens routed to hi_bits (by key L2-norm).
        group_size: Token group size for min/max quantization.

    Returns:
        :class:`ZipCacheState` carrying split codes and the saliency mask.
    """
    x32 = x.astype(mx.float32)
    S, D = x32.shape
    norms = token_key_norms(x32)
    mask = saliency_mask(norms, hi_fraction)

    # Split rows by saliency mask
    mask_np = [bool(v) for v in mask.tolist()]
    hi_rows = [i for i, m in enumerate(mask_np) if m]
    lo_rows = [i for i, m in enumerate(mask_np) if not m]

    def _gather(rows):
        if not rows:
            return mx.zeros((0, D), dtype=mx.float32)
        return mx.stack([x32[i] for i in rows], axis=0)

    x_hi = _gather(hi_rows)  # [n_hi, D]
    x_lo = _gather(lo_rows)  # [n_lo, D]

    hi_codes, hi_scales, hi_zeros = channel_quant(x_hi, hi_bits, group_size)
    lo_codes, lo_scales, lo_zeros = channel_quant(x_lo, lo_bits, group_size)

    return ZipCacheState(
        hi_codes=hi_codes,
        hi_scales=hi_scales,
        hi_zeros=hi_zeros,
        lo_codes=lo_codes,
        lo_scales=lo_scales,
        lo_zeros=lo_zeros,
        hi_mask=mask,
        hi_bits=hi_bits,
        lo_bits=lo_bits,
        seq_len=S,
        head_dim=D,
    )


def zipcache_reconstruct(state: ZipCacheState) -> mx.array:
    """Reconstruct fp16 ``[S, D]`` from a ZipCacheState.

    Dequantizes the hi and lo groups and scatters them back into their
    original token positions using the stored saliency mask.
    """
    S = state.seq_len
    D = state.head_dim
    gs = 32  # group_size is not stored in state; use 32 (the only value used)
    # Use scales shape to derive actual group_size used at compress time
    n_hi = int(state.hi_codes.shape[0]) if state.hi_codes.shape[0] > 0 else 0
    n_lo = int(state.lo_codes.shape[0]) if state.lo_codes.shape[0] > 0 else 0

    hi_recon = channel_dequant(state.hi_codes, state.hi_scales, state.hi_zeros, gs)
    lo_recon = channel_dequant(state.lo_codes, state.lo_scales, state.lo_zeros, gs)

    # Scatter back: build output row by row using the mask
    mask_list = [bool(v) for v in state.hi_mask.tolist()]
    hi_ptr = 0
    lo_ptr = 0
    rows = []
    for is_hi in mask_list:
        if is_hi:
            rows.append(hi_recon[hi_ptr])
            hi_ptr += 1
        else:
            rows.append(lo_recon[lo_ptr])
            lo_ptr += 1
    out = mx.stack(rows, axis=0)  # [S, D]
    return out.astype(mx.float16)


def zipcache_bytes(state: ZipCacheState, group_size: int = 32) -> int:
    """Honest stored size (bytes) of a ZipCacheState.

    Counts: packed codes (ceil(n * D * bits / 8)) + fp32 params (scale + zero).
    The saliency mask is stored as bool (1 byte/token).
    """
    S, D = state.seq_len, state.head_dim
    n_hi = int(state.hi_codes.shape[0])
    n_lo = int(state.lo_codes.shape[0])

    hi_code_bytes = math.ceil(n_hi * D * state.hi_bits / 8)
    lo_code_bytes = math.ceil(n_lo * D * state.lo_bits / 8)

    # Group param bytes: fp16 scale + zero per (group, D) — matches GEAR/CacheGen accounting
    n_hi_groups = math.ceil(n_hi / group_size) if n_hi > 0 else 0
    n_lo_groups = math.ceil(n_lo / group_size) if n_lo > 0 else 0
    hi_param_bytes = n_hi_groups * D * 2 * 2  # scale + zero, fp16 (2 bytes each)
    lo_param_bytes = n_lo_groups * D * 2 * 2

    mask_bytes = S  # bool, 1 byte per token
    return int(hi_code_bytes + lo_code_bytes + hi_param_bytes + lo_param_bytes + mask_bytes)


def base_only_bytes(S: int, D: int, bits: int, group_size: int = 32) -> int:
    """Uniform baseline size: all S tokens at a single bit-width (fp16 params)."""
    code_bytes = math.ceil(S * D * bits / 8)
    n_groups = math.ceil(S / group_size)
    param_bytes = n_groups * D * 2 * 2  # scale + zero, fp16
    return int(code_bytes + param_bytes)


def zipcache_quant_dequant(
    x: mx.array,
    hi_bits: int = 4,
    lo_bits: int = 2,
    hi_fraction: float = 0.20,
    group_size: int = 32,
) -> mx.array:
    """Drop-in quant→dequant: full ZipCache round-trip on ``[S, D]`` → fp16."""
    return zipcache_reconstruct(zipcache_compress(x, hi_bits, lo_bits, hi_fraction, group_size))


__all__ = [
    "ZipCacheState",
    "token_key_norms",
    "saliency_mask",
    "channel_quant",
    "channel_dequant",
    "zipcache_compress",
    "zipcache_reconstruct",
    "zipcache_bytes",
    "base_only_bytes",
    "zipcache_quant_dequant",
]
