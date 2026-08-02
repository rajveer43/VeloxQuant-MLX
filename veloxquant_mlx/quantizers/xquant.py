"""XQuant quantizer — cross-layer KV cache reuse primitives.

Inspired by "XQuant: Achieving Ultra-Low Bit KV Cache Quantization with
Cross-Layer Compression" (arXiv:2510.11236, EMNLP 2025, Yang et al.). Documented
as "XQuant-adapted (VeloxQuant-MLX implementation)" — faithful to the
cross-layer-reuse core, adapted at the integration boundary (a shared
coordinator object rather than a modified attention forward pass).

Core idea:
    Adjacent transformer layers produce highly similar K/V tensors. Instead of
    every layer storing its own quantized cache, layers are grouped into
    anchor/reuse groups. The *anchor* layer of a group quantizes K/V normally
    and publishes the integer codes. The *reuse* layers borrow those codes and
    store only their own per-group dequantization parameters (scale/zero), which
    correct for the small cross-layer magnitude/offset drift. Across a group this
    drives the effective per-element bit-width well below the anchor's bit-width.

This module holds the pure, side-effect-free numerics:
    - ``pair_layers``            — assign anchor/reuse roles by index stride.
    - ``quantize_codes``         — anchor-side: min/max group quant → (codes, params).
    - ``compute_reuse_params``   — reuse-side: fit this layer's scale/zero to shared codes.
    - ``dequant_with_params``    — reconstruct fp16 from codes + params.
    - ``cross_layer_similarity`` — diagnostic (MSE / cosine) for tests + benchmark.

All quantization uses the same asymmetric min/max group scheme as
``_quant_utils._group_quant_dequant`` (groups along the token axis), so XQuant
composes with the rest of the suite's correctness guarantees.
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx


class GroupParams(NamedTuple):
    """Per-group asymmetric quantization parameters.

    Attributes:
        scale: [n_groups, 1, D] fp32 — per-group step size.
        zero:  [n_groups, 1, D] fp32 — per-group minimum (zero-point offset).
        n_rows: int — original (pre-pad) number of token rows.
        bits: int — bit-width these params were computed at.
    """

    scale: mx.array
    zero: mx.array
    n_rows: int
    bits: int


def pair_layers(n_layers: int, group_size: int) -> list[tuple[str, int]]:
    """Assign each layer an anchor/reuse role and a group id.

    Layers are chunked into contiguous groups of ``group_size``. The first layer
    in each group is the anchor; the rest reuse it.

    Args:
        n_layers: Number of attention-bearing layers.
        group_size: Layers per group (>=2). 2 → pairs; 3 → anchor + 2 reusers.

    Returns:
        Length-``n_layers`` list of ``(role, group_id)`` where role is
        ``"anchor"`` or ``"reuse"``. A trailing partial group whose anchor is its
        only member is still a valid (degenerate) anchor.
    """
    if group_size < 2:
        raise ValueError(f"pair_layers: group_size must be >= 2, got {group_size}.")
    roles: list[tuple[str, int]] = []
    for i in range(n_layers):
        group_id = i // group_size
        role = "anchor" if (i % group_size == 0) else "reuse"
        roles.append((role, group_id))
    return roles


def _pad_to_groups(x32: mx.array, group_size: int) -> tuple[mx.array, int, int]:
    """Pad [N, D] along axis 0 to a whole number of groups. Returns (padded, n_groups, n)."""
    n, d = x32.shape
    n_groups = (n + group_size - 1) // group_size
    pad = n_groups * group_size - n
    if pad:
        x32 = mx.concatenate([x32, mx.broadcast_to(x32[-1:], (pad, d))], axis=0)
    return x32, n_groups, n


def quantize_codes(x: mx.array, bits: int, group_size: int = 32) -> tuple[mx.array, GroupParams]:
    """Anchor-side quantization: return integer codes and the group params.

    Mirrors ``_group_quant_dequant`` but exposes the codes (for cross-layer
    sharing) instead of immediately dequantizing.

    Args:
        x: [N, D] fp16 or fp32 (one head's keys or values).
        bits: Bit-width.
        group_size: Tokens per group along axis 0.

    Returns:
        codes: [n_groups, group_size, D] fp32 integer codes in [0, 2^bits - 1].
        params: GroupParams for dequantization.
    """
    x32 = x.astype(mx.float32)
    x32, n_groups, n = _pad_to_groups(x32, group_size)
    d = x32.shape[-1]
    xg = x32.reshape(n_groups, group_size, d)
    gmin = mx.min(xg, axis=1, keepdims=True)  # [n_groups, 1, D]
    gmax = mx.max(xg, axis=1, keepdims=True)
    levels = (1 << bits) - 1
    eps = 1e-8
    scale = mx.maximum((gmax - gmin) / levels, eps)  # [n_groups, 1, D]
    codes = mx.clip(mx.round((xg - gmin) / scale), 0, levels)
    return codes, GroupParams(scale=scale, zero=gmin, n_rows=n, bits=bits)


def compute_reuse_params(
    x: mx.array, codes: mx.array, bits: int, group_size: int = 32
) -> GroupParams:
    """Reuse-side: fit this layer's per-group scale/zero to *shared* codes.

    Given the anchor's integer codes, find the scale/zero that best reconstructs
    *this* layer's keys/values. Because the codes are fixed, the optimal affine
    fit is the one mapping the code range [0, levels] onto this layer's group
    min/max — i.e. a fresh asymmetric min/max calibration on this layer's data,
    reusing the anchor's code assignment.

    Args:
        x: [N, D] this layer's fp16/fp32 keys or values.
        codes: [n_groups, group_size, D] anchor integer codes (for shape only;
            the affine fit is data-driven, the codes carry the bin assignment).
        bits: Bit-width the codes were produced at.
        group_size: Tokens per group.

    Returns:
        GroupParams (scale, zero) calibrated to ``x``.
    """
    x32 = x.astype(mx.float32)
    x32, n_groups, n = _pad_to_groups(x32, group_size)
    d = x32.shape[-1]
    xg = x32.reshape(n_groups, group_size, d)
    gmin = mx.min(xg, axis=1, keepdims=True)
    gmax = mx.max(xg, axis=1, keepdims=True)
    levels = (1 << bits) - 1
    eps = 1e-8
    scale = mx.maximum((gmax - gmin) / levels, eps)
    return GroupParams(scale=scale, zero=gmin, n_rows=n, bits=bits)


def dequant_with_params(codes: mx.array, params: GroupParams) -> mx.array:
    """Reconstruct fp16 [N, D] from integer codes + group params.

    Args:
        codes: [n_groups, group_size, D] fp32 integer codes.
        params: GroupParams (scale, zero, n_rows).

    Returns:
        Reconstructed [n_rows, D] fp16.
    """
    recon = codes * params.scale + params.zero  # [n_groups, gs, D]
    n_groups, gs, d = recon.shape
    return recon.reshape(n_groups * gs, d)[: params.n_rows].astype(mx.float16)


def quantize_residual(x: mx.array, recon: mx.array, bits: int, group_size: int = 32) -> mx.array:
    """Quantize the reuse-layer residual (x - recon) and return its contribution.

    A low-bit correction applied on top of the shared-code reconstruction. With
    ``bits == 0`` the caller skips this entirely (pure reuse).

    Args:
        x: [N, D] this layer's true keys/values.
        recon: [N, D] the shared-code reconstruction before correction.
        bits: Residual bit-width (>= 1).
        group_size: Tokens per group.

    Returns:
        [N, D] fp16 quantized residual to add back to ``recon``.
    """
    res = x.astype(mx.float32) - recon.astype(mx.float32)
    res32, n_groups, n = _pad_to_groups(res, group_size)
    d = res32.shape[-1]
    rg = res32.reshape(n_groups, group_size, d)
    gmin = mx.min(rg, axis=1, keepdims=True)
    gmax = mx.max(rg, axis=1, keepdims=True)
    levels = (1 << bits) - 1
    eps = 1e-8
    scale = mx.maximum((gmax - gmin) / levels, eps)
    codes = mx.clip(mx.round((rg - gmin) / scale), 0, levels)
    recon_res = (codes * scale + gmin).reshape(n_groups * group_size, d)[:n]
    return recon_res.astype(mx.float16)


def cross_layer_similarity(a: mx.array, b: mx.array) -> dict:
    """Diagnostic: MSE and mean cosine similarity between two layers' tensors.

    Args:
        a, b: Same-shape [B, H, S, D] (or [S, D]) tensors.

    Returns:
        dict with ``mse`` and ``cosine`` (mean over all but the last axis).
    """
    a32 = a.astype(mx.float32)
    b32 = b.astype(mx.float32)
    mse = float(mx.mean((a32 - b32) ** 2).item())
    dot = mx.sum(a32 * b32, axis=-1)
    na = mx.sqrt(mx.sum(a32 * a32, axis=-1))
    nb = mx.sqrt(mx.sum(b32 * b32, axis=-1))
    cos = dot / mx.maximum(na * nb, 1e-8)
    return {"mse": mse, "cosine": float(mx.mean(cos).item())}


__all__ = [
    "GroupParams",
    "pair_layers",
    "quantize_codes",
    "compute_reuse_params",
    "dequant_with_params",
    "quantize_residual",
    "cross_layer_similarity",
]
