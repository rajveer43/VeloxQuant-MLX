"""VecInfer Metal kernels — codebook dequantize, quantize, and fused encode+decode.

Phase 1:
  - :func:`vecinfer_dequant_metal`  — O(N) codebook gather, no intermediate alloc.
  - :func:`vecinfer_quantize_metal` — O(N) nearest-centroid argmin; avoids the
    O(N * n_centroids * sub_dim) diff tensor that OOMs on Falcon3-7B VecInfer-2bit.

Phase 2 (fused encode+decode):
  - :func:`vecinfer_encode_decode_metal`        — key path: smooth→WHT→VQ→dequant→inv-WHT→smooth.
  - :func:`vecinfer_encode_decode_simple_metal` — value path: VQ only, no transforms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import mlx.core as mx


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return (Path(__file__).parent / "src" / filename).read_text()


# ---------------------------------------------------------------------------
# Shared kernel cache (keyed by (tag, *shape_params))
# ---------------------------------------------------------------------------
_cache: dict = {}


# ===========================================================================
# Metal source — 1. codebook gather (dequantize)
# ===========================================================================
# Grid: (N_total, 1, 1) — one thread per input index.
# Each thread copies sub_dim contiguous centroid components to the output.

_DEQUANT_SRC = _read_kernel_source("vecinfer_dequant.metal")


# ===========================================================================
# Metal source — 2. nearest-centroid argmin (quantize)
# ===========================================================================
# Grid: (N_total, 1, 1) — one thread per sub-vector.
# Accumulated squared distance stays in registers; only the winning index
# is written → peak memory O(N) instead of O(N * n_centroids * sub_dim).

_QUANTIZE_SRC = _read_kernel_source("vecinfer_quantize.metal")


# ===========================================================================
# Metal source — 3. fused key encode+decode (smooth + WHT + VQ + inv-WHT)
# ===========================================================================
# Grid:        (B * H * S, 1, 1) — one threadgroup per (batch, head, token).
# Threadgroup: (D, 1, 1)         — one thread per channel.
#
# Threadgroup memory (sizes injected at compile time via #define):
#   float buf_in[MAX_D]   — input after smooth-divide / dequant staging
#   float buf_tg[MAX_D]   — WHT output / inv-WHT output
#   uint  idx[MAX_N_SUB]  — winning centroid per sub-vector

_ENCODE_DECODE_FULL_SRC = _read_kernel_source("vecinfer_encode_decode_full.metal")


# ===========================================================================
# Metal source — 4. fused value encode+decode (VQ only, no transforms)
# ===========================================================================
# Grid:        (B * H * S, 1, 1)
# Threadgroup: (D, 1, 1)
#
# Values skip smooth/Hadamard per the VecInfer paper.
# Threadgroup memory: float buf[MAX_D] + uint idx[MAX_N_SUB]

_ENCODE_DECODE_SIMPLE_SRC = _read_kernel_source("vecinfer_encode_decode_simple.metal")


# ---------------------------------------------------------------------------
# Kernel factories
# ---------------------------------------------------------------------------


def _dequant_kernel(dtype: mx.Dtype):
    key = ("dequant", str(dtype))
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"vecinfer_dequant_{str(dtype).replace('.', '_')}",
            input_names=["indices", "codebook"],
            output_names=["out"],
            source=_DEQUANT_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


def _quantize_kernel(dtype: mx.Dtype):
    key = ("quantize", str(dtype))
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"vecinfer_quantize_{str(dtype).replace('.', '_')}",
            input_names=["x", "codebook"],
            output_names=["out"],
            source=_QUANTIZE_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


def _encode_decode_full_kernel(D: int, n_sub: int, sub_dim: int, n_centroids: int):
    key = ("enc_dec_full", D, n_sub, sub_dim, n_centroids)
    if key not in _cache:
        header = (
            f"#pragma METAL fp math_mode(relaxed)\n#define MAX_D {D}\n#define MAX_N_SUB {n_sub}\n"
        )
        _cache[key] = mx.fast.metal_kernel(
            name=f"vecinfer_enc_dec_full_d{D}_ns{n_sub}_sd{sub_dim}_nc{n_centroids}",
            input_names=["keys", "k_codebook", "smooth", "H_mat", "params"],
            output_names=["k_hat_out", "idx_out"],
            header=header,
            source=_ENCODE_DECODE_FULL_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


def _encode_decode_simple_kernel(D: int, n_sub: int, sub_dim: int, n_centroids: int):
    key = ("enc_dec_simple", D, n_sub, sub_dim, n_centroids)
    if key not in _cache:
        header = (
            f"#pragma METAL fp math_mode(relaxed)\n#define MAX_D {D}\n#define MAX_N_SUB {n_sub}\n"
        )
        _cache[key] = mx.fast.metal_kernel(
            name=f"vecinfer_enc_dec_simple_d{D}_ns{n_sub}_sd{sub_dim}_nc{n_centroids}",
            input_names=["values", "v_codebook", "params"],
            output_names=["v_hat_out", "idx_out"],
            header=header,
            source=_ENCODE_DECODE_SIMPLE_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def vecinfer_dequant_metal(
    indices: mx.array,
    codebook: mx.array,
    out_dtype: Optional[mx.Dtype] = None,
) -> mx.array:
    """Drop-in Metal replacement for ``dequantize_vq``.

    Args:
        indices:  ``[..., n_sub]`` codebook indices, promoted to uint32.
        codebook: ``[n_centroids, sub_dim]`` centroid table.
        out_dtype: Output dtype; defaults to ``codebook.dtype``.

    Returns:
        ``[..., n_sub * sub_dim]`` reconstruction.
    """
    if codebook.ndim != 2:
        raise ValueError(f"vecinfer_dequant_metal: codebook must be 2D, got {codebook.shape}")

    sub_dim = codebook.shape[1]
    n_total = indices.size
    flat_indices = indices.reshape(-1).astype(mx.uint32)

    if out_dtype is None:
        out_dtype = codebook.dtype
    cb = codebook.astype(out_dtype) if codebook.dtype != out_dtype else codebook

    outputs = _dequant_kernel(out_dtype)(
        inputs=[flat_indices, cb],
        output_shapes=[(n_total * sub_dim,)],
        output_dtypes=[out_dtype],
        grid=(n_total, 1, 1),
        threadgroup=(min(256, n_total), 1, 1),
    )
    *leading, n_sub = indices.shape
    return outputs[0].reshape(*leading, n_sub * sub_dim)


def vecinfer_quantize_metal(
    x: mx.array,
    codebook: mx.array,
    sub_dim: int,
) -> mx.array:
    """Drop-in Metal replacement for ``quantize_vq``.

    Computes squared distances in thread-local registers; peak memory O(N)
    instead of O(N * n_centroids * sub_dim).

    Args:
        x:        ``[..., D]`` input, D divisible by ``sub_dim``.
        codebook: ``[n_centroids, sub_dim]``.
        sub_dim:  Sub-vector dimension.

    Returns:
        ``[..., D // sub_dim]`` int32 indices.
    """
    *leading, D = x.shape
    if D % sub_dim != 0:
        raise ValueError(f"vecinfer_quantize_metal: D={D} not divisible by sub_dim={sub_dim}")
    if codebook.ndim != 2 or codebook.shape[1] != sub_dim:
        raise ValueError(
            f"vecinfer_quantize_metal: codebook must be [n_centroids, {sub_dim}], got {codebook.shape}"
        )

    n_sub = D // sub_dim
    flat_x = x.reshape(*leading, n_sub, sub_dim).reshape(-1, sub_dim)
    n_total = flat_x.shape[0]

    work_dtype = flat_x.dtype if flat_x.dtype in (mx.float16, mx.float32) else mx.float32
    flat_x_w = flat_x.astype(work_dtype) if flat_x.dtype != work_dtype else flat_x
    cb_w = codebook.astype(work_dtype) if codebook.dtype != work_dtype else codebook

    outputs = _quantize_kernel(work_dtype)(
        inputs=[flat_x_w, cb_w],
        output_shapes=[(n_total,)],
        output_dtypes=[mx.uint32],
        grid=(n_total, 1, 1),
        threadgroup=(min(256, max(1, n_total)), 1, 1),
    )
    return outputs[0].astype(mx.int32).reshape(*leading, n_sub)


def vecinfer_encode_decode_metal(
    keys: mx.array,
    k_codebook: mx.array,
    sub_dim: int,
    H_mat: mx.array,
    smooth: Optional[mx.array] = None,
) -> tuple[mx.array, mx.array]:
    """Fused key encode+decode: smooth → WHT → VQ → dequant → inv-WHT → smooth.

    Replaces 7 MLX graph nodes with a single Metal dispatch.

    Args:
        keys:       ``[B, H, S, D]`` fp16 or fp32.
        k_codebook: ``[n_centroids, sub_dim]`` fp32 centroids.
        sub_dim:    Sub-vector size; must divide D.
        H_mat:      ``[D, D]`` Walsh-Hadamard matrix (fp32).
        smooth:     ``[H, D]`` or ``[D]`` smooth factors, or ``None``.

    Returns:
        ``(k_hat [B,H,S,D] fp16, k_indices [B,H,S,n_sub] int32)``
    """
    if keys.ndim != 4:
        raise ValueError(f"vecinfer_encode_decode_metal: keys must be 4D, got {keys.shape}")
    B, H_dim, S, D = keys.shape
    n_sub = D // sub_dim
    n_centroids = k_codebook.shape[0]

    if D % sub_dim != 0:
        raise ValueError(f"vecinfer_encode_decode_metal: D={D} not divisible by sub_dim={sub_dim}")
    if D > 512:
        raise ValueError(f"vecinfer_encode_decode_metal: D={D} > 512 (threadgroup limit)")

    keys_f32 = keys.astype(mx.float32) if keys.dtype != mx.float32 else keys
    cb_f32 = k_codebook.astype(mx.float32) if k_codebook.dtype != mx.float32 else k_codebook
    H_f32 = H_mat.astype(mx.float32) if H_mat.dtype != mx.float32 else H_mat

    has_smooth, smooth_rows = 0, 1
    if smooth is not None:
        has_smooth = 1
        smooth_2d = smooth.reshape(1, D) if smooth.ndim == 1 else smooth
        smooth_2d = smooth_2d.astype(mx.float32)
        smooth_rows = smooth_2d.shape[0]
    else:
        smooth_2d = mx.ones((1, D), dtype=mx.float32)

    params = mx.array(
        [B, H_dim, S, D, n_sub, sub_dim, n_centroids, has_smooth, smooth_rows],
        dtype=mx.uint32,
    )
    n_tokens = B * H_dim * S

    outputs = _encode_decode_full_kernel(D, n_sub, sub_dim, n_centroids)(
        inputs=[keys_f32, cb_f32, smooth_2d, H_f32, params],
        output_shapes=[(B, H_dim, S, D), (B, H_dim, S, n_sub)],
        output_dtypes=[mx.float16, mx.uint32],
        # grid is in threads, not threadgroups (mx.fast.metal_kernel
        # convention) — one threadgroup of D threads per token, so the grid
        # must be n_tokens * D. Passing n_tokens here silently truncated to
        # floor(n_tokens / D) threadgroups (zero when n_tokens < D), leaving
        # most/all tokens' outputs at their uninitialized buffer contents.
        grid=(n_tokens * D, 1, 1),
        threadgroup=(D, 1, 1),
    )
    return outputs[0], outputs[1].astype(mx.int32)


def vecinfer_encode_decode_simple_metal(
    values: mx.array,
    v_codebook: mx.array,
    sub_dim: int,
) -> tuple[mx.array, mx.array]:
    """Fused value encode+decode: VQ quantize + dequantize in one pass.

    Args:
        values:     ``[B, H, S, D]`` fp16 or fp32.
        v_codebook: ``[n_centroids, sub_dim]`` fp32 centroids.
        sub_dim:    Sub-vector size; must divide D.

    Returns:
        ``(v_hat [B,H,S,D] fp16, v_indices [B,H,S,n_sub] uint32)``
    """
    if values.ndim != 4:
        raise ValueError(
            f"vecinfer_encode_decode_simple_metal: values must be 4D, got {values.shape}"
        )
    B, H, S, D = values.shape
    n_sub = D // sub_dim
    n_centroids = v_codebook.shape[0]

    if D % sub_dim != 0:
        raise ValueError(
            f"vecinfer_encode_decode_simple_metal: D={D} not divisible by sub_dim={sub_dim}"
        )
    if D > 512:
        raise ValueError(f"vecinfer_encode_decode_simple_metal: D={D} > 512 (threadgroup limit)")

    values_f32 = values.astype(mx.float32) if values.dtype != mx.float32 else values
    cb_f32 = v_codebook.astype(mx.float32) if v_codebook.dtype != mx.float32 else v_codebook

    params = mx.array([B, H, S, D, n_sub, sub_dim, n_centroids], dtype=mx.uint32)
    n_tokens = B * H * S

    outputs = _encode_decode_simple_kernel(D, n_sub, sub_dim, n_centroids)(
        inputs=[values_f32, cb_f32, params],
        output_shapes=[(B, H, S, D), (B, H, S, n_sub)],
        output_dtypes=[mx.float16, mx.uint32],
        # grid is in threads, not threadgroups — see matching comment in
        # vecinfer_encode_decode_metal.
        grid=(n_tokens * D, 1, 1),
        threadgroup=(D, 1, 1),
    )
    return outputs[0], outputs[1]


__all__ = [
    "vecinfer_dequant_metal",
    "vecinfer_quantize_metal",
    "vecinfer_encode_decode_metal",
    "vecinfer_encode_decode_simple_metal",
]
