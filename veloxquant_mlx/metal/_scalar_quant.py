"""Scalar quantization Metal kernels for TurboQuant.

Three kernels covering the scalar quantization hot-path used by
TurboQuantMSE, TurboQuantRVQ, and the Hadamard preconditioner:

  - :func:`turboquant_scalar_quantize`   — nearest-centroid encode, b=1–4 bits.
  - :func:`turboquant_scalar_dequantize` — centroid gather decode → fp16.
  - :func:`turboquant_hadamard_quantize` — fused Walsh-Hadamard + quantize in
    one threadgroup-local dispatch; eliminates the intermediate fp16 buffer.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return (Path(__file__).parent / "src" / filename).read_text()


_cache: dict = {}


# ===========================================================================
# Metal source — scalar quantize (nearest-centroid argmin)
# ===========================================================================
# Grid: (N, 1, 1) — one thread per element.
# Threadgroup: (256, 1, 1).
#
# Template <int B_BITS> lets the compiler statically unroll the centroid scan
# (max 16 centroids for b=4, all fit in registers).

_SCALAR_QUANTIZE_SRC = _read_kernel_source("scalar_quantize.metal")


# ===========================================================================
# Metal source — scalar dequantize (centroid gather)
# ===========================================================================
# Grid: (N, 1, 1) — one thread per element.
# Simple gather: x_hat[i] = fp16(centroids[indices[i]]).

_SCALAR_DEQUANTIZE_SRC = _read_kernel_source("scalar_dequantize.metal")


# ===========================================================================
# Metal source — fused Hadamard + scalar quantize
# ===========================================================================
# Grid:        (B * D, 1, 1) — MLX grid = total threads.
# Threadgroup: (D, 1, 1)    — D threads share one threadgroup (D ≤ 1024).
#
# MAX_D is injected as a compile-time #define so threadgroup float buf[MAX_D]
# is a legal static array.  D must be a power of 2.
#
# Pipeline:
#   1. Load x[b, lane] into threadgroup buffer; apply diagonal ±1 sign.
#   2. In-place WHT: range-based parallel butterfly (log2(D) barrier passes).
#      Each thread reads its partner BEFORE the barrier write, so there are
#      no data races.  Produces the same output as the sequential nested loop.
#   3. Scale by 1/√D (rsqrt for speed).
#   4. Nearest-centroid argmin in registers → write uint8 index.

_HADAMARD_QUANTIZE_SRC = _read_kernel_source("hadamard_quantize.metal")


# ---------------------------------------------------------------------------
# Kernel factories
# ---------------------------------------------------------------------------

def _scalar_quantize_kernel(b: int, dtype: mx.Dtype):
    key = ("scalar_quantize", b, str(dtype))
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"turboquant_scalar_quantize_b{b}_{str(dtype).replace('.', '_')}",
            input_names=["x", "centroids"],
            output_names=["indices"],
            source=_SCALAR_QUANTIZE_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


def _scalar_dequantize_kernel():
    key = "scalar_dequantize"
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name="turboquant_scalar_dequantize",
            input_names=["indices", "centroids"],
            output_names=["x_hat"],
            source=_SCALAR_DEQUANTIZE_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


def _hadamard_quantize_kernel(D: int, b: int):
    key = ("hadamard_quantize", D, b)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"turboquant_hadamard_quantize_d{D}_b{b}",
            input_names=["x", "diag", "centroids"],
            output_names=["indices"],
            header=f"#define MAX_D {D}\n",
            source=_HADAMARD_QUANTIZE_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def turboquant_scalar_quantize(
    x: mx.array,
    centroids: mx.array,
    b: int,
) -> mx.array:
    """Nearest-centroid scalar quantize into b-bit uint8 indices.

    Args:
        x:         ``[..., d]`` float input (any float dtype).
        centroids: ``[2^b]`` fp32 Lloyd-Max centroids.
        b:         Bits per index, 1–4.

    Returns:
        ``[..., d]`` uint8 indices.
    """
    if not (1 <= b <= 4):
        raise ValueError(f"turboquant_scalar_quantize: b must be 1–4, got {b}")
    expected = 1 << b
    if centroids.size != expected:
        raise ValueError(
            f"turboquant_scalar_quantize: expected {expected} centroids for b={b}, "
            f"got {centroids.size}"
        )
    *leading, d = x.shape
    # Only promote to fp32 if the input isn't already a Metal-native float
    # type: the kernel body does float(x[elem]) internally (see
    # _SCALAR_QUANTIZE_SRC), so fp16/fp32 input can be passed straight
    # through without an extra MLX cast dispatch beforehand.
    if x.dtype in (mx.float16, mx.float32):
        flat_x = x.reshape(-1)
    else:
        flat_x = x.reshape(-1).astype(mx.float32)
    cents_f32 = centroids.astype(mx.float32)
    N = flat_x.size

    outputs = _scalar_quantize_kernel(b, flat_x.dtype)(
        inputs=[flat_x, cents_f32],
        template=[("B_BITS", b)],
        grid=(N, 1, 1),
        threadgroup=(min(256, N), 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[mx.uint8],
    )
    return outputs[0].reshape(*leading, d)


def turboquant_scalar_dequantize(
    indices: mx.array,
    centroids: mx.array,
) -> mx.array:
    """Decode b-bit indices to fp16 via centroid gather.

    Args:
        indices:   ``[..., d]`` uint8 indices.
        centroids: ``[2^b]`` fp32 centroids.

    Returns:
        ``[..., d]`` fp16 reconstructed values.
    """
    flat_idx = indices.reshape(-1).astype(mx.uint8)
    cents_f32 = centroids.astype(mx.float32)
    N = flat_idx.size

    outputs = _scalar_dequantize_kernel()(
        inputs=[flat_idx, cents_f32],
        grid=(N, 1, 1),
        threadgroup=(min(256, N), 1, 1),
        output_shapes=[(N,)],
        output_dtypes=[mx.float16],
    )
    return outputs[0].reshape(indices.shape)


def turboquant_hadamard_quantize(
    x: mx.array,
    diag: mx.array,
    centroids: mx.array,
    b: int,
) -> mx.array:
    """Fused randomized Hadamard preconditioner + scalar quantize.

    Computes ``y = diag * H * x / sqrt(D)`` and nearest-centroid quantizes
    ``y`` in a single Metal dispatch — no intermediate allocation.

    Args:
        x:         ``[B, D]`` fp16 input. D must be a power of 2, ≤ 1024.
        diag:      ``[D]`` float ±1 diagonal signs.
        centroids: ``[2^b]`` fp32 Lloyd-Max centroids.
        b:         Bits per index, 1–4.

    Returns:
        ``[B, D]`` uint8 indices.
    """
    if x.ndim != 2:
        raise ValueError(f"turboquant_hadamard_quantize: x must be 2D, got {x.shape}")
    B, D = x.shape
    if D & (D - 1):
        raise ValueError(f"turboquant_hadamard_quantize: D={D} must be a power of 2")
    if D > 1024:
        raise ValueError(f"turboquant_hadamard_quantize: D={D} exceeds threadgroup limit 1024")
    if not (1 <= b <= 4):
        raise ValueError(f"turboquant_hadamard_quantize: b must be 1–4, got {b}")

    outputs = _hadamard_quantize_kernel(D, b)(
        inputs=[x.astype(mx.float32), diag.astype(mx.float32), centroids.astype(mx.float32)],
        template=[("B_BITS", b)],
        # MLX grid = total threads; B threadgroups × D threads each
        grid=(B * D, 1, 1),
        threadgroup=(D, 1, 1),
        output_shapes=[(B, D)],
        output_dtypes=[mx.uint8],
    )
    return outputs[0]


__all__ = [
    "turboquant_scalar_quantize",
    "turboquant_scalar_dequantize",
    "turboquant_hadamard_quantize",
]
