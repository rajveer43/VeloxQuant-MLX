"""QJL (Quantized Johnson-Lindenstrauss) Metal kernels.

Implements the two halves of the QJL attention estimator used by
TurboQuantProd:

  IP(q, k) ≈ ⟨q, x̂_mse⟩ + √(π/2)/m · ‖r‖ · ⟨S@q, sign(S@r)⟩

  - :func:`qjl_encode`         — compute sign(S @ x) (bit-packed) and ‖x‖.
  - :func:`qjl_inner_product`  — score pre-projected queries against encoded keys.

Grid convention (MLX)
---------------------
``mx.fast.metal_kernel`` dispatches ``grid`` as total threads, not total
threadgroups.  Both kernels use a threadgroup size of 32 (one simdgroup) and
set ``grid = n_threadgroups * 32`` accordingly.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return (Path(__file__).parent / "src" / filename).read_text()


_cache: dict = {}


# ===========================================================================
# Metal source — qjl_encode
# ===========================================================================
# Grid:        (B * ceil(m/32) * 32, 1, 1) — total threads
# Threadgroup: (32, 1, 1) — one simdgroup per threadgroup
#
# Each simdgroup handles 32 consecutive sketch dimensions.
# lane j owns sketch dimension simd_blk*32 + j, computes dot(S[j,:], x[b,:])
# via a scalar loop, then packs 32 sign bits into 4 consecutive uint8 bytes
# using simd_shuffle.
#
# Norm: the simdgroup with simd_blk==0 also computes ‖x‖ via simd_sum and
# lane 0 writes it to norms[b_idx].
#
# packed_signs layout: [B, ceil(m/8)] uint8, LSB-first bit order.

_QJL_ENCODE_SRC = _read_kernel_source("qjl_encode.metal")


# ===========================================================================
# Metal source — qjl_inner_product
# ===========================================================================
# Grid:        (H * S_kv * 32, 1, 1) — total threads
# Threadgroup: (32, 1, 1) — one simdgroup per (head, kv-slot)
#
# Each simdgroup computes:
#   acc = Σ_j  q_proj[h, j] * (2 * sign(S@k)[h,s,j] - 1)
#   score = √(π/2)/m * norms[s, h] * acc
#
# Bit unpacking: chunk=lane, lane+32, lane+64, … to stride across all m bits.

_QJL_INNER_PRODUCT_SRC = _read_kernel_source("qjl_inner_product.metal")


# ---------------------------------------------------------------------------
# Kernel factories
# ---------------------------------------------------------------------------


def _encode_kernel():
    key = "qjl_encode"
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name="qjl_encode",
            input_names=["x", "S"],
            output_names=["packed_signs", "norms"],
            source=_QJL_ENCODE_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


def _inner_product_kernel():
    key = "qjl_inner_product"
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name="qjl_inner_product",
            input_names=["q_proj", "packed_signs", "norms"],
            output_names=["scores"],
            source=_QJL_INNER_PRODUCT_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def qjl_encode(x: mx.array, S: mx.array) -> tuple[mx.array, mx.array]:
    """Compute 1-bit QJL encoding: sign(S @ x) bit-packed + ‖x‖.

    Args:
        x: ``[B, d]`` fp16 input vectors.
        S: ``[m, d]`` fp16 JL projection matrix. m must be divisible by 8.

    Returns:
        ``(packed_signs, norms)`` where:

        * ``packed_signs`` — ``[B, m//8]`` uint8 (LSB-first bit order)
        * ``norms``        — ``[B]`` fp16 Euclidean norms
    """
    if x.ndim != 2:
        raise ValueError(f"qjl_encode: x must be 2D, got {x.shape}")
    if S.ndim != 2:
        raise ValueError(f"qjl_encode: S must be 2D, got {S.shape}")
    B, d = x.shape
    m, d_s = S.shape
    if d != d_s:
        raise ValueError(f"qjl_encode: x.d={d} != S.d={d_s}")
    if m % 8 != 0:
        raise ValueError(f"qjl_encode: m={m} must be divisible by 8")

    n_simd_per_batch = (m + 31) // 32
    # MLX grid = total threads; threadgroup size = 32
    n_total_threads = B * n_simd_per_batch * 32
    n_sign_bytes = B * (m // 8)

    outputs = _encode_kernel()(
        inputs=[x.astype(mx.float16), S.astype(mx.float16)],
        grid=(n_total_threads, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n_sign_bytes,), (B,)],
        output_dtypes=[mx.uint8, mx.float16],
        init_value=0,
    )
    return outputs[0].reshape(B, m // 8), outputs[1]


def qjl_inner_product(
    q_proj: mx.array,
    packed_signs: mx.array,
    norms: mx.array,
) -> mx.array:
    """Compute unbiased QJL attention scores.

    Evaluates ``√(π/2)/m · norms[s,h] · ⟨q_proj[h,:], (2·signs−1)⟩``
    for every (head, kv-slot) pair.

    Args:
        q_proj:       ``[H, m]`` fp16 pre-projected queries (``S @ q``).
        packed_signs: ``[S_kv, H, m//8]`` uint8 bit-packed key signs
                      produced by :func:`qjl_encode`.
        norms:        ``[S_kv, H]`` fp16 key norms.

    Returns:
        ``[H, S_kv]`` fp16 attention scores.
    """
    if q_proj.ndim != 2:
        raise ValueError(f"qjl_inner_product: q_proj must be 2D [H,m], got {q_proj.shape}")
    if packed_signs.ndim != 3:
        raise ValueError(
            f"qjl_inner_product: packed_signs must be 3D [S_kv,H,m/8], got {packed_signs.shape}"
        )
    if norms.ndim != 2:
        raise ValueError(f"qjl_inner_product: norms must be 2D [S_kv,H], got {norms.shape}")

    H, m = q_proj.shape
    S_kv = norms.shape[0]
    n_tg = H * S_kv

    # packed_signs flat layout expected by kernel: [S_kv * H, m//8]
    ps_flat = packed_signs.astype(mx.uint8).reshape(S_kv * H, m // 8)

    outputs = _inner_product_kernel()(
        inputs=[q_proj.astype(mx.float16), ps_flat, norms.astype(mx.float16)],
        # MLX grid = total threads; threadgroup size = 32
        grid=(n_tg * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(H * S_kv,)],
        output_dtypes=[mx.float16],
    )
    return outputs[0].reshape(H, S_kv)


__all__ = [
    "qjl_encode",
    "qjl_inner_product",
]
