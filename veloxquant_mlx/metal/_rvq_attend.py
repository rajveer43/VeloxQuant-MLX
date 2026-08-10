"""Fused RVQ decode + attention Metal kernel for TurboQuantRVQ.

Replaces the two-dispatch pattern (decode keys → run SDPA) with a single
FlashAttention-style online-softmax pass that decodes keys on-the-fly from
two-stage RVQ indices and accumulates weighted values directly from value
codebook indices.

Public API:
  - :func:`turboquant_fused_rvq_decode_attend`
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return (Path(__file__).parent / "src" / filename).read_text()


_cache: dict = {}


# ===========================================================================
# Metal source — fused RVQ decode + FlashAttention attend
# ===========================================================================
# Grid:        (B * H * S_q * TG, 1, 1) — MLX grid = total threads.
# Threadgroup: (TG, 1, 1)               — TG = min(D, 32).
#
# Each threadgroup handles one query position (b, h, sq).
# Lanes stripe across the D-dimensional vectors in steps of TG.
#
# Per-lane state:
#   float my_out[8]  — output accumulator (max D/TG = 256/32 = 8 elems)
#   float running_m  — online softmax running max
#   float running_d  — online softmax running denominator
#
# Per kv-slot:
#   1. Decode k[i] = centroids1[k_indices1[i]] + centroids2[k_indices2[i]]
#   2. score = simd_sum(dot(q, k)) / sqrt(D)
#   3. Online softmax update (running_m, running_d, factor)
#   4. Decode v from v_codebook, accumulate w * v into my_out
#
# After all S_kv slots: divide my_out by running_d and write to out.
#
# Template parameters B_BITS1, B_BITS2, B_BITS_V are carried for future
# compile-time dispatch on centroid table size; currently unused in the body.

_FUSED_RVQ_ATTEND_SRC = _read_kernel_source("rvq_attend_fused.metal")


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------


def _rvq_attend_kernel(b1: int, b2: int, bv: int, D: int):
    key = ("fused_rvq_attend", b1, b2, bv, D)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"turboquant_fused_rvq_attend_b{b1}_{b2}_{bv}_d{D}",
            input_names=[
                "q",
                "k_indices1",
                "k_indices2",
                "centroids1",
                "centroids2",
                "v_indices",
                "v_codebook",
            ],
            output_names=["out"],
            source=_FUSED_RVQ_ATTEND_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def turboquant_fused_rvq_decode_attend(
    q: mx.array,
    k_indices1: mx.array,
    k_indices2: mx.array,
    centroids1: mx.array,
    centroids2: mx.array,
    v_indices: mx.array,
    v_codebook: mx.array,
    b1: int,
    b2: int,
    bv: int,
) -> mx.array:
    """Fused two-stage RVQ key decode + scaled-dot-product attention.

    Decodes keys on-the-fly from two-stage RVQ indices using an online
    softmax loop — no intermediate K_hat tensor is materialized.

    Args:
        q:          ``[B, H, S_q, D]`` fp16 queries (pre-rotated).
        k_indices1: ``[B, H, S_kv, D]`` uint8 first-stage key indices.
        k_indices2: ``[B, H, S_kv, D]`` uint8 second-stage key indices.
        centroids1: ``[2^b1]`` fp32 Gaussian centroids (stage 1).
        centroids2: ``[2^b2]`` fp32 Laplacian centroids (stage 2).
        v_indices:  ``[B, H, S_kv, D//sub_dim_v]`` uint8 value indices.
        v_codebook: ``[2^bv, sub_dim_v]`` fp16 value codebook.
        b1, b2, bv: Bit-widths for key stage 1, stage 2, and values.

    Returns:
        ``[B, H, S_q, D]`` fp16 attention output.
    """
    if q.ndim != 4:
        raise ValueError(f"turboquant_fused_rvq_decode_attend: q must be 4D, got {q.shape}")
    B, H, S_q, D = q.shape
    TG = min(D, 32)
    n_tg = B * H * S_q

    outputs = _rvq_attend_kernel(b1, b2, bv, D)(
        inputs=[
            q.astype(mx.float16),
            k_indices1.astype(mx.uint8),
            k_indices2.astype(mx.uint8),
            centroids1.astype(mx.float32),
            centroids2.astype(mx.float32),
            v_indices.astype(mx.uint8),
            v_codebook.astype(mx.float16),
        ],
        template=[("B_BITS1", b1), ("B_BITS2", b2), ("B_BITS_V", bv)],
        # MLX grid = total threads; TG threads per threadgroup
        grid=(n_tg * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[(B, H, S_q, D)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]


__all__ = [
    "turboquant_fused_rvq_decode_attend",
]
