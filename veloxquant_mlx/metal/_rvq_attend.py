"""Fused RVQ decode + attention Metal kernel for TurboQuantRVQ.

Replaces the two-dispatch pattern (decode keys → run SDPA) with a single
FlashAttention-style online-softmax pass that decodes keys on-the-fly from
two-stage RVQ indices and accumulates weighted values directly from value
codebook indices.

Public API:
  - :func:`turboquant_fused_rvq_decode_attend`
"""

from __future__ import annotations

import mlx.core as mx

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

_FUSED_RVQ_ATTEND_SRC = r"""
    constexpr int N_CENTS1  = 1 << B_BITS1;
    constexpr int N_CENTS2  = 1 << B_BITS2;
    constexpr int N_CENTS_V = 1 << B_BITS_V;

    uint tg      = threadgroup_position_in_grid.x;
    uint tg_lane = thread_position_in_threadgroup.x;

    uint B     = uint(q_shape[0]);
    uint H     = uint(q_shape[1]);
    uint S_q   = uint(q_shape[2]);
    uint D     = uint(q_shape[3]);
    uint S_kv  = uint(k_indices1_shape[2]);
    uint V_SUB = uint(v_codebook_shape[1]);
    uint n_sub_v = D / V_SUB;

    uint sq_idx = tg % S_q;
    uint h_idx  = (tg / S_q) % H;
    uint b_idx  = tg / (S_q * H);

    float inv_sqrt_d = metal::rsqrt(float(D));
    uint  TG         = threads_per_threadgroup.x;

    float running_m = -INFINITY;
    float running_d = 0.0f;

    // Per-lane output accumulator; max 8 slots (D=256, TG=32)
    float my_out[8];
    for (int i = 0; i < 8; ++i) my_out[i] = 0.0f;
    uint n_owned = (D + TG - 1) / TG;

    uint q_base    = ((b_idx * H + h_idx) * S_q + sq_idx) * D;
    uint k_base_bh = (b_idx * H + h_idx) * S_kv;
    uint v_base_bh = (b_idx * H + h_idx) * S_kv;

    for (uint sk = 0; sk < S_kv; ++sk) {
        // Decode key + partial dot product (each lane covers its strided dims)
        float partial_dot = 0.0f;
        for (uint i = tg_lane; i < D; i += TG) {
            uint  k_off = (k_base_bh + sk) * D + i;
            float ki    = centroids1[uint(k_indices1[k_off])]
                        + centroids2[uint(k_indices2[k_off])];
            partial_dot += float(q[q_base + i]) * ki;
        }
        float score = simd_sum(partial_dot) * inv_sqrt_d;

        // Online softmax update
        float m_new  = metal::max(running_m, score);
        float factor = metal::exp(running_m - m_new);
        float w      = metal::exp(score     - m_new);
        running_d    = running_d * factor + w;
        running_m    = m_new;

        // Rescale accumulated output
        for (uint i = 0; i < n_owned; ++i) my_out[i] *= factor;

        // Decode value + weighted accumulate
        for (uint i = tg_lane; i < D; i += TG) {
            uint  sub_i  = i / V_SUB;
            uint  comp_i = i % V_SUB;
            uint  v_off  = (v_base_bh + sk) * n_sub_v + sub_i;
            uint  cb_off = uint(v_indices[v_off]) * V_SUB + comp_i;
            float vi     = float(v_codebook[cb_off]);
            uint  out_i  = (i - tg_lane) / TG;
            my_out[out_i] += w * vi;
        }
    }

    // Normalize and write
    for (uint i = tg_lane; i < D; i += TG) {
        uint out_i   = (i - tg_lane) / TG;
        uint out_off = ((b_idx * H + h_idx) * S_q + sq_idx) * D + i;
        out[out_off] = half(my_out[out_i] / running_d);
    }
"""


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
