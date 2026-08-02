"""Tiled prefill attention over the asymmetric RaBitQ cache — simdgroup_matrix.

The prefill-shaped companion to :func:`rabitq_fused_attend`. Targets the
multi-turn VLM workload: S_q new-turn tokens attending over S_kv
compressed history slots (image tokens), a matmul-shaped problem where
the decode kernel's one-query-per-threadgroup scalar dots waste the
hardware matrix pipeline. Both matmuls (Q·K̂ᵀ and W·V̂) run on 8×8
``simdgroup_matrix`` tiles (MSL spec §2.4, §6.7); K is decoded from
1-bit packed signs and V from nibble-packed 4-bit codebook indices
on the fly inside the tile loop — no dequantized K or V matrix is ever
materialized.

Score model (differs from the decode kernel — exact dot, not Hamming):

    k_hat[j]    = signs(k_bits[j]) * k_mag[j]      (+-1 per dim)
    score[i][j] = (q[i] . k_hat[j]) * scale + k_const[j]

``scale`` is a plain scalar (fold 1/sqrt(D) here); it is multiplied
into Q once at staging. Values are nibble-packed only (the format
:func:`rabitq_pack_values` produces). Cross-attention only: every query
row attends over all S_kv slots (no causal mask) — new-token
self-attention belongs on the fp16 path.

Public API:
  - :func:`rabitq_prefill_attend`
"""

from __future__ import annotations

import mlx.core as mx

_cache: dict = {}


# ===========================================================================
# Metal source — FlashAttention-style tiles on simdgroup_matrix
# ===========================================================================
# Grid:        (B * H * ceil(S_q/32) * 32, NSG_C, 1) — MLX grid = threads.
# Threadgroup: (32, NSG_C=4, 1) — 4 SIMD-groups of 32 lanes.
#
# Each threadgroup owns one (b, h, 32-row query block); SIMD-group sg
# owns rows [sg*8, sg*8+8) of that block. All 4 SIMD-groups walk the kv
# chunks TOGETHER: each 8-slot chunk of K (then V) is decoded ONCE into
# a shared staging tile and consumed by all four 8-row Q blocks — this
# is what makes fused prefill viable, since decode work per threadgroup
# is S_kv*D regardless of how many query rows it serves. (v1 gave each
# SIMD-group its own chunks and re-decoded K/V once per 8 query rows —
# 16x more decode traffic; it lost to dequantize+SDPA by 5-10x.)
#
# Decode is byte-wise: one k_bits byte yields 8 dims, one v_idx byte
# yields 2 dims — lanes iterate bytes, not elements.
#
# simdgroup_matrix constraints honoured (spec §2.4): matrix ops run
# under uniform SIMD-group control flow, and since the element-to-thread
# mapping is unspecified, every elementwise step (softmax, rescale,
# accumulate) round-trips through threadgroup memory via
# simdgroup_store.
#
# Precision: Q/K̂/V̂ tiles and 8×8 MAC fragments are half (a QK̂ᵀ dot
# spans D<=128 scale-folded terms; a W·V̂ partial spans 8), the running
# output accumulator and softmax state are float. ~28 KB threadgroup
# memory at D=128; all-float tiles would exceed the 32 KB budget.

_RABITQ_PREFILL_SRC = r"""
    constexpr uint D  = uint(MAX_D);
    constexpr uint BQ = 8u;                                 // rows per SIMD-group
    constexpr uint BK = 8u;                                 // kv slots per chunk
    constexpr uint NB = uint(N_BYTES);

    threadgroup half  q_tile[NSG_C * BQ * MAX_D];           // 32 scale-folded rows
    threadgroup half  kv_tile[BK * MAX_D];                  // shared K̂, then V̂
    threadgroup half  s_tile[NSG_C][BQ * BK];               // raw QK̂ᵀ scores
    threadgroup half  w_tile[NSG_C][BQ * BK];               // softmax weights
    threadgroup half  p_tile[NSG_C][BQ * BK];               // W·V̂ chunk partial
    threadgroup float out_tile[NSG_C][BQ * MAX_D];          // running output
    threadgroup float m_run[NSG_C][BQ];
    threadgroup float d_run[NSG_C][BQ];
    threadgroup float f_row[NSG_C][BQ];

    uint tgx  = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint sg   = thread_position_in_threadgroup.y;
    uint tid  = sg * 32u + lane;
    constexpr uint N_THREADS = 32u * uint(NSG_C);

    uint B      = uint(q_shape[0]);
    uint H      = uint(q_shape[1]);
    uint S_q    = uint(q_shape[2]);
    uint S_kv   = uint(k_bits_shape[2]);
    uint BQ_TG  = uint(NSG_C) * BQ;                          // 32 rows / threadgroup
    uint n_qblk = (S_q + BQ_TG - 1u) / BQ_TG;
    (void)B;

    uint qblk  = tgx % n_qblk;
    uint h_idx = (tgx / n_qblk) % H;
    uint b_idx = tgx / (n_qblk * H);

    uint q_base  = ((b_idx * H + h_idx) * S_q) * D;
    uint kv_base = (b_idx * H + h_idx) * S_kv;
    float sc     = scale[0];

    // ---- Stage the 32-row Q block (scale folded; rows past S_q zeroed) ----
    for (uint idx = tid; idx < BQ_TG * D; idx += N_THREADS) {
        uint gq = qblk * BQ_TG + idx / D;
        q_tile[idx] = (gq < S_q)
            ? half(float(q[q_base + gq * D + (idx % D)]) * sc)
            : half(0.0f);
    }
    // ---- Init this sg's accumulator + softmax state ----
    for (uint idx = lane; idx < BQ * D; idx += 32u) out_tile[sg][idx] = 0.0f;
    if (lane < BQ) { m_run[sg][lane] = -INFINITY; d_run[sg][lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const threadgroup half *my_q = q_tile + sg * BQ * D;

    uint n_chunks = (S_kv + BK - 1u) / BK;
    for (uint c = 0; c < n_chunks; ++c) {
        uint s0 = c * BK;

        // 1. Decode K̂ chunk ONCE, byte-wise: one byte -> 8 dims of +-m_k.
        for (uint idx = tid; idx < BK * NB; idx += N_THREADS) {
            uint j = idx / NB, bcol = idx % NB, slot = s0 + j;
            uint base = j * D + bcol * 8u;
            if (slot < S_kv) {
                uint  byte = uint(k_bits[(kv_base + slot) * NB + bcol]);
                float mag  = k_mag[kv_base + slot];
                for (uint t = 0; t < 8u; ++t) {
                    kv_tile[base + t] =
                        half(((byte >> t) & 1u) != 0u ? mag : -mag);
                }
            } else {
                for (uint t = 0; t < 8u; ++t) kv_tile[base + t] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. QK̂ᵀ on the matrix units: acc(8 rows x 8 slots) over D/8 tiles.
        simdgroup_half8x8 acc = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0f));
        for (uint kt = 0; kt < D / 8u; ++kt) {
            simdgroup_half8x8 qf, kf;
            simdgroup_load(qf, my_q + kt * 8u, ulong(D));
            simdgroup_load(kf, kv_tile + kt * 8u, ulong(D), ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, qf, kf, acc);
        }
        simdgroup_store(acc, &s_tile[sg][0], ulong(BK));
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Online softmax (lanes 0..7 each own one of this sg's rows).
        if (lane < BQ) {
            uint r = lane;
            float s_j[8];
            float chunk_max = -INFINITY;
            for (uint j = 0; j < BK; ++j) {
                uint slot = s0 + j;
                s_j[j] = (slot < S_kv)
                    ? float(s_tile[sg][r * BK + j]) + k_const[kv_base + slot]
                    : -INFINITY;
                chunk_max = metal::max(chunk_max, s_j[j]);
            }
            float m_old  = m_run[sg][r];
            float m_new  = metal::max(m_old, chunk_max);
            float factor = metal::exp(m_old - m_new);
            float wsum   = 0.0f;
            for (uint j = 0; j < BK; ++j) {
                float w = metal::exp(s_j[j] - m_new);
                w_tile[sg][r * BK + j] = half(w);
                wsum += w;
            }
            d_run[sg][r] = d_run[sg][r] * factor + wsum;
            m_run[sg][r] = m_new;
            f_row[sg][r] = factor;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 4. Rescale the running output rows by this chunk's factor.
        for (uint idx = lane; idx < BQ * D; idx += 32u) {
            out_tile[sg][idx] *= f_row[sg][idx / D];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 5. Decode V̂ chunk ONCE over the same tile: one byte -> 2 dims.
        for (uint idx = tid; idx < BK * (D >> 1u); idx += N_THREADS) {
            uint j = idx / (D >> 1u), bcol = idx % (D >> 1u), slot = s0 + j;
            uint base = j * D + bcol * 2u;
            if (slot < S_kv) {
                uint byte = uint(v_idx[(kv_base + slot) * (D >> 1u) + bcol]);
                kv_tile[base]      = half(v_cents[byte & 0xFu]);
                kv_tile[base + 1u] = half(v_cents[byte >> 4u]);
            } else {
                kv_tile[base]      = half(0.0f);
                kv_tile[base + 1u] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 6. W·V̂ on the matrix units, per 8-column block; chunk partials
        //    (8-term dots, safe in half) accumulate into the float tile.
        for (uint dt = 0; dt < D / 8u; ++dt) {
            simdgroup_half8x8 wf, vf;
            simdgroup_half8x8 pacc = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0f));
            simdgroup_load(wf, &w_tile[sg][0], ulong(BK));
            simdgroup_load(vf, kv_tile + dt * 8u, ulong(D));
            simdgroup_multiply_accumulate(pacc, wf, vf, pacc);
            simdgroup_store(pacc, &p_tile[sg][0], ulong(BK));
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = lane; idx < BQ * BK; idx += 32u) {
                uint r = idx / BK, dc = idx % BK;
                out_tile[sg][r * D + dt * 8u + dc] += float(p_tile[sg][idx]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Each SIMD-group owns distinct rows — write them out directly ----
    for (uint idx = lane; idx < BQ * D; idx += 32u) {
        uint r  = idx / D;
        uint gq = qblk * BQ_TG + sg * BQ + r;
        if (gq >= S_q) continue;
        out[q_base + gq * D + (idx % D)] =
            half(out_tile[sg][idx] / d_run[sg][r]);
    }
"""


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

_N_SIMDGROUPS = 4


def _prefill_kernel(n_bytes: int, d: int):
    key = ("rabitq_prefill_attend", n_bytes, d)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"rabitq_prefill_attend_nb{n_bytes}_d{d}",
            input_names=[
                "q",
                "scale",
                "k_bits",
                "k_mag",
                "k_const",
                "v_idx",
                "v_cents",
            ],
            output_names=["out"],
            source=_RABITQ_PREFILL_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rabitq_prefill_attend(
    q: mx.array,  # [B, H, S_q, D]      fp16 — new-turn queries
    scale: mx.array,  # [1]                 fp32 — softmax scale (1/sqrt(D))
    k_bits: mx.array,  # [B, H, S_kv, D/8]   uint8 — packed 1-bit key signs
    k_mag: mx.array,  # [B, H, S_kv]        fp32 — per-key magnitude
    k_const: mx.array,  # [B, H, S_kv]        fp32 — additive score bias
    v_idx: mx.array,  # [B, H, S_kv, D/2]   uint8 — nibble-packed value indices
    v_cents: mx.array,  # [n_cents <= 16]     fp32 — scalar value codebook
) -> mx.array:
    """Tiled prefill attention over the compressed asymmetric cache.

    Cross-attention: every query row attends over all ``S_kv`` cached
    slots (no causal mask). Scores are exact dots with sign-decoded
    keys — ``(q . signs*k_mag) * scale + k_const`` — computed on
    simdgroup_matrix 8x8 tiles, as is the weight-value product. Values
    must be nibble-packed (see :func:`rabitq_pack_values`).

    Returns:
        ``[B, H, S_q, D]`` fp16 attention output.
    """
    if q.ndim != 4:
        raise ValueError(f"rabitq_prefill_attend: q must be 4D, got {q.shape}")
    B, H, S_q, D = q.shape
    if D % 8 != 0:
        raise ValueError(f"rabitq_prefill_attend: D={D} must be divisible by 8")
    if D > 128:
        raise ValueError(
            f"rabitq_prefill_attend: D={D} exceeds the 128 limit (threadgroup memory budget)"
        )
    n_bytes = D // 8

    if k_bits.ndim != 4 or k_bits.shape[:2] != (B, H) or k_bits.shape[3] != n_bytes:
        raise ValueError(
            f"rabitq_prefill_attend: k_bits must be [B, H, S_kv, {n_bytes}], got {k_bits.shape}"
        )
    S_kv = k_bits.shape[2]
    if k_mag.shape != (B, H, S_kv):
        raise ValueError(f"rabitq_prefill_attend: k_mag must be {(B, H, S_kv)}, got {k_mag.shape}")
    if k_const.shape != (B, H, S_kv):
        raise ValueError(
            f"rabitq_prefill_attend: k_const must be {(B, H, S_kv)}, got {k_const.shape}"
        )
    if v_idx.shape != (B, H, S_kv, D // 2):
        raise ValueError(
            f"rabitq_prefill_attend: v_idx must be nibble-packed "
            f"{(B, H, S_kv, D // 2)} (see rabitq_pack_values), got {v_idx.shape}"
        )
    if v_cents.ndim != 1 or v_cents.shape[0] > 16:
        raise ValueError(
            f"rabitq_prefill_attend: v_cents must be 1D with <= 16 entries, got {v_cents.shape}"
        )

    n_qblk = (S_q + 31) // 32
    n_tg = B * H * n_qblk

    outputs = _prefill_kernel(n_bytes, D)(
        inputs=[
            q.astype(mx.float16),
            scale.reshape(1).astype(mx.float32),
            k_bits.astype(mx.uint8),
            k_mag.astype(mx.float32),
            k_const.astype(mx.float32),
            v_idx.astype(mx.uint8),
            v_cents.astype(mx.float32),
        ],
        template=[("N_BYTES", n_bytes), ("MAX_D", D), ("NSG_C", _N_SIMDGROUPS)],
        # MLX grid = total threads; one (32 x NSG) threadgroup per q-block
        grid=(n_tg * 32, _N_SIMDGROUPS, 1),
        threadgroup=(32, _N_SIMDGROUPS, 1),
        output_shapes=[(B, H, S_q, D)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]


__all__ = ["rabitq_prefill_attend"]
