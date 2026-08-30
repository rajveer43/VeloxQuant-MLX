// flash_prefill.metal
// Extracted from veloxquant_mlx/metal/_flash_prefill.py (_FLASH_PREFILL_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).
//
// Plain-fp16 causal-only flash attention, specialized (no runtime
// branching) for exactly this repo's target: from-scratch prefill,
// D<=128, causal always on, no mask tensor, no attention sinks. Deltas
// versus the rabitq_prefill.metal scaffold this is built from:
//   1. BK is a template constant (KV_CHUNK, default 16, vs
//      rabitq_prefill.metal's fixed 8) — wider chunks mean fewer
//      threadgroup-barrier round trips per unit of K/V processed, at
//      the cost of more kv_tile/s_tile/w_tile threadgroup memory. Swept
//      per-D in _flash_prefill.py's _MEASURED_BK table (PHASE 6, see
//      blogs/prefill-roofline.md) rather than assumed.
//   2. exp2 softmax with the scale pre-folded by log2(e), and a causal
//      block-skip that drops fully-future KV chunks before any
//      load/matmul work (not just after, via masking) — both mirror
//      what MLX's own steel attention kernel already does.
//   3. The W.V step batches PDT depth-tiles (PDT*8 output columns) per
//      simdgroup_store instead of 1, cutting barrier count there by a
//      factor of PDT (16/PDT per chunk instead of 16). PDT is a
//      template constant chosen per head-dimension by
//      _flash_prefill.py's measured lookup table (_MEASURED_PDT), not a
//      runtime branch and not just "largest PDT that fits the 32KB
//      budget" — see that table's comment and blogs/prefill-roofline.md
//      for why a memory-budget proxy alone picked a worse value at D=64.
//   4. Online-softmax running state (m/d) lives in per-lane registers
//      instead of threadgroup arrays, broadcast via simd_shuffle only
//      where cross-lane visibility is actually needed — see the
//      declaration comment below.

    constexpr uint D   = uint(MAX_D);
    constexpr uint BQ  = uint(BQ_ROWS);                      // rows per SIMD-group (PHASE 6)
    constexpr uint BK  = uint(KV_CHUNK);                     // kv slots per chunk (PHASE 6)
    constexpr uint PDT = uint(P_DEPTH_TILES);                // depth-tiles batched per W.V store
    constexpr uint BKT = BK / 8u;                            // BK in units of 8x8 column-tiles
    constexpr uint BQT = BQ / 8u;                            // BQ in units of 8x8 row-tiles
    constexpr float kLog2E = 1.4426950408889634f;

    threadgroup half  q_tile[NSG_C * BQ * MAX_D];            // scale-folded Q rows
    threadgroup half  kv_tile[BK * MAX_D];                   // shared K, then V (plain fp16)
    threadgroup half  s_tile[NSG_C][BQ * BK];                // raw QK^T scores
    threadgroup half  w_tile[NSG_C][BQ * BK];                // softmax weights
    threadgroup half  p_tile[NSG_C][BQ * PDT * 8u];          // W.V chunk partial (PDT dt-tiles)
    threadgroup float out_tile[NSG_C][BQ * MAX_D];           // running output
    // Online-softmax running state (m/d) and this-chunk rescale factor
    // (f) live in per-lane registers, NOT threadgroup memory (PHASE 4):
    // each of lanes 0..7 owns exactly one query row across the whole
    // KV-chunk loop, so m_local/d_local need no cross-lane visibility
    // at all — only THIS lane ever reads its own value, every
    // iteration, so a plain thread-local persists correctly across
    // loop iterations with no barrier needed to protect it. factor
    // does need to reach lanes 8..31 too (the rescale step touches all
    // BQ*D output elements with 32 lanes, not just 8) — simd_shuffle
    // from lane r delivers that within the same simdgroup, no
    // threadgroup round trip required. Net: removes 3 threadgroup
    // arrays and narrows step 3->4's simdgroup_barrier to guarding
    // w_tile only.

    uint tgx  = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint sg   = thread_position_in_threadgroup.y;
    uint tid  = sg * 32u + lane;
    constexpr uint N_THREADS = 32u * uint(NSG_C);

    uint B      = uint(q_shape[0]);
    uint H      = uint(q_shape[1]);
    uint S_q    = uint(q_shape[2]);
    uint S_kv   = uint(k_shape[2]);
    uint BQ_TG  = uint(NSG_C) * BQ;
    uint n_qblk = (S_q + BQ_TG - 1u) / BQ_TG;
    (void)B;

    uint qblk  = tgx % n_qblk;
    uint h_idx = (tgx / n_qblk) % H;
    uint b_idx = tgx / (n_qblk * H);

    uint q_base  = ((b_idx * H + h_idx) * S_q) * D;
    uint kv_base = ((b_idx * H + h_idx) * S_kv) * D;
    // scale pre-folded with log2(e): softmax uses exp2 instead of exp
    // (matches MLX's own steel attention kernel — cheaper on Apple GPU ALUs).
    float sc = scale[0] * kLog2E;
    // Queries align to the tail of the KV cache (fused_sdpa.metal convention).
    int q_align = int(S_kv) - int(S_q);

    // ---- Stage the Q block (scale folded; rows past S_q zeroed) ----
    for (uint idx = tid; idx < BQ_TG * D; idx += N_THREADS) {
        uint gq = qblk * BQ_TG + idx / D;
        q_tile[idx] = (gq < S_q)
            ? half(float(q[q_base + gq * D + (idx % D)]) * sc)
            : half(0.0f);
    }
    for (uint idx = lane; idx < BQ * D; idx += 32u) out_tile[sg][idx] = 0.0f;
    float m_local = -INFINITY;
    float d_local = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const threadgroup half *my_q = q_tile + sg * BQ * D;

    // Causal block-skip: a threadgroup's query rows span
    // [qblk*BQ_TG, qblk*BQ_TG + BQ_TG). The last valid row's absolute
    // position bounds which KV chunks can contain any unmasked slot —
    // chunks entirely past that are skipped outright (no load, no
    // matmul, no softmax), not just masked after the fact.
    int last_row_abs = q_align + int(qblk * BQ_TG + BQ_TG - 1u);
    uint n_chunks_total = (S_kv + BK - 1u) / BK;
    uint n_chunks = 0u;
    if (last_row_abs >= 0) {
        uint visible = (uint(last_row_abs) + 1u + BK - 1u) / BK;
        n_chunks = metal::min(n_chunks_total, visible);
    }

    for (uint c = 0; c < n_chunks; ++c) {
        uint s0 = c * BK;

        // 1. Load K chunk (plain fp16, no decode).
        for (uint idx = tid; idx < BK * D; idx += N_THREADS) {
            uint j = idx / D, dcol = idx % D, slot = s0 + j;
            kv_tile[idx] = (slot < S_kv) ? k[kv_base + slot * D + dcol] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. QK^T on the matrix units: acc(BQ rows x BK slots), a
        //    BQT x BKT grid of 8x8 tiles, over D/8 depth tiles. BQT and
        //    BKT are compile-time constants (from the BQ_ROWS/KV_CHUNK
        //    template params), so this fully unrolls — no runtime
        //    branching. acc is flattened row-major [BQT][BKT] into a
        //    1D array (Metal disallows multi-dim fixed arrays of
        //    simdgroup matrices cleanly across all toolchains).
        simdgroup_half8x8 acc[BQT * BKT];
        for (uint i = 0; i < BQT * BKT; ++i) acc[i] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0f));
        for (uint kt = 0; kt < D / 8u; ++kt) {
            simdgroup_half8x8 qf[BQT];
            for (uint rt = 0; rt < BQT; ++rt) simdgroup_load(qf[rt], my_q + rt * 8u * D + kt * 8u, ulong(D));
            for (uint b = 0; b < BKT; ++b) {
                simdgroup_half8x8 kf;
                simdgroup_load(kf, kv_tile + kt * 8u + b * 8u * D, ulong(D), ulong2(0, 0), true);
                for (uint rt = 0; rt < BQT; ++rt) {
                    simdgroup_multiply_accumulate(acc[rt * BKT + b], qf[rt], kf, acc[rt * BKT + b]);
                }
            }
        }
        for (uint rt = 0; rt < BQT; ++rt) {
            for (uint b = 0; b < BKT; ++b) {
                simdgroup_store(acc[rt * BKT + b], &s_tile[sg][rt * 8u * BK + b * 8u], ulong(BK));
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Online softmax (exp2 form). lanes 0..7 each own one row,
        //    carrying m_local/d_local for that row across the whole
        //    chunk loop in registers (PHASE 4 — see declaration above).
        float factor = 0.0f;
        if (lane < BQ) {
            uint r = lane;
            int  q_abs = q_align + int(qblk * BQ_TG + sg * BQ + r);
            float s_j[BK];
            float chunk_max = -INFINITY;
            for (uint j = 0; j < BK; ++j) {
                uint slot = s0 + j;
                bool valid = slot < S_kv && int(slot) <= q_abs;
                s_j[j] = valid ? float(s_tile[sg][r * BK + j]) : -INFINITY;
                chunk_max = metal::max(chunk_max, s_j[j]);
            }
            float m_old = m_local;
            float m_new = metal::max(m_old, chunk_max);
            bool  chunk_empty = m_new == -INFINITY;
            factor = chunk_empty ? 0.0f : metal::fast::exp2(m_old - m_new);
            float wsum = 0.0f;
            for (uint j = 0; j < BK; ++j) {
                float w = chunk_empty ? 0.0f : metal::fast::exp2(s_j[j] - m_new);
                w_tile[sg][r * BK + j] = half(w);
                wsum += w;
            }
            d_local = d_local * factor + wsum;
            m_local = m_new;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 4. Rescale running output rows by this chunk's factor.
        //    factor only exists on lanes 0..7 (one per row); broadcast
        //    it to every lane within this simdgroup so all 32 lanes can
        //    rescale their share of the BQ*D output elements.
        for (uint idx = lane; idx < BQ * D; idx += 32u) {
            uint r = idx / D;
            out_tile[sg][idx] *= simd_shuffle(factor, r);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 5. Load V chunk (plain fp16, reuse kv_tile).
        for (uint idx = tid; idx < BK * D; idx += N_THREADS) {
            uint j = idx / D, dcol = idx % D, slot = s0 + j;
            kv_tile[idx] = (slot < S_kv) ? v[kv_base + slot * D + dcol] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 6. W.V on the matrix units. PDT depth-tiles (PDT*8 output
        //    columns) batched per simdgroup_store/barrier round trip,
        //    over a BQT x BKT grid of W row/column tiles (same
        //    generalization as step 2's QK^T — BQT=1 reduces this back
        //    to the original single-row-tile form). W (this sg's
        //    BQxBK weight tile) is loaded ONCE before the depth-tile
        //    loop, since it doesn't depend on dt — reused across every
        //    iteration below, flattened row-major [BQT][BKT]. V's BKT
        //    BKx8-row-tile halves are loaded once per depth-tile and
        //    MAC'd against the matching wf[] tile into that
        //    (row-tile, depth-tile)'s own 8x8 accumulator. PDT is
        //    chosen per-D by _flash_prefill.py to fit the 32KB
        //    threadgroup budget (see module header).
        simdgroup_half8x8 wf[BQT * BKT];
        for (uint rt = 0; rt < BQT; ++rt) {
            for (uint b = 0; b < BKT; ++b) {
                simdgroup_load(wf[rt * BKT + b], &w_tile[sg][rt * 8u * BK + b * 8u], ulong(BK));
            }
        }
        for (uint dt = 0; dt < D / 8u; dt += PDT) {
            for (uint rt = 0; rt < BQT; ++rt) {
                for (uint p = 0; p < PDT; ++p) {
                    simdgroup_half8x8 acc_p = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0f));
                    for (uint b = 0; b < BKT; ++b) {
                        simdgroup_half8x8 vt;
                        simdgroup_load(vt, kv_tile + (dt + p) * 8u + b * 8u * D, ulong(D));
                        simdgroup_multiply_accumulate(acc_p, wf[rt * BKT + b], vt, acc_p);
                    }
                    simdgroup_store(acc_p, &p_tile[sg][rt * 8u * PDT * 8u + p * 8u], ulong(PDT * 8u));
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = lane; idx < BQ * PDT * 8u; idx += 32u) {
                uint r = idx / (PDT * 8u), dc = idx % (PDT * 8u);
                out_tile[sg][r * D + dt * 8u + dc] += float(p_tile[sg][idx]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Write out: rows with n_chunks==0 (fully masked) get zeros ----
    // d_local is only meaningful on lanes 0..7 (one per row); broadcast
    // to every lane, same reasoning as step 4's factor broadcast. The
    // broadcast must run BEFORE the S_q bounds check (not after) so
    // every lane in the simdgroup executes it uniformly — skipping it
    // on some lanes via an early `continue` would be divergent control
    // flow into a simd_shuffle, which is undefined behavior.
    for (uint idx = lane; idx < BQ * D; idx += 32u) {
        uint r  = idx / D;
        float denom = simd_shuffle(d_local, r);
        uint gq = qblk * BQ_TG + sg * BQ + r;
        if (gq >= S_q) continue;
        out[q_base + gq * D + (idx % D)] =
            half(denom > 0.0f ? out_tile[sg][idx] / denom : 0.0f);
    }
