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
//   1. BK=16 instead of 8 — half as many threadgroup-barrier round
//      trips per unit of K/V processed.
//   2. exp2 softmax with the scale pre-folded by log2(e), and a causal
//      block-skip that drops fully-future KV chunks before any
//      load/matmul work (not just after, via masking) — both mirror
//      what MLX's own steel attention kernel already does.
//   3. The W.V step stores 2 depth-tiles (16 output columns) per
//      simdgroup_store instead of 1, halving barrier count there
//      (16 -> 8 per chunk) without growing threadgroup memory, since
//      p_tile's natural BQ*BK sizing already holds 16 columns.

    constexpr uint D   = uint(MAX_D);
    constexpr uint BQ  = 8u;                                 // rows per SIMD-group
    constexpr uint BK  = 16u;                                // kv slots per chunk
    constexpr float kLog2E = 1.4426950408889634f;

    threadgroup half  q_tile[NSG_C * BQ * MAX_D];            // scale-folded Q rows
    threadgroup half  kv_tile[BK * MAX_D];                   // shared K, then V (plain fp16)
    threadgroup half  s_tile[NSG_C][BQ * BK];                // raw QK^T scores
    threadgroup half  w_tile[NSG_C][BQ * BK];                // softmax weights
    threadgroup half  p_tile[NSG_C][BQ * BK];                // W.V chunk partial (2 dt-tiles)
    threadgroup float out_tile[NSG_C][BQ * MAX_D];           // running output
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
    if (lane < BQ) { m_run[sg][lane] = -INFINITY; d_run[sg][lane] = 0.0f; }
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

        // 2. QK^T on the matrix units: acc(8 rows x 16 slots), two 8x8
        //    column tiles side by side, over D/8 depth tiles.
        simdgroup_half8x8 acc0 = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0f));
        simdgroup_half8x8 acc1 = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0f));
        for (uint kt = 0; kt < D / 8u; ++kt) {
            simdgroup_half8x8 qf, kf0, kf1;
            simdgroup_load(qf, my_q + kt * 8u, ulong(D));
            simdgroup_load(kf0, kv_tile + kt * 8u, ulong(D), ulong2(0, 0), true);
            simdgroup_load(kf1, kv_tile + kt * 8u + 8u * D, ulong(D), ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc0, qf, kf0, acc0);
            simdgroup_multiply_accumulate(acc1, qf, kf1, acc1);
        }
        simdgroup_store(acc0, &s_tile[sg][0], ulong(BK));
        simdgroup_store(acc1, &s_tile[sg][8], ulong(BK));
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Online softmax (exp2 form). lanes 0..7 each own one row.
        if (lane < BQ) {
            uint r = lane;
            int  q_abs = q_align + int(qblk * BQ_TG + sg * BQ + r);
            float s_j[16];
            float chunk_max = -INFINITY;
            for (uint j = 0; j < BK; ++j) {
                uint slot = s0 + j;
                bool valid = slot < S_kv && int(slot) <= q_abs;
                s_j[j] = valid ? float(s_tile[sg][r * BK + j]) : -INFINITY;
                chunk_max = metal::max(chunk_max, s_j[j]);
            }
            float m_old = m_run[sg][r];
            float m_new = metal::max(m_old, chunk_max);
            bool  chunk_empty = m_new == -INFINITY;
            float factor = chunk_empty ? 0.0f : metal::fast::exp2(m_old - m_new);
            float wsum = 0.0f;
            for (uint j = 0; j < BK; ++j) {
                float w = chunk_empty ? 0.0f : metal::fast::exp2(s_j[j] - m_new);
                w_tile[sg][r * BK + j] = half(w);
                wsum += w;
            }
            d_run[sg][r] = d_run[sg][r] * factor + wsum;
            m_run[sg][r] = m_new;
            f_row[sg][r] = factor;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 4. Rescale running output rows by this chunk's factor.
        for (uint idx = lane; idx < BQ * D; idx += 32u) {
            out_tile[sg][idx] *= f_row[sg][idx / D];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 5. Load V chunk (plain fp16, reuse kv_tile).
        for (uint idx = tid; idx < BK * D; idx += N_THREADS) {
            uint j = idx / D, dcol = idx % D, slot = s0 + j;
            kv_tile[idx] = (slot < S_kv) ? v[kv_base + slot * D + dcol] : half(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 6. W.V on the matrix units. Two depth-tiles (16 output
        //    columns) per simdgroup_store — p_tile's BQ*BK=8*16
        //    natural size holds exactly that — halving barrier count
        //    in this loop versus storing one 8-wide tile at a time.
        for (uint dt = 0; dt < D / 8u; dt += 2u) {
            simdgroup_half8x8 wf0, wf1, vf0a, vf0b, vf1a, vf1b;
            simdgroup_half8x8 pacc0 = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0f));
            simdgroup_half8x8 pacc1 = make_filled_simdgroup_matrix<half, 8, 8>(half(0.0f));
            simdgroup_load(wf0, &w_tile[sg][0], ulong(BK));
            simdgroup_load(wf1, &w_tile[sg][8], ulong(BK));
            simdgroup_load(vf0a, kv_tile + dt * 8u, ulong(D));
            simdgroup_load(vf0b, kv_tile + dt * 8u + 8u * D, ulong(D));
            simdgroup_load(vf1a, kv_tile + (dt + 1u) * 8u, ulong(D));
            simdgroup_load(vf1b, kv_tile + (dt + 1u) * 8u + 8u * D, ulong(D));
            simdgroup_multiply_accumulate(pacc0, wf0, vf0a, pacc0);
            simdgroup_multiply_accumulate(pacc0, wf1, vf0b, pacc0);
            simdgroup_multiply_accumulate(pacc1, wf0, vf1a, pacc1);
            simdgroup_multiply_accumulate(pacc1, wf1, vf1b, pacc1);
            simdgroup_store(pacc0, &p_tile[sg][0], ulong(BK));
            simdgroup_store(pacc1, &p_tile[sg][8], ulong(BK));
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = lane; idx < BQ * BK; idx += 32u) {
                uint r = idx / BK, dc = idx % BK;
                out_tile[sg][r * D + dt * 8u + dc] += float(p_tile[sg][idx]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- Write out: rows with n_chunks==0 (fully masked) get zeros ----
    for (uint idx = lane; idx < BQ * D; idx += 32u) {
        uint r  = idx / D;
        uint gq = qblk * BQ_TG + sg * BQ + r;
        if (gq >= S_q) continue;
        float denom = d_run[sg][r];
        out[q_base + gq * D + (idx % D)] =
            half(denom > 0.0f ? out_tile[sg][idx] / denom : 0.0f);
    }
