// rabitq_prefill.metal
// Extracted from veloxquant_mlx/metal/_rabitq_prefill.py (_RABITQ_PREFILL_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

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
