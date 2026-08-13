// rvq_attend_fused.metal
// Extracted from veloxquant_mlx/metal/_rvq_attend.py (_FUSED_RVQ_ATTEND_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

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
