// scalar_affine_attend.metal
// Extracted from veloxquant_mlx/metal/_scalar_attend.py (_SCALAR_AFFINE_ATTEND_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    uint tg   = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint sg   = thread_position_in_threadgroup.y;

    uint B    = uint(q_shape[0]);
    uint H    = uint(q_shape[1]);
    uint S_q  = uint(q_shape[2]);
    uint D    = uint(q_shape[3]);
    uint S_kv = uint(k_codes_shape[2]);
    uint GK   = uint(k_scale_shape[2]);   // key groups along tokens
    uint GV   = uint(v_scale_shape[3]);   // value groups along channels
    (void)B;

    uint G        = uint(gsize[0]);       // group size
    float scale   = scale_arr[0];
    uint  NSG     = uint(NSG_C);

    uint sq_idx = tg % S_q;
    uint h_idx  = (tg / S_q) % H;
    uint b_idx  = tg / (S_q * H);

    uint q_base   = ((b_idx * H + h_idx) * S_q + sq_idx) * D;
    uint bh       = b_idx * H + h_idx;

    // ----- per-lane online-softmax state for this SIMD-group -----
    float running_m = -INFINITY;
    float running_d = 0.0f;
    float my_out[8];                       // D/32 <= 256/32 = 8
    for (int i = 0; i < 8; ++i) my_out[i] = 0.0f;
    uint n_owned = (D + 31u) / 32u;

    // ----- main loop: this SIMD-group strides kv by NSG -----
    for (uint sk = sg; sk < S_kv; sk += NSG) {
        // key group index along tokens for this slot
        uint kg = sk / G;

        // partial dot product over lane-owned dims
        float partial_dot = 0.0f;
        for (uint d = lane; d < D; d += 32u) {
            uint  code_off = (bh * S_kv + sk) * D + d;
            uint  ks_off   = (bh * GK + kg) * D + d;
            float k_hat = float(k_codes[code_off]) * k_scale[ks_off]
                        + k_zero[ks_off];
            partial_dot += float(q[q_base + d]) * k_hat;
        }
        float score = simd_sum(partial_dot) * scale;

        // online softmax update (every lane holds identical score)
        float m_new  = metal::max(running_m, score);
        float factor = metal::exp(running_m - m_new);
        float w      = metal::exp(score      - m_new);
        running_d    = running_d * factor + w;
        running_m    = m_new;
        for (uint i = 0; i < n_owned; ++i) my_out[i] *= factor;

        // value decode + weighted accumulate (per-token groups over channels)
        for (uint d = lane; d < D; d += 32u) {
            uint  vg      = d / G;
            uint  code_off = (bh * S_kv + sk) * D + d;
            uint  vs_off   = (bh * S_kv + sk) * GV + vg;
            float v_hat = float(v_codes[code_off]) * v_scale[vs_off]
                        + v_zero[vs_off];
            uint  out_i = (d - lane) / 32u;
            my_out[out_i] += w * v_hat;
        }
    }

    // ----- merge the NSG partial softmaxes through threadgroup memory -----
    // sh_o layout: [NSG_C][8][32]  (SIMD-group, owned-slot, lane).
    threadgroup float sh_m[NSG_C];         // one slot per SIMD-group
    threadgroup float sh_d[NSG_C];
    threadgroup float sh_o[NSG_C * 8 * 32];

    // stash this SIMD-group's per-lane partials
    for (uint i = 0; i < n_owned; ++i) {
        sh_o[(sg * 8u + i) * 32u + lane] = my_out[i];
    }
    if (lane == 0) { sh_m[sg] = running_m; sh_d[sg] = running_d; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SIMD-group 0 reduces across all groups
    if (sg == 0) {
        float gm = -INFINITY;
        for (uint s = 0; s < NSG; ++s) gm = metal::max(gm, sh_m[s]);
        float gd = 0.0f;
        for (uint s = 0; s < NSG; ++s) gd += sh_d[s] * metal::exp(sh_m[s] - gm);

        for (uint i = 0; i < n_owned; ++i) {
            float acc = 0.0f;
            for (uint s = 0; s < NSG; ++s) {
                acc += sh_o[(s * 8 + i) * 32 + lane]
                     * metal::exp(sh_m[s] - gm);
            }
            uint d = lane + i * 32u;
            if (d < D) {
                uint out_off = ((b_idx * H + h_idx) * S_q + sq_idx) * D + d;
                out[out_off] = half(acc / gd);
            }
        }
    }
