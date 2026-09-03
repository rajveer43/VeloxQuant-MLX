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
    uint H_q  = uint(q_shape[1]);
    uint S_q  = uint(q_shape[2]);
    uint D    = uint(q_shape[3]);
    uint H_kv = uint(k_codes_shape[1]);
    uint S_kv = uint(k_codes_shape[2]);
    uint GK   = uint(k_scale_shape[2]);   // key groups along tokens
    uint GV   = uint(v_scale_shape[3]);   // value groups along channels
    (void)B;

    uint G        = uint(gsize[0]);       // group size
    float scale   = scale_arr[0];
    uint  NSG     = uint(NSG_C);
    uint  HPK     = uint(HEADS_PER_KV_C);

    // Each threadgroup owns one (batch, kv-head, query-position) slot and
    // handles ALL HEADS_PER_KV_C query heads that share this kv head —
    // K/V codes are decoded once per (sk, d) and reused across those heads.
    uint sq_idx  = tg % S_q;
    uint hkv_idx = (tg / S_q) % H_kv;
    uint b_idx   = tg / (S_q * H_kv);

    uint bh_kv = b_idx * H_kv + hkv_idx;    // indexes k/v arrays (H_kv-wide)

    // ----- per-lane online-softmax state for this SIMD-group, per packed head -----
    float running_m[HEADS_PER_KV_C];
    float running_d[HEADS_PER_KV_C];
    float my_out[HEADS_PER_KV_C][8];        // D/32 <= 256/32 = 8
    for (uint hp = 0; hp < HPK; ++hp) {
        running_m[hp] = -INFINITY;
        running_d[hp] = 0.0f;
        for (int i = 0; i < 8; ++i) my_out[hp][i] = 0.0f;
    }
    uint n_owned = (D + 31u) / 32u;

    // ----- main loop: this SIMD-group strides kv by NSG -----
    for (uint sk = sg; sk < S_kv; sk += NSG) {
        // key group index along tokens for this slot
        uint kg = sk / G;

        // Decode K once per (sk, d) — shared across all HEADS_PER_KV_C heads —
        // and accumulate each head's partial dot product in the same pass.
        float partial_dot[HEADS_PER_KV_C];
        for (uint hp = 0; hp < HPK; ++hp) partial_dot[hp] = 0.0f;
        for (uint d = lane; d < D; d += 32u) {
            uint  code_off = (bh_kv * S_kv + sk) * D + d;
            uint  ks_off   = (bh_kv * GK + kg) * D + d;
            float k_hat = float(k_codes[code_off]) * k_scale[ks_off]
                        + k_zero[ks_off];
            for (uint hp = 0; hp < HPK; ++hp) {
                uint h_q_idx = hkv_idx * HPK + hp;
                uint q_off   = ((b_idx * H_q + h_q_idx) * S_q + sq_idx) * D + d;
                partial_dot[hp] += float(q[q_off]) * k_hat;
            }
        }

        float score[HEADS_PER_KV_C];
        for (uint hp = 0; hp < HPK; ++hp) {
            score[hp] = simd_sum(partial_dot[hp]) * scale;
        }

        // online softmax update, per packed head
        float w[HEADS_PER_KV_C];
        for (uint hp = 0; hp < HPK; ++hp) {
            float m_new  = metal::max(running_m[hp], score[hp]);
            float factor = metal::exp(running_m[hp] - m_new);
            w[hp]        = metal::exp(score[hp]      - m_new);
            running_d[hp] = running_d[hp] * factor + w[hp];
            running_m[hp] = m_new;
            for (uint i = 0; i < n_owned; ++i) my_out[hp][i] *= factor;
        }

        // Decode V once per (sk, d) — shared across all HEADS_PER_KV_C heads —
        // and accumulate each head's weighted value in the same pass.
        for (uint d = lane; d < D; d += 32u) {
            uint  vg      = d / G;
            uint  code_off = (bh_kv * S_kv + sk) * D + d;
            uint  vs_off   = (bh_kv * S_kv + sk) * GV + vg;
            float v_hat = float(v_codes[code_off]) * v_scale[vs_off]
                        + v_zero[vs_off];
            uint  out_i = (d - lane) / 32u;
            for (uint hp = 0; hp < HPK; ++hp) {
                my_out[hp][out_i] += w[hp] * v_hat;
            }
        }
    }

    // ----- merge the NSG partial softmaxes through threadgroup memory -----
    // sh_o layout: [NSG_C][HEADS_PER_KV_C][8][32]  (SIMD-group, packed head, owned-slot, lane).
    threadgroup float sh_m[NSG_C * HEADS_PER_KV_C];
    threadgroup float sh_d[NSG_C * HEADS_PER_KV_C];
    threadgroup float sh_o[NSG_C * HEADS_PER_KV_C * 8 * 32];

    // stash this SIMD-group's per-lane partials, per packed head
    for (uint hp = 0; hp < HPK; ++hp) {
        for (uint i = 0; i < n_owned; ++i) {
            sh_o[((sg * HPK + hp) * 8u + i) * 32u + lane] = my_out[hp][i];
        }
        if (lane == 0) {
            sh_m[sg * HPK + hp] = running_m[hp];
            sh_d[sg * HPK + hp] = running_d[hp];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SIMD-group 0 reduces across all groups, per packed head
    if (sg == 0) {
        for (uint hp = 0; hp < HPK; ++hp) {
            float gm = -INFINITY;
            for (uint s = 0; s < NSG; ++s) gm = metal::max(gm, sh_m[s * HPK + hp]);
            float gd = 0.0f;
            for (uint s = 0; s < NSG; ++s) gd += sh_d[s * HPK + hp] * metal::exp(sh_m[s * HPK + hp] - gm);

            uint h_q_idx = hkv_idx * HPK + hp;
            for (uint i = 0; i < n_owned; ++i) {
                float acc = 0.0f;
                for (uint s = 0; s < NSG; ++s) {
                    acc += sh_o[((s * HPK + hp) * 8 + i) * 32 + lane]
                         * metal::exp(sh_m[s * HPK + hp] - gm);
                }
                uint d = lane + i * 32u;
                if (d < D) {
                    uint out_off = ((b_idx * H_q + h_q_idx) * S_q + sq_idx) * D + d;
                    out[out_off] = half(acc / gd);
                }
            }
        }
    }
