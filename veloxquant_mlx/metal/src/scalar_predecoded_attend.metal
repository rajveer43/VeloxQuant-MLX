// scalar_predecoded_attend.metal
// Extracted from veloxquant_mlx/metal/_scalar_attend.py (_SCALAR_PREDECODED_ATTEND_SRC)
// -- see that file's docstring and issue #308 for the algorithm and spike
// rationale. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call
// time; it is NOT separately compiled (see issue #64).
//
// Flash-decoding SDPA over ALREADY-DECODED fp16 k_hat/v_hat (produced by
// scalar_affine_decode_once.metal), one threadgroup per (b, h_q, sq) --
// i.e. full B*H_q*S_q dispatch, matching the high-occupancy baseline
// measured in issue #307's addendum. This kernel does no on-the-fly
// dequantization; heads sharing a kv head each independently re-READ the
// same decoded fp16 rows (redundant DRAM reads, no redundant decode ALU
// work) -- the exact tradeoff issue #308 exists to measure against both
// on-the-fly redundant decode (307's unpacked baseline) and threadgroup
// packing (307's packed kernel).

    uint tg   = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint sg   = thread_position_in_threadgroup.y;

    uint B    = uint(q_shape[0]);
    uint H_q  = uint(q_shape[1]);
    uint S_q  = uint(q_shape[2]);
    uint D    = uint(q_shape[3]);
    uint H_kv = uint(k_hat_shape[1]);
    uint S_kv = uint(k_hat_shape[2]);
    (void)B;

    float scale = scale_arr[0];
    uint  NSG   = uint(NSG_C);
    uint  HPK   = uint(H_q / H_kv);

    uint sq_idx = tg % S_q;
    uint h_idx  = (tg / S_q) % H_q;
    uint b_idx  = tg / (S_q * H_q);
    uint hkv_idx = h_idx / HPK;
    uint bh_kv   = b_idx * H_kv + hkv_idx;   // indexes k_hat/v_hat (H_kv-wide)
    uint bh_q    = b_idx * H_q + h_idx;      // indexes q/out (H_q-wide)

    float running_m = -INFINITY;
    float running_d = 0.0f;
    // DSLOTS_C = ceil(D/32), baked in per-D by the kernel factory -- same
    // sizing as scalar_affine_attend.metal, rather than the D<=256 worst case.
    constexpr uint kDSlots = DSLOTS_C;
    float my_out[kDSlots];
    for (uint i = 0; i < kDSlots; ++i) my_out[i] = 0.0f;
    uint n_owned = kDSlots;

    for (uint sk = sg; sk < S_kv; sk += NSG) {
        float partial_dot = 0.0f;
        for (uint d = lane; d < D; d += 32u) {
            uint q_off = (bh_q * S_q + sq_idx) * D + d;
            uint k_off = (bh_kv * S_kv + sk) * D + d;
            partial_dot += float(q[q_off]) * float(k_hat[k_off]);
        }
        float score = simd_sum(partial_dot) * scale;

        float m_new  = metal::max(running_m, score);
        float factor = metal::exp(running_m - m_new);
        float w      = metal::exp(score - m_new);
        running_d    = running_d * factor + w;
        running_m    = m_new;
        for (uint i = 0; i < n_owned; ++i) my_out[i] *= factor;

        for (uint d = lane; d < D; d += 32u) {
            uint v_off  = (bh_kv * S_kv + sk) * D + d;
            uint out_i  = (d - lane) / 32u;
            my_out[out_i] += w * float(v_hat[v_off]);
        }
    }

    // ----- merge the NSG partial softmaxes through threadgroup memory -----
    // sh_o is half and holds out_s / d_s (a convex combination of V, bounded by
    // max|V|), exactly as in scalar_affine_attend.metal; the merge re-applies
    // d_s. sh_m/sh_d stay fp32 because they feed exp() rescaling.
    threadgroup float sh_m[NSG_C];
    threadgroup float sh_d[NSG_C];
    threadgroup half  sh_o[NSG_C * DSLOTS_C * 32];

    float inv_local = running_d > 0.0f ? 1.0f / running_d : 0.0f;
    for (uint i = 0; i < n_owned; ++i) {
        sh_o[(sg * kDSlots + i) * 32u + lane] = half(my_out[i] * inv_local);
    }
    if (lane == 0) {
        sh_m[sg] = running_m;
        sh_d[sg] = running_d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float gm = -INFINITY;
        for (uint s = 0; s < NSG; ++s) gm = metal::max(gm, sh_m[s]);
        float gd = 0.0f;
        for (uint s = 0; s < NSG; ++s) gd += sh_d[s] * metal::exp(sh_m[s] - gm);

        for (uint i = 0; i < n_owned; ++i) {
            float acc = 0.0f;
            for (uint s = 0; s < NSG; ++s) {
                acc += float(sh_o[(s * kDSlots + i) * 32 + lane])
                     * sh_d[s] * metal::exp(sh_m[s] - gm);
            }
            uint d = lane + i * 32u;
            if (d < D) {
                uint out_off = (bh_q * S_q + sq_idx) * D + d;
                out[out_off] = half(acc / gd);
            }
        }
    }
