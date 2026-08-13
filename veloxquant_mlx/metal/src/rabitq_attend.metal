// rabitq_attend.metal
// Extracted from veloxquant_mlx/metal/_rabitq_attend.py (_RABITQ_ATTEND_SRC) — see that file's
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
    uint S_kv = uint(k_bits_shape[2]);
    (void)B;

    uint sq_idx = tg % S_q;
    uint h_idx  = (tg / S_q) % H;
    uint b_idx  = tg / (S_q * H);

    uint q_base  = ((b_idx * H + h_idx) * S_q + sq_idx) * D;
    uint kv_base = (b_idx * H + h_idx) * S_kv;

    // Binarize the query once: lane j packs byte j (little-endian bit order,
    // q >= 0 -> bit 1, matching _pack_signs in the RaBitQ quantizer).
    uint my_qbyte = 0u;
    if (lane < uint(N_BYTES)) {
        for (uint t = 0; t < 8u; ++t) {
            float qv = float(q[q_base + lane * 8u + t]);
            my_qbyte |= (qv >= 0.0f ? 1u : 0u) << t;
        }
    }

    float qs = q_scale[(b_idx * H + h_idx) * S_q + sq_idx];

    float running_m = -INFINITY;
    float running_d = 0.0f;

    // Per-lane output accumulator; max 8 slots (D=256, 32 lanes)
    float my_out[8];
    for (int i = 0; i < 8; ++i) my_out[i] = 0.0f;
    uint n_owned = (D + 31u) / 32u;

    for (uint sk = sg; sk < S_kv; sk += uint(NSG_C)) {
        // 1. Hamming distance via XOR + popcount — keys stay packed.
        uint partial_ham = 0u;
        if (lane < uint(N_BYTES)) {
            uint xr = my_qbyte ^ uint(k_bits[(kv_base + sk) * uint(N_BYTES) + lane]);
            // popcount via bit manipulation (portable across Metal versions)
            uint v = xr;
            v = v - ((v >> 1u) & 0x55u);
            v = (v & 0x33u) + ((v >> 2u) & 0x33u);
            v = (v + (v >> 4u)) & 0x0Fu;
            partial_ham = v;
        }
        uint ham = simd_sum(partial_ham);

        // 2. Affine score: <q, k> estimate = (D - 2*ham) * m_q * m_k + bias.
        float score = (float(D) - 2.0f * float(ham)) * qs * k_mag[kv_base + sk]
                    + k_const[kv_base + sk];

        // 3. Online softmax update
        float m_new  = metal::max(running_m, score);
        float factor = metal::exp(running_m - m_new);
        float w      = metal::exp(score     - m_new);
        running_d    = running_d * factor + w;
        running_m    = m_new;

        for (uint i = 0; i < n_owned; ++i) my_out[i] *= factor;

        // 4. 4-bit value codebook gather + weighted accumulate.
        // V_PACKED is a compile-time constant, so the branch folds away:
        // packed stores two 4-bit indices per byte (lo nibble = even dim).
        for (uint i = lane; i < D; i += 32u) {
            uint idx;
            if (V_PACKED != 0) {
                uint byte = uint(v_idx[(kv_base + sk) * (D >> 1u) + (i >> 1u)]);
                idx = (i & 1u) ? (byte >> 4u) : (byte & 0xFu);
            } else {
                idx = uint(v_idx[(kv_base + sk) * D + i]);
            }
            float vi = v_cents[idx];
            my_out[(i - lane) / 32u] += w * vi;
        }
    }

    // Publish per-SIMD-group partials, then merge across SIMD-groups.
    threadgroup float tg_m[NSG_C];
    threadgroup float tg_d[NSG_C];
    threadgroup float tg_out[NSG_C * MAX_D];

    if (lane == 0u) {
        tg_m[sg] = running_m;
        tg_d[sg] = running_d;
    }
    for (uint i = lane; i < D; i += 32u) {
        tg_out[sg * uint(MAX_D) + i] = my_out[(i - lane) / 32u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_star = -INFINITY;
    for (uint s = 0; s < uint(NSG_C); ++s) {
        m_star = metal::max(m_star, tg_m[s]);
    }
    float d_star = 0.0f;
    for (uint s = 0; s < uint(NSG_C); ++s) {
        d_star += metal::exp(tg_m[s] - m_star) * tg_d[s];
    }

    uint flat = sg * 32u + lane;
    for (uint i = flat; i < D; i += 32u * uint(NSG_C)) {
        float acc = 0.0f;
        for (uint s = 0; s < uint(NSG_C); ++s) {
            acc += metal::exp(tg_m[s] - m_star) * tg_out[s * uint(MAX_D) + i];
        }
        out[q_base + i] = half(acc / d_star);
    }
