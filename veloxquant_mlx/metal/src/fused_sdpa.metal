// fused_sdpa.metal
// Extracted from veloxquant_mlx/metal/fused_sdpa.py (_FUSED_SDPA_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    // 32 lanes per threadgroup; one threadgroup per (B*H_q, S_q) output cell
    uint q_head_idx = thread_position_in_grid.x / 32;
    uint q_pos      = thread_position_in_grid.y;
    uint lane       = thread_position_in_threadgroup.x;

    // Unpack shape pack
    uint H_q       = params[0];
    uint H_kv      = params[1];
    uint S_q       = params[2];
    uint S_kv      = params[3];
    uint D         = params[4];
    uint n_sub     = params[5];
    uint sub_dim   = params[6];
    uint n_sub_v   = params[7];
    uint sub_dim_v = params[8];
    uint flags     = params[9];

    bool causal       = (flags & 1u) != 0u;
    bool use_window   = (flags & 2u) != 0u;
    uint window_width = slide_arr[0];
    float scale       = scale_arr[0];

    if (q_pos >= S_q) { return; }

    uint batch  = q_head_idx / H_q;
    uint h_q    = q_head_idx % H_q;
    uint h_kv   = (h_q * H_kv) / H_q;        // GQA integer div

    uint q_base   = q_head_idx * S_q * D + q_pos * D;
    uint k_base_b = batch * H_kv * S_kv;     // K row stride
    uint out_base = q_head_idx * S_q * D + q_pos * D;

    constexpr uint kNCentroids = LUT_N_CENTROIDS;     // compile-time
    constexpr uint kMaxLut     = LUT_MAX_SIZE;        // n_sub * n_centroids
    constexpr uint kDimsPerLane = MAX_D / 32;         // output dims owned by each lane

    // LUT lives in threadgroup (shared across all lanes in the SIMD group).
    // t_out and tg_d_shared have been replaced by lane-local registers:
    //   my_out[kDimsPerLane]  — each lane owns D/32 output dimensions
    //   running_d             — denominator accumulated in registers
    threadgroup float lut[kMaxLut];

    // -----------------------------------------------------------------
    // Phase 0: fill the per-query LUT cooperatively.
    //   lut[sub * n_centroids + c] = q_sub_vec dot k_codebook_row
    //
    // Stripe (sub, centroid) pairs across 32 lanes — each lane computes
    // one dot product independently.  The pragma enables FMA contraction
    // on the inner dot loop; relaxed still honors INF/NaN unlike fast.
    // -----------------------------------------------------------------
    uint lut_total = n_sub * kNCentroids;
    for (uint idx = lane; idx < lut_total; idx += 32) {
        uint sub = idx / kNCentroids;
        uint c   = idx % kNCentroids;
        uint q_sub_off = q_base + sub * sub_dim;
        uint cb_off    = c * sub_dim;
        float dot = 0.0f;
        for (uint i = 0; i < sub_dim; ++i) {
            dot += q[q_sub_off + i] * k_codebook[cb_off + i];
        }
        lut[idx] = dot;
    }

    // -----------------------------------------------------------------
    // Phase 1: initialize lane-local running stats.
    //
    // my_out holds D/32 output floats owned by this lane.
    // running_m and running_d are pure register scalars — no threadgroup
    // writes needed.  The single barrier here lets all lanes finish writing
    // the LUT before any lane reads it in Phase 2.
    // -----------------------------------------------------------------
    float my_out[kDimsPerLane];
    for (uint k = 0; k < kDimsPerLane; ++k) {
        my_out[k] = 0.0f;
    }
    float running_m = -INFINITY;
    float running_d = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);   // sole barrier: LUT ready

    // Convention: queries align to the tail of S_kv (standard decode pattern)
    uint q_abs = (S_kv - S_q) + q_pos;

    // -----------------------------------------------------------------
    // Phase 2: tiled online softmax + V accumulation — zero additional barriers
    // -----------------------------------------------------------------
    for (uint tile_start = 0; tile_start < S_kv; tile_start += 32) {
        uint k_pos = tile_start + lane;

        // Per-lane mask resolution
        float score = -INFINITY;
        bool valid = (k_pos < S_kv);
        if (valid && causal && k_pos > q_abs) valid = false;
        if (valid && use_window) {
            if (q_abs + 1u > window_width && k_pos + window_width < q_abs + 1u) {
                valid = false;
            }
        }
        if (valid) {
            uint k_row_idx = (k_base_b + h_kv * S_kv + k_pos) * n_sub;
            float s = 0.0f;
            for (uint sub = 0; sub < n_sub; ++sub) {
                uint c = k_indices[k_row_idx + sub];
                s += lut[sub * kNCentroids + c];
            }
            score = s * scale;
        }

        // SIMD-wide max — broadcasts result to all lanes, no barrier
        float tile_max = simd_max(score);
        if (!isfinite(tile_max)) { continue; }   // whole tile masked

        // Lane 0 computes the new max and rescale factor; simd_broadcast_first
        // propagates to all lanes without a threadgroup barrier.
        float m_old     = running_m;
        float m_new_l0  = max(m_old, tile_max);
        // metal::precise::exp guarantees exp(-inf)==0.0 regardless of math mode.
        float factor_l0 = isfinite(m_old) ? metal::precise::exp(m_old - m_new_l0) : 0.0f;
        float m_new  = simd_broadcast_first(m_new_l0);
        float factor = simd_broadcast_first(factor_l0);
        running_m = m_new;

        float w = metal::precise::exp(score - m_new);
        if (!valid) w = 0.0f;

        // Denominator: fully register-local, synced via simd_broadcast_first.
        // simd_sum broadcasts the tile sum to all lanes — no threadgroup write.
        float tile_w_sum = simd_sum(w);
        running_d = running_d * factor + simd_broadcast_first(tile_w_sum);

        // Rescale lane's owned output slice and accumulate V contribution.
        // Each lane owns output dimensions [lane*(D/32) .. (lane+1)*(D/32)-1].
        // No threadgroup array involved — pure register arithmetic.
        for (uint k = 0; k < kDimsPerLane; ++k) {
            my_out[k] *= factor;
        }

        // V accumulation: lane owns kDimsPerLane contiguous output dims.
        // Walk all 32 tile-mates via simd_shuffle to get their weights.
        for (uint k = 0; k < kDimsPerLane; ++k) {
            uint dim   = lane * kDimsPerLane + k;
            uint sub_v = dim / sub_dim_v;
            uint comp  = dim % sub_dim_v;
            float acc = 0.0f;
            for (uint l = 0; l < 32; ++l) {
                uint k_l = tile_start + l;
                if (k_l >= S_kv) break;
                float w_l = simd_shuffle(w, l);
                if (w_l == 0.0f) continue;
                uint v_row_idx = (k_base_b + h_kv * S_kv + k_l) * n_sub_v;
                uint v_c = v_indices[v_row_idx + sub_v];
                acc += w_l * v_codebook[v_c * sub_dim_v + comp];
            }
            my_out[k] += acc;
        }
        // No threadgroup barrier — all output state is lane-local.
    }

    // -----------------------------------------------------------------
    // Phase 3: normalize and write each lane's owned dims directly to device.
    // running_d is identical in all lanes (kept in sync via simd_broadcast_first).
    // -----------------------------------------------------------------
    float inv_d = 1.0f / max(running_d, 1e-20f);
    for (uint k = 0; k < kDimsPerLane; ++k) {
        uint dim = lane * kDimsPerLane + k;
        out[out_base + dim] = my_out[k] * inv_d;
    }
