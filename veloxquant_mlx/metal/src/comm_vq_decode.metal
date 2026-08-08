// comm_vq_decode.metal
// Extracted from veloxquant_mlx/metal/_comm_vq.py (_COMM_VQ_DECODE_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    uint flat = thread_position_in_grid.x;

    uint N = uint(indices_shape[0]);
    uint D = uint(N_CB) * uint(SUB_DIM);

    uint b_idx = flat / D;
    uint d_i   = flat % D;

    if (b_idx >= N) return;

    // Which sub-codebook owns dimension d_i?
    uint cb_i   = d_i / uint(SUB_DIM);
    uint comp_i = d_i % uint(SUB_DIM);

    // Gather centroid value (additive: one sub-codebook per segment)
    uint idx_val = uint(indices[b_idx * uint(N_CB) + cb_i]);
    uint cb_off  = cb_i * uint(CB_SIZE) * uint(SUB_DIM)
                 + idx_val * uint(SUB_DIM)
                 + comp_i;
    float x_val = float(codebook[cb_off]);

    // Apply RoPE: operate on paired dimensions (d_i, d_i + D/2) or (d_i - D/2, d_i)
    uint half_D = D / 2;
    uint pos    = uint(positions[b_idx]);
    uint freq_i = d_i % half_D;   // which frequency dimension

    float inv_f  = inv_freq[freq_i];
    float angle  = float(pos) * inv_f;
    float cos_v  = metal::cos(angle);
    float sin_v  = metal::sin(angle);

    // Partner dimension index
    uint partner_i = (d_i < half_D) ? (d_i + half_D) : (d_i - half_D);

    // We need the partner's pre-RoPE value. For the fused kernel we need a
    // two-phase approach: write pre-RoPE first, then apply RoPE.
    // Since we can't synchronise across threads for different d_i here, we
    // instead write the pre-RoPE value and let a second pass (or the caller)
    // apply RoPE. This is still a net win: we fuse the gather (O(N*D) reads
    // from a large indices array) into one dispatch.
    //
    // For the full fused path (gather + RoPE in one pass) the standard trick
    // is to have each thread compute BOTH halves of its dimension pair. We do
    // that here: thread for d_i < half_D also reads the codebook for d_i+half_D
    // and writes both rotated outputs. Threads for d_i >= half_D skip (output
    // already written by their lower-half partner).

    if (d_i < half_D) {
        // This thread handles the pair (d_i, d_i + half_D)
        uint cb_i2   = partner_i / uint(SUB_DIM);
        uint comp_i2 = partner_i % uint(SUB_DIM);
        uint idx2    = uint(indices[b_idx * uint(N_CB) + cb_i2]);
        uint cb_off2 = cb_i2 * uint(CB_SIZE) * uint(SUB_DIM)
                     + idx2 * uint(SUB_DIM)
                     + comp_i2;
        float x2 = float(codebook[cb_off2]);

        float out0 = x_val * cos_v - x2 * sin_v;
        float out1 = x_val * sin_v + x2 * cos_v;

        out[b_idx * D + d_i]            = half(out0);
        out[b_idx * D + partner_i]      = half(out1);
    }
    // threads with d_i >= half_D do nothing — already written above
