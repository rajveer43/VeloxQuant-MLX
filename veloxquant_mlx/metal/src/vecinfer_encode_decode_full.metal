// vecinfer_encode_decode_full.metal
// Extracted from veloxquant_mlx/metal/_vecinfer.py (_ENCODE_DECODE_FULL_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    threadgroup float buf_in[MAX_D];
    threadgroup float buf_tg[MAX_D];
    threadgroup uint  idx[MAX_N_SUB];

    uint tg_idx  = threadgroup_position_in_grid.x;
    uint lane    = thread_position_in_threadgroup.x;

    uint H           = params[1];
    uint S           = params[2];
    uint D           = params[3];
    uint n_sub       = params[4];
    uint sub_dim     = params[5];
    uint n_cents     = params[6];
    uint has_smooth  = params[7];
    uint smooth_rows = params[8];

    uint s_idx = tg_idx % S;
    uint h_idx = (tg_idx / S) % H;
    uint b_idx = tg_idx / (S * H);

    uint key_base    = ((b_idx * H + h_idx) * S + s_idx) * D;
    uint smooth_base = (has_smooth ? (h_idx % smooth_rows) * D : 0);

    // Phase A: load + optional smooth divide
    float val = float(keys[key_base + lane]);
    if (has_smooth) {
        float s = float(smooth[smooth_base + lane]);
        val = (s > 1e-8f) ? val / s : val;
    }
    buf_in[lane] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase B: WHT via matvec — thread lane computes dot(H_mat[lane, :], buf_in)
    {
        float dot = 0.0f;
        uint row_base = lane * D;
        for (uint c = 0; c < D; ++c) {
            dot += float(H_mat[row_base + c]) * buf_in[c];
        }
        buf_tg[lane] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase C: quantize — one leader per sub-vector scans all centroids
    uint my_sub  = lane / sub_dim;
    uint my_comp = lane % sub_dim;

    if (my_comp == 0 && my_sub < n_sub) {
        float best_dist = INFINITY;
        uint  best_c    = 0;
        uint  x_off     = my_sub * sub_dim;

        for (uint c = 0; c < n_cents; ++c) {
            float dist    = 0.0f;
            uint  cb_base = c * sub_dim;
            for (uint i = 0; i < sub_dim; ++i) {
                float d = buf_tg[x_off + i] - float(k_codebook[cb_base + i]);
                dist += d * d;
            }
            if (dist < best_dist) { best_dist = dist; best_c = c; }
        }
        idx[my_sub] = best_c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase D: dequantize — gather winning centroid into buf_in
    {
        uint c       = idx[my_sub];
        uint cb_base = c * sub_dim;
        buf_in[lane] = float(k_codebook[cb_base + my_comp]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase E: inv-WHT — H_mat.T[lane, c] = H_mat[c, lane]
    {
        float dot = 0.0f;
        for (uint c = 0; c < D; ++c) {
            dot += float(H_mat[c * D + lane]) * buf_in[c];
        }
        buf_tg[lane] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase F: inverse smooth multiply
    float out_val = buf_tg[lane];
    if (has_smooth) {
        float s = float(smooth[smooth_base + lane]);
        out_val *= s;
    }

    // Phase G: write outputs
    k_hat_out[key_base + lane] = half(out_val);

    if (my_comp == 0 && my_sub < n_sub) {
        uint idx_base = ((b_idx * H + h_idx) * S + s_idx) * n_sub;
        idx_out[idx_base + my_sub] = idx[my_sub];
    }
