// vecinfer_encode_decode_simple.metal
// Extracted from veloxquant_mlx/metal/_vecinfer.py (_ENCODE_DECODE_SIMPLE_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    threadgroup float buf[MAX_D];
    threadgroup uint  idx[MAX_N_SUB];

    uint tg_idx  = threadgroup_position_in_grid.x;
    uint lane    = thread_position_in_threadgroup.x;

    uint H       = params[1];
    uint S       = params[2];
    uint D       = params[3];
    uint n_sub   = params[4];
    uint sub_dim = params[5];
    uint n_cents = params[6];

    uint s_idx = tg_idx % S;
    uint h_idx = (tg_idx / S) % H;
    uint b_idx = tg_idx / (S * H);

    uint val_base = ((b_idx * H + h_idx) * S + s_idx) * D;

    buf[lane] = float(values[val_base + lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
                float d = buf[x_off + i] - float(v_codebook[cb_base + i]);
                dist += d * d;
            }
            if (dist < best_dist) { best_dist = dist; best_c = c; }
        }
        idx[my_sub] = best_c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (my_sub < n_sub) {
        uint c       = idx[my_sub];
        uint cb_base = c * sub_dim;
        buf[lane] = float(v_codebook[cb_base + my_comp]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    v_hat_out[val_base + lane] = half(buf[lane]);

    if (my_comp == 0 && my_sub < n_sub) {
        uint idx_base = ((b_idx * H + h_idx) * S + s_idx) * n_sub;
        idx_out[idx_base + my_sub] = idx[my_sub];
    }
