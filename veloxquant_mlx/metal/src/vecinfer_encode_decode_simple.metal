// vecinfer_encode_decode_simple.metal
// Extracted from veloxquant_mlx/metal/_vecinfer.py (_ENCODE_DECODE_SIMPLE_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).
//
// One thread per (token, sub-vector): nearest-centroid search, then write the
// index and the chosen centroid's sub_dim values. No threadgroup memory or
// barriers (the previous layout used D threads per token with only
// 1/sub_dim of them searching).

    uint gid     = thread_position_in_grid.x;
    uint n_sub   = params[4];
    uint sub_dim = params[5];
    uint n_cents = params[6];
    uint total   = params[0] * params[1] * params[2] * n_sub;
    if (gid >= total) return;

    uint x_base = gid * sub_dim;

    float best_dist = INFINITY;
    uint  best_c    = 0;
    for (uint c = 0; c < n_cents; ++c) {
        uint  cb_base = c * sub_dim;
        float dist    = 0.0f;
        for (uint i = 0; i < sub_dim; ++i) {
            float d = float(values[x_base + i]) - float(v_codebook[cb_base + i]);
            dist += d * d;
        }
        if (dist < best_dist) { best_dist = dist; best_c = c; }
    }

    idx_out[gid] = best_c;
    uint cb_base = best_c * sub_dim;
    for (uint i = 0; i < sub_dim; ++i) {
        v_hat_out[x_base + i] = half(float(v_codebook[cb_base + i]));
    }
