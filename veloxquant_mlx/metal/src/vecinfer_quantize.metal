// vecinfer_quantize.metal
// Extracted from veloxquant_mlx/metal/_vecinfer.py (_QUANTIZE_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    uint vec_idx = thread_position_in_grid.x;
    uint N_total = x_shape[0];
    if (vec_idx >= N_total) return;

    uint n_centroids = codebook_shape[0];
    uint sub_dim     = codebook_shape[1];
    uint x_base      = vec_idx * sub_dim;

    float best_dist = INFINITY;
    uint  best_idx  = 0;

    for (uint c = 0; c < n_centroids; ++c) {
        uint  cb_base = c * sub_dim;
        float dist    = 0.0f;
        for (uint i = 0; i < sub_dim; ++i) {
            float d = float(x[x_base + i]) - float(codebook[cb_base + i]);
            dist += d * d;
        }
        if (dist < best_dist) { best_dist = dist; best_idx = c; }
    }
    out[vec_idx] = best_idx;
