// scalar_quantize.metal
// Extracted from veloxquant_mlx/metal/_scalar_quant.py (_SCALAR_QUANTIZE_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    constexpr int N_CENTS = 1 << B_BITS;

    uint  elem     = thread_position_in_grid.x;
    float val      = float(x[elem]);
    int   best     = 0;
    float best_dist = INFINITY;

    for (int j = 0; j < N_CENTS; ++j) {
        float d    = val - centroids[j];
        float dist = d * d;
        if (dist < best_dist) { best_dist = dist; best = j; }
    }
    indices[elem] = uint8_t(best);
