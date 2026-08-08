// hadamard_quantize.metal
// Extracted from veloxquant_mlx/metal/_scalar_quant.py (_HADAMARD_QUANTIZE_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    constexpr int N_CENTS = 1 << B_BITS;

    threadgroup float buf[MAX_D];

    uint tg   = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint D    = uint(MAX_D);

    // 1. Load + diagonal sign flip
    float v = float(x[tg * D + lane]);
    v *= float(diag[lane]);
    buf[lane] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. In-place WHT: range-based parallel butterfly
    for (uint stride = 1; stride < D; stride <<= 1) {
        uint local    = lane % (stride << 1u);
        bool is_upper = local >= stride;
        uint partner  = is_upper ? (lane - stride) : (lane + stride);
        float a = buf[lane];
        float b = buf[partner];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[lane] = is_upper ? (b - a) : (a + b);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. Scale
    float y = buf[lane] * metal::rsqrt(float(D));

    // 4. Nearest-centroid argmin (register-local scan)
    int   best      = 0;
    float best_dist = INFINITY;
    for (int j = 0; j < N_CENTS; ++j) {
        float d    = y - centroids[j];
        float dist = d * d;
        if (dist < best_dist) { best_dist = dist; best = j; }
    }
    indices[tg * D + lane] = uint8_t(best);
