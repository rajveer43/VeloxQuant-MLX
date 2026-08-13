// rabitq_encode.metal
// Extracted from veloxquant_mlx/metal/_rabitq_encode.py (_RABITQ_ENCODE_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    threadgroup float buf[MAX_D];
    threadgroup float l1_partials[32];

    uint n    = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint D    = uint(MAX_D);

    // 1. Load + diagonal sign flip
    float v = float(keys[n * D + lane]) * diag[lane];
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

    // 4. Sign bits: one ballot per SIMD-group = 4 packed bytes
    uint sg = lane / 32u;
    uint sl = lane % 32u;
    simd_vote ballot = simd_ballot(y >= 0.0f);
    uint mask = uint(static_cast<simd_vote::vote_t>(ballot));

    if (sl == 0u) {
        uint start = sg * 4u;
        for (uint j = 0u; j < 4u && (start + j) < uint(N_BYTES); ++j) {
            k_bits[n * uint(N_BYTES) + start + j] = uint8_t((mask >> (8u * j)) & 0xFFu);
        }
    }

    // 5. L1 magnitude: simd_sum, then combine SIMD-group partials
    float partial = simd_sum(metal::abs(y));
    if (sl == 0u) l1_partials[sg] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0u) {
        uint n_sg = (D + 31u) / 32u;
        float total = 0.0f;
        for (uint s = 0; s < n_sg; ++s) total += l1_partials[s];
        k_mag[n] = total / float(D);
    }
