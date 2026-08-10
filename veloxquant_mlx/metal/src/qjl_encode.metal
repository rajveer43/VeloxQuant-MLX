// qjl_encode.metal
// Extracted from veloxquant_mlx/metal/_qjl.py (_QJL_ENCODE_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    uint flat_tg          = threadgroup_position_in_grid.x;
    uint lane             = thread_index_in_simdgroup;
    uint m                = uint(S_shape[0]);
    uint d                = uint(S_shape[1]);
    uint n_simd_per_batch = (m + 31u) / 32u;

    uint b_idx    = flat_tg / n_simd_per_batch;
    uint simd_blk = flat_tg % n_simd_per_batch;
    uint sketch_j = simd_blk * 32u + lane;

    // Dot product dot(S[sketch_j, :], x[b, :])
    float dot_val = 0.0f;
    if (sketch_j < m) {
        uint S_row = sketch_j * d;
        uint x_row = b_idx   * d;
        for (uint i = 0; i < d; ++i) {
            dot_val += float(S[S_row + i]) * float(x[x_row + i]);
        }
    }

    // Norm — only simd_blk 0 computes this; all 32 lanes share work via simd_sum
    if (simd_blk == 0) {
        float x_sq = 0.0f;
        uint  x_row = b_idx * d;
        for (uint i = lane; i < d; i += 32u) {
            float v = float(x[x_row + i]);
            x_sq += v * v;
        }
        float norm_sq = simd_sum(x_sq);
        if (lane == 0) {
            norms[b_idx] = half(metal::sqrt(norm_sq));
        }
    }

    // Pack 32 sign bits into 4 bytes (8 lanes → 1 byte, LSB-first)
    uint sign_bit    = (dot_val >= 0.0f) ? 1u : 0u;
    uint byte_in_blk = lane / 8u;
    uint bit_in_byte = lane % 8u;

    uint packed_byte = 0u;
    for (uint bit = 0; bit < 8u; ++bit) {
        uint src = byte_in_blk * 8u + bit;
        packed_byte |= (simd_shuffle(sign_bit, src) << bit);
    }

    if (bit_in_byte == 0 && sketch_j < m) {
        uint out_byte = b_idx * ((m + 7u) / 8u) + simd_blk * 4u + byte_in_blk;
        packed_signs[out_byte] = uint8_t(packed_byte);
    }
