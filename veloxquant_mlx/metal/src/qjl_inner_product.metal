// qjl_inner_product.metal
// Extracted from veloxquant_mlx/metal/_qjl.py (_QJL_INNER_PRODUCT_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    constexpr float SQRT_PI_OVER_2 = 1.2533141373155002f;

    uint flat_idx = threadgroup_position_in_grid.x;
    uint lane     = thread_index_in_simdgroup;
    uint H        = uint(q_proj_shape[0]);
    uint m        = uint(q_proj_shape[1]);
    uint S_kv     = uint(norms_shape[0]);

    uint h_idx = flat_idx % H;
    uint s_idx = flat_idx / H;

    float acc      = 0.0f;
    uint  sign_row = (s_idx * H + h_idx) * ((m + 7u) / 8u);
    uint  q_row    = h_idx * m;

    for (uint chunk = lane; chunk < m; chunk += 32u) {
        uint  byte_idx  = chunk / 8u;
        uint  bit_pos   = chunk % 8u;
        uint  sign_bit  = (uint(packed_signs[sign_row + byte_idx]) >> bit_pos) & 1u;
        float sign_pm1  = (sign_bit == 1u) ? 1.0f : -1.0f;
        acc += float(q_proj[q_row + chunk]) * sign_pm1;
    }

    float total = simd_sum(acc);

    if (lane == 0) {
        float norm  = float(norms[s_idx * H + h_idx]);
        float scale = SQRT_PI_OVER_2 / float(m);
        scores[h_idx * S_kv + s_idx] = half(scale * norm * total);
    }
