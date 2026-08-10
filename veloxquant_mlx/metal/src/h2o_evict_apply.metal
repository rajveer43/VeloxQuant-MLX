// h2o_evict_apply.metal
// Extracted from veloxquant_mlx/metal/_h2o_evict.py (_H2O_EVICT_APPLY_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).
//
// Dispatch 2 of 2 for H2O-adapted fused eviction (see
// H2O_METAL_KERNEL_TECH_SPEC.md section 3). Given evict_idx[bh] (from
// h2o_evict_reduce.metal) and the n_total = n_kept + 1 candidate rows
// (n_kept stored rows plus the appended new row, all already concatenated
// by the caller into *_mid arrays), produces the n_kept surviving rows:
//   - one thread per (bh, output_row) pair, grid = (BH * n_kept, 1, 1).
//   - source index = output_row if output_row < evict_idx[bh],
//                     output_row + 1 otherwise (skips the evicted row).
//   - position/RoPE handling per H2O_METAL_KERNEL_TECH_SPEC.md section 3
//     step 7: rows whose (source) position is <= evicted_pos are copied
//     bit-identically; rows whose position is > evicted_pos shift down by
//     exactly 1 and get re-rotated via the same delta-rotation formula as
//     rope_remap_positions (a2ats_rope.py) — rotate by
//     delta = new_position - old_position, NeoX-style (first/second-half
//     split) RoPE, matching MLX's default `traditional=False` convention.

    uint gid = thread_position_in_grid.x;

    uint n_kept = uint(keys_mid_shape[1]) - 1u;  // keys_mid: [BH, n_total, D]
    uint D      = uint(keys_mid_shape[2]);
    uint n_total = n_kept + 1u;

    uint bh  = gid / n_kept;
    uint j   = gid % n_kept;   // output row index within this bh group

    uint BH = uint(keys_mid_shape[0]);
    if (bh >= BH) return;

    int evict_i = evict_idx[bh];
    uint src = (j < uint(evict_i)) ? j : (j + 1u);

    int evicted_pos = positions_mid[bh * n_total + uint(evict_i)];
    int src_pos     = positions_mid[bh * n_total + src];
    bool needs_shift = src_pos > evicted_pos;
    int new_pos = needs_shift ? (src_pos - 1) : src_pos;

    // ---- values, scores, positions: straight copy (never rotated) ----
    uint out_row_off = bh * n_kept + j;
    uint src_row_off = bh * n_total + src;

    scores_out[out_row_off]    = scores_mid[src_row_off];
    positions_out[out_row_off] = new_pos;

    uint out_vd = out_row_off * D;
    uint src_vd = src_row_off * D;
    for (uint d = 0u; d < D; ++d) {
        values_out[out_vd + d] = values_mid[src_vd + d];
    }

    // ---- keys: bit-identical copy, or delta-rotate ----
    if (!needs_shift) {
        for (uint d = 0u; d < D; ++d) {
            keys_out[out_vd + d] = keys_mid[src_vd + d];
        }
        return;
    }

    // Delta rotation: rotate_by(new_pos - src_pos) == rotate_by(-1).
    // NeoX-style split: first half / second half pair (d, d + D/2).
    uint half_d = D / 2u;
    float base = rope_base_arr[0];
    float delta = float(new_pos - src_pos);

    for (uint d = 0u; d < half_d; ++d) {
        // Matches a2ats_rope.py's _rope_cos_sin exactly:
        // inv_freq[d] = 1 / base^(d / half_d).
        float inv_freq = 1.0f / metal::pow(base, float(d) / float(half_d));
        float angle = delta * inv_freq;
        float c = metal::cos(angle);
        float s = metal::sin(angle);

        float x1 = float(keys_mid[src_vd + d]);
        float x2 = float(keys_mid[src_vd + d + half_d]);

        keys_out[out_vd + d]         = half(x1 * c - x2 * s);
        keys_out[out_vd + d + half_d] = half(x1 * s + x2 * c);
    }
