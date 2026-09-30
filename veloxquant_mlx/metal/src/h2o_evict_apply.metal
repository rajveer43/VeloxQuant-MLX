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
//   - straight compaction only (#609): the caller (H2OKVCache) reports
//     `offset` as the true absolute step count, not the kept-row count, so
//     mlx_lm always rotates the next query at each survivor's true
//     distance. Renumbering survivors to a gap-free layout and re-rotating
//     their keys to match — the previous behavior of this kernel — changed
//     that true distance and desynced the dot product from the causal mask
//     (built from true positions). Every surviving row keeps the exact
//     position and rotation it arrived with; only the evicted row is
//     dropped. `rope_base_arr` is unused now (kept as an input for ABI
//     parity with the Python-side caller and the reduce kernel's config).

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

    // ---- values, scores, positions, keys: straight copy, no rotation ----
    uint out_row_off = bh * n_kept + j;
    uint src_row_off = bh * n_total + src;

    scores_out[out_row_off]    = scores_mid[src_row_off];
    positions_out[out_row_off] = positions_mid[src_row_off];

    uint out_vd = out_row_off * D;
    uint src_vd = src_row_off * D;
    for (uint d = 0u; d < D; ++d) {
        values_out[out_vd + d] = values_mid[src_vd + d];
        keys_out[out_vd + d]   = keys_mid[src_vd + d];
    }
