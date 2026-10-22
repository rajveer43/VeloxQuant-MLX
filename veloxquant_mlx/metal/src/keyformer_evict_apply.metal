// keyformer_evict_apply.metal
// Extracted from veloxquant_mlx/metal/_keyformer_evict.py (_KEYFORMER_EVICT_APPLY_SRC) —
// see that file's docstring for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).
//
// Dispatch 2 of 2 for Keyformer-adapted fused eviction. Identical compaction
// + conditional RoPE re-rotation logic to h2o_evict_apply.metal, with one
// extra parallel array compacted alongside keys/values/scores/positions:
// gumbel (the frozen per-position Gumbel noise — see quantizers/keyformer.py)
// must survive the same compaction so keyformer_update's invariant (every
// kept row has a gumbel draw that matches its original insertion position)
// holds after eviction, exactly like scores/positions do.
//
//   - one thread per (bh, output_row) pair, grid = (BH * n_kept, 1, 1).
//   - source index = output_row if output_row < evict_idx[bh],
//                     output_row + 1 otherwise (skips the evicted row).
//   - straight compaction only (#871): the cache reports `offset` as the true
//     absolute step count, so mlx_lm rotates the next query at each
//     survivor's true distance. Renumbering survivors and re-rotating their
//     keys (the previous behavior of this kernel) changed that distance.
//     Every surviving row keeps the exact position and rotation it arrived
//     with; only the evicted row is dropped. `rope_base_arr` is unused now
//     (kept as an input for ABI parity with the Python-side caller).

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

    // ---- values, scores, gumbel, positions, keys: straight copy, no rotation ----
    uint out_row_off = bh * n_kept + j;
    uint src_row_off = bh * n_total + src;

    scores_out[out_row_off]    = scores_mid[src_row_off];
    gumbel_out[out_row_off]    = gumbel_mid[src_row_off];
    positions_out[out_row_off] = positions_mid[src_row_off];

    uint out_vd = out_row_off * D;
    uint src_vd = src_row_off * D;
    for (uint d = 0u; d < D; ++d) {
        values_out[out_vd + d] = values_mid[src_vd + d];
        keys_out[out_vd + d]   = keys_mid[src_vd + d];
    }
