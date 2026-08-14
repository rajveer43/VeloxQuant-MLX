// qfilters_evict_apply.metal
// Read as plain text at import time via _read_kernel_source() and JIT-compiled
// by mx.fast.metal_kernel at call time; it is NOT separately compiled (same
// pattern as h2o_evict_apply.metal / keyformer_evict_apply.metal, issue #64).
//
// Dispatch 2 of 2 for Q-Filters fused eviction. Given the per-group scores
// from qfilters_score.metal and a per-group keep-threshold, compacts the
// surviving `budget` rows into the output arrays, preserving original
// temporal order (qfilters_update returns kept tokens in arrival order).
//
// Why this is a gather-by-scan and not h2o's index-shift
// -----------------------------------------------------
// H2O/Keyformer evict exactly ONE row per token, so an output row's source is
// the closed form `j + (j >= evict_idx)`. Q-Filters evicts a whole block down
// to budget at once, so there is no closed form: a thread must know how many
// survivors precede its output slot. One threadgroup per (batch*head) group
// cooperatively scans that group's scores and writes a compacted index list,
// then the copy phase gathers through it.
//
// Tie handling (the correctness crux)
// -----------------------------------
// Thresholding with `score >= thresh` keeps MORE than `budget` rows whenever
// the threshold value is duplicated -- and duplicate scores are common (all
// protected rows share +INFINITY). Overflowing the budget would silently
// break the cache's size guarantee. So rows are admitted in two tiers:
//   - strictly greater than the threshold: always kept;
//   - exactly equal to the threshold: kept only while the remaining quota
//     lasts, scanning low index -> high index.
// This reproduces mx.argsort's lowest-index-first tie-break, matching the
// pure-MLX path's selection on tied scores.
//
// Grid:        (BH * TG_SIZE, 1, 1) — one threadgroup per (batch*head) group.
// Threadgroup: (TG_SIZE, 1, 1).

    uint bh   = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint TG   = threads_per_threadgroup.x;

    uint BH      = uint(keys_mid_shape[0]);      // keys_mid: [BH, n_total, D]
    uint n_total = uint(keys_mid_shape[1]);
    uint D       = uint(keys_mid_shape[2]);
    // Passed explicitly: MLX exposes *_shape params for inputs only, so the
    // output's budget dimension is not readable from keys_out_shape.
    uint budget  = uint(budget_arr[0]);          // keys_out: [BH, budget, D]

    if (bh >= BH) return;

    float thresh = thresh_arr[bh];

    // ---- phase 1: one thread builds the compacted survivor index list ----
    // Serial within the group by design: the scan is inherently sequential
    // (each admission depends on the quota left by all lower indices), and
    // n_total is small enough that a parallel prefix-sum would cost more in
    // barriers than it saves. The D-element copy phase below is the real work
    // and is fully parallel.
    threadgroup uint keep_idx[QF_MAX_BUDGET];
    threadgroup uint n_kept_sh;

    if (lane == 0u) {
        uint n_strict = 0u;
        // Count strictly-greater rows first to size the equal-tier quota.
        for (uint i = 0u; i < n_total; ++i) {
            if (scores[bh * n_total + i] > thresh) n_strict += 1u;
        }
        uint eq_quota = (budget > n_strict) ? (budget - n_strict) : 0u;

        uint w = 0u;
        for (uint i = 0u; i < n_total && w < budget; ++i) {
            float s = scores[bh * n_total + i];
            bool take = false;
            if (s > thresh) {
                take = true;
            } else if (s == thresh && eq_quota > 0u) {
                take = true;
                eq_quota -= 1u;
            }
            if (take) {
                keep_idx[w] = i;
                w += 1u;
            }
        }
        n_kept_sh = w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint n_kept = n_kept_sh;

    // ---- phase 2: parallel gather of the surviving rows ----
    // Grid-stride over (row, dim) pairs so every lane stays busy even when
    // budget is small relative to the threadgroup size.
    uint work = n_kept * D;
    for (uint t = lane; t < work; t += TG) {
        uint j = t / D;          // output row
        uint d = t % D;          // element within the row
        uint src = keep_idx[j];

        uint out_off = (bh * budget + j) * D + d;
        uint src_off = (bh * n_total + src) * D + d;

        keys_out[out_off]   = keys_mid[src_off];
        values_out[out_off] = values_mid[src_off];
    }

    // Scores are carried forward per row (one element each, not D).
    for (uint j = lane; j < n_kept; j += TG) {
        scores_out[bh * budget + j] = scores[bh * n_total + keep_idx[j]];
    }

    // ---- phase 3: pad any unfilled tail ----
    // Reachable only if fewer than `budget` rows cleared the threshold (e.g. a
    // caller-supplied threshold above every score). Zero-fill keeps the output
    // deterministic instead of leaking uninitialized device memory.
    for (uint t = n_kept * D + lane; t < budget * D; t += TG) {
        uint j = t / D;
        uint d = t % D;
        uint out_off = (bh * budget + j) * D + d;
        keys_out[out_off]   = half(0.0);
        values_out[out_off] = half(0.0);
    }
    for (uint j = n_kept + lane; j < budget; j += TG) {
        scores_out[bh * budget + j] = -INFINITY;
    }
