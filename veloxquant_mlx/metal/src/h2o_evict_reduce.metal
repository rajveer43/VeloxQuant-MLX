// h2o_evict_reduce.metal
// Extracted from veloxquant_mlx/metal/_h2o_evict.py (_H2O_EVICT_REDUCE_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).
//
// Dispatch 1 of 2 for H2O-adapted fused eviction (see
// H2O_METAL_KERNEL_TECH_SPEC.md section 3 for the exact reference math this
// must match bit-for-bit). Finds, for each (batch*head) row group, the
// index of the minimum "protected" score among the n_total = n_kept + 1
// candidate rows (n_kept currently-stored rows plus the one new row that
// h2o_update conceptually appends before evicting). Two independent
// protections, each treated as +inf and unable to win the argmin, matching
// h2o_update's protection logic exactly:
//   - the first n_sink rows (sink tokens, protected by leading array index)
//   - the last n_grace rows (most-recently-arrived tokens, protected by
//     trailing array index -- positions are always kept sorted ascending
//     after eviction, so "most recent" == "highest array index"). This
//     gives every new token n_grace update steps to accumulate real
//     attention mass before it becomes eviction-eligible, fixing a real,
//     confirmed-on-real-models problem: without it, a brand-new token's
//     starting score of 0.0 makes it (almost always) the argmin target the
//     instant the cache is full, freezing the kept set on whichever tokens
//     filled the budget first and never admitting anything generated
//     afterward. See h2o.py's module docstring.
//
// One threadgroup handles one (batch*head) group. TG_SIZE threads grid-stride
// over the n_total candidate rows, each thread tracking its own local
// (min_value, min_index) pair, then a SIMD-group butterfly reduction
// (simd_shuffle_xor) combines the 32 lanes of each SIMD-group, and a final
// threadgroup-memory merge (same idiom as scalar_affine_attend.metal's
// cross-SIMD-group softmax merge) combines the NSG SIMD-groups into one
// (value, index) pair per threadgroup, written to evict_idx[bh].
//
// Tie-break: ties must resolve toward the LOWEST index, matching mx.argmin's
// documented behavior (h2o_update relies on mx.argmin(protected) today).
// This is enforced by only overwriting the running (min_value, min_index)
// when a strictly smaller value is found — later-encountered equal values
// never replace an earlier, lower index.

    uint bh    = threadgroup_position_in_grid.x;
    uint lane  = thread_position_in_threadgroup.x;
    uint sg    = thread_position_in_threadgroup.y;
    uint NSG   = uint(NSG_C);
    uint TG    = 32u * NSG;

    uint n_kept = uint(scores_mid_shape[1]);   // scores_mid: [BH, n_total]
    uint n_total = n_kept;                      // caller passes n_total in this axis
    uint n_sink  = uint(n_sink_arr[0]);
    uint n_grace = uint(n_grace_arr[0]);
    // Clamp so sink+grace protection can never exceed n_total, mirroring
    // h2o.py's min(n_sink, n_total) / min(grace, n_total) clamps.
    uint grace_start = (n_grace >= n_total) ? 0u : (n_total - n_grace);

    uint tid = sg * 32u + lane;

    float local_min = INFINITY;
    uint  local_idx  = 0xFFFFFFFFu;

    for (uint i = tid; i < n_total; i += TG) {
        bool protected_row = (i < n_sink) || (i >= grace_start);
        float v = protected_row ? INFINITY : scores_mid[bh * n_total + i];
        if (v < local_min) {
            local_min = v;
            local_idx = i;
        }
    }

    // SIMD-group butterfly reduction: combine 32 lanes -> 1 (value, index).
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other_min = simd_shuffle_xor(local_min, offset);
        uint  other_idx = simd_shuffle_xor(local_idx, offset);
        if (other_min < local_min ||
            (other_min == local_min && other_idx < local_idx)) {
            local_min = other_min;
            local_idx = other_idx;
        }
    }

    // Lane 0 of each SIMD-group now holds that group's (min, idx). Stash to
    // threadgroup memory and merge across SIMD-groups.
    threadgroup float sh_min[NSG_C];
    threadgroup uint  sh_idx[NSG_C];

    if (lane == 0u) {
        sh_min[sg] = local_min;
        sh_idx[sg] = local_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float best_min = sh_min[0];
        uint  best_idx = sh_idx[0];
        for (uint s = 1u; s < NSG; ++s) {
            if (sh_min[s] < best_min ||
                (sh_min[s] == best_min && sh_idx[s] < best_idx)) {
                best_min = sh_min[s];
                best_idx = sh_idx[s];
            }
        }
        evict_idx[bh] = int(best_idx);
    }
