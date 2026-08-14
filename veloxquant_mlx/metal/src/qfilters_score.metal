// qfilters_score.metal
// Read as plain text at import time via _read_kernel_source() and JIT-compiled
// by mx.fast.metal_kernel at call time; it is NOT separately compiled (same
// pattern as h2o_evict_reduce.metal / keyformer_evict_reduce.metal, issue #64).
//
// Dispatch 1 of 2 for Q-Filters fused eviction. Computes the projection score
//
//     score[bh][i] = sign * dot(keys[bh][i], filter_dir[bh])
//
// for every cached row, which is the Q-Filters ranking statistic (paper
// Theorem 3.3: E<Q_i, K_j> ~= kappa^h <K_j, u^h> with kappa^h > 0, so ranking
// by this projection ranks by expected attention logit).
//
// Unlike h2o/keyformer's dispatch 1, this is NOT a reduction to a single
// evicted index: Q-Filters evicts a whole block down to `budget` in one shot,
// so this kernel emits a full [BH, n_total] score array which dispatch 2
// thresholds against. Selecting the threshold itself stays on the MLX side
// (mx.sort/partition), which is where a well-tuned top-k already lives.
//
// Protection is folded in here rather than in the apply kernel so that the
// threshold computed downstream operates on already-protected values:
//   - the first n_sink rows (sink tokens, protected by leading array index)
//   - the last n_recent rows (most recent, protected by trailing array index)
// Both are forced to +INFINITY so they always survive a "keep the highest"
// selection, mirroring qfilters_update's `protect` array exactly.
//
// Grid:        (BH * n_total, 1, 1) — one thread per (group, row) pair.
// Threadgroup: (min(n_total, 256), 1, 1).
//
// Each thread walks D elements of one key row and accumulates the dot product
// in fp32 regardless of the fp16 key storage, matching the pure-MLX path's
// `keys.astype(mx.float32) @ filter_dir` accumulation dtype so the two agree
// to fp32 rounding rather than fp16.

    uint gid = thread_position_in_grid.x;

    uint BH      = uint(keys_shape[0]);        // keys: [BH, n_total, D]
    uint n_total = uint(keys_shape[1]);
    uint D       = uint(keys_shape[2]);

    uint total_rows = BH * n_total;
    if (gid >= total_rows) return;

    uint bh = gid / n_total;
    uint i  = gid % n_total;

    uint n_sink   = uint(n_sink_arr[0]);
    uint n_recent = uint(n_recent_arr[0]);
    float sign    = sign_arr[0];

    // Clamp so sink+recent protection can never exceed n_total, mirroring
    // qfilters.py's min(n_sink, n_total) / min(recent, ...) clamps.
    uint recent_start = (n_recent >= n_total) ? 0u : (n_total - n_recent);

    if (i < n_sink || i >= recent_start) {
        scores_out[gid] = INFINITY;
        return;
    }

    uint key_off    = (bh * n_total + i) * D;
    uint filter_off = bh * D;

    float acc = 0.0f;
    for (uint d = 0u; d < D; ++d) {
        acc += float(keys[key_off + d]) * filter_dir[filter_off + d];
    }

    scores_out[gid] = sign * acc;
