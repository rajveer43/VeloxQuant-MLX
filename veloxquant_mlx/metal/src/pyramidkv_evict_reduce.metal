// Sink-protected argmin for PyramidKV cumulative scores.
// One threadgroup handles one flattened batch/head group.
uint bh = uint(threadgroup_position_in_grid.x);
uint lane = thread_position_in_threadgroup.x;
uint sg = thread_position_in_threadgroup.y;
uint nsg = uint(NSG_C);
uint total_threads = 32u * nsg;
uint n_total = uint(scores_shape[1]);
uint n_sink = uint(n_sink_arr[0]);
uint tid = sg * 32u + lane;

float best = INFINITY;
uint best_idx = 0xffffffffu;
for (uint i = tid; i < n_total; i += total_threads) {
    if (i < n_sink) continue;
    float score = scores[bh * n_total + i];
    // NaNs sort as +infinity; ties select the earliest eligible row.
    if (isnan(score)) score = INFINITY;
    if (score < best || (score == best && i < best_idx)) {
        best = score;
        best_idx = i;
    }
}

for (uint offset = 16u; offset > 0u; offset >>= 1u) {
    float other = simd_shuffle_xor(best, offset);
    uint other_idx = simd_shuffle_xor(best_idx, offset);
    if (other < best || (other == best && other_idx < best_idx)) {
        best = other;
        best_idx = other_idx;
    }
}

threadgroup float shared_best[NSG_C];
threadgroup uint shared_idx[NSG_C];
if (lane == 0u) {
    shared_best[sg] = best;
    shared_idx[sg] = best_idx;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid == 0u) {
    float final_best = shared_best[0];
    uint final_idx = shared_idx[0];
    for (uint i = 1u; i < nsg; ++i) {
        if (shared_best[i] < final_best ||
            (shared_best[i] == final_best && shared_idx[i] < final_idx)) {
            final_best = shared_best[i];
            final_idx = shared_idx[i];
        }
    }
    evict_idx[bh] = int(final_idx);
}
