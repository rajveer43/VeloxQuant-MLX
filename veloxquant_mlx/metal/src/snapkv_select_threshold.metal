// One group per head. Global threshold supplied by MLX; all bookkeeping
// stays in this group. Exactly 256 threads / eight SIMD groups.
uint g = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
uint sg = lane / 32, sl = lane % 32;
uint n = uint(scores_shape[1]);
uint sink = uint(params[0]), k = uint(params[1]), count = sink + k;
float threshold = thresholds[g];
threadgroup uint sums[8];
threadgroup uint state[3]; // equal quota, preceding equals, preceding survivors
uint local_above = 0;
for (uint i = lane; i < n; i += 256) local_above += uint(scores[g*n+i] > threshold);
uint above = simd_sum(local_above);
if (sl == 31) sums[sg] = above;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) {
    uint total = 0;
    for (uint j = 0; j < 8; ++j) total += sums[j];
    state[0] = k - total;
    state[1] = 0;
    state[2] = sink;
}
for (uint i = lane; i < sink; i += 256) indices[g*count+i] = int(i);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint base = 0; base < n; base += 256) {
    uint i = base + lane;
    float v = i < n ? scores[g*n+i] : 0.0f;
    uint equal = uint(i < n && v == threshold);
    uint eq_rank = simd_prefix_inclusive_sum(equal);
    if (sl == 31) sums[sg] = eq_rank;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint eq_before = 0, eq_total = 0;
    for (uint j = 0; j < 8; ++j) {
        if (j < sg) eq_before += sums[j];
        eq_total += sums[j];
    }
    uint take = uint(i < n && (v > threshold ||
        (equal && state[1] + eq_before + eq_rank <= state[0])));
    // Finish all reads before reusing sums for the survivor scan.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rank = simd_prefix_inclusive_sum(take);
    if (sl == 31) sums[sg] = rank;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = 0, total = 0;
    for (uint j = 0; j < 8; ++j) {
        if (j < sg) before += sums[j];
        total += sums[j];
    }
    if (take) indices[g*count + state[2] + before + rank - 1] = int(sink + i);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        state[1] += eq_total;
        state[2] += total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
