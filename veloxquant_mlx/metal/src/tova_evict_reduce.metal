// One threadgroup per batch/KV-head; earliest-index ties at every level.
uint bh = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
uint sg = thread_position_in_threadgroup.y;
uint tid = sg * 32u + lane;
uint n = uint(weights_shape[1]);
float best = INFINITY;
uint index = 0xFFFFFFFFu;
for (uint i = sink[0] + tid; i < n; i += 32u * NSG) {
    float v = weights[bh * n + i];
    if (v < best || (v == best && i < index)) {
        best = v;
        index = i;
    }
}
for (uint delta = 16u; delta > 0u; delta >>= 1u) {
    float v = simd_shuffle_xor(best, delta);
    uint i = simd_shuffle_xor(index, delta);
    if (v < best || (v == best && i < index)) {
        best = v;
        index = i;
    }
}
threadgroup float minima[NSG];
threadgroup uint indices[NSG];
if (lane == 0u) {
    minima[sg] = best;
    indices[sg] = index;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0u) {
    best = minima[0];
    index = indices[0];
    for (uint s = 1u; s < NSG; ++s) {
        if (minima[s] < best || (minima[s] == best && indices[s] < index)) {
            best = minima[s];
            index = indices[s];
        }
    }
    // Never pass an invalid index to apply, including all-NaN input.
    evicted[bh] = index == 0xFFFFFFFFu ? sink[0] : index;
}
