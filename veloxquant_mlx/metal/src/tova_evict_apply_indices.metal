// Compact K and source lineage. Values are gathered once after the dependent
// multi-token eviction chain completes.
uint gid = thread_position_in_grid.x;
uint bh_count = uint(keys_shape[0]);
uint n = uint(keys_shape[1]);
uint d_count = uint(keys_shape[2]);
uint kept = n - 1u;
uint elements = bh_count * kept * d_count;
if (gid >= elements) return;
uint d = gid % d_count;
uint row = gid / d_count;
uint bh = row / kept;
uint j = row % kept;
uint src = j + uint(j >= evicted[bh]);
uint key_offset = (bh * n + src) * d_count + d;
keys_out[gid] = keys[key_offset];
if (d == 0u) {
    lineage_out[bh * kept + j] = lineage[bh * n + src];
}
