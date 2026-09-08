// Copy K from the candidate buffer and V from either the retained buffer or
// the virtual appended row. This removes a full V concatenate per eviction.
uint gid = thread_position_in_grid.x;
uint bh_count = uint(keys_shape[0]);
uint n = uint(keys_shape[1]);
uint d_count = uint(keys_shape[2]);
uint kept = n - 1u;
if (gid >= bh_count * kept * d_count) return;
uint d = gid % d_count;
uint row = gid / d_count;
uint bh = row / kept;
uint j = row % kept;
uint src = j + uint(j >= evicted[bh]);
uint key_offset = (bh * n + src) * d_count + d;
keys_out[gid] = keys[key_offset];
if (src < kept) {
    values_out[gid] = values_old[(bh * kept + src) * d_count + d];
} else {
    values_out[gid] = values_new[bh * d_count + d];
}
