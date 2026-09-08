// Adjacent lanes copy adjacent dimensions. No RoPE or score bookkeeping.
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
uint offset = (bh * n + src) * d_count + d;
keys_out[gid] = keys[offset];
values_out[gid] = values[offset];
