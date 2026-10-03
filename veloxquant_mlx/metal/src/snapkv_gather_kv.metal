// Shared index map, contiguous dimension lanes, independent K/V input types.
uint gid = thread_position_in_grid.x;
uint groups = uint(keys_shape[0]), n = uint(keys_shape[1]), d = uint(keys_shape[2]);
uint count = uint(indices_shape[1]);
if (gid >= groups * count * d) return;
uint dim = gid % d;
uint row = gid / d;
uint g = row / count;
// Clamp so a corrupt index reads a valid row of this group, never another group's.
uint src = min(uint(max(indices[row], 0)), n - 1);
keys_out[gid] = keys[(g*n + src)*d + dim];
values_out[gid] = values[(g*n + src)*d + dim];
