// Fixed-size ordered scatter. Prefix ranks give unique output slots.
uint gid = thread_position_in_grid.x;
uint n = uint(selected_shape[1]);
uint groups = uint(selected_shape[0]);
uint sink = uint(params[0]);
uint count = uint(params[1]);
uint width = n + sink;
if (gid >= groups * width) return;
uint g = gid / width;
uint i = gid % width;
if (i < sink) {
    indices[g * count + i] = int(i);
} else {
    uint offset = g * n + i - sink;
    if (selected[offset]) {
        uint target = sink + uint(ranks[offset]) - 1;
        if (target < count) indices[g * count + target] = int(i);
    }
}
