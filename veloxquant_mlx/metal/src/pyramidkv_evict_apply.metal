// Copy-only PyramidKV compaction. No positions or RoPE remapping are used.
uint gid = thread_position_in_grid.x;
uint n_kept = uint(keys_shape[1]) - 1u;
uint d = uint(keys_shape[2]);
uint bh = gid / (n_kept * d);
uint out_row = (gid / d) % n_kept;
uint col = gid % d;
if (bh >= uint(keys_shape[0])) return;

uint evict = uint(evict_idx[bh]);
uint src_row = (out_row < evict) ? out_row : out_row + 1u;
uint dst_base = (bh * n_kept + out_row) * d;
uint src_base = (bh * (n_kept + 1u) + src_row) * d;
keys_out[dst_base + col] = keys[src_base + col];
values_out[dst_base + col] = values[src_base + col];
if (col == 0u)
    scores_out[bh * n_kept + out_row] = scores[bh * (n_kept + 1u) + src_row];
