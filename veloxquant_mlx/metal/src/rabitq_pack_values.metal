// rabitq_pack_values.metal
// Extracted from veloxquant_mlx/metal/_rabitq_values.py (_PACK_VALUES_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    uint j = thread_position_in_grid.x;
    uint lo = uint(v_idx[2u * j])     & 0xFu;
    uint hi = uint(v_idx[2u * j + 1u]) & 0xFu;
    packed[j] = uint8_t(lo | (hi << 4u));
