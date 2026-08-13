// bit_unpack.metal
// Extracted from veloxquant_mlx/metal/_bit_packing.py (_UNPACK_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    constexpr int  ELEMS_PER_BYTE = 8 / B_BITS;
    constexpr uint MASK           = (1u << B_BITS) - 1u;

    uint elem_idx = thread_position_in_grid.x;
    uint byte_idx = elem_idx / ELEMS_PER_BYTE;
    uint bit_off  = (elem_idx % ELEMS_PER_BYTE) * B_BITS;

    indices[elem_idx] = uint8_t((uint(packed[byte_idx]) >> bit_off) & MASK);
