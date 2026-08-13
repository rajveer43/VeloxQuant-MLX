// bit_pack.metal
// Extracted from veloxquant_mlx/metal/_bit_packing.py (_PACK_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    constexpr int  ELEMS_PER_BYTE = 8 / B_BITS;
    constexpr uint MASK           = (1u << B_BITS) - 1u;

    uint byte_idx = thread_position_in_grid.x;
    uint base     = byte_idx * ELEMS_PER_BYTE;

    uint packed_byte = 0u;
    for (int i = 0; i < ELEMS_PER_BYTE; ++i) {
        uint val = uint(indices[base + i]) & MASK;
        packed_byte |= (val << (i * B_BITS));
    }
    packed[byte_idx] = uint8_t(packed_byte);
