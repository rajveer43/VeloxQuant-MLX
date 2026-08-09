// scalar_dequantize.metal
// Extracted from veloxquant_mlx/metal/_scalar_quant.py (_SCALAR_DEQUANTIZE_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    uint elem    = thread_position_in_grid.x;
    x_hat[elem]  = half(centroids[uint(indices[elem])]);
