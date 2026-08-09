// vecinfer_dequant.metal
// Extracted from veloxquant_mlx/metal/_vecinfer.py (_DEQUANT_SRC) — see that file's
// docstring/comments for the algorithm, memory-layout, and threadgroup
// strategy explanation. This file is read as plain text at import time via
// _read_kernel_source() and JIT-compiled by mx.fast.metal_kernel at call time;
// it is NOT separately compiled (see issue #64).

    uint flat_idx = thread_position_in_grid.x;
    uint N_total  = indices_shape[0];
    if (flat_idx >= N_total) return;

    uint sub_dim  = codebook_shape[1];
    uint code_idx = indices[flat_idx];
    uint cb_base  = code_idx * sub_dim;
    uint out_base = flat_idx * sub_dim;

    for (uint i = 0; i < sub_dim; ++i) {
        out[out_base + i] = codebook[cb_base + i];
    }
