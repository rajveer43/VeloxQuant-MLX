"""CommVQ decode Metal kernel — fused centroid gather + RoPE apply.

Replaces the Python loop in CommVQQuantizer._decode_batch + _apply_rope_mlx
with a single GPU dispatch:

  1. For each output element (b_idx, dim_i):
       a. Identify which sub-codebook owns dim_i → cb_i, comp_i
       b. Look up centroid: cb[cb_i, indices[b_idx, cb_i], comp_i]
       c. Accumulate across all sub-codebooks for this output position
       d. Apply RoPE in-place (paired complex multiply)

Grid: (N * D, 1, 1) rounded up to a threadgroup multiple — one thread per
      output scalar; the kernel bounds-checks and no-ops padding threads.
Threadgroup: (min(D, 256), 1, 1)

Public API:
  - :func:`comm_vq_decode_metal`
"""

from __future__ import annotations

import mlx.core as mx

from veloxquant_mlx.metal._kernel_utils import KernelCache, read_kernel_source


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return read_kernel_source(__file__, filename)


_cache = KernelCache()


# ===========================================================================
# Metal source — CommVQ decode + RoPE fused kernel
# ===========================================================================
# Template params (injected at compile time):
#   N_CB    — number of sub-codebooks
#   SUB_DIM — sub_dim = D / N_CB
#   CB_SIZE — codebook size (2^b); unused in body but available for dispatch
#
# Inputs (shape info available as <name>_shape[k]):
#   indices   [N, N_CB]            uint8
#   codebook  [N_CB, CB_SIZE, SUB_DIM] fp16
#   positions [N]                  int32 — token positions for RoPE
#   inv_freq  [D/2]                fp32  — RoPE inverse frequency table
#
# Output:
#   out [N, D] fp16

_COMM_VQ_DECODE_SRC = _read_kernel_source("comm_vq_decode.metal")


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------


def _comm_vq_kernel(n_cb: int, sub_dim: int, cb_size: int, D: int):
    key = ("comm_vq_decode", n_cb, sub_dim, cb_size, D)
    return _cache.get_or_create(
        key,
        lambda: mx.fast.metal_kernel(
            name=f"comm_vq_decode_ncb{n_cb}_sd{sub_dim}_k{cb_size}_d{D}",
            input_names=["indices", "codebook", "positions", "inv_freq"],
            output_names=["out"],
            source=_COMM_VQ_DECODE_SRC,
            ensure_row_contiguous=True,
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def comm_vq_decode_metal(
    indices: mx.array,
    codebook: mx.array,
    positions: mx.array,
    inv_freq: mx.array,
    n_cb: int,
    sub_dim: int,
    cb_size: int,
) -> mx.array:
    """Fused CommVQ centroid gather + RoPE decode Metal kernel.

    Args:
        indices:   ``[N, n_cb]`` uint8 sub-codebook indices.
        codebook:  ``[n_cb, cb_size, sub_dim]`` fp16 centroid table.
        positions: ``[N]`` int32 token positions for RoPE.
        inv_freq:  ``[D//2]`` fp32 RoPE inverse frequency table.
        n_cb:      Number of sub-codebooks.
        sub_dim:   Sub-dimension per codebook (D // n_cb).
        cb_size:   Codebook size (2^b).

    Returns:
        ``[N, D]`` fp16 decoded keys with RoPE applied.
    """
    N = indices.shape[0]
    D = n_cb * sub_dim
    total_threads = N * D  # one thread per output scalar; kernel handles pairs
    tg = min(D, 256)
    grid = ((total_threads + tg - 1) // tg) * tg

    outputs = _comm_vq_kernel(n_cb, sub_dim, cb_size, D)(
        inputs=[
            indices.astype(mx.uint8),
            codebook.astype(mx.float16),
            positions.astype(mx.int32),
            inv_freq.astype(mx.float32),
        ],
        template=[("N_CB", n_cb), ("SUB_DIM", sub_dim), ("CB_SIZE", cb_size)],
        grid=(grid, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(N, D)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]


__all__ = ["comm_vq_decode_metal"]
