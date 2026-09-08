"""TOVA selection and copy-only compaction, two dependent GPU dispatches.

Scoring and append remain MLX operations. Inputs are FP16 K/V and FP32
current-step weights; no position remapping or score history is involved.
"""

from functools import lru_cache
from pathlib import Path

import mlx.core as mx


@lru_cache(maxsize=4)
def _reduce_kernel(nsg):
    return mx.fast.metal_kernel(
        name=f"tova_evict_reduce_{nsg}",
        input_names=["weights", "sink"],
        output_names=["evicted"],
        header=f"#define NSG {nsg}\n",
        source=(Path(__file__).parent / "src/tova_evict_reduce.metal").read_text(),
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _apply_kernel():
    return mx.fast.metal_kernel(
        name="tova_evict_apply",
        input_names=["keys", "values", "evicted"],
        output_names=["keys_out", "values_out"],
        source=(Path(__file__).parent / "src/tova_evict_apply.metal").read_text(),
        ensure_row_contiguous=True,
    )


def tova_fused_evict(keys_mid, values_mid, weights, n_sink, *, nsg=4, stream=None):
    """Remove the earliest minimum non-sink weight in each ``[BH,N,D]`` group.

    K/V must be matching FP16 arrays; weights are ``[BH,N]`` FP32. Returns
    two FP16 ``[BH,N-1,D]`` arrays, copying survivors exactly. Odd D and
    noncontiguous inputs are supported. At least one non-sink is required.
    NaN weights are ignored; if all eligible weights are NaN, the earliest
    eligible row is removed. This safe invalid-input policy is not a claim
    of equivalence to MLX argmin's unspecified NaN ordering.
    """
    if keys_mid.ndim != 3 or keys_mid.shape != values_mid.shape:
        raise ValueError("tova_fused_evict: K/V must have matching [BH,N,D] shapes")
    bh, n, d = keys_mid.shape
    if weights.shape != (bh, n):
        raise ValueError("tova_fused_evict: weights must have shape [BH,N]")
    if keys_mid.dtype != mx.float16 or values_mid.dtype != mx.float16:
        raise ValueError("tova_fused_evict: K/V must be float16")
    if weights.dtype != mx.float32:
        raise ValueError("tova_fused_evict: weights must be float32")
    if bh < 1 or n < 1 or d < 1 or not isinstance(n_sink, int) or not 0 <= n_sink < n:
        raise ValueError("tova_fused_evict: require positive dimensions and 0 <= n_sink < N")
    if type(nsg) is not int or nsg not in (1, 2, 4, 8):
        raise ValueError("tova_fused_evict: nsg must be 1, 2, 4, or 8")
    if bh * n * d >= 2**32:
        raise ValueError("tova_fused_evict: input exceeds uint32 indexing range")
    if n == 1:
        return keys_mid[:, :0], values_mid[:, :0]
    (evicted,) = _reduce_kernel(nsg)(
        inputs=[weights, mx.array([n_sink], dtype=mx.uint32)],
        grid=(bh * 32, nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(bh,)],
        output_dtypes=[mx.uint32],
        stream=stream,
    )
    size = bh * (n - 1) * d
    return tuple(
        _apply_kernel()(
            inputs=[keys_mid, values_mid, evicted],
            grid=(((size + 255) // 256) * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(bh, n - 1, d)] * 2,
            output_dtypes=[mx.float16] * 2,
            stream=stream,
        )
    )
