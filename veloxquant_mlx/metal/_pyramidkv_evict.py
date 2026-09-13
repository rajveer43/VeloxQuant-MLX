"""Metal compaction primitive for PyramidKV's over-budget update."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mlx.core as mx

_SRC = Path(__file__).parent / "src"


@lru_cache(maxsize=8)
def _reduce(nsg: int):
    return mx.fast.metal_kernel(
        name=f"pyramidkv_evict_reduce_{nsg}",
        input_names=["scores", "n_sink_arr"],
        output_names=["evict_idx"],
        header=f"#define NSG_C {nsg}\n",
        source=(_SRC / "pyramidkv_evict_reduce.metal").read_text(),
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _apply():
    return mx.fast.metal_kernel(
        name="pyramidkv_evict_apply",
        input_names=["keys", "values", "scores", "evict_idx"],
        output_names=["keys_out", "values_out", "scores_out"],
        source=(_SRC / "pyramidkv_evict_apply.metal").read_text(),
        ensure_row_contiguous=True,
    )


def pyramidkv_fused_evict(keys_mid, values_mid, scores_mid, n_sink, *, nsg=4, stream=None):
    """Evict one lowest-scoring non-sink row from each ``[BH,N,D]`` group.

    ``keys_mid``/``values_mid`` contain the current cache plus one appended row;
    ``scores_mid`` contains cumulative FP32 scores with the appended score
    already set to zero. The returned arrays contain ``N - 1`` rows and retain
    original row order exactly. This deliberately does not renumber positions
    or apply RoPE: PyramidKV's cache state has no position-remap contract.
    NaNs are ranked as positive infinity; ties choose the earliest eligible
    row. This invalid-input policy is not MLX NaN-ordering equivalence.
    """
    if keys_mid.ndim != 3 or keys_mid.shape != values_mid.shape:
        raise ValueError("pyramidkv_fused_evict: K/V must have matching [BH,N,D] shapes")
    bh, n_total, d = keys_mid.shape
    if bh < 1 or d < 1 or bh * n_total * d >= 2**32:
        raise ValueError("pyramidkv_fused_evict: invalid dimensions or uint32 indexing overflow")
    if n_total < 2 or scores_mid.shape != (bh, n_total):
        raise ValueError("pyramidkv_fused_evict: invalid score shape")
    if keys_mid.dtype != mx.float16 or values_mid.dtype != mx.float16:
        raise ValueError("pyramidkv_fused_evict: K/V must be float16")
    if scores_mid.dtype != mx.float32:
        raise ValueError("pyramidkv_fused_evict: scores must be float32")
    if not isinstance(n_sink, int) or not 0 <= n_sink < n_total:
        raise ValueError("pyramidkv_fused_evict: invalid n_sink")
    if type(nsg) is not int or nsg not in (1, 2, 4, 8):
        raise ValueError("pyramidkv_fused_evict: nsg must be 1, 2, 4, or 8")
    (evict_idx,) = _reduce(nsg)(
        inputs=[scores_mid, mx.array([n_sink], dtype=mx.uint32)],
        grid=(bh * 32, nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(bh,)],
        output_dtypes=[mx.int32],
        stream=stream,
    )
    n_kept = n_total - 1
    tg = 256
    size = bh * n_kept * d
    return _apply()(
        inputs=[keys_mid, values_mid, scores_mid, evict_idx],
        grid=(((size + tg - 1) // tg) * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(bh, n_kept, d), (bh, n_kept, d), (bh, n_kept)],
        output_dtypes=[mx.float16, mx.float16, mx.float32],
        stream=stream,
    )


__all__ = ["pyramidkv_fused_evict"]
