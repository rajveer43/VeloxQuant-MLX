"""Experimental ordered compaction; MLX computes threshold and prefix ranks."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mlx.core as mx


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="snapkv_compact_indices",
        input_names=["selected", "ranks", "params"],
        output_names=["indices"],
        source=(Path(__file__).parent / "src/snapkv_compact_indices.metal").read_text(),
        ensure_row_contiguous=True,
    )


def compact_indices(selected, n_sink, k):
    """Internal: selected must contain exactly k true entries per group."""
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise ValueError("SnapKV Metal requires the default GPU device")
    groups, n = selected.shape
    count = n_sink + k
    if min(groups, n, k) <= 0 or n_sink < 0 or k > n:
        raise ValueError("Invalid compaction dimensions")
    if max(groups * (n + n_sink), groups * count) >= 2**31:
        raise ValueError("SnapKV index range exceeds int32")
    ranks = mx.cumsum(selected.astype(mx.int32), axis=-1)
    work = groups * (n + n_sink)
    return _kernel()(
        inputs=[selected, ranks, mx.array([n_sink, count], mx.int32)],
        grid=(((work + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(groups, count)],
        output_dtypes=[mx.int32],
    )[0]


@lru_cache(maxsize=2)
def _operation(name):
    selection = name == "snapkv_select_threshold"
    return mx.fast.metal_kernel(
        name=name,
        input_names=["scores", "thresholds", "params"]
        if selection
        else ["keys", "values", "indices"],
        output_names=["indices"] if selection else ["keys_out", "values_out"],
        source=(Path(__file__).parent / "src" / f"{name}.metal").read_text(),
        ensure_row_contiguous=True,
    )


def _check_gpu():
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise ValueError("SnapKV Metal requires the default GPU device")


def select_from_threshold(scores, thresholds, n_sink, k):
    """Internal: sanitized FP32 scores and exact kth threshold, per group."""
    _check_gpu()
    g, n = scores.shape
    count = n_sink + k
    if min(g, n, k) <= 0 or n_sink < 0 or k > n:
        raise ValueError("Invalid selection dimensions")
    if scores.dtype != mx.float32 or thresholds.shape != (g, 1) or thresholds.dtype != mx.float32:
        raise ValueError("Expected FP32 scores [G,N] and thresholds [G,1]")
    if max(g * n, g * count, n + n_sink) >= 2**31:
        raise ValueError("SnapKV index range exceeds int32")
    return _operation("snapkv_select_threshold")(
        inputs=[scores, thresholds, mx.array([n_sink, k], mx.int32)],
        grid=(g * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(g, count)],
        output_dtypes=[mx.int32],
    )[0]


def gather_kv(keys, values, indices, output_dtype):
    """Internal gather for validated maps produced by SnapKV selection."""
    _check_gpu()
    g, n, d = keys.shape
    if values.shape != keys.shape or indices.ndim != 2 or indices.shape[0] != g:
        raise ValueError("Mismatched K/V/index shapes")
    supported = (mx.float16, mx.bfloat16, mx.float32)
    if (
        keys.dtype not in supported
        or values.dtype not in supported
        or output_dtype not in supported
        or indices.dtype != mx.int32
    ):
        raise ValueError("Unsupported gather dtype")
    c = indices.shape[1]
    if min(g, n, d, c) <= 0 or max(g * n * d, g * c * d) >= 2**31:
        raise ValueError("Invalid gather dimensions or int32 overflow")
    work = g * c * d
    return _operation("snapkv_gather_kv")(
        inputs=[keys, values, indices],
        grid=(((work + 255) // 256) * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(g, c, d), (g, c, d)],
        output_dtypes=[output_dtype, output_dtype],
    )
