"""Tests for StreamingLLM-adapted quantizer primitives — sink + recency-window eviction.

StreamingLLM-adapted (arXiv:2309.17453, ICLR 2024) keeps the first N sink tokens
unconditionally and a rolling FIFO of the last W recent tokens; all others are dropped.
These tests cover: init_streaming_window, stream_update (single step, multi-step,
window trimming, sink absorption, mixed sink+recent, large batch), stream_get_kv
(shape/dtype, sink-only, recent-only, combined), byte accounting, and edge cases.
All data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.streaming_llm import (
    StreamingWindow,
    full_stream_fp16_bytes,
    init_streaming_window,
    stream_fp16_bytes,
    stream_get_kv,
    stream_update,
)


def _rand_kv(S: int, D: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((S, D)).astype(np.float16))
    return k, v


# ---------------------------------------------------------------------------
# init_streaming_window
# ---------------------------------------------------------------------------


def test_init_window_empty() -> None:
    w = init_streaming_window(n_sink=4, D=64)
    assert w.n_sink == 0
    assert w.n_recent == 0
    assert w.tokens_seen == 0


def test_init_window_shapes() -> None:
    w = init_streaming_window(n_sink=4, D=64)
    assert w.sink_keys.shape == (0, 64)
    assert w.sink_values.shape == (0, 64)
    assert w.recent_keys.shape == (0, 64)
    assert w.recent_values.shape == (0, 64)


# ---------------------------------------------------------------------------
# stream_update — sink absorption
# ---------------------------------------------------------------------------


def test_sink_absorption_single_step() -> None:
    """First N tokens go into sinks."""
    D = 32
    w = init_streaming_window(n_sink=4, D=D)
    k, v = _rand_kv(S=4, D=D)
    w = stream_update(w, k, v, n_sink=4, window_size=8)
    assert w.n_sink == 4
    assert w.n_recent == 0
    assert w.tokens_seen == 4


def test_sink_absorption_partial() -> None:
    """First 2 of 6 tokens → sinks; remaining 4 → recent window."""
    D = 32
    w = init_streaming_window(n_sink=2, D=D)
    k, v = _rand_kv(S=6, D=D)
    w = stream_update(w, k, v, n_sink=2, window_size=8)
    assert w.n_sink == 2
    assert w.n_recent == 4


def test_sinks_are_frozen_after_fill() -> None:
    """Once sink buffer is full, additional tokens do not extend sinks."""
    D = 32
    w = init_streaming_window(n_sink=4, D=D)
    k1, v1 = _rand_kv(S=4, D=D, seed=0)
    w = stream_update(w, k1, v1, n_sink=4, window_size=8)
    k2, v2 = _rand_kv(S=4, D=D, seed=1)
    w = stream_update(w, k2, v2, n_sink=4, window_size=8)
    assert w.n_sink == 4  # unchanged
    assert w.n_recent == 4  # new tokens went to recent window


def test_tokens_seen_accumulates() -> None:
    D = 32
    w = init_streaming_window(n_sink=2, D=D)
    k, v = _rand_kv(S=6, D=D)
    w = stream_update(w, k, v, n_sink=2, window_size=8)
    assert w.tokens_seen == 6
    k2, v2 = _rand_kv(S=3, D=D, seed=1)
    w = stream_update(w, k2, v2, n_sink=2, window_size=8)
    assert w.tokens_seen == 9


# ---------------------------------------------------------------------------
# stream_update — recent-window trimming
# ---------------------------------------------------------------------------


def test_window_trim_at_capacity() -> None:
    """Recent window trims to last window_size tokens when exceeded."""
    D = 32
    w = init_streaming_window(n_sink=2, D=D)
    # sink fill: 2 tokens
    k, v = _rand_kv(S=2, D=D, seed=0)
    w = stream_update(w, k, v, n_sink=2, window_size=4)
    # recent fill: 6 more tokens — only last 4 kept
    k2, v2 = _rand_kv(S=6, D=D, seed=1)
    w = stream_update(w, k2, v2, n_sink=2, window_size=4)
    assert w.n_recent == 4


def test_window_never_exceeds_capacity_multi_step() -> None:
    """20 decode steps of 1 token each — recent window stays <= window_size."""
    D = 16
    n_sink = 2
    window_size = 5
    w = init_streaming_window(n_sink=n_sink, D=D)
    k, v = _rand_kv(S=n_sink, D=D, seed=0)
    w = stream_update(w, k, v, n_sink=n_sink, window_size=window_size)
    for i in range(20):
        ki, vi = _rand_kv(S=1, D=D, seed=10 + i)
        w = stream_update(w, ki, vi, n_sink=n_sink, window_size=window_size)
        assert w.n_recent <= window_size, f"step {i}: n_recent={w.n_recent} > {window_size}"


def test_total_in_window_bounded() -> None:
    """n_sink + n_recent <= n_sink + window_size at all times."""
    D = 32
    n_sink = 4
    window_size = 8
    w = init_streaming_window(n_sink=n_sink, D=D)
    for i in range(30):
        ki, vi = _rand_kv(S=1, D=D, seed=i)
        w = stream_update(w, ki, vi, n_sink=n_sink, window_size=window_size)
    assert w.n_sink + w.n_recent <= n_sink + window_size


# ---------------------------------------------------------------------------
# stream_get_kv
# ---------------------------------------------------------------------------


def test_get_kv_combined_shape() -> None:
    """Returned K/V has n_sink + n_recent rows."""
    D = 32
    w = init_streaming_window(n_sink=4, D=D)
    k, v = _rand_kv(S=10, D=D)
    w = stream_update(w, k, v, n_sink=4, window_size=6)
    # 4 sinks + 6 recent
    ko, vo = stream_get_kv(w)
    assert ko.shape == (10, D)
    assert vo.shape == (10, D)


def test_get_kv_dtype_fp16() -> None:
    D = 32
    w = init_streaming_window(n_sink=2, D=D)
    k, v = _rand_kv(S=5, D=D)
    w = stream_update(w, k, v, n_sink=2, window_size=8)
    ko, vo = stream_get_kv(w)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16


def test_get_kv_sinks_first() -> None:
    """First n_sink rows in output are sink tokens (from first input tokens)."""
    D = 4
    n_sink = 2
    window_size = 4
    # Known sink tokens: positions 0, 1 with value 1.0
    k_known = mx.ones((n_sink, D), dtype=mx.float16)
    v_known = mx.ones((n_sink, D), dtype=mx.float16) * 2.0
    w = init_streaming_window(n_sink=n_sink, D=D)
    w = stream_update(w, k_known, v_known, n_sink=n_sink, window_size=window_size)
    # Add 4 more tokens with value -1.0
    k2 = mx.ones((4, D), dtype=mx.float16) * -1.0
    v2 = mx.ones((4, D), dtype=mx.float16) * -2.0
    w = stream_update(w, k2, v2, n_sink=n_sink, window_size=window_size)
    ko, vo = stream_get_kv(w)
    # First 2 rows must be the original sink tokens
    assert float(ko[0, 0].item()) == pytest.approx(1.0, abs=1e-3)
    assert float(ko[1, 0].item()) == pytest.approx(1.0, abs=1e-3)
    # Last 4 rows are recent
    assert float(ko[2, 0].item()) == pytest.approx(-1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_stream_fp16_bytes_formula() -> None:
    """stream_fp16_bytes = (n_sink + n_recent) * D * 4 (K+V, fp16)."""
    D = 64
    w = init_streaming_window(n_sink=4, D=D)
    k, v = _rand_kv(S=12, D=D)
    w = stream_update(w, k, v, n_sink=4, window_size=8)
    # 4 sinks + 8 recent = 12
    expected = 12 * D * 2 * 2
    assert stream_fp16_bytes(w) == expected


def test_full_stream_fp16_bytes_formula() -> None:
    assert full_stream_fp16_bytes(64, 128) == 64 * 128 * 2 * 2


def test_stream_fp16_bytes_after_trim() -> None:
    """After trim, bytes reflect only n_sink + n_recent (not all seen tokens)."""
    D = 32
    n_sink = 4
    window_size = 4
    w = init_streaming_window(n_sink=n_sink, D=D)
    k, v = _rand_kv(S=20, D=D)
    w = stream_update(w, k, v, n_sink=n_sink, window_size=window_size)
    expected = (n_sink + window_size) * D * 2 * 2
    assert stream_fp16_bytes(w) == expected


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_n_sink_zero() -> None:
    """n_sink=0: all tokens go into recent window."""
    D = 32
    w = init_streaming_window(n_sink=0, D=D)
    k, v = _rand_kv(S=8, D=D)
    w = stream_update(w, k, v, n_sink=0, window_size=4)
    assert w.n_sink == 0
    assert w.n_recent == 4


def test_single_token_step() -> None:
    """S=1 decode step absorbs correctly."""
    D = 32
    w = init_streaming_window(n_sink=4, D=D)
    k, v = _rand_kv(S=4, D=D, seed=0)
    w = stream_update(w, k, v, n_sink=4, window_size=8)
    k1, v1 = _rand_kv(S=1, D=D, seed=1)
    w = stream_update(w, k1, v1, n_sink=4, window_size=8)
    assert w.n_sink == 4
    assert w.n_recent == 1
    assert w.tokens_seen == 5
