"""Tests for StreamingLLMKVCache — sink + recency-window structural eviction.

StreamingLLM-adapted (arXiv:2309.17453, ICLR 2024) keeps n_sink initial tokens and
the last window_size tokens; all others are dropped. Tests cover: factory dispatch,
no .bits attribute, output shape bounded, output dtype fp16, sink-only phase,
decode accumulation within window, window trimming (overflow evicts oldest), byte
accounting (streaming_ratio, tokens_in_window), n_sink=0 edge case, determinism,
and for_model config propagation. All data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.streaming_llm_cache import StreamingLLMKVCache


def _make(**cfg):
    base = dict(
        method="streaming_llm",
        head_dim=64,
        stream_n_sink=4,
        stream_window_size=8,
    )
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_kv(S: int = 16, H: int = 2, D: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    V = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    return K, V


# ---------------------------------------------------------------------------
# Factory and interface
# ---------------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), StreamingLLMKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "streaming_ratio")
    assert hasattr(c, "tokens_in_window")


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape_sink_only() -> None:
    """Exactly n_sink tokens → output seq dim == n_sink."""
    c = _make(stream_n_sink=4, stream_window_size=8)
    k, v = _rand_kv(S=4, H=2, D=64)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 4
    assert vo.shape[2] == 4


def test_output_shape_bounded_by_sink_plus_window() -> None:
    """After many tokens, output seq dim <= n_sink + window_size."""
    c = _make(stream_n_sink=4, stream_window_size=8)
    # prefill 32 tokens
    k, v = _rand_kv(S=32, H=2, D=64)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] <= 4 + 8


def test_output_dtype_fp16() -> None:
    c = _make()
    k, v = _rand_kv(S=8)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16


# ---------------------------------------------------------------------------
# Window growth and trimming
# ---------------------------------------------------------------------------


def test_decode_grow_within_window() -> None:
    """Single-token decode steps grow output until window_size is reached."""
    c = _make(stream_n_sink=4, stream_window_size=6)
    # First fill sinks with 4 tokens
    k, v = _rand_kv(S=4, D=64)
    c.update_and_fetch(k, v)
    # Add 3 decode tokens — recent window grows 0→3
    for i in range(3):
        k1, v1 = _rand_kv(S=1, D=64, seed=10 + i)
        ko, vo = c.update_and_fetch(k1, v1)
    # seq dim = 4 sinks + 3 recent = 7
    assert ko.shape[2] == 7


def test_window_trims_oldest_recent() -> None:
    """Once recent window > window_size, oldest recent tokens are evicted."""
    c = _make(stream_n_sink=2, stream_window_size=4)
    # Fill sinks
    k, v = _rand_kv(S=2, D=64, seed=0)
    c.update_and_fetch(k, v)
    # Add 8 decode tokens — window fills and trims
    for i in range(8):
        k1, v1 = _rand_kv(S=1, D=64, seed=10 + i)
        ko, vo = c.update_and_fetch(k1, v1)
    # seq dim must be exactly n_sink + window_size = 2 + 4 = 6
    assert ko.shape[2] == 6


def test_tokens_in_window_bounded() -> None:
    """tokens_in_window never exceeds n_sink + window_size."""
    n_sink = 4
    window_size = 8
    c = _make(stream_n_sink=n_sink, stream_window_size=window_size)
    for i in range(30):
        k, v = _rand_kv(S=1, D=64, seed=i)
        c.update_and_fetch(k, v)
    assert c.tokens_in_window <= n_sink + window_size


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_streaming_ratio_equals_1_before_window_fills() -> None:
    """When all tokens fit in n_sink + window_size, ratio == 1."""
    c = _make(stream_n_sink=4, stream_window_size=100)
    k, v = _rand_kv(S=8, D=64)
    c.update_and_fetch(k, v)
    assert c.streaming_ratio == pytest.approx(1.0, rel=1e-3)


def test_streaming_ratio_gt_1_after_overflow() -> None:
    """After many tokens overflow the window, ratio > 1."""
    c = _make(stream_n_sink=4, stream_window_size=8)
    # 100 tokens — far more than 4 + 8 = 12
    k, v = _rand_kv(S=100, D=64)
    c.update_and_fetch(k, v)
    assert c.streaming_ratio > 1.0


def test_tokens_seen_accumulates() -> None:
    """tokens_seen grows by B * H * S per call."""
    c = _make(stream_n_sink=4, stream_window_size=8)
    k, v = _rand_kv(S=10, H=2, D=64)
    c.update_and_fetch(k, v)
    # B=1, H=2, S=10 → tokens_seen = 20
    assert c.tokens_seen == 20


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_n_sink_zero() -> None:
    """n_sink=0: all tokens go into recent window only."""
    c = _make(stream_n_sink=0, stream_window_size=8)
    k, v = _rand_kv(S=20, D=64)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 8
    assert c.tokens_in_window == 8


def test_large_prefill_trimmed_correctly() -> None:
    """Large prefill (S >> n_sink + window_size) trims to exact bound."""
    c = _make(stream_n_sink=4, stream_window_size=8)
    k, v = _rand_kv(S=1000, D=64)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 12  # 4 + 8


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic() -> None:
    k, v = _rand_kv(S=20)
    c1, c2 = _make(), _make()
    ko1, _ = c1.update_and_fetch(k, v)
    ko2, _ = c2.update_and_fetch(k, v)
    mse = float(mx.mean((ko1.astype(mx.float32) - ko2.astype(mx.float32)) ** 2).item())
    assert mse == pytest.approx(0.0, abs=0.0)


# ---------------------------------------------------------------------------
# for_model construction
# ---------------------------------------------------------------------------


def test_build_via_for_model_propagates_config() -> None:
    from veloxquant_mlx.cache.base import KVCacheBuilder

    class _Attn:
        head_dim = 64

    class _Layer:
        self_attn = _Attn()

    class _Model:
        layers = [_Layer(), _Layer()]

    cfg = KVCacheConfig(
        method="streaming_llm",
        head_dim=64,
        stream_n_sink=6,
        stream_window_size=128,
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, StreamingLLMKVCache) for c in caches)
    assert caches[0]._n_sink == 6
    assert caches[0]._window_size == 128
