"""Tests for SlidingWindowKVCache."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def base_cache():
    from veloxquant_mlx.cache.base import KVCacheBuilder

    return KVCacheBuilder().with_method("qjl").with_head_dim(64).with_jl_dim(64).build()


def test_sliding_window_evicts(base_cache) -> None:
    import mlx.core as mx
    from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache

    sw = SlidingWindowKVCache(base_cache, window_size=5)
    rng = np.random.default_rng(0)
    for i in range(10):
        k = mx.array(rng.standard_normal(64).astype(np.float16))
        v = mx.array(rng.standard_normal(64).astype(np.float16))
        sw.append(k, v)

    assert len(sw) == 5  # window of 5


def test_sliding_window_attend(base_cache) -> None:
    import mlx.core as mx
    from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache

    sw = SlidingWindowKVCache(base_cache, window_size=10)
    rng = np.random.default_rng(1)
    for i in range(20):
        k = mx.array(rng.standard_normal(64).astype(np.float16))
        v = mx.array(rng.standard_normal(64).astype(np.float16))
        sw.append(k, v)

    q = mx.array(rng.standard_normal(64).astype(np.float16))
    out = sw.attend(q)
    mx.eval(out)
    assert out.shape == (64,)


def test_sliding_window_invalid_size() -> None:
    from veloxquant_mlx.cache.base import KVCacheBuilder
    from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache

    cache = KVCacheBuilder().with_method("qjl").with_head_dim(64).with_jl_dim(64).build()
    with pytest.raises(ValueError):
        SlidingWindowKVCache(cache, window_size=0)


# ---------------------------------------------------------------------------
# Regression for #81: KVCacheConfig.sliding_window + a non-standalone method
# (mlx_lm-protocol, i.e. update_and_fetch-based) must raise a clear config
# error at construction time, not produce a cache broken at first use.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,extra",
    [
        ("h2o", {}),
        ("tova", {}),
        ("kivi", {}),
        ("snapkv", {}),
        ("streaming_llm", {}),
    ],
)
def test_sliding_window_rejected_for_mlx_lm_protocol_methods(method, extra) -> None:
    """Regression for #81's exact repro: sliding_window combined with any
    mlx_lm-protocol (update_and_fetch-based) method must raise
    QuantizerConfigError at KVCacheFactory.create(), not silently produce a
    cache with neither update_and_fetch (not on the wrapper) nor
    append_key/append_value (not on the inner cache) working."""
    from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
    from veloxquant_mlx.core.exceptions import QuantizerConfigError

    cfg = KVCacheConfig(method=method, head_dim=16, sliding_window=8, **extra)
    with pytest.raises(QuantizerConfigError, match="sliding_window"):
        KVCacheFactory.create(cfg)


def test_sliding_window_rejected_via_builder() -> None:
    """Same guard must apply through the KVCacheBuilder.build() entry point,
    since it delegates to KVCacheFactory.create()."""
    from veloxquant_mlx.cache.base import KVCacheBuilder
    from veloxquant_mlx.core.exceptions import QuantizerConfigError

    with pytest.raises(QuantizerConfigError, match="sliding_window"):
        (KVCacheBuilder().with_method("h2o").with_head_dim(16).with_sliding_window(8).build())


@pytest.mark.parametrize("method,extra", [("qjl", {"jl_dim": 8}), ("polar", {})])
def test_sliding_window_accepted_for_standalone_methods(method, extra) -> None:
    """sliding_window must still work end-to-end for STANDALONE_METHODS —
    the only family SlidingWindowKVCache's append_key/append_value/attend
    wrapping is actually compatible with."""
    import mlx.core as mx

    from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
    from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache

    d = 16
    cfg = KVCacheConfig(method=method, head_dim=d, sliding_window=4, bit_width_inlier=2, **extra)
    cache = KVCacheFactory.create(cfg)
    assert isinstance(cache, SlidingWindowKVCache)

    rng = np.random.default_rng(0)
    for _ in range(6):
        k = mx.array(rng.standard_normal(d).astype(np.float16))
        v = mx.array(rng.standard_normal(d).astype(np.float16))
        cache.append(k, v)
    assert len(cache) == 4

    q = mx.array(rng.standard_normal(d).astype(np.float16))
    out = cache.attend(q)
    mx.eval(out)
    assert out.shape == (d,)
