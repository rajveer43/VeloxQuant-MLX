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
