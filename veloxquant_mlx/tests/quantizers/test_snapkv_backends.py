"""Exact selection, independent K/V lineage and chunk parity."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.cache.snapkv_cache import SnapKVKVCache
from veloxquant_mlx.quantizers.snapkv import _snap_select_batched


@pytest.mark.parametrize("backend", ["mlx", "metal"])
def test_random_selection(backend):
    rng = np.random.default_rng(391)
    for _ in range(1000):
        n = int(rng.integers(1, 300))
        budget = int(rng.integers(-1, n + 5))
        sink = int(rng.integers(-1, n + 5))
        scores = mx.array(rng.integers(-4, 5, (3, n)).astype(np.float32))
        expected = _snap_select_batched(scores, budget, sink, backend="reference")
        actual = _snap_select_batched(scores, budget, sink, backend=backend)
        assert actual.tolist() == expected.tolist()


@pytest.mark.parametrize("backend", ["mlx", "metal"])
def test_nonfinite(backend):
    scores = mx.array([[float("nan"), float("inf"), -float("inf"), 0.0, -0.0, float("nan")]])
    for k in range(1, 7):
        assert (
            _snap_select_batched(scores, k, 0, backend=backend).tolist()
            == _snap_select_batched(scores, k, 0, backend="reference").tolist()
        )


@pytest.mark.parametrize("backend", ["mlx", "metal"])
def test_chunked_cache(backend):
    ref = SnapKVKVCache(KVCacheConfig(snap_budget=9, snap_n_sink=2, snap_backend="reference"))
    cache = SnapKVKVCache(KVCacheConfig(snap_budget=9, snap_n_sink=2, snap_backend=backend))
    rng = np.random.default_rng(17)
    offset = 0
    for n in [13, 7, 21, 1, 1, 3, 1]:
        keys = mx.array(rng.normal(size=(2, 3, n, 7)).astype(np.float32))
        values = mx.array(rng.integers(-100, 100, size=(2, 3, n, 7)).astype(np.float32))
        expected = ref.update_and_fetch(keys, values)
        actual = cache.update_and_fetch(keys, values)
        for a, e in zip(actual, expected, strict=True):
            assert a.tolist() == e.tolist()
        offset += n
        assert cache.offset == ref.offset == offset
        assert cache.tokens_kept == ref.tokens_kept
        assert cache.evicted_key_bytes == ref.evicted_key_bytes


@pytest.mark.parametrize("backend", ["mlx", "metal"])
def test_boundaries_and_strides(backend):
    for n in (0, 1, 2, 255, 256, 257, 511, 512, 513, 8193):
        scores = mx.zeros((2, n * 2), mx.float32)[:, ::2]
        for budget, sink in ((0, 0), (3, 9), (n + 1, 0), (n // 2, 1)):
            assert (
                _snap_select_batched(scores, budget, sink, backend=backend).tolist()
                == _snap_select_batched(scores, budget, sink, backend="reference").tolist()
            )


def test_long_decode_is_append_only():
    cache = SnapKVKVCache(KVCacheConfig(snap_budget=4, snap_n_sink=1))
    initial = mx.arange(16, dtype=mx.float32).reshape(1, 1, 8, 2)
    mx.eval(cache.update_and_fetch(initial, -initial))
    for i in range(2000):
        keys = mx.full((1, 1, 1, 2), i % 512, mx.float16)
        k, v = cache.update_and_fetch(keys, -keys)
        mx.eval(k, v)
    assert cache.offset == 2008
    assert k.shape[2] == 2004
    assert mx.array_equal(v, -k).item()


def test_bfloat16_storage_preserves_model_dtype():
    cache = SnapKVKVCache(KVCacheConfig(snap_budget=4, snap_n_sink=1, snap_dtype="auto"))
    keys = mx.arange(1 * 2 * 8 * 7, dtype=mx.bfloat16).reshape(1, 2, 8, 7)
    values = -keys
    kept_k, kept_v = cache.update_and_fetch(keys, values)
    mx.eval(kept_k, kept_v)
    assert kept_k.dtype == mx.bfloat16
    assert kept_v.dtype == mx.bfloat16


def test_float16_storage_remains_forceable():
    cache = SnapKVKVCache(KVCacheConfig(snap_budget=4, snap_n_sink=1, snap_dtype="float16"))
    keys = mx.arange(1 * 2 * 8 * 7, dtype=mx.bfloat16).reshape(1, 2, 8, 7)
    kept_k, kept_v = cache.update_and_fetch(keys, -keys)
    mx.eval(kept_k, kept_v)
    assert kept_k.dtype == mx.float16
    assert kept_v.dtype == mx.float16
