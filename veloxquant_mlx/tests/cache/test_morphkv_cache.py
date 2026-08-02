"""Tests for MorphKVKVCache (cache/morphkv_cache.py).

Covers: factory dispatch, config propagation, constant-size budget across B/H,
byte-accounting properties, prefill/decode both valid, construction guards,
no leftover .bits attribute, and the cache-level window=1 reduction sanity.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.morphkv_cache import MorphKVKVCache


def _kv(B, H, S, D, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


def _make(**kw):
    cfg = KVCacheConfig(method="morphkv", **kw)
    return KVCacheFactory.create(cfg)


# ---------------------------------------------------------------------------
# factory / construction
# ---------------------------------------------------------------------------
def test_factory_dispatch():
    cache = _make(morphkv_budget=16)
    assert isinstance(cache, MorphKVKVCache)


def test_no_bits_attribute():
    cache = _make()
    assert not hasattr(cache, "bits")


def test_construction_guard_bad_window():
    with pytest.raises(ValueError, match="window must be >= 1"):
        _make(morphkv_window=0)


def test_construction_guard_no_evictable_room():
    with pytest.raises(ValueError, match="no evictable positions"):
        _make(morphkv_budget=8, morphkv_n_sink=4, morphkv_window=4)


def test_config_defaults_propagate():
    cache = _make()
    assert cache._budget == 512 and cache._n_sink == 4 and cache._window == 8


def test_config_overrides_propagate():
    cache = _make(morphkv_budget=64, morphkv_n_sink=2, morphkv_window=16)
    assert cache._budget == 64 and cache._n_sink == 2 and cache._window == 16


# ---------------------------------------------------------------------------
# constant-size budget across shapes
# ---------------------------------------------------------------------------
def test_budget_respected_decode():
    cache = _make(morphkv_budget=12, morphkv_n_sink=2, morphkv_window=3)
    for i in range(40):
        k, v = _kv(1, 3, 1, 32, seed=i)
        K, V = cache.update_and_fetch(k, v)
        assert K.shape[2] <= 12 and V.shape[2] <= 12
        assert K.shape[:2] == (1, 3) and K.shape[3] == 32


def test_budget_respected_prefill_block():
    cache = _make(morphkv_budget=12, morphkv_n_sink=2, morphkv_window=3)
    k, v = _kv(2, 4, 50, 32, seed=1)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape == (2, 4, 12, 32)


def test_multi_head_independent():
    cache = _make(morphkv_budget=10, morphkv_n_sink=2, morphkv_window=3)
    for i in range(30):
        k, v = _kv(1, 4, 1, 16, seed=i)
        cache.update_and_fetch(k, v)
    assert cache.tokens_kept <= 10


# ---------------------------------------------------------------------------
# byte-accounting properties
# ---------------------------------------------------------------------------
def test_byte_accounting():
    cache = _make(morphkv_budget=16, morphkv_n_sink=2, morphkv_window=4)
    for i in range(60):
        k, v = _kv(1, 2, 1, 32, seed=i)
        cache.update_and_fetch(k, v)
    assert cache.tokens_seen == 60 * 2
    assert cache.morphkv_kept_bytes > 0
    assert cache.full_seq_bytes > cache.morphkv_kept_bytes
    assert cache.compression_ratio > 1.0


def test_compression_ratio_one_when_empty():
    cache = _make()
    assert cache.compression_ratio == 1.0
    assert cache.tokens_kept == 0


# ---------------------------------------------------------------------------
# prefill/decode both valid (no bit-for-bit equivalence claim — retention is
# path-dependent on the recent window)
# ---------------------------------------------------------------------------
def test_prefill_and_decode_both_within_budget():
    k_all, v_all = _kv(1, 2, 40, 24, seed=9)

    pf = _make(morphkv_budget=10, morphkv_n_sink=2, morphkv_window=4)
    Kp, _ = pf.update_and_fetch(k_all, v_all)

    dc = _make(morphkv_budget=10, morphkv_n_sink=2, morphkv_window=4)
    for t in range(40):
        Kd, _ = dc.update_and_fetch(k_all[:, :, t : t + 1], v_all[:, :, t : t + 1])

    assert Kp.shape[2] <= 10 and Kd.shape[2] <= 10


# ---------------------------------------------------------------------------
# window=1 reduction sanity at cache level (deterministic, budget honoured)
# ---------------------------------------------------------------------------
def test_window_one_deterministic_at_cache_level():
    ks = [_kv(1, 2, 1, 16, seed=i) for i in range(35)]

    def run():
        cache = _make(morphkv_budget=10, morphkv_n_sink=2, morphkv_window=1)
        for k, v in ks:
            K, _ = cache.update_and_fetch(k, v)
        return K

    a, b = run(), run()
    assert a.shape[2] <= 10
    assert bool(mx.all(a == b).item())
