"""Tests for KVzipKVCache (cache/kvzip_cache.py).

Covers: factory dispatch, config propagation, constant-size budget across B/H,
byte-accounting properties, prefill/decode both valid, construction guards,
no leftover .bits attribute, and the cache-level probe="latest" reduction sanity.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.kvzip_cache import KVzipKVCache


def _kv(B, H, S, D, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


def _make(**kw):
    cfg = KVCacheConfig(method="kvzip", **kw)
    return KVCacheFactory.create(cfg)


# ---------------------------------------------------------------------------
# factory / construction
# ---------------------------------------------------------------------------
def test_factory_dispatch():
    cache = _make(kvzip_budget=16)
    assert isinstance(cache, KVzipKVCache)


def test_no_bits_attribute():
    cache = _make()
    assert not hasattr(cache, "bits")


def test_construction_guard_bad_probe():
    with pytest.raises(ValueError, match="probe must be one of"):
        _make(kvzip_probe="bogus")


def test_construction_guard_sink_ge_budget():
    with pytest.raises(ValueError, match="n_sink .* must be < budget"):
        _make(kvzip_budget=8, kvzip_n_sink=8)


def test_config_defaults_propagate():
    cache = _make()
    assert cache._budget == 512 and cache._n_sink == 4 and cache._probe == "context"


def test_config_overrides_propagate():
    cache = _make(kvzip_budget=64, kvzip_n_sink=2, kvzip_probe="latest")
    assert cache._budget == 64 and cache._n_sink == 2 and cache._probe == "latest"


# ---------------------------------------------------------------------------
# constant-size budget across shapes
# ---------------------------------------------------------------------------
def test_budget_respected_decode():
    cache = _make(kvzip_budget=12, kvzip_n_sink=2, kvzip_probe="context")
    for i in range(40):
        k, v = _kv(1, 3, 1, 32, seed=i)
        K, V = cache.update_and_fetch(k, v)
        assert K.shape[2] <= 12 and V.shape[2] <= 12
        assert K.shape[:2] == (1, 3) and K.shape[3] == 32


def test_budget_respected_prefill_block():
    cache = _make(kvzip_budget=12, kvzip_n_sink=2, kvzip_probe="context")
    k, v = _kv(2, 4, 50, 32, seed=1)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape == (2, 4, 12, 32)


def test_multi_head_independent():
    cache = _make(kvzip_budget=10, kvzip_n_sink=2, kvzip_probe="context")
    for i in range(30):
        k, v = _kv(1, 4, 1, 16, seed=i)
        cache.update_and_fetch(k, v)
    assert cache.tokens_kept <= 10


# ---------------------------------------------------------------------------
# byte-accounting properties
# ---------------------------------------------------------------------------
def test_byte_accounting():
    cache = _make(kvzip_budget=16, kvzip_n_sink=2, kvzip_probe="context")
    for i in range(60):
        k, v = _kv(1, 2, 1, 32, seed=i)
        cache.update_and_fetch(k, v)
    assert cache.tokens_seen == 60 * 2
    assert cache.kvzip_kept_bytes > 0
    assert cache.full_seq_bytes > cache.kvzip_kept_bytes
    assert cache.compression_ratio > 1.0


def test_compression_ratio_one_when_empty():
    cache = _make()
    assert cache.compression_ratio == 1.0
    assert cache.tokens_kept == 0


# ---------------------------------------------------------------------------
# prefill/decode both valid (no bit-for-bit equivalence claim — retention is
# path-dependent on the reconstruction probe)
# ---------------------------------------------------------------------------
def test_prefill_and_decode_both_within_budget():
    k_all, v_all = _kv(1, 2, 40, 24, seed=9)

    pf = _make(kvzip_budget=10, kvzip_n_sink=2, kvzip_probe="context")
    Kp, _ = pf.update_and_fetch(k_all, v_all)

    dc = _make(kvzip_budget=10, kvzip_n_sink=2, kvzip_probe="context")
    for t in range(40):
        Kd, _ = dc.update_and_fetch(k_all[:, :, t : t + 1], v_all[:, :, t : t + 1])

    assert Kp.shape[2] <= 10 and Kd.shape[2] <= 10


# ---------------------------------------------------------------------------
# probe="latest" reduction sanity at cache level (deterministic, budget honoured)
# ---------------------------------------------------------------------------
def test_latest_probe_deterministic_at_cache_level():
    ks = [_kv(1, 2, 1, 16, seed=i) for i in range(35)]

    def run():
        cache = _make(kvzip_budget=10, kvzip_n_sink=2, kvzip_probe="latest")
        for k, v in ks:
            K, _ = cache.update_and_fetch(k, v)
        return K

    a, b = run(), run()
    assert a.shape[2] <= 10
    assert bool(mx.all(a == b).item())
