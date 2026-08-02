"""Tests for KeyformerKVCache (cache/keyformer_cache.py).

Covers: factory dispatch, config propagation, budget invariant across B/H,
byte-accounting properties, prefill/decode both valid, tau=0 determinism,
construction guards, and no leftover .bits attribute.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.keyformer_cache import KeyformerKVCache


def _kv(B, H, S, D, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


def _make(**kw):
    cfg = KVCacheConfig(method="keyformer", **kw)
    return KVCacheFactory.create(cfg)


# ---------------------------------------------------------------------------
# factory / construction
# ---------------------------------------------------------------------------
def test_factory_dispatch():
    cache = _make(keyformer_budget=16)
    assert isinstance(cache, KeyformerKVCache)


def test_no_bits_attribute():
    cache = _make()
    assert not hasattr(cache, "bits")


def test_construction_guard_negative_tau():
    with pytest.raises(ValueError, match="tau must be >= 0"):
        _make(keyformer_tau=-1.0)


def test_construction_guard_no_evictable_room():
    with pytest.raises(ValueError, match="no evictable positions"):
        _make(keyformer_budget=8, keyformer_n_sink=6, keyformer_recent=2)


def test_config_defaults_propagate():
    cache = _make()
    assert cache._budget == 512 and cache._n_sink == 4
    assert cache._recent == 0 and cache._tau == 1.0 and cache._seed == 0


# ---------------------------------------------------------------------------
# budget invariant across shapes
# ---------------------------------------------------------------------------
def test_budget_respected_decode():
    cache = _make(keyformer_budget=12, keyformer_n_sink=2)
    for i in range(40):
        k, v = _kv(1, 3, 1, 32, seed=i)
        K, V = cache.update_and_fetch(k, v)
        assert K.shape[2] <= 12 and V.shape[2] <= 12
        assert K.shape[:2] == (1, 3) and K.shape[3] == 32


def test_budget_respected_prefill_block():
    cache = _make(keyformer_budget=12, keyformer_n_sink=2)
    k, v = _kv(2, 4, 50, 32, seed=1)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape == (2, 4, 12, 32)


def test_multi_head_independent():
    cache = _make(keyformer_budget=10, keyformer_n_sink=2)
    for i in range(30):
        k, v = _kv(1, 4, 1, 16, seed=i)
        cache.update_and_fetch(k, v)
    # all heads capped at budget
    assert cache.tokens_kept <= 10


# ---------------------------------------------------------------------------
# byte-accounting properties
# ---------------------------------------------------------------------------
def test_byte_accounting():
    cache = _make(keyformer_budget=16, keyformer_n_sink=2)
    for i in range(60):
        k, v = _kv(1, 2, 1, 32, seed=i)
        cache.update_and_fetch(k, v)
    assert cache.tokens_seen == 60 * 2  # B*H*S summed
    assert cache.keyformer_kept_bytes > 0
    assert cache.full_seq_bytes > cache.keyformer_kept_bytes
    assert cache.compression_ratio > 1.0


def test_compression_ratio_one_when_empty():
    cache = _make()
    assert cache.compression_ratio == 1.0
    assert cache.tokens_kept == 0


# ---------------------------------------------------------------------------
# prefill/decode both valid (no bit-for-bit equivalence claim — Gumbel noise
# and path-dependent accumulation make them legitimately differ)
# ---------------------------------------------------------------------------
def test_prefill_and_decode_both_within_budget():
    k_all, v_all = _kv(1, 2, 40, 24, seed=9)

    pf = _make(keyformer_budget=10, keyformer_n_sink=2, keyformer_tau=1.0)
    Kp, _ = pf.update_and_fetch(k_all, v_all)

    dc = _make(keyformer_budget=10, keyformer_n_sink=2, keyformer_tau=1.0)
    for t in range(40):
        Kd, _ = dc.update_and_fetch(k_all[:, :, t : t + 1], v_all[:, :, t : t + 1])

    assert Kp.shape[2] <= 10 and Kd.shape[2] <= 10


# ---------------------------------------------------------------------------
# tau=0 determinism at cache level
# ---------------------------------------------------------------------------
def test_tau_zero_seed_invariant_at_cache_level():
    ks = [_kv(1, 2, 1, 16, seed=i) for i in range(35)]

    def run(seed):
        cache = _make(
            keyformer_budget=10, keyformer_n_sink=2, keyformer_tau=0.0, keyformer_seed=seed
        )
        for k, v in ks:
            K, _ = cache.update_and_fetch(k, v)
        return K

    assert bool(mx.all(run(0) == run(999)).item())
