"""Tests for CaMKVCache — cache-merging eviction (merge, don't drop).

Covers the single-layer cache (budget enforcement, sink preservation, byte
accounting, diagnostics, all merge modes), the factory route, the default
for_model path (one cache per layer, no coordinator), and the drop-mode == H2O
cache-level equivalence. All data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache import KVCacheBuilder, KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.cam_cache import CaMKVCache
from veloxquant_mlx.cache.h2o_cache import H2OKVCache


def _kv(B, H, S, D, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


class _MockAttn:
    def __init__(self, hd):
        self.head_dim = hd


class _MockLayer:
    def __init__(self, hd):
        self.self_attn = _MockAttn(hd)


class _MockModel:
    def __init__(self, n_layers, hd):
        self.layers = [_MockLayer(hd) for _ in range(n_layers)]


# ======================================================================
# Single-layer cache
# ======================================================================


def test_single_cache_reports_budget_and_mode():
    cfg = KVCacheConfig(
        method="cam", head_dim=16, cam_budget=32, cam_merge="sim_weighted", cam_n_sink=4
    )
    cache = CaMKVCache(cfg)
    assert cache.layer_budget == 32
    assert cache.merge_mode == "sim_weighted"


def test_single_cache_enforces_budget():
    cfg = KVCacheConfig(
        method="cam", head_dim=16, cam_budget=12, cam_merge="sim_weighted", cam_n_sink=2
    )
    cache = CaMKVCache(cfg)
    k, v = _kv(1, 2, 60, 16)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape[2] <= 12
    assert V.shape[2] <= 12


def test_single_cache_preserves_sinks():
    cfg = KVCacheConfig(
        method="cam", head_dim=8, cam_budget=10, cam_merge="sim_weighted", cam_n_sink=3
    )
    cache = CaMKVCache(cfg)
    k, v = _kv(1, 1, 60, 8, seed=2)
    cache.update_and_fetch(k, v)
    st = cache._states[0]
    assert bool(mx.all(st.keys[:3] == k[0, 0, :3].astype(mx.float16)).item())


def test_byte_accounting_and_ratio():
    cfg = KVCacheConfig(method="cam", head_dim=16, cam_budget=12, cam_n_sink=2)
    cache = CaMKVCache(cfg)
    k, v = _kv(1, 1, 40, 16)
    cache.update_and_fetch(k, v)
    assert cache.cam_kept_bytes > 0
    assert cache.full_seq_bytes >= cache.cam_kept_bytes
    assert cache.compression_ratio >= 1.0
    assert cache.tokens_seen == 40


def test_tokens_kept_diagnostic():
    cfg = KVCacheConfig(method="cam", head_dim=16, cam_budget=8, cam_n_sink=2)
    cache = CaMKVCache(cfg)
    k, v = _kv(1, 1, 20, 16)
    cache.update_and_fetch(k, v)
    assert cache.tokens_kept <= 8


def test_output_shapes_batch_and_heads():
    cfg = KVCacheConfig(method="cam", head_dim=8, cam_budget=6, cam_n_sink=1)
    cache = CaMKVCache(cfg)
    k, v = _kv(2, 3, 12, 8)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape[0] == 2 and K.shape[1] == 3 and K.shape[3] == 8
    assert V.shape[:2] == (2, 3)


def test_mean_mode_runs():
    cfg = KVCacheConfig(method="cam", head_dim=8, cam_budget=12, cam_n_sink=2, cam_merge="mean")
    cache = CaMKVCache(cfg)
    k, v = _kv(1, 2, 50, 8, seed=4)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape[2] <= 12


def test_merge_keys_flag_runs():
    cfg = KVCacheConfig(
        method="cam",
        head_dim=8,
        cam_budget=12,
        cam_n_sink=2,
        cam_merge="sim_weighted",
        cam_merge_keys=True,
    )
    cache = CaMKVCache(cfg)
    k, v = _kv(1, 2, 50, 8, seed=6)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape[2] <= 12


def test_prefill_then_decode():
    cfg = KVCacheConfig(method="cam", head_dim=8, cam_budget=12, cam_n_sink=2)
    cache = CaMKVCache(cfg)
    k, v = _kv(1, 2, 30, 8, seed=6)
    cache.update_and_fetch(k, v)
    for step in range(5):
        kd, vd = _kv(1, 2, 1, 8, seed=100 + step)
        K, V = cache.update_and_fetch(kd, vd)
        assert K.shape[2] <= 12
    assert cache.tokens_seen == (30 + 5) * 2


# ======================================================================
# Factory + for_model
# ======================================================================


def test_factory_creates_cam():
    cfg = KVCacheConfig(method="cam", head_dim=16, cam_budget=16)
    assert isinstance(KVCacheFactory.create(cfg), CaMKVCache)


def test_for_model_returns_cam_per_layer():
    cfg = KVCacheConfig(method="cam", head_dim=16, cam_budget=16)
    caches = KVCacheBuilder.for_model(_MockModel(4, 16), cfg)
    assert all(isinstance(c, CaMKVCache) for c in caches)
    assert len(caches) == 4


def test_for_model_budget_enforced():
    cfg = KVCacheConfig(method="cam", head_dim=16, cam_budget=16, cam_n_sink=4)
    caches = KVCacheBuilder.for_model(_MockModel(3, 16), cfg)
    for c in caches:
        k, v = _kv(1, 2, 48, 16, seed=8)
        c.update_and_fetch(k, v)
        assert c.tokens_kept <= 16


# ======================================================================
# drop mode == H2O (cache level)
# ======================================================================


@pytest.mark.parametrize("seed", [0, 1])
def test_cache_drop_mode_matches_h2o(seed):
    B, H, S, D, budget, n_sink = 1, 2, 40, 16, 8, 2
    k, v = _kv(B, H, S, D, seed=seed)

    cc = CaMKVCache(
        KVCacheConfig(
            method="cam", head_dim=D, cam_budget=budget, cam_n_sink=n_sink, cam_merge="drop"
        )
    )
    Kc, Vc = cc.update_and_fetch(k, v)

    hc = H2OKVCache(
        KVCacheConfig(
            method="h2o",
            head_dim=D,
            h2o_budget=budget,
            h2o_n_sink=n_sink,
            h2o_grace=0,
            h2o_decay=1.0,
        )
    )
    Kh, Vh = hc.update_and_fetch(k, v)

    assert Kc.shape == Kh.shape
    assert bool(mx.all(Kc == Kh).item())
    assert bool(mx.all(Vc == Vh).item())
