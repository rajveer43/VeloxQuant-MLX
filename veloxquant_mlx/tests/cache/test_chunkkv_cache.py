"""Tests for ChunkKVCache — chunk-level (semantic-block) eviction.

Covers the single-layer cache (budget enforcement, chunk alignment, sink
preservation, byte accounting, diagnostics, both score modes), the
KVCacheFactory route, the default KVCacheBuilder.for_model path (one cache per
layer, no coordinator), and the C=1 == H2O cache-level equivalence. All data is
synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache import KVCacheBuilder, KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.chunkkv_cache import ChunkKVCache
from veloxquant_mlx.cache.h2o_cache import H2OKVCache


def _kv(B, H, S, D, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


# ----- mock model ------------------------------------------------------


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


def test_single_cache_reports_budget_and_chunk():
    cfg = KVCacheConfig(
        method="chunkkv", head_dim=16, chunkkv_budget=32, chunkkv_chunk_size=8, chunkkv_n_sink=4
    )
    cache = ChunkKVCache(cfg)
    assert cache.layer_budget == 32
    assert cache.chunk_size == 8


def test_single_cache_enforces_budget():
    cfg = KVCacheConfig(
        method="chunkkv", head_dim=16, chunkkv_budget=12, chunkkv_chunk_size=4, chunkkv_n_sink=2
    )
    cache = ChunkKVCache(cfg)
    k, v = _kv(1, 2, 60, 16)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape[2] <= 12
    assert V.shape[2] <= 12


def test_single_cache_chunk_aligned_survivors():
    cfg = KVCacheConfig(
        method="chunkkv", head_dim=8, chunkkv_budget=20, chunkkv_chunk_size=4, chunkkv_n_sink=4
    )
    cache = ChunkKVCache(cfg)
    k, v = _kv(1, 1, 200, 8)
    cache.update_and_fetch(k, v)
    body = cache.tokens_kept - 4
    assert body % 4 == 0


def test_single_cache_preserves_sinks():
    cfg = KVCacheConfig(
        method="chunkkv", head_dim=8, chunkkv_budget=10, chunkkv_chunk_size=4, chunkkv_n_sink=3
    )
    cache = ChunkKVCache(cfg)
    k, v = _kv(1, 1, 60, 8, seed=2)
    cache.update_and_fetch(k, v)
    st = cache._states[0]
    assert bool(mx.all(st.keys[:3] == k[0, 0, :3].astype(mx.float16)).item())


def test_byte_accounting_and_ratio():
    cfg = KVCacheConfig(
        method="chunkkv", head_dim=16, chunkkv_budget=12, chunkkv_chunk_size=4, chunkkv_n_sink=2
    )
    cache = ChunkKVCache(cfg)
    k, v = _kv(1, 1, 40, 16)
    cache.update_and_fetch(k, v)
    assert cache.chunkkv_kept_bytes > 0
    assert cache.full_seq_bytes >= cache.chunkkv_kept_bytes
    assert cache.compression_ratio >= 1.0
    assert cache.tokens_seen == 40


def test_tokens_kept_diagnostic():
    cfg = KVCacheConfig(
        method="chunkkv", head_dim=16, chunkkv_budget=8, chunkkv_chunk_size=2, chunkkv_n_sink=2
    )
    cache = ChunkKVCache(cfg)
    k, v = _kv(1, 1, 20, 16)
    cache.update_and_fetch(k, v)
    assert cache.tokens_kept <= 8


def test_output_shapes_batch_and_heads():
    cfg = KVCacheConfig(
        method="chunkkv", head_dim=8, chunkkv_budget=6, chunkkv_chunk_size=2, chunkkv_n_sink=1
    )
    cache = ChunkKVCache(cfg)
    k, v = _kv(2, 3, 12, 8)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape[0] == 2 and K.shape[1] == 3 and K.shape[3] == 8
    assert V.shape[:2] == (2, 3)


def test_key_norm_score_mode():
    cfg = KVCacheConfig(
        method="chunkkv",
        head_dim=8,
        chunkkv_budget=12,
        chunkkv_chunk_size=4,
        chunkkv_n_sink=2,
        chunkkv_score="key_norm",
    )
    cache = ChunkKVCache(cfg)
    k, v = _kv(1, 2, 50, 8, seed=4)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape[2] <= 12


def test_prefill_then_decode():
    cfg = KVCacheConfig(
        method="chunkkv", head_dim=8, chunkkv_budget=12, chunkkv_chunk_size=4, chunkkv_n_sink=2
    )
    cache = ChunkKVCache(cfg)
    k, v = _kv(1, 2, 30, 8, seed=6)  # prefill
    cache.update_and_fetch(k, v)
    for step in range(5):  # decode
        kd, vd = _kv(1, 2, 1, 8, seed=100 + step)
        K, V = cache.update_and_fetch(kd, vd)
        assert K.shape[2] <= 12
    assert cache.tokens_seen == (30 + 5) * 2


# ======================================================================
# Factory + for_model
# ======================================================================


def test_factory_creates_chunkkv():
    cfg = KVCacheConfig(method="chunkkv", head_dim=16, chunkkv_budget=16)
    cache = KVCacheFactory.create(cfg)
    assert isinstance(cache, ChunkKVCache)


def test_for_model_returns_chunkkv_per_layer():
    cfg = KVCacheConfig(method="chunkkv", head_dim=16, chunkkv_budget=16, chunkkv_chunk_size=4)
    caches = KVCacheBuilder.for_model(_MockModel(4, 16), cfg)
    assert all(isinstance(c, ChunkKVCache) for c in caches)
    assert len(caches) == 4


def test_for_model_budget_enforced():
    cfg = KVCacheConfig(
        method="chunkkv", head_dim=16, chunkkv_budget=16, chunkkv_chunk_size=4, chunkkv_n_sink=4
    )
    caches = KVCacheBuilder.for_model(_MockModel(3, 16), cfg)
    for c in caches:
        k, v = _kv(1, 2, 48, 16, seed=8)
        c.update_and_fetch(k, v)
        assert c.tokens_kept <= 16


# ======================================================================
# C = 1  ==  H2O  (cache level)
# ======================================================================


@pytest.mark.parametrize("seed", [0, 1])
def test_cache_chunk_size_one_matches_h2o(seed):
    B, H, S, D, budget, n_sink = 1, 2, 40, 16, 8, 2
    k, v = _kv(B, H, S, D, seed=seed)

    cc = ChunkKVCache(
        KVCacheConfig(
            method="chunkkv",
            head_dim=D,
            chunkkv_budget=budget,
            chunkkv_chunk_size=1,
            chunkkv_n_sink=n_sink,
            chunkkv_score="attn_mass",
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
