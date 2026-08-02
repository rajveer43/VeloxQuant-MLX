"""Tests for SnapKVKVCache — prefill observation-window token eviction.

SnapKV-adapted retains only a budget of token positions from prefill (by
observation-window attention scoring) and always appends decode tokens. These
tests cover: factory dispatch, no .bits attribute, prefill output shape, decode
accumulation, output dtype, byte accounting, keep_rate, no-eviction short-seq
edge case, n_sink=0 edge case, decode-only path, determinism, and for_model
config propagation. All data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.snapkv_cache import SnapKVKVCache


def _make(**cfg):
    base = dict(
        method="snapkv",
        head_dim=128,
        snap_budget=16,
        snap_obs_window=8,
        snap_n_sink=2,
    )
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_kv(S: int = 64, H: int = 2, D: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    V = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    return K, V


# ---------------------------------------------------------------------------
# Factory and interface
# ---------------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), SnapKVKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "eviction_ratio")
    assert hasattr(c, "keep_rate")


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


def test_prefill_output_shape_evicted() -> None:
    """After prefill, seq dim should be min(budget, S)."""
    c = _make(snap_budget=16, snap_obs_window=8, snap_n_sink=2)
    k, v = _rand_kv(S=64, H=2, D=128)
    ko, vo = c.update_and_fetch(k, v)
    # mlx_lm accumulates: seq dim after one prefill = min(budget, S) = 16
    assert ko.shape[2] == 16
    assert vo.shape[2] == 16


def test_output_dtype_fp16() -> None:
    c = _make()
    k, v = _rand_kv(S=64)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16


def test_no_eviction_short_seq() -> None:
    """budget >= S: all tokens kept, seq dim == S after prefill."""
    c = _make(snap_budget=200, snap_obs_window=4, snap_n_sink=2)
    k, v = _rand_kv(S=10, H=2, D=128)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 10
    assert c.keep_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Decode accumulation
# ---------------------------------------------------------------------------


def test_decode_accumulation() -> None:
    """Decode tokens grow the seq dim by 1 each call."""
    c = _make(snap_budget=16, snap_obs_window=8, snap_n_sink=2)
    k, v = _rand_kv(S=64)
    c.update_and_fetch(k, v)  # prefill → 16 tokens kept
    for i in range(4):
        k1, v1 = _rand_kv(S=1, seed=100 + i)
        ko, vo = c.update_and_fetch(k1, v1)
    assert ko.shape[2] == 16 + 4


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_byte_ordering_eviction_ratio_gt_1() -> None:
    """After prefill with budget < S, eviction_ratio > 1."""
    c = _make(snap_budget=16, snap_obs_window=8, snap_n_sink=2)
    k, v = _rand_kv(S=64)
    c.update_and_fetch(k, v)
    assert c.eviction_ratio > 1.0
    assert c.evicted_key_bytes < c.full_key_bytes


def test_keep_rate_in_range() -> None:
    c = _make(snap_budget=16, snap_obs_window=8, snap_n_sink=2)
    k, v = _rand_kv(S=64)
    c.update_and_fetch(k, v)
    assert 0.0 < c.keep_rate <= 1.0


def test_keep_rate_no_eviction() -> None:
    c = _make(snap_budget=200)
    k, v = _rand_kv(S=10)
    c.update_and_fetch(k, v)
    assert c.keep_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_n_sink_zero() -> None:
    """n_sink=0 runs without error."""
    c = _make(snap_n_sink=0, snap_budget=16)
    k, v = _rand_kv(S=64)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 16


def test_decode_only_no_eviction() -> None:
    """Single-token first call (S=1) is treated as decode — no eviction."""
    c = _make(snap_budget=8)
    k, v = _rand_kv(S=1)
    ko, vo = c.update_and_fetch(k, v)
    assert c.keep_rate == pytest.approx(1.0)
    assert ko.shape[2] == 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic() -> None:
    k, v = _rand_kv(S=64)
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
        head_dim = 128

    class _Layer:
        self_attn = _Attn()

    class _Model:
        layers = [_Layer(), _Layer()]

    cfg = KVCacheConfig(
        method="snapkv",
        head_dim=128,
        snap_budget=32,
        snap_obs_window=16,
        snap_n_sink=3,
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, SnapKVKVCache) for c in caches)
    assert caches[0]._budget == 32
    assert caches[0]._obs_window == 16
    assert caches[0]._n_sink == 3
