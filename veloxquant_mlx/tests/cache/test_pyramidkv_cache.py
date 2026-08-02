"""Tests for PyramidKVCache — layer-adaptive budget attention-mass eviction.

PyramidKV-adapted (arXiv:2406.02069) allocates a pyramid of per-layer budgets
(large early, small deep, fixed mean) resolved at for_model build time, evicting
within each layer via H2O cumulative-attention-mass scoring. Tests cover: factory
dispatch, interface attributes, single-layer fallback (== uniform budget), output
shape bounded by budget, output dtype fp16, sink protection, budget enforcement
across steps, byte accounting, and — the distinguishing feature — for_model
producing per-layer caches with a decreasing pyramid of budgets. All data synthetic.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.pyramidkv_cache import PyramidKVCache


def _make(**cfg):
    base = dict(method="pyramidkv", head_dim=32, pyramid_budget=8, pyramid_n_sink=2)
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_kv(S: int = 4, H: int = 2, D: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    V = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    return K, V


class _Attn:
    head_dim = 32


class _Layer:
    def __init__(self):
        self.self_attn = _Attn()


class _Model:
    def __init__(self, n):
        self.layers = [_Layer() for _ in range(n)]


# ---------------------------------------------------------------------------
# Factory and interface
# ---------------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), PyramidKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "compression_ratio")
    assert hasattr(c, "tokens_kept")
    assert hasattr(c, "pyramid_kept_bytes")
    assert hasattr(c, "layer_budget")


def test_single_layer_falls_back_to_avg_budget() -> None:
    """Factory.create (no layer context) uses pyramid_budget as the budget."""
    c = _make(pyramid_budget=13)
    assert c.layer_budget == 13


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape_below_budget() -> None:
    c = _make(pyramid_budget=16, pyramid_n_sink=2)
    k, v = _rand_kv(S=6, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 6
    assert vo.shape[2] == 6


def test_output_shape_bounded_by_budget() -> None:
    budget = 8
    c = _make(pyramid_budget=budget, pyramid_n_sink=2)
    k, v = _rand_kv(S=20, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] <= budget


def test_output_dtype_fp16() -> None:
    c = _make()
    k, v = _rand_kv(S=4)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16


def test_output_batch_head_dims_preserved() -> None:
    c = _make(pyramid_budget=16, pyramid_n_sink=0)
    k, v = _rand_kv(S=4, H=4, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[0] == 1
    assert ko.shape[1] == 4
    assert ko.shape[3] == 32


# ---------------------------------------------------------------------------
# Budget enforcement across steps
# ---------------------------------------------------------------------------


def test_budget_enforced_after_many_steps() -> None:
    budget = 10
    c = _make(pyramid_budget=budget, pyramid_n_sink=3)
    for i in range(30):
        k, v = _rand_kv(S=1, H=2, D=32, seed=i)
        ko, vo = c.update_and_fetch(k, v)
        assert ko.shape[2] <= budget, f"step {i}: seq={ko.shape[2]} > {budget}"


def test_tokens_kept_bounded_by_budget() -> None:
    budget = 8
    c = _make(pyramid_budget=budget, pyramid_n_sink=2)
    k, v = _rand_kv(S=20, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.tokens_kept <= budget


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_compression_ratio_equals_1_below_budget() -> None:
    c = _make(pyramid_budget=32, pyramid_n_sink=0)
    k, v = _rand_kv(S=4, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio == pytest.approx(1.0, rel=1e-3)


def test_compression_ratio_gt_1_after_evictions() -> None:
    c = _make(pyramid_budget=8, pyramid_n_sink=2)
    k, v = _rand_kv(S=100, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio > 1.0


def test_tokens_seen_accumulates() -> None:
    c = _make(pyramid_budget=32)
    k, v = _rand_kv(S=6, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.tokens_seen == 12  # B=1, H=2, S=6


def test_pyramid_kept_bytes_positive_after_update() -> None:
    c = _make(pyramid_budget=16)
    k, v = _rand_kv(S=4, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.pyramid_kept_bytes > 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic() -> None:
    k, v = _rand_kv(S=12, H=2, D=32)
    ko1, _ = _make().update_and_fetch(k, v)
    ko2, _ = _make().update_and_fetch(k, v)
    mse = float(mx.mean((ko1.astype(mx.float32) - ko2.astype(mx.float32)) ** 2).item())
    assert mse == pytest.approx(0.0, abs=0.0)


# ---------------------------------------------------------------------------
# for_model — the pyramid (distinguishing feature)
# ---------------------------------------------------------------------------


def test_for_model_returns_pyramidkv_caches() -> None:
    cfg = KVCacheConfig(method="pyramidkv", head_dim=32, pyramid_budget=256, pyramid_n_sink=4)
    caches = KVCacheBuilder.for_model(_Model(8), cfg)
    assert all(isinstance(c, PyramidKVCache) for c in caches)


def test_for_model_budgets_form_decreasing_pyramid() -> None:
    """Early-layer caches get a larger budget than deep-layer caches."""
    cfg = KVCacheConfig(
        method="pyramidkv", head_dim=32, pyramid_budget=256, pyramid_n_sink=4, pyramid_beta=2.0
    )
    caches = KVCacheBuilder.for_model(_Model(12), cfg)
    budgets = [c.layer_budget for c in caches]
    for i in range(len(budgets) - 1):
        assert budgets[i] >= budgets[i + 1], f"layer {i}={budgets[i]} < {i + 1}={budgets[i + 1]}"
    assert budgets[0] > budgets[-1]


def test_for_model_budget_mean_near_avg() -> None:
    """The per-layer budgets average to roughly pyramid_budget."""
    avg = 256
    cfg = KVCacheConfig(
        method="pyramidkv", head_dim=32, pyramid_budget=avg, pyramid_n_sink=4, pyramid_beta=2.0
    )
    caches = KVCacheBuilder.for_model(_Model(16), cfg)
    budgets = [c.layer_budget for c in caches]
    mean = sum(budgets) / len(budgets)
    assert abs(mean - avg) / avg < 0.05


def test_for_model_flat_beta_gives_uniform_budgets() -> None:
    """beta=1.0 → every layer gets the same budget (== uniform H2O)."""
    cfg = KVCacheConfig(
        method="pyramidkv", head_dim=32, pyramid_budget=200, pyramid_n_sink=4, pyramid_beta=1.0
    )
    caches = KVCacheBuilder.for_model(_Model(10), cfg)
    budgets = [c.layer_budget for c in caches]
    assert all(b == 200 for b in budgets)


def test_for_model_early_layer_keeps_more_tokens() -> None:
    """Feeding the same long sequence, the early-layer cache retains more tokens."""
    cfg = KVCacheConfig(
        method="pyramidkv", head_dim=32, pyramid_budget=64, pyramid_n_sink=4, pyramid_beta=2.5
    )
    caches = KVCacheBuilder.for_model(_Model(12), cfg)
    k, v = _rand_kv(S=200, H=2, D=32, seed=3)
    caches[0].update_and_fetch(k, v)
    caches[-1].update_and_fetch(k, v)
    assert caches[0].tokens_kept > caches[-1].tokens_kept
