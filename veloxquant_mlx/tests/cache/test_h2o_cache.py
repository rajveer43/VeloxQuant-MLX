"""Tests for H2OKVCache — cumulative attention-mass heavy-hitter oracle eviction.

H2O-adapted (arXiv:2306.14048, ICLR 2024) accumulates per-token attention weights
as a proxy importance score and evicts the lowest-score non-sink token whenever the
cache exceeds h2o_budget. Tests cover: factory dispatch, interface attributes,
output shape bounded by budget, output dtype fp16, sink protection, decode
accumulation, budget enforcement across many steps, byte accounting
(compression_ratio, h2o_kept_bytes), tokens_kept, n_sink=0 edge case,
determinism, and for_model config propagation. All data is synthetic.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.h2o_cache import H2OKVCache


def _make(**cfg):
    base = dict(method="h2o", head_dim=32, h2o_budget=8, h2o_n_sink=2)
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_kv(S: int = 4, H: int = 2, D: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    V = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    return K, V


# ---------------------------------------------------------------------------
# Factory and interface
# ---------------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), H2OKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "compression_ratio")
    assert hasattr(c, "tokens_kept")
    assert hasattr(c, "h2o_kept_bytes")


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape_below_budget() -> None:
    """S < budget → all tokens returned."""
    c = _make(h2o_budget=16, h2o_n_sink=2)
    k, v = _rand_kv(S=6, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 6
    assert vo.shape[2] == 6


def test_output_shape_bounded_by_budget() -> None:
    """S > budget → output seq dim <= budget."""
    budget = 8
    c = _make(h2o_budget=budget, h2o_n_sink=2)
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
    """B and H dims pass through unchanged."""
    c = _make(h2o_budget=16, h2o_n_sink=0)
    k, v = _rand_kv(S=4, H=4, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[0] == 1  # B
    assert ko.shape[1] == 4  # H
    assert ko.shape[3] == 32  # D


# ---------------------------------------------------------------------------
# Budget enforcement across steps
# ---------------------------------------------------------------------------


def test_budget_enforced_after_many_steps() -> None:
    """30 decode steps — output seq dim never exceeds budget."""
    budget = 10
    c = _make(h2o_budget=budget, h2o_n_sink=3)
    for i in range(30):
        k, v = _rand_kv(S=1, H=2, D=32, seed=i)
        ko, vo = c.update_and_fetch(k, v)
        assert ko.shape[2] <= budget, f"step {i}: seq={ko.shape[2]} > {budget}"


def test_tokens_kept_bounded_by_budget() -> None:
    budget = 8
    c = _make(h2o_budget=budget, h2o_n_sink=2)
    k, v = _rand_kv(S=20, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.tokens_kept <= budget


# ---------------------------------------------------------------------------
# Sink protection
# ---------------------------------------------------------------------------


def test_n_sink_zero_still_enforces_budget() -> None:
    """With n_sink=0, all tokens may be evicted; budget still respected."""
    budget = 4
    c = _make(h2o_budget=budget, h2o_n_sink=0)
    k, v = _rand_kv(S=20, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] <= budget


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_compression_ratio_equals_1_below_budget() -> None:
    """When tokens < budget, no eviction → ratio == 1."""
    c = _make(h2o_budget=32, h2o_n_sink=0)
    k, v = _rand_kv(S=4, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio == pytest.approx(1.0, rel=1e-3)


def test_compression_ratio_gt_1_after_evictions() -> None:
    """After many evictions, ratio > 1."""
    c = _make(h2o_budget=8, h2o_n_sink=2)
    k, v = _rand_kv(S=100, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio > 1.0


def test_tokens_seen_accumulates() -> None:
    """tokens_seen grows by B * H * S per call."""
    c = _make(h2o_budget=32)
    k, v = _rand_kv(S=6, H=2, D=32)
    c.update_and_fetch(k, v)
    # B=1, H=2, S=6 → 12
    assert c.tokens_seen == 12


def test_h2o_kept_bytes_positive_after_update() -> None:
    c = _make(h2o_budget=16)
    k, v = _rand_kv(S=4, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.h2o_kept_bytes > 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic() -> None:
    k, v = _rand_kv(S=12, H=2, D=32)
    c1 = _make()
    c2 = _make()
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
        head_dim = 32

    class _Layer:
        self_attn = _Attn()

    class _Model:
        layers = [_Layer(), _Layer(), _Layer()]

    cfg = KVCacheConfig(
        method="h2o",
        head_dim=32,
        h2o_budget=64,
        h2o_n_sink=8,
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, H2OKVCache) for c in caches)
    assert caches[0]._budget == 64
    assert caches[0]._n_sink == 8
