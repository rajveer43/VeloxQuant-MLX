"""Tests for AMCKVCache — saliency-driven per-token tiered rank + precision.

AMC-adapted (arXiv:2607.10109, no verified venue) never evicts tokens: every
token seen is retained, with its rank/bit-width set by its saliency tier.
Tests cover: factory dispatch, interface attributes, output shape (always ==
tokens seen, unlike every eviction method), output dtype fp16, tier-count
observability, byte accounting, determinism, query-aware + adaptive-threshold
opt-in paths, and for_model config propagation. All data is synthetic.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.amc_cache import AMCKVCache
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory


def _make(**cfg):
    base = dict(method="amc", head_dim=32)
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
    assert isinstance(_make(), AMCKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "compression_ratio")
    assert hasattr(c, "tokens_kept")
    assert hasattr(c, "amc_kept_bytes")
    assert hasattr(c, "tokens_high")
    assert hasattr(c, "tokens_mid")
    assert hasattr(c, "tokens_low")


# ---------------------------------------------------------------------------
# Shape and dtype — no eviction, output always == tokens seen
# ---------------------------------------------------------------------------


def test_output_shape_equals_tokens_seen_prefill() -> None:
    c = _make()
    k, v = _rand_kv(S=6, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 6
    assert vo.shape[2] == 6


def test_output_shape_never_shrinks_across_steps() -> None:
    """Unlike every eviction method, seq dim only ever grows."""
    c = _make()
    k, v = _rand_kv(S=10, H=2, D=32, seed=0)
    ko, _ = c.update_and_fetch(k, v)
    assert ko.shape[2] == 10
    for i in range(5):
        k, v = _rand_kv(S=1, H=2, D=32, seed=50 + i)
        ko, vo = c.update_and_fetch(k, v)
        assert ko.shape[2] == 10 + i + 1


def test_no_eviction_all_tokens_retained() -> None:
    """Direct proof of the compression-only design: cache size always equals
    cumulative tokens passed, never clamped down."""
    c = _make()
    total = 0
    for i, S in enumerate([8, 1, 1, 1, 5, 1]):
        k, v = _rand_kv(S=S, H=1, D=32, seed=i)
        ko, vo = c.update_and_fetch(k, v)
        total += S
        assert ko.shape[2] == total
        assert vo.shape[2] == total


def test_output_dtype_fp16() -> None:
    c = _make()
    k, v = _rand_kv(S=4)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16


def test_output_batch_head_dims_preserved() -> None:
    c = _make()
    k, v = _rand_kv(S=4, H=4, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[0] == 1
    assert ko.shape[1] == 4
    assert ko.shape[3] == 32


# ---------------------------------------------------------------------------
# Tier distribution
# ---------------------------------------------------------------------------


def test_tier_counts_sum_to_tokens_seen() -> None:
    c = _make(amc_k_high=0.20, amc_k_mid=0.30)
    k, v = _rand_kv(S=20, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.tokens_high + c.tokens_mid + c.tokens_low == c.tokens_seen


def test_tier_counts_roughly_match_percentiles() -> None:
    c = _make(amc_k_high=0.20, amc_k_mid=0.30)
    k, v = _rand_kv(S=200, H=1, D=32, seed=42)
    c.update_and_fetch(k, v)
    total = c.tokens_high + c.tokens_mid + c.tokens_low
    assert c.tokens_high / total == pytest.approx(0.20, abs=0.05)
    assert c.tokens_mid / total == pytest.approx(0.30, abs=0.05)


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_compression_ratio_gt_1() -> None:
    c = _make()
    k, v = _rand_kv(S=64, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio > 1.0


def test_tokens_seen_accumulates() -> None:
    c = _make()
    k, v = _rand_kv(S=6, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.tokens_seen == 12  # B=1, H=2, S=6


def test_amc_kept_bytes_positive_after_update() -> None:
    c = _make()
    k, v = _rand_kv(S=4, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.amc_kept_bytes > 0


def test_tokens_kept_matches_tokens_per_head() -> None:
    c = _make()
    k, v = _rand_kv(S=15, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.tokens_kept == 15


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
# Opt-in paths: query-aware saliency, adaptive thresholds
# ---------------------------------------------------------------------------


def test_query_aware_saliency_opt_in_runs_without_crash() -> None:
    c = _make(amc_use_query_saliency=True, amc_query_alpha=0.4)
    k, v = _rand_kv(S=10, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 10
    assert not bool(mx.any(mx.isnan(ko)).item())


def test_adaptive_thresholds_requires_calib_variance() -> None:
    with pytest.raises(ValueError):
        _make(amc_adaptive_thresholds=True, amc_calib_variance=None)


def test_k_high_above_one_rejected() -> None:
    with pytest.raises(ValueError, match="amc_k_high"):
        _make(amc_k_high=1.5)


def test_k_high_negative_rejected() -> None:
    with pytest.raises(ValueError, match="amc_k_high"):
        _make(amc_k_high=-0.1)


def test_k_high_plus_k_mid_above_one_rejected() -> None:
    """amc_assign_tiers silently clamps n_high/n_mid to fit N tokens, so an
    invalid k_high + k_mid > 1.0 config would otherwise pass through
    unnoticed instead of surfacing as a config error."""
    with pytest.raises(ValueError, match="amc_k_high \\+ amc_k_mid"):
        _make(amc_k_high=0.8, amc_k_mid=0.5)


def test_adaptive_thresholds_opt_in_runs_without_crash() -> None:
    c = _make(amc_adaptive_thresholds=True, amc_calib_variance=0.05, amc_threshold_window=16)
    k, v = _rand_kv(S=20, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 20
    assert not bool(mx.any(mx.isnan(ko)).item())


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
        method="amc",
        head_dim=32,
        amc_k_high=0.15,
        amc_k_mid=0.35,
        amc_group_size=16,
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, AMCKVCache) for c in caches)
    assert caches[0]._k_high == pytest.approx(0.15)
    assert caches[0]._k_mid == pytest.approx(0.35)
    assert caches[0]._group_size == 16


def test_factory_smoke_compression_ratio_positive_both_kv() -> None:
    """End-to-end factory smoke test through both K and V."""
    c = _make()
    k, v = _rand_kv(S=64, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 64
    assert vo.shape[2] == 64
    assert c.compression_ratio > 1.0
