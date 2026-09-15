"""Tests for SqueezeAttentionCache — 2D layer×token data-driven budget eviction.

Covers the single-layer cache (fallback uniform budget, byte accounting,
diagnostics), the SqueezeCoordinator (concentration reporting, one-shot finalise,
resolved-budget lookup) and the end-to-end KVCacheBuilder.for_model path (shared
coordinator, data-driven per-layer budgets, broad layer keeps more than
concentrated, strength=0 uniform). All data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache import KVCacheBuilder, KVCacheConfig
from veloxquant_mlx.cache.squeeze_cache import SqueezeAttentionCache
from veloxquant_mlx.cache.squeeze_coordinator import SqueezeCoordinator


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


def _prefill_concentration(li, n_layers, B, H, S, D, seed=0):
    """Keys that grow more concentrated (clustered) with layer depth."""
    frac = li / max(n_layers - 1, 1)
    rng = np.random.default_rng(seed + li)
    base = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32))
    direction = mx.ones((B, H, 1, D))
    k = ((1 - frac) * base + frac * 3.0 * direction).astype(mx.float16)
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


# ======================================================================
# Single-layer cache (no coordinator → uniform fallback)
# ======================================================================


def test_single_cache_uses_fallback_budget():
    cfg = KVCacheConfig(method="squeeze", head_dim=16, squeeze_budget=32, squeeze_n_sink=4)
    cache = SqueezeAttentionCache(cfg)
    assert cache.layer_budget == 32


def test_single_cache_enforces_budget():
    cfg = KVCacheConfig(method="squeeze", head_dim=16, squeeze_budget=8, squeeze_n_sink=2)
    cache = SqueezeAttentionCache(cfg)
    k, v = _kv(1, 2, 20, 16)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape[2] <= 8
    assert V.shape[2] <= 8


def test_single_cache_resolved_override():
    cfg = KVCacheConfig(
        method="squeeze", head_dim=16, squeeze_budget=64, squeeze_resolved_budget=10
    )
    cache = SqueezeAttentionCache(cfg)
    assert cache.layer_budget == 10


def test_byte_accounting_and_ratio():
    cfg = KVCacheConfig(method="squeeze", head_dim=16, squeeze_budget=8, squeeze_n_sink=2)
    cache = SqueezeAttentionCache(cfg)
    k, v = _kv(1, 1, 40, 16)
    cache.update_and_fetch(k, v)
    assert cache.squeeze_kept_bytes > 0
    assert cache.full_seq_bytes >= cache.squeeze_kept_bytes
    assert cache.compression_ratio >= 1.0
    assert cache.tokens_seen == 40


def test_tokens_kept_diagnostic():
    cfg = KVCacheConfig(method="squeeze", head_dim=16, squeeze_budget=6, squeeze_n_sink=2)
    cache = SqueezeAttentionCache(cfg)
    k, v = _kv(1, 1, 20, 16)
    cache.update_and_fetch(k, v)
    assert cache.tokens_kept <= 6


def test_output_shapes_batch_and_heads():
    cfg = KVCacheConfig(method="squeeze", head_dim=8, squeeze_budget=5, squeeze_n_sink=1)
    cache = SqueezeAttentionCache(cfg)
    k, v = _kv(2, 3, 12, 8)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape[0] == 2 and K.shape[1] == 3 and K.shape[3] == 8
    assert V.shape[:2] == (2, 3)


# ======================================================================
# SqueezeCoordinator
# ======================================================================


def test_coordinator_not_finalized_until_all_report():
    coord = SqueezeCoordinator(n_layers=3, avg_budget=100, n_sink=4, strength=1.0)
    coord.report_concentration(0, 0.1)
    assert not coord.is_finalized
    assert coord.resolved_budget(0) is None
    coord.report_concentration(1, 0.5)
    assert not coord.is_finalized
    coord.report_concentration(2, 0.9)
    assert coord.is_finalized


def test_coordinator_resolves_budgets():
    coord = SqueezeCoordinator(n_layers=3, avg_budget=100, n_sink=4, strength=1.0)
    coord.report_concentration(0, 0.1)  # broad → more
    coord.report_concentration(1, 0.5)
    coord.report_concentration(2, 0.9)  # concentrated → less
    assert coord.resolved_budget(0) > coord.resolved_budget(2)


def test_coordinator_report_idempotent():
    """First report per layer wins; later (decode) reports are ignored."""
    coord = SqueezeCoordinator(n_layers=2, avg_budget=100, n_sink=4, strength=1.0)
    coord.report_concentration(0, 0.1)
    coord.report_concentration(1, 0.9)
    b_before = coord.resolved_budget(0)
    coord.report_concentration(0, 0.99)  # ignored — already finalised
    assert coord.resolved_budget(0) == b_before


def test_coordinator_strength_zero_uniform():
    coord = SqueezeCoordinator(n_layers=4, avg_budget=128, n_sink=4, strength=0.0)
    for i, c in enumerate([0.1, 0.4, 0.7, 0.95]):
        coord.report_concentration(i, c)
    budgets = [coord.resolved_budget(i) for i in range(4)]
    assert budgets == [128, 128, 128, 128]


def test_coordinator_reset():
    coord = SqueezeCoordinator(n_layers=2, avg_budget=100, n_sink=4, strength=1.0)
    coord.report_concentration(0, 0.1)
    coord.report_concentration(1, 0.9)
    assert coord.is_finalized
    coord.reset()
    assert not coord.is_finalized
    assert coord.resolved_budget(0) is None


# ======================================================================
# for_model — end-to-end
# ======================================================================


def _build_and_run(n_layers, hd, strength, avg_budget=32, n_sink=4):
    model = _MockModel(n_layers, hd)
    cfg = KVCacheConfig(
        method="squeeze",
        head_dim=hd,
        squeeze_budget=avg_budget,
        squeeze_n_sink=n_sink,
        squeeze_strength=strength,
    )
    caches = KVCacheBuilder.for_model(model, cfg)
    B, H, S = 1, 2, 48
    # prefill (all layers report)
    for li, c in enumerate(caches):
        k, v = _prefill_concentration(li, n_layers, B, H, S, hd)
        c.update_and_fetch(k, v)
    # one decode step (layers adopt resolved budget)
    for c in caches:
        k, v = _kv(B, H, 1, hd, seed=99)
        c.update_and_fetch(k, v)
    return caches


def test_for_model_returns_squeeze_caches():
    caches = _build_and_run(5, 16, strength=1.0)
    assert all(isinstance(c, SqueezeAttentionCache) for c in caches)


def test_for_model_shares_one_coordinator():
    caches = _build_and_run(4, 16, strength=1.0)
    coords = {id(c._coordinator) for c in caches}
    assert len(coords) == 1


def test_for_model_data_driven_budgets_vary():
    """With real concentration variation the budgets are not all equal."""
    caches = _build_and_run(6, 16, strength=1.0)
    budgets = [c.layer_budget for c in caches]
    assert len(set(budgets)) > 1


def test_for_model_broad_layer_keeps_more():
    """The broad (early) layer ends with a larger budget than the concentrated one."""
    caches = _build_and_run(6, 16, strength=1.0)
    assert caches[0].layer_budget > caches[-1].layer_budget


def test_for_model_mean_budget_near_avg():
    caches = _build_and_run(8, 16, strength=1.0, avg_budget=32)
    budgets = [c.layer_budget for c in caches]
    # floor can nudge the mean up; it should never fall below avg.
    assert sum(budgets) / len(budgets) >= 32 * 0.95


def test_for_model_strength_zero_uniform_budgets():
    caches = _build_and_run(6, 16, strength=0.0, avg_budget=32)
    budgets = [c.layer_budget for c in caches]
    assert all(b == 32 for b in budgets)


def test_for_model_concentration_reported():
    caches = _build_and_run(5, 16, strength=1.0)
    concs = [c.concentration for c in caches]
    # deep clustered layers should report higher concentration than the first
    assert concs[-1] > concs[0]


def test_for_model_budget_enforced_after_rebudget():
    caches = _build_and_run(6, 16, strength=1.0)
    for c in caches:
        assert c.tokens_kept <= c.layer_budget


# ======================================================================
# Batching guard (see VeloxQuant-MLX#358) — VeloxQuant-Studio issue #28
# ======================================================================


def test_is_trimmable_false():
    cfg = KVCacheConfig(method="squeeze", head_dim=16, squeeze_budget=32, squeeze_n_sink=4)
    cache = SqueezeAttentionCache(cfg)
    assert cache.is_trimmable() is False


def test_not_batchable_via_mlx_lm_server_probe():
    cfg = KVCacheConfig(method="squeeze", head_dim=16, squeeze_budget=32, squeeze_n_sink=4)
    cache = SqueezeAttentionCache(cfg)
    # mlx_lm.server's hasattr(cache, "merge") probe must see this as absent —
    # a property that raises on access makes hasattr() return False.
    assert not hasattr(cache, "merge")
    with pytest.raises(AttributeError):
        cache.merge  # noqa: B018 — accessing the property is the point


def test_merge_on_empty_cache_would_silently_substitute_if_inherited():
    """Guards against regressing to the base classmethod: on a batch of brand-new
    (empty) caches, ``mlx_lm``'s ``KVCache.merge()`` silently returns a plain
    ``BatchKVCache`` instead of raising or preserving SqueezeAttention behaviour —
    exactly the substitution the ``merge`` property above must prevent.
    """
    from mlx_lm.models.cache import BatchKVCache
    from mlx_lm.models.cache import KVCache as _MLXKVCache

    cfg = KVCacheConfig(method="squeeze", head_dim=16, squeeze_budget=32, squeeze_n_sink=4)
    cache = SqueezeAttentionCache(cfg)
    assert cache.size() == 0
    merged = _MLXKVCache.merge.__func__(SqueezeAttentionCache, [cache])
    assert isinstance(merged, BatchKVCache)
    assert not isinstance(merged, SqueezeAttentionCache)


# ======================================================================
# field_is_relevant / config_fields leak — VeloxQuant-Studio issue #28
# ======================================================================


def test_squeeze_resolved_budget_not_relevant():
    """squeeze_resolved_budget is an internal field the coordinator resolves at
    runtime, not a user-facing knob — `--set squeeze_resolved_budget=N` or the
    app's parameter editor must not treat it as a valid override, else it could
    silently pin every layer to one fixed budget and bypass the 2D data-driven
    reallocation entirely with no indication anything unusual happened.

    field_is_relevant is the direct guard used by `--set` validation.
    get_method(...).config_fields is what the app's parameter editor actually
    renders — it must consult the curated _CONFIG_FIELDS entry (now present
    for "squeeze"), not the uncurated name-prefix fallback
    (_default_config_fields), which cannot tell squeeze_resolved_budget apart
    from squeeze_budget/squeeze_n_sink/squeeze_strength.
    """
    from veloxquant_mlx.cache.registry import field_is_relevant, get_method

    assert field_is_relevant("squeeze", "squeeze_resolved_budget") is False
    assert "squeeze_resolved_budget" not in get_method("squeeze").config_fields


def test_squeeze_user_facing_fields_still_relevant():
    from veloxquant_mlx.cache.registry import field_is_relevant

    for name in ("squeeze_budget", "squeeze_n_sink", "squeeze_strength"):
        assert field_is_relevant("squeeze", name) is True
