"""Tests for CaMKVCache — cache-merging eviction (merge, don't drop).

Covers the single-layer cache (budget enforcement, sink preservation, byte
accounting, diagnostics, all merge modes), the Eq. 14 merge gate (default-on,
disableable, deterministic across runs), the factory route, the default
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
    """STORED seq dim <= budget. The call's own RETURN value is deliberately
    NOT capped at budget (see #370): mlx_lm's attention mask for this call
    is fixed before eviction can run, so update_and_fetch defers eviction to
    storage only and returns the full pre-eviction set for this call's own
    (already correctly masked) attention — here, the first-ever call, so
    the return is exactly the S=60 raw incoming tokens."""
    cfg = KVCacheConfig(
        method="cam", head_dim=16, cam_budget=12, cam_merge="sim_weighted", cam_n_sink=2
    )
    cache = CaMKVCache(cfg)
    k, v = _kv(1, 2, 60, 16)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape[2] == 60
    assert V.shape[2] == 60
    assert cache.tokens_kept <= 12


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
    # RETURN is deliberately un-evicted (#370); STORAGE is capped.
    assert K.shape[2] == 50
    assert cache.tokens_kept <= 12


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
    # RETURN is deliberately un-evicted (#370); STORAGE is capped.
    assert K.shape[2] == 50
    assert cache.tokens_kept <= 12


def test_prefill_then_decode():
    """STORED seq dim stays <= budget across prefill + decode. Each step's
    own RETURN is one token larger than what's stored (the previous step's
    kept set + this step's 1 new token, un-evicted — see #370's
    deferred-eviction fix in update_and_fetch's docstring)."""
    cfg = KVCacheConfig(method="cam", head_dim=8, cam_budget=12, cam_n_sink=2)
    cache = CaMKVCache(cfg)
    k, v = _kv(1, 2, 30, 8, seed=6)
    cache.update_and_fetch(k, v)
    for step in range(5):
        stored_before = cache.tokens_kept
        kd, vd = _kv(1, 2, 1, 8, seed=100 + step)
        K, V = cache.update_and_fetch(kd, vd)
        assert cache.tokens_kept <= 12
        assert K.shape[2] == stored_before + 1
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


# ======================================================================
# Merge gate (Eq. 14 Bernoulli mask)
# ======================================================================


def test_cache_merge_gate_default_on():
    cfg = KVCacheConfig(method="cam", head_dim=16, cam_budget=16)
    cache = CaMKVCache(cfg)
    assert cache.merge_gate is True


def test_cache_merge_gate_off_reports_false():
    cfg = KVCacheConfig(method="cam", head_dim=16, cam_budget=16, cam_merge_gate=False)
    cache = CaMKVCache(cfg)
    assert cache.merge_gate is False


def test_cache_gate_off_matches_old_unconditional_merge_shape():
    """merge_gate=False still enforces budget (in STORAGE) the same way as
    gated merging. RETURN is deliberately un-evicted (#370) — both configs'
    first-ever call returns the full S=50 raw incoming tokens."""
    cfg_gated = KVCacheConfig(method="cam", head_dim=16, cam_budget=12, cam_n_sink=2)
    cfg_ungated = KVCacheConfig(
        method="cam", head_dim=16, cam_budget=12, cam_n_sink=2, cam_merge_gate=False
    )
    k, v = _kv(1, 2, 50, 16, seed=15)
    cache_g = CaMKVCache(cfg_gated)
    cache_u = CaMKVCache(cfg_ungated)
    Kg, Vg = cache_g.update_and_fetch(k, v)
    Ku, Vu = cache_u.update_and_fetch(k, v)
    assert Kg.shape == Ku.shape == (1, 2, 50, 16)
    assert cache_g.tokens_kept == cache_u.tokens_kept == 12


def test_cache_gate_is_deterministic_across_runs():
    cfg = KVCacheConfig(method="cam", head_dim=16, cam_budget=12, cam_n_sink=2, seed=42)
    k, v = _kv(1, 2, 50, 16, seed=15)
    K1, V1 = CaMKVCache(cfg).update_and_fetch(k, v)
    K2, V2 = CaMKVCache(cfg).update_and_fetch(k, v)
    assert bool(mx.all(K1 == K2).item())
    assert bool(mx.all(V1 == V2).item())


# ======================================================================
# Attention mask correctness (#370)
# ======================================================================


def test_make_mask_before_any_call_falls_back_to_base() -> None:
    from mlx_lm.models.base import create_attention_mask

    cfg = KVCacheConfig(method="cam", head_dim=8, cam_budget=8, cam_n_sink=2)
    c = CaMKVCache(cfg)
    h_fake = mx.zeros((1, 5, 4))
    assert create_attention_mask(h_fake, c) == "causal"


def test_every_call_returns_full_unevicted_set_for_own_attention() -> None:
    """Neither the first nor any later multi-token call may shrink what it
    RETURNS below its own pre-eviction count — mlx_lm's mask for that call
    is fixed (based on the previous call's true kept positions) before this
    call's own eviction/merge can run, and only what's returned matches
    that fixed mask's shape."""
    budget = 6
    cfg = KVCacheConfig(method="cam", head_dim=8, cam_budget=budget, cam_n_sink=1)
    c = CaMKVCache(cfg)
    k1, v1 = _kv(1, 1, 5, 8, seed=1)
    ko1, _ = c.update_and_fetch(k1, v1)
    assert ko1.shape[2] == 5  # first call: nothing stored yet to concat onto
    assert c.tokens_kept <= budget

    k2, v2 = _kv(1, 1, 5, 8, seed=2)
    ko2, _ = c.update_and_fetch(k2, v2)
    # returned == (previously stored, <= budget) ++ (this call's 5 new)
    prev_stored = min(5, budget)
    assert ko2.shape[2] == prev_stored + 5
    assert c.tokens_kept <= budget


def test_make_mask_after_eviction_is_position_correct_explicit_array() -> None:
    from mlx_lm.models.base import create_attention_mask

    budget = 6
    cfg = KVCacheConfig(method="cam", head_dim=8, cam_budget=budget, cam_n_sink=1)
    c = CaMKVCache(cfg)
    k1, v1 = _kv(1, 1, 15, 8, seed=3)
    c.update_and_fetch(k1, v1)
    assert c.tokens_kept <= budget
    kept_positions = c._kept_positions[0].tolist()

    h_fake = mx.zeros((1, 3, 4))
    mask = create_attention_mask(h_fake, c)
    assert isinstance(mask, mx.array)
    n_stored = c.tokens_kept
    assert mask.shape == (1, 1, 3, n_stored + 3)

    query_positions = [c.offset + i for i in range(3)]
    key_positions = kept_positions + [c.offset + i for i in range(3)]
    expected = [[kj <= qi for kj in key_positions] for qi in query_positions]
    assert mask[0, 0].tolist() == expected


def test_make_mask_single_query_returns_none() -> None:
    from mlx_lm.models.base import create_attention_mask

    cfg = KVCacheConfig(method="cam", head_dim=8, cam_budget=6, cam_n_sink=1)
    c = CaMKVCache(cfg)
    k, v = _kv(1, 1, 15, 8, seed=3)
    c.update_and_fetch(k, v)

    h_fake = mx.zeros((1, 1, 4))
    assert create_attention_mask(h_fake, c) is None


def test_kept_positions_valid_after_merge_with_merging_enabled():
    """With merging enabled (not drop mode) and the gate off (unconditional
    merge, maximizing how often merges actually happen), every tracked
    position must stay within [0, true_offset) and sinks must remain
    exactly [0, n_sink) — the merge-position semantics (survivor position
    becomes max(survivor, loser)) must never produce an out-of-range or
    sink-corrupting position."""
    budget, n_sink = 6, 2
    cfg = KVCacheConfig(
        method="cam",
        head_dim=8,
        cam_budget=budget,
        cam_n_sink=n_sink,
        cam_merge="sim_weighted",
        cam_merge_gate=False,  # unconditional merge — exercises the merge-position path every time
    )
    c = CaMKVCache(cfg)
    k, v = _kv(1, 1, 40, 8, seed=9)
    c.update_and_fetch(k, v)
    assert c.tokens_kept <= budget
    positions = c._kept_positions[0].tolist()
    assert positions[:n_sink] == list(range(n_sink))
    assert all(0 <= p < c.offset for p in positions)
    assert len(positions) == len(set(positions))  # no duplicate positions
