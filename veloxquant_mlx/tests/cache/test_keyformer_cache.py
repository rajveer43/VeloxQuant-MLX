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
    assert cache._recent == 0 and cache._seed == 0
    assert cache._tau_init == 1.0 and cache._tau_end == 1.0 and cache._anneal_steps == 0
    assert cache._rope_base == 10000.0


def test_keyformer_tau_alias_overrides_init_end():
    cache = _make(keyformer_tau=3.0, keyformer_tau_init=1.0, keyformer_tau_end=2.0)
    assert cache._tau_init == 3.0 and cache._tau_end == 3.0


def test_tau_init_end_annealing_propagates():
    cache = _make(keyformer_tau_init=1.0, keyformer_tau_end=2.0, keyformer_anneal_steps=50)
    assert cache._tau_init == 1.0 and cache._tau_end == 2.0 and cache._anneal_steps == 50


def test_rope_base_propagates():
    cache = _make(keyformer_rope_base=500000.0)
    assert cache._rope_base == 500000.0


# ---------------------------------------------------------------------------
# budget invariant across shapes
# ---------------------------------------------------------------------------
def test_budget_respected_decode():
    """The STORED state (cache.keys) is capped at budget every step. The
    per-call RETURN value can exceed budget by up to S rows right after an
    eviction (#610, deferred-eviction pattern) — here S=1, so at most
    budget+1 — since it must match the shape mlx_lm's mask already assumed."""
    cache = _make(keyformer_budget=12, keyformer_n_sink=2)
    for i in range(40):
        k, v = _kv(1, 3, 1, 32, seed=i)
        K, V = cache.update_and_fetch(k, v)
        assert K.shape[2] <= 13 and V.shape[2] <= 13
        assert cache.keys.shape[2] <= 12
        assert K.shape[:2] == (1, 3) and K.shape[3] == 32


def test_budget_respected_prefill_block():
    """update_and_fetch's RETURN value for a multi-token call is the full
    pre-eviction concatenation (#610, matching the #370 pattern) — the mask
    mlx_lm built for this call already assumes all 50 keys are present.
    What gets STORED (and returned to the NEXT call) is capped at budget."""
    cache = _make(keyformer_budget=12, keyformer_n_sink=2)
    k, v = _kv(2, 4, 50, 32, seed=1)
    K, V = cache.update_and_fetch(k, v)
    assert K.shape == (2, 4, 50, 32)
    assert cache.keys.shape == (2, 4, 12, 32)


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
    """The per-call RETURN value is the full pre-eviction set (#610); the
    STORED state (cache.keys) is what stays capped at budget in both modes."""
    k_all, v_all = _kv(1, 2, 40, 24, seed=9)

    pf = _make(keyformer_budget=10, keyformer_n_sink=2, keyformer_tau=1.0)
    pf.update_and_fetch(k_all, v_all)

    dc = _make(keyformer_budget=10, keyformer_n_sink=2, keyformer_tau=1.0)
    for t in range(40):
        dc.update_and_fetch(k_all[:, :, t : t + 1], v_all[:, :, t : t + 1])

    assert pf.keys.shape[2] <= 10 and dc.keys.shape[2] <= 10


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


# ---------------------------------------------------------------------------
# temperature annealing propagates and evolves per-head at the cache level
# ---------------------------------------------------------------------------
def test_annealing_changes_kept_set_vs_constant_tau():
    ks = [_kv(1, 2, 1, 16, seed=i) for i in range(60)]

    def run(**kw):
        cache = _make(keyformer_budget=10, keyformer_n_sink=2, keyformer_seed=0, **kw)
        for k, v in ks:
            K, _ = cache.update_and_fetch(k, v)
        return K

    constant = run(keyformer_tau=1.0)
    annealed = run(keyformer_tau_init=1.0, keyformer_tau_end=6.0, keyformer_anneal_steps=30)
    # Different effective temperature schedule -> different eviction decisions
    # across 60 steps is expected (not a bit-for-bit equivalence claim).
    assert constant.shape == annealed.shape
    assert not bool(mx.all(constant == annealed).item())


# ---------------------------------------------------------------------------
# #610 regression: prefill eviction must not desync the inherited "causal"
# mask from the returned key count
# ---------------------------------------------------------------------------
def test_prefill_eviction_matches_uncompressed_reference_at_position_zero():
    """End-to-end regression for #610: position 0's logits depend only on
    token 0 under any causal attention in which token 0 is kept (it is a
    sink here), so a single-prefill-call run through KeyformerKVCache must
    produce IDENTICAL position-0 logits to a plain, uncompressed KVCache —
    regardless of how the rest of the 32-token prompt is evicted. Before the
    fix, eviction inside the prefill call shrank the returned key count
    below N while mlx_lm's mask (already fixed for this call as the
    "causal" string) still assumed N keys, corrupting every query's
    attention including position 0 (max error ~3.18 on this exact repro,
    matching the issue's own reported number)."""
    from mlx_lm.models import llama
    from mlx_lm.models.cache import KVCache as PlainKVCache

    from veloxquant_mlx.cache.base import KVCacheBuilder

    mx.random.seed(0)
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=128,
        num_hidden_layers=2,
        intermediate_size=256,
        num_attention_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=100,
        head_dim=32,
        num_key_value_heads=2,
    )
    model = llama.Model(args)
    model.set_dtype(mx.float16)
    P = 32
    tokens = mx.random.randint(0, 100, (1, P))
    ref = model(tokens, cache=[PlainKVCache() for _ in range(2)])

    cfg = KVCacheConfig(method="keyformer", head_dim=32, keyformer_budget=16, keyformer_n_sink=2)
    caches = KVCacheBuilder.for_model(model, cfg)
    out = model(tokens, cache=caches)

    d0 = float(mx.abs(out[0, 0].astype(mx.float32) - ref[0, 0].astype(mx.float32)).max())
    assert d0 < 1e-3, f"position-0 logits diverged from the uncompressed reference by {d0}"


def test_annealing_is_reproducible_per_head():
    ks = [_kv(1, 3, 1, 16, seed=i) for i in range(40)]

    def run():
        cache = _make(
            keyformer_budget=8,
            keyformer_n_sink=1,
            keyformer_tau_init=1.0,
            keyformer_tau_end=3.0,
            keyformer_anneal_steps=20,
            keyformer_seed=5,
        )
        for k, v in ks:
            K, _ = cache.update_and_fetch(k, v)
        return K

    assert bool(mx.all(run() == run()).item())
