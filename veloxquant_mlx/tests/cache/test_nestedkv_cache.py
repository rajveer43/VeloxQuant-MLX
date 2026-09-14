"""Tests for NestedKVKVCache — multi-scale ensembled prefill eviction.

NestedKV-adapted (arXiv:2605.26678, no verified peer-reviewed venue as of
2026-07-14 — a one-time, user-directed exception) compresses the KV cache
ONCE at the end of prefill: each head is scored by three parallel key-only
continuum-memory anomaly signals, combined via a head-adaptive blend and a
per-token surprise gate, then independently keeps its own top-
nestedkv_budget scoring tokens — UNIFORM per-head budget, like H2O/CurDKV
(see #21: the paper's cross-head budget competition produced ragged
per-head lengths that were zero-padded to stack into one tensor, but that
padding became real unmasked attention entries mlx_lm's shared-mask-per-
model architecture cannot correctly exclude per layer, so this class no
longer uses it). Decode tokens are always appended, unscored — the cache
is NOT bounded during decode, unlike H2O/CurDKV.
test_decode_growth_unbounded_past_prefill_budget and
test_prefill_budget_bounds_only_prefill_output prove this directly. Tests
also cover: factory dispatch, interface attributes, shape/dtype, uniform
per-head budget reaching every head, byte accounting, determinism, and
for_model config propagation. All data is synthetic.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.nestedkv_cache import NestedKVKVCache


def _make(**cfg):
    base = {"method": "nestedkv", "head_dim": 32, "nestedkv_budget": 8, "nestedkv_n_sink": 2}
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
    assert isinstance(_make(), NestedKVKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "compression_ratio")
    assert hasattr(c, "tokens_kept")
    assert hasattr(c, "nestedkv_kept_bytes")


# ---------------------------------------------------------------------------
# Shape and dtype — prefill compression
# ---------------------------------------------------------------------------


def test_output_shape_below_budget() -> None:
    """S < budget at prefill -> all tokens returned."""
    c = _make(nestedkv_budget=16, nestedkv_n_sink=2)
    k, v = _rand_kv(S=6, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 6
    assert vo.shape[2] == 6


def test_output_shape_bounded_by_budget_at_prefill() -> None:
    """S > budget at prefill -> output seq dim == total layer budget (budget * H
    tokens allocated across H heads, but each head's own output is <= its
    allocated share; check the aggregate per-head cap loosely via <= S)."""
    budget = 8
    c = _make(nestedkv_budget=budget, nestedkv_n_sink=2)
    k, v = _rand_kv(S=30, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] <= 30
    # Each head individually should be compressed well below the original S.
    assert ko.shape[2] < 30


def test_output_dtype_fp16() -> None:
    c = _make()
    k, v = _rand_kv(S=4)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16


def test_output_batch_head_dims_preserved() -> None:
    c = _make(nestedkv_budget=16, nestedkv_n_sink=0)
    k, v = _rand_kv(S=20, H=4, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[0] == 1  # B
    assert ko.shape[1] == 4  # H
    assert ko.shape[3] == 32  # D


# ---------------------------------------------------------------------------
# One-shot prefill / decode-append design — the core proof
# ---------------------------------------------------------------------------


def test_prefill_budget_bounds_only_prefill_output() -> None:
    """The prefill call's output size should be roughly bounded by the total
    layer budget (budget * H), not by S."""
    budget = 6
    H = 2
    c = _make(nestedkv_budget=budget, nestedkv_n_sink=1)
    k, v = _rand_kv(S=40, H=H, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] <= budget + 2  # small slack for rounding in per-head allocation


def test_decode_growth_unbounded_past_prefill_budget() -> None:
    """After prefill compression, appending many decode tokens must grow the
    cache PAST the prefill budget — proving decode tokens are never rescored
    or re-evicted (the paper's one-shot design, Appendix A). The per-head
    prefill size can legitimately exceed the nominal per-head `budget` for
    a single, concentrated-score head (the cross-head competition can award
    it more than its equal share — see test_budget_allocation_favors_high_score_head
    at the quantizer level) — the real invariant is the padded head dim
    growing by exactly n_decode_tokens after prefill, not a tight per-head cap."""
    budget = 5
    H = 2
    c = _make(nestedkv_budget=budget, nestedkv_n_sink=1)
    k, v = _rand_kv(S=30, H=H, D=32, seed=0)  # prefill
    ko, vo = c.update_and_fetch(k, v)
    prefill_size = ko.shape[2]
    assert prefill_size <= budget * H  # total layer budget is the real cap

    for i in range(15):
        k, v = _rand_kv(S=1, H=H, D=32, seed=50 + i)  # decode
        ko, vo = c.update_and_fetch(k, v)

    assert ko.shape[2] == prefill_size + 15, (
        "decode tokens must be appended unconditionally, growing the cache past the prefill budget"
    )


def test_tokens_kept_bounded_at_prefill_only() -> None:
    budget = 8
    H = 2
    c = _make(nestedkv_budget=budget, nestedkv_n_sink=2)
    k, v = _rand_kv(S=30, H=H, D=32)
    c.update_and_fetch(k, v)
    assert c.tokens_kept <= budget * H  # total layer budget is the real cap


# ---------------------------------------------------------------------------
# Cross-head budget competition reaches every head
# ---------------------------------------------------------------------------


def test_all_heads_receive_nonzero_output_after_prefill() -> None:
    """Every head keeps its own uniform nestedkv_budget after prefill
    compression (see #21), even under an aggressive budget."""
    c = _make(nestedkv_budget=4, nestedkv_n_sink=0, nestedkv_safeguard_alpha=0.20)
    k, v = _rand_kv(S=40, H=4, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[1] == 4  # H dim intact
    assert ko.shape[2] > 0


# ---------------------------------------------------------------------------
# Sink protection
# ---------------------------------------------------------------------------


def test_n_sink_zero_still_compresses() -> None:
    """With n_sink=0, prefill compression still runs and bounds output."""
    budget = 4
    c = _make(nestedkv_budget=budget, nestedkv_n_sink=0)
    k, v = _rand_kv(S=30, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] <= 30


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_compression_ratio_equals_1_below_budget() -> None:
    c = _make(nestedkv_budget=32, nestedkv_n_sink=0)
    k, v = _rand_kv(S=4, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio == pytest.approx(1.0, rel=1e-3)


def test_compression_ratio_gt_1_after_prefill_eviction() -> None:
    c = _make(nestedkv_budget=8, nestedkv_n_sink=2)
    k, v = _rand_kv(S=100, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio > 1.0


def test_tokens_seen_accumulates() -> None:
    c = _make(nestedkv_budget=32)
    k, v = _rand_kv(S=6, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.tokens_seen == 12  # B=1, H=2, S=6


def test_nestedkv_kept_bytes_positive_after_update() -> None:
    c = _make(nestedkv_budget=16)
    k, v = _rand_kv(S=20, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.nestedkv_kept_bytes > 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic() -> None:
    k, v = _rand_kv(S=20, H=2, D=32)
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
        method="nestedkv",
        head_dim=32,
        nestedkv_budget=64,
        nestedkv_n_sink=8,
        nestedkv_window=32,
        nestedkv_beta=2.5,
        nestedkv_tau=0.5,
        nestedkv_kappa=8.0,
        nestedkv_safeguard_alpha=0.15,
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, NestedKVKVCache) for c in caches)
    assert caches[0]._budget == 64
    assert caches[0]._n_sink == 8
    assert caches[0]._window == 32
    assert caches[0]._beta == 2.5
    assert caches[0]._tau == 0.5
    assert caches[0]._kappa == 8.0
    assert caches[0]._safeguard_alpha == 0.15


def test_factory_smoke_compression_ratio_positive_both_kv() -> None:
    """End-to-end factory smoke test: compression_ratio > 1 at a reasonable
    budget, exercising both K and V through the full factory path."""
    c = _make(nestedkv_budget=8, nestedkv_n_sink=2)
    k, v = _rand_kv(S=64, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] <= 30
    assert vo.shape[2] <= 30
    assert c.compression_ratio > 1.0


# ---------------------------------------------------------------------------
# Regression for VeloxQuant-Studio issue #21: uniform per-head budget /
# no fake zero-padding rows leaking into attention.
# ---------------------------------------------------------------------------
def test_prefill_keeps_every_head_at_uniform_length() -> None:
    """Regression for issue #21. An earlier version let NestedKV's
    cross-head budget competition give heads different kept lengths, then
    zero-padded shorter heads (at the front) purely to stack them into one
    tensor. Those zero rows were never masked out of attention, so they
    silently corrupted output for any head that got less than the layer's
    max share -- reproduced directly against unpadded per-head state:
    heads legitimately came back with lengths like [30, 31, 38, 39, 30, 40,
    16, 32] out of a shared max of 40. This class now keeps every head at
    the same nestedkv_budget so no padding is ever constructed. Uses
    per-head score profiles specifically shaped to have driven ragged
    allocation under the old cross-head competition.
    """
    H, S, D, budget = 6, 60, 16, 10
    rng = np.random.default_rng(3)
    # Deliberately different per-head magnitude/variance profiles -- the
    # kind of spread that used to produce a wide head_budgets range under
    # nestedkv_allocate_head_budgets's cross-head competition.
    scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    K = np.stack(
        [rng.standard_normal((S, D)).astype(np.float32) * s for s in scales], axis=0
    )[None]
    V = np.stack(
        [rng.standard_normal((S, D)).astype(np.float32) * s for s in scales], axis=0
    )[None]
    c = _make(nestedkv_budget=budget, nestedkv_n_sink=2)
    ko, vo = c.update_and_fetch(mx.array(K.astype(np.float16)), mx.array(V.astype(np.float16)))

    assert ko.shape == (1, H, budget, D)
    assert vo.shape == (1, H, budget, D)
    lens = [int(st.keys.shape[0]) for st in c._states]
    assert lens == [budget] * H

    # No spurious all-zero rows anywhere (the old padded rows were exactly
    # this: real-looking cache content that was actually fake).
    Knp = np.array(ko)
    for h in range(H):
        zero_rows = np.all(Knp[0, h] == 0, axis=-1)
        assert not zero_rows.any(), f"head {h} has {zero_rows.sum()} spurious zero rows"


def test_decode_stays_uniform_length_after_prefill() -> None:
    """Decode append must also stay uniform across heads (see #21):
    every head enters decode at the same prefill length and grows by the
    same S each call."""
    c = _make(nestedkv_budget=6, nestedkv_n_sink=1)
    k0, v0 = _rand_kv(S=20, H=3, D=32, seed=5)
    c.update_and_fetch(k0, v0)
    for i in range(4):
        k1, v1 = _rand_kv(S=1, H=3, D=32, seed=100 + i)
        ko, vo = c.update_and_fetch(k1, v1)
    lens = [int(st.keys.shape[0]) for st in c._states]
    assert len(set(lens)) == 1
    assert ko.shape[2] == lens[0]


def test_not_batchable_via_mlx_lm_server_probe() -> None:
    """`mlx_lm.server`'s `ModelProvider.load()` decides whether a method is
    batchable purely via `hasattr(cache, "merge")` on a probe instance. The
    base `KVCache` this inherits from defines `merge()` as a classmethod
    returning a plain `BatchKVCache` — oblivious to this class's
    multi-scale ensembled eviction state. Left inherited, every request
    (even a lone one — `BatchGenerator` merges a batch of 1 too) would
    silently replace this cache with that generic one: no eviction, no
    sink protection, unlimited growth, while the server still believes it
    is running `nestedkv`. `hasattr` must see `merge` as absent so the
    server routes `nestedkv` through its sequential path instead, where
    this class runs correctly.
    """
    c = _make()
    assert not hasattr(c, "merge")
    with pytest.raises(AttributeError):
        c.merge
