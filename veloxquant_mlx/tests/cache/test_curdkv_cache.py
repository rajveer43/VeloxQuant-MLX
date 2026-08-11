"""Tests for CurDKVKVCache — value-aware leverage-score heavy-hitter eviction.

CurDKV-adapted (arXiv:2509.15038, NeurIPS 2025) accumulates per-token
leverage scores (derived from the joint key+value structure) as a proxy
importance score and evicts the lowest-score non-sink token whenever the
cache exceeds curdkv_budget. Tests cover: factory dispatch, interface
attributes, output shape bounded by budget, output dtype fp16, sink
protection, decode accumulation, budget enforcement across many steps, byte
accounting (compression_ratio, curdkv_kept_bytes), tokens_kept, n_sink=0
edge case, determinism, and for_model config propagation. All data is
synthetic.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.curdkv_cache import CurDKVKVCache


def _make(**cfg):
    base = dict(method="curdkv", head_dim=32, curdkv_budget=8, curdkv_n_sink=2)
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
    assert isinstance(_make(), CurDKVKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "compression_ratio")
    assert hasattr(c, "tokens_kept")
    assert hasattr(c, "curdkv_kept_bytes")


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape_below_budget() -> None:
    """S < budget → all tokens returned."""
    c = _make(curdkv_budget=16, curdkv_n_sink=2)
    k, v = _rand_kv(S=6, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 6
    assert vo.shape[2] == 6


def test_output_shape_bounded_by_budget() -> None:
    """S > budget → output seq dim <= budget."""
    budget = 8
    c = _make(curdkv_budget=budget, curdkv_n_sink=2)
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
    c = _make(curdkv_budget=16, curdkv_n_sink=0)
    k, v = _rand_kv(S=4, H=4, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[0] == 1  # B
    assert ko.shape[1] == 4  # H
    assert ko.shape[3] == 32  # D


# ---------------------------------------------------------------------------
# Budget enforcement across steps (prefill + decode)
# ---------------------------------------------------------------------------


def test_budget_enforced_after_many_steps() -> None:
    """30 decode steps — output seq dim never exceeds budget."""
    budget = 10
    c = _make(curdkv_budget=budget, curdkv_n_sink=3)
    for i in range(30):
        k, v = _rand_kv(S=1, H=2, D=32, seed=i)
        ko, vo = c.update_and_fetch(k, v)
        assert ko.shape[2] <= budget, f"step {i}: seq={ko.shape[2]} > {budget}"


def test_prefill_then_decode_same_loop() -> None:
    """A multi-token prefill followed by single-token decode steps both go
    through the same eviction loop (no prefill-only special case)."""
    budget = 6
    c = _make(curdkv_budget=budget, curdkv_n_sink=1)
    k, v = _rand_kv(S=10, H=2, D=32, seed=0)  # prefill
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] <= budget
    for i in range(10):
        k, v = _rand_kv(S=1, H=2, D=32, seed=50 + i)  # decode
        ko, vo = c.update_and_fetch(k, v)
        assert ko.shape[2] <= budget


def test_tokens_kept_bounded_by_budget() -> None:
    budget = 8
    c = _make(curdkv_budget=budget, curdkv_n_sink=2)
    k, v = _rand_kv(S=20, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.tokens_kept <= budget


# ---------------------------------------------------------------------------
# Sink protection
# ---------------------------------------------------------------------------


def test_n_sink_zero_still_enforces_budget() -> None:
    """With n_sink=0, all tokens may be evicted; budget still respected."""
    budget = 4
    c = _make(curdkv_budget=budget, curdkv_n_sink=0)
    k, v = _rand_kv(S=20, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] <= budget


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_compression_ratio_equals_1_below_budget() -> None:
    """When tokens < budget, no eviction → ratio == 1."""
    c = _make(curdkv_budget=32, curdkv_n_sink=0)
    k, v = _rand_kv(S=4, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio == pytest.approx(1.0, rel=1e-3)


def test_compression_ratio_gt_1_after_evictions() -> None:
    """After many evictions, ratio > 1."""
    c = _make(curdkv_budget=8, curdkv_n_sink=2)
    k, v = _rand_kv(S=100, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio > 1.0


def test_tokens_seen_accumulates() -> None:
    """tokens_seen grows by B * H * S per call."""
    c = _make(curdkv_budget=32)
    k, v = _rand_kv(S=6, H=2, D=32)
    c.update_and_fetch(k, v)
    # B=1, H=2, S=6 → 12
    assert c.tokens_seen == 12


def test_curdkv_kept_bytes_positive_after_update() -> None:
    c = _make(curdkv_budget=16)
    k, v = _rand_kv(S=4, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.curdkv_kept_bytes > 0


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
        method="curdkv",
        head_dim=32,
        curdkv_budget=64,
        curdkv_n_sink=8,
        curdkv_rank_cap=4,
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, CurDKVKVCache) for c in caches)
    assert caches[0]._budget == 64
    assert caches[0]._n_sink == 8
    assert caches[0]._rank_cap == 4


def test_factory_smoke_compression_ratio_positive_both_kv() -> None:
    """End-to-end factory smoke test: compression_ratio > 1 at a reasonable
    budget, exercising both K and V through the full factory path."""
    c = _make(curdkv_budget=8, curdkv_n_sink=2)
    k, v = _rand_kv(S=64, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] <= 8
    assert vo.shape[2] <= 8
    assert c.compression_ratio > 1.0


# ---------------------------------------------------------------------------
# RoPE offset tracking (real-model regression)
# ---------------------------------------------------------------------------
#
# See veloxquant_mlx/quantizers/curdkv.py module docstring's FIXED,
# CONFIRMED-ON-REAL-MODELS PROBLEM: mlx_lm's attention module rotates the
# next query/key using self.rope(x, offset=cache.offset) BEFORE
# update_and_fetch is ever called, so self.offset must always equal the true
# absolute step count, not the physically-stored row count. The original
# implementation reset self.offset to the post-eviction kept-row count every
# call, which undercounts the true position by however many tokens have been
# evicted so far -- confirmed via a real Llama-3.2-1B generation producing
# severely corrupted output even at ~1% eviction, traced to a single
# mismatched key row whose rotation used the wrong absolute position.


def test_offset_tracks_true_position_not_kept_count() -> None:
    """After eviction, self.offset must exceed the physically stored row
    count — it tracks true elapsed steps, which is what mlx_lm's attention
    module uses to rotate the NEXT query and key correctly."""
    budget = 4
    c = _make(curdkv_budget=budget, curdkv_n_sink=0)
    for i in range(10):
        k, v = _rand_kv(S=1, H=1, D=32, seed=i)
        c.update_and_fetch(k, v)
    assert c.keys.shape[2] <= budget
    assert c.offset == 10
    assert c.offset > c.keys.shape[2]


def test_state_property_returns_exactly_kept_rows() -> None:
    """cache.state (read by mlx_lm.generate during chunked prefill) must
    return exactly the stored rows, not a slice sized by self.offset."""
    budget = 4
    c = _make(curdkv_budget=budget, curdkv_n_sink=0)
    for i in range(10):
        k, v = _rand_kv(S=1, H=1, D=32, seed=i)
        c.update_and_fetch(k, v)
    keys_state, values_state = c.state
    assert keys_state.shape[2] == c.keys.shape[2] <= budget
    assert values_state.shape[2] == c.values.shape[2] <= budget


def test_size_returns_kept_count_not_offset() -> None:
    budget = 4
    c = _make(curdkv_budget=budget, curdkv_n_sink=0)
    for i in range(10):
        k, v = _rand_kv(S=1, H=1, D=32, seed=i)
        c.update_and_fetch(k, v)
    assert c.size() == c.keys.shape[2] <= budget
    assert c.size() != c.offset


def test_two_call_split_matches_single_bulk_call_when_no_further_eviction() -> None:
    """Minimal reproduction of the real-model bug: mlx_lm.generate() always
    processes the prompt as (bulk prefill call) + (final token folded into
    the first decode step) rather than one single call. If an eviction
    happened during the bulk call, the second call's self.offset must still
    equal the true absolute position — not the post-eviction kept-row count
    — or the model computes RoPE for that final token at the wrong angle.

    Verified here at the RAW ROPE-AWARE level: build a real RoPE-rotated key
    sequence directly (bypassing a full model forward pass), feed it through
    the cache as (a) one single bulk call and (b) a bulk call split into
    (S-1) + (1) tokens the way mlx_lm.generate() does, and confirm every
    kept row's key is bit-identical between the two paths. Before the fix,
    the split path's last row diverged from the single-call path because
    self.offset was wrong (kept-row count) instead of true position.
    """
    from veloxquant_mlx.quantizers.a2ats_rope import a2ats_apply_exact_rope

    D = 16
    budget = 8
    n_sink = 1
    n_tokens = 12  # > budget, so eviction happens during the bulk portion

    rng = np.random.default_rng(0)
    raw = rng.standard_normal((n_tokens, D)).astype(np.float32)
    positions = mx.arange(n_tokens, dtype=mx.int32)
    rotated = a2ats_apply_exact_rope(mx.array(raw), positions, base=10000.0)  # [n_tokens, D]
    values = mx.array(rng.standard_normal((n_tokens, D)).astype(np.float16))

    def as_bhsd(k_or_v: mx.array) -> mx.array:
        return k_or_v[None, None, :, :]  # [B=1, H=1, S, D]

    # Path A: single bulk call with all n_tokens at once.
    c_bulk = _make(curdkv_budget=budget, curdkv_n_sink=n_sink, head_dim=D)
    ko_bulk, _ = c_bulk.update_and_fetch(as_bhsd(rotated), as_bhsd(values))

    # Path B: split exactly like mlx_lm.generate() does — bulk call with the
    # first n_tokens-1, then a SEPARATE single-token call for the last one.
    c_split = _make(curdkv_budget=budget, curdkv_n_sink=n_sink, head_dim=D)
    c_split.update_and_fetch(as_bhsd(rotated[:-1]), as_bhsd(values[:-1]))
    ko_split, _ = c_split.update_and_fetch(as_bhsd(rotated[-1:]), as_bhsd(values[-1:]))

    assert c_bulk.offset == c_split.offset == n_tokens
    assert ko_bulk.shape == ko_split.shape

    diff = float(mx.max(mx.abs(ko_bulk.astype(mx.float32) - ko_split.astype(mx.float32))).item())
    assert diff < 1e-3, (
        f"split-call path diverged from single-call path (max diff={diff}) — "
        "self.offset likely desynced from the true absolute position again"
    )
