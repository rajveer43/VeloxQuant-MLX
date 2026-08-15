"""Tests for TOVAKVCache — current-step attention-weight token eviction.

TOVA-adapted (arXiv:2401.06104) scores each token by the attention weight it
receives at the current step (memoryless — no accumulation) and evicts the
lowest-weight non-sink token whenever the cache exceeds tova_budget. Tests cover:
factory dispatch, interface attributes, output shape bounded by budget, output
dtype fp16, sink protection, decode budget enforcement across many steps, byte
accounting (compression_ratio, tova_kept_bytes), tokens_kept, n_sink=0 edge case,
determinism, and for_model config propagation. All data is synthetic.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.tova_cache import TOVAKVCache


def _make(**cfg):
    base = dict(method="tova", head_dim=32, tova_budget=8, tova_n_sink=2)
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
    assert isinstance(_make(), TOVAKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "compression_ratio")
    assert hasattr(c, "tokens_kept")
    assert hasattr(c, "tova_kept_bytes")


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


def test_output_shape_below_budget() -> None:
    """S < budget → all tokens returned."""
    c = _make(tova_budget=16, tova_n_sink=2)
    k, v = _rand_kv(S=6, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 6
    assert vo.shape[2] == 6


def test_output_shape_bounded_by_budget() -> None:
    """S > budget → output seq dim <= budget."""
    budget = 8
    c = _make(tova_budget=budget, tova_n_sink=2)
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
    c = _make(tova_budget=16, tova_n_sink=0)
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
    c = _make(tova_budget=budget, tova_n_sink=3)
    for i in range(30):
        k, v = _rand_kv(S=1, H=2, D=32, seed=i)
        ko, vo = c.update_and_fetch(k, v)
        assert ko.shape[2] <= budget, f"step {i}: seq={ko.shape[2]} > {budget}"


def test_tokens_kept_bounded_by_budget() -> None:
    budget = 8
    c = _make(tova_budget=budget, tova_n_sink=2)
    k, v = _rand_kv(S=20, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.tokens_kept <= budget


# ---------------------------------------------------------------------------
# Sink protection
# ---------------------------------------------------------------------------


def test_n_sink_zero_still_enforces_budget() -> None:
    """With n_sink=0, all tokens may be evicted; budget still respected."""
    budget = 4
    c = _make(tova_budget=budget, tova_n_sink=0)
    k, v = _rand_kv(S=20, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] <= budget


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_compression_ratio_equals_1_below_budget() -> None:
    """When tokens < budget, no eviction → ratio == 1."""
    c = _make(tova_budget=32, tova_n_sink=0)
    k, v = _rand_kv(S=4, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio == pytest.approx(1.0, rel=1e-3)


def test_compression_ratio_gt_1_after_evictions() -> None:
    """After many evictions, ratio > 1."""
    c = _make(tova_budget=8, tova_n_sink=2)
    k, v = _rand_kv(S=100, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio > 1.0


def test_tokens_seen_accumulates() -> None:
    """tokens_seen grows by B * H * S per call."""
    c = _make(tova_budget=32)
    k, v = _rand_kv(S=6, H=2, D=32)
    c.update_and_fetch(k, v)
    # B=1, H=2, S=6 → 12
    assert c.tokens_seen == 12


def test_tova_kept_bytes_positive_after_update() -> None:
    c = _make(tova_budget=16)
    k, v = _rand_kv(S=4, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.tova_kept_bytes > 0


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
        method="tova",
        head_dim=32,
        tova_budget=64,
        tova_n_sink=8,
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, TOVAKVCache) for c in caches)
    assert caches[0]._budget == 64
    assert caches[0]._n_sink == 8


# ==================================================================
# RoPE position bookkeeping (#171, #175)
# ==================================================================


def test_offset_tracks_true_position_after_eviction() -> None:
    """``cache.offset`` must be the true token position, not the retained count.

    mlx_lm rotates both the query and the incoming key at ``offset=cache.offset``
    *before* calling ``update_and_fetch``. Before this fix, ``self.offset`` was
    left at whatever the base ``KVCache.update_and_fetch`` set it to — the
    number of RETAINED rows — so once eviction pinned the kept count at the
    budget, the offset stopped advancing and every later token was rotated at
    the wrong (stale) position while its true position kept climbing. This
    reproduces that drift without the fix: without ``_true_offset`` tracking,
    ``cache.offset`` would stall at ``budget`` instead of tracking ``t + 1``.
    """
    budget = 8
    cache = _make(tova_budget=budget, tova_n_sink=2)

    n_steps = 5 * budget
    for t in range(n_steps):
        k, v = _rand_kv(S=1, H=2, D=32, seed=100 + t)
        cache.update_and_fetch(k, v)
        assert cache.offset == t + 1, (
            f"offset {cache.offset} != true position {t + 1} — RoPE would be wrong"
        )

    assert cache.tokens_kept <= budget


def test_offset_advances_by_block_size_on_prefill() -> None:
    """A multi-token block advances the offset by S, not by rows retained."""
    S, budget = 100, 16
    cache = _make(tova_budget=budget, tova_n_sink=2)

    k, v = _rand_kv(S=S, H=2, D=32, seed=7)
    cache.update_and_fetch(k, v)
    assert cache.offset == S
    assert cache.tokens_kept <= budget

    k2, v2 = _rand_kv(S=1, H=2, D=32, seed=8)
    cache.update_and_fetch(k2, v2)
    assert cache.offset == S + 1


def test_offset_survives_prefill_then_decode_mix() -> None:
    """Offset stays the true position across a prefill block followed by
    many decode steps, even though eviction is active throughout."""
    S, budget = 40, 12
    cache = _make(tova_budget=budget, tova_n_sink=2)

    k, v = _rand_kv(S=S, H=2, D=32, seed=9)
    cache.update_and_fetch(k, v)
    assert cache.offset == S

    for t in range(30):
        kd, vd = _rand_kv(S=1, H=2, D=32, seed=200 + t)
        cache.update_and_fetch(kd, vd)
        assert cache.offset == S + t + 1
    assert cache.tokens_kept <= budget
