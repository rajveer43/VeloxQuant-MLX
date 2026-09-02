"""Tests for AgeTieredKVCache — position/age-gated 3-tier precision (issue #256).

AgeTieredKV never evicts tokens: every token seen is retained, with its
bit-width set purely by ``age = current_position - token_position`` relative
to two configurable boundaries. Tests cover: factory dispatch, interface
attributes, output shape (always == tokens seen, unlike eviction methods),
output dtype fp16, age-boundary validation, tier assignment/observability,
byte accounting, determinism, re-tiering as tokens age across a boundary,
and for_model config propagation. All data is synthetic.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.age_tiered_cache import AgeTieredKVCache
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory


def _make(**cfg):
    base = dict(method="age_tiered", head_dim=32)
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_kv(S: int = 4, H: int = 2, D: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    V = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    return K, V


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_recent_boundary_zero_rejected() -> None:
    with pytest.raises(ValueError, match="age_recent_boundary"):
        _make(age_recent_boundary=0)


def test_recent_boundary_negative_rejected() -> None:
    with pytest.raises(ValueError, match="age_recent_boundary"):
        _make(age_recent_boundary=-5)


def test_mid_boundary_below_recent_boundary_rejected() -> None:
    with pytest.raises(ValueError, match="age_mid_boundary"):
        _make(age_recent_boundary=100, age_mid_boundary=50)


def test_mid_boundary_equal_recent_boundary_accepted() -> None:
    # age_mid_boundary == age_recent_boundary means the MID tier is empty
    # (every age either < boundary -> RECENT, or >= boundary -> OLD), which
    # is a degenerate-but-valid two-tier configuration, not an error.
    c = _make(age_recent_boundary=50, age_mid_boundary=50)
    assert isinstance(c, AgeTieredKVCache)


@pytest.mark.parametrize("field", ["age_bits_recent", "age_bits_mid", "age_bits_old"])
def test_bits_out_of_range_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _make(**{field: 0})
    with pytest.raises(ValueError, match=field):
        _make(**{field: 17})


# ---------------------------------------------------------------------------
# Factory and interface
# ---------------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), AgeTieredKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "compression_ratio")
    assert hasattr(c, "tokens_kept")
    assert hasattr(c, "age_tiered_bytes")
    assert hasattr(c, "tokens_recent")
    assert hasattr(c, "tokens_mid")
    assert hasattr(c, "tokens_old")


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
    c = _make()
    k, v = _rand_kv(S=10, H=2, D=32, seed=0)
    ko, _ = c.update_and_fetch(k, v)
    assert ko.shape[2] == 10
    for i in range(5):
        k, v = _rand_kv(S=1, H=2, D=32, seed=50 + i)
        ko, vo = c.update_and_fetch(k, v)
        assert ko.shape[2] == 10 + i + 1


def test_no_eviction_all_tokens_retained() -> None:
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
# Tier assignment
# ---------------------------------------------------------------------------


def test_short_sequence_is_entirely_recent() -> None:
    """No token can have aged past the recent boundary if fewer tokens than
    the boundary have ever been written."""
    c = _make(age_recent_boundary=128, age_mid_boundary=1024)
    k, v = _rand_kv(S=10, H=1, D=32)
    c.update_and_fetch(k, v)
    assert c.tokens_recent == 10
    assert c.tokens_mid == 0
    assert c.tokens_old == 0


def test_tier_counts_sum_to_tokens_seen() -> None:
    """tokens_recent/mid/old sum across all heads (like tokens_seen), while
    tokens_kept reports only the (b=0, h=0) head's count — see their
    respective docstrings."""
    c = _make(age_recent_boundary=4, age_mid_boundary=8)
    k, v = _rand_kv(S=20, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.tokens_recent + c.tokens_mid + c.tokens_old == c.tokens_seen


def test_tier_boundaries_partition_by_age() -> None:
    """With small boundaries and a longer sequence, all three tiers should
    be populated and match the expected counts by direct age arithmetic."""
    c = _make(age_recent_boundary=3, age_mid_boundary=6)
    k, v = _rand_kv(S=15, H=1, D=32)
    c.update_and_fetch(k, v)
    # positions 0..14 written, current_position == 15.
    # age[i] = 15 - (i+1); recent: age < 3 -> i in {12,13,14} (3 tokens)
    # mid: 3 <= age < 6 -> i in {9,10,11} (3 tokens); old: the rest (9 tokens)
    assert c.tokens_recent == 3
    assert c.tokens_mid == 3
    assert c.tokens_old == 9


def test_tokens_age_from_recent_to_mid_to_old_across_steps() -> None:
    """A token written early should transition RECENT -> MID -> OLD as more
    tokens are appended, purely from the accumulating position count."""
    c = _make(age_recent_boundary=2, age_mid_boundary=4)
    k, v = _rand_kv(S=1, H=1, D=32, seed=1)
    c.update_and_fetch(k, v)  # token 0 written; age(token0) == 0 -> RECENT
    assert c.tokens_recent == 1

    k, v = _rand_kv(S=1, H=1, D=32, seed=2)
    c.update_and_fetch(k, v)  # token0 age==1 (still RECENT), token1 age==0
    assert c.tokens_recent == 2

    k, v = _rand_kv(S=1, H=1, D=32, seed=3)
    c.update_and_fetch(k, v)  # token0 age==2 -> MID now
    assert c.tokens_mid == 1

    k, v = _rand_kv(S=1, H=1, D=32, seed=4)
    c.update_and_fetch(k, v)  # token0 age==3 -> still MID (< 4)
    assert c.tokens_mid >= 1

    k, v = _rand_kv(S=1, H=1, D=32, seed=5)
    c.update_and_fetch(k, v)  # token0 age==4 -> OLD now
    assert c.tokens_old >= 1


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_compression_ratio_gt_1_when_bits_below_16() -> None:
    c = _make(age_bits_recent=8, age_bits_mid=4, age_bits_old=2)
    k, v = _rand_kv(S=64, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio > 1.0


def test_compression_ratio_is_1_when_all_bits_16() -> None:
    c = _make(age_bits_recent=16, age_bits_mid=16, age_bits_old=16)
    k, v = _rand_kv(S=64, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio == pytest.approx(1.0, abs=0.05)


def test_tokens_seen_accumulates() -> None:
    c = _make()
    k, v = _rand_kv(S=6, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.tokens_seen == 12  # B=1, H=2, S=6


def test_age_tiered_bytes_positive_after_update() -> None:
    c = _make()
    k, v = _rand_kv(S=4, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.age_tiered_bytes > 0


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


def test_no_nans_across_multiple_steps() -> None:
    c = _make(age_recent_boundary=3, age_mid_boundary=6)
    for i in range(10):
        k, v = _rand_kv(S=1, H=2, D=32, seed=i)
        ko, vo = c.update_and_fetch(k, v)
        assert not bool(mx.any(mx.isnan(ko)).item())
        assert not bool(mx.any(mx.isnan(vo)).item())


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
        method="age_tiered",
        head_dim=32,
        age_recent_boundary=16,
        age_mid_boundary=64,
        age_bits_recent=8,
        age_bits_mid=4,
        age_bits_old=2,
        age_group_size=16,
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, AgeTieredKVCache) for c in caches)
    assert caches[0]._age_recent_boundary == 16
    assert caches[0]._age_mid_boundary == 64
    assert caches[0]._group_size == 16


def test_factory_smoke_compression_ratio_positive_both_kv() -> None:
    """End-to-end factory smoke test through both K and V."""
    c = _make()
    k, v = _rand_kv(S=64, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 64
    assert vo.shape[2] == 64
    assert c.compression_ratio > 1.0


# ---------------------------------------------------------------------------
# is_trimmable
# ---------------------------------------------------------------------------


def test_is_not_trimmable() -> None:
    c = _make()
    assert c.is_trimmable() is False
