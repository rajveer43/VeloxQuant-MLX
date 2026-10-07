"""Tests for QJLKVCache.

Covers the standard append/attend/memory_bytes path plus a regression for
#82: once more tokens have been appended than the configured ``capacity``,
the RingBuffer-backed storage silently evicts the oldest entries, so
``attend()``/``memory_bytes()``/``__len__`` must use the buffer's actual
(capacity-capped) live size, not the unbounded lifetime append counter.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory


def _make(**cfg):
    base = {"method": "qjl", "head_dim": 8, "jl_dim": 8, "seed": 0}
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_vec(D: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(D).astype(np.float16))


def test_factory_dispatch() -> None:
    from veloxquant_mlx.cache.qjl_cache import QJLKVCache

    assert isinstance(_make(), QJLKVCache)


def test_append_and_attend_within_capacity() -> None:
    cache = _make(capacity=100)
    for i in range(10):
        cache.append_key(_rand_vec(seed=i))
        cache.append_value(_rand_vec(seed=100 + i))
    assert len(cache) == 10

    out = cache.attend(_rand_vec(seed=999))
    mx.eval(out)
    assert out.shape == (8,)


def test_empty_cache_attend_returns_zeros() -> None:
    cache = _make(capacity=10)
    out = cache.attend(_rand_vec())
    mx.eval(out)
    assert bool(mx.all(out == 0))


# ---------------------------------------------------------------------------
# Regression for #82
# ---------------------------------------------------------------------------


def test_attend_past_capacity_does_not_raise() -> None:
    """Regression for #82: attend() must not IndexError once more tokens
    have been appended than the ring buffer's capacity."""
    cache = _make(capacity=4)
    for i in range(6):
        cache.append_key(_rand_vec(seed=i))
        cache.append_value(_rand_vec(seed=100 + i))

    out = cache.attend(_rand_vec(seed=999))  # must not raise
    mx.eval(out)
    assert out.shape == (8,)


def test_len_capped_at_capacity() -> None:
    """len() must reflect the ring buffer's live (capped) size, not the
    unbounded lifetime append count."""
    cache = _make(capacity=4)
    for i in range(6):
        cache.append_key(_rand_vec(seed=i))
        cache.append_value(_rand_vec(seed=100 + i))
    assert len(cache) == 4


def test_memory_bytes_capped_at_capacity() -> None:
    """memory_bytes() must scale with the capped live size, not the
    unbounded lifetime append count (else it over-reports usage)."""
    cache = _make(capacity=4)
    for i in range(4):
        cache.append_key(_rand_vec(seed=i))
        cache.append_value(_rand_vec(seed=100 + i))
    bytes_at_capacity = cache.memory_bytes()

    for i in range(4, 6):
        cache.append_key(_rand_vec(seed=i))
        cache.append_value(_rand_vec(seed=100 + i))
    bytes_past_capacity = cache.memory_bytes()

    assert bytes_past_capacity == bytes_at_capacity, (
        "memory_bytes() must plateau once past capacity, not keep growing "
        "with the unbounded lifetime token count"
    )


def test_attend_correct_after_eviction() -> None:
    """After capacity overflow, attend() must only ever see the surviving
    (most recent `capacity`) tokens — no crash, and a deterministic result
    across repeated calls."""
    cache = _make(capacity=3)
    for i in range(5):
        cache.append_key(_rand_vec(seed=i))
        cache.append_value(_rand_vec(seed=100 + i))
    assert len(cache) == 3

    q = _rand_vec(seed=999)
    out1 = cache.attend(q)
    out2 = cache.attend(q)
    mx.eval(out1, out2)
    assert np.array_equal(np.array(out1), np.array(out2))


def test_reset_returns_to_empty_state() -> None:
    """Regression for #274: reset() (called by SlidingWindowKVCache on every
    window advance) must return the cache to a genuinely empty, reusable
    state — len 0, memory_bytes 0 — not merely a same-sized buffer whose
    old contents were never actually cleared."""
    cache = _make(capacity=10)
    for i in range(6):
        cache.append_key(_rand_vec(seed=i))
        cache.append_value(_rand_vec(seed=100 + i))
    assert len(cache) == 6

    cache.reset()
    assert len(cache) == 0
    assert cache.memory_bytes() == 0

    for i in range(3):
        cache.append_key(_rand_vec(seed=200 + i))
        cache.append_value(_rand_vec(seed=300 + i))
    assert len(cache) == 3
    out = cache.attend(_rand_vec(seed=999))
    mx.eval(out)
    assert out.shape == (8,)


def test_multi_row_append_raises_instead_of_dropping_rows():
    """#772: a (n, d) input used to keep only row 0 (keys) / flatten (values)."""
    import mlx.core as mx
    import pytest

    from veloxquant_mlx.cache.base import KVCacheConfig
    from veloxquant_mlx.cache.polar_cache import PolarQuantKVCache
    from veloxquant_mlx.cache.qjl_cache import QJLKVCache

    for cls in (QJLKVCache, PolarQuantKVCache):
        c = cls(KVCacheConfig(head_dim=64))
        bad = mx.random.normal((4, 64)).astype(mx.float16)
        with pytest.raises(ValueError, match="append_key"):
            c.append_key(bad)
        with pytest.raises(ValueError, match="append_value"):
            c.append_value(bad)
        assert len(c) == 0
        one = mx.random.normal((1, 64)).astype(mx.float16)  # (1, d) is still fine
        c.append_key(one)
        c.append_value(one)
        assert len(c) == 1
        assert c.attend(one[0]).shape == (64,)


def test_repr_reports_live_token_count_after_ring_buffer_wrap():
    """#774: repr used a monotonically growing counter and disagreed with len()."""
    import mlx.core as mx

    from veloxquant_mlx.cache.base import KVCacheConfig
    from veloxquant_mlx.cache.qjl_cache import QJLKVCache

    c = QJLKVCache(KVCacheConfig(head_dim=64, capacity=4))
    for _ in range(10):
        v = mx.random.normal((64,)).astype(mx.float16)
        c.append_key(v)
        c.append_value(v)
    assert len(c) == 4
    assert "n_tokens=4" in repr(c)
