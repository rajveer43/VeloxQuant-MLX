"""Tests for PolarQuantKVCache.

Covers the standard append/attend/memory_bytes path plus a regression for
#82: once more tokens have been appended than the configured ``capacity``,
the RingBuffer-backed storage silently evicts the oldest entries, so
``attend()``/``memory_bytes()``/``__len__`` must use the buffer's actual
(capacity-capped) live size, not the unbounded lifetime append counter.

head_dim must be divisible by 2**n_levels (16, per RecursivePolarTransform's
default depth) for PolarQuantizer.encode to accept it.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

_D = 16


def _make(**cfg):
    base = dict(method="polar", head_dim=_D, bit_width_inlier=2, seed=0)
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_vec(D: int = _D, seed: int = 0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(D).astype(np.float16))


def test_factory_dispatch() -> None:
    from veloxquant_mlx.cache.polar_cache import PolarQuantKVCache

    assert isinstance(_make(), PolarQuantKVCache)


def test_append_and_attend_within_capacity() -> None:
    cache = _make(capacity=100)
    for i in range(10):
        cache.append_key(_rand_vec(seed=i))
        cache.append_value(_rand_vec(seed=100 + i))
    assert len(cache) == 10

    out = cache.attend(_rand_vec(seed=999))
    mx.eval(out)
    assert out.shape == (_D,)


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
    assert out.shape == (_D,)


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
    unbounded lifetime append count (else it over-reports usage — the
    milder, non-crashing form of the same bug flagged in #82)."""
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
