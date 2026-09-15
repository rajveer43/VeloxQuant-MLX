"""Tests for SnapKVKVCache — prefill observation-window token eviction.

SnapKV-adapted retains only a budget of token positions from prefill (by
observation-window attention scoring) and always appends decode tokens. These
tests cover: factory dispatch, no .bits attribute, prefill output shape, decode
accumulation, output dtype, byte accounting, keep_rate, no-eviction short-seq
edge case, n_sink=0 edge case, decode-only path, determinism, and for_model
config propagation. All data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.snapkv_cache import SnapKVKVCache


def _make(**cfg):
    base = {
        "method": "snapkv",
        "head_dim": 128,
        "snap_budget": 16,
        "snap_obs_window": 8,
        "snap_n_sink": 2,
    }
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_kv(S: int = 64, H: int = 2, D: int = 128, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    V = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    return K, V


# ---------------------------------------------------------------------------
# Factory and interface
# ---------------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), SnapKVKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "eviction_ratio")
    assert hasattr(c, "keep_rate")


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


def test_prefill_output_shape_evicted() -> None:
    """After prefill, seq dim should be min(budget, S)."""
    c = _make(snap_budget=16, snap_obs_window=8, snap_n_sink=2)
    k, v = _rand_kv(S=64, H=2, D=128)
    ko, vo = c.update_and_fetch(k, v)
    # mlx_lm accumulates: seq dim after one prefill = min(budget, S) = 16
    assert ko.shape[2] == 16
    assert vo.shape[2] == 16


def test_output_dtype_fp16() -> None:
    c = _make()
    k, v = _rand_kv(S=64)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16


def test_no_eviction_short_seq() -> None:
    """budget >= S: all tokens kept, seq dim == S after prefill."""
    c = _make(snap_budget=200, snap_obs_window=4, snap_n_sink=2)
    k, v = _rand_kv(S=10, H=2, D=128)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 10
    assert c.keep_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Chunked prefill (#84)
# ---------------------------------------------------------------------------


def test_chunked_prefill_budget_stays_capped() -> None:
    """Regression for #84: mlx_lm's chunked prefill calls update_and_fetch
    once per prefill_step_size chunk. The retained token count must stay
    capped at snap_budget across multiple S>1 calls, not grow by up to
    budget per chunk."""
    c = _make(snap_budget=10, snap_obs_window=2, snap_n_sink=1)
    k1, v1 = _rand_kv(S=50, H=1, D=8, seed=1)
    k2, v2 = _rand_kv(S=50, H=1, D=8, seed=2)
    k3, v3 = _rand_kv(S=50, H=1, D=8, seed=3)

    ko1, _ = c.update_and_fetch(k1, v1)
    assert ko1.shape[2] == 10

    ko2, _ = c.update_and_fetch(k2, v2)
    assert ko2.shape[2] == 10, (
        f"budget violated after 2nd prefill chunk: kept {ko2.shape[2]} > snap_budget=10"
    )

    ko3, _ = c.update_and_fetch(k3, v3)
    assert ko3.shape[2] == 10, (
        f"budget violated after 3rd prefill chunk: kept {ko3.shape[2]} > snap_budget=10"
    )
    # Since #171 ``offset`` reports the TRUE absolute token position (what
    # mlx_lm rotates RoPE at), not the retained row count — 150 tokens were
    # seen even though only 10 rows survive.
    assert c.offset == 150


def test_chunked_prefill_then_decode_appends() -> None:
    """After multi-chunk prefill re-caps the budget, decode tokens (S==1)
    must still always append, never evict."""
    c = _make(snap_budget=10, snap_obs_window=2, snap_n_sink=1)
    k1, v1 = _rand_kv(S=50, H=1, D=8, seed=1)
    k2, v2 = _rand_kv(S=50, H=1, D=8, seed=2)
    c.update_and_fetch(k1, v1)
    c.update_and_fetch(k2, v2)
    # ``offset`` is the true absolute position since #171, not the row count:
    # 100 tokens seen, 10 rows retained.
    assert c.offset == 100

    for i in range(5):
        k1d, v1d = _rand_kv(S=1, H=1, D=8, seed=200 + i)
        ko, _ = c.update_and_fetch(k1d, v1d)
    assert ko.shape[2] == 15
    assert c.offset == 105


def test_chunked_prefill_sink_anchored_at_true_start() -> None:
    """The sink token planted at true sequence position 0 must still be
    protected after a later chunk re-runs eviction over the concatenated
    kept + new tokens — sink anchoring must not drift to the chunk-local
    position 0 of a later chunk."""
    c = _make(snap_budget=10, snap_obs_window=2, snap_n_sink=1, head_dim=8)
    rng = np.random.default_rng(7)
    k1 = rng.standard_normal((1, 1, 50, 8)).astype(np.float32)
    k1[:, :, 0, :] = 50.0  # true sequence-start sink token
    v1 = rng.standard_normal((1, 1, 50, 8)).astype(np.float32)
    k2 = rng.standard_normal((1, 1, 50, 8)).astype(np.float32)
    v2 = rng.standard_normal((1, 1, 50, 8)).astype(np.float32)

    c.update_and_fetch(mx.array(k1.astype(np.float16)), mx.array(v1.astype(np.float16)))
    ko2, _ = c.update_and_fetch(mx.array(k2.astype(np.float16)), mx.array(v2.astype(np.float16)))
    mx.eval(ko2)

    ko2_np = np.array(ko2).astype(np.float32)
    assert np.any(np.all(np.isclose(ko2_np[0, 0], k1[0, 0, 0, :], atol=1e-2), axis=-1)), (
        "true sequence-start sink token must still be retained after a later chunk"
    )


# ---------------------------------------------------------------------------
# Decode accumulation
# ---------------------------------------------------------------------------


def test_decode_accumulation() -> None:
    """Decode tokens grow the seq dim by 1 each call."""
    c = _make(snap_budget=16, snap_obs_window=8, snap_n_sink=2)
    k, v = _rand_kv(S=64)
    c.update_and_fetch(k, v)  # prefill → 16 tokens kept
    for i in range(4):
        k1, v1 = _rand_kv(S=1, seed=100 + i)
        ko, vo = c.update_and_fetch(k1, v1)
    assert ko.shape[2] == 16 + 4


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_byte_ordering_eviction_ratio_gt_1() -> None:
    """After prefill with budget < S, eviction_ratio > 1."""
    c = _make(snap_budget=16, snap_obs_window=8, snap_n_sink=2)
    k, v = _rand_kv(S=64)
    c.update_and_fetch(k, v)
    assert c.eviction_ratio > 1.0
    assert c.evicted_key_bytes < c.full_key_bytes


def test_keep_rate_in_range() -> None:
    c = _make(snap_budget=16, snap_obs_window=8, snap_n_sink=2)
    k, v = _rand_kv(S=64)
    c.update_and_fetch(k, v)
    assert 0.0 < c.keep_rate <= 1.0


def test_keep_rate_no_eviction() -> None:
    c = _make(snap_budget=200)
    k, v = _rand_kv(S=10)
    c.update_and_fetch(k, v)
    assert c.keep_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_n_sink_zero() -> None:
    """n_sink=0 runs without error."""
    c = _make(snap_n_sink=0, snap_budget=16)
    k, v = _rand_kv(S=64)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] == 16


def test_decode_only_no_eviction() -> None:
    """Single-token first call (S=1) is treated as decode — no eviction."""
    c = _make(snap_budget=8)
    k, v = _rand_kv(S=1)
    ko, vo = c.update_and_fetch(k, v)
    assert c.keep_rate == pytest.approx(1.0)
    assert ko.shape[2] == 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic() -> None:
    k, v = _rand_kv(S=64)
    c1, c2 = _make(), _make()
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
        head_dim = 128

    class _Layer:
        self_attn = _Attn()

    class _Model:
        layers = [_Layer(), _Layer()]

    cfg = KVCacheConfig(
        method="snapkv",
        head_dim=128,
        snap_budget=32,
        snap_obs_window=16,
        snap_n_sink=3,
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, SnapKVKVCache) for c in caches)
    assert caches[0]._budget == 32
    assert caches[0]._obs_window == 16
    assert caches[0]._n_sink == 3


def test_offset_tracks_true_position_not_retained_rows() -> None:
    """Regression for #171: ``offset`` must report the true absolute token
    position, since mlx_lm rotates RoPE at ``offset=cache.offset`` before
    update_and_fetch runs.

    Before the fix, offset carried the RETAINED ROW COUNT, so after prefill
    compression dropped tokens it lagged the true position by exactly the
    number evicted — every subsequent token was rotated at the wrong
    position, and the error persisted for the rest of the sequence.
    """
    c = _make(snap_budget=16, snap_obs_window=4, snap_n_sink=2)
    k, v = _rand_kv(S=64, H=1, D=8, seed=7)
    ko, _ = c.update_and_fetch(k, v)

    # Compression really did drop rows — otherwise this test proves nothing.
    assert ko.shape[2] == 16
    assert c.offset == 64, "offset must be the true position, not the row count"

    # Each decode token advances the position by exactly one, with no drift
    # accumulating relative to the retained row count.
    for i in range(20):
        k1, v1 = _rand_kv(S=1, H=1, D=8, seed=300 + i)
        c.update_and_fetch(k1, v1)
        assert c.offset == 64 + i + 1, f"position drift at decode step {i}"


# ---------------------------------------------------------------------------
# Batching + trim guards (issue #27)
# ---------------------------------------------------------------------------


def test_not_batchable_via_mlx_lm_server_probe() -> None:
    c = _make()
    assert not hasattr(c, "merge")
    with pytest.raises(AttributeError):
        c.merge


def test_merge_on_empty_cache_would_silently_substitute_if_inherited() -> None:
    from mlx_lm.models.cache import BatchKVCache
    from mlx_lm.models.cache import KVCache as _MLXKVCache

    c = _make()
    assert c.size() == 0
    merged = _MLXKVCache.merge.__func__(SnapKVKVCache, [c])
    assert isinstance(merged, BatchKVCache)
    assert not isinstance(merged, SnapKVKVCache)


def test_is_trimmable_false() -> None:
    c = _make()
    assert c.is_trimmable() is False


def test_trim_would_return_garbage_rows_if_trimmable() -> None:
    """Reproduces the corruption directly: trim() reads ``offset`` (the true
    absolute position) to clamp ``n``, then writes it back through the
    property setter, which stores it as the RETAINED ROW COUNT instead —
    leaving ``_row_offset`` larger than the number of rows ever written as
    soon as eviction has dropped anything, and the next update_and_fetch
    returns a slice reaching into stale buffer rows.
    """
    c = _make(snap_budget=4, snap_obs_window=2, snap_n_sink=1)
    k, v = _rand_kv(S=20, H=2, D=8, seed=1)
    c.update_and_fetch(k, v)

    assert c._true_offset == 20
    assert c._row_offset == 4  # eviction kept only the budget

    n = c.trim(3)  # base-class KVCache.trim: n = min(self.offset, 3)
    assert n == 3
    # Corrupted: only 4 rows were ever real, but row_offset is now 17.
    assert c._row_offset == 17
    assert c._row_offset > 4

    k2, v2 = _rand_kv(S=1, H=2, D=8, seed=99)
    out_k, _ = c.update_and_fetch(k2, v2)
    # The returned slice reaches past the 4 real rows into uninitialized
    # buffer space -- silent corruption, not a crash.
    assert out_k.shape[2] == 18
    assert out_k.shape[2] > 5
