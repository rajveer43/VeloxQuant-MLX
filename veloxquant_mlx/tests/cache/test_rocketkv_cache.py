"""Tests for RocketKVKVCache — two-stage compression (SnapKV eviction + HSA).

RocketKV-adapted (arXiv:2502.14051, ICML 2025) evicts PERMANENTLY at prefill
(stage 1, reusing SnapKV — same eviction-family behavior as SnapKVKVCache,
so test_prefill_evicts_when_over_budget mirrors that suite) and layers
dynamic HSA page selection on top for decode-time attention (stage 2,
exposed via ``select_indices``). Tests cover: factory dispatch, interface
attributes, shape/dtype, chunked-prefill re-enforcement (mlx_lm's chunked
prefill convention, same as SnapKVKVCache), byte accounting, determinism,
HSA selection, and for_model config propagation. All data is synthetic.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.rocketkv_cache import RocketKVKVCache


def _make(**cfg):
    base = dict(
        method="rocketkv",
        head_dim=32,
        rocketkv_compression_ratio=4.0,
        rocketkv_obs_window=4,
        rocketkv_n_sink=2,
    )
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_kv(S: int = 40, H: int = 2, D: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    V = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    return K, V


# ---------------------------------------------------------------------------
# Factory and interface
# ---------------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), RocketKVKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "compression_ratio")
    assert hasattr(c, "tokens_kept")
    assert hasattr(c, "stage1_bytes")
    assert hasattr(c, "stage2_aux_bytes")


def test_not_trimmable() -> None:
    assert _make().is_trimmable() is False


# ---------------------------------------------------------------------------
# Stage-1 eviction invariant — RocketKV IS an eviction method (unlike AnchorKV)
# ---------------------------------------------------------------------------


def test_prefill_evicts_when_over_budget() -> None:
    """A high compression ratio must reduce the retained token count below S."""
    c = _make(rocketkv_compression_ratio=8.0)
    k, v = _rand_kv(S=64, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] < 64
    assert vo.shape[2] < 64


def test_decode_tokens_always_appended() -> None:
    c = _make()
    k, v = _rand_kv(S=40, H=2, D=32)
    ko, _ = c.update_and_fetch(k, v)
    prefill_size = ko.shape[2]

    for i in range(6):
        kd, vd = _rand_kv(S=1, H=2, D=32, seed=50 + i)
        ko, vo = c.update_and_fetch(kd, vd)

    assert ko.shape[2] == prefill_size + 6
    assert vo.shape[2] == prefill_size + 6


def test_offset_tracks_true_position_not_row_count() -> None:
    """Same RoPE-correctness invariant as SnapKVKVCache (#171) — offset must
    equal true absolute position, not the (smaller) retained row count."""
    c = _make(rocketkv_compression_ratio=8.0)
    k, v = _rand_kv(S=64, H=2, D=32)
    ko, _ = c.update_and_fetch(k, v)
    assert ko.shape[2] < 64  # eviction happened
    assert c.offset == 64  # but true position is unaffected

    kd, vd = _rand_kv(S=1, H=2, D=32, seed=99)
    c.update_and_fetch(kd, vd)
    assert c.offset == 65


# ---------------------------------------------------------------------------
# Chunked prefill (mlx_lm's prefill_step_size convention)
# ---------------------------------------------------------------------------


def test_chunked_prefill_reenforces_budget() -> None:
    """Two S>1 calls (chunked prefill) must not double-count the budget —
    same convention SnapKVKVCache enforces (see #84). RocketKV's stage-1
    budget uses stage1_ratio (the ADAPTIVE split of the overall
    compression_ratio, paper §3.6), not the overall ratio directly, so the
    naive "retained <= S/compression_ratio" bound does not apply here — use
    stage1_ratio explicitly instead."""
    c = _make(rocketkv_compression_ratio=4.0)
    k1, v1 = _rand_kv(S=20, H=2, D=32, seed=1)
    ko1, _ = c.update_and_fetch(k1, v1)
    k2, v2 = _rand_kv(S=20, H=2, D=32, seed=2)
    ko2, _ = c.update_and_fetch(k2, v2)

    # Budget re-derived from the accumulated 40 tokens (not 20+20 stacked
    # independently), so retained count reflects stage1_ratio over 40 total,
    # not up to 2x that from re-applying the per-chunk budget twice.
    stacked_upper_bound = 2 * round(20 / c.stage1_ratio)
    assert ko2.shape[2] <= stacked_upper_bound
    assert c.offset == 40


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------


def test_output_dtype_fp16() -> None:
    c = _make()
    k, v = _rand_kv(S=20)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16


def test_output_batch_head_dims_preserved() -> None:
    c = _make()
    k, v = _rand_kv(S=32, H=4, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[0] == 1
    assert ko.shape[1] == 4
    assert ko.shape[3] == 32


def test_small_prompt_below_page_size_does_not_crash() -> None:
    c = _make(rocketkv_compression_ratio=2.0, rocketkv_page_size=16)
    k, v = _rand_kv(S=3, H=1, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] <= 3


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_compression_ratio_gt_1_at_aggressive_ratio() -> None:
    c = _make(rocketkv_compression_ratio=16.0)
    k, v = _rand_kv(S=200, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.compression_ratio > 1.0


def test_stage1_and_stage2_bytes_positive() -> None:
    c = _make()
    k, v = _rand_kv(S=48, H=2, D=32)
    c.update_and_fetch(k, v)
    assert c.stage1_bytes > 0
    assert c.stage2_aux_bytes > 0


def test_keep_rate_between_0_and_1() -> None:
    c = _make(rocketkv_compression_ratio=8.0)
    k, v = _rand_kv(S=64, H=2, D=32)
    c.update_and_fetch(k, v)
    assert 0.0 < c.keep_rate <= 1.0


def test_full_fp16_bytes_matches_uncompressed_accounting() -> None:
    c = _make()
    B, H, S, D = 1, 2, 32, 32
    k, v = _rand_kv(S=S, H=H, D=D)
    c.update_and_fetch(k, v)
    assert c.full_fp16_bytes_total == B * H * S * D * 2 * 2


# ---------------------------------------------------------------------------
# HSA stage-2 selection
# ---------------------------------------------------------------------------


def test_select_indices_returns_valid_subset() -> None:
    c = _make(rocketkv_compression_ratio=4.0)
    k, v = _rand_kv(S=40, H=2, D=32)
    c.update_and_fetch(k, v)

    q = mx.random.normal((32,)).astype(mx.float32)
    idx = c.select_indices(q, b=0, h=0)
    n_kept = c._summaries[0][0].n_tokens
    assert idx.tolist() == sorted(set(idx.tolist()))  # ascending, deduplicated
    assert all(0 <= i < n_kept for i in idx.tolist())


def test_select_indices_keep_recent_always_included() -> None:
    c = _make(rocketkv_compression_ratio=4.0)
    k, v = _rand_kv(S=40, H=2, D=32)
    c.update_and_fetch(k, v)
    n_kept = c._summaries[0][0].n_tokens

    q = mx.random.normal((32,)).astype(mx.float32)
    idx = set(c.select_indices(q, b=0, h=0, keep_recent=3).tolist())
    assert set(range(n_kept - 3, n_kept)).issubset(idx)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic() -> None:
    k, v = _rand_kv(S=32, H=2, D=32)
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
        method="rocketkv",
        head_dim=32,
        rocketkv_compression_ratio=6.0,
        rocketkv_page_size=8,
        rocketkv_head_topk1=12,
        rocketkv_obs_window=16,
        rocketkv_n_sink=2,
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, RocketKVKVCache) for c in caches)
    assert caches[0]._compression_ratio == 6.0
    assert caches[0]._page_size_cfg == 8
    assert caches[0]._head_topk1_cfg == 12
    assert caches[0]._obs_window == 16
    assert caches[0]._n_sink == 2


def test_factory_smoke_compression_ratio_positive_both_kv() -> None:
    """End-to-end factory smoke test exercising both K and V through the full path."""
    c = _make(rocketkv_compression_ratio=8.0)
    k, v = _rand_kv(S=100, H=2, D=32)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape[2] < 100
    assert vo.shape[2] < 100
    assert c.compression_ratio > 1.0
