"""Tests for QFiltersKVCache — query-agnostic projection eviction wrapper.

Covers mlx_lm protocol shape/dtype preservation, budget enforcement,
pre-calibration passthrough, sink protection, the honest prefill-vs-decode
NON-equivalence (both valid, not bit-for-bit — Q-Filters is path-dependent),
the sign ablation knob, the mechanism test under paper-like geometry, byte
accounting, build-time validation, and for_model wiring.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.qfilters_cache import QFiltersKVCache


def _kv(B, H, S, D, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


def _make(**cfg):
    base = dict(
        method="qfilters",
        head_dim=64,
        qfilters_budget=16,
        qfilters_n_sink=2,
        qfilters_calib_tokens=16,
    )
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


# ------------------------------------------------------------------
# Protocol basics
# ------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), QFiltersKVCache)


def test_shape_dtype_preserved() -> None:
    cache = _make(qfilters_budget=128)
    k, v = _kv(1, 4, 64, 64)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert ko.shape == (1, 4, 64, 64)  # under budget: everything kept
    assert vo.shape == (1, 4, 64, 64)
    assert ko.dtype == mx.float16


def test_no_bits_leak() -> None:
    assert not hasattr(_make(), "bits")


def test_under_budget_bit_for_bit_passthrough() -> None:
    cache = _make(qfilters_budget=128)
    k, v = _kv(1, 2, 32, 64, seed=1)
    ko, vo = cache.update_and_fetch(k, v)
    assert np.array_equal(np.array(ko), np.array(k))
    assert np.array_equal(np.array(vo), np.array(v))


def test_pre_calibration_passthrough() -> None:
    """Below calib_tokens, no filter yet — nothing evicted even over budget."""
    cache = _make(qfilters_budget=8, qfilters_calib_tokens=200)
    k, v = _kv(1, 2, 50, 64, seed=2)  # 50 > budget 8, but < calib 200
    ko, _ = cache.update_and_fetch(k, v)
    assert ko.shape[2] == 50


# ------------------------------------------------------------------
# Eviction behavior
# ------------------------------------------------------------------


def test_budget_enforced_after_long_prefill() -> None:
    cache = _make(qfilters_budget=16)
    k, v = _kv(1, 2, 200, 64, seed=3)
    ko, _ = cache.update_and_fetch(k, v)
    assert ko.shape[2] == 16
    assert cache.tokens_kept == 16


def test_sinks_retained_across_heavy_eviction() -> None:
    B, H, S, D, n_sink = 1, 1, 120, 64, 3
    k, v = _kv(B, H, S, D, seed=4)
    cache = _make(qfilters_budget=8, qfilters_n_sink=n_sink, head_dim=D)
    ko, _ = cache.update_and_fetch(k, v)
    assert np.array_equal(np.array(ko[0, 0, :n_sink]), np.array(k[0, 0, :n_sink]))


def test_decode_accumulation_caps_at_budget() -> None:
    cache = _make(qfilters_budget=8, qfilters_n_sink=2, qfilters_calib_tokens=8)
    for t in range(30):
        k, v = _kv(1, 2, 1, 64, seed=100 + t)
        ko, _ = cache.update_and_fetch(k, v)
    assert ko.shape[2] == 8
    assert cache.tokens_seen == 30 * 2


def test_prefill_decode_both_valid_not_equivalent() -> None:
    """Q-Filters is path-DEPENDENT: prefill and decode may freeze different
    filters. We assert both stay within budget and both freeze a valid
    unit-norm filter — NOT that they are bit-for-bit equal."""
    S = 80
    k, v = _kv(1, 1, S, 64, seed=5)

    block = _make(qfilters_budget=20, qfilters_n_sink=2, head_dim=64)
    ka, _ = block.update_and_fetch(k, v)

    stream = _make(qfilters_budget=20, qfilters_n_sink=2, head_dim=64)
    for t in range(S):
        kb, _ = stream.update_and_fetch(k[:, :, t : t + 1, :], v[:, :, t : t + 1, :])
    mx.eval(ka, kb)

    assert ka.shape[2] <= 20 and kb.shape[2] <= 20
    for st in (block._states[0], stream._states[0]):
        d = np.array(st.filter_dir)
        assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-4


def test_sign_differs_and_respects_budget() -> None:
    k, v = _kv(1, 2, 80, 64, seed=6)
    pos = _make(qfilters_budget=16, qfilters_sign=1)
    neg = _make(qfilters_budget=16, qfilters_sign=-1)
    kp, _ = pos.update_and_fetch(k, v)
    kn, _ = neg.update_and_fetch(k, v)
    assert kp.shape[2] == 16 and kn.shape[2] == 16
    assert not np.array_equal(np.array(kp), np.array(kn))


# ------------------------------------------------------------------
# Mechanism test under paper-like geometry
# ------------------------------------------------------------------


def _attn_out(q, k, v):
    scale = 1.0 / math.sqrt(float(k.shape[-1]))
    w = mx.softmax((q @ k.T) * scale, axis=-1)
    return w @ v


def test_projection_scorer_beats_random_under_paper_geometry() -> None:
    """Construct the QK anisotropy the paper (arXiv:2503.02812) reports:
    important tokens carry a large projection onto the dominant axis and align
    with the query cluster; the rest are near-orthogonal noise.

    The honest claim (given a KEY-derived filter): the key-SVD recovers the
    dominant *axis* but not which *end* is important — that sign is exactly
    what a query would disambiguate and the cache never sees one. So the
    ``sign`` knob is a real ablation, and the correct-sign cache's attention
    output must clearly beat random eviction. We assert the *better of the two
    signs* beats random by a wide margin (not that +1 specifically wins) —
    that is the truthful statement for a query-agnostic key-only estimator."""
    rng = np.random.default_rng(7)
    D, S, n_imp = 64, 256, 24
    mu = rng.standard_normal(D).astype(np.float32)
    mu /= np.linalg.norm(mu)

    k = np.zeros((S, D), dtype=np.float32)
    imp_idx = rng.choice(S, size=n_imp, replace=False)
    imp_mask = np.zeros(S, dtype=bool)
    imp_mask[imp_idx] = True
    k[imp_mask] = 4.0 * mu + 0.2 * rng.standard_normal((n_imp, D))
    k[~imp_mask] = 0.3 * rng.standard_normal((S - n_imp, D))
    v = rng.standard_normal((S, D)).astype(np.float32)
    q = (mu + 0.1 * rng.standard_normal((16, D))).astype(np.float32)

    kk = mx.array(k[None, None].astype(np.float16))
    vv = mx.array(v[None, None].astype(np.float16))
    ref = _attn_out(mx.array(q), mx.array(k), mx.array(v))
    budget = n_imp + 4

    def _pert(out) -> float:
        rn = ref / (mx.sqrt(mx.sum(ref * ref, -1, keepdims=True)) + 1e-8)
        on = out / (mx.sqrt(mx.sum(out * out, -1, keepdims=True)) + 1e-8)
        return float(mx.mean(1.0 - mx.sum(rn * on, -1)).item())

    def sign_pert(sign: int) -> float:
        cache = _make(
            qfilters_budget=budget,
            qfilters_n_sink=0,
            qfilters_sign=sign,
            qfilters_calib_tokens=32,
            head_dim=D,
        )
        ko, vo = cache.update_and_fetch(kk, vv)
        return _pert(
            _attn_out(mx.array(q), ko[0, 0].astype(mx.float32), vo[0, 0].astype(mx.float32))
        )

    # Random-eviction baseline at the same budget.
    rperts = []
    for s in range(8):
        idx = np.sort(np.random.default_rng(s).choice(S, budget, replace=False))
        rperts.append(_pert(_attn_out(mx.array(q), mx.array(k[idx]), mx.array(v[idx]))))
    random_pert = float(np.mean(rperts))

    best_sign = min(sign_pert(1), sign_pert(-1))
    assert best_sign < 0.5 * random_pert


# ------------------------------------------------------------------
# Accounting / validation / wiring
# ------------------------------------------------------------------


def test_compression_ratio_math() -> None:
    D = 64
    cache = _make(qfilters_budget=16, head_dim=D)
    k, v = _kv(1, 2, 128, D, seed=8)
    cache.update_and_fetch(k, v)
    # 2 heads: K+V fp16 for 16 tokens each, plus D float32 filter per head.
    expected = 2 * (16 * D * 2 * 2 + D * 4)
    assert cache.qfilters_kept_bytes == expected
    assert cache.full_seq_bytes == 2 * 128 * D * 2 * 2


def test_determinism() -> None:
    k, v = _kv(1, 2, 80, 64, seed=9)
    a, b = _make(), _make()
    ka, va = a.update_and_fetch(k, v)
    kb, vb = b.update_and_fetch(k, v)
    assert np.array_equal(np.array(ka), np.array(kb))
    assert np.array_equal(np.array(va), np.array(vb))


def test_build_time_validation() -> None:
    with pytest.raises(ValueError, match="sign"):
        _make(qfilters_sign=3)
    with pytest.raises(ValueError, match="evictable"):
        _make(qfilters_budget=8, qfilters_n_sink=4, qfilters_recent=4)


class _ToyAttn:
    def __init__(self, head_dim):
        self.head_dim = head_dim


class _ToyLayer:
    def __init__(self, head_dim=64):
        self.self_attn = _ToyAttn(head_dim)


class _ToyNorm:
    pass


class _ToyInner:
    def __init__(self):
        self.layers = [_ToyLayer(), _ToyNorm(), _ToyLayer()]


class _ToyModel:
    def __init__(self):
        self.model = _ToyInner()
        self.args = None


def test_for_model_wiring_and_fallback() -> None:
    from mlx_lm.models.cache import KVCache as _FallbackCache

    caches = KVCacheBuilder.for_model(_ToyModel(), KVCacheConfig(method="qfilters", head_dim=64))
    assert len(caches) == 3
    assert isinstance(caches[0], QFiltersKVCache)
    assert type(caches[1]) is _FallbackCache
    assert isinstance(caches[2], QFiltersKVCache)


# ==================================================================
# Calibrated (paper-faithful) filters — see quantizers/qfilters_calibration.py
# ==================================================================


def _calibrated_filters(H: int, D: int, seed: int = 0) -> mx.array:
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((H, D))
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return mx.array(f.astype(np.float32))


def _kv_block(B: int, H: int, S: int, D: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


def test_calibrated_filter_evicts_without_warmup() -> None:
    """A calibrated filter is frozen before token 0 — no calibration window.

    The key-SVD fallback must first observe `calib_tokens` keys, during which
    the cache grows past its budget. A supplied filter removes that window
    entirely.
    """
    B, H, S, D = 1, 4, 80, 16
    cfg = KVCacheConfig(method="qfilters", head_dim=D, qfilters_budget=32, qfilters_n_sink=2)
    k, v = _kv_block(B, H, S, D, seed=1)

    calibrated = QFiltersKVCache(cfg, filters=_calibrated_filters(H, D))
    k_out, _ = calibrated.update_and_fetch(k, v)
    assert k_out.shape[2] == 32

    # Fallback: still inside its calibration window, so nothing is evicted.
    fallback = QFiltersKVCache(cfg)
    k_fb, _ = fallback.update_and_fetch(k, v)
    assert k_fb.shape[2] == S


def test_calibrated_filter_is_path_independent() -> None:
    """Prefill-in-one-block and token-by-token decode agree bit-for-bit.

    This is the guarantee the key-SVD fallback cannot make: its direction
    depends on which chunk first crosses the calibration threshold.
    """
    B, H, S, D = 1, 4, 80, 16
    cfg = KVCacheConfig(method="qfilters", head_dim=D, qfilters_budget=32, qfilters_n_sink=2)
    filters = _calibrated_filters(H, D, seed=2)
    k, v = _kv_block(B, H, S, D, seed=3)

    prefill = QFiltersKVCache(cfg, filters=filters)
    k_prefill, v_prefill = prefill.update_and_fetch(k, v)

    decode = QFiltersKVCache(cfg, filters=filters)
    k_decode = v_decode = None
    for t in range(S):
        k_decode, v_decode = decode.update_and_fetch(k[:, :, t : t + 1], v[:, :, t : t + 1])

    assert np.array_equal(np.array(k_prefill), np.array(k_decode))
    assert np.array_equal(np.array(v_prefill), np.array(v_decode))


def test_calibrated_selection_matches_projection_ranking() -> None:
    """The kept set is the budget highest-projection rows against the filter."""
    B, H, S, D, budget = 1, 2, 40, 8, 12
    cfg = KVCacheConfig(method="qfilters", head_dim=D, qfilters_budget=budget, qfilters_n_sink=0)
    filters = _calibrated_filters(H, D, seed=4)
    k, v = _kv_block(B, H, S, D, seed=5)

    cache = QFiltersKVCache(cfg, filters=filters)
    k_out, _ = cache.update_and_fetch(k, v)

    for h in range(H):
        scores = np.array(k, dtype=np.float32)[0, h] @ np.array(filters, dtype=np.float32)[h]
        keep = np.sort(np.argsort(scores, kind="stable")[S - budget :])
        assert np.array_equal(np.array(k)[0, h][keep], np.array(k_out)[0, h])


def test_calibrated_filters_reject_head_layout_mismatch() -> None:
    """Filters from another model must fail loudly, not silently mis-evict."""
    cfg = KVCacheConfig(method="qfilters", head_dim=16, qfilters_budget=16)
    cache = QFiltersKVCache(cfg, filters=_calibrated_filters(8, 16))
    k, v = _kv_block(1, 4, 20, 16, seed=6)  # 4 heads, filters have 8

    with pytest.raises(ValueError, match="calibrated on a different model"):
        cache.update_and_fetch(k, v)


def test_calibrated_filters_reject_wrong_rank() -> None:
    cfg = KVCacheConfig(method="qfilters", head_dim=16, qfilters_budget=16)
    with pytest.raises(ValueError, match=r"2-D \[H_kv, D\]"):
        QFiltersKVCache(cfg, filters=mx.zeros((4,), dtype=mx.float32))
