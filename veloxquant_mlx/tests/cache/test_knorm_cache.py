"""Tests for L2NormKVCache — intrinsic key-norm eviction wrapper.

Covers mlx_lm protocol shape/dtype preservation, budget enforcement, sink
protection, the prefill-vs-decode bit-for-bit path-independence invariant,
the keep="low"/"high" ablation knob, the mechanism test under paper-like
geometry, byte accounting, build-time validation, and for_model wiring.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.knorm_cache import L2NormKVCache


def _kv(B, H, S, D, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))
    return k, v


def _make(**cfg):
    base = dict(method="knorm", head_dim=64, knorm_budget=16, knorm_n_sink=2)
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


# ------------------------------------------------------------------
# Protocol basics
# ------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), L2NormKVCache)


def test_shape_dtype_preserved() -> None:
    cache = _make(knorm_budget=128)
    k, v = _kv(1, 4, 64, 64)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert ko.shape == (1, 4, 64, 64)  # under budget: everything kept
    assert vo.shape == (1, 4, 64, 64)
    assert ko.dtype == mx.float16


def test_no_bits_leak() -> None:
    cache = _make()
    assert not hasattr(cache, "bits")


def test_under_budget_bitfor_bit_passthrough() -> None:
    cache = _make(knorm_budget=128)
    k, v = _kv(1, 2, 32, 64, seed=1)
    ko, vo = cache.update_and_fetch(k, v)
    assert np.array_equal(np.array(ko), np.array(k))
    assert np.array_equal(np.array(vo), np.array(v))


# ------------------------------------------------------------------
# Eviction behavior
# ------------------------------------------------------------------


def test_budget_enforced_after_long_prefill() -> None:
    cache = _make(knorm_budget=16)
    k, v = _kv(1, 2, 200, 64, seed=2)
    ko, vo = cache.update_and_fetch(k, v)
    assert ko.shape[2] == 16
    assert cache.tokens_kept == 16


def test_sinks_retained_across_heavy_eviction() -> None:
    B, H, S, D, n_sink = 1, 1, 100, 64, 3
    rng = np.random.default_rng(3)
    k = rng.standard_normal((B, H, S, D)).astype(np.float16)
    k[:, :, :n_sink, :] *= 40.0  # sinks have enormous norms
    v = rng.standard_normal((B, H, S, D)).astype(np.float16)
    cache = _make(knorm_budget=8, knorm_n_sink=n_sink)
    ko, _ = cache.update_and_fetch(mx.array(k), mx.array(v))
    assert np.array_equal(np.array(ko[0, 0, :n_sink]), k[0, 0, :n_sink])


def test_decode_accumulation_caps_at_budget() -> None:
    cache = _make(knorm_budget=8, knorm_n_sink=2)
    for t in range(24):
        k, v = _kv(1, 2, 1, 64, seed=100 + t)
        ko, vo = cache.update_and_fetch(k, v)
    assert ko.shape[2] == 8
    assert cache.tokens_seen == 24 * 2  # per-head positions summed


def test_prefill_decode_bit_for_bit_equivalence() -> None:
    """The path-independence invariant: with knorm_recent=0 the kept cache is
    identical whether tokens arrive as one prefill block or one at a time."""
    S = 50
    k, v = _kv(1, 2, S, 64, seed=4)

    block = _make(knorm_budget=12, knorm_n_sink=2)
    ka, va = block.update_and_fetch(k, v)

    stream = _make(knorm_budget=12, knorm_n_sink=2)
    for t in range(S):
        kb, vb = stream.update_and_fetch(k[:, :, t : t + 1, :], v[:, :, t : t + 1, :])
    mx.eval(ka, va, kb, vb)
    assert np.array_equal(np.array(ka), np.array(kb))
    assert np.array_equal(np.array(va), np.array(vb))


def test_keep_high_differs_and_respects_budget() -> None:
    k, v = _kv(1, 2, 64, 64, seed=5)
    lo = _make(knorm_budget=16, knorm_keep="low")
    hi = _make(knorm_budget=16, knorm_keep="high")
    klo, _ = lo.update_and_fetch(k, v)
    khi, _ = hi.update_and_fetch(k, v)
    assert klo.shape[2] == 16 and khi.shape[2] == 16
    assert not np.array_equal(np.array(klo), np.array(khi))


# ------------------------------------------------------------------
# Mechanism test under paper-like geometry
# ------------------------------------------------------------------


def _attn_out(q, k, v):
    scale = 1.0 / math.sqrt(float(k.shape[-1]))
    w = mx.softmax((q @ k.T) * scale, axis=-1)
    return w @ v


def test_keep_low_beats_keep_high_under_paper_geometry() -> None:
    """Construct the geometry the paper reports in trained LMs (Devoto et
    al., EMNLP 2024): low-norm keys aligned with the query distribution,
    high-norm keys anti-aligned. Under it, the keep-low cache's attention
    output must be closer to the full-cache output than keep-high's.
    The geometry is constructed — the correlation claim is the paper's."""
    rng = np.random.default_rng(6)
    D, S, n_imp = 64, 96, 24
    mu = rng.standard_normal(D).astype(np.float32)
    mu /= np.linalg.norm(mu)

    k = np.zeros((S, D), dtype=np.float32)
    imp_idx = rng.choice(S, size=n_imp, replace=False)
    imp_mask = np.zeros(S, dtype=bool)
    imp_mask[imp_idx] = True
    # Important: low norm, aligned. Unimportant: high norm, anti-aligned.
    k[imp_mask] = 0.5 * mu + 0.05 * rng.standard_normal((n_imp, D))
    k[~imp_mask] = 3.0 * (-mu + 0.3 * rng.standard_normal((S - n_imp, D)))
    v = rng.standard_normal((S, D)).astype(np.float32)
    q = (mu + 0.1 * rng.standard_normal((16, D))).astype(np.float32)

    kk = mx.array(k[None, None].astype(np.float16))
    vv = mx.array(v[None, None].astype(np.float16))
    ref = _attn_out(mx.array(q), mx.array(k), mx.array(v))

    def perturbation(keep: str) -> float:
        cache = _make(knorm_budget=n_imp + 4, knorm_n_sink=0, knorm_keep=keep, head_dim=D)
        ko, vo = cache.update_and_fetch(kk, vv)
        out = _attn_out(mx.array(q), ko[0, 0].astype(mx.float32), vo[0, 0].astype(mx.float32))
        rn = ref / (mx.sqrt(mx.sum(ref * ref, -1, keepdims=True)) + 1e-8)
        on = out / (mx.sqrt(mx.sum(out * out, -1, keepdims=True)) + 1e-8)
        return float(mx.mean(1.0 - mx.sum(rn * on, -1)).item())

    assert perturbation("low") < perturbation("high")


# ------------------------------------------------------------------
# Accounting / validation / wiring
# ------------------------------------------------------------------


def test_compression_ratio_math() -> None:
    cache = _make(knorm_budget=16)
    k, v = _kv(1, 2, 128, 64, seed=7)
    cache.update_and_fetch(k, v)
    assert cache.knorm_kept_bytes == 2 * 16 * 64 * 2 * 2  # 2 heads
    assert cache.full_seq_bytes == 2 * 128 * 64 * 2 * 2
    assert abs(cache.compression_ratio - 8.0) < 1e-9


def test_determinism() -> None:
    k, v = _kv(1, 2, 80, 64, seed=8)
    a, b = _make(), _make()
    ka, va = a.update_and_fetch(k, v)
    kb, vb = b.update_and_fetch(k, v)
    assert np.array_equal(np.array(ka), np.array(kb))
    assert np.array_equal(np.array(va), np.array(vb))


def test_build_time_validation() -> None:
    with pytest.raises(ValueError, match="keep"):
        _make(knorm_keep="middle")
    with pytest.raises(ValueError, match="evictable"):
        _make(knorm_budget=8, knorm_n_sink=4, knorm_recent=4)


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

    caches = KVCacheBuilder.for_model(_ToyModel(), KVCacheConfig(method="knorm", head_dim=64))
    assert len(caches) == 3
    assert isinstance(caches[0], L2NormKVCache)
    assert type(caches[1]) is _FallbackCache
    assert isinstance(caches[2], L2NormKVCache)
