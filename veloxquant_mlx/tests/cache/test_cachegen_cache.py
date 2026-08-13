"""Tests for CacheGenKVCache — entropy-coded byte accounting over group quant.

CacheGen's reconstructed K/V are identical to plain group quant; its
contribution is the entropy-coded *byte accounting*. These tests cover the
lossless-reconstruction property, the token-locality entropy win, the
never-worse-than-fixed-width cap, the usual factory/shape/accounting checks,
and — the distinguishing feature — for_model producing per-layer caches with
a non-increasing bit-width schedule (§5.1.2/§5.2 layer-wise sensitivity).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.cachegen_cache import CacheGenKVCache


def _make(**cfg):
    base = dict(method="cachegen", head_dim=64, cachegen_bits=4, cachegen_group_size=32)
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_kv(S=128, H=2, D=64, seed=0):
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((1, H, S, D)).astype(np.float16)
    V = rng.standard_normal((1, H, S, D)).astype(np.float16)
    return mx.array(K), mx.array(V)


class _Attn:
    head_dim = 64


class _Layer:
    def __init__(self):
        self.self_attn = _Attn()


class _Model:
    def __init__(self, n):
        self.layers = [_Layer() for _ in range(n)]


def _corr_kv(S=128, H=2, D=64, seed=0):
    """Token-correlated KV (random walk) — the locality CacheGen exploits."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal((S, 1, D)).astype(np.float32) * 0.15, axis=0)
    K = np.repeat(walk, H, axis=1).transpose(1, 0, 2)[None]
    V = np.repeat(walk, H, axis=1).transpose(1, 0, 2)[None]
    return mx.array(K.astype(np.float16)), mx.array(V.astype(np.float16))


# ------------------------------------------------------------------
# Factory and interface
# ------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), CacheGenKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "assigned_avg_bits")


def test_output_shape_preserved() -> None:
    c = _make()
    K, V = _rand_kv(S=64, H=2, D=64)
    ko, vo = c.update_and_fetch(K, V)
    mx.eval(ko, vo)
    assert ko.shape == (1, 2, 64, 64)
    assert vo.shape == (1, 2, 64, 64)
    assert ko.dtype == mx.float16


# ------------------------------------------------------------------
# Reconstruction is exactly plain group quant (lossless over codes)
# ------------------------------------------------------------------


def test_reconstruction_matches_group_quant() -> None:
    from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant

    K, V = _rand_kv(S=64, H=1, D=64, seed=3)
    c = _make()
    ko, _ = c.update_and_fetch(K, V)
    mx.eval(ko)
    ref = _group_quant_dequant(K[0, 0].astype(mx.float32), b=4, group_size=32)
    mx.eval(ref)
    assert np.allclose(np.array(ko[0, 0]), np.array(ref), atol=1e-3)


# ------------------------------------------------------------------
# Token-locality entropy win
# ------------------------------------------------------------------


def test_delta_lowers_entropy_on_correlated_data() -> None:
    from veloxquant_mlx.quantizers.cachegen import (
        quantize_to_codes,
        token_delta,
        symbol_entropy_bits,
    )

    K, _ = _corr_kv(S=128, H=1, D=64)
    st = quantize_to_codes(K[0, 0], bits=4, group_size=32)
    flat = st.codes.reshape(-1, 64)[:128]
    e_raw = symbol_entropy_bits(flat)
    e_delta = symbol_entropy_bits(token_delta(flat))
    assert e_delta < e_raw, f"delta entropy {e_delta:.3f} should be < raw {e_raw:.3f}"


def test_entropy_savings_positive_on_correlated_data() -> None:
    c = _make()
    K, V = _corr_kv(S=256, H=2, D=64)
    c.update_and_fetch(K, V)
    assert c.entropy_savings > 0.0
    assert c.compressed_key_bytes < c.fixed_width_key_bytes


# ------------------------------------------------------------------
# Never-worse-than-fixed-width cap (incompressible data)
# ------------------------------------------------------------------


def test_savings_never_negative_on_random_data() -> None:
    c = _make()
    K, V = _rand_kv(S=64, H=2, D=64)
    c.update_and_fetch(K, V)
    assert c.entropy_savings >= 0.0
    assert c.compressed_key_bytes <= c.fixed_width_key_bytes


# ------------------------------------------------------------------
# Byte accounting ordering
# ------------------------------------------------------------------


def test_byte_accounting_ordering() -> None:
    c = _make()
    K, V = _corr_kv(S=128, H=2, D=64)
    c.update_and_fetch(K, V)
    assert 0 < c.compressed_key_bytes <= c.fixed_width_key_bytes < c.fp16_key_bytes
    assert 0 < c.compressed_value_bytes <= c.fixed_width_value_bytes < c.fp16_value_bytes


def test_assigned_avg_bits_below_fp16() -> None:
    c = _make(cachegen_bits=4)
    K, V = _corr_kv(S=256, H=2, D=64)
    c.update_and_fetch(K, V)
    assert c.assigned_avg_bits < 16.0


def test_use_delta_false_path() -> None:
    c = _make(cachegen_use_delta=False)
    K, V = _corr_kv(S=128, H=2, D=64)
    c.update_and_fetch(K, V)
    assert c.compressed_key_bytes <= c.fixed_width_key_bytes


# ------------------------------------------------------------------
# Decode + determinism
# ------------------------------------------------------------------


def test_decode_after_prefill() -> None:
    c = _make()
    Kp, Vp = _rand_kv(S=32, H=2, D=64)
    c.update_and_fetch(Kp, Vp)
    for step in range(4):
        Kd, Vd = _rand_kv(S=1, H=2, D=64, seed=step + 10)
        ko, vo = c.update_and_fetch(Kd, Vd)
        mx.eval(ko, vo)
        assert not mx.any(mx.isnan(ko)).item()


def test_determinism() -> None:
    K, V = _rand_kv(S=64, H=2, D=64, seed=7)
    c1, c2 = _make(), _make()
    ko1, _ = c1.update_and_fetch(K, V)
    ko2, _ = c2.update_and_fetch(K, V)
    mx.eval(ko1, ko2)
    assert np.allclose(np.array(ko1), np.array(ko2), atol=1e-4)
    assert c1.compressed_key_bytes == c2.compressed_key_bytes


# ------------------------------------------------------------------
# Single-layer fallback (== uniform cachegen_bits)
# ------------------------------------------------------------------


def test_single_layer_falls_back_to_uniform_bits() -> None:
    """Factory.create (no layer context) uses cachegen_bits as the bit-width."""
    c = _make(cachegen_bits=5)
    assert c._bits == 5


# ------------------------------------------------------------------
# for_model — per-layer bit-width schedule (§5.1.2/§5.2, the distinguishing feature)
# ------------------------------------------------------------------


def test_for_model_returns_cachegen_caches() -> None:
    cfg = KVCacheConfig(method="cachegen", head_dim=64, cachegen_bits=4)
    caches = KVCacheBuilder.for_model(_Model(8), cfg)
    assert len(caches) == 8
    assert all(isinstance(c, CacheGenKVCache) for c in caches)


def test_for_model_bits_non_increasing_with_depth() -> None:
    cfg = KVCacheConfig(method="cachegen", head_dim=64, cachegen_bits=4, cachegen_layer_groups=3)
    caches = KVCacheBuilder.for_model(_Model(12), cfg)
    bits = [c._bits for c in caches]
    assert bits == sorted(bits, reverse=True)
    assert bits[0] == 4
    assert bits[-1] < bits[0]


def test_for_model_shallow_layer_keeps_base_bits() -> None:
    cfg = KVCacheConfig(method="cachegen", head_dim=64, cachegen_bits=6)
    caches = KVCacheBuilder.for_model(_Model(9), cfg)
    assert caches[0]._bits == 6
