"""Tests for ZipCacheKVCache — saliency-adaptive per-token mixed-precision.

ZipCache-adapted routes high-norm tokens to hi_bits and low-norm tokens to
lo_bits within the quantized space. These tests cover: factory dispatch, shape
preservation, byte ordering, decode accumulation, edge-case hi_fractions, the
values-off path, mask handling, and construction via both KVCacheFactory.create
and KVCacheBuilder.for_model. All data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.zipcache_cache import ZipCacheKVCache


def _make(**cfg):
    base = dict(
        method="zipcache",
        head_dim=128,
        zipcache_hi_bits=4,
        zipcache_lo_bits=2,
        zipcache_hi_fraction=0.20,
        zipcache_group_size=32,
    )
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_kv(S=64, H=2, D=128, seed=0):
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    V = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    return K, V


# ---------------------------------------------------------------------------
# Factory and interface
# ---------------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), ZipCacheKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "effective_avg_bits")


def test_output_shape_preserved() -> None:
    c = _make()
    k, v = _rand_kv()
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape == k.shape
    assert vo.shape == v.shape


def test_output_dtype_fp16() -> None:
    c = _make()
    k, v = _rand_kv()
    ko, vo = c.update_and_fetch(k, v)
    assert ko.dtype == mx.float16
    assert vo.dtype == mx.float16


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_byte_ordering_compressed_lt_fp16() -> None:
    c = _make()
    k, v = _rand_kv()
    c.update_and_fetch(k, v)
    assert c.compressed_key_bytes < c.fp16_key_bytes
    assert c.compression_ratio > 1.0


def test_byte_ordering_compressed_gte_baseline() -> None:
    """Mixed-bit keys >= all-lo-bit baseline (hi-bit tokens add overhead vs all-lo)."""
    c = _make(zipcache_hi_fraction=0.2)
    k, v = _rand_kv()
    c.update_and_fetch(k, v)
    assert c.compressed_key_bytes >= c.baseline_key_bytes


def test_effective_avg_bits_in_range() -> None:
    c = _make(zipcache_hi_bits=4, zipcache_lo_bits=2, zipcache_hi_fraction=0.2)
    k, v = _rand_kv()
    c.update_and_fetch(k, v)
    avg = c.effective_avg_bits
    assert 2.0 <= avg <= 4.0


# ---------------------------------------------------------------------------
# Values-off path
# ---------------------------------------------------------------------------


def test_values_off_passthrough() -> None:
    c = _make(zipcache_quantize_values=False)
    k, v = _rand_kv()
    ko, vo = c.update_and_fetch(k, v)
    assert float(mx.mean((vo.astype(mx.float32) - v.astype(mx.float32)) ** 2).item()) == 0.0
    assert c.compressed_value_bytes == 0
    assert c.fp16_value_bytes > 0


# ---------------------------------------------------------------------------
# Edge-case hi_fraction values
# ---------------------------------------------------------------------------


def test_hi_fraction_zero() -> None:
    """hi_fraction=0 (all lo_bits) runs without error."""
    c = _make(zipcache_hi_fraction=0.0)
    k, v = _rand_kv()
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape == k.shape


def test_hi_fraction_one() -> None:
    """hi_fraction=1 (all hi_bits) runs without error."""
    c = _make(zipcache_hi_fraction=1.0)
    k, v = _rand_kv()
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape == k.shape


# ---------------------------------------------------------------------------
# Decode accumulation
# ---------------------------------------------------------------------------


def test_decode_accumulation() -> None:
    c = _make()
    k, v = _rand_kv(S=32)
    c.update_and_fetch(k, v)
    for i in range(3):
        k1, v1 = _rand_kv(S=1, seed=100 + i)
        ko, vo = c.update_and_fetch(k1, v1)
    assert ko.shape[2] == 32 + 3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic() -> None:
    k, v = _rand_kv()
    c1, c2 = _make(), _make()
    ko1, _ = c1.update_and_fetch(k, v)
    ko2, _ = c2.update_and_fetch(k, v)
    mse = float(mx.mean((ko1.astype(mx.float32) - ko2.astype(mx.float32)) ** 2).item())
    assert mse == pytest.approx(0.0, abs=0.0)


# ---------------------------------------------------------------------------
# for_model construction
# ---------------------------------------------------------------------------


def test_build_via_for_model_propagates_config() -> None:
    """KVCacheBuilder.for_model must carry the zipcache_* fields."""
    from veloxquant_mlx.cache.base import KVCacheBuilder

    class _Attn:
        head_dim = 128

    class _Layer:
        self_attn = _Attn()

    class _Model:
        layers = [_Layer(), _Layer()]

    cfg = KVCacheConfig(
        method="zipcache",
        head_dim=128,
        zipcache_hi_bits=4,
        zipcache_lo_bits=2,
        zipcache_hi_fraction=0.25,
        zipcache_group_size=32,
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, ZipCacheKVCache) for c in caches)
    assert caches[0]._hi_bits == 4
    assert caches[0]._lo_bits == 2
    assert caches[0]._hi_fraction == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Config validation — zipcache_hi_fraction must be in [0, 1]
# ---------------------------------------------------------------------------


def test_hi_fraction_above_one_rejected() -> None:
    with pytest.raises(ValueError, match="zipcache_hi_fraction"):
        _make(zipcache_hi_fraction=1.5)


def test_hi_fraction_negative_rejected() -> None:
    with pytest.raises(ValueError, match="zipcache_hi_fraction"):
        _make(zipcache_hi_fraction=-0.2)
