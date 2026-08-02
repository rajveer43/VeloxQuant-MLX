"""Tests for GEARKVCache — error-feedback compression over a base group quant.

GEAR's reconstructed K/V genuinely recover quality the base bit-width loses
(unlike CacheGen, whose reconstruction is identical to group quant). These tests
cover factory dispatch, shape preservation, the quality-recovery property, byte
accounting, the values-off path, decode accumulation, determinism, and
construction via both KVCacheFactory.create and KVCacheBuilder.for_model. All
data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.gear_cache import GEARKVCache


def _make(**cfg):
    base = dict(
        method="gear",
        head_dim=128,
        gear_bits=2,
        gear_rank=8,
        gear_sparse_fraction=0.005,
        gear_group_size=32,
    )
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _lowrank_kv(S=128, H=2, D=128, r=6, seed=0):
    """Low-rank + small-noise KV — the regime GEAR's error feedback helps."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((H, S, r)).astype(np.float32)
    B = rng.standard_normal((H, r, D)).astype(np.float32)
    K = (A @ B + 0.03 * rng.standard_normal((H, S, D))).astype(np.float16)[None]
    V = (A @ B + 0.03 * rng.standard_normal((H, S, D))).astype(np.float16)[None]
    return mx.array(K), mx.array(V)


# ------------------------------------------------------------------
# Factory and interface
# ------------------------------------------------------------------


def test_factory_dispatch() -> None:
    assert isinstance(_make(), GEARKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "assigned_avg_bits")


def test_output_shape_preserved() -> None:
    c = _make()
    k, v = _lowrank_kv()
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape == k.shape and vo.shape == v.shape


# ------------------------------------------------------------------
# Core claim: GEAR recovers quality the base bit-width loses
# ------------------------------------------------------------------


def test_error_recovery_positive() -> None:
    c = _make()
    k, v = _lowrank_kv()
    c.update_and_fetch(k, v)
    assert 0.0 < c.error_recovery_ratio <= 1.0


def test_beats_naive_base_reconstruction() -> None:
    """The reconstructed keys are closer to the originals than base quant alone."""
    from veloxquant_mlx.quantizers.cachegen import cachegen_quant_dequant

    c = _make()
    k, v = _lowrank_kv()
    ko, _ = c.update_and_fetch(k, v)

    def mse(a, b):
        return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())

    base = mx.stack(
        [
            mx.stack([cachegen_quant_dequant(k[b, h], 2, 32) for h in range(k.shape[1])])
            for b in range(k.shape[0])
        ]
    )
    assert mse(ko, k) < mse(base, k)


# ------------------------------------------------------------------
# Byte accounting
# ------------------------------------------------------------------


def test_byte_accounting_ordering() -> None:
    c = _make()
    k, v = _lowrank_kv()
    c.update_and_fetch(k, v)
    assert c.base_only_key_bytes <= c.compressed_key_bytes <= c.fp16_key_bytes
    assert c.assigned_avg_bits <= 16.0


def test_values_off_keeps_values_fp16() -> None:
    c = _make(gear_quantize_values=False)
    k, v = _lowrank_kv()
    ko, vo = c.update_and_fetch(k, v)
    # values pass through unchanged (lossless)
    assert float(mx.mean((vo.astype(mx.float32) - v.astype(mx.float32)) ** 2).item()) == 0.0
    assert c.compressed_value_bytes == 0
    assert c.fp16_value_bytes > 0


# ------------------------------------------------------------------
# Decode and robustness
# ------------------------------------------------------------------


def test_decode_accumulation() -> None:
    c = _make()
    k, v = _lowrank_kv(S=64)
    c.update_and_fetch(k, v)
    for i in range(4):
        k1, v1 = _lowrank_kv(S=1, seed=100 + i)
        ko, vo = c.update_and_fetch(k1, v1)
    assert ko.shape[2] == 64 + 4


def test_deterministic() -> None:
    k, v = _lowrank_kv()
    c1, c2 = _make(), _make()
    ko1, _ = c1.update_and_fetch(k, v)
    ko2, _ = c2.update_and_fetch(k, v)
    assert float(mx.mean((ko1.astype(mx.float32) - ko2.astype(mx.float32)) ** 2).item()) == 0.0


def test_build_via_for_model_propagates_config() -> None:
    """KVCacheBuilder.for_model must carry the gear_* fields (replace path)."""
    from veloxquant_mlx.cache.base import KVCacheBuilder

    class _Attn:
        head_dim = 128

    class _Layer:
        self_attn = _Attn()

    class _Model:
        layers = [_Layer(), _Layer()]

    cfg = KVCacheConfig(
        method="gear", head_dim=128, gear_bits=2, gear_rank=8, gear_sparse_fraction=0.005
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, GEARKVCache) for c in caches)
    assert caches[0]._rank == 8
    assert caches[0]._sparse_frac == pytest.approx(0.005)
