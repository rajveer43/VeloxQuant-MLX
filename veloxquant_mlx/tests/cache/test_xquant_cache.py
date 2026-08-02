"""Tests for XQuantKVCache — cross-layer KV cache reuse.

16 tests covering:
  1.  Factory dispatch (degenerate anchor)
  2.  for_model produces correct anchor/reuse pairing across layers
  3.  Coordinator publish/fetch round-trips codes exactly
  4.  Anchor output shape (prefill + decode)
  5.  Reuse output shape (prefill + decode)
  6.  Values reconstructed (shape + finite)
  7.  Reuse residual=0 reconstructs within tolerance of its own quant
  8.  Reuse residual>0 strictly lower MSE than residual=0 (correlated layers)
  9.  Correlated layer pairs — reuse MSE ~= per-layer self-quant MSE (reuse near-free)
  10. Uncorrelated pairs — residual path recovers quality (negative control)
  11. Byte accounting — reuse bytes << anchor bytes
  12. effective_pair_bits < base quantizer bits (reuse layer)
  13. Decode after prefill — anchor + reuser stay synchronized over steps
  14. Coordinator token budget — exceeding max_ctx raises
  15. group_size=3 (1 anchor -> 2 reusers) pairing + reconstruction
  16. Determinism
"""

from __future__ import annotations

import pytest
import numpy as np
import mlx.core as mx

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory, KVCacheBuilder
from veloxquant_mlx.cache.xquant_cache import XQuantKVCache
from veloxquant_mlx.cache.xquant_coordinator import XQuantCoordinator
from veloxquant_mlx.quantizers.xquant import (
    pair_layers,
    quantize_codes,
    dequant_with_params,
)
from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cfg(**kwargs) -> KVCacheConfig:
    d = dict(method="xquant", head_dim=64)
    d.update(kwargs)
    return KVCacheConfig(**d)


def _pair(coord, base_bits=2, residual_bits=0, group_id=0):
    cfg = _cfg(xquant_base_bits=base_bits, xquant_residual_bits=residual_bits)
    a = XQuantKVCache(cfg, role="anchor", group_id=group_id, coordinator=coord)
    r = XQuantKVCache(cfg, role="reuse", group_id=group_id, coordinator=coord)
    return a, r


def _rand(B=1, H=2, S=32, D=64, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))


def _mse(a, b):
    return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())


class _MockAttn:
    def __init__(self, head_dim):
        self.head_dim = head_dim


class _MockLayer:
    def __init__(self, head_dim=64):
        self.self_attn = _MockAttn(head_dim)


class _MockModel:
    def __init__(self, n_layers, head_dim=64):
        self.layers = [_MockLayer(head_dim) for _ in range(n_layers)]
        self.args = None


# ---------------------------------------------------------------------------
# Test 1 — factory dispatch (degenerate anchor)
# ---------------------------------------------------------------------------
def test_factory_dispatch():
    cache = KVCacheFactory.create(_cfg())
    assert isinstance(cache, XQuantKVCache)
    assert cache.role == "anchor"  # no coordinator → degenerate anchor


# ---------------------------------------------------------------------------
# Test 2 — for_model pairing
# ---------------------------------------------------------------------------
def test_for_model_pairing():
    model = _MockModel(n_layers=6, head_dim=64)
    caches = KVCacheBuilder.for_model(model, _cfg(xquant_group_size=2))
    assert len(caches) == 6
    roles = [(c.role, c.group_id) for c in caches]
    assert roles == [
        ("anchor", 0),
        ("reuse", 0),
        ("anchor", 1),
        ("reuse", 1),
        ("anchor", 2),
        ("reuse", 2),
    ]
    # All share one coordinator instance
    coords = {id(c._coord) for c in caches}
    assert len(coords) == 1


# ---------------------------------------------------------------------------
# Test 3 — coordinator publish/fetch round-trips
# ---------------------------------------------------------------------------
def test_coordinator_round_trip():
    coord = XQuantCoordinator()
    codes, params = quantize_codes(_rand(1, 1, 16, 64)[0, 0], bits=2, group_size=32)
    coord.register_anchor(0, token_start=0, n_tokens=16, codes=codes, params=params)
    seg = coord.fetch_anchor(0, 0)
    assert seg is not None
    np.testing.assert_array_equal(np.array(seg.codes.tolist()), np.array(codes.tolist()))
    assert coord.published_tokens(0) == 16
    assert coord.fetch_anchor(0, 999) is None  # missing offset


# ---------------------------------------------------------------------------
# Test 4 — anchor output shape (prefill + decode)
# ---------------------------------------------------------------------------
def test_anchor_shape_prefill_decode():
    coord = XQuantCoordinator()
    a, _ = _pair(coord)
    ko, vo = a.update_and_fetch(_rand(1, 2, 32, 64), _rand(1, 2, 32, 64, seed=1))
    assert ko.shape == (1, 2, 32, 64) and vo.shape == (1, 2, 32, 64)
    ko2, _ = a.update_and_fetch(_rand(1, 2, 1, 64, seed=2), _rand(1, 2, 1, 64, seed=3))
    assert ko2.shape == (1, 2, 33, 64)


# ---------------------------------------------------------------------------
# Test 5 — reuse output shape (prefill + decode)
# ---------------------------------------------------------------------------
def test_reuse_shape_prefill_decode():
    coord = XQuantCoordinator()
    a, r = _pair(coord)
    k = _rand(1, 2, 32, 64)
    v = _rand(1, 2, 32, 64, seed=1)
    a.update_and_fetch(k, v)
    ko, vo = r.update_and_fetch(k, v)
    assert ko.shape == (1, 2, 32, 64) and vo.shape == (1, 2, 32, 64)
    # decode step
    kd = _rand(1, 2, 1, 64, seed=5)
    vd = _rand(1, 2, 1, 64, seed=6)
    a.update_and_fetch(kd, vd)
    ko2, _ = r.update_and_fetch(kd, vd)
    assert ko2.shape == (1, 2, 33, 64)


# ---------------------------------------------------------------------------
# Test 6 — values reconstructed (shape + finite)
# ---------------------------------------------------------------------------
def test_values_reconstructed():
    coord = XQuantCoordinator()
    a, r = _pair(coord)
    k = _rand(1, 2, 32, 64)
    v = _rand(1, 2, 32, 64, seed=1)
    a.update_and_fetch(k, v)
    _, vo = r.update_and_fetch(k, v)
    assert vo.shape == v.shape
    assert bool(mx.all(mx.isfinite(vo)).item())


# ---------------------------------------------------------------------------
# Test 7 — reuse residual=0 within tolerance of own quant
# ---------------------------------------------------------------------------
def test_reuse_residual0_within_tolerance():
    coord = XQuantCoordinator()
    a, r = _pair(coord, base_bits=2, residual_bits=0)
    rng = np.random.default_rng(7)
    base = rng.standard_normal((1, 2, 64, 64)).astype(np.float32)
    k = mx.array(base.astype(np.float16))
    a.update_and_fetch(k, k)
    ko, _ = r.update_and_fetch(k, k)  # identical layer → reuse is essentially own quant
    self_q = _group_quant_dequant(k[0, 0], 2, 32)
    mse_reuse = _mse(ko[0, 0], k[0, 0])
    mse_self = _mse(self_q, k[0, 0])
    # When the reuse layer's data == anchor's, shared codes reconstruct as well
    # as self-quant (within fp16 rounding).
    assert mse_reuse <= mse_self * 1.05 + 1e-4, (
        f"reuse MSE {mse_reuse:.5f} should be ~= self-quant MSE {mse_self:.5f}"
    )


# ---------------------------------------------------------------------------
# Test 8 — residual>0 strictly lower MSE than residual=0 (correlated layers)
# ---------------------------------------------------------------------------
def test_residual_lowers_mse_correlated():
    rng = np.random.default_rng(8)
    base = rng.standard_normal((1, 2, 64, 64)).astype(np.float32)
    reuse_keys = (base + 0.02 * rng.standard_normal(base.shape)).astype(np.float16)

    def run(residual_bits):
        coord = XQuantCoordinator()
        a, r = _pair(coord, base_bits=2, residual_bits=residual_bits)
        a.update_and_fetch(mx.array(base.astype(np.float16)), mx.array(base.astype(np.float16)))
        ko, _ = r.update_and_fetch(mx.array(reuse_keys), mx.array(reuse_keys))
        return _mse(ko, mx.array(reuse_keys))

    assert run(2) < run(0), "residual=2 should lower MSE vs residual=0 on correlated layers"


# ---------------------------------------------------------------------------
# Test 9 — correlated pairs: reuse MSE ~= self-quant MSE (reuse near-free)
# ---------------------------------------------------------------------------
def test_correlated_reuse_near_self_quant():
    rng = np.random.default_rng(9)
    base = rng.standard_normal((1, 2, 64, 64)).astype(np.float32)
    reuse_keys = (base + 0.02 * rng.standard_normal(base.shape)).astype(np.float16)

    coord = XQuantCoordinator()
    a, r = _pair(coord, base_bits=2, residual_bits=0)
    a.update_and_fetch(mx.array(base.astype(np.float16)), mx.array(base.astype(np.float16)))
    ko, _ = r.update_and_fetch(mx.array(reuse_keys), mx.array(reuse_keys))
    mse_reuse = _mse(ko, mx.array(reuse_keys))

    self_q = _group_quant_dequant(mx.array(reuse_keys)[0, 0], 2, 32)
    mse_self = _mse(self_q, mx.array(reuse_keys)[0, 0])
    # Reuse over a near-identical anchor costs almost nothing vs self-quant.
    assert mse_reuse < mse_self * 1.5, (
        f"reuse MSE {mse_reuse:.5f} should be close to self-quant {mse_self:.5f}"
    )


# ---------------------------------------------------------------------------
# Test 10 — uncorrelated pairs: residual recovers quality (negative control)
# ---------------------------------------------------------------------------
def test_uncorrelated_residual_recovers():
    rng = np.random.default_rng(10)
    base = rng.standard_normal((1, 2, 64, 64)).astype(np.float32)
    reuse_keys = (base + 1.0 * rng.standard_normal(base.shape)).astype(np.float16)  # uncorrelated

    def run(residual_bits):
        coord = XQuantCoordinator()
        a, r = _pair(coord, base_bits=2, residual_bits=residual_bits)
        a.update_and_fetch(mx.array(base.astype(np.float16)), mx.array(base.astype(np.float16)))
        ko, _ = r.update_and_fetch(mx.array(reuse_keys), mx.array(reuse_keys))
        return _mse(ko, mx.array(reuse_keys))

    mse_no_res = run(0)
    mse_res4 = run(4)
    # A sufficient residual recovers quality even when layers are uncorrelated.
    assert mse_res4 < mse_no_res * 0.1, (
        f"residual=4 ({mse_res4:.5f}) should recover quality vs residual=0 ({mse_no_res:.5f})"
    )


# ---------------------------------------------------------------------------
# Test 11 — byte accounting: reuse << anchor
# ---------------------------------------------------------------------------
def test_byte_accounting_reuse_less_than_anchor():
    coord = XQuantCoordinator()
    a, r = _pair(coord)
    k = _rand(1, 2, 64, 64)
    v = _rand(1, 2, 64, 64, seed=1)
    a.update_and_fetch(k, v)
    r.update_and_fetch(k, v)
    assert r.compressed_key_bytes < a.compressed_key_bytes
    assert a.compressed_key_bytes < a.fp16_key_bytes


# ---------------------------------------------------------------------------
# Test 12 — effective_pair_bits < base bits (reuse layer)
# ---------------------------------------------------------------------------
def test_effective_pair_bits_below_base():
    coord = XQuantCoordinator()
    a, r = _pair(coord, base_bits=2, residual_bits=0)
    k = _rand(1, 2, 64, 64)
    v = _rand(1, 2, 64, 64, seed=1)
    a.update_and_fetch(k, v)
    r.update_and_fetch(k, v)
    assert r.effective_pair_bits < 2.0, f"reuse effective bits {r.effective_pair_bits}"


# ---------------------------------------------------------------------------
# Test 13 — decode synchronization over multiple steps
# ---------------------------------------------------------------------------
def test_decode_synchronization():
    coord = XQuantCoordinator()
    a, r = _pair(coord)
    a.update_and_fetch(_rand(1, 2, 20, 64), _rand(1, 2, 20, 64, seed=1))
    r.update_and_fetch(_rand(1, 2, 20, 64), _rand(1, 2, 20, 64, seed=1))
    for step in range(5):
        kd = _rand(1, 2, 1, 64, seed=100 + step)
        vd = _rand(1, 2, 1, 64, seed=200 + step)
        a.update_and_fetch(kd, vd)
        ko, _ = r.update_and_fetch(kd, vd)
        expected_S = 20 + step + 1
        assert ko.shape[2] == expected_S
    assert coord.published_tokens(0) == 25


# ---------------------------------------------------------------------------
# Test 14 — coordinator token budget raises
# ---------------------------------------------------------------------------
def test_coordinator_budget_raises():
    coord = XQuantCoordinator(max_ctx=16)
    a, _ = _pair(coord)
    a.update_and_fetch(_rand(1, 2, 16, 64), _rand(1, 2, 16, 64, seed=1))  # fills budget
    with pytest.raises(RuntimeError, match="max_ctx"):
        a.update_and_fetch(_rand(1, 2, 1, 64, seed=2), _rand(1, 2, 1, 64, seed=3))


# ---------------------------------------------------------------------------
# Test 15 — group_size=3 pairing + reconstruction
# ---------------------------------------------------------------------------
def test_group_size_three():
    assert pair_layers(6, 3) == [
        ("anchor", 0),
        ("reuse", 0),
        ("reuse", 0),
        ("anchor", 1),
        ("reuse", 1),
        ("reuse", 1),
    ]
    model = _MockModel(n_layers=6, head_dim=64)
    caches = KVCacheBuilder.for_model(model, _cfg(xquant_group_size=3))
    # one anchor feeds two reusers in group 0
    k = _rand(1, 2, 32, 64)
    v = _rand(1, 2, 32, 64, seed=1)
    out0, _ = caches[0].update_and_fetch(k, v)  # anchor
    out1, _ = caches[1].update_and_fetch(k, v)  # reuse
    out2, _ = caches[2].update_and_fetch(k, v)  # reuse
    assert out0.shape == out1.shape == out2.shape == (1, 2, 32, 64)
    assert caches[1].role == "reuse" and caches[2].role == "reuse"


# ---------------------------------------------------------------------------
# Test 16 — determinism
# ---------------------------------------------------------------------------
def test_determinism():
    k = _rand(1, 2, 32, 64, seed=77)
    v = _rand(1, 2, 32, 64, seed=88)

    def run():
        coord = XQuantCoordinator()
        a, r = _pair(coord)
        a.update_and_fetch(k, v)
        ko, vo = r.update_and_fetch(k, v)
        return np.array(ko.tolist()), np.array(vo.tolist())

    k1, v1 = run()
    k2, v2 = run()
    np.testing.assert_array_equal(k1, k2)
    np.testing.assert_array_equal(v1, v2)
