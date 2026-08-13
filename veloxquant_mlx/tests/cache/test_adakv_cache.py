"""Tests for AdaKVCache — per-head adaptive bit allocation over KIVI.

14 tests covering:
  1.  Factory dispatch via KVCacheFactory
  2.  Output shape preserved after prefill
  3.  Output shape preserved after decode
  4.  Values unchanged (AdaKV-proxy compresses keys only)
  5.  High-importance heads get more bits than low-importance heads
  6.  Average bits equals target (within ±0.5 due to rounding)
  7.  lo_bit-only when all heads equal importance — degrades to uniform target
  8.  MSE lower on the high-importance head than if it had been given lo_bit
  9.  Running norm accumulator correctness vs ground-truth variance
  10. Decode after prefill — sequential accumulation produces correct shape
  11. Byte accounting — compressed_key_bytes < fp16_key_bytes
  12. assigned_avg_bits within [lo_bit, hi_bit]
  13. Single-head model — trivially assigns target_avg_bits (snapped)
  14. Determinism — identical inputs produce identical outputs
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.adakv_cache import AdaKVCache
from veloxquant_mlx.quantizers.adakv import (
    allocate_head_bits,
    compute_head_norm_variance,
    quantize_head,
)
from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_cfg(**kwargs) -> KVCacheConfig:
    defaults = dict(method="adakv", head_dim=64)
    defaults.update(kwargs)
    return KVCacheConfig(**defaults)


def _keys(B=1, H=4, S=32, D=64, seed=0) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))


def _values(B=1, H=4, S=32, D=64, seed=1) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))


def _heterogeneous_keys(B=1, H=4, S=64, D=64, seed=3) -> mx.array:
    """Keys where head importance (inter-token norm variance) increases with h.

    Head h gets its per-token norm scaled by an h-dependent jittered factor,
    so higher heads have larger inter-token norm variance.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((B, H, S, D)).astype(np.float32)
    for h in range(H):
        # Per-token magnitude scale; spread grows with h → norm variance grows.
        spread = 0.05 + h * 0.6
        scale = (1.0 + spread * rng.standard_normal((B, S, 1))).astype(np.float32)
        data[:, h, :, :] = data[:, h, :, :] * np.abs(scale)
    return mx.array(data.astype(np.float16))


# ---------------------------------------------------------------------------
# Test 1 — factory dispatch
# ---------------------------------------------------------------------------
def test_factory_dispatch():
    cache = KVCacheFactory.create(_make_cfg())
    assert isinstance(cache, AdaKVCache)


# ---------------------------------------------------------------------------
# Test 2 — output shape after prefill
# ---------------------------------------------------------------------------
def test_output_shape_prefill():
    cache = AdaKVCache(_make_cfg())
    k = _keys(B=1, H=4, S=32, D=64)
    v = _values(B=1, H=4, S=32, D=64)
    k_out, v_out = cache.update_and_fetch(k, v)
    assert k_out.shape == (1, 4, 32, 64)
    assert v_out.shape == (1, 4, 32, 64)


# ---------------------------------------------------------------------------
# Test 3 — output shape after decode
# ---------------------------------------------------------------------------
def test_output_shape_decode():
    cache = AdaKVCache(_make_cfg())
    cache.update_and_fetch(_keys(B=1, H=4, S=16, D=64), _values(B=1, H=4, S=16, D=64))
    k_dec = _keys(B=1, H=4, S=1, D=64, seed=99)
    v_dec = _values(B=1, H=4, S=1, D=64, seed=100)
    k_out, v_out = cache.update_and_fetch(k_dec, v_dec)
    assert k_out.shape == (1, 4, 17, 64)
    assert v_out.shape == (1, 4, 17, 64)


# ---------------------------------------------------------------------------
# Test 4 — values unchanged
# ---------------------------------------------------------------------------
def test_values_unchanged():
    cache = AdaKVCache(_make_cfg())
    k = _keys()
    v = _values()
    _, v_out = cache.update_and_fetch(k, v)
    assert np.allclose(
        np.array(v_out[0, 0, :, :].tolist()),
        np.array(v[0, 0, :, :].tolist()),
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# Test 5 — high-importance heads get more bits than low-importance heads
# ---------------------------------------------------------------------------
def test_high_importance_heads_get_more_bits():
    cache = AdaKVCache(_make_cfg(adakv_target_avg_bits=3.0, head_dim=64))
    k = _heterogeneous_keys(B=1, H=4, S=64, D=64)
    v = _values(B=1, H=4, S=64, D=64)
    cache.update_and_fetch(k, v)
    bits = cache.head_bits
    # Head 0 (lowest norm variance) should not exceed head 3 (highest).
    assert bits[3] >= bits[0], f"head_bits={bits} — high-importance head got fewer bits"
    assert bits[3] > bits[0], (
        f"head_bits={bits} — expected strictly more bits on the high-importance head"
    )


# ---------------------------------------------------------------------------
# Test 6 — average bits ≈ target (within ±0.5)
# ---------------------------------------------------------------------------
def test_average_bits_matches_target():
    for target in (2.0, 2.5, 3.0):
        cache = AdaKVCache(_make_cfg(adakv_target_avg_bits=target, head_dim=64))
        k = _heterogeneous_keys(B=1, H=8, S=64, D=64, seed=11)
        v = _values(B=1, H=8, S=64, D=64, seed=12)
        cache.update_and_fetch(k, v)
        assert abs(cache.assigned_avg_bits - target) <= 0.5, (
            f"target={target}, assigned={cache.assigned_avg_bits}, bits={cache.head_bits}"
        )


# ---------------------------------------------------------------------------
# Test 7 — equal importance degrades to uniform target allocation
# ---------------------------------------------------------------------------
def test_equal_importance_uniform_allocation():
    # All heads equal importance → uniform target. With target == lo_bit, every
    # head should snap to lo_bit.
    bits = allocate_head_bits(
        head_importance=[1.0, 1.0, 1.0, 1.0],
        target_avg_bits=2.0,
        allowed_bits=[2, 3, 4],
        n_heads=4,
    )
    assert bits == [2, 2, 2, 2], f"expected all lo_bit, got {bits}"

    # All-zero importance → also uniform.
    bits_zero = allocate_head_bits(
        head_importance=[0.0, 0.0, 0.0, 0.0],
        target_avg_bits=2.0,
        allowed_bits=[2, 3, 4],
        n_heads=4,
    )
    assert bits_zero == [2, 2, 2, 2], f"expected all lo_bit, got {bits_zero}"


# ---------------------------------------------------------------------------
# Test 8 — high-importance head: assigned bits give lower MSE than lo_bit
# ---------------------------------------------------------------------------
def test_high_importance_head_lower_mse_than_lo_bit():
    cache = AdaKVCache(_make_cfg(adakv_target_avg_bits=3.0, head_dim=64))
    k = _heterogeneous_keys(B=1, H=4, S=64, D=64, seed=21)
    v = _values(B=1, H=4, S=64, D=64, seed=22)
    cache.update_and_fetch(k, v)

    bits = cache.head_bits
    hi_head = int(np.argmax(cache.head_importance))
    assert bits[hi_head] > cache._lo_bit, (
        f"high-importance head {hi_head} got only {bits[hi_head]} bits; bits={bits}"
    )

    orig = np.array(k[0, hi_head].tolist())
    recon_assigned = np.array(quantize_head(k[0, hi_head], bits[hi_head], 32).tolist())
    recon_lo = np.array(_group_quant_dequant(k[0, hi_head], cache._lo_bit, 32).tolist())

    mse_assigned = float(np.mean((orig - recon_assigned) ** 2))
    mse_lo = float(np.mean((orig - recon_lo) ** 2))
    assert mse_assigned < mse_lo, (
        f"assigned-bit MSE {mse_assigned:.6f} should be < lo_bit MSE {mse_lo:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 9 — running norm accumulator correctness vs ground truth
# ---------------------------------------------------------------------------
def test_running_norm_accumulator_correctness():
    H, S, D = 3, 50, 64
    rng = np.random.default_rng(33)
    data = rng.standard_normal((1, H, S, D)).astype(np.float32)
    keys = mx.array(data)

    # Ground-truth inter-token norm variance per head.
    norms = np.sqrt((data[0] ** 2).sum(axis=-1))  # [H, S]
    gt_var = norms.var(axis=1)  # [H]

    cache = AdaKVCache(_make_cfg(head_dim=D))
    cache._update_norm_accumulators(keys)
    acc_var = np.array(cache.head_importance)

    np.testing.assert_allclose(acc_var, gt_var, rtol=1e-3, atol=1e-3)

    # And the standalone quantizer helper agrees.
    direct = np.array(compute_head_norm_variance(keys).tolist())
    np.testing.assert_allclose(direct, gt_var, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Test 10 — decode after prefill accumulates shapes correctly
# ---------------------------------------------------------------------------
def test_decode_after_prefill_accumulates():
    cache = AdaKVCache(_make_cfg())
    cache.update_and_fetch(_keys(B=1, H=4, S=20, D=64), _values(B=1, H=4, S=20, D=64))
    for step in range(5):
        k_dec = _keys(B=1, H=4, S=1, D=64, seed=200 + step)
        v_dec = _values(B=1, H=4, S=1, D=64, seed=300 + step)
        k_out, v_out = cache.update_and_fetch(k_dec, v_dec)
        expected_S = 20 + step + 1
        assert k_out.shape[2] == expected_S
        assert v_out.shape[2] == expected_S
    # Accumulated token count reflects prefill + decode steps.
    assert cache._n_tokens == 25


# ---------------------------------------------------------------------------
# Test 11 — byte accounting: compressed < fp16
# ---------------------------------------------------------------------------
def test_byte_accounting_compressed_less_than_fp16():
    cache = AdaKVCache(_make_cfg())
    cache.update_and_fetch(_keys(B=1, H=4, S=64, D=64), _values(B=1, H=4, S=64, D=64))
    assert cache.compressed_key_bytes < cache.fp16_key_bytes, (
        f"compressed={cache.compressed_key_bytes} should be < fp16={cache.fp16_key_bytes}"
    )


# ---------------------------------------------------------------------------
# Test 12 — assigned_avg_bits within [lo_bit, hi_bit]
# ---------------------------------------------------------------------------
def test_assigned_avg_bits_in_range():
    cache = AdaKVCache(_make_cfg(adakv_target_avg_bits=3.0))
    cache.update_and_fetch(_heterogeneous_keys(B=1, H=4, S=64, D=64), _values(B=1, H=4, S=64, D=64))
    avg = cache.assigned_avg_bits
    assert cache._lo_bit <= avg <= cache._hi_bit, (
        f"assigned_avg_bits={avg} out of range [{cache._lo_bit}, {cache._hi_bit}]"
    )


# ---------------------------------------------------------------------------
# Test 13 — single-head model trivially assigns target (snapped)
# ---------------------------------------------------------------------------
def test_single_head_assigns_target():
    cache = AdaKVCache(_make_cfg(adakv_target_avg_bits=3.0, head_dim=64))
    cache.update_and_fetch(_keys(B=1, H=1, S=32, D=64), _values(B=1, H=1, S=32, D=64))
    # Target 3.0 snaps exactly to allowed mid_bit=3.
    assert cache.head_bits == [3], f"single-head bits={cache.head_bits}"

    # A non-allowed target snaps to the nearest allowed value.
    bits = allocate_head_bits([5.0], target_avg_bits=2.4, allowed_bits=[2, 3, 4], n_heads=1)
    assert bits == [2], f"expected [2], got {bits}"


# ---------------------------------------------------------------------------
# Test 14 — determinism
# ---------------------------------------------------------------------------
def test_determinism():
    k = _heterogeneous_keys(B=1, H=4, S=32, D=64, seed=77)
    v = _values(B=1, H=4, S=32, D=64, seed=88)

    cache1 = AdaKVCache(_make_cfg())
    k_out1, v_out1 = cache1.update_and_fetch(k, v)
    cache2 = AdaKVCache(_make_cfg())
    k_out2, v_out2 = cache2.update_and_fetch(k, v)

    np.testing.assert_array_equal(np.array(k_out1.tolist()), np.array(k_out2.tolist()))
    np.testing.assert_array_equal(np.array(v_out1.tolist()), np.array(v_out2.tolist()))
    assert cache1.head_bits == cache2.head_bits
