"""Tests for KittyKVCache — dynamic channel-wise mixed-precision key quantization.

14 tests covering:
  1.  Factory dispatch via KVCacheFactory
  2.  Output shape preserved after prefill
  3.  Output shape preserved after decode
  4.  Values unchanged (Kitty compresses keys only)
  5.  Channel ranking — high-variance channels are correctly identified
  6.  Hi channels incur lower quantization error than lo channels (more bits)
  7.  MSE on high-variance synthetic data — Kitty outperforms uniform 2-bit
  8.  Running variance accumulator correctness after multiple decode steps
  9.  Decode after prefill — sequential key accumulation produces correct shape
  10. Byte accounting — compressed_key_bytes < fp16_key_bytes
  11. assigned_avg_bits is in [2.0, 4.0] at default settings
  12. hi_fraction=0.0 degrades to uniform lo_bit (all channels lo)
  13. hi_fraction=1.0 degrades to uniform hi_bit (all channels hi)
  14. Determinism — identical inputs produce identical outputs
"""

from __future__ import annotations

import pytest
import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.kitty_cache import KittyKVCache
from veloxquant_mlx.quantizers.kitty import (
    compute_running_variance,
    rank_channels_by_sensitivity,
    quantize_mixed_channels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_cfg(**kwargs) -> KVCacheConfig:
    defaults = dict(method="kitty", head_dim=64)
    defaults.update(kwargs)
    return KVCacheConfig(**defaults)


def _keys(B=1, H=2, S=32, D=64, seed=0) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))


def _values(B=1, H=2, S=32, D=64, seed=1) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((B, H, S, D)).astype(np.float16))


# ---------------------------------------------------------------------------
# Test 1 — factory dispatch
# ---------------------------------------------------------------------------
def test_factory_dispatch():
    cfg = _make_cfg()
    cache = KVCacheFactory.create(cfg)
    assert isinstance(cache, KittyKVCache)


# ---------------------------------------------------------------------------
# Test 2 — output shape after prefill
# ---------------------------------------------------------------------------
def test_output_shape_prefill():
    cache = KittyKVCache(_make_cfg())
    k = _keys(B=1, H=2, S=32, D=64)
    v = _values(B=1, H=2, S=32, D=64)
    k_out, v_out = cache.update_and_fetch(k, v)
    assert k_out.shape == (1, 2, 32, 64)
    assert v_out.shape == (1, 2, 32, 64)


# ---------------------------------------------------------------------------
# Test 3 — output shape after decode
# ---------------------------------------------------------------------------
def test_output_shape_decode():
    cache = KittyKVCache(_make_cfg())
    k_pre = _keys(B=1, H=2, S=16, D=64)
    v_pre = _values(B=1, H=2, S=16, D=64)
    cache.update_and_fetch(k_pre, v_pre)
    # Decode step: S=1
    k_dec = _keys(B=1, H=2, S=1, D=64, seed=99)
    v_dec = _values(B=1, H=2, S=1, D=64, seed=100)
    k_out, v_out = cache.update_and_fetch(k_dec, v_dec)
    assert k_out.shape == (1, 2, 17, 64)
    assert v_out.shape == (1, 2, 17, 64)


# ---------------------------------------------------------------------------
# Test 4 — values unchanged
# ---------------------------------------------------------------------------
def test_values_unchanged():
    cache = KittyKVCache(_make_cfg())
    k = _keys()
    v = _values()
    _, v_out = cache.update_and_fetch(k, v)
    # Values should be identical (fp16 passthrough)
    assert np.allclose(
        np.array(v_out[0, 0, :, :].tolist()),
        np.array(v[0, 0, :, :].tolist()),
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# Test 5 — channel ranking identifies high-variance channels
# ---------------------------------------------------------------------------
def test_channel_ranking_selects_high_variance():
    D = 64
    S = 128
    rng = np.random.default_rng(7)
    # First 8 channels have 10× higher variance
    data = rng.standard_normal((S, D)).astype(np.float32)
    data[:, :8] *= 10.0
    keys = mx.array(data)
    hi_idx, lo_idx = rank_channels_by_sensitivity(keys, hi_fraction=0.125)
    # All 8 high-variance channels should land in hi_idx
    assert set(range(8)).issubset(set(hi_idx)), (
        f"Expected high-variance channels 0-7 in hi_idx, got {hi_idx[:10]}"
    )


# ---------------------------------------------------------------------------
# Test 6 — hi channels have lower quant error than lo channels
# ---------------------------------------------------------------------------
def test_hi_channels_lower_error_than_lo():
    D = 64
    S = 64
    rng = np.random.default_rng(42)
    data = rng.standard_normal((S, D)).astype(np.float16)
    keys = mx.array(data)

    hi_fraction = 0.25
    hi_idx, lo_idx = rank_channels_by_sensitivity(keys, hi_fraction=hi_fraction)
    k_q = quantize_mixed_channels(keys, hi_idx, lo_idx, hi_bit=4, lo_bit=2, group_size=32)

    orig = np.array(data)
    recon = np.array(k_q.tolist())

    mse_hi = float(np.mean((orig[:, hi_idx] - recon[:, hi_idx]) ** 2))
    mse_lo = float(np.mean((orig[:, lo_idx] - recon[:, lo_idx]) ** 2))
    # 4-bit hi should have lower error than 2-bit lo
    assert mse_hi < mse_lo, f"Expected mse_hi({mse_hi:.6f}) < mse_lo({mse_lo:.6f})"


# ---------------------------------------------------------------------------
# Test 7 — Kitty MSE < uniform 2-bit on high-variance data
# ---------------------------------------------------------------------------
def test_kitty_better_mse_than_uniform_2bit_on_high_variance_data():
    D = 64
    S = 128
    rng = np.random.default_rng(13)
    # High variance in first 16 channels — these benefit from 4-bit
    data = rng.standard_normal((S, D)).astype(np.float32)
    data[:, :16] *= 8.0
    keys = mx.array(data.astype(np.float16))

    hi_idx, lo_idx = rank_channels_by_sensitivity(keys, hi_fraction=0.25)
    k_kitty = quantize_mixed_channels(keys, hi_idx, lo_idx, hi_bit=4, lo_bit=2, group_size=32)

    from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant

    k_uniform = _group_quant_dequant(keys, b=2, group_size=32)

    orig = np.array(data)
    mse_kitty = float(np.mean((orig - np.array(k_kitty.tolist())) ** 2))
    mse_uniform = float(np.mean((orig - np.array(k_uniform.tolist())) ** 2))
    assert mse_kitty < mse_uniform, (
        f"Kitty MSE {mse_kitty:.6f} should be < uniform 2-bit MSE {mse_uniform:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 8 — running variance accumulator correctness
# ---------------------------------------------------------------------------
def test_running_variance_accumulator():
    D = 32
    S = 64
    rng = np.random.default_rng(55)
    data = rng.standard_normal((S, D)).astype(np.float32)

    # Ground truth variance
    gt_var = np.var(data, axis=0)

    # Simulate accumulator
    key_sum = mx.zeros((D,), dtype=mx.float32)
    key_sq_sum = mx.zeros((D,), dtype=mx.float32)
    n = 0
    for i in range(S):
        row = mx.array(data[i])
        key_sum = key_sum + row
        key_sq_sum = key_sq_sum + row * row
        n += 1

    var_acc = compute_running_variance(key_sum, key_sq_sum, n)
    var_arr = np.array(var_acc.tolist())
    # Should match ground truth to within float32 rounding
    np.testing.assert_allclose(var_arr, gt_var, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 9 — decode after prefill accumulates shapes correctly
# ---------------------------------------------------------------------------
def test_decode_after_prefill_accumulates():
    cache = KittyKVCache(_make_cfg())
    k_pre = _keys(B=1, H=2, S=20, D=64)
    v_pre = _values(B=1, H=2, S=20, D=64)
    cache.update_and_fetch(k_pre, v_pre)

    for step in range(5):
        k_dec = _keys(B=1, H=2, S=1, D=64, seed=200 + step)
        v_dec = _values(B=1, H=2, S=1, D=64, seed=300 + step)
        k_out, v_out = cache.update_and_fetch(k_dec, v_dec)
        expected_S = 20 + step + 1
        assert k_out.shape[2] == expected_S, (
            f"Step {step}: expected S={expected_S}, got {k_out.shape[2]}"
        )
        assert v_out.shape[2] == expected_S


# ---------------------------------------------------------------------------
# Test 10 — byte accounting: compressed < fp16
# ---------------------------------------------------------------------------
def test_byte_accounting_compressed_less_than_fp16():
    cache = KittyKVCache(_make_cfg())
    k = _keys(B=1, H=2, S=64, D=64)
    v = _values(B=1, H=2, S=64, D=64)
    cache.update_and_fetch(k, v)
    assert cache.compressed_key_bytes < cache.fp16_key_bytes, (
        f"compressed={cache.compressed_key_bytes} should be < fp16={cache.fp16_key_bytes}"
    )


# ---------------------------------------------------------------------------
# Test 11 — assigned_avg_bits in [2.0, 4.0] at defaults
# ---------------------------------------------------------------------------
def test_assigned_avg_bits_in_range():
    cache = KittyKVCache(_make_cfg())
    avg = cache.assigned_avg_bits
    assert 2.0 <= avg <= 4.0, f"assigned_avg_bits={avg} out of range [2.0, 4.0]"


# ---------------------------------------------------------------------------
# Test 12 — hi_fraction=0.0 → nearly all channels at lo_bit (≤1 hi channel)
# ---------------------------------------------------------------------------
def test_hi_fraction_zero_nearly_all_lo_bit():
    D = 64
    cfg = _make_cfg(kitty_hi_fraction=0.0, kitty_hi_bit=4, kitty_lo_bit=2, head_dim=D)
    cache = KittyKVCache(cfg)
    k = _keys(B=1, H=1, S=32, D=D)
    v = _values(B=1, H=1, S=32, D=D)
    cache.update_and_fetch(k, v)
    # With hi_fraction=0.0, n_hi = max(1, int(D * 0.0)) = 1 (guard prevents 0).
    # avg_bits should be very close to lo_bit (1 channel out of 64 at hi_bit).
    avg = cache.assigned_avg_bits
    # (1*4 + 63*2) / 64 = 2.03125
    expected = (1 * 4 + (D - 1) * 2) / D
    assert abs(avg - expected) < 1e-6, f"avg_bits={avg}, expected≈{expected}"


# ---------------------------------------------------------------------------
# Test 13 — hi_fraction=1.0 → all channels at hi_bit
# ---------------------------------------------------------------------------
def test_hi_fraction_one_all_hi_bit():
    cfg = _make_cfg(kitty_hi_fraction=1.0, kitty_hi_bit=4, kitty_lo_bit=2)
    cache = KittyKVCache(cfg)
    k = _keys(B=1, H=1, S=32, D=64)
    v = _values(B=1, H=1, S=32, D=64)
    k_out, _ = cache.update_and_fetch(k, v)

    from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant

    k_ref = _group_quant_dequant(k[0, 0], b=4, group_size=32)

    out_np = np.array(k_out[0, 0].tolist())
    ref_np = np.array(k_ref.tolist())
    np.testing.assert_allclose(out_np, ref_np, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Test 14 — determinism
# ---------------------------------------------------------------------------
def test_determinism():
    k = _keys(B=1, H=2, S=32, D=64, seed=77)
    v = _values(B=1, H=2, S=32, D=64, seed=88)

    cache1 = KittyKVCache(_make_cfg())
    k_out1, v_out1 = cache1.update_and_fetch(k, v)

    cache2 = KittyKVCache(_make_cfg())
    k_out2, v_out2 = cache2.update_and_fetch(k, v)

    np.testing.assert_array_equal(np.array(k_out1.tolist()), np.array(k_out2.tolist()))
    np.testing.assert_array_equal(np.array(v_out1.tolist()), np.array(v_out2.tolist()))


# ---------------------------------------------------------------------------
# Config validation — kitty_hi_fraction must be in [0, 1]
# ---------------------------------------------------------------------------


def test_hi_fraction_above_one_rejected() -> None:
    """kitty_hi_fraction > 1.0 makes n_hi > D, so n_lo = D - n_hi goes
    negative and silently corrupts byte accounting instead of erroring."""
    with pytest.raises(ValueError, match="kitty_hi_fraction"):
        KittyKVCache(_make_cfg(kitty_hi_fraction=1.5))


def test_hi_fraction_negative_rejected() -> None:
    with pytest.raises(ValueError, match="kitty_hi_fraction"):
        KittyKVCache(_make_cfg(kitty_hi_fraction=-0.2))
