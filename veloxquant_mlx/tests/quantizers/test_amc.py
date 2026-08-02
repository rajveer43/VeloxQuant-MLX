"""Tests for the AMC-adapted quantizer core — saliency, tiering, rank
masking, quantization, and closed-loop threshold adaptation.

AMC-adapted (arXiv:2607.10109, no verified venue) assigns each token a tier
(High/Mid/Low) from an L1-norm saliency score, then applies per-tier rank
masking + quantization. Unlike every eviction method in this repo, no token
is ever dropped. All data is synthetic.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.amc import (
    HIGH,
    LOW,
    MID,
    AMC_TIERS,
    amc_adaptive_thresholds,
    amc_apply_rank_mask,
    amc_assign_tiers,
    amc_fp16_bytes,
    amc_pack_low_tier,
    amc_quantize_tier,
    amc_query_aware_saliency,
    amc_saliency,
    full_amc_fp16_bytes,
    init_amc_threshold_state,
)


def _mat(rows, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(np.array(rows, dtype=np.float32)) if isinstance(rows, list) else None


# ---------------------------------------------------------------------------
# Saliency — Eq. 1-2
# ---------------------------------------------------------------------------


def test_saliency_matches_l1_norm_definition() -> None:
    x = mx.array(np.array([[1.0, -1.0, 1.0, -1.0], [0.1, 0.1, 0.1, 0.1]], dtype=np.float32))
    s = amc_saliency(x)
    assert s.tolist() == pytest.approx([1.0, 0.1], abs=1e-6)


def test_saliency_clamped_to_unit_interval() -> None:
    x = mx.array(np.array([[10.0, 10.0], [0.0, 0.0]], dtype=np.float32))
    s = amc_saliency(x)
    assert float(s[0].item()) <= 1.0
    assert float(s[1].item()) >= 0.0


def test_query_aware_saliency_downweights_high_magnitude_irrelevant_tokens() -> None:
    # Token 0: high magnitude, key orthogonal to query (irrelevant).
    # Token 1: moderate magnitude, key parallel to query (relevant).
    d = 8
    x = np.zeros((2, d), dtype=np.float32)
    x[0, :] = 0.9  # high |x| (within [0, 1] so clamping doesn't erase the gap)
    x[1, :] = 0.3  # moderate |x|
    keys = np.zeros((2, d), dtype=np.float32)
    keys[0, 0] = 1.0  # orthogonal-ish to query below
    keys[1, :] = 1.0  # parallel to query
    query = np.ones(d, dtype=np.float32)

    x_mx = mx.array(x)
    k_mx = mx.array(keys)
    q_mx = mx.array(query)

    mag_only = amc_saliency(x_mx)
    query_aware = amc_query_aware_saliency(x_mx, k_mx, q_mx, alpha=0.3)

    # Under magnitude-only scoring token 0 > token 1.
    assert float(mag_only[0].item()) > float(mag_only[1].item())
    # Under query-aware scoring (alpha=0.3, semantic term dominates), the gap
    # should shrink or reverse because token 1's key is far more aligned with
    # the query than token 0's.
    gap_mag = float(mag_only[0].item()) - float(mag_only[1].item())
    gap_query = float(query_aware[0].item()) - float(query_aware[1].item())
    assert gap_query < gap_mag


def test_query_aware_saliency_guards_zero_norm_key() -> None:
    x = mx.array(np.array([[1.0, 1.0]], dtype=np.float32))
    keys = mx.array(np.array([[0.0, 0.0]], dtype=np.float32))  # zero-norm key
    query = mx.array(np.array([1.0, 0.0], dtype=np.float32))
    s = amc_query_aware_saliency(x, keys, query, alpha=0.5)
    assert not math.isnan(float(s[0].item()))


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------


def test_tier_assignment_respects_percentiles() -> None:
    n = 1000
    rng = np.random.default_rng(9)
    saliency = mx.array(rng.uniform(0, 1, size=n).astype(np.float32))
    tiers = amc_assign_tiers(saliency, k_high=0.20, k_mid=0.30)
    counts = {HIGH: 0, MID: 0, LOW: 0}
    for t in tiers:
        counts[t] += 1
    assert counts[HIGH] == pytest.approx(200, abs=5)
    assert counts[MID] == pytest.approx(300, abs=5)
    assert counts[LOW] == pytest.approx(500, abs=10)


def test_high_tier_tokens_survive_full_precision() -> None:
    # Construct clearly separated saliency values.
    saliency = mx.array(
        np.array([0.9, 0.85, 0.5, 0.5, 0.5, 0.1, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    )
    tiers = amc_assign_tiers(saliency, k_high=0.20, k_mid=0.30)
    # Top-2 (indices 0, 1) must be HIGH.
    assert tiers[0] == HIGH
    assert tiers[1] == HIGH
    # Bottom scorers must be LOW.
    assert tiers[-1] == LOW
    assert tiers[-2] == LOW


def test_tier_assignment_empty_input() -> None:
    saliency = mx.array(np.array([], dtype=np.float32))
    tiers = amc_assign_tiers(saliency)
    assert tiers == []


def test_tier_assignment_single_token() -> None:
    saliency = mx.array(np.array([0.5], dtype=np.float32))
    tiers = amc_assign_tiers(saliency, k_high=0.20, k_mid=0.30)
    assert len(tiers) == 1
    assert tiers[0] in (HIGH, MID, LOW)


# ---------------------------------------------------------------------------
# Rank masking — Eq. 6
# ---------------------------------------------------------------------------


def test_rank_mask_zeros_tail_channels() -> None:
    x = mx.array(np.ones((2, 8), dtype=np.float32))
    out = amc_apply_rank_mask(x, rank=3)
    out_np = np.array(out)
    assert np.all(out_np[:, :3] == 1.0)
    assert np.all(out_np[:, 3:] == 0.0)


def test_rank_mask_full_rank_is_identity() -> None:
    x = mx.array(np.random.default_rng(10).standard_normal((3, 5)).astype(np.float32))
    out = amc_apply_rank_mask(x, rank=5)
    assert np.allclose(np.array(out), np.array(x))


def test_rank_mask_clamps_rank_above_dim() -> None:
    x = mx.array(np.ones((2, 4), dtype=np.float32))
    out = amc_apply_rank_mask(x, rank=100)
    assert np.allclose(np.array(out), np.array(x))


def test_rank_mask_zero_rank_zeros_everything() -> None:
    x = mx.array(np.ones((2, 4), dtype=np.float32))
    out = amc_apply_rank_mask(x, rank=0)
    assert np.all(np.array(out) == 0.0)


# ---------------------------------------------------------------------------
# Quantization — Eq. 7
# ---------------------------------------------------------------------------


def test_quantize_tier_16bit_is_passthrough_fp16() -> None:
    x = mx.array(np.array([[1.5, -2.5]], dtype=np.float32))
    out = amc_quantize_tier(x, bits=16)
    assert out.dtype == mx.float16


def test_quantize_tier_4bit_reduces_precision() -> None:
    rng = np.random.default_rng(11)
    x = mx.array(rng.standard_normal((32, 8)).astype(np.float32))
    out = amc_quantize_tier(x, bits=4, group_size=32)
    assert out.dtype == mx.float16
    # Quantized values should differ from the original (lossy).
    assert not np.allclose(np.array(out), np.array(x), atol=1e-4)


def test_quantize_tier_output_shape_preserved() -> None:
    x = mx.array(np.random.default_rng(12).standard_normal((10, 6)).astype(np.float32))
    for bits in (16, 8, 4):
        out = amc_quantize_tier(x, bits=bits)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Closed-loop adaptive thresholds — Eq. 4-5
# ---------------------------------------------------------------------------


def test_adaptive_thresholds_widen_on_high_variance_sequences() -> None:
    state = init_amc_threshold_state(window_size=32, calib_variance=0.01)
    high_var_saliency = mx.array(np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32))
    tau_h, tau_l, state = amc_adaptive_thresholds(
        tau_high_base=0.8,
        tau_low_base=0.5,
        state=state,
        new_saliency_values=high_var_saliency,
        gamma=0.1,
    )
    # High variance vs. tiny calib variance -> ratio >> 1 -> ln(ratio) > 0
    # -> thresholds depressed (lower than base).
    assert tau_h < 0.8
    assert tau_l < 0.5


def test_adaptive_thresholds_narrow_on_low_variance_sequences() -> None:
    state = init_amc_threshold_state(window_size=32, calib_variance=1.0)
    low_var_saliency = mx.array(np.array([0.5, 0.5001, 0.4999, 0.5, 0.5, 0.5001], dtype=np.float32))
    tau_h, tau_l, state = amc_adaptive_thresholds(
        tau_high_base=0.8,
        tau_low_base=0.5,
        state=state,
        new_saliency_values=low_var_saliency,
        gamma=0.1,
    )
    # Very low variance vs. calib variance 1.0 -> ratio << 1 -> ln(ratio) < 0
    # -> thresholds raised (higher than base).
    assert tau_h > 0.8
    assert tau_l > 0.5


def test_adaptive_thresholds_guard_degenerate_zero_variance() -> None:
    state = init_amc_threshold_state(window_size=8, calib_variance=1e-10)
    zero_var = mx.array(np.array([0.5, 0.5, 0.5], dtype=np.float32))
    tau_h, tau_l, state = amc_adaptive_thresholds(
        tau_high_base=0.8,
        tau_low_base=0.5,
        state=state,
        new_saliency_values=zero_var,
        gamma=0.1,
    )
    assert not math.isnan(tau_h)
    assert not math.isinf(tau_h)
    assert not math.isnan(tau_l)
    assert not math.isinf(tau_l)


def test_adaptive_thresholds_single_value_no_crash() -> None:
    state = init_amc_threshold_state(window_size=8, calib_variance=1.0)
    single = mx.array(np.array([0.5], dtype=np.float32))
    tau_h, tau_l, state = amc_adaptive_thresholds(
        tau_high_base=0.8,
        tau_low_base=0.5,
        state=state,
        new_saliency_values=single,
        gamma=0.1,
    )
    assert tau_h == pytest.approx(0.8)  # < 2 samples in window -> no adjustment yet
    assert tau_l == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Bit-packing (Low tier, dsa.BitPackBuffer)
# ---------------------------------------------------------------------------


def test_bitpack_roundtrip_low_tier() -> None:
    from veloxquant_mlx.dsa.bit_pack import BitPackBuffer

    rng = np.random.default_rng(13)
    codes = rng.integers(0, 16, size=37).astype(np.uint8)
    packed_bytes, n = amc_pack_low_tier(mx.array(codes))
    assert n == 37
    packer = BitPackBuffer(4)
    unpacked = packer.unpack(np.frombuffer(packed_bytes, dtype=np.uint8), n)
    assert np.array_equal(unpacked, codes)


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_amc_fp16_bytes_all_high_tier_equals_full_rank() -> None:
    # All-HIGH tier at D=128 uses full rank 128 @ 16-bit == fp16 full cost.
    counts = {HIGH: 10, MID: 0, LOW: 0}
    b = amc_fp16_bytes(counts, head_dim=128)
    full = full_amc_fp16_bytes(tokens_seen=10, head_dim=128)
    assert b == full


def test_amc_fp16_bytes_all_low_tier_much_smaller() -> None:
    counts_high = {HIGH: 10, MID: 0, LOW: 0}
    counts_low = {HIGH: 0, MID: 0, LOW: 10}
    b_high = amc_fp16_bytes(counts_high, head_dim=128)
    b_low = amc_fp16_bytes(counts_low, head_dim=128)
    assert b_low < b_high


def test_full_amc_fp16_bytes_scales_with_tokens_and_dim() -> None:
    assert full_amc_fp16_bytes(100, 128) == 100 * 128 * 2 * 2
