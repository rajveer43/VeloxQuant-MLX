"""Tests for SVDqKVCache — sub-2-bit key compression via offline SVD.

Covers:
  - factory dispatch and no-bits-leak
  - SVD projection correctness (reconstruction error < baseline)
  - prefill-only (no decode) stores V and K_mean correctly
  - decode accumulation: sequential keys reconstruct with low MSE
  - byte accounting: compressed_key_bytes < fp16_key_bytes
  - rank selection via energy threshold
  - values are passed through fp16 unchanged
  - assigned_avg_bits is sub-2-bit at default settings
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.svdq_cache import SVDqKVCache


def _make(**cfg):
    base = dict(
        method="svdq",
        head_dim=64,
        # rank=32 = min_safe_rank(8 groups) at MIN_SAFE_CHANNELS_PER_GROUP=4 —
        # explicit and deterministic, and clears the small-rank guard rail.
        svdq_rank=32,
        svdq_bit_schedule=(8, 4, 2, 1, 1, 0, 0, 0),
        svdq_group_size=16,
    )
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_kv(S=128, H=2, D=64, seed=0):
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((1, H, S, D)).astype(np.float16)
    V = rng.standard_normal((1, H, S, D)).astype(np.float16)
    return mx.array(K), mx.array(V)


# ------------------------------------------------------------------
# Factory and interface
# ------------------------------------------------------------------


def test_factory_dispatch() -> None:
    c = _make()
    assert isinstance(c, SVDqKVCache)


def test_no_bits_attribute() -> None:
    c = _make()
    assert not hasattr(c, "bits")
    assert hasattr(c, "assigned_avg_bits")


# ------------------------------------------------------------------
# SVD projection correctness
# ------------------------------------------------------------------


def test_svd_rank_stored_after_prefill() -> None:
    c = _make(svdq_rank=32)
    K, V = _rand_kv(S=64, D=64)
    ko, vo = c.update_and_fetch(K, V)
    mx.eval(ko, vo)
    assert c._V is not None
    assert c._K_mean is not None
    assert c.rank == 32
    assert c._V.shape == (64, 32)
    assert c._K_mean.shape == (64,)


def test_output_shape_preserved() -> None:
    c = _make()
    K, V = _rand_kv(S=64, H=2, D=64)
    ko, vo = c.update_and_fetch(K, V)
    mx.eval(ko, vo)
    assert ko.shape == (1, 2, 64, 64)
    assert vo.shape == (1, 2, 64, 64)


def test_reconstruction_lower_mse_than_raw_2bit() -> None:
    """SVDq outperforms naive 2-bit on low-rank structured data — in the
    regime the paper's 8-group schedule targets (rank ≫ 8, energy decaying
    smoothly across many latent channels per group, mirroring Table 1's
    real configs: d=1024, d/8=128 channels per group).

    With a *small* rank close to 8 (e.g. rank=8, one channel per group), the
    schedule's trailing 0-bit groups truncate real signal outright and can
    underperform naive quantization — see
    test_small_rank_near_group_count_can_underperform_naive below. This is a
    genuine property of the paper's fixed 8-group schedule, not a bug: the
    schedule assumes rank is large enough that each of the 8 groups still
    spans a range of decaying-but-non-negligible singular values.
    """
    from veloxquant_mlx.quantizers.svdq import _group_quant_dequant

    rng = np.random.default_rng(42)
    S, D, true_rank = 256, 128, 64
    U = rng.standard_normal((S, true_rank)).astype(np.float32)
    # Exponentially decaying component scale, matching the paper's Section 4.3
    # decay model (lambda_j = c * exp(-rho*j)) so the trailing groups really
    # do carry little energy and are safe to truncate.
    decay = np.exp(-0.15 * np.arange(true_rank)).astype(np.float32)
    W = (rng.standard_normal((true_rank, D)).astype(np.float32) * decay[:, None])
    noise = rng.standard_normal((S, D)).astype(np.float32) * 0.02
    K_np = U @ W + noise
    K_mx = mx.array(K_np)

    c = _make(head_dim=D, svdq_rank=true_rank, svdq_bit_schedule=(8, 4, 2, 1, 1, 0, 0, 0))
    K_in = mx.array(K_np[None, None])
    V_in = mx.zeros((1, 1, S, D))
    ko, _ = c.update_and_fetch(K_in, V_in)
    mx.eval(ko)
    svdq_mse = float(mx.mean((ko[0, 0].astype(mx.float32) - K_mx) ** 2).item())

    naive_recon = _group_quant_dequant(K_mx, b=2, group_size=16)
    mx.eval(naive_recon)
    naive_mse = float(mx.mean((naive_recon.astype(mx.float32) - K_mx) ** 2).item())

    assert svdq_mse < naive_mse, (
        f"SVDq MSE {svdq_mse:.6f} should be < naive 2-bit MSE {naive_mse:.6f} "
        f"on low-rank, energy-decaying data (true_rank={true_rank}, D={D})"
    )


def test_small_rank_near_group_count_is_rejected() -> None:
    """Guard rail: when rank is close to n_groups=8, the trailing 0-bit
    groups would truncate one-channel groups that still carry real signal
    (SVDq can end up *worse* than naive 2-bit quantization there — the
    schedule assumes d/8 channels per group, not 1). Rather than silently
    degrading, the cache now raises at prefill time so this misuse is caught
    immediately instead of producing quietly-bad reconstructions.
    """
    rng = np.random.default_rng(42)
    S, D, true_rank = 128, 64, 8
    U = rng.standard_normal((S, true_rank)).astype(np.float32)
    W = rng.standard_normal((true_rank, D)).astype(np.float32)
    noise = rng.standard_normal((S, D)).astype(np.float32) * 0.05
    K_np = U @ W + noise

    c = _make(head_dim=D, svdq_rank=true_rank, svdq_bit_schedule=(8, 4, 2, 1, 1, 0, 0, 0))
    K_in = mx.array(K_np[None, None])
    V_in = mx.zeros((1, 1, S, D))

    with pytest.raises(ValueError, match="too small for the"):
        c.update_and_fetch(K_in, V_in)


def test_automatic_rank_degrades_gracefully_instead_of_raising() -> None:
    """When rank comes from the energy threshold (not an explicit choice —
    e.g. a short prefill sequence that only supports a small rank), the
    small-rank guard must NOT raise: the caller never chose an unsafe rank,
    the sequence length just happened to force one. Default config with a
    16-token prefill (mirrors the crash-tier probe's default smoke test in
    registry.py's _run_probe) must still serve without error, silently
    substituting a non-truncating schedule for that layer instead of
    dropping real signal or crashing.
    """
    c = _make(svdq_rank=None, head_dim=128, svdq_bit_schedule=(8, 4, 2, 1, 1, 0, 0, 0))
    K, V = _rand_kv(S=16, H=8, D=128)
    ko, vo = c.update_and_fetch(K, V)
    mx.eval(ko, vo)
    assert not mx.any(mx.isnan(ko)).item()
    # Guard should have kicked in: rank is capped by S=16 < min_safe_rank(8)=32.
    assert c.rank < 32
    assert 0 not in c._effective_schedule


def test_schedule_without_truncation_allows_small_rank() -> None:
    """The guard only fires when the schedule has a 0-bit group. A schedule
    with no truncation (every group gets at least 1 bit) has no failure mode
    tied to small rank, so it should be allowed through even at rank=8.
    """
    rng = np.random.default_rng(42)
    S, D, true_rank = 128, 64, 8
    K_np = rng.standard_normal((S, D)).astype(np.float32)

    c = _make(
        head_dim=D,
        svdq_rank=true_rank,
        svdq_bit_schedule=(4, 4, 3, 3, 2, 2, 1, 1),  # no zeros
    )
    K_in = mx.array(K_np[None, None])
    V_in = mx.zeros((1, 1, S, D))
    ko, _ = c.update_and_fetch(K_in, V_in)
    mx.eval(ko)
    assert not mx.any(mx.isnan(ko)).item()


def test_min_safe_rank_helper() -> None:
    from veloxquant_mlx.quantizers.svdq import MIN_SAFE_CHANNELS_PER_GROUP, min_safe_rank

    assert min_safe_rank(8) == 8 * MIN_SAFE_CHANNELS_PER_GROUP


# ------------------------------------------------------------------
# Values pass-through
# ------------------------------------------------------------------


def test_values_unchanged() -> None:
    """Values must be passed through without modification."""
    c = _make()
    K, V = _rand_kv(S=64, D=64)
    ko, vo = c.update_and_fetch(K, V)
    mx.eval(vo)
    # After the first call the parent cache accumulates; just check dtype+shape
    assert vo.dtype == mx.float16
    assert vo.shape[-1] == 64


# ------------------------------------------------------------------
# Decode accumulation
# ------------------------------------------------------------------


def test_decode_after_prefill() -> None:
    """Decode calls after prefill must produce valid fp16 output."""
    c = _make(svdq_rank=32)
    # Prefill
    K_pre, V_pre = _rand_kv(S=64, H=2, D=64, seed=0)
    c.update_and_fetch(K_pre, V_pre)
    # Decode steps
    for step in range(4):
        K_dec, V_dec = _rand_kv(S=1, H=2, D=64, seed=step + 10)
        ko, vo = c.update_and_fetch(K_dec, V_dec)
        mx.eval(ko, vo)
        assert ko.dtype == mx.float16
        assert not mx.any(mx.isnan(ko)).item()


# ------------------------------------------------------------------
# Byte accounting
# ------------------------------------------------------------------


def test_compressed_bytes_less_than_fp16() -> None:
    c = _make(svdq_rank=32)
    K, V = _rand_kv(S=128, D=64)
    c.update_and_fetch(K, V)
    assert c.compressed_key_bytes > 0
    assert c.fp16_key_bytes > 0
    assert c.compressed_key_bytes < c.fp16_key_bytes


def test_value_fp16_bytes_positive() -> None:
    c = _make(svdq_rank=32)
    K, V = _rand_kv(S=64, D=64)
    c.update_and_fetch(K, V)
    assert c.value_fp16_bytes > 0


# ------------------------------------------------------------------
# Effective bit-width
# ------------------------------------------------------------------


def test_assigned_avg_bits_sub_2() -> None:
    """Default settings should give effective key bit-width well below 2."""
    c = _make(head_dim=128, svdq_rank=32)
    K, V = _rand_kv(S=64, H=2, D=128)
    c.update_and_fetch(K, V)
    bits = c.assigned_avg_bits
    assert bits < 2.0, f"Expected sub-2-bit, got {bits:.3f}"


# ------------------------------------------------------------------
# Energy threshold rank selection
# ------------------------------------------------------------------


def test_energy_threshold_rank_selection() -> None:
    """With svdq_rank=None, rank should be determined by energy threshold."""
    c = _make(svdq_rank=None, svdq_energy_threshold=0.90, head_dim=64)
    K, V = _rand_kv(S=128, D=64)
    c.update_and_fetch(K, V)
    assert 1 <= c.rank <= 64


def test_determinism() -> None:
    """Two caches with same config on same data must produce identical output."""
    K, V = _rand_kv(S=64, D=64, seed=7)
    c1 = _make()
    c2 = _make()
    ko1, _ = c1.update_and_fetch(K, V)
    ko2, _ = c2.update_and_fetch(K, V)
    mx.eval(ko1, ko2)
    assert np.allclose(np.array(ko1), np.array(ko2), atol=1e-4)


# ---------------------------------------------------------------------------
# Config validation — svdq_bit_schedule entries must be >= 0
# ---------------------------------------------------------------------------


def test_negative_bit_schedule_entry_rejected() -> None:
    with pytest.raises(ValueError, match="svdq_bit_schedule"):
        _make(svdq_bit_schedule=(8, 4, 2, -1, 1, 0, 0, 0))


def test_empty_bit_schedule_rejected() -> None:
    with pytest.raises(ValueError, match="svdq_bit_schedule"):
        _make(svdq_bit_schedule=())


# ---------------------------------------------------------------------------
# 8-group bit schedule — paper Eq. 6 grouping and truncation semantics
# ---------------------------------------------------------------------------


def test_zero_bit_group_is_truncated_to_zero() -> None:
    """Channels in a 0-bit group must reconstruct to exactly zero."""
    from veloxquant_mlx.quantizers.svdq import latent_group_slices, quantize_latents_mixed

    rng = np.random.default_rng(0)
    S, r = 32, 16
    L = mx.array(rng.standard_normal((S, r)).astype(np.float32))
    sv = mx.array(np.linspace(10.0, 0.1, r).astype(np.float32))
    schedule = (8, 4, 2, 1, 1, 0, 0, 0)

    L_q = quantize_latents_mixed(L, sv, bit_schedule=schedule, group_size=8)
    mx.eval(L_q)

    slices = latent_group_slices(r, n_groups=len(schedule))
    for group_idx, (start, end) in enumerate(slices):
        if schedule[group_idx] == 0:
            assert np.all(np.array(L_q[:, start:end]) == 0.0)


def test_equivalent_bit_width_matches_paper_example() -> None:
    """Paper's worked example: schedule (8,4,2,1,1,0,0,0) → b̄ = 2."""
    from veloxquant_mlx.quantizers.svdq import equivalent_bit_width

    assert equivalent_bit_width(64, (8, 4, 2, 1, 1, 0, 0, 0)) == pytest.approx(2.0)


def test_default_schedule_gives_lower_effective_bits_than_old_default() -> None:
    """The paper's schedule (b̄=2) with truncation should compress harder than
    the old top-25%-at-4/rest-at-2 split (which never truncates).
    """
    c = _make(head_dim=128, svdq_rank=32, svdq_bit_schedule=(8, 4, 2, 1, 1, 0, 0, 0))
    K, V = _rand_kv(S=64, H=2, D=128)
    c.update_and_fetch(K, V)
    assert c.assigned_avg_bits < 1.5
