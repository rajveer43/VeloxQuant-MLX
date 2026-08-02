"""Unit tests for SKVQ-adapted channel-reorder + clipped group quantization.

Covers:
  - permutation validity, inversion, and gather round-trip
  - sorted-by-range permutation shrinks within-group range spread
  - alpha=1 clip window == plain asymmetric min/max group quantization
  - per-group clip search never worse than alpha=1 (search metric)
  - reconstruction error monotone in bits; high-bit round-trip accuracy
  - reorder helps on heterogeneous channels, ~nothing on homogeneous
  - ragged-group channel padding; shapes/dtypes; determinism; guards
  - analytic byte helpers
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.skvq import (
    DEFAULT_ALPHA_GRID,
    apply_permutation,
    channel_permutation,
    clipped_group_dequant,
    clipped_group_quant,
    invert_permutation,
    skvq_compressed_bytes,
    skvq_fp16_bytes,
    skvq_round_trip,
)


def _rows(n, d, seed=0, channel_scales=None):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    if channel_scales is not None:
        x = x * channel_scales[None, :].astype(np.float32)
    return mx.array(x)


def _het_scales(d, seed=3):
    """Smoothly heterogeneous per-channel scales (three decades), shuffled
    so similar scales are NOT contiguous. Sorting then tightens most groups
    — the SKVQ reordering premise."""
    rng = np.random.default_rng(seed)
    scales = np.logspace(-2, 1, d)
    rng.shuffle(scales)
    return scales


def _mse(a: mx.array, b: mx.array) -> float:
    return float(mx.mean(mx.square(a.astype(mx.float32) - b.astype(mx.float32))).item())


# ------------------------------------------------------------------
# Permutation primitives
# ------------------------------------------------------------------


def test_permutation_valid_and_inverse() -> None:
    x = _rows(64, 32, seed=1, channel_scales=_het_scales(32))
    perm = channel_permutation(x)
    p = np.array(perm)
    assert sorted(p.tolist()) == list(range(32))
    inv = invert_permutation(perm)
    assert np.array_equal(np.array(inv)[p], np.arange(32))
    # gather round-trip: x -> permuted -> back
    back = apply_permutation(apply_permutation(x, perm), inv)
    assert np.array_equal(np.array(back), np.array(x))


def test_permutation_sorts_by_dynamic_range() -> None:
    x = _rows(128, 16, seed=2, channel_scales=_het_scales(16))
    perm = channel_permutation(x)
    xn = np.array(x)
    rng_per_ch = xn.max(axis=0) - xn.min(axis=0)
    assert np.all(np.diff(rng_per_ch[np.array(perm)]) >= 0)


def test_sorted_permutation_shrinks_group_range_spread() -> None:
    d, gs = 32, 8
    x = _rows(256, d, seed=4, channel_scales=_het_scales(d))
    xn = np.array(x)
    rng_per_ch = xn.max(axis=0) - xn.min(axis=0)

    def spread(order):
        r = rng_per_ch[order].reshape(d // gs, gs)
        return float((r.max(axis=1) - r.min(axis=1)).sum())

    assert spread(np.array(channel_permutation(x))) < spread(np.arange(d))


# ------------------------------------------------------------------
# Clipped group quantization
# ------------------------------------------------------------------


def test_alpha1_equals_plain_minmax() -> None:
    x = _rows(64, 32, seed=5, channel_scales=_het_scales(32))
    codes, lo, scale = clipped_group_quant(x, bits=2, group_size=8, alphas=(1.0,))

    # Plain asymmetric min/max reference (numpy, float32 arithmetic)
    xn = np.array(x).reshape(64, 4, 8)
    gmin = xn.min(axis=-1, keepdims=True)
    gmax = xn.max(axis=-1, keepdims=True)
    ref_scale = np.maximum((gmax - gmin) / 3.0, 1e-8)
    ref_codes = np.clip(np.round((xn - gmin) / ref_scale), 0, 3)

    got = np.array(codes, dtype=np.float64).reshape(64, 4, 8)
    # mid - range/2 == gmin only up to 1 ulp in float32, so allow codes to
    # differ by at most one step on an isolated boundary element.
    assert np.mean(got == ref_codes) > 0.999
    assert np.max(np.abs(got - ref_codes)) <= 1.0
    recon = np.array(clipped_group_dequant(codes, lo, scale, 8, 32)).reshape(64, 4, 8)
    ref_recon = ref_codes * ref_scale + gmin
    assert np.all(np.abs(recon - ref_recon) <= ref_scale + 1e-5)


def test_clip_search_never_worse_than_alpha1() -> None:
    # Groups with a single extreme outlier are exactly where clipping wins.
    rng = np.random.default_rng(6)
    x = rng.standard_normal((128, 32)).astype(np.float32)
    x[rng.random((128, 32)) < 0.03] *= 20.0  # sparse outliers
    x = mx.array(x)

    def per_group_mse(alphas):
        codes, lo, scale = clipped_group_quant(x, bits=2, group_size=8, alphas=alphas)
        recon = clipped_group_dequant(codes, lo, scale, 8, 32)
        err = mx.square(recon - x.astype(mx.float32)).reshape(128, 4, 8)
        return np.array(mx.mean(err, axis=-1))

    searched = per_group_mse(DEFAULT_ALPHA_GRID)
    plain = per_group_mse((1.0,))
    assert np.all(searched <= plain + 1e-7)
    # ...and on this outlier-heavy data the search strictly wins somewhere.
    assert searched.sum() < plain.sum()


def test_error_monotone_in_bits() -> None:
    x = _rows(128, 64, seed=7)
    errs = [_mse(skvq_round_trip(x, None, bits=b, group_size=16), x) for b in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-4  # 8-bit round trip is near-exact


def test_reorder_helps_heterogeneous_not_homogeneous() -> None:
    d = 64
    het = _rows(256, d, seed=8, channel_scales=_het_scales(d, seed=9))
    hom = _rows(256, d, seed=8)

    def improvement(x):
        perm = channel_permutation(x)
        base = _mse(skvq_round_trip(x, None, bits=2, group_size=16), x)
        reord = _mse(skvq_round_trip(x, perm, bits=2, group_size=16), x)
        return (base - reord) / base

    imp_het = improvement(het)
    imp_hom = improvement(hom)
    assert imp_het > 0.05  # real win under smooth heterogeneity
    assert imp_het > imp_hom + 0.03  # and clearly larger than the control


def test_reorder_large_win_on_per_channel_snr() -> None:
    """With a few dominant outlier channels, absolute MSE is dominated by
    the outlier channels whichever way groups are cut — but reordering
    rescues the *small* channels: normalized per-channel error (channel MSE
    over channel variance) collapses once small channels stop sharing a
    group range with an outlier. Outlier count == group_size, so sorting
    isolates them into exactly one group and every small channel is clean;
    unsorted, the shuffle contaminates every group."""
    d = 64
    rng = np.random.default_rng(14)
    scales = np.full(d, 0.05)
    scales[:16] = 8.0
    rng.shuffle(scales)
    x = _rows(256, d, seed=15, channel_scales=scales)

    def norm_err(perm):
        recon = np.array(skvq_round_trip(x, perm, bits=2, group_size=16), dtype=np.float32)
        err = np.mean((recon - np.array(x)) ** 2, axis=0)
        var = np.maximum(np.array(x).var(axis=0), 1e-12)
        return float(np.mean(err / var))

    assert norm_err(channel_permutation(x)) < 0.5 * norm_err(None)


def test_ragged_group_padding_round_trip() -> None:
    x = _rows(16, 10, seed=10)  # d=10, gs=4 -> padded to 12
    codes, lo, scale = clipped_group_quant(x, bits=4, group_size=4)
    assert codes.shape == (16, 12)
    assert lo.shape == (16, 3) and scale.shape == (16, 3)
    recon = clipped_group_dequant(codes, lo, scale, 4, 10)
    assert recon.shape == (16, 10)
    assert _mse(recon, x) < _mse(mx.zeros_like(x), x)


def test_shapes_dtypes_and_fp16_preserved() -> None:
    x = _rows(32, 32, seed=11).astype(mx.float16)
    codes, lo, scale = clipped_group_quant(x, bits=2, group_size=8)
    assert codes.dtype == mx.uint8
    assert lo.dtype == mx.float32 and scale.dtype == mx.float32
    out = skvq_round_trip(x, channel_permutation(x), bits=2, group_size=8)
    assert out.dtype == mx.float16 and out.shape == x.shape


def test_determinism() -> None:
    x = _rows(64, 32, seed=12, channel_scales=_het_scales(32))
    perm = channel_permutation(x)
    a = skvq_round_trip(x, perm, bits=2, group_size=8)
    b = skvq_round_trip(x, perm, bits=2, group_size=8)
    assert np.array_equal(np.array(a), np.array(b))


def test_guards() -> None:
    x = _rows(8, 16, seed=13)
    with pytest.raises(ValueError):
        clipped_group_quant(x, bits=0, group_size=8)
    with pytest.raises(ValueError):
        clipped_group_quant(x, bits=9, group_size=8)
    with pytest.raises(ValueError):
        clipped_group_quant(x, bits=2, group_size=0)
    with pytest.raises(ValueError):
        clipped_group_quant(x, bits=2, group_size=8, alphas=())
    with pytest.raises(ValueError):
        clipped_group_quant(x, bits=2, group_size=8, alphas=(0.9, 0.0))
    with pytest.raises(ValueError):
        clipped_group_quant(x, bits=2, group_size=8, alphas=(1.2,))


def test_byte_helpers() -> None:
    # 100 tokens, d=64, 2 bits, gs=16 -> 4 groups
    assert skvq_compressed_bytes(100, 64, 2, 16) == (math.ceil(100 * 64 * 2 / 8) + 100 * 4 * 2 * 2)
    assert skvq_fp16_bytes(100, 64) == 100 * 64 * 2
    # ragged: d=10, gs=4 -> 3 groups
    assert skvq_compressed_bytes(10, 10, 4, 4) == math.ceil(10 * 10 * 4 / 8) + 10 * 3 * 4
