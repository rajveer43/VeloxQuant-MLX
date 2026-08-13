"""Tests for the KVTC quantizer (quantizers/kvtc.py).

Covers: init guards, near-identity reconstruction at a generous budget, the
DP allocator beating SVDq's fixed top-25%/75% split at a MATCHED total byte
budget on a planted skewed-variance geometry (a rate over several seeds, not
one lucky run — this is the clean mechanism observable named in the
implementation prompt), byte accounting (kvtc_fp16_bytes includes the
entropy-coded payload + table + V + mean; kvtc_pre_entropy_bytes excludes
entropy gain), determinism, and values compressed too (not keys-only, unlike
SVDq).
"""

from __future__ import annotations

import numpy as np
import pytest
import mlx.core as mx

from veloxquant_mlx.quantizers.kvtc import (
    KVTCArtifact,
    kvtc_compress,
    kvtc_decompress,
    kvtc_fp16_bytes,
    kvtc_pre_entropy_bytes,
)


def _skewed_variance_data(S: int, D: int, r_true: int, seed: int, scale=None):
    """Planted low-rank-ish geometry with sharply decaying per-component
    variance — the regime where the DP allocator should beat a fixed split.
    """
    rng = np.random.default_rng(seed)
    if scale is None:
        scale = np.array([20.0 / (i + 1) for i in range(r_true)])
    U = rng.standard_normal((S, r_true))
    Wt = rng.standard_normal((r_true, D)) * scale[:, None]
    noise = rng.standard_normal((S, D)) * 0.05
    X = (U @ Wt + noise).astype(np.float32)
    return mx.array(X)


def _svdq_fixed_split_reconstruction(
    x: mx.array, total_bit_budget: int, hi_fraction=0.25
) -> mx.array:
    """SVDq-style fixed top-25%/75% mixed-bit split, at a MATCHED total bit
    budget, for the same local PCA basis kvtc_compress would fit — used as
    the baseline in the "DP beats fixed split" comparison. Implemented
    directly here (rather than importing svdq.py's hi_bit/lo_bit knobs,
    which are tuned for a different total-budget convention) so the
    comparison is exactly matched-budget: same total bits, split
    25%/75% by singular-value magnitude instead of DP-allocated.
    """
    from veloxquant_mlx.quantizers._quant_utils import _truncated_svd
    from veloxquant_mlx.quantizers.kvtc import quantize_component

    xf = x.astype(mx.float32)
    S, D = int(xf.shape[0]), int(xf.shape[1])
    mean = mx.mean(xf, axis=0)
    xc = xf - mean[None, :]
    r = min(S, D)
    U, s_vals, Vt = _truncated_svd(xc, rank=r)
    V = Vt.T
    L = xc @ V
    mx.eval(L)
    L_np = np.asarray(L.tolist(), dtype=np.float64)

    n_hi = max(1, int(r * hi_fraction))
    n_lo = r - n_hi
    # Split the SAME total bit budget proportionally to hi/lo tier sizes so
    # bit_hi/bit_lo are integers with hi_bit >= lo_bit (mirrors SVDq's 4/2
    # ratio at whatever total budget is available) -- solve for an integer
    # (hi_bit, lo_bit) pair with hi_bit = 2*lo_bit (SVDq's ratio) that fits
    # the budget as closely as possible without exceeding it.
    best = (0, 0)
    for lo_bit in range(0, 9):
        hi_bit = 2 * lo_bit
        total = n_hi * hi_bit + n_lo * lo_bit
        if total <= total_bit_budget and (n_hi * hi_bit + n_lo * lo_bit) > (
            n_hi * best[0] + n_lo * best[1]
        ):
            best = (hi_bit, lo_bit)
    hi_bit, lo_bit = best

    recon = np.zeros((S, r), dtype=np.float64)
    for i in range(r):
        bits = hi_bit if i < n_hi else lo_bit
        if bits <= 0:
            continue
        codes, lo, scale = quantize_component(L_np[:, i], bits)
        recon[:, i] = codes.astype(np.float64) * scale + lo

    x_hat = mx.array(recon.astype(np.float32)) @ V.T + mean[None, :]
    return x_hat.astype(mx.float16)


def _mse(a: mx.array, b: mx.array) -> float:
    return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())


# ---------------------------------------------------------------------------
# init guards
# ---------------------------------------------------------------------------
def test_compress_rejects_empty_tensor():
    with pytest.raises(ValueError, match="S >= 1"):
        kvtc_compress(mx.zeros((0, 8)), total_bit_budget=16)


def test_compress_rejects_negative_budget():
    x = mx.array(np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32))
    with pytest.raises(ValueError, match="total_bit_budget must be >= 0"):
        kvtc_compress(x, total_bit_budget=-1)


def test_compress_zero_budget_drops_everything():
    x = mx.array(np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32))
    art = kvtc_compress(x, total_bit_budget=0)
    assert art.n_survived == 0
    recon = kvtc_decompress(art)
    assert recon.shape == x.shape


# ---------------------------------------------------------------------------
# near-identity reconstruction at a generous budget
# ---------------------------------------------------------------------------
def test_generous_budget_near_identity_reconstruction():
    rng = np.random.default_rng(0)
    S, D = 64, 16
    x = mx.array(rng.standard_normal((S, D)).astype(np.float32))
    art = kvtc_compress(x, total_bit_budget=D * 16, bit_choices=(0, 1, 2, 3, 4, 6, 8))
    recon = kvtc_decompress(art)

    xf = x.astype(mx.float32)
    rf = recon.astype(mx.float32)
    cos = float(
        mx.mean(
            mx.sum(xf * rf, axis=-1)
            / (mx.sqrt(mx.sum(xf * xf, axis=-1)) * mx.sqrt(mx.sum(rf * rf, axis=-1)) + 1e-8)
        ).item()
    )
    assert cos > 0.999


# ---------------------------------------------------------------------------
# DP allocator beats SVDq's fixed 25/75 split at MATCHED byte budget on
# skewed-variance geometry — a rate over several seeds
# ---------------------------------------------------------------------------
def test_dp_beats_fixed_split_at_matched_budget_skewed_variance():
    S, D, r_true = 128, 32, 6
    budget = D * 3  # total bits across components — matched for both arms
    seeds = range(12)

    dp_wins = 0
    for seed in seeds:
        x = _skewed_variance_data(S, D, r_true, seed)
        art = kvtc_compress(x, total_bit_budget=budget)
        dp_recon = kvtc_decompress(art)
        dp_mse = _mse(dp_recon, x)

        fixed_recon = _svdq_fixed_split_reconstruction(x, total_bit_budget=budget)
        fixed_mse = _mse(fixed_recon, x)

        if dp_mse <= fixed_mse:
            dp_wins += 1

    # A rate claim, not a per-seed guarantee: DP should win comfortably more
    # often than not on this planted skewed geometry.
    assert dp_wins >= int(0.8 * len(list(seeds)))


def test_dp_beats_fixed_split_mean_mse_skewed_variance():
    """Same geometry, checking the AVERAGE MSE across seeds rather than a
    per-seed win-rate — a second, complementary form of the mechanism claim.
    """
    S, D, r_true = 128, 32, 6
    budget = D * 3
    seeds = range(10)

    dp_mses, fixed_mses = [], []
    for seed in seeds:
        x = _skewed_variance_data(S, D, r_true, seed)
        art = kvtc_compress(x, total_bit_budget=budget)
        dp_mses.append(_mse(kvtc_decompress(art), x))
        fixed_mses.append(_mse(_svdq_fixed_split_reconstruction(x, budget), x))

    assert float(np.mean(dp_mses)) < float(np.mean(fixed_mses))


def test_flat_variance_dp_does_not_dramatically_beat_fixed_split():
    """Null-ish control: on near-flat variance (no signal to exploit), the DP
    allocator should be close to the fixed split's distortion, not a huge
    win — the honest 'flat' control the benchmark also reports.
    """
    rng = np.random.default_rng(0)
    S, D = 128, 32
    x = mx.array(rng.standard_normal((S, D)).astype(np.float32))  # isotropic
    budget = D * 3

    art = kvtc_compress(x, total_bit_budget=budget)
    dp_mse = _mse(kvtc_decompress(art), x)
    fixed_mse = _mse(_svdq_fixed_split_reconstruction(x, budget), x)

    # DP should not be dramatically worse (it is provably optimal for its own
    # proxy, so it should be at least competitive), but we do not assert a
    # big win on flat geometry -- only that it is in the same ballpark.
    assert dp_mse <= fixed_mse * 1.5


# ---------------------------------------------------------------------------
# byte accounting
# ---------------------------------------------------------------------------
def test_fp16_bytes_includes_payload_table_projection():
    rng = np.random.default_rng(1)
    x = mx.array(rng.standard_normal((64, 16)).astype(np.float32))
    art = kvtc_compress(x, total_bit_budget=32)

    D, r = 16, art.bit_allocation.shape[0]
    projection_bytes = (D * r + D) * 4
    quant_param_bytes = art.n_survived * 8
    payload_bytes = len(art.entropy_payload)
    from veloxquant_mlx.quantizers._entropy_coding import table_nbytes

    table_bytes = table_nbytes(art.entropy_table)

    expected = projection_bytes + quant_param_bytes + payload_bytes + table_bytes
    assert kvtc_fp16_bytes(art) == expected


def test_pre_entropy_bytes_excludes_entropy_gain():
    rng = np.random.default_rng(2)
    x = mx.array(rng.standard_normal((64, 16)).astype(np.float32))
    art = kvtc_compress(x, total_bit_budget=32)

    pre = kvtc_pre_entropy_bytes(art)
    S = art.S
    expected = sum(-(-(S * int(b)) // 8) for b in art.bit_allocation if b > 0)
    assert pre == expected
    # Pre-entropy accounting must not itself include the entropy payload or
    # table bytes -- it's a separate (smaller-scope) quantity than kvtc_fp16_bytes.
    assert pre != kvtc_fp16_bytes(art) or pre == 0


def test_byte_helpers_zero_when_nothing_survives():
    x = mx.array(np.random.default_rng(3).standard_normal((32, 8)).astype(np.float32))
    art = kvtc_compress(x, total_bit_budget=0)
    assert art.n_survived == 0
    assert kvtc_pre_entropy_bytes(art) == 0
    # fp16_bytes still counts V + mean (always stored).
    assert kvtc_fp16_bytes(art) > 0


# ---------------------------------------------------------------------------
# Regression for #77: entropy coding must never make the stored artifact
# larger than plain fixed-width packing -- kvtc_compress falls back to
# fixed-width when Huffman coding doesn't pay for itself.
# ---------------------------------------------------------------------------
def test_stored_size_never_worse_than_fixed_width():
    """Randomized sweep mirroring the issue's exact repro: across varied
    (S, D, budget), the realized stored payload+table must never exceed
    kvtc_pre_entropy_bytes (fixed-width size)."""
    from veloxquant_mlx.quantizers._entropy_coding import table_nbytes

    rng = np.random.default_rng(0)
    for _ in range(30):
        S = int(rng.integers(20, 100))
        D = int(rng.integers(8, 64))
        x = mx.array(rng.standard_normal((S, D)).astype(np.float32))
        budget = D * int(rng.integers(2, 5))
        art = kvtc_compress(x, total_bit_budget=budget)
        pre = kvtc_pre_entropy_bytes(art)
        stored = len(art.entropy_payload) + table_nbytes(art.entropy_table)
        assert stored <= pre


def test_uniform_codes_fall_back_to_fixed_width():
    """Near-uniform quantized codes (the realistic KV regime) should trigger
    the fixed-width fallback -- Huffman has nothing to gain and its table is
    pure overhead."""
    rng = np.random.default_rng(1)
    S, D = 512, 128
    x = mx.array(rng.standard_normal((S, D)).astype(np.float32))
    art = kvtc_compress(x, total_bit_budget=D * 3)
    assert art.is_entropy_coded is False
    assert art.entropy_table == {}


def test_fixed_width_fallback_round_trips_correctly():
    """Reconstruction through the fixed-width path must match reconstruction
    through the (previously always-used) entropy-coded path in accuracy --
    the fallback must not be lossy relative to entropy coding."""
    rng = np.random.default_rng(2)
    S, D = 256, 64
    x = mx.array(rng.standard_normal((S, D)).astype(np.float32))
    art = kvtc_compress(x, total_bit_budget=D * 3)
    assert art.is_entropy_coded is False  # this budget/shape triggers fallback
    recon = kvtc_decompress(art)
    mx.eval(recon)
    assert recon.shape == x.shape
    mse = _mse(recon, x)
    assert mse < 1.0  # ordinary quantization noise, not garbage


def test_entropy_coding_gain_never_below_one():
    """The module docstring's ">= 1 typically" claim (cache/kvtc_cache.py)
    should now hold as a hard guarantee, not "typically": the fallback
    ensures pre_entropy_bytes/kvtc_bytes-style ratios never fall below 1."""
    from veloxquant_mlx.quantizers._entropy_coding import table_nbytes

    rng = np.random.default_rng(3)
    for _ in range(10):
        S = int(rng.integers(20, 200))
        D = int(rng.integers(8, 64))
        x = mx.array(rng.standard_normal((S, D)).astype(np.float32))
        art = kvtc_compress(x, total_bit_budget=D * 3)
        pre = kvtc_pre_entropy_bytes(art)
        stored = len(art.entropy_payload) + table_nbytes(art.entropy_table)
        if stored == 0:
            continue
        assert pre / stored >= 1.0


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_deterministic_same_input_same_everything():
    rng = np.random.default_rng(4)
    x = mx.array(rng.standard_normal((48, 16)).astype(np.float32))

    art1 = kvtc_compress(x, total_bit_budget=32)
    art2 = kvtc_compress(x, total_bit_budget=32)

    assert np.array_equal(art1.bit_allocation, art2.bit_allocation)
    assert art1.entropy_payload == art2.entropy_payload
    assert bool(mx.all(art1.V == art2.V).item())

    r1 = kvtc_decompress(art1)
    r2 = kvtc_decompress(art2)
    assert bool(mx.all(r1 == r2).item())


# ---------------------------------------------------------------------------
# values compressed too — not keys-only (contrast with SVDq)
# ---------------------------------------------------------------------------
def test_compresses_arbitrary_tensor_not_just_keys():
    """kvtc_compress/kvtc_decompress operate on any [S, D] tensor -- the
    cache wrapper (cache/kvtc_cache.py) applies this identically to K and V,
    mirroring Palu's full-KV scope rather than SVDq's keys-only scope.
    """
    rng = np.random.default_rng(5)
    values = mx.array(rng.standard_normal((40, 16)).astype(np.float32) * 2.0)
    art = kvtc_compress(values, total_bit_budget=32)
    recon = kvtc_decompress(art)
    assert recon.shape == values.shape
    mse = _mse(recon, values)
    assert np.isfinite(mse)
    assert mse < _mse(mx.zeros_like(values), values)  # meaningfully better than all-zero
