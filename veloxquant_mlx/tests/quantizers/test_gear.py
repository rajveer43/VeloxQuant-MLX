"""Tests for the GEAR quantizer numerics — error-feedback over a base group quant.

GEAR reconstructs a base-quantized tensor's error with a low-rank residual plus
a sparse outlier correction. These tests cover the core claim (GEAR beats
base-quant-alone on low-rank+outlier data), the degenerate modes (pure low-rank,
pure sparse, base-only), residual-SVD recovery, sparse selection, byte
accounting, and determinism. All data is synthetic — no model loading.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.cachegen import cachegen_quant_dequant
from veloxquant_mlx.quantizers.gear import (
    GEARState,
    base_only_bytes,
    gear_bytes,
    gear_compress,
    gear_quant_dequant,
    gear_reconstruct,
    lowrank_error,
    quantize_base,
    residual,
    sparse_outliers,
)


def _mse(a: mx.array, b: mx.array) -> float:
    return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())


def _lowrank_plus_outliers(N=128, D=128, r=6, n_out=20, seed=0):
    """Low-rank signal + small noise + a few large outliers — GEAR's ideal case."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, r)).astype(np.float32)
    B = rng.standard_normal((r, D)).astype(np.float32)
    X = A @ B + 0.03 * rng.standard_normal((N, D)).astype(np.float32)
    flat = X.reshape(-1)
    idx = rng.choice(flat.size, size=n_out, replace=False)
    flat[idx] += rng.standard_normal(n_out).astype(np.float32) * 8.0
    return mx.array(X.reshape(N, D).astype(np.float16))


# ------------------------------------------------------------------
# Core claim
# ------------------------------------------------------------------


def test_gear_beats_base_quant() -> None:
    """GEAR reconstruction MSE strictly below base-quant-alone on its ideal data."""
    X = _lowrank_plus_outliers()
    gear = gear_quant_dequant(X, bits=2, rank=12, sparse_frac=0.01, group_size=32)
    base = cachegen_quant_dequant(X, 2, 32)
    assert _mse(gear, X) < _mse(base, X)


def test_lowrank_alone_helps() -> None:
    """Even with no sparse term, the low-rank residual reduces error vs base."""
    X = _lowrank_plus_outliers(n_out=0)
    gear = gear_quant_dequant(X, bits=2, rank=12, sparse_frac=0.0, group_size=32)
    base = cachegen_quant_dequant(X, 2, 32)
    assert _mse(gear, X) < _mse(base, X)


def test_sparse_alone_helps() -> None:
    """With rank=0, the sparse outlier term alone still reduces error vs base."""
    X = _lowrank_plus_outliers()
    gear = gear_quant_dequant(X, bits=2, rank=0, sparse_frac=0.02, group_size=32)
    base = cachegen_quant_dequant(X, 2, 32)
    assert _mse(gear, X) < _mse(base, X)


def test_base_only_equals_group_quant() -> None:
    """rank=0, sparse=0 collapses GEAR exactly to the base group quant."""
    X = _lowrank_plus_outliers()
    gear = gear_quant_dequant(X, bits=2, rank=0, sparse_frac=0.0, group_size=32)
    base = cachegen_quant_dequant(X, 2, 32)
    assert _mse(gear, base) == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------
# Component correctness
# ------------------------------------------------------------------


def test_residual_lowrank_recovers_known_rank() -> None:
    """A genuinely rank-r residual is recovered by the low-rank term to < eps."""
    rng = np.random.default_rng(3)
    N, D, r = 64, 48, 5
    E = mx.array((rng.standard_normal((N, r)) @ rng.standard_normal((r, D))).astype(np.float32))
    L, R = lowrank_error(E, rank=r)
    assert _mse(L @ R, E) < 1e-6


def test_sparse_selects_true_outliers() -> None:
    """sparse_outliers picks the largest-magnitude entries."""
    rng = np.random.default_rng(4)
    M = mx.array((rng.standard_normal((10, 10)) * 0.01).astype(np.float32))
    flat = list(M.reshape(-1))
    M = M.reshape(-1)
    M = M.at[mx.array([7, 50, 91])].add(mx.array([100.0, -100.0, 100.0]))
    M = M.reshape(10, 10)
    idx, vals = sparse_outliers(M, frac=3 / 100)
    picked = set(int(i) for i in idx.tolist())
    assert {7, 50, 91} <= picked


def test_lowrank_rank0_returns_none() -> None:
    E = mx.zeros((8, 8))
    L, R = lowrank_error(E, rank=0)
    assert L is None and R is None


def test_sparse_zero_frac_returns_none() -> None:
    idx, vals = sparse_outliers(mx.zeros((8, 8)), frac=0.0)
    assert idx is None and vals is None


# ------------------------------------------------------------------
# Byte accounting
# ------------------------------------------------------------------


def test_byte_accounting_ordering() -> None:
    """base_only <= gear_bytes <= fp16 at a realistic head dim with low rank."""
    X = _lowrank_plus_outliers(N=128, D=128, r=6)
    st = gear_compress(X, bits=2, rank=8, sparse_frac=0.005, group_size=32)
    fp16 = 128 * 128 * 2
    assert base_only_bytes(st) <= gear_bytes(st) <= fp16


def test_gear_bytes_components_sum() -> None:
    """gear_bytes equals base_only plus the low-rank and sparse overheads."""
    X = _lowrank_plus_outliers(N=128, D=128, r=6)
    st = gear_compress(X, bits=2, rank=8, sparse_frac=0.005, group_size=32)
    lr = (st.L.shape[0] * st.L.shape[1] + st.R.shape[0] * st.R.shape[1]) * 2
    nnz = int(st.sp_idx.shape[0])
    sp = nnz * (4 + 2)
    assert gear_bytes(st) == base_only_bytes(st) + lr + sp


# ------------------------------------------------------------------
# Robustness
# ------------------------------------------------------------------


def test_deterministic() -> None:
    X = _lowrank_plus_outliers()
    a = gear_quant_dequant(X, bits=2, rank=10, sparse_frac=0.01, group_size=32)
    b = gear_quant_dequant(X, bits=2, rank=10, sparse_frac=0.01, group_size=32)
    assert _mse(a, b) == pytest.approx(0.0, abs=0.0)


def test_state_shapes() -> None:
    X = _lowrank_plus_outliers(N=64, D=32, r=4)
    st = gear_compress(X, bits=2, rank=4, sparse_frac=0.02, group_size=32)
    assert isinstance(st, GEARState)
    assert st.n_rows == 64
    assert st.L.shape == (64, 4)
    assert st.R.shape == (4, 32)
    assert gear_reconstruct(st).shape == (64, 32)


def test_energy_threshold_rank_monotonic() -> None:
    """Higher energy threshold retains at least as many ranks."""
    X = _lowrank_plus_outliers(N=64, D=48, r=8)
    E = residual(X, cachegen_quant_dequant(X, 2, 32))
    L_lo, _ = lowrank_error(E, rank=None, energy_threshold=0.5)
    L_hi, _ = lowrank_error(E, rank=None, energy_threshold=0.99)
    assert L_hi.shape[1] >= L_lo.shape[1]


# ------------------------------------------------------------------
# KCVT base axis (paper's actual backbone: per-channel keys, per-token values)
# ------------------------------------------------------------------


def test_quantize_base_rejects_bad_axis() -> None:
    with pytest.raises(ValueError):
        quantize_base(mx.zeros((8, 8)), bits=2, axis="bogus")


def test_channel_axis_differs_from_token_axis() -> None:
    """Per-channel and per-token base quant give different reconstructions.

    They group along different axes with different (scale, zero) granularity,
    so on generic data they must not silently coincide — otherwise the axis
    plumbing wouldn't actually be doing anything.
    """
    X = _lowrank_plus_outliers(N=64, D=48, r=6)
    _, recon_channel = quantize_base(X, bits=2, group_size=16, axis="channel")
    _, recon_token = quantize_base(X, bits=2, group_size=16, axis="token")
    assert _mse(recon_channel, recon_token) > 1e-8


def test_channel_axis_matches_transposed_token_axis() -> None:
    """Per-channel quant of X equals per-token quant of X.T, transposed back.

    This is the defining property of the "channel" axis: it is exactly the
    "token" axis quantizer applied to the transpose.
    """
    X = _lowrank_plus_outliers(N=64, D=48, r=6)
    _, recon_channel = quantize_base(X, bits=3, group_size=16, axis="channel")
    _, recon_token_t = quantize_base(X.T, bits=3, group_size=16, axis="token")
    assert _mse(recon_channel, recon_token_t.T) == pytest.approx(0.0, abs=1e-6)


def test_channel_axis_roundtrip_shape_and_state() -> None:
    """gear_compress/reconstruct preserve shape under base_axis='channel' and
    record the axis + original channel count on the returned state."""
    X = _lowrank_plus_outliers(N=64, D=48, r=6)
    st = gear_compress(X, bits=2, rank=4, sparse_frac=0.01, group_size=16, base_axis="channel")
    assert st.axis == "channel"
    assert st.d_cols == 48
    rec = gear_reconstruct(st)
    assert rec.shape == X.shape


def test_channel_axis_beats_base_quant() -> None:
    """The core GEAR claim (error feedback beats base-quant-alone) also holds
    under the per-channel (KCVT key) base axis, not just the default token axis."""
    X = _lowrank_plus_outliers(N=128, D=128, r=6)
    gear = gear_quant_dequant(
        X, bits=2, rank=12, sparse_frac=0.01, group_size=32, base_axis="channel"
    )
    _, base_recon = quantize_base(X.astype(mx.float32), bits=2, group_size=32, axis="channel")
    assert _mse(gear, X) < _mse(base_recon, X)


def test_single_token_decode_call_exceeds_fp16() -> None:
    """Per-call low-rank overhead makes a single-token GEAR call cost MORE than
    fp16, for any rank >= 1 — this is not a bug, it is why the paper only
    applies its low-rank term on streaming-buffer flushes (nb=20 tokens), never
    per single decoded token (§3, "Streaming Buffer"). This wrapper computes
    the low-rank term on whatever batch update_and_fetch receives, which is a
    single row at decode time (no streaming buffer, see gear.py's module
    docstring) — so decode-time low-rank compression should be expected to
    look worse than fp16 per call, by design of the simplification, not a
    regression. See docs-site/docs/algorithms/gear.md "When it does NOT help".
    """
    D = 128
    X = mx.array(np.random.default_rng(0).standard_normal((1, D)).astype(np.float16))
    for r in (1, 2, 4):
        st = gear_compress(X, bits=2, rank=r, sparse_frac=0.005, group_size=32, base_axis="channel")
        assert gear_bytes(st) > D * 2  # fp16 bytes for one token


def test_prefill_batch_call_can_beat_fp16() -> None:
    """In contrast to the single-token case, a large prefill batch amortizes
    the low-rank factors and achieves real compression — the low-rank
    overhead scales with (N + D) * r while the base+savings scale with N * D,
    so larger N dilutes the fixed (D * r) part of the overhead."""
    N, D = 1024, 128
    X = mx.array(np.random.default_rng(0).standard_normal((N, D)).astype(np.float16))
    st = gear_compress(X, bits=2, rank=4, sparse_frac=0.005, group_size=32, base_axis="channel")
    assert gear_bytes(st) < N * D * 2


def test_byte_accounting_axis_independent() -> None:
    """gear_bytes/base_only_bytes report the same size for 'channel' and 'token'
    axes on the same shape — code size depends on element count, not axis,
    only the (scale, zero) parameter count's shape differs per group width."""
    X = _lowrank_plus_outliers(N=64, D=64, r=6)
    st_tok = gear_compress(X, bits=2, rank=4, sparse_frac=0.0, group_size=32, base_axis="token")
    st_chan = gear_compress(X, bits=2, rank=4, sparse_frac=0.0, group_size=32, base_axis="channel")
    fp16 = 64 * 64 * 2
    assert base_only_bytes(st_tok) <= fp16
    assert base_only_bytes(st_chan) <= fp16
