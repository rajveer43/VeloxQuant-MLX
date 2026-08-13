"""Tests for the ZipCache-adapted quantizer numerics.

ZipCache-adapted routes high-norm tokens to hi_bits and low-norm tokens to
lo_bits within the quantized space. These tests cover: saliency mask
correctness, channel quant round-trips, compress/reconstruct shapes,
uniform-bits edge cases, byte ordering, values-off path, and determinism.
All data is synthetic — no model loading.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.zipcache import (
    ZipCacheState,
    base_only_bytes,
    channel_dequant,
    channel_quant,
    saliency_mask,
    token_key_norms,
    zipcache_bytes,
    zipcache_compress,
    zipcache_quant_dequant,
    zipcache_reconstruct,
)


def _rand(shape, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float32) * scale)


def _mse(a: mx.array, b: mx.array) -> float:
    return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())


# ---------------------------------------------------------------------------
# Saliency helpers
# ---------------------------------------------------------------------------


def test_token_key_norms_shape() -> None:
    x = _rand((16, 64))
    norms = token_key_norms(x)
    assert norms.shape == (16,)
    assert norms.dtype == mx.float32


def test_saliency_mask_selects_top_fraction() -> None:
    """saliency_mask marks exactly ceil(S * hi_fraction) tokens as True."""
    S, D = 20, 32
    rng = np.random.default_rng(1)
    norms = mx.array(rng.standard_normal(S).astype(np.float32) ** 2 + 0.01)
    hi_fraction = 0.25
    mask = saliency_mask(norms, hi_fraction)
    assert mask.shape == (S,)
    n_hi = int(mask.astype(mx.int32).sum().item())
    assert n_hi == math.ceil(S * hi_fraction)


def test_saliency_mask_selects_highest_norms() -> None:
    """The selected tokens are genuinely the top-k by norm."""
    S = 10
    norms = mx.array([float(i) for i in range(S)], dtype=mx.float32)  # 0..9
    mask = saliency_mask(norms, hi_fraction=0.3)
    selected = [i for i, m in enumerate(mask.tolist()) if m]
    # top 3 (ceil(10*0.3)=3) should be 7, 8, 9
    assert set(selected) == {7, 8, 9}


def test_saliency_mask_zero_fraction() -> None:
    norms = mx.ones((8,), dtype=mx.float32)
    mask = saliency_mask(norms, hi_fraction=0.0)
    assert int(mask.astype(mx.int32).sum().item()) == 0


def test_saliency_mask_full_fraction() -> None:
    norms = mx.ones((8,), dtype=mx.float32)
    mask = saliency_mask(norms, hi_fraction=1.0)
    assert int(mask.astype(mx.int32).sum().item()) == 8


# ---------------------------------------------------------------------------
# channel_quant / channel_dequant
# ---------------------------------------------------------------------------


def test_channel_quant_4bit_near_lossless() -> None:
    """4-bit channel quant round-trip has cosine > 0.995 on smooth data."""
    x = _rand((64, 128), seed=2, scale=1.0)
    codes, scales, zeros = channel_quant(x, bits=4, group_size=32)
    recon = channel_dequant(codes, scales, zeros, group_size=32)
    x_norm = x / (mx.linalg.norm(x.reshape(-1)) + 1e-8)
    r_norm = recon.astype(mx.float32) / (
        mx.linalg.norm(recon.astype(mx.float32).reshape(-1)) + 1e-8
    )
    cosine = float((x_norm.reshape(-1) * r_norm.reshape(-1)).sum().item())
    assert cosine > 0.995


def test_channel_quant_2bit_lossy_bounded() -> None:
    """2-bit channel quant round-trip is lossy but cosine > 0.8 on smooth data."""
    x = _rand((64, 128), seed=3)
    codes, scales, zeros = channel_quant(x, bits=2, group_size=32)
    recon = channel_dequant(codes, scales, zeros, group_size=32)
    x_norm = x / (mx.linalg.norm(x.reshape(-1)) + 1e-8)
    r_norm = recon.astype(mx.float32) / (
        mx.linalg.norm(recon.astype(mx.float32).reshape(-1)) + 1e-8
    )
    cosine = float((x_norm.reshape(-1) * r_norm.reshape(-1)).sum().item())
    assert cosine > 0.8


def test_channel_quant_empty_input() -> None:
    """Empty (0-token) input returns empty tensors without error."""
    codes, scales, zeros = channel_quant(mx.zeros((0, 64)), bits=4, group_size=32)
    assert codes.shape == (0, 64)
    assert scales.shape == (0, 64)


# ---------------------------------------------------------------------------
# zipcache_compress / zipcache_reconstruct
# ---------------------------------------------------------------------------


def test_compress_returns_state() -> None:
    x = _rand((32, 64))
    state = zipcache_compress(x, hi_bits=4, lo_bits=2, hi_fraction=0.25, group_size=32)
    assert isinstance(state, ZipCacheState)
    assert state.seq_len == 32
    assert state.head_dim == 64


def test_reconstruct_shape_preserved() -> None:
    S, D = 48, 128
    x = _rand((S, D))
    state = zipcache_compress(x, hi_bits=4, lo_bits=2, hi_fraction=0.2)
    recon = zipcache_reconstruct(state)
    assert recon.shape == (S, D)
    assert recon.dtype == mx.float16


def test_uniform_hi_bits_roundtrip() -> None:
    """hi_fraction=1 (all hi_bits) has lower MSE than hi_fraction=0 (all lo_bits)."""
    x = _rand((64, 128))
    mse_hi = _mse(zipcache_quant_dequant(x, hi_bits=4, lo_bits=2, hi_fraction=1.0), x)
    mse_lo = _mse(zipcache_quant_dequant(x, hi_bits=4, lo_bits=2, hi_fraction=0.0), x)
    assert mse_hi < mse_lo


def test_hi_fraction_zero_no_error() -> None:
    """hi_fraction=0 (all lo_bits) runs without error."""
    x = _rand((32, 64))
    out = zipcache_quant_dequant(x, hi_bits=4, lo_bits=2, hi_fraction=0.0)
    assert out.shape == (32, 64)


def test_hi_fraction_one_no_error() -> None:
    """hi_fraction=1 (all hi_bits) runs without error."""
    x = _rand((32, 64))
    out = zipcache_quant_dequant(x, hi_bits=4, lo_bits=2, hi_fraction=1.0)
    assert out.shape == (32, 64)


# ---------------------------------------------------------------------------
# Regression for #78: zipcache_reconstruct hardcoded group_size=32, silently
# desyncing dequant grouping from the group_size actually used at compress
# time for any non-default zipcache_group_size.
# ---------------------------------------------------------------------------


def test_state_stores_compress_time_group_size() -> None:
    x = _rand((40, 8))
    state = zipcache_compress(x, hi_bits=4, lo_bits=2, hi_fraction=0.2, group_size=8)
    assert state.group_size == 8


def test_reconstruct_uses_matching_group_size_not_hardcoded_32() -> None:
    """A non-default group_size must reconstruct with error comparable to the
    default group_size=32 case, not the ~2.6x-higher error a hardcoded
    mismatched grouping produces (the exact repro from #78)."""
    x = _rand((40, 8), seed=0)

    state_gs8 = zipcache_compress(x, hi_bits=4, lo_bits=2, hi_fraction=0.2, group_size=8)
    recon_gs8 = zipcache_reconstruct(state_gs8)
    mse_gs8 = _mse(recon_gs8, x)

    state_gs32 = zipcache_compress(x, hi_bits=4, lo_bits=2, hi_fraction=0.2, group_size=32)
    recon_gs32 = zipcache_reconstruct(state_gs32)
    mse_gs32 = _mse(recon_gs32, x)

    # Both are ordinary quantization noise, same order of magnitude — not the
    # inflated error a group_size mismatch produces.
    assert mse_gs8 < mse_gs32 * 2


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_byte_ordering_compressed_lt_fp16() -> None:
    """Mixed-bit stored size is strictly below fp16."""
    S, D = 128, 128
    x = _rand((S, D))
    state = zipcache_compress(x, hi_bits=4, lo_bits=2, hi_fraction=0.2)
    comp = zipcache_bytes(state, group_size=32)
    fp16 = S * D * 2
    assert comp < fp16


def test_byte_ordering_between_lo_and_fp16() -> None:
    """Mixed-bit bytes > all-lo-bit baseline (saliency hi-bits add overhead)."""
    S, D = 128, 128
    x = _rand((S, D))
    state = zipcache_compress(x, hi_bits=4, lo_bits=2, hi_fraction=0.2)
    comp = zipcache_bytes(state, group_size=32)
    all_lo = base_only_bytes(S, D, bits=2, group_size=32)
    fp16 = S * D * 2
    assert comp >= all_lo
    assert comp < fp16


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic() -> None:
    x = _rand((64, 128), seed=7)
    out1 = zipcache_quant_dequant(x, hi_bits=4, lo_bits=2, hi_fraction=0.2)
    out2 = zipcache_quant_dequant(x, hi_bits=4, lo_bits=2, hi_fraction=0.2)
    assert _mse(out1, out2) == pytest.approx(0.0, abs=0.0)
