"""Unit tests for CacheGen entropy-coding primitives."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.cachegen import (
    cachegen_quant_dequant,
    dequant_codes,
    entropy_coded_bytes,
    fixed_width_bytes,
    layer_group_bits,
    quantize_to_codes,
    symbol_entropy_bits,
    token_delta,
)


def test_quantize_dequant_roundtrip_shapes() -> None:
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((40, 32)).astype(np.float32))
    st = quantize_to_codes(x, bits=4, group_size=16)
    recon = dequant_codes(st)
    assert recon.shape == (40, 32)
    assert recon.dtype == mx.float16


def test_codes_in_range() -> None:
    rng = np.random.default_rng(1)
    x = mx.array(rng.standard_normal((32, 16)).astype(np.float32))
    st = quantize_to_codes(x, bits=3, group_size=16)
    mx.eval(st.codes)
    assert float(mx.min(st.codes).item()) >= 0
    assert float(mx.max(st.codes).item()) <= (1 << 3) - 1


def test_token_delta_reversible() -> None:
    rng = np.random.default_rng(2)
    codes = mx.array(rng.integers(0, 16, (20, 8)).astype(np.float32))
    delta = token_delta(codes)
    recovered = mx.cumsum(delta, axis=0)
    mx.eval(recovered)
    assert np.allclose(np.array(recovered), np.array(codes))


def test_entropy_zero_for_constant() -> None:
    s = mx.zeros((100,), dtype=mx.int32)
    assert symbol_entropy_bits(s) == pytest.approx(0.0, abs=1e-9)


def test_entropy_matches_uniform_two_symbols() -> None:
    # 50/50 two symbols → 1 bit
    s = mx.array(np.array([0, 1] * 50, dtype=np.int32))
    assert symbol_entropy_bits(s) == pytest.approx(1.0, abs=1e-6)


def test_entropy_bounded_by_log2_alphabet() -> None:
    rng = np.random.default_rng(3)
    s = mx.array(rng.integers(0, 16, (1000,)).astype(np.int32))
    assert symbol_entropy_bits(s) <= math.log2(16) + 1e-6


def test_entropy_bytes_capped_at_fixed_width() -> None:
    rng = np.random.default_rng(4)
    x = mx.array(rng.standard_normal((64, 32)).astype(np.float32))  # incompressible
    st = quantize_to_codes(x, bits=4, group_size=32)
    assert entropy_coded_bytes(st, use_delta=True) <= fixed_width_bytes(st)


def test_entropy_bytes_smaller_on_correlated() -> None:
    rng = np.random.default_rng(5)
    walk = np.cumsum(rng.standard_normal((128, 32)).astype(np.float32) * 0.1, axis=0)
    st = quantize_to_codes(mx.array(walk), bits=4, group_size=32)
    assert entropy_coded_bytes(st, use_delta=True) < fixed_width_bytes(st)


def test_drop_in_matches_group_quant() -> None:
    from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant

    rng = np.random.default_rng(6)
    x = mx.array(rng.standard_normal((48, 32)).astype(np.float32))
    a = cachegen_quant_dequant(x, 4, 16)
    b = _group_quant_dequant(x, 4, 16)
    mx.eval(a, b)
    assert bool(mx.all(a == b).item())


# ------------------------------------------------------------------
# Per-channel entropy grouping (§5.1.3)
# ------------------------------------------------------------------


def test_per_channel_entropy_smaller_or_equal_pooled() -> None:
    """Grouping by channel should never be worse than pooling all channels."""
    rng = np.random.default_rng(7)
    # Channels with very different scales/distributions: pooling should blur
    # the per-channel structure and yield higher (or equal) entropy.
    walk = np.cumsum(rng.standard_normal((128, 32)).astype(np.float32), axis=0)
    scales = np.concatenate([np.full(16, 0.05), np.full(16, 5.0)]).astype(np.float32)
    x = walk * scales
    st = quantize_to_codes(mx.array(x), bits=4, group_size=32)
    per_channel = entropy_coded_bytes(st, use_delta=True, per_channel=True)
    pooled = entropy_coded_bytes(st, use_delta=True, per_channel=False)
    assert per_channel <= pooled


def test_per_channel_entropy_capped_at_fixed_width() -> None:
    rng = np.random.default_rng(8)
    x = mx.array(rng.standard_normal((64, 16)).astype(np.float32))
    st = quantize_to_codes(x, bits=4, group_size=32)
    assert entropy_coded_bytes(st, use_delta=True, per_channel=True) <= fixed_width_bytes(st)


# ------------------------------------------------------------------
# Layer-wise bit schedule (§5.1.2/§5.2)
# ------------------------------------------------------------------


def test_layer_group_bits_non_increasing() -> None:
    schedule = layer_group_bits(n_layers=24, base_bits=4, n_groups=3)
    assert len(schedule) == 24
    assert schedule == sorted(schedule, reverse=True)
    assert schedule[0] == 4
    assert schedule[-1] < schedule[0]


def test_layer_group_bits_floored_at_two() -> None:
    schedule = layer_group_bits(n_layers=9, base_bits=3, n_groups=3)
    assert min(schedule) >= 2


def test_layer_group_bits_empty() -> None:
    assert layer_group_bits(n_layers=0, base_bits=4) == []


def test_layer_group_bits_single_layer() -> None:
    assert layer_group_bits(n_layers=1, base_bits=4) == [4]
