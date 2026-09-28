"""Unit tests for CacheGen entropy-coding primitives."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.cachegen import (
    cachegen_quant_dequant,
    dequant_codes,
    dequant_codes_batched,
    entropy_coded_bytes,
    entropy_coded_bytes_batched,
    fixed_width_bytes,
    fixed_width_bytes_batched,
    layer_group_bits,
    quantize_to_codes,
    quantize_to_codes_batched,
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


# ------------------------------------------------------------------
# Batched (B*H) parity — quantize_to_codes_batched / dequant_codes_batched /
# entropy_coded_bytes_batched / fixed_width_bytes_batched vs. the per-head
# loop they replace in CacheGenKVCache._quant_and_account.
# ------------------------------------------------------------------


def _looped_quant_dequant(x: mx.array, bits: int, gs: int) -> mx.array:
    g = x.shape[0]
    return mx.stack([dequant_codes(quantize_to_codes(x[i], bits, gs)) for i in range(g)])


def _looped_bytes(x: mx.array, bits: int, gs: int, use_delta: bool, per_channel: bool):
    g = x.shape[0]
    comp, fixed = [], []
    for i in range(g):
        st = quantize_to_codes(x[i], bits, gs)
        comp.append(entropy_coded_bytes(st, use_delta=use_delta, per_channel=per_channel))
        fixed.append(fixed_width_bytes(st))
    return comp, fixed


@pytest.mark.parametrize(
    "G,S,D,bits,gs",
    [
        (3, 1, 8, 4, 32),  # decode: single token
        (4, 33, 16, 3, 32),  # S not a multiple of group_size
        (2, 65, 24, 8, 32),  # 8-bit: largest alphabet
        (1, 32, 8, 2, 16),  # 2-bit: smallest alphabet, group_size == S
        (5, 5, 8, 4, 32),  # S < group_size: single partial group
        (6, 130, 48, 4, 32),  # general case, several heads
    ],
)
def test_batched_dequant_matches_looped(G: int, S: int, D: int, bits: int, gs: int) -> None:
    rng = np.random.default_rng(100 + G + S + D)
    x = mx.array(rng.standard_normal((G, S, D)).astype(np.float32))
    recon_loop = _looped_quant_dequant(x, bits, gs)
    recon_batch = dequant_codes_batched(quantize_to_codes_batched(x, bits, gs))
    mx.eval(recon_loop, recon_batch)
    assert mx.array_equal(recon_loop, recon_batch).item()


@pytest.mark.parametrize("use_delta", [True, False])
@pytest.mark.parametrize("per_channel", [True, False])
@pytest.mark.parametrize(
    "G,S,D,bits,gs",
    [
        (3, 1, 8, 4, 32),
        (4, 33, 16, 3, 32),
        (2, 65, 24, 8, 32),
        (1, 32, 8, 2, 16),
        (5, 5, 8, 4, 32),
    ],
)
def test_batched_bytes_match_looped_random(
    G: int, S: int, D: int, bits: int, gs: int, per_channel: bool, use_delta: bool
) -> None:
    rng = np.random.default_rng(200 + G + S + D)
    x = mx.array(rng.standard_normal((G, S, D)).astype(np.float32))
    comp_loop, fixed_loop = _looped_bytes(x, bits, gs, use_delta, per_channel)
    stream = quantize_to_codes_batched(x, bits, gs)
    comp_batch = entropy_coded_bytes_batched(stream, use_delta=use_delta, per_channel=per_channel)
    fixed_batch = fixed_width_bytes_batched(stream)
    assert comp_loop == comp_batch
    assert fixed_loop == fixed_batch


@pytest.mark.parametrize("use_delta", [True, False])
@pytest.mark.parametrize("per_channel", [True, False])
def test_batched_bytes_match_looped_correlated(use_delta: bool, per_channel: bool) -> None:
    """Correlated (random-walk) data — the case where entropy coding
    actually beats fixed-width, exercising the interesting branch rather
    than just the incompressible-data cap."""
    rng = np.random.default_rng(7)
    G, S, D, bits, gs = 5, 200, 40, 4, 32
    walk = np.cumsum(rng.standard_normal((G, S, D)).astype(np.float32) * 0.1, axis=1)
    x = mx.array(walk)
    comp_loop, fixed_loop = _looped_bytes(x, bits, gs, use_delta, per_channel)
    stream = quantize_to_codes_batched(x, bits, gs)
    comp_batch = entropy_coded_bytes_batched(stream, use_delta=use_delta, per_channel=per_channel)
    fixed_batch = fixed_width_bytes_batched(stream)
    assert comp_loop == comp_batch
    assert fixed_loop == fixed_batch
    # sanity: this data should actually show entropy coding beating fixed-width
    assert sum(comp_loop) < sum(fixed_loop)
