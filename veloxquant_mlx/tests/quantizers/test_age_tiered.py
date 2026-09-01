"""Tests for veloxquant_mlx.quantizers.age_tiered — the tiering/quantize
primitives AgeTieredKVCache wraps (issue #256)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.age_tiered import (
    MID,
    OLD,
    RECENT,
    age_tier_quantize,
    age_tiered_bytes,
    assign_age_tiers,
    default_age_tiers,
    full_fp16_bytes,
)


def test_assign_age_tiers_boundaries() -> None:
    ages = [0, 1, 2, 3, 4, 5]
    tiers = assign_age_tiers(ages, age_recent_boundary=2, age_mid_boundary=4)
    assert tiers == [RECENT, RECENT, MID, MID, OLD, OLD]


def test_assign_age_tiers_empty() -> None:
    assert assign_age_tiers([], age_recent_boundary=2, age_mid_boundary=4) == []


def test_assign_age_tiers_all_recent() -> None:
    ages = [0, 1]
    tiers = assign_age_tiers(ages, age_recent_boundary=100, age_mid_boundary=200)
    assert tiers == [RECENT, RECENT]


def test_default_age_tiers_bits() -> None:
    tiers = default_age_tiers(8, 4, 2)
    by_id = {t.tier: t.bits for t in tiers}
    assert by_id[RECENT] == 8
    assert by_id[MID] == 4
    assert by_id[OLD] == 2


def test_age_tier_quantize_roundtrips_to_fp16() -> None:
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((16, 32)).astype(np.float32))
    out = age_tier_quantize(x, bits=4, group_size=8)
    assert out.dtype == mx.float16
    assert out.shape == x.shape


def test_age_tier_quantize_16_bit_is_lossless_cast() -> None:
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((8, 16)).astype(np.float16))
    out = age_tier_quantize(x, bits=16, group_size=8)
    assert bool(mx.all(out == x).item())


def test_age_tier_quantize_empty_input() -> None:
    x = mx.zeros((0, 32), dtype=mx.float16)
    out = age_tier_quantize(x, bits=4, group_size=8)
    assert out.shape == (0, 32)


def test_age_tier_quantize_lower_bits_more_error() -> None:
    """Coarser bit-widths should not produce *less* quantization error than
    finer ones on the same data — a basic sanity check on tier ordering."""
    rng = np.random.default_rng(0)
    x = mx.array(rng.standard_normal((64, 32)).astype(np.float32))
    x16 = x.astype(mx.float16)

    out8 = age_tier_quantize(x16, bits=8, group_size=32)
    out2 = age_tier_quantize(x16, bits=2, group_size=32)

    err8 = float(mx.mean((out8.astype(mx.float32) - x) ** 2).item())
    err2 = float(mx.mean((out2.astype(mx.float32) - x) ** 2).item())
    assert err2 >= err8


def test_age_tiered_bytes_all_one_tier() -> None:
    tiers = default_age_tiers(8, 4, 2)
    b = age_tiered_bytes({RECENT: 10, MID: 0, OLD: 0}, tiers, head_dim=32)
    # RECENT: 8 bits/elem * 32 elems = 256 bits = 32 bytes/token, K+V = 64
    assert b == 10 * 64


def test_age_tiered_bytes_empty() -> None:
    tiers = default_age_tiers(8, 4, 2)
    assert age_tiered_bytes({RECENT: 0, MID: 0, OLD: 0}, tiers, head_dim=32) == 0


def test_full_fp16_bytes() -> None:
    assert full_fp16_bytes(tokens_seen=10, head_dim=32) == 10 * 32 * 2 * 2


def test_age_tiered_bytes_less_than_full_fp16_for_compressed_tiers() -> None:
    tiers = default_age_tiers(8, 4, 2)
    n = 100
    compressed = age_tiered_bytes({RECENT: 0, MID: 0, OLD: n}, tiers, head_dim=32)
    full = full_fp16_bytes(n, head_dim=32)
    assert compressed < full
