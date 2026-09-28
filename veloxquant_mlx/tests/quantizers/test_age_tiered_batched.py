"""Parity checks for AgeTieredKV's batched (B*H) per-group re-quantization.

``AgeTieredKVCache.update_and_fetch`` used to loop ``for b in range(B): for
h in range(H):``, and per head loop ``for g in range(n_groups):`` inside
``_requantize`` to quantize each fixed-size group at its own (head-invariant)
tier's bit-width, reusing a group's previous output byte-for-byte when its
tier hadn't changed (#397). ``age_tier_quantize_batched`` replaces the inner
loop with one batched group-quant-dequant call per distinct tier (always
exactly 3: RECENT/MID/OLD) over the whole ``[G, N, D]`` array, selected per
group via ``mx.where``; the cache's ``_requantize`` batches the "did this
group's tier change" splice the same way. These tests confirm the quantizer
primitive matches the original per-group loop bit-for-bit, including groups
whose members span more than one individual token tier (the group's tier is
defined by its first/oldest token only, matching the original's
``tiers[start]`` semantics) and bits>=16 (the no-op cast path, which
``_group_quant_dequant_batched`` does not special-case on its own).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.quantizers.age_tiered import (
    age_tier_quantize,
    age_tier_quantize_batched,
    assign_age_tiers,
    default_age_tiers,
)


def _make(**cfg):
    base = {"method": "age_tiered", "head_dim": 16}
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


# ---------------------------------------------------------------------------
# 1. Quantizer primitive: looped per-group vs. batched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,d,gs,bits_r,bits_m,bits_o,rb,mb,cp",
    [
        (100, 16, 8, 8, 4, 2, 20, 50, 60),  # mixed-tier group (see test module docstring)
        (32, 8, 32, 8, 4, 2, 10, 20, 40),  # single full group
        (1, 12, 8, 8, 4, 2, 5, 10, 3),  # single token
        (150, 20, 16, 16, 8, 4, 30, 90, 100),  # bits_recent=16: no-op-cast tier present
        (64, 8, 4, 4, 2, 1, 8, 24, 50),  # small group_size, low bit-widths
    ],
)
def test_batched_matches_looped_quantizer(
    n: int, d: int, gs: int, bits_r: int, bits_m: int, bits_o: int, rb: int, mb: int, cp: int
) -> None:
    rng = np.random.default_rng(100 + n + d)
    raw = mx.array(rng.standard_normal((n, d)).astype(np.float32))
    tiers_cfg = default_age_tiers(bits_r, bits_m, bits_o)
    by_tier = {c.tier: c.bits for c in tiers_cfg}
    ages = [cp - (i + 1) for i in range(n)]
    tiers = assign_age_tiers(ages, rb, mb)

    n_groups = (n + gs - 1) // gs
    out_chunks = []
    group_tier = []
    for g in range(n_groups):
        start, end = g * gs, min((g + 1) * gs, n)
        gt = tiers[start]
        group_tier.append(gt)
        out_chunks.append(age_tier_quantize(raw[start:end], by_tier[gt], gs))
    ref = mx.concatenate(out_chunks, axis=0)

    batched = age_tier_quantize_batched(raw[None], group_tier, tiers_cfg, gs)[0]
    mx.eval(ref, batched)
    assert mx.array_equal(ref, batched).item()


@pytest.mark.parametrize("G", [1, 3, 5])
def test_batched_quantizer_leading_axis_independent(G: int) -> None:
    """Different rows of the G axis (independent random content) must each
    be quantized exactly as if processed alone -- no cross-row contamination."""
    n, d, gs = 40, 12, 8
    tiers_cfg = default_age_tiers(8, 4, 2)
    by_tier = {c.tier: c.bits for c in tiers_cfg}
    rng = np.random.default_rng(7)
    raw = mx.array(rng.standard_normal((G, n, d)).astype(np.float32))
    ages = [50 - (i + 1) for i in range(n)]
    tiers = assign_age_tiers(ages, 15, 35)
    n_groups = (n + gs - 1) // gs
    group_tier = [tiers[g * gs] for g in range(n_groups)]

    batched = age_tier_quantize_batched(raw, group_tier, tiers_cfg, gs)
    mx.eval(batched)

    for g in range(G):
        out_chunks = []
        for gi in range(n_groups):
            start, end = gi * gs, min((gi + 1) * gs, n)
            out_chunks.append(age_tier_quantize(raw[g, start:end], by_tier[group_tier[gi]], gs))
        ref_row = mx.concatenate(out_chunks, axis=0)
        mx.eval(ref_row)
        assert mx.array_equal(ref_row, batched[g]).item()


def test_batched_quantizer_empty() -> None:
    tiers_cfg = default_age_tiers(8, 4, 2)
    raw = mx.zeros((2, 0, 8))
    out = age_tier_quantize_batched(raw, [], tiers_cfg, 32)
    assert out.shape == (2, 0, 8)


# ---------------------------------------------------------------------------
# 2. Cache-level: multi-head batching doesn't cross-contaminate heads
# ---------------------------------------------------------------------------


def test_multihead_cache_matches_per_head_reference() -> None:
    """Run AgeTieredKVCache with B*H > 1 heads of independent random data and
    confirm each head's output matches what a single-head cache produces
    when fed that head's data alone -- the key risk in batching over B*H is
    one head's content leaking into another's quantization."""
    H, D = 4, 16
    gs = 8
    cfg = {
        "age_recent_boundary": 6,
        "age_mid_boundary": 14,
        "age_bits_recent": 8,
        "age_bits_mid": 4,
        "age_bits_old": 2,
        "age_group_size": gs,
        "head_dim": D,
    }

    rng = np.random.default_rng(11)
    n_steps = 25
    all_k = [rng.standard_normal((n_steps, D)).astype(np.float16) for _ in range(H)]
    all_v = [rng.standard_normal((n_steps, D)).astype(np.float16) for _ in range(H)]

    multi = _make(**cfg)
    ko_multi = vo_multi = None
    for t in range(n_steps):
        k = mx.array(np.stack([all_k[h][t] for h in range(H)])[None, :, None, :])  # [1,H,1,D]
        v = mx.array(np.stack([all_v[h][t] for h in range(H)])[None, :, None, :])
        ko_multi, vo_multi = multi.update_and_fetch(k, v)

    for h in range(H):
        single = _make(**cfg)
        ko_single = vo_single = None
        for t in range(n_steps):
            k = mx.array(all_k[h][t][None, None, None, :])  # [1,1,1,D]
            v = mx.array(all_v[h][t][None, None, None, :])
            ko_single, vo_single = single.update_and_fetch(k, v)

        multi_h_k = np.array(ko_multi[0, h])
        multi_h_v = np.array(vo_multi[0, h])
        single_k = np.array(ko_single[0, 0])
        single_v = np.array(vo_single[0, 0])
        assert np.array_equal(multi_h_k, single_k), f"head {h} key mismatch"
        assert np.array_equal(multi_h_v, single_v), f"head {h} value mismatch"
