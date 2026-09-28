"""Parity checks for AMC's batched (B*H) saliency scoring, tier assignment,
and rank-mask + quantize compression.

``AMCKVCache.update_and_fetch`` used to loop ``for b in range(B): for h in
range(H):``, scoring saliency, assigning tiers via a Python
:class:`~veloxquant_mlx.dsa.heap.MaxHeap` top-k (itself looping
``.tolist()`` per head), and then compressing tokens with a further
``for i in range(n):`` loop calling ``_group_quant_dequant`` on single-row
``[1, D]`` slices. ``amc_assign_tiers_batched`` replaces the heap-based
ranking with one ``mx.argsort`` call over all ``G = B*H`` rows;
``amc_compress_tokens_batched`` replaces the nested per-token loop with one
batched pass per tier (there are always exactly 3) selected via
``mx.where``. These tests confirm both are bit-for-bit equivalent to the
original per-row/per-token loops on tie-free saliency (the realistic case —
saliency is a continuous function of float activations, so exact ties are
measure-zero); exact-tie inputs are a known, documented divergence (heap pop
order on equal priority is an unspecified :class:`MaxHeap` array-layout
artifact, not a tie-break guarantee either version promises).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.amc import (
    HIGH,
    LOW,
    MID,
    _tier_config_for_dim,
    amc_apply_rank_mask,
    amc_assign_tiers,
    amc_assign_tiers_batched,
    amc_compress_tokens_batched,
    amc_quantize_tier,
    amc_query_aware_saliency,
    amc_query_aware_saliency_batched,
    amc_saliency,
)

# ---------------------------------------------------------------------------
# 1. Tier assignment: heap-based (looped) vs. argsort-based (batched)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "G,N,k_high,k_mid",
    [
        (1, 8, 0.20, 0.30),
        (5, 12, 0.20, 0.30),
        (3, 1, 0.25, 0.25),
        (4, 40, 0.125, 0.5),
        (2, 3, 0.5, 0.5),
    ],
)
def test_batched_tier_assignment_matches_looped_tie_free(
    G: int, N: int, k_high: float, k_mid: float
) -> None:
    rng = np.random.default_rng(100 + G + N)
    # Continuous saliency values: ties are measure-zero.
    saliency = mx.array(rng.uniform(0.01, 0.99, (G, N)).astype(np.float32))

    ref = [amc_assign_tiers(saliency[g], k_high, k_mid) for g in range(G)]
    batched = amc_assign_tiers_batched(saliency, k_high, k_mid)
    mx.eval(batched)

    assert batched.tolist() == ref


def test_batched_tier_assignment_empty() -> None:
    saliency = mx.zeros((3, 0))
    batched = amc_assign_tiers_batched(saliency, 0.2, 0.3)
    assert batched.shape == (3, 0)


# ---------------------------------------------------------------------------
# 2. Compression: per-token loop vs. batched mask-select
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "G,N,D",
    [
        (1, 10, 16),
        (4, 8, 24),
        (3, 1, 12),  # decode-shaped: single token per group
        (2, 20, 32),
    ],
)
def test_batched_compression_matches_looped_tie_free(G: int, N: int, D: int) -> None:
    tier_configs = {t: _tier_config_for_dim(t, D) for t in (HIGH, MID, LOW)}
    rng = np.random.default_rng(200 + G + N + D)
    x = mx.array(rng.standard_normal((G, N, D)).astype(np.float32))
    saliency = amc_saliency(x)

    ref_groups = []
    for g in range(G):
        tiers = amc_assign_tiers(saliency[g], 0.2, 0.3)
        out_rows = []
        for i in range(N):
            cfg = tier_configs[tiers[i]]
            row = x[g, i : i + 1]
            row = amc_apply_rank_mask(row, cfg.rank)
            row = amc_quantize_tier(row, cfg.bits, 32)
            out_rows.append(row)
        ref_groups.append(mx.concatenate(out_rows, axis=0))
    ref = mx.stack(ref_groups, axis=0)

    tiers_batched = amc_assign_tiers_batched(saliency, 0.2, 0.3)
    out = amc_compress_tokens_batched(x, tiers_batched, tier_configs)

    mx.eval(ref, out)
    assert mx.array_equal(ref, out).item()


def test_batched_compression_2d_single_group() -> None:
    """amc_compress_tokens_batched also accepts unbatched [N, D] input."""
    D = 16
    tier_configs = {t: _tier_config_for_dim(t, D) for t in (HIGH, MID, LOW)}
    rng = np.random.default_rng(9)
    x = mx.array(rng.standard_normal((10, D)).astype(np.float32))
    saliency = amc_saliency(x)
    tiers = amc_assign_tiers(saliency, 0.2, 0.3)

    out_rows = []
    for i in range(10):
        cfg = tier_configs[tiers[i]]
        row = x[i : i + 1]
        row = amc_apply_rank_mask(row, cfg.rank)
        row = amc_quantize_tier(row, cfg.bits, 32)
        out_rows.append(row)
    ref = mx.concatenate(out_rows, axis=0)

    tiers_mx = mx.array(tiers)
    out = amc_compress_tokens_batched(x, tiers_mx, tier_configs)
    mx.eval(ref, out)
    assert ref.shape == out.shape
    assert mx.array_equal(ref, out).item()


# ---------------------------------------------------------------------------
# 3. Query-aware saliency: looped vs. batched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("G,N,D,alpha", [(1, 8, 16, 0.5), (4, 6, 24, 0.25), (3, 1, 12, 0.8)])
def test_batched_query_aware_saliency_matches_looped(G: int, N: int, D: int, alpha: float) -> None:
    rng = np.random.default_rng(300 + G + N + D)
    x = mx.array(rng.standard_normal((G, N, D)).astype(np.float32))
    keys = mx.array(rng.standard_normal((G, N, D)).astype(np.float32))
    query = mx.mean(keys.astype(mx.float32), axis=1)  # [G, D]

    ref = mx.stack(
        [amc_query_aware_saliency(x[g], keys[g], query[g], alpha=alpha) for g in range(G)]
    )
    batched = amc_query_aware_saliency_batched(x, keys, query, alpha=alpha)
    mx.eval(ref, batched)

    # Batched matmul reduction order differs slightly from per-row dot
    # products (fp32 accumulation order) -- verified tight, not bit-exact.
    assert mx.max(mx.abs(ref.astype(mx.float32) - batched.astype(mx.float32))).item() < 1e-5
