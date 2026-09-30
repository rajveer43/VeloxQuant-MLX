"""Parity checks for Kitty's batched (B*H) channel ranking + mixed-precision quant.

``KittyKVCache._quantize_keys`` used to loop ``for b in range(B): for h in
range(H):``, and its decode branch additionally did ``.tolist()`` + Python
``sorted()`` per head to rank channels by running variance — a host sync and
a pure-Python sort on every decode token. ``hi_mask_from_variance_batched``
replaces that with one ``mx.argsort`` call over all ``B*H`` rows, and
``quantize_mixed_channels_batched`` replaces the per-row
``quantize_mixed_channels`` (which itself gathers ragged hi/lo index lists)
with one batched group-quant at each bit-width selected via a boolean mask.
These tests confirm both are bit-for-bit equivalent to the original per-row
loop, for the prefill (batch-variance) and decode (running-variance) ranking
paths, including tie-heavy variance where sort stability matters.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.kitty import (
    compute_running_variance,
    hi_mask_from_variance_batched,
    quantize_mixed_channels,
    quantize_mixed_channels_batched,
    rank_channels_by_sensitivity,
)


def _looped_rank(var_h: mx.array, D: int, hi_fraction: float) -> tuple[list[int], list[int]]:
    """Exactly the decode-branch ranking KittyKVCache._quantize_keys used to do."""
    var_list = var_h.tolist()
    sorted_idx = sorted(range(D), key=lambda i: -var_list[i])
    n_hi = max(1, int(D * hi_fraction))
    hi_idx = sorted(sorted_idx[:n_hi])
    lo_idx = sorted(sorted_idx[n_hi:])
    return hi_idx, lo_idx


# ---------------------------------------------------------------------------
# 1. Prefill path: rank_channels_by_sensitivity (looped) vs.
#    hi_mask_from_variance_batched (batched over B*H)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "G,S,D,hi_fraction,hi_bit,lo_bit,gs",
    [
        (1, 32, 16, 0.25, 4, 2, 8),
        (5, 40, 16, 0.25, 4, 2, 8),
        (3, 64, 32, 0.125, 4, 2, 16),
        (2, 20, 8, 0.5, 6, 3, 32),  # group_size > S: single partial group
        (4, 1, 12, 0.25, 4, 2, 8),  # S=1: decode-shaped input through the prefill path
    ],
)
def test_batched_prefill_ranking_matches_looped(
    G: int, S: int, D: int, hi_fraction: float, hi_bit: int, lo_bit: int, gs: int
) -> None:
    rng = np.random.default_rng(100 + G + S + D)
    keys = mx.array(rng.standard_normal((G, S, D)).astype(np.float32))

    outs = []
    for i in range(G):
        hi_idx, lo_idx = rank_channels_by_sensitivity(keys[i], hi_fraction)
        outs.append(
            quantize_mixed_channels(
                keys[i], hi_idx, lo_idx, hi_bit=hi_bit, lo_bit=lo_bit, group_size=gs
            )
        )
    ref = mx.stack(outs, axis=0)

    variance = mx.var(keys.astype(mx.float32), axis=1)
    hi_mask = hi_mask_from_variance_batched(variance, hi_fraction)
    batched = quantize_mixed_channels_batched(
        keys, hi_mask, hi_bit=hi_bit, lo_bit=lo_bit, group_size=gs
    )

    mx.eval(ref, batched)
    assert mx.array_equal(ref, batched).item()


# ---------------------------------------------------------------------------
# 2. Decode path: running-variance ranking (looped, per-head) vs. batched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,H,D,hi_fraction,hi_bit,lo_bit,gs,n",
    [
        (1, 4, 12, 0.25, 4, 2, 8, 10),
        (3, 5, 16, 0.25, 4, 2, 8, 50),
        (2, 8, 32, 0.125, 4, 2, 16, 5),  # n small (near the n<2 boundary from above)
        (2, 3, 20, 0.5, 6, 3, 32, 100),
    ],
)
def test_batched_decode_ranking_matches_looped(
    B: int, H: int, D: int, hi_fraction: float, hi_bit: int, lo_bit: int, gs: int, n: int
) -> None:
    rng = np.random.default_rng(200 + B + H + D)
    key_sum = mx.array(rng.standard_normal((H, D)).astype(np.float32))
    key_sq_sum = mx.array((rng.standard_normal((H, D)) ** 2).astype(np.float32)) + key_sum * key_sum
    keys = mx.array(rng.standard_normal((B, H, 1, D)).astype(np.float32))
    flat = keys.reshape(B * H, 1, D)

    outs = []
    for b in range(B):
        for h in range(H):
            var_h = compute_running_variance(key_sum[h], key_sq_sum[h], n)
            hi_idx, lo_idx = _looped_rank(var_h, D, hi_fraction)
            outs.append(
                quantize_mixed_channels(
                    keys[b, h], hi_idx, lo_idx, hi_bit=hi_bit, lo_bit=lo_bit, group_size=gs
                )
            )
    ref = mx.stack(outs, axis=0)

    var_full = compute_running_variance(key_sum, key_sq_sum, n)
    variance = mx.broadcast_to(var_full[None, :, :], (B, H, D)).reshape(B * H, D)
    hi_mask = hi_mask_from_variance_batched(variance, hi_fraction)
    batched = quantize_mixed_channels_batched(
        flat, hi_mask, hi_bit=hi_bit, lo_bit=lo_bit, group_size=gs
    )

    mx.eval(ref, batched)
    assert mx.array_equal(ref, batched).item()


def test_batched_decode_ranking_matches_looped_tie_heavy() -> None:
    """All-zero variance (e.g. n<2, or genuinely constant channels): every
    channel ties, so ranking degenerates to pure index order. Confirms
    mx.argsort's tie-break agrees with Python's stable sorted()."""
    B, H, D, hi_fraction, hi_bit, lo_bit, gs = 2, 4, 16, 0.25, 4, 2, 8
    rng = np.random.default_rng(9)
    key_sum = mx.zeros((H, D))
    key_sq_sum = mx.zeros((H, D))
    keys = mx.array(rng.standard_normal((B, H, 1, D)).astype(np.float32))
    flat = keys.reshape(B * H, 1, D)

    var_full = compute_running_variance(key_sum, key_sq_sum, 0)  # n<2 -> zeros
    outs = []
    for b in range(B):
        for h in range(H):
            hi_idx, lo_idx = _looped_rank(var_full[h], D, hi_fraction)
            outs.append(
                quantize_mixed_channels(
                    keys[b, h], hi_idx, lo_idx, hi_bit=hi_bit, lo_bit=lo_bit, group_size=gs
                )
            )
    ref = mx.stack(outs, axis=0)

    variance = mx.broadcast_to(var_full[None, :, :], (B, H, D)).reshape(B * H, D)
    hi_mask = hi_mask_from_variance_batched(variance, hi_fraction)
    batched = quantize_mixed_channels_batched(
        flat, hi_mask, hi_bit=hi_bit, lo_bit=lo_bit, group_size=gs
    )

    mx.eval(ref, batched)
    assert mx.array_equal(ref, batched).item()


# ---------------------------------------------------------------------------
# 3. Hi-mask channel count sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hi_fraction", [0.0, 0.125, 0.25, 0.5, 1.0])
def test_hi_mask_channel_count_matches_formula(hi_fraction: float) -> None:
    G, D = 6, 32
    rng = np.random.default_rng(3)
    variance = mx.array(rng.uniform(0, 1, (G, D)).astype(np.float32))
    hi_mask = hi_mask_from_variance_batched(variance, hi_fraction)
    n_hi_expected = max(1, int(D * hi_fraction))
    counts = mx.sum(hi_mask.astype(mx.int32), axis=-1)
    mx.eval(counts)
    assert all(c == n_hi_expected for c in counts.tolist())
