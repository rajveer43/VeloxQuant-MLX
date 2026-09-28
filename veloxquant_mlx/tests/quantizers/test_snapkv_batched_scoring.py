"""Parity checks for SnapKV's opt-in ``batched_scoring`` path.

``_snapkv_compress_batched(..., batched_scoring=True)`` replaces the default
per-head Python loop over ``obs_window_attention_scores`` with a single
batched matmul across all ``B*H`` heads at once. Both paths compute the same
math (same matmul, same softmax, same mean-pool), issued differently — a
loop of small matmuls versus one batched matmul. Per
``docs/SNAPKV_METAL_FINDINGS.md``, this was shipped opt-in pending a check
that MLX's batched-matmul kernel doesn't reduce in a different float32 order
closely enough to flip a near-tie top-k selection boundary in
``snap_select_indices`` (which ranks by exact `==` threshold membership, so a
sub-ULP score perturbation could genuinely move a token across the cutoff).
These tests are that check.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.snapkv import (
    _snap_select_batched,
    _snapkv_compress_batched,
    obs_window_attention_scores,
)


def _rand_kv_batch(B: int, H: int, S: int, D: int, seed: int) -> tuple[mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32))
    V = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32))
    return K, V


def _looped_scores(flat_k: mx.array, obs_window: int) -> mx.array:
    """The per-head reference: exactly what batched_scoring=False computes."""
    bh = flat_k.shape[0]
    return mx.stack([obs_window_attention_scores(flat_k[g], obs_window) for g in range(bh)])


def _batched_scores(flat_k: mx.array, obs_window: int) -> mx.array:
    """The batched path's score computation, isolated from selection/gather."""
    import math

    S = flat_k.shape[1]
    D = flat_k.shape[2]
    w = min(max(obs_window, 1), S)
    k32 = flat_k.astype(mx.float32)
    logits = (k32[:, -w:] @ mx.swapaxes(k32, -1, -2)) / math.sqrt(D)
    return mx.mean(mx.softmax(logits, axis=-1), axis=-2)


# ---------------------------------------------------------------------------
# 1. Score-level parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,H,S,D",
    [
        (1, 1, 64, 128),
        (1, 8, 128, 64),
        (2, 4, 256, 32),
        (1, 64, 512, 128),  # large B*H, matching PyramidKV-benchmark head counts
    ],
)
def test_batched_scores_match_looped_scores(B: int, H: int, S: int, D: int) -> None:
    K, _ = _rand_kv_batch(B, H, S, D, seed=1)
    flat_k = K.reshape(B * H, S, D)
    looped = _looped_scores(flat_k, obs_window=16)
    batched = _batched_scores(flat_k, obs_window=16)
    # float32 matmul is not associative: a batched matmul over B*H and a
    # Python loop of B*H individual matmuls can reduce in different orders.
    # rtol=1e-5 is ~10x float32 machine epsilon (1.19e-7) per accumulated
    # dot product of length D<=128 — generous enough to absorb reordering,
    # tight enough to catch a real algorithmic divergence.
    np.testing.assert_allclose(
        np.array(looped.tolist()), np.array(batched.tolist()), rtol=1e-5, atol=1e-6
    )


def test_batched_scores_match_looped_scores_many_seeds() -> None:
    """Broader sweep: many seeds/shapes at once, still checking every value."""
    for seed in range(20):
        rng = np.random.default_rng(seed)
        B = int(rng.integers(1, 3))
        H = int(rng.integers(1, 33))
        S = int(rng.integers(8, 300))
        D = int(rng.integers(8, 128))
        K, _ = _rand_kv_batch(B, H, S, D, seed=seed + 1000)
        flat_k = K.reshape(B * H, S, D)
        w = int(rng.integers(1, S + 1))
        looped = _looped_scores(flat_k, obs_window=w)
        batched = _batched_scores(flat_k, obs_window=w)
        np.testing.assert_allclose(
            np.array(looped.tolist()),
            np.array(batched.tolist()),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"seed={seed} B={B} H={H} S={S} D={D} w={w}",
        )


# ---------------------------------------------------------------------------
# 2. Selection parity under near-ties
# ---------------------------------------------------------------------------


def test_selection_parity_deliberate_near_tie() -> None:
    """Construct keys designed to produce near-duplicate scores at the exact
    top-k cutoff, then confirm both scoring paths select identical indices.

    Near-duplicate scores are engineered by making two key rows almost
    identical (one a copy of the other plus a tiny perturbation), which
    drives their post-softmax importance scores to within float32 ULP-level
    distance of each other — the scenario where a reduction-order difference
    between the looped and batched matmul could plausibly flip which side of
    `_snap_select_batched`'s `==` threshold comparison a score falls on.
    """
    B, H, S, D = 1, 16, 128, 64
    budget, n_sink = 32, 4
    rng = np.random.default_rng(7)
    K = rng.standard_normal((B, H, S, D)).astype(np.float32)
    # For every head, clone a token near the likely budget-th/[budget+1]-th
    # rank boundary and perturb it by ~1 ULP so its score nearly ties its
    # source token's score.
    for h in range(H):
        src = budget - 1
        dup = budget
        K[0, h, dup] = K[0, h, src] * (1.0 + np.float32(1e-7))
    K_mx = mx.array(K)
    flat_k = K_mx.reshape(B * H, S, D)

    scores_looped = _looped_scores(flat_k, obs_window=16)
    scores_batched = _batched_scores(flat_k, obs_window=16)

    idx_looped = _snap_select_batched(scores_looped, budget, n_sink, backend="mlx")
    idx_batched = _snap_select_batched(scores_batched, budget, n_sink, backend="mlx")

    assert idx_looped.tolist() == idx_batched.tolist()


def test_selection_parity_random_trials() -> None:
    """Broad randomized sweep: any selection-boundary disagreement across
    thousands of trials would indicate the batched path is unsafe to
    default on, regardless of whether a hand-built near-tie triggers it."""
    n_trials = 2000
    disagreements = []
    rng = np.random.default_rng(42)
    for trial in range(n_trials):
        B = 1
        H = int(rng.integers(1, 9))
        S = int(rng.integers(8, 130))
        D = int(rng.integers(8, 65))
        budget = int(rng.integers(1, S + 1))
        n_sink = int(rng.integers(0, min(budget, 8) + 1))
        w = int(rng.integers(1, S + 1))
        K, _ = _rand_kv_batch(B, H, S, D, seed=trial)
        flat_k = K.reshape(B * H, S, D)

        scores_looped = _looped_scores(flat_k, obs_window=w)
        scores_batched = _batched_scores(flat_k, obs_window=w)

        idx_looped = _snap_select_batched(scores_looped, budget, n_sink, backend="mlx")
        idx_batched = _snap_select_batched(scores_batched, budget, n_sink, backend="mlx")

        if idx_looped.tolist() != idx_batched.tolist():
            disagreements.append((trial, B, H, S, D, budget, n_sink, w))

    assert not disagreements, (
        f"{len(disagreements)}/{n_trials} trials disagreed on selection "
        f"between looped and batched scoring: {disagreements[:10]}"
    )


# ---------------------------------------------------------------------------
# 3. End-to-end compress parity (full function, both scoring modes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,H,S,D,budget,n_sink,obs_window",
    [
        (1, 8, 64, 128, 16, 2, 8),
        (2, 4, 200, 32, 50, 4, 16),
        (1, 32, 512, 64, 128, 8, 32),
    ],
)
def test_compress_batched_end_to_end_parity(
    B: int, H: int, S: int, D: int, budget: int, n_sink: int, obs_window: int
) -> None:
    K, V = _rand_kv_batch(B, H, S, D, seed=99)
    k_loop, v_loop, idx_loop = _snapkv_compress_batched(
        K, V, budget, obs_window, n_sink, backend="mlx", batched_scoring=False,
        return_indices=True,
    )
    k_batch, v_batch, idx_batch = _snapkv_compress_batched(
        K, V, budget, obs_window, n_sink, backend="mlx", batched_scoring=True,
        return_indices=True,
    )
    assert idx_loop.tolist() == idx_batch.tolist()
    assert k_loop.tolist() == k_batch.tolist()
    assert v_loop.tolist() == v_batch.tolist()
