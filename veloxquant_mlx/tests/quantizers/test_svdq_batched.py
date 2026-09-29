"""Parity checks for SVDq's batched per-head project/quantize/reconstruct (#562).

``SVDqKVCache._project_quantize_reconstruct`` used to loop ``for h in
range(H):``, projecting each head's keys through its own already-fitted SVD
basis, mixed-bit quantizing, and reconstructing — run on both prefill and
every decode step. Distinct from the SVD *fit* itself (``_run_prefill_svd``,
tracked separately in #569): this covers only the per-token application of
an already-fitted, ragged-rank basis.

``project_quantize_reconstruct_batched`` batches this by zero-padding every
head's projection basis to the batch's max rank for the projection/
reconstruction matmuls (a head's padded columns are exact zeros, so they
never leak into another head's output), then quantizing per distinct
``(rank, effective_schedule)`` group — since a head's own rank determines
its channel-group boundaries, heads with different ranks or (after the
small-rank degradation guard) different schedules cannot share one
quantize call.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.svdq import (
    DEFAULT_BIT_SCHEDULE,
    project_quantize_reconstruct_batched,
    quantize_latents_mixed,
    reconstruct_keys,
    svd_compress_keys,
)


def _ref_loop(keys_hsd, V_list, K_mean_list, schedules, group_size):
    H = keys_hsd.shape[0]
    out = []
    for h in range(H):
        k_h = keys_hsd[h].astype(mx.float32)
        k_centered = k_h - K_mean_list[h][None, :]
        L = k_centered @ V_list[h]
        L_q = quantize_latents_mixed(L, None, bit_schedule=schedules[h], group_size=group_size)
        out.append(reconstruct_keys(L_q, V_list[h], K_mean_list[h]))
    return mx.stack(out, axis=0)


def _fit_heads(rng, H, D, n_fit, ranks=None, energy_threshold=None):
    V_list, K_mean_list = [], []
    for h in range(H):
        k_h = mx.array(rng.standard_normal((n_fit, D)).astype(np.float32))
        rank = ranks[h] if ranks is not None else None
        et = energy_threshold if energy_threshold is not None else 0.95
        _, V, K_mean, _ = svd_compress_keys(k_h, rank=rank, energy_threshold=et)
        V_list.append(V)
        K_mean_list.append(K_mean)
    return V_list, K_mean_list


@pytest.mark.parametrize(
    "desc,make_case",
    [
        ("uniform_rank_schedule", "uniform"),
        ("ragged_ranks_energy_threshold", "ragged"),
        ("ragged_ranks_and_schedules", "ragged_schedule"),
        ("varied_explicit_ranks", "explicit_varied"),
    ],
)
def test_batched_matches_looped(desc: str, make_case: str) -> None:
    rng = np.random.default_rng(11)
    H, S, D = 6, 30, 16
    keys = mx.array(rng.standard_normal((H, S, D)).astype(np.float32))
    group_size = 8

    if make_case == "uniform":
        V_list, K_mean_list = _fit_heads(rng, H, D, 40, ranks=[8] * H)
        schedules = [DEFAULT_BIT_SCHEDULE] * H
    elif make_case == "ragged":
        V_list, K_mean_list = _fit_heads(rng, H, D, 40, energy_threshold=0.9)
        schedules = [DEFAULT_BIT_SCHEDULE] * H
    elif make_case == "ragged_schedule":
        V_list, K_mean_list = _fit_heads(rng, H, D, 40, energy_threshold=0.9)
        schedules = [
            DEFAULT_BIT_SCHEDULE if h % 2 == 0 else tuple(max(b, 1) for b in DEFAULT_BIT_SCHEDULE)
            for h in range(H)
        ]
    else:  # explicit_varied
        target_ranks = [2, 8, 16, 3, 12, 5]
        V_list, K_mean_list = _fit_heads(rng, H, D, 40, ranks=target_ranks)
        schedules = [DEFAULT_BIT_SCHEDULE] * H

    ref = _ref_loop(keys, V_list, K_mean_list, schedules, group_size)
    batched = project_quantize_reconstruct_batched(keys, V_list, K_mean_list, schedules, group_size)
    mx.eval(ref, batched)
    assert mx.array_equal(ref, batched).item()


def test_batched_matches_looped_single_head() -> None:
    rng = np.random.default_rng(3)
    H, S, D = 1, 20, 16
    keys = mx.array(rng.standard_normal((H, S, D)).astype(np.float32))
    V_list, K_mean_list = _fit_heads(rng, H, D, 30, ranks=[6])
    schedules = [DEFAULT_BIT_SCHEDULE]

    ref = _ref_loop(keys, V_list, K_mean_list, schedules, 8)
    batched = project_quantize_reconstruct_batched(keys, V_list, K_mean_list, schedules, 8)
    mx.eval(ref, batched)
    assert mx.array_equal(ref, batched).item()


def test_batched_matches_looped_decode_shaped() -> None:
    """S=1 -- the actual decode hot path this fix targets."""
    rng = np.random.default_rng(5)
    H, S, D = 8, 1, 16
    keys = mx.array(rng.standard_normal((H, S, D)).astype(np.float32))
    V_list, K_mean_list = _fit_heads(rng, H, D, 40, energy_threshold=0.9)
    schedules = [DEFAULT_BIT_SCHEDULE] * H

    ref = _ref_loop(keys, V_list, K_mean_list, schedules, 8)
    batched = project_quantize_reconstruct_batched(keys, V_list, K_mean_list, schedules, 8)
    mx.eval(ref, batched)
    assert mx.array_equal(ref, batched).item()
