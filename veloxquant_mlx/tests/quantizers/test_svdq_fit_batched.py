"""Equivalence tests: batched [H, S, D] SVDq SVD-fit vs. the per-head
reference function (#569).

``svd_compress_keys_batched`` replaces ``SVDqKVCache._run_prefill_svd``'s
``for h in range(H):`` loop, each iteration calling ``svd_compress_keys``
once per head. Each head fits its own independent SVD basis (see
``svdq_cache.py``'s docstring on cross-head correlation) — no cross-head
coupling — so batching the SVD dispatch itself across all H heads is a pure
vectorization, not a policy change. Distinct from #562 (batching the
per-token project/quantize/reconstruct *application* of an already-fitted
basis, covered by test_svdq_batched.py) — this covers only the fit itself.

Each head can legitimately land at a different rank under energy-threshold
auto-rank (different heads' key distributions have different effective
dimensionality); verified directly via the "ragged" test cases below.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.svdq_cache import SVDqKVCache
from veloxquant_mlx.quantizers.svdq import (
    DEFAULT_BIT_SCHEDULE as DEFAULT_BIT_SCHEDULE_CHECK,
    svd_compress_keys,
    svd_compress_keys_batched,
)


def _rand(shape, seed):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float32))


@pytest.mark.parametrize(
    "desc,H,S,D,rank,energy,seed",
    [
        ("uniform_explicit_rank", 4, 30, 16, 8, 0.95, 0),
        ("energy_threshold_ragged", 5, 50, 32, None, 0.9, 1),
        ("aggressive_truncation_ragged", 4, 100, 16, None, 0.5, 2),
        ("small_S_relative_to_D", 3, 10, 16, None, 0.99, 3),
        ("single_head", 1, 20, 16, None, 0.95, 4),
        ("generous_energy_full_rank", 3, 20, 8, None, 0.999, 5),
    ],
)
def test_svd_compress_keys_batched_matches_reference(desc, H, S, D, rank, energy, seed):
    keys = _rand((H, S, D), seed)
    L_b, V_b, K_b, s_b = svd_compress_keys_batched(keys, rank=rank, energy_threshold=energy)

    for h in range(H):
        L_r, V_r, K_r, s_r = svd_compress_keys(keys[h], rank=rank, energy_threshold=energy)
        assert L_b[h].shape == L_r.shape, f"{desc}: head {h} L shape"
        assert V_b[h].shape == V_r.shape, f"{desc}: head {h} V shape"

        kd = float(mx.max(mx.abs(K_b[h] - K_r)).item())
        assert kd < 1e-3, f"{desc}: head {h} K_mean diverged by {kd}"

        if s_r.shape[0] > 0:
            sd = float(mx.max(mx.abs(s_b[h] - s_r)).item())
            assert sd < 1e-2, f"{desc}: head {h} singular values diverged by {sd}"

        # V has SVD sign ambiguity per component; compare via reconstruction
        # (the real correctness signal) instead of raw V/L equality.
        recon_b = L_b[h] @ V_b[h].T + K_b[h]
        recon_r = L_r @ V_r.T + K_r
        rd = float(mx.max(mx.abs(recon_b - recon_r)).item())
        assert rd < 1e-2, f"{desc}: head {h} reconstruction diverged by {rd}"


def test_svd_compress_keys_batched_ragged_ranks_differ_across_heads():
    """Sanity check that per-head effective rank actually differs when heads
    have different intrinsic dimensionality (otherwise the ragged-rank
    zero-pad/truncate-to-own-rank path wouldn't be tested at all). Construct
    heads directly: head h's keys live in a rank-(h+1) subspace, so
    energy-threshold auto-rank should recover approximately that rank per
    head, differing head to head."""
    S, D = 60, 16
    rng = np.random.default_rng(9)
    head_ranks = [1, 3, 6, 10]
    rows = []
    for r in head_ranks:
        basis = rng.standard_normal((r, D)).astype(np.float32)
        coeffs = rng.standard_normal((S, r)).astype(np.float32)
        rows.append(coeffs @ basis)
    keys = mx.array(np.stack(rows, axis=0))

    _, V_b, _, _ = svd_compress_keys_batched(keys, rank=None, energy_threshold=0.999)
    ranks = [int(V.shape[1]) for V in V_b]
    assert len(set(ranks)) > 1, f"expected ragged ranks, got uniform {ranks}"
    assert ranks == sorted(ranks), f"expected ranks to increase with head_ranks, got {ranks}"


# ---------------------------------------------------------------------------
# Real SVDqKVCache end-to-end vs. the true per-head reference (svd_compress_keys
# called directly in a loop, matching _run_prefill_svd's pre-batching form).
# ---------------------------------------------------------------------------


def _make_cache(**cfg):
    base = {"method": "svdq", "head_dim": 32, "svdq_rank": 32}
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


@pytest.mark.parametrize("H,seed", [(1, 0), (3, 1), (8, 2)])
def test_real_cache_prefill_matches_reference_loop(H, seed):
    """Uses an explicit svdq_rank=32 (== min_safe_rank for the default
    8-group schedule) so _resolve_safe_schedule's small-rank guard never
    degrades the schedule — keeping this test focused on the fit-batching
    equivalence (#569), not the unrelated schedule-degradation logic."""
    D, S = 32, 60
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float32)).astype(mx.float16)
    v = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float32)).astype(mx.float16)

    cache = _make_cache(head_dim=D)
    k_out, v_out = cache.update_and_fetch(k, v)
    assert cache._effective_schedule == [DEFAULT_BIT_SCHEDULE_CHECK] * H, (
        "test assumption violated: schedule was degraded, reference below would not match"
    )

    # Reference: fit each head's SVD independently (the pre-batching
    # per-head algorithm), then project/quantize/reconstruct exactly as the
    # cache's own (already-batched, #562) apply path does — isolating this
    # test to the fit step, which is what #569 batches.
    from veloxquant_mlx.quantizers.svdq import DEFAULT_BIT_SCHEDULE, project_quantize_reconstruct_batched

    V_list, K_mean_list = [], []
    for h in range(H):
        _, V, K_mean, _ = svd_compress_keys(k[0, h].astype(mx.float32), rank=32)
        V_list.append(V)
        K_mean_list.append(K_mean)
    schedules = [DEFAULT_BIT_SCHEDULE] * H
    ref_k = project_quantize_reconstruct_batched(
        k[0].astype(mx.float32), V_list, K_mean_list, schedules, 32
    )[None]

    kd = float(mx.max(mx.abs(k_out.astype(mx.float32) - ref_k.astype(mx.float32))).item())
    assert kd < 1e-1, f"H={H} seed={seed}: prefill output diverged by {kd}"
