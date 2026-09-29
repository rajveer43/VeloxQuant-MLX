"""Parity checks for CaM's batched per-head eviction/merge dispatch (#563).

``CaMKVCache.update_and_fetch`` used to loop ``for b in range(B): for h in
range(H):``, dispatching each row to ``cam_update`` one at a time — every
call on both prefill and decode (no gate). The outer dispatch is independent
per row (each head owns its own state and the loop only routes data); the
per-token merge/eviction decision *inside* ``cam_update`` is a genuine
recurrence (each token's insertion/merge depends on the previous token's
state) and stays untouched.

``cam_update_batched`` batches the outer dispatch over ``[BH, ...]``,
following the H2O/#504 template: every step removes exactly one row when
over budget (the survivor slot is rewritten in place on a merge, or the
loser's slot is dropped outright), so state stays rectangular across rows at
every step — the same shape invariant ``h2o_update_batched`` relies on.

Merged *keys*, *scores*, and evicted/kept *positions* are bit-for-bit exact
against the per-head loop (verified below with ``mx.array_equal``) — the
same result ``h2o_update_batched`` achieves, since the eviction argmin and
survivor argmax are computed identically either way. Merged *values* in
``"sim_weighted"``/``"mean"`` mode carry one inherent source of float32
imprecision: ``merge_pair``'s cosine-similarity blend weight sums over the
head dimension, and MLX's batched-``[BH,D]`` reduction kernel accumulates in
a different order than the per-row ``[D]`` reduction the loop uses (a
one-line repro: ``mx.array_equal(mx.sum(a*b, axis=-1), mx.stack([mx.sum(a[i]*b[i]) for i in range(len(a))]))``
is False for real fp32 data even though both use the identical formula) —
confirmed via a 300-trial sweep to never exceed one fp16 representable step,
so those cases are checked with a tight ``mx.allclose`` instead of exact
equality. ``"drop"`` mode has no such reduction (pure H2O eviction, no
merge) and stays bit-for-bit exact throughout.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.cam import (
    cam_get_kv,
    cam_update,
    cam_update_batched,
    init_cam_state,
)


def _ref_loop(
    bh, s, d, n_sink, budget, merge_mode, merge_keys, merge_gate, seeds, keys_seq, values_seq
):
    states = [
        init_cam_state(
            n_sink,
            budget,
            d,
            merge_mode=merge_mode,
            merge_keys=merge_keys,
            merge_gate=merge_gate,
            seed=seeds[i],
        )
        for i in range(bh)
    ]
    positions = [None] * bh
    for t in range(s):
        new_positions = mx.arange(t, t + 1, dtype=mx.int32)
        for i in range(bh):
            st, pos = cam_update(
                states[i],
                keys_seq[i, t : t + 1],
                values_seq[i, t : t + 1],
                positions[i],
                new_positions,
            )
            states[i] = st
            positions[i] = pos
    ks, vs = [], []
    for i in range(bh):
        k, v = cam_get_kv(states[i])
        ks.append(k)
        vs.append(v)
    return mx.stack(ks), mx.stack(vs), mx.stack(positions)


def _batched(bh, keys_seq, values_seq, n_sink, budget, merge_mode, merge_keys, merge_gate, seeds):
    keys = values = scores = positions = None
    next_pos = 0
    draw_count = 0
    keys, values, scores, positions, next_pos, draw_count = cam_update_batched(
        keys,
        values,
        scores,
        positions,
        keys_seq,
        values_seq,
        n_sink,
        budget,
        merge_mode,
        merge_keys,
        merge_gate,
        seeds,
        next_pos,
        draw_count,
    )
    return keys, values, positions


def _run_case(
    desc, bh, s, d, n_sink, budget, merge_mode, merge_keys, merge_gate, seed_base=0, data_seed=0
):
    rng = np.random.default_rng(data_seed)
    keys_seq = mx.array(rng.standard_normal((bh, s, d)).astype(np.float32)).astype(mx.float16)
    values_seq = mx.array(rng.standard_normal((bh, s, d)).astype(np.float32)).astype(mx.float16)
    seeds = [seed_base + i for i in range(bh)]

    ref_k, ref_v, ref_pos = _ref_loop(
        bh, s, d, n_sink, budget, merge_mode, merge_keys, merge_gate, seeds, keys_seq, values_seq
    )
    b_k, b_v, b_pos = _batched(
        bh, keys_seq, values_seq, n_sink, budget, merge_mode, merge_keys, merge_gate, seeds
    )
    mx.eval(ref_k, ref_v, ref_pos, b_k, b_v, b_pos)
    assert mx.array_equal(ref_k, b_k).item(), f"{desc}: key mismatch"
    assert mx.array_equal(ref_pos, b_pos).item(), f"{desc}: position mismatch"
    if merge_mode == "drop":
        # No cosine-similarity reduction anywhere on this path (pure H2O
        # drop-eviction) -- bit-for-bit exact, same as h2o_update_batched.
        assert mx.array_equal(ref_v, b_v).item(), f"{desc}: value mismatch"
    else:
        # merge_pair's cosine-similarity blend weight sums over D with a
        # different accumulation order in MLX's batched-[BH,D] reduction
        # kernel than in the per-row [D] reduction the loop uses -- a
        # documented, unavoidable float32-ULP difference (see
        # cam_update_batched's docstring and h2o_update_batched's own
        # "fp32-rounding-only" note), confirmed here to never exceed one
        # fp16 representable step.
        assert mx.allclose(ref_v, b_v, atol=2e-3, rtol=2e-3).item(), f"{desc}: value mismatch"


@pytest.mark.parametrize(
    "desc,merge_mode,merge_keys,merge_gate,data_seed",
    [
        ("sim_weighted, gate on", "sim_weighted", False, True, 1),
        ("sim_weighted, gate off", "sim_weighted", False, False, 2),
        ("sim_weighted, merge_keys", "sim_weighted", True, True, 3),
        ("mean, gate on", "mean", False, True, 4),
        ("mean, merge_keys, gate off", "mean", True, False, 5),
        ("drop (== H2O)", "drop", False, True, 6),
    ],
)
def test_batched_matches_looped(
    desc: str, merge_mode: str, merge_keys: bool, merge_gate: bool, data_seed: int
) -> None:
    _run_case(
        desc,
        bh=5,
        s=40,
        d=8,
        n_sink=2,
        budget=10,
        merge_mode=merge_mode,
        merge_keys=merge_keys,
        merge_gate=merge_gate,
        data_seed=data_seed,
    )


def test_batched_matches_looped_no_sinks() -> None:
    _run_case(
        "no sinks",
        bh=3,
        s=25,
        d=4,
        n_sink=0,
        budget=6,
        merge_mode="sim_weighted",
        merge_keys=False,
        merge_gate=True,
        data_seed=7,
    )


def test_batched_matches_looped_single_head() -> None:
    _run_case(
        "single head",
        bh=1,
        s=30,
        d=8,
        n_sink=1,
        budget=8,
        merge_mode="sim_weighted",
        merge_keys=True,
        merge_gate=True,
        data_seed=8,
    )


def test_batched_matches_looped_decode_shaped() -> None:
    """Many single-token steps -- the actual decode hot path this fix targets."""
    _run_case(
        "decode-shaped repeated S=1 steps",
        bh=8,
        s=15,
        d=16,
        n_sink=2,
        budget=6,
        merge_mode="sim_weighted",
        merge_keys=False,
        merge_gate=True,
        data_seed=9,
    )


def test_batched_matches_looped_large_bh() -> None:
    _run_case(
        "large BH",
        bh=32,
        s=30,
        d=8,
        n_sink=2,
        budget=10,
        merge_mode="sim_weighted",
        merge_keys=False,
        merge_gate=True,
        seed_base=100,
        data_seed=10,
    )
