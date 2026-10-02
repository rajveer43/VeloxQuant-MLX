"""Parity tests for the fused H2O eviction Metal kernel (h2o_fused_evict).

The kernel must reproduce h2o_update's per-token eviction branch
(veloxquant_mlx/quantizers/h2o.py) bit-for-bit: sink-protected argmin over
the "mid" state (n_kept stored rows + 1 appended row), evict the winner, and
compact the rest verbatim — no re-rotation, no renumbering (#609: an earlier
version of both the kernel and this file expected survivors to be
renumbered to a gap-free layout and re-rotated to match; that behavior was
removed because it silently changed a survivor's true distance from any
future query). See paper/research/H2O_METAL_KERNEL_TECH_SPEC.md for the full
design and the T1-T8 test plan this file implements (T1-T7; T8, the
real-model regression, lives outside the unit test suite — see the
PR/issue writeup).
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import h2o_fused_evict
from veloxquant_mlx.quantizers.a2ats_rope import a2ats_apply_exact_rope

pytestmark = [
    pytest.mark.metal,
    pytest.mark.skipif(
        not metal_available(),
        reason="Metal compute kernels not available on this build of mlx.",
    ),
]


def _make_fingerprinted(n_total: int, D: int, seed: int = 0):
    """n_total tokens with unique keys and an exact-integer fingerprint in
    value[:, 0] = 1..n_total, so we can identify which original token
    survives after eviction without relying on approximate matching."""
    rng = np.random.default_rng(seed)
    raw_keys = mx.array(rng.standard_normal((n_total, D)).astype(np.float32))
    fingerprints = np.zeros((n_total, D), dtype=np.float32)
    for i in range(n_total):
        fingerprints[i, 0] = i + 1.0
    raw_values = mx.array(fingerprints)
    positions = mx.arange(n_total, dtype=mx.int32)
    # This suite checks the fp16 kernel's bit-exactness against a fp16
    # reference, so fix the rotation's own dtype here rather than relying on
    # a2ats_apply_exact_rope's (correct, since #622) input-dtype passthrough.
    rotated_keys = a2ats_apply_exact_rope(raw_keys, positions, base=10000.0).astype(mx.float16)
    return raw_keys, raw_values, rotated_keys, positions


# ---------------------------------------------------------------------------
# T1 — bit-for-bit vs. the reference eviction math (interior + newest evicted)
# ---------------------------------------------------------------------------


def test_evict_newest_token_when_it_is_the_minimum():
    """Newest-arrival eviction (the common case per h2o_update's early-token-
    freeze property — see module docstring in h2o.py): scores_mid's last row
    is the global minimum (0.0), so it is evicted and all other rows are
    untouched (no shift, no rotation)."""
    D = 8
    raw_keys, raw_values, rotated_keys, positions = _make_fingerprinted(5, D)
    scores_mid = mx.array([[5.0, 5.0, 0.001, 5.0, 0.0]], dtype=mx.float32)

    ko, vo, so, po = h2o_fused_evict(
        rotated_keys[None].astype(mx.float16),
        raw_values[None].astype(mx.float16),
        scores_mid,
        positions[None],
        n_sink=0,
        rope_base=10000.0,
    )
    mx.eval(ko, vo, so, po)

    assert po.tolist() == [[0, 1, 2, 3]]
    assert so.tolist() == [[5.0, 5.0, pytest.approx(0.001, abs=1e-4), 5.0]]
    # Evicted row (index 4, the newest arrival) leaves rows 0-3 untouched —
    # nothing after the eviction point, so all survivors are bit-identical.
    diff = float(
        mx.max(mx.abs(ko[0].astype(mx.float32) - rotated_keys[:4].astype(mx.float32))).item()
    )
    assert diff == 0.0


def test_interior_eviction_leaves_survivors_exactly_unrotated():
    """Force eviction of an INTERIOR row (index 2 of 5), the rarer but real
    case (e.g. with n_sink > 0). Every surviving key must come back
    bit-identical to how it arrived, and its position must stay its true
    original position — no renumbering to a gap-free range (#609)."""
    D = 8
    raw_keys, raw_values, rotated_keys, positions = _make_fingerprinted(5, D)
    scores_mid = mx.array([[5.0, 5.0, 0.001, 5.0, 5.0]], dtype=mx.float32)

    ko, vo, so, po = h2o_fused_evict(
        rotated_keys[None].astype(mx.float16),
        raw_values[None].astype(mx.float16),
        scores_mid,
        positions[None],
        n_sink=0,
        rope_base=10000.0,
    )
    mx.eval(ko, vo, so, po)

    # Position 2 (the evicted token) leaves a real gap — not renumbered.
    assert po.tolist() == [[0, 1, 3, 4]]
    kept_fp = np.array(vo.astype(mx.float32))[0, :, 0]
    assert 3.0 not in kept_fp  # token originally at index 2 (fp=3.0) is gone

    for row, fp in enumerate(kept_fp):
        orig_idx = int(round(fp)) - 1
        err = float(mx.max(mx.abs(ko[0, row].astype(mx.float32) - rotated_keys[orig_idx])).item())
        assert err < 1e-6, f"row {row} (orig token {orig_idx}): survivor key was altered, err={err}"


# ---------------------------------------------------------------------------
# T2 — output shape is always exactly n_kept rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_total", [2, 5, 9])
def test_output_shape_is_n_total_minus_one(n_total):
    D = 16
    _, raw_values, rotated_keys, positions = _make_fingerprinted(n_total, D)
    scores_mid = mx.arange(n_total, dtype=mx.float32)[None] + 0.1

    ko, vo, so, po = h2o_fused_evict(
        rotated_keys[None].astype(mx.float16),
        raw_values[None].astype(mx.float16),
        scores_mid,
        positions[None],
        n_sink=0,
        rope_base=10000.0,
    )
    mx.eval(ko, vo, so, po)
    assert ko.shape == (1, n_total - 1, D)
    assert vo.shape == (1, n_total - 1, D)
    assert so.shape == (1, n_total - 1)
    assert po.shape == (1, n_total - 1)


# ---------------------------------------------------------------------------
# T3 — sink invariant: protected rows can never be evicted
# ---------------------------------------------------------------------------


def test_sink_protection():
    """n_sink=2: even though index 0 holds the numeric minimum score, it is
    protected and the real (non-sink) minimum, index 2, is evicted instead."""
    D = 8
    keys_mid = mx.zeros((1, 5, D), dtype=mx.float16)
    values_mid = mx.zeros((1, 5, D), dtype=mx.float16)
    scores_mid = mx.array([[0.0001, 5.0, 0.5, 5.0, 5.0]], dtype=mx.float32)
    positions_mid = mx.array([[0, 1, 2, 3, 4]], dtype=mx.int32)

    _, _, so, po = h2o_fused_evict(
        keys_mid, values_mid, scores_mid, positions_mid, n_sink=2, rope_base=10000.0
    )
    mx.eval(so, po)
    # Position 2 (the evicted, non-sink, real-minimum row) leaves a real
    # gap — not renumbered (#609).
    assert po.tolist() == [[0, 1, 3, 4]]
    assert so[0, 0].item() == pytest.approx(0.0001, abs=1e-6)  # sink row survived


def test_n_sink_zero_allows_all_evictions():
    D = 8
    keys_mid = mx.zeros((1, 3, D), dtype=mx.float16)
    values_mid = mx.zeros((1, 3, D), dtype=mx.float16)
    scores_mid = mx.array([[0.0001, 5.0, 5.0]], dtype=mx.float32)
    positions_mid = mx.array([[0, 1, 2]], dtype=mx.int32)

    _, _, so, po = h2o_fused_evict(
        keys_mid, values_mid, scores_mid, positions_mid, n_sink=0, rope_base=10000.0
    )
    mx.eval(so, po)
    assert so.tolist() == [[5.0, 5.0]]  # the 0.0001 row (index 0) WAS evicted


# ---------------------------------------------------------------------------
# T4 — untouched rows (before the eviction gap) are bit-identical, not
# merely numerically close
# ---------------------------------------------------------------------------


def test_untouched_rows_are_exact_copies():
    D = 8
    _, raw_values, rotated_keys, positions = _make_fingerprinted(5, D)
    # Evict index 3 (interior, not the first or last row). Post-#609, EVERY
    # survivor is an exact copy — rows 0,1,2 (before the gap) and row 4
    # (after the gap) alike. Nothing is shifted or re-rotated anymore.
    scores_mid = mx.array([[5.0, 5.0, 5.0, 0.001, 5.0]], dtype=mx.float32)

    ko, _, _, po = h2o_fused_evict(
        rotated_keys[None].astype(mx.float16),
        raw_values[None].astype(mx.float16),
        scores_mid,
        positions[None],
        n_sink=0,
        rope_base=10000.0,
    )
    mx.eval(ko, po)

    assert po.tolist() == [[0, 1, 2, 4]]  # real gap at the evicted position 3
    orig_rows = [0, 1, 2, 4]
    for out_row, orig_row in enumerate(orig_rows):
        diff = float(
            mx.max(
                mx.abs(
                    ko[0, out_row].astype(mx.float32) - rotated_keys[orig_row].astype(mx.float32)
                )
            ).item()
        )
        assert diff == 0.0, (
            f"row {out_row} (orig {orig_row}) should be bit-identical, got diff={diff}"
        )


# ---------------------------------------------------------------------------
# T5 — determinism
# ---------------------------------------------------------------------------


def test_deterministic():
    D = 16
    _, raw_values, rotated_keys, positions = _make_fingerprinted(6, D)
    scores_mid = mx.array([[3.0, 1.0, 4.0, 0.001, 2.0, 5.0]], dtype=mx.float32)

    out1 = h2o_fused_evict(
        rotated_keys[None].astype(mx.float16),
        raw_values[None].astype(mx.float16),
        scores_mid,
        positions[None],
        n_sink=0,
        rope_base=10000.0,
    )
    out2 = h2o_fused_evict(
        rotated_keys[None].astype(mx.float16),
        raw_values[None].astype(mx.float16),
        scores_mid,
        positions[None],
        n_sink=0,
        rope_base=10000.0,
    )
    for a, b in zip(out1, out2, strict=True):
        mx.eval(a, b)
        assert float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()) == 0.0


# ---------------------------------------------------------------------------
# T6 — large n_total stress test (exercises the grid-stride reduction loop)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_total", [128, 1000, 2048])
def test_large_n_total_finds_correct_minimum(n_total):
    D = 16
    rng = np.random.default_rng(0)
    scores_np = rng.uniform(1.0, 100.0, size=(2, n_total)).astype(np.float32)
    plant_idx = [n_total // 3, n_total - 2]
    for g, idx in enumerate(plant_idx):
        scores_np[g, idx] = 1e-4
    scores_mid = mx.array(scores_np)
    keys_mid = mx.array(rng.standard_normal((2, n_total, D)).astype(np.float16))
    values_mid = mx.array(rng.standard_normal((2, n_total, D)).astype(np.float16))
    positions_mid = mx.array(np.tile(np.arange(n_total), (2, 1)).astype(np.int32))

    _, _, so, _ = h2o_fused_evict(
        keys_mid, values_mid, scores_mid, positions_mid, n_sink=0, rope_base=10000.0, nsg=4
    )
    mx.eval(so)
    for g in range(2):
        ref = int(mx.argmin(scores_mid[g]).item())
        assert ref == plant_idx[g]
        assert float(mx.min(so[g]).item()) > 1e-4  # planted minimum is gone


# ---------------------------------------------------------------------------
# T7 — tie-break behavior matches mx.argmin (lowest index wins)
# ---------------------------------------------------------------------------


def test_tie_break_matches_mx_argmin():
    D = 8
    keys_mid = mx.zeros((1, 5, D), dtype=mx.float16)
    values_mid = mx.zeros((1, 5, D), dtype=mx.float16)
    scores_tie = mx.array([[1.0, 0.5, 0.5, 1.0, 1.0]], dtype=mx.float32)
    positions_mid = mx.array([[0, 1, 2, 3, 4]], dtype=mx.int32)

    ref_evict = int(mx.argmin(scores_tie[0]).item())
    expected_scores = [scores_tie[0, i].item() for i in range(5) if i != ref_evict]

    _, _, so, _ = h2o_fused_evict(
        keys_mid, values_mid, scores_tie, positions_mid, n_sink=0, rope_base=10000.0
    )
    mx.eval(so)
    assert so.tolist()[0] == expected_scores


# ---------------------------------------------------------------------------
# Multi-group independence (BH > 1 groups handled independently)
# ---------------------------------------------------------------------------


def test_multiple_bh_groups_are_independent():
    D = 8
    keys_mid = mx.zeros((3, 5, D), dtype=mx.float16)
    values_mid = mx.zeros((3, 5, D), dtype=mx.float16)
    scores_mid = mx.array(
        [
            [5.0, 0.1, 5.0, 5.0, 5.0],
            [5.0, 5.0, 0.1, 5.0, 5.0],
            [0.1, 5.0, 5.0, 5.0, 5.0],
        ],
        dtype=mx.float32,
    )
    positions_mid = mx.array([[0, 1, 2, 3, 4]] * 3, dtype=mx.int32)

    _, _, so, _ = h2o_fused_evict(
        keys_mid, values_mid, scores_mid, positions_mid, n_sink=0, rope_base=10000.0
    )
    mx.eval(so)
    for g in range(3):
        assert 0.1 not in so[g].tolist()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_non_3d_keys():
    with pytest.raises(ValueError):
        h2o_fused_evict(
            mx.zeros((5, 8), dtype=mx.float16),
            mx.zeros((1, 5, 8), dtype=mx.float16),
            mx.zeros((1, 5), dtype=mx.float32),
            mx.zeros((1, 5), dtype=mx.int32),
            n_sink=0,
            rope_base=10000.0,
        )


def test_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        h2o_fused_evict(
            mx.zeros((1, 5, 7), dtype=mx.float16),
            mx.zeros((1, 5, 7), dtype=mx.float16),
            mx.zeros((1, 5), dtype=mx.float32),
            mx.zeros((1, 5), dtype=mx.int32),
            n_sink=0,
            rope_base=10000.0,
        )


def test_rejects_single_row_input():
    """n_total=1 implies n_kept=0 -- nothing to evict from."""
    with pytest.raises(ValueError):
        h2o_fused_evict(
            mx.zeros((1, 1, 8), dtype=mx.float16),
            mx.zeros((1, 1, 8), dtype=mx.float16),
            mx.zeros((1, 1), dtype=mx.float32),
            mx.zeros((1, 1), dtype=mx.int32),
            n_sink=0,
            rope_base=10000.0,
        )


# ---------------------------------------------------------------------------
# Benchmark (printed, not asserted) — kernel vs. the pure-MLX per-token
# eviction branch it replaces
# ---------------------------------------------------------------------------


def test_h2o_evict_benchmark(capsys):
    from veloxquant_mlx.quantizers.h2o import H2OState, h2o_update

    D = 128
    n_kept = 512

    def _timeit(fn, iters=50, warmup=10):
        for _ in range(warmup):
            mx.eval(fn())
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        mx.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    rng = np.random.default_rng(0)
    keys = mx.array(rng.standard_normal((n_kept, D)).astype(np.float16))
    values = mx.array(rng.standard_normal((n_kept, D)).astype(np.float16))
    scores = mx.array(rng.uniform(0.1, 5.0, size=(n_kept,)).astype(np.float32))
    positions = mx.arange(n_kept, dtype=mx.int32)

    def _mlx_path():
        st = H2OState(
            keys=keys,
            values=values,
            scores=scores,
            positions=positions,
            n_sink=4,
            budget=n_kept,
            rope_base=10000.0,
            next_pos=n_kept,
        )
        new_k = mx.array(rng.standard_normal((1, D)).astype(np.float16))
        new_v = mx.array(rng.standard_normal((1, D)).astype(np.float16))
        out = h2o_update(st, new_k, new_v)
        return out.keys

    keys_mid = mx.concatenate([keys, keys[:1]], axis=0)[None]
    values_mid = mx.concatenate([values, values[:1]], axis=0)[None]
    scores_mid = mx.concatenate([scores, mx.zeros((1,))], axis=0)[None]
    positions_mid = mx.concatenate([positions, mx.array([n_kept])], axis=0)[None].astype(mx.int32)

    def _kernel_path():
        ko, _, _, _ = h2o_fused_evict(
            keys_mid.astype(mx.float16),
            values_mid.astype(mx.float16),
            scores_mid,
            positions_mid,
            n_sink=4,
            rope_base=10000.0,
        )
        return ko

    t_mlx = _timeit(_mlx_path)
    t_kernel = _timeit(_kernel_path)
    with capsys.disabled():
        print(f"\n# H2O fused eviction  |  n_kept={n_kept} D={D}  |  MLX {mx.__version__}")
        print("| path | ms/call |")
        print("|------|---------|")
        print(f"| Python loop (h2o_update) | {t_mlx:.4f} |")
        print(f"| fused Metal kernel       | {t_kernel:.4f} |")
        print(f"| speedup | {t_mlx / t_kernel:.2f}x |")


def _no_candidate_inputs():
    mx.random.seed(0)
    k = mx.random.normal((2, 6, 8)).astype(mx.float16)
    v = mx.random.normal((2, 6, 8)).astype(mx.float16)
    pos = mx.array([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]], dtype=mx.int32)
    return k, v, pos


def test_h2o_all_protected_matches_mlx_and_stays_in_group() -> None:
    """With no evictable row the kernel must not use index -1 (#650)."""
    from veloxquant_mlx.quantizers import h2o

    k, v, pos = _no_candidate_inputs()
    s = mx.random.uniform(shape=(2, 6))
    _, _, _, po = h2o_fused_evict(k, v, s, pos, n_sink=2, rope_base=10000.0, grace=4)
    _, _, _, pr = h2o._evict_via_mlx_batched(k, v, s, pos, 2, 10000.0, 4)
    assert po.tolist() == pr.tolist()


def test_h2o_nan_scores_evict_an_unprotected_row() -> None:
    k, v, pos = _no_candidate_inputs()
    s = mx.full((2, 6), float("nan"))
    ko, _, _, po = h2o_fused_evict(k, v, s, pos, n_sink=1, rope_base=10000.0)
    # First eligible row (index 1) is evicted; the sink survives.
    assert po.tolist() == [[0, 2, 3, 4, 5], [10, 12, 13, 14, 15]]
    assert po[0].tolist()[0] == 0 and po[1].tolist()[0] == 10
    assert all(0 <= p <= 5 for p in po[0].tolist())
    assert all(10 <= p <= 15 for p in po[1].tolist())
    assert bool(mx.all(mx.isfinite(ko.astype(mx.float32))).item())


def test_batched_metal_matches_mlx_twin_within_one_fp16_ulp():
    """Documented contract (#652): values/scores/positions exact, keys within
    1 fp16 ULP — the kernel rotates in fp32, the MLX twin in fp16."""
    from veloxquant_mlx.quantizers import h2o

    BH, N, D = 3, 300, 128
    mx.random.seed(1)
    k = mx.random.normal((BH, N, D)).astype(mx.float16)
    v = mx.random.normal((BH, N, D)).astype(mx.float16)
    s = mx.random.uniform(shape=(BH, N))
    pos = (mx.arange(N)[None] + 1000).astype(mx.int32) * mx.ones((BH, 1), dtype=mx.int32)
    ref = h2o._evict_via_mlx_batched(k, v, s, pos, 0, 500000.0, 0)
    got = h2o._metal_evict_batched(k, v, s, pos, 0, 500000.0, 0)
    for a, b in zip(ref[1:], got[1:], strict=True):  # values, scores, positions
        assert mx.array_equal(a, b).item()
    a, b = ref[0].astype(mx.float32), got[0].astype(mx.float32)
    assert bool(mx.all(mx.abs(a - b) <= 2.0**-9 * mx.maximum(mx.abs(a), mx.abs(b)) + 1e-6).item())
