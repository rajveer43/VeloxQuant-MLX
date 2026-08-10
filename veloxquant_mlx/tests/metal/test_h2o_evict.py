"""Parity tests for the fused H2O eviction Metal kernel (h2o_fused_evict).

The kernel must reproduce h2o_update's per-token eviction branch
(veloxquant_mlx/quantizers/h2o.py) bit-for-bit: sink-protected argmin over
the "mid" state (n_kept stored rows + 1 appended row), evict the winner,
and re-rotate exactly the rows whose position shifted (rows before the
eviction gap are untouched — bit-identical, not just numerically close).
See paper/research/H2O_METAL_KERNEL_TECH_SPEC.md for the full design and
the T1-T8 test plan this file implements (T1-T7; T8, the real-model
regression, lives outside the unit test suite — see the PR/issue writeup).
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import h2o_fused_evict
from veloxquant_mlx.quantizers.a2ats_rope import a2ats_apply_exact_rope, rope_remap_positions

pytestmark = pytest.mark.skipif(
    not metal_available(),
    reason="Metal compute kernels not available on this build of mlx.",
)


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
    rotated_keys = a2ats_apply_exact_rope(raw_keys, positions, base=10000.0)
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


def test_interior_eviction_recovers_exact_original_keys():
    """Force eviction of an INTERIOR row (index 2 of 5), the rarer but real
    case (e.g. with n_sink > 0). Every surviving key, de-rotated at its new
    position, must recover its exact original pre-rotation value."""
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

    assert po.tolist() == [[0, 1, 2, 3]]
    kept_fp = np.array(vo.astype(mx.float32))[0, :, 0]
    assert 3.0 not in kept_fp  # token originally at index 2 (fp=3.0) is gone

    recovered = rope_remap_positions(
        ko[0].astype(mx.float32), po[0], mx.zeros_like(po[0]), base=10000.0
    )
    for row, fp in enumerate(kept_fp):
        orig_idx = int(round(fp)) - 1
        err = float(mx.max(mx.abs(recovered[row] - raw_keys[orig_idx])).item())
        assert err < 1e-2, f"row {row} (orig token {orig_idx}): recon error {err}"


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
    assert po.tolist() == [[0, 1, 2, 3]]
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
    # Evict index 3 (interior, not the first or last row) so rows 0,1,2
    # (before the gap) must be untouched and rows mapping from index 4
    # (after the gap) must shift + rotate.
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

    for row in range(3):  # positions 0, 1, 2 — all before the evicted position 3
        diff = float(
            mx.max(
                mx.abs(ko[0, row].astype(mx.float32) - rotated_keys[row].astype(mx.float32))
            ).item()
        )
        assert diff == 0.0, f"row {row} should be bit-identical, got diff={diff}"


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
    for a, b in zip(out1, out2):
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
        print(f"| path | ms/call |")
        print(f"|------|---------|")
        print(f"| Python loop (h2o_update) | {t_mlx:.4f} |")
        print(f"| fused Metal kernel       | {t_kernel:.4f} |")
        print(f"| speedup | {t_mlx / t_kernel:.2f}x |")
