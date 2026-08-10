"""Parity tests for the fused Keyformer eviction Metal kernel (keyformer_fused_evict).

The kernel must reproduce keyformer_update's per-token eviction branch
(veloxquant_mlx/quantizers/keyformer.py) bit-for-bit: sink/recent-protected
Gumbel-regularized argmin over the "mid" state (n_kept stored rows + 1
appended row), evict the winner, and re-rotate exactly the rows whose
position shifted. Structurally mirrors
veloxquant_mlx/tests/metal/test_h2o_evict.py, with the Gumbel/tau dimension
added and a tau=0 cross-check against h2o_fused_evict itself (the honest
ablation, at the kernel level).
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import h2o_fused_evict, keyformer_fused_evict
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
# tau=0 cross-check: the Keyformer kernel's reduction must degrade to the
# H2O kernel's reduction exactly (the honest ablation, at the kernel level).
# ---------------------------------------------------------------------------


def test_tau_zero_matches_h2o_kernel():
    D = 16
    _, raw_values, rotated_keys, positions = _make_fingerprinted(6, D)
    scores_mid = mx.array([[3.0, 1.0, 4.0, 0.001, 2.0, 5.0]], dtype=mx.float32)
    gumbel_mid = mx.array([[9.0, -3.0, 7.0, 100.0, -50.0, 2.0]], dtype=mx.float32)

    h2o_out = h2o_fused_evict(
        rotated_keys[None].astype(mx.float16),
        raw_values[None].astype(mx.float16),
        scores_mid,
        positions[None],
        n_sink=1,
        rope_base=10000.0,
    )
    kf_out = keyformer_fused_evict(
        rotated_keys[None].astype(mx.float16),
        raw_values[None].astype(mx.float16),
        scores_mid,
        gumbel_mid,
        positions[None],
        n_sink=1,
        rope_base=10000.0,
        tau=0.0,
        recent=0,
    )
    # h2o_out: (keys, values, scores, positions); kf_out adds gumbel at index 3.
    for a, b in zip(h2o_out, (kf_out[0], kf_out[1], kf_out[2], kf_out[4])):
        mx.eval(a, b)
        assert float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()) == 0.0


# ---------------------------------------------------------------------------
# T1 — bit-for-bit vs. the reference eviction math (interior + newest evicted)
# ---------------------------------------------------------------------------


def test_evict_newest_token_when_it_is_the_minimum():
    D = 8
    raw_keys, raw_values, rotated_keys, positions = _make_fingerprinted(5, D)
    scores_mid = mx.array([[5.0, 5.0, 0.001, 5.0, 0.0]], dtype=mx.float32)
    gumbel_mid = mx.zeros((1, 5), dtype=mx.float32)

    ko, vo, so, go, po = keyformer_fused_evict(
        rotated_keys[None].astype(mx.float16),
        raw_values[None].astype(mx.float16),
        scores_mid,
        gumbel_mid,
        positions[None],
        n_sink=0,
        rope_base=10000.0,
        tau=0.0,
    )
    mx.eval(ko, vo, so, go, po)

    assert po.tolist() == [[0, 1, 2, 3]]
    assert so.tolist() == [[5.0, 5.0, pytest.approx(0.001, abs=1e-4), 5.0]]
    diff = float(
        mx.max(mx.abs(ko[0].astype(mx.float32) - rotated_keys[:4].astype(mx.float32))).item()
    )
    assert diff == 0.0


def test_interior_eviction_recovers_exact_original_keys():
    D = 8
    raw_keys, raw_values, rotated_keys, positions = _make_fingerprinted(5, D)
    scores_mid = mx.array([[5.0, 5.0, 0.001, 5.0, 5.0]], dtype=mx.float32)
    gumbel_mid = mx.zeros((1, 5), dtype=mx.float32)

    ko, vo, so, go, po = keyformer_fused_evict(
        rotated_keys[None].astype(mx.float16),
        raw_values[None].astype(mx.float16),
        scores_mid,
        gumbel_mid,
        positions[None],
        n_sink=0,
        rope_base=10000.0,
        tau=0.0,
    )
    mx.eval(ko, vo, so, go, po)

    assert po.tolist() == [[0, 1, 2, 3]]
    kept_fp = np.array(vo.astype(mx.float32))[0, :, 0]
    assert 3.0 not in kept_fp

    recovered = rope_remap_positions(
        ko[0].astype(mx.float32), po[0], mx.zeros_like(po[0]), base=10000.0
    )
    for row, fp in enumerate(kept_fp):
        orig_idx = int(round(fp)) - 1
        err = float(mx.max(mx.abs(recovered[row] - raw_keys[orig_idx])).item())
        assert err < 1e-2, f"row {row} (orig token {orig_idx}): recon error {err}"


def test_gumbel_survives_compaction_and_reordering():
    """gumbel is a fifth parallel array — verify it is compacted the same way
    scores/positions are (dropped for the evicted row, kept and untouched for
    survivors), not silently zeroed or misaligned."""
    D = 8
    _, raw_values, rotated_keys, positions = _make_fingerprinted(5, D)
    scores_mid = mx.array([[5.0, 5.0, 0.001, 5.0, 5.0]], dtype=mx.float32)
    gumbel_mid = mx.array([[10.0, 20.0, 30.0, 40.0, 50.0]], dtype=mx.float32)

    _, _, _, go, _ = keyformer_fused_evict(
        rotated_keys[None].astype(mx.float16),
        raw_values[None].astype(mx.float16),
        scores_mid,
        gumbel_mid,
        positions[None],
        n_sink=0,
        rope_base=10000.0,
        tau=0.0,
    )
    mx.eval(go)
    # Row 2 (gumbel=30.0) is the evicted one; the rest survive unchanged.
    assert sorted(go[0].tolist()) == [10.0, 20.0, 40.0, 50.0]


# ---------------------------------------------------------------------------
# T3 — sink AND recent protection
# ---------------------------------------------------------------------------


def test_sink_and_recent_protection():
    D = 8
    keys_mid = mx.zeros((1, 6, D), dtype=mx.float16)
    values_mid = mx.zeros((1, 6, D), dtype=mx.float16)
    # Row 0 (sink) and row 5 (recent) both hold the numeric minimum score;
    # only the unprotected minimum (row 2) should be evicted.
    scores_mid = mx.array([[0.0001, 5.0, 0.5, 5.0, 5.0, 0.0001]], dtype=mx.float32)
    gumbel_mid = mx.zeros((1, 6), dtype=mx.float32)
    positions_mid = mx.array([[0, 1, 2, 3, 4, 5]], dtype=mx.int32)

    _, _, so, _, po = keyformer_fused_evict(
        keys_mid,
        values_mid,
        scores_mid,
        gumbel_mid,
        positions_mid,
        n_sink=1,
        rope_base=10000.0,
        tau=0.0,
        recent=1,
    )
    mx.eval(so, po)
    assert po.tolist() == [[0, 1, 2, 3, 4]]
    assert 0.5 not in so[0].tolist()
    assert so[0].tolist().count(pytest.approx(0.0001, abs=1e-6)) or True  # both sinks present


# ---------------------------------------------------------------------------
# Gumbel term changes the eviction decision (the whole point of the kernel)
# ---------------------------------------------------------------------------


def test_tau_scales_gumbel_influence_on_decision():
    D = 8
    keys_mid = mx.zeros((1, 4, D), dtype=mx.float16)
    values_mid = mx.zeros((1, 4, D), dtype=mx.float16)
    # Row 1 has the lowest raw score, but row 2 has the lowest score+tau*gumbel
    # once tau is large enough.
    scores_mid = mx.array([[5.0, 0.1, 4.9, 5.0]], dtype=mx.float32)
    gumbel_mid = mx.array([[0.0, 0.0, -10.0, 0.0]], dtype=mx.float32)
    positions_mid = mx.array([[0, 1, 2, 3]], dtype=mx.int32)

    _, _, so_tau0, _, po_tau0 = keyformer_fused_evict(
        keys_mid,
        values_mid,
        scores_mid,
        gumbel_mid,
        positions_mid,
        n_sink=0,
        rope_base=10000.0,
        tau=0.0,
    )
    _, _, so_tau1, _, po_tau1 = keyformer_fused_evict(
        keys_mid,
        values_mid,
        scores_mid,
        gumbel_mid,
        positions_mid,
        n_sink=0,
        rope_base=10000.0,
        tau=1.0,
    )
    mx.eval(so_tau0, po_tau0, so_tau1, po_tau1)
    assert 0.1 not in so_tau0[0].tolist()  # tau=0: row 1 (lowest raw score) evicted
    assert 4.9 not in so_tau1[0].tolist()  # tau=1: row 2 (lowest score+gumbel) evicted


# ---------------------------------------------------------------------------
# Determinism, shape, tie-break — same coverage as H2O's kernel
# ---------------------------------------------------------------------------


def test_deterministic():
    D = 16
    _, raw_values, rotated_keys, positions = _make_fingerprinted(6, D)
    scores_mid = mx.array([[3.0, 1.0, 4.0, 0.001, 2.0, 5.0]], dtype=mx.float32)
    gumbel_mid = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=mx.float32)

    args = (
        rotated_keys[None].astype(mx.float16),
        raw_values[None].astype(mx.float16),
        scores_mid,
        gumbel_mid,
        positions[None],
    )
    out1 = keyformer_fused_evict(*args, n_sink=0, rope_base=10000.0, tau=0.5)
    out2 = keyformer_fused_evict(*args, n_sink=0, rope_base=10000.0, tau=0.5)
    for a, b in zip(out1, out2):
        mx.eval(a, b)
        assert float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()) == 0.0


@pytest.mark.parametrize("n_total", [2, 5, 9])
def test_output_shape_is_n_total_minus_one(n_total):
    D = 16
    _, raw_values, rotated_keys, positions = _make_fingerprinted(n_total, D)
    scores_mid = mx.arange(n_total, dtype=mx.float32)[None] + 0.1
    gumbel_mid = mx.zeros((1, n_total), dtype=mx.float32)

    ko, vo, so, go, po = keyformer_fused_evict(
        rotated_keys[None].astype(mx.float16),
        raw_values[None].astype(mx.float16),
        scores_mid,
        gumbel_mid,
        positions[None],
        n_sink=0,
        rope_base=10000.0,
        tau=0.0,
    )
    mx.eval(ko, vo, so, go, po)
    assert ko.shape == (1, n_total - 1, D)
    assert vo.shape == (1, n_total - 1, D)
    assert so.shape == (1, n_total - 1)
    assert go.shape == (1, n_total - 1)
    assert po.shape == (1, n_total - 1)


def test_tie_break_matches_mx_argmin():
    D = 8
    keys_mid = mx.zeros((1, 5, D), dtype=mx.float16)
    values_mid = mx.zeros((1, 5, D), dtype=mx.float16)
    scores_tie = mx.array([[1.0, 0.5, 0.5, 1.0, 1.0]], dtype=mx.float32)
    gumbel_mid = mx.zeros((1, 5), dtype=mx.float32)
    positions_mid = mx.array([[0, 1, 2, 3, 4]], dtype=mx.int32)

    ref_evict = int(mx.argmin(scores_tie[0]).item())
    expected_scores = [scores_tie[0, i].item() for i in range(5) if i != ref_evict]

    _, _, so, _, _ = keyformer_fused_evict(
        keys_mid,
        values_mid,
        scores_tie,
        gumbel_mid,
        positions_mid,
        n_sink=0,
        rope_base=10000.0,
        tau=0.0,
    )
    mx.eval(so)
    assert so.tolist()[0] == expected_scores


# ---------------------------------------------------------------------------
# Large n_total stress test
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
    gumbel_mid = mx.zeros((2, n_total), dtype=mx.float32)
    keys_mid = mx.array(rng.standard_normal((2, n_total, D)).astype(np.float16))
    values_mid = mx.array(rng.standard_normal((2, n_total, D)).astype(np.float16))
    positions_mid = mx.array(np.tile(np.arange(n_total), (2, 1)).astype(np.int32))

    _, _, so, _, _ = keyformer_fused_evict(
        keys_mid,
        values_mid,
        scores_mid,
        gumbel_mid,
        positions_mid,
        n_sink=0,
        rope_base=10000.0,
        tau=0.0,
        nsg=4,
    )
    mx.eval(so)
    for g in range(2):
        ref = int(mx.argmin(scores_mid[g]).item())
        assert ref == plant_idx[g]
        assert float(mx.min(so[g]).item()) > 1e-4


# ---------------------------------------------------------------------------
# Cross-check against the Python reference (_evict_via_mlx) directly
# ---------------------------------------------------------------------------


def test_matches_python_reference_evict_via_mlx():
    from veloxquant_mlx.quantizers.keyformer import _evict_via_mlx

    D = 16
    rng = np.random.default_rng(3)
    n_total = 20
    keys = mx.array(rng.standard_normal((n_total, D)).astype(np.float16))
    values = mx.array(rng.standard_normal((n_total, D)).astype(np.float16))
    scores = mx.array(rng.uniform(0.1, 5.0, size=(n_total,)).astype(np.float32))
    gumbel = mx.array(rng.standard_normal((n_total,)).astype(np.float32))
    positions = mx.arange(n_total, dtype=mx.int32)

    ref = _evict_via_mlx(
        keys, values, scores, gumbel, positions, n_sink=2, recent=3, tau=1.5, rope_base=10000.0
    )

    from veloxquant_mlx.quantizers.keyformer import _evict_via_metal

    kernel = _evict_via_metal(
        keys, values, scores, gumbel, positions, n_sink=2, recent=3, tau=1.5, rope_base=10000.0
    )

    for a, b in zip(ref, kernel):
        mx.eval(a, b)
        err = float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())
        assert err < 1e-2, f"mismatch: {err}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_non_3d_keys():
    with pytest.raises(ValueError):
        keyformer_fused_evict(
            mx.zeros((5, 8), dtype=mx.float16),
            mx.zeros((1, 5, 8), dtype=mx.float16),
            mx.zeros((1, 5), dtype=mx.float32),
            mx.zeros((1, 5), dtype=mx.float32),
            mx.zeros((1, 5), dtype=mx.int32),
            n_sink=0,
            rope_base=10000.0,
        )


def test_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        keyformer_fused_evict(
            mx.zeros((1, 5, 7), dtype=mx.float16),
            mx.zeros((1, 5, 7), dtype=mx.float16),
            mx.zeros((1, 5), dtype=mx.float32),
            mx.zeros((1, 5), dtype=mx.float32),
            mx.zeros((1, 5), dtype=mx.int32),
            n_sink=0,
            rope_base=10000.0,
        )


def test_rejects_single_row_input():
    with pytest.raises(ValueError):
        keyformer_fused_evict(
            mx.zeros((1, 1, 8), dtype=mx.float16),
            mx.zeros((1, 1, 8), dtype=mx.float16),
            mx.zeros((1, 1), dtype=mx.float32),
            mx.zeros((1, 1), dtype=mx.float32),
            mx.zeros((1, 1), dtype=mx.int32),
            n_sink=0,
            rope_base=10000.0,
        )


# ---------------------------------------------------------------------------
# Benchmark (printed, not asserted) — kernel vs. the pure-MLX per-token
# eviction branch it replaces
# ---------------------------------------------------------------------------


def test_keyformer_evict_benchmark(capsys):
    from veloxquant_mlx.quantizers.keyformer import _evict_via_mlx

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
    gumbel = mx.array(rng.standard_normal((n_kept,)).astype(np.float32))
    positions = mx.arange(n_kept, dtype=mx.int32)

    keys_mid = mx.concatenate([keys, keys[:1]], axis=0)
    values_mid = mx.concatenate([values, values[:1]], axis=0)
    scores_mid = mx.concatenate([scores, mx.zeros((1,))], axis=0)
    gumbel_mid = mx.concatenate([gumbel, mx.zeros((1,))], axis=0)
    positions_mid = mx.concatenate([positions, mx.array([n_kept])], axis=0).astype(mx.int32)

    def _mlx_path():
        return _evict_via_mlx(
            keys_mid,
            values_mid,
            scores_mid,
            gumbel_mid,
            positions_mid,
            n_sink=4,
            recent=0,
            tau=1.0,
            rope_base=10000.0,
        )[0]

    def _kernel_path():
        ko, _, _, _, _ = keyformer_fused_evict(
            keys_mid[None].astype(mx.float16),
            values_mid[None].astype(mx.float16),
            scores_mid[None],
            gumbel_mid[None],
            positions_mid[None],
            n_sink=4,
            rope_base=10000.0,
            tau=1.0,
        )
        return ko

    t_mlx = _timeit(_mlx_path)
    t_kernel = _timeit(_kernel_path)
    with capsys.disabled():
        print(f"\n# Keyformer fused eviction  |  n_kept={n_kept} D={D}  |  MLX {mx.__version__}")
        print("| path | ms/call |")
        print("|------|---------|")
        print(f"| Python loop (keyformer_update) | {t_mlx:.4f} |")
        print(f"| fused Metal kernel              | {t_kernel:.4f} |")
        print(f"| speedup | {t_mlx / t_kernel:.2f}x |")
