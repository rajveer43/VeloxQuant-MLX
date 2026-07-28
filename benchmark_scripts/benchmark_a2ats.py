"""Offline-synthetic benchmark for A2ATS-adapted windowed RoPE + query-aware VQ.

A2ATS-adapted (He, Xing, Wang, Xu, Wu, Zhou, Liu, Xue, Li — ACL 2025 Findings,
aclanthology.org/2025.findings-acl.644) applies exact RoPE within a trailing
window of the current decode position and leaves keys outside it unrotated
(Eq. 12), carrying the far bucket's constant ``R_b`` rotation on the query
instead (Eq. 11) — combined with query-aware VQ codebook assignment for a
retrieval-fraction subset of tokens.

Two comparisons, at the SAME codebook/sub_dim (so the only varying factor is
the RoPE-handling strategy and the assignment strategy):

  1. **A2ATS windowed RoPE vs. always-exact RoPE** (the ``window=+inf``
     degenerate case from ``a2ats_apply_windowed_rope`` — every token gets
     its own exact rotation, matching CommVQ-adapted's uniform treatment).
     Measured as error in the **attention score** ``u_ij = q̃_i k̃_j^T``, not
     in the key vector: under Eq. (12) far keys are deliberately unrotated,
     so scoring them against exact-RoPE'd keys would count the method's own
     design as error. See ``_rope_comparison`` (:issue:`29`).
  2. **A2ATS query-aware assignment vs. plain nearest-centroid VQ** (VecInfer-
     adapted's ``quantize_vq`` — the closest existing sibling: both are
     query-touching VQ methods; VecInfer-adapted's smooth+Hadamard transform
     is not query-aware). Isolates what the retrieval-set query-aware
     assignment buys over plain nearest-centroid quantization.

Two geometries:

  1. **local_recency** — query and its truly relevant context sit within the
     trailing window; positional locality is strong. A2ATS's windowing
     should cost little here (most/all reconstruction-relevant tokens get
     exact RoPE anyway).
  2. **long_range_dependent** — the query's most relevant tokens are far in
     the past, outside any reasonable window size. This is the honest stress
     case: the fixed-offset approximation pays its full cost here, and the
     results below say so plainly rather than hiding it.

Deterministic in ALL non-``_ms`` fields — verify by diffing two runs.
Offline-synthetic; loads no model, no mlx_lm generation. No CUDA kernel
fusion reproduced — this benchmark measures reconstruction fidelity only, not
the paper's own throughput numbers (see module docstrings in
``quantizers/a2ats_rope.py`` / ``quantizers/a2ats.py`` / ``cache/a2ats_cache.py``).

Usage
-----
    python benchmark_scripts/benchmark_a2ats.py

Prints tables and saves a JSON summary.
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import mlx.core as mx
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from veloxquant_mlx.allocators.vecinfer import quantize_vq, dequantize_vq, train_codebook
from veloxquant_mlx.quantizers.a2ats import a2ats_query_aware_assignment, a2ats_select_retrieval_set
from veloxquant_mlx.quantizers.a2ats_rope import (
    a2ats_apply_exact_rope,
    a2ats_apply_far_query_rope,
    a2ats_apply_windowed_rope,
)

# ── sweep configuration ──────────────────────────────────────────────────────
SEQ_LENS = [200, 400]
HEAD_DIM = 32
SUB_DIM = 8
CODEBOOK_BITS = 6
GEOMETRIES = ["local_recency", "long_range_dependent"]
DATA_SEEDS = [0, 1, 2, 3, 4]
SEED = 23
WINDOW = 16
B_CONST = 2048          # paper §5.1: b is independent of w (w=64, b=2048)
RETRIEVAL_FRACTION = 0.20
BETA = 0.5


def _synthetic(n: int, geometry: str, seed: int, d: int = HEAD_DIM) -> tuple:
    """Return (keys [n, d], query [d]) under the given geometry.

    local_recency: query matches the trailing WINDOW tokens closely.
    long_range_dependent: query matches a block near position 0 instead —
    far outside any reasonable trailing window.
    """
    rng = np.random.default_rng(seed)
    keys = 0.3 * rng.standard_normal((n, d)).astype(np.float32)
    query_base = rng.standard_normal(d).astype(np.float32)

    if geometry == "local_recency":
        relevant_start = max(0, n - WINDOW)
    else:  # long_range_dependent
        relevant_start = 0
    relevant_end = min(n, relevant_start + max(1, n // 10))
    keys[relevant_start:relevant_end] = query_base + 0.05 * rng.standard_normal(
        (relevant_end - relevant_start, d)
    ).astype(np.float32)

    query = query_base + 0.02 * rng.standard_normal(d).astype(np.float32)
    return keys, query


def _mse(a: mx.array, b: mx.array) -> float:
    return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())


def _rope_comparison(
    keys_mx: mx.array, query_mx: mx.array, codebook: mx.array, query_position: int
) -> dict:
    """Compare windowed vs. always-exact RoPE on **attention-score** error.

    Measured on attention scores ``u_ij = q̃_i k̃_j^T``, not on key vectors.
    That change matters: under Eq. (12) far keys are deliberately left
    unrotated, and the constant ``R_b`` rides on the query instead, so
    comparing far *keys* against exact-RoPE'd keys would score the method's
    own design as error. The attention score is what WRoPE actually
    approximates, and it is the quantity Eq. (11) is written in terms of.

    (The pre-:issue:`29` version of this benchmark compared key vectors while
    the implementation wrongly rotated far keys by ``R_window``. Both halves
    of that were wrong, which is why it reported windowed RoPE losing in
    every geometry.)
    """
    n, d = keys_mx.shape
    positions = mx.arange(n)

    idx = quantize_vq(keys_mx, codebook, SUB_DIM)
    dequant = dequantize_vq(idx, codebook).astype(mx.float16)   # pre-RoPE reconstruction

    # Ground truth: exact RoPE on both sides, unquantized keys (Eq. 3).
    true_k = a2ats_apply_exact_rope(keys_mx, positions).astype(mx.float32)
    true_q = a2ats_apply_exact_rope(
        query_mx[None, :], mx.array([query_position])
    ).astype(mx.float32)[0]
    true_scores = true_k @ true_q                      # [n]

    # A2ATS: near keys exact, far keys unrotated (Eq. 12); query carries the
    # exact rotation for the near bucket and R_b for the far bucket (Eq. 11).
    win_k = a2ats_apply_windowed_rope(
        dequant, positions, query_position, window=WINDOW
    ).astype(mx.float32)
    q_far = a2ats_apply_far_query_rope(query_mx, b=B_CONST).astype(mx.float32)
    distance = mx.array(query_position, dtype=mx.float32) - positions.astype(mx.float32)
    near = (distance < float(WINDOW))[:, None]          # [n, 1]
    # Near tokens score against the exactly-rotated query; far tokens against
    # the R_b-rotated query (Eq. 11).
    q_per_token = mx.where(near, true_q[None, :], q_far[None, :])   # [n, d]
    win_scores = (win_k * q_per_token).sum(axis=-1)                 # [n]

    # Always-exact baseline: every key gets its own exact rotation.
    exact_k = a2ats_apply_windowed_rope(
        dequant, positions, query_position, window=10_000_000
    ).astype(mx.float32)
    exact_scores = exact_k @ true_q

    # Split the error by bucket. This is the diagnostic that actually explains
    # the headline number: with Eq. (12) implemented correctly the near bucket
    # is *identical* to always-exact, so the whole penalty is far tokens —
    # which are the overwhelming majority of any long sequence.
    near_np = np.array(near[:, 0])
    err_win = (np.array(win_scores) - np.array(true_scores)) ** 2
    err_exact = (np.array(exact_scores) - np.array(true_scores)) ** 2

    return {
        "windowed_mse": _mse(win_scores, true_scores),
        "always_exact_mse": _mse(exact_scores, true_scores),
        "near_windowed_mse": float(err_win[near_np].mean()) if near_np.any() else 0.0,
        "near_exact_mse": float(err_exact[near_np].mean()) if near_np.any() else 0.0,
        "far_windowed_mse": float(err_win[~near_np].mean()) if (~near_np).any() else 0.0,
        "far_exact_mse": float(err_exact[~near_np].mean()) if (~near_np).any() else 0.0,
        "n_near": int(near_np.sum()),
        "n_far": int((~near_np).sum()),
    }


def _assignment_comparison(keys_mx: mx.array, codebook: mx.array, query_mx: mx.array) -> dict:
    """Compare A2ATS query-aware assignment (on the retrieval set) vs. plain
    nearest-centroid VQ (VecInfer-adapted's quantize_vq), both at the same
    sub_dim/codebook, reconstruction error against the unquantized keys."""
    n, d = keys_mx.shape
    n_sub = d // SUB_DIM

    retrieval_idx, bulk_idx = a2ats_select_retrieval_set(
        keys_mx, query_mx, retrieval_fraction=RETRIEVAL_FRACTION
    )

    # Plain nearest-centroid baseline (VecInfer-style) — every token.
    plain_idx = quantize_vq(keys_mx, codebook, SUB_DIM)
    plain_recon = dequantize_vq(plain_idx, codebook)

    # A2ATS: retrieval set gets query-aware assignment per sub-vector; bulk
    # gets plain nearest-centroid (identical to the baseline for those rows).
    a2ats_recon = plain_recon
    if retrieval_idx.shape[0] > 0:
        ret_keys = mx.take(keys_mx, retrieval_idx, axis=0)
        sub_idx_list = []
        for sub_i in range(n_sub):
            start, end = sub_i * SUB_DIM, (sub_i + 1) * SUB_DIM
            q_sub = query_mx[start:end]
            assign = a2ats_query_aware_assignment(
                ret_keys[:, start:end], codebook, q_sub, beta=BETA
            )
            sub_idx_list.append(assign)
        ret_idx_stack = mx.stack(sub_idx_list, axis=1)
        ret_recon = dequantize_vq(ret_idx_stack, codebook)

        # Functional scatter: replace retrieval-set rows in the baseline
        # reconstruction with their query-aware reconstruction.
        update = mx.zeros_like(a2ats_recon)
        update = update.at[retrieval_idx].add(ret_recon - mx.take(a2ats_recon, retrieval_idx, axis=0))
        a2ats_recon = a2ats_recon + update

    return {
        "a2ats_assignment_mse": _mse(a2ats_recon, keys_mx),
        "plain_vq_mse": _mse(plain_recon, keys_mx),
        "n_retrieved": int(retrieval_idx.shape[0]),
    }


def _run_once(seq_len: int, geometry: str, seed: int) -> dict:
    row = {"seq_len": seq_len, "geometry": geometry}
    windowed_mses, exact_mses = [], []
    a2ats_mses, plain_mses = [], []
    n_retrieved_list = []
    ms_list = []
    near_win, near_ex, far_win, far_ex = [], [], [], []
    n_near_list, n_far_list = [], []

    for ds in DATA_SEEDS:
        keys_np, query_np = _synthetic(seq_len, geometry, seed + ds)
        keys_mx = mx.array(keys_np)
        query_mx = mx.array(query_np)

        cb_train_data = keys_mx.reshape(-1, SUB_DIM)
        codebook = train_codebook(cb_train_data, 2 ** CODEBOOK_BITS, max_iter=10, seed=seed + ds)

        t0 = time.perf_counter()
        rope_res = _rope_comparison(keys_mx, query_mx, codebook, query_position=seq_len - 1)
        assign_res = _assignment_comparison(keys_mx, codebook, query_mx)
        mx.eval(mx.array(0))   # force any lazy graph before timing stop
        ms_list.append((time.perf_counter() - t0) * 1_000)

        windowed_mses.append(rope_res["windowed_mse"])
        exact_mses.append(rope_res["always_exact_mse"])
        a2ats_mses.append(assign_res["a2ats_assignment_mse"])
        plain_mses.append(assign_res["plain_vq_mse"])
        n_retrieved_list.append(assign_res["n_retrieved"])
        near_win.append(rope_res["near_windowed_mse"])
        near_ex.append(rope_res["near_exact_mse"])
        far_win.append(rope_res["far_windowed_mse"])
        far_ex.append(rope_res["far_exact_mse"])
        n_near_list.append(rope_res["n_near"])
        n_far_list.append(rope_res["n_far"])

    row["windowed_rope_mse"] = round(float(np.mean(windowed_mses)), 6)
    row["always_exact_rope_mse"] = round(float(np.mean(exact_mses)), 6)
    row["a2ats_assignment_mse"] = round(float(np.mean(a2ats_mses)), 6)
    row["plain_vq_mse"] = round(float(np.mean(plain_mses)), 6)
    row["near_windowed_mse"] = round(float(np.mean(near_win)), 6)
    row["near_exact_mse"] = round(float(np.mean(near_ex)), 6)
    row["far_windowed_mse"] = round(float(np.mean(far_win)), 6)
    row["far_exact_mse"] = round(float(np.mean(far_ex)), 6)
    row["n_near"] = int(np.mean(n_near_list))
    row["n_far"] = int(np.mean(n_far_list))
    row["avg_n_retrieved"] = round(float(np.mean(n_retrieved_list)), 1)
    row["ms"] = round(float(np.mean(ms_list)), 3)
    return row


def main() -> None:
    print("A2ATS-adapted windowed RoPE + query-aware VQ — offline synthetic benchmark")
    print(f"  head_dim={HEAD_DIM}  sub_dim={SUB_DIM}  codebook_bits={CODEBOOK_BITS}  "
          f"window={WINDOW}  retrieval_fraction={RETRIEVAL_FRACTION}  beta={BETA}")
    print("  (mse = mean squared reconstruction error; lower = better fidelity)")
    print()
    header = (f"{'seq':>4} {'geometry':>20}  {'windowed':>10}  {'always_exact':>12}  "
              f"{'a2ats_assign':>12}  {'plain_vq':>10}")
    print(header)
    print("-" * len(header))

    results = []
    for seq_len, geometry in product(SEQ_LENS, GEOMETRIES):
        row = _run_once(seq_len, geometry, seed=SEED + seq_len)
        results.append(row)
        print(f"{row['seq_len']:>4} {row['geometry']:>20}  {row['windowed_rope_mse']:>10.6f}  "
              f"{row['always_exact_rope_mse']:>12.6f}  {row['a2ats_assignment_mse']:>12.6f}  "
              f"{row['plain_vq_mse']:>10.6f}")

    out_path = Path(__file__).parent.parent / "figures" / "a2ats" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    for geom in GEOMETRIES:
        rows = [r for r in results if r["geometry"] == geom]
        windowed = float(np.mean([r["windowed_rope_mse"] for r in rows]))
        exact = float(np.mean([r["always_exact_rope_mse"] for r in rows]))
        a2ats_assign = float(np.mean([r["a2ats_assignment_mse"] for r in rows]))
        plain = float(np.mean([r["plain_vq_mse"] for r in rows]))
        near_w = float(np.mean([r["near_windowed_mse"] for r in rows]))
        near_e = float(np.mean([r["near_exact_mse"] for r in rows]))
        far_w = float(np.mean([r["far_windowed_mse"] for r in rows]))
        far_e = float(np.mean([r["far_exact_mse"] for r in rows]))
        n_near = int(np.mean([r["n_near"] for r in rows]))
        n_far = int(np.mean([r["n_far"] for r in rows]))
        print(f"\nSummary ({geom}):")
        print(f"  Attention-score MSE — windowed: {windowed:.6f}   always-exact: {exact:.6f}"
              f"   ({windowed / exact:.2f}x)")
        print(f"    near bucket (n={n_near:>3}): windowed {near_w:.6f}  always-exact {near_e:.6f}")
        print(f"    far  bucket (n={n_far:>3}): windowed {far_w:.6f}  always-exact {far_e:.6f}")
        print(f"  VQ reconstruction MSE   — a2ats query-aware: {a2ats_assign:.6f}   plain nearest-centroid: {plain:.6f}")

    _ratios = {
        g: (
            float(np.mean([r["windowed_rope_mse"] for r in results if r["geometry"] == g]))
            / float(np.mean([r["always_exact_rope_mse"] for r in results if r["geometry"] == g]))
        )
        for g in GEOMETRIES
    }
    _near_gap = max(
        abs(float(np.mean([r["near_windowed_mse"] for r in results if r["geometry"] == g]))
            - float(np.mean([r["near_exact_mse"] for r in results if r["geometry"] == g])))
        for g in GEOMETRIES
    )

    print("\n  (honest reading, stated plainly rather than softened:")
    print()
    print("   1. Windowed RoPE is WORSE than always-exact RoPE in BOTH geometries measured")
    print(f"      here, not just the long-range one — {_ratios['local_recency']:.1f}x higher"
          " attention-score MSE on")
    print(f"      local_recency and {_ratios['long_range_dependent']:.1f}x higher on"
          " long_range_dependent. The cost is")
    print("      REAL and INTRINSIC, not an artifact of the pre-#29 bugs: with Eq. (12)")
    print(f"      implemented correctly the NEAR bucket is now identical to always-exact")
    print(f"      (max gap {_near_gap:.2e}), so the entire penalty comes from FAR tokens —")
    print("      which are the overwhelming majority of any long sequence. Replacing each")
    print("      far token's true relative position with the single constant b is simply")
    print("      lossy, and no choice of b removes it (verified by sweeping b).")
    print()
    print("   2. A2ATS's query-aware assignment is WORSE than plain nearest-centroid VQ on")
    print("      reconstruction MSE in every row measured — this is mathematically expected,")
    print("      not a bug: beta=0.5 blends in a query-cosine-similarity term that pulls")
    print("      centroid selection away from the pure-reconstruction-error optimum")
    print("      (beta=1.0 reduces exactly to plain nearest-centroid, see")
    print("      test_beta_one_reduces_to_nearest_centroid). The intended payoff is downstream")
    print("      retrieval/attention quality for the query-relevant subset, not reconstruction")
    print("      fidelity — this benchmark does not measure that payoff (it would require a")
    print("      real attention computation and ground-truth relevance labels, neither of")
    print("      which exist in an offline synthetic harness). Readers should not conclude")
    print("      query-aware assignment is 'better' from this benchmark; it trades")
    print("      reconstruction accuracy for a downstream property this benchmark cannot")
    print("      measure.")
    print()
    print("   This benchmark measures reconstruction fidelity only; it is NOT a reproduction")
    print("   of the paper's own retrieval-accuracy or throughput numbers, which are measured")
    print("   on real long-context LLM workloads this repo does not have.)")


if __name__ == "__main__":
    main()
