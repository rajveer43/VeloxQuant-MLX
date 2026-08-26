"""Offline-synthetic benchmark for RocketKV-adapted two-stage compression.

RocketKV (Behnam, Fu, Zhao, Tsai, Yu, Tumanov; NVIDIA/Georgia Tech; ICML
2025, arXiv:2502.14051) claims that, AT A FIXED OVERALL TOKEN BUDGET, splitting
compression into a coarse eviction pass followed by a dynamic per-step
top-k selection pass (Hybrid Sparse Attention, HSA) over the survivors beats
reaching that same final budget via eviction alone — because dynamic top-k
prediction is far more accurate once run over a small, pre-filtered
candidate set than over the full sequence (paper §3.1's CDF analysis: the
union of per-step top-k indices actually needed, once eviction has already
run, is much smaller than the budget eviction alone would need to hit the
same accuracy).

This benchmark measures that claim directly and offline (no model loaded):
for a synthetic (query, K, V) triple and a target overall compression ratio
``c``, compare the true softmax-attention output of

  1. **SnapKV-only-at-c** — stage-1 eviction (this repo's existing
     SnapKVKVCache primitives) run DIRECTLY to the full ``c``-ratio budget
     (matching RocketKV's FINAL retained-token count), attention computed
     over that evicted set with no further narrowing, and
  2. **RocketKV (SnapKV + HSA)** — stage-1 eviction to the SMALLER
     ``stage1_ratio`` budget (paper §3.6's adaptive split of ``c``), THEN
     stage-2 HSA page selection narrows attention to a k2-page subset of
     those survivors, landing at the SAME final token count as arm 1,

against the TRUE softmax-attention output over the full, uncompressed
sequence. Both arms retain the same number of tokens at the end — the
comparison isolates whether *how* a fixed budget is reached (eviction alone
vs. eviction-then-dynamic-selection) changes accuracy, which is RocketKV's
actual claim.

Two data geometries:
  1. **uniform_relevance** — keys/values drawn i.i.d.; no structure favors
     either arm. Baseline reconstruction fidelity with no adversarial setup.
  2. **sparse_peaked** — a few key positions dominate the true query's
     attention mass (the common regime long-context tasks actually exercise:
     most tokens are near-irrelevant to any one query). HSA's per-step
     re-selection should track the true peak more accurately than SnapKV's
     one-shot prefill selection, especially as the query drifts.

Primary field: relative attention-output error ``||y - y_hat|| / ||y||``
against the TRUE (uncompressed, dense) softmax-attention output for a
held-out query — the metric RocketKV's own Figure 1 (qasper score vs token
budget) is a downstream proxy for.

**Explicitly NOT a model-level benchmark.** The paper's LongBench/NIAH/
RULER/SCBench numbers are the paper's own, measured on Llama-3.1-8B-Instruct,
Mistral-7B-Instruct-v0.2, and LongChat-7B-v1.5 on NVIDIA A100/H100 GPUs — not
reproduced here.

Usage
-----
    python benchmark_scripts/benchmark_rocketkv.py

Prints tables and saves a JSON summary to figures/rocketkv/results.json.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from veloxquant_mlx.quantizers.rocketkv import (
    build_paged_summary,
    hsa_approx_scores,
    select_topk_pages,
    split_compression_ratio,
    split_hsa_dims,
)
from veloxquant_mlx.quantizers.snapkv import snapkv_compress

HEAD_DIM = 32
SEQ_LENS = [128, 256]
COMPRESSION_RATIOS = [4.0, 8.0, 16.0]
OBS_WINDOW = 8
N_SINK = 2
DATA_SEEDS = [0, 1, 2, 3, 4]
SEED = 23


def _softmax_attn(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """Standard dense softmax attention: q [D], k/v [S, D] -> [D]."""
    D = q.shape[0]
    scale = 1.0 / math.sqrt(D)
    logits = (k @ q) * scale  # [S]
    weights = mx.softmax(logits, axis=-1)
    return weights @ v  # [D]


def _uniform_relevance_kv(S: int, D: int, seed: int):
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((S, D)).astype(np.float32)
    V = rng.standard_normal((S, D)).astype(np.float32)
    Q = rng.standard_normal((D,)).astype(np.float32)
    return mx.array(K), mx.array(V), mx.array(Q)


def _sparse_peaked_kv(S: int, D: int, seed: int, n_peaks: int = 3):
    """A handful of key positions strongly aligned with the query; the rest near-orthogonal noise."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((D,)).astype(np.float32)
    Q /= np.linalg.norm(Q) + 1e-8

    K = rng.standard_normal((S, D)).astype(np.float32) * 0.3
    V = rng.standard_normal((S, D)).astype(np.float32)
    peak_idx = rng.choice(S, size=n_peaks, replace=False)
    for i in peak_idx:
        K[i] = Q * 4.0 + rng.standard_normal(D).astype(np.float32) * 0.1
    return mx.array(K), mx.array(V), mx.array(Q)


def _run_once(seq_len: int, compression_ratio: float, geometry: str, seed: int) -> dict:
    D = HEAD_DIM
    if geometry == "uniform_relevance":
        K, V, Q = _uniform_relevance_kv(seq_len, D, seed)
    else:
        K, V, Q = _sparse_peaked_kv(seq_len, D, seed)

    y_true = _softmax_attn(Q, K, V)
    mx.eval(y_true)

    stage1_ratio, stage2_ratio = split_compression_ratio(compression_ratio)

    # --- RocketKV: evict to the SMALLER stage-1 sub-budget, then HSA-narrow ---
    stage1_budget = max(1, int(round(seq_len / stage1_ratio)))

    t0 = time.perf_counter()
    state = snapkv_compress(K, V, budget=stage1_budget, obs_window=OBS_WINDOW, n_sink=N_SINK)
    mx.eval(state.kept_keys)
    t1 = time.perf_counter()

    page_size, head_dim_ratio = split_hsa_dims(stage2_ratio)
    summary = build_paged_summary(state.kept_keys, page_size)
    n_pages = int(summary.page_max.shape[0])
    head_topk1 = max(1, int(round(D / head_dim_ratio)))
    k2 = max(1, int(round(n_pages / head_dim_ratio)))

    t2 = time.perf_counter()
    scores = hsa_approx_scores(Q, summary, head_topk1)
    pages = select_topk_pages(scores, k2)
    mx.eval(pages)
    t3 = time.perf_counter()

    selected_tokens: list[int] = []
    for p in pages.tolist():
        start = p * page_size
        end = min(start + page_size, state.n_kept)
        selected_tokens.extend(range(start, end))
    selected_tokens = sorted(set(selected_tokens))

    hsa_keys = state.kept_keys[mx.array(selected_tokens, dtype=mx.int32)].astype(mx.float32)
    hsa_values = state.kept_values[mx.array(selected_tokens, dtype=mx.int32)].astype(mx.float32)
    y_rocketkv = _softmax_attn(Q, hsa_keys, hsa_values)
    mx.eval(y_rocketkv)

    # --- SnapKV-only: evict DIRECTLY to the SAME final token count RocketKV
    # landed on, so both arms retain an equal number of tokens — the fair
    # comparison the paper's claim is actually about. ---
    final_budget = max(1, len(selected_tokens))
    state_direct = snapkv_compress(K, V, budget=final_budget, obs_window=OBS_WINDOW, n_sink=N_SINK)
    mx.eval(state_direct.kept_keys)
    y_snapkv = _softmax_attn(
        Q, state_direct.kept_keys.astype(mx.float32), state_direct.kept_values.astype(mx.float32)
    )
    mx.eval(y_snapkv)

    err_snapkv = float(mx.sqrt(mx.sum((y_snapkv - y_true) ** 2)).item()) / (
        float(mx.sqrt(mx.sum(y_true**2)).item()) + 1e-8
    )
    err_rocketkv = float(mx.sqrt(mx.sum((y_rocketkv - y_true) ** 2)).item()) / (
        float(mx.sqrt(mx.sum(y_true**2)).item()) + 1e-8
    )

    return {
        "seq_len": seq_len,
        "compression_ratio": compression_ratio,
        "geometry": geometry,
        "stage1_ratio": round(stage1_ratio, 3),
        "stage2_ratio": round(stage2_ratio, 3),
        "n_kept_stage1": state.n_kept,
        "n_kept_final": len(selected_tokens),
        "n_kept_snapkv_only": state_direct.n_kept,
        "rel_error_snapkv_only": round(err_snapkv, 4),
        "rel_error_rocketkv": round(err_rocketkv, 4),
        "ms_stage1": round((t1 - t0) * 1000, 4),
        "ms_stage2_hsa": round((t3 - t2) * 1000, 4),
    }


def main() -> None:
    print("RocketKV-adapted two-stage compression — offline synthetic benchmark")
    print("  (SnapKV eviction + Hybrid Sparse Attention; arXiv:2502.14051, ICML 2025)")
    print(
        f"  head_dim={HEAD_DIM}  obs_window={OBS_WINDOW}  n_sink={N_SINK}  data_seeds={DATA_SEEDS}"
    )
    print("  (rel_error = ||y_hat - y_exact|| / ||y_exact|| against the TRUE dense softmax output;")
    print("   lower = better; both arms retain the SAME final token count)")
    print()
    header = f"{'seq':>4} {'ratio':>6} {'geometry':>18}  {'snapkv_only':>12}  {'rocketkv':>9}  {'n_final':>8}"
    print(header)
    print("-" * len(header))

    results = []
    for seq_len in SEQ_LENS:
        for ratio in COMPRESSION_RATIOS:
            for geometry in ["uniform_relevance", "sparse_peaked"]:
                for data_seed in DATA_SEEDS:
                    row = _run_once(seq_len, ratio, geometry, seed=SEED + data_seed + seq_len)
                    results.append(row)
                print(
                    f"{seq_len:>4} {ratio:>6.1f} {geometry:>18}  "
                    f"{np.mean([r['rel_error_snapkv_only'] for r in results[-len(DATA_SEEDS) :]]):>12.4f}  "
                    f"{np.mean([r['rel_error_rocketkv'] for r in results[-len(DATA_SEEDS) :]]):>9.4f}  "
                    f"{results[-1]['n_kept_final']:>8}"
                )

    out_dir = Path(__file__).parent.parent / "figures" / "rocketkv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    for geom in ["uniform_relevance", "sparse_peaked"]:
        rows = [r for r in results if r["geometry"] == geom]
        snap_err = float(np.mean([r["rel_error_snapkv_only"] for r in rows]))
        rocket_err = float(np.mean([r["rel_error_rocketkv"] for r in rows]))
        print(
            f"\nSummary ({geom}): SnapKV-only mean rel-error {snap_err:.4f}  vs  RocketKV {rocket_err:.4f}"
        )

    print(
        "\n  (This is an offline-synthetic, cache-primitive-level benchmark — no model is\n"
        "   loaded and no LongBench/NIAH/RULER/SCBench numbers are reproduced; those are\n"
        "   the paper's own (Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2,\n"
        "   LongChat-7B-v1.5, NVIDIA A100/H100). Both arms retain the SAME final token\n"
        "   count — rocketkv reaches it via eviction-then-HSA-narrowing, snapkv_only via\n"
        "   eviction alone — isolating whether the two-stage path improves accuracy at a\n"
        "   fixed final budget, which is RocketKV's actual claim.)"
    )


if __name__ == "__main__":
    main()
