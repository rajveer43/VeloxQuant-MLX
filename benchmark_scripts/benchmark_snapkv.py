"""Offline SnapKV-adapted benchmark — prefill eviction coverage and throughput.

SnapKV-adapted (arXiv:2404.14469-adapted, Yuan et al.) retains only a budget
of token positions from prefill by scoring them via observation-window attention
(key-as-query proxy). This harness is **fully offline** — it loads no model and
allocates only small synthetic KV matrices — so it runs in a few hundred MB of RAM.

It measures:
  - kept bytes vs full fp16 bytes and eviction ratio (storage savings)
  - keep rate (fraction of tokens retained)
  - attention coverage: fraction of total attention mass in the kept set vs a
    random-budget baseline (how well the eviction targets the right tokens)
  - compress+select throughput (per-head, µs) across configs

Results are written to ``results_snapkv.json`` next to this script.

NOT YET RUN on hardware — no numbers are claimed in docs/CHANGELOG until this
is executed and its ``results_snapkv.json`` is committed.

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_snapkv.py
    PYTHONPATH=. python benchmark_scripts/benchmark_snapkv.py --seq 512 --heads 4 --dim 128
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from veloxquant_mlx.quantizers.snapkv import (
    full_fp16_bytes,
    obs_window_attention_scores,
    snap_select_indices,
    snapkv_compress,
    snapkv_fp16_bytes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_kv(S: int, D: int, seed: int = 0) -> tuple[mx.array, mx.array]:
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((S, D)).astype(np.float32))
    V = mx.array(rng.standard_normal((S, D)).astype(np.float32))
    return K, V


def _high_attention_outlier_kv(
    S: int, D: int, seed: int = 0, n_outliers: int = 10
) -> tuple[mx.array, mx.array]:
    """Synthetic data where a small cluster of tokens has very high self-attention mass."""
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((S, D)).astype(np.float32)
    V = rng.standard_normal((S, D)).astype(np.float32)
    idx = rng.choice(S, size=n_outliers, replace=False)
    K[idx] *= 6.0  # inflate norms → these tokens dominate self-attention
    return mx.array(K), mx.array(V)


def _attention_coverage(keys: mx.array, kept_indices: mx.array, obs_window: int) -> float:
    """Fraction of total obs-window attention mass falling on kept tokens."""
    scores = obs_window_attention_scores(keys, obs_window)
    total = float(mx.sum(scores).item())
    if total == 0:
        return 1.0
    kept_idx = [int(i) for i in kept_indices.tolist()]
    kept_mass = sum(float(scores[i].item()) for i in kept_idx)
    return kept_mass / total


def _random_coverage(keys: mx.array, budget: int, obs_window: int, seed: int = 42) -> float:
    """Attention coverage for a random-budget baseline (same keep count, random positions)."""
    S = keys.shape[0]
    scores = obs_window_attention_scores(keys, obs_window)
    total = float(mx.sum(scores).item())
    if total == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    rand_idx = rng.choice(S, size=min(budget, S), replace=False).tolist()
    rand_mass = sum(float(scores[i].item()) for i in rand_idx)
    return rand_mass / total


def _benchmark_one(
    keys: mx.array,
    values: mx.array,
    budget: int,
    obs_window: int,
    n_sink: int,
    n_rep: int = 10,
) -> dict:
    S, D = keys.shape

    # Warmup
    for _ in range(2):
        state = snapkv_compress(keys, values, budget, obs_window, n_sink)
        mx.eval(state.kept_keys)

    t0 = time.perf_counter()
    for _ in range(n_rep):
        state = snapkv_compress(keys, values, budget, obs_window, n_sink)
        mx.eval(state.kept_keys)
    t1 = time.perf_counter()

    kept = snapkv_fp16_bytes(state)
    full = full_fp16_bytes(S, D)
    cov = _attention_coverage(keys, state.kept_indices, obs_window)
    rand_cov = _random_coverage(keys, budget, obs_window)

    return {
        "seq_len": S,
        "head_dim": D,
        "budget": budget,
        "obs_window": obs_window,
        "n_sink": n_sink,
        "n_kept": state.n_kept,
        "keep_rate": round(state.n_kept / S, 4),
        "kept_bytes": kept,
        "full_bytes": full,
        "eviction_ratio": round(full / kept, 3) if kept > 0 else 1.0,
        "attention_coverage_snapkv": round(cov, 4),
        "attention_coverage_random": round(rand_cov, 4),
        "coverage_lift_vs_random": round(cov - rand_cov, 4),
        "ms_per_head": round((t1 - t0) / n_rep * 1000, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SnapKV-adapted offline benchmark")
    parser.add_argument("--seq", type=int, default=128, help="Base sequence length")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads to sweep")
    parser.add_argument("--dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--n_rep", type=int, default=10, help="Timing repetitions")
    args = parser.parse_args()

    seq_lens = [args.seq, args.seq * 4, args.seq * 16]
    budget_fractions = [0.25, 0.50, 0.75]
    obs_window = 32
    n_sink = 4
    D = args.dim

    results = []
    print(f"\nSnapKV-adapted offline benchmark  (NOT YET RUN on dedicated hardware)\n")
    print(
        f"{'S':>6}  {'budget':>8}  {'kept':>6}  {'ratio':>8}  "
        f"{'cov_snap':>10}  {'cov_rand':>10}  {'lift':>8}  {'ms/head':>9}"
    )
    print("-" * 85)

    for S in seq_lens:
        for bf in budget_fractions:
            budget = max(1, int(S * bf))
            n_out = max(1, int(S * 0.10))
            keys, values = _high_attention_outlier_kv(S, D, seed=42, n_outliers=n_out)
            r = _benchmark_one(keys, values, budget, obs_window, n_sink, args.n_rep)
            results.append(r)
            print(
                f"{S:>6}  {budget:>8}  {r['n_kept']:>6}  {r['eviction_ratio']:>8.3f}  "
                f"{r['attention_coverage_snapkv']:>10.4f}  "
                f"{r['attention_coverage_random']:>10.4f}  "
                f"{r['coverage_lift_vs_random']:>8.4f}  "
                f"{r['ms_per_head']:>9.4f}"
            )

    out_path = Path(__file__).parent / "results_snapkv.json"
    summary = {
        "note": (
            "NOT YET RUN on dedicated Apple Silicon hardware. "
            "Numbers above are from the development machine at benchmark-script "
            "execution time. Do not cite in paper until hardware results are committed."
        ),
        "hardware": platform.node(),
        "platform": platform.platform(),
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
