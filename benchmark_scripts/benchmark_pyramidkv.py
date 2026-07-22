"""Offline-synthetic benchmark for PyramidKV-adapted KV cache.

Sweeps (n_layers, seq_len, avg_budget, beta) on synthetic fp16 K/V data,
building a full per-layer pyramid via KVCacheBuilder.for_model against a mock
model, then feeding the same sequence to every layer cache. Reports the budget
schedule, per-layer kept tokens, and compression ratio. No real model required.

Usage
-----
    python benchmark_scripts/benchmark_pyramidkv.py

Results print a table and save a JSON summary.
"""
from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig
from veloxquant_mlx.quantizers.pyramidkv import pyramid_budgets

# ── sweep configuration ──────────────────────────────────────────────────────
N_LAYERS_LIST = [12, 32]
SEQ_LENS      = [512, 1024]
AVG_BUDGETS   = [128, 256]
BETAS         = [1.0, 2.0, 3.0]
N_HEADS       = 8
HEAD_DIM      = 128
N_SINK        = 4


class _Attn:
    head_dim = HEAD_DIM


class _Layer:
    def __init__(self):
        self.self_attn = _Attn()


class _Model:
    def __init__(self, n):
        self.layers = [_Layer() for _ in range(n)]


def _rand_kv(S: int, H: int = N_HEADS, D: int = HEAD_DIM, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    V = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    mx.eval(K, V)
    return K, V


def _run_once(n_layers: int, seq_len: int, avg_budget: int, beta: float) -> dict:
    cfg = KVCacheConfig(
        method="pyramidkv",
        head_dim=HEAD_DIM,
        pyramid_budget=avg_budget,
        pyramid_n_sink=N_SINK,
        pyramid_beta=beta,
    )
    caches = KVCacheBuilder.for_model(_Model(n_layers), cfg)
    schedule = pyramid_budgets(n_layers, avg_budget, N_SINK, beta)

    K, V = _rand_kv(S=seq_len)

    t0 = time.perf_counter()
    kept = []
    for c in caches:
        ko, vo = c.update_and_fetch(K, V)
        mx.eval(ko, vo)
        kept.append(c.tokens_kept)
    latency_ms = (time.perf_counter() - t0) * 1_000

    total_kept = sum(kept)
    total_full = n_layers * seq_len
    ratio = total_full / total_kept if total_kept else 1.0

    return {
        "n_layers":        n_layers,
        "seq_len":         seq_len,
        "avg_budget":      avg_budget,
        "beta":            beta,
        "budget_first":    schedule[0],
        "budget_last":     schedule[-1],
        "budget_mean":     round(sum(schedule) / len(schedule), 1),
        "kept_first":      kept[0],
        "kept_last":       kept[-1],
        "compression_ratio": round(ratio, 3),
        "latency_ms_all_layers": round(latency_ms, 2),
    }


def main() -> None:
    print("PyramidKV-adapted KV Cache — offline synthetic benchmark")
    print(f"  n_heads={N_HEADS}  head_dim={HEAD_DIM}  n_sink={N_SINK}")
    print()
    header = (f"{'layers':>6}  {'seq':>5}  {'avg':>5}  {'beta':>4}  "
              f"{'b_first':>7}  {'b_last':>6}  {'kept_1st':>8}  {'kept_last':>9}  "
              f"{'ratio':>6}  {'ms':>7}")
    print(header)
    print("-" * len(header))

    results = []
    for n_layers, seq_len, avg_budget, beta in product(
        N_LAYERS_LIST, SEQ_LENS, AVG_BUDGETS, BETAS
    ):
        if avg_budget > seq_len:
            continue
        row = _run_once(n_layers, seq_len, avg_budget, beta)
        results.append(row)
        print(
            f"{row['n_layers']:>6}  {row['seq_len']:>5}  {row['avg_budget']:>5}  "
            f"{row['beta']:>4.1f}  {row['budget_first']:>7}  {row['budget_last']:>6}  "
            f"{row['kept_first']:>8}  {row['kept_last']:>9}  "
            f"{row['compression_ratio']:>5.2f}x  {row['latency_ms_all_layers']:>7.1f}"
        )

    out_path = Path(__file__).parent.parent / "figures" / "pyramidkv" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
