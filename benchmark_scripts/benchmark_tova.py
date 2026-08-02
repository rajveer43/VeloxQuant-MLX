"""Offline-synthetic benchmark for TOVA-adapted KV cache.

Sweeps (seq_len, tova_budget, tova_n_sink) on synthetic fp16 K/V data,
measuring update latency and compression ratio. No model required.

Usage
-----
    python benchmark_scripts/benchmark_tova.py

Results print a table and optionally save a JSON summary.
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

# ── sweep configuration ──────────────────────────────────────────────────────
SEQ_LENS = [128, 512, 1024, 2048]
BUDGETS = [64, 128, 256, 512]
N_SINKS = [0, 4]
N_HEADS = 8
HEAD_DIM = 128
N_WARMUP = 2
N_TIMED = 5


def _rand_kv(S: int, H: int = N_HEADS, D: int = HEAD_DIM, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    V = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    mx.eval(K, V)
    return K, V


def _run_once(seq_len: int, budget: int, n_sink: int) -> dict:
    cfg = KVCacheConfig(
        method="tova",
        head_dim=HEAD_DIM,
        tova_budget=budget,
        tova_n_sink=n_sink,
    )
    K, V = _rand_kv(S=seq_len)

    # Warmup
    for _ in range(N_WARMUP):
        cache = KVCacheFactory.create(cfg)
        ko, vo = cache.update_and_fetch(K, V)
        mx.eval(ko, vo)

    # Timed runs
    times = []
    for _ in range(N_TIMED):
        cache = KVCacheFactory.create(cfg)
        t0 = time.perf_counter()
        ko, vo = cache.update_and_fetch(K, V)
        mx.eval(ko, vo)
        times.append((time.perf_counter() - t0) * 1_000)

    ratio = cache.compression_ratio
    kept = cache.tokens_kept

    return {
        "seq_len": seq_len,
        "tova_budget": budget,
        "tova_n_sink": n_sink,
        "kept_tokens": kept,
        "compression_ratio": round(ratio, 3),
        "latency_ms_mean": round(float(np.mean(times)), 3),
        "latency_ms_min": round(float(np.min(times)), 3),
    }


def main() -> None:
    print("TOVA-adapted KV Cache — offline synthetic benchmark")
    print(f"  n_heads={N_HEADS}  head_dim={HEAD_DIM}  warmup={N_WARMUP}  timed={N_TIMED}")
    print()
    header = f"{'seq_len':>8}  {'budget':>8}  {'n_sink':>7}  {'kept':>6}  {'ratio':>7}  {'ms_mean':>9}  {'ms_min':>7}"
    print(header)
    print("-" * len(header))

    results = []
    for seq_len, budget, n_sink in product(SEQ_LENS, BUDGETS, N_SINKS):
        if budget > seq_len:
            continue  # trivial: no eviction possible
        row = _run_once(seq_len, budget, n_sink)
        results.append(row)
        print(
            f"{row['seq_len']:>8}  {row['tova_budget']:>8}  {row['tova_n_sink']:>7}  "
            f"{row['kept_tokens']:>6}  {row['compression_ratio']:>7.2f}x  "
            f"{row['latency_ms_mean']:>9.3f}  {row['latency_ms_min']:>7.3f}"
        )

    out_path = Path(__file__).parent.parent / "figures" / "tova" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
