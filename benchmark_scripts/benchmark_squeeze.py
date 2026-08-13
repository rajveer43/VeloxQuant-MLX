"""Offline-synthetic benchmark for SqueezeAttention-adapted KV cache.

Sweeps (n_layers, seq_len, avg_budget, strength) on synthetic fp16 K/V data,
building a shared-coordinator per-layer cache set via KVCacheBuilder.for_model
against a mock model. Each layer is fed keys of *increasing concentration* with
depth (early layers broad, deep layers clustered) so the data-driven
reallocation is exercised end to end. Reports the measured concentration, the
resolved budget schedule, per-layer kept tokens, and compression ratio. No real
model required.

Usage
-----
    python benchmark_scripts/benchmark_squeeze.py

Results print a table and save a JSON summary. Note the wall-clock numbers are
dominated by the O(S^2) pure-Python eviction loop run across all layers — a
prefill-batch worst case, not a per-decode-step cost.
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig

# ── sweep configuration ──────────────────────────────────────────────────────
N_LAYERS_LIST = [12, 32]
SEQ_LENS = [256, 512]
AVG_BUDGETS = [64, 128]
STRENGTHS = [0.0, 0.5, 1.0]
N_HEADS = 8
HEAD_DIM = 128
N_SINK = 4


class _Attn:
    head_dim = HEAD_DIM


class _Layer:
    def __init__(self):
        self.self_attn = _Attn()


class _Model:
    def __init__(self, n):
        self.layers = [_Layer() for _ in range(n)]


def _concentration_kv(layer_idx: int, n_layers: int, S: int, seed: int = 0):
    """Keys that grow more concentrated (clustered) with layer depth.

    Early layers → near-random directions (broad, low concentration).
    Deep layers  → dominated by one direction (clustered, high concentration).
    """
    frac = layer_idx / max(n_layers - 1, 1)
    rng = np.random.default_rng(seed + layer_idx)
    base = mx.array(rng.standard_normal((1, N_HEADS, S, HEAD_DIM)).astype(np.float32))
    direction = mx.ones((1, N_HEADS, 1, HEAD_DIM))
    K = ((1 - frac) * base + frac * 3.0 * direction).astype(mx.float16)
    V = mx.array(rng.standard_normal((1, N_HEADS, S, HEAD_DIM)).astype(np.float16))
    mx.eval(K, V)
    return K, V


def _run_once(n_layers: int, seq_len: int, avg_budget: int, strength: float) -> dict:
    cfg = KVCacheConfig(
        method="squeeze",
        head_dim=HEAD_DIM,
        squeeze_budget=avg_budget,
        squeeze_n_sink=N_SINK,
        squeeze_strength=strength,
    )
    caches = KVCacheBuilder.for_model(_Model(n_layers), cfg)

    t0 = time.perf_counter()
    # Prefill: every layer reports concentration.
    for li, c in enumerate(caches):
        K, V = _concentration_kv(li, n_layers, seq_len)
        ko, vo = c.update_and_fetch(K, V)
        mx.eval(ko, vo)
    # One decode step so layers adopt the resolved budget and re-trim.
    for c in caches:
        rng = np.random.default_rng(777)
        K = mx.array(rng.standard_normal((1, N_HEADS, 1, HEAD_DIM)).astype(np.float16))
        V = mx.array(rng.standard_normal((1, N_HEADS, 1, HEAD_DIM)).astype(np.float16))
        ko, vo = c.update_and_fetch(K, V)
        mx.eval(ko, vo)
    latency_ms = (time.perf_counter() - t0) * 1_000

    budgets = [c.layer_budget for c in caches]
    concs = [c.concentration for c in caches]
    kept = [c.tokens_kept for c in caches]

    total_kept = sum(kept)
    total_full = n_layers * (seq_len + 1)
    ratio = total_full / total_kept if total_kept else 1.0

    return {
        "n_layers": n_layers,
        "seq_len": seq_len,
        "avg_budget": avg_budget,
        "strength": strength,
        "conc_first": round(concs[0], 3),
        "conc_last": round(concs[-1], 3),
        "budget_first": budgets[0],
        "budget_last": budgets[-1],
        "budget_mean": round(sum(budgets) / len(budgets), 1),
        "kept_first": kept[0],
        "kept_last": kept[-1],
        "compression_ratio": round(ratio, 3),
        "latency_ms_all_layers": round(latency_ms, 2),
    }


def main() -> None:
    print("SqueezeAttention-adapted KV Cache — offline synthetic benchmark")
    print(f"  n_heads={N_HEADS}  head_dim={HEAD_DIM}  n_sink={N_SINK}")
    print("  (keys grow more concentrated with layer depth)")
    print()
    header = (
        f"{'layers':>6}  {'seq':>5}  {'avg':>5}  {'str':>4}  "
        f"{'c_1st':>5}  {'c_last':>6}  {'b_first':>7}  {'b_last':>6}  "
        f"{'kept_1st':>8}  {'kept_last':>9}  {'ratio':>6}  {'ms':>8}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for n_layers, seq_len, avg_budget, strength in product(
        N_LAYERS_LIST, SEQ_LENS, AVG_BUDGETS, STRENGTHS
    ):
        if avg_budget > seq_len:
            continue
        row = _run_once(n_layers, seq_len, avg_budget, strength)
        results.append(row)
        print(
            f"{row['n_layers']:>6}  {row['seq_len']:>5}  {row['avg_budget']:>5}  "
            f"{row['strength']:>4.1f}  {row['conc_first']:>5.2f}  {row['conc_last']:>6.2f}  "
            f"{row['budget_first']:>7}  {row['budget_last']:>6}  "
            f"{row['kept_first']:>8}  {row['kept_last']:>9}  "
            f"{row['compression_ratio']:>5.2f}x  {row['latency_ms_all_layers']:>8.1f}"
        )

    out_path = Path(__file__).parent.parent / "figures" / "squeeze" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
