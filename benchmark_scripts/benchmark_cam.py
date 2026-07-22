"""Offline-synthetic benchmark for CaM-adapted KV cache (cache merging).

Sweeps (seq_len, budget, merge_mode) on synthetic fp16 K/V and, for each config,
compares the compressed cache's attention output against the *full* (uncompressed)
cache's output — the perturbation CaM is designed to reduce. The token-level H2O
baseline (== CaM in "drop" mode) is the reference: CaM's claim is that merging the
evicted token into a survivor perturbs the output *less* than dropping it.

Metric: ``perturbation`` = mean cosine distance (1 - cos) between the compressed
attention output and the full-cache output, over a set of random probe queries.
Lower is better. ``gain_vs_drop`` = drop's perturbation - this mode's perturbation
(positive = merging helped). No real model required.

Usage
-----
    python benchmark_scripts/benchmark_cam.py

Prints a table and saves a JSON summary. Wall-clock is dominated by the O(S^2)
pure-Python merge loop (a prefill worst case), not a per-decode-step cost.
"""
from __future__ import annotations

import json
import math
import time
from itertools import product
from pathlib import Path

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.cache.cam_cache import CaMKVCache

# ── sweep configuration ──────────────────────────────────────────────────────
SEQ_LENS    = [256, 512, 1024]
BUDGETS     = [64, 128]
MERGE_MODES = ["drop", "mean", "sim_weighted"]
N_HEADS     = 4
HEAD_DIM    = 64
N_SINK      = 4
N_PROBES    = 32


def _synthetic_kv(S: int, seed: int = 0):
    """One layer's K/V: a few salient contiguous spans in a broad background."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((1, N_HEADS, S, HEAD_DIM)).astype(np.float32)
    direction = rng.standard_normal((HEAD_DIM,)).astype(np.float32)
    direction /= np.linalg.norm(direction)
    for start in (S // 6, S // 2, (4 * S) // 5):
        span = slice(start, min(start + S // 12, S))
        base[:, :, span, :] += 4.0 * direction
    K = mx.array(base.astype(np.float16))
    V = mx.array(rng.standard_normal((1, N_HEADS, S, HEAD_DIM)).astype(np.float16))
    mx.eval(K, V)
    return K, V


def _attn_output(query, keys, values):
    """Softmax attention output of probe queries over a K/V set.

    query:  [P, D], keys/values: [n, D] → returns [P, D].
    """
    q = query.astype(mx.float32)
    k = keys.astype(mx.float32)
    v = values.astype(mx.float32)
    scale = 1.0 / math.sqrt(float(k.shape[-1]))
    logits = (q @ k.T) * scale               # [P, n]
    w = mx.softmax(logits, axis=-1)          # [P, n]
    return w @ v                             # [P, D]


def _perturbation(probe, full_k, full_v, comp_k, comp_v):
    """Mean cosine distance between compressed-cache and full-cache attention outputs."""
    ref = _attn_output(probe, full_k, full_v)     # [P, D]
    got = _attn_output(probe, comp_k, comp_v)     # [P, D]
    rn = ref / (mx.sqrt(mx.sum(ref * ref, axis=-1, keepdims=True)) + 1e-8)
    gn = got / (mx.sqrt(mx.sum(got * got, axis=-1, keepdims=True)) + 1e-8)
    cos = mx.sum(rn * gn, axis=-1)                # [P]
    return float(mx.mean(1.0 - cos).item())


def _run_once(seq_len, budget, merge_mode, drop_ref=None) -> dict:
    K, V = _synthetic_kv(seq_len, seed=budget)
    full_k, full_v = K[0, 0], V[0, 0]             # head 0 full cache
    rng = np.random.default_rng(seq_len + budget)
    probe = mx.array(rng.standard_normal((N_PROBES, HEAD_DIM)).astype(np.float16))

    cfg = KVCacheConfig(
        method="cam", head_dim=HEAD_DIM, cam_budget=budget, cam_n_sink=N_SINK,
        cam_merge=merge_mode,
    )
    cache = CaMKVCache(cfg)
    t0 = time.perf_counter()
    Ko, Vo = cache.update_and_fetch(K, V)
    mx.eval(Ko, Vo)
    latency_ms = (time.perf_counter() - t0) * 1_000

    comp_k, comp_v = cache._states[0].keys, cache._states[0].values
    pert = _perturbation(probe, full_k, full_v, comp_k, comp_v)
    gain = None if drop_ref is None else round(drop_ref - pert, 5)

    return {
        "seq_len":           seq_len,
        "budget":            budget,
        "merge_mode":        merge_mode,
        "tokens_kept":       cache.tokens_kept,
        "compression_ratio": round(cache.compression_ratio, 3),
        "perturbation":      round(pert, 5),
        "gain_vs_drop":      gain,
        "latency_ms":        round(latency_ms, 2),
    }


def main() -> None:
    print("CaM-adapted KV Cache — offline synthetic benchmark")
    print(f"  n_heads={N_HEADS}  head_dim={HEAD_DIM}  n_sink={N_SINK}  probes={N_PROBES}")
    print("  (perturbation = cosine distance of attn output vs full cache; lower better)")
    print("  (drop == H2O baseline; gain_vs_drop > 0 means merging helped)")
    print()
    header = (f"{'seq':>5}  {'budget':>6}  {'mode':>13}  {'kept':>5}  {'ratio':>6}  "
              f"{'perturb':>8}  {'gain_v_drop':>11}  {'ms':>7}")
    print(header)
    print("-" * len(header))

    results = []
    for seq_len, budget in product(SEQ_LENS, BUDGETS):
        if budget >= seq_len:
            continue
        # run drop first to get the reference perturbation for this shape
        drop_row = _run_once(seq_len, budget, "drop")
        drop_ref = drop_row["perturbation"]
        for merge_mode in MERGE_MODES:
            row = drop_row if merge_mode == "drop" else _run_once(
                seq_len, budget, merge_mode, drop_ref=drop_ref)
            if merge_mode == "drop":
                row = {**row, "gain_vs_drop": 0.0}
            results.append(row)
            print(
                f"{row['seq_len']:>5}  {row['budget']:>6}  {row['merge_mode']:>13}  "
                f"{row['tokens_kept']:>5}  {row['compression_ratio']:>5.2f}x  "
                f"{row['perturbation']:>8.5f}  {str(row['gain_vs_drop']):>11}  "
                f"{row['latency_ms']:>7.1f}"
            )

    out_path = Path(__file__).parent.parent / "figures" / "cam" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
