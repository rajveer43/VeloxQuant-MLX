"""Offline-synthetic benchmark for L2Norm-adapted intrinsic key-norm eviction.

The paper's claim (Devoto et al., EMNLP 2024, arXiv:2406.11430) is an
*empirical property of trained decoder LMs*: keys with low L2 norm tend to
receive high attention, so keeping the lowest-norm tokens preserves the
attention output. That correlation cannot be validated on synthetic data —
so this harness runs two regimes and reports both honestly:

  1. **paper_like** — geometry constructed to *exhibit* the paper's reported
     correlation (low-norm keys aligned with the probe-query cluster,
     high-norm keys anti-aligned). Here keep-low should clearly beat
     keep-high and random eviction: this validates the *machinery* given the
     paper's geometry, not the geometry itself.
  2. **isotropic** — plain Gaussian keys, where norm carries no importance
     signal. Here keep-low should show ~no advantage over random. Reporting
     this control is the point: no fabricated advantage.

Arms at matched budget: keep="low" (the method), keep="high" (inverted
ablation), random eviction (seeded), and H2O-adapted (the repo's
accumulating-score reference). Metric: output perturbation — mean
(1 − cosine) of probe-query attention output vs the full uncompressed cache —
the same metric family as the CaM/ChunkKV/xKV benchmarks.

**Explicitly NOT a model-level perplexity/throughput benchmark.**

Usage
-----
    python benchmark_scripts/benchmark_knorm.py

Prints tables and saves a JSON summary.
"""
from __future__ import annotations

import json
import math
import sys
import time
from itertools import product
from pathlib import Path

import mlx.core as mx
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

# ── sweep configuration ──────────────────────────────────────────────────────
SEQ_LENS   = [256, 512, 1024]
BUDGETS    = [64, 128]
GEOMETRIES = ["paper_like", "isotropic"]
HEAD_DIM   = 64
N_SINK     = 4
N_PROBES   = 32
SEED       = 7


def _synthetic(S: int, geometry: str, seed: int, budget: int = 0):
    """Keys/values [S, D] + probe queries [N_PROBES, D]."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((S, HEAD_DIM)).astype(np.float32)

    if geometry == "isotropic":
        k = rng.standard_normal((S, HEAD_DIM)).astype(np.float32)
        q = rng.standard_normal((N_PROBES, HEAD_DIM)).astype(np.float32)
        return k, v, q

    # paper_like: an "important" subset of tokens gets low norm + alignment
    # with the query cluster; the rest high norm + anti-alignment. This is
    # the correlation the paper reports in trained LMs, constructed
    # explicitly so the machinery can be exercised. The important set is
    # capped at half the budget so the sweep tests the paper's operating
    # regime (budget covers the important tokens) — when the important set
    # exceeds the budget, no selection rule can retain it and the comparison
    # degenerates for every arm.
    mu = rng.standard_normal(HEAD_DIM).astype(np.float32)
    mu /= np.linalg.norm(mu)
    n_imp = min(S // 4, budget // 2) if budget else S // 4
    imp = rng.choice(S, size=n_imp, replace=False)
    mask = np.zeros(S, dtype=bool)
    mask[imp] = True
    k = np.zeros((S, HEAD_DIM), dtype=np.float32)
    k[mask] = 0.5 * mu + 0.05 * rng.standard_normal((n_imp, HEAD_DIM))
    k[~mask] = 3.0 * (-mu + 0.3 * rng.standard_normal((S - n_imp, HEAD_DIM)))
    q = (mu + 0.1 * rng.standard_normal((N_PROBES, HEAD_DIM))).astype(np.float32)
    return k, v, q


def _attn_out(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    scale = 1.0 / math.sqrt(float(k.shape[-1]))
    w = mx.softmax((q @ k.T) * scale, axis=-1)
    return w @ v


def _perturbation(q, full_k, full_v, kept_k, kept_v) -> float:
    ref = _attn_out(q, full_k, full_v)
    got = _attn_out(q, kept_k, kept_v)
    rn = ref / (mx.sqrt(mx.sum(ref * ref, -1, keepdims=True)) + 1e-8)
    gn = got / (mx.sqrt(mx.sum(got * got, -1, keepdims=True)) + 1e-8)
    return float(mx.mean(1.0 - mx.sum(rn * gn, -1)).item())


def _run_cache(method_cfg: dict, k: np.ndarray, v: np.ndarray):
    cache = KVCacheFactory.create(KVCacheConfig(head_dim=HEAD_DIM, **method_cfg))
    kk = mx.array(k[None, None].astype(np.float16))
    vv = mx.array(v[None, None].astype(np.float16))
    t0 = time.perf_counter()
    ko, vo = cache.update_and_fetch(kk, vv)
    mx.eval(ko, vo)
    ms = (time.perf_counter() - t0) * 1_000
    return ko[0, 0].astype(mx.float32), vo[0, 0].astype(mx.float32), ms


def _random_evict(k: np.ndarray, v: np.ndarray, budget: int, seed: int):
    rng = np.random.default_rng(seed)
    S = k.shape[0]
    keep = np.sort(np.concatenate([
        np.arange(min(N_SINK, S)),
        N_SINK + rng.choice(S - N_SINK, size=budget - N_SINK, replace=False),
    ]))
    return mx.array(k[keep]), mx.array(v[keep])


def _run_once(S: int, budget: int, geometry: str, seed: int) -> dict:
    k, v, q = _synthetic(S, geometry, seed, budget=budget)
    qq = mx.array(q)
    full_k, full_v = mx.array(k), mx.array(v)

    klo, vlo, ms_lo = _run_cache(
        dict(method="knorm", knorm_budget=budget, knorm_n_sink=N_SINK,
             knorm_keep="low"), k, v)
    khi, vhi, _ = _run_cache(
        dict(method="knorm", knorm_budget=budget, knorm_n_sink=N_SINK,
             knorm_keep="high"), k, v)
    kh2o, vh2o, ms_h2o = _run_cache(
        dict(method="h2o", h2o_budget=budget, h2o_n_sink=N_SINK), k, v)
    krnd, vrnd = _random_evict(k, v, budget, seed + 99)

    return {
        "seq_len":            S,
        "budget":             budget,
        "geometry":           geometry,
        "pert_keep_low":      round(_perturbation(qq, full_k, full_v, klo, vlo), 5),
        "pert_keep_high":     round(_perturbation(qq, full_k, full_v, khi, vhi), 5),
        "pert_random":        round(_perturbation(qq, full_k, full_v, krnd, vrnd), 5),
        "pert_h2o":           round(_perturbation(qq, full_k, full_v, kh2o, vh2o), 5),
        "compression_ratio":  round(S / budget, 2),
        "knorm_ms":           round(ms_lo, 2),
        "h2o_ms":             round(ms_h2o, 2),
    }


def main() -> None:
    print("L2Norm-adapted intrinsic key-norm eviction — offline synthetic benchmark")
    print(f"  head_dim={HEAD_DIM}  n_sink={N_SINK}  probes={N_PROBES}")
    print("  (perturbation = 1 - cosine of probe attention output vs full cache; lower = better)")
    print()
    header = (f"{'seq':>5}  {'budget':>6}  {'geometry':>10}  {'keep_low':>9}  "
              f"{'keep_high':>9}  {'random':>8}  {'h2o':>8}  {'knorm_ms':>8}  {'h2o_ms':>8}")
    print(header)
    print("-" * len(header))

    results = []
    for S, budget, geometry in product(SEQ_LENS, BUDGETS, GEOMETRIES):
        row = _run_once(S, budget, geometry, seed=SEED + S)
        results.append(row)
        print(
            f"{row['seq_len']:>5}  {row['budget']:>6}  {row['geometry']:>10}  "
            f"{row['pert_keep_low']:>9.5f}  {row['pert_keep_high']:>9.5f}  "
            f"{row['pert_random']:>8.5f}  {row['pert_h2o']:>8.5f}  "
            f"{row['knorm_ms']:>8.1f}  {row['h2o_ms']:>8.1f}"
        )

    out_path = Path(__file__).parent.parent / "figures" / "knorm" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    for geom in GEOMETRIES:
        rows = [r for r in results if r["geometry"] == geom]
        d_rand = np.mean([r["pert_random"] - r["pert_keep_low"] for r in rows])
        d_high = np.mean([r["pert_keep_high"] - r["pert_keep_low"] for r in rows])
        print(f"\nSummary ({geom}):")
        print(f"  keep-low perturbation advantage vs random:    {d_rand:+.5f}")
        print(f"  keep-low perturbation advantage vs keep-high: {d_high:+.5f}")
    print("\n  (honest reading: the advantage exists only under paper_like geometry.")
    print("   Under the isotropic control the direction actually REVERSES — softmax")
    print("   favors high-norm keys on isotropic Gaussians, so keep-low underperforms")
    print("   even random eviction there. The method's value rests entirely on the")
    print("   low-norm ⇒ high-attention geometry the paper reports in trained LMs —")
    print("   that correlation is the paper's claim, not this benchmark's.)")


if __name__ == "__main__":
    main()
