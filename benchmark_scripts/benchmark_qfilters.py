"""Offline-synthetic benchmark for Q-Filters-adapted projection eviction.

The paper's premise (arXiv:2503.02812, preprint) is an *empirical property of
trained attention heads*: the (Query, Key) distribution is anisotropic, so a
single per-head direction predicts attention. That claim cannot be validated
on synthetic data — and, crucially, THIS ADAPTATION ESTIMATES THE DIRECTION
FROM KEYS (the cache never sees queries), which recovers the dominant *axis*
but not which *end* is important. So this harness reports both honestly:

  1. **paper_like** — geometry constructed to *exhibit* the anisotropy:
     "important" tokens carry a large projection onto a dominant axis and
     align with the probe-query cluster; the rest are near-orthogonal noise.
     Here the correct-sign Q-Filter should clearly beat random eviction.
  2. **isotropic** — plain Gaussian keys, no dominant importance axis. Here
     the method should show ~no advantage over random. Reporting this control
     is the point: no fabricated advantage.

Because the key-SVD direction is sign-ambiguous, we run BOTH sign arms
(sign=+1, sign=-1) and also report ``pert_qfilters_best`` = the better of the
two (what the method achieves once the sign is chosen). The ``filter_cosine``
field measures how well the key-SVD direction recovered the planted axis —
the honest gauge of whether the key-derived estimator stands in for the
paper's query-derived one.

Arms at matched budget: sign=+1, sign=-1, best-of-sign, random eviction
(seeded), KNorm-adapted and H2O-adapted references. Metric: output
perturbation — mean (1 − cosine) of probe-query attention output vs the full
uncompressed cache — the same metric family as the CaM/ChunkKV/xKV/KNorm
benchmarks.

**Explicitly NOT a model-level perplexity/throughput benchmark.**

Usage
-----
    python benchmark_scripts/benchmark_qfilters.py

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
from veloxquant_mlx.quantizers.qfilters import estimate_filter_dir

# ── sweep configuration ──────────────────────────────────────────────────────
SEQ_LENS   = [256, 512, 1024]
BUDGETS    = [64, 128]
GEOMETRIES = ["paper_like", "isotropic"]
HEAD_DIM   = 64
N_SINK     = 4
N_PROBES   = 32
CALIB      = 64
SEED       = 7


def _synthetic(S: int, geometry: str, seed: int, budget: int = 0):
    """Keys/values [S, D] + probe queries [N_PROBES, D] + planted axis or None."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((S, HEAD_DIM)).astype(np.float32)

    if geometry == "isotropic":
        k = rng.standard_normal((S, HEAD_DIM)).astype(np.float32)
        q = rng.standard_normal((N_PROBES, HEAD_DIM)).astype(np.float32)
        return k, v, q, None

    # paper_like: an "important" subset carries a large projection onto a
    # dominant axis mu and aligns with the query cluster; the rest are
    # near-orthogonal noise. mu is thus the top singular direction, and high
    # |projection| == important == attended-to (the anisotropy the paper
    # reports in trained heads, constructed explicitly). Important set capped
    # at half the budget so the sweep tests the paper's operating regime.
    mu = rng.standard_normal(HEAD_DIM).astype(np.float32)
    mu /= np.linalg.norm(mu)
    n_imp = min(S // 4, budget // 2) if budget else S // 4
    imp = rng.choice(S, size=n_imp, replace=False)
    mask = np.zeros(S, dtype=bool)
    mask[imp] = True
    k = np.zeros((S, HEAD_DIM), dtype=np.float32)
    k[mask] = 4.0 * mu + 0.2 * rng.standard_normal((n_imp, HEAD_DIM))
    k[~mask] = 0.3 * rng.standard_normal((S - n_imp, HEAD_DIM))
    q = (mu + 0.1 * rng.standard_normal((N_PROBES, HEAD_DIM))).astype(np.float32)
    return k, v, q, mu


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
    k, v, q, mu = _synthetic(S, geometry, seed, budget=budget)
    qq = mx.array(q)
    full_k, full_v = mx.array(k), mx.array(v)

    def qf(sign):
        return _run_cache(
            dict(method="qfilters", qfilters_budget=budget,
                 qfilters_n_sink=N_SINK, qfilters_calib_tokens=CALIB,
                 qfilters_sign=sign), k, v)

    kp, vp, ms_qf = qf(1)
    kn, vn, _ = qf(-1)
    p_pos = _perturbation(qq, full_k, full_v, kp, vp)
    p_neg = _perturbation(qq, full_k, full_v, kn, vn)

    kkn, vkn, ms_knorm = _run_cache(
        dict(method="knorm", knorm_budget=budget, knorm_n_sink=N_SINK,
             knorm_keep="low"), k, v)
    kh2o, vh2o, ms_h2o = _run_cache(
        dict(method="h2o", h2o_budget=budget, h2o_n_sink=N_SINK), k, v)
    krnd, vrnd = _random_evict(k, v, budget, seed + 99)

    # filter_cosine: how well the key-SVD direction recovered the planted axis
    # (paper_like only; None for isotropic where there is no planted axis).
    filter_cos = None
    if mu is not None:
        d = np.array(estimate_filter_dir(mx.array(k[:CALIB])))
        filter_cos = round(abs(float(d @ mu)), 4)

    return {
        "seq_len":              S,
        "budget":               budget,
        "geometry":             geometry,
        "pert_qfilters_pos":    round(p_pos, 5),
        "pert_qfilters_neg":    round(p_neg, 5),
        "pert_qfilters_best":   round(min(p_pos, p_neg), 5),
        "pert_knorm":           round(_perturbation(qq, full_k, full_v, kkn, vkn), 5),
        "pert_h2o":             round(_perturbation(qq, full_k, full_v, kh2o, vh2o), 5),
        "pert_random":          round(_perturbation(qq, full_k, full_v, krnd, vrnd), 5),
        "filter_cosine":        filter_cos,
        "compression_ratio":    round(S / budget, 2),
        "qfilters_ms":          round(ms_qf, 2),
        "knorm_ms":             round(ms_knorm, 2),
        "h2o_ms":               round(ms_h2o, 2),
    }


def main() -> None:
    print("Q-Filters-adapted query-agnostic projection eviction — offline synthetic benchmark")
    print(f"  head_dim={HEAD_DIM}  n_sink={N_SINK}  calib={CALIB}  probes={N_PROBES}")
    print("  (perturbation = 1 - cosine of probe attention output vs full cache; lower = better)")
    print("  (filter_cosine = |cos(key-SVD dir, planted axis)|; higher = key estimator recovers the axis)")
    print()
    header = (f"{'seq':>5}  {'budget':>6}  {'geometry':>10}  {'qf_best':>8}  "
              f"{'qf_pos':>8}  {'qf_neg':>8}  {'knorm':>8}  {'h2o':>8}  "
              f"{'random':>8}  {'flt_cos':>7}  {'qf_ms':>7}")
    print(header)
    print("-" * len(header))

    results = []
    for S, budget, geometry in product(SEQ_LENS, BUDGETS, GEOMETRIES):
        row = _run_once(S, budget, geometry, seed=SEED + S)
        results.append(row)
        fc = f"{row['filter_cosine']:.4f}" if row["filter_cosine"] is not None else "   -  "
        print(
            f"{row['seq_len']:>5}  {row['budget']:>6}  {row['geometry']:>10}  "
            f"{row['pert_qfilters_best']:>8.5f}  {row['pert_qfilters_pos']:>8.5f}  "
            f"{row['pert_qfilters_neg']:>8.5f}  {row['pert_knorm']:>8.5f}  "
            f"{row['pert_h2o']:>8.5f}  {row['pert_random']:>8.5f}  {fc:>7}  "
            f"{row['qfilters_ms']:>7.1f}"
        )

    out_path = Path(__file__).parent.parent / "figures" / "qfilters" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    for geom in GEOMETRIES:
        rows = [r for r in results if r["geometry"] == geom]
        d_rand = np.mean([r["pert_random"] - r["pert_qfilters_best"] for r in rows])
        print(f"\nSummary ({geom}):")
        print(f"  best-sign Q-Filter perturbation advantage vs random: {d_rand:+.5f}")
        if geom == "paper_like":
            fc = np.mean([r["filter_cosine"] for r in rows])
            print(f"  mean filter_cosine (key-SVD recovers planted axis):  {fc:.4f}")
        else:
            print("  (any small residual here is the best-of-two-signs selection bonus,")
            print("   not a real importance signal — the raw single-sign arms hover at random)")

    print("\n  (honest reading: the key-derived Q-Filter recovers the dominant AXIS but")
    print("   NOT which end is important — that sign is exactly what a query would")
    print("   disambiguate and the cache never sees one. So the win shows only for the")
    print("   correct sign arm (pert_qfilters_best), and only under paper_like geometry;")
    print("   under isotropic there is no advantage. The anisotropy claim is the paper's,")
    print("   the query→key estimator substitution is ours — see filter_cosine.)")


if __name__ == "__main__":
    main()
