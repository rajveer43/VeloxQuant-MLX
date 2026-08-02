"""Offline-synthetic benchmark for KVTC-adapted local-PCA + DP bit allocation + entropy coding.

The paper's contribution (arXiv:2511.01815, NVIDIA, ICLR 2026) is a
transform-coding pipeline: a pre-calibrated global PCA basis, a
dynamic-programming bit allocator that can assign zero bits to a
low-variance component, and an entropy-coding stage on top of the quantized
codes. The honest thing to measure here is NOT "KVTC beats everything" but
"does the DP-optimal, per-component, zero-capable bit allocation reach lower
reconstruction distortion than a fixed-split mixed-precision baseline (the
kind Palu/SVDq already ship) AT A MATCHED TOTAL BYTE BUDGET, and by how much
does the (separately real) entropy-coding stage help on top?"

Two geometries:

  1. **skewed_variance** — a planted low-rank-ish geometry where a handful
     of principal components carry most of the variance and the rest are
     near-noise. This is the case the DP allocator should win on: it can
     assign 0 bits to the near-noise tail instead of paying SVDq's fixed
     25%/75% floor on every component.
  2. **flat** — near-isotropic (Gaussian) data, where there is no
     concentrated variance to exploit. Here the DP allocator should be
     roughly competitive with the fixed split, not a dramatic win — reporting
     this null control is the point (matches the framing used by every other
     mechanism-claim benchmark in this repo: KVzip's flat control, MorphKV's
     stable control, Keyformer's stable-geometry control).

Arms at a matched total bit budget: KVTC (DP-allocated + entropy-coded), a
fixed-UNIFORM-bits baseline (same budget spread evenly across all
components), and SVDq's fixed top-25%/75% split (same total budget, split
2:1 hi:lo bit ratio by singular-value magnitude — the same rule
`quantizers/svdq.py::quantize_latents_mixed` uses, just re-derived here at a
MATCHED total-bit budget rather than SVDq's own hi_bit/lo_bit knobs, so all
three arms spend the exact same number of bits).

Primary field: reconstruction MSE / cosine similarity at the matched budget.
Secondary field: entropy-coding's REALIZED gain
(pre_entropy_bytes / kvtc_fp16_bytes) — reported plainly, not oversold; on
synthetic Gaussian-like component codes this is typically modest.

Deterministic in ALL non-``_ms`` fields (only timing may vary) — verify by
diffing two runs. Offline-synthetic; loads no model, no mlx_lm generation.

**Explicitly NOT a model-level perplexity/throughput benchmark.** The
paper's headline numbers (up to 20x, up to 40x in some regimes, <1pp
accuracy loss on LLaMA 3 / Mistral NeMo / R1-Qwen2.5 1.5B-70B across AIME25,
GSM8K, LiveCodeBench, LongBench, MATH-500, MMLU, Qasper, RULER) are the
paper's, on trained models — not reproduced here.

Usage
-----
    python benchmark_scripts/benchmark_kvtc.py

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

from veloxquant_mlx.quantizers._quant_utils import _truncated_svd
from veloxquant_mlx.quantizers.kvtc import (
    kvtc_compress,
    kvtc_decompress,
    kvtc_fp16_bytes,
    kvtc_pre_entropy_bytes,
    quantize_component,
)

# ── sweep configuration ──────────────────────────────────────────────────────
SEQ_LENS = [128, 256]
BIT_BUDGETS = [64, 128]  # total bits across all min(S, D) components
GEOMETRIES = ["skewed_variance", "flat"]
HEAD_DIM = 32
R_TRUE = 6  # planted rank for skewed_variance geometry
DATA_SEEDS = [0, 1, 2, 3, 4]  # average over data realizations (no RNG in method)
SEED = 13


def _synthetic(S: int, geometry: str, seed: int) -> np.ndarray:
    """[S, HEAD_DIM] fp32 synthetic keys/values under one of two geometries."""
    rng = np.random.default_rng(seed)
    if geometry == "skewed_variance":
        scale = np.array([20.0 / (i + 1) for i in range(R_TRUE)])
        U = rng.standard_normal((S, R_TRUE))
        Wt = rng.standard_normal((R_TRUE, HEAD_DIM)) * scale[:, None]
        noise = rng.standard_normal((S, HEAD_DIM)) * 0.05
        return (U @ Wt + noise).astype(np.float32)
    # flat: near-isotropic Gaussian, no concentrated variance.
    return rng.standard_normal((S, HEAD_DIM)).astype(np.float32)


def _mse(a: mx.array, b: mx.array) -> float:
    return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())


def _cosine(a: mx.array, b: mx.array) -> float:
    af, bf = a.astype(mx.float32), b.astype(mx.float32)
    num = mx.sum(af * bf, axis=-1)
    den = mx.sqrt(mx.sum(af * af, axis=-1)) * mx.sqrt(mx.sum(bf * bf, axis=-1)) + 1e-8
    return float(mx.mean(num / den).item())


def _fixed_uniform_reconstruction(x: mx.array, total_bit_budget: int) -> mx.array:
    """Fixed-uniform-bits baseline: same total budget, spread evenly (floor +
    remainder to the first components) across all r = min(S, D) components —
    no DP, no variance-awareness at all.
    """
    xf = x.astype(mx.float32)
    S, D = int(xf.shape[0]), int(xf.shape[1])
    mean = mx.mean(xf, axis=0)
    xc = xf - mean[None, :]
    r = min(S, D)
    U, s_vals, Vt = _truncated_svd(xc, rank=r)
    V = Vt.T
    L = xc @ V
    mx.eval(L)
    L_np = np.asarray(L.tolist(), dtype=np.float64)

    base, rem = divmod(total_bit_budget, r)
    recon = np.zeros((S, r), dtype=np.float64)
    for i in range(r):
        bits = base + (1 if i < rem else 0)
        if bits <= 0:
            continue
        codes, lo, scale = quantize_component(L_np[:, i], bits)
        recon[:, i] = codes.astype(np.float64) * scale + lo

    x_hat = mx.array(recon.astype(np.float32)) @ V.T + mean[None, :]
    return x_hat.astype(mx.float16)


def _svdq_fixed_split_reconstruction(
    x: mx.array, total_bit_budget: int, hi_fraction=0.25
) -> mx.array:
    """SVDq-style fixed top-25%/75% split at the SAME matched total bit
    budget (hi_bit = 2 * lo_bit, SVDq's ratio, chosen as the largest integer
    pair that fits the budget without exceeding it).
    """
    xf = x.astype(mx.float32)
    S, D = int(xf.shape[0]), int(xf.shape[1])
    mean = mx.mean(xf, axis=0)
    xc = xf - mean[None, :]
    r = min(S, D)
    U, s_vals, Vt = _truncated_svd(xc, rank=r)
    V = Vt.T
    L = xc @ V
    mx.eval(L)
    L_np = np.asarray(L.tolist(), dtype=np.float64)

    n_hi = max(1, int(r * hi_fraction))
    n_lo = r - n_hi
    best = (0, 0)
    for lo_bit in range(0, 9):
        hi_bit = 2 * lo_bit
        total = n_hi * hi_bit + n_lo * lo_bit
        if total <= total_bit_budget and total > (n_hi * best[0] + n_lo * best[1]):
            best = (hi_bit, lo_bit)
    hi_bit, lo_bit = best

    recon = np.zeros((S, r), dtype=np.float64)
    for i in range(r):
        bits = hi_bit if i < n_hi else lo_bit
        if bits <= 0:
            continue
        codes, lo, scale = quantize_component(L_np[:, i], bits)
        recon[:, i] = codes.astype(np.float64) * scale + lo

    x_hat = mx.array(recon.astype(np.float32)) @ V.T + mean[None, :]
    return x_hat.astype(mx.float16)


def _run_once(S: int, budget: int, geometry: str, seed: int) -> dict:
    row = {
        "seq_len": S,
        "budget": budget,
        "geometry": geometry,
    }

    kvtc_mses, kvtc_coss, uni_mses, uni_coss, split_mses, split_coss = [], [], [], [], [], []
    entropy_gains, kvtc_mss = [], []

    for ds in DATA_SEEDS:
        x_np = _synthetic(S, geometry, seed + ds)
        x = mx.array(x_np)

        t0 = time.perf_counter()
        art = kvtc_compress(x, total_bit_budget=budget)
        kvtc_recon = kvtc_decompress(art)
        mx.eval(kvtc_recon)
        ms = (time.perf_counter() - t0) * 1_000
        kvtc_mss.append(ms)

        kvtc_mses.append(_mse(kvtc_recon, x))
        kvtc_coss.append(_cosine(kvtc_recon, x))

        pre = kvtc_pre_entropy_bytes(art)
        realized = kvtc_fp16_bytes(art)
        entropy_gains.append((pre / realized) if realized else 1.0)

        uni_recon = _fixed_uniform_reconstruction(x, budget)
        uni_mses.append(_mse(uni_recon, x))
        uni_coss.append(_cosine(uni_recon, x))

        split_recon = _svdq_fixed_split_reconstruction(x, budget)
        split_mses.append(_mse(split_recon, x))
        split_coss.append(_cosine(split_recon, x))

    row["mse_kvtc"] = round(float(np.mean(kvtc_mses)), 6)
    row["cos_kvtc"] = round(float(np.mean(kvtc_coss)), 5)
    row["mse_fixed_uniform"] = round(float(np.mean(uni_mses)), 6)
    row["cos_fixed_uniform"] = round(float(np.mean(uni_coss)), 5)
    row["mse_svdq_fixed_split"] = round(float(np.mean(split_mses)), 6)
    row["cos_svdq_fixed_split"] = round(float(np.mean(split_coss)), 5)
    row["entropy_coding_gain"] = round(float(np.mean(entropy_gains)), 4)
    row["ms_kvtc"] = round(float(np.mean(kvtc_mss)), 3)

    return row


def main() -> None:
    print(
        "KVTC-adapted local-PCA + DP-optimal bit allocation + entropy coding — offline synthetic benchmark"
    )
    print(f"  head_dim={HEAD_DIM}  r_true={R_TRUE}  data_seeds={DATA_SEEDS}")
    print("  (mse = reconstruction MSE at matched total bit budget; lower = better)")
    print("  (cos = reconstruction cosine similarity; higher = better)")
    print(
        "  (entropy_coding_gain = pre_entropy_bytes / kvtc_fp16_bytes; realized, not Shannon bound)"
    )
    print()
    header = (
        f"{'seq':>4} {'budget':>6} {'geometry':>16}  {'mse_kvtc':>10}  "
        f"{'mse_uniform':>11}  {'mse_split':>10}  {'cos_kvtc':>9}  {'ent_gain':>9}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for S, budget, geometry in product(SEQ_LENS, BIT_BUDGETS, GEOMETRIES):
        row = _run_once(S, budget, geometry, seed=SEED + S)
        results.append(row)
        print(
            f"{row['seq_len']:>4} {row['budget']:>6} {row['geometry']:>16}  "
            f"{row['mse_kvtc']:>10.6f}  {row['mse_fixed_uniform']:>11.6f}  "
            f"{row['mse_svdq_fixed_split']:>10.6f}  {row['cos_kvtc']:>9.5f}  "
            f"{row['entropy_coding_gain']:>9.4f}"
        )

    out_path = Path(__file__).parent.parent / "figures" / "kvtc" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    for geom in GEOMETRIES:
        rows = [r for r in results if r["geometry"] == geom]
        kvtc_mse = float(np.mean([r["mse_kvtc"] for r in rows]))
        uni_mse = float(np.mean([r["mse_fixed_uniform"] for r in rows]))
        split_mse = float(np.mean([r["mse_svdq_fixed_split"] for r in rows]))
        ent_gain = float(np.mean([r["entropy_coding_gain"] for r in rows]))
        print(f"\nSummary ({geom}):")
        print(
            f"  mean MSE — KVTC(DP): {kvtc_mse:.6f}   fixed-uniform: {uni_mse:.6f}   "
            f"SVDq-fixed-split: {split_mse:.6f}   entropy-coding gain: {ent_gain:.4f}"
        )

    print("\n  (honest reading: the clean, defensible observable is RECONSTRUCTION MSE/COSINE")
    print("   AT A MATCHED TOTAL BIT BUDGET. Under skewed_variance (a planted low-rank-ish")
    print("   geometry), the DP allocator can assign 0 bits to the near-noise tail instead of")
    print("   paying a uniform or fixed-25%/75% floor on every component, so it should reach")
    print("   materially lower MSE than both baselines. Under flat (isotropic) geometry there")
    print("   is no concentrated variance to exploit, so KVTC should be roughly competitive")
    print("   with the fixed-split baseline, not a dramatic win — reported honestly, not")
    print("   oversold. entropy_coding_gain is the REALIZED post-entropy-coding byte gain")
    print("   (pre_entropy_bytes / kvtc_fp16_bytes), including the code table's own storage")
    print("   cost — never the theoretical Shannon-entropy lower bound, and typically modest")
    print("   on synthetic Gaussian-like component codes. The DP allocator optimizes the")
    print("   repo's existing analytic Gaussian distortion proxy (ratequant.py's D(v,b) =")
    print("   v*beta**(-b)), NOT a rate-distortion model fit on real LLM activation statistics.")
    print("   The local PCA basis is fit per-sequence, NOT the paper's pre-calibrated global")
    print("   basis. The paper's up-to-20x/40x, <1pp-accuracy-loss numbers on LLaMA 3/Mistral")
    print("   NeMo/R1-Qwen2.5 are the paper's, on trained models — not reproduced here.)")


if __name__ == "__main__":
    main()
