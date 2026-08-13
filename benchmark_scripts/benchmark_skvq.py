"""Offline-synthetic benchmark for SKVQ-adapted sliding-window quantization.

SKVQ's two mechanisms — channel reordering and clipped dynamic quantization
— only pay off when channels are *heterogeneous* (real K/V have a few
dominant channels; that is the paper's premise, shared with KIVI/KVQuant).
That premise cannot be validated on synthetic data, so this harness runs two
regimes and reports both honestly:

  1. **heterogeneous** — per-channel scales spread smoothly over 2.5
     decades (logspace, shuffled so similar scales are not contiguous).
     Reordering should clearly reduce reconstruction error here: this
     validates the *machinery* given the paper's premise, not the premise
     itself. (Extreme few-outlier constructions are deliberately avoided:
     there, concentrating all outliers into one group can *widen* that
     group's range and absolute MSE becomes a coin flip — the normalized
     per-channel metric below is what reordering reliably rescues.)
  2. **homogeneous** — unit channel scales, where reordering has nothing to
     sort. It should buy ~nothing. Reporting this control is the point: no
     fabricated advantage.

Arms at matched bits / group size / window:
  - skvq            reorder + per-group clip search (the method)
  - skvq_noreorder  clip search only
  - skvq_noclip     reorder only (alpha = 1)
  - skvq_plain      both off — plain per-token group quant behind the window
  - kivi            the repo's KIVI-adapted reference (per-channel keys,
                    per-token values) at the same bits/group

Metrics over the region quantized by *every* arm: key reconstruction MSE,
key **normalized** per-channel error (channel MSE / channel variance — the
metric that shows what reordering buys: small channels stop sharing a group
range with large ones), attention output perturbation (mean 1 − cosine of
probe attention output vs the full fp16 cache, same metric family as the
knorm/CaM/xKV benchmarks), compressed bytes/token, and wall-clock flush
time.

Coverage note, stated up front: SKVQ quantizes every token that ages past
the window (tail = S mod window stays fp16), while the repo's KIVI wrapper
keeps the trailing ``residual_length`` tokens of the incoming block fp16.
At S ≡ 0 (mod window) SKVQ has quantized *everything* and KIVI still holds
128 fp16 tokens — a small built-in perturbation advantage for KIVI that the
MSE-over-common-region metric avoids. Both are reported.

**Explicitly NOT a model-level perplexity/throughput benchmark.**

Usage
-----
    python benchmark_scripts/benchmark_skvq.py

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
SEQ_LENS = [512, 1024]
BITS = [2, 4]
REGIMES = ["heterogeneous", "homogeneous"]
HEAD_DIM = 64
N_HEADS = 2
GROUP = 16
WINDOW = 128
N_SINK = 5
N_PROBES = 32
SEED = 11


def _synthetic(S: int, regime: str, seed: int):
    """K/V [1, H, S, D] fp16 + probe queries [P, D] fp32."""
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((1, N_HEADS, S, HEAD_DIM)).astype(np.float32)
    v = rng.standard_normal((1, N_HEADS, S, HEAD_DIM)).astype(np.float32)
    if regime == "heterogeneous":
        # Smooth 2.5-decade per-channel scale spread, shuffled so similar
        # scales are not contiguous (the reordering premise).
        scales = np.logspace(-1.5, 1, HEAD_DIM).astype(np.float32)
        rng.shuffle(scales)
        k = k * scales[None, None, None, :]
        v = v * scales[None, None, None, :]
    q = rng.standard_normal((N_PROBES, HEAD_DIM)).astype(np.float32)
    return (
        mx.array(k.astype(np.float16)),
        mx.array(v.astype(np.float16)),
        mx.array(q),
    )


def _attn_out(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    scale = 1.0 / math.sqrt(float(k.shape[-1]))
    w = mx.softmax((q @ k.T) * scale, axis=-1)
    return w @ v


def _perturbation(q, k_full, v_full, k_got, v_got) -> float:
    """Mean 1 − cosine of probe attention output vs the fp16 cache,
    averaged over heads."""
    vals = []
    for h in range(k_full.shape[1]):
        ref = _attn_out(q, k_full[0, h].astype(mx.float32), v_full[0, h].astype(mx.float32))
        got = _attn_out(q, k_got[0, h].astype(mx.float32), v_got[0, h].astype(mx.float32))
        rn = ref / (mx.sqrt(mx.sum(ref * ref, -1, keepdims=True)) + 1e-8)
        gn = got / (mx.sqrt(mx.sum(got * got, -1, keepdims=True)) + 1e-8)
        vals.append(float(mx.mean(1.0 - mx.sum(rn * gn, -1)).item()))
    return float(np.mean(vals))


def _run_arm(cfg: dict, k, v):
    cache = KVCacheFactory.create(KVCacheConfig(head_dim=HEAD_DIM, **cfg))
    t0 = time.perf_counter()
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    ms = (time.perf_counter() - t0) * 1_000
    return cache, ko, vo, ms


def _key_errs(k, ko, region: slice) -> tuple:
    """(absolute MSE, per-channel normalized MSE) over the common region."""
    a = np.array(k[:, :, region, :], dtype=np.float32)
    b = np.array(ko[:, :, region, :], dtype=np.float32)
    err_ch = np.mean((b - a) ** 2, axis=(0, 1, 2))
    var_ch = np.maximum(a.var(axis=(0, 1, 2)), 1e-12)
    return float(np.mean((b - a) ** 2)), float(np.mean(err_ch / var_ch))


def _run_once(S: int, bits: int, regime: str, seed: int) -> dict:
    k, v, q = _synthetic(S, regime, seed)

    skvq_base = dict(
        method="skvq",
        skvq_bits_key=bits,
        skvq_bits_value=bits,
        skvq_group_size=GROUP,
        skvq_window=WINDOW,
        skvq_n_sink=N_SINK,
        skvq_max_ctx=max(8192, S),
    )
    arms = {
        "skvq": dict(skvq_base),
        "skvq_noreorder": dict(skvq_base, skvq_reorder=False),
        "skvq_noclip": dict(skvq_base, skvq_clip_search=False),
        "skvq_plain": dict(skvq_base, skvq_reorder=False, skvq_clip_search=False),
        "kivi": dict(
            method="kivi", bit_width_inlier=bits, kivi_group_size=GROUP, residual_length=WINDOW
        ),
    }

    outs, ms = {}, {}
    caches = {}
    for name, cfg in arms.items():
        caches[name], ko, vo, ms[name] = _run_arm(cfg, k, v)
        outs[name] = (ko, vo)

    # Region quantized by EVERY arm: past the SKVQ sinks, before both the
    # SKVQ flush frontier and KIVI's fp16 residual tail.
    q_end = caches["skvq"].quantized_tokens
    common = slice(N_SINK, min(q_end, S - WINDOW))

    row = {
        "seq_len": S,
        "bits": bits,
        "regime": regime,
        "quantized_tokens_skvq": q_end,
        "common_region": [common.start, common.stop],
    }
    for name in arms:
        mse, nmse = _key_errs(k, outs[name][0], common)
        row[f"key_mse_{name}"] = round(mse, 6)
        row[f"key_nmse_{name}"] = round(nmse, 6)
        row[f"pert_{name}"] = round(_perturbation(q, k, v, outs[name][0], outs[name][1]), 6)
    sk = caches["skvq"]
    row["skvq_bytes_per_token"] = round(
        (
            sk.compressed_key_bytes
            + sk.compressed_value_bytes
            + sk.residual_fp16_bytes
            + sk.perm_bytes
        )
        / sk.tokens_seen,
        2,
    )
    row["fp16_bytes_per_token"] = HEAD_DIM * 2 * 2 * N_HEADS
    row["skvq_assigned_avg_bits"] = round(sk.assigned_avg_bits, 3)
    row["skvq_ms"] = round(ms["skvq"], 2)
    row["kivi_ms"] = round(ms["kivi"], 2)
    return row


def main() -> None:
    print("SKVQ-adapted sliding-window quantization — offline synthetic benchmark")
    print(
        f"  head_dim={HEAD_DIM} heads={N_HEADS} group={GROUP} "
        f"window={WINDOW} n_sink={N_SINK} probes={N_PROBES}"
    )
    print(
        "  (key MSE over the region quantized by every arm; perturbation = "
        "1 - cosine vs full fp16 cache; lower = better)"
    )
    print()

    results = []
    hdr = (
        f"{'seq':>5} {'bits':>4} {'regime':>13}  {'skvq':>9} {'noreord':>9} "
        f"{'noclip':>9} {'plain':>9} {'kivi':>9}   metric"
    )
    print(hdr)
    print("-" * len(hdr))
    for S, bits, regime in product(SEQ_LENS, BITS, REGIMES):
        row = _run_once(S, bits, regime, seed=SEED + S)
        results.append(row)
        print(
            f"{S:>5} {bits:>4} {regime:>13}  "
            f"{row['key_mse_skvq']:>9.5f} {row['key_mse_skvq_noreorder']:>9.5f} "
            f"{row['key_mse_skvq_noclip']:>9.5f} {row['key_mse_skvq_plain']:>9.5f} "
            f"{row['key_mse_kivi']:>9.5f}   key_mse"
        )
        print(
            f"{'':>5} {'':>4} {'':>13}  "
            f"{row['key_nmse_skvq']:>9.3f} {row['key_nmse_skvq_noreorder']:>9.3f} "
            f"{row['key_nmse_skvq_noclip']:>9.3f} {row['key_nmse_skvq_plain']:>9.3f} "
            f"{row['key_nmse_kivi']:>9.3f}   key_nmse"
        )
        print(
            f"{'':>5} {'':>4} {'':>13}  "
            f"{row['pert_skvq']:>9.5f} {row['pert_skvq_noreorder']:>9.5f} "
            f"{row['pert_skvq_noclip']:>9.5f} {row['pert_skvq_plain']:>9.5f} "
            f"{row['pert_kivi']:>9.5f}   pert"
        )

    out_path = Path(__file__).parent.parent / "figures" / "skvq" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    for regime in REGIMES:
        rows = [r for r in results if r["regime"] == regime]
        gain_reorder = np.mean(
            [
                (r["key_mse_skvq_noreorder"] - r["key_mse_skvq"])
                / max(r["key_mse_skvq_noreorder"], 1e-12)
                for r in rows
            ]
        )
        gain_clip = np.mean(
            [
                (r["key_mse_skvq_noclip"] - r["key_mse_skvq"])
                / max(r["key_mse_skvq_noclip"], 1e-12)
                for r in rows
            ]
        )
        print(f"\nSummary ({regime}):")
        print(f"  mean key-MSE reduction from reordering (given clip): {gain_reorder:+.1%}")
        print(f"  mean key-MSE reduction from clip search (given reorder): {gain_clip:+.1%}")

    print("\n  (honest reading: reordering's win exists only under channel")
    print("   heterogeneity — the homogeneous control should show ~0%. Whether")
    print("   real transformer K/V exhibit the heterogeneous regime is the")
    print("   paper's claim (and KIVI's/KVQuant's), not this benchmark's. The")
    print("   KIVI column is a genuinely different scheme (per-channel key")
    print("   groups along tokens) and may win rows; reported as measured.)")


if __name__ == "__main__":
    main()
