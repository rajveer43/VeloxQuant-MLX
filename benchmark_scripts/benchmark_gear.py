"""Offline GEAR benchmark — error-feedback reconstruction quality vs base quant.

GEAR (arXiv:2403.05527-adapted, Kang et al.) makes a low-bit base quantizer
near-lossless via a low-rank residual + sparse outlier correction. This harness
is **fully offline** — it loads no model and allocates only small synthetic KV
matrices — so it runs in a few hundred MB of RAM. It measures, on synthetic
low-rank-plus-outlier data (the regime GEAR targets), separately for the
paper's KCVT backbone axes (keys: per-channel, values: per-token):

  - reconstruction MSE: GEAR vs base-quant-alone vs fp16
  - stored bytes: GEAR (codes + low-rank factors + sparse triples) vs base-only
    vs fp16, and the effective bits/element
  - error-recovery ratio (fraction of base quantization error removed)
  - compress+reconstruct throughput (per-head, synthetic) across configs

Results are written to ``results_gear.json`` next to this script. **Not yet
run** on hardware — no numbers are claimed in the docs/CHANGELOG until this is
executed and its ``results_gear.json`` committed.

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_gear.py
    PYTHONPATH=. python benchmark_scripts/benchmark_gear.py --seq 256 --heads 8 --dim 128
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from veloxquant_mlx.quantizers.gear import (
    base_only_bytes,
    gear_bytes,
    gear_compress,
    gear_reconstruct,
    quantize_base,
)


def _synth_head(S: int, D: int, r: int, n_out: int, seed: int) -> mx.array:
    """Low-rank signal + small noise + sparse large outliers — GEAR's ideal case."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((S, r)).astype(np.float32)
    B = rng.standard_normal((r, D)).astype(np.float32)
    X = A @ B + 0.03 * rng.standard_normal((S, D)).astype(np.float32)
    flat = X.reshape(-1)
    idx = rng.choice(flat.size, size=n_out, replace=False)
    flat[idx] += rng.standard_normal(n_out).astype(np.float32) * 8.0
    return mx.array(X.reshape(S, D).astype(np.float16))


def _mse(a: mx.array, b: mx.array) -> float:
    return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())


def run(seq: int, heads: int, dim: int, signal_rank: int) -> dict:
    configs = [
        {"name": "gear_2bit_r8_sp0.5", "bits": 2, "rank": 8, "sparse": 0.005},
        {"name": "gear_2bit_r16_sp1.0", "bits": 2, "rank": 16, "sparse": 0.01},
        {"name": "gear_3bit_r8_sp0.5", "bits": 3, "rank": 8, "sparse": 0.005},
        {"name": "gear_4bit_r4_sp0.2", "bits": 4, "rank": 4, "sparse": 0.002},
    ]
    n_out = max(1, int(seq * dim * 0.003))
    heads_data = [_synth_head(seq, dim, signal_rank, n_out, seed=h) for h in range(heads)]
    fp16_bytes = seq * dim * 2
    # KCVT backbone: keys quantized per-channel, values per-token (§2 of the
    # paper). Benchmarked separately since they are genuinely different
    # quantizers, not a cosmetic label — see gear.py's quantize_base.
    axes = [("key", "channel"), ("value", "token")]

    results = []
    for cfg in configs:
        for tensor_kind, base_axis in axes:
            mse_gear = mse_base = comp = base = 0.0
            err_base_sq = err_after_sq = 0.0
            t0 = time.perf_counter()
            for X in heads_data:
                st = gear_compress(
                    X,
                    bits=cfg["bits"],
                    rank=cfg["rank"],
                    sparse_frac=cfg["sparse"],
                    group_size=32,
                    base_axis=base_axis,
                )
                rec = gear_reconstruct(st)
                _, base_rec = quantize_base(
                    X.astype(mx.float32), cfg["bits"], 32, axis=base_axis
                )
                mse_gear += _mse(rec, X)
                mse_base += _mse(base_rec, X)
                comp += gear_bytes(st)
                base += base_only_bytes(st)
                err_base_sq += float(
                    mx.sum((X.astype(mx.float32) - base_rec.astype(mx.float32)) ** 2).item()
                )
                err_after_sq += float(
                    mx.sum((X.astype(mx.float32) - rec.astype(mx.float32)) ** 2).item()
                )
            mx.eval()
            elapsed = time.perf_counter() - t0

            results.append(
                {
                    "config": cfg["name"],
                    "tensor_kind": tensor_kind,
                    "base_axis": base_axis,
                    "bits": cfg["bits"],
                    "rank": cfg["rank"],
                    "sparse_fraction": cfg["sparse"],
                    "mse_gear": mse_gear / heads,
                    "mse_base_only": mse_base / heads,
                    "mse_improvement_pct": round(100 * (1 - (mse_gear / mse_base)), 2)
                    if mse_base
                    else 0.0,
                    "stored_bytes_gear": int(comp),
                    "stored_bytes_base_only": int(base),
                    "stored_bytes_fp16": int(fp16_bytes * heads),
                    "effective_bits": round(16.0 * comp / (fp16_bytes * heads), 3),
                    "error_recovery_ratio": round(1 - err_after_sq / err_base_sq, 4)
                    if err_base_sq
                    else 0.0,
                    "compress_reconstruct_sec_per_head": round(elapsed / heads, 5),
                }
            )
    return {
        "harness": "offline-synthetic (no model loaded)",
        "backbone": "KCVT (keys per-channel, values per-token)",
        "platform": platform.platform(),
        "seq": seq,
        "heads": heads,
        "dim": dim,
        "signal_rank": signal_rank,
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--signal-rank", type=int, default=12)
    args = ap.parse_args()

    out = run(args.seq, args.heads, args.dim, args.signal_rank)
    print(json.dumps(out, indent=2))
    dest = Path(__file__).with_name("results_gear.json")
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
