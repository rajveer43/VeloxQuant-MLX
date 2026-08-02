"""Offline ZipCache-adapted benchmark — saliency-routing reconstruction quality.

ZipCache-adapted (arXiv:2405.14256-adapted, He et al.) routes high-key-norm
tokens to hi_bits and low-key-norm tokens to lo_bits within the quantized space.
This harness is **fully offline** — it loads no model and allocates only small
synthetic KV matrices — so it runs in a few hundred MB of RAM. It measures:

  - reconstruction MSE: ZipCache-adapted vs uniform-lo-bit vs uniform-hi-bit vs fp16
  - stored bytes: mixed-bit vs uniform-lo baseline vs fp16, and effective bits/element
  - compress+reconstruct throughput (per-head, synthetic) across configs

Results are written to ``results_zipcache.json`` next to this script.

NOT YET RUN on hardware — no numbers are claimed in docs/CHANGELOG until this
is executed and its ``results_zipcache.json`` is committed.

Usage::

    PYTHONPATH=. python benchmark_scripts/benchmark_zipcache.py
    PYTHONPATH=. python benchmark_scripts/benchmark_zipcache.py --seq 512 --heads 4 --dim 128
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from veloxquant_mlx.quantizers.zipcache import (
    base_only_bytes,
    zipcache_bytes,
    zipcache_compress,
    zipcache_reconstruct,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_kv(S: int, D: int, seed: int = 0) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((S, D)).astype(np.float32))


def _high_norm_outlier_kv(S: int, D: int, seed: int = 0, n_outliers: int = 10) -> mx.array:
    """Synthetic data with a small fraction of high-norm tokens — ZipCache's ideal case."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((S, D)).astype(np.float32)
    idx = rng.choice(S, size=n_outliers, replace=False)
    x[idx] *= 5.0  # inflate these tokens' norms
    return mx.array(x)


def _mse(a: mx.array, b: mx.array) -> float:
    return float(mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2).item())


def _benchmark_one(
    x: mx.array, hi_bits: int, lo_bits: int, hi_fraction: float, group_size: int, n_rep: int = 10
) -> dict:
    S, D = x.shape
    # Warmup
    for _ in range(2):
        state = zipcache_compress(x, hi_bits, lo_bits, hi_fraction, group_size)
        mx.eval(zipcache_reconstruct(state))

    # Timed
    t0 = time.perf_counter()
    for _ in range(n_rep):
        state = zipcache_compress(x, hi_bits, lo_bits, hi_fraction, group_size)
        mx.eval(zipcache_reconstruct(state))
    t1 = time.perf_counter()

    recon = zipcache_reconstruct(state)
    mx.eval(recon)

    # Uniform baselines for comparison
    lo_state = zipcache_compress(x, lo_bits, lo_bits, 1.0, group_size)
    lo_recon = zipcache_reconstruct(lo_state)
    mx.eval(lo_recon)
    hi_state = zipcache_compress(x, hi_bits, hi_bits, 1.0, group_size)
    hi_recon = zipcache_reconstruct(hi_state)
    mx.eval(hi_recon)

    comp = zipcache_bytes(state, group_size)
    fp16 = S * D * 2
    base_lo = base_only_bytes(S, D, lo_bits, group_size)
    base_hi = base_only_bytes(S, D, hi_bits, group_size)
    eff_bits = 16.0 * comp / fp16

    return {
        "seq_len": S,
        "head_dim": D,
        "hi_bits": hi_bits,
        "lo_bits": lo_bits,
        "hi_fraction": hi_fraction,
        "group_size": group_size,
        "mse_zipcache": _mse(recon, x),
        "mse_uniform_lo": _mse(lo_recon, x),
        "mse_uniform_hi": _mse(hi_recon, x),
        "mse_fp16": 0.0,
        "bytes_zipcache": comp,
        "bytes_uniform_lo": base_lo,
        "bytes_uniform_hi": base_hi,
        "bytes_fp16": fp16,
        "effective_bits": round(eff_bits, 3),
        "compression_ratio_vs_fp16": round(fp16 / comp, 3),
        "ms_per_head": round((t1 - t0) / n_rep * 1000, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ZipCache-adapted offline benchmark")
    parser.add_argument("--seq", type=int, default=128, help="Sequence length")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads to sweep")
    parser.add_argument("--dim", type=int, default=128, help="Head dim")
    parser.add_argument("--n_rep", type=int, default=10, help="Timing repetitions")
    args = parser.parse_args()

    seq_lens = [args.seq, args.seq * 4, args.seq * 16]
    hi_fractions = [0.1, 0.2, 0.3]
    hi_bits = 4
    lo_bits = 2
    group_size = 32
    D = args.dim

    results = []
    print(f"\nZipCache-adapted offline benchmark  (NOT YET RUN on dedicated hardware)\n")
    print(
        f"{'S':>6}  {'hi_frac':>8}  {'eff_bits':>9}  {'comp_ratio':>10}  "
        f"{'MSE_zip':>12}  {'MSE_lo':>12}  {'ms/head':>9}"
    )
    print("-" * 80)

    for S in seq_lens:
        for hi_frac in hi_fractions:
            # Use high-norm-outlier data (ZipCache's intended regime)
            n_out = max(1, int(S * hi_frac))
            x = _high_norm_outlier_kv(S, D, seed=42, n_outliers=n_out)
            r = _benchmark_one(x, hi_bits, lo_bits, hi_frac, group_size, args.n_rep)
            results.append(r)
            print(
                f"{S:>6}  {hi_frac:>8.2f}  {r['effective_bits']:>9.3f}  "
                f"{r['compression_ratio_vs_fp16']:>10.3f}  "
                f"{r['mse_zipcache']:>12.6f}  {r['mse_uniform_lo']:>12.6f}  "
                f"{r['ms_per_head']:>9.4f}"
            )

    out_path = Path(__file__).parent / "results_zipcache.json"
    summary = {
        "note": "NOT YET RUN on dedicated Apple Silicon hardware. "
        "Numbers above are from the development machine at benchmark-script execution time. "
        "Do not cite in paper until hardware results are committed.",
        "hardware": platform.node(),
        "platform": platform.platform(),
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
