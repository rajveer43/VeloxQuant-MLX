"""Attend latency and memory benchmark across sequence lengths.

Compares four configurations:
  baseline   — no optimizations
  vectorized — enable_vectorized_attend=True
  fused      — enable_vectorized_attend=True + enable_fused_query_dot=True
  all        — fused + enable_outlier_two_stream=True

Usage::

    python -m veloxquant_mlx.benchmarks.attend_benchmark
    python -m veloxquant_mlx.benchmarks.attend_benchmark --method turboquant_mse --bits 2
    python -m veloxquant_mlx.benchmarks.attend_benchmark --seq_lens 64 256 1024 4096
"""

from __future__ import annotations

import argparse
import time
from typing import List

import numpy as np


_SEQ_LENS: List[int] = [128, 512, 1000, 2048]
_N_ATTEND_CALLS = 20  # per measurement
_N_OUTLIER_CHANNELS = 4
_N_CALIB_TOKENS = 50  # short so calibration completes inside every seq_len


def _build(
    method: str,
    d: int,
    bits: int,
    jl_dim: int,
    seed: int,
    *,
    vectorized: bool = False,
    fused: bool = False,
    outlier: bool = False,
):
    from veloxquant_mlx.cache.base import KVCacheBuilder

    return (
        KVCacheBuilder()
        .with_method(method)
        .with_head_dim(d)
        .with_bit_width(inlier=bits)
        .with_jl_dim(jl_dim)
        .with_seed(seed)
        .with_vectorized_attend(vectorized)
        .with_fused_query_dot(fused)
        .with_outlier_two_stream(outlier)
        .with_n_outlier_channels(_N_OUTLIER_CHANNELS)
        .with_n_calib_tokens(_N_CALIB_TOKENS)
        .build()
    )


def _fill(cache, keys, vals) -> None:
    for i in range(len(keys)):
        cache.append(keys[i], vals[i])


def _measure_attend_ms(cache, q, n_calls: int) -> float:
    import mlx.core as mx

    # Warm-up
    mx.eval(cache.attend(q))

    t0 = time.perf_counter()
    for _ in range(n_calls):
        mx.eval(cache.attend(q))
    return (time.perf_counter() - t0) * 1_000.0 / n_calls


def _correctness_check(cache_a, cache_b, q, label_a: str, label_b: str) -> None:
    """Assert two caches produce numerically close attend outputs."""
    import mlx.core as mx

    out_a = np.array(cache_a.attend(q))
    out_b = np.array(cache_b.attend(q))
    mx.eval()
    try:
        np.testing.assert_allclose(out_a, out_b, rtol=5e-3, atol=5e-3)
        print(
            f"  Correctness {label_a} vs {label_b}: OK (max_diff={np.max(np.abs(out_a - out_b)):.5f})"
        )
    except AssertionError as e:
        print(f"  Correctness {label_a} vs {label_b}: FAIL — {e}")


def run(
    method: str,
    d: int,
    bits: int,
    jl_dim: int,
    seed: int,
    seq_lens: List[int],
    n_calls: int,
    correctness: bool,
) -> None:
    import mlx.core as mx

    rng = np.random.default_rng(seed)

    configs = {
        "baseline": dict(vectorized=False, fused=False, outlier=False),
        "vectorized": dict(vectorized=True, fused=False, outlier=False),
        "fused": dict(vectorized=True, fused=True, outlier=False),
        "all_opts": dict(vectorized=True, fused=True, outlier=True),
    }
    # turboquant_mse doesn't have a fused path; skip 'fused'/'all_opts' for it.
    if method == "turboquant_mse":
        configs = {
            "baseline": dict(vectorized=False, fused=False, outlier=False),
            "vectorized": dict(vectorized=True, fused=False, outlier=False),
            "all_opts": dict(vectorized=True, fused=False, outlier=True),
        }

    col_w = 14
    header = f"{'seq_len':>8}  " + "  ".join(f"{k:>{col_w}}" for k in configs)
    print(f"\n=== attend latency (ms/call) — method={method}, d={d}, bits={bits} ===")
    print(header)
    print("-" * len(header))

    for seq_len in seq_lens:
        keys = [mx.array(rng.standard_normal(d).astype(np.float16)) for _ in range(seq_len)]
        vals = [mx.array(rng.standard_normal(d).astype(np.float16)) for _ in range(seq_len)]
        q = mx.array(rng.standard_normal(d).astype(np.float16))

        caches = {}
        latencies = {}
        for name, flags in configs.items():
            c = _build(method, d, bits, jl_dim, seed, **flags)
            _fill(c, keys, vals)
            caches[name] = c
            latencies[name] = _measure_attend_ms(c, q, n_calls)

        row = f"{seq_len:>8}  " + "  ".join(f"{latencies[k]:>{col_w}.3f}" for k in configs)
        print(row)

        # Speedup summary
        base_ms = latencies["baseline"]
        for name, ms in latencies.items():
            if name != "baseline":
                print(f"          {name:>12}: {base_ms / max(ms, 1e-9):.2f}× speedup vs baseline")

        # Optional correctness check
        if correctness and len(caches) >= 2:
            names = list(caches.keys())
            _correctness_check(caches[names[0]], caches[names[1]], q, names[0], names[1])

        # Memory footprint
        print(
            f"          memory (bytes): "
            + ", ".join(f"{n}={c.memory_bytes()}" for n, c in caches.items())
        )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant attend latency sweep")
    parser.add_argument(
        "--method", default="turboquant_prod", choices=["turboquant_prod", "turboquant_mse"]
    )
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--jl_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_lens", type=int, nargs="*", default=_SEQ_LENS)
    parser.add_argument("--n_calls", type=int, default=_N_ATTEND_CALLS)
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Run cross-config correctness checks at each seq_len",
    )
    args = parser.parse_args()

    run(
        method=args.method,
        d=args.head_dim,
        bits=args.bits,
        jl_dim=args.jl_dim,
        seed=args.seed,
        seq_lens=args.seq_lens,
        n_calls=args.n_calls,
        correctness=args.correctness,
    )


if __name__ == "__main__":
    main()
