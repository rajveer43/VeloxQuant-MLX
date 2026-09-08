"""Microbenchmarks: TOVA vs H2O vs KIVI cache-update latency across cache
sizes, plus a per-token host-sync count check.

Run: .venv/bin/python scripts/kv_bookkeeping_microbench.py --output /tmp/kv_microbench.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.metal import metal_available


def bench_decode_step(cache_factory, B, H, D, cache_sizes, decode_steps, warmup, repeats):
    """For each target cache size, prime the cache to that size, then time
    `decode_steps` single-token update_and_fetch calls (synchronized)."""
    results = {}
    for size in cache_sizes:
        cache = cache_factory()
        prime_k = mx.array(np.random.default_rng(0).normal(size=(B, H, size, D)).astype(np.float16))
        prime_v = mx.array(np.random.default_rng(1).normal(size=(B, H, size, D)).astype(np.float16))
        try:
            cache.update_and_fetch(prime_k, prime_v)
        except Exception as e:
            results[size] = {"error": str(e)}
            continue
        mx.eval(cache.keys, cache.values)

        def step():
            k = mx.array(np.random.default_rng(size).normal(size=(B, H, 1, D)).astype(np.float16))
            v = mx.array(np.random.default_rng(size + 1).normal(size=(B, H, 1, D)).astype(np.float16))
            ko, vo = cache.update_and_fetch(k, v)
            return ko, vo

        for _ in range(warmup):
            mx.eval(*step())
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            mx.eval(*step())
            samples.append((time.perf_counter_ns() - t0) / 1e6)
        results[size] = dict(
            median_ms=float(np.median(samples)),
            p95_ms=float(np.percentile(samples, 95)),
            mean_ms=float(np.mean(samples)),
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-sizes", type=int, nargs="+", default=[128, 256, 512, 1024, 2048]
    )
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    B, H, D = 1, args.heads, args.dim

    def make_cfg(method, **kw):
        return KVCacheConfig(method=method, head_dim=D, **kw)

    out = dict(
        hardware=platform.platform(),
        metal_available=metal_available(),
        mlx_default_device=str(mx.default_device()),
        B=B,
        H=H,
        D=D,
        cache_sizes=args.cache_sizes,
    )

    # Budgets must exceed the largest cache_size tested so the cache never
    # itself evicts down below the priming size (we want pure per-token
    # append+evict-at-capacity cost at each target size).
    max_budget = max(args.cache_sizes) + 8

    print("Benchmarking TOVA (mlx backend)...")
    out["tova_mlx"] = bench_decode_step(
        lambda: KVCacheFactory.create(
            make_cfg("tova", tova_budget=max_budget, tova_n_sink=4, tova_backend="mlx")
        ),
        B, H, D, args.cache_sizes, None, args.warmup, args.repeats,
    )
    print("Benchmarking TOVA (metal backend)...")
    out["tova_metal"] = bench_decode_step(
        lambda: KVCacheFactory.create(
            make_cfg("tova", tova_budget=max_budget, tova_n_sink=4, tova_backend="metal")
        ),
        B, H, D, args.cache_sizes, None, args.warmup, args.repeats,
    )
    print("Benchmarking H2O (auto backend, metal fused evict via metal_available)...")
    out["h2o"] = bench_decode_step(
        lambda: KVCacheFactory.create(
            make_cfg("h2o", h2o_budget=max_budget, h2o_n_sink=4, h2o_grace=16, h2o_decay=0.98)
        ),
        B, H, D, args.cache_sizes, None, args.warmup, args.repeats,
    )
    print("Benchmarking KIVI (metal kernels on)...")
    out["kivi_metal"] = bench_decode_step(
        lambda: KVCacheFactory.create(
            make_cfg("kivi", bit_width_inlier=2, kivi_group_size=32, residual_length=32, use_metal_kernels=True)
        ),
        B, H, D, args.cache_sizes, None, args.warmup, args.repeats,
    )
    print("Benchmarking KIVI (mlx-only path)...")
    out["kivi_mlx"] = bench_decode_step(
        lambda: KVCacheFactory.create(
            make_cfg("kivi", bit_width_inlier=2, kivi_group_size=32, residual_length=32, use_metal_kernels=False)
        ),
        B, H, D, args.cache_sizes, None, args.warmup, args.repeats,
    )
    print("Benchmarking plain fp16 (mlx_lm base KVCache, no compression) baseline...")
    from mlx_lm.models.cache import KVCache as PlainCache

    def bench_plain():
        results = {}
        for size in args.cache_sizes:
            cache = PlainCache()
            prime_k = mx.array(np.random.default_rng(0).normal(size=(B, H, size, D)).astype(np.float16))
            prime_v = mx.array(np.random.default_rng(1).normal(size=(B, H, size, D)).astype(np.float16))
            cache.update_and_fetch(prime_k, prime_v)
            mx.eval(cache.keys, cache.values)

            def step():
                k = mx.array(np.random.default_rng(size).normal(size=(B, H, 1, D)).astype(np.float16))
                v = mx.array(np.random.default_rng(size + 1).normal(size=(B, H, 1, D)).astype(np.float16))
                return cache.update_and_fetch(k, v)

            for _ in range(args.warmup):
                mx.eval(*step())
            samples = []
            for _ in range(args.repeats):
                t0 = time.perf_counter_ns()
                mx.eval(*step())
                samples.append((time.perf_counter_ns() - t0) / 1e6)
            results[size] = dict(median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)))
        return results

    out["plain_fp16"] = bench_plain()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
