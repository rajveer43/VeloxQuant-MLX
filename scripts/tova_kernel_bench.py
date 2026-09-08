"""Reproducible TOVA selection, full-update, cache, and decode-chain timings.

Run from the repository: .venv/bin/python scripts/tova_kernel_bench.py --output /tmp/tova.json
All times are synchronized wall times, including Python/MLX dispatch overhead.
"""

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.metal import tova_fused_evict
from veloxquant_mlx.quantizers.tova import _evict_mlx, _tova_update_batched


def timing(fn, warmup, repeats):
    start = time.perf_counter_ns()
    mx.eval(*fn())
    first_ms = (time.perf_counter_ns() - start) / 1e6
    for _ in range(warmup):
        mx.eval(*fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        mx.eval(*fn())
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return dict(
        first_call_ms=first_ms,
        median_ms=float(np.median(samples)),
        p95_ms=float(np.percentile(samples, 95)),
        samples_ms=samples,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--heads", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["reference", "mlx", "metal"],
        default=["reference", "mlx", "metal"],
    )
    args = parser.parse_args()
    if (
        min(args.budgets) <= 4
        or min(args.heads) < 1
        or min(args.repeats, args.decode_steps, args.dim) < 1
    ):
        parser.error("budgets must exceed 4; heads/repeats/decode-steps/dim must be positive")
    rng = np.random.default_rng(42)
    results = []
    for h in args.heads:
        for budget in args.budgets:
            d = args.dim
            k = mx.array(rng.normal(size=(h, budget, d)).astype(np.float16))
            v = mx.array(rng.normal(size=k.shape).astype(np.float16))
            new_k = mx.array(rng.normal(size=(h, args.decode_steps, d)).astype(np.float16))
            new_v = mx.array(rng.normal(size=new_k.shape).astype(np.float16))
            mid_k = mx.concatenate([k, new_k[:, :1]], axis=1)
            mid_v = mx.concatenate([v, new_v[:, :1]], axis=1)
            w = mx.softmax(mx.array(rng.normal(size=(h, budget + 1)).astype(np.float32)))
            mx.eval(k, v, new_k, new_v, mid_k, mid_v, w)
            for backend in args.backends:
                update = partial(
                    _tova_update_batched,
                    k,
                    v,
                    new_k[:, :1],
                    new_v[:, :1],
                    4,
                    budget,
                    backend=backend,
                )
                chain = partial(
                    _tova_update_batched, k, v, new_k, new_v, 4, budget, backend=backend
                )

                cfg = KVCacheConfig(
                    method="tova",
                    head_dim=d,
                    tova_budget=budget,
                    tova_n_sink=4,
                    tova_backend=backend,
                )
                cache = KVCacheFactory.create(cfg)
                cache.update_and_fetch(k[None], v[None])

                def cache_step(cache=cache, k=k, v=v, budget=budget, new_k=new_k, new_v=new_v):
                    # Same pre-materialized starting state for every sample.
                    # Restore outside the timer would omit work inconsistently;
                    # assign the same two buffers here for every backend.
                    cache.keys, cache.values = k[None], v[None]
                    cache._true_offset = budget
                    cache.offset = budget
                    return cache.update_and_fetch(new_k[None, :, :1], new_v[None, :, :1])

                stages = {"update": update, "cache": cache_step, "decode_chain": chain}
                if backend != "reference":
                    evict = _evict_mlx if backend == "mlx" else tova_fused_evict
                    stages["select_apply"] = partial(evict, mid_k, mid_v, w, 4)
                for stage, fn in stages.items():
                    result = dict(
                        heads=h,
                        dim=d,
                        budget=budget,
                        backend=backend,
                        stage=stage,
                        steps=args.decode_steps if stage == "decode_chain" else 1,
                        **timing(fn, args.warmup, args.repeats),
                    )
                    results.append(result)
                    print(
                        f"H={h} N={budget} {backend:9} {stage:12} {result['median_ms']:.4f} ms",
                        flush=True,
                    )
    report = dict(
        device=mx.device_info(),
        macos=platform.mac_ver()[0],
        python=platform.python_version(),
        mlx=importlib.metadata.version("mlx"),
        mlx_lm=importlib.metadata.version("mlx-lm"),
        seed=42,
        warmup=args.warmup,
        repeats=args.repeats,
        note="first_call_ms includes dispatch/evaluation; it is not isolated compilation time",
        results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
