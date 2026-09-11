"""Synchronized PyramidKV eviction benchmark; timings include Python dispatch."""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from veloxquant_mlx.metal._pyramidkv_evict import pyramidkv_fused_evict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 512, 2048, 4096, 8192])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if min(*args.budgets, args.heads, args.dim, args.repeats) < 1 or args.warmup < 0:
        parser.error("dimensions and repeats must be positive; warmup must be nonnegative")
    mx.random.seed(42)
    rows = []
    for budget in args.budgets:
        h, n, d = args.heads, budget + 1, args.dim
        sink = min(4, budget)
        k = mx.random.normal((h, n, d)).astype(mx.float16)
        v = mx.random.normal((h, n, d)).astype(mx.float16)
        s = mx.random.uniform(shape=(h, n)).astype(mx.float32)
        mx.eval(k, v, s)

        def baseline(k=k, v=v, s=s, sink=sink, n=n):
            ev = mx.argmin(s[:, sink:], axis=1) + sink
            j = mx.arange(n - 1)[None, :]
            idx = j + (j >= ev[:, None])
            return (
                mx.take_along_axis(k, idx[:, :, None], axis=1),
                mx.take_along_axis(v, idx[:, :, None], axis=1),
                mx.take_along_axis(s, idx, axis=1),
            )

        def metal(k=k, v=v, s=s, sink=sink):
            return pyramidkv_fused_evict(k, v, s, sink)

        ref, out = baseline(), metal()
        mx.eval(*ref, *out)
        if not all(mx.array_equal(a, b).item() for a, b in zip(ref, out, strict=True)):
            raise AssertionError("Metal output differs from MLX")
        for _ in range(args.warmup):
            mx.eval(*baseline())
            mx.eval(*metal())
        samples = {"mlx": [], "metal": []}
        for i in range(args.repeats):
            order = [("mlx", baseline), ("metal", metal)]
            for name, fn in order if i % 2 == 0 else reversed(order):
                start = time.perf_counter_ns()
                mx.eval(*fn())
                samples[name].append((time.perf_counter_ns() - start) / 1e6)
        row = {"budget": budget, "exact_parity": True, "samples_ms": samples}
        row["median_ms"] = {name: statistics.median(vals) for name, vals in samples.items()}
        rows.append(row)
        print(budget, row["median_ms"], flush=True)
    args.output.write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "args": vars(args) | {"output": str(args.output)},
                "results": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
