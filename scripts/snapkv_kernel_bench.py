"""Synchronized SnapKV selection and full cache benchmark; no model required."""

import argparse
import json
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.cache.snapkv_cache import SnapKVKVCache
from veloxquant_mlx.quantizers.snapkv import _snap_select_batched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    mx.random.seed(92)
    rows = []
    for n in (512, 2048, 8192):
        budget = min(512, n // 4)
        keys = mx.random.normal((1, 8, n, 128)).astype(mx.float16)
        values = mx.random.normal(keys.shape).astype(mx.float16)
        scores = mx.random.uniform(shape=(8, n))
        mx.eval(keys, values, scores)
        for stage in ("selection", "cache"):
            samples = {b: [] for b in ("reference", "mlx", "metal")}
            for trial in range(args.repeats + 5):
                for backend in list(samples)[:: 1 if trial % 2 else -1]:
                    cache = SnapKVKVCache(KVCacheConfig(snap_budget=budget, snap_backend=backend))
                    start = time.perf_counter_ns()
                    out = (
                        _snap_select_batched(scores, budget, 4, backend=backend)
                        if stage == "selection"
                        else cache.update_and_fetch(keys, values)
                    )
                    mx.eval(out)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    if trial >= 5:
                        samples[backend].append(elapsed)
            for backend, timings in samples.items():
                rows.append(
                    dict(
                        n=n,
                        budget=budget,
                        groups=8,
                        dim=128,
                        stage=stage,
                        backend=backend,
                        median_ms=statistics.median(timings),
                        p95_ms=sorted(timings)[min(len(timings) - 1, int(len(timings) * 0.95))],
                        samples_ms=timings,
                    )
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                platform=platform.platform(),
                mlx=version("mlx"),
                mlx_lm=version("mlx-lm"),
                device=mx.metal.device_info(),
                warmups=5,
                results=rows,
            ),
            indent=2,
        )
    )
    print(args.output)


if __name__ == "__main__":
    main()
