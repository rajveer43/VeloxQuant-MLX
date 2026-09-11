"""Compare historical and current apply; requires git commit db86edf locally."""

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx

from veloxquant_mlx.metal._pyramidkv_evict import _apply, _reduce

old = mx.fast.metal_kernel(
    name="pyramid_old_apply_probe",
    input_names=["keys", "values", "scores", "evict_idx"],
    output_names=["keys_out", "values_out", "scores_out"],
    source=subprocess.check_output(
        ["git", "show", "db86edf:veloxquant_mlx/metal/src/pyramidkv_evict_apply.metal"], text=True
    ),
    ensure_row_contiguous=True,
)
rows = []
for budget in (4096, 8192):
    h, d, n = 8, 128, budget + 1
    k = mx.ones((h, n, d), dtype=mx.float16)
    v = -k
    s = mx.ones((h, n), dtype=mx.float32)
    ev = mx.full((h,), budget // 2, dtype=mx.int32)
    sink = mx.array([4], dtype=mx.uint32)
    mx.eval(k, v, s, ev, sink)

    def apply(fn, size, k=k, v=v, s=s, ev=ev, h=h, budget=budget, d=d):
        return fn(
            inputs=[k, v, s, ev],
            grid=(((size + 255) // 256) * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(h, budget, d), (h, budget, d), (h, budget)],
            output_dtypes=[mx.float16, mx.float16, mx.float32],
        )

    funcs = {
        "old_apply": lambda apply=apply, h=h, budget=budget: apply(old, h * budget),
        "new_apply": lambda apply=apply, h=h, budget=budget, d=d: apply(_apply(), h * budget * d),
        "reduce": lambda s=s, sink=sink, h=h: _reduce(4)(
            inputs=[s, sink],
            grid=(h * 32, 4, 1),
            threadgroup=(32, 4, 1),
            output_shapes=[(h,)],
            output_dtypes=[mx.int32],
        ),
    }
    for fn in funcs.values():
        for _ in range(10):
            mx.eval(*fn())
    samples = {name: [] for name in funcs}
    for i in range(100):
        for name in list(funcs) if i % 2 == 0 else list(funcs)[::-1]:
            start = time.perf_counter_ns()
            mx.eval(*funcs[name]())
            samples[name].append((time.perf_counter_ns() - start) / 1e6)
    row = dict(
        budget=budget,
        median_ms={n: statistics.median(s) for n, s in samples.items()},
        samples_ms=samples,
    )
    rows.append(row)
    print(budget, row["median_ms"])
Path("/tmp/pyramid_stage_results.json").write_text(json.dumps(rows, indent=2))
