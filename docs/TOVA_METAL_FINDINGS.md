# TOVA eviction: MLX and Metal implementation

Measured 2026-09-08 on Apple M4, MLX 0.32.0 and mlx-lm 0.31.3. These are
synthetic cache timings, not pretrained-model generation throughput or quality
measurements. The implementation preserves this repository's key-as-query TOVA
adaptation.

## What changed

TOVA now has an MLX argmin/gather path that keeps the eviction decision on the
device, plus two Metal dispatches for sink-protected argmin and copy-only K/V
compaction. Scoring still uses MLX FP32 matmul and softmax; the Metal pair does
not include scoring or append. There is no RoPE remapping, cumulative score
history, or new position buffer.

The cache batches all batch/KV-head groups together. It absorbs the below-budget
prefix in bulk, then processes tokens sequentially, materializing every 32
evictions within a call to bound lazy graph growth. It stores retained buffers
directly instead of copying them into a fresh padded append buffer each step.
Per-head compatibility states are views into those buffers. `size()` and `.state`
report actual retained rows; `offset` continues to count absolute incoming tokens.

Configure `KVCacheConfig(tova_backend=...)`, or pass `backend=...` to
`tova_update`:

| Backend | Behavior |
|---|---|
| `auto` (default) | Metal for multi-token updates on an available default GPU; MLX for single-token updates and CPU/unavailable-Metal builds. |
| `mlx` | GPU-only argmin/gather on GPU, ordinary MLX execution on CPU. |
| `metal` | Force the two custom eviction kernels; raises if the default GPU/custom-kernel capability is unavailable. |
| `reference` | Original sequential Python-driven algorithm, retained for parity and benchmarking. |

Zero/negative-budget and negative-sink legacy cases retain the reference path;
they are outside the accelerated domain. This preserves the existing zero-budget
bootstrap inconsistency rather than silently fixing it inside a performance patch.
Unexpected compilation failures are not swallowed by an automatic retry.

## Measurements

The clean sweep used seed 42, 8 warmup evaluations, and 30 measured samples per
case, with synchronized outputs and no concurrent test suite. `D=128`, batch 1.
Raw samples, p95, first-call times, versions, and hardware metadata are in
[`benchmarks/tova_m4.json`](benchmarks/tova_m4.json).

| KV heads | Budget | Single cache update, reference / MLX / Metal (ms) | 64 sequential evictions, MLX / Metal (ms) |
|---:|---:|---:|---:|
| 1 | 128 | 0.448 / 0.269 / 0.254 | 3.917 / 2.415 |
| 1 | 512 | 0.489 / 0.282 / 0.265 | 4.427 / 2.674 |
| 1 | 2048 | 0.598 / 0.334 / 0.297 | 6.527 / 4.850 |
| 8 | 128 | 2.411 / 0.334 / 0.297 | 4.936 / 3.537 |
| 8 | 512 | 2.604 / 0.468 / 0.448 | 13.852 / 10.105 |
| 8 | 2048 | 3.628 / 1.158 / 1.046 | 49.020 / 36.075 |

The 64-token measurement includes scoring, append, eviction and compaction on an
already-full cache, in one sequential multi-token update with graph flushes. It
is not 64 model decode steps. The reference cache measurement uses the same new
cache bookkeeping as the other backends and the original per-head numerical
algorithm; it does not reproduce the old padded-buffer overhead.

Metal reduced multi-token elapsed time by approximately 26–40% against the
GPU-only MLX baseline across this sweep. Single-token results are less decisive
and showed variability in an earlier exploratory run, so automatic decode uses
MLX. Single-token MLX still removes the original host decisions and batches
heads: at eight heads/budget 512 it measured 0.468 ms versus 2.604 ms for the
reference algorithm in the clean run. These observations justify the initial
multi-token dispatch policy; they do not establish optimal dispatch on other
Apple GPU generations.

Additional reverse-order (Metal then MLX) sweeps used eight heads, budgets 128
and 512, eight warmups and 20 samples. At D=64, Metal reduced multi-token time
from 4.328/7.390 ms to 3.166/5.217 ms. At D=256 it reduced 7.818/26.501 ms to
7.040/22.602 ms. Some single-token cases regressed, further supporting the
conservative single-token MLX default. Samples are in
[`tova_m4_d64.json`](benchmarks/tova_m4_d64.json) and
[`tova_m4_d256.json`](benchmarks/tova_m4_d256.json).

The script reports synchronized Python/MLX wall time. First-call timing is not
isolated shader compilation time, and the tests do not collect GPU hardware
occupancy counters. Thermal/power state is uncontrolled. Treat absolute numbers
and small differences accordingly.

## Correctness and limits

Tests cover exact retained identities and copied values, ties across SIMD groups
and grid-stride iterations, zero/underflow scores, first/interior/newest eviction,
odd dimensions, noncontiguous arrays, nondefault GPU streams, compiled selection,
single-candidate empty output, large budgets, direct FP32 proxy inputs, batched
head routing, CPU fallback, cache state/offset/bytes, and 2,048-token graph growth.

The installed mlx-lm Llama model implementation was exercised with tiny random
FP16 weights, two layers and GQA (four query heads/two KV heads). Reference, MLX
and Metal produced identical logits through four-token prefill and 128 further
decode steps. This validates model plumbing, not pretrained-model quality or
throughput. No local MLX-format pretrained weights were found in the inspected
standard model/cache locations; no model was downloaded.

The public Metal primitive requires matching FP16 `[BH,N,D]` K/V, FP32 `[BH,N]`
weights, and at least one eligible non-sink. Output is `[BH,N-1,D]`. It supports
odd D and bounded uint32 indexing. Available SIMD-group counts are 1, 2, 4, 8.
All-infinite eligible scores select the first eligible row. NaNs are ignored;
all-NaN eligible scores also select the first eligible row, preventing invalid
index propagation. Nonfinite scoring is outside the finite-score parity claim;
MLX argmin has different/unspecified NaN ordering.

K/V-only state restoration uses retained count as its absolute-offset estimate,
matching the information available in that interface. Exact mid-history restore
requires separate absolute-position metadata and remains unsupported. Existing
cache-level causal/chunk-prefill and explicit-mask limitations are unchanged:
there is no model interception or per-query retained-position mask redesign.
Changing the eviction algorithm, the zero-budget behavior, or the mask model is
outside this implementation.

## Reproduce

```sh
.venv/bin/python scripts/tova_kernel_bench.py --output /tmp/tova.json --repeats 30 --warmup 8
.venv/bin/python -m pytest -q veloxquant_mlx/tests/metal/test_tova_evict.py veloxquant_mlx/tests/quantizers/test_tova_backends.py veloxquant_mlx/tests/quantizers/test_tova.py veloxquant_mlx/tests/cache/test_tova_cache.py
.venv/bin/python -m build --wheel --outdir /tmp/tova-wheel
```

Run GPU commands in a session with Metal device access. The environment's
`pytest` executable has a stale interpreter path, so use `python -m pytest`.
Package data already includes `metal/src/*.metal`; no dependency-floor or build
configuration change is required. Only MLX 0.32.0 was executed in this validation;
older versions use capability detection but were not tested here.

Repository-wide Ruff formatting passes. TOVA implementation/test files pass Ruff
lint. `cache/base.py` retains the same 50 pre-existing lint findings as HEAD;
the only change there is the backend configuration field.

Validation completed: 149 tests passed across the TOVA suites, related
KVzip/MorphKV parity suites, sliding-window/prefix-cache regressions, and an
installed-wheel GPU smoke test. The wheel was built, installed into a temporary
target, and its TOVA sources checked byte-for-byte against the final checkout.
Both shaders loaded and executed from the installed artifact. No release or
publication was performed.
