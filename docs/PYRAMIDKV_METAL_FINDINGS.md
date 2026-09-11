# PyramidKV Metal regression investigation

The original apply kernel assigned one full row to each thread. At D=128,
neighboring SIMD lanes accessed FP16 elements 256 bytes apart. Changing the
mapping to one element per thread makes neighboring lanes copy adjacent
dimensions; one lane per row copies its FP32 score. The reduction and its
two-dispatch dependency remain intact.

An isolated same-session A/B on Apple M4 measured these synchronized host wall
medians (100 trials, 10 warmups, BH=8, D=128):

| Budget | Original apply ms | Elementwise apply ms | Reduction alone ms |
|---|---:|---:|---:|
| 4096 | 1.520 | 0.668 | 0.200 |
| 8192 | 2.933 | 1.013 | 0.245 |

This isolates the apply mapping as a major cause of the regression. No GPU
hardware counters were collected; transaction efficiency is an interpretation
of the mapping and timings, not a measured counter. Isolated synchronized
stage latencies must not be summed to predict combined latency.

Across three fresh combined-primitive runs, budget 4096 Metal medians were
0.692–0.726 ms versus MLX 0.819–0.828 ms; budget 8192 Metal was 1.033–1.042 ms
versus MLX 1.276–1.288 ms. The large-budget regression is reversed on this
machine. Budget 2048 showed substantial run-to-run timing variation, retained
in the raw data. These compare resident finite random scores, not full scoring
or admission-policy costs. The MLX baseline uses an eligible score slice for
argmin, avoiding the former protected-score allocation.

## Correctness and integration

Sinks are now excluded explicitly from reduction, fixing their selection when
all eligible scores were infinity. NaNs rank as positive infinity; ties resolve
to the earliest eligible row. This is a documented invalid-input policy, not
a claim of equivalence to MLX NaN ordering. Wrapper checks now reject zero BH/D
and uint32 indexing overflow.

`KVCacheConfig(pyramid_backend="metal")` now invokes the kernel from the real
cache update path. `mlx` uses device argmin/gather; `auto` selects MLX.
The existing `reference` default is retained. Accelerated paths now batch append
and selection/compaction across heads. Per-head GEMV scoring is preserved to
avoid numerical drift near ties; token steps remain sequential, with graph
materialization every 32 steps. FP16 storage, budget schedules, and existing
offset behavior are unchanged. Existing offset/mask semantics have
not been certified for general long-context generation by this change.

191 tests passed: kernel ties, infinity/NaN behavior, odd dimensions,
noncontiguous inputs, explicit stream, invalid dimensions, update-history
parity, cache dispatch/chunk parity, six-head update histories over 83 tokens,
and existing quantizer/cache/state-write-through suites.

## Real-model probes

Local Qwen3-8B-4bit and GLM-4-9B-0414-4bit were run with a repeated short
prompt capped at 64 tokens, average layer budget 16, and four greedy decode
steps. Each backend ran twice in opposite order, with fresh caches. These are
short parity probes without separate warmup trials or quality evaluation.

All accelerated runs had zero final-prefill logit error and identical greedy
tokens against the reference. Metal recorded 13,248 calls per Qwen run and
3,680 per GLM run, proving the integrated kernel executed.

Second-run decode wall times were approximately Qwen: reference 166.9,
MLX 83.2, Metal 82.0 ms/step; GLM: reference 90.5, MLX 60.8, Metal 61.9.
The reduction in host synchronization benefits both accelerated backends.
These runs do not demonstrate a reliable Metal-over-MLX model speedup.

## Reproduction and remaining work

Run from the repository using `.venv/bin/python`. GPU execution requires an
execution context with Metal access; sandbox initialization failure is not
evidence that the Mac lacks a GPU.

```
.venv/bin/python scripts/pyramidkv_kernel_bench.py --output /tmp/pyramid.json
.venv/bin/python scripts/pyramidkv_stage_probe.py
.venv/bin/python scripts/pyramidkv_model_probe.py LOCAL_MODEL /tmp/model.json --chunk 64 --prompt-tokens 128 --budget 64 --decode-steps 16 --warmups 1 --repeats 3
```

Raw data are in `docs/benchmarks/pyramidkv_elementwise*.json`,
`pyramid_stage_results.json`, `pyramid_qwen_probe.json`, and
`pyramid_glm_probe.json`. Model JSON records exact snapshot paths.

## Batched-path warmed model results

After integration, both local models were evaluated with 128 prompt tokens,
64-token prefill chunks, average layer budget 64, and 16 greedy decode steps.
One warmup and three measured trials per backend used fresh caches and
alternating backend order. Median measured host-wall times:

| Model | Backend | Prefill ms | Decode ms/step |
|---|---|---:|---:|
| Qwen3-8B | reference | 8131.4 | 173.64 |
| Qwen3-8B | MLX | 3346.7 | 88.96 |
| Qwen3-8B | Metal | 3382.6 | 86.36 |
| GLM-4-9B | reference | 3676.2 | 92.16 |
| GLM-4-9B | MLX | 2697.2 | 70.72 |
| GLM-4-9B | Metal | 2387.6 | 71.51 |

Every run had zero final-prefill logit error and identical generated tokens
against reference. Metal ran 2880 kernel calls per Qwen run and 3200 per GLM
run. Qwen decode improved about 3% over MLX while prefill regressed about 1%;
GLM prefill improved about 11% while decode regressed about 1%. Three measured
trials on one prompt are limited evidence, not a general throughput claim.
Raw measurements: `benchmarks/pyramid_qwen_warmed.json` and
`benchmarks/pyramid_glm_warmed.json`.

Exact local checkpoint revisions for these runs:

- `mlx-community/Qwen3-8B-4bit`: `545dc4251c05440727734bcd94334791f6ab0192`.
- `mlx-community/GLM-4-9B-0414-4bit`: `fecf35efe04b11a7e11ae434bf30693ac4c38e2c`.

The benchmark page's latest table uses medians of the three records with
`warmup: false` per backend. Older short probes are retained as historical
evidence and are not combined with these warmed measurements.

The wheel built successfully using isolated build dependencies, and both
packaged PyramidKV Metal sources match the working tree byte-for-byte.
Focused Ruff checks and diff whitespace checks pass.

Further experiments, rather than prerequisites for this regression fix:
parameter sweeps, rotating-buffer measurements, broader natural-language
quality evaluations, and native-dtype optimization. The conservative backend
default remains appropriate because model-level MLX versus Metal results are
mixed. Full offset/mask semantics and true-query PyramidKV fidelity are outside
this backend acceleration change.
