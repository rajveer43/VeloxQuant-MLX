# SnapKV device selection findings

Source baseline: `cfa9c49`. Measured on Apple M4; software/device details and all samples are in [raw results](benchmarks/snapkv_selection.json).

## Implemented behavior

- `quantizers/snapkv.py`: exact threshold membership, earliest-index ties, GPU gathers, and a forceable Python selector. NaNs now explicitly rank as negative infinity; this is a defined invalid-score policy, not legacy Python NaN-sort parity.
- `cache/snapkv_cache.py`: batched selection and gather across B×H, retaining per-head scoring to preserve the original matmul behavior. Multi-token chunks reselect from retained plus incoming rows. Decode remains append-only.
- `metal/_snapkv_select.py` and `src/snapkv_compact_indices.metal`: experimental prefix-rank scatter. MLX still computes threshold, masks and prefix ranks. This is not a fully custom top-k shader.
- `snap_backend="auto"` selects MLX. Explicit `metal` selects the experimental GPU compaction; `reference` forces Python sorting. No tensor host reads occur in the MLX/Metal selection/gather routes.

## Measurements

Median synchronized milliseconds, 1 batch × 8 KV heads, D=128, five warmups and 20 measured samples per route. Fresh first-prefill cache each sample. Budgets are 128 at N=512 and 512 otherwise. Observation window is the default 32.

| N | Stage | Python selector | MLX | Metal |
|---:|---|---:|---:|---:|
| 512 | selection | 0.3834 | 0.6102 | 0.4809 |
| 512 | cache | 1.0282 | 0.4397 | 0.4211 |
| 2048 | selection | 2.1140 | 0.5478 | 0.4511 |
| 2048 | cache | 3.4936 | 0.8356 | 0.8265 |
| 8192 | selection | 8.7523 | 1.1728 | 0.8562 |
| 8192 | cache | 11.8274 | 2.3872 | 2.2330 |

The reference here uses the preserved Python selector with the new batched gather; it is not the entire original pre-change implementation. Complete-cache Metal improvement over MLX is approximately 1–6%, below the predeclared 10% promotion target. MLX remains automatic. Selection-only improvements do not justify promotion. These are synthetic first-prefill measurements, not TTFT or model decode throughput.

## Validation and limits

- 49 targeted quantizer/cache/protocol tests passed, including 1,000 randomized three-group cases per accelerated backend, nonfinite scores, boundary/strided inputs, multi-chunk K/V parity, and 2,000 append-only decode updates.
- Built `veloxquant_mlx-0.81.1-py3-none-any.whl`, installed without dependencies into `/tmp/snapkv-installed`, and executed `/tmp/snapkv_installed_smoke.py` outside the checkout. The installed Metal selector compiled and returned exact expected indices; the cache smoke also passed.
- Targeted changed-code Ruff checks pass. `cache/base.py` has pre-existing lint findings; the only change there is the backend field.
- FP32 scorer inputs retain their precision until output gathering casts to FP16. Empty direct compression returns an empty result; zero head dimension is rejected.
- MLX dependency-floor compatibility, other Apple GPUs, explicit streams/compiled execution, model-level mask correctness, process peak memory, cold compilation, and later-chunk performance remain unmeasured. No production-quality or full-model speedup claim is made.
- Existing singleton-before-first-prefill behavior and cache protocol semantics were not redesigned. The full bookkeeping audit remains separate.
- Metal uses immutable MLX-managed arrays, no shared mutable scratch, no barriers, and int32-bounded indexing. The internal scatter requires exactly k selected entries per group, established by the threshold/tie construction. Arbitrary external selection masks are not a supported public API.

## Reproduction

```sh
.venv/bin/python scripts/snapkv_kernel_bench.py --output docs/benchmarks/snapkv_selection.json
.venv/bin/python -m pytest -q veloxquant_mlx/tests/quantizers/test_snapkv.py veloxquant_mlx/tests/quantizers/test_snapkv_backends.py veloxquant_mlx/tests/cache/test_snapkv_cache.py veloxquant_mlx/tests/cache/test_prefix_cache_reuse.py
```

## Local open-model validation

Executed `scripts/snapkv_model_check.py` on the cached MLX 4-bit Llama-3.2-1B-Instruct snapshot `08231374eeacb049a0eade7922910865b8fce912`. No model download. [Raw model results](benchmarks/snapkv_llama_model_check.json).

All 12 runs completed: plain cache and SnapKV reference/MLX/Metal, with no eviction (budget 512), eviction (budget 64), and chunked prefill (budget 64, chunk 48). The actual repeated-text prompt contained 171 tokens; each run generated 16 greedy tokens. MLX and Metal had zero maximum final-prefill logit error against the SnapKV reference and identical generated tokens in every scenario. Absolute offsets reached 187.

Single-run prefill timings for eviction were reference 198.0 ms, MLX 187.1 ms, Metal 186.1 ms; chunked prefill was 250.2/229.4/228.4 ms respectively. These are functional smoke timings without statistical warmup/control, not evidence of a stable speedup. Decode took approximately 8.8–9.4 ms per step. The manual loop includes scalar token reads and is not a production generation throughput benchmark.

The unchunked eviction output diverged from plain-cache output, equally across all SnapKV backends. This proves backend parity on this workload, not general algorithm quality or mask correctness. The prompt was intentionally repetitive and not a quality evaluation. Peak process memory, broader prompts/models, and warmed repeated timing remain unmeasured.

## Repeated Llama trials

144 additional runs completed: two warmups plus ten measured trials for each backend/scenario, alternating forward/reverse backend order. All runs completed successfully. All 60 measured MLX/Metal runs had zero final-prefill logit error and identical 16-token generation against the corresponding reference. [Raw samples and summary](benchmarks/snapkv_llama_repeated.json).

| Budget / chunk | Backend | Median prefill ms | Prefill min–max ms | Median decode ms/step |
|---|---|---:|---:|---:|
| 512 / 192 | plain | 185.23 | 181.74–186.86 | 10.51 |
| 512 / 192 | reference | 188.72 | 183.03–190.25 | 10.78 |
| 512 / 192 | mlx | 188.51 | 184.12–192.32 | 10.42 |
| 512 / 192 | metal | 188.48 | 183.52–190.80 | 10.61 |
| 64 / 192 | plain | 188.61 | 185.77–190.54 | 10.67 |
| 64 / 192 | reference | 229.31 | 212.22–231.85 | 10.74 |
| 64 / 192 | mlx | 194.60 | 190.85–198.65 | 10.65 |
| 64 / 192 | metal | 196.62 | 193.01–200.62 | 10.76 |
| 64 / 48 | plain | 236.51 | 234.33–238.92 | 10.69 |
| 64 / 48 | reference | 331.76 | 308.21–342.05 | 11.01 |
| 64 / 48 | mlx | 257.88 | 255.93–260.80 | 10.51 |
| 64 / 48 | metal | 259.58 | 251.67–261.56 | 10.61 |

These repeats use the same 171-token prompt and local model as the initial smoke. They broaden timing evidence, not prompt/quality coverage. Ten samples give a coarse p95 (nearest-rank p95 equals the maximum). Backend order alternates, but system load and thermal state are not controlled; small differences must not be interpreted as robust speedups. Auto dispatch remains MLX.

Reproduce with the model snapshot path and `--repeats 10 --warmups 2 --output docs/benchmarks/snapkv_llama_repeated.json` on `scripts/snapkv_model_check.py`.

## Larger open models

Downloaded and tested MLX 4-bit Qwen3-8B and GLM-4-9B-0414: 168 successful runs including warmups, exact accelerated/reference parity in all measured comparisons. See [8–9B results](SNAPKV_8B_9B_MODEL_RESULTS.md) for revisions, timing ranges, and the observed Qwen whole-cache-path slowdown. Metal remains experimental.

## BF16 preservation and fused-kernel follow-up

The Qwen3-8B probe isolated the earlier decode regression: SnapKV forced BF16 model K/V to FP16, causing MLX attention to produce FP32 output. `snap_dtype="auto"` now preserves BF16 when the model supplies BF16 K/V; `snap_dtype="float16"` keeps the legacy route available. A controlled probe measured about 47–56 ms/step for native BF16 versus 68–78 ms/step for forced FP16.

The Metal path now includes threshold/tie compaction and a combined K/V gather. `snap_batched_scoring` is available as an opt-in experiment; it changes matmul batching and must be checked for near-tie parity before defaulting. The default scorer remains per-head for numerical parity.

Post-fix Qwen results are in [raw data](benchmarks/snapkv_qwen8b_after_dtype.json). One warmup and three measured trials per route/scenario passed. MLX/Metal matched the reference's first-token logits and generated tokens in every measured case. The native-dtype change restores decode timing near the plain cache; Metal remains mixed and is still not automatic.

The post-fusion synthetic benchmark ([raw data](benchmarks/snapkv_selection_after_fused.json)) measured Metal selection at 0.25, 0.25 and 0.78 ms versus MLX at 0.39, 0.41 and 1.14 ms for N=512, 2048 and 8192. Complete cache medians were 0.41, 0.80 and 2.20 ms for Metal versus 0.44, 0.85 and 2.32 ms for MLX. This is a microbenchmark, not model throughput; automatic routing remains conservative until more model trials support a stable win.
