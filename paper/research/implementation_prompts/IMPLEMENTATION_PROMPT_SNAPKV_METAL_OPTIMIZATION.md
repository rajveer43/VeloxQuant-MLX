# End-to-end implementation prompt: SnapKV selection and gather on MLX / Apple silicon

Repository: `/Users/rajveerrathod/Work/personal_projects/turboquant_mac_implementation`

Source-inspection baseline: `cfa9c49`, 2026-09-09. This is an engineering specification, not an implementation or a benchmark result. Recheck the checkout before starting. This expands Phase B of [the eviction roadmap](IMPLEMENTATION_PROMPT_MLX_EVICTION_METAL.md) as the next work item after TOVA.

## 1. Copy-ready assignment

Implement and validate SnapKV-adapted prefill selection acceleration end to end. Remove Python reads of score and index tensors, replace per-row construction with device gathers, batch independent heads within a layer, and evaluate a custom Metal selection/compaction implementation against an exact GPU-only MLX baseline. Integrate the winning route into the real cache, configuration, tests, benchmarks, packaging, and documentation.

Preserve the repository's current SnapKV-adapted algorithm. This task does not implement the paper's true-query interception, max-pool smoothing, a decode eviction policy, or a different quantization format. A measured MLX-only winner is a successful outcome; do not enable a slower Metal implementation to satisfy the word “kernel.”

Work autonomously through reversible implementation and validation. Inspect applicable repository instructions and working-tree changes first. Preserve unrelated work. Do not download large models, publish, release, or commit unless separately authorized. Use local model assets when available and state unavailable validation explicitly.

## 2. Establish the source contract before editing

Read these files and record relevant functions and line references in the findings document:

| File | Purpose |
|---|---|
| `veloxquant_mlx/quantizers/snapkv.py` | Scoring, selection, state, gather and byte accounting |
| `veloxquant_mlx/cache/snapkv_cache.py` | First prefill, repeated chunks, decode append, offsets and counters |
| `veloxquant_mlx/cache/base.py` | Configuration, factory, per-layer propagation |
| `veloxquant_mlx/metal/_qfilters_evict.py` and its shaders | Existing threshold compaction, limits and tie logic |
| `veloxquant_mlx/metal/_tova_evict.py` and its shaders | Wrapper, lazy loading and launch conventions |
| `veloxquant_mlx/metal/{__init__,kernels,_warmup}.py` | Exports, capability checks and optional warmup |
| `veloxquant_mlx/tests/quantizers/test_snapkv.py` | Existing numerical expectations |
| `veloxquant_mlx/tests/cache/test_snapkv_cache.py` | Chunking and cache expectations |
| `veloxquant_mlx/tests/cache/test_prefix_cache_reuse.py` | Eviction/cache protocol regressions |
| `benchmark_scripts/benchmark_snapkv.py` | Existing benchmark limitations |
| `scripts/tova_kernel_bench.py` | Reusable benchmark conventions |
| `pyproject.toml`, `docs-site/docs/algorithms/snapkv.md` | Dependency, packaging and user contract |

Also inspect the installed `mlx_lm.models.cache.KVCache` and actual model attention call sites. Read local API documentation/signatures for MLX sorting, partition, cumulative scans, gathers, streams and `mx.fast.metal_kernel`. Verify compatibility with the declared dependency floor (`mlx>=0.18` at this baseline); do not assume installed-version behavior applies to every supported version. Feature-detect acceleration or justify a tested dependency change.

Source-verified starting observations:

- `snap_select_indices` calls `scores.tolist()` and Python stable descending `sorted()`.
- `snapkv_compress` calls `indices.tolist()` again and constructs each K/V output with Python row loops and `mx.stack`.
- `_process_prefill` and `_process_prefill_chunk` loop over batch and heads, invoking that work separately.
- Every `S > 1` update is treated as prefill. Later prefill chunks reselect from retained rows plus incoming rows. This is not necessarily one selection per entire prompt.
- `S == 1` appends without eviction. Retained cache length can therefore exceed the prefill budget during decode.
- `_true_offset` is absolute processed length; `_row_offset` is the base cache write cursor. Base storage capacity may exceed both retained length and returned slice length.
- Q-Filters uses MLX sort to obtain a threshold; its shader is not an existing standalone Metal top-k implementation.
- The current benchmark evaluates only `state.kept_keys`, and its `--heads` argument does not produce a real batched-head workload. It is not sufficient evidence for complete K/V cache speedup.

Confirm these against the live checkout. Separate proven source facts, measured costs, and hypotheses. Do not repeat stale TOVA host-synchronization claims from the older roadmap.

## 3. Exact numerical and state invariants

Use `G = B * H_kv`, `N` for candidate rows, `D` for head dimension, `C` for retained count, `s` for effective sinks and `k = C - s` for dynamic survivors. Do not confuse KV heads with query heads.

For nonempty candidates, preserve:

```text
C = min(max(budget, 1), N)
s = min(max(n_sink, 0), C)
w = min(max(obs_window, 1), N)
K32 = FP32(candidate keys)
Qproxy = last w rows of K32
scores[i] = mean_q softmax(Qproxy @ K32.T / sqrt(D))[q, i]
```

The scorer uses the supplied key precision before FP16 output storage. Do not cast direct FP32 inputs to FP16 before scoring. Later prefill concatenation may combine prior FP16 survivors with higher-precision incoming keys; preserve existing promotion behavior or explicitly isolate a tested change.

Selection must satisfy all of these:

1. Every leading sink row survives, regardless of score.
2. Exactly `k` non-sinks survive, ordered for membership by `(score descending, candidate index ascending)`.
3. Returned indices are unique, ascending in candidate order, length `C`, and `int32`.
4. Both K and V use the identical index map. Gathered values equal the reference after the same FP16 conversion.
5. Scores are the mean of individually normalized attention distributions. Do not replace this with softmax of averaged logits, raw-dot-product ranking, causal proxy attention, or cumulative TOVA/H2O importance.
6. No new protection for the observation window or newest token is introduced.
7. Keys already carry original-position rotations. Do not renumber or re-rotate survivors.

`SnapKVState.kept_indices` is an index into the supplied candidate matrix. On repeated cache chunks this does not by itself encode global original positions. Preserve this public meaning; clarify misleading documentation if needed. Offset correctness alone does not prove causal-mask correctness.

Characterize empty `N`, zero `D`, zero groups, mismatched K/V, invalid ranks and configuration before launching. `snap_select_indices` currently returns an empty range for `N=0`, while compression's empty stack/scoring behavior is not a sound established empty-cache contract. Either retain documented rejection or add a separately tested consistent empty result. Never issue a zero-sized Metal launch or read a nonexistent threshold.

Define the supported numerical domain explicitly. Finite scores, duplicate scores, signed zeros, subnormals, and near ties need exact selection parity. Python sorting of NaNs is not a reliable total order. Choose a documented deterministic device policy for NaNs/infinities or retain a clearly scoped reference contract; ensure invalid values can never produce out-of-range/uninitialized output indices. Do not perform a per-call host finite scan. Test nonfinite behavior separately and do not claim exact legacy NaN parity without proving it.

## 4. Stage A: reference and GPU-only MLX baseline

Preserve an independently executable slow reference selector based on Python stable sorting. Keep reference reads out of accelerated routes. Retain public positional-call compatibility; add backend selection as a keyword-only argument if needed.

Implement fixed-shape, exact device selection first. For `[G,N]` scores, operate on non-sink scores only:

```text
if C == N: return broadcast(arange(N))
if k == 0: return broadcast(arange(s))
t = kth-largest(non_sink_scores), separately per group
above = non_sink_scores > t
equal = non_sink_scores == t
remaining = k - sum(above, axis=-1)
tie_rank = cumsum(equal, axis=-1)       # inclusive, ascending original index
selected_dynamic = above OR (equal AND tie_rank <= remaining)
selected = sink_mask OR selected_dynamic
indices = sort(where(selected, arange(N), sentinel), axis=-1)[:, :C]
```

Use an integer sentinel larger than every valid index and within dtype bounds. The shape of `remaining` must broadcast per group, not across heads. Threshold partition/sort stability is irrelevant to this method because it uses the threshold value and explicitly resolves boundary ties. Verify actual MLX API and dtype behavior rather than assuming a `topk` signature.

For finite inputs, prove `sum(above) <= k`, enough threshold-equal entries exist to fill `remaining`, and exactly `C` entries precede sentinels. The final integer sort guarantees temporal order without dynamic output cardinality. Short-circuit no-selection cases before scoring where semantically safe.

Replace row stacking with `mx.take` or verified `mx.take_along_axis` semantics. Single-head inputs remain supported, preferably through a shared batched implementation. Set `n_kept` from output shape. No `.tolist()`, `.item()`, NumPy conversion, or Python branching on tensor values is allowed in `mlx`, `metal`, or `auto` accelerated execution. Shape/configuration-derived Python integers are allowed.

Measure this baseline before designing additional kernels. It may remove most of the existing overhead without custom code.

## 5. Stage B: implement a bounded, exact Metal candidate

Use the repository's source-body `.metal` / `mx.fast.metal_kernel` pattern with lazy cached factories. Proposed wrapper contract:

```python
snapkv_select_indices(scores, budget, n_sink, *, stream=None)
# scores: [G,N] FP32; output: [G,C] int32, ascending candidate indices

snapkv_gather(keys, values, indices, *, stream=None)
# keys/values: [G,N,D], validated supported dtype
# indices: [G,C] int32; outputs: [G,C,D] FP16
```

These are proposed internal APIs; match existing naming and stream conventions. Keep standalone selection testable independently of attention scoring. Do not call a sort-plus-several-dispatch pipeline “one fused kernel.”

### Preferred first experiment: threshold plus parallel ordered compaction

Reuse the MLX threshold from Stage A. Eliminate the final sentinel sort through a multi-dispatch compaction design:

1. Divide each group's candidate sequence into contiguous tiles. For each tile compute counts of non-sink scores above and equal to the threshold. Use the same numeric comparator everywhere.
2. Scan tile counts in sequence order per group. Obtain the global above-count, each tile's preceding equal-count, and `remaining = k - total_above`. Determine each tile's survivor count including its sink rows and the allowed earliest equal-score rows. Compute exclusive survivor offsets.
3. Within each tile compute local equal ranks and local selected ranks. Write each selected candidate index to `tile_survivor_offset + local_selected_rank`. This yields an ascending `[G,C]` index map without host cardinality reads.
4. Gather K and V with the map, or benchmark a proven alternative that fuses index emission and copy while retaining enough parallelism across `D`.

Use separate dependent dispatches for cross-threadgroup communication. No device-wide spin barriers, races, or assumptions that threadgroups execute in order. If the tile-count scan is itself too large for one group, use a hierarchical scan or route the shape to MLX. Bounds on tiles and scratch must be explicit.

Derive and test the inclusive/exclusive conventions in every scan. Indices from separate tiles must occupy disjoint contiguous output intervals. Handle sinks spanning tiles and threshold ties crossing every tile boundary. An atomic append counter is insufficient because it does not preserve temporal order.

The first version can retain MLX threshold sorting. Name this “Metal compaction with MLX threshold selection.” Profile before replacing the threshold stage itself.

### Alternative: small-shape bitonic selection

Benchmark a power-of-two padded sorting network only for shapes whose score/index records fit documented threadgroup resource limits. Use a total comparator: valid candidates before padding; higher score first; lower candidate index first for equal scores. Exclude sinks from dynamic ranking and add them back explicitly. Padding must lose even when real scores are negative infinity under the supported policy.

After selecting `k` dynamic indices, still restore ascending candidate order. Include this stage and K/V gather in measurements. Specify exact threadgroup memory, barriers and work per stage; one uint32 index plus float score already costs eight bytes per padded candidate before scratch. Do not attempt an entire long-context sort in one threadgroup.

For large shapes, a tiled top-k/global-merge design is optional and requires a proof: retaining each tile's top `min(k,tile_length)` candidates is sufficient because a discarded candidate has at least `k` globally better candidates in its own tile under the same total order. Quantify the candidate explosion when `k` is large. Merge exactly, then restore temporal order. Avoid repeated `k` full-array argmax passes unless a measured tiny-`k` specialization justifies them.

### Gather layout and precision

Start with flattened contiguous output elements:

```text
d = gid % D
row = gid / D
j = row % C
g = row / C
src = indices[g,C-position j]
Kout[g,j,d] = FP16(K[g,src,d])
Vout[g,j,d] = FP16(V[g,src,d])
```

Use conventional array indexing in actual code; the notation above describes the mapping. Guard rounded-up threads and validate index-map shape. Internally produced maps must be proven in range. Do not launch an unchecked low-level gather on arbitrary public index buffers without a defined safety contract.

Contiguous output lanes should read contiguous dimensions of each selected row. Compare scalar mapping against vectorized loads/stores only where alignment, strides and tail guards prove safety. Support odd dimensions through the baseline; do not inherit TOVA/H2O RoPE restrictions. Avoid one thread serially copying an entire large row without measuring the occupancy cost.

### Launch, lifetime and resource checklist

- Use explicit supported device/stream behavior; CPU execution routes to MLX and forced Metal on CPU reports a clear unsupported request.
- Validate ranks, shapes, dtypes, group counts, candidate count, sink/budget clamps, integer multiplication bounds, and supported indexing range before launch.
- Distinguish int32 output-index bounds from uint32 flattened-offset bounds. Use safe wider arithmetic or reject/fallback for products beyond implementation limits.
- All lanes reach threadgroup barriers uniformly. Initialize inactive lanes to neutral values and every valid output exactly once.
- Check installed pipeline thread limits and threadgroup-memory limits. Benchmark a small legal set of group sizes; do not assume 1024 threads is always valid or fast.
- Scratch is per group/tile and allocated through MLX outputs, not unsafe global mutable buffers. No input mutation, accidental aliasing, or untracked temporary lifetime.
- Count row-contiguity conversions, scalar parameter arrays and scratch initialization in complete timings.
- Keep compilation caches bounded by a small set of useful specializations. Do not specialize on arbitrary score values or every sequence length.
- Repeated asynchronous calls and concurrent supported streams must not share writable scratch.

## 6. Stage C: real cache integration

Introduce an internal batched compressor for `[B,H,N,D]` or flattened `[G,N,D]` with independent scores and selections per group. Preserve the existing single-head API and `SnapKVState`. Batch only heads whose inputs already exist within one layer; transformer layers are sequential dependencies.

First prefill: batch scoring, select and gather once across groups. Later chunks: slice prior storage to `_row_offset`, concatenate retained and incoming tensors along the candidate axis, then run batched compression. Never include padded capacity rows. Update counters using Python-known shapes rather than scanning output arrays.

Preserve `_true_offset += incoming_S` exactly once and `_row_offset == retained rows` after replacement. Maintain the `_in_base` try/finally discipline or replace it only with equally tested protocol handling. Public state, `size`, masks, non-trimmability, state restore and prefix reuse must follow their actual supported contracts. Unsupported cache reuse must fail explicitly rather than silently losing absolute positions.

Test `S=1` followed by `S>1`, repeated singleton updates, empty update, and multiple prefill chunks. The initial singleton-then-prefill case may expose an existing discrepancy because `_prefill_done` remains false. Characterize and isolate correctness fixes; do not silently call changed behavior exact performance parity.

Inspect the actual attention integration during a compressed multi-token update. Does attention consume the full incoming chunk or only its selected subset? Does its mask use original positions, retained positions, or an expected uncompressed length? Compare to the reference backend and a small actual attention call. Existing mask bugs must be reported and resolved or clearly gate supported use; an optimized selector does not prove the overall cache algorithm correct.

Keep decode's append-only behavior. Any default decode latency change should be measurement noise or reduced incidental overhead, not a claimed direct top-k improvement. Do not add a ring buffer or quantized cache format in this task.

Add `snap_backend` with `auto`, `mlx`, `metal`, `reference` if consistent with repository conventions. Verify config factory and per-layer propagation. `auto` uses measured supported routes; forced `metal` exposes unsupported shape/capability or shader errors clearly. Catch known capability absence only; broad exception swallowing hides kernel defects. Preserve optional/lazy Metal imports on unsupported hosts.

## 7. Correctness validation required before promotion

Separate selection parity from scorer parity. Identical supplied scores must produce exact identical indices. Batched scoring can change floating reduction order; score allclose alone is insufficient if the changed ranking selects different rows. Test exact selected membership for adversarial near ties; retain a compatible scorer or constrain/document any experimental numerical change.

Use deterministic Python reference tests and at least 1,000 randomized selection cases spanning:

- `N=0,1,2`, tile boundaries ±1, non-power-of-two lengths and representative long lengths.
- Budgets negative/zero, one, below sinks, `N-1`, `N`, above `N`, and shape-limit boundaries.
- Sinks negative/zero/one, equal to budget, spanning tiles, and above budget/length according to existing clamping.
- Ascending, descending, all-equal, few distinct values, boundary ties, negative values, signed zero, subnormal and near-equal finite FP32 scores.
- Distinct per-group winners, `B>1`, multiple heads, and independent batches.
- Nonfinite values under the stated policy, with no invalid writes even when every candidate is invalid.
- Noncontiguous inputs, supported dtypes, unsupported dtypes, odd `D`, and integer/resource boundary rejection.

Use independent K and V fingerprints representable exactly in FP16; encode larger token IDs across multiple small components rather than casting large integers to FP16. Assert every retained K/V pair belongs to the same original token and unselected rows do not appear. Test multi-chunk sequences against a reference that tracks global token lineage separately from candidate indices.

Run cache sequences beyond ten times the budget, mixing chunk sizes then thousands of singleton appends. Decode memory is expected to grow under this policy: do not incorrectly demand a bounded plateau. For repeated recompressed prefill or repeated fixed-shape independent selection, test released allocations and graph retention separately; synchronize and account for allocator caching before calling growth a leak.

Check no accelerated source path contains tensor-to-host reads; use profiling/instrumentation where available rather than relying only on textual searches. Test lazy graph construction, normal evaluation, supported stream use, and supported compiled execution. Do not promise `mx.compile` compatibility without execution.

Suggested tests: new `tests/metal/test_snapkv_select.py` and `tests/quantizers/test_snapkv_backends.py`, plus extensions to existing SnapKV quantizer/cache tests. Run relevant cache offset/mask/prefix regressions. If shared Q-Filters/TOVA code changes, run those consumers' tests too. Avoid generalizing primitives until at least two consumers actually need identical semantics.

## 8. Performance and memory experiment design

Create `scripts/snapkv_kernel_bench.py` with selectable backend/stage, reproducible seeds, JSON output, and hardware/software metadata. Fix or clearly separate the old benchmark. Evaluate all K/V/index outputs, not keys alone. Exclude coverage diagnostics and host correctness conversion from timed regions.

Compare the preserved Python reference, strongest exact MLX implementation, forced Metal candidate and automatic integrated cache. Measure:

1. Scoring alone.
2. Selection from already materialized identical scores.
3. Gather from already materialized identical indices.
4. Selection plus gather, including threshold, scans, scratch and temporal ordering.
5. Complete compression including scorer, casts and contiguity conversions.
6. First and subsequent `update_and_fetch`, including base-cache storage copies and counters.
7. Model prefill/TTFT and decode separately where local assets permit.

Warm compilation separately and report cold-start cost. Pre-evaluate common inputs for component timing, construct fresh operation graphs per repeat, synchronize every measured output, and do not time reevaluation of an already materialized result as a kernel execution. Also measure realistic lazy chunk chains. Run backends in alternating order with repeated trials; report median, p95, spread, repeat count and variation. Python graph-building time and synchronized wall time are different measurements; device time requires actual profiling support.

Primary shapes: `B=1`, `H_kv=8`, `D=128`, `N=512,2048,8192`, budget `128,512,2048` where applicable, observation window `16,32,64`. Add `G=1,4,16,32`, `B=2`, `D=64,256`, odd correctness shapes, `N=32768` only within memory limits, and no-eviction / sinks-only cases. Use a focused matrix first rather than an unbounded Cartesian product. Include different budget fractions and tie-heavy scores for performance sensitivity.

For chunked prefill, record incoming chunk length, prior retained count, total absolute tokens and candidate length separately. Selection runs at most once per multi-token update per group in the unbatched reference, not once per decode token. Report layer counts only for real model measurements.

Memory model, labeled as logical estimates:

```text
FP32 scores:                     4 * G * N bytes
one FP32 attention-sized tensor: 4 * G * w * N bytes
int32 selected indices:          4 * G * C bytes
FP16 K/V output storage:         4 * G * C * D bytes
FP16 K/V gather read + write:    8 * G * C * D bytes (minimum)
```

For FP32 input gather, minimum K/V reads plus FP16 writes are `12 * G * C * D` bytes. Threshold sorting, masks, scans, casts, concatenation and base storage add traffic and allocations. For example `G=8,w=32,N=32768` makes a single FP32 attention-sized tensor 32 MiB; logits and softmax outputs may overlap in lifetime. Inspect actual liveness rather than assuming only one allocation.

Measure peak MLX allocation and process memory separately when APIs are available; record allocator-cache treatment, baseline memory, and whether outputs remain live. Do not equate theoretical retained-cache savings with peak process savings. Unified memory removes a discrete PCIe-transfer model, not synchronization or memory-traffic costs.

If scorer memory becomes limiting, first evaluate bounded groups of heads as a throughput/memory tradeoff. A streamed/online-softmax scorer is a separate optional phase because its reduction order can change selection. Do not hide a scoring algorithm rewrite inside the selection optimization.

For actual inference, use identical local model, prompts, chunk size, output length, precision and sampling settings. Record model identifier and revision when known. Compare reference and accelerated SnapKV for implementation parity. Full-cache comparisons measure the eviction approximation as well as implementation changes; report logit error/token agreement separately. Do not claim compression-plus-eviction composition works without tracing an actual supported integration.

Estimate the opportunity before enabling Metal. If selection/gather is fraction `f` of prefill latency and its speedup is `r`, the optimistic total speedup is `1 / ((1-f) + f/r)`. Use measured `f`; do not invent it. No model availability means end-to-end speedup remains unverified.

## 9. Promotion, rollout and completion gates

Predeclare a promotion target: at least 10% median complete-cache improvement over the strongest equivalent MLX route, exceeding run-to-run variability, on representative routed shapes, with no material p95 or peak-memory regression. This is an acceptance target, not a predicted result. Small-shape wins alone do not justify routing long-context shapes to the same kernel.

If Metal loses, keep MLX automatic, retain Metal only as an explicitly experimental measured candidate or remove it. Use simple shape/resource dispatch rules supported by data; no expensive per-call autotuning. Backend overrides make rollback immediate. CPU and unsupported-device behavior remain usable through MLX. Document remaining sync points and supported numerical/resource limits.

Proposed changed files, subject to the measured winning design:

- `veloxquant_mlx/quantizers/snapkv.py`, `cache/snapkv_cache.py`, `cache/base.py`.
- `veloxquant_mlx/metal/_snapkv_select.py` and only the necessary `src/snapkv_*.metal` sources.
- Metal lazy exports/facade; warmup only if useful and capability-safe.
- SnapKV quantizer, cache and Metal tests; focused protocol regressions.
- `scripts/snapkv_kernel_bench.py`, optional correction to the existing benchmark.
- `docs/SNAPKV_METAL_FINDINGS.md`, raw benchmark JSON, and algorithm documentation.

Avoid unrelated TOVA, H2O, CAM, PyramidKV, attention, quantization and model changes. Any essential shared fix must have independent rationale and regression evidence. Do not broaden this phase into the repository-wide bookkeeping audit.

Run targeted tests, repository lint/format checks and diff validation. Build a wheel, inspect packaged shader paths, install into an isolated location, and execute an actual installed-artifact test from outside the checkout. Verify imported module paths and test execution count. Running a Python file that only defines test functions is not a smoke test. Exercise both real selection and real cache routing from the installed package; dependency absence must be reported honestly.

Final findings must contain: source baseline; invariants; implemented design; exact backend routes; supported versions/shapes/dtypes; shader resource/scratch bounds; selection and cache parity results; complete benchmark tables with raw data; cold compilation and memory results; model validation or its absence; remaining risks; and reproduction commands. Explain any existing bug separately from the performance change.

Completion requires integrated removal of both host roundtrips, exact supported-domain selection and aligned gathers, tested batched/chunked cache behavior, a measured automatic-backend decision, packaging verification, and truthful documentation. Do not stop after creating shaders, showing isolated top-k latency, or reporting theoretical speedup.
