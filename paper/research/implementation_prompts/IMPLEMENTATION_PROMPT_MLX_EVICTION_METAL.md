# End-to-end implementation prompt: MLX eviction kernels on Apple silicon

Repository: `/Users/rajveerrathod/Work/personal_projects/turboquant_mac_implementation`

Source review baseline: `e0d1d3f`, 2026-09-08. This document is an implementation brief based on source inspection, not a claim that new kernels or performance measurements already exist. Recheck the checkout before implementing.

## Copy-ready task

You are implementing production-quality KV-cache eviction acceleration for VeloxQuant-MLX on Apple silicon. Complete the code, runtime integration, correctness validation, reproducible benchmarks, packaging verification, and documentation. Start with TOVA, then remove SnapKV's host roundtrips. Extend only the genuinely reusable parts to other eviction methods, using the phase gates below.

Optimize measured cache and generation latency while preserving the repository's current research-adapted algorithms. Keep an executable MLX reference and a supported fallback. A standalone shader that the real cache never calls is incomplete. A custom kernel that loses to an equivalent GPU-only MLX implementation should remain experimental or be removed from automatic dispatch, with the result documented.

Use the existing repository environment and conventions. Inspect applicable `AGENTS.md` files and the working tree first; preserve unrelated changes. Do not publish, release, or download large models as part of routine kernel development. Use locally available models for integration measurements and record any unavailable validation explicitly.

## 1. Establish the actual starting point

Read these implementation files and their associated tests before editing:

- `veloxquant_mlx/metal/_h2o_evict.py`, `src/h2o_evict_reduce.metal`, `src/h2o_evict_apply.metal`.
- `veloxquant_mlx/metal/_keyformer_evict.py` and its reduce/apply sources for another existing specialization.
- `veloxquant_mlx/metal/_qfilters_evict.py`, `src/qfilters_score.metal`, `src/qfilters_evict_apply.metal`.
- `veloxquant_mlx/metal/__init__.py`, `kernels.py`, `_warmup.py`, and package-data configuration in `pyproject.toml`.
- `veloxquant_mlx/quantizers/{tova,snapkv,pyramidkv,chunkkv,cam,kvzip,streaming_llm,h2o}.py`.
- The corresponding files under `veloxquant_mlx/cache/`, plus `chunkkv_coordinator.py`, cache configuration/factory code, and relevant cache protocol tests.
- `veloxquant_mlx/tests/metal/test_h2o_evict.py`, the matching quantizer/cache tests, and `test_chunkkv_coordinator.py`.
- `benchmark_scripts/benchmark_{tova,snapkv,pyramidkv,chunkkv,cam,kvzip,streaming_llm}.py`, `scripts/kv_kernel_roofline_bench.py`, `docs/KV_KERNEL_ROOFLINE_FINDINGS.md`, and `blogs/prefill-roofline.md`.

The H2O wrapper refers to `paper/research/H2O_METAL_KERNEL_TECH_SPEC.md`, but that file was not present in this checkout during this review. Treat executable code and tests as the available reference; do not invent the missing specification.

Produce a compact audit identifying synchronization sites, scoring cost, dispatches, copies, state contracts, and candidate batching boundaries. Distinguish findings from hypotheses. These source-verified corrections must inform the implementation:

| Method | Actual behavior in this checkout | Reuse boundary |
|---|---|---|
| TOVA | Append, recompute current-step softmax over all candidates including the incoming key, sink-protected argmin, remove one row. `.item()` and a Python index list force the decision onto the host. | H2O's reduction structure is useful; use a copy-only apply. TOVA has no persistent score or position arrays. |
| SnapKV | Observation-window softmax followed by mean pooling, CPU stable descending ranking, temporal-order gather. `snapkv_compress` performs another `.tolist()` and per-row stacking. Cache compression reruns on every `S > 1` prefill chunk over retained plus incoming rows; `S == 1` appends. | Exact top-k membership plus ordered gather; not a single-argmin port. |
| PyramidKV | Cumulative attention over existing rows before append, bootstrap score one, appended score zero, per-layer budgets, single-row eviction, no RoPE remap. | Reduce and copy, including score compaction; preserve its own scoring policy. |
| ChunkKV | Attention-mass or fixed key-norm scoring; mean score per non-sink chunk; remove whole chunks, possibly a short tail; optional leader/follower index reuse. | Chunk reduction and range compaction with variable lengths; needs a separate layout design. |
| CAM | Cumulative scoring, loser argmin, optional cosine-nearest eligible survivor, optional seeded merge gate, blend K/V as configured, accumulate merged score, compact. | Drop mode is simple. Merge modes require additional dependent work and exact RNG bookkeeping. |
| KVzip | `latest` uses the newest stored key as query; `context` builds all-key attention, normalizes each probe row, then takes max over probes. | Shared selection/apply is useful, but context scoring is a separate quadratic computation. |
| StreamingLLM | Positional sink/recent buffers; current implementation still loops through incoming tokens to construct Python lists. | First vectorize the MLX bookkeeping and slicing. Benchmark before adding Metal. |

Do not describe Q-Filters as an existing standalone Metal top-k: its score shader emits scores and its wrapper still uses MLX selection operations. Inspect its tie handling before borrowing its compaction design.

H2O's two eviction kernels do not perform the running attention-sum update themselves. That update happens upstream. Porting TOVA therefore means changing the scorer and state wiring as well as dropping H2O's score/position outputs and RoPE work. Existing H2O also has grace/decay behavior that must not leak into PyramidKV, CAM, or TOVA through reuse.

## 2. MLX and Apple silicon requirements

Use `mx.fast.metal_kernel` and the existing source-file/JIT wrapper pattern. Keep `.metal` files as source bodies loaded from `metal/src`; do not introduce a separate native extension or external shader compiler for this work. Preserve lazy imports and capability checks.

Check installed MLX and mlx-lm versions and actual API availability. `pyproject.toml` currently declares `mlx>=0.18` and `mlx-lm>=0.31.3`; current online documentation is not proof that every declared dependency version supports a new API. Feature-detect optional acceleration, or explicitly justify and validate a dependency-floor change. Verify sorting, indexing, stream, compilation, and custom-kernel behavior against the installed version.

The MLX custom-kernel API supports explicit input/output names, output shapes and dtypes, launch geometry, specialization, and row-contiguity handling. Account for any copies caused by `ensure_row_contiguous=True` in complete-path measurements. Use explicit GPU stream/device handling consistently with the installed API; avoid silently moving CPU requests to GPU. See the [MLX custom-kernel guide](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html) and [API reference](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html).

Apply these engineering constraints:

- Use FP32 scoring/reduction with the existing FP16 cache-storage contract. Accept caller dtypes only where existing APIs accept them and preserve the timing of casts. Do not expand this task into a BF16 cache redesign.
- Keep eviction indices, ranks, masks, and gather maps as GPU arrays. The accelerated hot path must not call `.item()`, `.tolist()`, NumPy conversion, or Python truth conversion on device-derived values. Python integers from shapes/configuration are fine.
- Validate dimensions, matching shapes, supported dtypes, sink/budget ranges, and specialization bounds before launching. Handle empty outputs without a zero-sized Metal dispatch. Reject unsupported inputs or use an explicit correct fallback.
- Output shapes must be known at dispatch time. Do not pretend a device scalar can directly determine a Python output shape without synchronization.
- Use immutable outputs and MLX dependency tracking. Do not mutate input arrays from custom shaders or depend on accidental aliasing/donation.
- Compile/cache kernel factories and a bounded set of useful specializations. Do not create a new entry point keyed by every token count or every changing scalar value without measuring compilation/cache growth.
- Start with the existing Apple-oriented 32-lane SIMD reduction pattern. Tune a small legal set of threadgroup sizes on the actual device; a wrapper accepting `nsg <= 32` is not proof that every pipeline can launch 1,024 threads efficiently or legally.
- Keep all threads participating in a threadgroup barrier on convergent control flow. A threadgroup barrier does not synchronize separate threadgroups. Use separate dependent dispatches for global reduce/apply handoff. Apple explicitly warns that incorrect synchronization can cause memory corruption or wrong results in its [Apple silicon porting guidance](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon).
- Write each output element exactly once, guard rounded-up grids, and initialize every reduction participant and every output. Use safe index arithmetic for the supported shape range.
- Scope reductions and shared memory to actual need. For copy-only apply, compare contiguous-element mapping against H2O's one-thread-per-row serial loop over `D`.
- Do not copy H2O's even-`D` restriction into TOVA: that restriction exists for RoPE pairing, which TOVA does not perform.
- Removing host reads can create a long unevaluated graph. Preserve bounded prefill materialization, informed by H2O's `_EVAL_FLUSH_INTERVAL = 32`, and measure a suitable interval. Materialization at chunk boundaries is different from reading each eviction decision onto the CPU.

## 3. Phase A — ship TOVA end to end

### A1. Preserve the exact reference

Retain a forceable reference path before changing it. For each incoming token, append its FP16 K/V, compute the existing query-proxy softmax, mask leading sinks, choose the earliest minimum, and compact survivors in temporal order. The newest token is eligible; add neither grace protection nor cumulative scores.

The quantizer uses the incoming key cast to FP32 as proxy, while the stored candidate keys are rounded to FP16 before scoring. The cache currently converts inputs to FP16 before invoking the quantizer. Preserve this distinction when supporting direct FP32 quantizer callers; do not silently substitute the rounded stored key for an unrounded proxy.

Preserve `TovaState`'s public contract: keys, values, sink count, budget. Do not add persistent scores or position buffers merely to match H2O. Preserve `_true_offset`, externally visible `offset`, state access, byte accounting, and non-trimmability in `TOVAKVCache`.

Characterize edge cases first. In particular, initialization accepts `n_sink=0, budget=0`, but bootstrap currently inserts one row unconditionally. This is an existing contract inconsistency, not a reason to launch an invalid kernel. Pin current behavior for the performance change or make a separately documented correctness fix with its own test; do not silently change it while claiming exact parity. Inspect empty updates and invalid negative configuration similarly.

### A2. Build a GPU-only MLX baseline

Keep the existing scorer, but leave `mx.argmin` on the GPU. For `N` candidate rows and `e` the device argmin, construct:

```text
j = arange(N - 1)
source = j + (j >= e)
K_out = gather(K_candidates, source)
V_out = gather(V_candidates, source)
```

For `[BH,N,D]`, use a batched gather with a separate `e` per group. Verify installed MLX indexing semantics. This baseline removes host synchronization and Python index construction without custom Metal, and is the fair competitor for the new kernels.

### A3. Implement the reduce/apply pair

Proposed files:

- `veloxquant_mlx/metal/_tova_evict.py`
- `veloxquant_mlx/metal/src/tova_evict_reduce.metal`
- `veloxquant_mlx/metal/src/tova_evict_apply.metal`
- `veloxquant_mlx/tests/metal/test_tova_evict.py`

The file split may change if a small shared primitive proves cleaner. Do not introduce a many-policy framework before validating this first consumer.

Start with a wrapper contract equivalent to:

```text
tova_fused_evict(
    keys_mid:   [BH, N, D] fp16,
    values_mid: [BH, N, D] fp16,
    weights:    [BH, N]    fp32,
    n_sink: int,
    ...validated launch/stream options...
) -> (keys_out: [BH,N-1,D] fp16, values_out: [BH,N-1,D] fp16)
```

This is two dispatches for **selection and compaction**, not two dispatches for the complete update: MLX scoring and concatenation remain additional work. Name and document the API accurately.

Reduction: one threadgroup per `BH` initially, grid-stride over `N`, reduce `(weight,index)` with earliest-index tie breaking, and exclude sink rows. Local and inter-lane comparisons must use the same comparator. The existing H2O local accumulator only accepts `v < INFINITY`; blindly copying it can leave `UINT_MAX` for all-infinite/invalid inputs. Define finite-score fast-path expectations, test infinities/NaNs, and prevent invalid indices from reaching apply. Do not add a host finite-value scan to every update; use a safe device policy or a documented validated contract.

Apply: compute `source = j + (j >= evict_idx[bh])`; copy only K/V. Flatten output elements for a coalesced baseline:

```text
gid -> d = gid % D, row = gid / D
j = row % (N - 1), bh = row / (N - 1)
source = j + (j >= evict_idx[bh])
```

Read the selected source element and write the corresponding K and V elements. Compare this layout with row/vector alternatives only through measurement. Preserve untouched values bit-for-bit. No trigonometry, position renumbering, score-history update, or auxiliary output array is necessary.

### A4. Wire the real batched cache

The current cache calls `tova_update` inside nested batch/head loops. Merely exposing a `[BH,...]` kernel while invoking it with `BH=1` in that loop leaves substantial overhead untouched.

Add an internal batched update path for uniform `[B,H,S,D]` inputs. Batch scoring and selection across the actual KV-head axis, not query-head count. Keep token steps sequential whenever eviction changes the next step's candidate set. Absorb the below-budget prefix with vectorized concatenation because TOVA has no cumulative score history there. Then process remaining tokens in order, with bounded graph materialization.

Preserve per-head compatibility state if public/tests depend on it, while avoiding repeated full stack/unstack copies in steady-state decode. Audit the base class's rebuilt-buffer copy separately before changing it. Any direct buffer-management optimization must keep `.state`, `size()`, mask construction, offsets, and cache reuse valid. Do not assume that retained-row count equals absolute position.

Do not claim decode calls across sequential transformer layers are independently batchable: later layers depend on earlier hidden states. Batch only work whose inputs are actually available together.

### A5. Treat further fusion as a measured second step

After A1–A4 pass, profile scorer, append, reduce, and apply independently. Consider virtual append—reading old K/V plus the incoming row directly—if concatenation traffic matters. Consider a tiled score kernel only if the MLX scorer is material in the profile.

Do not replace `argmin(softmax(logits))` with `argmin(logits)` as an unconditional exact optimization. Although softmax preserves order in real arithmetic, finite-precision underflow and rounding can create ties that select a different earliest token. Different dot-product reduction orders can also change near-tie decisions. Keep the MLX softmax in the default parity path unless the new implementation passes adversarial selection tests; describe any approximate variant explicitly and keep it opt-in.

Likewise, a one-threadgroup score/select/copy kernel is a possible small-shape experiment, not automatically faster. Assess register pressure, threadgroup memory, copy parallelism, and the small number of KV heads before enabling it.

## 4. Phase B — SnapKV exact selection and gather

First replace both host roundtrips with a GPU-only implementation. Preserve:

- `budget = min(max(budget,1),S)` and `n_sink = min(max(n_sink,0),budget)`, including the empty-input consequences.
- Sink inclusion even if their scores are low.
- Highest scores among non-sinks; equal scores prefer the lower original index because the current Python sort is stable over ascending indices.
- Final output indices sorted in ascending temporal order, `int32`, unique, and of exact requested cardinality.
- Mean of per-query softmax distributions from the observation window. It is not softmax of mean logits, and it is not a causal scorer unless the reference changes explicitly.
- Decode append behavior and repeated prefill-chunk compression over retained plus new rows, with absolute offset bookkeeping intact.

Replace `indices.tolist()` and per-row `mx.stack` in `snapkv_compress` with GPU gather. Derive `n_kept` from known output shape. Batch heads where practical; retain the single-head public API.

Use MLX partition/sort primitives first. Do not assume undocumented stable ordering from `argsort` or arbitrary membership from `argpartition` is equivalent at ties. One exact finite-score strategy is:

1. Partition **only non-sink scores** to obtain the `k`th-largest threshold, where `k = budget - n_sink`; short-circuit `k=0` and keep-all cases.
2. Keep scores strictly above threshold.
3. Compute the remaining count on GPU. Among scores equal to threshold, use an ascending-index prefix count to admit exactly that many earliest ties.
4. Include sinks and produce exactly `budget` sorted indices. Use fixed-size sorting of selected indices with a sentinel, or a proven prefix-compaction primitive; avoid a dynamic-size boolean indexing operation that reintroduces a host count read.
5. Gather K/V on device.

Characterize nonfinite-score behavior separately. Do not claim Python sorting of NaNs implements an ordinary total order. Preserve supported finite inputs and explicitly document any invalid-input contract.

Only add `_snapkv_select.py` / `snapkv_select*.metal` if custom selection beats this complete MLX baseline. For small bounded arrays, benchmark a padded bitonic network with comparator `(score descending,index ascending)` and invalid padded lanes always losing. For larger arrays, use tiled candidate selection plus exact global merge, or threshold selection plus prefix compaction. Specify scratch sizes and prove global membership and tie ordering. Do not put an entire 32k-token sort into one threadgroup or use repeated `k` global argmax passes without a measured small-`k` justification.

Account for the final temporal sort/gather and all intermediate kernels in latency. The deliverable is an integrated prefill improvement; a GPU-only MLX winner is a valid completed result even if no custom top-k is shipped.

## 5. Phase C — extend by actual algorithm

### PyramidKV: next simplest integration

Reuse protected argmin and copy-only compaction with an optional FP32 score array. Preserve the allocator and per-layer budget schedule. Keep bootstrap score one, accumulation over the existing cache, and appended score zero. Do not inherit H2O's grace, decay, position remapping, or other newer policy changes merely because older docstrings say the methods are equivalent. Scores must still update below budget, so TOVA's cheap bulk-fill shortcut does not transfer directly.

Batch within a layer with a common budget. Do not require every layer to have the same output shape. A flat budget schedule is not sufficient evidence of full parity with today's H2O implementation.

### KVzip: separate selection from scoring

Accelerate `latest` using the TOVA-style post-append scorer/select/apply when its cast order and caller contract match. Preserve `pos` and cache metadata.

For `context`, initially reuse only selection/compaction. Its dominant scorer forms `[N,N]` logits and attention. If profiling justifies further work, design a separate exact tiled scorer: compute each probe row's normalization across all key tiles before comparing its normalized probabilities in the max-over-probe reduction. A tile-local softmax or max of unnormalized logits changes the policy. Bound scratch memory; consider a two-pass normalization/importance design with deterministic staged maxima. Compare against MLX matmul + softmax + max, and measure the cost of recomputation versus quadratic temporary storage.

### ChunkKV: design lengths before kernels

Preserve score modes, mean pooling of ragged chunks, earliest-chunk argmin tie breaking, and repeated partitioning after removal. Do not confuse streaming eviction ties with `chunkkv_keep_mask`, whose keep-ranking ties favor newer chunks and whose greedy packing can skip a chunk that does not fit.

A full chunk and a ragged tail have different widths. Which chunk loses depends on device scores, so survivor length can be data-dependent and can differ across heads. Before implementing, document how the cache currently trims heads to a common length and how the coordinator records/reuses per-step indices.

Implement a fixed-width fast path only for shapes where removed width and output length are provably known from host metadata. For general ragged cases, choose between a correct fallback and a fully integrated padded-buffer plus device-length/mask representation. Do not hide a scalar synchronization behind an output-shape helper, and do not expose padded rows to attention as real tokens. Preserve the public Python-list coordinator interface at an explicit compatibility boundary if a new internal device-index representation is introduced.

Required cases: chunk size one, short final chunk, sinks splitting the prefix, cache ending below budget, both score modes, independently differing head choices, leader/follower reuse, and multiple evictions from an initially overfull state where supported.

### CAM: preserve merge and RNG semantics

Ship drop-mode reuse first. For merge modes, use dependent stages: loser selection; cosine-nearest eligible survivor selection excluding sinks and loser; gate and blend; final compaction with exactly one writer for the modified survivor. Additional dispatches are acceptable if they remove host decisions and win overall.

Match existing cosine formulas, including their different epsilon placement in target selection versus blend-weight calculation. Preserve `mean`, `sim_weighted`, `merge_keys`, no-target behavior, score addition only when a merge actually occurs, and the precise seed/draw-counter progression.

Generate the existing keyed random draw using MLX on device and pass it into a kernel if needed. Do not substitute an unrelated Metal PRNG and call it seeded parity. Python float arithmetic currently participates in probability/weight calculation; test threshold cases before replacing it with FP32 arithmetic. Keep any modes without proven compatible behavior on a clearly reported fallback.

### StreamingLLM: vectorize and measure

Replace the per-token list construction with shape-derived sink and recent slices, concatenate once per buffer, and trim the recent tail. Preserve transition from partially filled sinks to the recent window, dtype conversions, tokens-seen accounting, and offset behavior.

The current code only trims when `window_size > 0`; zero does not currently mean an empty recent buffer. Characterize or separately fix this contract instead of changing it accidentally. Measure vectorized MLX update plus `stream_get_kv` plus cache integration before deciding whether a copy kernel is useful.

A ring buffer is a separate design requiring chronological presentation to attention and compatible masking. It is not a free O(1) replacement if every fetch must still gather the full window. Leave StreamingLLM on MLX when that is the best measured implementation.

## 6. Correctness and integration gates

Add meaningful parameterized tests that compare an independent reference with forced MLX and forced Metal paths. Do not rely only on two wrappers calling the same new helper. Skip GPU tests cleanly when Metal is unavailable; exercise fallback behavior separately.

For TOVA, cover `BH` values including 1, 8, and non-power-of-two groups; representative `D` values 32, 64, 96, 128, 256 plus an odd dimension; budgets 1, small odd sizes, and realistic 128–4096; no sinks, normal sinks, and `budget-1` sinks; below-budget fill, exact boundary, first overflow, long decode, and multi-token updates crossing the boundary. Use a targeted matrix rather than an enormous Cartesian product.

Force earliest/interior/newest eviction; uniform weights; equal minima in different SIMD groups and grid-stride iterations; tiny softmax probabilities; signed zero; near ties; noncontiguous inputs; and direct FP32 quantizer input. Assert exact retained identities, ordering, shape, dtype, sink preservation, and bitwise copy equality. Floating-point scorer closeness alone does not establish eviction parity.

For SnapKV, test all-equal scores, threshold ties crossing tile boundaries, budgets below sink requests, keep-all, zero/negative clamp inputs, empty selection inputs, observation windows beyond sequence length, non-power-of-two `S`, large `S`, exact cardinality, temporal sorting, and gathered-value identity. Test multiple prefill chunks followed by decode, not just a single compression call.

At cache level, test factory-created caches, `[B,H,S,D]` shapes, GQA models' actual KV-head counts, absolute offset advancing by incoming `S`, retained row count, `.state`, supported state restoration/reuse behavior, mask/size methods, byte accounting, and non-trimmability. Run the existing regression suites for every modified method.

Run a locally available mlx-lm model through prefill and at least 128 decode steps after capacity is reached, with forced reference and accelerated backends. Record model identifier, configuration, prompt, tokenizer, seed, versions, and chunk size. Compare retained-state/logit behavior and deterministic output where appropriate. Similar-looking generated text alone is insufficient evidence. Preserve the documented key-as-query adaptations; kernel acceleration does not establish paper fidelity or model quality.

Run a long-prefill resource test to ensure removal of `.item()` has not created unbounded graph growth. Test stream behavior and `mx.compile` compatibility where supported; document unsupported compilation rather than implying it works.

## 7. Benchmark design and promotion criteria

Add a reproducible benchmark script, for example `scripts/eviction_kernel_bench.py`, with explicit backend selection, seeds, shape lists, warmup/repetition counts, and JSON output. Preserve the original reference for measurement. Record three alternatives:

1. Current/reference Python-driven implementation.
2. GPU-only vectorized MLX implementation.
3. Custom Metal implementation, where present.

Separate these measurements:

- Selection/apply microbenchmark on identical already-materialized inputs.
- Full scorer + append + selection + compaction operation.
- `update_and_fetch`, including real packing/state/buffer work.
- Model prefill/TTFT and steady-state decode throughput.

Warm each tested specialization outside steady-state timing. Report cold compilation separately. Evaluate inputs before measurement and synchronize/materialize all outputs at the end of each timed sample. Also measure an amortized sequence of dependent updates with appropriate boundary evaluation; do not compare per-operation synchronized MLX timing with asynchronously queued Metal work.

For repeated calls, prevent accidentally reusing already-evaluated outputs. Reset state consistently for independent samples, and use identical token streams for evolving-cache runs. Report median and p95 latency, dispersion, trial counts, and confidence or run-to-run variability. A baseline deliberately burdened with extra synchronization is not evidence of kernel superiority.

Suggested targeted shapes:

- TOVA/Pyramid decode: `B=1,2,4`, `H_kv=1,4,8,16,32`, `D=64,128,256`, budgets 128–4096; emphasize `B=1,H_kv=8,D=128` and boundary/odd shapes.
- TOVA prefill: incoming `S=1,16,128,512,2048`, including many evictions and several chunk boundaries.
- SnapKV: candidate length 512, 2048, 8192, 32768; budgets 128, 512, 2048; observation window 16, 32, 64; include scoring memory in feasibility checks.
- KVzip context: a separate memory-aware sweep; do not allocate a huge quadratic batch casually.
- StreamingLLM: decode and large prefill chunks, measuring complete fetch as well as update.

Record Apple chip/GPU, macOS, Python, MLX, mlx-lm, available memory, and thermal/power conditions where observable. Do not infer current hardware from old benchmark documents. Test on additional Apple generations only if hardware is actually available; qualify single-machine results.

Estimate traffic by stage. A minimal FP16 K/V copy reads and writes roughly `8 * BH * (N-1) * D` bytes; reduction additionally reads roughly `4 * BH * N` score bytes, plus indices and repeated index loads. Full update also includes concatenation, scoring, contiguous conversions, and wrapper copies. Label this a logical traffic estimate, not a hardware bandwidth-counter measurement. Profile dispatch count and GPU occupancy with available tooling before asserting a bottleneck.

Promotion requires exact correctness for the claimed domain and a reproducible improvement over the strongest equivalent MLX path at representative shapes. Use a predeclared decision rule, such as at least 10% median complete-cache improvement exceeding measured noise, with no material regression in routed shapes; this is a project acceptance target, not a promised result. Verify model-level benefit or clearly report when weight/attention costs hide the cache improvement. Use simple measured shape dispatch only when justified; keep slower cases on MLX.

Do not report synthetic batching across unavailable layer inputs as a real generation optimization. Do not generalize this repository's earlier attention roofline numbers to eviction kernels without measurement.

## 8. Delivery checklist

Deliver reviewed implementation files, real cache/factory routing, lazy exports through the appropriate Metal modules, a reliable way to force backends for tests/benchmarks, and supported fallback behavior. Capability absence may select fallback; unexpected shader failures must be visible during development and in forced-Metal tests rather than being swallowed by a broad exception.

Run targeted quantizer/cache/Metal tests for changed methods, existing H2O/Q-Filters tests if shared primitives change, relevant cache protocol regressions, and the repository's required lint checks. Record exact commands and outcomes. Build a wheel and verify all newly referenced `.metal` sources are included under the existing `metal/src/*.metal` package-data rule; test loading from the installed artifact when feasible.

Add a findings document with measured tables and links to raw JSON results. Explain which paths are automatic, experimental, or fallback; supported shape/dtype/version ranges; numerical and algorithmic guarantees; resource bounds; and any remaining synchronization. Distinguish existing source bugs, new fixes, and deliberate unchanged behavior.

Completion means TOVA and SnapKV's mandatory phases are integrated and validated; every later phase has either a tested implementation or a concrete measured/technical reason for retaining its fallback, with remaining work stated plainly. Do not mark CAM merge or general ragged ChunkKV accelerated merely because their simple special cases work.

End with a concise report: what changed, why it wins or does not, exact correctness/benchmark coverage, packaging status, limitations, and commands to reproduce. Never fabricate unrun tests, unsupported hardware results, or generation speedups.
