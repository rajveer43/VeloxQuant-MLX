# TOVA Metal optimization — engineering implementation prompt

Repository: `/Users/rajveerrathod/Work/personal_projects/turboquant_mac_implementation`

Reviewed baseline: `2e3cbd1`, 2026-09-08. This brief starts from the implemented
TOVA backends and their committed measurements. It proposes optimization work;
none of the proposed gains below should be presented as already measured.

## Copy-ready assignment

Act as the principal engineer responsible for MLX/Metal inference performance
and numerical correctness. Optimize the existing TOVA implementation from its
current working baseline through tested integration and reproducible evidence.
Use the judgment expected of an experienced GPU engineer: quantify overhead,
reduce unnecessary work, choose layouts from access patterns, and keep only
changes that improve representative complete workloads.

The objective is lower cache-update latency and better sustained multi-token
throughput on Apple silicon without changing TOVA's eviction policy, retained
identities, FP16 storage contract, or absolute RoPE offsets. Deliver functioning
code, tests, benchmarks, installed-wheel validation, and a findings report.
Do not stop at a shader prototype or a list of recommendations.

Scope is TOVA. Preserve unrelated working-tree changes. Do not rewrite H2O,
SnapKV, attention, model integration, or other cache families to make this task
look more successful. Shared abstractions are justified only by a demonstrated
need. Keep all experiments forceable and retain the existing MLX and Metal
implementations as comparators until replacements are validated.

## 1. Read and freeze the baseline

Inspect applicable repository instructions and capture HEAD, working-tree
status, machine, software versions, and relevant configuration before edits.
Read:

- `veloxquant_mlx/metal/_tova_evict.py` and both `metal/src/tova_evict_*.metal` files.
- `veloxquant_mlx/quantizers/tova.py`, especially `_tova_update_reference`,
  `_evict_mlx`, `_tova_update_batched`, and `_resolve_backend`.
- `veloxquant_mlx/cache/tova_cache.py` and the TOVA configuration/factory wiring.
- `veloxquant_mlx/tests/metal/test_tova_evict.py`,
  `tests/quantizers/test_tova_backends.py`, `tests/quantizers/test_tova.py`,
  and `tests/cache/test_tova_cache.py`.
- `scripts/tova_kernel_bench.py`, `docs/TOVA_METAL_FINDINGS.md`, and all three
  `docs/benchmarks/tova_m4*.json` datasets.
- Installed MLX custom-kernel/compilation APIs and installed mlx-lm cache/model
  code. Use actual API support, not assumptions from the newest online examples.

Current implementation facts:

1. Keys and values are already batched across `G = B * H_kv`. Do not claim
   introducing head batching as a new optimization.
2. Each over-budget token still materializes appended K and V with MLX
   concatenations, casts candidate K to FP32, runs matmul and softmax, then
   selects and compacts. Two Metal dispatches cover only the last two steps.
3. Reduction uses one threadgroup per G, defaults to four 32-lane SIMD groups,
   and merges SIMD results serially in thread zero after a threadgroup barrier.
4. Apply already uses adjacent lanes for adjacent dimensions. It is not the old
   H2O one-thread-per-row serial-copy layout. It performs runtime quotient and
   remainder operations to recover dimension, row, and group.
5. The wrapper allocates a sink scalar array on each eviction, calls cached
   kernel factories, declares new outputs, and enforces row contiguity.
6. Values do not participate in TOVA scoring. Nevertheless, V is appended and
   compacted on every token of a multi-token call.
7. The cache already removed its padded base-buffer copy. It still rebuilds
   per-head `TovaState` views each call and resolves backend capabilities through
   Python. Measure whether that matters before changing compatibility state.
8. Below-budget input is already absorbed in bulk. The post-capacity token loop
   must remain sequential because each eviction changes the next score input.
9. `_EVAL_FLUSH_INTERVAL` is 32 within a call. It bounds lazy graph growth;
   removing it is not a free speedup.
10. `auto` currently chooses Metal for multi-token GPU updates and MLX for
    single-token updates. It is a conservative measured policy, not an immutable
    specification or a reason to force Metal onto all decode shapes.

Historical measurements on the M4 are context, not acceptance data for this pass:

| Case | MLX | Existing Metal | Interpretation |
|---|---:|---:|---|
| G=8, D=128, budget=512, one cache update | 0.4681 ms | 0.4477 ms | Small difference; synchronization and host overhead matter. |
| Same shape, 64 sequential eviction steps in one update | 13.8521 ms | 10.1047 ms | Existing multi-token advantage; not 64 model decode steps. |
| G=8, D=64, budget=128, one cache update | 0.3129 ms | 0.4746 ms | A measured Metal regression; do not generalize the multi-token win. |
| G=8, D=256, budget=128, 64 eviction steps | 7.8179 ms | 7.0402 ms | Less headroom than the D=128 example. |

Some raw p95/median ratios exceed 2–3. Reproduce with stronger controls before
making dispatch decisions from small median differences.

## 2. Write the contracts before tuning

Use `M` for retained capacity, `N=M+1` for candidates at steady state, `D` for
head dimension, `G=B*H_kv`, and `T` for incoming token count.

For each overflow step, preserve this exact operational contract:

```text
K_candidate = concatenate(K_retained, fp16(incoming_key))
q_proxy     = fp32(incoming_key)
logits      = matmul(fp32(K_candidate), q_proxy) * (1 / sqrt(D))
weights     = softmax(logits)
evict       = earliest minimum eligible weight after sink protection
retain      = all rows except evict, in the same temporal order
```

The cache rounds incoming K to FP16 before entering this quantizer path. Direct
quantizer callers may supply FP32 incoming keys, whose proxy must remain
unrounded even though their stored candidate row is FP16. Preserve both APIs.
The newest row is eligible. Add no grace window, score decay, history, or RoPE
operation. Surviving FP16 rows must remain bitwise copies.

Preserve the public wrapper's low-level nonfinite policy: ignore NaN weights,
choose the earliest eligible row if every eligible weight is NaN, and handle
infinite ties without emitting an invalid index. The high-level finite-score
reference and the low-level invalid-input policy are separate test contracts.

Keep empty updates, odd D, noncontiguous inputs, CPU fallback, explicit Metal
failure behavior, byte accounting, true offsets, `size()`, `.state`, and
non-trimmability working. Existing zero/negative-budget fallbacks and limitations
on restoring evicted absolute positions are not optimization opportunities.
Do not silently repair a different policy or mask bug while claiming parity.

Separate numerical promises:

- **Mandatory:** selection and copy transformations preserve exact identities
  for identical supplied weights, including ties.
- **Mandatory for automatic routing:** complete updates pass independent
  reference comparisons with exact retained identities on adversarial and
  realistic traces. Report which domain and MLX versions were tested.
- **Experimental only:** scorer variants that change reduction or softmax
  arithmetic and can alter decisions. Small logit error is not an adequate
  substitute for identity parity.

## 3. Improve measurement before optimizing

Extend the existing harness rather than replacing its baseline numbers with
incomparable new measurements.

### 3.1 Measure four workload levels

1. **Kernel components:** reduction alone with an evaluated index output; apply
   alone with supplied evaluated eviction indices; complete select/apply.
2. **Full update:** scorer, append, casts, reduction and compaction, including
   any wrapper conversions or scratch setup.
3. **Actual cache calls:** steady-state `update_and_fetch`, below-budget fill,
   mixed prefill chunks, and sequences of real single-token calls.
4. **Model execution:** prefill/TTFT and sustained decode through mlx-lm, using
   identical model, tokens, cache policy and evaluation boundaries.

The existing `decode_chain` is a multi-token numerical update; rename or label
it clearly and retain it. Add a distinct `cache_decode_sequence` that calls the
cache repeatedly with `S=1`. Also measure a full prefill starting from an empty
cache and crossing capacity. Do not infer a true cache-prefill improvement from
a numerical-loop benchmark alone.

### 3.2 Control the experiment

- Pre-materialize inputs and warm all tested specializations. Measure cold JIT
  separately in fresh processes; do not equate first invocation with compiler
  time.
- Alternate or randomize candidate/baseline order within trials; also run
  separate process repetitions. Keep workloads, seeds and evaluation cadence
  identical. Avoid concurrent GPU tests, browser rendering benchmarks, builds,
  or other intentional load during timing.
- Use both per-call synchronized latency and realistic dependent sequences
  with boundary evaluation. Evaluate every output required by the workload;
  prevent accidental reuse of already-evaluated results.
- Reset all mutable cache bookkeeping consistently when resetting samples.
  The current script resets K/V and offset but not all counters/views. Add a
  dedicated deterministic setup/reset mechanism or construct fresh equivalent
  cache state outside each timing interval. Charge the same setup costs to
  every candidate where setup is part of the measured workload.
- Keep synthetic random weights for isolated selection timing, but use actual
  scorer-produced weights and realistic eviction-index distributions for
  complete-path measurements. An always-evict-newest toy case is insufficient.
- Record all samples, median, p95, variability, independent run IDs, backend,
  specialization, trace/commit identifiers, and memory metrics available in
  the installed MLX. Use paired trial statistics; do not bootstrap correlated
  individual GPU calls as though they were independent machines/runs.
- A target of 30–100 timed samples per case and at least three independent
  repetitions is reasonable. Adapt run counts to variance and runtime rather
  than executing an enormous Cartesian product blindly.

Use Metal System Trace or available GPU profiling/capture tools to distinguish
CPU submission, command-buffer scheduling, GPU kernel time, copying and
synchronization. Record tool availability and exact method. Never label a kernel
occupancy-bound from wall time alone, subtract two noisy medians to manufacture
a GPU duration, or present guessed counters as observed evidence.

### 3.3 Workload matrix

Prioritize `G=1,8`, `D=64,128,256`, `M=128,512,2048`, `T=1,2,16,64`.
Extend targeted cases to G=3,16,32; D=7,33,96; M=1,127,513,4096;
partial-fill and capacity crossings; and long T=2048 resource tests. Include
GQA layouts where H_q differs from H_kv. Do not expand every axis simultaneously.

A candidate tuned on M4 must retain a generic fallback. Run other Apple
generations only if available and identify exactly what was measured.

## 4. Build a cost model and choose experiments

Draw the actual MLX graph for one overflow step and label reads, writes, casts,
dispatches and evaluation points. These are logical traffic estimates, not
claims about DRAM transactions:

| Operation | Approximate logical bytes |
|---|---:|
| Materialize FP16 K append | `4*G*N*D` |
| Materialize FP16 V append | `4*G*N*D` |
| Cast candidate K FP16→FP32 | `6*G*N*D` |
| Read FP32 candidate K for scoring | `4*G*N*D` plus proxy traffic |
| Compact FP16 K and V | `8*G*M*D` |
| Read FP32 weights for argmin | `4*G*N` |

Cache reuse, compiler fusion, allocation strategy, vectorization and actual
memory accesses can change physical traffic. Confirm which intermediate arrays
are materialized. Do not sum component timing medians and call that a predicted
full-update time. Use Amdahl's law to bound an experiment's possible overall
benefit once stage proportions are measured.

Rank experiments by avoidable traffic/dispatch and expected risk:

1. Defer V movement across multi-token updates.
2. Remove materialized V append through virtual-source gathering.
3. Tune the existing reduction and compaction layouts with bounded variants.
4. Reduce measured Python/graph-building overhead and evaluate compilation.
5. Fuse selection/apply for small shapes using one threadgroup per head.
6. Remove K append/cast only with a scorer-compatible design and numerical
   evidence; custom scoring is a separate, higher-risk project phase.

This ordering is provisional. If profiling identifies a different dominant
cost, document it and reorder. For every experiment state hypothesis, changed
cost, launch/layout, expected risk, comparison, and stop condition before coding.

## 5. First major experiment: defer values until the end of a chunk

This optimization follows directly from TOVA's data dependencies: the scorer
reads only keys. Intermediate values are never observed outside a single
`update_and_fetch` call.

For a multi-token call, retain the initial V and incoming V as immutable source
buffers. Maintain a per-head `uint32` lineage map for the active rows instead
of moving V on every eviction:

```text
initial map: [0, 1, ..., initial_length-1]
incoming row at local step t: source ID = initial_length + t
after selecting e: compact K and map, preserving order and source IDs
end of call: gather retained V once from initial V or incoming V by source ID
```

Do not concatenate the full value source merely to simplify addressing if
branching between two source buffers avoids that materialization. Convert
incoming V to the existing FP16 storage semantics before final output; conversion
can occur in the final gather if it produces exactly the reference result.
Support each head's independent lineage and test head/base offsets thoroughly.

Proposed internal implementation:

- An apply variant writes retained K and lineage IDs rather than retained K/V.
- Each lineage element has one writer. If apply is mapped per K element, only
  the designated row lane writes the lineage; avoid redundant conflicting
  writes even when they happen to contain identical bits.
- Treat the incoming lineage ID as a virtual appended element, avoiding a
  separate map concatenation where practical.
- A final gather writes `[G,n_kept,D]` FP16 V, coalescing dimensions for each
  chosen source row.
- Batch-fill without eviction can retain the existing cheap MLX path.
- At a graph-flush boundary, evaluate the live K and lineage. Do not materialize
  intermediate V just to reuse the old flush function.

Compare existing full-update traffic against moving roughly O(G*M) map words
per step plus one O(G*M*D) V gather at the end. The optimization adds lineage
storage, map traffic and a final dispatch; for T=1 it may lose. Benchmark T=2
and short chunks as well as long ones. Route only the winning regimes.

The temporary representation lives within one call. Keep public `TovaState`,
returned buffers, cache byte accounting and `.state` unchanged. Preserve pure
update semantics: aliases to input K/V must remain unmodified.

An index-only representation for **keys** is a separate experiment. It would
make scoring indirect and may require a new scorer or repeated gathers,
changing locality and reduction order. Do not silently extend the V argument
to K or claim both can skip compaction at no cost.

## 6. Virtual append: avoid copying data just to remove one row

Implement a lower-risk virtual-V-append variant before modifying scoring:

```text
selection still receives weights from the unchanged materialized K scorer
for output row j:
    src = j + (j >= evicted_index)
    V_out[j] = old_V[src] if src < old_length else new_V[0]
```

Keep candidate K materialized initially, so score arithmetic and ordering are
unchanged. Apply can gather K from that candidate and V from old/new buffers.
This can eliminate the standalone V concatenate without introducing lineage.
Measure it as an independent ablation before combining with deferred V.

For virtual **K** append, explicitly list the scorer choices:

1. Continue to materialize K for the old scorer: this preserves scoring but
   does not eliminate the K-append cost. Do not claim otherwise.
2. Compute old-row and new-row logits separately: candidate dot products can
   change rounding because matrix shape/reduction path changed. Test complete
   eviction identity, not just close logits.
3. Introduce a custom scorer reading old K plus virtual new K. This belongs to
   the numerical-risk phase below.

If you keep an FP32 key shadow within a chunk or fuse compaction with generation
of the next FP32 scorer input, account for wider copies, extra outputs and peak
memory. Every shadow element must represent the rounded FP16 stored key, never
an unrounded direct FP32 incoming key. Benchmark whether avoided casts outweigh
the larger working set; do not assume FP32 storage is universally faster.

## 7. Tune existing Metal primitives without changing their math

### 7.1 Reduction

Sweep `nsg` over the supported 1,2,4,8 choices. Implement a true one-SIMD-group
specialization that avoids unnecessary threadgroup staging and barrier when
nsg=1. Keep `(value,index)` tie semantics at every level and preserve the
nonfinite safety rule.

For multiple SIMD groups, compare the existing thread-zero serial merge with
a first-SIMD-group merge of the partials. The latter requires all participating
lanes to receive initialized values, correct unused-lane sentinels, and a
uniform synchronization point. With at most eight partials the serial merge may
already be cheaper; retain it if measurement says so.

Consider one SIMD group per head with multiple heads inside one threadgroup
only for tiny reductions and enough heads. Distinguish SIMD-wide and
threadgroup-wide barriers when heads share a threadgroup. Do not confuse more
threads inside one group with more independently schedulable threadgroups.

Only consider a split-N reduction for genuinely large N/small G when trace
evidence justifies another dispatch and partial buffers. Weight reduction is
O(G*N), whereas copying/scoring is O(G*N*D); it may be the wrong place to spend
complexity. A single threadgroup can grid-stride arbitrary N; N need not fit
into threadgroup memory or equal the thread count.

### 7.2 Compaction

Keep current contiguous-element apply as a comparator. Try a two-dimensional
grid with group/head along y and flattened row-dimension along x to remove
division by total rows from the per-element path:

```text
bh = thread_position_in_grid.y
x  = thread_position_in_grid.x
j  = x / D
d  = x % D
src = j + (j >= evicted[bh])
```

Evaluate bounded D specializations for 64,128,256 so constant division can be
optimized, plus a generic arbitrary-D fallback. Sweep threadgroups of 64,128,
256 threads where supported and meaningful. Changing launch geometry needs
boundary tests for both dimensions.

Evaluate 2- or 4-element loads/stores per thread only after proving source and
destination alignment. `ensure_row_contiguous=True` does not by itself prove
that every sliced buffer's base address has the alignment of a vector pointer
cast. Do not reinterpret to `half4*` on an unsupported assumption. Use safe
scalar loads/vector construction or a proven alignment predicate and fallback.
Handle row-end tails without crossing the eviction discontinuity or another
head's rows. Preserve exact FP16 bit patterns.

Do not add shared-memory K/V staging to a single-use copy without identifying
reuse. More instructions or vector types are not performance evidence. Measure
larger per-thread work against register pressure, number of threadgroups and
short-shape utilization.

### 7.3 Small-shape selection/apply fusion

Prototype one threadgroup per head: reduce weights, publish one selected index
inside that group, then cooperatively copy that head's rows. This legitimately
removes a dispatch and device index buffer because the consumers are in the
same threadgroup.

It also restricts apply parallelism to G threadgroups. At small G or large M*D
that can overwhelm any dispatch saving. Use global input/output buffers and
grid-stride copies; do not allocate O(M*D) threadgroup arrays merely to fit the
entire cache. Keep it bounded to measured shapes and compare against the
two-dispatch implementation including scoring and cache overhead.

Never fuse separate threadgroups through spin-waiting flags or a fictitious
global barrier. A threadgroup barrier synchronizes only its own group; Apple
warns that incorrect synchronization and out-of-bounds access can produce
incorrect results or memory corruption. See the [Apple silicon Metal porting
guide](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon).

## 8. Python, MLX graph and specialization overhead

Profile factory lookup, repeated imports/backend resolution, sink scalar
creation, output setup, compatibility views and graph construction separately
from GPU execution. Move invariant setup out of the token loop when it matters.
Use bounded caches keyed by true specialization/configuration, with correct
device/stream handling. Do not cache arrays in a way that pins thousands of
contexts or mixes CPU/GPU use unexpectedly.

Possible experiments:

- Reuse one sink scalar per batched update instead of creating it for every
  step; consider a small set of sink constants only if compile-cache growth is
  bounded.
- Hoist invariant scorer scale, backend selection and callable lookup.
- Compile a pure array-to-array fixed-shape update using installed `mx.compile`
  where supported. Keep object mutation, cache accounting and evaluation
  boundaries outside that compiled function.
- Benchmark short fixed-length compiled chunks against ordinary lazy
  construction. Control graph size and compilation time; do not unroll a
  2,048-token dependency chain into one enormous compiled specialization.
- Sweep flush intervals such as 16,32,64,128 under long-prefill and memory
  stress. Record peak memory/resource behavior and tail latency. Preserve a
  safe bound and default if larger intervals bring little benefit.

MLX custom kernels expose output shapes, specialization and contiguity handling;
inspect the installed signatures and account for contiguity copies rather than
assuming strided views are free. Current MLX documentation also distinguishes
safe from relaxed/fast compilation math. Do not enable fast math globally in a
selection-sensitive pipeline. See the [MLX custom Metal kernel documentation](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).

Array output shapes remain host metadata. Do not add a device-dependent output
length that requires a hidden scalar read. Do not mutate input arrays or rely
on undocumented output aliasing/donation to simulate an in-place compact.

## 9. Higher-risk phase: scorer fusion

Start only after measuring whether scoring/casts now dominate the improved
path. A scorer writing logits from FP16 K with FP32 accumulation might remove
the full K cast and enable virtual K append. A further score/select fusion
might avoid materializing weights. These are hypotheses with numerical costs.

Design candidates around the actual single-proxy GEMV shape, not a large GEMM:

- One SIMD group cooperatively computes a key-row dot product, with adjacent
  lanes reading adjacent D values; evaluate several rows per group where useful.
- Tile rows across threadgroups when enough work exists. Specify partial
  outputs, normalization/reduction stages and total dispatch count.
- Compare query reuse in registers/shared memory against its load and barrier
  costs. D=64/128/256 is small; elaborate staging may lose to caches.
- Do not apply `simdgroup_matrix` machinery merely because attention GEMMs use
  it. TOVA's one-proxy scoring has different reuse and occupancy properties.

Preserve normalization across **all** candidate rows, including protected sinks
and the newly appended key. Protection affects eviction eligibility; removing
sinks from the softmax denominator changes probabilities and floating-point
ties even though those rows cannot be evicted.

Mathematically `argmin(softmax(logits))` agrees with `argmin(logits)`, but that
does not establish equivalence to finite-precision MLX. Underflow, rounded
probability ties, multiplication versus division, dot-product reassociation,
FMA and exponential implementation can change the earliest selected row.
Even dropping a common normalizer can remove ties caused by division rounding.

Therefore:

1. Keep the exact existing scorer as a forceable path.
2. First compare logits and weights diagnostically, then compare selected
   identities and complete retained trajectories.
3. Construct adversarial equal/near-equal logits, extreme FP16 keys, direct
   FP32 proxies, underflowed weights, signed zero and nonfinite cases.
4. Test sequence lengths large enough to cross MLX kernel/tiling thresholds.
5. If identity parity cannot be maintained, do not silently promote a logit-only
   or fused-softmax variant. Report it as an approximate opt-in experiment, or
   leave it unshipped.

A parity-preserving V movement optimization is valuable even if exact scorer
fusion proves impractical. Do not weaken the algorithm to satisfy a speed target.

## 10. Verification and dispatch policy

### 10.1 Independent correctness

Run existing suites and add focused tests for changed representations and
launches. The reference must not call the new helper it is meant to validate.

For deferred V, fingerprint values independently of keys, force each possible
eviction region, and compare final V to sequential reference across heads,
chunk sizes, partial fills, zero/one retained row where supported, and long
chunks. Validate lineage source IDs and final gather on GPU, using host reads
only in the test harness. Check that input buffers remain unchanged and that
no intermediate padded rows escape as valid cache rows.

For layout/reduction changes, force ties across lanes, SIMD groups, workgroup
boundaries and grid-stride iterations. Test shape values immediately below,
equal to and above vector/tile/threadgroup boundaries. Include sliced
row-contiguous inputs with nonzero offsets and noncontiguous inputs requiring
copies. Test all initialized outputs and supported stream/compile paths.

Keep the existing tiny mlx-lm Llama GQA forward-parity test. Add real-cache
multi-token transitions and long decode. If locally available pretrained MLX
weights permit full-model measurement, use them and record identifiers;
otherwise clearly distinguish synthetic model plumbing from quality evidence.
Do not download large models automatically or claim trained-model throughput
from random-weight tests.

Run CPU/unavailable-kernel fallback tests and wheel-install GPU smoke tests.
Verify every new shader is included by the `metal/src/*.metal` package-data
rule. Test actual installed sources, not just checkout imports. Keep a clean
failure path for unsupported dtype/device/shape/launch variants.

### 10.2 Promotion rule

Compare every candidate with both the existing Metal path and the strongest
GPU-only MLX path. The original Python-driven reference is for correctness and
historical context; beating it alone does not justify new shader complexity.

Predeclare promotion criteria before the final measurement run. A useful target
is at least 10% improvement in the intended complete workload, supported by
independent runs and uncertainty estimates, with no material p95 regression or
unbounded memory increase. Treat 10% as a project decision threshold, not a
predicted gain. Smaller improvements may be worthwhile for very simple changes
with strong evidence, but justify that decision explicitly.

Use a compact deterministic dispatch table keyed only by relevant shape/device
properties. The number of actual overflow steps may be a better variable than
raw incoming S for deferred-V routing. Separate kernel tuning from algorithm
selection; avoid one cached kernel per exact token count. Keep a generic fallback
and forced-path tests for every route. Do not autotune on live user decode.

GPU family and pipeline resource limits vary. Check legal threadgroup sizes and
available features rather than assuming a device-wide maximum guarantees every
kernel can use it. Apple documents pipeline-dependent limits in [Calculating
threadgroup and grid sizes](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes)
and publishes [Metal feature tables](https://developer.apple.com/metal/limits/).

## 11. Required deliverables and execution order

Deliver work in reviewable stages, keeping each hypothesis independently
testable:

1. Baseline audit and improved benchmark harness, including actual cache
   multi-token and single-token-sequence workloads.
2. Virtual V append and deferred-V lineage prototypes, each measured alone.
3. Bounded reduction/apply tuning with a recorded configuration sweep.
4. Measured host/graph improvements and optional small-shape selection/apply
   fusion.
5. Scorer-fusion investigation only if the profile justifies it; report exact
   versus approximate behavior honestly.
6. Combined ablations, final dispatch policy, regression tests, packaging,
   raw measurements, documentation and reproduction commands.

Suggested files may include `metal/src/tova_evict_apply_indices.metal`,
`tova_gather_values.metal`, wrapper additions in `_tova_evict.py`, and numerical
integration in `quantizers/tova.py`. Choose names based on the final design;
do not add empty placeholders for abandoned experiments.

The final report must contain:

- The measured bottleneck and evidence, not just the optimization attempted.
- Before/after complete-workload median and p95, hardware/software, trial
  counts, dispatches, logical traffic model, and peak-memory impact.
- A small ablation table showing which changes matter alone and together.
- Exact algorithmic contracts, test commands/results, installed-wheel status,
  automatic dispatch coverage and fallback behavior.
- Rejected variants, their measured outcome, and any unresolved numerical or
  hardware limitations. Do not leave dead experimental paths in the default
  execution graph.

Finish the authorized implementation and validation without asking for routine
layout choices. Stop an experiment when evidence shows it is not worthwhile,
but continue independent useful work. If no additional kernel variant beats the
current implementation, deliver the measurement improvements and a precise
negative result. Fabricated speedups, selectively synchronized baselines and
changed eviction semantics are not acceptable completion criteria.
