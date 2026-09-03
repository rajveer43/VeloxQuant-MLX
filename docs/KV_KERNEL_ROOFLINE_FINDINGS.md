# KV-Cache Kernel Roofline Analysis (issue #259)

## TL;DR

VeloxQuant's quantize/dequantize/fused-attend Metal kernels are, as
expected, **overwhelmingly memory-bound by construction** — none of them
does enough arithmetic per byte touched to ever be compute-bound (argmin
over ≤16 centroids, a min/max reduction, or a dot product against a long
but thin K/V cache). Measured arithmetic intensity across all kernels
tested is **1–11 FLOPs/byte**, far below the point where any Apple Silicon
GPU's compute throughput would become the constraint. Two of the four
measured kernels reach 55–102% of this machine's calibrated memory
bandwidth at large sizes, confirming the roofline model's prediction
directly. The interesting exception — `scalar_fused_decode_attend`, the
fused dequantize+attention kernel — **never becomes memory-bound even at
16k cached tokens**, topping out at 6% of peak bandwidth. That is not a
bandwidth problem; the occupancy sweep in this analysis shows it is
**dispatch/occupancy-bound**: realistic single-token decode shapes
(`B*H*S_q` small) launch too few threadgroups to fill a 10-core GPU,
regardless of how much data any one threadgroup has to move.

This complements, not duplicates, the existing prefill roofline
([blogs/prefill-roofline.md](../blogs/prefill-roofline.md)), which applied
issue #259's methodology to the *compute-bound* causal-prefill regime
(`S_q ≈ S_kv`, both large — dominated by big matmuls). This analysis covers
what #259 actually names: the KV-cache read/write kernels, which are a
different regime (decode: one query against a long cache, low reuse).

## Method

Following the same self-calibration principle as the prefill roofline
(never trust an unverified spec-sheet bandwidth number):

1. **Theoretical bytes moved** and **FLOPs** are derived analytically from
   each kernel's own `.metal` source — input/output shapes, dtypes, and
   documented per-element work (see each kernel's docstring in
   `scripts/kv_kernel_roofline_bench.py`).
2. **Arithmetic intensity** = FLOPs / bytes.
3. **Achieved bandwidth** = bytes / measured latency; achieved GFLOP/s =
   FLOPs / measured latency.
4. **Calibrated bandwidth peak**: measured directly on this GPU, this run,
   via a large elementwise op (`a * 2.0` at 100M–300M fp16 elements) —
   mirrors `_calibrate_matmul_peak()` in `prefill_roofline_bench.py`, which
   found the naive spec-sheet-derived FLOPs number was off by ~3x. On this
   machine: **~97–99 GB/s achieved** (some run-to-run variance, same
   caveat the prefill roofline noted — see below).
5. **Classification**: ≥60% of calibrated peak bandwidth ⇒ memory-bound.
   Below that, latency within ~2x of the same kernel's smallest-tested-size
   latency ⇒ launch-bound (fixed dispatch overhead dominates). Otherwise ⇒
   occupancy-limited — neither bandwidth- nor launch-dominated, confirmed
   separately via a threadgroup-count sweep at fixed problem size.

Hardware: one Apple M4 (10-core GPU, 24 GB), the same machine used for the
prefill roofline — see that document's own caveat about run-to-run
variance from thermal/scheduling effects; numbers here were reproduced
across two runs and agreed within a few percent except where noted.

## Results

### `turboquant_scalar_quantize` — nearest-centroid encode (argmin over 2^b centroids)

| N | b | latency | GB/s | % peak | AI (F/B) | class |
|---|---|---|---|---|---|---|
| 10,000 | 2 | 0.20 ms | 0.2 | 0.2% | 2.67 | launch-bound |
| 1,000,000 | 2 | 0.25 ms | 12.2 | 12.3% | 2.67 | launch-bound |
| 16,000,000 | 2 | 0.92 ms | 51.9 | 52.2% | 2.67 | occupancy-limited (near memory-bound) |
| 10,000 | 4 | 0.18 ms | 0.2 | 0.2% | 10.67 | launch-bound |
| 1,000,000 | 4 | 0.24 ms | 12.4 | 12.5% | 10.67 | launch-bound |
| 16,000,000 | 4 | 0.88 ms | 54.6 | 54.9% | 10.67 | occupancy-limited (near memory-bound) |

Even at b=4 (16 centroids scanned per element, the highest FLOPs/byte
tested), AI stays at 10.67 — nowhere near compute-bound. Bandwidth climbs
toward ~55% of peak as N grows and plateaus; it doesn't clearly cross the
60% memory-bound threshold at the largest size tested here, so more
headroom may exist at even larger N (not tested — see Limitations).

### `turboquant_scalar_dequantize` — centroid gather decode (zero arithmetic)

| N | b | latency | GB/s | % peak | class |
|---|---|---|---|---|---|
| 10,000 | 2/4 | ~0.17 ms | 0.2 | 0.2% | launch-bound |
| 1,000,000 | 2/4 | ~0.23 ms | 13.0 | 13.2% | launch-bound |
| 16,000,000 | 2 | 0.78 ms | 61.5 | 61.8% | **memory-bound** |
| 16,000,000 | 4 | 0.79 ms | 60.6 | 61.0% | **memory-bound** |

Zero FLOPs per element by design (a pure gather) — this is the purest test
of the memory-bound hypothesis in this suite, and it confirms it directly:
at large N the kernel crosses the 60% threshold and sits within ~40% of
this machine's calibrated bandwidth ceiling, doing no arithmetic at all.

### `kivi_group_quant_dequant` — group-affine quantize+dequantize (fused, group_size=32)

| shape (BH×S×D) | latency | GB/s | % peak | AI (F/B) | class |
|---|---|---|---|---|---|
| 8×128×128 | 0.20 ms | 4.0 | 4.0% | 1.17 | launch-bound |
| 32×512×128 | 0.32 ms | 38.8 | 39.0% | 1.17 | launch-bound |
| 32×4096×128 | 0.99 ms | 101.6 | **102.2%** | 1.17 | **memory-bound** |

The cleanest confirmation in this analysis: at the largest tested shape
(32 batch-heads × 4096 tokens × 128 channels — a realistic long-context
KIVI cache size) this kernel **matches or slightly exceeds** the
calibrated bandwidth ceiling (102%, within measurement/calibration noise
of 100%). It's genuinely memory-bound and, at this size, essentially
optimal against the bandwidth roofline — there is no headroom left for a
faster kernel to capture without moving fewer bytes.

### `scalar_fused_decode_attend` — fused group-affine decode + attention

| S_kv | D | latency | GB/s | % peak | AI (F/B) | class |
|---|---|---|---|---|---|---|
| 128 | 128 | 0.25 ms | 1.3 | 1.3% | 1.62 | launch-bound |
| 2,048 | 128 | 0.98 ms | 5.3 | 5.4% | 1.62 | occupancy-limited |
| 16,384 | 128 | 6.92 ms | 6.1 | 6.1% | 1.62 | occupancy-limited |

This is the interesting negative result. Unlike the other three kernels,
**bandwidth utilization never climbs meaningfully with problem size** — it
goes from 1.3% to 6.1% as S_kv grows 128x, nowhere near the 55–100%
other kernels reach at comparable byte counts. AI stays low (1.62), so
it's not compute-bound either. The kernel dispatches exactly one
threadgroup per `(batch, head, query position)` — at a realistic single
decode step (`B=1, H=8, S_q=1`), that's only **8 threadgroups total**, far
too few to fill a 10-core GPU regardless of how many bytes each
threadgroup streams through S_kv.

**Occupancy sweep** (fixed S_kv=16384, D=128, varying dispatched threadgroup count):

| B | H | S_q | threadgroups | latency | GB/s | % peak |
|---|---|---|---|---|---|---|
| 1 | 8 | 1 | 8 | 6.95 ms | 6.0 | 6.1% |
| 1 | 32 | 1 | 32 | 7.74 ms | 21.7 | 21.8% |
| 4 | 32 | 1 | 128 | 18.4 ms | 36.5 | 36.7% |

Bandwidth utilization scales with threadgroup count, not with S_kv or
bytes moved per threadgroup — direct evidence this kernel is
**occupancy/dispatch-bound**, not memory- or compute-bound, at the shapes
that matter for real single-token decode (few heads, batch size 1). This
matches the kernel's own design comment
(`_scalar_attend.py`: "A single 32-lane pass over S_kv under-fills the GPU
for decode shapes... so the kv axis is split flash-decoding style") —
that split (NSG_C SIMD-groups per threadgroup) helps *within* a
threadgroup, but doesn't add threadgroups, so it can't fix under-dispatch
at the B*H*S_q level tested here.

## Classification summary

| Kernel | Regime | Bottleneck at realistic scale |
|---|---|---|
| `turboquant_scalar_quantize` | argmin, AI 2.67–10.67 | memory-bound (trending toward it, ~55% at largest N tested) |
| `turboquant_scalar_dequantize` | pure gather, AI ≈ 0 | **memory-bound** (confirmed, ~61% of peak) |
| `kivi_group_quant_dequant` | group reduce + fused write, AI 1.17 | **memory-bound** (confirmed, ~102% of peak) |
| `scalar_fused_decode_attend` | fused decode+attend, AI 1.62 | **occupancy/dispatch-bound**, not bandwidth — confirmed via threadgroup-count sweep |

No kernel measured here is compute-bound, and none showed evidence of
synchronization-bound behavior in isolation (the fused-attend kernel's
online-softmax merge uses one `threadgroup_barrier` per threadgroup,
amortized over its full S_kv loop — negligible at the S_kv sizes tested).
The one real surprise is that **the fused-attend kernel's problem isn't
memory bandwidth at all** — it's that a single-query decode step doesn't
generate enough parallel work to fill a modern GPU's core count, no matter
how efficiently each unit of that work is streamed.

## Addendum: GQA head-packing experiment (issue #307, part 2)

Issue #307 proposed two fixes on top of this analysis. Part 2 — GQA-style
head-packing, where the `heads_per_kv` query heads sharing one kv head are
packed into a single threadgroup so each K/V code is decoded once and
reused across those heads instead of redundantly redecoded per head — has
been implemented in `scalar_fused_decode_attend`
(`veloxquant_mlx/metal/src/scalar_affine_attend.metal`,
`veloxquant_mlx/metal/_scalar_attend.py`) and is correctness-verified:
43/43 tests pass, including new GQA-shaped parity cases
(`H_q/H_kv ∈ {(8,2), (32,4), (32,8)}`, max abs error < 2e-3 against a
numpy reference) and confirmation that `H_q == H_kv` (plain MHA) produces
unchanged output on every pre-existing test.

**The performance result is a clear negative, and it directly confirms
this document's own occupancy diagnosis rather than contradicting it.**
Packing trades away threadgroup *count* to save K/V byte traffic — but
byte traffic isn't the bottleneck at these shapes; occupancy is. Packing
`H_q=32, H_kv=4` (`heads_per_kv=8`) into one threadgroup per kv-head drops
total dispatched threadgroups to `B*H_kv*S_q = 4` — *below* the
`B=1,H=8,S_q=1` (8 threadgroups) row this document already measured at
6.1% of peak bandwidth. Measured on the same M4 hardware, `S_kv=16384`:

| Variant | threadgroups | latency | vs. packed |
|---|---|---|---|
| packed (nsg=2) | 4 | 43.8 ms | 1.0x |
| unpacked, 32 separate 1-threadgroup dispatches (nsg=2) | 32 (across 32 dispatches) | 16.3 ms | **2.7x faster** |
| unpacked, 32 separate 1-threadgroup dispatches (nsg=8) | 32 (across 32 dispatches) | 9.4 ms | **4.7x faster** |

The "unpacked" baseline here is deliberately the redundant-redecode
anti-pattern (each query head independently redecodes the same shared
K/V codes) — i.e. strictly *more* total DRAM traffic than packed — and it
still wins by 2.7-4.7x, because 32 independent small dispatches give the
GPU scheduler enough concurrent work to fill its cores, while 4 large
dispatches (even without any redundant work inside them) do not. This
matches this document's Recommendation #2 below almost exactly: dispatch
count is the lever that matters at these shapes, not bytes moved.

**Conclusion: GQA head-packing, as scoped in issue #307 part 2, is not a
net win in isolation and should not be adopted as implemented.** It
remains correct and available (useful if a future caller needs the
byte-traffic reduction for a different reason, e.g. once part 1's
cross-layer batching fixes occupancy first — packing would then compose
with a large multi-layer dispatch rather than being the sole source of
threadgroup count). But taken alone, it makes the exact metric this
document identified as the actual bottleneck (threadgroup count) worse in
exchange for optimizing a metric (K/V byte traffic) that isn't yet the
constraint. Issue #307 should be updated to reflect this: part 1
(cross-layer batched dispatch, which increases threadgroup count) is the
correct next step, not an optional companion to part 2 — packing should
be revisited only after occupancy is fixed, to see whether it adds
further gains once dispatch count is no longer the limiter.

Reproduction: `pytest veloxquant_mlx/tests/metal/test_scalar_attend.py -v
-k gqa` for correctness; the packing-vs-unpacked timing table prints from
`test_scalar_attend_gqa_packing_benchmark`.

**Why this isn't fixable with a smarter packing design.** Before
concluding, we checked whether some other threadgroup layout could get
both K/V-decode sharing *and* full `B*H_q*S_q` threadgroup count
simultaneously. It cannot, on Metal, and this is an architectural limit,
not a gap in this design:

- Metal `threadgroup`-address-space memory is scoped strictly to threads
  co-resident in one threadgroup; there is no barrier, SIMD-group
  primitive, imageblock mechanism, or Metal 3/4 feature that lets two
  independently-dispatched threadgroups read each other's on-chip state.
  The only way to make a value visible across a threadgroup boundary is
  `device` (DRAM) memory — exactly the round-trip this kernel family
  exists to avoid.
- GPU occupancy at these shapes is gated by threadgroup *count* (the
  hardware scheduler needs enough independently-schedulable units to
  spread across the GPU's cores), not by making individual threadgroups
  internally wider. A "fatter" threadgroup (more SIMD-groups packed in to
  add real concurrency instead of a serial per-head loop) is still one
  scheduling unit landing on one core — it doesn't recover the cross-core
  distribution that dispatching 32 separate threadgroups gives for free.
- **This is confirmed by MLX's own production kernel.** We read MLX's
  vendored `sdpa_vector.h` (the Metal kernel `mx.fast.scaled_dot_product_
  attention` itself dispatches to at decode/small-S_q shapes) directly:
  it uses one threadgroup per query head with **no K/V-decode sharing
  across heads that share a kv-head** — `gqa_factor` is used only for
  pointer arithmetic (`kv_head_idx = q_head_idx / gqa_factor`), and every
  threadgroup independently re-reads the same K/V data. Where MLX's own
  2-pass long-context variant *does* need state visible across
  independently-scheduled threadgroups (merging partial softmax across a
  KV-axis split), it goes through a `device`-space scratch buffer between
  kernel launches — the same DRAM round-trip this analysis says is the
  only way to do it. MLX had every incentive to solve this if it were
  solvable; it makes the same redundant-refetch-for-occupancy tradeoff
  this document now recommends.

So the recommendation isn't "revert and try again later" in the sense of
expecting a different kernel design to close the gap — it's that
head-packing and full decode-shape occupancy are in genuine tension on
this hardware, and occupancy should win until part 1 changes what
"occupancy" costs (batching across layers changes `S_q`/dispatch
structure enough that revisiting packing at that point is a fair
question, not a re-run of this same experiment).

## Addendum: SIMD-shuffle spike and two-pass decode-once (issue #308)

Issue #308 asked whether `simd_shuffle`/`quad_shuffle` — a genuine, cheap
(single hardware crossbar instruction, no barrier) cross-*lane* exchange
confirmed to exist unconditionally on all Apple Silicon since MSL 2.2 —
could give head-packing's K/V-decode-sharing benefit without sacrificing
threadgroup count the way part 2 above does.

**It cannot, and this is a straightforward consequence of the same
architectural fact already established above, not a new limit.**
`simd_shuffle` exchanges data only between lanes already co-resident in
one SIMD-group, which is a subset of one threadgroup. Under the
full-occupancy dispatch shape (`B*H_q*S_q` threadgroups, one per query
head) that this document recommends, a single threadgroup by construction
only ever holds *one* query head's work for its entire lifetime — there is
no other query head's data anywhere in that threadgroup's address space to
shuffle with. Getting any cross-head sharing still requires those heads to
be co-resident in one threadgroup, which means dropping to `B*H_kv*S_q`
threadgroups — reproducing part 2's already-measured tradeoff, just with a
cheaper merge mechanism, not a way to avoid the threadgroup-count cost.

**A genuinely different two-pass alternative was implemented and measured
instead**, trading a different resource: one extra DRAM round-trip instead
of threadgroup count. `scalar_decode_once` (new:
`veloxquant_mlx/metal/src/scalar_affine_decode_once.metal`) decodes every
K/V code exactly once into a `[B, H_kv, S_kv, D]` fp16 device buffer,
dispatched flat over all elements — occupancy-friendly by construction,
since it scales with `S_kv` rather than `S_q`. `scalar_predecoded_attend`
(new: `veloxquant_mlx/metal/src/scalar_predecoded_attend.metal`) then runs
the *original* full-occupancy `B*H_q*S_q` dispatch against the
already-decoded buffer — heads sharing a kv head still redundantly re-read
the same decoded fp16 rows from DRAM, but no longer redundantly re-decode
them (no dequant multiply-add repeated per head).

Correctness: 10/10 new tests pass, including parity against the same
numpy reference used throughout this document across three head ratios
(`(H_q,H_kv) ∈ {(4,4), (8,2), (32,4)}`) and `S_kv ∈ {64, 512, 2048}`.

**Result: a genuine, S_kv-dependent crossover, not a uniform win or
loss.** Measured on the same M4 hardware at `H_q=32, H_kv=4, D=128, nsg=2`
against the unpacked-redundant baseline (this document's fastest
measurement so far):

| S_kv | unpacked ms | two-pass ms | two-pass vs. unpacked |
|------|-------------|-------------|------------------------|
| 256  | 1.22 | 0.38 | **3.18x faster** |
| 1024 | 1.33 | 0.98 | **1.36x faster** |
| 2048 | 2.05 | 1.94 | **1.06x faster** |
| 3072 | 2.77 | 3.27 | 0.85x (slower) |
| 4096 | 3.63 | 4.41 | 0.82x (slower) |
| 8192 | 7.25 | 9.10 | 0.80x (slower) |
| 16384 | 13.89 | 17.44 | 0.80x (slower) |

The crossover (~S_kv 2048-3072 at this shape) holds directionally across
`(H_q,H_kv) ∈ {(32,4), (32,8), (8,2)}` — two-pass wins by 1.1-3.2x at
short/moderate context and loses by 0.7-0.85x at long context in every
ratio tested. The mechanism: the decode pass's own read-then-write cost
scales with `S_kv * H_kv * D` regardless of `heads_per_kv`, while the
*savings* (redundant on-the-fly decode arithmetic avoided) scale with
`S_kv * H_kv * D * (heads_per_kv - 1)`. At small `S_kv` the saved
redundant-decode ALU work outweighs the extra round-trip; at large `S_kv`
the fixed per-byte round-trip overhead — a real DRAM write plus a second
real DRAM read that redundant on-the-fly decode never pays at all —
dominates over the saved arithmetic, which is cheap relative to a byte
round-trip in the first place.

**Disposition:** not adopted for the same reason part 2 wasn't — this
repo's realistic decode targets (long-context KV caches, `S_kv` in the
thousands to tens of thousands) sit past the crossover, where two-pass is
a net loss. Kept as correct, tested code (same rationale as part 2:
available groundwork, not dead weight) rather than reverted, since the
short-context win is real and could matter for a caller with a small
context budget, and because `scalar_decode_once` in particular could be a
useful building block outside this specific attend pairing. Reproduction:
`pytest veloxquant_mlx/tests/metal/test_scalar_attend.py -v -k
"decode_once or two_pass"` for correctness; the crossover table prints
from `test_scalar_attend_two_pass_benchmark`.

**Why real-model end-to-end integration was not attempted, despite the
short-context win above being real.** All numbers in this document and
its addenda are synthetic microbenchmarks: a fresh `q`/`k`/`v` call per
measurement, decoding K/V from scratch every time. A real decode loop
does not look like this — `mlx_lm`'s standard `KVCache` materializes and
*retains* the dequantized fp16 `K_hat`/`V_hat` tensor across decode
steps, so per-step dequantization cost is already amortized to
effectively zero on the standard path. Any fused kernel that redoes
decode work (or, for the two-pass design, pays a fresh DRAM round-trip)
on every call is competing against a baseline that mostly isn't doing
that redundant work in the first place during real generation — a
materially different comparison than this document's isolated-call
benchmarks.

This is not a hypothesis — **this exact experiment has already been run
in this repo, on a real model, for the closest sibling kernel family**.
`veloxquant_mlx/cache/vecinfer_cache.py`'s `fused_sdpa` opt-in path pairs
a fused Metal decode+attend kernel (`metal/fused_sdpa.py`) with codebook
(VQ) index storage, structurally the same shape of integration this
document's kernels would need for KIVI. `patch_mlx_lm_for_fused_sdpa()`
(`veloxquant_mlx/metal/fused_sdpa.py:317-351`) is the dispatcher that
would route real `mlx_lm.generate()` calls to it — and its own inline
comment records the result: *"Profiling on Llama-3.1-8B showed that the
fused kernel cannot beat MLX SDPA on an already-materialized K_hat
tensor because the per-step dequant cost is amortized to zero by
mlx_lm's persistent cache buffer."* The dispatcher is coded as a
pass-through no-op today specifically because of this measured result —
not a stub awaiting completion.

Given a first-party, already-measured result on this exact failure mode
for a structurally identical integration, re-running it for
`scalar_predecoded_attend`/`scalar_fused_decode_attend` against KIVI was
judged unlikely to produce new information proportional to the
integration cost (wiring compressed-state storage into `KIVIKVCache`,
a monkeypatch dispatcher, and real-model benchmark plumbing — see
`docs/RVQ_PACKED_STORAGE_FINDINGS.md` for how much work the analogous
`turboquant_rvq` conversion took). If a future change alters the premise
— e.g. part 1's cross-layer batched dispatch changes what "amortized"
means by changing the call granularity itself, or a caller emerges that
genuinely cannot retain a materialized fp16 `K_hat` (the memory-bound
case `fused_sdpa`'s docstring already names as its own remaining
rationale) — this is the reference point to revisit, not a closed
question in principle.

## Recommendation

1. **Kernels already at or near the bandwidth roofline** (`kivi_group_quant_dequant`
   at 102%, `turboquant_scalar_dequantize` at 61%) are not worth further
   micro-optimization — the only way to speed them up is to move fewer
   bytes (smaller codes / narrower groups / avoiding materializing
   intermediates), not to tune the kernel's arithmetic.
2. **`scalar_fused_decode_attend`'s bottleneck is occupancy, not
   bandwidth or FLOPs** — the fix is architectural, not a kernel-internals
   tune. The obvious lever from the occupancy sweep: dispatch more
   threadgroups per call by processing multiple layers or multiple
   requests in one kernel launch (batching across what's currently
   separate per-layer Python-level calls), rather than trying to make a
   single (B=1, H=8, S_q=1) dispatch stream memory faster — there simply
   isn't enough independent work in that shape to fill the GPU, at any
   bandwidth. This is a concrete, scoped follow-up for a future issue, not
   attempted here (this issue is investigation-only, per its own framing).
3. **`turboquant_scalar_quantize` didn't clearly cross the memory-bound
   threshold even at 16M elements** (52–55%, vs. 61–102% for the other
   two bandwidth-confirmed kernels) — worth a follow-up at larger N to see
   whether it plateaus below the ceiling (suggesting real headroom, e.g.
   from its per-element centroid-scan loop not being fully hidden by
   memory latency) or simply needs more elements to amortize dispatch
   overhead the same way the others did. Not resolved here.

## Limitations

- **One machine, one GPU tier.** All numbers are from a base 10-core Apple
  M4, 24 GB — the same caveat the prefill roofline gives. Occupancy
  headroom in particular is core-count-dependent; a higher-tier Apple GPU
  (more cores) would likely show occupancy-limited kernels hitting the
  ceiling at even larger threadgroup counts than tested here.
- **No GPU-level profiling.** Metal System Trace / GPU performance
  counters (real occupancy %, actual achieved bandwidth per counter
  rather than inferred from wall-clock latency) weren't used — the
  bandwidth/occupancy story here is inferred from black-box latency
  measurements and the kernels' own dispatch-shape documentation, the
  same limitation the prefill roofline's Step 4 flagged for its own
  root-cause hypothesis.
- **`turboquant_scalar_quantize`'s trend at N > 16M is untested** (see
  Recommendation #3).
- **Synchronization-bound behavior wasn't isolated with a dedicated
  experiment** (e.g. varying `NSG_C` and measuring barrier count vs.
  throughput directly) — the conclusion that no kernel here is
  synchronization-bound rests on reading the source's barrier structure,
  not a targeted measurement.

## Reproduction

```
python scripts/kv_kernel_roofline_bench.py
```

Deterministic inputs (fixed `np.random.default_rng` seeds); re-run 2-3x
and compare, per the prefill roofline's own observed run-to-run variance
on this hardware.
