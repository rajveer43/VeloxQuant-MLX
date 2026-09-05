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

## Addendum: cross-layer batched decode-attend dispatch (issue #307, part 1)

This closes the one lever Recommendation #2 (below) named as unblocked and
not yet attempted at the time this document was first written: **dispatch
more threadgroups per call by batching the independent per-layer
`scalar_fused_decode_attend` calls a real decode step already makes (one
per transformer layer, typically 28-80 layers) into a single kernel
launch**, rather than trying to make a single `(B=1, H=8, S_q=1)` dispatch
stream memory faster with no more independent work to give it.

**Mechanism.** `scalar_fused_decode_attend_batched`
(`veloxquant_mlx/metal/_scalar_attend.py`,
`veloxquant_mlx/metal/src/scalar_affine_attend_batched.metal`) adds a new
outermost `NL` (num_layers) grid axis on top of the existing
`(sq_idx, hkv_idx, b_idx)` derivation, so threadgroup count becomes
`NL * B * H_kv * S_q` instead of `B * H_kv * S_q`. The caller pre-stacks
each layer's tensors along a new leading axis (`mx.stack`); the kernel
does not gather scattered per-layer arrays itself. This changes nothing
about the math, the memory layout, or the per-(layer, batch, kv-head,
query-position) work one threadgroup does — same total bytes moved, same
total FLOPs, summed over layers either way. Verified bit-identical (not
just within tolerance) against calling the single-layer kernel `NL` times
in a loop and stacking the outputs, for `NL ∈ {1, 4, 32}`
(`test_scalar_attend_batched_parity_vs_single_layer_loop`), plus numpy
reference parity across four `(H_q, H_kv)` ratios and an adversarial
`NL=3, B=2` combined-indexing test to rule out a swapped layer/batch
stride order.

**Result: a consistent, positive speedup across every shape tested — the
occupancy hypothesis holds.** Measured on the same M4 hardware, `D=128`,
`nsg=2`, `group_size=32`:

| H_kv | H_q/H_kv | S_kv | NL | sequential ms | batched ms | speedup |
|------|----------|------|----|---------------:|-----------:|--------:|
| 2 | 1 | 128 | 32 | 0.62 | 0.30 | **2.06x** |
| 2 | 1 | 128 | 80 | 1.01 | 0.44 | **2.28x** |
| 2 | 1 | 2048 | 32 | 2.69 | 2.11 | **1.27x** |
| 2 | 1 | 2048 | 80 | 6.12 | 3.28 | **1.87x** |
| 2 | 1 | 16384 | 32 | 53.88 | 15.66 | **3.44x** |
| 2 | 1 | 16384 | 80 | 107.75 | 23.89 | **4.51x** |
| 2 | 8 | 128 | 32 | 1.24 | 0.97 | **1.28x** |
| 2 | 8 | 2048 | 32 | 11.80 | 11.28 | **1.05x** |
| 2 | 8 | 16384 | 32 | 174.67 | 88.36 | **1.98x** |
| 2 | 8 | 16384 | 80 | 349.41 | 218.67 | **1.60x** |
| 8 | 1 | 128 | 32 | 0.98 | 0.55 | **1.79x** |
| 8 | 1 | 2048 | 32 | 7.27 | 4.76 | **1.53x** |
| 8 | 1 | 16384 | 32 | 115.57 | 35.79 | **3.23x** |
| 8 | 1 | 16384 | 80 | 289.66 | 83.91 | **3.45x** |
| 8 | 8 | 128 | 32 | 3.37 | 2.82 | **1.20x** |
| 8 | 8 | 2048 | 80 | 95.91 | 93.25 | **1.03x** |
| 8 | 8 | 16384 | 32 | 501.15 | 229.55 | **2.18x** |
| 8 | 8 | 16384 | 80 | 1247.28 | 322.54 | **3.87x** |

Every cell tested is a win — no null or negative result was found,
unlike part 2 (GQA head-packing) and the two-pass spike above. The win
grows with `S_kv` (largest at `S_kv=16384`, up to 4.5x) and shrinks
towards ~1.0-1.3x at small `S_kv` combined with a high `H_q/H_kv` ratio
(e.g. `H_kv=2, ratio=8, S_kv=2048`), where the sequential path's own
threadgroup count (`H_kv * heads_per_kv`-scaled work per layer, before
batching) is already large enough that batching adds proportionally less
headroom, and fixed per-dispatch overhead is a larger share of both
numbers' latency. Achieved bandwidth in the batched case reached up to
~40% of the calibrated peak at the best shapes tested — still well short
of memory-bound, consistent with occupancy (not bandwidth) remaining the
governing constraint even after batching, just a less severe one.

**The stacking cost is real and must be netted against the win.**
`mx.stack`-ing `NL` independent per-layer arrays into the batched layout
is not free — measured standalone (not folded into the batched kernel's
own timing above), at `H_kv=4, ratio=8`:

| S_kv | NL | mx.stack ms |
|------|----|--------------:|
| 128 | 32 | 1.19 |
| 128 | 80 | 1.01 |
| 2048 | 32 | 25.38 |
| 2048 | 80 | 5.34 |
| 16384 | 32 | 14.91 |
| 16384 | 80 | 36.41 |

This cost is noisy and does not cleanly scale with `NL * S_kv` the way
the kernel's own bytes-moved does (compare `S_kv=2048, NL=32` at 25.4ms
against `S_kv=2048, NL=80` at 5.3ms) — consistent with `mx.stack`'s cost
being dominated by MLX's own graph-construction/dispatch overhead for a
list of `NL` separate concatenation inputs rather than by the bytes
actually copied, and warrants a dedicated investigation if this technique
is pursued further. At the shapes where it is large (tens of ms), it can
erode or exceed the kernel-level win shown above once netted in — e.g. at
`S_kv=2048, NL=32`, a ~0.6ms kernel-level saving (2.69ms → 2.11ms in the
first table, same shape family) is dwarfed by a 25ms stacking cost, were
that stacking paid on every decode step.

**Whether that stacking cost is paid once or every step depends entirely
on the KV-cache implementation, and here the news is unfavorable for an
easy end-to-end win.** Verified by reading `mlx_lm.models.cache` directly
rather than assumed: `mlx_lm`'s stock `KVCache` is instantiated **one
independent object per layer**, held in a plain Python list (e.g.
`[KVCache() for _ in range(len(self.model.layers))]`, `cache.py`'s own
default `make_prompt_cache`), each with its own independently-growing
`.keys`/`.values` buffer. There is no shared, layer-stacked backing
buffer anywhere in the stock cache — so a real integration built on it
would need to `mx.stack()` every layer's state **fresh on every decode
step**, not once at cache-construction time. That makes the stacking cost
above a **per-step tax**, not a one-time layout change, which is the same
"amortization" failure mode the SIMD-shuffle addendum's real-model section
already found for the sibling `fused_sdpa` kernel — just arriving via a
different mechanism (stacking overhead here, vs. redundant dequant there).

**Disposition: the kernel-level technique is a real, verified win in
isolation and is shipped as tested, documented code — but it is NOT
claimed to speed up real `mlx_lm` generation today.** Closing the gap
would require either (a) a custom layer-stacked cache implementation that
keeps per-layer K/V contiguously stacked as it grows (avoiding the
per-step `mx.stack`, at the cost of a bespoke cache class diverging from
`mlx_lm`'s stock `KVCache`), or (b) amortizing the one-time stack cost
across enough decode steps between cache reallocations that the per-step
tax becomes negligible relative to the per-step kernel win — neither is
attempted here; both are scoped future work, exactly as this document's
prior addenda flagged real end-to-end integration as a separate,
larger effort rather than assumed. Reproduction:
`python benchmark_scripts/benchmark_crosslayer_decode_batch.py` (full
`H_kv × ratio × S_kv × NL` grid spanning current open-weight model
depths — the run captured in the tables above used a representative
subset for tractable turnaround, not the full grid, on a single session);
correctness: `pytest veloxquant_mlx/tests/metal/test_scalar_attend.py -v
-k batched`.

**No claim, here or anywhere else in this repo, is made that
`scalar_fused_decode_attend_batched` "outperforms" MLX's own SDPA, any
other library's kernel, or any other kernel in this repo.** The only
valid claim is the one measured above: a same-total-work, before/after
threadgroup-count comparison against this repo's own existing
single-layer dispatch shape, which is exactly what the tables report.

## Addendum: cross-layer batching cannot reach real single-request decode, but the underlying (single-layer) kernel measurably speeds up real multi-request decode

The addendum above measured `scalar_fused_decode_attend_batched` (the new
`NL`-axis kernel) in isolation and flagged real `mlx_lm` end-to-end
integration as unresolved. Attempting that integration surfaced a hard
**structural** blocker, not an engineering gap: reading `mlx_lm`'s own
model code (`TransformerBlock.__call__`, every `models/*.py`) shows layer
`L+1`'s attention input is `q_proj(norm(layer_L_output))`, where
`layer_L_output` is layer `L`'s **full** block output — attention,
residual, MLP, residual — not just its attention output. There is no
reordering of a standard pre-norm decoder transformer that lets multiple
layers' attention be grouped into one dispatch while still computing the
same model; doing so would require changing what the model computes, which
fails this repo's correctness bar. **Cross-layer batched decode-attend
therefore cannot speed up real single-request decode latency on any
standard transformer — this is true regardless of kernel quality,
occupancy tuning, or any future optimization of the batching mechanism
itself.** This is a stronger and more useful conclusion than "not yet
integrated": it closes the cross-layer half of Recommendation #2 below as
a dead end for single-request serving, not a pending follow-up.

**The other half of Recommendation #2 — batching across concurrent
*requests* rather than layers — has no such blocker, and was verified for
real on a real model.** Concurrent requests are independent by
construction (no residual-stream dependency between them), and the
*existing, already-shipped* `scalar_fused_decode_attend` (not the new
batched kernel — the original single-layer one) already carries a `B`
axis in its dispatch grid (`n_tg = B * H_kv * S_q`). No cache in this repo
previously routed real `mlx_lm` generation through this kernel
(`KIVIKVCache` uses `kivi_group_quant_dequant` + standard SDPA instead;
`fused_sdpa`'s dispatcher for the sibling VecInfer kernel is a documented
no-op for the amortization reason given in the SIMD-shuffle addendum
above). A minimal purpose-built cache (`_ScalarAttendKIVICache` in
`benchmark_scripts/benchmark_real_model_scalar_attend.py`) was built to
test this directly: it stores KIVI-quantized (2-bit, group_size=32) K/V
codes incrementally (quantizing only newly-aged `group_size`-aligned
blocks each step, not re-quantizing the full history — an earlier version
of this script did the latter and added ~25ms/step of pure quantization
overhead across 36 layers, which would have silently contaminated the
result with a test-harness artifact rather than measuring the kernel; the
fix mirrors `KIVIKVCache._quantization_boundary()`'s existing incremental-
flush discipline). `mlx_lm.models.base.scaled_dot_product_attention` was
monkeypatched — at both its canonical definition and the loaded model's
own module, since `from .base import scaled_dot_product_attention` binds
a plain name at each model module's import time, not a live reference
(patching only `mlx_lm.models.base` after `load()` has no effect,
confirmed empirically: 0 routed calls until both bindings were patched) —
so that decode-shape (`S_q == 1`) attention over this cache dispatches to
either (a) dequantize-to-fp16 then standard MLX SDPA (the baseline — what
a real KIVI-style cache does today) or (b) `scalar_fused_decode_attend`
directly against the quantized codes (fused), with **both arms verified
to produce bit-identical greedy-decoded token sequences** before any
timing was trusted.

Measured on `mlx-community/Qwen3-4B-4bit` (36 layers, `H_q=32, H_kv=8`,
real M4 hardware, `prompt_len=256`, 30 decode steps, real end-to-end
tokens/sec including embeddings/MLPs/o_proj/sampling/per-step
quantization — not an isolated kernel call):

| B (concurrent requests) | TTFT baseline | decode tok/s baseline | TTFT fused | decode tok/s fused | speedup |
|---|---|---|---|---|---|
| 1  | 0.79s | 17.1  | 0.75s | 25.5  | **1.50x** |
| 4  | 3.01s | 25.3  | 3.08s | 48.1  | **1.90x** |
| 16 | 12.20s | 34.3 | 12.31s | 108.1 | **3.16x** |
| 32 | 25.36s | 39.1 | 27.86s | 149.9 | **3.83x** |

**This is a real, positive, end-to-end result on a real model** — not a
synthetic microbenchmark — and the speedup growing with `B` (1.50x → 3.83x)
is exactly the occupancy signature this document's synthetic `B`-sweep
predicted (more concurrent requests → more threadgroups dispatched → more
of the fused kernel's headroom realized), now confirmed through a real
forward pass rather than isolated kernel calls. TTFT is essentially
unaffected in both arms, as expected — prefill uses `S_q > 1` and never
routes through this decode-only kernel; the small TTFT increase at `B=32`
(25.36s → 27.86s) is run-to-run variance from sharing the machine across
back-to-back large-batch prefills, not a fused-kernel effect (prefill
computation is identical between arms).

**Caveats, stated plainly rather than glossed over:**
- `_ScalarAttendKIVICache` is a minimal benchmark harness, not a
  shippable cache: it has no fp16 residual window (real KIVI keeps recent
  tokens exact via `residual_length`; this quantizes as soon as
  `group_size` tokens are available), so its *output quality* is not
  representative of production KIVI — greedy decode degenerates
  (observed repeating tokens after enough steps) faster than the real
  `KIVIKVCache` would. This is irrelevant to the *timing* comparison,
  since both arms decode from the identical (lossy) quantized state and
  were verified to produce identical tokens — but the numbers above
  should not be read as "KIVI is 1.5-3.8x faster in production," only as
  "the fused kernel is 1.5-3.8x faster than dequant+SDPA for the *same*
  quantized state, on a real model."
- Only one model, one machine, one shape family (`prompt_len=256`,
  `group_size=32`, `bits=2`) was tested. The isolated per-layer
  comparison at this shape (`S_kv≈64, B=4`) showed the fused kernel at
  only ~1.15x over a **plain fp16 KVCache with no quantization at all**
  (vs. 2.57x over the KIVI-dequant baseline) — meaning most of this
  result's headroom comes from avoiding the dequant materialization cost
  specifically, not from the attend computation itself being dramatically
  faster in absolute terms. A model/shape where dequant cost is smaller
  relative to attend cost (larger `D`, different `bits`) could show a
  smaller margin.
- This does not resolve whether a *production* multi-request serving
  integration (proper request scheduling, padding/masking for
  variable-length sequences within a batch, KV-cache eviction under
  concurrent load) would preserve this win — this benchmark uses
  identical-length prompts and no eviction, real serving traffic differs
  on both counts.

**Disposition:** the multi-request half of Recommendation #2 is
confirmed as a real, positive lever on a real model, using only the
already-shipped (non-batched) `scalar_fused_decode_attend` kernel — no
new kernel work was needed, only the cache/dispatch wiring this repo
had not yet built for this kernel family. Building a production-grade
version (a real cache with a residual window, wired through
`KVCacheBuilder`, tested against variable-length concurrent requests) is
scoped future work, not attempted here.

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
   requests in one kernel launch, rather than trying to make a single
   (B=1, H=8, S_q=1) dispatch stream memory faster — there simply isn't
   enough independent work in that shape to fill the GPU, at any
   bandwidth. **Both halves of this lever are now resolved, with opposite
   outcomes.** The cross-layer half was implemented, measured as a real
   kernel-level win (1.0-4.5x, "Addendum: cross-layer batched
   decode-attend dispatch"), and then shown to be a **structural dead
   end for real single-request serving** — a standard transformer's
   residual stream makes layer L+1's attention depend on layer L's full
   block output, so real layers' attention can never be grouped into one
   dispatch without changing the model's output (see the addendum
   immediately above). The multi-*request* half has no such blocker and
   was **confirmed as a real, positive win on a real model**: routing
   `mlx_lm` decode-step attention through the existing (non-batched)
   `scalar_fused_decode_attend` on Qwen3-4B-4bit measured **1.50x-3.83x**
   real end-to-end decode tokens/sec, growing with concurrent request
   count `B ∈ {1,4,16,32}` — see the same addendum for the full table,
   caveats, and the apples-to-apples methodology (both arms verified to
   produce bit-identical tokens before timing was trusted). A
   production-grade integration (real residual-window cache, wired
   through `KVCacheBuilder`, variable-length concurrent requests) remains
   future work.
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
python benchmark_scripts/benchmark_crosslayer_decode_batch.py       # cross-layer batching addendum
python -m benchmark_scripts.benchmark_real_model_scalar_attend      # real-model multi-request addendum
```

The real-model script must be run as a module (`python -m
benchmark_scripts...`, not `python benchmark_scripts/....py`) — running it
by path inserts the script's own directory at `sys.path[0]` ahead of the
repo root, which can resolve `veloxquant_mlx` to a stale installed copy
instead of the repo source if one exists in the active environment's
`site-packages` (encountered directly while building this addendum; see
that script's own import guard).

Deterministic inputs (fixed `np.random.default_rng` seeds); re-run 2-3x
and compare, per the prefill roofline's own observed run-to-run variance
on this hardware.
