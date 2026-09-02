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
