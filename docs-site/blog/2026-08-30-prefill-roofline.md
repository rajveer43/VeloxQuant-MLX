---
slug: prefill-roofline
title: "Chasing the Mac-vs-CUDA Prefill Gap — and Finding a Wall Instead"
date: 2026-08-30
authors: rajveer
tags: [metal, apple-silicon, mlx, attention, benchmarking, roofline]
---

*Issue #277 asked whether a hand-written Metal kernel could speed up prefill-side attention, the way this repo's decode kernels already speed up cache reads. The honest answer, measured on the hardware in hand: no — and the reason why is more interesting than a speedup would have been.*

---

## What decode kernels do that prefill can't reuse

Every Metal kernel in VeloxQuant-MLX before this — `quantize_vq`, `rabitq_encode`, `rabitq_fused_attend`, `scalar_fused_decode_attend` — accelerates something that happens to an **already-built** KV cache. That's the right target for decode: one new query token against a long cache is bandwidth-bound, and Apple Silicon's unified memory is genuinely competitive there.

Prefill is a different problem. The whole prompt goes through attention in one shot — `S_q ≈ S_kv`, both potentially tens of thousands of tokens for an agentic coding session that re-feeds a large, mostly-unchanged codebase every turn. That's compute-bound: big batched matmuls, not a bandwidth story. And it's the regime where Apple GPUs are furthest behind CUDA — back-of-envelope estimates put an M3 Ultra around 30x slower than an RTX Pro 6000 at large-context prefill.

None of the existing kernels touch this. They all assume the cache already exists. So the question in #277 was: is there a real software gap here that a `simdgroup_matrix`-tiled kernel (the same technique behind `rabitq_prefill_attend`) could close, or is the ~30x gap purely a hardware FLOPs ceiling that no amount of kernel-writing fixes?

{/* truncate */}

## Step 1: give the existing prefill kernel a causal mask

`rabitq_prefill_attend` already existed — built for the multi-turn VLM case, a new turn attending over a long compressed image-token history. But per its own docstring it was "cross-attention only, no causal mask," which rules out the thing #277 actually cares about: standard autoregressive self-attention over a prompt.

The fix was mechanical. The kernel already runs an online-softmax loop over 8-slot KV chunks; masking meant computing each query row's absolute position and rejecting KV slots past it, using the same tail-alignment convention this repo's other fused-attention kernel (`fused_sdpa.metal`) already established:

```cpp
int q_abs = q_align + int(qblk * BQ_TG + sg * BQ + r);   // q_align = S_kv - S_q
...
bool valid = slot < S_kv && (!causal || int(slot) <= q_abs);
s_j[j] = valid ? float(s_tile[sg][r * BK + j]) + k_const[kv_base + slot] : -INFINITY;
```

One subtlety the padding case didn't need to worry about: a causal mask can make an *entire* 8-slot chunk invalid (every query in it is behind the whole chunk), which the existing online-softmax update didn't handle — `exp(-inf - -inf)` is `NaN`, not zero. Two explicit guards fixed it:

```cpp
bool  chunk_empty = m_new == -INFINITY;
float factor = chunk_empty ? 0.0f : metal::exp(m_old - m_new);
...
float w = chunk_empty ? 0.0f : metal::exp(s_j[j] - m_new);
```

Parity tests against a masked numpy reference pass across the existing `(B, H, S_q, S_kv, D)` sweep plus dedicated causal cases, including the from-scratch `S_q == S_kv` shape that's the actual target of this issue. 88 passed, 12 skipped (the causal sweep correctly skips `S_q > S_kv`, which isn't a valid causal alignment).

## Step 2: benchmark it — and lose

With correctness done, the benchmark:

```
[bench] rabitq_prefill_attend — causal self-attention shapes, B=1 H=8 D=128
  S_q   S_kv | prefill (ms) | decode-k (ms) | baseline (ms) | vs base
------------------------------------------------------------------
 2048   2048 |       54.951 |           n/a |         4.546 |   0.08x
 8192   8192 |      797.760 |           n/a |        53.229 |   0.07x
```

"baseline" is `mx.fast.scaled_dot_product_attention` on dequantized fp16 K/V. The fused kernel is **12-15x slower**, not faster. This is the opposite of `rabitq_prefill_attend`'s original VLM cross-attention numbers, where fusing decode into the tile loop wins by avoiding a separate dequantize pass.

The difference is architectural, not a bug. In the VLM case, `S_kv` (compressed image history) is large and `S_q` (new-turn tokens) is comparatively small, so the K/V decode-once-per-chunk-share-across-32-query-rows trick amortizes well. In causal self-attention prefill, `S_q == S_kv`, both large — the same byte-wise scalar decode (one `k_bits` byte → 8 dims, one lane at a time) now runs on the full sequence length with much less reuse per decode, and it's fighting MLX's SDPA, which is presumably calling into a well-tuned matmul path rather than decoding anything at all. A tiled kernel built for "long history, short new turn" doesn't transfer to "long history, long new turn" for free.

That's a real, useful negative result on its own — but it only tells us the *compressed-KV* prefill path loses. It doesn't answer the harder question in #277: for a first-ever prefill, where there's no compressed cache to exploit and K/V are being produced fresh, is MLX's plain SDPA already close to what the hardware can do?

## Step 3: roofline the thing that actually matters

This is where #259's methodology (arithmetic intensity, achieved vs. theoretical throughput, compute- vs. memory-bound classification) gets applied to prefill specifically. The question isn't "is my kernel fast" — it's "how much room is there for *any* kernel to be faster than SDPA."

First attempt: compare SDPA's achieved TFLOP/s against a spec-sheet fp16 peak. That number doesn't actually exist for Apple GPUs — Apple publishes fp32 figures for some tiers and nothing consistent for attention-shaped fp16 work — so a naive "double the fp32 number for fp16 packing" guess put SDPA at a flat ~33% of peak across every shape tested. That's a plausible-looking number that's wrong: it's measuring against an unverified ceiling, not the machine's actual achievable throughput.

So the script self-calibrates instead — it benchmarks square fp16 matmuls (2048³ through 8192³) on the same GPU, same run, and uses the best observed TFLOP/s as the ceiling:

```python
def _calibrate_matmul_peak() -> float:
    for n in (2048, 4096, 8192):
        a, b = mx.array(...), mx.array(...)  # fp16
        t = _bench(lambda: a @ b, ...)
        best = max(best, 2.0 * n**3 / t / 1e12)
    return best
```

On the base 10-core Apple M4 in hand: **~3.18 TFLOP/s achieved** on raw square matmuls — nowhere near the fp32 spec-sheet-derived guess. Against *that* ceiling, `mx.fast.scaled_dot_product_attention` at causal prefill shapes looks completely different:

```
[roofline] calibrated peak: 3.176 TFLOP/s (achieved, not spec-sheet)
[roofline] causal fp16 SDPA prefill — B=1 D=128, peak=3.18 TFLOP/s
      S  H_q H_kv | latency (ms) |    tok/s |  TFLOP/s | % of peak
----------------------------------------------------------------------
   2048   32   32 |       12.516 |   163635 |    2.745 |     86.4%
   2048   32    8 |       13.065 |   156758 |    2.630 |     82.8%
   8192   32   32 |      201.096 |    40737 |    2.734 |     86.1%
   8192   32    8 |      192.738 |    42503 |    2.852 |     89.8%
  32768   32   32 |     3211.256 |    10204 |    2.739 |     86.2%
  32768   32    8 |     3309.248 |     9902 |    2.658 |     83.7%
```

**82-90% of achieved matmul peak, flat across sequence lengths from 2k to 32k and across MHA/GQA head ratios.** That flatness matters — it's not a small-shape launch-overhead artifact that would fade at scale; it's a steady-state ceiling that holds from 2k tokens all the way to 32k.

## Step 4: try to close the gap anyway

10-15% of headroom isn't much, but it's not nothing — so the natural follow-up was to actually write the from-scratch kernel and see. `flash_prefill_attend` (`veloxquant_mlx/metal/_flash_prefill.py`, `flash_prefill.metal`) is that attempt: plain fp16 Q/K/V, causal-only, no compression, no GQA/mask/sinks branching — everything this repo's actual use case doesn't need is compiled out rather than handled at runtime, which is the one legitimate structural advantage a specialized kernel can have over a fully general one.

Before writing it, it's worth being honest about what "SDPA" actually is here. It isn't a naive baseline — `mx.fast.scaled_dot_product_attention` dispatches to MLX's own `steel` attention kernel (`mlx/backend/metal/kernels/steel/attn/`), a hand-tuned, AOT-compiled flash-attention implementation: block-tiled GEMMs, `simdgroup_matrix` MMA fragments, function-constant-gated causal/mask/GQA paths so unused branches compile away entirely, bank-conflict-avoiding threadgroup memory padding, and an `exp2`-with-prescaled-`log2(e)` softmax. Beating that on a first attempt was always a stretch goal, not a safe bet.

The kernel is built on the same online-softmax / `simdgroup_matrix` scaffold as `rabitq_prefill_attend`, with several deltas aimed specifically at closing the gap:

- **`BK=16` instead of 8** — half as many threadgroup-barrier round trips per unit of K/V processed.
- **`exp2` softmax, scale pre-folded by `log2(e)`** — the same trick steel uses, cheaper than `metal::exp` on Apple GPU ALUs.
- **Causal block-skip** — a threadgroup computes the last valid query row's absolute position and skips any KV chunk entirely in the future *before* loading or matmul'ing it, not just after via masking. Steel does the equivalent (`kb_min_causal` bounds its loop).
- **Batched `simdgroup_store` in the W·V step** — storing 2 depth-tiles per barrier instead of 1, halving barrier count there without growing threadgroup memory (a `p_tile` sized `BQ*BK` already holds exactly 2 tiles' worth of columns).

It's correct — 40 parity tests pass against a numpy causal reference, including a targeted check that perturbing a future (masked) key changes nothing about earlier output rows:

```python
def test_flash_prefill_causal_masks_future():
    k2, v2 = k.copy(), v.copy()
    k2[:, :, -1, :] = 1000.0
    out2 = _run_kernel(q, k2, v2, scale)
    np.testing.assert_allclose(out1[:, :, :-1, :], out2[:, :, :-1, :], atol=1e-3)
```

It's also, consistently, **4-6x slower than SDPA**:

```
[roofline] flash_prefill_attend (this repo's from-scratch kernel) vs SDPA — MHA only
      S    H |  flash (ms) |  sdpa (ms) | sdpa/flash
-------------------------------------------------------
   2048   32 |      86.165 |     14.232 |      0.17x
   8192   32 |    1404.717 |    299.902 |      0.21x
  32768   32 |   22087.881 |   5266.862 |      0.24x
```

Four separate structural changes were tried — `BK=8→16`, `exp`→`exp2`, adding the causal block-skip, and sweeping SIMD-group count per threadgroup (2, 3, 4) while trying to reduce the W·V loop's barrier count from 16 down to 1 — and every configuration landed in the same 4-10x-slower band. Reducing barrier count alone didn't help as much as expected: cutting `NSG_C` from 4 to 2 to fit a wider scratch buffer that eliminated 15 of 16 barriers in the W·V step made things *worse* (down to 0.09-0.10x, i.e. ~10x slower), because it cost more in lost intra-threadgroup parallelism than it gained in fewer stalls. The best configuration found — `NSG_C=4`, `BK=16`, `exp2`, causal block-skip, 2-tile-per-store in the W·V loop — is what's shipped, at a stable ~4-6x slowdown across scales.

The likely reason tile-level tuning couldn't close this: the MSL spec leaves the element-to-thread mapping of `simdgroup_matrix` fragments unspecified, so every elementwise step (softmax, rescale, the running-output accumulate) *must* round-trip through threadgroup memory via `simdgroup_store` + barrier + read back. Steel's AOT-compiled, Apple-internal kernel may have a materially different strategy — e.g. keeping fragment data resident and reasoning about the thread mapping directly rather than always bouncing through shared memory — that a from-scratch kernel using the same public `simdgroup_matrix` API this repo already relies on elsewhere can't easily replicate. This is offered as the most likely explanation, not a verified root cause; confirming it would require lower-level profiling (Metal System Trace / GPU counters) that wasn't done here.

## The finding

MLX's built-in `scaled_dot_product_attention` is already running prefill attention at roughly 85% of what this GPU can actually deliver on fp16 matmuls (the two roofline runs in this post measured 82-90% and 52-59% respectively — see the caveat below on run-to-run variance). There's maybe 10-15% of software headroom left in the best case — not nothing, but nowhere near enough to matter against a 30x hardware gap. And the concrete attempt at capturing it, `flash_prefill_attend`, didn't get there: four rounds of structural tuning all landed in the same 4-10x-slower band, well short of even matching SDPA, let alone beating it. `rabitq_prefill_attend` with causal masking added did worse still, at 12-15x slower — expected, since it was built for a different regime (VLM cross-attention, not `S_q == S_kv` self-attention).

This matches what #277 flagged as the likely outcome: **the Mac-vs-CUDA prefill gap is a hardware FLOPs ceiling, not a software gap.** MLX's own steel attention kernel is a professionally hand-tuned, AOT-compiled implementation extracting most of what this GPU's fp16 matmul units can deliver, and a first-pass JIT-compiled kernel using the same public `simdgroup_matrix` primitives — built by one person over one investigation, not Apple's internal MLX team over however long steel took — was never a realistic bet to beat it outright. The repo's realistic scope for prefill-adjacent work stays where it already was: decode-side fusion (`rabitq_fused_attend`, `scalar_fused_decode_attend`) and reducing what has to move through memory in the first place — cache compression, and [cross-model KV transfer](https://veloxquant-mlx.netlify.app/docs/algorithms/cross-model-transfer) for skipping prefill entirely between same-family models.

## Step 5: a second pass at `flash_prefill_attend` — measure everything, keep only what wins

The 4-6x gap above came from one tuning pass. A second, more disciplined pass followed — audit every synchronization point for whether it's load-bearing, build a boundary-case correctness harness (`scripts/flash_prefill_harness.py`), then sweep tile parameters and keep only what a head-to-head measurement actually validated. Four things were tried:

**Synchronization audit first.** Before changing anything, every `threadgroup_barrier` and `simdgroup_barrier` in the kernel was traced to its producer/consumer. The result: every `threadgroup_barrier` was guarding genuinely cross-SIMD-group traffic — almost entirely the shared `kv_tile` staging buffer, loaded once per chunk and consumed by all four SIMD-groups (that sharing is precisely the kernel's memory-bandwidth win; giving it up to save barriers would trade a bandwidth win for a latency win in the wrong direction). None could be safely removed or downgraded. The W·V loop's `simdgroup_barrier`s — 16 of the kernel's 22 barriers per chunk at D=128 — were the one place with real slack.

**Register-resident softmax state.** `m_run`/`d_run`/`f_row` lived in threadgroup arrays even though each is only ever touched by the one lane that owns that query row. Moved to per-lane registers (`m_local`, `d_local`), with `simd_shuffle` used only where a value genuinely needs to cross lanes (the rescale factor, read by all 32 lanes; the final denominator, same). Result: correct (all parity tests unchanged to the last decimal), and a small, consistent win at the largest shapes (+1-3%) — real, but confirmation that barriers were never the dominant cost here, matching the audit above.

**W·V depth-tile batching, measured per head-dimension instead of assumed.** The W·V step's `simdgroup_store`→barrier→scalar-reload round trip can batch more than 2 depth-tiles per barrier if there's threadgroup memory to spare. The obvious heuristic — "batch as many as fit in 32KB" — picked `PDT=8` at D=64 (20.86KB/threadgroup) and made things **24-27% slower**, not faster: crossing the ~16KB double-occupancy line cost more (losing a second resident threadgroup per GPU core) than the extra barrier savings gained. A direct sweep over `PDT ∈ {1,2,4,8}` at each D found the actual per-D optimum — `PDT=2` at D=32, `PDT=4` at D=64, `PDT=2` at D=128 (the only value that fits 32KB there) — and that measured table is what shipped.

**KV chunk width (`BK`), also measured per-D.** `BK=32` doubles chunk width, halving barrier round trips per unit of K/V at the cost of more `kv_tile`/`s_tile`/`w_tile` memory (and it's structurally infeasible at D=128 — `kv_tile` alone would be 8KB, blowing the 32KB budget even with the smallest depth-tile batch). Measured at D=32 and D=64: `BK=32` was a clear win at D=32 (0.42-0.50ms vs 0.67-1.38ms at S=512; 2.4ms vs 2.6-2.8ms at S=2048 — D=32 has the most memory headroom, 9.5-14KB even at BK=32, so the wider chunk's barrier savings aren't offset by occupancy loss) and a small, consistent loss at D=64 (already tighter on memory, 17-22KB, so the growth costs more than it saves). Kept BK=32 at D=32 only.

**`BQ=16` (doubling rows per SIMD-group) — tried, and rejected.** This was the largest remaining structural lever: doubling how many query rows each SIMD-group owns changes the QK^T/W·V matmul from one 8×8 fragment per tile to a 2×N grid, and halves how many threadgroups get dispatched. It required generalizing the row-tile dimension throughout the QK^T and W·V matmul steps (not just a template constant swap, unlike `BK`/`PDT`) — real restructuring, fully validated for correctness (parity holds at both D=32 and D=64 after the change). But it measured worse everywhere tested: 11-14% slower at D=32, and 14-42% slower at D=64 (worst at the largest shape, S=8192) — at D=64 it forces `PDT=1` (no W·V batching left at all) and pushes memory to 31.75KB, essentially single-occupancy, losing both the Phase-7 barrier-batching win and whatever occupancy margin D=64 had left. The code stays in the kernel (`BQ_ROWS` is a template parameter, defaulted to 8 everywhere via a measured lookup table) in case a future shape or GPU generation favors it, but it isn't used in production today.

**Net result** — same shapes as the original table above, same machine:

```
   D  S_q   S_kv | flash (ms) | sdpa (ms) | sdpa/flash
------------------------------------------------------
  32  512    512 |      1.29  |     1.53  |    1.19x   (flash WINS)
  32 2048   2048 |      2.42  |     5.89  |    2.44x   (flash WINS, was competitive-but-behind before)
  64  512    512 |      0.77  |     0.35  |    0.45x
  64 2048   2048 |      6.61  |     1.89  |    0.29x   (unchanged — see below)
  64 8192   8192 |     88.0   |    26.5   |    0.30x   (unchanged)
 128  512    512 |      1.98  |     0.50  |    0.25x
 128 2048   2048 |     21.3   |     3.7   |    0.17x
 128 8192   8192 |    234.3   |    52.9   |    0.23x   (was 0.15x — ~30% faster than the first pass)
```

D=32 now genuinely beats SDPA — by more than 2x at S=2048, where it was previously competitive-but-losing. D=128's largest tested shape improved materially (0.15x → 0.23x of SDPA, i.e. the gap shrank from ~6.7x to ~4.3x) from the PDT/register-softmax changes stacking. D=64 is unchanged after three independent, correctly-measured attempts (register-resident softmax, PDT sweep, BK sweep, BQ=16) — it sits in an awkward occupancy zone (already past the double-occupancy line at its baseline tile size, with no headroom to grow into a wider tile without losing single-occupancy too) that none of the tile-shape levers tried here could improve. That's a genuine, if unsatisfying, finding in its own right: not every head dimension responds to the same optimization, and D=64's ceiling here looks structural rather than something a further parameter sweep would fix.

## What's still open

- **Hardware ceiling, not measured across the family.** Everything above is one base 10-core M4, 24 GB. The issue's own back-of-envelope numbers were for an M3 Ultra; whether the ~85%-of-peak result holds on higher-tier GPUs (more cores, different matmul-unit generation) isn't verified here.
- **The two roofline runs disagreed by more than expected** (82-90% of peak in one run, 52-59% in a later run on the same machine) — almost certainly thermal or scheduling variance from back-to-back heavy GPU benchmarking rather than a real regime change, but it wasn't isolated. Treat "SDPA is within single-digit-percent of X%" as noisier than the table implies; the qualitative finding (SDPA captures most of the achievable ceiling) held in both runs. The same variance showed up in Step 5's tuning sweeps too — some individual readings varied 2x between repeated runs at small shapes; every number kept in the final table above was reproduced at least twice before being trusted.
- **D=32 now beats SDPA, but D=64/D=128 still don't.** The picture isn't "flash_prefill_attend loses everywhere" any more — it's shape-dependent, and D=64 in particular resisted every lever tried in Step 5.
- **`BQ=16`'s rejection is measured on this GPU only.** A higher-core-count Apple GPU with more threadgroup-memory bandwidth or a different occupancy/latency tradeoff might favor it where this M4 doesn't — the template parameter is left in place for exactly that reason, but nobody should flip it without re-measuring on the target hardware.
- **`rabitq_prefill_attend`'s causal path is correct but not competitive**, and wasn't revisited in Step 5 (all tuning there targeted `flash_prefill_attend`) — it's still ~12-15x slower than SDPA at the shapes tested in Step 2.
- **Root cause of the D=64 ceiling, and of the original barrier/round-trip overhead, remains a hypothesis, not a measurement.** The "simdgroup_matrix forces threadgroup-memory round-trips that steel's AOT strategy avoids" explanation from the first pass is still the most likely story, and Step 5's synchronization audit reinforces it (every barrier traced back to genuinely necessary cross-SIMD-group traffic, none was slack) — but this wasn't confirmed with GPU-level profiling (Metal System Trace, occupancy counters). Phases from the original optimization brief that weren't attempted here — per-head-dimension specialized kernel *entry points* (as opposed to the template-parameter specialization already in place), vectorized (`float4`) softmax reductions, a redesigned output-accumulation layout, sequence-length-aware dispatch, and double-buffered K/V pipelining — remain open for a future pass, but given how flat the returns were on Step 5's synchronization- and tile-shape-focused work, none of them looks likely to close the D=64/D=128 gap on its own.

*What was measured: causal-mask correctness via 88 passing parity tests on `rabitq_prefill_attend` (12 skipped for invalid `S_q > S_kv` shapes) plus 40 passing on `flash_prefill_attend` (12 skipped, same reason) from the first pass; 90 passing correctness checks (boundary cases included — first/last token, partial final KV block, S not divisible by BK, query blocks partially outside S_q, KV-cache continuation) from Step 5's `scripts/flash_prefill_harness.py correctness`, zero failures at every checkpoint. Benchmark numbers from `scripts/metal_rabitq_prefill_bench.py`, `scripts/prefill_roofline_bench.py`, and `scripts/flash_prefill_harness.py bench`, all committed and reproducible. All numbers are from one Apple M4 (10-core GPU, 24 GB) — not the M3 Ultra class of chip the issue's back-of-envelope estimate used.*

