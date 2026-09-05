# `nsg` Autotuning — Benchmark Report

Optimization of `scalar_fused_decode_attend`, VeloxQuant-MLX's fused
group-affine (KIVI-style) decode + attention Metal kernel.

**TL;DR** — The kernel's `nsg` parameter (SIMD-groups per threadgroup) shipped
at a fixed default of 4. It was the one occupancy lever no prior experiment had
tuned. Two threadgroup-memory footprint fixes unblocked higher values, and an
autotuned default now selects from the dispatch shape. Kernel-level speedup vs.
the old default is **1.2–4.2x**; real end-to-end decode throughput on
Qwen3-4B-4bit improves **1.07x–1.85x** depending on context length, and
**1.48x–5.31x** against the dequantize-then-SDPA path a KIVI-style cache uses
today. All 3352 tests pass; output is verified against a numpy reference.

---

## 1. Hardware and software

Every number in this document was measured on one machine, in the sessions
described. No number is extrapolated, scaled, or carried over from another
device.

| | |
|---|---|
| Chip | Apple M4 (MacBook Air, `Mac16,13`) |
| GPU | 10-core, architecture `applegpu_g16g` |
| CPU | 10-core (4 performance + 6 efficiency) |
| Unified memory | 24 GB (25.77 GB reported; 19.07 GB max recommended working set) |
| macOS | 26.5.2 (build 25F84) |
| Metal | Metal 4 |
| MLX | 0.32.0 |
| **Calibrated memory bandwidth** | **92.3–93.6 GB/s** |

> **Bandwidth is measured, never assumed.** The peak above is re-derived at the
> start of every benchmark run via a large fp16 elementwise multiply
> (100M elements, read + write), following the same self-calibration principle
> as `docs/KV_KERNEL_ROOFLINE_FINDINGS.md`. Run-to-run variance across sessions
> was 92.3–93.6 GB/s (~1.4%); each table below is internally consistent with
> the peak measured in its own run.

> **Note on the paper draft.** This is *not* the "M1 Max, 64 GB" configuration
> described in the Open-TQ-Metal manuscript. On this 24 GB machine the paper's
> 70B-at-128K experiments are physically impossible (~53 GB required). All
> claims here are scoped to what this hardware actually ran.

### Measurement methodology

All timings come from one shared harness so methodology cannot drift between
experiments:

- **Warmup**: 3–5 untimed iterations before any measurement (Metal kernels are
  JIT-compiled by `mx.fast.metal_kernel` on first call).
- **Synchronization**: `mx.eval()` **and** `mx.synchronize()` inside the timed
  region, so what is measured is GPU execution, not MLX lazy-graph
  construction.
- **Samples**: 10–25 timed iterations per cell.
- **Reported statistic**: **median**. p10/p90/min/max/stdev are captured for
  every cell; medians are quoted throughout because they are robust to the
  thermal and scheduling noise this machine shows under sustained load.
- **Inputs**: deterministic (`np.random.default_rng` with fixed seeds).

---

## 2. Baseline: what was actually wrong

### 2.1 The kernel is occupancy-bound, and this reproduces

`docs/KV_KERNEL_ROOFLINE_FINDINGS.md` (issue #259) concluded that
`scalar_fused_decode_attend` is dispatch/occupancy-bound rather than
bandwidth-bound. That was re-derived from scratch here rather than taken on
trust, and it reproduces.

**Experiment 1 — vary `S_kv` at a fixed decode shape** (B=1, H_q=8, H_kv=8;
8 threadgroups; D=128, `nsg=4`). If the kernel were bandwidth-bound, bandwidth
utilization would climb toward peak as the problem grows. It does not:

| `S_kv` | median (ms) | GB/s | % of peak |
|---:|---:|---:|---:|
| 128 | 0.256 | 1.28 | 1.4% |
| 512 | 0.385 | 3.40 | 3.7% |
| 2 048 | 0.987 | 5.31 | 5.7% |
| 8 192 | 3.754 | 5.59 | 6.0% |
| 16 384 | 7.059 | 5.94 | 6.4% |

128× more data moved buys 5% more bandwidth utilization. The kernel is not
bandwidth-limited.

**Experiment 2 — fix `S_kv`=16 384, vary threadgroup count.** This is the
discriminating test:

| B | H_q | H_kv | threadgroups | median (ms) | GB/s | % of peak |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 8 | 7.044 | 5.95 | 6.4% |
| 1 | 32 | 32 | 32 | 7.647 | 21.94 | 23.7% |
| 4 | 32 | 32 | 128 | 18.346 | 36.58 | 39.5% |
| 8 | 32 | 32 | 256 | 35.595 | 37.71 | 40.7% |

Bandwidth tracks **threadgroup count**, not bytes moved. This matches the prior
document's measurements within noise (it reported 6.1% / 21.8% / 36.7% for the
first three rows). Diagnosis confirmed.

### 2.2 The gap: `nsg` was never tuned

Total GPU concurrency is roughly `n_tg × nsg`, where:

- `n_tg = B × H_kv × S_q` — the number of dispatched threadgroups
- `nsg` — SIMD-groups **inside** each threadgroup, splitting the KV axis
  flash-decoding style

Every prior experiment in this repo moved the *first* term. Head-packing traded
it away (measured 2.7–4.7× **slower**); cross-layer and multi-request batching
added to it. **All of them held `nsg` at 4.**

This matters because at a real single-request decode step `n_tg` is only 4–8
threadgroups on a 10-core GPU. And critically — unlike head-packing — raising
`nsg` costs **no threadgroup count**. It composes with the dispatch lever
rather than competing against it.

A direct probe confirmed the default was far from optimal (B=1, H_q=8, H_kv=8,
`S_kv`=16 384):

| `nsg` | median (ms) | % of peak | vs. `nsg=4` |
|---:|---:|---:|---:|
| 1 | 26.109 | 1.7% | 0.27× |
| 2 | 13.409 | 3.4% | 0.52× |
| **4** *(shipped default)* | **7.013** | **6.5%** | **1.00×** |
| 8 | 3.837 | 11.8% | 1.83× |
| 16 | 3.197 | 14.2% | **2.19×** |
| 32 | — | — | *rejected: budget* |

Two problems surfaced at once: the default was ~2× off, **and** the higher
values were being rejected by the threadgroup-memory budget check on exactly
the GQA shapes real models use.

---

## 3. What changed

The 32 KB threadgroup-memory budget — not diminishing returns — was the binding
constraint on `nsg`. Two footprint fixes removed it.

### 3.1 `DSLOTS_C` specialization

`my_out` (per-lane registers) and `sh_o` (the cross-SIMD-group softmax merge
buffer) were both sized to **8 slots**, the worst case for `D ≤ 256`:

```c
float my_out[HEADS_PER_KV_C][8];                    // D/32 <= 256/32 = 8
threadgroup float sh_o[NSG_C * HEADS_PER_KV_C * 8 * 32];
```

But the kernel is **already compiled per-`D`** — the factory keys its cache on
`D` — it simply never passed `D` to the compiler. Injecting
`DSLOTS_C = ceil(D/32)` as a `#define` sizes both arrays to the actual head
dimension. At the near-universal `D=128` that is 4 slots instead of 8: **half**
the per-lane register footprint and half the threadgroup buffer.

### 3.2 `sh_o` stored as `half`

`sh_o` held fp32 partials that are rescaled, summed across SIMD-groups, divided
by the global denominator, and emitted as `half` regardless — so fp32 bought no
precision that survived to the output.

The one real hazard is overflow: an unnormalized partial sum grows with the
number of KV slots a SIMD-group visited and could exceed `half`'s ~65504
ceiling at long context. This is avoided by dividing each partial by **its own**
`running_d` before the store, making it a convex combination of decoded V values
— bounded by `max|V|` regardless of `S_kv`. The merge then re-applies `d_s`
explicitly:

```c
// store:  sh_o = out_s / d_s          (bounded)
// merge:  acc += sh_o * d_s * exp(m_s - gm)
```

This is an exact algebraic rebalancing, not an approximation. `sh_m` (running
max) and `sh_d` (softmax denominator) deliberately **stay fp32** — they feed
`exp()` rescaling, where fp16's narrow range and mantissa would degrade the
online-softmax correction.

**Combined effect**: the merge buffer shrinks 4× at `D=128`, raising the
admissible `nsg` from 8 → 16 at `heads_per_kv=4`, and 4 → 8 at
`heads_per_kv=8`.

### 3.3 `_auto_nsg`: the default is now derived, not guessed

`nsg` now defaults to `None`, which selects from the dispatch shape. The policy
comes from the sweep in §4.1, not from intuition: **take the largest `nsg` the
budget admits when under-dispatched; back off to 8 once `n_tg ≥ 32`.** An
explicit integer still pins the value for benchmarking or unusual shapes.

---

## 4. Kernel-level results

### 4.1 Full `nsg` sweep

D=128, `group_size=32`, B=1 unless noted. Every legal `nsg` measured per cell;
**bold** marks the selected optimum. Times are medians in ms.

| shape | `hpk` | `n_tg` | `S_kv` | nsg=1 | nsg=2 | nsg=4 | nsg=8 | nsg=16 | nsg=32 | best | **vs nsg=4** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B=1 H_q=8 H_kv=8 | 1 | 8 | 256 | 1.763 | 1.012 | 0.687 | 0.483 | 0.376 | **0.375** | 32 | **1.83×** |
| B=1 H_q=8 H_kv=8 | 1 | 8 | 1 024 | 2.671 | 1.222 | 0.724 | 0.466 | 0.340 | **0.310** | 32 | **2.33×** |
| B=1 H_q=8 H_kv=8 | 1 | 8 | 4 096 | 6.639 | 3.495 | 1.886 | 1.137 | 0.729 | **0.598** | 32 | **3.15×** |
| B=1 H_q=8 H_kv=8 | 1 | 8 | 16 384 | 25.216 | 12.896 | 6.733 | 3.697 | 2.155 | **1.605** | 32 | **4.19×** |
| B=1 H_q=32 H_kv=8 | 4 | 8 | 256 | 0.744 | 0.470 | 0.350 | 0.291 | **0.276** | — | 16 | **1.27×** |
| B=1 H_q=32 H_kv=8 | 4 | 8 | 1 024 | 2.131 | 1.207 | 0.744 | 0.498 | **0.396** | — | 16 | **1.88×** |
| B=1 H_q=32 H_kv=8 | 4 | 8 | 4 096 | 8.841 | 4.455 | 2.398 | 1.356 | **0.928** | — | 16 | **2.59×** |
| B=1 H_q=32 H_kv=8 | 4 | 8 | 16 384 | 33.308 | 17.154 | 8.735 | 4.670 | **2.944** | — | 16 | **2.97×** |
| B=1 H_q=28 H_kv=4 | 7 | 4 | 256 | 0.826 | 0.552 | 0.541 | 0.300 | **0.285** | — | 16 | **1.90×** |
| B=1 H_q=28 H_kv=4 | 7 | 4 | 1 024 | 3.009 | 1.509 | 0.968 | 0.598 | **0.473** | — | 16 | **2.05×** |
| B=1 H_q=28 H_kv=4 | 7 | 4 | 4 096 | 10.558 | 5.227 | 2.843 | 1.587 | **1.225** | — | 16 | **2.32×** |
| B=1 H_q=28 H_kv=4 | 7 | 4 | 16 384 | 41.734 | 20.986 | 11.010 | 6.105 | **4.166** | — | 16 | **2.64×** |
| B=1 H_q=32 H_kv=4 | 8 | 4 | 256 | 0.901 | 0.547 | 0.392 | **0.320** | — | — | 8 | **1.22×** |
| B=1 H_q=32 H_kv=4 | 8 | 4 | 1 024 | 2.850 | 1.560 | 0.928 | **0.599** | — | — | 8 | **1.55×** |
| B=1 H_q=32 H_kv=4 | 8 | 4 | 4 096 | 11.281 | 5.822 | 3.005 | **1.734** | — | — | 8 | **1.73×** |
| B=1 H_q=32 H_kv=4 | 8 | 4 | 16 384 | 44.768 | 22.586 | 11.531 | **6.267** | — | — | 8 | **1.84×** |
| B=1 H_q=16 H_kv=16 | 1 | 16 | 256 | 0.611 | 0.443 | 0.362 | 0.327 | 0.317 | **0.314** | 32 | 1.15× |
| B=1 H_q=16 H_kv=16 | 1 | 16 | 1 024 | 1.720 | 1.043 | 0.662 | 0.462 | 0.419 | **0.409** | 32 | **1.62×** |
| B=1 H_q=16 H_kv=16 | 1 | 16 | 4 096 | 6.523 | 3.459 | 1.951 | 1.207 | 0.964 | **0.954** | 32 | **2.04×** |
| B=1 H_q=16 H_kv=16 | 1 | 16 | 16 384 | 25.298 | 13.040 | 6.879 | 3.852 | **2.846** | 2.922 | 16 | **2.42×** |
| B=4 H_q=32 H_kv=8 | 4 | 32 | 256 | 0.756 | 0.482 | 0.382 | **0.380** | 0.384 | — | 8 | 1.00× |
| B=4 H_q=32 H_kv=8 | 4 | 32 | 1 024 | 2.371 | 1.357 | 1.002 | 0.921 | **0.916** | — | 16 | 1.09× |
| B=4 H_q=32 H_kv=8 | 4 | 32 | 4 096 | 8.573 | 4.668 | 2.852 | **2.714** | 2.754 | — | 8 | 1.05× |
| B=4 H_q=32 H_kv=8 | 4 | 32 | 16 384 | 33.704 | 17.973 | 10.778 | **10.357** | 10.589 | — | 8 | 1.04× |
| B=8 H_q=32 H_kv=8 | 4 | 64 | 256 | 0.850 | 0.609 | 0.589 | **0.540** | 0.562 | — | 8 | 1.09× |
| B=8 H_q=32 H_kv=8 | 4 | 64 | 1 024 | 2.533 | 1.551 | 1.436 | **1.237** | 1.339 | — | 8 | 1.16× |
| B=8 H_q=32 H_kv=8 | 4 | 64 | 4 096 | 9.256 | 5.304 | 4.994 | **4.183** | 4.592 | — | 8 | 1.19× |
| B=8 H_q=32 H_kv=8 | 4 | 64 | 16 384 | 36.929 | 20.575 | 19.446 | **16.305** | 18.024 | — | 8 | 1.19× |

*(`—` = rejected by the threadgroup-memory budget at that `hpk`.)*

### 4.2 Reading the sweep

Three patterns, all consistent with the occupancy diagnosis:

1. **The win is largest where `n_tg` is smallest.** At `n_tg`=8 the gain
   reaches 4.19×; at `n_tg`=64 it flattens to ~1.19×. This is the defining
   signature of an occupancy-limited kernel — extra intra-threadgroup width only
   pays while the dispatch is failing to fill the GPU.

2. **The win grows with `S_kv`.** At `n_tg`=8, `hpk`=1: 1.83× at `S_kv`=256
   rising to 4.19× at 16 384. Longer contexts give each SIMD-group more
   independent KV slots to chew through, so added width converts into real
   parallel work rather than idle lanes.

3. **The optimum is "largest legal `nsg`" until saturation.** `hpk`=1 shapes
   want 32; `hpk`=4 want 16; `hpk`=8 want 8 — in each case the budget ceiling.
   Past `n_tg ≥ 32` the optimum settles at 8 and larger values slightly regress
   (B=8, `S_kv`=16 384: 16.305 ms at `nsg`=8 vs 18.024 ms at `nsg`=16).

This directly produced the `_auto_nsg` policy, with the `n_tg ≥ 32` saturation
threshold read off rows 21–28 rather than chosen.

### 4.3 Achieved bandwidth

Best-`nsg` bandwidth utilization, B=1, D=128 (peak 93.6 GB/s that run):

| shape | `S_kv` | `nsg=4` % peak | best-`nsg` % peak |
|---|---:|---:|---:|
| H_q=8 H_kv=8 (`hpk`=1) | 16 384 | 6.4% | **27.4%** |
| H_q=8 H_kv=8 (`hpk`=1) | 4 096 | 5.8% | **18.7%** |
| H_q=32 H_kv=8 (`hpk`=4) | 16 384 | 4.6% | **9.6%** |

Utilization improves 2–4×, but even the best case reaches only ~27% of peak.
**The kernel remains occupancy-limited, not bandwidth-limited** — this change
reduces the severity of the constraint without removing it. Any claim that this
kernel is now "near the roofline" would be false.

---

## 5. Real-model end-to-end results

Kernel microbenchmarks are necessary but not sufficient — a kernel can win in
isolation and lose in real inference. These numbers are **real
`mlx_lm` generation**: full forward passes including embeddings, MLPs, QKV/O
projections, sampling, and per-step quantization.

### 5.1 Setup

| | |
|---|---|
| Model | `mlx-community/Qwen3-4B-4bit` |
| Layers | 36 |
| Heads | `H_q`=32, `H_kv`=8 (GQA ratio 4), `D`=128 |
| KV quantization | KIVI-style group-affine, `group_size`=32 |
| Harness | `benchmark_scripts/benchmark_real_model_scalar_attend.py` |
| Decode steps | 20 (context sweep) / 30 (batch sweep) |

**All arms attend over the identical KIVI-quantized state.** The only
difference is the attend path:

- **dequant** — dequantize to fp16, then standard MLX SDPA. This is what a real
  KIVI-style cache does today.
- **fused `nsg=4`** — `scalar_fused_decode_attend` at the previously shipped
  default.
- **fused auto** — `scalar_fused_decode_attend` with `nsg=None` (this work).

### 5.2 Context-length sweep (B=1, single-request decode)

Decode throughput, tokens/sec:

| `prompt_len` | dequant | fused `nsg=4` | **fused auto** | auto vs `nsg=4` | auto vs dequant |
|---:|---:|---:|---:|---:|---:|
| 256 | 17.4 | 24.0 | **25.7** | 1.07× | 1.48× |
| 1 024 | 8.1 | 16.7 | **22.8** | **1.37×** | **2.82×** |
| 2 048 | 4.8 | 12.4 | **18.7** | **1.51×** | **3.93×** |
| 4 096 | 2.6 | 7.4 | **13.6** | **1.85×** | **5.31×** |

The end-to-end win **grows with context**, mirroring the kernel sweep — 1.07×
at short context rising to 1.85× at `prompt_len`=4096 over the previously
shipped default, and 5.31× over the dequantize-then-SDPA path.

> ### ⚠️ The short-context number is reported deliberately
>
> The first real-model run measured only **1.01–1.08×** and looked like a null
> result. The cause was methodological, not physical: the stock harness
> hardcodes `prompt_len=256`, which holds `S_kv` at ~256–288 — the **shortest**
> point in the sweep and the one with the least headroom (kernel-level gain
> there is just 1.22–1.27×). Additionally at B≥4 the threadgroup count already
> saturates the GPU.
>
> The 1.07× figure is kept in this table rather than dropped, because it is
> what anyone re-running the stock harness will see. **Anyone benchmarking this
> kernel must sweep `prompt_len`; a single short-context reading understates
> the effect by roughly 1.7×.**

### 5.3 Batch sweep (`prompt_len`=256, 30 decode steps)

| B | dequant tok/s | fused `nsg=4` | fused auto | auto vs `nsg=4` | auto vs dequant |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.8 | 23.9 | 25.9 | 1.08× | 1.45× |
| 4 | 25.2 | 49.3 | 49.7 | 1.01× | 1.97× |
| 8 | 25.9 | 52.0 | 52.4 | 1.01× | 2.03× |

At short context with B≥4 the autotune adds essentially nothing (1.01×) — the
dispatch already fills the GPU, exactly as §4.2's saturation pattern predicts.
The fused kernel's advantage over the dequant baseline (1.45–2.03×) persists,
but that advantage predates this work.

**Honest summary of where this optimization does and does not help:**

| regime | benefit |
|---|---|
| Long context, single request | **Large** (up to 1.85× end-to-end) |
| Short context, single request | Small (~1.07×) |
| Short context, batched (B≥4) | **None** (~1.01×) |

---

## 6. Memory

This work is a **scheduling** change: same bytes moved, same FLOPs, same
quantized representation. **It does not change memory footprint.** The KV cache
figures below are the pre-existing property of the group-affine quantized cache
this kernel reads, included for completeness.

Qwen3-4B (36 layers, `H_kv`=8, `D`=128, `group_size`=32), K+V:

| context | fp16 KV | quantized KV (codes + scale/zero) | reduction |
|---:|---:|---:|---:|
| 256 | 37.7 MB | 23.6 MB | 1.60× |
| 1 024 | 151.0 MB | 94.4 MB | 1.60× |
| 2 048 | 302.0 MB | 188.7 MB | 1.60× |
| 4 096 | 604.0 MB | 377.5 MB | 1.60× |
| 16 384 | 2 415.9 MB | 1 509.9 MB | 1.60× |

The 1.60× (not 2×) reflects fp32 scale/zero metadata amortized over
`group_size`=32. Threadgroup memory per threadgroup dropped 4× at `D`=128
(the enabling change of §3), but that is on-chip scratch, not DRAM.

---

## 7. Correctness

Performance that breaks numerics is rejected. Every change here is gated on
parity against an independent numpy reference implementation.

| check | result |
|---|---|
| `test_scalar_attend.py` | **194 passed** |
| Full Metal suite | **939 passed**, 114 skipped |
| Full repository suite | **3352 passed**, 118 skipped |
| Lint (`ruff`) | 49 findings before **and** after — none introduced |

Specific guarantees verified:

- **Numpy-reference parity** across `S_kv ∈ {64, 512, 2048}`, `(H_q,H_kv)` ∈
  {(8,2), (32,4), (32,8)}, and every legal `nsg` — max abs error < 2e-3.
- **Expanded GQA coverage**: 40 additional parity cases at the newly-admissible
  `nsg=8`, which previously could not be tested because the budget rejected
  them.
- **Bit-identical single-layer ↔ batched parity restored.** The `half` `sh_o`
  change initially broke this (the batched kernel had not yet been updated);
  applying the identical change to `scalar_affine_attend_batched.metal` restored
  exact equality for `NL ∈ {1, 4, 32}`.
- **`half` `sh_o` precision cost**: max deviation **2.4e-4**, against a 2e-3
  tolerance — a 8× margin. Accepted because the `nsg=16` it unlocks is worth
  1.65–2.41× on GQA shapes (measured separately before committing to the
  tradeoff).
- **Three new invariant tests** on `_auto_nsg` (110 parametrized cases):
  1. never exceeds the threadgroup-memory budget, across
     `D ∈ {64,128,256}` × `hpk ∈ {1,2,4,7,8,16}` × `n_tg ∈ {1,4,8,31,32,128}`;
  2. always widens when under-dispatched relative to saturated;
  3. `nsg=None` is **bit-identical** to pinning the value it selects — the
     autotuner changes scheduling only, never results.

---

## 8. Reproduction

```bash
# Correctness (194 tests, includes the GQA + autotuner invariants)
pytest veloxquant_mlx/tests/metal/test_scalar_attend.py -v

# Autotuner invariants only
pytest veloxquant_mlx/tests/metal/test_scalar_attend.py -v -k auto_nsg

# Kernel roofline / occupancy baseline
python scripts/kv_kernel_roofline_bench.py

# Real-model end-to-end (must run as a module — see note below)
python -m benchmark_scripts.benchmark_real_model_scalar_attend
```

The real-model script **must** be run as a module. Running it by path puts the
script's directory at `sys.path[0]` ahead of the repo root, which can resolve
`veloxquant_mlx` to a stale installed copy in site-packages.

To reproduce §5.2's context sweep, vary `prompt_len` — the script's default of
256 sits at the minimum of the headroom curve (see the warning in §5.2).

Re-run each benchmark 2–3× and compare: this machine shows real run-to-run
variance from thermal and scheduling effects.

---

## 9. Limitations

Stated plainly, because a benchmark report that hides its caveats is not
useful.

- **One machine, one GPU tier.** Base 10-core M4. The `n_tg ≥ 32` saturation
  threshold in `_auto_nsg` is core-count-dependent; a higher-tier Apple GPU
  would saturate later and likely want higher `nsg` across more shapes. The
  policy would need re-fitting there.
- **One model family end-to-end.** Qwen3-4B-4bit only. A model with different
  `D`, GQA ratio, or bit-width could show a different margin. `D=128` in
  particular is what makes the `DSLOTS_C` win a clean 2×; `D=256` models gain
  less.
- **Static heuristic, not a runtime search.** `_auto_nsg` is fit to §4.1's
  table. Shapes far outside it (`heads_per_kv > 8`, very large `D`) fall back
  to conservative values that were not separately optimized.
- **No GPU-level profiling.** The occupancy story is inferred from black-box
  wall-clock latency and dispatch-shape reasoning, not from Metal System Trace
  counters or hardware occupancy metrics. This is the same limitation
  `KV_KERNEL_ROOFLINE_FINDINGS.md` flags for its own analysis.
- **Benchmark-harness cache is not production KIVI.** The real-model harness's
  `_ScalarAttendKIVICache` has no fp16 residual window, so its *output quality*
  is not representative of production KIVI. This does not affect the timing
  comparison (both arms decode from identical quantized state and were verified
  to produce identical tokens), but the throughput numbers should be read as
  "the fused kernel is Nx faster for the *same* quantized state," not "KIVI is
  Nx faster in production."
- **Prefill untouched.** This kernel is decode-only (`S_q`=1). TTFT is
  unaffected in all measurements, as expected.

---

## 10. What was *not* pursued, and why

`docs/KV_KERNEL_ROOFLINE_FINDINGS.md` already measured several optimization
directions as losses. They were not re-run:

| direction | prior measured outcome |
|---|---|
| GQA head-packing | **2.7–4.7× slower** — trades away threadgroup count, the binding constraint |
| SIMD-shuffle cross-head sharing | Architecturally impossible — `simd_shuffle` is intra-SIMD-group only |
| Two-pass decode-once + predecoded-attend | Crossover at `S_kv`≈2048–3072; **0.80×** at long context |
| Cross-layer batched dispatch | Real 1.0–4.5× kernel win, but a **structural dead end** for single-request decode: layer L+1's attention depends on layer L's *full* block output (attention + residual + MLP), so layers cannot be grouped without changing what the model computes |

Additionally, `kivi_group_quant_dequant` was previously measured at ~102% of
calibrated peak bandwidth — genuinely memory-bound, with no headroom for kernel
tuning to capture.

**The most promising untouched direction is a prefill-specialized path**
(`S_q > 1`), a genuinely different regime — large query tiles, high reuse,
matmul-shaped — that neither this work nor any prior addendum has addressed.
