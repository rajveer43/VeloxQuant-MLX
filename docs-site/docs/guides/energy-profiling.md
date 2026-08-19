---
id: energy-profiling
title: Energy Profiling Guide
sidebar_label: Energy Profiling
slug: /guides/energy-profiling
---

# Energy Profiling Guide

This guide covers `benchmark_scripts/benchmark_energy.py` and the
`veloxquant_mlx.profiling` package: a measurement harness for comparing the
energy and throughput cost of FP16 KV cache against VeloxQuant-compressed KV on
Apple Silicon.

It is a **measurement** harness. It does not optimise anything, and it contains
no Metal kernel work.

## What the numbers mean — and what they do not

Read this section before quoting any figure this harness produces.

| Metric | Status | Caveat |
|---|---|---|
| `tokens_per_s` | **Measured** | Decode only; prefill is timed separately. |
| `peak_memory_mb` | **Measured** | `mx.get_peak_memory()`, reset per arm. |
| `kv_bytes_per_token` | **DERIVED** | Computed from cache geometry. **Not** observed traffic. |
| `energy_j`, `j_per_token` | **Sampled estimate** | Integrated package power. Requires root. |

Three limitations matter, and none of them are incidental:

1. **Energy is a sampled integration, not a hardware energy counter.** The
   sampler computes `J = mean_package_W × elapsed_s` from `powermetrics`
   readings taken at a fixed interval. Power excursions shorter than that
   interval are averaged away, so the sampling interval bounds the resolution.

2. **Attribution is whole-package.** `powermetrics` reports package-level
   power. Every other process running on the machine contributes to the total.
   Close other workloads before measuring, and treat the absolute number as an
   upper bound on what inference alone costs.

3. **Bandwidth is derived from cache geometry, not measured.** MLX exposes no
   bytes-moved counter — its entire `mx.metal` surface is allocation-side
   (`get_active_memory`, `get_peak_memory`, `set_cache_limit`, …), none of it
   traffic-side. `powermetrics` reports power and residency, not DRAM bytes/s.
   So `kv_bytes_per_token` is an analytical model of what the cache *should*
   read, assuming each decode step reads the whole resident cache once. Use it
   to compare arms against each other, never as an absolute traffic figure.

## Running it

### Unprivileged

Gives throughput, peak memory, and derived bytes/token. No `sudo` needed.

```bash
python benchmark_scripts/benchmark_energy.py mlx-community/Qwen3-8B-4bit \
    --reps 3 --max-tokens 200 --context-tokens 4096
```

Every energy field prints as `n/a (requires sudo)` — never as a number, and
never as a dash that could be misread as zero.

### Privileged

Adds `energy_j` and `j_per_token`. The harness **cannot elevate itself**, so
run it under `sudo` directly:

```bash
sudo python benchmark_scripts/benchmark_energy.py mlx-community/Qwen3-8B-4bit \
    --reps 3 --max-tokens 200 --context-tokens 4096
```

Note the interpreter: if the project runs in a virtualenv, use its Python
explicitly (`sudo .venv/bin/python ...`). Plain `sudo python` picks up system
Python, which will not have `mlx` installed.

### Always check sampling coverage

`powermetrics` streams a plist document per interval into a pipe. If any of
that output is lost, the surviving samples cover only part of the run and the
energy figure is correspondingly incomplete. The harness reports this:

```
WARNING: power sampling covered only 42% of this run (17.8s of 42.1s).
Energy is under-reported for this arm.
```

Per-arm coverage is also recorded in `energy_benchmark_results.json` under
`meta.power_sampling_coverage`, and is available programmatically as
`PowerSampler.coverage` (`None` when nothing was sampled, `1.0` when every
interval was captured).

**Treat any arm below ~90% coverage as unreliable rather than as a low-energy
result.** Energy integrates over the window the samples actually cover, so
incomplete sampling under-reports — it does not invent energy for unobserved
intervals, but it does understate the run. Widely varying coverage across
repetitions of the same arm is the signature to watch for: identical work
reporting very different joules means the sampler dropped data, not that the
hardware behaved differently.

### Why `--context-tokens` matters

Eviction arms only diverge from the baseline once the sequence exceeds their
budget. With a short prompt, a Q-Filters arm at budget 512 evicts nothing and
correctly reports bytes/token identical to FP16 — which looks like the method
does nothing. Pad the context past the budget to measure the regime the method
is actually for.

## What the arms are

| Arm | Cache | Mechanism |
|---|---|---|
| A | stock `mlx_lm` `KVCache` | FP16 baseline |
| B1 | `KVCacheConfig(method="kivi", bit_width_inlier=4)` | **Quantization** — scales bytes/token by the bit ratio |
| B2 | `KVCacheConfig(method="qfilters", qfilters_budget=512)` | **Eviction** — caps bytes/token at the budget |

B1 and B2 are both present because they reduce traffic by *different*
mechanisms, and distinguishing them is the single most useful thing this
harness does. Quantization scales traffic — it still grows with sequence
length, just more slowly. Eviction caps it — past the budget, traffic stops
growing with sequence length entirely.

**Arm C (a fused Metal kernel) is deliberately not implemented.** Its
precondition is a profiled bottleneck identified from arms A and B. Building it
before this harness reports would mean optimising a bottleneck nobody has
demonstrated exists.

## Confound controls

The harness is built around two confounds that would otherwise produce
plausible, wrong numbers:

- **Warm-up runs are discarded** before every measured arm. First-run Metal
  kernel compilation and page-in are large, one-off costs that would otherwise
  be charged entirely to whichever arm happened to run first.
- **Arms are interleaved**, not blocked. Running `A,A,A,B,B,B` on a
  sustained-load M4 confounds "ran later, when the chip was hotter" with "used
  more energy". The harness runs `A,B1,B2,A,B1,B2,…` and reports **medians with
  spread**, not means.

## Measuring real bandwidth (escalation path)

If you genuinely need *measured* memory bandwidth rather than the derived
figure — which Step 2 kernel work would — MLX exposes `mx.metal.start_capture()`
and `mx.metal.stop_capture()`. These write a GPU trace that Xcode Instruments
can open, and Instruments' Metal System Trace **can** show real bandwidth
counters.

This path is interactive and not scriptable, which is why the harness does not
build on it. Use it as a targeted follow-up once A/B identifies something worth
investigating.

## Unprivileged degradation

`powermetrics` requires root, and this repo's test suite does not. Every
energy-measuring path degrades to `None` rather than crashing or reporting
zero:

```python
from veloxquant_mlx.profiling import PowerSampler

with PowerSampler() as sampler:
    ...  # your workload

sampler.energy_joules()   # None when unprivileged — never 0.0
```

The `None`-not-`0.0` distinction is enforced by tests. A silent zero would
travel downstream as a real measurement showing that inference is free, which
is exactly the kind of fabricated number this harness exists to avoid.

`PowerSampler.__exit__` never raises: a profiling failure must not fail the run
being profiled.

## Results

<!-- RESULTS-TABLE-START -->

Apple M4 (10-core GPU, 25.77 GB unified) · MLX 0.32.0 ·
`mlx-community/Qwen3-8B-4bit` (36 layers, 8 KV heads, head_dim 128) ·
4096-token prompt, 200 decode tokens · 3 interleaved reps, medians reported.

| Arm | tok/s | Peak MB | KV B/tok (derived) | J | J/token |
|---|---|---|---|---|---|
| A — fp16 baseline | 16.77 | 6846.7 | 633,470,976 | n/a | n/a |
| B1 — KIVI 4-bit | 15.22 | 6701.7 | 210,935,808 | n/a | n/a |
| B2 — Q-Filters, budget 512 | 10.60 | 7981.7 | 75,497,472 | n/a | n/a |

**J and J/token are NOT YET RUN.** They require `sudo`, which was not available
in the session that produced this table. Every other column is real.

### What this shows — and what it does not

**Derived KV traffic behaves as modelled.** KIVI 4-bit reads 33% of the
baseline's bytes/token; Q-Filters at budget 512 reads 12%, and — unlike
quantization — its figure does not grow with sequence length. The two
mechanisms are cleanly distinguishable, which was the harness's main design
goal.

**Neither compressed arm is faster. Both are slower.** This is the finding, and
it is a negative one:

- KIVI 4-bit costs **~9% throughput** (15.22 vs 16.77 tok/s) despite reading a
  third of the bytes.
- Q-Filters costs **~37% throughput** (10.60 vs 16.77 tok/s) despite reading an
  eighth of the bytes, and uses **more** peak memory than the baseline
  (7981.7 vs 6846.7 MB).

At this scale, on this hardware, **reducing KV bytes did not buy speed** — the
per-step dequantization and eviction-scoring work costs more than the traffic
it saves. Decode at 4K context on an 8B model is evidently not bandwidth-bound
enough for the traffic reduction to dominate.

Q-Filters using more memory than the cache it compresses is worth flagging
specifically: the eviction path holds scoring state and retained-window buffers
alongside the cache, and at a 512-token budget against a 4296-token sequence
that overhead exceeds what eviction gives back.

**This does not establish anything about J/token.** Throughput is not energy: a
slower arm at lower power can still win on joules, which is precisely why the
energy column exists. Whether compression saves energy on this hardware remains
**open** until the privileged run is done. The hypothesis that dequantization
cost exceeds traffic saved is now *supported on the throughput axis* — which is
the outcome that would most change the plan — but it has not been tested on the
energy axis at all.

**Caveats.** Single model, single context length, single budget. Thermal drift
is visible within the run (the baseline fell 18.72 → 15.43 tok/s across three
interleaved reps), which is why medians over interleaved arms are reported
rather than means over blocks.

<!-- RESULTS-TABLE-END -->
