---
id: amc
title: AMC-adapted
sidebar_label: AMC-adapted
slug: /algorithms/amc
---

# AMC-adapted — Saliency-Driven Tiered Rank + Precision

**Method id:** `amc` · **New in 0.38.0** · *Inspired by* ["Adaptive Model
Compression (AMC): Saliency-Driven Resource Allocation for Ultra-Low-Power
Transformer Inference" (Hu, Yuan, Hu, Yin, Li, Suchter — Apple;
arXiv:2607.10109)](https://arxiv.org/abs/2607.10109) — **AMC-adapted
(VeloxQuant-MLX implementation)**, not a faithful port.

AMC scores every token by activation magnitude and routes it into one of
three tiers — **High** (full rank + 16-bit), **Mid** (rank 43 + 8-bit), or
**Low** (rank 8 + 4-bit). Unlike every eviction method in this library, it
**never drops a token** — it only spends less rank/precision on the ones
that matter less. That makes it a good fit when you want a bounded-memory
guarantee with zero risk of losing a token you'll need later, while still
compressing the bulk of the sequence hard.

:::info[Two things worth knowing before you use this]
This method traces to a preprint that has not yet cleared peer review, and
this software port implements only the algorithmic half of the source paper
— not the custom 45nm chip it was designed alongside. Neither changes what
the code does; both are detailed under [Good to know](#good-to-know) below.
:::

## How it works

Every call to `update_and_fetch` — prefill batch or single decode token
alike — runs the same four-step pipeline:

1. **Score.** Each token's saliency is the mean absolute value of its
   activations (L1-norm), clamped to `[0, 1]`.
2. **Tier.** The top `amc_k_high` fraction of tokens (by score) get the High
   tier, the next `amc_k_mid` fraction get Mid, everyone else gets Low.
3. **Rank mask.** Low/Mid-tier tokens get their least-informative channels
   zeroed out — safe only because an offline calibration step has already
   sorted channels by variance (see [Calibration](#calibration-required),
   below).
4. **Quantize.** Each tier's surviving channels are quantized to that tier's
   bit-width (16 / 8 / 4).

| Tier | Fraction | Rank | Bits |
|---|---|---|---|
| High | top 20% | 128 | 16 |
| Mid | next 30% | 43 | 8 |
| Low | remaining 50% | 8 | 4 |

(Ranks shown at `head_dim=128`; scaled proportionally for other head dims.)

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="amc",
    head_dim=128,
    amc_k_high=0.20,  # top percentile -> High tier (rank 128, 16-bit)
    amc_k_mid=0.30,   # next percentile -> Mid tier (rank 43, 8-bit)
    # remaining 50% -> Low tier (rank 8, 4-bit)
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches

response = mlx_lm.generate(model, tokenizer, prompt="...", max_tokens=120)
```

No `.bits` attribute — AMC returns fp16 K/V directly; the rank mask and
quantization are simulated as a quantize-then-dequantize round-trip, same
convention as every other method in this library.

### Optional: query-aware scoring and adaptive thresholds

Two extra knobs exist for cases where pure activation magnitude is a poor
proxy for importance (e.g. repetitive punctuation spiking the raw score):

```python
config = KVCacheConfig(
    method="amc",
    head_dim=128,
    amc_use_query_saliency=True,   # blend magnitude with query relevance
    amc_query_alpha=0.5,           # 1.0 = pure magnitude, 0.0 = pure relevance
    amc_adaptive_thresholds=True,  # widen/narrow tiers with sequence complexity
    amc_threshold_window=64,
    amc_gamma=0.1,
    amc_calib_variance=0.05,       # REQUIRED when amc_adaptive_thresholds=True
)
```

These are off by default; the default path matches the paper's primary
reported configuration. When `amc_use_query_saliency=True`, the cache uses
the mean of the current step's keys as a stand-in for a true query vector
(no query is visible at the cache-wrapper level — same approximation
H2O-adapted and SnapKV-adapted make).

## Calibration required

AMC's rank mask assumes channels are already sorted from most- to
least-informative — otherwise it truncates arbitrary channels, not the
genuinely low-variance ones. Run this once, offline, on representative data
before deploying:

```python
from veloxquant_mlx.quantizers.amc_calibration import (
    amc_calibrate_channel_order,
    amc_permute_weights,
)

perm = amc_calibrate_channel_order(calib_activations)  # [n_calib, D] -> [D]
weight = amc_permute_weights(weight, perm)
```

This is a one-time, zero-runtime-cost step (same category as
[Palu](../algorithms/palu)'s and [SVDq](../algorithms/svdq)'s calibration
requirements) — `AMCKVCache` does not auto-run it for you.

## Byte accounting

- `amc_kept_bytes` — actual bytes stored across all heads (fp16-equivalent
  K + V, per tier).
- `full_seq_bytes` — hypothetical fp16 full-rank K + V cost if AMC were
  never applied.
- `compression_ratio` — `full_seq_bytes / amc_kept_bytes` (> 1 = savings).
- `tokens_seen` / `tokens_kept` — total tokens seen vs. retained (always
  equal — AMC never evicts).
- `tokens_high` / `tokens_mid` / `tokens_low` — cumulative per-tier counts,
  for observability.

## Benchmark

`benchmark_scripts/benchmark_amc.py` (results in `figures/amc/results.json`)
compares AMC's tiered compression against a **matched-average-byte-budget
uniform baseline** across two synthetic geometries, measuring reconstruction
MSE:

- **Sparse outlier** (10% of tokens are large-magnitude, the rest small —
  the geometry AMC's mechanism is built for): AMC beats the matched-budget
  baseline by roughly **8×** lower MSE.
- **Uniform magnitude** (every token statistically identical — no saliency
  signal to exploit): AMC comes out roughly **100× worse** than the uniform
  baseline. This isn't a bug — a fixed percentile split still has to route
  half the tokens into the Low tier even when there's nothing genuine to
  distinguish them by rank order of noise. If your workload doesn't have a
  real saliency signal, AMC is the wrong tool (see
  [When to use it](#when-to-use-it)).

This is an offline, synthetic, deterministic benchmark — not a reproduction
of the source paper's hardware-measured numbers (see
[Good to know](#good-to-know)).

## When to use it

Reach for AMC when you want **every token retained** — no eviction risk —
but want to spend more precision on high-magnitude tokens and less on
low-magnitude ones, as a middle ground between full fp16 and uniform
aggressive quantization.

**Skip it** if your activations don't have a real saliency signal to
exploit (see the benchmark's uniform-magnitude result above) — an eviction
method like [H2O](../algorithms/h2o) or [CurDKV](../algorithms/curdkv) gives
a bounded-memory guarantee regardless of distribution, and
[KIVI](../algorithms/kivi) or [SKVQ](../algorithms/skvq) give
distribution-agnostic uniform compression without the calibration step.

| Method | Ever evicts | Adapts rank + bits jointly | Needs calibration |
|--------|:---:|:---:|:---:|
| [H2O](../algorithms/h2o) / [CurDKV](../algorithms/curdkv) | Yes | No | No |
| [Palu](../algorithms/palu) | No | Rank only | Yes |
| [KIVI](../algorithms/kivi) | No | Bits only | No |
| **AMC-adapted** | **No** | **Yes, from one saliency score** | **Yes** |

AMC is also the first method in this library where a **single** per-token
score drives both rank and bit-width at once — every other rank-adaptive
method ([Palu](../algorithms/palu)) or bit-width-adaptive method
([KIVI](../algorithms/kivi), [SKVQ](../algorithms/skvq),
[RateQuant](../guides/mixed-precision)) picks one axis, not both.

## Good to know

A few things worth being upfront about, in case they affect your decision
to use this method:

- **Preprint, not yet peer-reviewed.** As of 2026-07-14, arXiv:2607.10109
  is a single revision (submitted 2026-07-11) with no record of acceptance
  anywhere. It's also filed under `cs.IR` (Information Retrieval), an
  unusual category for what is fundamentally a hardware paper. This repo
  normally requires a verified peer-reviewed venue before shipping a
  method; AMC (like [NestedKV-adapted](../algorithms/nestedkv) before it)
  is a stated, one-time exception. None of this affects correctness of the
  code below — it's a provenance note, not a code-quality one.
- **Software half only.** Roughly half of AMC's source paper (45nm CMOS
  RTL, Verilog clock-gating, a custom systolic array, an SRAM write-back
  buffer, and all of its energy-per-joule figures) describes a physical
  chip. None of that exists here — there's no silicon layer in a pure MLX
  library. What's implemented is the software half: the saliency scoring,
  tiering, rank masking, and quantization math (Sections II-A and III of
  the paper). The paper's own headline numbers — **59.2% energy reduction,
  2.24× throughput, 3.6% accuracy cost** — are measured on that physical
  chip and are not reproduced by this port; see the
  [benchmark](#benchmark) above for this repo's own measured numbers
  instead.
- Query-aware saliency and adaptive thresholds are opt-in and lightly
  tested against real workloads — validate on your own data before relying
  on them in production.

## Evidence

54 tests pass across `test_amc.py`, `test_amc_calibration.py`, and
`test_amc_cache.py`, covering: saliency scoring against the paper's
definition, tier-assignment percentile accuracy, calibration ordering
channels correctly by variance, rank masking preserving high-variance
channels post-calibration, query-aware scoring reordering
magnitude-vs-relevance tokens as expected, adaptive thresholds moving in the
correct direction under high/low variance, 4-bit bit-packing round-trips,
zero-eviction guarantee across mixed prefill/decode sequences, and
determinism. No trained-model benchmark has been run — see
[Benchmark](#benchmark) for what has.
