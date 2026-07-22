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
arXiv:2607.10109)](https://arxiv.org/abs/2607.10109) —
**AMC-adapted (VeloxQuant-MLX implementation)**, not a faithful port.

:::warning[No verified peer-reviewed venue]
This is the **second** method in VeloxQuant-MLX (2 of 40) that does not trace
to a verified peer-reviewed venue — the first was
[NestedKV-adapted](../algorithms/nestedkv). As of 2026-07-14, the paper is a
single arXiv revision (submitted 2026-07-11, verified live 3 days later) with
no Comments/journal-ref field indicating acceptance anywhere. It is also
filed under `cs.IR` (Information Retrieval), an unusual category for what is
fundamentally a hardware architecture paper — noted as a minor oddity, not a
disqualifier. Every other method in this repo required a live-verified venue
before implementation — this one ships as a **one-time, user-directed
exception** to that standing rule, at the user's explicit direction. The next
method survey reverts to requiring a verified venue — this is not a new
precedent.
:::

:::warning[Hardware/RTL half of the source paper is entirely out of scope]
Roughly half of AMC's source paper (Sections IV-V: 45nm CMOS RTL, Verilog
clock-gating, the Precision-Gated Systolic Array, the Narrow-Width SRAM
write-back buffer, all pJ/µJ energy figures, the EDAP/Pareto silicon
comparisons) targets physical silicon. **None of that is implemented here** —
VeloxQuant-MLX is a pure-software MLX library with no RTL/silicon layer. This
module ports only the software saliency engine and the rank/precision scaling
math (Sections II-A and III). The paper's headline 59.2%-energy /
2.24x-throughput / 3.6%-accuracy trade-off is measured on the paper's own
45nm hardware simulation — **not reproduced by this port**.
:::

AMC joins the repo's per-token adaptive-precision family
([Palu](../algorithms/palu) [rank-only], [KIVI](../algorithms/kivi)/
[SKVQ](../algorithms/skvq) [bit-width-only], [RateQuant](../guides/mixed-precision)
[per-layer bit allocation]) with a genuinely new mechanism: a **single
per-token saliency scalar drives both rank and bit-width simultaneously**,
via three discrete tiers. It is also the first method in this repo whose
family is **compression-only, not eviction** — every token is retained
forever; only its effective precision varies.

## Where it sits — the mechanism gap

| Method | Adapts rank? | Adapts bit-width? | Granularity | Ever evicts? |
|---|:---:|:---:|---|:---:|
| [Palu](../algorithms/palu) | Yes | No (fixed mixed-bit latents) | per-layer/group | No |
| [KIVI](../algorithms/kivi) | No | Yes (asymmetric groups) | per-channel/token | No |
| [SKVQ](../algorithms/skvq) | No | Yes | per-channel/token | No |
| [RateQuant](../guides/mixed-precision) | No | Yes | per-layer | No |
| [H2O](../algorithms/h2o) / [CurDKV](../algorithms/curdkv) / [NestedKV](../algorithms/nestedkv) | N/A | N/A | per-token | **Yes** |
| **AMC-adapted** | **Yes** | **Yes** | **per-token, single saliency score drives both** | **No** |

## :warning: The honesty crux — read this first

1. **Unpublished preprint, 3 days old at verification, no venue.** See the
   warning banner above — the headline exception for this method, stated
   first.
2. **Hardware/RTL half of the paper entirely out of scope.** No
   clock-gating, no systolic array, no 45nm silicon, no pJ/µJ energy numbers
   reproduced anywhere in this implementation. This is roughly half of the
   source paper, not a minor omission — see the second warning banner above.
3. **Compression-only, never eviction.** Unlike every eviction method in
   this repo, AMC never drops a token — only its rank/precision is reduced.
   This is a structurally different family; see the mechanism-gap table
   above.
4. **Query-aware saliency (Eq. 3) and closed-loop adaptive thresholds (Eq.
   4-5) are opt-in, off by default.** The default path is pure
   magnitude-only scoring (Eq. 1-2), matching the paper's primary reported
   configuration. When `amc_use_query_saliency=True`, the cache wrapper uses
   the mean of the current step's keys as a proxy query (no true query
   vector is visible at the cache-wrapper level) — the same category of
   approximation as [H2O](../algorithms/h2o)/[SnapKV](../algorithms/snapkv)/
   [CurDKV](../algorithms/curdkv)'s key-as-query proxy.
5. **Offline SVD/PCA channel-order calibration required** for the rank mask
   to be safe — see
   [`amc_calibration.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/veloxquant_mlx/quantizers/amc_calibration.py).
   Without running `amc_calibrate_channel_order` on representative data
   first, rank masking truncates arbitrary, not lowest-variance, channels —
   the same category of requirement as
   [Palu](../algorithms/palu)/[SVDq](../algorithms/svdq)/RaBitQ's
   calibration step. This module ships the calibration function; the
   `AMCKVCache` wrapper itself does not currently auto-invoke it (see
   Adaptation notes).
6. **`cs.IR` category mismatch** — a minor oddity noted in the venue banner
   above, not a disqualifier.
7. **A real, honestly-reported weakness found during benchmark
   construction**: on activation distributions with no genuine saliency
   signal (every token statistically identical in magnitude), AMC's fixed
   20/30/50 percentile split still routes half the tokens into the
   aggressive Low tier — purely by rank order of noise — and comes out
   **worse** than a matched-budget uniform baseline, not merely neutral. See
   the benchmark section below for the full finding. This is a structural
   property of percentile-based tiering when the saliency signal is
   uninformative, not an implementation bug — the paper itself only ever
   evaluates on natural-language activations, where the magnitude heuristic
   is claimed to correlate with importance.
8. Nothing here is validated on a trained model or real hardware. The
   paper's own energy/throughput/accuracy numbers (Section VI, 45nm RTL
   simulation, 3-layer synthetic transformer) are the **paper's** — never
   quoted as this repo's own.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="amc",
    head_dim=128,
    amc_k_high=0.20,              # top percentile -> High tier (rank 128, 16-bit)
    amc_k_mid=0.30,                # next percentile -> Mid tier (rank 43, 8-bit)
    # remaining 50% -> Low tier (rank 8, 4-bit)
    amc_use_query_saliency=False,  # opt-in: Eq. 3 query-aware blend
    amc_adaptive_thresholds=False, # opt-in: Eq. 4-5 closed-loop threshold adjustment
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

Opt-in query-aware + adaptive-threshold path:

```python
config = KVCacheConfig(
    method="amc",
    head_dim=128,
    amc_use_query_saliency=True,
    amc_query_alpha=0.5,            # Eq. 3 balance coefficient
    amc_adaptive_thresholds=True,
    amc_threshold_window=64,        # trailing window for variance tracking
    amc_gamma=0.1,                  # threshold attenuation factor
    amc_calib_variance=0.05,        # REQUIRED when amc_adaptive_thresholds=True
)
```

Single-layer, no coordinator — the default `for_model` path returns one
`AMCKVCache` per attention layer. No `.bits` attribute — stores and returns
fp16 K/V directly (rank mask + quantize is simulated as a quantize-then-
dequantize round-trip, same convention as every other method here).

## How it works

Every call to `update_and_fetch` — prefill batch or single decode token
alike — runs the same per-token pipeline (unlike
[NestedKV](../algorithms/nestedkv)'s one-shot-prefill design):

1. **Saliency score** (Eq. 1-2): `S_i = clamp(mean(|x_i|), 0, 1)` — the
   L1-norm of each token's key activation. Optionally blended with
   query-aware cosine similarity (Eq. 3, opt-in).
2. **Tier assignment** (Algorithm 1 Phase II): the top `amc_k_high` fraction
   of tokens (by saliency) get the High tier, the next `amc_k_mid` fraction
   get Mid, the rest get Low. Implemented via
   [`dsa.MaxHeap`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/veloxquant_mlx/dsa/heap.py)
   top-k selection rather than a full sort.
3. **Rank masking** (Eq. 6): zero out channels beyond the tier's rank, on
   channels already reordered by the offline calibration permutation (see
   [`amc_calibration.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/veloxquant_mlx/quantizers/amc_calibration.py))
   so the surviving prefix is the highest-variance subspace.
4. **Precision scaling** (Eq. 7): quantize the rank-masked activation to the
   tier's bit-width (16/8/4), reusing this repo's shared group quantizer.
5. (Opt-in) **Closed-loop threshold adaptation** (Eq. 4-5): a trailing
   window of saliency values (backed by
   [`dsa.RingBuffer`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/veloxquant_mlx/dsa/ring_buffer.py))
   tracks sequence-level activation variance and widens/narrows the
   High/Mid allocation window accordingly.

Tier table (Algorithm 1's Golden Model values at `head_dim=128`; scaled
proportionally for other `head_dim`):

| Tier | Fraction | Rank | Bits |
|---|---|---|---|
| High | top 20% | 128 | 16 |
| Mid | next 30% | 43 | 8 |
| Low | remaining 50% | 8 | 4 |

## Byte accounting

- `amc_kept_bytes` — actual bytes stored across all heads (fp16-equivalent
  K + V, per-tier).
- `full_seq_bytes` — hypothetical fp16 full-rank K + V cost if AMC were
  never applied.
- `compression_ratio` — `full_seq_bytes / amc_kept_bytes` (> 1 = savings).
- `tokens_seen` — total token positions ever passed to `update_and_fetch`.
- `tokens_kept` — tokens in the (B=0, H=0) head's cache; always equals
  `tokens_seen` per head (no eviction).
- `tokens_high` / `tokens_mid` / `tokens_low` — cumulative per-tier token
  counts, for observability.

## Benchmark — honestly reported, including the part that didn't work

`benchmark_scripts/benchmark_amc.py` (results in
`figures/amc/results.json`) sweeps sequence length
(200/400) across two geometries and compares AMC's tiered compression
against a **matched-average-byte-budget uniform baseline** (fixed rank+bits
for every token, sized to AMC's own average byte cost), measuring
reconstruction MSE:

- **`sparse_outlier`** (10% of tokens are large-magnitude outliers, the rest
  small): AMC beats the matched-budget uniform baseline by roughly **8x**
  lower MSE — the geometry the method's mechanism is designed to exploit,
  and it does.
- **`uniform_magnitude`** (every token statistically identical in
  magnitude — no saliency signal to exploit): AMC is **worse** than the
  uniform baseline by roughly **100x** higher MSE, **not merely neutral** —
  stated plainly, not softened. The fixed 20/30/50 percentile split still
  routes half the tokens into the aggressive Low tier purely by rank order
  of noise, while the uniform baseline spreads the same average byte cost
  evenly. This is a real, structural weakness of percentile-based tiering
  when the underlying saliency signal is uninformative — the paper itself
  only evaluates on natural-language activations, where the magnitude
  heuristic is claimed to correlate with importance; it does not claim
  robustness on distributions where that correlation is absent, and neither
  do we.

Deterministic in all non-`_ms` fields, verified by diffing two runs.
Offline-synthetic; loads no model, no mlx_lm generation. **Not** a
reproduction of the paper's energy/throughput/accuracy numbers.

## Adaptation notes — what we do NOT implement

- The entire hardware/RTL half of the paper (Sections IV-V): 45nm CMOS RTL,
  Verilog clock-gating logic, the Precision-Gated Systolic Array, the
  Narrow-Width SRAM write-back buffer, the Saliency-Aware Controller (SAC)
  as physical logic, all pJ/µJ energy figures (Tables I-II, Eq. 8-17), the
  EDAP/Pareto silicon comparisons (Fig. 4-5, Table V), and the
  LLM.int8/AWQ/H2O/StreamingLLM/Quest/RankDyna/DiP/DynamicViT comparative
  numbers reported against the paper's own hardware/software baselines.
- `AMCKVCache` does not auto-invoke `amc_calibrate_channel_order` — callers
  must run the offline calibration pass themselves and apply
  `amc_permute_weights` before deployment for the rank mask to be
  meaningful (see honesty crux, point 5).
- Any trained-model perplexity/throughput/accuracy benchmark. The paper's
  own numbers are hardware-measured on a specific 3-layer synthetic
  transformer setup (`num-samples=4000, seq-len=32, vocab-size=16`) — not
  reproduced here.
- No PyTorch/CUDA reference kept; pure MLX from the start.

## Evidence

All claims trace to passing tests across
`veloxquant_mlx/tests/quantizers/test_amc_calibration.py` (9 tests),
`veloxquant_mlx/tests/quantizers/test_amc.py` (23 tests), and
`veloxquant_mlx/tests/cache/test_amc_cache.py` (19 tests):

- **`test_calibration_orders_channels_by_variance`** /
  **`test_permuted_columns_have_descending_variance`** — direct proof the
  offline calibration correctly identifies the highest-variance channels.
- **`test_high_tier_tokens_survive_full_precision`** — proves clearly
  high-saliency tokens get full rank/precision and low-saliency tokens get
  the aggressive Low tier.
- **`test_query_aware_saliency_downweights_high_magnitude_irrelevant_tokens`**
  — proves the opt-in Eq. 3 blend does something a pure-magnitude score
  cannot: reorders a high-magnitude-but-query-irrelevant token below a
  moderate-magnitude-but-query-relevant one.
- **`test_adaptive_thresholds_widen_on_high_variance_sequences`** /
  **`test_adaptive_thresholds_narrow_on_low_variance_sequences`** — proves
  the opt-in Eq. 4-5 closed loop moves thresholds in the correct direction.
- **`test_no_eviction_all_tokens_retained`** — direct proof of the
  compression-only design (honesty crux, point 3): cache size always equals
  cumulative tokens passed, across a mixed prefill+decode sequence.
- **`test_bitpack_roundtrip_low_tier`** — proves the 4-bit Low tier's
  `dsa.BitPackBuffer`-backed dense packing round-trips correctly.
- Degenerate zero-variance and small-`n` inputs produce finite scores, no
  NaN/crash.
- Standard suite: byte accounting, determinism, `for_model` config
  propagation (all `amc_*` fields), factory dispatch, factory smoke test.

**No model-level benchmark has been run.** `benchmark_scripts/benchmark_amc.py`
is offline-synthetic and deterministic in all non-timing fields —
reconstruction MSE only, not perplexity or throughput on a real model.

## When to use it

AMC-adapted is for workloads where you want **every** token retained (no
eviction risk) but want to spend more precision on high-magnitude tokens and
less on low-magnitude ones — a middle ground between full fp16 and
uniform aggressive quantization. It is **not** a good fit for activation
distributions with no genuine saliency signal (see honesty crux, point 7 and
the benchmark's `uniform_magnitude` result) — for a bounded-memory guarantee
regardless of distribution, prefer an eviction method like
[H2O](../algorithms/h2o) or [CurDKV](../algorithms/curdkv); for
distribution-agnostic uniform compression, prefer [KIVI](../algorithms/kivi)
or [SKVQ](../algorithms/skvq).

| Method | Ever evicts | Adapts rank + bits jointly | Verified venue |
|--------|:---:|:---:|:---:|
| [H2O](../algorithms/h2o) / [CurDKV](../algorithms/curdkv) | Yes | No | Yes |
| [Palu](../algorithms/palu) | No | Rank only | Yes |
| [KIVI](../algorithms/kivi) | No | Bits only | Yes |
| **AMC-adapted** | **No** | **Yes, from one saliency score** | **No (this method + NestedKV only)** |
