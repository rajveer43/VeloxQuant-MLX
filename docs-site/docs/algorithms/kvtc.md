---
id: kvtc
title: KVTC-adapted
sidebar_label: KVTC-adapted
slug: /algorithms/kvtc
---

# KVTC-adapted — Local PCA + DP-Optimal Bit Allocation + Entropy Coding

**Method id:** `kvtc` · **New in 0.35.0** · *Inspired by* ["KV Cache Transform
Coding for Compact Storage in LLM Inference" (NVIDIA, **ICLR 2026**, accepted
poster, arXiv:2511.01815)](https://arxiv.org/abs/2511.01815) —
**KVTC-adapted (VeloxQuant-MLX implementation)**, not a faithful port.

KVTC joins the repo's low-rank / spectral family
([Palu](../algorithms/palu), [SVDq](../algorithms/svdq),
[SpectralQuant](../algorithms/spectral)) with a genuinely new mechanism axis:
instead of a fixed, hand-chosen mixed-bit split, it computes a
**dynamic-programming-optimal, per-component, budget-constrained** bit
allocation — one that can assign a component **exactly 0 bits** (dropping it
entirely) — and adds a real, measured **entropy-coding** stage on top of the
quantized codes.

## Where it sits — the mechanism gap

| Method | Split rule | Can zero a component? | Entropy coding? |
|---|---|---|---|
| [SVDq](../algorithms/svdq) | fixed top-25%/75% by singular value | No | No |
| [Palu](../algorithms/palu) | fixed top-25%/75% by singular value (K **and** V) | No | No |
| [SpectralQuant](../algorithms/spectral) | binary signal/noise cutoff via participation ratio, uniform bits per half | No | No |
| `ratequant`'s waterfilling (`allocators/ratequant.py`, `spectral/bit_allocator.py::water_fill_bits`) | closed-form, continuous, **per-layer** proportional allocation | No | No |
| **KVTC-adapted** | **DP-optimal, discrete, per-component** allocation under a hard total-bit budget | **Yes** | **Yes (order-0 Huffman)** |

All four existing allocators use a fixed or closed-form-continuous split —
none compute a **provably optimal discrete allocation** for a given total-bit
budget, and none can zero an individual low-variance component while another
gets more bits than the "high" tier. KVTC's axis: given per-component
variances from a local PCA and a total bit budget, use **dynamic
programming** to choose an integer bit-width per component (including 0)
that minimizes total expected distortion — then **entropy-code** the
resulting codes for a further lossless size reduction.

## :warning: The honesty crux — read this first

1. **Local (per-sequence) PCA, not the paper's pre-calibrated global basis.**
   The paper fits one PCA basis offline on a calibration corpus and reuses it
   for all future caches at inference. This library has no calibration
   pipeline wired into `KVCacheBuilder.for_model`, so the basis is fit
   **online from the sequence's own prefill keys/values** — the same
   "fit-locally, no calibration set" limitation [SVDq](../algorithms/svdq)
   and [Palu](../algorithms/palu) already document.
2. **The DP allocator optimizes an analytic distortion proxy, not a
   real-activation-fit rate-distortion model.** `allocators/kvtc_dp.py`'s DP
   is exact and real — it correctly finds the budget-constrained minimum of
   the objective it is given. What is a *proxy* is the objective itself: the
   reused `D(v, b) = v · β^(-b)` Gaussian-quantization distortion curve from
   `allocators/ratequant.py::fit_distortion_curve`, not a curve fit on real
   LLM activation statistics as the paper does. **One canonical distortion
   curve** — this module imports the constant rather than re-deriving one.
3. **Entropy coding is a real, measured, lossless order-0 Huffman coder**
   (`quantizers/_entropy_coding.py`, stdlib `heapq`, no external dependency),
   not the paper's (possibly more sophisticated) scheme, and never the
   theoretical Shannon-entropy bound. We report the **realized**
   post-entropy-coding byte count, including the code table's own storage
   cost — never hidden.
4. **Both K and V**, mirroring [Palu](../algorithms/palu) (not
   [SVDq](../algorithms/svdq)'s keys-only scope) — the paper compresses both.
5. **Not path-dependent** (contrast with the eviction family
   [H2O](../algorithms/h2o)/[TOVA](../algorithms/tova)/[MorphKV](../algorithms/morphkv)/[KVzip](../algorithms/kvzip)):
   the PCA basis and DP-derived bit allocation are fixed once at prefill and
   reused, unchanged, for every subsequent token — pinned by a determinism
   test.
6. Nothing here is validated on a trained model. The paper's headline
   numbers (**up to 20× — up to 40× in some regimes — compression at under
   1pp accuracy loss** on LLaMA 3 / Mistral NeMo / R1-Qwen2.5 1.5B–70B across
   AIME25, GSM8K, LiveCodeBench, LongBench, MATH-500, MMLU, Qasper, RULER)
   are the **paper's, on trained models** — never quoted as this repo's own.

## The uniform-variance collapse (pinned)

When every principal component has **equal** variance (no signal to
exploit) and `bit_choices` is a contiguous integer range, the DP-optimal
allocation collapses **exactly** to `floor(total_bit_budget / n_components)`
bits per component, with the remainder distributed one extra bit to the
first components in index order — precisely what a naive uniform splitter
would produce. This is the analogue of [SVDq](../algorithms/svdq)'s
fixed-split baseline, [MorphKV](../algorithms/morphkv)'s `window=1`==TOVA,
and [KVzip](../algorithms/kvzip)'s `probe="latest"`==TOVA reductions — a
pinned test, not just documentation
(`tests/allocators/test_kvtc_dp.py::test_uniform_variance_collapses_to_floor_plus_remainder`).
We do **not** claim any other collapse (e.g. to SVDq's fixed 25/75 split) —
that split is a *different, non-optimal* allocation, and the DP allocator
should **beat** it whenever variance is non-uniform (see below).

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="kvtc",
    head_dim=128,
    kvtc_bit_budget=512,     # total bits per token across all components (K, V independently)
    kvtc_bit_choices=(0, 1, 2, 3, 4, 6, 8),  # allowed per-component bit-widths (0 = drop)
    kvtc_beta=3.5,           # distortion decay constant, shared with ratequant.py
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

Single-layer, no coordinator — the default `for_model` path returns one
`KVTCKVCache` per attention layer.

## How it works

**Prefill** (first call, `S > 1`), for keys and values independently:

1. Center the batch and run truncated SVD (reusing
   `quantizers/_quant_utils.py::_truncated_svd`, the same shared helper
   [SVDq](../algorithms/svdq)/[Palu](../algorithms/palu)/[GEAR](../algorithms/gear)
   already use) with **no fixed-energy truncation** — `r = min(S, D)`. The
   DP allocator itself decides how many components survive.
2. Project into latents `L = (x − mean) @ V → [S, r]`.
3. Compute per-component sample variance from the singular values
   (`v_i = s_i² / S`).
4. Call `allocators.kvtc_dp.dp_allocate_bits(variances, total_bit_budget)` to
   get an integer bit-width per component (may be 0).
5. Quantize each surviving (bits > 0) component independently with a
   min/max affine integer coder. **0-bit components are dropped from
   storage entirely** — no zero-filled placeholder is stored.
6. **Entropy-code** the concatenated integer codes of all surviving
   components (order-0 Huffman).

**Decode** (subsequent tokens): project the new key/value through the
**already-fitted, frozen** basis and bit allocation, quantize, and grow the
entropy-coded store. Reconstruction: entropy-decode → dequantize each
surviving component → zero-fill dropped components → un-project
(`latents @ V.T + mean`).

## The matched-budget-distortion observable

The clean, defensible claim: **at a matched total byte budget**, the DP
allocator reaches lower reconstruction distortion than a fixed-uniform-bits
baseline and than SVDq's fixed top-25%/75% split, on a planted
non-uniform-variance ("skewed") geometry — because it can zero low-variance
components instead of paying a uniform or fixed-tier floor. On a "flat"
(near-isotropic) control with no concentrated variance, KVTC is only roughly
competitive with the fixed split, not a dramatic win — this null control is
reported honestly, not oversold (the same discipline as
[KVzip](../algorithms/kvzip)'s flat control and
[MorphKV](../algorithms/morphkv)'s stable control).

`benchmark_scripts/benchmark_kvtc.py` (results in
`figures/kvtc/results.json`) sweeps sequence length
(128/256) and total bit budget (64/128) across `geometry ∈ {skewed_variance,
flat}`, three arms (KVTC-DP, fixed-uniform, SVDq-fixed-split) at the exact
same matched budget:

- **`skewed_variance`** (budget=64): KVTC(DP) reaches mean MSE **≈0.027**
  across seeds vs **≈87.6** (fixed-uniform) and **≈84.4** (SVDq-fixed-split)
  — the DP allocator wins by roughly three orders of magnitude by zeroing
  the near-noise tail instead of paying either fixed floor.
- **`flat`**: KVTC(DP) mean MSE **≈0.18** vs fixed-uniform **≈0.14** and
  SVDq-fixed-split **≈1.22** — no concentrated variance to exploit, so KVTC
  is in the same ballpark as the uniform baseline (not a dramatic win),
  while both beat the fixed-split arm, which pays its 25%/75% tier
  regardless of whether the data has any two-tier structure.
- **Entropy-coding realized gain** (`pre_entropy_bytes / kvtc_fp16_bytes`,
  including the code table's own cost): **≈0.15–0.50** across the sweep —
  modest and regime-dependent, reported plainly, never claimed at the
  Shannon-entropy bound.

This mechanism claim is also pinned as a test
(`tests/quantizers/test_kvtc.py::test_dp_beats_fixed_split_at_matched_budget_skewed_variance`)
as a **rate over several seeds**, not a single lucky run.

## Byte accounting

- `kvtc_bytes` — realized total stored bytes (K + V, all heads/batches):
  projection basis (`V` + `mean`, fp32) + per-surviving-component quant
  params (min/scale, fp32) + **realized entropy-coded payload** + entropy
  code table. Never the pre-entropy-coding fixed-width size.
- `pre_entropy_bytes` — the fixed-width (pre-entropy-coding) size, for
  comparison.
- `entropy_coding_gain` — `pre_entropy_bytes / kvtc_bytes` (the realized,
  honestly-scoped entropy-coding gain).
- `full_seq_bytes` / `compression_ratio` — hypothetical fp16 cost and the
  resulting ratio (> 1 means savings over fp16).

## Adaptation notes — what we do NOT implement

- The paper's pre-calibrated global PCA basis fit across a calibration
  corpus — replaced by a per-sequence local-PCA proxy (same limitation
  [SVDq](../algorithms/svdq) already documents).
- The paper's rate-distortion model fit on real model activation statistics
  — replaced by the repo's existing analytic Gaussian distortion-curve proxy
  (`fit_distortion_curve` / `α·β^(-b)`), reused rather than re-derived.
- A sophisticated adaptive/context-modeled entropy coder — a simple order-0
  Huffman coder instead.
- Any trained-model perplexity/throughput/accuracy benchmark. The paper's
  headline numbers are the paper's, on trained models — offline-synthetic
  matched-budget-distortion and entropy-coding-gain numbers only, here.

## Evidence

All claims trace to passing tests across
`veloxquant_mlx/tests/allocators/test_kvtc_dp.py` (32 tests),
`veloxquant_mlx/tests/quantizers/test_entropy_coding.py` (15 tests),
`veloxquant_mlx/tests/quantizers/test_kvtc.py` (12 tests), and
`veloxquant_mlx/tests/cache/test_kvtc_cache.py` (14 tests):

- Uniform-variance collapse to `floor(budget/n) [+ remainder]`, matched
  against brute-force optimality on small `n`, and local-optimality (no
  single-component reallocation lowers total distortion further).
- Monotonicity: a strictly higher-variance component never gets fewer bits.
- A near-zero-variance component can be assigned exactly 0 bits under a
  tight budget.
- Entropy coder: lossless round-trip across alphabets/sizes, table overhead
  counted, degenerate single-symbol/empty inputs handled.
- **The DP allocator beats SVDq's fixed top-25%/75% split at a matched total
  byte budget** on planted skewed-variance geometry — a rate over several
  seeds, not one lucky run.
- Basis/allocation frozen after prefill, reused unchanged across decode
  steps (not path-dependent) — pinned explicitly.
- `compression_ratio > 1` on structured (low-rank) long-sequence synthetic
  data, for both K and V.

**No model-level benchmark has been run.** `benchmark_scripts/benchmark_kvtc.py`
is offline-synthetic and deterministic in all non-timing fields (verified by
diffing two runs) — matched-budget reconstruction MSE/cosine and
entropy-coding-gain numbers only, not perplexity or throughput on a real
model.

## When to use it

KVTC is for workloads where the KV cache's principal-component variance is
genuinely skewed (structured, low-rank-ish data) and you want the allocator
itself to decide which components are worth keeping, rather than
hand-tuning a fixed split. Where [SVDq](../algorithms/svdq) and
[Palu](../algorithms/palu) always pay a 25%/75% floor and
[SpectralQuant](../algorithms/spectral) uses a binary signal/noise cutoff,
KVTC can drop a component to exactly zero bits and adds a real (if modest)
entropy-coding pass on top.

| Method | Allocation rule | Zero-bit components | Entropy coding |
|--------|------------------|----------------------|-----------------|
| [SVDq](../algorithms/svdq) | fixed top-25%/75% | No | No |
| [Palu](../algorithms/palu) | fixed top-25%/75% (K + V) | No | No |
| [SpectralQuant](../algorithms/spectral) | binary signal/noise, uniform per half | No | No |
| **KVTC-adapted** | **DP-optimal per component** | **Yes** | **Yes (order-0)** |
