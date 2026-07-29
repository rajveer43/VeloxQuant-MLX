---
id: adakv
title: AdaKV-proxy — Per-Head Adaptive Bit Allocation
sidebar_label: AdaKV-proxy
slug: /algorithms/adakv
---

# AdaKV-proxy — Per-Head Adaptive Bit Allocation

**Available since:** v0.12.0  
**Paper:** arXiv:2407.11550 (Ada-KV, NeurIPS 2025) — VeloxQuant-MLX implements a *proxy* adaptation, not a faithful port (see [Adaptation notes](#adaptation-notes)).  
**Effective key bits:** configurable target (default 2.5; usable range 2.0–4.0) → 4×–8× key bandwidth reduction  
**Calibration:** None — zero-shot, works on any model immediately.

:::warning Pick a target strictly between `lo_bit` and `hi_bit`
Per-head adaptation is only possible when `adakv_target_avg_bits` lies **strictly inside** `(adakv_lo_bit, adakv_hi_bit)`. At either endpoint the budget `H × target` can only be met by giving every head the same bit-width, so the allocation is uniform and AdaKV-proxy degrades to plain [KIVI](./kivi) — no matter what the importance signal says.

The default is `2.5` for this reason. Setting `2.0` (== `lo_bit`) or `4.0` (== `hi_bit`) now emits a `UserWarning` rather than silently flattening.
:::

---

## Quick start

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

config = KVCacheConfig(
    method="adakv",
    head_dim=128,               # set to your model's head dimension
    adakv_target_avg_bits=2.5,  # must be strictly inside (lo_bit, hi_bit)
    adakv_lo_bit=2,             # minimum bits any head can get
    adakv_mid_bit=3,            # middle tier (set == hi for a 2-tier set)
    adakv_hi_bit=4,             # maximum bits any head can get
    adakv_group_size=32,
    adakv_importance="norm_variance",  # or "attention_entropy" — see below
)

# Build one cache per model layer (mlx_lm style)
caches = KVCacheBuilder.for_model(model, config)
```

For `mlx_lm.generate`:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
caches = KVCacheBuilder.for_model(model, KVCacheConfig(method="adakv"))

output = generate(model, tokenizer, prompt="Tell me about KV caches", kv_cache=caches)
```

---

## How it works

### Intuition

Where [Kitty](./kitty) adapts precision **across channels within a head**, AdaKV-proxy adapts precision **across heads**. Not every attention head is equally sensitive to key quantization. AdaKV-proxy scores each head, then solves a per-head bit budget so the heads that need precision get it and the budget you save elsewhere pays for it — all while the average bits/element stays at your configured target.

### Head importance: two signals, two different criteria

The signal is selected by `adakv_importance`. **They are not interchangeable — they rank heads in opposite directions**, so pick deliberately.

#### `"norm_variance"` (default) — quantization sensitivity

```
head_importance[h] = Var_t( ||k_t||_2  for t in seen tokens )
```

Computed online from running accumulators — no calibration corpus, no attention weights, O(H) per step.

What it measures is **quantization sensitivity**: a head whose key magnitudes vary widely has a wider dynamic range inside each min/max quantization group, so a fixed bit-width incurs more reconstruction error and extra bits buy more improvement. That is a sound basis for allocating *bits*.

:::note This is not the paper's criterion
Ada-KV shifts budget toward attention-**dispersed** heads (§3.3, Fig. 1b). High key-norm variance is the signature of a few outlier-magnitude tokens dominating the `q·k` logits — i.e. an attention-**sparse** head, which the paper would give *less* budget. The two signals are empirically anti-correlated; earlier versions of this page described norm-variance as "a proxy for high attention entropy", which was incorrect.

`norm_variance` is a different, defensible objective — not an approximation of the paper's. If you want the paper's ranking, use `attention_entropy`.
:::

#### `"attention_entropy"` — the paper's criterion

Estimates per-head attention entropy over an observation window, normalised by `ln(S)` to `[0, 1]`. Higher entropy = more dispersed = more budget, matching Ada-KV §3.3.

Real query states aren't visible at `update_and_fetch` time, so this reuses the keys-as-proxy-queries substitution already established by [SnapKV-adapted](./snapkv): the last `adakv_obs_window` key rows stand in for queries. Still an approximation, but it carries the correct sign. Costs one `[w, S]` attention matrix per head at prefill; at decode a single token carries no attention distribution, so the prefill estimate is retained.

### Budget allocation

Computed once at the end of prefill and updated every decode step:

1. **Rank-normalise** importances to a bounded, mean-centred spread `z[h]`. Only the *ordering* survives.
2. **Place** each head at `target + spread × z[h]`, where `spread` is the widest value keeping every head inside `[lo_bit, hi_bit]`. The mean lands on `target` by construction.
3. **Snap** each to the nearest value in the allowed set `{lo_bit, mid_bit, hi_bit}`.
4. **Greedy round-trip correction.** If the integer total over/undershoots `n_heads × target_avg_bits`, repeatedly move the head whose real budget is closest to the next allowed boundary one step in the corrective direction, breaking ties on importance, until no single step gets the total closer to target.

Why rank-normalisation rather than scaling raw importance: key-norm variances span orders of magnitude, so proportional scaling followed by a clamp pins nearly every head to `lo_bit` or `hi_bit` and discards the interior ordering that steps 3–4 exist to act on. Ranks are bounded and order-preserving, so one extreme head cannot flatten the allocation.

Each head's keys are then quantized with KIVI-style asymmetric min/max group quantization at its assigned bit-width and reconstructed to fp16 for downstream SDPA.

### Running state

Updated every `update_and_fetch` call:

| State | Shape | Meaning |
|---|---|---|
| `norm_sum` | `[H]` | running sum of per-token key L2 norms per head |
| `norm_sq_sum` | `[H]` | running sum of squared norms per head |
| `n_tokens` | scalar | total tokens seen |
| `head_bits` | `[H]` | current per-head bit assignment (recomputed each call) |

Variance is recovered as `E[norm²] − E[norm]²` from these accumulators in O(H) per step.

### Effective bit-width

```
assigned_avg_bits = sum_h head_bits[h] / H

where head_bits[h] in {lo_bit, mid_bit, hi_bit}
and   sum_h head_bits[h] ~= H * target_avg_bits
```

Because bits are integers snapped to the allowed set, `assigned_avg_bits` matches the target to within a fraction of a bit (rounding). Key bandwidth reduction vs fp16 is `16 / assigned_avg_bits` (e.g. 8× at 2.0, 5.3× at 3.0).

---

## Configuration reference

| Parameter | Default | Description |
|---|---|---|
| `adakv_target_avg_bits` | `2.5` | Global average bits/element target. The allocator constrains the per-head sum to `H × this`. **Must be strictly inside `(lo_bit, hi_bit)`** or the allocation is uniform. |
| `adakv_lo_bit` | `2` | Minimum bits any head can receive. |
| `adakv_mid_bit` | `3` | Middle tier. Set equal to `hi_bit` for a 2-tier `{lo, hi}` set. |
| `adakv_hi_bit` | `4` | Maximum bits any head can receive. |
| `adakv_group_size` | `32` | Number of tokens per quantization group (along the sequence axis). |
| `adakv_update_interval` | `1` | Recompute the head allocation every N tokens. Wired but currently always recomputes every step (see [What is not implemented](#what-is-not-implemented)). |
| `adakv_importance` | `"norm_variance"` | Importance signal: `"norm_variance"` (quantization sensitivity) or `"attention_entropy"` (the paper's dispersion criterion). |
| `adakv_obs_window` | `32` | Observation-window size for `"attention_entropy"`. Ignored otherwise. |

### Tuning the target

With the default allowed set `{2, 3, 4}`:

| `target_avg_bits` | Adaptive? | Typical spread | Key compression |
|---|---|---|---|
| 2.0 | ❌ **no** — `== lo_bit` | all 2-bit (identical to KIVI) | 8× |
| 2.25 | ✅ yes | mostly 2-bit, a few 3-bit | 7.1× |
| **2.5** (default) | ✅ yes | mix of 2/3-bit | 6.4× |
| 3.0 | ✅ yes | mix of 2/3/4-bit | 5.3× |
| 3.5 | ✅ yes | mix of 3/4-bit | 4.6× |
| 4.0 | ❌ **no** — `== hi_bit` | all 4-bit (identical to KIVI) | 4× |

If you want a 2.0-bit average *and* per-head adaptation, lower `adakv_lo_bit` to `1` so `2.0` sits strictly inside the range.

---

## Comparison with related methods

| | AdaKV-proxy | Kitty | SVDq | KIVI |
|---|---|---|---|---|
| Adaptation axis | Per **head** | Per **channel** | Latent (SVD) | None (uniform) |
| Key space | Original (no projection) | Original | Latent (SVD) | Original |
| Effective bits | target (2.25–3.5) | ~2.5 | ~1.25 | 2.0 |
| Key compression | 4.6×–7.1× | 6.4× | 12.8× | 8× |
| Calibration | None | None | SVD at prefill | None |
| Importance signal | Key-norm variance or attention entropy (online) | Per-channel variance (online) | Singular value magnitude | Uniform |
| Values compressed | No | No | No | Yes (group quant) |

**When to use AdaKV-proxy over KIVI:** When some heads are far more sensitive to key quantization than others. AdaKV-proxy spends the same average budget as KIVI but concentrates it where it helps; KIVI quantizes every head at the same bit-width.

**When to use AdaKV-proxy over Kitty:** When the dominant non-uniformity in your model is *across heads* rather than *across channels within a head*. The two are complementary — Kitty redistributes bits inside a head, AdaKV-proxy redistributes them between heads. AdaKV-proxy also lets you name an exact average-bit target rather than deriving it from a channel fraction.

**When to use Kitty/SVDq instead:** Kitty if per-channel variance dominates; SVDq if you need sub-2-bit keys and can absorb the prefill SVD.

---

## Adaptation notes

VeloxQuant-MLX's implementation is a **proxy** for Ada-KV, documented here:

1. **Bit budget, not eviction budget.** True Ada-KV (arXiv:2407.11550) adapts the per-head *eviction* budget — how many tokens each head keeps — using softmax attention weights. Those weights are not available inside the `update_and_fetch` contract. We instead adapt the per-head *bit* budget, which fits the cache-only contract while preserving the core idea: give more resources to the heads that need them.

2. **Importance signal is not the paper's.** With the default `"norm_variance"`, head importance is inter-token key-norm variance — a *quantization-sensitivity* signal that is anti-correlated with Ada-KV's attention-dispersion criterion (see [Head importance](#head-importance-two-signals-two-different-criteria)). `"attention_entropy"` provides a signal with the paper's sign, at the cost of an observation-window attention matrix per head at prefill.

3. **Online recomputation.** The allocation is recomputed every step from running sum/sum-of-squares accumulators (O(H) per step), not from a one-time offline pass.

4. **No variable-length attention.** The paper's §3.5 flattened cache layout and custom CUDA kernels for variable-sized per-head caches have no analogue here — because AdaKV-proxy allocates *bits* rather than *token counts*, every head keeps the same number of tokens and standard SDPA applies unchanged. The efficiency question the paper's §4.4 answers does not arise.

### What is not implemented

- **True Ada-KV head-adaptive eviction budget** — needs softmax attention weights, outside the cache contract. Documented as the theoretical basis only.
- **Cross-layer budget sharing** — a layer with uniformly low-importance heads could in principle donate budget to another layer. Out of scope: it would break the single-wrapper-per-layer contract.
- **`update_interval > 1` caching** — the bit assignment is recomputed every step by default. The `adakv_update_interval` field is wired through config, but caching the assignment across N steps is a future optimisation.

---

## Evidence

| Claim | Source | Status |
|---|---|---|
| High-importance heads receive more bits than low-importance heads | Test `test_high_importance_heads_get_more_bits` | ✅ Verified |
| **The default config actually adapts** (regression, issue #31) | `test_default_config_is_adaptive` | ✅ Verified |
| **Degenerate target warns instead of silently flattening** | `test_degenerate_target_warns_and_is_uniform` | ✅ Verified |
| **Allocation is monotone in importance** (no saturation) | `test_allocation_monotone_in_importance` | ✅ Verified |
| **Allocation is permutation-equivariant** over distinct importances | `test_allocation_permutation_equivariant` | ✅ Verified |
| **Budget met exactly** where the allowed set permits | `test_budget_met_exactly` | ✅ Verified |
| **`attention_entropy` ranks dispersed above sparse** (paper's sign) | `test_attention_entropy_ranks_dispersed_above_sparse` | ✅ Verified |
| **`norm_variance` is anti-correlated with the paper's criterion** | `test_norm_variance_is_anticorrelated_with_paper_criterion` | ✅ Verified (documented caveat) |
| Average bits matches target within ±0.5 | `test_average_bits_matches_target` | ✅ Verified |
| Equal importance degrades to uniform target allocation | `test_equal_importance_uniform_allocation` | ✅ Verified |
| Assigned bits give lower MSE than lo_bit on the high-importance head | `test_high_importance_head_lower_mse_than_lo_bit` | ✅ Verified |
| Running norm accumulator matches ground-truth variance | `test_running_norm_accumulator_correctness` | ✅ Verified |
| Output shape preserved (prefill + decode) | Tests 2, 3, 10 | ✅ Verified |
| Values unchanged | Test 4 | ✅ Verified |
| `assigned_avg_bits` within `[lo_bit, hi_bit]` | `test_assigned_avg_bits_in_range` | ✅ Verified |
| Single-head model trivially assigns target | `test_single_head_assigns_target` | ✅ Verified |
| Determinism | Test 14 | ✅ Verified |
| Throughput vs KIVI/Kitty on M-series | `benchmark_scripts/benchmark_adakv.py` | Run locally |

---

## Next steps

- [Kitty](./kitty) — per-channel mixed precision within a head (complementary axis)
- [KIVI](./kivi) — uniform 2-bit group quantization (the baseline AdaKV-proxy layers on)
- [SVDq](./svdq) — sub-2-bit keys via offline SVD
- [Algorithm overview](./overview) — full method comparison
- [mlx_lm integration guide](../guides/mlx-lm-integration)
