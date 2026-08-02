---
id: adakv
title: AdaKV-proxy — Per-Head Adaptive Bit Allocation
sidebar_label: AdaKV-proxy
slug: /algorithms/adakv
---

# AdaKV-proxy — Per-Head Adaptive Bit Allocation

**Available since:** v0.12.0  
**Paper:** arXiv:2407.11550 (Ada-KV, 2024) — VeloxQuant-MLX implements a *proxy* adaptation, not a faithful port (see [Adaptation notes](#adaptation-notes)).  
**Effective key bits:** configurable target (default 2.0; common 2.0–3.0) → 5.3×–8× key bandwidth reduction  
**Calibration:** None — zero-shot, works on any model immediately.

---

## Quick start

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

config = KVCacheConfig(
    method="adakv",
    head_dim=128,  # set to your model's head dimension
    adakv_target_avg_bits=2.0,  # global average bits/element target
    adakv_lo_bit=2,  # minimum bits any head can get
    adakv_mid_bit=3,  # middle tier (set == hi for a 2-tier set)
    adakv_hi_bit=4,  # maximum bits any head can get
    adakv_group_size=32,
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

Where [Kitty](./kitty) adapts precision **across channels within a head**, AdaKV-proxy adapts precision **across heads**. Not every attention head is equally sensitive to key quantization. Heads whose key magnitudes vary widely from token to token tend to spread attention over many positions (high attention entropy) — quantizing them coarsely costs more quality. AdaKV-proxy uses the inter-token key-norm variance as an attention-free proxy for head importance, then solves a per-head bit budget so that the heads that need precision get it and the budget you save on flat heads pays for it elsewhere — all while the average bits/element stays at your configured target.

### Head importance proxy

For each head `h`, compute the variance of the per-token key L2 norm across the tokens seen so far:

```
head_importance[h] = Var_t( ||k_t||_2  for t in seen tokens )
```

averaged over the batch. This is computed online from running accumulators — no calibration corpus, no attention weights.

### Budget allocation

Computed once at the end of prefill and updated every decode step:

1. **Normalise** importances to sum to 1.
2. **Scale** to the global budget: multiply by `n_heads × target_avg_bits` to get a real-valued per-head bit budget `b[h]`, then clamp to `[lo_bit, hi_bit]`.
3. **Snap** each `b[h]` to the nearest value in the allowed set `{lo_bit, mid_bit, hi_bit}`.
4. **Greedy round-trip correction.** If the integer total over/undershoots `n_heads × target_avg_bits`, repeatedly move the head whose real budget is closest to the next allowed boundary one step in the corrective direction, stopping when no single step gets the total closer to target.

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
| `adakv_target_avg_bits` | `2.0` | Global average bits/element target. The allocator constrains the per-head sum to `H × this`. |
| `adakv_lo_bit` | `2` | Minimum bits any head can receive. |
| `adakv_mid_bit` | `3` | Middle tier. Set equal to `hi_bit` for a 2-tier `{lo, hi}` set. |
| `adakv_hi_bit` | `4` | Maximum bits any head can receive. |
| `adakv_group_size` | `32` | Number of tokens per quantization group (along the sequence axis). |
| `adakv_update_interval` | `1` | Recompute the head allocation every N tokens. Wired but currently always recomputes every step (see [What is not implemented](#what-is-not-implemented)). |

### Tuning the target

| `target_avg_bits` | Allowed set | Typical spread | Key compression |
|---|---|---|---|
| 2.0 | {2, 3, 4} | mostly 2-bit, a few 3/4-bit | 8× |
| 2.5 | {2, 3, 4} | mix of 2/3-bit | 6.4× |
| 3.0 | {2, 3, 4} | mix of 2/3/4-bit | 5.3× |

---

## Comparison with related methods

| | AdaKV-proxy | Kitty | SVDq | KIVI |
|---|---|---|---|---|
| Adaptation axis | Per **head** | Per **channel** | Latent (SVD) | None (uniform) |
| Key space | Original (no projection) | Original | Latent (SVD) | Original |
| Effective bits | target (2.0–3.0) | ~2.5 | ~1.25 | 2.0 |
| Key compression | 5.3×–8× | 6.4× | 12.8× | 8× |
| Calibration | None | None | SVD at prefill | None |
| Importance signal | Inter-token key-norm variance (online) | Per-channel variance (online) | Singular value magnitude | Uniform |
| Values compressed | No | No | No | Yes (group quant) |

**When to use AdaKV-proxy over KIVI:** When some heads are far more sensitive to key quantization than others. AdaKV-proxy spends the same average budget as KIVI but concentrates it where it helps; KIVI quantizes every head at the same bit-width.

**When to use AdaKV-proxy over Kitty:** When the dominant non-uniformity in your model is *across heads* rather than *across channels within a head*. The two are complementary — Kitty redistributes bits inside a head, AdaKV-proxy redistributes them between heads. AdaKV-proxy also lets you name an exact average-bit target rather than deriving it from a channel fraction.

**When to use Kitty/SVDq instead:** Kitty if per-channel variance dominates; SVDq if you need sub-2-bit keys and can absorb the prefill SVD.

---

## Adaptation notes

VeloxQuant-MLX's implementation is a **proxy** for Ada-KV, documented here:

1. **Bit budget, not eviction budget.** True Ada-KV (arXiv:2407.11550) adapts the per-head *eviction* budget — how many tokens each head keeps — using softmax attention weights. Those weights are not available inside the `update_and_fetch` contract. We instead adapt the per-head *bit* budget, which fits the cache-only contract while preserving the core idea: give more resources to the heads that need them.

2. **Attention-free importance proxy.** Head importance is the inter-token key-norm variance, computed online from running accumulators — a proxy for attention entropy that needs no attention scores and no calibration corpus.

3. **Online recomputation.** The allocation is recomputed every step from running sum/sum-of-squares accumulators (O(H) per step), not from a one-time offline pass.

### What is not implemented

- **True Ada-KV head-adaptive eviction budget** — needs softmax attention weights, outside the cache contract. Documented as the theoretical basis only.
- **Cross-layer budget sharing** — a layer with uniformly low-importance heads could in principle donate budget to another layer. Out of scope: it would break the single-wrapper-per-layer contract.
- **`update_interval > 1` caching** — the bit assignment is recomputed every step by default. The `adakv_update_interval` field is wired through config, but caching the assignment across N steps is a future optimisation.

---

## Evidence

| Claim | Source | Status |
|---|---|---|
| High-importance heads receive more bits than low-importance heads | Test `test_high_importance_heads_get_more_bits` | ✅ Verified |
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
