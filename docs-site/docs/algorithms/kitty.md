---
id: kitty
title: Kitty — Dynamic Channel-wise Mixed-Precision
sidebar_label: Kitty
slug: /algorithms/kitty
---

# Kitty — Dynamic Channel-wise Mixed-Precision Keys

**Available since:** v0.11.0  
**Paper:** arXiv:2511.18643 (Nov 2025, unreviewed preprint) — VeloxQuant-MLX implementation is an adaptation, not a faithful port.  
**Effective key bits:** ~2.5 bits/element (default) → **6.4× key bandwidth reduction**  
**Calibration:** None — zero-shot, works on any model immediately.

---

## Quick start

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

config = KVCacheConfig(
    method="kitty",
    head_dim=128,  # set to your model's head dimension
    kitty_hi_fraction=0.25,  # top 25% channels get 4-bit (default)
    kitty_hi_bit=4,
    kitty_lo_bit=2,
    kitty_group_size=32,
)

# Build one cache per model layer (mlx_lm style)
caches = KVCacheBuilder.for_model(model, config)
```

For `mlx_lm.generate`:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
caches = KVCacheBuilder.for_model(model, KVCacheConfig(method="kitty"))

output = generate(model, tokenizer, prompt="Tell me about KV caches", kv_cache=caches)
```

---

## How it works

### Intuition

Not all key channels carry equal information. Channels with high variance across tokens encode more discriminative attention patterns — quantizing them coarsely loses more quality than quantizing low-variance channels. Kitty exploits this by allocating more bits to high-variance channels at every step, with no offline calibration.

### Prefill phase

When the first batch of keys arrives (shape `[B, H, S, D]`):

1. **Rank channels by variance.** For each attention head `h`, compute per-channel variance across the sequence axis:

   `σ²_j = Var(K_h[:, j])  for j in [0, D)`

2. **Split into hi/lo channel sets.** Top `hi_fraction × D` channels (by σ²) → 4-bit. Remaining → 2-bit.

3. **Quantize each set.** Asymmetric min/max group quantization with `group_size` tokens per group, independently per channel set.

4. **Reconstruct fp16 keys** and forward to the underlying `mlx_lm` KVCache.

5. **Initialise running accumulators** `key_sum [H, D]` and `key_sq_sum [H, D]` from the prefill batch for use in decode steps.

### Decode phase

For each new single-token key (`S=1`):

1. **Update running accumulators** with the incoming key.
2. **Re-derive channel ranking** from updated running variance:

   `σ²_j ≈ (Σ k_j²) / n − ((Σ k_j) / n)²`

3. **Quantize** the new key at the updated mixed-precision assignment.
4. **Reconstruct fp16** and pass to the underlying cache.

### Effective bit-width

`avg_bits = hi_fraction × hi_bit + (1 − hi_fraction) × lo_bit`

At defaults (`hi_fraction=0.25, hi_bit=4, lo_bit=2`):

`avg_bits = 0.25 × 4 + 0.75 × 2 = 2.5 bits/element`

Key bandwidth reduction vs fp16: `16 / 2.5 = 6.4×`

---

## Configuration reference

| Parameter | Default | Description |
|---|---|---|
| `kitty_hi_fraction` | `0.25` | Fraction of channels routed to `hi_bit`. 0.25 → top-25% by variance get 4-bit. |
| `kitty_hi_bit` | `4` | Bit width for high-variance channels. |
| `kitty_lo_bit` | `2` | Bit width for low-variance channels. |
| `kitty_group_size` | `32` | Number of tokens per quantization group (along sequence axis). |

### Tuning avg_bits

| `hi_fraction` | `hi_bit` | `lo_bit` | `avg_bits` | Key compression |
|---|---|---|---|---|
| 0.0 | 4 | 2 | ~2.0 | 8× |
| 0.25 | 4 | 2 | 2.5 | 6.4× |
| 0.5 | 4 | 2 | 3.0 | 5.3× |
| 0.25 | 8 | 4 | 5.0 | 3.2× |

---

## Comparison with related methods

| | Kitty | SVDq | KIVI |
|---|---|---|---|
| Key space | Original (no projection) | Latent (SVD) | Original |
| Effective bits | ~2.5 | ~1.25 | 2.0 |
| Key compression | 6.4× | 12.8× | 8× |
| Calibration | None | SVD at prefill | None |
| Sensitivity signal | Per-channel variance (online) | Singular value magnitude | Uniform |
| Values compressed | No | No | Yes (group quant) |

**When to use Kitty over SVDq:** When you want zero calibration overhead (no prefill SVD) and slightly higher quality at 2.5-bit vs SVDq's 1.25-bit latent compression. Kitty is the better choice for short-to-medium sequences where SVDq's prefill SVD cost is non-trivial.

**When to use Kitty over KIVI:** When your key distributions have strongly non-uniform channel variance. Kitty adaptively allocates more bits to the channels that matter most; KIVI quantizes all channels uniformly.

---

## Adaptation notes

VeloxQuant-MLX's implementation differs from the paper in two ways, both documented here:

1. **Online channel ranking (vs. offline calibration).** The paper optionally pre-computes channel sensitivity from a calibration corpus. We compute it online from the incoming key tensor — simpler, zero-setup, and competitive for keys with stable inter-channel variance ordering across contexts.

2. **Running variance accumulator.** During decode, channel rankings are updated incrementally using Welford-style sum/sum-of-squares accumulators instead of recomputing variance from scratch per step. This is O(D) per step instead of O(S·D).

**Not implemented:** cross-layer sensitivity sharing (breaks single-wrapper contract); static calibration path (deferred — requires offline corpus).

---

## Evidence

| Claim | Source | Status |
|---|---|---|
| 2-bit mixed-precision outperforms uniform 2-bit on high-variance channels | Unit test `test_kitty_better_mse_than_uniform_2bit_on_high_variance_data` | ✅ Verified |
| avg_bits = 2.5 at default settings | `test_assigned_avg_bits_in_range` | ✅ Verified |
| Output shape preserved (prefill + decode) | Tests 2, 3 | ✅ Verified |
| Values unchanged | Test 4 | ✅ Verified |
| High-variance channels correctly identified | Test 5 | ✅ Verified |
| Hi channels lower error than lo channels | Test 6 | ✅ Verified |
| Determinism | Test 14 | ✅ Verified |
| Throughput vs KIVI/SVDq on M-series | `benchmark_scripts/benchmark_kitty.py` | Run locally |

---

## Next steps

- [SVDq](./svdq) — sub-2-bit keys via offline SVD (more compression, prefill cost)
- [KIVI](./kivi) — uniform 2-bit group quantization (simpler, no channel ranking)
- [Algorithm overview](./overview) — full method comparison
- [mlx_lm integration guide](../guides/mlx-lm-integration)
