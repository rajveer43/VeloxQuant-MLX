---
id: xquant
title: XQuant — Cross-Layer KV Cache Reuse
sidebar_label: XQuant
slug: /algorithms/xquant
description: XQuant is VeloxQuant-MLX's first cross-layer method, pairing transformer layers into anchor/reuse groups so reuse layers share the anchor's quantized codes and store only their own scale/zero plus a residual, reaching roughly 4.5 effective key bits at the safe default.
keywords: [xquant, cross-layer kv cache reuse, anchor reuse groups, layer coordinator, effective bit-width, sub-2-bit keys]
---

# XQuant — Cross-Layer KV Cache Reuse

**Available since:** v0.13.0  
**Paper:** arXiv:2510.11236 (EMNLP 2025, Yang et al.) — VeloxQuant-MLX implementation is faithful to the cross-layer-reuse core, adapted at the integration boundary (see [Adaptation notes](#adaptation-notes)).  
**Effective key bits:** ≈4.5 bits/element (group_size=2, base_bits=2) at the safe default `residual_bits=4`; sub-2-bit is reachable with `residual_bits=0`, but that setting is unsafe on real models — see the warning below.  
**Calibration:** None — zero-shot, works on any model immediately.

This is VeloxQuant-MLX's **first cross-layer method**. Every other algorithm operates strictly within one layer's `update_and_fetch`; XQuant has layers *coordinate*.

---

## Quick start

XQuant requires cross-layer coordination, so it must be built with `KVCacheBuilder.for_model()` (which constructs the shared coordinator and assigns anchor/reuse roles):

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

config = KVCacheConfig(
    method="xquant",
    xquant_group_size=2,  # layers per anchor/reuse group (2 = pairs)
    xquant_base_bits=2,  # anchor quantizer bit-width
    xquant_residual_bits=4,  # reuse-layer correction residual (default; see warning below)
    xquant_group_quant_size=32,
)

caches = KVCacheBuilder.for_model(model, config)
```

For `mlx_lm.generate`:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
caches = KVCacheBuilder.for_model(model, KVCacheConfig(method="xquant"))

output = generate(model, tokenizer, prompt="Tell me about KV caches", kv_cache=caches)
```

:::note[Single-cache construction]
`KVCacheFactory.create(KVCacheConfig(method="xquant"))` returns a *degenerate anchor* (no coordinator) — useful for unit-testing the anchor path, but it performs no cross-layer reuse. Use `for_model` for the real method.
:::

---

## How it works

### Intuition

Adjacent transformer layers are *assumed* to produce similar key/value tensors — the premise XQuant's cross-layer reuse is built on. XQuant exploits this by grouping layers into **anchor / reuse** groups: the anchor pays the full quantization cost and publishes its integer codes; the reuse layers borrow those codes and store only their own dequantization parameters (a fresh scale/zero), which correct for cross-layer drift.

:::warning[Measured correlation is weak on real models]
On real checkpoints (Qwen2.5-0.5B, Llama-3.2-1B), measured cosine similarity between adjacent layers' real keys is **~0, sometimes negative** — not "highly similar." Pure reuse (`xquant_residual_bits=0`) then reconstructs mostly noise and generation degrades to incoherent output, at *every* `xquant_base_bits`, including near-lossless settings (see [VeloxQuant-MLX#380](https://github.com/veloxquant/veloxquant-mlx/issues/380)). Always measure with `cross_layer_similarity` before trusting `residual_bits=0` for a given model; the shipped default is `4`, which recovers coherent output even with no correlation at all.
:::

Across a group with a sufficient residual, the *effective* per-element bit-width still falls below the anchor's — just not all the way down to the param-only floor pure reuse would imply.

### Layer pairing

At build time, attention-bearing layers are chunked into contiguous groups of `group_size`. The first layer in each group is the **anchor**; the rest **reuse** it:

```
group_size=2:  [anchor 0][reuse 0] [anchor 1][reuse 1] ...
group_size=3:  [anchor 0][reuse 0][reuse 0] [anchor 1][reuse 1][reuse 1] ...
```

### Anchor layer

On each `update_and_fetch(keys, values)`:

1. Quantize K and V with asymmetric min/max group quantization (the same scheme as KIVI) → integer codes + per-group scale/zero.
2. Publish the codes to the shared `XQuantCoordinator`, keyed by `(group_id, token_start)`.
3. Return the fp16 reconstruction as usual.

### Reuse layer

On each `update_and_fetch(keys, values)`:

1. Fetch the paired anchor's codes for the same token range from the coordinator.
2. Fit this layer's **own** per-group scale/zero against its incoming K/V (the codes are shared; the dequant *parameters* are per-layer — this is what corrects cross-layer magnitude/offset drift).
3. Dequantize the shared codes with this layer's params. Add a quantized low-bit residual (`xquant_residual_bits`, default 4) — see the correlation warning above for why 0 is unsafe on real models.
4. Store only `(scale, zero, [residual_codes])` — never a full code tensor. This is the byte win.

If the anchor has not yet published a step (e.g. mis-ordered iteration), the reuse layer falls back to self-quantization, so correctness never depends on iteration order.

### Effective bit-width

```
per_layer_bits(reuse)  = (param_bytes + residual_bytes) * 8 / (n_elements)
per_layer_bits(anchor) = base_bits + amortized param overhead

group_effective_bits   = ( anchor_bits + (group_size - 1) * reuse_bits ) / group_size

At group_size=2, base_bits=2, residual_bits=0 (unsafe, see warning above):
  anchor charges base_bits + amortized params -> effective_pair_bits ~= 3.0
  reuse layer charges only scale/zero (param-only) -> effective_pair_bits ~= 1.0
  group effective ~= (3.0 + 1.0) / 2 = 2.0 bits/element

At group_size=2, base_bits=2, residual_bits=4 (shipped default):
  anchor unchanged -> effective_pair_bits ~= 3.0
  reuse layer charges scale/zero + a 4-bit residual -> effective_pair_bits ~= 6.0
  group effective ~= (3.0 + 6.0) / 2 = 4.5 bits/element
```

(`effective_pair_bits` measured directly from the accounting properties, not a closed-form approximation — param overhead depends on `xquant_group_quant_size`.)

`effective_pair_bits` on each cache reports the bits actually charged to that layer (`16 × compressed_key_bytes / fp16_key_bytes`).

---

## Configuration reference

| Parameter | Default | Description |
|---|---|---|
| `xquant_group_size` | `2` | Layers per anchor/reuse group. 2 → pairs; 3 → one anchor feeds two reusers. |
| `xquant_base_bits` | `2` | Anchor quantizer bit-width. |
| `xquant_residual_bits` | `4` | Reuse-layer correction residual. **0 is unsafe on real models** (see warning above) — measured near-zero cross-layer correlation means pure reuse reconstructs mostly noise. 4 is the validated floor that recovers coherent output with no correlation assumed; only lower it after confirming strong correlation for your specific model with `cross_layer_similarity`. |
| `xquant_group_quant_size` | `32` | Tokens per quantization group (along the sequence axis). |
| `xquant_max_ctx` | `8192` | Per-group token budget. Exceeding it raises `RuntimeError`. |

### Tuning the residual

Group effective bits measured via `effective_pair_bits` (not a closed-form estimate — depends on `xquant_group_quant_size`):

| `group_size` | `base_bits` | `residual_bits` | Group effective bits | When |
|---|---|---|---|---|
| 2 | 2 | 4 (default) | ~4.5 | Safe default; no correlation assumed |
| 2 | 2 | 1 | ~3.0 | Only if you've measured strong correlation for this model |
| 3 | 2 | 0 | ~1.7 | **Unsafe** — pure reuse; aggressive compression, real models degrade to incoherent output |
| 2 | 2 | 0 | ~2.0 | **Unsafe** — pure reuse (former default; see VeloxQuant-MLX#380) |

---

## Comparison with related methods

| | XQuant | SVDq | Kitty | KIVI |
|---|---|---|---|---|
| Compression axis | **Cross-layer** | Latent (SVD) | Per-channel | None (uniform) |
| Key space | Original | Latent (SVD) | Original | Original |
| Effective key bits | ~4.5 (group, safe default) / ~1.0–1.4 (group, `residual_bits=0`, unsafe) | ~1.25 | ~2.5 | 2.0 |
| Key compression | ~3.5× (safe default) / 11×–16× (`residual_bits=0`, unsafe) | 12.8× | 6.4× | 8× |
| Calibration | None | SVD at prefill | None | None |
| Coordinates layers | **Yes (coordinator)** | No | No | No |
| Values compressed | Yes | No | No | Yes |

**When to use XQuant over KIVI/SVDq/Kitty:** When you've *measured* (via `cross_layer_similarity`) that adjacent layers in your model are strongly correlated and want the lowest effective bit-width available — that measurement is not something to assume, real checkpoints checked so far show near-zero correlation. Otherwise, XQuant at its safe default is still cross-layer-aware but closer to KIVI's compression ratio. XQuant is the only method that exploits *inter-layer* redundancy — an axis orthogonal to every other method, so it can in principle compose with them (anchor quantizer swapped for SVDq/Kitty) in future.

**When to prefer the others:** Single-layer methods (KIVI, SVDq, Kitty) have no cross-layer coupling and are simpler to reason about. If your model's layers are weakly correlated, XQuant with `residual_bits=0` will degrade — measure cross-layer similarity first (the benchmark reports it) or raise `residual_bits`.

---

## Adaptation notes

XQuant's reference implementation couples layers inside a modified attention forward pass. VeloxQuant-MLX's contract is one cache object per layer, iterated independently by `mlx_lm.generate`. Our adaptation:

1. **Shared coordinator, not a forward-pass hook.** All `XQuantKVCache` instances of a model hold a reference to a single `XQuantCoordinator`, injected at `for_model()` build time. Anchors publish to it; reusers subscribe. The per-layer `update_and_fetch` signature is unchanged, so `mlx_lm.generate` stays untouched — the same "force-multiplier via a side channel" pattern KIVI-Sink used for sinks, generalized to cross-layer state.

2. **Fixed index-stride pairing.** Layers pair by index (`group_size` stride). The paper can select pairs by measured similarity; learned pairing is a future option (needs a calibration pass).

### What is not implemented

- **Learned layer pairing** — pairing is by fixed index stride, not measured similarity. Documented as a future calibration-based option.
- **Attention-output reuse** — XQuant variants also reuse attention outputs, not just KV. Out of scope: it requires intercepting the model forward pass, breaking the cache-only contract.
- **Cross-model / cross-request sharing** — the coordinator is per-model-instance, per-generation. No persistence.

---

## Evidence

| Claim | Source | Status |
|---|---|---|
| `for_model` assigns correct anchor/reuse pairing | Test `test_for_model_pairing` | ✅ Verified |
| Coordinator publish/fetch round-trips codes exactly | `test_coordinator_round_trip` | ✅ Verified |
| Output shape preserved (anchor + reuse, prefill + decode) | Tests 4, 5, 13 | ✅ Verified |
| Reuse with residual=0 ≈ self-quant on identical data | `test_reuse_residual0_within_tolerance` | ✅ Verified |
| Residual lowers MSE on correlated layers | `test_residual_lowers_mse_correlated` | ✅ Verified |
| Correlated reuse ≈ self-quant (near-free) | `test_correlated_reuse_near_self_quant` | ✅ Verified |
| Residual recovers quality on uncorrelated pairs (negative control) | `test_uncorrelated_residual_recovers` | ✅ Verified |
| Reuse bytes far below anchor bytes | `test_byte_accounting_reuse_less_than_anchor` | ✅ Verified |
| `effective_pair_bits` below base bits (reuse) | `test_effective_pair_bits_below_base` | ✅ Verified |
| Anchor/reuser stay synchronized over decode steps | `test_decode_synchronization` | ✅ Verified |
| Coordinator token budget enforced | `test_coordinator_budget_raises` | ✅ Verified |
| `group_size=3` (1→2) pairing + reconstruction | `test_group_size_three` | ✅ Verified |
| Determinism | `test_determinism` | ✅ Verified |
| Throughput + cross-layer similarity on M-series | `benchmark_scripts/benchmark_xquant.py` | Run locally |

---

## Next steps

- [SVDq](./svdq) — sub-2-bit keys via offline SVD (a candidate anchor quantizer)
- [Kitty](./kitty) — per-channel mixed precision (orthogonal axis)
- [KIVI](./kivi) — uniform 2-bit group quantization (the default anchor scheme)
- [xKV](./xkv) — a third cross-layer route: jointly factorizes a *group* of layers into one shared SVD basis, rather than reusing one anchor's codes
- [Algorithm overview](./overview) — full method comparison
- [mlx_lm integration guide](../guides/mlx-lm-integration)
