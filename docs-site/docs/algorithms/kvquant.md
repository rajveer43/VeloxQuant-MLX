---
id: kvquant
title: KVQuant-NUQ — Non-Uniform Quantization + Outlier Isolation
sidebar_label: KVQuant-NUQ
slug: /algorithms/kvquant
---

# KVQuant-NUQ — Non-Uniform Quantization + Outlier Isolation

**Available since:** v0.14.0  
**Paper:** arXiv:2401.18079 (NeurIPS 2024, Hooper et al.) — VeloxQuant-MLX implements the four cache-observable pillars (per-channel keys / per-token values, NUQ datatype, per-vector dense-and-sparse outlier isolation, and Attention Sink-Aware quantization); pre-RoPE key quantization is documented out of scope.  
**Effective key bits:** 2–4 (non-uniform) → near-fp16 quality at 3-bit on heavy-tailed K/V  
**Calibration:** None — signpost levels fit online from the prefill batch.

This is VeloxQuant-MLX's **first non-uniform-datatype method**. Every other quantizer in the suite snaps values to *uniformly* spaced levels; KVQuant places the levels where the data actually is.

---

## Quick start

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

config = KVCacheConfig(
    method="kvquant",
    head_dim=128,  # set to your model's head dimension
    kvquant_bits=3,  # base NUQ bit-width
    kvquant_outlier_fraction=0.01,  # top 1% by magnitude kept fp16
    kvquant_lloyd_iters=8,  # Lloyd-Max iterations for level fitting
)

caches = KVCacheBuilder.for_model(model, config)
```

For `mlx_lm.generate`:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
caches = KVCacheBuilder.for_model(model, KVCacheConfig(method="kvquant"))

output = generate(model, tokenizer, prompt="Tell me about KV caches", kv_cache=caches)
```

---

## How it works

### Intuition

LLM key and value distributions are sharply non-uniform — bell-shaped with heavy tails. Uniform min/max quantization spaces its levels evenly across the range, wasting most of them on the sparse tails and starving the dense center. KVQuant fixes both problems:

1. **NUQ (non-uniform datatype)** places the levels where the mass is.
2. **Dense-and-Sparse isolation** carves out the few extreme outliers so they cannot stretch the level range.
3. **Attention Sink-Aware quantization** keeps the first few tokens exact, because the model is disproportionately sensitive to error there.

### Non-uniform levels (Lloyd-Max)

For a quantization group, KVQuant fits `2^bits` signpost levels that minimize reconstruction error for the *observed* distribution, via 1-D Lloyd-Max (k-means):

1. **Quantile initialization** — levels seeded at evenly spaced quantiles of the data (deterministic).
2. **Assign / update sweeps** — each value is assigned to its nearest level; each level moves to the mean of its assigned values. Distortion is monotone non-increasing across sweeps (Lloyd's lemma).

Quantize = index of nearest signpost (`bits` bits). Dequantize = table lookup.

### Dense-and-Sparse outlier isolation

Before fitting, the top `outlier_fraction` of elements by magnitude (per channel/token) are removed to an fp16 sparse side-channel and excluded from the level fit. This stops a handful of outliers from inflating the level range — the same failure mode KIVI-Sink addresses for tokens, applied here at the *element* granularity.

Selection is by **rank**, not by comparing against the k-th largest value. A value threshold over-selects whenever magnitudes tie — a constant channel would send *every* element to the fp16 side-channel while the byte accounting still charged only `k`. Rank selection keeps it at exactly `k` per vector.

At decode a per-channel key column holds a single sample, so a top-k within that column is undefined. Decode keys are therefore screened against the **per-channel threshold frozen at prefill**, so they keep the same outlier protection prefill keys got rather than silently losing it.

### Attention Sink-Aware quantization

After the first few layers, LLMs dump a large share of attention onto the opening tokens regardless of their semantic content — the *attention sink* effect. The model is correspondingly sensitive to quantization error at those positions, so the first `kvquant_n_sink` tokens are stored in **exact fp16**.

Following the paper, sink tokens are also **excluded from the level fit and from the outlier thresholds**. Otherwise the very tokens kept exact would still skew the datatype derived for every other token. The paper reports this matters most at low bit-widths (2-bit) and when outlier isolation is disabled.

Sinks are a property of the *sequence*, so protection applies only to the leading positions of the prefill — decode tokens are always quantized.

### Quantization axes

Matching KVQuant's asymmetry (the same axes KIVI uses):

- **Keys — per-channel.** Each head-dim channel gets its own levels (sample axis = tokens). Channels have stable, distinct distributions, so key levels are fit at prefill and **frozen** for decode (like SVDq's frozen projection).
- **Values — per-token.** Each token gets its own levels (sample axis = channels). Per-token levels are inherently re-fit every call.

### Effective bit-width

```
effective_bits = bits + table_overhead + outlier_overhead + sink_overhead

table_overhead   = 2^bits level entries (fp16) per channel (keys) / per token (values),
                   amortized over the tokens stored
outlier_overhead = realized outlier count * (fp16 value + position index), amortized
sink_overhead    = kvquant_n_sink rows stored raw fp16, amortized

At bits=3, outlier_fraction=0.01, n_sink=1:
  effective_bits ~= 3.5 bits/element at long context (~4.5x compression)
```

The outlier term is charged from the **outliers actually produced**, not the nominal `S * D * outlier_fraction` — the rank-based split rounds per column and the decode path uses a carried-over threshold, so realized counts differ from the nominal figure. Accounting follows the real side-channel.

Measured amortization (`head_dim=64`, `bits=3`, `outlier_fraction=0.01`):

| Context | `effective_bits` | Compression |
|---|---|---|
| 128 | 4.48 | 3.6× |
| 512 | 3.68 | 4.4× |
| 1024 | 3.54 | 4.5× |
| 4096 | 3.55 | 4.5× |

At short context the level table dominates; it amortizes as context grows. `effective_bits` reports the realized rate.

---

## Configuration reference

| Parameter | Default | Description |
|---|---|---|
| `kvquant_bits` | `3` | Base NUQ bit-width. Produces `2^bits` signpost levels. |
| `kvquant_outlier_fraction` | `0.01` | Top-magnitude fraction kept fp16 and excluded from the level fit. 0 = pure NUQ. |
| `kvquant_group_size` | `32` | Group size for per-channel/per-token fitting. |
| `kvquant_lloyd_iters` | `8` | Lloyd-Max iterations. More iters = tighter levels, diminishing returns. |
| `kvquant_refit_interval` | `0` | Refit key levels every N decode steps. 0 = freeze prefill levels (recommended). |
| `kvquant_n_sink` | `1` | Leading attention-sink tokens kept exact in fp16 and excluded from the level fit (paper §3.5). 0 = off. |

### Tuning

| `bits` | `outlier_fraction` | Quality | When |
|---|---|---|---|
| 2 | 0.01 | good | Aggressive compression on heavy-tailed K/V |
| 3 | 0.01 | near-fp16 | Recommended default — best quality per bit |
| 4 | 0.01 | very high | Quality-critical |
| 3 | 0.0 | good | Ablation / distributions without extreme outliers |

`kvquant_n_sink=1` (the default) is nearly free — one fp16 row amortizes to nothing over a long context. The paper finds sink protection matters most at **2-bit** and when `outlier_fraction=0`, so keep it on when pushing bits down; raise it to 4 if you see quality loss concentrated at the start of the sequence.

---

## Comparison with related methods

| | KVQuant-NUQ | KIVI | SVDq | Kitty |
|---|---|---|---|---|
| Level placement | **Non-uniform (data-fit)** | Uniform min/max | Uniform (latent) | Uniform (per-channel) |
| Outlier handling | **Dense/sparse isolation** | Residual window | None | None |
| Sink protection | **fp16 sink tokens** | None | None | None |
| Key axis | Per-channel | Per-channel | Latent (SVD) | Per-channel |
| Value axis | Per-token | Per-token | fp16 | fp16 |
| Effective key bits | 2–4 | 2.0 | ~1.25 | ~2.5 |
| Calibration | None | None | SVD at prefill | None |

**When to use KVQuant over KIVI:** Whenever K/V are non-uniform (essentially always). At the same bit-width, non-uniform levels strictly reduce reconstruction error — measured 73% lower MSE than uniform at 3-bit on Laplacian data (see [Evidence](#evidence)). The cost is the level-table overhead, which amortizes over long context.

**When to prefer the others:** KIVI is simpler and has no level-fit cost — better for very short sequences where the table overhead dominates. SVDq reaches lower effective bits via low-rank latent projection (a different axis — NUQ could in principle quantize SVDq's latents).

---

## Adaptation notes

### Paper fidelity map

| Paper component | Status | Notes |
|---|---|---|
| §3.1 Per-Channel Key Quantization | ✅ Implemented | Keys per-channel, values per-token |
| §3.3 nuqX non-uniform datatype | ⚙️ Adapted | Lloyd-Max fit online, unweighted (see below) |
| §3.4 Per-Vector Dense-and-Sparse | ✅ Implemented | Threshold per channel (keys) / per token (values) |
| §3.5 Attention Sink-Aware | ✅ Implemented | `kvquant_n_sink`, excluded from the fit |
| §3.6 Offline calibration | ⚙️ Adapted | Fit online from prefill; zero calibration |
| §3.2 Pre-RoPE Key Quantization | ❌ Not implemented | Requires a model-forward hook |

**Adaptations:**

1. **Online level fitting (vs offline calibration).** The paper fits levels on a calibration corpus; we fit online from the prefill batch — zero setup, consistent with the suite's other adapted methods. Key levels are frozen after prefill, which plays the role of the paper's offline per-channel calibration.

2. **Per-channel codebooks (vs shared datatype + per-vector rescale).** The paper derives one datatype per layer and rescales it per vector. We fit levels independently per channel/token. This is strictly more expressive at equal code width, but stores a larger table — the cost is visible in `effective_bits` and amortizes with context.

### What is not implemented

- **Pre-RoPE key quantization (§3.2)** — KVQuant quantizes keys *before* rotary embedding (more quantization-friendly), then applies RoPE after dequant. This needs a model-forward hook to intercept pre-RoPE keys, outside the cache-only `update_and_fetch` contract. Our cache sees post-RoPE keys only. The paper attributes 0.82 perplexity of its LLaMA-7B 3-bit gain to this pillar, so our post-RoPE keys give up that portion of the benefit.
- **Fisher-information sensitivity weighting (Eq. 1)** — the paper weights each squared error by the diagonal Fisher term `F_ii`, computed from gradients on a calibration set. Our Lloyd-Max fit is the **unweighted special case** (all `F_ii` equal). Adding it would require a calibration pass with gradients.

---

## Evidence

| Claim | Source | Status |
|---|---|---|
| NUQ lower MSE than uniform at equal bits on non-uniform data | Test `test_nuq_beats_uniform_on_nonuniform` | ✅ Verified |
| NUQ not materially worse than uniform on uniform data | `test_nuq_not_worse_on_uniform` | ✅ Verified |
| Lloyd-Max distortion monotone non-increasing | `test_lloyd_max_monotone` | ✅ Verified |
| Dense/sparse split selects true top-k by magnitude | `test_split_selects_top_k` | ✅ Verified |
| Outlier isolation lowers MSE on heavy tails | `test_outlier_isolation_lowers_mse` | ✅ Verified |
| `outlier_fraction=0` reduces to plain NUQ | `test_outlier_fraction_zero_pure_nuq` | ✅ Verified |
| Level-table determinism | `test_level_table_determinism` | ✅ Verified |
| Frozen key levels across decode + correct accumulation | `test_decode_frozen_key_levels` | ✅ Verified |
| Outlier split exact under ties (no over-selection) | `test_split_exact_under_ties` | ✅ Verified |
| Attention sink tokens bit-exact in fp16 | `test_attention_sink_exact` | ✅ Verified |
| Sink off at `n_sink=0`; never applied to decode tokens | `test_sink_disabled_and_not_applied_at_decode` | ✅ Verified |
| Decode keys keep outlier protection (frozen threshold) | `test_decode_keys_keep_outlier_protection` | ✅ Verified |
| Sink tokens excluded from the level fit | `test_sink_excluded_from_level_fit` | ✅ Verified |
| Accounting tracks realized outliers + fp16 sink rows | `test_accounting_tracks_realized_outliers_and_sinks` | ✅ Verified |
| Per-channel (key) vs per-token (value) axes | `test_key_value_axes` | ✅ Verified |
| Byte accounting compressed below fp16 | `test_byte_accounting` | ✅ Verified |
| `effective_bits` within `[bits, bits + overhead]` | `test_effective_bits_range` | ✅ Verified |
| Determinism | `test_determinism` | ✅ Verified |
| ~73% MSE reduction vs uniform at 3-bit (Laplacian) | `benchmark_scripts/benchmark_kvquant.py` | Verified offline |
| Throughput + memory on M-series | `benchmark_scripts/benchmark_kvquant.py` | Run locally |

---

## Next steps

- [KIVI](./kivi) — uniform 2-bit group quantization (the uniform baseline NUQ improves on)
- [KIVI-Sink](./kivi-sink) — token-granularity outlier (sink) protection
- [SVDq](./svdq) — sub-2-bit keys via low-rank latent projection
- [Algorithm overview](./overview) — full method comparison
- [mlx_lm integration guide](../guides/mlx-lm-integration)
