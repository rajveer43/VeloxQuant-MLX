---
id: mixed-precision
title: Mixed-Precision Guide
sidebar_label: Mixed Precision
slug: /guides/mixed-precision
---

# Mixed-Precision Guide

Mixed-precision quantization assigns different bit rates to different layers based on their sensitivity to quantization noise. This achieves better accuracy than uniform quantization at the same average memory footprint.

## Why mixed precision?

Not all transformer layers are equally sensitive to quantization. Uniform low-bit quantization wastes bits on insensitive layers and starves sensitive ones. Mixed precision solves this with a per-layer bit assignment that minimises total distortion at a fixed average bit rate.

:::note[There is no `method="ratequant"`]
Mixed precision in this repo is implemented as a **per-layer `bit_width_inlier` list**, consumed by `KVCacheBuilder.for_model(...)` for any existing method (typically `turboquant_rvq`) — not a separate cache method. See the [RateQuant algorithm page](../algorithms/ratequant) for the full mechanism.
:::

## Method 1 — RateQuant (automatic allocation)

RateQuant probes per-layer sensitivity and allocates bits via reverse-waterfilling:

```python
import mlx_lm
from veloxquant_mlx.allocators.ratequant import (
    calibrate_layer_sensitivities,
    allocate_bits_ratequant,
)
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

sensitivities = calibrate_layer_sensitivities(model, tokenizer, seq_len=256)

bit_allocation = allocate_bits_ratequant(
    sensitivities,
    target_avg_bits=2.0,
    beta=3.5,
    bit_choices=(1, 2, 3),
)

config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=bit_allocation)
caches = KVCacheBuilder.for_model(model, config)  # one cache per layer, sized per-layer
```

See the [RateQuant algorithm page](../algorithms/ratequant) for the full reference.

## Method 2 — Manual allocation

If you know which layers are sensitive (from profiling or domain knowledge), skip the sensitivity probe and set `bit_width_inlier` directly to a `list[int]` of length `n_layers`:

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

n_layers = 32  # e.g. Llama-3.1-8B
bit_allocation = [1] * n_layers
for sensitive_layer in [8, 9, 10, 11, 12, 13, 14, 15, 16]:
    bit_allocation[sensitive_layer] = 4

config = KVCacheConfig(
    method="turboquant_rvq",
    bit_width_inlier=bit_allocation,
)
caches = KVCacheBuilder.for_model(model, config)
```

## Outlier token handling

:::warning[Not currently automated]
There is no built-in mechanism that detects outlier tokens and routes them to a higher-bit quantizer automatically. `KeyNormObserver` can tell you *whether* your keys are heterogeneous enough for mixed-precision to help (via `heterogeneity_ratio`), but it does not feed back into cache construction on its own.
:::

```python
from veloxquant_mlx.observers.key_norm import KeyNormObserver
from veloxquant_mlx.observers.base import QuantizationEvent

observer = KeyNormObserver()

# Feed it per-token key norm² however you compute it in your own pipeline
observer.on_event(QuantizationEvent(
    stage="key_norm",
    input_shape=keys.shape,
    metadata={"key_l2_norm_sq": per_token_norm_sq},
))

report = observer.report()
print(f"Heterogeneity ratio: {report.heterogeneity_ratio:.2f}")
# Per RateQuant Theorem 3: well above ~2.0 means mixed-precision allocation
# is likely to produce measurable gains over uniform quantization.
```

If you need actual per-token bit routing (not per-layer), several of the repo's eviction/mixed-bit methods do this natively — e.g. [ZipCache-adapted](../algorithms/zipcache) (`hi_fraction` tokens by key-norm get `hi_bits`, the rest get `lo_bits`) or [AdaKV-proxy](../algorithms/adakv) (per-head adaptive bits). See the [algorithm overview](../algorithms/overview) for the full list.

## Adaptive codebook variant

`TurboQuantProdAdaptive` is a thin subclass of `TurboQuantProd` that simply defaults `use_adaptive_codebook=True` — it is **not** runtime-adaptive to an observer's distortion feedback; there is no `base_bits`/`max_bits`/`distortion_threshold`/`observer` constructor argument.

```python
from veloxquant_mlx.quantizers.turboquant_prod import TurboQuantProdAdaptive

quantizer = TurboQuantProdAdaptive(d=128, b=3, seed=42)
# equivalent to: TurboQuantProd(d=128, b=3, seed=42, use_adaptive_codebook=True)
```

## See also

- [RateQuant algorithm](../algorithms/ratequant)
- [Observers guide](../guides/observers)
- [Allocators API](../api/allocators)
- [Observers API](../api/observers-api)
