---
id: ratequant
title: RateQuant
sidebar_label: RateQuant
slug: /algorithms/ratequant
---

# RateQuant

RateQuant is a **per-layer mixed-precision allocator**, not a standalone `method=`. It measures which layers are most sensitive to quantization, then allocates a per-layer bit budget via reverse-waterfilling — sensitive layers get more bits, insensitive layers get fewer. The resulting bit list is passed to any existing method (typically `turboquant_rvq`) as `bit_width_inlier`.

:::note[Allocator, not a cache method]
There is no `method="ratequant"`. RateQuant produces a `list[int]` that you assign to `KVCacheConfig.bit_width_inlier`, then build per-layer caches with `KVCacheBuilder.for_model(model, config)` instead of `KVCacheBuilder.build(model, config)`.
:::

## How it works

1. **Sensitivity probing** — `calibrate_layer_sensitivities(model, tokenizer, ...)` runs a short forward pass over calibration prompts and returns a per-layer sensitivity weight (higher = more error-prone at a fixed bit-width, so it should get more bits).

2. **Distortion-rate constant** — The reverse-waterfilling formula needs a decay constant `β` for the underlying quantizer's distortion-rate curve. `fit_distortion_curve(head_dim, ...)` estimates it from synthetic data — but its own docstring recommends skipping this for production use and passing the paper-reported constant directly (`β≈3.5` for TurboQuant RVQ, `β≈5.0` for KIVI/QuaRot-style methods).

3. **Reverse-waterfilling** — `allocate_bits_ratequant(sensitivities, target_avg_bits, beta, bit_choices)` solves the closed-form allocation from Theorem 2 of the RateQuant paper and returns a `list[int]` — one bit-width per layer, averaging to `target_avg_bits`.

4. **Per-layer cache construction** — Pass that list as `bit_width_inlier` and build with `KVCacheBuilder.for_model(...)`, which dispatches to `KVCacheFactory.create()` once per layer with the corresponding scalar bit-width.

## Key properties

| Property | Value |
|---|---|
| Calibration | Sensitivity probe over calibration prompts (duration depends on prompt count/length) |
| Output | `list[int]`, one bit-width per attention layer |
| Underlying method | Any method whose `KVCacheFactory.create()` path accepts a scalar `bit_width_inlier` (e.g. `turboquant_rvq`) |
| Build entry point | `KVCacheBuilder.for_model(model, config)`, not `.build()` |

## Step 1 — Calibrate and allocate

```python
import mlx_lm
from veloxquant_mlx.allocators.ratequant import (
    calibrate_layer_sensitivities,
    allocate_bits_ratequant,
)

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

sensitivities = calibrate_layer_sensitivities(
    model, tokenizer,
    seq_len=256,
)

bit_allocation = allocate_bits_ratequant(
    sensitivities,
    target_avg_bits=2.0,
    beta=3.5,               # paper-reported constant for TurboQuant RVQ
    bit_choices=(1, 2, 3),
)
print(bit_allocation)
# [3, 2, 1, ..., 2]  — one entry per attention layer
```

## Step 2 — Inference

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

config = KVCacheConfig(
    method="turboquant_rvq",
    bit_width_inlier=bit_allocation,  # list[int], one per layer
)
caches = KVCacheBuilder.for_model(model, config)

response = mlx_lm.generate(
    model, tokenizer,
    prompt="Write a detailed analysis of the economic impacts of automation.",
    max_tokens=1024,
    kv_cache=caches,
)
```

## Configuration reference

`allocate_bits_ratequant` parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sensitivities` | iterable of `float` | — | Per-layer weights from `calibrate_layer_sensitivities()` (required) |
| `target_avg_bits` | `float` | — | Desired mean bits/dim across layers (required) |
| `beta` | `float` | `3.5` | Distortion-rate decay constant for the underlying quantizer |
| `bit_choices` | `tuple[int, ...]` | `(1, 2, 3)` | Allowed integer bit-widths |

`calibrate_layer_sensitivities` parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | — | — | Loaded `mlx_lm` model (required) |
| `tokenizer` | — | — | Matching tokenizer (required) |
| `prompts` | `Optional[list]` | `None` (→ built-in defaults) | Calibration strings |
| `seq_len` | `int` | `256` | Max tokens per prompt |
| `verbose` | `bool` | `False` | Print per-sequence progress |

## Outlier-aware allocation

RateQuant's bit list is orthogonal to per-token outlier handling — there is currently no built-in hook that automatically routes high-key-norm tokens to a different bit-width. If you want to track which layers have heterogeneous key norms (a signal for whether RateQuant will help at all), use `KeyNormObserver` and check `report().heterogeneity_ratio` — see the [Observers guide](../guides/observers).

## When to use RateQuant

**Use RateQuant when:**
- Quality is the primary objective
- You can run a calibration pass over representative prompts
- You want fine-grained control over the accuracy-memory tradeoff
- You are dealing with models that have heterogeneous layer sensitivities

**Consider alternatives when:**
- Zero calibration required → [TurboQuant RVQ](../algorithms/rvq) with a single, uniform `bit_width_inlier`
- Maximum compression → [VecInfer](../algorithms/vecinfer)

## See also

- [Mixed-precision guide](../guides/mixed-precision)
- [Calibration guide](../guides/calibration)
- [Allocators API](../api/allocators)
- [Observers guide](../guides/observers)
