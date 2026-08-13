---
id: polarquant
title: PolarQuant
sidebar_label: PolarQuant
slug: /algorithms/polarquant
---

# PolarQuant

PolarQuant uses **recursive polar coordinate decomposition** to represent keys as angles rather than Cartesian coordinates. This is particularly effective for models where keys form geometric clusters on a sphere — a distribution that standard scalar quantizers handle poorly.

## How it works

1. **Rotation** — Each key vector is first rotated by a random orthogonal matrix (decorrelates dimensions before the polar transform).

2. **Recursive polar decomposition** — The rotated vector is decomposed level by level into angles; each level's angles are quantized against a per-level codebook (`n_levels` codebooks total, sized `2**b` each).

3. **Geometric reconstruction** — Decoding reconstructs the original direction by composing the quantized angles in reverse order. The final radius (norm) is stored separately.

## Key properties

| Property | Value |
|---|---|
| Calibration | None |
| Bit-width | `b` bits per level, `n_levels` levels total |
| Best for | Models with spherical/normalized key geometry |

## Quickstart

```python
import mlx_lm
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Phi-3-mini-4k-instruct-4bit")

config = KVCacheConfig(
    method="polar",
    bit_width_inlier=2,
)
cache = KVCacheBuilder.build(model, config)

response = mlx_lm.generate(
    model,
    tokenizer,
    prompt="What are the main differences between Python and Go?",
    max_tokens=400,
    kv_cache=cache,
)
```

## Using the quantizer directly

```python
import mlx.core as mx
from veloxquant_mlx.quantizers.polarquant import PolarQuantizer

d = 64  # Phi-3 mini head_dim
quantizer = PolarQuantizer(d=d, b=2, seed=42)

keys = mx.array(mx.random.normal(shape=(4, d)))  # [batch, d] — 2D, not 4D

encoded = quantizer.encode(keys)
decoded = quantizer.decode(encoded)
```

## When to use PolarQuant

**Use PolarQuant when:**
- Key vectors are distributed approximately on a hypersphere (unit norm)
- The model uses normalised attention (Phi-3, Gemma-2 style)
- You want low-bit keys without calibration and without the JL approximation

**Consider [TurboQuant RVQ](../algorithms/rvq) instead when:**
- Keys are not spherically distributed (most Llama/Mistral variants)
- You need both key and value compression at high quality

## Configuration reference

`KVCacheConfig` fields (when `method="polar"`) — PolarQuant reuses the shared fields, it has no dedicated `polar_*` config block:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bit_width_inlier` | `int` | `2` | Bits per polar level |
| `head_dim` | `int` | `128` | Key/value dimension |
| `seed` | `int` | `42` | Random seed for the rotation matrix |

`PolarQuantizer` constructor:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `d` | `int` | — | Vector dimension (required) |
| `b` | `int` | `2` | Bits per level |
| `n_levels` | `int` | (module default) | Number of recursive polar levels |
| `seed` | `int` | `42` | Random seed |

## See also

- [CommVQ — RoPE compatibility](../algorithms/commvq)
- [TurboQuant RVQ — better quality for non-spherical keys](../algorithms/rvq)
- [Quantizers API](../api/quantizers)
