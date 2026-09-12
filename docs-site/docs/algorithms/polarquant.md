---
id: polarquant
title: PolarQuant
sidebar_label: PolarQuant
slug: /algorithms/polarquant
description: PolarQuant rotates key vectors and recursively decomposes them into quantized angles rather than Cartesian coordinates, making it best suited to models whose keys form spherical or normalized geometric clusters.
keywords: [polarquant, polar coordinate decomposition, spherical key geometry, angle quantization, recursive codebook, normalized attention]
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

:::warning[Standalone method — not `mlx_lm.generate()`-compatible]
`method="polar"` is one of the library's `STANDALONE_METHODS`: `PolarQuantKVCache` implements VeloxQuant's own `append_key`/`append_value`/`attend` interface, not `mlx_lm`'s `update_and_fetch` protocol. `KVCacheBuilder.for_model()` and `patch_model_kv_cache()` both reject it with `QuantizerConfigError`, so it cannot be wired into `mlx_lm.generate()`. Build it directly via `KVCacheFactory.create()` and drive it with `append_key`/`append_value`/`attend`, as shown below.
:::

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

config = KVCacheConfig(
    method="polar",
    head_dim=64,  # match your model's per-head dimension
    bit_width_inlier=2,
)
cache = KVCacheFactory.create(config)

# Drive it directly, one key/value pair at a time (fp16 vectors, shape [head_dim])
cache.append_key(key_vector)
cache.append_value(value_vector)
output = cache.attend(query_vector)
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
