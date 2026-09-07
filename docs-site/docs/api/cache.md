---
id: cache
title: Cache API
sidebar_label: Cache
slug: /api/cache
description: Python API reference for veloxquant_mlx.cache, covering the KVCacheConfig dataclass, the KVCacheFactory and KVCacheBuilder classes, and the concrete cache implementations TurboQuantRVQKVCache, VecInferKVCache, SpectralQuantKVCache, PolarQuantKVCache, QJLKVCache, and SlidingWindowKVCache.
keywords: [cache, KVCacheConfig, KVCacheBuilder, "API reference", "python api", KVCacheFactory, KV cache]
---

# Cache API

`veloxquant_mlx.cache`

The cache module provides the configuration system, factory, builder, and all KV cache implementations.

---

## KVCacheConfig

```python
from veloxquant_mlx.cache.base import KVCacheConfig
```

Dataclass that describes a quantization configuration.

```python
@dataclass
class KVCacheConfig:
    method: str
    bits: int = 1
    value_bits: int = 2
    num_residuals: int = 2
    use_hadamard: bool = True
    codebook: ndarray | None = None
    smooth_factors: ndarray | None = None
    rotations: list | None = None
    bit_allocation: dict[str, int] | None = None
    outlier_observer: KeyNormObserver | None = None
    outlier_bits: int = 8
    sketch_dim: int = 64
    num_clusters: int = 64
    num_subspaces: int | None = None
    use_fused_sdpa: bool = True
    signal_bits: int = 4
    noise_bits: int = 1
    seed: int = 0
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | Required | Algorithm name: `"turboquant_rvq"`, `"vecinfer"`, `"ratequant"`, `"spectral"`, `"rabitq"`, `"qjl"`, `"polarquant"`, `"commvq"` |
| `bits` | `int` | `1` | Key bit rate |
| `value_bits` | `int` | `2` | Value bit rate. `16` = fp16 (no compression) |
| `num_residuals` | `int` | `2` | RVQ residual passes (TurboQuant RVQ only) |
| `use_hadamard` | `bool` | `True` | Apply Walsh-Hadamard before quantization |
| `codebook` | `ndarray` | `None` | Trained product codebook (VecInfer required) |
| `smooth_factors` | `ndarray` | `None` | Per-channel scaling (VecInfer required) |
| `rotations` | `list` | `None` | SVD rotations (SpectralQuant required) |
| `bit_allocation` | `dict` | `None` | Per-layer bit map (RateQuant) |
| `sketch_dim` | `int` | `64` | JL sketch dimension (QJL) |
| `num_clusters` | `int` | `64` | IVF clusters (RaBitQ) |
| `signal_bits` | `int` | `4` | Bits for signal dimensions (SpectralQuant) |
| `noise_bits` | `int` | `1` | Bits for noise dimensions (SpectralQuant) |

---

## KVCacheFactory

```python
from veloxquant_mlx.cache.base import KVCacheFactory
```

Factory that maps a `KVCacheConfig` to a concrete `KVCache` instance.

### `KVCacheFactory.create`

```python
@staticmethod
def create(
    config: KVCacheConfig,
    num_heads: int,
    head_dim: int,
    max_seq_len: int = 8192,
) -> KVCache
```

Creates a single-layer KV cache.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `config` | `KVCacheConfig` | Quantization configuration |
| `num_heads` | `int` | Number of KV heads for this layer |
| `head_dim` | `int` | Dimension per attention head |
| `max_seq_len` | `int` | Pre-allocated sequence length |

**Returns:** A concrete `KVCache` subclass matching `config.method`.

---

## KVCacheBuilder

```python
from veloxquant_mlx.cache.base import KVCacheBuilder
```

High-level builder that inspects a model and creates per-layer caches automatically.

### `KVCacheBuilder.build`

```python
@staticmethod
def build(
    model,
    config: KVCacheConfig,
    max_seq_len: int = 8192,
) -> list[KVCache]
```

Creates one `KVCache` per transformer layer, matching layer-specific head counts and head dims.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `model` | mlx_lm model | Model loaded with `mlx_lm.load()` |
| `config` | `KVCacheConfig` | Quantization configuration |
| `max_seq_len` | `int` | Pre-allocated sequence length per cache |

**Returns:** `list[KVCache]` — one per layer, pass directly to `mlx_lm.generate(kv_cache=...)`.

**Example:**

```python
import mlx_lm
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")
config = KVCacheConfig(method="turboquant_rvq", bits=1)
cache = KVCacheBuilder.build(model, config)
# cache is a list of 28 TurboQuantRVQKVCache instances (one per Llama layer)
```

---

## Cache classes

### TurboQuantRVQKVCache

```python
from veloxquant_mlx.cache.turboquant_rvq_cache import TurboQuantRVQKVCache
```

KV cache backed by [TurboQuant RVQ](../algorithms/rvq). Writes compressed keys/values on each attention step and provides dequantized tensors for attention computation.

### VecInferKVCache

```python
from veloxquant_mlx.cache.vecinfer_cache import VecInferKVCache
```

[VecInfer](../algorithms/vecinfer) cache with smooth scaling + product VQ. Requires pre-trained codebook and smooth factors.

### SpectralQuantKVCache

```python
from veloxquant_mlx.cache.spectral_cache import SpectralQuantKVCache
```

[SpectralQuant](../algorithms/spectral) cache. Requires per-layer rotation matrices from `calibrate_spectral_rotation()`.

### PolarQuantKVCache

```python
from veloxquant_mlx.cache.polar_cache import PolarQuantKVCache
```

[PolarQuant](../algorithms/polarquant) cache. Zero calibration; encodes keys as polar angles.

### QJLKVCache

```python
from veloxquant_mlx.cache.qjl_cache import QJLKVCache
```

[QJL](../algorithms/qjl) 1-bit sign sketch cache.

### SlidingWindowKVCache

```python
from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache
```

Token eviction wrapper for any KVCache. See [Sliding Window guide](../guides/sliding-window).

---

## See also

- [mlx_lm integration guide](../guides/mlx-lm-integration)
- [API — Quantizers](../api/quantizers)
- [API — Core abstractions](../api/core-api)
