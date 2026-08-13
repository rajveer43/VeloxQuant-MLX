---
id: vecinfer
title: VecInfer
sidebar_label: VecInfer
slug: /algorithms/vecinfer
---

# VecInfer

VecInfer is VeloxQuant-MLX's highest-compression algorithm. It combines **product vector quantization** with per-channel smooth scaling and Metal GPU kernels.

:::warning[Apple Silicon required]
VecInfer's Metal kernels (`vecinfer_quantize_metal`, `vecinfer_encode_decode_metal`) are only available on macOS M-series; the pure-MLX fallback path still works elsewhere.
:::

:::note
The source itself documents that the paper's CUDA kernel fusion (Section 3.3) isn't portable to Metal — on Apple Silicon the win from VecInfer is **memory compression**, not a decode speedup over fp16 (see `veloxquant_mlx/cache/vecinfer_cache.py` module docstring).
:::

## How it works

1. **Smooth scaling** — Before quantization, each key channel is scaled by `λᵢ = √max|Kᵢ|` (`calibrate_smooth_factors`). This suppresses outlier channels that would otherwise dominate the codebook, similar to the technique from SmoothQuant.

2. **Walsh-Hadamard transform** — The scaled keys are rotated by a WHT matrix to decorrelate dimensions, making the distribution more uniform across subspaces.

3. **Product VQ (PVQ)** — The head dimension is split into sub-vectors of `key_sub_dim` (keys) / `value_sub_dim` (values) elements. Each sub-vector is independently quantized against a trained sub-codebook (`train_codebook`). The result is a short integer index per sub-vector.

4. **Metal-accelerated lookup** — `vecinfer_quantize_metal` and `vecinfer_encode_decode_metal` run the nearest-centroid search and encode/decode on GPU; `compute_query_lut` precomputes a query-codebook inner-product lookup table for the attention step.

## Key properties

| Property | Value |
|---|---|
| Calibration | Smooth-factor + codebook training (one-time, offline) |
| Key bits | `key_codebook_bits` (default `12`, i.e. `2**12` centroids) |
| Value bits | `value_codebook_bits` (default `8`) |
| Key sub-vector dim | `key_sub_dim` (default `4`) |
| Value sub-vector dim | `value_sub_dim` (default `8`) |
| Metal kernels | `vecinfer_quantize_metal`, `vecinfer_encode_decode_metal` |

## Calibration (one-time setup)

VecInfer needs a trained codebook and smooth factors. Calibrate on a representative sample of the model's actual key/value activations, then persist the arrays for reuse (`np.savez` is the simplest option — there's no dedicated VecInfer artifact-store method for this yet).

```python
import numpy as np
import mlx.core as mx
from veloxquant_mlx.allocators.vecinfer import (
    calibrate_smooth_factors,
    train_codebook,
)

head_dim = 128
key_sub_dim = 4
value_sub_dim = 8
key_bits = 12
value_bits = 8

# In practice, collect these from the model's real key/value activations
# over a calibration prompt set — shape [n_tokens, n_heads, head_dim].
keys_calib = mx.array(
    np.random.default_rng(0).standard_normal((4096, 8, head_dim)).astype(np.float32)
)
values_calib = mx.array(
    np.random.default_rng(1).standard_normal((4096, 8, head_dim)).astype(np.float32)
)

smooth_factors = calibrate_smooth_factors(keys_calib)

k_subs = keys_calib.reshape(-1, key_sub_dim)
v_subs = values_calib.reshape(-1, value_sub_dim)
key_codebook = train_codebook(k_subs, n_centroids=2**key_bits, seed=42)
value_codebook = train_codebook(v_subs, n_centroids=2**value_bits, seed=43)

np.savez(
    "vecinfer_artifacts.npz",
    smooth=np.asarray(smooth_factors),
    key_cb=np.asarray(key_codebook),
    value_cb=np.asarray(value_codebook),
)
```

## Inference

```python
import mlx_lm
import numpy as np
import mlx.core as mx
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

data = np.load("vecinfer_artifacts.npz")

config = KVCacheConfig(
    method="vecinfer",
    key_sub_dim=4,
    value_sub_dim=8,
    key_codebook_bits=12,
    value_codebook_bits=8,
    smooth_factors=mx.array(data["smooth"]),
    key_codebook=mx.array(data["key_cb"]),
    value_codebook=mx.array(data["value_cb"]),
)
cache = KVCacheBuilder.build(model, config)

response = mlx_lm.generate(
    model,
    tokenizer,
    prompt="Summarise the history of calculus in 300 words.",
    max_tokens=400,
    kv_cache=cache,
)
```

If `smooth_factors`/`key_codebook`/`value_codebook` are left as `None`, `VecInferKVCache` falls back to an identity smoothing factor and a randomly-initialized codebook — usable for testing the plumbing, not for real quality.

## Fused SDPA — current status

`patch_mlx_lm_for_fused_sdpa()` exists and monkey-patches `mlx_lm`'s attention dispatcher to route to a cache's `fused_sdpa()` method when available:

```python
from veloxquant_mlx.metal.fused_sdpa import patch_mlx_lm_for_fused_sdpa

patch_mlx_lm_for_fused_sdpa()  # no arguments; patches the mlx_lm module globally
```

:::warning[Currently a no-op for VecInfer's default path]
The dispatcher's own source comment states that as of the current version it does **not** change VecInfer's live generation loop: `VecInferKVCache.update_and_fetch` already returns a standard fp16 tensor, and profiling on Llama-3.1-8B showed the fused kernel can't beat MLX's SDPA on an already-materialized tensor. `cache.fused_sdpa(q)` remains callable directly for memory-bound configurations that want to skip materializing the fp16 buffer, but the automatic dispatch doesn't currently take that path for VecInfer.
:::

## Configuration reference

`KVCacheConfig` fields (when `method="vecinfer"`):

| Parameter | Type | Description |
|---|---|---|
| `key_sub_dim` | `int` | Key sub-vector dimension. Default `4` |
| `value_sub_dim` | `int` | Value sub-vector dimension. Default `8` |
| `key_codebook_bits` | `int` | Key codebook size = `2**key_codebook_bits`. Default `12` |
| `value_codebook_bits` | `int` | Value codebook size = `2**value_codebook_bits`. Default `8` |
| `smooth_factors` | `mx.array \| np.ndarray \| None` | Per-channel scaling from `calibrate_smooth_factors()`. Identity if `None` |
| `key_codebook` | `mx.array \| np.ndarray \| None` | Trained key codebook from `train_codebook()`. Random if `None` (testing only) |
| `value_codebook` | `mx.array \| np.ndarray \| None` | Trained value codebook from `train_codebook()`. Random if `None` (testing only) |
| `residual_length` | `int` | Default `128` |

## When to use VecInfer

**Use VecInfer when:**
- You can run an offline calibration pass and persist the codebook/smooth-factor artifacts
- Maximizing memory compression is the primary objective
- Context lengths are moderate (up to 8k)

**Consider alternatives when:**
- Zero calibration is required → [TurboQuant RVQ](../algorithms/rvq)
- Context exceeds 8k → [SpectralQuant](../algorithms/spectral)

## See also

- [Calibration guide](../guides/calibration)
- [Metal kernels guide](../guides/metal-kernels)
- [Allocators API](../api/allocators)
- [Metal API](../api/metal-api)
