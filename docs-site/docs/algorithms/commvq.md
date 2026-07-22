---
id: commvq
title: CommVQ
sidebar_label: CommVQ
slug: /algorithms/commvq
---

# CommVQ

CommVQ (Commutative VQ) is designed for models using **Rotary Position Embeddings (RoPE)**. Standard quantization loses positional information because RoPE is applied after keys are written to the cache. CommVQ uses a residual VQ structure that **commutes with RoPE** — so position embeddings can be applied to quantized codes without dequantizing first.

:::note[Quantizer-level API only]
`CommVQQuantizer` is not yet wired into `KVCacheConfig`/`KVCacheBuilder` — there is no `method="comm_vq"` cache option. Use the quantizer directly, as shown below.
:::

## How it works

Standard KV cache flow with RoPE:
```
k_raw → apply_rope(position) → cache → dequant → attention
```

CommVQ flow:
```
k_raw → apply_rope → CommVQ_encode → cache → CommVQ_decode → attention
```

CommVQ encodes the RoPE-rotated key directly via a multi-codebook residual VQ. The codebook is structured so that applying a rotation to the centroid approximates the rotation of the residual — making position-aware decoding possible without storing per-token position metadata beyond the token index itself.

The Metal kernel `comm_vq_decode_metal` fuses centroid gathering and RoPE application in a single GPU pass.

## Key properties

| Property | Value |
|---|---|
| Calibration | One-time `fit()` to train the residual codebooks |
| Key bits | `b` bits per codebook × `n_codebooks` (e.g. `b=4, n_codebooks=4` on `head_dim=128` ≈ 4-bit keys) |
| Value bits | Not handled by this quantizer (pair with another value quantizer or keep fp16) |
| RoPE compatible | Yes — position applied at decode time |
| Metal kernel | `comm_vq_decode_metal` |

## Using the quantizer directly

```python
import numpy as np
import mlx.core as mx
from veloxquant_mlx.quantizers.comm_vq import CommVQQuantizer

d = 128  # head_dim (must be even)

quantizer = CommVQQuantizer(d=d, b=4, n_codebooks=4, seed=42)

# One-time calibration: train the residual codebooks on a representative
# sample of PRE-RoPE keys
rng = np.random.default_rng(0)
calib_keys = rng.standard_normal((2048, d)).astype(np.float16)
quantizer.fit(mx.array(calib_keys))

# Keys passed to encode() should be pre-RoPE; positions are stored
# alongside the codes so RoPE can be applied at decode time
keys = mx.array(rng.standard_normal((512, d)).astype(np.float16))  # [N, D]
positions = mx.arange(512, dtype=mx.int32)

encoded = quantizer.encode(keys, positions=positions)
decoded = quantizer.decode(encoded)  # RoPE-applied fp16 keys
```

## Why RoPE compatibility matters

With standard quantization, you must dequantize before applying RoPE (or store enough metadata to apply it correctly after the fact). CommVQ's codebook structure lets RoPE be applied during decode directly to the quantized code, so the cache never needs to store a separately-rotated copy.

This is particularly valuable for:
- Grouped Query Attention (GQA) models where KV heads are shared across many query heads
- Very long contexts (16k+) where any per-token metadata overhead compounds
- Deployment scenarios where minimising cache format complexity matters

## Configuration reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `d` | `int` | — | Key vector dimension, must be even (required) |
| `b` | `int` | `8` | Bits per codebook per residual pass |
| `n_codebooks` | `int` | `4` | Number of residual VQ codebooks |
| `seed` | `int` | `42` | Random seed |
| `rope_base` | `float` | `10000.0` | RoPE base frequency (must match the model) |
| `n_em_iters` | `int` | `50` | EM iterations used when fitting each codebook |

## When to use CommVQ

**Use CommVQ when:**
- The model uses RoPE positional encoding (Llama, Mistral, Qwen, Phi)
- You want to avoid re-deriving RoPE-rotated keys from scratch at decode
- You can pair it with a separate value-quantization strategy (or keep values at fp16)

**Consider [TurboQuant RVQ](../algorithms/rvq) instead when:**
- You want a single `method=` string wired through `KVCacheConfig` today (CommVQ is quantizer-level only)
- You need both key and value compression from the same method
- You want higher quality at equal bits

## See also

- [PolarQuant — geometric decomposition](../algorithms/polarquant)
- [Metal API — `comm_vq_decode_metal`](../api/metal-api)
