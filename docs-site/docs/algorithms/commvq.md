---
id: commvq
title: CommVQ
sidebar_label: CommVQ
slug: /algorithms/commvq
description: CommVQ-style RoPE-aware product vector quantizer for keys. Each codebook owns a disjoint slice of the head dimension. It is not the paper's additive scheme and does not enforce RoPE commutativity.
keywords: [commvq, product vq, rotary position embeddings, rotary position embeddings, metal kernel, quantizer-level api]
---

# CommVQ

CommVQ here is a **RoPE-aware product vector quantizer** for keys. It is inspired by the CommVQ paper but is not a faithful port.

:::caution[What is and is not implemented]
- It is plain **product quantization**: each of the `n_codebooks` sub-codebooks owns a disjoint slice of the head dimension, and decoding concatenates the slices. The paper's additive/residual scheme (each stage encoding the previous stage's error) is not implemented.
- Codebooks are trained on **pre-RoPE** keys and RoPE is applied once after decoding. No commutativity constraint is enforced. An earlier projection step claimed to do this but acted on the wrong dimension pairs, and was removed (reconstruction MSE rose about 2% on Gaussian data).
- `compression_ratio` counts the index bytes only. `stored_compression_ratio` also counts the 4-byte position that `encode()` returns (16x vs 32x at d=64, 4 codebooks). Codebook bytes are excluded from both.

Tracking issue: #756.
:::

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

CommVQ encodes pre-RoPE keys with one codebook per slice of the head dimension, and RoPE is applied once to the decoded vector at the stored position.

The Metal kernel `comm_vq_decode_metal` fuses centroid gathering and RoPE application in a single GPU pass.

## Key properties

| Property | Value |
|---|---|
| Calibration | One-time `fit()` to train the sub-codebooks |
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

# One-time calibration: train the sub-codebooks on a representative
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
