---
id: rabitq
title: RaBitQ
sidebar_label: RaBitQ
slug: /algorithms/rabitq
---

# RaBitQ

RaBitQ achieves **1-bit key compression** using a randomised Hadamard transform followed by binary sign packing with IVF (Inverted File Index) clustering. It delivers 6× total KV compression (keys at 1 bit, values at fp16) with zero calibration beyond a one-time `fit()` pass to train the IVF centroids.

:::warning[Apple Silicon required]
Hamming distance scoring uses `rabitq_hamming_score` — a Metal kernel with native XOR+popcount instructions.
:::

:::note[Quantizer-level API only]
`RaBitQQuantizer` is not yet wired into `KVCacheConfig`/`KVCacheBuilder` — there is no `method="rabitq"` cache option. Use the quantizer directly, as shown below.
:::

## How it works

1. **Randomised Hadamard transform** — Keys are multiplied by a random sign matrix then Walsh-Hadamard transformed. This spreads the energy uniformly and makes 1-bit sign encoding close to optimal (Johnson-Lindenstrauss guarantee).

2. **IVF clustering (`fit`)** — Keys are organised into `nlist` Voronoi cells via k-means. Each key stores its cluster ID plus the 1-bit residual within the cell — improving inner-product approximation over flat 1-bit encoding.

3. **1-bit sign packing** — The rotated residual is encoded as its sign and packed into bytes (`indices`, `[N, D//8]` uint8), giving 8× memory reduction over fp16 keys.

4. **Hamming distance scoring** — Attention scores are approximated by XOR+popcount Hamming distance between packed query bits and each packed key, run on Metal GPU cores via `rabitq_hamming_score`.

## Key properties

| Property | Value |
|---|---|
| Calibration | One-time `fit()` to train IVF centroids |
| Key bits | 1 (+ cluster ID + norm metadata overhead) |
| Value bits | Not handled by this quantizer (pair with another value quantizer or keep fp16) |
| Metal kernel | `rabitq_hamming_score` |

## Using the quantizer directly

```python
import numpy as np
import mlx.core as mx
from veloxquant_mlx.quantizers.rabitq import RaBitQQuantizer

d = 128  # head_dim

quantizer = RaBitQQuantizer(d=d, nlist=64, nprobe=8, rerank=32, seed=42)

# One-time calibration: train IVF centroids on a representative key sample
rng = np.random.default_rng(0)
calib_keys = rng.standard_normal((2048, d)).astype(np.float16)
quantizer.fit(mx.array(calib_keys))

keys = mx.array(rng.standard_normal((512, d)).astype(np.float16))  # [N, D]

encoded = quantizer.encode(keys)
# encoded.indices: packed sign bits, uint8  [N, D//8]
# encoded.norm:    [centroid_id, Cx, L1] per key, float32  [N, 3]

decoded = quantizer.decode(encoded)
```

## Configuration reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `d` | `int` | — | Key vector dimension (required) |
| `nlist` | `int` | `64` | IVF cluster count. Higher = better accuracy, more memory/compute at `fit()` |
| `nprobe` | `int` | `8` | Clusters probed per query at search time |
| `rerank` | `int` | `32` | Candidates kept for exact rerank after Hamming scoring |
| `seed` | `int` | `42` | Random seed for the Hadamard rotation and k-means init |

## When to use RaBitQ

**Use RaBitQ when:**
- 1-bit key compression is the goal
- You can pair it with a separate value-quantization strategy (or keep values at fp16)
- A one-time `fit()` calibration pass is acceptable

**Consider TurboQuant RVQ instead when:**
- You want a single `method=` string wired through `KVCacheConfig` today (RaBitQ is quantizer-level only)
- You need both keys and values compressed by the same method
- Quality at 1-bit is important (RVQ outperforms RaBitQ at equal compression)

## See also

- [QJL — simpler 1-bit method, wired into `KVCacheConfig`](../algorithms/qjl)
- [NSNQuant — universal-codebook VQ](../algorithms/nsnquant): differs because it adapts the data to a fixed codebook (NSN Gaussianization), not the codebook (or geometry) to the data
- [TurboQuant RVQ — better quality at same bits](../algorithms/rvq)
- [Metal API — rabitq_hamming_score](../api/metal-api)
