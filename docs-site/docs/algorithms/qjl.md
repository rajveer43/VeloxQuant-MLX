---
id: qjl
title: QJL
sidebar_label: QJL
slug: /algorithms/qjl
---

# QJL

QJL (Quantized Johnson-Lindenstrauss) is the **simplest algorithm** in VeloxQuant-MLX. It uses a random Johnson-Lindenstrauss projection to reduce each key to a 1-bit sign sketch. No calibration, no codebook, no hyperparameters beyond sketch dimension.

## How it works

1. **Random projection** — Each key vector `k ∈ ℝᵈ` is projected to a lower-dimensional space: `z = Ak` where `A ∈ ℝ^{m×d}` is a random Gaussian matrix with `m < d`.

2. **Sign sketch** — The projected vector's sign is taken: `b = sign(z) ∈ {-1, +1}ᵐ`. This is packed into bit strings.

3. **Inner product approximation** — For a query `q`, the attention score `⟨q, k⟩` is approximated as:
   ```
   ⟨q, k⟩ ≈ (d/m) · ⟨Aq, b⟩
   ```
   This is the Johnson-Lindenstrauss lemma applied to inner products: the approximation error is bounded by `O(1/√m)`.

## Key properties

| Property | Value |
|---|---|
| Calibration | None |
| Key bits | 1 |
| Value bits | fp16 (default) or 2/4 |
| Sketch dimension | 64–256 (configurable) |
| Compression | 8–16× keys |
| Theoretical guarantee | JL lemma — bounded inner product error |

## Quickstart

```python
import mlx_lm
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="qjl",
    jl_dim=64,   # sketch dimension (m). Larger = better quality, more memory.
                 # Defaults to head_dim if left unset.
)
cache = KVCacheBuilder.build(model, config)

response = mlx_lm.generate(
    model, tokenizer,
    prompt="List 10 interesting facts about quantum computing.",
    max_tokens=300,
    kv_cache=cache,
)
```

## Using the quantizer directly

```python
import mlx.core as mx
from veloxquant_mlx.quantizers.qjl import QJLQuantizer

quantizer = QJLQuantizer(d=128, m=64, seed=42)

keys = mx.random.normal(shape=(512, 128))  # [N, D]

encoded = quantizer.encode(keys)
decoded = quantizer.decode(encoded)  # approximation, not exact reconstruction
```

:::note
`decode()` returns an approximation suitable for inner product computation — it does not reconstruct the original key vector exactly.
:::

## Configuration reference

`KVCacheConfig` field (when `method="qjl"`):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `jl_dim` | `Optional[int]` | `None` (→ `head_dim`) | Sketch dimension `m`. Must be ≤ `head_dim` |

`QJLQuantizer` constructor:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `d` | `int` | — | Key vector dimension (required) |
| `m` | `int` | `128` | Sketch dimension |
| `b` | `int` | `1` | Bits per sketch dimension |
| `seed` | `int` | `42` | Random seed for projection matrix `A` |

## Sketch dimension tradeoffs

| `jl_dim` (`m`) | Memory (128-dim keys) | Quality |
|---|---|---|
| 32 | 0.25 bit/dim | Poor — for large batches only |
| 64 | 0.5 bit/dim | Acceptable |
| 128 | 1 bit/dim | Good — matches head_dim |
| 256 | 2 bits/dim | Excellent but less compression |

## When to use QJL

**Use QJL when:**
- Simplicity is paramount (fewest moving parts, no tuning)
- Prototyping a new integration
- You want a theoretical guarantee on inner product approximation error

**Consider [TurboQuant RVQ](../algorithms/rvq) instead when:**
- Quality matters — RVQ consistently outperforms QJL at equal bits
- You are moving to production

## See also

- [RaBitQ — better 1-bit method with IVF](../algorithms/rabitq)
- [TurboQuant RVQ — best zero-calibration quality](../algorithms/rvq)
- [Quantizers API](../api/quantizers)
- [Metal API — `qjl_encode`, `qjl_inner_product`](../api/metal-api)
