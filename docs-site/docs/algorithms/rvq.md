---
id: rvq
title: TurboQuant RVQ
sidebar_label: TurboQuant RVQ
slug: /algorithms/rvq
---

# TurboQuant RVQ

TurboQuant RVQ is the **recommended default algorithm** in VeloxQuant-MLX. It uses Residual Vector Quantization with analytical codebooks — no calibration required, works on any model out of the box.

:::warning[Apple Silicon required]
Requires macOS M-series (MLX/Metal). Rotation runs on MLX's built-in `mx.hadamard_transform`, which is Metal-accelerated automatically.
:::

## How it works

1. **Hadamard rotation** — Keys are multiplied by a Walsh-Hadamard matrix to spread any outlier energy evenly across dimensions. This makes the distribution more Gaussian, which is ideal for the codebooks.

2. **First-pass RVQ (Gaussian codebook)** — The rotated key vector is quantized with a Lloyd-Max Gaussian codebook at `bit_width_inlier` precision. The codebook is precomputed analytically — no training needed.

3. **Residual (Laplacian codebook)** — The quantization error from the first pass is encoded with a Laplacian codebook. Laplacian distributions have heavier tails, which better model residual distributions.

4. **Packed key storage** — Both codebook-index streams are bit-packed into compact `uint32` words rather than stored as full-size fp16 tensors. Values are stored uncompressed. Keys are unpacked back to fp16 only for the instant they're needed during attention.

## Key properties

| Property | Value |
|---|---|
| Calibration | None |
| Key bits | 1, 2, or 3+ (`bit_width_inlier`) |
| Compression ratio | 7.5× (1-bit) to 4× (2-bit) |
| Measured memory savings (1-bit, 4k-token context) | 12.8% lower peak memory than fp16 |
| Quality (cosine similarity) | 0.92 (1-bit) – 0.98 (2-bit) |

## Quickstart

```python
import mlx_lm
from veloxquant_mlx.cache import KVCacheConfig
from veloxquant_mlx.integration.mlx_lm_patch import patch_model_kv_cache

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="turboquant_rvq",
    bit_width_inlier=1,  # 1-bit keys (7.5x compression)
    seed=42,
)

# Overrides model.make_cache() so mlx_lm builds the quantized cache automatically
patch_model_kv_cache(model, config)

response = mlx_lm.generate(
    model,
    tokenizer,
    prompt="Explain transformer attention in one paragraph.",
    max_tokens=512,
)
print(response)
```

## Configuration reference

```python
KVCacheConfig(
    method="turboquant_rvq",
    head_dim=128,        # Attention head dimension. Default: 128
    bit_width_inlier=1,  # Bits per key dimension per RVQ stage. Default: 2
    seed=42,             # Random seed for the rotation matrix. Default: 42
)
```

`TurboQuantRVQ` always runs two RVQ stages (Gaussian + Laplacian residual) at
the same bit-width — there is no separate `value_bits`/`num_residuals` knob;
values always pass through as plain fp16.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `head_dim` | `int` | `128` | Attention head dimension |
| `bit_width_inlier` | `int` | `2` | Bits per key dimension, per RVQ stage (total key storage cost scales with `2 * bit_width_inlier`) |
| `seed` | `int` | `42` | Random seed for the Hadamard/rotation preconditioner |

## Using the quantizer directly

```python
import mlx.core as mx
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ

quantizer = TurboQuantRVQ(d=128, b=1, seed=42, use_hadamard=True)

# keys: [batch, d] -- flatten (batch, heads, seq_len, head_dim) to this shape first
keys = mx.random.normal(shape=(512, 128))

encoded = quantizer.encode(keys)
decoded = quantizer.decode(encoded)

# Measure cosine similarity
cos_sim = mx.mean(
    mx.sum(keys * decoded, axis=-1)
    / (mx.linalg.norm(keys, axis=-1) * mx.linalg.norm(decoded, axis=-1))
).item()
print(f"Cosine similarity: {cos_sim:.4f}")  # ~0.92 at b=1, ~0.98 at b=2
```

## When to use TurboQuant RVQ

**Use RVQ when:**
- You want to get started immediately with no calibration
- You are running on an unfamiliar model and do not have calibration data
- Memory is tight (up to 7.5× compression at 1-bit, with measured lower peak memory than fp16)
- Quality is important — RVQ consistently outperforms QJL and RaBitQ at the same bit rate

**Consider alternatives when:**
- Maximum throughput matters more than setup time → [VecInfer](../algorithms/vecinfer)
- You have 1–3 minutes for calibration and want the absolute best accuracy → [RateQuant](../algorithms/ratequant)
- Context length exceeds 8k → [SpectralQuant](../algorithms/spectral)

## Benchmark results

Measured on `mlx-community/Llama-3.2-1B-Instruct-4bit`, a 4,000-token prompt plus 100 generated tokens, peak memory as reported by MLX:

| Configuration | Peak memory | vs. fp16 |
|---|---|---|
| fp16 (no compression) | 1608 MB | — |
| mlx-lm's built-in 4-bit cache | 2537 MB | 58% higher |
| TurboQuant RVQ, 1-bit | **1402 MB** | **12.8% lower** |

At this context length, `mlx_lm`'s own built-in quantized cache actually uses *more* memory than no compression at all — RVQ's packed storage avoids that overhead. Results will vary with context length and model size.

Compression ratio and quality by bit-width (`d=128`):

| Bits | Compression | Cosine similarity |
|---|---|---|
| 1-bit | 7.5× | ~0.92 |
| 2-bit | 4× | ~0.98 |

## See also

- [Core concepts — RVQ explained](../getting-started/concepts)
- [mlx_lm integration](../guides/mlx-lm-integration)
- [API — TurboQuantRVQ](../api/quantizers)
