---
id: rvq
title: TurboQuant RVQ
sidebar_label: TurboQuant RVQ
slug: /algorithms/rvq
---

# TurboQuant RVQ

TurboQuant RVQ is the **recommended default algorithm** in VeloxQuant-MLX. It uses Residual Vector Quantization with analytical codebooks — no calibration required, works on any model out of the box.

:::warning[Apple Silicon required]
Metal kernels are used for scalar quantization and Hadamard rotation. Requires macOS M-series.
:::

## How it works

1. **Hadamard rotation** — Keys are multiplied by a Walsh-Hadamard matrix to spread any outlier energy evenly across dimensions. This makes the distribution more Gaussian, which is ideal for the codebooks.

2. **First-pass RVQ (Gaussian codebook)** — The rotated key vector is quantized with a Lloyd-Max Gaussian codebook at `bits` precision. The codebook is precomputed analytically — no training needed.

3. **Residual (Laplacian codebook)** — The quantization error from the first pass is encoded with a Laplacian codebook. Laplacian distributions have heavier tails, which better model residual distributions.

4. **Metal fused dequant + attention** — During the attention step, Metal kernels dequantize K directly and compute scaled dot-product attention without materialising the full fp16 cache.

## Key properties

| Property | Value |
|---|---|
| Calibration | None |
| Key bits | 1, 2, or 3+ |
| Value bits | 2 (default) or 4 |
| Compression ratio | 7.5× (1-bit) to 4× (2-bit) |
| Quality (cosine sim) | 0.97–0.99 |
| Metal kernel | Yes — `turboquant_hadamard_quantize` |

## Quickstart

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder
import mlx_lm

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="turboquant_rvq",
    bits=1,  # 1-bit keys (7.5× compression)
    value_bits=2,  # 2-bit values
)
cache = KVCacheBuilder.build(model, config)

response = mlx_lm.generate(
    model,
    tokenizer,
    prompt="Explain transformer attention in one paragraph.",
    max_tokens=512,
    kv_cache=cache,
)
print(response)
```

## Configuration reference

```python
KVCacheConfig(
    method="turboquant_rvq",
    bits=1,  # Key quantization bits (1, 2, 3). Default: 1
    value_bits=2,  # Value quantization bits (2, 4). Default: 2
    num_residuals=2,  # Number of RVQ passes. Default: 2
    use_hadamard=True,  # Apply WHT before quantization. Default: True
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bits` | `int` | `1` | Bits per key dimension per residual pass |
| `value_bits` | `int` | `2` | Bits per value dimension |
| `num_residuals` | `int` | `2` | Number of RVQ residual passes (total bits = bits × num_residuals) |
| `use_hadamard` | `bool` | `True` | Apply Walsh-Hadamard transform before quantization |

## Using the quantizer directly

```python
import mlx.core as mx
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ

quantizer = TurboQuantRVQ(bits=1, num_residuals=2, use_hadamard=True)

# keys: [batch, heads, seq_len, head_dim]
keys = mx.random.normal(shape=(1, 8, 512, 128))

encoded = quantizer.encode(keys)
decoded = quantizer.decode(encoded)

# Measure cosine similarity
cos_sim = mx.mean(
    mx.sum(keys * decoded, axis=-1)
    / (mx.linalg.norm(keys, axis=-1) * mx.linalg.norm(decoded, axis=-1))
).item()
print(f"Cosine similarity: {cos_sim:.4f}")  # typically 0.97-0.99
```

## When to use TurboQuant RVQ

**Use RVQ when:**
- You want to get started immediately with no calibration
- You are running on an unfamiliar model and do not have calibration data
- Memory is very tight (1-bit achieves 7.5× compression)
- Quality is important — RVQ consistently outperforms QJL and RaBitQ at the same bit rate

**Consider alternatives when:**
- Maximum throughput matters more than setup time → [VecInfer](../algorithms/vecinfer)
- You have 1–3 minutes for calibration and want the absolute best accuracy → [RateQuant](../algorithms/ratequant)
- Context length exceeds 8k → [SpectralQuant](../algorithms/spectral)

## Benchmark results

On Llama-3.1-8B at 4096 context, measured on M3 Pro (source: BENCHMARK_RESULTS.md):

| Bits | Memory | Compression | Perplexity delta |
|---|---|---|---|
| fp16 (baseline) | 536 MB | 1× | 0.00 |
| RVQ 2-bit | 134 MB | 4× | +0.08 |
| RVQ 1-bit | 71 MB | 7.5× | +0.21 |

## See also

- [Core concepts — RVQ explained](../getting-started/concepts)
- [mlx_lm integration](../guides/mlx-lm-integration)
- [API — TurboQuantRVQ](../api/quantizers)
- [Metal kernel — `turboquant_hadamard_quantize`](../api/metal-api)
