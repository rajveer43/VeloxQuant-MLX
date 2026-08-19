---
id: concepts
title: Core Concepts
sidebar_label: Core Concepts
slug: /getting-started/concepts
---

# Core Concepts

This page explains the fundamental ideas behind VeloxQuant-MLX without assuming prior knowledge of quantization theory. If you are already familiar with KV caches and vector quantization, skip to [Algorithm Overview](../algorithms/overview).

## What is a KV cache?

During autoregressive generation, a transformer computes **key** and **value** matrices at every layer for every token. Without caching, generating token `t` requires recomputing keys/values for all `t-1` prior tokens — quadratic cost.

The **KV cache** stores these matrices. Once a key/value pair is computed, it is written to the cache and reused for all future tokens. Generation becomes linear in sequence length at the cost of memory:

```
memory = num_layers × num_kv_heads × head_dim × seq_len × 2 (K+V) × 2 bytes (fp16)
```

For Llama-3.1-8B at 4096 context: `32 × 8 × 128 × 4096 × 4 = 536 MB`. At 32k context that becomes **4.2 GB** — often the binding constraint on Apple Silicon.

## Why compress the KV cache?

Compressing the KV cache trades a small quality loss for a large memory saving. The quality loss comes from quantization noise: instead of storing exact fp16 values, you store an approximation. Modern quantization methods are designed so the attention output changes by less than 1% even at 1-bit compression.

The memory saving is immediate: at 1 bit per value (vs 16 bits), you get a **16× reduction** in cache size. This enables:

- Longer context windows on the same hardware
- Running larger models that would otherwise OOM
- Fitting more concurrent inference sessions on one machine

## What is vector quantization?

**Scalar quantization** maps each number independently to the nearest value in a finite codebook. At 4-bit precision, the codebook has 16 values — each value is stored as a 4-bit index.

**Vector quantization (VQ)** groups numbers into vectors and maps the whole vector to the nearest centroid in a learned codebook. Because natural data has correlations between adjacent dimensions, VQ achieves much lower distortion than scalar quantization at the same bit rate.

**Residual VQ (RVQ)** applies VQ in multiple passes. The first quantizer encodes the raw vector; the second encodes the residual (error); the third encodes the residual of the residual. Each pass uses 1 bit, so three passes give effectively 3-bit fidelity with a simple 1-bit codebook structure.

## The compression-quality tradeoff

All quantization methods sit on a **rate-distortion curve**: lower bits = smaller memory, higher distortion. VeloxQuant-MLX algorithms vary in where they sit on this curve:

```
Quality
  ▲
  │  SpectralQuant (8-bit)
  │   VecInfer (4-bit)
  │    RateQuant (mixed)
  │     TurboQuant RVQ (2-bit)
  │      CommVQ / PolarQuant
  │       QJL / RaBitQ (1-bit)
  └────────────────────────────► Memory savings
```

The right choice depends on your use case. For most applications, **TurboQuant RVQ at 1–2 bits** is the best starting point — zero calibration, strong quality, 7–16× compression.

## What is calibration?

Some algorithms require a **calibration step** before inference: they analyse a small sample of real activations (typically 64–128 sequences) to learn model-specific statistics.

Examples:
- **VecInfer** trains a product-VQ codebook on real key distributions
- **SpectralQuant** computes an SVD rotation that aligns key dimensions with high-variance directions
- **RateQuant** probes per-layer sensitivity to allocate bits where distortion hurts most

Calibration artifacts (rotation matrices, codebooks, smooth factors) are saved to disk and reused across inference sessions. Zero-calibration methods (TurboQuant RVQ, QJL, RaBitQ, PolarQuant) use fixed analytical codebooks — no calibration needed, works on any model out of the box.

## Metal GPU kernels

VeloxQuant-MLX compiles Metal shaders at runtime using `mx.fast.metal_kernel`. These kernels run quantization and dequantization directly on the GPU:

- **13× faster** than equivalent MLX Python ops for VecInfer product VQ
- Fused decode+attention (`metal_fused_sdpa`) avoids materialising the full fp16 cache in VRAM
- Hamming distance for RaBitQ uses native XOR+popcount instructions

Metal kernels are loaded lazily — the first call to an algorithm triggers JIT compilation (typically 200–800 ms). Subsequent calls use the cached compiled kernel.

## Key abstractions

| Abstraction | Class | Role |
|---|---|---|
| Quantizer | `veloxquant_mlx.core.abstractions.Quantizer` | Encode/decode one tensor |
| KV Cache | `veloxquant_mlx.core.abstractions.KVCache` | Per-layer cache storing compressed K+V |
| Preconditioner | `veloxquant_mlx.core.abstractions.Preconditioner` | Linear transform applied before quantization |
| Codebook | `veloxquant_mlx.core.abstractions.Codebook` | Mapping from vectors to indices |
| Artifact Store | `veloxquant_mlx.artifacts.base.ArtifactStore` | Load/save calibration artifacts |

See [API Reference — Core](../api/core-api) for the full interface documentation.

## Next steps

- [Choose an algorithm](../algorithms/overview)
- [vs. llama.cpp / plain mlx_lm](./comparison) — why compress the cache instead of using what you already have
- [mlx_lm integration guide](../guides/mlx-lm-integration)
- [Calibration guide](../guides/calibration)
