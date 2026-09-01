---
id: intro
title: What is VeloxQuant-MLX?
sidebar_label: Introduction
slug: /getting-started/intro
---

# What is VeloxQuant-MLX?

VeloxQuant-MLX is a **KV cache compression library** for Apple Silicon (M-series Macs). It implements 43 compression methods — quantizers, token-eviction caches, and cross-layer merging — that compress the key-value cache used during LLM inference, reducing peak memory by up to **98%** while maintaining near-lossless output quality.

LLMs like Llama, Mistral, and Qwen store past context in a KV cache that grows linearly with sequence length. On a MacBook M3 Pro with 18 GB unified memory, a 7B model at 8k context can consume 14 GB of cache alone — leaving almost no room for anything else. VeloxQuant-MLX compresses that cache on-the-fly with Metal GPU kernels, making long-context inference practical on consumer hardware.

```bash
pip install veloxquant-mlx
```

```python
from veloxquant_mlx.cache import KVCacheConfig
from veloxquant_mlx.integration.mlx_lm_patch import patch_model_kv_cache

config = KVCacheConfig(method="turboquant_rvq")  # zero-calibration, drop-in
patch_model_kv_cache(model, config)
```

All 43 methods share this same API — just swap `method="..."` — see the [5-minute quickstart](../getting-started/quickstart) for a full working example.

## Why Apple Silicon?

Apple's M-series chips have a unique advantage: **unified memory**. The GPU and CPU share the same memory pool, which means there is no PCIe bandwidth bottleneck between host and device. VeloxQuant-MLX is built specifically around this architecture:

- Metal GPU kernels run quantization/dequantization directly on the Neural Engine and GPU cores
- MLX — Apple's ML framework — provides the tensor primitives; VeloxQuant-MLX sits on top of it
- Quantized KV cache stays in unified memory, accessed by both the attention kernel and the quantizer with zero copies

## Key metrics

| Metric | Value |
|---|---|
| Max key cache compression | 16× (VecInfer 1-bit) |
| Metal kernel speedup | 13× faster quantization |
| Peak memory reduction | up to 98% |
| RVQ-1bit compression | 7.5× with zero calibration |
| RaBitQ full KV | 6× (keys + values) |
| Validated models | 12 (Llama, Mistral, Qwen, Phi, Gemma 3/4, Falcon) |
| Test suite | 3,321 passing tests |

## Algorithm overview

VeloxQuant-MLX ships 43 methods across three families, each adapted from a published paper:

- **Quantization** (22 methods) — compress every token's key/value vectors to low bits.
- **Low-rank & cross-layer** (6 methods) — compress across the hidden dimension or model depth.
- **Token eviction & merging** (15 methods) — drop or merge low-value tokens outright.

The six flagship methods below cover the most common tradeoffs — full comparison, decision guide, and all 43 methods (including every eviction/merging algorithm) are in the [Algorithm Overview](../algorithms/overview):

| Algorithm | Bits | Calibration | Best for |
|---|---|---|---|
| **TurboQuant RVQ** | 1–3+ | None | General purpose, drop-in replacement |
| **VecInfer** | 1–4 | Codebook training | Maximum throughput |
| **RateQuant** | mixed | 90 seconds | Mixed-precision accuracy-memory tradeoffs |
| **SpectralQuant** | 2–8 | SVD rotation | High-accuracy long context |
| **RaBitQ** | 1 | None | Key-only extreme compression |
| **SnapKV-adapted** | fp16 (kept tokens) | None | Token eviction — fixed memory budget regardless of context length |

## Next steps

- [Install VeloxQuant-MLX](../getting-started/installation)
- [5-minute quickstart](../getting-started/quickstart)
- [Core concepts — KV cache, quantization](../getting-started/concepts)
- [Full algorithm comparison — all 43 methods](../algorithms/overview)
