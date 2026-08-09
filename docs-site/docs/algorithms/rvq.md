---
id: rvq
title: TurboQuant RVQ
sidebar_label: TurboQuant RVQ
slug: /algorithms/rvq
---

# TurboQuant RVQ

TurboQuant RVQ is the **recommended default algorithm** in VeloxQuant-MLX. It uses Residual Vector Quantization with analytical codebooks — no calibration required, works on any model out of the box.

:::warning[Apple Silicon required]
Requires macOS M-series (MLX/Metal). This method uses MLX's built-in `mx.hadamard_transform` for rotation (Metal-accelerated by MLX itself); it does not use the hand-written custom Metal kernel (`turboquant_hadamard_quantize`) that the standalone `turboquant_prod`/`turboquant_mse` methods use — those are a different pair of classes that happen to share the "TurboQuant" name lineage.
:::

## How it works

1. **Hadamard rotation** — Keys are multiplied by a Walsh-Hadamard matrix to spread any outlier energy evenly across dimensions. This makes the distribution more Gaussian, which is ideal for the codebooks.

2. **First-pass RVQ (Gaussian codebook)** — The rotated key vector is quantized with a Lloyd-Max Gaussian codebook at `bit_width_inlier` precision. The codebook is precomputed analytically — no training needed.

3. **Residual (Laplacian codebook)** — The quantization error from the first pass is encoded with a Laplacian codebook. Laplacian distributions have heavier tails, which better model residual distributions.

4. **Packed storage** — Both codebook-index streams are bit-packed into `uint32` words (not stored as dequantized fp16); values pass through unchanged. Dequantization (codebook gather + inverse rotation) happens transiently on every fetch, the same tradeoff `mlx_lm`'s own native `QuantizedKVCache` accepts. See [issue #27](https://github.com/rajveer43/VeloxQuant-MLX/issues/27) and [docs/RVQ_PACKED_STORAGE_FINDINGS.md](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/docs/RVQ_PACKED_STORAGE_FINDINGS.md) for the full investigation.

:::info[Accounting vs. measured]
Earlier versions of this cache dequantized keys back to fp16 immediately after encoding and stored that — the compression ratio was bit-width *accounting*, not a reduction in actual resident memory. As of the packed-storage fix, keys are genuinely stored packed; see the measured benchmark below for real `mx.get_peak_memory()` numbers, not an estimate.
:::

## Key properties

| Property | Value |
|---|---|
| Calibration | None |
| Key bits | 1, 2, or 3+ (`bit_width_inlier`) |
| Storage | Genuinely packed (bit-packed `uint32` index streams) — not dequantized fp16 |
| Compression ratio (accounting) | 7.5× (1-bit) to 4× (2-bit) |
| Measured peak-memory reduction (1-bit, 4k-token context) | -12.8% vs. fp16 baseline |
| Quality (cosine sim) | 0.92 (1-bit) – 0.98 (2-bit) |

## Quickstart

```python
import mlx_lm
from veloxquant_mlx.cache import KVCacheConfig
from veloxquant_mlx.integration.mlx_lm_patch import patch_model_kv_cache

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="turboquant_rvq",
    bit_width_inlier=1,  # 1-bit keys (7.5x accounting compression)
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
- Memory is tight (measured -12.8% peak RSS vs. fp16 at 1-bit, plus 7.5× accounting compression on top)
- Quality is important — RVQ consistently outperforms QJL and RaBitQ at the same bit rate

**Consider alternatives when:**
- Maximum throughput matters more than setup time → [VecInfer](../algorithms/vecinfer)
- You have 1–3 minutes for calibration and want the absolute best accuracy → [RateQuant](../algorithms/ratequant)
- Context length exceeds 8k → [SpectralQuant](../algorithms/spectral)

## Benchmark results

:::caution[Previous table removed]
This section previously cited a Llama-3.1-8B / 4096-context / M3 Pro table (536 MB / 134 MB / 71 MB) attributed to `BENCHMARK_RESULTS.md`. That file does not contain those figures — the table could not be traced to an actual measurement and has been replaced below with numbers verified during the packed-storage fix, methodology included. If you have a real Llama-3.1-8B-scale measurement, please contribute it rather than restoring the old table.
:::

**Measured (not accounting) peak memory** — `mx.get_peak_memory()`, gc-controlled across repeated runs, `mlx-community/Llama-3.2-1B-Instruct-4bit`, 4002-token prompt + 100 decode tokens (full methodology, including two measurement false starts worth avoiding: [docs/RVQ_PACKED_STORAGE_FINDINGS.md](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/docs/RVQ_PACKED_STORAGE_FINDINGS.md)):

| Configuration | Peak memory | vs. fp16 |
|---|---|---|
| fp16 baseline (`mlx_lm.models.cache.KVCache`) | 1608.4 MB | — |
| `mlx_lm`'s native `QuantizedKVCache(bits=4)` | 2537.2 MB | +57.7% (worse — see note below) |
| TurboQuant RVQ, b=1, packed storage | **1402.3 MB** | **-12.8%** |

`mlx_lm`'s own native quantized cache is *worse* than fp16 at this prompt length because its `mx.quantize()` call materializes full intermediate buffers on a single large prefill call; this cache writes packed storage incrementally per call and does not pay that cost. Treat this as one measured data point with its methodology stated, not a universal claim across all context lengths or prefill chunking strategies — see the findings doc for the full discussion.

Separately, **accounting-ratio / quality** numbers (bit-width compression vs. cosine similarity, not a memory measurement) at `d=128`:

| Bits | Accounting compression | Cosine similarity |
|---|---|---|
| RVQ 1-bit | 7.5× | ~0.92 |
| RVQ 2-bit | 4× | ~0.98 |

## See also

- [Core concepts — RVQ explained](../getting-started/concepts)
- [mlx_lm integration](../guides/mlx-lm-integration)
- [API — TurboQuantRVQ](../api/quantizers)
- [Packed storage investigation — docs/RVQ_PACKED_STORAGE_FINDINGS.md](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/docs/RVQ_PACKED_STORAGE_FINDINGS.md)
