---
id: validation-report
title: Validation Report
sidebar_label: Validation Report
slug: /guides/validation-report
description: Defines the KV cache and quantization formulas precisely, and shows how to reproduce compression and throughput numbers honestly with scripts/validate_kv_memory.py on measured Apple Silicon results.
keywords: [validation report, kv memory formula, compression claim, accounting bytes, validate_kv_memory]
---

# Validation Report: KV Cache Quantization on Apple Silicon

This page explains what VeloxQuant-MLX does, how to measure it honestly, and
how to reproduce numbers with `scripts/validate_kv_memory.py`.

No marketing softeners: every claim is a definition, a formula, or a number
from a committed script. After you run the harness, cite
`figures/validation/<model>/results.json`.

## 1. What the KV cache is

During autoregressive generation, each transformer layer computes **key** (K)
and **value** (V) vectors for every token. The **KV cache** stores those
tensors so each new token only computes its own K/V and attends over history.

```text
bytes ≈ 2 × n_layers × n_kv_heads × head_dim × seq_len × 2
```

`seq_len` is tokens currently in the cache (prompt + generated).

## 2. What quantization does

**Quantization** stores fewer bits (or codebook indices) instead of fp16 so
the same RAM can hold a longer `seq_len`.

**Eviction** drops or merges tokens (different axis). **Cross-layer** methods
share state across depth.

Integration is three lines: build caches with `KVCacheConfig` /
`KVCacheBuilder.for_model`, pass them to `mlx_lm.generate` as `prompt_cache`.

## 3. Where reduction happens

| Axis | What shrinks | Examples |
| --- | --- | --- |
| Keys only | Packed key bytes | RVQ, VecInfer keys |
| Keys + values | Full KV packed bytes | RaBitQ, KIVI |
| Token count | `seq_len` | H2O, StreamingLLM |
| Across layers | Shared state | XQuant, MiniCache, xKV |

### Honest storage behavior (default RVQ / VecInfer)

1. Quantize incoming tensors.
2. Dequantize back to fp16.
3. Store in the parent `mlx_lm` `KVCache`.
4. Report `compressed_key_bytes` as packed-format **accounting**.

Headline **~7.5×** (RVQ-1bit, `d=128`) and **~16×** (VecInfer-1bit,
`sub_dim=8`) are accounting ratios unless a packed/fused path is active and
measured.

## Measured example (M4 Pro, 48 GB)

From committed `figures/validation/Llama-3.2-3B-Instruct-4bit/results.json`
(`max_tokens=64`, 120 tokens in cache):

| Arm | tok/s | Peak MB | Keys fp16 (acct) | Keys compressed (acct) | Claim |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp16 | 58.2 | 1954 | n/a | n/a | 1.00× |
| RVQ-1bit | 81.1 | 1855 | 6.562 MB | 0.872 MB | **7.53×** |
| VecInfer-1bit | 49.2 | 1926 | 6.562 MB | 0.410 MB | **16.00×** |

```text
claim = fp16_key_bytes / compressed_key_bytes
```

## 4. Before / after metrics

| Metric | Meaning | JSON field |
| --- | --- | --- |
| Tokens in cache | Context length held | `tokens_in_cache_max` |
| Keys fp16 size | Uncompressed key accounting | `fp16_key_bytes`, `fp16_key_mb` |
| Keys compressed size | Packed key accounting | `compressed_key_bytes`, `compressed_key_mb` |
| Compression claim | `fp16 / compressed` | `key_compression` |
| MLX peak MB | Weights + activations + temps | `peak_mb` |
| Throughput | tok/s | `throughput_tok_s` |
| Preview | Quality spot-check | `output_preview` |

At short generation, weights dominate peak MB. Grow prefill
(`--prompt-repeat`) before claiming cache-driven RAM wins.

## 5. Method defaults

| Goal | Method |
| --- | --- |
| Everyday, zero calibration | `turboquant_rvq` b=1 |
| Max key accounting | `vecinfer` 1-bit (calibrate) |
| Quality at moderate rate | `spectral` |
| Max context in tight RAM | `rabitq` or eviction |
| Constant memory | `streaming_llm` / `h2o` |

## 6. Reproduce

```bash
PYTHONPATH=. python scripts/validate_kv_memory.py \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --max-tokens 128
```

Writes `figures/validation/Llama-3.2-3B-Instruct-4bit/results.json`.

Also see the fuller write-up in the repo: [`docs/validation-report.md`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/docs/validation-report.md).

## 7. Formula check

RVQ (`d=128`, `b=1`): `(128×2) / (ceil(128×2×1/8)+2) ≈ 7.5×`.

VecInfer (`d=128`, `d_k=8`, 8-bit indices): `256 / 16 = 16×`.

Matching JSON ratios confirms counters, not resident RSS.
