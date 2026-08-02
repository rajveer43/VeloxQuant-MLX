---
id: quickstart
title: 5-Minute Quickstart
sidebar_label: Quickstart
slug: /getting-started/quickstart
---

# 5-Minute Quickstart

This guide gets you from a fresh install to compressed LLM inference in five minutes. You will load a model with `mlx_lm`, attach a TurboQuant RVQ KV cache, generate text, and print memory statistics.

:::note[Prerequisites]
Complete [Installation](../getting-started/installation) first. You need `mlx_lm` installed (`pip install mlx-lm`) and a model downloaded locally (e.g. `mlx-community/Llama-3.2-3B-Instruct-4bit`).
:::

:::tip[No Python? Start here instead]
`veloxquant panel` runs a local Start/Stop web UI and OpenAI-compatible server
— pick a model and method, no code required. See the
[Control Panel guide](../guides/control-panel).
:::

## Step 1 — Load a model

```python
import mlx_lm

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")
```

## Step 2 — Create a compressed KV cache

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

config = KVCacheConfig(
    method="turboquant_rvq",  # zero-calibration 1-bit RVQ
    bits=1,  # 1-bit keys, 2-bit values (default)
)

# Build per-layer cache matching the model architecture
cache = KVCacheBuilder.build(model, config)
```

## Step 3 — Generate with compression

```python
prompt = "Explain the key-value cache in large language models in simple terms."

response = mlx_lm.generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=512,
    kv_cache=cache,  # drop-in replacement for the default cache
    verbose=True,
)

print(response)
```

## Step 4 — Inspect memory savings

```python
from veloxquant_mlx.observers.memory import MemoryObserver

observer = MemoryObserver()
observer.attach(cache)

# Run a longer generation to see the savings
response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=2048, kv_cache=cache)

report = observer.report()
print(f"Peak compressed memory : {report.peak_compressed_mb:.1f} MB")
print(f"Equivalent fp16 memory : {report.peak_fp16_mb:.1f} MB")
print(f"Compression ratio      : {report.compression_ratio:.1f}×")
```

Example output on M3 Pro (Llama-3.2-3B, 2048 tokens):

```
Peak compressed memory : 48.3 MB
Equivalent fp16 memory : 362.0 MB
Compression ratio      : 7.5×
```

## Full script

```python
import mlx_lm
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder
from veloxquant_mlx.observers.memory import MemoryObserver

# Load model
model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Configure compressed cache
config = KVCacheConfig(method="turboquant_rvq", bits=1)
cache = KVCacheBuilder.build(model, config)

# Attach memory observer
observer = MemoryObserver()
observer.attach(cache)

# Generate
prompt = "Write a short story about a robot learning to paint."
response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=1024, kv_cache=cache)
print(response)

# Print stats
report = observer.report()
print(
    f"\nMemory: {report.peak_compressed_mb:.1f} MB "
    f"(vs {report.peak_fp16_mb:.1f} MB fp16, "
    f"{report.compression_ratio:.1f}× compression)"
)
```

## What just happened?

- `KVCacheConfig` describes which algorithm and bit-width to use
- `KVCacheBuilder.build()` creates one cache per transformer layer, matching the model's `num_key_value_heads` and `head_dim`
- During generation, each attention layer writes compressed keys/values via Metal GPU kernels instead of storing raw fp16 tensors
- The `MemoryObserver` tracks peak allocation and reports the savings

## Try a stronger algorithm

For higher accuracy at a slightly higher compute cost, switch to VecInfer (requires a one-time codebook training step):

```python
from veloxquant_mlx.allocators.vecinfer import train_codebook, calibrate_smooth_factors

# One-time calibration (save and reuse across sessions)
smooth_factors = calibrate_smooth_factors(model, tokenizer, num_samples=64)
codebook = train_codebook(model, tokenizer, smooth_factors, num_samples=128)

config = KVCacheConfig(
    method="vecinfer",
    bits=2,
    codebook=codebook,
    smooth_factors=smooth_factors,
)
cache = KVCacheBuilder.build(model, config)
```

See [VecInfer algorithm docs](../algorithms/vecinfer) and the [mlx_lm integration guide](../guides/mlx-lm-integration) for full details.

## Next steps

- [Core concepts — KV cache, quantization](../getting-started/concepts)
- [Choose the right algorithm](../algorithms/overview)
- [mlx_lm deep integration guide](../guides/mlx-lm-integration)
