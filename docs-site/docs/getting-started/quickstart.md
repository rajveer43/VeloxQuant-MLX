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

## Step 2 — Wire in a compressed KV cache

```python
from veloxquant_mlx.cache import KVCacheConfig
from veloxquant_mlx.integration.mlx_lm_patch import patch_model_kv_cache

config = KVCacheConfig(
    method="turboquant_rvq",  # zero-calibration 1-bit RVQ
    bit_width_inlier=1,  # 1-bit inlier channels
    seed=42,
)

# Overrides model.make_cache() so mlx_lm builds the quantized cache automatically
patch_model_kv_cache(model, config)
```

## Step 3 — Generate with compression

```python
prompt = "Explain the key-value cache in large language models in simple terms."

response = mlx_lm.generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=512,
    verbose=True,
)

print(response)
```

No cache argument needed — `patch_model_kv_cache` already rewired `model.make_cache()`, so every `generate()` / `stream_generate()` call on this model builds a fresh quantized cache automatically.

## Step 4 — Inspect memory savings

```python
from veloxquant_mlx.observers.memory import MemoryObserver

observer = MemoryObserver()
config = KVCacheConfig(
    method="turboquant_rvq",
    bit_width_inlier=1,
    seed=42,
    observers=[observer],
)
patch_model_kv_cache(model, config)

# Run a longer generation to see the savings
response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=2048)

report = observer.report()
print(report)  # {'quantize': <bytes>, 'dequantize': <bytes>, ...} per pipeline stage
print(f"Peak single-stage delta: {observer.peak_delta_bytes()} bytes")
```

`report()` returns accounting deltas per pipeline stage (bytes saved by storing quantized values instead of fp16), not a resident-RAM measurement — see [Choose the right algorithm](../algorithms/overview) for which methods reduce accounting size only vs. actual resident memory before treating this as an Activity-Monitor-visible number.

## Full script

```python
import mlx_lm
from veloxquant_mlx.cache import KVCacheConfig
from veloxquant_mlx.integration.mlx_lm_patch import patch_model_kv_cache
from veloxquant_mlx.observers.memory import MemoryObserver

# Load model
model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Configure compressed cache + memory accounting
observer = MemoryObserver()
config = KVCacheConfig(
    method="turboquant_rvq",
    bit_width_inlier=1,
    seed=42,
    observers=[observer],
)
patch_model_kv_cache(model, config)

# Generate — no cache argument needed, model.make_cache() is already patched
prompt = "Write a short story about a robot learning to paint."
response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=1024)
print(response)

# Print accounting stats
print(f"\nMemory deltas by stage: {observer.report()}")
```

## What just happened?

- `KVCacheConfig` describes which algorithm and bit-width to use
- `patch_model_kv_cache()` overrides `model.make_cache()` so `mlx_lm.generate()` / `stream_generate()` build a quantized cache automatically, with no per-call cache argument
- During generation, each attention layer writes compressed keys/values via Metal GPU kernels instead of storing raw fp16 tensors
- The `MemoryObserver`, attached via `KVCacheConfig(observers=[...])`, records per-stage byte deltas you can inspect with `.report()`

## Try a stronger algorithm

For higher accuracy at a slightly higher compute cost, switch to VecInfer. It requires a one-time codebook training step (`calibrate_smooth_factors` + `train_codebook` from `veloxquant_mlx.allocators.vecinfer`) before it can be passed to `KVCacheConfig(method="vecinfer", ...)` — see the [VecInfer algorithm docs](../algorithms/vecinfer) for the full, runnable calibration snippet and the [mlx_lm integration guide](../guides/mlx-lm-integration) for wiring it into `patch_model_kv_cache`.

## Next steps

- [Core concepts — KV cache, quantization](../getting-started/concepts)
- [Choose the right algorithm](../algorithms/overview)
- [mlx_lm deep integration guide](../guides/mlx-lm-integration)
