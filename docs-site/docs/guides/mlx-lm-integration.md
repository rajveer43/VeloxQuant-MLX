---
id: mlx-lm-integration
title: mlx_lm Integration
sidebar_label: mlx_lm Integration
slug: /guides/mlx-lm-integration
---

# mlx_lm Integration

VeloxQuant-MLX is designed to work as a drop-in extension for `mlx_lm`. This guide covers the three integration patterns: the `KVCacheBuilder` helper, the `mlx_lm_patch` monkey-patch, and the fused SDPA kernel.

## Pattern 1 — KVCacheBuilder (recommended)

`KVCacheBuilder` is the primary integration point. It inspects the model's config and constructs one `KVCache` per transformer layer, matching `num_key_value_heads` and `head_dim` automatically.

```python
import mlx_lm
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(method="turboquant_rvq", bits=1, value_bits=2)
cache = KVCacheBuilder.build(model, config)

# Pass cache directly to mlx_lm.generate
response = mlx_lm.generate(
    model,
    tokenizer,
    prompt="Hello, world!",
    max_tokens=256,
    kv_cache=cache,
)
```

`KVCacheBuilder.build()` works with any model that exposes `model.args.num_hidden_layers`, `model.args.num_key_value_heads`, and `model.args.head_dim` — which covers all major mlx_lm model families.

## Pattern 2 — mlx_lm monkey-patch

The monkey-patch approach automatically intercepts the default KV cache creation inside mlx_lm and replaces it with VeloxQuant-MLX caches. This requires zero changes to your generation call:

```python
import mlx_lm
from veloxquant_mlx.integration.mlx_lm_patch import patch_model_kv_cache
from veloxquant_mlx.cache import KVCacheConfig

model, tokenizer = mlx_lm.load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
patch_model_kv_cache(model, config)  # overrides model.make_cache()

# No cache argument needed — mlx_lm.generate() builds the quantized cache
response = mlx_lm.generate(model, tokenizer, prompt="...", max_tokens=512)
```

:::tip
The monkey-patch is useful when integrating with third-party code that calls `mlx_lm.generate` directly and does not expose a cache argument.
:::

## Vision-language models (mlx-vlm)

`patch_vlm_kv_cache` wires VeloxQuant caches into [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) models (Qwen2-VL, LLaVA, etc.). mlx-vlm's single-prompt generation builds its cache through `model.language_model.make_cache()` — the patch overrides exactly that hook (verified against mlx-vlm 0.6.5):

```python
from mlx_vlm import load, generate
from veloxquant_mlx.integration.mlx_vlm_patch import patch_vlm_kv_cache
from veloxquant_mlx.cache import KVCacheConfig

model, processor = load("mlx-community/Qwen2-VL-2B-Instruct-4bit")
config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=2, seed=42)
patch_vlm_kv_cache(model, config)

output = generate(model, processor, prompt, image)
```

Behaviour to know:

- **Fresh caches per generation.** Unlike the text patch, `make_cache` rebuilds the cache list on every call, so repeated `generate()` calls never leak KV state between generations.
- **Single-prompt path only.** mlx-vlm's batched/session path converts caches with its own `to_batch_cache()`, which rejects foreign cache types — so batched generation keeps mlx-vlm's built-in caches (its native `kv_bits` quantization still works there). The top-level model is deliberately left unpatched to keep that path safe.
- **Eviction methods warn.** Token-dropping methods (`snapkv`, `h2o`, `pyramidkv`, …) may discard image tokens from the prompt prefix; the patch emits a `UserWarning` and quantization-only methods are recommended for multimodal prompts.

## Pattern 3 — Fused SDPA

`patch_mlx_lm_for_fused_sdpa` replaces the attention computation with a Metal kernel that dequantizes keys/values and computes attention in a single GPU pass:

```python
from veloxquant_mlx.metal.fused_sdpa import patch_mlx_lm_for_fused_sdpa

# Call once after loading the model
patch_mlx_lm_for_fused_sdpa(model)

# Subsequent generate calls use the fused kernel automatically
response = mlx_lm.generate(model, tokenizer, prompt="...", max_tokens=1024, kv_cache=cache)
```

Fused SDPA is most beneficial when:
- The KV cache is large (long sequences, many layers)
- You are using VecInfer (the fused kernel is optimised for its codebook format)
- Throughput is the priority over latency on individual calls

Check compatibility before patching:

```python
from veloxquant_mlx.metal.fused_sdpa import supports_shape

# Verify your model's attention shape is supported
is_supported = supports_shape(
    batch=1,
    heads=model.args.num_attention_heads,
    seq_len=4096,
    head_dim=model.args.head_dim,
)
print(f"Fused SDPA supported: {is_supported}")
```

## Streaming generation

VeloxQuant-MLX caches work transparently with mlx_lm's streaming API:

```python
for token in mlx_lm.stream_generate(
    model,
    tokenizer,
    prompt="Tell me a very long story.",
    max_tokens=4096,
    kv_cache=cache,
):
    print(token, end="", flush=True)
```

## Multi-turn conversations

For multi-turn chat, reuse the same cache across turns. The cache grows across turns but retains compression:

```python
config = KVCacheConfig(method="turboquant_rvq", bits=1)
cache = KVCacheBuilder.build(model, config)

turns = [
    "What is the capital of France?",
    "And what is it known for?",
    "What's the population?",
]

for turn in turns:
    response = mlx_lm.generate(model, tokenizer, prompt=turn, max_tokens=200, kv_cache=cache)
    print(f"User: {turn}\nAssistant: {response}\n")
    # cache now contains compressed K/V for all prior turns
```

:::warning
Cache capacity is bounded by `max_seq_len`. If the conversation exceeds this, use [SlidingWindowKVCache](../guides/sliding-window) to evict old tokens.
:::

## Supported models

All mlx_lm model families have been validated:

| Model family | Recommended config |
|---|---|
| Llama 3.1 / 3.2 / 3.3 | `method="turboquant_rvq", bits=1` |
| Mistral 7B / Mixtral | `method="vecinfer", bits=2` |
| Qwen 2.5 (7–72B) | `method="spectral", signal_bits=4` |
| Phi-3 / Phi-3.5 Mini | `method="commvq", bits=2` |
| Gemma 2B / 7B | `method="turboquant_rvq", bits=2` |
| Falcon 7B | `method="ratequant", target_bits=2.0` |

## See also

- [5-minute quickstart](../getting-started/quickstart)
- [Metal kernels guide](../guides/metal-kernels)
- [API — KVCacheBuilder](../api/cache)
