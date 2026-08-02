# Plan: VLM KV-Cache Quantization Support (TurboQuant, PolarQuant, QJL)

## Context

The project currently quantizes KV caches for text-only LLMs. The user wants to extend
TurboQuant, PolarQuant, and QJL to Vision Language Models (VLMs), targeting
**Qwen2-VL / Qwen3-VL** first. The goal is to compress the KV cache for multimodal
inference without changing the core quantization math.

**Key insight from exploration:** mlx_lm already handles VLMs via the same
`update_and_fetch(keys, values)` interface as text-only models. Vision features are
pre-projected into the language model's embedding space before the first transformer
layer — there is no separate "vision KV cache". The language model then attends over
both image tokens and text tokens in the same sequence. This means:

- The KV cache shape is still `(B, H, S, D)` — S just includes image patch tokens
- No cross-attention cache is needed
- The quantizer operates identically on image and text key vectors
- The only structural difference is that image tokens flood the sequence during prefill
  (e.g. 256–1024 image patches → large S on the first call), then S grows by 1 per decode step

The required changes are **shallow** — no changes to core quantizers, no new
cache types. Only cache construction and the benchmark layer need updating.

---

## Why the Core Quantizers Need No Changes

All three algorithms already operate on `(batch, d)` tensors:

| Algorithm | Input to quantizer | VLM compatibility |
|---|---|---|
| TurboQuant (MSE/Prod/RVQ) | `(B*H*S, D)` after reshape | Already works — D is head_dim, independent of token type |
| PolarQuant | `(d,)` per vector | Already works |
| QJL | `(d,)` per vector sign sketch | Already works |

The image patch key vectors after ViT projection into the language model's hidden space
follow the same distribution as text key vectors (both pass through the same attention
projection matrix). The rotation + Lloyd-Max codebook quantizes them identically.

---

## What Must Change

### 1. Per-layer head_dim detection for heterogeneous VLMs

**Problem:** Current `build_caches()` takes a single scalar `head_dim` and applies it
to all layers uniformly. Qwen2-VL has vision encoder layers with a potentially different
`head_dim` than language decoder layers.

**Fix:** Inspect each layer individually when building the cache list.

**File:** [benchmark_scripts/benchmark_core.py](../benchmark_scripts/benchmark_core.py)

```python
def build_vlm_caches(model, bits: int, use_rvq: bool = False, seed: int = 42):
    layers = model.model.layers
    caches = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            # Non-attention layer — use standard fp16 cache
            from mlx_lm.models.cache import KVCache as _MLXKVCache

            caches.append(_MLXKVCache())
            continue
        n_kv = getattr(attn, "n_kv_heads", None) or model.args.num_key_value_heads
        hd = getattr(attn, "head_dim", None) or (
            model.args.hidden_size // model.args.num_attention_heads
        )
        cls = TurboQuantRVQMLXKVCache if use_rvq else TurboQuantMLXKVCache
        caches.append(cls(n_kv_heads=n_kv, head_dim=hd, bits=bits, seed=seed + i))
    return caches
```

### 2. `KVCacheBuilder.for_model()` helper

**Problem:** Users integrating manually must know `head_dim` per layer. VLMs make this
non-trivial.

**Fix:** Add a static method to `KVCacheBuilder` that introspects the model:

**File:** [mlx_kv_quant/cache/base.py](../mlx_kv_quant/cache/base.py)

```python
@staticmethod
def for_model(model, config: "KVCacheConfig") -> list:
    """Build one KVCache per language-model layer, sized per-layer.
    Works for text-only and VLM models (Qwen2-VL, Qwen3-VL, etc.)."""
    layers = model.model.layers
    caches = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            from mlx_lm.models.cache import KVCache as _FallbackCache

            caches.append(_FallbackCache())
            continue
        hd = getattr(attn, "head_dim", None) or (
            model.args.hidden_size // model.args.num_attention_heads
        )
        layer_cfg = KVCacheConfig(
            method=config.method,
            head_dim=hd,
            bit_width_inlier=config.bit_width_inlier,
            seed=config.seed + i,
        )
        caches.append(KVCacheFactory.create(layer_cfg))
    return caches
```

### 3. New VLM benchmark script

**New file:** `benchmark_scripts/benchmark_qwen2_vl.py`

- Load `mlx-community/Qwen2-VL-2B-Instruct` (fits on 16GB) using `mlx_lm.load()`
- Construct a text+image prompt using mlx_lm's processor/tokenizer for VLMs
- Build cache list via `build_vlm_caches(model, bits, use_rvq)`
- Run 4 configs: fp16, TurboQuant RVQ 2-bit, TurboQuant 4-bit, PolarQuant 4-bit
- Measure tok/s and token completeness (out of 200)
- Save 6 figures to `figures/updated_tests/qwen2_vl/`

### 4. Synthetic VLM test (fast validation)

**File:** [benchmark_scripts/test_2bit_improvements.py](../benchmark_scripts/test_2bit_improvements.py)

Add a test section that synthesizes image-like key tensors (larger norms, 256 token
sequence, same head_dim) and verifies:
- RVQ 2-bit cosine similarity ≥ 0.96
- `build_vlm_caches()` returns correct number of caches with correct head_dim

---

## Implementation Order

1. Add `KVCacheBuilder.for_model()` to [mlx_kv_quant/cache/base.py](../mlx_kv_quant/cache/base.py)
2. Add `build_vlm_caches()` to [benchmark_scripts/benchmark_core.py](../benchmark_scripts/benchmark_core.py)
3. Add synthetic VLM test to [benchmark_scripts/test_2bit_improvements.py](../benchmark_scripts/test_2bit_improvements.py)
4. Write [benchmark_scripts/benchmark_qwen2_vl.py](../benchmark_scripts/benchmark_qwen2_vl.py)

---

## Files to be Modified

| File | Change |
|---|---|
| [mlx_kv_quant/cache/base.py](../mlx_kv_quant/cache/base.py) | Add `KVCacheBuilder.for_model()` static method |
| [benchmark_scripts/benchmark_core.py](../benchmark_scripts/benchmark_core.py) | Add `build_vlm_caches()` function |
| [benchmark_scripts/benchmark_qwen2_vl.py](../benchmark_scripts/benchmark_qwen2_vl.py) | New — VLM benchmark script |
| [benchmark_scripts/test_2bit_improvements.py](../benchmark_scripts/test_2bit_improvements.py) | Add synthetic VLM key test |

No changes to quantizers, codebooks, preconditioners, or core abstractions.

---

## Verification

1. `python3 benchmark_scripts/test_2bit_improvements.py` — synthetic VLM key test passes
2. `python3 benchmark_scripts/benchmark_qwen2_vl.py` — runs end-to-end without error
3. RVQ 2-bit produces ≥ 195 coherent tokens on the VLM prompt
4. tok/s for RVQ 2-bit is within 15% of fp16 (same bandwidth-bound ceiling as text models)

---

## Out of Scope

- Cross-attention cache (not used by Qwen2-VL in mlx_lm — vision is embedding-injected)
- Image token eviction / sliding window for vision tokens
- Fused Metal kernel for rotation+quantize
- Models other than Qwen2-VL in this round (Gemma3, Pixtral, Kimi-VL later)
