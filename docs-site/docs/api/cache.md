---
id: cache
title: Cache API
sidebar_label: Cache
slug: /api/cache
description: Python API reference for veloxquant_mlx.cache, covering the KVCacheConfig dataclass, the KVCacheFactory and KVCacheBuilder classes, and the concrete cache implementations TurboQuantRVQKVCache, VecInferKVCache, SpectralQuantKVCache, PolarQuantKVCache, QJLKVCache, and SlidingWindowKVCache.
keywords: [cache, KVCacheConfig, KVCacheBuilder, "API reference", "python api", KVCacheFactory, KV cache]
---

# Cache API

`veloxquant_mlx.cache`

The cache module provides the configuration system, factory, builder, and all KV cache implementations. The registry (`veloxquant_mlx.cache.registry`) currently tracks **43 methods** sharing one config surface; this page documents the core construction API plus a curated subset of 6 cache classes. See [Registry](#registry-introspection) below for how to enumerate the full method list.

---

## KVCacheConfig

```python
from veloxquant_mlx.cache.base import KVCacheConfig
```

Single dataclass holding hyperparameters for every method in the registry (over 40 quantization/eviction methods share this one config surface), each namespacing its own fields with a method-specific prefix (e.g. `kivi_group_size`, `svdq_rank`, `kvquant_bits`).

### Core parameters

These apply across most methods; everything else is method-specific (see below).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | `MethodName` (str Literal) | `"turboquant_rvq"` | Algorithm name. One of 43 values, e.g. `"turboquant_rvq"`, `"vecinfer"`, `"spectral"`, `"polar"`, `"qjl"`, `"kivi"`, `"kvquant"`, `"h2o"`, `"snapkv"`, etc. |
| `head_dim` | `int` | `128` | Attention head dimension (d) |
| `bit_width_inlier` | `int \| list` | `2` | Bit-width for inlier channels. A single `int` applies uniformly; a `list[int]` of length `n_layers` gives per-layer (RateQuant-style) allocation — only consumed by `KVCacheBuilder.for_model()`, rejected by `KVCacheFactory.create()` |
| `bit_width_outlier` | `int \| None` | `None` | Bit-width for outlier channels (`None` → same as inlier) |
| `jl_dim` | `int \| None` | `None` | Johnson-Lindenstrauss projection dimension (QJL) |
| `n_outlier_channels` | `int \| None` | `None` | Number of outlier channels to detect |
| `n_calib_tokens` | `int \| None` | `None` | Calibration token count for outlier activation |
| `enable_vectorized_attend` | `bool` | `True` | Vectorized packed-key unpack in `attend()` |
| `enable_outlier_two_stream` | `bool` | `False` | Outlier/inlier split cache after calibration |
| `enable_fused_query_dot` | `bool` | `False` | Fused rotated-query + codebook-dot path |
| `seed` | `int` | `42` | Random seed |
| `dtype` | `Any` | `None` | MLX dtype for computations |
| `capacity` | `int \| None` | `None` | Maximum tokens to store (`None` → unlimited) |
| `sliding_window` | `int \| None` | `None` | If set, wrap the cache with sliding-window eviction (only valid for [standalone methods](#standalone-methods)) |
| `store` | `ArtifactStore \| None` | `None` | `ArtifactStore` to load precomputed artifacts from |
| `observers` | `list` | `[]` | List of `QuantizationObserver` instances |
| `use_metal_kernels` | `bool \| None` | `None` | Metal fast-path for VecInfer quantize/dequant: `None` auto-detects, `True` requires it, `False` forces pure MLX |
| `fused_sdpa` | `bool \| None` | `False` | Enable the fused dequant+SDPA Metal kernel path |
| `fused_sdpa_max_ctx` | `int` | `8192` | Pre-allocated index ring-buffer capacity (tokens) when `fused_sdpa=True` |
| `fused_sdpa_memory_bound` | `bool` | `False` | Skip fp16 K/V materialization entirely (requires `fused_sdpa=True` and `patch_mlx_lm_for_fused_sdpa()` active) |

### Method-specific parameters (selected)

Every other field follows a `{method}_*` naming prefix (with a small number of aliases, e.g. `snapkv`'s fields are `snap_*`, `streaming_llm`'s are `stream_*`, `pyramidkv`'s are `pyramid_*`). A representative sample relevant to the cache classes documented below:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `key_sub_dim` | `int` | `4` | VecInfer key sub-vector dimension |
| `value_sub_dim` | `int` | `8` | VecInfer value sub-vector dimension |
| `key_codebook_bits` | `int` | `12` | VecInfer key codebook bits |
| `value_codebook_bits` | `int` | `8` | VecInfer value codebook bits |
| `residual_length` | `int` | `128` | VecInfer recent tokens kept uncompressed |
| `kivi_group_size` | `int` | `32` | KIVI min/max group size |
| `spectral_key_d_eff` | `int` | `4` | SpectralQuant signal dimensions for keys |
| `spectral_val_d_eff` | `int` | `50` | SpectralQuant signal dimensions for values |
| `spectral_apply_qjl` | `bool` | `True` | Apply QJL on signal dims only |
| `spectral_model_name` | `str` | `"model"` | Identifier for rotation cache on disk |
| `kvquant_bits` | `int` | `3` | KVQuant-NUQ base bit-width |
| `kvquant_group_size` | `int` | `32` | KVQuant-NUQ group size for per-channel/per-token fitting |
| `kvquant_outlier_fraction` | `float` | `0.01` | KVQuant-NUQ top-magnitude fraction kept fp16 |
| `gear_bits` | `int` | `2` | GEAR ultra-low base bit-width |
| `gear_group_size` | `int` | `32` | GEAR base group-quant token group size |
| `svdq_group_size` | `int` | `32` | SVDq group size for latent quantization |

This is a small sample — the full field list runs to roughly 200 fields across 43 methods (KIVI, SVDq, Kitty, AdaKV, XQuant, KVQuant, PALU, CacheGen, MiniCache, GEAR, ZipCache, SnapKV, StreamingLLM, H2O, TOVA, PyramidKV, SqueezeAttention, ChunkKV, CaM, xKV, NSNQuant, K-norm, SKVQ, Q-Filters, Keyformer, MorphKV, KVzip, KVTC, CurDKV, NestedKV, AMC, A2ATS, AnchorKV, RocketKV, AgeTieredKV, and more). Read `veloxquant_mlx/cache/base.py`'s `KVCacheConfig` dataclass directly for the authoritative, complete list, or use [`registry.describe_field()`](#registry-introspection) to introspect a single field programmatically.

### Standalone methods

```python
from veloxquant_mlx.cache.base import STANDALONE_METHODS
# frozenset({"turboquant_prod", "turboquant_mse", "polar", "qjl", "spectral"})
```

Five methods implement VeloxQuant's own `KVCache` ABC (`append_key`/`append_value`/`attend`/`memory_bytes`) instead of `mlx_lm`'s serving protocol (`update_and_fetch`/`nbytes`/`state`/`trim`/`merge`/`meta_state`). Because `mlx_lm.generate()` drives caches purely through the latter interface, a standalone-method cache cannot be used with `KVCacheBuilder.for_model()` or `patch_model_kv_cache` — both reject standalone methods by raising `QuantizerConfigError` rather than failing deep inside generation. `KVCacheFactory.create()` is the only valid construction path for these methods (direct/research use, not `mlx_lm` serving). Note this means **SpectralQuantKVCache, PolarQuantKVCache, and QJLKVCache — three of the six cache classes below — are standalone and cannot be built via `for_model()`.**

---

## KVCacheFactory

```python
from veloxquant_mlx.cache.base import KVCacheFactory
```

Factory that maps a `KVCacheConfig` to a concrete cache instance.

### `KVCacheFactory.create`

```python
@staticmethod
def create(config: KVCacheConfig) -> KVCache | _MLXKVCache
```

Instantiate a single-layer KV cache from `config.method`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `config` | `KVCacheConfig` | Quantization configuration; `config.method` selects the concrete class |

**Returns:** A concrete cache instance. For methods in `STANDALONE_METHODS`, this is VeloxQuant's own `KVCache` (from `core.abstractions`); for every other method, it's an `mlx_lm.models.cache.KVCache` subclass.

**Raises:** `QuantizerConfigError` if `config.method` is unknown, if `config.bit_width_inlier` is a list for a non-`vecinfer` method (list-form per-layer allocation is only consumed by `KVCacheBuilder.for_model()`), or if `config.sliding_window` is set for a non-standalone method.

Note there is **no** `num_heads`, `head_dim` (as a call argument — it's read from `config.head_dim`), or `max_seq_len` parameter — the signature takes only `config`.

If `config.sliding_window` is set (and `config.method` is a standalone method), the returned cache is wrapped in `SlidingWindowKVCache`.

---

## KVCacheBuilder

```python
from veloxquant_mlx.cache.base import KVCacheBuilder
```

Provides two distinct APIs:

1. A **fluent instance builder** (`with_method(...)`, `with_head_dim(...)`, ... `.build()`) for constructing a single validated cache.
2. A **static per-model builder**, `KVCacheBuilder.for_model(model, config)`, which inspects a loaded model and constructs one cache per transformer layer automatically.

There is no `KVCacheBuilder.build(model, config, ...)` static method — `.build()` is an instance method with no arguments, called after configuring the builder via the `with_*` methods; per-model construction is done through `for_model()`, documented below.

### `KVCacheBuilder.for_model`

```python
@staticmethod
def for_model(model, config: KVCacheConfig) -> list
```

Builds one cache per language-model layer, sized per-layer (works for text-only and VLM models, e.g. Qwen2-VL, Qwen3-VL, Mistral). Layers without a `self_attn`/`attn` attribute (MoE gates, hybrid-attention slots such as GatedDeltaNet, etc.) fall back to the model's own native cache slot (or a plain fp16 `mlx_lm` `KVCache` if none is available), so the returned list length always matches `len(model.layers)`.

If `config.bit_width_inlier` is a `list[int]`, element `i` is used for attention layer `i`; the list length must equal the number of attention-bearing layers.

Methods that need cross-layer coordination (`xquant`, `minicache`, `pyramidkv`, `cachegen`, `squeeze`, `xkv`, and `chunkkv` when `chunkkv_reuse_layers > 1`) are routed through internal per-method builders that construct a shared coordinator and assign per-layer roles; every other method builds each layer's cache independently via `KVCacheFactory.create()`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `model` | mlx_lm model | Model loaded with `mlx_lm.load()` |
| `config` | `KVCacheConfig` | Quantization configuration (`head_dim` is overridden per-layer from the model) |

**Returns:** `list` of cache instances, one per language-model layer, passable to `mlx_lm.generate(prompt_cache=...)`.

**Raises:** `QuantizerConfigError` if `config.method` is in [`STANDALONE_METHODS`](#standalone-methods) — standalone methods must be constructed directly via `KVCacheFactory.create()` instead.

There is no `max_seq_len` parameter — caches are token-appended incrementally rather than pre-allocated to a fixed sequence length.

**Example:**

```python
import mlx_lm
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")
config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1)
caches = KVCacheBuilder.for_model(model, config)
# caches is a list of one TurboQuantRVQKVCache per Llama layer
```

### Fluent instance builder

```python
cache = (
    KVCacheBuilder()
    .with_method("turboquant_prod")
    .with_head_dim(128)
    .with_bit_width(inlier=2, outlier=3)
    .with_jl_dim(128)
    .with_seed(42)
    .build()
)
```

Chainable setters: `with_method(method)`, `with_head_dim(d)`, `with_bit_width(inlier, outlier=None)`, `with_jl_dim(m)`, `with_n_outlier_channels(n)`, `with_n_calib_tokens(n)`, `with_vectorized_attend(enabled=True)`, `with_outlier_two_stream(enabled=True)`, `with_fused_query_dot(enabled=True)`, `with_seed(seed)`, `with_precision(dtype)`, `with_capacity(max_tokens)`, `with_artifact_store(store)`, `with_observer(observer)`, `with_sliding_window(window_size)`. Each returns `self`.

`build()` validates the accumulated config (head_dim must be a power of 2, `bit_width_inlier` list entries must all be ints ≥ 1, `jl_dim <= head_dim`, `n_outlier_channels < head_dim`, etc.), then calls `KVCacheFactory.create()` and returns the resulting cache. It rejects list-form `bit_width_inlier` implicitly by delegating to `KVCacheFactory.create()`, which raises for lists on non-`vecinfer` methods.

---

## Cache classes

The following 6 classes are a curated subset for illustration; the registry (`veloxquant_mlx.cache.registry`) currently lists 43 methods in total. See [Registry introspection](#registry-introspection) below to enumerate all of them.

### TurboQuantRVQKVCache

```python
from veloxquant_mlx.cache.turboquant_rvq_cache import TurboQuantRVQKVCache
```

Residual vector quantization cache; the library's default serving method (`method="turboquant_rvq"`). Subclasses `mlx_lm.models.cache.KVCache` (implements `update_and_fetch`), so it is fully compatible with `KVCacheBuilder.for_model()` and `mlx_lm.generate()`.

### VecInferKVCache

```python
from veloxquant_mlx.cache.vecinfer_cache import VecInferKVCache
```

Codebook vector-quantization cache for aggressive compression (`method="vecinfer"`), configured via `key_sub_dim`, `value_sub_dim`, `key_codebook_bits`, `value_codebook_bits`, `residual_length`. Subclasses `mlx_lm.models.cache.KVCache`; compatible with `for_model()`.

### SpectralQuantKVCache

```python
from veloxquant_mlx.cache.spectral_cache import SpectralQuantKVCache
```

Spectral-domain transform cache (`method="spectral"`), configured via `spectral_key_d_eff`, `spectral_val_d_eff`, `spectral_apply_qjl`, `spectral_model_name`. **This is a [standalone method](#standalone-methods)** — it implements VeloxQuant's own `KVCache` ABC, not `mlx_lm`'s protocol, so it must be constructed via `KVCacheFactory.create()`, not `KVCacheBuilder.for_model()`.

### PolarQuantKVCache

```python
from veloxquant_mlx.cache.polar_cache import PolarQuantKVCache
```

Polar-coordinate encoding of key vectors (`method="polar"`). **Standalone method** — construct via `KVCacheFactory.create()` only.

### QJLKVCache

```python
from veloxquant_mlx.cache.qjl_cache import QJLKVCache
```

Johnson-Lindenstrauss sketch cache with 1-bit quantization (`method="qjl"`), configured via `jl_dim` and `seed`. **Standalone method** — construct via `KVCacheFactory.create()` only.

### SlidingWindowKVCache

```python
from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache
```

Token-eviction wrapper for a standalone `KVCache`. Applied automatically by `KVCacheFactory.create()` when `config.sliding_window` is set on a standalone-method config; only compatible with standalone methods (see [Standalone methods](#standalone-methods)) since it wraps VeloxQuant's `append_key`/`append_value`/`attend` interface, not `mlx_lm`'s `update_and_fetch` protocol.

---

## Registry introspection

```python
from veloxquant_mlx.cache.registry import (
    get_method, list_methods, all_method_names, describe_field, field_is_relevant,
)
```

`veloxquant_mlx.cache.registry` derives a live, code-accurate catalog of all 43 methods rather than a hand-maintained table:

- `all_method_names() -> list[str]` — every method name `KVCacheConfig.method` accepts, read directly from its `Literal` type annotation.
- `get_method(name) -> MethodInfo` — family, serve tier (probed against a real cache instance, not declared), blurb, relevant config fields, paper-deviation notes, and telemetry coverage for one method.
- `list_methods(*, servable_only=False, family=None) -> list[MethodInfo]` — all methods, optionally filtered, sorted servable-first then by name.
- `describe_field(name) -> dict` — type, default, and optionality for one `KVCacheConfig` field, derived by reading the dataclass via `dataclasses.fields()` and `typing.get_type_hints()`. Correctly recognizes an `Optional[X]` field written as either legacy `typing.Union[X, None]` or PEP 604 `X | None` syntax — both resolve to a `types.UnionType`/`typing.Union` origin check, so all 20 `Optional`-typed fields in `KVCacheConfig` (regardless of which syntax they use) are reported with `"optional": True`, not just one style.
- `field_is_relevant(method, name) -> bool` — whether a given config field has any effect for a given method.

`veloxquant_mlx` ships [PEP 561](https://peps.python.org/pep-0561/) type marker support (`py.typed`), so these signatures are fully type-checkable by `mypy`/`pyright` in consuming projects.

---

## See also

- [mlx_lm integration guide](../guides/mlx-lm-integration)
- [API — Quantizers](../api/quantizers)
- [API — Core abstractions](../api/core-api)
