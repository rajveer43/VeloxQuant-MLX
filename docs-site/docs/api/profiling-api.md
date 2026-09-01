---
id: profiling-api
title: Profiling API
sidebar_label: Profiling
slug: /api/profiling-api
---

# Profiling API

`veloxquant_mlx.profiling`

Kernel-level timing and memory profiling for KV caches. `KVCacheProfiler` transparently wraps any `KVCache` instance — no changes to the wrapped cache's implementation required.

---

## KVCacheProfiler

```python
from veloxquant_mlx import KVCacheProfiler
# or: from veloxquant_mlx.profiling import KVCacheProfiler
```

Implements the `KVCache` interface (`append_key`, `append_value`, `attend`, `memory_bytes`, `append`, `__len__`) by delegating to a wrapped cache while recording per-call latency and memory.

### Constructor

```python
KVCacheProfiler(cache: KVCache, head_dim: int | None = None, layer_id: Any = 0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cache` | `KVCache` | required | The cache instance to profile |
| `head_dim` | `int \| None` | `None` | Head dimension, used for the fp16 compression baseline (2 bytes/element). Falls back to `cache._d` if present, else `0` (disables `compression_ratio`) |
| `layer_id` | `Any` | `0` | Label attached to the resulting `LayerProfile`, used in multi-layer reports |

### Methods

```python
def append_key(self, k: Any) -> None
def append_value(self, v: Any) -> None
def attend(self, q: Any) -> Any
def memory_bytes(self) -> int
def profile(self) -> LayerProfile
def reset(self) -> None
```

**`append_key(k)`** — Times the wrapped cache's `append_key`, increments `n_quantize_calls`/`tokens_written`, accumulates `quantize_ms_total`, adds `2 * head_dim` to `fp16_baseline_bytes`, and updates `peak_memory_bytes`.

**`append_value(v)`** — Times the wrapped cache's `append_value`, accumulates `write_ms_total`, and updates `peak_memory_bytes`.

**`attend(q)`** — Times the wrapped cache's `attend`, increments `n_dequantize_calls`, accumulates `dequantize_ms_total`. Returns the wrapped call's result unchanged.

**`memory_bytes()`** — Passes through to the wrapped cache.

**`profile()`** — Returns the accumulated `LayerProfile` for this instance.

**`reset()`** — Clears accumulated stats. Does not affect the wrapped cache or its stored data.

Any attribute not defined on `KVCacheProfiler` (e.g. a method-specific `fused_sdpa()`) is forwarded to the wrapped cache via `__getattr__`.

---

## MLXCacheProfiler

```python
from veloxquant_mlx import MLXCacheProfiler
# or: from veloxquant_mlx.profiling import MLXCacheProfiler
```

Wraps an `mlx_lm.models.cache.KVCache`-style cache — the interface servable methods actually implement (`update_and_fetch(keys, values) -> (keys, values)`), driven by real `mlx_lm.generate()` runs. Unlike `KVCacheProfiler`'s standalone `append_key` / `append_value` / `attend` triad, `update_and_fetch` fuses quantize, store, and dequantize into one call, so there is no separate latency to attribute to each step — `MLXCacheProfiler` times the whole call and records it as `quantize_ms_total` with `LayerProfile.is_fused=True`; `dequantize_ms_total` and `write_ms_total` stay `0`.

### Constructor

```python
MLXCacheProfiler(cache: Any, layer_id: Any = 0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `cache` | `mlx_lm.models.cache.KVCache`-style | required | The cache instance to profile (e.g. one entry from `KVCacheBuilder.for_model()`) |
| `layer_id` | `Any` | `0` | Label attached to the resulting `LayerProfile` |

### Methods

```python
def update_and_fetch(self, keys: Any, values: Any) -> Any
def profile(self) -> LayerProfile
```

**`update_and_fetch(keys, values)`** — Times the wrapped cache's `update_and_fetch` (forcing evaluation of the returned arrays via `mx.eval` so the timing reflects real compute, not lazy-graph deferral), increments `n_quantize_calls`/`tokens_written`, accumulates the elapsed time into `quantize_ms_total`, adds `n_tokens * 2 * head_dim` to `fp16_baseline_bytes`, and updates `peak_memory_bytes` from the wrapped cache's `nbytes`. Returns the wrapped call's result unchanged.

**`profile()`** — Returns the accumulated `LayerProfile` for this instance.

Substitute a list of `MLXCacheProfiler` instances directly for the list `KVCacheBuilder.for_model()` returns and pass it as `mlx_lm.generate(..., prompt_cache=profilers)`. Any attribute not defined on `MLXCacheProfiler` (e.g. `make_mask`, `trim`, `is_trimmable`, `meta_state`, `state`) is forwarded to the wrapped cache via `__getattr__`, so it substitutes transparently wherever the real cache is expected.

See also: [`veloxquant profile` CLI](../guides/profiling#the-veloxquant-profile-cli), which wires this up end to end against a real model.

---

## LayerProfile

```python
from veloxquant_mlx import LayerProfile
```

```python
@dataclass
class LayerProfile:
    layer_id: Any
    n_quantize_calls: int = 0
    n_dequantize_calls: int = 0
    quantize_ms_total: float = 0.0
    dequantize_ms_total: float = 0.0
    write_ms_total: float = 0.0
    peak_memory_bytes: int = 0
    tokens_written: int = 0
    fp16_baseline_bytes: int = 0
    is_fused: bool = False
```

| Field / Property | Type | Description |
|---|---|---|
| `layer_id` | `Any` | Label identifying the layer |
| `n_quantize_calls` | `int` | Number of `append_key` (or `update_and_fetch`) calls |
| `n_dequantize_calls` | `int` | Number of `attend` calls. Always `0` when `is_fused` is `True` |
| `quantize_ms_total` | `float` | Cumulative `append_key` wall time, ms. For `MLXCacheProfiler` (`is_fused=True`), this holds the *combined* quantize+dequantize+write time instead — there is no separate number to split it into |
| `dequantize_ms_total` | `float` | Cumulative `attend` wall time, ms. Always `0` when `is_fused` is `True` |
| `write_ms_total` | `float` | Cumulative `append_value` wall time, ms. Always `0` when `is_fused` is `True` |
| `peak_memory_bytes` | `int` | Largest `memory_bytes()` / `nbytes` observed after any call |
| `tokens_written` | `int` | Number of `append_key` / `update_and_fetch` calls (proxy for tokens stored) |
| `fp16_baseline_bytes` | `int` | `tokens_written * 2 * head_dim` — what the same tokens would cost in fp16 |
| `is_fused` | `bool` | `True` when `quantize_ms_total` is a combined measurement (`MLXCacheProfiler`) rather than quantize-only (`KVCacheProfiler`) |
| `quantize_ms_mean` (property) | `float` | `quantize_ms_total / n_quantize_calls`, or `0.0` if no calls |
| `dequantize_ms_mean` (property) | `float` | `dequantize_ms_total / n_dequantize_calls`, or `0.0` if no calls |
| `compression_ratio` (property) | `float` | `fp16_baseline_bytes / peak_memory_bytes`, or `0.0` if `peak_memory_bytes <= 0` |

---

## ProfileReport

```python
from veloxquant_mlx import ProfileReport
```

```python
@dataclass
class ProfileReport:
    layers: list[LayerProfile] = field(default_factory=list)
    elapsed_s: float = 0.0
```

| Field / Property | Type | Description |
|---|---|---|
| `layers` | `list[LayerProfile]` | Per-layer profiles, in layer order |
| `elapsed_s` | `float` | Total wall time covered by the profiling session (caller-supplied) |
| `total_bytes_written` (property) | `int` | Sum of `peak_memory_bytes` across layers |
| `total_tokens` (property) | `int` | Sum of `tokens_written` across layers |
| `tokens_per_sec` (property) | `float` | `total_tokens / elapsed_s`, or `0.0` if `elapsed_s <= 0` |
| `overall_compression_ratio` (property) | `float` | Sum of `fp16_baseline_bytes` across layers divided by `total_bytes_written`, or `0.0` if that's `<= 0` |

---

## profile_layers

```python
from veloxquant_mlx import profile_layers
```

```python
def profile_layers(
    profilers: list[KVCacheProfiler | MLXCacheProfiler], elapsed_s: float = 0.0
) -> ProfileReport
```

Aggregates a list of `KVCacheProfiler` or `MLXCacheProfiler` instances (one per model layer) into a single `ProfileReport`. Both wrapper types report into the same `LayerProfile` shape, so a list can be aggregated uniformly regardless of which interface it wraps.

---

## format_profile_table

```python
from veloxquant_mlx import format_profile_table
```

```python
def format_profile_table(report: ProfileReport) -> str
```

Renders a `ProfileReport` as a fixed-width table:

```
Layer       Quantize    Dequantize   Memory
-----------------------------------------------
Layer 0     12.3 µs     8.1 µs       1.20 MB
Layer 1     11.8 µs     7.9 µs       1.20 MB
-----------------------------------------------
Total tokens:        4096
Total memory:        38.40 MB
Compression ratio:   6.83x
Tokens/sec:          812.4
```

The totals block (tokens, memory, compression ratio, tokens/sec) is only appended when `report.layers` is non-empty; `tokens_per_sec` is only shown when `report.elapsed_s > 0`.

---

## Example — profiling a standalone cache end to end

```python
from veloxquant_mlx import KVCacheBuilder, KVCacheProfiler
from veloxquant_mlx.profiling import profile_layers, format_profile_table

cache = (
    KVCacheBuilder()
    .with_method("turboquant_prod")
    .with_head_dim(64)
    .with_bit_width(inlier=2)
    .with_jl_dim(64)
    .build()
)
profiled = KVCacheProfiler(cache, head_dim=64, layer_id=0)

for k, v in zip(keys, values):
    profiled.append(k, v)
profiled.attend(query)

report = profile_layers([profiled], elapsed_s=0.05)
print(format_profile_table(report))
```

---

## See also

- [Profiling guide](../guides/profiling)
- [Observers API](./observers-api)
- [Cache API](./cache)
