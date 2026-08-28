---
id: profiling
title: KV-Cache Profiling
sidebar_label: Profiling
slug: /guides/profiling
---

# KV-Cache Profiling

`veloxquant_mlx.profiling.KVCacheProfiler` gives you kernel-level visibility into where time and memory go inside a KV cache — quantization latency, dequantization latency, write latency, peak memory, and derived compression ratio — without editing the cache implementation itself.

## Why a wrapper, not built-in instrumentation

VeloxQuant-MLX ships 40+ KV-cache methods (`turboquant_rvq`, `kivi`, `h2o`, `palu`, …). Rather than adding timing code to every one of them, `KVCacheProfiler` **wraps** any existing `KVCache` instance and intercepts `append_key`, `append_value`, and `attend` — the three calls every method implements. This means:

- Zero changes to cache implementations.
- Works with any method returned by `KVCacheFactory.create()` or `KVCacheBuilder.build()`.
- Drops into `KVCacheBuilder.for_model()` output by wrapping each per-layer cache.
- Unrecognized attributes (e.g. a method-specific `fused_sdpa()`) forward straight through to the wrapped cache.

## Basic usage

```python
from veloxquant_mlx import KVCacheBuilder, KVCacheProfiler

cache = (
    KVCacheBuilder()
    .with_method("turboquant_rvq")
    .with_head_dim(128)
    .with_bit_width(inlier=2)
    .build()
)

profiled = KVCacheProfiler(cache, head_dim=128, layer_id=0)

for k, v in zip(keys, values):
    profiled.append(k, v)  # append_key + append_value, timed

out = profiled.attend(query)  # timed

report = profiled.profile()
print(f"Quantize latency (mean): {report.quantize_ms_mean * 1000:.1f} µs")
print(f"Dequantize latency (mean): {report.dequantize_ms_mean * 1000:.1f} µs")
print(f"Peak memory: {report.peak_memory_bytes} bytes")
print(f"Compression ratio: {report.compression_ratio:.2f}x")
```

`KVCacheProfiler` implements the same `KVCache` interface as the object it wraps (`append_key`, `append_value`, `attend`, `memory_bytes`, `append`, `__len__`), so it's a drop-in substitute anywhere a `KVCache` is expected.

## Profiling every layer of a model

`KVCacheBuilder.for_model()` returns one cache per layer. Wrap each one to get a per-layer breakdown:

```python
from veloxquant_mlx import KVCacheBuilder, KVCacheConfig, KVCacheProfiler
from veloxquant_mlx.profiling import profile_layers, format_profile_table

config = KVCacheConfig(method="kivi", bit_width_inlier=2)
caches = KVCacheBuilder.for_model(model, config)
profilers = [
    KVCacheProfiler(c, head_dim=config.head_dim, layer_id=i)
    for i, c in enumerate(caches)
]

# ... run generation, feeding keys/values/queries through `profilers` instead
# of `caches` directly ...

report = profile_layers(profilers, elapsed_s=total_wall_time)
print(format_profile_table(report))
```

Example output:

```
Layer       Quantize    Dequantize   Memory
-----------------------------------------------
Layer 0     12.3 µs     8.1 µs       1.20 MB
Layer 1     11.8 µs     7.9 µs       1.20 MB
...
-----------------------------------------------
Total tokens:        4096
Total memory:        38.40 MB
Compression ratio:   6.83x
Tokens/sec:          812.4
```

## What gets measured

| Metric | Source |
|---|---|
| Quantization latency | Wall time inside each `append_key()` call |
| Dequantization latency | Wall time inside each `attend()` call |
| Write latency | Wall time inside each `append_value()` call |
| Peak KV memory | Largest `memory_bytes()` seen after any call |
| Compression ratio | `(tokens_written × 2 × head_dim) / peak_memory_bytes` — actual bytes vs. an fp16 baseline |
| Tokens/sec | `total_tokens / elapsed_s`, computed at the `profile_layers()` aggregation step (you supply `elapsed_s`) |

:::note[Cache-allocation latency and byte-level read/write counts]
The current `KVCache` interface doesn't expose a separate allocation step or granular byte-level read/write counters, so those two items from the original proposal aren't broken out individually — `memory_bytes()` peak captures the net effect of allocation instead. If your cache subclass tracks these separately, read `profiler.profile()` fields directly and extend as needed.
:::

## Resetting between runs

```python
profiled.reset()  # clears accumulated stats; does not touch the wrapped cache's data
```

## See also

- [Profiling API](../api/profiling-api)
- [Observers guide](./observers) — event-driven metrics (distortion, per-stage latency/memory) for use inside your own pipeline
- [Benchmarking guide](./benchmarking)
