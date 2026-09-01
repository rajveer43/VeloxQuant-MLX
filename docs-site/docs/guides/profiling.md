---
id: profiling
title: KV-Cache Profiling
sidebar_label: Profiling
slug: /guides/profiling
---

# KV-Cache Profiling

If your compressed KV cache is slower than you expected, or eating more memory than it should, `KVCacheProfiler` tells you exactly where the time and bytes are going — per layer, without changing any code in the cache itself.

Wrap your cache, run it as normal, and print a report:

```python
from veloxquant_mlx import KVCacheBuilder, KVCacheProfiler

cache = KVCacheBuilder().with_method("kivi").with_head_dim(128).build()
profiled = KVCacheProfiler(cache, head_dim=128)

for k, v in zip(keys, values):
    profiled.append(k, v)
profiled.attend(query)

report = profiled.profile()
print(f"Quantize:    {report.quantize_ms_mean * 1000:.1f} µs/call")
print(f"Dequantize:  {report.dequantize_ms_mean * 1000:.1f} µs/call")
print(f"Peak memory: {report.peak_memory_bytes / 1024:.1f} KB")
print(f"Compression: {report.compression_ratio:.2f}x vs fp16")
```

That's the whole workflow: construct your cache as usual, wrap it in `KVCacheProfiler`, use it exactly like a normal cache (`.append(k, v)`, `.attend(q)`), then read `.profile()` for the numbers.

## Profiling a full model, layer by layer

Most of the time you care about a per-layer breakdown, not just one cache. `KVCacheBuilder.for_model()` gives you one cache per layer — wrap each one the same way, then let `profile_layers()` and `format_profile_table()` do the aggregation and printing for you:

```python
from veloxquant_mlx import KVCacheBuilder, KVCacheConfig, KVCacheProfiler
from veloxquant_mlx.profiling import profile_layers, format_profile_table

config = KVCacheConfig(method="kivi", bit_width_inlier=2)
caches = KVCacheBuilder.for_model(model, config)
profilers = [KVCacheProfiler(c, head_dim=config.head_dim, layer_id=i) for i, c in enumerate(caches)]

# Run your normal generation loop, using `profilers[i]` wherever you'd
# normally use `caches[i]` — everything else about your code stays the same.
...

report = profile_layers(profilers, elapsed_s=total_wall_time)
print(format_profile_table(report))
```

That prints a table you can read at a glance:

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

If one layer's quantize time or memory jumps out from the rest, that's your bottleneck — no guessing, no separate benchmarking harness.

## What each number means

| Metric | What it tells you |
|---|---|
| **Quantize** (`quantize_ms_mean`) | Average time to compress one key vector into the cache. High here means your bit-width/method choice is CPU/GPU-bound on writes. |
| **Dequantize** (`dequantize_ms_mean`) | Average time to reconstruct/attend over the cache for one query. High here shows up directly as slower generation. |
| **Memory** (`peak_memory_bytes`) | The largest footprint the cache ever reached — the number that actually matters for "will this fit." |
| **Compression ratio** | How much smaller the cache is than storing the same tokens in fp16. Lower than expected? Check your bit-width config. |
| **Tokens/sec** | Overall throughput across all profiled layers, for a given wall-clock window you supply. |

## Resetting between runs

Comparing two configs in the same session? Reset the stats without losing the cache's actual data:

```python
profiled.reset()  # clears accumulated timing/memory stats; the cache itself is untouched
```

## Good to know

- **No code changes required anywhere else.** `KVCacheProfiler` wraps any of the 40+ cache methods (`turboquant_rvq`, `kivi`, `h2o`, `palu`, …) the same way — it works because every method implements the same `append_key` / `append_value` / `attend` calls under the hood.
- **It's a drop-in replacement.** `KVCacheProfiler` behaves like a normal cache (same `append`, `attend`, `memory_bytes`, `__len__`), so you can pass it anywhere a cache is expected. Anything it doesn't handle itself (like a method-specific `fused_sdpa()`) is passed straight through to the real cache.
- **Two metrics from the original proposal aren't broken out separately**: a dedicated "cache allocation latency" and byte-level read/write counters. The current cache interface doesn't expose those as distinct steps, so `peak_memory_bytes` captures their net effect instead. If you need finer detail, `profiler.profile()` returns a plain dataclass you can extend.

## Profiling a real `mlx_lm.generate()` run — `MLXCacheProfiler`

`KVCacheProfiler` above wraps the standalone `append_key` / `append_value` / `attend` interface used by `benchmark.py` and research code. Servable methods — the ones `veloxquant serve` and `veloxquant profile` drive through a real `mlx_lm.generate()` call — implement a different, fused interface instead: `update_and_fetch(keys, values)`, which quantizes, stores, and dequantizes in one call. There's no separate quantize/dequantize/write split to hook into there, so a second wrapper, `MLXCacheProfiler`, times `update_and_fetch` as a single unit and reports it as `compute_ms_total` (`LayerProfile.is_fused=True`) rather than fabricating three numbers that don't exist.

```python
from veloxquant_mlx import KVCacheBuilder, KVCacheConfig, MLXCacheProfiler
from veloxquant_mlx.profiling import profile_layers, format_profile_table
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
config = KVCacheConfig(method="kivi", bit_width_inlier=2)
caches = KVCacheBuilder.for_model(model, config)
profilers = [MLXCacheProfiler(c, layer_id=i) for i, c in enumerate(caches)]

generate(model, tokenizer, prompt="Explain unified memory.", max_tokens=64, prompt_cache=profilers)

report = profile_layers(profilers, elapsed_s=...)
print(format_profile_table(report))
```

`profilers` substitutes directly for the cache list in `prompt_cache=` — nothing else about the generation call changes.

### The `veloxquant profile` CLI

For a quick one-off check without writing a script, `veloxquant profile` wires this up end to end and prints a JSON summary:

```bash
veloxquant profile \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --method kivi --bits 2 \
  --prompt "Explain unified memory on Apple Silicon." \
  --max-tokens 64
```

It loads the model, builds one `MLXCacheProfiler`-wrapped cache per layer via `KVCacheBuilder.for_model`, runs a real generation pass, and emits (to stdout) a stable, `schema_version`-tagged JSON payload — the same control-panel contract `veloxquant methods --json` and `veloxquant serve` use — with per-layer `compute_latency_ms` / `peak_memory_bytes` / `compression_ratio` plus a `summary` block. Only methods that implement the `mlx_lm` serving contract can be profiled this way (`veloxquant methods --servable-only` lists valid choices); passing a non-servable method fails fast, before the model loads.

Use `--set FIELD=VALUE` (repeatable) to override any method-specific `KVCacheConfig` field, e.g. `--set kivi_group_size=64`.

As with every other accounting number in this library, the JSON payload's `accounting_note` field is a standing reminder that these caches store dequantized fp16 tensors — the reported bytes measure compression fidelity, not runtime RSS saved.

## See also

- [Profiling API reference](../api/profiling-api) — full field/method list
- [Observers guide](./observers) — for custom event-driven metrics inside your own pipeline (distortion, arbitrary stage timing)
- [Benchmarking guide](./benchmarking)
