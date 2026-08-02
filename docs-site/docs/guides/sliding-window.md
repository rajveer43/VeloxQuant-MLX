---
id: sliding-window
title: Sliding Window Cache
sidebar_label: Sliding Window
slug: /guides/sliding-window
---

# Sliding Window Cache

`SlidingWindowKVCache` wraps any VeloxQuant-MLX cache with FIFO token eviction. When the sequence length exceeds the window size, the oldest token is evicted, keeping memory bounded regardless of generation length.

:::note[FIFO only]
`SlidingWindowKVCache` implements one eviction policy — oldest token evicted first. There is no configurable eviction strategy (no attention-based or fixed-prefix mode).
:::

## Why use a sliding window?

Standard KV caches grow linearly with sequence length. Even with compression, a very long conversation can exhaust memory on a memory-constrained Mac. The sliding window bounds the cache size at the cost of losing access to tokens outside the window.

## How it works

`SlidingWindowKVCache` keeps the raw (pre-quantization) key/value vectors for the current window in a ring buffer. Because the wrapped cache doesn't support random deletion, evicting the oldest token means **rebuilding the inner cache from scratch** from the remaining window contents — this happens once per eviction, not once per token, but it does mean the wrapped cache is re-populated every `window_size` tokens. This makes it suitable for inference, not training.

## Basic usage

`SlidingWindowKVCache` wraps the repo's lower-level, per-token `KVCache` protocol (`append_key`/`append_value`/`attend`/`memory_bytes`) — the caches that implement it directly are `turboquant_prod`/`turboquant_mse` (`TurboQuantKVCache`), `polar`, `qjl`, and `spectral`. The `_MLXKVCache`-based caches (e.g. `turboquant_rvq`, `vecinfer`) built for `mlx_lm`'s `update_and_fetch` protocol don't implement `memory_bytes()` and aren't a fit for this wrapper today.

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache

# Create the inner compressed cache
config = KVCacheConfig(method="turboquant_prod", bit_width_inlier=2, head_dim=128)
inner_cache = KVCacheFactory.create(config)

# Wrap it with a sliding window of 2048 tokens
cache = SlidingWindowKVCache(inner_cache, window_size=2048)

# Token-by-token insertion (the interface this cache implements)
cache.append_key(key_vector)  # shape (d,), fp16
cache.append_value(value_vector)  # shape (d,), fp16 — triggers eviction if window is full

output = cache.attend(query_vector)  # delegates to the inner cache
```

:::note
`SlidingWindowKVCache` implements the repo's own `KVCache` protocol (`append_key`/`append_value`/`attend`), which is per-token — it is not currently plumbed through `KVCacheBuilder.build()`/`for_model()` for direct use as an `mlx_lm` `kv_cache=` argument. Check `veloxquant_mlx/core/abstractions.py` for the full `KVCache` interface before wiring it into a generation loop.
:::

## Inspecting state

`SlidingWindowKVCache` exposes only:

| Method | Returns |
|---|---|
| `len(cache)` | Current number of stored tokens, capped at `window_size` |
| `cache.memory_bytes()` | Memory usage of the inner (windowed) cache |
| `repr(cache)` | `SlidingWindowKVCache(window=..., n_stored=..., total_seen=...)` |

There is no `eviction_stats()` method or eviction-count tracking beyond what `repr()` shows via `total_seen`.

```python
print(len(cache))  # tokens currently held (<= window_size)
print(cache.memory_bytes())  # bytes used by the inner cache
print(cache)  # SlidingWindowKVCache(window=2048, n_stored=2048, total_seen=5000)
```

## Configuration reference

| Parameter | Type | Description |
|---|---|---|
| `inner` | `KVCache` | The underlying cache to wrap (positional, first argument) |
| `window_size` | `int` | Maximum tokens to keep; must be `>= 1` |

## See also

- [mlx_lm integration](../guides/mlx-lm-integration)
- [Observers — memory tracking](../guides/observers)
