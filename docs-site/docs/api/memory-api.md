---
id: memory-api
title: Memory (Block Pool) API
sidebar_label: Memory / Block Pool
slug: /api/memory-api
---

# Memory (Block Pool) API

`veloxquant_mlx.memory`

Fixed-size block allocation and reuse for KV-cache storage — the
memory-allocation analogue of the bit-level compression the rest of
VeloxQuant-MLX provides. Pre-allocates a fixed pool of blocks up front so
generation never pays for per-token `malloc`/`free`, tracks reuse and
fragmentation across requests, and lets a single pool host blocks written
in different compression formats side by side. The allocator is
thread-safe, can grow on demand under `PoolConfig.grow_on_exhaustion`
instead of hard-failing, guards against accidental owner-id collisions,
can shrink back down via `shrink()`, and exposes a bounded history of
stats snapshots plus a Prometheus exporter for observability.

---

## Overview

| Class | Purpose |
|---|---|
| [`PoolConfig`](#poolconfig) | Sizing/behavior config for a pool (block size, block count, K/V separation) |
| [`BlockPoolAllocator`](#blockpoolallocator) | Pure-Python block bookkeeping — allocate, free, reuse, per-owner tracking, stats |
| [`Block`](#block) | A single fixed-size block handed out by the allocator |
| [`AllocationStats`](#allocationstats) | Running allocation/reuse/fragmentation counters |
| [`MLXBlockStorage`](#mlxblockstorage) | Pairs with a pool to provide the actual `mx.array` buffers, one per block id |
| [`PooledKVCache`](#pooledkvcache) | Wraps any existing VeloxQuant `KVCache` so its appends check out/return blocks from a shared pool |
| [`PoolBackedKVCache`](#poolbackedkvcache) | Drop-in `mlx_lm.models.cache.KVCache` replacement — puts the pool directly in `mlx_lm.generate()`'s decode hot path |
| [`build_pooled_caches`](#poolbackedkvcache) | Builds one `PoolBackedKVCache` per model layer, sharing one pool/owner |

`BlockPoolAllocator`, `PoolConfig`, `Block`, and `AllocationStats` have no
MLX dependency — they are plain Python and can be used, tested, or reasoned
about independently of `mx.array`. `MLXBlockStorage`, `PooledKVCache`, and
`PoolBackedKVCache` are the MLX-backed pieces that plug the pool into
actual cache storage.

`PooledKVCache` and `PoolBackedKVCache` serve different interfaces and are
not interchangeable: `PooledKVCache` wraps VeloxQuant's own `KVCache` ABC
(`append_key`/`append_value`/`attend`), implemented only by the 5
"standalone" methods (`turboquant_prod`, `turboquant_mse`, `polar`, `qjl`,
`spectral` — see `STANDALONE_METHODS` in `veloxquant_mlx/cache/base.py`),
which `mlx_lm.generate()` never drives. `PoolBackedKVCache` implements
`mlx_lm`'s own `update_and_fetch` protocol directly, so it's the one to use
when you want the pool actually driving a real model's live generation.

---

## PoolConfig

```python
from veloxquant_mlx.memory import PoolConfig


@dataclass
class PoolConfig:
    block_size: int = 16
    n_blocks: int = 1024
    separate_kv: bool = True
    grow_on_exhaustion: bool = False
    max_blocks: Optional[int] = None
```

**Fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `block_size` | `int` | `16` | Number of token slots stored per block |
| `n_blocks` | `int` | `1024` | Total number of blocks the pool manages initially |
| `separate_kv` | `bool` | `True` | If `True`, keys and values are allocated from independent free lists so K-heavy and V-heavy workloads never compete for the same blocks. If `False`, a single free list backs both, which packs tighter when K/V pressure is uneven. |
| `grow_on_exhaustion` | `bool` | `False` | If `True`, an `allocate()` call that would otherwise raise `BlockPoolExhaustedError` instead grows the exhausted stream's block count just enough to satisfy the request (capped by `max_blocks`). |
| `max_blocks` | `Optional[int]` | `None` | Upper bound on total blocks once grown. `None` means unbounded growth. Ignored when `grow_on_exhaustion` is `False`. Growth that would exceed this cap still raises `BlockPoolExhaustedError`. |

Raises `ValueError` if `block_size < 1`, `n_blocks < 1`, or `max_blocks < n_blocks`.

---

## BlockPoolAllocator

```python
from veloxquant_mlx.memory import BlockPoolAllocator, PoolConfig

pool = BlockPoolAllocator(PoolConfig(block_size=16, n_blocks=256))
```

Pre-allocates `n_blocks` fixed-size blocks and hands them out on request.
Freed blocks return to a free list (LIFO — a just-freed block tends to be
cache-hot) and are reused by later allocations, so steady-state generation
performs zero new-memory allocation once the pool is warm.

**Thread safety:** every mutating and read method acquires an internal
lock, so one pool can be shared across threads or concurrent request
handlers without external synchronization. The lock is a plain
(non-reentrant) `threading.Lock` — don't call back into the pool from
inside a callback invoked while holding it.

### `allocate`

```python
def allocate(
    self,
    stream: str,
    n_tokens: int,
    owner: int,
    format: str = "fp16",
) -> list[Block]
```

Allocate enough blocks to hold `n_tokens` slots (`ceil(n_tokens / block_size)`
blocks). All-or-nothing: if fewer blocks are free than required, no blocks
are handed out.

| Parameter | Type | Description |
|---|---|---|
| `stream` | `str` | `"k"` or `"v"` (ignored, treated as `"kv"`, when `separate_kv=False`) |
| `n_tokens` | `int` | Number of token slots needed |
| `owner` | `int` | Opaque id (e.g. request id) the blocks are checked out to |
| `format` | `str` | Compression-format tag recorded on each block (`"fp16"`, `"int8"`, `"int4"`, `"int2"`, `"int1"`, ...) |

**Raises:** `BlockPoolExhaustedError` if not enough free blocks remain and
the pool can't grow enough to cover the gap (see `PoolConfig.grow_on_exhaustion`
/ `max_blocks`); `ValueError` if `n_tokens <= 0` or `stream` is invalid for
the pool's `separate_kv` setting.

Repeated `allocate()` calls for an owner that's already active on the pool
are treated as that same request continuing to grow (the normal case — see
[`PoolBackedKVCache`](#poolbackedkvcache)) and never raise. See
[`register_owner`](#register_owner--release_owner) for real collision
detection between two different callers.

### `free` / `free_all`

```python
def free(self, blocks: list[Block]) -> None
def free_all(self, owner: int) -> None
```

Return blocks to the free list for their stream. `free_all` returns every
block currently checked out to `owner` in one call — the pattern to use
when a request finishes. `free` is idempotent (freeing an already-free
block is a no-op).

### `blocks_for` / `n_free`

```python
def blocks_for(self, owner: int) -> list[Block]
def n_free(self, stream: str = "k") -> int
```

`blocks_for` returns the blocks currently checked out to `owner`, in
allocation order. `n_free` returns the number of free blocks remaining for
a stream.

### `register_owner` / `release_owner`

```python
def register_owner(self, owner: int) -> None
def release_owner(self, owner: int) -> None
```

`allocate()` calls `register_owner()` automatically for any owner id not
yet seen, but it can't tell "the same request growing" apart from "two
different callers colliding on the same id" — a second `allocate()` for an
already-active owner is always treated as the former. Call
`register_owner()` yourself, once, before the first `allocate()` for an
owner, if you want the latter case to raise immediately:
`OwnerAlreadyActiveError` is raised on a second `register_owner()` call
for an id that's still active (has checked-out blocks, or was registered
and not yet released). `release_owner()` clears an owner from the active
set without freeing any blocks — `free_all()` calls it automatically once
an owner's last block is freed.

### `shrink`

```python
def shrink(self, stream: str, target_free: int) -> int
```

Permanently retires fully-free blocks from `stream` down to `target_free`
free blocks remaining, and returns the number actually retired. Only
blocks currently on the free list are eligible — in-use blocks are never
touched, so this never breaks a live request. Use it to give memory back
when a pool was sized for peak load that turned out much higher than
steady-state usage. Retired blocks are removed from the pool's bookkeeping
entirely, so any external backing storage indexed by block id (e.g.
`MLXBlockStorage`) may drop the corresponding buffer. Raises `ValueError`
if `target_free < 0`.

### `stats` / `history`

A `BlockPoolAllocator` exposes its running counters as `pool.stats`, an
[`AllocationStats`](#allocationstats) instance updated on every
`allocate`/`free`/`shrink` call. `pool.history` is a bounded `deque`
(most-recent 256 entries) of independent [`AllocationStats.snapshot()`](#allocationstats)
copies, one appended per mutating call — use it to see recent
fragmentation/exhaustion trends rather than only the current instant. See
[`AllocationStats.to_prometheus`](#allocationstats) for exporting either
the live `stats` or any `history` entry.

---

## Block

```python
@dataclass
class Block:
    block_id: int
    stream: str = "kv"
    format: str = "fp16"
    n_used: int = 0
    owner: Optional[int] = None
    ever_allocated: bool = False
```

A single fixed-size block of KV-cache storage. `block_id` is stable for the
life of the pool — blocks are reused in place, never moved. `n_used` is the
number of token slots currently occupied (`<= block_size`); `owner` is
`None` when the block is free.

---

## AllocationStats

```python
@dataclass
class AllocationStats:
    n_blocks: int
    n_allocations: int = 0
    n_frees: int = 0
    n_reused: int = 0
    n_exhausted: int = 0
    n_grown: int = 0
    n_retired: int = 0
    peak_blocks_in_use: int = 0
```

| Field | Description |
|---|---|
| `n_blocks` | Total blocks currently managed by the pool |
| `n_allocations` | Cumulative successful `allocate()` calls |
| `n_frees` | Cumulative `free()` calls |
| `n_reused` | Allocations satisfied by a block that had previously been freed and returned to the pool (vs. a block never handed out before) |
| `n_exhausted` | Allocation attempts that failed because no free block was available (and the pool couldn't grow enough to cover the gap) |
| `n_grown` | Blocks added to the pool by exhaustion-triggered growth (see `PoolConfig.grow_on_exhaustion`) |
| `n_retired` | Blocks permanently removed from the pool by [`shrink()`](#shrink) |
| `peak_blocks_in_use` | High-water mark of concurrently allocated blocks |

**Methods:**

```python
def blocks_in_use(self) -> int                          # n_allocations - n_frees
def fragmentation(self) -> float                        # fraction of managed blocks that are allocated but not full
def snapshot(self) -> AllocationStats                    # independent copy, detached from further mutation
def to_prometheus(self, prefix: str = "veloxquant_block_pool") -> str  # Prometheus text-exposition format
```

`to_prometheus()` renders gauges (`blocks_in_use`, `blocks_total`,
`fragmentation`, `peak_blocks_in_use`) and counters (`allocations_total`,
`frees_total`, `reused_total`, `exhausted_total`, `grown_total`,
`retired_total`) as newline-terminated Prometheus text, suitable for
serving directly from a `/metrics` endpoint:

```python
print(pool.stats.to_prometheus())
# # TYPE veloxquant_block_pool_blocks_in_use gauge
# veloxquant_block_pool_blocks_in_use 3
# ...
```

```python
pool = BlockPoolAllocator(PoolConfig(block_size=16, n_blocks=64))
pool.allocate(stream="k", n_tokens=40, owner=1)
print(pool.stats)
# AllocationStats(in_use=3/64, allocations=3, frees=0, reused=0,
#                  exhausted=0, grown=0, retired=0, peak=3, fragmentation=0.016)
```

---

## MLXBlockStorage

```python
from veloxquant_mlx.memory import MLXBlockStorage

storage = MLXBlockStorage(pool, head_dim=128)
```

Allocates one `mx.array` buffer per block id, shaped `(block_size, head_dim)`,
lazily on first write. A block recycled by the pool reuses its existing
buffer in place — writing into a reused block overwrites its previous
contents without a new `mx.array` allocation. A block's buffer is only
reallocated when the compression format assigned to that block id actually
changes.

### `write` / `read`

```python
def write(self, block: Block, offset: int, values) -> None
def read(self, block: Block)
```

`write` writes `values` (`mx.array` of shape `(n, head_dim)`) into `block`
starting at token-slot `offset`; raises `ValueError` on overflow past
`block_size`. `read` returns the live (used) slice of `block`'s buffer,
shape `(block.n_used, head_dim)`.

### `resident_bytes` / `release`

```python
def resident_bytes(self) -> int
def release(self, block: Block) -> None
```

`resident_bytes` returns the total bytes currently materialized across all
allocated buffers — only blocks that have had a buffer allocated (lazily,
on first write/read), so this reflects actual resident footprint, not the
pool's full pre-allocated capacity. `release` drops the buffer for a freed
block so MLX can reclaim its memory immediately, rather than leaving it
around for silent reuse the next time that block id is handed out.

---

## PooledKVCache

```python
from veloxquant_mlx.memory import PooledKVCache

cache = PooledKVCache(inner, pool, owner=request_id, format="int2")
```

Wraps any VeloxQuant `KVCache` with block-pool-backed memory accounting.
The wrapped cache still owns and drives its own storage arrays — its
compression format can be anything registered with `KVCacheFactory`; this
wrapper checks out and returns fixed-size blocks from a shared
`BlockPoolAllocator` in lock-step with token appends, so a multi-request
server can track allocation counts, reuse, and fragmentation across every
active cache from one pool.

Block granularity is `pool.config.block_size` tokens — a new block is
requested only when the previous one fills up, and every block this cache
holds is returned to the pool in one call (`release()`) when the request
completes. No block is ever allocated or freed on a per-token basis.

| Constructor argument | Type | Description |
|---|---|---|
| `inner` | `KVCache` | The underlying cache that performs actual compression |
| `pool` | `BlockPoolAllocator` | Shared pool to draw block accounting from |
| `owner` | `int` | Opaque id identifying this cache's request/sequence — must be unique per concurrent cache |
| `format` | `str` | Compression-format tag recorded on each block this cache checks out |

**Methods:** implements the standard `KVCache` interface
(`append_key`, `append_value`, `append`, `attend`, `memory_bytes`, `__len__`),
plus:

```python
def release(self) -> None          # return every block held back to the pool
def n_blocks_held(self) -> int     # total K + V blocks currently checked out
```

---

## PoolBackedKVCache

```python
from veloxquant_mlx.memory import PoolBackedKVCache, build_pooled_caches

cache = PoolBackedKVCache(pool, owner=request_id)
```

Drop-in replacement for `mlx_lm.models.cache.KVCache`, the class
`mlx_lm.generate()` uses by default for every layer's cache. `mlx_lm`'s
stock `KVCache` grows its backing `mx.array` in fixed `step=256`-token
chunks, hardcoded per instance, with no visibility into how many times
that growth happened or how much of the last chunk went unused.
`PoolBackedKVCache` keeps the same contiguous-buffer growth strategy —
attention needs one contiguous `(B, n_kv_heads, seq_len, head_dim)` tensor
every step, and RoPE/masking read `cache.offset` directly, so growth can't
be replaced with true block-paged storage without paying a gather cost
every decode step — but routes every growth step through
`pool.allocate()`, so it shows up in the pool's `AllocationStats`, and the
growth chunk size becomes the pool's configured `block_size` instead of a
hardcoded constant.

This is the class that actually puts `BlockPoolAllocator` in a real
model's decode hot path — unlike `PooledKVCache` (see above), which only
targets the 5 standalone methods, `PoolBackedKVCache` implements
`mlx_lm`'s `update_and_fetch` protocol directly, so it works as a
`prompt_cache` for `mlx_lm.generate()` on any model.

| Constructor argument | Type | Description |
|---|---|---|
| `pool` | `BlockPoolAllocator` | Shared pool to draw growth accounting from |
| `owner` | `int` | Opaque id identifying this cache's request/sequence — must be unique per concurrent cache |
| `step` | `Optional[int]` | Token-chunk growth size; defaults to `pool.config.block_size` |

**Methods:** implements the full `mlx_lm.models.cache.KVCache` contract
(`update_and_fetch`, `state`, `meta_state`, `is_trimmable`, `trim`,
`make_mask`, `empty`, `nbytes`, `size`), verified bit-for-bit identical
output against the stock class on the same input sequence, plus:

```python
def release(self) -> None   # return every block this cache's growth checked out
```

### `build_pooled_caches`

```python
def build_pooled_caches(
    model, pool: BlockPoolAllocator, owner: int, step: Optional[int] = None
) -> list[PoolBackedKVCache]
```

Builds one `PoolBackedKVCache` per language-model layer, all sharing the
same `pool` and `owner` — the drop-in replacement for
`mlx_lm.models.cache.make_prompt_cache()` / `model.make_cache()` when you
want pool-tracked growth. Because every layer shares one `owner`, calling
`.release()` on any single layer's cache frees every layer's blocks for
this request at once.

---

## Usage

```python
from veloxquant_mlx import KVCacheBuilder, KVCacheConfig
from veloxquant_mlx.memory import BlockPoolAllocator, PoolConfig, PooledKVCache

pool = BlockPoolAllocator(PoolConfig(block_size=16, n_blocks=512))

inner = KVCacheBuilder().with_method("kivi").with_head_dim(128).build()
cache = PooledKVCache(inner, pool, owner=request_id, format="int2")

for token in tokens:
    cache.append(key, value)

# ... generation for this request finishes ...
cache.release()  # blocks become available for reuse by the next request
```

Serving multiple concurrent requests from one pool: give each request a
distinct `owner`, and call `release()` when each finishes so its blocks
return to the shared free list for the next request.

```python
print(pool.stats)
# AllocationStats(in_use=..., allocations=..., frees=..., reused=...,
#                  exhausted=0, peak=..., fragmentation=...)
```

A synthetic benchmark comparing this against naive per-token allocation
(allocation count, peak memory, fragmentation, throughput) lives at
[`benchmark_scripts/benchmark_block_pool.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/benchmark_scripts/benchmark_block_pool.py),
with results committed at `figures/block_pool/results.json`.

### Driving real `mlx_lm.generate()` traffic through the pool

```python
import mlx_lm
from veloxquant_mlx.memory import BlockPoolAllocator, PoolConfig, build_pooled_caches

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")
pool = BlockPoolAllocator(PoolConfig(block_size=16, n_blocks=8192))

cache = build_pooled_caches(model, pool, owner=request_id)
text = mlx_lm.generate(model, tokenizer, prompt, prompt_cache=cache)

# ... request finishes ...
cache[0].release()  # every layer shares one owner, so any one releases all
```

A second, sequential request sharing the same `pool` (a fresh
`build_pooled_caches(model, pool, owner=another_id)`) draws its growth
from the blocks the first request just released — `pool.stats.n_reused`
reflects that reuse happening on a real model's real generation, not a
synthetic replay. An end-to-end benchmark comparing `PoolBackedKVCache`
against a stock `KVCache` on the same prompt/model (generation tokens/sec,
peak memory, and cross-request reuse fraction) lives at
[`benchmark_scripts/benchmark_pool_backed_kvcache.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/benchmark_scripts/benchmark_pool_backed_kvcache.py),
with results committed at `figures/block_pool/pool_backed_kvcache_results.json`.

---

## See also

- [Allocators API](./allocators) — bit-level (not memory) allocation: RateQuant/VecInfer calibration
- [Cache API](./cache) — `KVCacheBuilder`, `KVCacheConfig`, `KVCacheFactory`
- [Exceptions API](./exceptions-api) — `BlockPoolExhaustedError`
