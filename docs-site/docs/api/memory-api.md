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
in different compression formats side by side.

---

## Overview

| Class | Purpose |
|---|---|
| [`PoolConfig`](#poolconfig) | Sizing/behavior config for a pool (block size, block count, K/V separation) |
| [`BlockPoolAllocator`](#blockpoolallocator) | Pure-Python block bookkeeping — allocate, free, reuse, per-owner tracking, stats |
| [`Block`](#block) | A single fixed-size block handed out by the allocator |
| [`AllocationStats`](#allocationstats) | Running allocation/reuse/fragmentation counters |
| [`MLXBlockStorage`](#mlxblockstorage) | Pairs with a pool to provide the actual `mx.array` buffers, one per block id |
| [`PooledKVCache`](#pooledkvcache) | Wraps any existing `KVCache` so its appends check out/return blocks from a shared pool |

`BlockPoolAllocator`, `PoolConfig`, `Block`, and `AllocationStats` have no
MLX dependency — they are plain Python and can be used, tested, or reasoned
about independently of `mx.array`. `MLXBlockStorage` and `PooledKVCache`
are the MLX-backed pieces that plug the pool into actual cache storage.

---

## PoolConfig

```python
from veloxquant_mlx.memory import PoolConfig

@dataclass
class PoolConfig:
    block_size: int = 16
    n_blocks: int = 1024
    separate_kv: bool = True
```

**Fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `block_size` | `int` | `16` | Number of token slots stored per block |
| `n_blocks` | `int` | `1024` | Total number of blocks the pool manages |
| `separate_kv` | `bool` | `True` | If `True`, keys and values are allocated from independent free lists so K-heavy and V-heavy workloads never compete for the same blocks. If `False`, a single free list backs both, which packs tighter when K/V pressure is uneven. |

Raises `ValueError` if `block_size < 1` or `n_blocks < 1`.

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

**Raises:** `BlockPoolExhaustedError` if not enough free blocks remain;
`ValueError` if `n_tokens <= 0` or `stream` is invalid for the pool's
`separate_kv` setting.

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

### `stats`

A `BlockPoolAllocator` exposes its running counters as `pool.stats`, an
[`AllocationStats`](#allocationstats) instance updated on every
`allocate`/`free` call.

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
    peak_blocks_in_use: int = 0
```

| Field | Description |
|---|---|
| `n_blocks` | Total blocks managed by the pool |
| `n_allocations` | Cumulative successful `allocate()` calls |
| `n_frees` | Cumulative `free()` calls |
| `n_reused` | Allocations satisfied by a block that had previously been freed and returned to the pool (vs. a block never handed out before) |
| `n_exhausted` | Allocation attempts that failed because no free block was available |
| `peak_blocks_in_use` | High-water mark of concurrently allocated blocks |

**Methods:**

```python
def blocks_in_use(self) -> int      # n_allocations - n_frees
def fragmentation(self) -> float    # fraction of managed blocks that are allocated but not full
```

```python
pool = BlockPoolAllocator(PoolConfig(block_size=16, n_blocks=64))
pool.allocate(stream="k", n_tokens=40, owner=1)
print(pool.stats)
# AllocationStats(in_use=3/64, allocations=3, frees=0, reused=0,
#                  exhausted=0, peak=3, fragmentation=0.016)
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

## Usage

```python
from veloxquant_mlx import KVCacheBuilder, KVCacheConfig
from veloxquant_mlx.memory import BlockPoolAllocator, PoolConfig, PooledKVCache

pool = BlockPoolAllocator(PoolConfig(block_size=16, n_blocks=512))

inner = (
    KVCacheBuilder()
    .with_method("kivi")
    .with_head_dim(128)
    .build()
)
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

---

## See also

- [Allocators API](./allocators) — bit-level (not memory) allocation: RateQuant/VecInfer calibration
- [Cache API](./cache) — `KVCacheBuilder`, `KVCacheConfig`, `KVCacheFactory`
- [Exceptions API](./exceptions-api) — `BlockPoolExhaustedError`
