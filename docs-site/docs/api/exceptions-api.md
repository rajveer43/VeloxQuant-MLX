---
id: exceptions-api
title: Exceptions API
sidebar_label: Exceptions
slug: /api/exceptions-api
description: Python API reference for veloxquant_mlx.core.exceptions, covering ArtifactNotFoundError, CodebookDimensionMismatch, CyclicPipelineError, QuantizerConfigError, BlockPoolExhaustedError, and OwnerAlreadyActiveError.
keywords: [exceptions, "API reference", "python api", error handling, BlockPoolExhaustedError]
---

# Exceptions API

`veloxquant_mlx.core.exceptions`

---

## Exception hierarchy

There is no common VeloxQuant-specific base exception. Each of the six
exceptions independently subclasses a Python builtin, grouped here by that
builtin:

```
ValueError
├── QuantizerConfigError
├── CodebookDimensionMismatch
└── OwnerAlreadyActiveError

FileNotFoundError
└── ArtifactNotFoundError

RuntimeError
├── CyclicPipelineError
└── BlockPoolExhaustedError
```

Catch a specific exception, or catch its builtin category if you want to
handle a broader class of errors — for example, `except ValueError` also
catches `QuantizerConfigError`, `CodebookDimensionMismatch`, and
`OwnerAlreadyActiveError`, since all three subclass it.

---

## ArtifactNotFoundError

```python
from veloxquant_mlx.core.exceptions import ArtifactNotFoundError
```

Raised when a required precomputed artifact (rotation matrix, codebook, or JL matrix) is not found in the artifact store.

**When raised:**
- `NpyArtifactStore.load_rotation_matrix()` / `load_codebook()` / `load_jl_matrix()` when the backing `.npy` file does not exist on disk
- The equivalent lookups on `MemoryArtifactStore` when the key was never saved

```python
from veloxquant_mlx.core.exceptions import ArtifactNotFoundError
from veloxquant_mlx.artifacts.npy_store import NpyArtifactStore

store = NpyArtifactStore("./artifacts/")
try:
    codebook = store.load_codebook(distribution="gaussian", b=4, d=128)
except ArtifactNotFoundError:
    print("Codebook not found. Run calibration first:")
    print("  python -m veloxquant_mlx precompute --head_dim 128 --bits 4")
```

---

## CodebookDimensionMismatch

```python
from veloxquant_mlx.core.exceptions import CodebookDimensionMismatch
```

Raised when a codebook's shape does not match the expected dimension — for example, when constructing a `ScalarCodebook` from centroids that aren't a 1-D array, or whose count isn't a power of two.

**When raised:**
- `ScalarCodebook(centroids)` is given a `centroids` array with more than one dimension
- `ScalarCodebook(centroids)` is given a number of centroids that is not a power of 2 (so it doesn't correspond to a whole number of bits)

**Fix:** Re-run calibration to regenerate a codebook with the expected shape, or check the array you're passing in.

---

## CyclicPipelineError

```python
from veloxquant_mlx.core.exceptions import CyclicPipelineError
```

Raised when a `QuantizationGraph` (see `veloxquant_mlx/dsa/dag.py`) contains a cycle and cannot be reduced to a valid topological order.

**When raised:**
- Calling the graph's topological-sort step on a `QuantizationGraph` whose `set_next()` / `add_edge()` calls introduced a cycle

---

## QuantizerConfigError

```python
from veloxquant_mlx.core.exceptions import QuantizerConfigError
```

Raised when a quantizer or KV cache is misconfigured — the most common exception in the library. It covers a range of validation failures across `KVCacheFactory`, `KVCacheBuilder`, `QuantizerFactory`, `PreconditionerFactory`, `CodebookFactory`, and individual quantizers/caches.

**When raised:**
- An unknown `method` is passed to `KVCacheConfig` / `KVCacheFactory.create()`
- `head_dim` is not a power of 2, or `bit_width_inlier` is invalid (e.g. `< 1`, or an empty/mixed-type list)
- `jl_dim` or `n_outlier_channels` is out of range relative to `head_dim`
- `sliding_window` is set for a method that doesn't support it
- A quantizer- or preconditioner-specific constraint fails (e.g. `KIVIQuantizer`'s `b` outside `[1, 8]`, or a `PreconditionerFactory` call missing a required kwarg like `Pi` or `S`)

```python
from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig

try:
    config = KVCacheConfig(method="vecinfer", head_dim=100)  # not a power of 2
    caches = KVCacheBuilder.for_model(model, config)
except QuantizerConfigError as e:
    print(e)
    # "KVCacheBuilder: head_dim=100 must be a power of 2."
```

---

## BlockPoolExhaustedError

```python
from veloxquant_mlx.core.exceptions import BlockPoolExhaustedError
```

Raised when a `BlockPoolAllocator` has no free blocks left to satisfy an
`allocate()` call. Allocation is all-or-nothing — no blocks are handed out
on failure.

**When raised:**
- `pool.allocate(...)` requests more blocks than are currently free for the
  requested stream (`"k"` or `"v"`)
- A `PooledKVCache` needs a new block mid-generation and the shared pool is
  fully checked out by other requests

**Fix:** Increase `PoolConfig.n_blocks`, reduce concurrent request count, or
call `release()` / `free_all()` on finished requests sooner.

```python
from veloxquant_mlx.core.exceptions import BlockPoolExhaustedError
from veloxquant_mlx.memory import BlockPoolAllocator, PoolConfig

pool = BlockPoolAllocator(PoolConfig(block_size=16, n_blocks=4))
try:
    pool.allocate(stream="k", n_tokens=1000, owner=1)
except BlockPoolExhaustedError as e:
    print(e)
```

See [Memory (Block Pool) API](./memory-api) for the full allocator reference.

---

## OwnerAlreadyActiveError

```python
from veloxquant_mlx.core.exceptions import OwnerAlreadyActiveError
```

Raised when an owner id is registered on a `BlockPoolAllocator` while it's
still checked out to another (unreleased) caller — catches two different
requests accidentally reusing the same owner id, which would otherwise let
one silently free the other's blocks via `free_all()`.

**When raised:**
- `pool.register_owner(owner)` is called twice for the same `owner` before
  a `free_all(owner)` / `release_owner(owner)` in between

**Not raised for:** a second `pool.allocate(..., owner=owner)` call for an
owner that's already active — that's the normal pattern for a single
request growing over multiple calls (e.g. `PoolBackedKVCache` appending
more blocks), and `allocate()` can't otherwise distinguish it from a
collision. Call `register_owner()` yourself up front if you want
collisions between different callers caught immediately.

**Fix:** Use a different owner id (e.g. a monotonically increasing request
counter that isn't reused until confirmed released), or call
`pool.release_owner(owner)` / `pool.free_all(owner)` before reusing the id.

```python
from veloxquant_mlx.core.exceptions import OwnerAlreadyActiveError
from veloxquant_mlx.memory import BlockPoolAllocator, PoolConfig

pool = BlockPoolAllocator(PoolConfig(block_size=16, n_blocks=64))
pool.register_owner(request_id)
try:
    pool.register_owner(request_id)  # still active -- likely an id collision
except OwnerAlreadyActiveError as e:
    print(e)
```

See [Memory (Block Pool) API](./memory-api) for the full allocator reference.

---

## See also

- [Installation — troubleshooting](../getting-started/installation)
- [Calibration guide](../guides/calibration)
- [Core API](../api/core-api)
