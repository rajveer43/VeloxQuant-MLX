---
id: exceptions-api
title: Exceptions API
sidebar_label: Exceptions
slug: /api/exceptions-api
---

# Exceptions API

`veloxquant_mlx.core.exceptions`

---

## Exception hierarchy

```
Exception
└── VeloxQuantError          # base for all library errors
    ├── ArtifactNotFoundError
    ├── CodebookDimensionMismatch
    ├── CyclicPipelineError
    ├── QuantizerConfigError
    ├── MetalUnavailableError
    └── BlockPoolExhaustedError
```

---

## VeloxQuantError

```python
from veloxquant_mlx.core.exceptions import VeloxQuantError
```

Base exception class. Catch this to handle any VeloxQuant-MLX error:

```python
try:
    cache = KVCacheBuilder.build(model, config)
except VeloxQuantError as e:
    print(f"VeloxQuant error: {e}")
```

---

## ArtifactNotFoundError

```python
from veloxquant_mlx.core.exceptions import ArtifactNotFoundError
```

Raised when a required calibration artifact (codebook, rotation, sensitivity map) is not found in the `ArtifactStore`.

**When raised:**
- Calling `VecInferKVCache` without a pre-trained codebook
- `load_cached_rotations()` when the path does not exist
- `NpyArtifactStore.load()` for a key that was never saved

```python
from veloxquant_mlx.core.exceptions import ArtifactNotFoundError
from veloxquant_mlx.artifacts.npy_store import NpyArtifactStore

store = NpyArtifactStore("./artifacts/")
try:
    codebook = store.load("vecinfer_codebook")
except ArtifactNotFoundError:
    print("Codebook not found. Run calibration first:")
    print("  python -m veloxquant_mlx precompute --method vecinfer --model ...")
```

---

## CodebookDimensionMismatch

```python
from veloxquant_mlx.core.exceptions import CodebookDimensionMismatch
```

Raised when a loaded codebook's dimensions do not match the current model's head dimensions.

**When raised:**
- Codebook was trained on a different model (different `head_dim` or `num_subspaces`)
- Codebook was trained with a different `num_subspaces` than specified in `KVCacheConfig`

**Message format:** `"Codebook shape [32, 8, 256, 16] incompatible with head_dim=128, num_subspaces=8"`

**Fix:** Re-run calibration with the correct model and configuration.

---

## CyclicPipelineError

```python
from veloxquant_mlx.core.exceptions import CyclicPipelineError
```

Raised when a `CompositeQuantizer` or the quantization DAG contains a cycle.

**When raised:**
- Building a `CompositeQuantizer` where a quantizer references itself (directly or transitively)
- Misconfigured custom pipeline using the `dag.py` utilities

---

## QuantizerConfigError

```python
from veloxquant_mlx.core.exceptions import QuantizerConfigError
```

Raised when a `KVCacheConfig` is invalid for the requested method.

**When raised:**
- `method="vecinfer"` without `codebook` or `smooth_factors`
- `method="spectral"` without `rotations`
- `bits` value not supported by the algorithm (e.g., `bits=5` for RVQ)
- `num_subspaces` does not divide `head_dim` evenly

**Message format:** `"VecInfer requires 'codebook' and 'smooth_factors' in KVCacheConfig"`

```python
from veloxquant_mlx.core.exceptions import QuantizerConfigError

try:
    config = KVCacheConfig(method="vecinfer")  # missing codebook
    cache = KVCacheBuilder.build(model, config)
except QuantizerConfigError as e:
    print(e)
    # "VecInfer requires 'codebook' and 'smooth_factors' in KVCacheConfig"
```

---

## MetalUnavailableError

```python
from veloxquant_mlx.core.exceptions import MetalUnavailableError
```

Raised when a Metal kernel is called on a device where Metal is not available.

**When raised:**
- Calling `vecinfer_quantize_metal()` on an Intel Mac or in a VM
- `patch_mlx_lm_for_fused_sdpa()` on an unsupported device

**Fix:** Run on macOS with an Apple M-series chip. See [Installation troubleshooting](../getting-started/installation).

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

## See also

- [Installation — troubleshooting](../getting-started/installation)
- [Calibration guide](../guides/calibration)
- [Core API](../api/core-api)
