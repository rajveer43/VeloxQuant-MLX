---
id: core-api
title: Core Abstractions API
sidebar_label: Core
slug: /api/core-api
description: Python API reference for veloxquant_mlx.core, covering the nine abstract base classes (Quantizer, Preconditioner, Codebook, KVCache, Transform, QuantizationHandler, InnerProductStrategy, CodebookStrategy, ArtifactStore, QuantizationObserver), the EncodedVector/QuantizationContext/TransformResult context types, the QuantizerRegistry/CodebookRegistry/PreconditionerRegistry, and the veloxquant_mlx CLI commands.
keywords: [core abstractions, Quantizer, KVCache, "API reference", "python api", QuantizerRegistry, EncodedVector, CLI]
---

# Core Abstractions API

`veloxquant_mlx.core`

Framework-level primitives shared across the whole quantization pipeline: the
abstract base classes every quantizer/codebook/cache/store implements, the
data-carrying types passed between pipeline stages, and the name-to-class
registries used to select implementations by string key.

---

## Abstract base classes

`veloxquant_mlx.core.abstractions`

This module declares **nine** ABCs. All concrete implementations across the
codebase subclass one of these rather than being duck-typed against a
concrete class — program to these interfaces when building custom
integrations.

### Quantizer

```python
from veloxquant_mlx.core.abstractions import Quantizer
```

Abstract base class for all vector quantizers.

```python
class Quantizer(ABC):
    @abstractmethod
    def encode(self, x: Any) -> EncodedVector: ...

    @abstractmethod
    def decode(self, ev: EncodedVector) -> Any: ...

    @abstractmethod
    def estimate_inner_product(self, q: Any, ev: EncodedVector) -> Any: ...
```

- `encode(x)` — `x` is an input array of shape `(batch, d)`, fp16. Returns an
  `EncodedVector` containing the compressed representation.
- `decode(ev)` — `ev` is an `EncodedVector` produced by `encode()`. Returns
  the reconstructed array of shape `(batch, d)`, fp16.
- `estimate_inner_product(q, ev)` — `q` is a query vector of shape `(d,)` or
  `(1, d)`, fp16; `ev` is an encoded key cache where `batch_size == n_keys`.
  Returns estimated inner products, shape `(batch_size,)`, fp16, without
  fully decoding the cache.

### Preconditioner

```python
from veloxquant_mlx.core.abstractions import Preconditioner
```

Abstract base class for linear preconditioners (rotation/JL sketch), applied
before quantization.

```python
class Preconditioner(ABC):
    @abstractmethod
    def apply(self, x: Any) -> Any: ...

    @abstractmethod
    def apply_inverse(self, y: Any) -> Any: ...
```

- `apply(x)` — forward transform. `x` shape `(batch, d)`, returns shape
  `(batch, out_dim)`.
- `apply_inverse(y)` — inverse (transpose) transform. `y` shape
  `(batch, out_dim)`, returns the reconstructed array of shape `(batch, d)`.

Note the method is `apply_inverse`, not `inverse`.

### Codebook

```python
from veloxquant_mlx.core.abstractions import Codebook
```

Abstract base class for scalar codebooks.

```python
class Codebook(ABC):
    @abstractmethod
    def quantize(self, y: Any) -> Any: ...

    @abstractmethod
    def dequantize(self, idx: Any) -> Any: ...
```

- `quantize(y)` — maps coordinates to nearest-centroid indices. `y` shape
  `(batch, d)`, returns an index array of shape `(batch, d)`, dtype `uint8`.
- `dequantize(idx)` — retrieves centroid values for given indices. `idx`
  shape `(batch, d)` uint8, returns a centroid array of shape `(batch, d)`.

### KVCache

```python
from veloxquant_mlx.core.abstractions import KVCache
```

Abstract base class for KV cache implementations.

```python
class KVCache(ABC):
    @abstractmethod
    def append_key(self, k: Any) -> None: ...

    @abstractmethod
    def append_value(self, v: Any) -> None: ...

    @abstractmethod
    def attend(self, q: Any) -> Any: ...

    @abstractmethod
    def memory_bytes(self) -> int: ...

    def append(self, k: Any, v: Any) -> None:
        self.append_key(k)
        self.append_value(v)

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
```

- `append_key(k)` / `append_value(v)` — append a key or value vector, shape
  `(d,)`, fp16, to the cache.
- `attend(q)` — compute the attention-weighted value for a query vector `q`
  of shape `(d,)`, fp16. Returns the attention output, shape `(d,)`, fp16.
- `memory_bytes()` — return the current memory footprint of the cache in
  bytes.
- `append(k, v)` — **concrete** convenience method (not abstract) that calls
  `append_key(k)` then `append_value(v)` in one call.

There is no `update()`/`state` API on `KVCache` — callers append keys/values
one at a time and call `attend()` per query.

### Transform

```python
from veloxquant_mlx.core.abstractions import Transform
```

Abstract base class for invertible vector transforms, used by PolarQuant.

```python
class Transform(ABC):
    @abstractmethod
    def forward(self, x: Any) -> TransformResult: ...

    @abstractmethod
    def inverse(self, result: TransformResult) -> Any: ...
```

- `forward(x)` — `x` shape `(batch, d)`. Returns a `TransformResult`
  encapsulating all intermediate and final values (angles, radius, level
  count) needed to invert the transform.
- `inverse(result)` — reconstructs the original vector, shape `(batch, d)`,
  from a `TransformResult` produced by a prior `forward()` call.

### QuantizationHandler

```python
from veloxquant_mlx.core.abstractions import QuantizationHandler
```

Abstract base for Chain of Responsibility pipeline stages (see
`veloxquant_mlx.handlers`). Subclasses implement `handle()` to mutate a
`QuantizationContext` and call `_pass_to_next()` to continue the chain.

```python
class QuantizationHandler(ABC):
    _next: QuantizationHandler | None = None

    def set_next(self, handler: QuantizationHandler) -> QuantizationHandler: ...

    @abstractmethod
    def handle(self, ctx: QuantizationContext) -> QuantizationContext: ...

    def _pass_to_next(self, ctx: QuantizationContext) -> QuantizationContext: ...

    @property
    @abstractmethod
    def handler_name(self) -> str: ...
```

- `set_next(handler)` — attaches the next handler and returns it, enabling
  the fluent `a.set_next(b).set_next(c)` idiom.
- `handle(ctx)` — process the mutable context and optionally pass it
  downstream; returns the (possibly mutated) context.
- `_pass_to_next(ctx)` — forwards the context to the next handler if one is
  attached, otherwise returns `ctx` unchanged (end of chain).
- `handler_name` — abstract property; a human-readable stage name used in
  DAG and Observer events.

### InnerProductStrategy

```python
from veloxquant_mlx.core.abstractions import InnerProductStrategy
```

Strategy for estimating inner products between queries and encoded keys.

```python
class InnerProductStrategy(ABC):
    @abstractmethod
    def estimate(self, q: Any, encoded: EncodedVector) -> Any: ...
```

- `estimate(q, encoded)` — `q` shape `(d,)` or `(1, d)`. Returns estimated
  ⟨q, k⟩ inner products for each encoded key, shape `(batch_size,)`.

### CodebookStrategy

```python
from veloxquant_mlx.core.abstractions import CodebookStrategy
```

Strategy for computing optimal codebook centroids.

```python
class CodebookStrategy(ABC):
    @abstractmethod
    def compute_centroids(self, b: int, d: int) -> Any: ...
```

- `compute_centroids(b, d)` — `b` is the bit-width (number of bits per
  code), `d` is the vector dimension (used to set distribution variance).
  Returns a numpy array of shape `(2^b,)` containing sorted centroids.

### ArtifactStore

```python
from veloxquant_mlx.core.abstractions import ArtifactStore
```

DAO interface for loading and saving precomputed quantization artifacts
(rotation matrices, codebooks, JL projection matrices).

```python
class ArtifactStore(ABC):
    @abstractmethod
    def load_rotation_matrix(self, d: int, seed: int) -> Any: ...

    @abstractmethod
    def save_rotation_matrix(self, Pi: Any, d: int, seed: int) -> None: ...

    @abstractmethod
    def load_codebook(self, distribution: str, b: int, d: int) -> Any: ...

    @abstractmethod
    def save_codebook(self, cb: Any, distribution: str, b: int, d: int) -> None: ...

    @abstractmethod
    def load_jl_matrix(self, d: int, m: int, seed: int) -> Any: ...

    @abstractmethod
    def save_jl_matrix(self, S: Any, d: int, m: int, seed: int) -> None: ...

    @abstractmethod
    def exists(self, artifact_type: str, **kwargs: Any) -> bool: ...
```

- `load_rotation_matrix(d, seed)` / `save_rotation_matrix(Pi, d, seed)` —
  load/persist a rotation matrix of shape `(d, d)`, fp16. Load raises
  `ArtifactNotFoundError` if the artifact does not exist.
- `load_codebook(distribution, b, d)` / `save_codebook(cb, distribution, b, d)`
  — load/persist a codebook of centroids, shape `(2^b,)`, fp16, keyed by
  distribution name (e.g. `'gaussian'`, `'beta'`), bit-width, and dimension.
- `load_jl_matrix(d, m, seed)` / `save_jl_matrix(S, d, m, seed)` —
  load/persist a JL projection matrix of shape `(m, d)`, fp16, mapping input
  dimension `d` to sketch dimension `m`.
- `exists(artifact_type, **kwargs)` — check whether a specific artifact
  exists in the store. `artifact_type` is one of `'rotation'`, `'codebook'`,
  `'jl'`; `**kwargs` are the identifying parameters (`d`, `seed`, `b`,
  `distribution`, `m`).

### QuantizationObserver

```python
from veloxquant_mlx.core.abstractions import QuantizationObserver
```

Observer for pipeline events (timing, memory, distortion).

```python
class QuantizationObserver(ABC):
    @abstractmethod
    def on_event(self, event: Any) -> None: ...
```

- `on_event(event)` — handle a quantization pipeline event (a
  `QuantizationEvent` dataclass instance).

---

## Context types

`veloxquant_mlx.core.context`

Data-carrying types shared across the quantization handler chain and
quantizers. These dataclasses decouple the handler pipeline and quantizer
implementations from MLX's array type via lazy import.

### QuantizationContext

```python
from veloxquant_mlx.core.context import QuantizationContext
```

The mutable payload passed between `QuantizationHandler` stages (see
`veloxquant_mlx.handlers`) as a vector is encoded or decoded — **not** a
request-scoped context keyed by layer name/step/config.

```python
@dataclass
class QuantizationContext:
    x_original: Any  # mx.array (batch, d)
    mode: Literal["encode", "decode"]
    x_current: Any  # mx.array (batch, d)
    norm: Any | None = None  # mx.array (batch,)
    rotated: Any | None = None  # mx.array (batch, d)
    indices: Any | None = None  # mx.array (batch, d) uint8
    signs: Any | None = None  # mx.array (batch, m) int8
    residual_norm: Any | None = None  # mx.array (batch,)
    angles: list[Any] | None = None  # list of mx.array per level
    final_radius: Any | None = None  # mx.array (batch,)
    outlier_idx: np.ndarray | None = None
    packed_bits: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Field meanings:

- `x_original` — original input vectors, shape `(batch, d)`.
- `mode` — whether the chain is encoding or decoding.
- `x_current` — working copy mutated by each handler.
- `norm` — L2 norm stored by `NormalizationHandler`, shape `(batch,)`.
- `rotated` — vector after `RotationHandler`, shape `(batch, d)`.
- `indices` — codebook indices, shape `(batch, d)` uint8.
- `signs` — QJL sign bits, shape `(batch, m)` int8.
- `residual_norm` — residual L2 norm for the QJL stage, shape `(batch,)`.
- `angles` — per-level polar angles from `PolarTransformHandler`.
- `final_radius` — scalar radius after all polar levels, shape `(batch,)`.
- `outlier_idx` — channel positions of outlier channels.
- `packed_bits` — bit-packed index array from `BitPackingHandler`.
- `metadata` — arbitrary stage-specific metadata.

### EncodedVector

```python
from veloxquant_mlx.core.context import EncodedVector
```

The typed, memory-accountable output of `Quantizer.encode()`. Different
quantizer types populate different subsets of fields.

```python
@dataclass
class EncodedVector:
    quantizer_type: str
    batch_size: int
    dim: int
    indices: Any | None = None
    norm: Any | None = None
    signs: Any | None = None
    residual_norm: Any | None = None
    angles: list[Any] | None = None
    final_radius: Any | None = None
    outlier_idx: np.ndarray | None = None
    outlier_encoded: EncodedVector | None = None
    inlier_encoded: EncodedVector | None = None

    def memory_bytes(self) -> int: ...
```

Field meanings:

- `quantizer_type` — registry key of the producing quantizer.
- `batch_size` — number of vectors encoded.
- `dim` — original vector dimensionality.
- `indices` — scalar codebook indices, shape `(batch, d)` uint8.
- `norm` — per-vector L2 norm, shape `(batch,)` fp16.
- `signs` — QJL sign bits, shape `(batch, m)` int8.
- `residual_norm` — QJL residual norm, shape `(batch,)` fp16.
- `angles` — PolarQuant level angles, a list of `(batch, d/2^ℓ)` arrays.
- `final_radius` — PolarQuant scalar radius, shape `(batch,)` fp16.
- `outlier_idx` — outlier channel indices for `CompositeQuantizer`.
- `outlier_encoded` / `inlier_encoded` — nested `EncodedVector` instances for
  the outlier/inlier channel splits used by `CompositeQuantizer`.

`memory_bytes()` computes the exact byte footprint of the encoded
representation: it sums the `nbytes` of every populated array field
(handling both `numpy.ndarray` and `mx.array`, inferring itemsize from
dtype for MLX arrays), and recurses into `outlier_encoded`/`inlier_encoded`
when present. This is genuinely useful for measuring compression ratio
without decoding.

### TransformResult

```python
from veloxquant_mlx.core.context import TransformResult
```

Output of `RecursivePolarTransform.forward()` — i.e. the `Transform` ABC's
`forward()` output, specific to the recursive polar decomposition used by
PolarQuant. It is **not** generic pre-transform metadata and is not produced
by `Preconditioner.apply()`.

```python
@dataclass
class TransformResult:
    angles: list[Any]  # list of mx.array
    final_radius: Any  # mx.array (batch,)
    n_levels: int
```

- `angles` — list of angle arrays, one per polar level. `angles[0]` has
  shape `(batch, d/2)`; `angles[ℓ]` has shape `(batch, d/2^(ℓ+1))`.
- `final_radius` — scalar radius at the end of recursion, shape `(batch,)`.
- `n_levels` — number of polar recursion levels applied.

`Transform.inverse()` consumes a `TransformResult` to reconstruct the
original vector.

---

## Registry

`veloxquant_mlx.core.registry`

Thread-safe name-to-class registries used to decouple config from
implementation. This lets quantizers, codebooks, and preconditioners be
selected by string key (e.g. from a config file) without importing every
concrete class up front, and lets each registry family maintain its own
independent namespace.

```python
from veloxquant_mlx.core.registry import (
    QuantizerRegistry,
    CodebookRegistry,
    PreconditionerRegistry,
)
```

All three are `_BaseRegistry` subclasses exposing the same interface:

```python
class _BaseRegistry:
    @classmethod
    def register(cls, name: str): ...  # class decorator

    @classmethod
    def get(cls, name: str) -> type: ...  # raises KeyError if not registered

    @classmethod
    def list_names(cls) -> list[str]: ...

    @classmethod
    def is_registered(cls, name: str) -> bool: ...
```

- `register(name)` — class decorator that registers a concrete class under
  `name`. Raises `KeyError` if `name` is already registered by a different
  class.
- `get(name)` — retrieve a registered class by name. Raises `KeyError` if
  `name` is not registered; the error message lists the available names.
- `list_names()` — return all registered names, sorted alphabetically.
- `is_registered(name)` — return `True` if `name` is registered.

Example — registering and looking up a custom quantizer:

```python
from veloxquant_mlx.core.abstractions import Quantizer
from veloxquant_mlx.core.registry import QuantizerRegistry

@QuantizerRegistry.register("my_quantizer")
class MyQuantizer(Quantizer):
    def encode(self, x): ...
    def decode(self, ev): ...
    def estimate_inner_product(self, q, ev): ...

# Look up the class by name and instantiate it directly
cls = QuantizerRegistry.get("my_quantizer")
q = cls(bits=2)
```

There is **no** `QuantizerFactory` in `veloxquant_mlx.core.registry` — the
core registries only map names to classes via `get()`, they do not
instantiate. (A separate, narrower `QuantizerFactory` exists in
`veloxquant_mlx.quantizers.base`; see the [Quantizers API](../api/quantizers)
for that distinct, higher-level factory. Do not conflate the two — the
`core.registry` classes are the low-level plugin registries used across
quantizers, codebooks, and preconditioners alike.)

`CodebookRegistry` registers `CodebookStrategy` concrete classes, and
`PreconditionerRegistry` registers `Preconditioner` concrete classes,
following the same `register`/`get`/`list_names`/`is_registered` interface.

---

## CLI reference

The `veloxquant_mlx` package exposes **nine** CLI commands via
`python -m veloxquant_mlx <command>`, dispatched by `veloxquant_mlx/__main__.py`:

```
python -m veloxquant_mlx {precompute|benchmark|recommend|auto-config|methods|serve|profile|panel|worker}
```

Each command maps to a module under `veloxquant_mlx/cli/`:

| Command | Module |
| --- | --- |
| `precompute` | `veloxquant_mlx.cli.precompute` |
| `benchmark` | `veloxquant_mlx.cli.benchmark` |
| `recommend` | `veloxquant_mlx.cli.recommend` |
| `auto-config` | `veloxquant_mlx.cli.auto_config` |
| `methods` | `veloxquant_mlx.cli.methods` |
| `serve` | `veloxquant_mlx.cli.serve` |
| `profile` | `veloxquant_mlx.cli.profile` |
| `panel` | `veloxquant_mlx.cli.panel` |
| `worker` | `veloxquant_mlx.cli.worker` |

Flags are documented below for `precompute` and `benchmark`; see the
corresponding module under `veloxquant_mlx/cli/` for the others.

### `precompute`

```bash
python -m veloxquant_mlx precompute \
    [--head_dim HEAD_DIM] \
    [--bits BITS [BITS ...]] \
    [--jl_dim JL_DIM] \
    [--seed SEED] \
    [--output_dir OUTPUT_DIR]
```

Precomputes rotation matrices, JL matrices, and codebooks and saves them as
artifacts.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--head_dim` | int | `128` | Attention head dimension. |
| `--bits` | int (nargs `+`) | `[1, 2, 3, 4]` | Bit-widths to precompute. |
| `--jl_dim` | int | `128` | JL projection dimension. |
| `--seed` | int | `42` | Random seed. |
| `--output_dir` | str | `./artifacts/` | Output directory. |

### `benchmark`

```bash
python -m veloxquant_mlx benchmark \
    [--method {turboquant_prod,turboquant_mse,qjl,polar}] \
    [--head_dim HEAD_DIM] \
    [--bits BITS] \
    [--jl_dim JL_DIM] \
    [--seq_len SEQ_LEN] \
    [--seq_lens SEQ_LEN [SEQ_LEN ...]] \
    [--seed SEED] \
    [--compare_optimized] \
    [--n_outlier_channels N] \
    [--n_calib_tokens N]
```

Benchmarks KV cache encode/decode latency and memory, printing a table of
`seq_len | baseline_attend_ms | optimized_attend_ms | speedup` (the
optimized column is only populated when `--compare_optimized` is passed).

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--method` | choice | `turboquant_prod` | One of `turboquant_prod`, `turboquant_mse`, `qjl`, `polar`. |
| `--head_dim` | int | `128` | Attention head dimension. |
| `--bits` | int | `3` | Inlier bit-width. |
| `--jl_dim` | int | `128` | JL projection dimension. |
| `--seq_len` | int | `1000` | Sequence length for a single run. |
| `--seq_lens` | int (nargs `*`) | `None` | Multiple sequence lengths to sweep; overrides `--seq_len` when given. |
| `--seed` | int | `42` | Random seed. |
| `--compare_optimized` | flag | off | Also build and time a vectorized/fused/outlier-two-stream optimized cache alongside the baseline. |
| `--n_outlier_channels` | int | `4` | Number of outlier channels for the optimized cache. |
| `--n_calib_tokens` | int | `200` | Number of calibration tokens for the optimized cache. |

There is no `--model`, `--output`, `--value-bits`, or `--num-runs` flag on
either command — the cache is built via `KVCacheBuilder` directly from
these arguments and benchmarked against synthetic random keys/values, not a
loaded model checkpoint. See [Benchmarking guide](../guides/benchmarking).

---

## See also

- [API — Quantizers](../api/quantizers)
- [API — Cache](../api/cache)
- [API — Exceptions](../api/exceptions-api)
