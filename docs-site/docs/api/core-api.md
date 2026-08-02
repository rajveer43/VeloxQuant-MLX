---
id: core-api
title: Core Abstractions API
sidebar_label: Core
slug: /api/core-api
---

# Core Abstractions API

`veloxquant_mlx.core`

---

## Abstract base classes

`veloxquant_mlx.core.abstractions`

All concrete implementations subclass these ABCs. You should program to these interfaces when building custom integrations.

### Quantizer

```python
from veloxquant_mlx.core.abstractions import Quantizer
```

```python
class Quantizer(ABC):
    @abstractmethod
    def encode(self, x: mx.array) -> EncodedVector: ...

    @abstractmethod
    def decode(self, encoded: EncodedVector) -> mx.array: ...

    def encode_values(self, x: mx.array) -> EncodedVector:
        return self.encode(x)

    def decode_values(self, encoded: EncodedVector) -> mx.array:
        return self.decode(encoded)
```

All quantizers implement `encode` and `decode` with these signatures:
- `encode(x)` — input shape `[batch, heads, seq, head_dim]`, returns `EncodedVector`
- `decode(encoded)` — returns `mx.array` of shape `[batch, heads, seq, head_dim]`

### KVCache

```python
from veloxquant_mlx.core.abstractions import KVCache
```

```python
class KVCache(ABC):
    @abstractmethod
    def update(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]: ...

    @property
    @abstractmethod
    def state(self) -> tuple[mx.array, mx.array]: ...
```

`update(keys, values)` is called once per generation step. It writes the new keys/values to the compressed cache and returns the full (dequantized) cache for attention computation.

### Preconditioner

```python
from veloxquant_mlx.core.abstractions import Preconditioner
```

Linear transforms applied before quantization.

```python
class Preconditioner(ABC):
    @abstractmethod
    def apply(self, x: mx.array) -> mx.array: ...

    @abstractmethod
    def inverse(self, x: mx.array) -> mx.array: ...
```

### Codebook

```python
from veloxquant_mlx.core.abstractions import Codebook
```

```python
class Codebook(ABC):
    @abstractmethod
    def quantize(self, x: mx.array) -> mx.array: ...  # returns indices

    @abstractmethod
    def dequantize(self, indices: mx.array) -> mx.array: ...
```

---

## Context types

`veloxquant_mlx.core.context`

### EncodedVector

```python
from veloxquant_mlx.core.context import EncodedVector
```

```python
@dataclass
class EncodedVector:
    indices: mx.array  # packed integer codes
    scale: mx.array | None  # per-channel or per-block scale
    metadata: dict  # algorithm-specific extra data
    original_shape: tuple[int, ...]
    dtype: mx.Dtype  # original dtype (usually mx.float16)
```

`EncodedVector` is the currency passed between `encode()` and `decode()`. Different algorithms store different things in `metadata` (e.g., cluster IDs for RaBitQ, rotation info for SpectralQuant).

### QuantizationContext

```python
from veloxquant_mlx.core.context import QuantizationContext
```

Request-scoped context passed through a quantization pipeline.

```python
@dataclass
class QuantizationContext:
    layer_name: str
    step: int  # generation step (0-indexed)
    config: KVCacheConfig
    artifacts: ArtifactStore
```

### TransformResult

```python
from veloxquant_mlx.core.context import TransformResult
```

Output of a `Preconditioner.apply()` call, including pre-transform metadata needed for the inverse.

---

## Registry

```python
from veloxquant_mlx.core.registry import QuantizerRegistry
```

Plugin registry for quantizer discovery.

```python
# Register a custom quantizer
@QuantizerRegistry.register("my_quantizer")
class MyQuantizer(Quantizer):
    def encode(self, x): ...
    def decode(self, encoded): ...


# Create by name
q = QuantizerFactory.create("my_quantizer", bits=2)
```

---

## CLI reference

The `veloxquant_mlx` package exposes two CLI commands via `python -m veloxquant_mlx`:

### `precompute`

```bash
python -m veloxquant_mlx precompute \
    --method {vecinfer,spectral,ratequant} \
    --model MODEL_PATH \
    --output ARTIFACT_DIR \
    [--num-samples N] \
    [--sequence-length L] \
    [--target-bits BITS]
```

Runs calibration for the specified method and saves artifacts.

### `benchmark`

```bash
python -m veloxquant_mlx benchmark \
    --model MODEL_PATH \
    --method METHOD \
    --bits BITS \
    [--value-bits BITS] \
    [--seq-len SEQ] \
    [--num-runs N] \
    [--output JSON_PATH]
```

Benchmarks a configuration and prints/saves metrics. See [Benchmarking guide](../guides/benchmarking).

---

## See also

- [API — Quantizers](../api/quantizers)
- [API — Cache](../api/cache)
- [API — Exceptions](../api/exceptions-api)
