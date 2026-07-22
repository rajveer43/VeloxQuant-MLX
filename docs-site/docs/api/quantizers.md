---
id: quantizers
title: Quantizers API
sidebar_label: Quantizers
slug: /api/quantizers
---

# Quantizers API

`veloxquant_mlx.quantizers`

All quantizers implement the `Quantizer` abstract base class (`encode`, `decode`, `estimate_inner_product`) and take `x` of shape `(batch, d)` — not `(batch, heads, seq, head_dim)`; flatten the head/seq dims before calling `encode`.

---

## QuantizerFactory

```python
from veloxquant_mlx.quantizers.base import QuantizerFactory
```

### `QuantizerFactory.create`

```python
@staticmethod
def create(
    method: Literal["qjl", "turboquant_mse", "turboquant_prod", "polar"],
    d: int,
    b: int = 2,
    m: Optional[int] = None,
    seed: int = 42,
    store: Optional[ArtifactStore] = None,
    **kwargs,
) -> Quantizer
```

Registered names: `"qjl"`, `"turboquant_mse"`, `"turboquant_prod"`, `"polar"`. `TurboQuantRVQ`, `RaBitQQuantizer`, `CommVQQuantizer`, and `CompositeQuantizer` are **not** registered in this factory — construct them directly from their own module.

```python
q = QuantizerFactory.create("turboquant_mse", d=128, b=2, seed=42)
q = QuantizerFactory.create("turboquant_prod", d=128, b=3, m=128, seed=42)
q = QuantizerFactory.create("polar", d=128, b=2, seed=42)
```

---

## TurboQuantRVQ

```python
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ
```

Two-pass Residual VQ with Gaussian analytical codebooks.

### Constructor

```python
TurboQuantRVQ(
    d: int,
    b: int = 2,
    seed: int = 42,
    m: int = 0,               # unused, kept for factory API compatibility
    store: Optional[ArtifactStore] = None,
    use_hadamard: bool = False,
    residual_scale: Optional[float] = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `d` | `int` | — | Vector dimension (required) |
| `b` | `int` | `2` | Bits per residual pass |
| `seed` | `int` | `42` | Random seed |
| `use_hadamard` | `bool` | `False` | Apply Walsh-Hadamard rotation before quantizing |
| `residual_scale` | `Optional[float]` | `None` | Override for the residual scaling factor |

### Methods

```python
def encode(self, x: Any) -> EncodedVector: ...
def decode(self, ev: EncodedVector) -> Any: ...
```

`encode(x)` takes `x` of shape `(batch, d)` (fp16), returns an `EncodedVector` with stage-1 and stage-2 codes. `decode(ev)` reconstructs approximate vectors of shape `(batch, d)`.

```python
import mlx.core as mx
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ

q = TurboQuantRVQ(d=128, b=1, seed=42)
x = mx.array(mx.random.normal(shape=(512, 128)))  # [batch, d]
encoded = q.encode(x)
decoded = q.decode(encoded)
```

---

## TurboQuantMSE

```python
from veloxquant_mlx.quantizers.turboquant_mse import TurboQuantMSE
```

MSE-optimal scalar quantization via Lloyd-Max codebooks, with optional Walsh-Hadamard rotation. No residual pass.

### Constructor

```python
TurboQuantMSE(
    d: int,
    b: int = 2,
    seed: int = 42,
    m: int = 128,
    store: Optional[ArtifactStore] = None,
    use_beta: bool = False,
    use_hadamard: bool = False,
)
```

---

## TurboQuantProd

```python
from veloxquant_mlx.quantizers.turboquant_prod import TurboQuantProd
```

Product VQ: Lloyd-Max scalar centroids for the primary pass, with an optional adaptive codebook and Hadamard rotation.

### Constructor

```python
TurboQuantProd(
    d: int,
    b: int = 3,
    m: Optional[int] = None,       # defaults via TurboQuantProd.m_default(d, b)
    seed: int = 42,
    store: Optional[ArtifactStore] = None,
    use_hadamard: bool = False,
    use_adaptive_codebook: bool = False,
    n_calib: int = 64,
)
```

### TurboQuantProdAdaptive

```python
from veloxquant_mlx.quantizers.turboquant_prod import TurboQuantProdAdaptive
```

A thin subclass of `TurboQuantProd` that simply defaults `use_adaptive_codebook=True`. It takes the exact same constructor arguments as `TurboQuantProd` — there is no separate `base_bits`/`max_bits`/`distortion_threshold`/`observer` API, and it does not react to an observer at runtime.

```python
q = TurboQuantProdAdaptive(d=128, b=3, seed=42)
# equivalent to TurboQuantProd(d=128, b=3, seed=42, use_adaptive_codebook=True)
```

---

## RaBitQQuantizer

```python
from veloxquant_mlx.quantizers.rabitq import RaBitQQuantizer
```

Randomised Hadamard transform + 1-bit sign packing with IVF clustering. Not wired into `KVCacheConfig` — see the [RaBitQ algorithm page](../algorithms/rabitq).

### Constructor

```python
RaBitQQuantizer(
    d: int,
    nlist: int = 64,
    nprobe: int = 8,
    rerank: int = 32,
    seed: int = 42,
)
```

Requires a one-time `fit(keys_calib: mx.array, max_samples: Optional[int] = None)` call to train IVF centroids before `encode()`/`decode()`.

### Methods

```python
def fit(self, keys_calib: Any, max_samples: Optional[int] = None) -> None: ...
def encode(self, keys: Any, **kwargs) -> EncodedVector: ...
def decode(self, ev: EncodedVector) -> Any: ...
```

`EncodedVector.indices` — packed sign bits, uint8, shape `[N, D//8]`.
`EncodedVector.norm` — `[centroid_id, Cx, L1]` per key, float32, shape `[N, 3]`.

---

## CommVQQuantizer

```python
from veloxquant_mlx.quantizers.comm_vq import CommVQQuantizer
```

RoPE-commutative residual VQ. Not wired into `KVCacheConfig` — see the [CommVQ algorithm page](../algorithms/commvq).

### Constructor

```python
CommVQQuantizer(
    d: int,               # must be even (required by RoPE)
    b: int = 8,
    n_codebooks: int = 4,
    seed: int = 42,
    rope_base: float = 10000.0,
    n_em_iters: int = 50,
)
```

Requires a one-time `fit(keys_calib: mx.array)` call (on pre-RoPE keys) before `encode()`/`decode()`.

### Methods

```python
def fit(self, keys_calib: Any) -> None: ...
def encode(self, x: Any, positions: Optional[Any] = None) -> EncodedVector: ...
def decode(self, ev: EncodedVector) -> Any: ...
```

`encode` expects pre-RoPE keys; `positions` (defaults to `0..N-1` if omitted) is stored in the `EncodedVector.norm` field so `decode` can apply RoPE at reconstruction time.

---

## PolarQuantizer

```python
from veloxquant_mlx.quantizers.polarquant import PolarQuantizer
```

Recursive polar coordinate decomposition.

### Constructor

```python
PolarQuantizer(
    d: int,
    b: int = 2,
    m: int = 128,
    seed: int = 42,
    n_levels: int = DEFAULT_POLAR_LEVELS,
    store: Optional[ArtifactStore] = None,
    use_hadamard: bool = False,
)
```

---

## QJLQuantizer

```python
from veloxquant_mlx.quantizers.qjl import QJLQuantizer
```

Johnson-Lindenstrauss 1-bit sign sketch.

### Constructor

```python
QJLQuantizer(
    d: int,
    m: int = 128,
    seed: int = 42,
    b: int = 1,
    store: Optional[ArtifactStore] = None,
)
```

---

## CompositeQuantizer

```python
from veloxquant_mlx.quantizers.composite import CompositeQuantizer
```

**Not** a residual chain. Routes outlier and inlier channels of the same vector to two different quantizers — the outlier channels (by index) go to one quantizer, the rest go to another.

### Constructor

```python
CompositeQuantizer(
    outlier_quantizer: Quantizer,
    inlier_quantizer: Quantizer,
    outlier_idx: np.ndarray,
    total_dim: int,
)
```

```python
import numpy as np
from veloxquant_mlx.quantizers.composite import CompositeQuantizer
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ
from veloxquant_mlx.quantizers.qjl import QJLQuantizer

total_dim = 128
outlier_idx = np.array([3, 17, 42, 88])

q = CompositeQuantizer(
    outlier_quantizer=TurboQuantRVQ(d=len(outlier_idx), b=4, seed=42),
    inlier_quantizer=QJLQuantizer(d=total_dim - len(outlier_idx), m=64, seed=42),
    outlier_idx=outlier_idx,
    total_dim=total_dim,
)
encoded = q.encode(x)  # x: (batch, total_dim)
decoded = q.decode(encoded)
```

---

## See also

- [Algorithm pages](../algorithms/overview)
- [Cache API](../api/cache)
- [Core API](../api/core-api)
