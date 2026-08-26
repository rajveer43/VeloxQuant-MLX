---
id: allocators
title: Allocators API
sidebar_label: Allocators
slug: /api/allocators
---

# Allocators API

`veloxquant_mlx.allocators`

The allocators module provides calibration and bit-allocation functions for VecInfer and RateQuant.

---

## RateQuant allocator

`veloxquant_mlx.allocators.ratequant`

### `calibrate_layer_sensitivities`

```python
def calibrate_layer_sensitivities(
    model,
    tokenizer,
    num_samples: int = 32,
    sequence_length: int = 512,
    noise_scale: float = 0.1,
    device: str = "gpu",
) -> dict[str, SensitivityResult]
```

Probes each transformer layer's sensitivity to quantization noise by perturbing KV cache entries and measuring output change.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | mlx_lm model | Required | Loaded model |
| `tokenizer` | tokenizer | Required | Loaded tokenizer |
| `num_samples` | `int` | `32` | Number of calibration sequences |
| `sequence_length` | `int` | `512` | Tokens per sequence |
| `noise_scale` | `float` | `0.1` | Magnitude of perturbation (fraction of key std) |
| `device` | `str` | `"gpu"` | `"gpu"` or `"cpu"` |

**Returns:** `dict[str, SensitivityResult]` — keyed by layer name (`"layer_0"` ... `"layer_N"`).

`SensitivityResult` fields:
- `mean_sensitivity: float` — average output change per unit noise
- `std_sensitivity: float` — variance across samples
- `layer_name: str`

---

### `fit_distortion_curve`

```python
def fit_distortion_curve(
    sensitivities: dict[str, SensitivityResult],
    bit_rates: list[float] = [1.0, 1.5, 2.0, 3.0, 4.0],
) -> dict[str, DistortionCurve]
```

Fits a parametric rate-distortion model `D(r) = α · exp(-β · r)` per layer.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sensitivities` | `dict` | Required | Output of `calibrate_layer_sensitivities` |
| `bit_rates` | `list[float]` | `[1.0, 1.5, 2.0, 3.0, 4.0]` | Bit rates to evaluate on the curve |

**Returns:** `dict[str, DistortionCurve]` — parametric distortion models per layer.

---

### `allocate_bits_ratequant`

```python
def allocate_bits_ratequant(
    distortion_curves: dict[str, DistortionCurve],
    target_bits: float = 2.0,
    min_bits: int = 1,
    max_bits: int = 4,
) -> dict[str, int]
```

Solves the reverse-waterfilling optimisation: allocate bits across layers to minimise total distortion at `target_bits` average.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `distortion_curves` | `dict` | Required | Output of `fit_distortion_curve` |
| `target_bits` | `float` | `2.0` | Target average bit rate across all layers |
| `min_bits` | `int` | `1` | Floor for any single layer |
| `max_bits` | `int` | `4` | Ceiling for any single layer |

**Returns:** `dict[str, int]` — per-layer integer bit assignment. Example: `{"layer_0": 2, "layer_1": 3, ..., "layer_31": 1}`.

```python
from veloxquant_mlx.allocators.ratequant import (
    calibrate_layer_sensitivities,
    fit_distortion_curve,
    allocate_bits_ratequant,
)

sensitivities = calibrate_layer_sensitivities(model, tokenizer, num_samples=32)
curves = fit_distortion_curve(sensitivities)
allocation = allocate_bits_ratequant(curves, target_bits=2.0)
```

---

## VecInfer allocator

`veloxquant_mlx.allocators.vecinfer`

### `calibrate_smooth_factors`

```python
def calibrate_smooth_factors(
    model,
    tokenizer,
    num_samples: int = 64,
    sequence_length: int = 256,
) -> ndarray
```

Computes per-channel smooth scaling factors `λᵢ = √max|Kᵢ|` by observing key activations across calibration samples.

**Returns:** `ndarray` of shape `[num_layers, head_dim]` — scale factor per layer per channel.

---

### `train_codebook`

```python
def train_codebook(
    model,
    tokenizer,
    smooth_factors: ndarray,
    num_samples: int = 128,
    num_centroids: int = 256,
    num_subspaces: int = 8,
    sequence_length: int = 256,
    max_iter: int = 100,
) -> ndarray
```

Trains a product-VQ codebook via K-means on collected key activations.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `smooth_factors` | `ndarray` | Required | From `calibrate_smooth_factors` |
| `num_samples` | `int` | `128` | Number of calibration sequences |
| `num_centroids` | `int` | `256` | Centroids per subspace (2^k for k-bit codes) |
| `num_subspaces` | `int` | `8` | Number of product VQ partitions |
| `max_iter` | `int` | `100` | K-means iterations |

**Returns:** `ndarray` of shape `[num_layers, num_subspaces, num_centroids, subspace_dim]`.

---

### Other VecInfer utilities

```python
from veloxquant_mlx.allocators.vecinfer import (
    walsh_hadamard_matrix,  # (d,) -> ndarray [d, d] WHT matrix
    apply_dual_transform_keys,  # smooth + Hadamard rotation for keys
    apply_dual_transform_queries,  # inverse dual transform for queries
    quantize_vq,  # product VQ encoding
    dequantize_vq,  # codebook lookup decoding
    compute_query_lut,  # precompute query-codebook LUT for MIPS
)
```

---

## See also

- [RateQuant algorithm](../algorithms/ratequant)
- [VecInfer algorithm](../algorithms/vecinfer)
- [Calibration guide](../guides/calibration)
- [Mixed-precision guide](../guides/mixed-precision)
- [Memory (Block Pool) API](./memory-api) — bit-allocation here is distinct from the KV-cache *memory*-block allocator in `veloxquant_mlx.memory`
