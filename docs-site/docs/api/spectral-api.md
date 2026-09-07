---
id: spectral-api
title: SpectralQuant API
sidebar_label: SpectralQuant
slug: /api/spectral-api
description: Python API reference for veloxquant_mlx.spectral, covering the SpectralQuantizer eigenvector-rotated quantizer, calibration functions for computing per-layer PCA rotations, and water-filling bit allocation helpers.
keywords: [spectral, spectralquant, SpectralQuantizer, "API reference", "python api", water-filling, PCA rotation]
---

# SpectralQuant API

`veloxquant_mlx.spectral`

---

## SpectralQuantizer

```python
from veloxquant_mlx.spectral.spectral_quant import SpectralQuantizer
```

Eigenvector-rotated quantizer with separate signal and noise codebooks.

### Constructor

```python
SpectralQuantizer(
    rotation: SpectralRotation,
    signal_bits: int = 4,
    noise_bits: int = 1,
    use_water_filling: bool = False,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `rotation` | `SpectralRotation` | Required | Per-layer rotation from calibration |
| `signal_bits` | `int` | `4` | Bits for high-variance dimensions |
| `noise_bits` | `int` | `1` | Bits for low-variance dimensions |
| `use_water_filling` | `bool` | `False` | Use per-dim water-filling allocation |

### Methods

```python
def encode(self, keys: mx.array) -> EncodedVector: ...
def decode(self, encoded: EncodedVector) -> mx.array: ...
```

---

## calibrate_spectral_rotation

```python
from veloxquant_mlx.spectral.calibrate import calibrate_spectral_rotation
```

```python
def calibrate_spectral_rotation(
    model,
    tokenizer,
    num_samples: int = 64,
    sequence_length: int = 1024,
    device: str = "gpu",
) -> list[SpectralRotation]
```

Collects key activations and computes the PCA rotation matrix per layer via SVD.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | mlx_lm model | Required | Loaded model |
| `tokenizer` | tokenizer | Required | Loaded tokenizer |
| `num_samples` | `int` | `64` | Calibration sequences |
| `sequence_length` | `int` | `1024` | Tokens per sequence |
| `device` | `str` | `"gpu"` | `"gpu"` or `"cpu"` |

**Returns:** `list[SpectralRotation]` — one per transformer layer.

`SpectralRotation` fields:
- `rotation_matrix: mx.array` — shape `[head_dim, head_dim]`
- `eigenvalues: mx.array` — shape `[head_dim]`, sorted descending
- `head_dim: int`
- `layer_name: str`

---

## calibrate_from_vectors

```python
from veloxquant_mlx.spectral.calibrate import calibrate_from_vectors
```

```python
def calibrate_from_vectors(
    key_vectors: list[mx.array],
) -> list[SpectralRotation]
```

Compute rotation from pre-collected key vectors instead of running a forward pass. Useful when key activations are already available.

---

## save_rotations / load_cached_rotations

```python
from veloxquant_mlx.spectral.calibrate import save_rotations, load_cached_rotations
```

```python
def save_rotations(rotations: list[SpectralRotation], path: str) -> None: ...
def load_cached_rotations(path: str) -> list[SpectralRotation]: ...
```

Persist rotation matrices to disk and reload them. Uses NumPy `.npy` format.

```python
save_rotations(rotations, "./artifacts/spectral/")
rotations = load_cached_rotations("./artifacts/spectral/")
```

---

## compute_participation_ratio

```python
from veloxquant_mlx.spectral.participation_ratio import compute_participation_ratio
```

```python
def compute_participation_ratio(eigenvalues: mx.array) -> float
```

Measures how many effective dimensions concentrate the variance:

```
PR = (Σ λᵢ)² / (d · Σ λᵢ²)
```

Returns a value in `[1/d, 1.0]`. Close to `1/d` means energy concentrated in few dims; close to `1.0` means uniform distribution.

---

## compute_spectral_gap

```python
from veloxquant_mlx.spectral.participation_ratio import compute_spectral_gap
```

```python
def compute_spectral_gap(eigenvalues: mx.array) -> int
```

Finds the index of the largest drop in consecutive eigenvalues — the boundary between "signal" and "noise" subspaces.

---

## water_fill_bits

```python
from veloxquant_mlx.spectral.bit_allocator import water_fill_bits
```

```python
def water_fill_bits(
    eigenvalues: mx.array,
    target_avg_bits: float,
    min_bits: int = 1,
    max_bits: int = 8,
) -> list[int]
```

Water-filling bit allocation: assigns more bits to dimensions with higher eigenvalues.

**Returns:** `list[int]` of length `head_dim` — bits per dimension.

```python
from veloxquant_mlx.spectral.bit_allocator import water_fill_bits

bits_per_dim = water_fill_bits(
    eigenvalues=rotations[0].eigenvalues,
    target_avg_bits=3.0,
)
print(bits_per_dim[:8])  # e.g. [8, 8, 6, 4, 2, 1, 1, 1]
```

---

## See also

- [SpectralQuant algorithm](../algorithms/spectral)
- [Calibration guide](../guides/calibration)
- [API — Cache](../api/cache)
