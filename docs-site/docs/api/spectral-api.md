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

Eigenvector-rotated quantizer with separate signal/noise codebooks and an optional QJL error-correction sketch on the signal residual (Algorithm 1 of the SpectralQuant paper, "3% Is All You Need: Breaking TurboQuant's Compression Limit via Spectral Structure").

### Constructor

```python
SpectralQuantizer(
    d: int,
    b_signal: int = 3,
    b_noise: int = 3,
    rotation: np.ndarray | None = None,
    d_s: int = 4,
    apply_qjl: bool = False,
    jl_dim: int | None = None,
    seed: int = 42,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `d` | `int` | Required | Head dimension |
| `b_signal` | `int` | `3` | Bit-width for signal dimensions (paper default) |
| `b_noise` | `int` | `3` | Bit-width for noise dimensions (paper default) |
| `rotation` | `np.ndarray \| None` | `None` | Eigenvector matrix `U` from `eigh(Σ)` (or its transpose), shape `(d, d)`, float32. If `None`, falls back to a random orthogonal rotation (degrades to TurboQuant-like behavior) |
| `d_s` | `int` | `4` | Number of signal dimensions (typically `ceil(participation_ratio)`), clamped to `[1, d]` |
| `apply_qjl` | `bool` | `False` | If `True`, store a QJL sign sketch of the signal quantization residual for error correction. The paper's primary config (`SQ_noQJL_v3`) uses `False` |
| `jl_dim` | `int \| None` | `None` | JL sketch dimension `m`; only used when `apply_qjl=True`. Defaults to `d_s` |
| `seed` | `int` | `42` | Random seed for the fallback random rotation and the JL matrix |

There is no `signal_bits`/`noise_bits`/`use_water_filling` constructor API, and rotation is passed as a plain `np.ndarray` — there is no `SpectralRotation` dataclass anywhere in the codebase.

### Methods

```python
def encode(self, x: Any) -> EncodedVector: ...
def decode(self, ev: EncodedVector) -> Any: ...
def estimate_inner_product(self, q: Any, ev: EncodedVector) -> Any: ...
def compression_ratio(self) -> float: ...
```

**`encode(x)`** — `x` has shape `(batch, d)` (or `(d,)`, which is broadcast to `(1, d)`), fp16 or fp32. Applies the spectral rotation `h̃ = U^T h`, splits into signal dims `h̃[:d_s]` and noise dims `h̃[d_s:]`, quantizes each with its own Gaussian codebook after per-vector abs-max scaling, and (if `apply_qjl=True`) computes a QJL sign sketch of the signal quantization residual. Returns an `EncodedVector` with `indices` (uint8, all `d` dims concatenated), `norm` (signal scale per vector), `final_radius` (noise scale per vector), and optionally `signs`/`residual_norm` when QJL is enabled.

**`decode(ev)`** — Reconstructs a `(batch, d)` fp16 array: dequantizes signal and noise dims separately (applying the QJL residual correction to the signal dims if present), concatenates, and applies the inverse rotation `ĥ = U · h̃̂`.

**`estimate_inner_product(q, ev)`** — Estimates `⟨q, k⟩` for all encoded keys directly in the rotated basis (`⟨q, k⟩ = ⟨q̃, k̃⟩`), without fully decoding `k`, adding the QJL correction term when available.

**`compression_ratio()`** — Returns fp16 bits (`16 * d`) divided by the compressed bit budget (`d_s * b_signal + (d - d_s) * b_noise`, plus `m` JL sign bits and 16 residual-norm bits if `apply_qjl=True`). Per-vector fp16 scale fields (`norm`/`final_radius`) are treated as small fixed overhead and are **not** included in this count, matching the paper's Table 2 accounting. This is a different method from `RaBitQQuantizer.compression_ratio` in `veloxquant_mlx/quantizers/rabitq.py` — that one returns 16x for the 1-bit sign-packed bits alone (~6x once per-key metadata overhead is included), an unrelated 1-bit algorithm, not SpectralQuant's rotation/water-filling scheme.

---

## calibrate_spectral_rotation

```python
from veloxquant_mlx.spectral.calibrate import calibrate_spectral_rotation
```

```python
def calibrate_spectral_rotation(
    model: Any,
    calibration_tokens: Any,
    n_tokens: int = 512,
    model_name: str = "model",
    force_recompute: bool = False,
) -> dict[int, tuple]
```

Runs calibration tokens through an mlx-lm model, collecting per-layer key/value activations via a wrapped cache (`collect_kv_vectors_mlx`), then computes a PCA rotation (via SVD) for each layer's keys and values separately.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | mlx_lm model | Required | Loaded model instance |
| `calibration_tokens` | array-like | Required | Token IDs, shape `(seq_len,)` or `(1, seq_len)` |
| `n_tokens` | `int` | `512` | Maximum KV vectors collected per layer |
| `model_name` | `str` | `"model"` | Cache key used to save/load rotations from disk |
| `force_recompute` | `bool` | `False` | If `True`, ignore any cached rotations for `model_name` |

**Returns:** `dict[int, tuple]` mapping `layer_idx -> (key_U, val_U, key_eigenvalues, val_eigenvalues, key_d_s, val_d_s)`, where:
- `key_U` / `val_U`: `(d, d)` float32, columns are eigenvectors sorted by descending eigenvalue
- `key_eigenvalues` / `val_eigenvalues`: `(d,)` float64, descending
- `key_d_s` / `val_d_s`: `int`, `ceil(participation_ratio)` signal-dimension count for keys/values respectively

If fewer than 4 vectors were collected for a layer, that layer falls back to a random orthogonal rotation with uniform eigenvalues and a paper-default `d_s` (4 for keys, 50 for values).

There is no `list[SpectralRotation]` return type, no `SpectralRotation` dataclass, and no `tokenizer`, `num_samples`, `sequence_length`, or `device` parameters — calibration works from already-tokenized `calibration_tokens`, and both keys and values are calibrated together in the same call.

---

## calibrate_from_vectors

```python
from veloxquant_mlx.spectral.calibrate import calibrate_from_vectors
```

```python
def calibrate_from_vectors(
    key_vectors: dict[int, np.ndarray],
    val_vectors: dict[int, np.ndarray],
    model_name: str = "synthetic",
) -> dict[int, tuple]
```

Builds the same calibration result as `calibrate_spectral_rotation`, but from pre-collected KV arrays instead of running a model forward pass. `key_vectors` and `val_vectors` are dicts mapping `layer_idx -> (N, d)` float32 arrays (not a flat `list[mx.array]`). Returns the same `dict[int, tuple]` format described above, and also persists the result to disk under `model_name` via `save_rotations`.

---

## save_rotations / load_cached_rotations

```python
from veloxquant_mlx.spectral.calibrate import save_rotations, load_cached_rotations
```

```python
def save_rotations(model_name: str, rotations: dict[int, tuple]) -> None: ...
def load_cached_rotations(model_name: str) -> dict | None: ...
```

Persist rotation matrices and eigenvalues to a single on-disk cache file and reload them. The cache path is derived from `model_name` (sanitized to replace `/` and `\` with `_`) under `$VELOXQUANT_CACHE_DIR/spectral/<safe_name>/rotations.npz` (defaults to `~/.cache/veloxquant/spectral/<safe_name>/rotations.npz` if the env var isn't set) — uses NumPy's compressed `.npz` archive format, **not** individual `.npy` files, and takes a `model_name` string key rather than an arbitrary directory `path`.

```python
save_rotations("my-model", rotations)
rotations = load_cached_rotations("my-model")  # returns None if not cached
```

---

## compute_participation_ratio

```python
from veloxquant_mlx.spectral.participation_ratio import compute_participation_ratio
```

```python
def compute_participation_ratio(vectors: np.ndarray) -> float
```

Takes raw vectors — shape `(n_samples, d)`, fp32 or fp16 — not a pre-computed eigenvalue array. Internally mean-centers the vectors, computes the sample covariance, and eigendecomposes it via `np.linalg.eigvalsh`, then applies:

```
d_eff = (Σ λᵢ)² / Σ λᵢ²
```

Returns a float in `[1, d]`: equal to `d` when eigenvalues are all equal (variance fully spread across dimensions), equal to `1` when a single dimension dominates. (This is the reciprocal scaling of a normalized `[1/d, 1]` ratio — the actual return value is an effective dimension count, not a fraction.)

---

## compute_spectral_gap

```python
from veloxquant_mlx.spectral.participation_ratio import compute_spectral_gap
```

```python
def compute_spectral_gap(vectors: np.ndarray) -> tuple[int, np.ndarray]
```

Also takes raw vectors (`(n_samples, d)`), not eigenvalues, and does **not** find "the largest drop in consecutive eigenvalues." It computes the covariance eigenspectrum (descending), calls `compute_participation_ratio` on the same vectors, and rounds that ratio to the nearest integer to get `d_eff`. Returns `(d_eff, eigenvalues)` — an `int` cutoff paired with the full descending eigenvalue array, not a bare `int`.

---

## water_fill_bits

```python
from veloxquant_mlx.spectral.bit_allocator import water_fill_bits
```

```python
def water_fill_bits(
    eigenvalues: np.ndarray,
    total_bit_budget: int,
    min_bits: int = 1,
    max_bits: int = 8,
) -> np.ndarray
```

Water-filling bit allocation: iteratively assigns bits proportionally to each dimension's eigenvalue (more signal → more bits), redistributing budget away from dimensions that hit `max_bits` until the allocation converges, then reconciles any rounding remainder against the exact requested budget by greedily nudging individual dimensions.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `eigenvalues` | `np.ndarray` | Required | Per-dimension variance/eigenvalue, shape `(d,)`, non-negative |
| `total_bit_budget` | `int` | Required | **Total** bits to distribute across all `d` dimensions (not a per-dimension average) |
| `min_bits` | `int` | `1` | Minimum bits per dimension |
| `max_bits` | `int` | `8` | Maximum bits per dimension |

**Returns:** `np.ndarray` of shape `(d,)`, dtype `int32` — bits per dimension. (Not a `list[int]`, and the budget parameter is a total bit count, not a `target_avg_bits` float.)

```python
from veloxquant_mlx.spectral.bit_allocator import water_fill_bits

d = len(rotations[0][2])  # key_eigenvalues for layer 0
bits_per_dim = water_fill_bits(
    eigenvalues=rotations[0][2],  # key_eigenvalues
    total_bit_budget=3 * d,       # e.g. average of 3 bits/dim over d dims
)
print(bits_per_dim[:8])
```

---

## See also

- [SpectralQuant algorithm](../algorithms/spectral)
- [Calibration guide](../guides/calibration)
- [API — Cache](../api/cache)
