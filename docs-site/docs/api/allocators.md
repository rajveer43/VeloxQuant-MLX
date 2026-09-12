---
id: allocators
title: Allocators API
sidebar_label: Allocators
slug: /api/allocators
description: Python API reference for veloxquant_mlx.allocators, covering RateQuant's calibrate_layer_sensitivities/allocate_bits_ratequant/fit_distortion_curve functions, VecInfer's smooth-calibration/Hadamard/product-VQ primitives, and KVTC's DP-optimal per-component bit allocator.
keywords: [allocators, RateQuant, VecInfer, KVTC, "API reference", "python api", bit allocation, calibration, codebook]
---

# Allocators API

`veloxquant_mlx.allocators`

The allocators module provides calibration and bit-allocation functions for RateQuant, VecInfer, and KVTC.

---

## RateQuant allocator

`veloxquant_mlx.allocators.ratequant`

Per-layer bit allocation adapted from RateQuant (arxiv:2605.06675). `calibrate_layer_sensitivities` runs one forward pass over calibration prompts and returns a per-layer activation-norm sensitivity proxy; `allocate_bits_ratequant` turns those weights into integer bit-widths via a closed-form reverse-waterfilling rule (the paper's Theorem 2). `fit_distortion_curve` is an optional helper for estimating the decay constant `beta` from synthetic data — most users can skip it and pass the paper-reported default (`beta=3.5`).

### `calibrate_layer_sensitivities`

```python
def calibrate_layer_sensitivities(
    model,
    tokenizer,
    prompts: list | None = None,
    seq_len: int = 256,
    verbose: bool = False,
) -> list[float]
```

Runs a calibration forward pass and returns per-layer sensitivity. Internally this swaps in a probe KV cache that accumulates the mean-squared per-token key L2 norm for each layer; higher norm implies larger absolute reconstruction error at a given bit-width, so it is used as an activation-based sensitivity proxy (the paper's Table 5 "activation-based" variant — the gradient-based proxy is not used because it would require backprop through `mlx_lm.generate`).

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | mlx_lm model | Required | Loaded model (e.g. from `mlx_lm.load()`) |
| `tokenizer` | tokenizer | Required | Matching tokenizer |
| `prompts` | `list \| None` | `None` | Calibration strings. Defaults to 8 built-in general-domain prompts (history, science, CS) |
| `seq_len` | `int` | `256` | Max tokens per prompt (truncated if longer) |
| `verbose` | `bool` | `False` | Print per-sequence progress |

**Returns:** `list[float]` of length `n_attention_layers`, each strictly greater than 0. Higher values indicate layers whose key cache is more error-prone at fixed bit-width and should receive more bits.

---

### `allocate_bits_ratequant`

```python
def allocate_bits_ratequant(
    sensitivities,
    target_avg_bits: float,
    beta: float = 3.5,
    bit_choices: tuple = (1, 2, 3),
) -> list[int]
```

Allocates per-layer bit-widths via RateQuant's Theorem 2 closed-form reverse waterfilling. The continuous solution is:

```
b_i = b̄ + (ln w_i − ln_w_bar) / ln(β)
```

which is rounded to the nearest member of `bit_choices` per layer, then re-balanced with greedy +1/−1-step adjustments (always moving to the actual next/previous value in `bit_choices`, not a bare increment) so the integer total exactly matches `round(target_avg_bits * N)`.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sensitivities` | iterable of `float` | Required | Per-layer sensitivity weights `w_i > 0`, typically from `calibrate_layer_sensitivities` |
| `target_avg_bits` | `float` | Required | Desired mean bits/dim across layers; may be fractional (e.g. `1.5`) |
| `beta` | `float` | `3.5` | Distortion-rate decay constant for the underlying quantizer. Paper-reported values: 3.5 for TurboQuant, 5.0 for KIVI/QuaRot. A mismatched β can invert allocation ordering |
| `bit_choices` | `tuple` | `(1, 2, 3)` | Allowed integer bit-widths |

**Returns:** `list[int]` of length `len(sensitivities)`, one integer bit-width per layer — pass directly as `KVCacheConfig.bit_width_inlier`.

**Raises:** `ValueError` if any sensitivity weight is `<= 0`, or if `bit_choices` is empty.

```python
from veloxquant_mlx.allocators import (
    allocate_bits_ratequant,
    calibrate_layer_sensitivities,
)

weights = calibrate_layer_sensitivities(model, tokenizer)
alloc = allocate_bits_ratequant(weights, target_avg_bits=1.5)
```

---

### `fit_distortion_curve`

```python
def fit_distortion_curve(
    head_dim: int,
    bit_choices: tuple = (1, 2, 3),
    seed: int = 0,
    n_samples: int = 64,
) -> tuple[float, float]
```

Optional helper that fits `D(b) = α·β^(-b)` on synthetic unit-norm Gaussian keys, by encoding/decoding them through `TurboQuantRVQ` at each bit-width in `bit_choices` and fitting the decay via log-linear least squares. For `TurboQuantRVQ` at `d=128` this recovers β ≈ 3.5, matching the paper. For production use it is usually simpler to skip this fit and pass `beta=3.5` directly to `allocate_bits_ratequant`.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `head_dim` | `int` | Required | Key dimension to synthesize calibration vectors for |
| `bit_choices` | `tuple` | `(1, 2, 3)` | Bit-widths to fit the curve on |
| `seed` | `int` | `0` | RNG seed for synthetic data and the quantizer |
| `n_samples` | `int` | `64` | Number of synthetic calibration vectors |

**Returns:** `tuple[float, float]` — `(alpha, beta)`.

---

## VecInfer allocator

`veloxquant_mlx.allocators.vecinfer`

Algorithmic primitives for VecInfer (arxiv:2510.06175) KV cache compression on MLX: smooth-factor calibration, an orthonormal Walsh-Hadamard transform, and product vector quantization. The paper's fused CUDA kernel is not portable to Metal, so this module provides only the math; standard `mlx_lm` SDPA runs on the dequantized result. Pipeline for the key cache:

1. Calibrate a per-(head, channel) smooth factor `λ` offline from a representative key sample.
2. Build a Walsh-Hadamard matrix `H` of size `head_dim × head_dim`.
3. Train a product-VQ codebook on smooth- and Hadamard-transformed keys.
4. At inference: `K̃ = (K / λ) @ H`, quantized via the codebook; queries get the inverse-facing transform `q̃ = (q · λ) @ H` so that `q̃ @ K̃.T == q @ K.T`.

### `calibrate_smooth_factors`

```python
def calibrate_smooth_factors(keys_calib: mx.array, eps: float = 1e-4) -> mx.array
```

Computes the per-(head, channel) smooth scaling factor `λᵢ = sqrt(max_t |K[t, ..., i]|)` from a sample of calibration keys.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `keys_calib` | `mx.array` | Required | Calibration keys, shape `[n_tokens, n_heads, head_dim]` or `[n_tokens, head_dim]` (single-head) |
| `eps` | `float` | `1e-4` | Floor for the per-channel max, to avoid divide-by-zero |

**Returns:** `mx.array` shaped to match the input head layout — `[n_heads, head_dim]` or `[head_dim]`.

**Raises:** `ValueError` if `keys_calib` is not 2D or 3D.

---

### `walsh_hadamard_matrix`

```python
def walsh_hadamard_matrix(d: int, dtype=mx.float32) -> mx.array
```

Constructs an orthonormal Walsh-Hadamard matrix via the recursive form `H_1 = [[1]]`, `H_{2k} = (1/√2) · [[H_k, H_k], [H_k, -H_k]]`.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `d` | `int` | Required | Output dimension; must be a power of 2 |
| `dtype` | mx dtype | `mx.float32` | Dtype of the returned matrix |

**Returns:** `mx.array` of shape `[d, d]` satisfying `H @ H.T == I`.

**Raises:** `ValueError` if `d` is not a power of 2.

---

### `apply_dual_transform_keys`

```python
def apply_dual_transform_keys(K: mx.array, smooth: mx.array, H: mx.array) -> mx.array
```

Applies `K̃ = (K / λ) @ H` (smooth, then Hadamard rotation). `smooth` may be `[head_dim]` (broadcast across heads) or `[n_heads, head_dim]` (per-head, matched against `K`'s `-3` axis); if the head counts don't line up (e.g. GQA where `smooth` was calibrated on Q heads), `smooth` is averaged across the head axis instead of erroring.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `K` | `mx.array` | Keys, shape `[..., head_dim]` |
| `smooth` | `mx.array` | `[head_dim]` or `[n_heads, head_dim]`, from `calibrate_smooth_factors` |
| `H` | `mx.array` | Walsh-Hadamard matrix `[head_dim, head_dim]` |

**Returns:** Transformed keys, same shape as `K`.

**Raises:** `ValueError` if `smooth` is not 1D/2D, or cannot be broadcast against `K`.

---

### `apply_dual_transform_queries`

```python
def apply_dual_transform_queries(q: mx.array, smooth: mx.array, H: mx.array) -> mx.array
```

Applies `q̃ = (q · λ) @ H` so that `q̃ @ K̃.T == q @ K.T`. Same broadcasting rules as `apply_dual_transform_keys`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `q` | `mx.array` | Queries, shape `[..., head_dim]` |
| `smooth` | `mx.array` | Same convention as `apply_dual_transform_keys` |
| `H` | `mx.array` | Walsh-Hadamard matrix `[head_dim, head_dim]` |

**Returns:** Transformed queries, same shape as `q`.

**Raises:** `ValueError` if `smooth` is not 1D/2D, or cannot be broadcast against `q`.

---

### `train_codebook`

```python
def train_codebook(x: mx.array, n_centroids: int, max_iter: int = 30, seed: int = 42) -> mx.array
```

Trains a VQ codebook on flat sub-vector samples using a pure-numpy Lloyd's k-means (chunked assignment, empty-cluster reseeding, early-stopping on inertia convergence).

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x` | `mx.array` | Required | Flat `[n_samples, sub_dim]` training sub-vectors |
| `n_centroids` | `int` | Required | Codebook size, typically `2**b` for `b`-bit codes |
| `max_iter` | `int` | `30` | K-means iteration cap |
| `seed` | `int` | `42` | RNG seed |

**Returns:** `mx.array` of shape `[n_centroids, sub_dim]`.

**Raises:** `ValueError` if `x` is not 2D.

---

### `quantize_vq`

```python
def quantize_vq(x: mx.array, codebook: mx.array, sub_dim: int) -> mx.array
```

Encodes `x` as nearest-centroid indices under a product-VQ scheme (chunked argmin to bound memory for large codebooks).

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `x` | `mx.array` | `[..., D]` where `D` is divisible by `sub_dim` |
| `codebook` | `mx.array` | `[n_centroids, sub_dim]` |
| `sub_dim` | `int` | Sub-vector dimension |

**Returns:** `mx.array` of shape `[..., D // sub_dim]`, `int32` indices into `codebook`.

**Raises:** `ValueError` if `D` is not divisible by `sub_dim`.

---

### `dequantize_vq`

```python
def dequantize_vq(indices: mx.array, codebook: mx.array) -> mx.array
```

Reconstructs vectors from codebook indices via a gather.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `indices` | `mx.array` | `[..., n_sub]` `int32` codebook indices |
| `codebook` | `mx.array` | `[n_centroids, sub_dim]` |

**Returns:** `mx.array` of shape `[..., n_sub * sub_dim]`.

---

### `compute_query_lut`

```python
def compute_query_lut(q_tilde: mx.array, codebook: mx.array, sub_dim: int) -> mx.array
```

Precomputes `q_sub @ codebook.T` so that attention scores can be evaluated via lookup instead of dequantizing keys first. The attention score for a token with codebook indices `idx[n_sub]` is `lut[token, range(n_sub), idx].sum()`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `q_tilde` | `mx.array` | Transformed query `[..., D]` |
| `codebook` | `mx.array` | `[n_centroids, sub_dim]` |
| `sub_dim` | `int` | Must match codebook width |

**Returns:** `mx.array` of shape `[..., n_sub, n_centroids]`.

**Raises:** `ValueError` if `D` is not divisible by `sub_dim`.

```python
from veloxquant_mlx.allocators.vecinfer import (
    calibrate_smooth_factors,
    walsh_hadamard_matrix,
    apply_dual_transform_keys,
    train_codebook,
    quantize_vq,
)

smooth = calibrate_smooth_factors(keys_calib)
H = walsh_hadamard_matrix(head_dim)
keys_transformed = apply_dual_transform_keys(keys_calib, smooth, H)
codebook = train_codebook(keys_transformed.reshape(-1, sub_dim), n_centroids=256)
codes = quantize_vq(keys_transformed, codebook, sub_dim=sub_dim)
```

---

## KVTC allocator

`veloxquant_mlx.allocators.kvtc_dp`

DP-optimal per-component bit allocator adapted from "KV Cache Transform Coding for Compact Storage in LLM Inference" (arXiv:2511.01815, ICLR 2026). Unlike `allocate_bits_ratequant` (closed-form, continuous, per-*layer*), `dp_allocate_bits` computes an exact, discrete, per-*component* allocation — including assigning exactly **0** bits to a component (dropping it) while another gets more than a "high" tier. The distortion model it minimizes is the repo's own analytic Gaussian quantization-distortion proxy (the same curve `fit_distortion_curve` estimates), not the source paper's rate-distortion model fit on real activation statistics — see the module docstring for the full scope statement.

### `dp_allocate_bits`

```python
def dp_allocate_bits(
    variances: np.ndarray,
    total_bit_budget: int,
    bit_choices: tuple[int, ...] = DEFAULT_BIT_CHOICES,
    beta: float = DEFAULT_BETA,
) -> np.ndarray
```

Finds the DP-optimal integer bit-width per component under a total bit budget. Minimizes `sum_i D(variances[i], bits[i])` subject to `sum_i bits[i] <= total_bit_budget` and `bits[i]` drawn from `bit_choices`, using dynamic programming over (component index, cumulative budget spent) — `O(n_components * total_bit_budget * len(bit_choices))`, exact. The per-component distortion proxy is `D(v, b) = v * beta ** (-b)` for `b > 0`, and `D(v, 0) = v` (dropping a component keeps its full variance as error).

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `variances` | `np.ndarray` | Required | Per-component variance (e.g. squared singular values from a local PCA), shape `[n_components]`, non-negative |
| `total_bit_budget` | `int` | Required | Total integer bits available across all components; must be `>= 0` |
| `bit_choices` | `tuple[int, ...]` | `DEFAULT_BIT_CHOICES` = `(0, 1, 2, 3, 4, 6, 8)` | Allowed integer bit-widths per component. Must include `0` for a component to be droppable |
| `beta` | `float` | `DEFAULT_BETA` = `3.5` | Distortion decay constant for `D(v, b) = v * beta ** (-b)` — the same constant used by `ratequant.py`'s curve |

**Returns:** `np.ndarray[int]` of shape `[n_components]`, values drawn from `bit_choices`, summing to at most `total_bit_budget`.

**Raises:** `ValueError` if `total_bit_budget < 0`, any variance is negative, `variances` is empty, or `bit_choices` is empty or contains a negative value.

```python
from veloxquant_mlx.allocators.kvtc_dp import dp_allocate_bits

bits = dp_allocate_bits(variances, total_bit_budget=32)
```

---

## See also

- [RateQuant algorithm](../algorithms/ratequant)
- [VecInfer algorithm](../algorithms/vecinfer)
- [Calibration guide](../guides/calibration)
- [Mixed-precision guide](../guides/mixed-precision)
- [Memory (Block Pool) API](./memory-api) — bit-allocation here is distinct from the KV-cache *memory*-block allocator in `veloxquant_mlx.memory`
