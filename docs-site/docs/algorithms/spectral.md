---
id: spectral
title: SpectralQuant
sidebar_label: SpectralQuant
slug: /algorithms/spectral
---

# SpectralQuant

SpectralQuant uses **eigenvector rotation** to align the key distribution with the quantizer's assumptions. By rotating keys into the PCA basis, it separates high-variance "signal" dimensions from low-variance "noise" dimensions and applies separate codebooks to each group — achieving high fidelity at long context lengths.

:::warning[Apple Silicon required]
Requires macOS M-series for Metal kernels and efficient MLX SVD.
:::

## How it works

1. **SVD rotation calibration** — `calibrate_spectral_rotation()` collects key/value activations from calibration tokens and computes eigenvectors via SVD, per layer. These form the rotation matrices `U` (key) and `U` (value).

2. **Participation ratio** — `compute_participation_ratio()` measures the effective dimensionality `d_eff = (Σλᵢ)² / Σλᵢ²` of the collected vectors. A high participation ratio (close to `head_dim`) means keys are uniform; a low ratio means energy is concentrated in a few directions. This is what sizes `spectral_key_d_eff`/`spectral_val_d_eff` below.

3. **Signal/noise split** — The top `d_s` rotated dimensions (by eigenvalue) are treated as "signal"; the rest are "noise". Both groups are quantized with the same bit-width today (`bit_width_inlier`), so the split currently controls *which dimensions* get the (optional) QJL treatment, not a differing bit-width per group.

4. **Rotation + quantize** — At inference, each key/value is rotated by its layer's `U`, then encoded via `SpectralQuantizer` (optionally applying QJL sign-sketching on the signal dimensions when `spectral_apply_qjl=True`).

5. **Calibration injection** — `SpectralQuantKVCache` is constructed with a *random* orthogonal rotation by default. Call `cache.calibrate(rotation_entry)` per layer (with the tuple returned by `calibrate_spectral_rotation()`) to inject the real, data-derived rotation before running inference.

## Key properties

| Property | Value |
|---|---|
| Calibration | SVD rotation over calibration tokens; ~15s per the source paper (§3.1) |
| Key bits | `bit_width_inlier` (shared with values; no separate signal/noise bit-width yet) |
| Signal dimension (keys) | `spectral_key_d_eff` (default `4`) |
| Signal dimension (values) | `spectral_val_d_eff` (default `50`) |
| QJL on signal dims | `spectral_apply_qjl` (default `True`) |
| Best for | Long context (8k+), high-fidelity inference |

## Calibration

```python
import mlx_lm
from veloxquant_mlx.spectral.calibrate import (
    calibrate_spectral_rotation,
    save_rotations,
)

model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-7B-Instruct-4bit")
calibration_tokens = tokenizer.encode("Some representative calibration text...")

# Calibrate rotation matrices per layer
rotations = calibrate_spectral_rotation(
    model,
    calibration_tokens,
    n_tokens=512,
    model_name="qwen2.5-7b",
)

# Save for reuse — writes to the on-disk rotation cache keyed by model_name
save_rotations(model_name="qwen2.5-7b", rotations=rotations)
print(f"Calibrated rotations for {len(rotations)} layers")
```

## Inspect spectral properties

```python
from veloxquant_mlx.spectral.participation_ratio import (
    compute_participation_ratio,
    compute_spectral_gap,
)

# vectors: [n_samples, head_dim] collected key activations for a layer
pr = compute_participation_ratio(vectors)
print(f"Effective dimensionality: {pr:.2f} / {vectors.shape[-1]}")
```

## Inference

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder
from veloxquant_mlx.spectral.calibrate import load_cached_rotations

config = KVCacheConfig(
    method="spectral",
    bit_width_inlier=3,
    spectral_key_d_eff=4,  # signal dimensions for keys
    spectral_val_d_eff=50,  # signal dimensions for values
    spectral_apply_qjl=True,  # apply QJL sign-sketch on signal dims
)
cache = KVCacheBuilder.build(model, config)

# Inject the real, calibrated rotation (falls back to a random orthogonal
# rotation otherwise — correct interface, uncalibrated quality)
rotations = load_cached_rotations("qwen2.5-7b")
if rotations is not None:
    cache.calibrate(rotations[0])  # per-layer; repeat per transformer layer

response = mlx_lm.generate(
    model,
    tokenizer,
    prompt="Write a comprehensive essay on the history of mathematics.",
    max_tokens=2000,
    kv_cache=cache,
)
```

## Water-filling bit allocation

`water_fill_bits` allocates a total bit budget across dimensions proportionally to eigenvalue (signal strength). It is a standalone utility in `veloxquant_mlx.spectral.bit_allocator` — it is not currently wired into `SpectralQuantKVCache`'s per-dimension quantization automatically.

```python
from veloxquant_mlx.spectral.bit_allocator import water_fill_bits

# eigenvalues: per-dimension variance from calibrate_spectral_rotation()
per_dim_bits = water_fill_bits(
    eigenvalues,
    total_bit_budget=3 * len(eigenvalues),  # e.g. avg 3 bits/dim
    min_bits=1,
    max_bits=8,
)
print(per_dim_bits)  # e.g. [8, 8, 6, 4, 2, 1, 1, ...]
```

## Configuration reference

`KVCacheConfig` fields (when `method="spectral"`):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bit_width_inlier` | `int` | `2` | Bit-width used for both signal and noise dimensions |
| `spectral_key_d_eff` | `int` | `4` | Signal dimension count for keys |
| `spectral_val_d_eff` | `int` | `50` | Signal dimension count for values |
| `spectral_apply_qjl` | `bool` | `True` | Apply QJL sign-sketch on key signal dimensions |
| `spectral_model_name` | `str` | `"model"` | Identifier used for the on-disk rotation cache |

Calibration is a separate, explicit step: call `cache.calibrate(rotation_entry)` after `KVCacheBuilder.build(...)` for each layer, using the tuples from `calibrate_spectral_rotation()` or `load_cached_rotations()`. Without it, the cache runs with a random orthogonal rotation instead of a data-derived one.

## When to use SpectralQuant

**Use SpectralQuant when:**
- Context length exceeds 8k tokens
- Perplexity must be minimised (long sequences amplify accumulation errors)
- The model's key distribution is low-rank (check `compute_participation_ratio`)
- You can spend time calibrating and explicitly injecting rotations per layer

**Consider alternatives when:**
- Zero calibration required → [TurboQuant RVQ](../algorithms/rvq)
- Maximum compression is the goal → [VecInfer](../algorithms/vecinfer)

SpectralQuant's signal/noise split is a **binary** cutoff (participation-ratio
derived) with **uniform bits within each half**. [KVTC-adapted](../algorithms/kvtc)
takes the same local-PCA starting point but replaces the binary cutoff with a
**dynamic-programming-optimal** bit-width *per individual component* (not just
two tiers) — including exactly 0 bits for a component — and adds a
zero-calibration entropy-coding stage on top.

## See also

- [Calibration guide](../guides/calibration)
- [Spectral API](../api/spectral-api)
