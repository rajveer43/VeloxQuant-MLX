---
id: calibration
title: Calibration Guide
sidebar_label: Calibration
slug: /guides/calibration
---

# Calibration Guide

Some VeloxQuant-MLX algorithms require a calibration step before inference. This guide explains when calibration is needed and how to collect, save, and reuse calibration artifacts using the real functions each algorithm exposes.

## Which algorithms need calibration?

| Algorithm | Calibration needed | What is calibrated |
|---|---|---|
| TurboQuant RVQ / MSE / Prod | No | Fixed analytical codebooks |
| QJL | No | Fixed random projection |
| RaBitQ | Yes — `fit()` | IVF centroids (one-time, no model forward pass needed) |
| PolarQuant | No | Fixed rotation + per-level codebooks (random init) |
| CommVQ | Yes — `fit()` | Residual VQ codebooks (one-time, no model forward pass needed) |
| **VecInfer** | **Yes** | Smooth factors + product codebook, from real key/value activations |
| **RateQuant** | **Yes** | Per-layer sensitivity weights → bit allocation |
| **SpectralQuant** | **Yes** | Per-layer SVD rotation matrices, from real key/value activations |

RaBitQ and CommVQ calibrate via `.fit()` on a plain array of sample vectors (no model or tokenizer needed) — see their algorithm pages ([RaBitQ](../algorithms/rabitq), [CommVQ](../algorithms/commvq)) for the exact call. VecInfer and SpectralQuant calibrate against a model's actual key/value activations, shown below.

## VecInfer calibration

```python
import numpy as np
import mlx.core as mx
from veloxquant_mlx.allocators.vecinfer import calibrate_smooth_factors, train_codebook

head_dim = 128
key_sub_dim = 4
value_sub_dim = 8

# Collect real key/value activations from your model — shape
# [n_tokens, n_heads, head_dim]. calibrate_smooth_factors/train_codebook
# take raw arrays; there is no built-in "run the model and collect" helper,
# so you hook into your own forward pass (or use synthetic data for testing).
keys_calib = mx.array(np.random.default_rng(0).standard_normal(
    (4096, 8, head_dim)).astype(np.float32))
values_calib = mx.array(np.random.default_rng(1).standard_normal(
    (4096, 8, head_dim)).astype(np.float32))

smooth_factors = calibrate_smooth_factors(keys_calib)

key_codebook = train_codebook(
    keys_calib.reshape(-1, key_sub_dim), n_centroids=2 ** 12, seed=42,
)
value_codebook = train_codebook(
    values_calib.reshape(-1, value_sub_dim), n_centroids=2 ** 8, seed=43,
)

np.savez(
    "vecinfer_artifacts.npz",
    smooth=np.asarray(smooth_factors),
    key_cb=np.asarray(key_codebook),
    value_cb=np.asarray(value_codebook),
)
```

See the [VecInfer page](../algorithms/vecinfer) for the full calibrate → build → run flow.

## RateQuant calibration

```python
from veloxquant_mlx.allocators.ratequant import (
    calibrate_layer_sensitivities,
    allocate_bits_ratequant,
)

sensitivities = calibrate_layer_sensitivities(model, tokenizer, seq_len=256)

bit_allocation = allocate_bits_ratequant(
    sensitivities,
    target_avg_bits=2.0,
    beta=3.5,  # paper-reported constant; see fit_distortion_curve() docstring
               # for why fitting it from scratch is usually unnecessary
)
```

`bit_allocation` is a plain `list[int]` — pass it as `KVCacheConfig.bit_width_inlier` and build with `KVCacheBuilder.for_model(...)` (see the [RateQuant page](../algorithms/ratequant)). There's no separate artifact format to save beyond that list, so a simple `json.dump`/`np.save` is enough if you want to persist it.

## SpectralQuant calibration

```python
from veloxquant_mlx.spectral.calibrate import calibrate_spectral_rotation, save_rotations

calibration_tokens = tokenizer.encode("representative calibration text...")

rotations = calibrate_spectral_rotation(
    model,
    calibration_tokens,
    n_tokens=512,
    model_name="my-model",
)

save_rotations(model_name="my-model", rotations=rotations)  # writes to the on-disk rotation cache
```

See the [SpectralQuant page](../algorithms/spectral) for how to load the cached rotations back and inject them via `cache.calibrate(...)`.

## Loading calibration artifacts

```python
import numpy as np
from veloxquant_mlx.spectral.calibrate import load_cached_rotations

# VecInfer — plain npz, however you saved it
data = np.load("vecinfer_artifacts.npz")
smooth_factors, key_codebook, value_codebook = data["smooth"], data["key_cb"], data["value_cb"]

# SpectralQuant — keyed by the model_name passed to save_rotations()
rotations = load_cached_rotations("my-model")
```

There is no generic artifact store with freeform `save(name, value)`/`load(name)` methods for these use cases — `NpyArtifactStore` (`veloxquant_mlx/artifacts/npy_store.py`) exposes a fixed, typed API instead: `save_rotation_matrix`/`load_rotation_matrix`, `save_codebook`/`load_codebook`, `save_jl_matrix`/`load_jl_matrix`, keyed by `(d, seed)`/`(distribution, b, d)`/`(d, m, seed)` respectively. It backs the zero-calibration methods' analytical artifacts (see the CLI section below), not VecInfer/RateQuant/SpectralQuant's calibration outputs.

## The `precompute` CLI

`python -m veloxquant_mlx precompute` exists, but it precomputes the **zero-calibration** artifacts shared by `turboquant_mse`/`turboquant_prod`/`qjl` (analytical Gaussian/Beta codebooks, rotation matrix, JL matrix) — it does not run VecInfer/RateQuant/SpectralQuant's per-model calibration.

```bash
python -m veloxquant_mlx precompute \
    --head_dim 128 \
    --bits 1 2 3 4 \
    --jl_dim 128 \
    --seed 42 \
    --output_dir ./artifacts/
```

| Flag | Default | Description |
|---|---|---|
| `--head_dim` | `128` | Attention head dimension |
| `--bits` | `1 2 3 4` | Bit-widths to precompute codebooks for |
| `--jl_dim` | `128` | JL projection dimension |
| `--seed` | `42` | Random seed |
| `--output_dir` | `./artifacts/` | Output directory for the `.npy` files, read via `NpyArtifactStore` |

## Artifact reuse across sessions

- SpectralQuant rotations are cached on disk keyed by `model_name` and reused indefinitely (they don't expire).
- VecInfer's codebook/smooth-factor arrays are whatever you saved them as — reuse them for the same model and quantization config.
- RateQuant's bit allocation is tied to the sensitivity calibration run — recalibrate if the model or target average bit rate changes.

## When to recalibrate

| Situation | Recalibrate? |
|---|---|
| Same model, new prompt domain | Optional — usually not needed |
| Updated model weights (fine-tune) | Yes |
| Different model family | Yes |
| Different quantization bit rate (RateQuant) | Yes — re-run `allocate_bits_ratequant` with the new `target_avg_bits` |
| Updated VeloxQuant-MLX version | Check `CHANGELOG.md` |

## See also

- [VecInfer algorithm](../algorithms/vecinfer)
- [RateQuant algorithm](../algorithms/ratequant)
- [SpectralQuant algorithm](../algorithms/spectral)
- [Allocators API](../api/allocators)
- [Spectral API](../api/spectral-api)
