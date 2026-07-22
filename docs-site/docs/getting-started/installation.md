---
id: installation
title: Installation
sidebar_label: Installation
slug: /getting-started/installation
---

# Installation

:::warning[Apple Silicon required]
VeloxQuant-MLX uses Metal GPU kernels compiled at runtime. It requires **macOS on an Apple M-series chip** (M1 or later). Intel Macs and Linux are not supported.
:::

## Requirements

| Requirement | Version |
|---|---|
| macOS | 13 Ventura or later |
| Apple Silicon | M1, M2, M3, M4 (any tier) |
| Python | 3.11 or 3.12 |
| MLX | ≥ 0.18 |
| NumPy | ≥ 1.26 |

## Install from PyPI

```bash
pip install veloxquant-mlx
```

This installs the library and its runtime dependencies (MLX, NumPy, psutil).

## Install from source

Use this if you want the latest development version or plan to contribute.

```bash
git clone https://github.com/rajveer43/VeloxQuant-MLX.git
cd VeloxQuant-MLX
pip install -e ".[dev]"
```

The `[dev]` extra installs SciPy (for codebook training) and the full test suite dependencies.

## Verify the installation

```python
import veloxquant_mlx
print(veloxquant_mlx.__version__)   # should print the version you just installed

import mlx.core as mx
print(mx.default_device())           # Device(gpu, 0)
```

If you see `Device(cpu, 0)` instead of `gpu`, MLX is not using Metal. See the troubleshooting section below.

## Install mlx_lm (optional but recommended)

Most users will want to run VeloxQuant-MLX with `mlx_lm` for model loading and generation:

```bash
pip install mlx-lm
```

Check compatibility:

```bash
python -c "import mlx_lm; print(mlx_lm.__version__)"
```

## Troubleshooting

### `Device(cpu, 0)` — Metal not active

MLX falls back to CPU when Metal is unavailable. Common causes:

1. **Running in a VM or Docker** — Metal is not forwarded; run natively on the Mac host.
2. **Rosetta 2 Python** — Install an arm64 Python via Homebrew or the official installer:
   ```bash
   brew install python@3.12
   ```
3. **Conda environment** — Use miniforge (arm64) instead of standard Anaconda:
   ```bash
   brew install miniforge
   conda create -n velox python=3.12
   conda activate velox
   pip install veloxquant-mlx
   ```

### `ImportError: No module named 'mlx'`

MLX is only published for Apple Silicon. If `pip install mlx` fails, you are on an incompatible platform.

### Metal kernel compilation errors

VeloxQuant-MLX compiles Metal kernels on first use with `mx.fast.metal_kernel`. If you see compilation errors:

```bash
# Ensure Xcode Command Line Tools are installed
xcode-select --install

# Confirm Metal availability
python -c "import mlx.core as mx; mx.fast.metal_kernel"
```

### `ArtifactNotFoundError` on first run

Some algorithms (VecInfer, SpectralQuant) require calibration artifacts before
first use. There is no single `precompute --method ... --model ...` command —
each method calibrates differently:

- **VecInfer** — calibrate smooth factors and train a codebook from your
  model's own key/value activations (`calibrate_smooth_factors`,
  `train_codebook`).
- **SpectralQuant** — calibrate per-layer rotation matrices
  (`calibrate_spectral_rotation`, cached to disk by model name).
- **Zero-calibration methods** (TurboQuant, QJL, PolarQuant) can precompute
  their analytical codebooks/rotation matrices ahead of time with:
  ```bash
  python -m veloxquant_mlx precompute --head_dim 128 --bits 1 2 3 4 --jl_dim 128 --output_dir ./artifacts/
  ```

See the [Calibration Guide](../guides/calibration) for the full, verified
walkthrough of each method's real calibration flow.

## Next steps

- [5-minute quickstart](../getting-started/quickstart)
- [Core concepts](../getting-started/concepts)
