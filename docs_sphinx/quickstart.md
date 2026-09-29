# Quickstart

## Installation

```bash
pip install VeloxQuant-MLX
```

Requirements: Apple Silicon M1+, Python >= 3.10, MLX >= 0.18, NumPy >= 1.26.

Source install, conda/miniforge, Metal troubleshooting, and verifying the install
are covered in the [installation guide](https://veloxquant.dev/docs/getting-started/installation).

## Usage

RVQ 1-bit, 7.5x compression, no calibration (recommended default):

```python
import mlx_lm
from veloxquant_mlx import KVCacheBuilder, KVCacheConfig

model, tokenizer = mlx_lm.load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches

response = mlx_lm.generate(model, tokenizer, prompt="Explain relativity simply.", max_tokens=200)
```

No Python, using the control panel:

```bash
veloxquant panel     # local web UI at http://127.0.0.1:7860
```

For the full method library, algorithm comparisons, and guides, see
[veloxquant.dev](https://veloxquant.dev/). This site covers the API reference only —
see {doc}`api/index`.
