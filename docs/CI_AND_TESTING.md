# CI and testing policy

VeloxQuant-MLX targets **Apple Silicon** (M1+). End-to-end `mlx_lm`
generation and Metal kernel parity tests need a real Mac GPU.

## What should run where

| Suite | Where | Notes |
| --- | --- | --- |
| Pure Python unit tests (no Metal) | Linux CI or macOS CI | Examples: `tests/non_metal/test_mac_recommender.py`, many quantizer math tests |
| Metal parity / kernel tests | Apple Silicon only | Skip or mark `metal` on headless/Linux runners |
| End-to-end generation benches | Local macOS | Scripts under `benchmark_scripts/` and `scripts/validate_kv_memory.py` |

## Guidance for contributors

1. Always run `python -m pytest veloxquant_mlx/tests -q` on a Mac before a PR
   that touches caches, Metal, or generation paths.
2. Number claims need a reproducible script + committed `results.json`
   (see CONTRIBUTING).
3. Do not assume GitHub-hosted Linux runners can execute Metal kernels.

## Suggested follow-up CI (not required for Phase 1)

- A workflow that runs a **non-Metal** pytest selection on Linux (lint + pure
  Python modules).
- Document `pytest -m "not metal"` once Metal tests are consistently marked.
- Keep release publishing (PyPI) separate from e2e benches.
