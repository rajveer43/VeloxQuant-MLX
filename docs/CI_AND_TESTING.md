# CI and testing policy

VeloxQuant-MLX targets **Apple Silicon** (M1+). End-to-end `mlx_lm`
generation and Metal kernel parity tests need a real Mac GPU.

## Two test directories, and why

The repo has two, separately-run test trees:

| Directory | Suite | Runner | Run via |
| --- | --- | --- | --- |
| `veloxquant_mlx/tests/` | Main suite (MLX-dependent) | `macos-14` (Apple Silicon) | `pytest` (`testpaths` in `pyproject.toml`) |
| `tests/non_metal/` | Pure-Python, MLX-free tests | `ubuntu-latest` | `.github/workflows/non-metal-unit.yml` |

This split is **intentional, not a stray duplicate** — do not merge the two
directories or delete `tests/non_metal/` as "cleanup."

**Why `tests/non_metal/` has to live outside `veloxquant_mlx/`:** collecting
any module under `veloxquant_mlx/` triggers `veloxquant_mlx/__init__.py`,
which imports `mlx`. `mlx` only installs/runs on Apple Silicon, so any test
file under `veloxquant_mlx/tests/` requires the expensive `macos-14` runner
just to be *collected*, even if the test itself never touches MLX. Modules
that are genuinely pure Python (e.g. `veloxquant_mlx/tools/mac_recommender.py`,
a RAM/method recommender with no MLX dependency) can still get fast, cheap
CI coverage by keeping their tests in `tests/non_metal/` instead, loaded via
`importlib.util.spec_from_file_location` (not a normal import) specifically
to avoid pulling in `veloxquant_mlx/__init__.py`'s import chain. This keeps
`non-metal-unit.yml` running on a plain `ubuntu-latest` runner with no `mlx`
install at all.

`pyproject.toml`'s `testpaths = ["veloxquant_mlx/tests"]` means a plain
`pytest` invocation at the repo root never picks up `tests/non_metal/` — only
the dedicated `non-metal-unit` workflow does, with explicit isolation flags
(`--noconftest --import-mode=importlib -c /dev/null -o addopts=`) so it
doesn't pick up the project's own pytest config.

**When adding a new test, choose based on the module under test, not the
test's own content:**

- If the module you're testing imports `mlx` (directly or transitively,
  including anything under `veloxquant_mlx/` that isn't leaf-level pure
  Python) → `veloxquant_mlx/tests/`.
- If the module is genuinely pure Python with **no** MLX dependency (a CLI
  tool, config parser, recommender, etc.) → consider adding it to
  `tests/non_metal/` as well (or instead, if it has no MLX-dependent
  behavior to test) so it gets cheap, fast Linux coverage. Follow the
  `importlib.util.spec_from_file_location` loading pattern in
  `tests/non_metal/test_mac_recommender.py` to avoid importing
  `veloxquant_mlx/__init__.py`.

## What should run where

| Suite | Where | Notes |
| --- | --- | --- |
| Pure Python unit tests (no Metal) | Linux CI (`non-metal-unit.yml`) or macOS CI | Examples: `tests/non_metal/test_mac_recommender.py`, many quantizer math tests under `veloxquant_mlx/tests/` |
| Metal parity / kernel tests | Apple Silicon only | Skip or mark `metal` on headless/Linux runners |
| End-to-end generation benches | Local macOS | Scripts under `benchmark_scripts/` and `scripts/validate_kv_memory.py` |

## Guidance for contributors

1. Always run `python -m pytest veloxquant_mlx/tests -q` on a Mac before a PR
   that touches caches, Metal, or generation paths.
2. Number claims need a reproducible script + committed `results.json`
   (see CONTRIBUTING).
3. Do not assume GitHub-hosted Linux runners can execute Metal kernels.

## Suggested follow-up CI (not required for Phase 1)

- Document `pytest -m "not metal"` once Metal tests are consistently marked.
- Keep release publishing (PyPI) separate from e2e benches.
