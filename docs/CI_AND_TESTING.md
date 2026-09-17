# CI and testing policy

VeloxQuant-MLX targets **Apple Silicon** (M1+). End-to-end `mlx_lm`
generation and Metal kernel parity tests need a real Mac GPU.

## Two test directories, and why

The repo has two, separately-run test trees:

| Directory | Suite | Runner | Run via |
| --- | --- | --- | --- |
| `veloxquant_mlx/tests/` | Main suite (MLX-dependent) | `macos-14` (Apple Silicon) | `pytest` (`testpaths` in `pyproject.toml`); on every push/PR to `master` via `.github/workflows/mlx-tests.yml`, and again as a release gate in `release.yml` |
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
  - If your module needs a stand-in for a `veloxquant_mlx` parent package
    (so a `from veloxquant_mlx.x.y import z` inside it resolves), remove
    the stand-in `sys.modules` entries again once your module has finished
    loading (see `tests/non_metal/test_block_pool.py`). `non-metal-unit.yml`
    runs every file in this directory in one pytest process, so a stand-in
    left behind at module-collection time shadows the real package for
    every other file collected afterward — #425 was exactly this: a fake,
    submodule-less `veloxquant_mlx` leaking past its own file and breaking
    `test_mac_recommender.py`'s import of the real package.

## What should run where

| Suite | Where | Notes |
| --- | --- | --- |
| Pure Python unit tests (no Metal) | Linux CI (`non-metal-unit.yml`) or macOS CI | Examples: `tests/non_metal/test_mac_recommender.py`, `tests/non_metal/test_block_pool.py`, many quantizer math tests under `veloxquant_mlx/tests/` |
| Metal parity / kernel tests (correctness) | GitHub-hosted `macos-14`/`macos-15` (via `mlx-tests.yml`, every PR) or any real Apple Silicon | Validated in issue #395: hosted runners expose a genuine, correct — if paravirtualized — Metal device. Selectable via `pytest -m metal` / `-m "not metal"`. |
| Metal kernel **performance**/timing numbers | Real Apple Silicon only, never hosted runners | The hosted runner's device reports as `"Apple Paravirtual device"` with a synthetic `memory_size`; its timing characteristics are unknown and not assumed representative. See `docs/BENCHMARK_INFRASTRUCTURE_FEASIBILITY.md` Lane B. |
| End-to-end generation benches | Local macOS / dedicated bare-metal | Scripts under `benchmark_scripts/` and `scripts/validate_kv_memory.py` |

## Guidance for contributors

1. Always run `python -m pytest veloxquant_mlx/tests -q` on a Mac before a PR
   that touches caches, Metal, or generation paths.
2. Number claims need a reproducible script + committed `results.json`
   (see CONTRIBUTING).
3. Do not assume GitHub-hosted Linux runners can execute Metal kernels.

## The `metal` marker

Tests that need a real Metal device carry both a module-level
`pytest.mark.metal` marker and the pre-existing
`pytest.mark.skipif(not metal_available(), ...)`. Keep both: the marker lets
CI *select* (`-m metal`) or *deselect* (`-m "not metal"`) this subset
explicitly (e.g. to isolate a hosted-runner correctness job from a
bare-metal performance job), while the `skipif` still makes the test suite
self-skip correctly on a real machine that genuinely lacks Metal (older
Intel Mac, CI misconfiguration) even if nobody passed `-m`. Registered in
`pyproject.toml`'s `[tool.pytest.ini_options] markers`.

Do not mark a test `metal` if it specifically tests the *absence* of Metal
(e.g. `test_use_metal_kernels_true_without_metal_raises` in
`test_qfilters_cache.py`) — it should keep its own `skipif(metal_available(), ...)`
guard and no `metal` marker, so `-m metal` never selects it.

## Suggested follow-up CI (not required for Phase 1)

- Keep release publishing (PyPI) separate from e2e benches.
