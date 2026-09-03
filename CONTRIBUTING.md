# Contributing to VeloxQuant-MLX

Thanks for your interest in contributing. VeloxQuant-MLX is a KV-cache
quantization library for `mlx_lm` on Apple Silicon, and contributions of all
kinds are welcome: bug reports, new quantization methods, benchmarks on
additional models, documentation, and performance work.

## Issue-first workflow

For anything beyond a tiny typo fix, **open an issue first** using one of the
templates under `.github/ISSUE_TEMPLATE/`:

| Template | Use when |
| --- | --- |
| Bug report | Something is broken or incorrect |
| Feature request | New API, method, or capability |
| Validation run | Recording a reproducible comparison |

Then create a feature branch and open a PR that references the issue
(`Closes #N`). This keeps discussion searchable and reviewable.

### Suggested labels

Maintainers may apply these labels (create them in the GitHub UI as needed):

- `bug`, `enhancement`, `docs`, `validation`, `metal`, `good first issue`
- `method:*` for work tied to a specific algorithm (e.g. `method:rvq`)

Do not add a heavy project board until issue volume justifies it.

## Reporting bugs and requesting features

Please open an issue at
<https://github.com/rajveer43/VeloxQuant-MLX/issues>. Prefer the issue
templates. For bugs, include:

- your hardware (chip + RAM) and macOS version,
- `python`, `mlx`, and `mlx_lm` versions,
- the model id you were running,
- a minimal snippet that reproduces the problem, and
- the full error output.

## Accounting vs resident memory

Every compression or memory claim in an issue or PR must say which metric
was measured:

1. **Key-byte accounting** — `fp16_key_bytes / compressed_key_bytes` on the
   cache objects. This is what many benches print as `key_x`. It reflects
   packed-format size, not necessarily process RSS.
2. **Full-KV accounting** — keys + values (+ residual fp16 windows when the
   method keeps them).
3. **MLX peak memory** — `mx.get_peak_memory()` (weights, activations, and
   temporary tensors dominate at short context).
4. **Resident / OS RSS** — Activity Monitor or `ps` RSS. Use long contexts
   before claiming the user will see large RAM savings.

Default **RVQ** and **VecInfer** paths quantize then dequantize into the
parent `mlx_lm` fp16 `KVCache`. Headline ratios such as 7.5× or 16× are
**key accounting** unless a packed or fused storage path is active and
measured. Do not describe accounting ratios as resident RAM savings.

We do not merge number claims that lack a reproducible script and a
committed `results.json` (see Submitting changes).

## Getting set up

Requires Apple Silicon (M1 or later), Python ≥ 3.11, and MLX ≥ 0.18.

```bash
git clone https://github.com/rajveer43/VeloxQuant-MLX
cd VeloxQuant-MLX
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Fork first if you do not have write access, then add
`upstream` pointing at `rajveer43/VeloxQuant-MLX`.

## Code style (pre-commit)

Python code is formatted and linted with [Ruff](https://docs.astral.sh/ruff/)
(config in `pyproject.toml`). Install the git hook once after setting up your
virtualenv:

```bash
pre-commit install
```

This runs `ruff format` and `ruff check --fix` on staged files before every
commit. To run it manually against the whole repo (e.g. before opening a
PR):

```bash
pre-commit run --all-files
```

## Running the tests

```bash
python -m pytest veloxquant_mlx/tests -q
```

New code should come with tests that mirror the existing conventions in
`veloxquant_mlx/tests/` (shape/dtype preservation, reconstruction-quality
bounds on seeded synthetic data, and for any Metal kernel a parity test
against the pure-MLX path). Seed all randomness: determinism is treated as a
correctness requirement.

End-to-end `mlx_lm` generation benches require Apple Silicon and are run
locally.

The repo has two, deliberately separate test directories —
`veloxquant_mlx/tests/` (MLX-dependent, requires Apple Silicon) and
`tests/non_metal/` (pure Python, MLX-free, runs on plain Linux CI). See
[`docs/CI_AND_TESTING.md`](docs/CI_AND_TESTING.md#two-test-directories-and-why)
for why the split exists and which one a new test belongs in.

## Submitting changes

1. Open or reference an issue for non-trivial work.
2. Fork (if needed) and create a feature branch
   (`chore/...`, `docs/...`, `feat/...`, `fix/...`).
3. Make your change with accompanying tests and documentation.
4. Ensure the full test suite passes locally.
5. Open a pull request describing the change and, for any performance or
   compression claim, the committed `results.json` it traces to. We do not
   merge numbers that are not reproducible from a script in the repository.

## Adding a new quantization method

A new method typically consists of:

- a `Quantizer` subclass in `veloxquant_mlx/quantizers/`, registered with
  `QuantizerRegistry`,
- an `mlx_lm`-compatible cache wrapper in `veloxquant_mlx/cache/` with byte
  accounting,
- wiring into `KVCacheConfig` / `KVCacheFactory`,
- tests, and
- a benchmark script emitting `figures/<method>/<model>/results.json`.

See `paper/research/surveys/` for examples of how a method is scoped and
chosen before implementation (the highest-numbered `NEW_METHOD_SURVEY_V*.md`
is the most current).

## Commit conventions

This repo uses [Conventional Commits](https://www.conventionalcommits.org/)
and version numbers are bumped automatically from commit messages on merge
to `master` (see `.github/workflows/release.yml`) — so the prefix you use
has a real effect, not just a style preference.

Every commit (or, if a PR is squash-merged, the squash-merge title) must
start with one of:

| Prefix | Effect | Example |
| --- | --- | --- |
| `feat(scope): ...` | Minor version bump | `feat(kitty): dynamic channel-wise mixed-precision` |
| `fix(scope): ...` | Patch version bump | `fix(cache): bounds-check fraction configs` |
| `feat!: ...` or a `BREAKING CHANGE:` footer | Major version bump | `feat!: rename KVCacheConfig.bits to bit_width_inlier` |
| `docs: ...`, `chore: ...`, `test: ...`, `refactor: ...` | No version bump | `docs: fix broken link in README` |

`scope` is optional but encouraged — use the method name (`kitty`, `xquant`,
`cache`, `metal`, `docs-site`, …) so the changelog groups related changes.
A commit-lint check runs on every PR (`.github/workflows/commitlint.yml`)
and will fail the PR if no commit matches this pattern.

Reference the relevant issue number when one exists (`Closes #12`). Avoid
commits that mix unrelated changes — each commit/PR should map to one
semver-meaningful change so the automated changelog entry stays accurate.

**`landing/` changes are not package releases.** The commit parser looks
only at the `feat`/`fix`/`perf` prefix, not at which files changed, so
`feat(landing): ...` or `fix(landing): ...` still bumps `VeloxQuant-MLX`'s
version and republishes to PyPI even though nothing in `veloxquant_mlx/`
changed. Use `chore(landing): ...` or `docs(landing): ...` for landing-page
work instead — both are recognized types with no version effect. (The
release workflow also skips entirely for pushes that touch only
`landing/`, `docs/`, or `*.md` as a second line of defense, but the commit
type should still be correct.)

## Code of conduct

Please be respectful and constructive. We follow the
[Contributor Covenant v2.1](CODE_OF_CONDUCT.md).
