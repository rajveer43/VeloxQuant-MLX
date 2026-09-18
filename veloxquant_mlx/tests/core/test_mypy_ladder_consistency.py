"""Keep the mypy strictness ladder's two module lists in agreement.

The ladder added in #438 names the same four modules twice:

  - `module = [...]` in the `[[tool.mypy.overrides]]` block of pyproject.toml,
    which carries the strictness flags and is what a local `mypy` run obeys;
  - the file list in the blocking `mypy-strict-modules` job in
    `.github/workflows/lint.yml`, which is what CI actually invokes.

Both must name the same set, and the failure when they do not is silent in the
worst direction. Adding a module to pyproject.toml but not the workflow means
CI never checks it -- the job passes while the intended gate does not exist.
Adding it to the workflow but not pyproject.toml means CI checks it with
default (non-strict) settings, so it passes for the wrong reason. Neither
mistake produces a red build; the ladder just quietly stops being a ladder.

This is the same class of drift as #439 (prose quoting metadata that later
moved), so it gets the same treatment: a test, not a convention.

Deliberately NOT covered:
  - Whether the modules actually pass. That is the `mypy-strict-modules` job's
    job, and duplicating it here would mean running mypy inside the unit-test
    suite on every matrix leg.
  - The specific strictness flags. They live only in pyproject.toml -- the
    workflow intentionally does not repeat them -- so there is no second copy
    to drift.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = ROOT / "pyproject.toml"
LINT_WORKFLOW = ROOT / ".github" / "workflows" / "lint.yml"

# Parsed with `re`, not `tomllib`/`yaml`: this suite runs on the 3.10 leg of the
# CI matrix, where `tomllib` is not stdlib, and PyYAML is not a declared
# dependency of this project at all. A bare import of either would ImportError
# at *collection*, taking down the whole run rather than one test. Same
# reasoning as test_python_floor_consistency.py.


def _ladder_modules_from_pyproject() -> set[str]:
    """Dotted module names in the `[[tool.mypy.overrides]]` ladder block."""
    text = PYPROJECT.read_text()
    block = re.search(
        r"\[\[tool\.mypy\.overrides\]\]\s*\nmodule\s*=\s*\[(?P<body>[^\]]*)\]",
        text,
    )
    assert block, (
        "Could not find a `[[tool.mypy.overrides]]` block with a `module = [...]` "
        "list in pyproject.toml. If the ladder moved or was removed, update this "
        "test -- do not delete it without also removing the mypy-strict-modules "
        "job in .github/workflows/lint.yml."
    )
    return set(re.findall(r'"([^"]+)"', block.group("body")))


def _ladder_paths_from_workflow() -> set[str]:
    """Module paths passed to mypy by the blocking `mypy-strict-modules` job."""
    text = LINT_WORKFLOW.read_text()
    job = re.search(
        r"\n  mypy-strict-modules:\n(?P<body>(?:.*\n)*?)(?=\n  \S|\Z)",
        text,
    )
    assert job, (
        "Could not find the `mypy-strict-modules` job in "
        ".github/workflows/lint.yml. If it was renamed, update this test; if it "
        "was removed, the ladder in pyproject.toml is no longer enforced by CI."
    )
    paths = set(re.findall(r"(veloxquant_mlx/[\w/]+\.py)", job.group("body")))
    assert paths, (
        "The mypy-strict-modules job names no veloxquant_mlx/*.py files. Either "
        "the invocation changed shape (update this pattern) or the job now "
        "checks nothing while still reporting success."
    )
    return paths


def _to_dotted(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".")


def test_ladder_lists_match() -> None:
    """pyproject's override list and the CI job check the same modules."""
    configured = _ladder_modules_from_pyproject()
    invoked = {_to_dotted(p) for p in _ladder_paths_from_workflow()}

    missing_from_ci = configured - invoked
    assert not missing_from_ci, (
        f"These modules are on the strictness ladder in pyproject.toml but are "
        f"not checked by the mypy-strict-modules job: {sorted(missing_from_ci)}. "
        f"CI reports success without ever checking them. Add them to the job in "
        f".github/workflows/lint.yml."
    )

    missing_from_config = invoked - configured
    assert not missing_from_config, (
        f"These modules are checked by the mypy-strict-modules job but have no "
        f"`[[tool.mypy.overrides]]` entry: {sorted(missing_from_config)}. The "
        f"job runs them with default settings, so it passes without applying "
        f"any of the ladder's strictness. Add them to the override in "
        f"pyproject.toml."
    )


def test_ladder_modules_exist() -> None:
    """Every module named on the ladder is a real file.

    A renamed or deleted module leaves mypy checking nothing under that name
    while both lists still agree with each other, so the test above would not
    catch it.
    """
    for dotted in sorted(_ladder_modules_from_pyproject()):
        path = ROOT / (dotted.replace(".", "/") + ".py")
        assert path.exists(), (
            f"Ladder names `{dotted}` but {path.relative_to(ROOT)} does not "
            f"exist. If the module moved, update both pyproject.toml and the "
            f"mypy-strict-modules job."
        )


def test_ladder_job_is_blocking() -> None:
    """The ladder job must not carry `continue-on-error`.

    The whole-package `mypy` job in the same file is deliberately
    reporting-only. This one exists precisely to block, and copying that
    `continue-on-error: true` line into it would turn the gate back into a
    status light without changing anything visible.
    """
    text = LINT_WORKFLOW.read_text()
    job = re.search(r"\n  mypy-strict-modules:\n(?P<body>(?:.*\n)*?)(?=\n  \S|\Z)", text)
    assert job, "Could not find the `mypy-strict-modules` job."
    assert "continue-on-error" not in job.group("body"), (
        "The mypy-strict-modules job has `continue-on-error`, which makes it "
        "non-blocking. That defeats its only purpose -- the reporting-only "
        "whole-package `mypy` job already covers visibility."
    )
