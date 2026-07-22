"""Sync README badges to the current version and live test count.

Run as the last step of the release workflow (.github/workflows/release.yml),
after python-semantic-release has already bumped pyproject.toml and tagged
the release, so the version this script reads is the new one.

Usage:
    python scripts/sync_release_badges.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PYPROJECT = ROOT / "pyproject.toml"


def _current_version() -> str:
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def _live_test_count() -> int:
    """Count collected tests via `pytest --collect-only -q`.

    Does not execute any test — safe to run without a Metal GPU.
    """
    result = subprocess.run(
        ["python", "-m", "pytest", "veloxquant_mlx/tests", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"(\d+) tests? collected", result.stdout)
    if not match:
        raise RuntimeError(
            f"sync_release_badges: could not parse test count from pytest output:\n{result.stdout}"
        )
    return int(match.group(1))


def main() -> None:
    version = _current_version()
    test_count = _live_test_count()

    text = README.read_text()

    text, n_tests = re.subn(
        r"badge/tests-\d+%20passing-",
        f"badge/tests-{test_count}%20passing-",
        text,
    )
    text, n_changelog = re.subn(
        r"badge/changelog-[\d.]+-",
        f"badge/changelog-{version}-",
        text,
    )

    if n_tests == 0:
        print("sync_release_badges: tests badge pattern not found, skipping", file=sys.stderr)
    if n_changelog == 0:
        print("sync_release_badges: changelog badge pattern not found, skipping", file=sys.stderr)

    README.write_text(text)
    print(f"Synced badges: tests={test_count}, changelog={version}")


if __name__ == "__main__":
    main()
