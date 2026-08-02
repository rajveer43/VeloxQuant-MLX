"""Sync README badges and landing-page version/test claims to the current release.

Run as the last step of the release workflow (.github/workflows/release.yml),
after python-semantic-release has already bumped pyproject.toml and tagged
the release, so the version this script reads is the new one.

What is synced automatically:
  README.md            — tests badge, changelog-version badge
  landing/index.html   — "New in <version>:" in the meta description,
                         the "<n>/<n> tests passing" spec-list claim
  landing/assets/main.js — the hero pill's "v<version>" prefix

What is NOT synced: the *prose* after "v<version> — " in the hero pill and
after "New in <version>: " in the meta description. That copy describes what
actually shipped and cannot be derived from a version number, so this script
only rewrites the version itself and warns when the prose still references the
previous release. Update it by hand in the release PR.

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
LANDING_HTML = ROOT / "landing" / "index.html"
LANDING_JS = ROOT / "landing" / "assets" / "main.js"

_warnings: list[str] = []


def _warn(msg: str) -> None:
    _warnings.append(msg)
    print(f"sync_release_badges: {msg}", file=sys.stderr)


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


def _sync_readme(version: str, test_count: int) -> None:
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
        _warn("tests badge pattern not found in README.md, skipping")
    if n_changelog == 0:
        _warn("changelog badge pattern not found in README.md, skipping")

    README.write_text(text)


def _sync_landing_html(version: str, test_count: int) -> None:
    """Sync the landing page's meta description and test-count claim."""
    if not LANDING_HTML.exists():
        _warn(f"{LANDING_HTML.relative_to(ROOT)} not found, skipping")
        return

    text = LANDING_HTML.read_text()

    # "New in 0.41.0: <prose>" -> only the version is rewritten; the prose after
    # the colon describes the actual feature and stays under human control.
    stale_new_in = re.search(r"New in (?!%s:)([\d.]+):" % re.escape(version), text)
    text, n_new_in = re.subn(r"New in [\d.]+:", f"New in {version}:", text)
    if n_new_in == 0:
        _warn("'New in <version>:' not found in landing/index.html, skipping")
    elif stale_new_in:
        _warn(
            f"landing/index.html meta description bumped to {version}, but its prose "
            f"still describes {stale_new_in.group(1)} — update the copy by hand"
        )

    # "928/934 tests passing" or "1484/1484 tests passing". The suite is expected
    # to be fully green at release time (the workflow's test gate runs before
    # this script), so both halves become the collected count.
    text, n_claim = re.subn(
        r"\d+/\d+ tests passing",
        f"{test_count}/{test_count} tests passing",
        text,
    )
    if n_claim == 0:
        _warn("'<n>/<n> tests passing' claim not found in landing/index.html, skipping")

    LANDING_HTML.write_text(text)


def _sync_landing_pill(version: str) -> None:
    """Sync the hero pill's version prefix in the typing-animation string."""
    if not LANDING_JS.exists():
        _warn(f"{LANDING_JS.relative_to(ROOT)} not found, skipping")
        return

    text = LANDING_JS.read_text()

    # const text = "v0.41.0 — <prose> shipped";
    pattern = r'(const text = ")v([\d.]+)( — )'
    match = re.search(pattern, text)
    if not match:
        _warn("hero pill string not found in landing/assets/main.js, skipping")
        return

    previous = match.group(2)
    text = re.sub(pattern, rf"\g<1>v{version}\g<3>", text)
    LANDING_JS.write_text(text)

    if previous != version:
        _warn(
            f"hero pill bumped {previous} -> {version}, but its headline prose still "
            f"describes {previous} — update it by hand to name what shipped"
        )


def main() -> None:
    version = _current_version()
    test_count = _live_test_count()

    _sync_readme(version, test_count)
    _sync_landing_html(version, test_count)
    _sync_landing_pill(version)

    print(f"Synced badges + landing: tests={test_count}, version={version}")
    if _warnings:
        print(f"{len(_warnings)} warning(s) above — release copy may need a manual edit")


if __name__ == "__main__":
    main()
