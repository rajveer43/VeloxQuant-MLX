"""Sync README badges and landing-page version/test claims to the current release.

Run as the last step of the release workflow (.github/workflows/release.yml),
after python-semantic-release has already bumped pyproject.toml and tagged
the release, so the version this script reads is the new one.

What is synced automatically:
  README.md            — tests badge, changelog-version badge
  landing/index.html   — "New in <version>:" in the meta description,
                         the "<n>/<n> tests passing" spec-list claim,
                         the hero badge's "v<version>" prefix (#hero-badge
                         data-text attribute and its textContent fallback)

What is NOT synced: the *prose* after "v<version> — " in the hero badge and
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
    """Sync the hero badge's version prefix in landing/index.html.

    The badge markup is:
        <div class="badge" id="hero-badge" data-text="v0.41.0 — <prose> shipped">v0.41.0 — <prose> shipped</div>
    `initBadgeTyping()` in landing/assets/main.js reads `dataset.text` (falling
    back to `textContent`) to drive the typing animation, so both copies of the
    string inside the tag must carry the same version — there is no separate
    version constant in main.js to sync.
    """
    if not LANDING_HTML.exists():
        _warn(f"{LANDING_HTML.relative_to(ROOT)} not found, skipping")
        return

    text = LANDING_HTML.read_text()

    # Matches the id="hero-badge" div's opening tag through its inner text,
    # capturing each "v<version> — " occurrence (data-text attribute, then
    # the visible textContent) so both are rewritten together.
    badge_pattern = re.compile(
        r'(<div class="badge" id="hero-badge" data-text="v)([\d.]+)( — [^"]*">v)([\d.]+)( — )'
    )
    match = badge_pattern.search(text)
    if not match:
        _warn("hero badge not found in landing/index.html, skipping")
        return

    previous = match.group(2)
    text = badge_pattern.sub(
        rf"\g<1>{version}\g<3>{version}\g<5>",
        text,
    )
    LANDING_HTML.write_text(text)

    if previous != version:
        _warn(
            f"hero badge bumped {previous} -> {version}, but its headline prose still "
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
