"""Guard the documented Python floor against drifting from `requires-python`.

Regression (#439): 26ec112 lowered `requires-python` from ">=3.11" to ">=3.10"
but did not sweep the places that quote that value back. The stale "3.11"
survived in pyproject.toml's own [tool.mypy] comment and in two live
user-facing surfaces -- the docs-site install table and the landing page's
requirement badge and FAQ -- which told a 3.10 user they were unsupported when
`pip install` in fact works for them. Nothing failed, because no check ties
prose to metadata; the drift was found only by reading the file.

This asserts that every *floor claim* in the live docs agrees with
`requires-python`, so the next bump fails here instead of shipping a
contradiction.

Deliberately NOT covered:
  - Dated posts under blogs/ and docs-site/blog/. They were accurate when
    published and their version line sits inside the environment description
    of a benchmark run; rewriting them to match today's metadata would
    falsify the record of what was actually executed.
  - Example install commands (`brew install python@3.12`,
    `conda create -n velox python=3.12`). Naming a specific known-good
    version to install is not a claim about the supported range, so the
    patterns below match only stated floors/ranges, never a command.
  - `[tool.mypy] python_version`, which is a type-checking target and
    intentionally 3.12 while the floor is 3.10 (see the comment on it). Its
    prose is covered by test_mypy_comment_states_the_real_floor below, which
    checks the comment does not misname the floor -- not that it equals it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = ROOT / "pyproject.toml"

# Parsed with `re`, not `tomllib`: this suite runs on the 3.10 leg of the CI
# matrix, and `tomllib` is stdlib only from 3.11. Pulling in a `tomli` backport
# purely for two scalar fields would add a dependency to defend a floor -- and
# a bare `import tomllib` would ImportError at *collection* on exactly the
# oldest-Python job this test exists to protect.


def _floor() -> tuple[int, int]:
    """The (major, minor) floor declared by `requires-python`."""
    m = re.search(r'^requires-python\s*=\s*"([^"]+)"', PYPROJECT.read_text(), re.M)
    assert m, "Could not find `requires-python` in pyproject.toml."
    spec = m.group(1).strip()
    v = re.fullmatch(r">=\s*(\d+)\.(\d+)", spec)
    assert v, (
        f"requires-python is {spec!r}, which this test cannot parse. It only "
        f"understands a simple '>=X.Y' floor; if the project moves to a "
        f"compound specifier, teach this helper the new form."
    )
    return int(v.group(1)), int(v.group(2))


def _supported_minors() -> list[int]:
    """Minor versions claimed via `Programming Language :: Python ::` trove classifiers."""
    minors = [
        int(m.group(2))
        for m in re.finditer(
            r'"Programming Language :: Python :: (\d+)\.(\d+)"', PYPROJECT.read_text()
        )
    ]
    return sorted(minors)


# Each entry: (path, regex capturing a claimed floor minor version in group
# "minor"). Patterns are written to match a *stated floor or range* only.
_FLOOR_CLAIMS: list[tuple[str, str]] = [
    # "Requirements: Apple Silicon M1+, Python >= 3.10, ..."
    (
        "README.md",
        r"Python\s*(?:>=|≥)\s*3\.(?P<minor>\d+)",
    ),
    # "Apple Silicon M1+ . Python 3.10+ . 43 methods . MIT License"
    (
        "README.md",
        r"Python\s*3\.(?P<minor>\d+)\+",
    ),
    # "| Python | 3.10 - 3.14 |" in the requirements table.
    (
        "docs-site/docs/getting-started/installation.md",
        r"\|\s*Python\s*\|\s*3\.(?P<minor>\d+)\s*[–-]",
    ),
    # "<span>PYTHON 3.10+</span>" requirement badge.
    (
        "landing/index.html",
        r"PYTHON\s*3\.(?P<minor>\d+)\+",
    ),
    # FAQ answer, duplicated in the JSON-LD block and the visible copy.
    (
        "landing/index.html",
        r"Python\s*3\.(?P<minor>\d+)\s*or\s*newer",
    ),
]


@pytest.mark.parametrize(("relpath", "pattern"), _FLOOR_CLAIMS, ids=lambda v: str(v)[:60])
def test_documented_floor_matches_requires_python(relpath: str, pattern: str) -> None:
    """Every stated floor in the live docs equals `requires-python`'s floor."""
    _, floor_minor = _floor()
    path = ROOT / relpath
    assert path.exists(), f"{relpath} is missing; update _FLOOR_CLAIMS if it moved."

    matches = list(re.finditer(pattern, path.read_text()))
    assert matches, (
        f"No Python-floor claim matching {pattern!r} found in {relpath}. The "
        f"wording probably changed -- update the pattern in _FLOOR_CLAIMS so "
        f"this stays a real check instead of silently passing."
    )

    for m in matches:
        claimed = int(m.group("minor"))
        assert claimed == floor_minor, (
            f"{relpath} advertises Python 3.{claimed} but `requires-python` is "
            f">=3.{floor_minor} (in {m.group(0)!r}). A reader is being told the "
            f"wrong floor -- update the doc, or the floor, so they agree."
        )


def test_landing_faq_copies_stay_in_sync() -> None:
    """The landing FAQ answer exists twice: JSON-LD and visible copy.

    They must carry the same requirement text, or the structured data Google
    reads disagrees with the page a human reads.
    """
    text = (ROOT / "landing" / "index.html").read_text()
    claims = re.findall(r"Python\s*3\.(\d+)\s*or\s*newer", text)
    assert len(claims) == 2, (
        f"Expected the FAQ requirement sentence twice (JSON-LD + visible copy), "
        f"found {len(claims)}. If the page was restructured, update this test."
    )
    assert claims[0] == claims[1], (
        f"Landing FAQ disagrees with itself: structured data says 3.{claims[0]}, "
        f"visible copy says 3.{claims[1]}."
    )


def test_classifiers_start_at_the_floor() -> None:
    """The lowest `Programming Language :: Python` classifier is the floor.

    A classifier list that starts above `requires-python` tells PyPI's filters
    a different story than the installer enforces.
    """
    _, floor_minor = _floor()
    minors = _supported_minors()
    assert minors, "No `Programming Language :: Python :: X.Y` classifiers found."
    assert minors[0] == floor_minor, (
        f"Lowest Python classifier is 3.{minors[0]} but `requires-python` is "
        f">=3.{floor_minor}. pip would install on 3.{floor_minor} while PyPI's "
        f"metadata claims it is unsupported."
    )
    assert minors == list(range(minors[0], minors[-1] + 1)), (
        f"Python classifiers have a gap: {minors}. Either a version is genuinely "
        f"unsupported (then `requires-python` should say so) or one was forgotten."
    )


def test_mypy_comment_states_the_real_floor() -> None:
    """The [tool.mypy] rationale must not misname the floor it contrasts with.

    `python_version` is deliberately *above* the floor (numpy's stubs need a
    3.12 target), and the comment explains why by referring back to
    `requires-python`. That reference is what went stale in #439: the reasoning
    stayed correct while the number it cited did not, which makes a reader
    check the floor, find a different value, and distrust the whole comment.
    """
    text = PYPROJECT.read_text()
    m = re.search(r"\[tool\.mypy\]\n(?P<body>(?:#.*\n|\s*\n)*)", text)
    assert m, "Could not locate the [tool.mypy] comment block."
    comment = m.group("body")

    _, floor_minor = _floor()
    # Any "3.N floor" phrasing in the rationale must name the true floor.
    for cited in re.findall(r"3\.(\d+)\s+floor", comment):
        assert int(cited) == floor_minor, (
            f"[tool.mypy] comment calls 3.{cited} the floor, but `requires-python` "
            f"is >=3.{floor_minor}. Update the comment (see #439)."
        )
