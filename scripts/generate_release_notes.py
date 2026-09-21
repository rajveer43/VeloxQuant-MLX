"""Generate a terse, one-line-per-entry RELEASE_NOTES.md from CHANGELOG.md.

CHANGELOG.md (5000+ lines and growing) is optimized for git-blame archaeology:
full Conventional-Commit subjects grouped by category, each with a linked
commit/PR. That's the wrong shape for "what changed between 0.90.0 and
0.91.0?" -- this script mechanically re-derives a terser view from the same
already-parsed sections python-semantic-release writes, so there is nothing
to hand-author or keep in sync by hand. See #475.

What it does NOT do: this only reads/writes local files. It is not wired
into release.yml -- run it manually (e.g. before a docs push) or add it as a
release step later once the terse format has been used for a few releases.

Usage:
    python scripts/generate_release_notes.py [--limit N]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def github_anchor(heading: str) -> str:
    """Reproduce GitHub's Markdown heading-to-anchor slug algorithm.

    Lowercase, drop everything but word chars/spaces/hyphens, collapse
    whitespace to single hyphens. Verified against GitHub's rendering of
    "## v0.91.0 (2026-09-21)" -> "#v0910-2026-09-21" (the dots in the
    version are dropped entirely, not turned into hyphens).
    """
    slug = heading.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"\s+", "-", slug.strip())


ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
OUTPUT = ROOT / "RELEASE_NOTES.md"

# Matches "## v0.91.0 (2026-09-21)" -- semantic-release's auto-generated
# heading. Deliberately does not match the older hand-written
# "## [<version>] — <date>" style or "## [Unreleased]" (see release.yml's
# matching awk step for the same two-format note).
VERSION_HEADING = re.compile(r"^## v(?P<version>\S+) \((?P<date>[^)]+)\)\s*$")
CATEGORY_HEADING = re.compile(r"^### (?P<category>.+)$")
BULLET_START = re.compile(r"^- (?:\*\*(?P<scope>[^*]+)\*\*: )?(?P<text>.+)$")
# A bullet's commit/PR links can appear on their own line(s) once the subject
# wraps; recognize and drop those rather than appending them to the subject.
LINK_ONLY_LINE = re.compile(r"^\s*\(\[|^\s*\[`")
# ...or the link can start mid-line, right after the wrapped subject text on
# the same continuation line (e.g. "...mask=\"causal\" (issue\n  #370) ([#392](...").
# Strip from the first "([`<pr-num>`](" or "([`<sha>`](" onward so only the
# subject prose survives.
TRAILING_LINK = re.compile(r"\s*\(\[[`#].*$")

# Order controls the terse summary's section order; category names are
# python-semantic-release's own (from commit-type -> section mapping in
# pyproject.toml's [tool.semantic_release.changelog] / its defaults).
CATEGORY_ORDER = [
    "Features",
    "Bug Fixes",
    "Performance Improvements",
    "Documentation",
    "Continuous Integration",
    "Code Style",
    "Refactoring",
    "Testing",
    "Chores",
]


def parse_changelog(text: str) -> list[dict]:
    """Return [{version, date, categories: {name: [subject, ...]}}, ...]."""
    versions: list[dict] = []
    current: dict | None = None
    current_category: str | None = None
    pending_subject: list[str] | None = None

    def flush_bullet() -> None:
        nonlocal pending_subject
        if pending_subject and current is not None and current_category is not None:
            subject = " ".join(pending_subject).strip()
            current["categories"].setdefault(current_category, []).append(subject)
        pending_subject = None

    for line in text.splitlines():
        v = VERSION_HEADING.match(line)
        if v:
            flush_bullet()
            current = {"version": v["version"], "date": v["date"], "categories": {}}
            versions.append(current)
            current_category = None
            continue

        if line.startswith("## "):
            # "## [Unreleased]" or any other top-level heading: not a
            # released version, so nothing after it belongs to `current`
            # until the next real version heading is seen.
            flush_bullet()
            current = None
            current_category = None
            continue

        if current is None:
            continue

        c = CATEGORY_HEADING.match(line)
        if c:
            flush_bullet()
            current_category = c["category"].strip()
            continue

        b = BULLET_START.match(line)
        if b:
            flush_bullet()
            scope = b["scope"]
            text_part = TRAILING_LINK.sub("", b["text"].strip())
            pending_subject = [f"**{scope}**: {text_part}" if scope else text_part]
            if text_part != b["text"].strip():
                flush_bullet()
            continue

        if pending_subject is not None:
            if LINK_ONLY_LINE.match(line) or not line.strip():
                flush_bullet()
            else:
                stripped = TRAILING_LINK.sub("", line.strip())
                pending_subject.append(stripped)
                if stripped != line.strip():
                    # The link start was found mid-line, so this continuation
                    # line ends the subject -- nothing after it is prose.
                    flush_bullet()

    flush_bullet()
    return versions


def render(versions: list[dict], limit: int | None) -> str:
    lines = [
        "# Release Notes",
        "",
        "Terse, per-version summary generated from `CHANGELOG.md` Conventional"
        " Commit subjects -- see `scripts/generate_release_notes.py`. For full"
        " detail (why a change was made, all commits/PRs), follow the link to"
        " the matching `CHANGELOG.md` section.",
        "",
    ]

    shown = versions if limit is None else versions[:limit]
    for entry in shown:
        heading = f"v{entry['version']} ({entry['date']})"
        anchor = github_anchor(heading)
        lines.append(f"## {heading}")
        lines.append("")
        # CHANGELOG.md legitimately lists the same subject twice when the
        # same commit landed on more than one merged PR (e.g. a cherry-pick
        # or a squash-then-recommit) -- semantic-release parses every commit
        # since the last tag, not one-per-subject. That duplication is exactly
        # the noise this terse view exists to strip, so dedupe per version
        # while keeping first-seen order.
        seen: set[str] = set()
        ordered_categories = list(CATEGORY_ORDER) + [
            c for c in entry["categories"] if c not in CATEGORY_ORDER
        ]
        for category in ordered_categories:
            for subject in entry["categories"].get(category, []):
                if subject in seen:
                    continue
                seen.add(subject)
                lines.append(f"- {subject}")
        lines.append("")
        lines.append(f"[Full changelog entry](CHANGELOG.md#{anchor})")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only include the N most recent versions (default: all).",
    )
    args = parser.parse_args()

    versions = parse_changelog(CHANGELOG.read_text())
    OUTPUT.write_text(render(versions, args.limit))
    print(f"Wrote {OUTPUT.relative_to(ROOT)} ({len(versions)} versions).")


if __name__ == "__main__":
    main()
