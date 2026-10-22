"""Every workflow action is pinned to a full commit SHA (#848).

Tags such as ``release/v1`` are mutable. The release workflow runs
``pypa/gh-action-pypi-publish`` with ``id-token: write``, so a moved tag would
run with trusted-publishing credentials. Dependabot (``.github/dependabot.yml``)
keeps the SHAs current.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
USES = re.compile(r"^\s*(?:-\s+)?uses:\s*(\S+)", re.M)
SHA = re.compile(r"@[0-9a-f]{40}$")


def _uses():
    for wf in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        text = "\n".join(
            line for line in wf.read_text().splitlines() if not line.lstrip().startswith("#")
        )
        for ref in USES.findall(text):
            if not ref.startswith("./"):  # local reusable workflows have no ref
                yield wf.name, ref


def test_all_actions_are_pinned_to_commit_shas():
    unpinned = [f"{wf}: {ref}" for wf, ref in _uses() if not SHA.search(ref)]
    assert not unpinned, "pin to a full commit SHA with the tag in a comment:\n" + "\n".join(
        unpinned
    )


def test_dependabot_keeps_pinned_actions_current():
    cfg = (ROOT / ".github" / "dependabot.yml").read_text()
    assert "package-ecosystem: github-actions" in cfg
