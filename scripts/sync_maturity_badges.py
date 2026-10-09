"""Keep the maturity badge on each per-method docs page in sync with the registry.

The badge is the line between ``<!-- maturity:start -->`` and
``<!-- maturity:end -->`` directly under the page's H1. Maturity is read from
``veloxquant_mlx.cache.registry`` so the docs cannot disagree with
``veloxquant methods``.

Usage:
    python scripts/sync_maturity_badges.py          # rewrite pages
    python scripts/sync_maturity_badges.py --check  # exit 1 if any page is stale
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "docs-site" / "docs" / "algorithms"

#: Docs page id -> registry method, where the id is not just the name with
#: underscores for dashes. Pages with no registry method (commvq, rabitq,
#: ratequant, overview, cross-model-transfer) are skipped, not badged.
_ID_TO_METHOD = {
    "rvq": "turboquant_rvq",
    "polarquant": "polar",
    "age-tiered": "age_tiered",
    "kivi-sink": "kivi_sink",
}

_COLOR = {"stable": "22c55e", "beta": "eab308", "experimental": "f97316"}
_BLOCK = re.compile(r"<!-- maturity:start -->.*?<!-- maturity:end -->\n?", re.S)
_H1 = re.compile(r"^# .+\n", re.M)


def badge(maturity: str) -> str:
    """The marked block for one maturity value."""
    img = f"https://img.shields.io/badge/maturity-{maturity}-{_COLOR[maturity]}?style=flat-square"
    return (
        "<!-- maturity:start -->\n"
        f"![Maturity: {maturity}]({img}) "
        "[What do these labels mean?](/docs/algorithms/overview#maturity)\n"
        "<!-- maturity:end -->\n"
    )


def render(text: str, maturity: str) -> str:
    """Return ``text`` with its maturity block inserted or replaced."""
    block = badge(maturity)
    if _BLOCK.search(text):
        return _BLOCK.sub(lambda _m: block, text, count=1)
    h1 = _H1.search(text)
    if h1 is None:
        raise ValueError("no H1 found")
    return text[: h1.end()] + "\n" + block + text[h1.end() :]


def page_methods() -> dict[Path, str]:
    """Map each badge-able docs page to its registry method name."""
    from veloxquant_mlx.cache.registry import all_method_names

    known = set(all_method_names())
    out: dict[Path, str] = {}
    for path in sorted(PAGES.glob("*.md")):
        m = re.search(r"^id: (\S+)", path.read_text(), re.M)
        if m is None:
            continue
        method = _ID_TO_METHOD.get(m.group(1), m.group(1).replace("-", "_"))
        if method in known:
            out[path] = method
    return out


def main(argv: list[str]) -> int:
    """Rewrite pages, or with ``--check`` report stale ones."""
    from veloxquant_mlx.cache.registry import static_method_info

    check = "--check" in argv
    stale = []
    for path, method in page_methods().items():
        old = path.read_text()
        new = render(old, static_method_info(method).maturity.value)
        if new != old:
            stale.append(path.name)
            if not check:
                path.write_text(new)
    if stale:
        print(("stale: " if check else "updated: ") + ", ".join(stale))
    return 1 if (check and stale) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
