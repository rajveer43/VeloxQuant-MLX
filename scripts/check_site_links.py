#!/usr/bin/env python3
"""Check that every veloxquant-mlx.netlify.app URL in the repo actually resolves.

Exists because a reader hit a 404 before we did. Two links in README.md pointed
at ``/algorithms/cross-model-transfer`` while the docs site serves everything
under ``/docs/`` (``baseUrl: '/docs/'`` in docusaurus.config.ts, and netlify.toml
copies ``docs-site/build/*`` into ``dist/docs/``). Docusaurus' own
``onBrokenLinks: 'throw'`` cannot catch these: they are absolute URLs in files
outside the docs site, so nothing validates them at build time.

Checks two things:

* **Absolute site URLs** in tracked Markdown/HTML resolve (HEAD, following
  redirects). Needs network; skipped with ``--offline``.
* **Markdown anchor links** (``[text](#anchor)``) match a heading in the same
  file. Pure text, always runs.

Usage:
    python scripts/check_site_links.py [--offline] [PATH ...]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

SITE = "https://veloxquant-mlx.netlify.app"
URL_RE = re.compile(rf"{re.escape(SITE)}[^\s)\"'<>\]]*")
ANCHOR_RE = re.compile(r"\[[^\]]*\]\(#([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.M)
# Trailing punctuation that belongs to the prose, not the URL.
TRAILING = ".,;:!?"


def slugify(text: str) -> str:
    """GitHub's heading-anchor rules: strip formatting, lowercase, hyphenate."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Punctuation is deleted in place, NOT collapsed: GitHub turns
    # "Project & governance" into "project--governance" (two hyphens), because
    # the "&" vanishes but the spaces either side each become a hyphen.
    # Collapsing whitespace first would yield one hyphen and flag every such
    # heading as a broken anchor.
    text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"\s", "-", text)


def tracked_files(paths: list[str]) -> list[Path]:
    """Git-tracked .md/.html files, so build output and node_modules stay out."""
    out = subprocess.run(
        ["git", "ls-files", *(paths or ["."])],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [Path(f) for f in out if f.endswith((".md", ".mdx", ".html"))]


def check_url(url: str) -> tuple[str, int | str]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "link-check"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return url, r.status
    except urllib.error.HTTPError as e:
        # Some hosts reject HEAD but serve GET; retry before calling it broken.
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "link-check"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return url, r.status
        except Exception:
            return url, e.code
    except Exception as e:  # noqa: BLE001 - report any failure verbatim
        return url, type(e).__name__


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--offline", action="store_true", help="skip network checks")
    args = ap.parse_args()

    files = tracked_files(args.paths)
    failures: list[str] = []

    # ---- anchors ----
    for f in files:
        if f.suffix not in {".md", ".mdx"}:
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        slugs = {slugify(m.group(2)) for m in HEADING_RE.finditer(text)}
        for m in ANCHOR_RE.finditer(text):
            if m.group(1) not in slugs:
                failures.append(f"{f}: missing anchor #{m.group(1)}")

    # ---- absolute site URLs ----
    urls: dict[str, list[Path]] = {}
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in URL_RE.finditer(text):
            url = m.group(0).rstrip(TRAILING)
            # Glob/placeholder fragments in prose are not real links.
            if "*" in url:
                continue
            urls.setdefault(url, []).append(f)

    print(f"{len(files)} files, {len(urls)} distinct site URLs")

    if not args.offline and urls:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for url, status in ex.map(check_url, urls):
                if status != 200:
                    where = ", ".join(str(p) for p in sorted(set(urls[url])))
                    failures.append(f"{where}: {status} {url}")

    if failures:
        print(f"\n{len(failures)} broken link(s):", file=sys.stderr)
        for f in sorted(failures):
            print(f"  {f}", file=sys.stderr)
        return 1

    print("All links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
