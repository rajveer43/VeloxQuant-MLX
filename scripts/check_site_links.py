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
import time
import urllib.error
import urllib.request
from pathlib import Path

SITE = "https://veloxquant-mlx.netlify.app"
URL_RE = re.compile(rf"{re.escape(SITE)}[^\s)\"'<>\]]*")
ANCHOR_RE = re.compile(r"\[[^\]]*\]\(#([^)]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.M)
# Trailing punctuation that belongs to the prose, not the URL.
TRAILING = ".,;:!?"

# Netlify throttles concurrent bursts, and a checker that reports throttling as
# a broken link fails CI on working URLs. Keep concurrency low, retry transport
# errors, and only trust an HTTP status the server actually returned.
TIMEOUT = 15
RETRIES = 3
BACKOFF = 1.0
WORKERS = 4
# Whole-run budget. Past this the remaining URLs are reported unreachable
# rather than retried: a link check is not worth a CI slot that hangs, and an
# unreachable URL never fails the build anyway.
DEADLINE_S = 180


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


_deadline = float("inf")


def _fetch(url: str, method: str) -> int:
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "link-check"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.status


def check_url(url: str) -> tuple[str, int | str]:
    """Resolve one URL, retrying transient network failures.

    A link check that reports every dropped connection as a broken link is
    worse than no check: it fails CI on working links and trains people to
    ignore it. Netlify throttles bursts, so timeouts and RemoteDisconnected
    are expected noise rather than evidence about the link. Only an HTTP
    status the server actually returned counts as a verdict; transport errors
    are retried with backoff and reported only if every attempt fails.
    """
    last: int | str = "unknown"
    for attempt in range(RETRIES):
        if time.monotonic() > _deadline:
            return url, last if last != "unknown" else "SkippedPastDeadline"
        try:
            return url, _fetch(url, "HEAD")
        except urllib.error.HTTPError as e:
            # A status the server actually returned is a verdict, not noise --
            # report it without retrying. The exception is a host that rejects
            # HEAD while serving GET; confirm those with a GET before believing
            # the link is broken.
            if e.code not in {403, 405, 501}:
                return url, e.code
            try:
                return url, _fetch(url, "GET")
            except urllib.error.HTTPError as e2:
                return url, e2.code
            except Exception:  # noqa: BLE001 - transport failure, fall through to retry
                last = type(e).__name__
        except Exception as e:  # noqa: BLE001 - transport failure, retry
            last = type(e).__name__
        if attempt < RETRIES - 1:
            time.sleep(BACKOFF * (2**attempt))
    return url, last


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

    unreachable: list[str] = []
    if not args.offline and urls:
        global _deadline
        _deadline = time.monotonic() + DEADLINE_S
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for url, status in ex.map(check_url, urls):
                where = ", ".join(str(p) for p in sorted(set(urls[url])))
                if isinstance(status, int):
                    # urlopen follows redirects, so a 2xx here means the URL
                    # resolves to a real page. Anything else is the server's
                    # own verdict and is a genuine broken link.
                    if not 200 <= status < 300:
                        failures.append(f"{where}: {status} {url}")
                else:
                    # Never resolved after RETRIES attempts -- DNS, TLS, timeout
                    # or a dropped connection. That says nothing about whether
                    # the link is correct, so it must not fail the build; a
                    # flaky checker gets ignored, which is worse than none.
                    unreachable.append(f"{where}: {status} {url}")

    if unreachable:
        print(
            f"\n{len(unreachable)} URL(s) unreachable after {RETRIES} attempts "
            "(network, not necessarily broken links):",
            file=sys.stderr,
        )
        for u in sorted(unreachable):
            print(f"  {u}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} broken link(s):", file=sys.stderr)
        for f in sorted(failures):
            print(f"  {f}", file=sys.stderr)
        return 1

    print("All links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
