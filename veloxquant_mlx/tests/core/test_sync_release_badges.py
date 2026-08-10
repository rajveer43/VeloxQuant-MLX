"""Tests for scripts/sync_release_badges.py's landing-page hero badge sync.

Regression: _sync_landing_pill() used to search landing/assets/main.js for a
`const text = "v<version> — ..."` pattern that never existed there — the real
hero badge is a `data-text` attribute on `#hero-badge` in landing/index.html,
read by initBadgeTyping() in main.js. The mismatch meant the sync silently
no-op'd (a stderr warning, not a failure) on every release, so the badge
drifted from the actual shipped version. This exercises the fixed function
against fixture HTML instead of the real landing page, so a future regression
fails the test suite instead of failing silently in CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "sync_release_badges.py"


def _load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_release_badges", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sync_release_badges = _load_sync_module()

_HERO_BADGE_HTML = """<!doctype html>
<html>
<body>
    <div class="hero-content">
      <div class="badge" id="hero-badge" data-text="v0.42.1 — fused KIVI-style Metal attention kernel shipped">v0.42.1 — fused KIVI-style Metal attention kernel shipped</div>
    </div>
</body>
</html>
"""


class TestSyncLandingPill:
    """Regression: hero badge sync must target landing/index.html's
    #hero-badge div, not a nonexistent main.js constant."""

    def test_rewrites_both_occurrences_of_version(self, tmp_path, monkeypatch) -> None:
        landing_html = tmp_path / "index.html"
        landing_html.write_text(_HERO_BADGE_HTML)
        monkeypatch.setattr(sync_release_badges, "LANDING_HTML", landing_html)

        sync_release_badges._sync_landing_pill("0.45.0")

        text = landing_html.read_text()
        assert 'data-text="v0.45.0 — fused KIVI-style Metal attention kernel shipped"' in text
        assert ">v0.45.0 — fused KIVI-style Metal attention kernel shipped<" in text
        assert "0.42.1" not in text

    def test_preserves_prose_after_version(self, tmp_path, monkeypatch) -> None:
        landing_html = tmp_path / "index.html"
        landing_html.write_text(_HERO_BADGE_HTML)
        monkeypatch.setattr(sync_release_badges, "LANDING_HTML", landing_html)

        sync_release_badges._sync_landing_pill("0.45.0")

        text = landing_html.read_text()
        assert "fused KIVI-style Metal attention kernel shipped" in text

    def test_warns_when_prose_still_describes_previous_version(self, tmp_path, monkeypatch) -> None:
        landing_html = tmp_path / "index.html"
        landing_html.write_text(_HERO_BADGE_HTML)
        monkeypatch.setattr(sync_release_badges, "LANDING_HTML", landing_html)
        monkeypatch.setattr(sync_release_badges, "_warnings", [])

        sync_release_badges._sync_landing_pill("0.45.0")

        assert any(
            "0.42.1 -> 0.45.0" in w and "headline prose" in w
            for w in sync_release_badges._warnings
        )

    def test_no_warning_when_version_unchanged(self, tmp_path, monkeypatch) -> None:
        landing_html = tmp_path / "index.html"
        landing_html.write_text(_HERO_BADGE_HTML)
        monkeypatch.setattr(sync_release_badges, "LANDING_HTML", landing_html)
        monkeypatch.setattr(sync_release_badges, "_warnings", [])

        sync_release_badges._sync_landing_pill("0.42.1")

        assert sync_release_badges._warnings == []

    def test_warns_and_skips_when_badge_missing(self, tmp_path, monkeypatch) -> None:
        landing_html = tmp_path / "index.html"
        landing_html.write_text("<html><body>no badge here</body></html>")
        monkeypatch.setattr(sync_release_badges, "LANDING_HTML", landing_html)
        monkeypatch.setattr(sync_release_badges, "_warnings", [])

        sync_release_badges._sync_landing_pill("0.45.0")

        assert any("hero badge not found" in w for w in sync_release_badges._warnings)

    def test_does_not_reference_main_js(self) -> None:
        """LANDING_JS was removed — the sync has one source of truth now."""
        assert not hasattr(sync_release_badges, "LANDING_JS")


@pytest.mark.parametrize("version", ["0.45.0", "1.0.0", "0.42.10"])
def test_rewrites_to_arbitrary_versions(tmp_path, monkeypatch, version) -> None:
    landing_html = tmp_path / "index.html"
    landing_html.write_text(_HERO_BADGE_HTML)
    monkeypatch.setattr(sync_release_badges, "LANDING_HTML", landing_html)

    sync_release_badges._sync_landing_pill(version)

    text = landing_html.read_text()
    assert f'data-text="v{version} — ' in text
    assert f">v{version} — " in text
