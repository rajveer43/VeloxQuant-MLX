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
            "0.42.1 -> 0.45.0" in w and "headline prose" in w for w in sync_release_badges._warnings
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


_VERSION_HTML = """<!doctype html>
<html>
<head>
  <script type="application/ld+json">
  {
    "@type": "SoftwareApplication",
    "softwareVersion": "0.70.0",
    "license": "https://opensource.org/licenses/MIT"
  }
  </script>
</head>
<body>
  <div class="card-meta">v0.67.0 · Metal Linked</div>
  <div class="eco-cli-line">&gt;_ @veloxquant/sdk (v0.7.1)</div>
</body>
</html>
"""


class TestSyncLandingVersions:
    """#941: the JSON-LD softwareVersion and the Studio mockup label drifted
    to 0.70.0 / v0.67.0 while the package was 0.95.1."""

    def test_rewrites_json_ld_and_mockup_label(self, tmp_path, monkeypatch) -> None:
        landing_html = tmp_path / "index.html"
        landing_html.write_text(_VERSION_HTML)
        monkeypatch.setattr(sync_release_badges, "LANDING_HTML", landing_html)

        sync_release_badges._sync_landing_versions("0.96.0")

        text = landing_html.read_text()
        assert '"softwareVersion": "0.96.0"' in text
        assert "v0.96.0 · Metal Linked" in text
        assert "0.70.0" not in text
        assert "v0.67.0" not in text

    def test_leaves_separately_versioned_sdk_strings_alone(self, tmp_path, monkeypatch) -> None:
        landing_html = tmp_path / "index.html"
        landing_html.write_text(_VERSION_HTML)
        monkeypatch.setattr(sync_release_badges, "LANDING_HTML", landing_html)

        sync_release_badges._sync_landing_versions("0.96.0")

        assert "@veloxquant/sdk (v0.7.1)" in landing_html.read_text()

    def test_is_idempotent(self, tmp_path, monkeypatch) -> None:
        landing_html = tmp_path / "index.html"
        landing_html.write_text(_VERSION_HTML)
        monkeypatch.setattr(sync_release_badges, "LANDING_HTML", landing_html)

        sync_release_badges._sync_landing_versions("0.96.0")
        once = landing_html.read_text()
        sync_release_badges._sync_landing_versions("0.96.0")

        assert landing_html.read_text() == once

    def test_warns_for_each_missing_target(self, tmp_path, monkeypatch) -> None:
        landing_html = tmp_path / "index.html"
        landing_html.write_text("<html><body>nothing to sync</body></html>")
        monkeypatch.setattr(sync_release_badges, "LANDING_HTML", landing_html)
        monkeypatch.setattr(sync_release_badges, "_warnings", [])

        sync_release_badges._sync_landing_versions("0.96.0")

        assert any("softwareVersion not found" in w for w in sync_release_badges._warnings)
        assert any("Metal Linked" in w for w in sync_release_badges._warnings)


def test_real_landing_page_matches_package_version() -> None:
    """Drift guard: the checked-in landing page must carry the package
    version in the two places the release sync owns."""
    version = sync_release_badges._current_version()
    text = (_SCRIPT_PATH.parents[1] / "landing" / "index.html").read_text()

    assert f'"softwareVersion": "{version}"' in text
    assert f"v{version} · Metal Linked" in text
