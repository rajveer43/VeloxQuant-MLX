"""Per-method docs pages must show the registry's maturity (scripts/sync_maturity_badges.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "sync_maturity_badges.py"
_spec = importlib.util.spec_from_file_location("sync_maturity_badges", _SCRIPT)
assert _spec is not None and _spec.loader is not None
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


def test_render_is_idempotent_and_replaces():
    text = "---\nid: x\n---\n\n# X\n\nbody\n"
    once = sync.render(text, "beta")
    assert once.count("maturity:start") == 1
    assert sync.render(once, "beta") == once
    assert "maturity-stable" in sync.render(once, "stable")


def test_docs_pages_match_registry():
    from veloxquant_mlx.cache.registry import static_method_info

    pages = sync.page_methods()
    if not pages:
        pytest.skip("docs-site not present")
    for path, method in pages.items():
        want = sync.render(path.read_text(), static_method_info(method).maturity.value)
        assert path.read_text() == want, f"{path.name} is stale; run scripts/sync_maturity_badges.py"
