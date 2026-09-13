"""Tests for build_stats_payload() — the /v1/kv/stats presentation logic.

Pure-function tests, no HTTP server or real model needed: they exercise the
same honesty rules as the memory panel (docs/control-panel-enhancements.md
rules 7-9) using fake per-layer cache stand-ins.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from veloxquant_mlx.cli.telemetry import build_stats_payload


def _keys_and_values_cache():
    return SimpleNamespace(
        compressed_key_bytes=100,
        fp16_key_bytes=800,
        compressed_value_bytes=120,
        fp16_value_bytes=800,
    )


def _keys_only_cache():
    return SimpleNamespace(compressed_key_bytes=100, fp16_key_bytes=800)


def _eviction_cache(seen, kept):
    return SimpleNamespace(tokens_seen=seen, tokens_kept=kept)


def _bare_cache():
    """A cache with no telemetry attributes at all (e.g. a crash-tier stand-in)."""
    return SimpleNamespace()


def test_no_live_caches_reports_not_generating_yet():
    payload = build_stats_payload(caches=None, method="turboquant_rvq", bits=2)
    assert payload["coverage"] == "none"
    assert payload["not_reported_reason"] == "no server is generating yet"
    assert payload["keys"] is None
    assert payload["tokens"] is None


def test_keys_and_values_coverage_reports_both_ratios(monkeypatch):
    caches = [_keys_and_values_cache(), _keys_and_values_cache()]
    _patch_coverage(monkeypatch, "keys_and_values")

    payload = build_stats_payload(caches=caches, method="kivi", bits=2)

    assert payload["coverage"] == "keys_and_values"
    assert payload["keys"]["compressed_bytes"] == 200
    assert payload["keys"]["fp16_bytes"] == 1600
    assert payload["keys"]["ratio"] == 8.0
    assert payload["values"]["compressed_bytes"] == 240
    assert payload["values"]["ratio"] == pytest.approx(round(1600 / 240, 2), abs=0.01)
    assert payload["tokens"] is None


def test_keys_only_coverage_never_reports_values(monkeypatch):
    caches = [_keys_only_cache()]
    _patch_coverage(monkeypatch, "keys_only")

    payload = build_stats_payload(caches=caches, method="turboquant_rvq", bits=2)

    assert payload["coverage"] == "keys_only"
    assert payload["keys"] is not None
    # The whole point of this branch: a key-only method must never populate
    # "values", which a naive client could mistake for a whole-cache ratio.
    assert payload["values"] is None


def test_eviction_method_reports_tokens_not_bytes(monkeypatch):
    caches = [_eviction_cache(seen=100, kept=64), _eviction_cache(seen=100, kept=64)]
    _patch_coverage(monkeypatch, "none")

    payload = build_stats_payload(caches=caches, method="h2o", bits=None)

    assert payload["coverage"] == "none"
    assert payload["keys"] is None
    assert payload["values"] is None
    assert payload["tokens"] == {"seen": 200, "retained": 128}
    assert "not_reported_reason" not in payload


def test_method_with_no_counters_at_all_states_absence_not_zero(monkeypatch):
    """A method with neither byte nor token counters must not render as 0.

    This is rule 8 from docs/control-panel-enhancements.md made literal: a
    bare SimpleNamespace has no attributes at all, so a naive implementation
    that used getattr(..., 0) everywhere would silently report all-zero
    tokens here, indistinguishable from "everything was evicted". This test
    fails if that regression is reintroduced.
    """
    caches = [_bare_cache()]
    _patch_coverage(monkeypatch, "none")

    payload = build_stats_payload(caches=caches, method="mystery_method", bits=None)

    assert payload["tokens"] is None
    assert payload["not_reported_reason"] == "this method does not report byte or token counters"


def test_ratio_is_none_without_a_baseline():
    from veloxquant_mlx.cache.registry import TelemetryCoverage
    from veloxquant_mlx.cli.telemetry import _byte_counts

    # compressed bytes present, no fp16 baseline recorded yet (e.g. cache
    # just constructed, no tokens processed) -- ratio must be None, not a
    # divide-by-zero or a fabricated 0.0.
    cache = SimpleNamespace(compressed_key_bytes=0, fp16_key_bytes=0)
    keys, values = _byte_counts([cache], TelemetryCoverage.KEYS_ONLY)
    assert keys["ratio"] is None


def test_memory_block_is_always_measured():
    payload = build_stats_payload(caches=None, method=None, bits=None)
    assert payload["memory"]["source"] == "measured"


def _patch_coverage(monkeypatch, coverage_value: str):
    from veloxquant_mlx.cache.registry import MethodFamily, MethodInfo, ServeTier, TelemetryCoverage

    coverage = TelemetryCoverage(coverage_value)
    fake_info = MethodInfo(
        name="fake",
        family=MethodFamily.QUANTIZATION,
        serve_tier=ServeTier.ACCOUNTING_ONLY,
        blurb="fake method for tests",
        coverage=coverage,
    )
    # build_stats_payload imports get_method from veloxquant_mlx.cache.registry
    # *inside* the function body, so patching the registry module's attribute
    # (rather than veloxquant_mlx.cli.telemetry.get_method, which doesn't
    # exist as a module-level name) is what the function actually resolves.
    monkeypatch.setattr("veloxquant_mlx.cache.registry.get_method", lambda name: fake_info)
