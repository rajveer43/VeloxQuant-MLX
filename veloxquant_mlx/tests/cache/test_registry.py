"""Tests for the KV-cache method registry (#33 Phase 0).

The point of these tests is drift detection. The registry drives which methods
the control panel offers and which ones ``veloxquant serve`` refuses, so a
cache silently changing base class must fail the build rather than quietly
change what the UI promises.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from veloxquant_mlx.cache.registry import (
    DEFAULT_SERVE_METHOD,
    MethodFamily,
    ServeTier,
    all_method_names,
    get_method,
    list_methods,
    probe_serve_tier,
)

# Locked against the audit in issue #27, re-verified by probe.
EXPECTED_TOTAL = 40
EXPECTED_CRASHING = {
    "turboquant_prod",
    "turboquant_mse",
    "polar",
    "qjl",
    "spectral",
}


def test_all_methods_discovered():
    assert len(all_method_names()) == EXPECTED_TOTAL


def test_crash_tier_matches_issue_27():
    """The 35/5 split is a published claim; hold the line on it.

    If this fails because a standalone cache gained the mlx_lm contract, that
    is good news — update EXPECTED_CRASHING and the docs matrix together.
    """
    crashing = {i.name for i in list_methods() if not i.serve_tier.is_servable}
    assert crashing == EXPECTED_CRASHING

    servable = [i for i in list_methods() if i.serve_tier.is_servable]
    assert len(servable) == EXPECTED_TOTAL - len(EXPECTED_CRASHING)


def test_no_method_claims_honest_bytes_yet():
    """Guards #27's credibility rule.

    Until compressed storage is real (#27 option (d)), no method may advertise
    honest byte accounting. Flipping a method to HONEST_BYTES without doing the
    storage work would put a false memory-savings claim in the UI.
    """
    assert all(i.serve_tier is not ServeTier.HONEST_BYTES for i in list_methods())


def test_every_servable_method_is_accounting_only():
    for info in list_methods(servable_only=True):
        assert info.serve_tier is ServeTier.ACCOUNTING_ONLY


def test_default_serve_method_is_servable():
    """The launcher default must work, and must differ from the library default."""
    info = get_method(DEFAULT_SERVE_METHOD)
    assert info.serve_tier.is_servable

    from veloxquant_mlx.cache.base import KVCacheConfig

    assert KVCacheConfig().method in EXPECTED_CRASHING, (
        "library default is expected to be crash-tier; if that changed, the "
        "serve default and its accompanying docs note should be revisited"
    )
    assert DEFAULT_SERVE_METHOD != KVCacheConfig().method


def test_unsupported_methods_explain_themselves():
    for info in list_methods():
        if not info.serve_tier.is_servable:
            assert info.unsupported_reason, f"{info.name} refuses without a reason"


def test_probe_is_memoized():
    first = probe_serve_tier("turboquant_rvq")
    assert probe_serve_tier("turboquant_rvq") is first


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        get_method("definitely_not_a_method")


def test_every_method_has_family_and_blurb():
    for info in list_methods():
        assert isinstance(info.family, MethodFamily)
        assert info.blurb and not info.blurb.endswith("KV-cache method."), (
            f"{info.name} is missing an editorial blurb"
        )


def test_filters():
    evictions = list_methods(family=MethodFamily.EVICTION)
    assert evictions and all(i.family is MethodFamily.EVICTION for i in evictions)


def test_servable_methods_sort_first():
    infos = list_methods()
    servable_flags = [i.serve_tier.is_servable for i in infos]
    assert servable_flags == sorted(servable_flags, reverse=True)


def test_adapted_methods_carry_deviation_notes():
    """ "-adapted" is an honesty claim; keep it attached to a real explanation."""
    for name in ("adakv", "a2ats"):
        info = get_method(name)
        assert info.is_adapted
        assert info.paper_deviation


def test_to_dict_is_json_serializable():
    payload = [i.to_dict() for i in list_methods()]
    round_tripped = json.loads(json.dumps(payload))
    assert len(round_tripped) == EXPECTED_TOTAL
    assert {"name", "serve_tier", "is_servable", "docs_url"} <= set(round_tripped[0])


def test_methods_cli_json_contract():
    """The macOS panel decodes this; breaking its shape breaks the app."""
    proc = subprocess.run(
        [sys.executable, "-m", "veloxquant_mlx", "methods", "--json"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == 1
    assert payload["accounting_only"] is True
    assert payload["default_serve_method"] == DEFAULT_SERVE_METHOD
    assert len(payload["methods"]) == EXPECTED_TOTAL
