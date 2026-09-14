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
    field_is_relevant,
    get_method,
    list_methods,
    probe_serve_tier,
)

# Locked against the audit in issue #27, re-verified by probe.
# 43 as of age_tiered (issue #256): position/age-gated 3-tier precision.
EXPECTED_TOTAL = 43
EXPECTED_CRASHING = {
    "turboquant_prod",
    "turboquant_mse",
    "polar",
    "qjl",
    "spectral",
}

# Eviction/compression caches that deliberately return is_trimmable() -> False
# (17e83c3): trim() would roll back the base class's offset bookkeeping without
# reverting internal per-token eviction state. They serve correctly — the probe
# used to misreport them as CRASHES, which told users 15 working methods were
# unavailable (#152).
EXPECTED_NOT_TRIMMABLE = {
    "age_tiered",
    "amc",
    "anchorkv",
    "cam",
    "chunkkv",
    "curdkv",
    "h2o",
    "keyformer",
    "knorm",
    "kvzip",
    "morphkv",
    "nestedkv",
    "pyramidkv",
    "qfilters",
    "rocketkv",
    "squeeze",
    "streaming_llm",
    "tova",
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


def test_not_trimmable_methods_are_servable():
    """The 15 eviction caches serve correctly; only trim() is unavailable (#152).

    This is the regression guard for the bug that blocked the release gate:
    the probe conflated "refuses to be trimmed" with "crashes", so these
    methods were reported as unavailable in the control panel and gated out
    of `veloxquant serve`.
    """
    not_trimmable = {i.name for i in list_methods() if i.serve_tier is ServeTier.NOT_TRIMMABLE}
    assert not_trimmable == EXPECTED_NOT_TRIMMABLE

    for info in list_methods():
        if info.serve_tier is ServeTier.NOT_TRIMMABLE:
            assert info.serve_tier.is_servable, f"{info.name} must remain servable"
            assert not info.serve_tier.is_trimmable
            assert info.unsupported_reason, f"{info.name} must explain the limitation"


def test_servable_and_trimmable_are_independent():
    """A tier may be servable-but-not-trimmable; only CRASHES is unservable."""
    assert ServeTier.NOT_TRIMMABLE.is_servable
    assert not ServeTier.NOT_TRIMMABLE.is_trimmable
    assert not ServeTier.CRASHES.is_servable
    assert ServeTier.ACCOUNTING_ONLY.is_servable
    assert ServeTier.ACCOUNTING_ONLY.is_trimmable


def test_not_trimmable_label_does_not_read_as_unavailable():
    """The UI string must not tell users a working method is unavailable."""
    label = ServeTier.NOT_TRIMMABLE.label
    assert "available" in label and "not available" not in label


def test_no_method_claims_honest_bytes_yet():
    """Guards #27's credibility rule.

    Until compressed storage is real (#27 option (d)), no method may advertise
    honest byte accounting. Flipping a method to HONEST_BYTES without doing the
    storage work would put a false memory-savings claim in the UI.
    """
    assert all(i.serve_tier is not ServeTier.HONEST_BYTES for i in list_methods())


def test_every_servable_method_is_accounting_only():
    """No servable method may claim honest compressed bytes yet (#27 option (d)).

    NOT_TRIMMABLE is accepted alongside ACCOUNTING_ONLY: it is an orthogonal
    fact about ``trim()`` support, not a byte-accounting claim (#152). What
    this test actually guards is that nothing reaches HONEST_BYTES early.
    """
    for info in list_methods(servable_only=True):
        assert info.serve_tier in (
            ServeTier.ACCOUNTING_ONLY,
            ServeTier.NOT_TRIMMABLE,
        ), f"{info.name} is servable at unexpected tier {info.serve_tier}"


def test_default_serve_method_is_servable():
    """The launcher default must work under ``mlx_lm.server``.

    This used to also assert the *library* default was crash-tier, which was
    true when the library defaulted to ``turboquant_prod``. Since f6e9434 it
    defaults to ``turboquant_rvq``, which is servable — so that assertion was
    pinning a fact that had already stopped being true rather than protecting
    anything. What matters is only that the serve default itself is servable.
    """
    assert get_method(DEFAULT_SERVE_METHOD).serve_tier.is_servable

    from veloxquant_mlx.cache.base import KVCacheConfig

    # The library default must also be servable — if it ever regresses to a
    # crash-tier method, `veloxquant serve` and the three-line README example
    # would disagree about what works.
    assert get_method(KVCacheConfig().method).serve_tier.is_servable


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


class TestFieldIsRelevant:
    """#345: KVCacheConfig has no built-in way to tell a field is irrelevant
    to the active method (e.g. kivi_group_size while method="h2o"). These pin
    field_is_relevant's three lookup tiers: curated _CONFIG_FIELDS entries,
    the prefix fallback for uncurated methods, and always-relevant generics.
    """

    def test_generic_fields_always_relevant(self):
        for method in ("h2o", "kivi", "turboquant_rvq"):
            assert field_is_relevant(method, "bit_width_inlier")
            assert field_is_relevant(method, "seed")

    def test_field_without_underscore_is_generic(self):
        assert field_is_relevant("h2o", "capacity")

    def test_curated_method_uses_explicit_list(self):
        assert field_is_relevant("kivi", "kivi_group_size")
        assert not field_is_relevant("kivi", "svdq_rank")

    def test_curated_method_rejects_other_methods_own_field(self):
        assert not field_is_relevant("svdq", "kivi_group_size")

    def test_uncurated_method_falls_back_to_prefix(self):
        # h2o is not in _CONFIG_FIELDS, so this exercises the prefix branch.
        assert field_is_relevant("h2o", "h2o_budget")
        assert not field_is_relevant("h2o", "tova_budget")

    def test_prefix_alias_methods(self):
        """snapkv/streaming_llm/pyramidkv/nsnquant fields don't share the
        method name. nsnquant (nsn_*) found via
        test_uncurated_methods_still_expose_their_own_fields, which failed
        before _FIELD_PREFIX_ALIAS had an entry for it -- same root cause
        as the gear bug (issue #9), just the alias-lookup side of it rather
        than the config_fields-fallback side."""
        assert field_is_relevant("snapkv", "snap_budget")
        assert field_is_relevant("streaming_llm", "stream_n_sink")
        assert field_is_relevant("pyramidkv", "pyramid_beta")
        assert field_is_relevant("nsnquant", "nsn_bits")
        assert not field_is_relevant("snapkv", "stream_n_sink")

    def test_every_config_fields_entry_exists_on_the_dataclass(self):
        """Catches _CONFIG_FIELDS drifting from KVCacheConfig's real fields --
        e.g. a field renamed in base.py but not updated in the registry."""
        import dataclasses

        from veloxquant_mlx.cache.base import KVCacheConfig
        from veloxquant_mlx.cache.registry import _CONFIG_FIELDS

        valid = {f.name for f in dataclasses.fields(KVCacheConfig)}
        for method, fields in _CONFIG_FIELDS.items():
            for name in fields:
                assert name in valid, (method, name)

    def test_uncurated_methods_still_expose_their_own_fields(self):
        """Regression for the gear bug (issue #9 in VeloxQuant-Studio):
        get_method().config_fields used to fall back to _GENERIC_FIELDS only
        for any method missing from _CONFIG_FIELDS, silently hiding every
        method-specific knob from field_schema (what the macOS app's
        parameter editor renders from) -- even though field_is_relevant()
        already promised a name-prefix fallback for exactly this case.

        gear has 6 real fields (gear_bits, gear_rank, ...) that were
        completely invisible to any UI before this. Every method not in
        _CONFIG_FIELDS must expose at least one of its own prefixed fields.
        """
        from veloxquant_mlx.cache.registry import _CONFIG_FIELDS, _GENERIC_FIELDS

        for method in all_method_names():
            if method in _CONFIG_FIELDS:
                continue
            info = get_method(method)
            method_specific = [f for f in info.config_fields if f not in _GENERIC_FIELDS]
            assert method_specific, (
                f"{method} is not in _CONFIG_FIELDS and exposed no "
                f"method-specific fields: {info.config_fields}"
            )

    def test_gear_config_fields_include_all_six_real_knobs(self):
        """gear specifically (issue #9): the method this bug was found on."""
        fields = set(get_method("gear").config_fields)
        assert fields == {
            "bit_width_inlier",
            "seed",
            "gear_bits",
            "gear_rank",
            "gear_energy_threshold",
            "gear_sparse_fraction",
            "gear_group_size",
            "gear_quantize_values",
        }

    def test_kivi_sink_includes_n_sink_tokens(self):
        """Regression for VeloxQuant-Studio issue #14: kivi_sink IS in
        _CONFIG_FIELDS (unlike gear/#9's uncurated-method bug), but its
        explicit list was missing n_sink_tokens -- SinkProtectedKVCache's
        own sink-count knob (default 5), read directly in its __init__.
        Because the name has no kivi_sink_ prefix, field_is_relevant's
        prefix fallback would never have caught it either; a curated
        method's list is used verbatim, with no fallback.
        """
        fields = set(get_method("kivi_sink").config_fields)
        assert "n_sink_tokens" in fields
        assert field_is_relevant("kivi_sink", "n_sink_tokens")

    def test_kvquant_config_fields_include_n_sink(self):
        """Regression for VeloxQuant-Studio issue #16: kvquant IS in
        _CONFIG_FIELDS, but its explicit list was missing kvquant_n_sink --
        KVQuantKVCache's Attention Sink-Aware knob (default 1, paper §3.5),
        read directly in its __init__ and documented/tested elsewhere in the
        repo. Because the field has a kvquant_ prefix, field_is_relevant's
        prefix fallback *would* have caught it for an uncurated method, but
        a curated method's list is used verbatim, with no fallback.
        """
        fields = set(get_method("kvquant").config_fields)
        assert "kvquant_n_sink" in fields
        assert field_is_relevant("kvquant", "kvquant_n_sink")

    def test_tuple_valued_fields_describe_as_array_not_unknown(self):
        """Regression for VeloxQuant-Studio issue #17: describe_field()'s
        type-mapping dict only covered int/float/bool/str, so any
        KVCacheConfig field annotated bare `tuple` (kvtc_bit_choices,
        svdq_bit_schedule) fell through to `"type": "unknown"`. That alone
        was mostly cosmetic (the macOS app's editor renders every field as a
        generic text field regardless of `type`), but Swift's `JSONValue`
        decoder only recognizes bool/int/double/string and silently
        collapses anything else -- including a real JSON array default like
        `[0, 1, 2, 3, 4, 6, 8]` -- to `.null`, which the app then renders as
        a blank default instead of the real one. `"array"` at least gives a
        future Swift-side switch on `type` something correct to key off of.
        """
        from veloxquant_mlx.cache.registry import describe_field

        for name, expected_default in [
            ("kvtc_bit_choices", (0, 1, 2, 3, 4, 6, 8)),
            ("svdq_bit_schedule", (8, 4, 2, 1, 1, 0, 0, 0)),
        ]:
            desc = describe_field(name)
            assert desc["type"] == "array", (name, desc)
            assert desc["default"] == expected_default, (name, desc)
