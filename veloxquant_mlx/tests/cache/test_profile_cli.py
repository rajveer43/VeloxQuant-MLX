"""Tests for ``veloxquant profile`` argument handling and refusal rules (#45).

These cover everything up to the model load. Actually running the profiler
needs model weights, so that path is exercised manually rather than in CI.
"""

from __future__ import annotations

import pytest

from veloxquant_mlx.cache.registry import DEFAULT_SERVE_METHOD
from veloxquant_mlx.cli import profile as profile_cli


def test_default_method_is_the_servable_one():
    args = profile_cli.build_parser().parse_args(["--model", "some/model"])
    assert args.method == DEFAULT_SERVE_METHOD
    assert args.bits == 2
    assert args.max_tokens == 64


def test_validate_method_rejects_standalone():
    """turboquant_prod isn't an mlx_lm.models.cache.KVCache — can't be profiled live."""
    with pytest.raises(SystemExit) as excinfo:
        profile_cli.validate_method("turboquant_prod")

    message = str(excinfo.value)
    assert "cannot be profiled through a live model run" in message


def test_validate_method_rejects_unknown():
    with pytest.raises(SystemExit):
        profile_cli.validate_method("not_a_real_method")


def test_validate_method_accepts_servable():
    profile_cli.validate_method(DEFAULT_SERVE_METHOD)


def test_build_config_applies_overrides():
    args = profile_cli.build_parser().parse_args(
        ["--model", "m/x", "--method", "kivi", "--bits", "3", "--set", "kivi_group_size=64"]
    )
    config = profile_cli.build_config(args)
    assert config.method == "kivi"
    assert config.bit_width_inlier == 3
    assert config.kivi_group_size == 64


def test_parse_overrides_rejects_reserved_fields():
    with pytest.raises(SystemExit):
        profile_cli.parse_overrides(["method=kivi"])


def test_parse_overrides_rejects_malformed_pair():
    with pytest.raises(SystemExit):
        profile_cli.parse_overrides(["not-a-kv-pair"])
