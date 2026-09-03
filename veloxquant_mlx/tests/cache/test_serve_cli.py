"""Tests for ``veloxquant serve`` argument handling and refusal rules (#27/#34).

These cover everything up to the model load. Actually starting a server needs
model weights, so that path is exercised manually rather than in CI.
"""

from __future__ import annotations

import json

import pytest

from veloxquant_mlx.cache.registry import DEFAULT_SERVE_METHOD, ServeTier, probe_serve_tier
from veloxquant_mlx.cli import serve as serve_cli


def test_default_method_is_the_servable_one():
    args = serve_cli.build_parser().parse_args(["--model", "some/model"])
    assert args.method == DEFAULT_SERVE_METHOD
    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_validate_method_rejects_crash_tier():
    """No silent fp16 fallback: an unservable method must stop the process."""
    with pytest.raises(SystemExit) as excinfo:
        serve_cli.validate_method("turboquant_prod")

    message = str(excinfo.value)
    assert "cannot be served" in message
    assert "No fp16 fallback" in message


def test_validate_method_rejects_unknown():
    with pytest.raises(SystemExit):
        serve_cli.validate_method("not_a_real_method")


def test_validate_method_accepts_servable():
    serve_cli.validate_method(DEFAULT_SERVE_METHOD)


def test_ready_handshake_shape(capsys):
    """The panel parses this line to flip Starting -> Running."""
    args = serve_cli.build_parser().parse_args(
        ["--model", "m/x", "--port", "9999", "--method", "kivi", "--bits", "3"]
    )
    serve_cli.emit_ready(args, n_caches=16)

    line = capsys.readouterr().out.strip()
    assert line.startswith(serve_cli.READY_PREFIX)

    payload = json.loads(line[len(serve_cli.READY_PREFIX) :])
    assert payload["method"] == "kivi"
    assert payload["bits"] == 3
    assert payload["layer_caches"] == 16
    assert payload["endpoints"]["openai_base_url"] == "http://127.0.0.1:9999/v1"

    # The honesty flag must ride along with the numbers, not be optional.
    assert payload["accounting_only"] is True
    assert "not runtime memory saved" in payload["accounting_note"]


def test_ready_handshake_advertises_only_real_endpoints(capsys):
    """#34: do not invent endpoints mlx_lm.server does not serve.

    /health and /metrics are NOT part of mlx_lm.server, so they must not appear
    in the handshake the panel renders copy-buttons from.
    """
    args = serve_cli.build_parser().parse_args(["--model", "m/x"])
    serve_cli.emit_ready(args, n_caches=1)

    line = capsys.readouterr().out.strip()
    payload = json.loads(line[len(serve_cli.READY_PREFIX) :])

    assert "health" not in payload["endpoints"]
    assert "metrics" not in payload["endpoints"]


def test_mlx_server_defaults_are_readable():
    """We inherit mlx_lm's arg defaults; if that breaks, serving breaks."""
    import mlx_lm.server as server_module

    parser = serve_cli._capture_mlx_parser(server_module)
    namespace = parser.parse_args([])

    # Fields mlx_lm.server reads off cli_args at request time.
    for field in (
        "draft_model",
        "pipeline",
        "prompt_cache_size",
        "prompt_cache_bytes",
        "temp",
        "top_p",
    ):
        assert hasattr(namespace, field), f"missing {field}"


def test_mlx_server_args_override_ours():
    args = serve_cli.build_parser().parse_args(
        ["--model", "m/x", "--port", "1234", "--max-tokens", "77"]
    )
    ns = serve_cli._mlx_server_args(args)

    assert ns.model == "m/x"
    assert ns.port == 1234
    assert ns.max_tokens == 77
    assert hasattr(ns, "draft_model")


def test_prompt_cache_size_default_is_ten():
    args = serve_cli.build_parser().parse_args(["--model", "some/model"])
    assert args.prompt_cache_size == 10


def test_prompt_cache_size_flag_reaches_mlx_namespace():
    args = serve_cli.build_parser().parse_args(["--model", "m/x", "--prompt-cache-size", "25"])
    ns = serve_cli._mlx_server_args(args)
    assert ns.prompt_cache_size == 25


def test_prompt_cache_bytes_default_is_none():
    args = serve_cli.build_parser().parse_args(["--model", "some/model"])
    assert args.prompt_cache_bytes is None


def test_prompt_cache_bytes_flag_parses_size_suffix_and_reaches_namespace():
    args = serve_cli.build_parser().parse_args(["--model", "m/x", "--prompt-cache-bytes", "2G"])
    assert args.prompt_cache_bytes == 2_000_000_000

    ns = serve_cli._mlx_server_args(args)
    assert ns.prompt_cache_bytes == 2_000_000_000


def test_prompt_cache_bytes_inert_flag_warns(monkeypatch, capsys):
    """--prompt-cache-bytes is parsed but not applied by the installed
    mlx_lm server (verified by reading mlx_lm/server.py directly: it builds
    LRUPromptCache with only prompt_cache_size). Setting it must warn, not
    silently pretend it works -- this repo's "no silent fallback" rule."""
    monkeypatch.setattr(serve_cli, "run_server", lambda args: None)

    serve_cli.main(["--model", "some/model", "--prompt-cache-bytes", "1G"])

    err = capsys.readouterr().err
    assert "--prompt-cache-bytes is parsed but not applied" in err


def test_prompt_cache_bytes_unset_does_not_warn(monkeypatch, capsys):
    monkeypatch.setattr(serve_cli, "run_server", lambda args: None)

    serve_cli.main(["--model", "some/model"])

    err = capsys.readouterr().err
    assert "--prompt-cache-bytes is parsed but not applied" not in err


def test_not_trimmable_method_gate_used_by_serve_warning():
    """_Provider._load (only reachable with real model weights, see module
    docstring) warns when probe_serve_tier(args.method) is NOT_TRIMMABLE --
    this pins the gating condition itself, since the warning site can't be
    exercised without a real model load."""
    assert probe_serve_tier("h2o") is ServeTier.NOT_TRIMMABLE
    assert probe_serve_tier(DEFAULT_SERVE_METHOD) is not ServeTier.NOT_TRIMMABLE
