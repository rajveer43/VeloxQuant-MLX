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
    assert payload["endpoints"]["kv_stats"] == "http://127.0.0.1:9999/v1/kv/stats"

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


def test_set_irrelevant_field_warns_but_still_applies(capsys):
    """#345: KVCacheConfig accepts any field regardless of method, so
    --set kivi_group_size=64 with --method h2o previously succeeded with no
    indication the flag had no effect. It must now warn -- but the value
    still applies (unused, harmless) rather than being silently dropped or
    hard-rejected, since the relevance table does not yet cover every method."""
    overrides = serve_cli.parse_overrides(["kivi_group_size=64"], method="h2o")

    assert overrides == {"kivi_group_size": 64}
    err = capsys.readouterr().err
    assert "'kivi_group_size' has no effect for method 'h2o'" in err


def test_set_relevant_field_does_not_warn(capsys):
    overrides = serve_cli.parse_overrides(["kivi_group_size=64"], method="kivi")

    assert overrides == {"kivi_group_size": 64}
    err = capsys.readouterr().err
    assert err == ""


def test_set_generic_field_never_warns(capsys):
    """bit_width_inlier/seed apply to every method, regardless of curation."""
    overrides = serve_cli.parse_overrides(["seed=7"], method="h2o")

    assert overrides == {"seed": 7}
    err = capsys.readouterr().err
    assert err == ""


def test_set_without_method_skips_relevance_check(capsys):
    """method=None (the prior default) preserves old behavior exactly."""
    overrides = serve_cli.parse_overrides(["kivi_group_size=64"])

    assert overrides == {"kivi_group_size": 64}
    err = capsys.readouterr().err
    assert err == ""


def _make_fake_model(n_layers: int = 2, n_heads: int = 2, head_dim: int = 32):
    from types import SimpleNamespace

    hidden_size = n_heads * head_dim
    layers = [
        SimpleNamespace(self_attn=SimpleNamespace(head_dim=head_dim)) for _ in range(n_layers)
    ]
    args = SimpleNamespace(hidden_size=hidden_size, num_attention_heads=n_heads)
    return SimpleNamespace(layers=layers, args=args)


def test_attach_cache_records_live_caches_on_each_call():
    """#36: /v1/kv/stats reads whichever cache list is currently live.

    ResponseGenerator._generate calls model.make_cache() fresh per request
    (see cli/telemetry.py's module docstring) -- attach_cache must record
    the *result of that call*, not just the throwaway probe it uses to count
    layers, or telemetry would describe a discarded cache instead of the
    one actually in use.
    """
    from veloxquant_mlx.cache.base import KVCacheConfig
    from veloxquant_mlx.cli import telemetry

    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=2, seed=42)
    model = _make_fake_model(n_layers=3)

    telemetry._LIVE_CACHES = None
    n_layers = serve_cli.attach_cache(model, config)
    assert n_layers == 3
    # The probe used only to count layers must not itself count as "live" --
    # only a real make_cache() call (mlx_lm's own contract) should.
    assert telemetry._LIVE_CACHES is None

    caches = model.make_cache()
    assert len(caches) == 3
    assert telemetry._LIVE_CACHES is caches
    assert telemetry._LIVE_METHOD == "turboquant_rvq"
    assert telemetry._LIVE_BITS == 2

    # A second request (mlx_lm calls make_cache() fresh per request) replaces
    # the recorded list, not append.
    second = model.make_cache()
    assert telemetry._LIVE_CACHES is second
    assert telemetry._LIVE_CACHES is not caches


def test_attach_cache_does_not_recurse_into_its_own_wrapper(recwarn):
    """Regression test: for_model() probes model.make_cache as a
    hybrid-attention-layer heuristic (see its own docstring). Once
    attach_cache patches model.make_cache, a naive implementation has that
    probe find its own wrapper and recurse until RecursionError -- silently
    caught by for_model's broad `except Exception` and falling back
    correctly by accident, but wasting a full recursion-limit's worth of
    stack frames and an exception catch on every single call.

    This asserts the fix: repeated make_cache() calls on the same model
    produce no warnings at all, not just "eventually falls back okay".
    """
    from veloxquant_mlx.cache.base import KVCacheConfig

    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=2, seed=42)
    model = _make_fake_model(n_layers=3)
    serve_cli.attach_cache(model, config)

    for _ in range(3):
        model.make_cache()

    assert len(recwarn) == 0, [str(w.message) for w in recwarn]


def test_attach_cache_preserves_a_real_native_make_cache():
    """Hybrid-attention models (Qwen3.5/qwen3_next, mamba hybrids) define
    their own make_cache for non-attention slots (see for_model's docstring
    in cache/base.py). attach_cache's restore-around-the-probe trick must
    not lose that native method -- for_model still needs to see and call it,
    just not our own wrapper standing in for it.
    """
    from veloxquant_mlx.cache.base import KVCacheConfig

    model = _make_fake_model(n_layers=2)
    calls = []

    def native_make_cache():
        calls.append(1)
        return  # a plain object's native cache isn't a real hybrid list here

    model.make_cache = native_make_cache
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=2, seed=42)

    # attach_cache's own upfront layer-count probe already calls the native
    # make_cache once, before model.make_cache is even patched.
    serve_cli.attach_cache(model, config)
    assert calls == [1]

    model.make_cache()

    # for_model's probe inside our wrapper (getattr(model, "make_cache",
    # None) -> call it) reached the real native function a second time, not
    # our wrapper calling itself.
    assert calls == [1, 1]
