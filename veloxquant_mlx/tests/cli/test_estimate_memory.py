"""Tests for the ``estimate-memory`` CLI."""

from __future__ import annotations

import json

from veloxquant_mlx.cli import estimate_memory as estimate_memory_cli


def _args(extra: list[str] | None = None) -> list[str]:
    base = ["--n-layers", "8", "--n-query-heads", "8", "--n-kv-heads", "8", "--head-dim", "128"]
    return base + (extra or [])


def test_estimate_memory_json(capsys):
    estimate_memory_cli.main([*_args(), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "model" in payload and "workload" in payload
    strategies = payload["strategies"]
    assert isinstance(strategies, dict)
    assert len(strategies) >= 5
    first = next(iter(strategies.values()))
    for key in ("method", "baseline_bytes", "compressed_bytes", "savings_percent", "confidence"):
        assert key in first


def test_estimate_memory_top_limit(capsys):
    estimate_memory_cli.main([*_args(), "--top", "2", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["strategies"]) == 2


def test_estimate_memory_plain_text(capsys):
    estimate_memory_cli.main([*_args(), "--top", "3"])
    out = capsys.readouterr().out
    assert "method" in out and "zipcache" in out


def test_estimate_memory_batch_scales(capsys):
    estimate_memory_cli.main([*_args(), "--context", "1024", "--batch", "4", "--top", "1", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["workload"]["batch_size"] == 4


def test_estimate_memory_model_config(capsys, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "architectures": ["MistralForCausalLM"],
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "hidden_size": 1024,
            }
        )
    )
    estimate_memory_cli.main(["--model-config", str(config), "--top", "1", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"]["architecture"] == "mistral"


def test_estimate_memory_mha_vs_gqa_baseline(capsys):
    estimate_memory_cli.main(
        [*_args(), "--n-kv-heads", "8", "--n-query-heads", "8", "--top", "1", "--json"]
    )
    mha = json.loads(capsys.readouterr().out)["strategies"]
    estimate_memory_cli.main(
        [*_args(), "--n-kv-heads", "2", "--n-query-heads", "8", "--top", "1", "--json"]
    )
    gqa = json.loads(capsys.readouterr().out)["strategies"]
    base = lambda d: max(e["baseline_bytes"] for e in d.values())  # noqa: E731
    assert base(mha) > base(gqa)
