"""Tests for the ``recommend`` CLI in both auto and legacy modes."""

from __future__ import annotations

import json

import pytest

from veloxquant_mlx.cli import recommend as recommend_cli
from veloxquant_mlx.tools import mac_recommender


def _recommend_args(extra: list[str]) -> list[str]:
    return ["--auto", "--no-probe", *extra]


def _auto_json(capsys, extra: list[str]) -> dict:
    recommend_cli.main(_recommend_args([*extra, "--json"]))
    return json.loads(capsys.readouterr().out)


def test_auto_balanced_json(capsys):
    payload = _auto_json(
        capsys,
        [
            "--n-layers",
            "16",
            "--n-query-heads",
            "16",
            "--n-kv-heads",
            "16",
            "--head-dim",
            "128",
            "--objective",
            "balanced",
            "--context",
            "4096",
            "--generation",
            "256",
        ],
    )
    assert payload["objective"] == "balanced"
    assert len(payload["ranked"]) == 3
    assert payload["ranked"][0]["method"]
    assert payload["model"]["attention_type"] == "mha"


def test_auto_memory_json(capsys):
    payload = _auto_json(
        capsys,
        [
            "--n-layers",
            "16",
            "--n-query-heads",
            "16",
            "--n-kv-heads",
            "16",
            "--head-dim",
            "128",
            "--objective",
            "memory",
        ],
    )
    # memory objective: the top pick must be an aggressive reducer
    assert payload["ranked"][0]["memory"]["savings_percent"] >= 90


def test_auto_explain_text(capsys):
    recommend_cli.main(
        _recommend_args(
            [
                "--n-layers",
                "8",
                "--n-query-heads",
                "8",
                "--n-kv-heads",
                "8",
                "--head-dim",
                "128",
                "--explain",
            ]
        )
    )
    out = capsys.readouterr().out
    assert "RECOMMENDATION" in out
    assert "Model:" in out
    assert "Workload:" in out


def test_auto_model_config_path(capsys, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "num_key_value_heads": 4,
                "hidden_size": 2048,
            }
        )
    )
    recommend_cli.main(["--model-config", str(config), "--no-probe", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"]["architecture"] == "llama"
    assert payload["model"]["attention_type"] == "gqa"


def test_auto_gqa_from_kv_heads(capsys):
    payload = _auto_json(capsys, ["--n-query-heads", "32", "--n-kv-heads", "8"])
    assert payload["model"]["attention_type"] == "gqa"


def test_auto_quality_prefers_retention(capsys):
    from veloxquant_mlx.planning.memory_estimator import method_quant_bits

    payload = _auto_json(
        capsys,
        [
            "--n-layers",
            "16",
            "--n-query-heads",
            "16",
            "--n-kv-heads",
            "16",
            "--objective",
            "quality",
        ],
    )
    top = payload["ranked"][0]
    # quality-ranked top pick keeps all tokens (must not be eviction-first)
    assert method_quant_bits(top["method"])[2] is False
    assert top["objective_scores"]["quality"] >= 0.5


def test_legacy_mode_json(capsys):
    recommend_cli.main(
        ["--chip", "M4", "--ram-gb", "24", "--model-class", "7B", "--goal", "everyday", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert "request" in payload
    assert "recommendation" in payload
    assert "method" in payload["recommendation"]


def test_legacy_no_args_raises(capsys):
    import pytest

    with pytest.raises(SystemExit):
        recommend_cli.main([])


def test_auto_never_raises_with_bare_geometry(capsys):
    payload = _auto_json(capsys, ["--n-layers", "32", "--n-query-heads", "32", "--n-kv-heads", "8"])
    assert isinstance(payload["ranked"], list)
    assert len(payload["ranked"]) == 3


def test_auto_latency_never_empty(capsys):
    payload = _auto_json(
        capsys, ["--objective", "latency", "--n-query-heads", "16", "--n-kv-heads", "16"]
    )
    assert len(payload["ranked"]) == 3


def test_cli_argparse_choices_match_recommender():
    # Regression test for issue #391: cli/recommend.py's argparse choices=
    # for --chip and --ram-gb must not drift from mac_recommender.py's own
    # ChipFamily / ALLOWED_RAM_GB. Lives here (not tests/non_metal/) because
    # recommend.py imports veloxquant_mlx.planning, which is not leaf-level
    # pure Python -- see docs/CI_AND_TESTING.md#two-test-directories-and-why.
    assert set(recommend_cli._CHIP_CHOICES) == {"M1", "M2", "M3", "M4"}
    assert set(recommend_cli.ALLOWED_RAM_GB) == set(mac_recommender.ALLOWED_RAM_GB)
    assert set(recommend_cli.MODEL_WEIGHT_GB_4BIT) == set(mac_recommender.MODEL_WEIGHT_GB_4BIT)


_CONFIG_KNOBS = {
    "bit_width_inlier",
    "kivi_group_size",
    "stream_n_sink",
    "stream_window_size",
    "seed",
}


@pytest.mark.parametrize("model_class", ["1B", "7B", "14B"])
@pytest.mark.parametrize("ram_gb", [16, 24, 64])
@pytest.mark.parametrize("goal", mac_recommender.ruleset_dict()["goals"])
def test_every_goal_recommends_a_servable_method(goal, ram_gb, model_class):
    """The recommender must only name methods the package can build and serve (#632)."""
    from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
    from veloxquant_mlx.cache.registry import get_method

    req = mac_recommender.RecommendRequest(
        chip="M4", ram_gb=ram_gb, model_class=model_class, goal=goal
    )
    r = mac_recommender.recommend(req)
    assert get_method(r.method).serve_tier.is_servable
    cfg = {"method": r.method, **{k: v for k, v in r.knobs.items() if k in _CONFIG_KNOBS}}
    if r.method != "vecinfer":  # vecinfer needs a calibration pass before create()
        KVCacheFactory.create(KVCacheConfig(**cfg))


def test_ruleset_defaults_are_registered_methods():
    from veloxquant_mlx.cache.registry import get_method

    for entry in mac_recommender.ruleset_dict()["defaults"].values():
        assert get_method(entry["method"]).serve_tier.is_servable
