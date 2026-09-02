"""Tests for ``veloxquant auto-config`` argument handling and JSON output (#253/#44)."""

from __future__ import annotations

import json

from veloxquant_mlx.cli import auto_config as auto_config_cli
from veloxquant_mlx.config.auto_config import LONG_CONTEXT_TOKENS, SHORT_CONTEXT_TOKENS


def test_defaults_produce_mid_band_kivi(capsys):
    auto_config_cli.main(["--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["workload"] == {
        "head_dim": 128,
        "seq_len": 4096,
        "n_layers": 1,
        "batch_size": 1,
    }
    assert payload["config"]["method"] == "kivi"
    assert payload["config"]["head_dim"] == 128
    assert "reason" in payload
    assert len(payload["reason"]) > 0


def test_short_context_selects_turboquant_rvq(capsys):
    auto_config_cli.main(["--seq-len", str(SHORT_CONTEXT_TOKENS - 1), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["method"] == "turboquant_rvq"
    assert payload["config"]["bit_width_inlier"] == 4


def test_long_context_selects_kvquant(capsys):
    auto_config_cli.main(["--seq-len", str(LONG_CONTEXT_TOKENS), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["method"] == "kvquant"
    assert payload["config"]["kvquant_bits"] == 3


def test_explicit_memory_pressure_forces_gear(capsys):
    # A tiny total-memory override relative to a large multi-layer/batch
    # workload should trip the pressure rule regardless of seq_len band.
    auto_config_cli.main(
        [
            "--head-dim",
            "128",
            "--seq-len",
            "100",
            "--n-layers",
            "32",
            "--batch-size",
            "8",
            "--total-memory-bytes",
            "1024",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["method"] == "gear"
    assert payload["config"]["gear_bits"] == 2
    assert "memory pressure" in payload["reason"]
    assert payload["hardware"]["total_memory_bytes"] == 1024


def test_active_memory_bytes_ignored_without_total_override(capsys):
    # --active-memory-bytes alone (no --total-memory-bytes) should not
    # override auto-detection, since active memory without a total is
    # meaningless for the pressure calculation.
    auto_config_cli.main(["--active-memory-bytes", "999", "--json"])
    payload = json.loads(capsys.readouterr().out)
    # Falls back to real hardware auto-detection; just assert it didn't crash
    # and produced a valid pool method.
    assert payload["config"]["method"] in {"turboquant_rvq", "kivi", "kvquant", "gear"}


def test_large_head_dim_doubles_group_size(capsys):
    auto_config_cli.main(["--head-dim", "256", "--seq-len", "8000", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["method"] == "kivi"
    assert payload["config"]["kivi_group_size"] == 64


def test_non_json_output_does_not_crash(capsys):
    auto_config_cli.main(["--seq-len", "100"])
    out = capsys.readouterr().out
    assert "auto-config" in out
    assert "method=" in out


def test_config_dict_excludes_non_serializable_fields():
    from veloxquant_mlx.cache.base import KVCacheConfig

    config = KVCacheConfig(method="kivi", head_dim=128, kivi_group_size=32)
    result = auto_config_cli._config_to_dict(config)
    assert "observers" not in result
    assert "store" not in result
    assert "dtype" not in result
    assert result["method"] == "kivi"
    assert result["kivi_group_size"] == 32


def test_config_dict_omits_fields_not_relevant_to_selected_method():
    from veloxquant_mlx.cache.base import KVCacheConfig

    config = KVCacheConfig(method="turboquant_rvq", head_dim=128, bit_width_inlier=4)
    result = auto_config_cli._config_to_dict(config)
    assert "kivi_group_size" not in result
    assert "gear_bits" not in result
