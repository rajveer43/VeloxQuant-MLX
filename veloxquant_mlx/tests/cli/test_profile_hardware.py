"""Tests for the ``profile-hardware`` CLI."""

from __future__ import annotations

import json

from veloxquant_mlx.cli import profile_hardware as profile_hardware_cli


def test_profile_hardware_json(capsys):
    profile_hardware_cli.main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "chip" in payload
    assert "total_memory_bytes" in payload
    assert "metal_available" in payload
    assert "chip_generation" in payload


def test_profile_hardware_plain_text(capsys):
    profile_hardware_cli.main([])
    out = capsys.readouterr().out
    assert "chip" in out.lower()


def test_measure_bandwidth_json(capsys):
    profile_hardware_cli.main(["--measure-bandwidth", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "measured_bandwidth_gbps" in payload
    assert payload["measured_bandwidth_gbps"] > 0
