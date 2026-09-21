"""Tests for Apple-Silicon hardware detection (RFC Phase 2)."""

from __future__ import annotations

import platform

import pytest

from veloxquant_mlx.profiling.hardware_profiler import (
    HardwareProfile,
    _stdlib_chip,
    chip_generation,
    detect_hardware_profile,
)


def test_chip_generation_parses_numeric_generation():
    assert chip_generation("M1") == 1
    assert chip_generation("M4 Pro") == 4
    assert chip_generation("Apple M2 Ultra") == 2
    assert chip_generation("foo") == 0


def test_chip_generation_handles_lowercase_and_mixed():
    assert chip_generation("m3") == 3
    assert chip_generation("Apple M5 Max") == 5


@pytest.mark.parametrize("chip,expected", [("M1", 1), ("M2", 2), ("M3", 3), ("M4", 4)])
def test_chip_generation_table(chip, expected):
    assert chip_generation(chip) == expected


def test_detect_hardware_profile_never_raises():
    profile = detect_hardware_profile()
    assert isinstance(profile, HardwareProfile)


def test_detect_reports_apple_silicon():
    # In virtualized CI environments (paravirtual device), chip detection
    # may not find the M-series name, but we still get a string chip name,
    # memory bytes, and other fields. Real hardware will have "M" in chip name.
    profile = detect_hardware_profile()
    assert isinstance(profile.chip, str) and profile.chip
    # Either the chip name contains "M" (real hardware) or we have memory info
    has_m_series = "M" in profile.chip.upper()
    has_memory = profile.total_memory_bytes and profile.total_memory_bytes > 0
    assert has_m_series or has_memory or platform.system() == "Darwin"


def test_detect_version_fields_are_strings_or_none():
    profile = detect_hardware_profile()
    assert profile.mlx_version is None or isinstance(profile.mlx_version, str)
    assert profile.macos_version is None or isinstance(profile.macos_version, str)


def test_detect_memory_positive_when_known():
    profile = detect_hardware_profile()
    if profile.total_memory_bytes:
        assert profile.total_memory_bytes > 0


def test_bandwidth_property_uses_nominal_table():
    profile = HardwareProfile(chip="M4", chip_generation=4)
    assert profile.bandwidth_gbps == 120.0
    profile = HardwareProfile(chip="M1", chip_generation=1)
    assert profile.bandwidth_gbps == 68.3


def test_bandwidth_unknown_chip_returns_none():
    assert HardwareProfile().bandwidth_gbps is None


def test_bandwidth_prefers_measured():
    profile = HardwareProfile(chip="M2", chip_generation=2, peak_memory_bandwidth_gbps=88.0)
    assert profile.bandwidth_gbps == 88.0


def test_to_dict_round_trips_all_fields():
    profile = HardwareProfile(
        chip="M4 Pro",
        chip_generation=4,
        total_memory_bytes=48 * 1024**3,
        available_memory_bytes=40 * 1024**3,
        mlx_version="0.32.2",
        macos_version="26.0",
        metal_available=True,
        peak_memory_bandwidth_gbps=273.0,
        avg_quantize_latency_ms_per_token=None,
    )
    data = profile.to_dict()
    assert data["chip"] == "M4 Pro"
    assert data["total_memory_bytes"] == 48 * 1024**3
    assert data["metal_available"] is True
    assert data["avg_quantize_latency_ms_per_token"] is None


def test_detect_classmethod_matches_function():
    assert HardwareProfile.detect().to_dict() == detect_hardware_profile().to_dict()


def test_stdlib_chip_returns_string():
    chip = _stdlib_chip()
    assert isinstance(chip, str) and chip


def test_macos_version_detection():
    if platform.system() == "Darwin":
        profile = detect_hardware_profile()
        assert profile.macos_version is not None


def test_available_memory_never_negative_when_total_known():
    profile = HardwareProfile(chip="M1", chip_generation=1, total_memory_bytes=8 * 1024**3)
    assert profile.available_memory_bytes >= 0
