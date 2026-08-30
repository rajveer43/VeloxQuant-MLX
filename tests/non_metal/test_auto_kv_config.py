"""Unit tests for the hardware-aware auto KV-config selector (no MLX required).

Lives outside veloxquant_mlx/tests/ on purpose — see
docs/CI_AND_TESTING.md#two-test-directories-and-why.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Repo-root tests/ so pytest does not import veloxquant_mlx/__init__.py (needs mlx).
_MOD_PATH = Path(__file__).resolve().parents[2] / "veloxquant_mlx" / "tools" / "auto_kv_config.py"
_SPEC = importlib.util.spec_from_file_location("auto_kv_config", _MOD_PATH)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["auto_kv_config"] = _mod
_SPEC.loader.exec_module(_mod)

WorkloadProfile = _mod.WorkloadProfile
HardwareProfile = _mod.HardwareProfile
select_kv_config = _mod.select_kv_config
estimate_kv_fp16_gb = _mod.estimate_kv_fp16_gb
to_kv_cache_config = _mod.to_kv_cache_config


def test_estimate_kv_fp16_gb_mistral_like():
    # 2 * 32 * 8 * 128 * 2048 * 2 bytes = 256 MiB = 0.25 GiB
    gb = estimate_kv_fp16_gb(WorkloadProfile(seq_len=2048, head_dim=128, n_layers=32, n_kv_heads=8))
    assert abs(gb - 0.25) < 1e-6


def test_short_context_prefers_higher_precision_kivi():
    r = select_kv_config(
        WorkloadProfile(seq_len=2048, head_dim=128, n_layers=32, n_kv_heads=8),
        HardwareProfile(ram_gb=64),
    )
    assert r.context_regime == "short"
    assert r.method == "kivi"
    assert r.bit_width == 4
    assert r.knobs["kivi_group_size"] == r.group_size


def test_long_context_prefers_aggressive_compression():
    r = select_kv_config(
        WorkloadProfile(seq_len=65536, head_dim=128, n_layers=32, n_kv_heads=8),
        HardwareProfile(ram_gb=64),
    )
    assert r.context_regime == "long"
    assert r.method == "turboquant_rvq"
    assert r.bit_width == 1


def test_large_head_dim_gets_a_coarser_group_size():
    small = select_kv_config(
        WorkloadProfile(seq_len=2048, head_dim=64, n_layers=32, n_kv_heads=8),
        HardwareProfile(ram_gb=64),
    )
    large = select_kv_config(
        WorkloadProfile(seq_len=2048, head_dim=256, n_layers=32, n_kv_heads=8),
        HardwareProfile(ram_gb=64),
    )
    assert small.group_size <= 64
    assert large.group_size <= 256
    # head_dim=256 admits the full preferred group size (64 | 256); a
    # smaller head_dim that still divides evenly should not get *more*
    # than the same preferred cap.
    assert large.group_size == 64


def test_group_size_always_divides_head_dim():
    for head_dim in (48, 96, 100, 127, 130, 192, 384):
        r = select_kv_config(
            WorkloadProfile(seq_len=2048, head_dim=head_dim, n_layers=32, n_kv_heads=8),
            HardwareProfile(ram_gb=64),
        )
        assert head_dim % r.group_size == 0, (head_dim, r.group_size)


def test_memory_pressure_lowers_bit_width():
    workload = WorkloadProfile(seq_len=4096, head_dim=128, n_layers=80, n_kv_heads=8)

    # Plenty of RAM: short context keeps the 4-bit default.
    roomy = select_kv_config(workload, HardwareProfile(ram_gb=256))
    assert roomy.bit_width == 4
    assert roomy.memory_pressure_ratio <= 1.0

    # Same workload, tiny machine: pressure forces bit-width down and
    # method switches off kivi (kivi is reserved for the no-pressure path).
    tight = select_kv_config(workload, HardwareProfile(ram_gb=8))
    assert tight.bit_width < roomy.bit_width
    assert tight.memory_pressure_ratio > 1.0
    assert any("pressure" in w.lower() for w in tight.warnings)


def test_extreme_pressure_falls_back_to_streaming_llm():
    r = select_kv_config(
        WorkloadProfile(seq_len=200000, head_dim=128, n_layers=80, n_kv_heads=8),
        HardwareProfile(ram_gb=8),
    )
    assert r.method == "streaming_llm"
    assert r.knobs["stream_window_size"] > 0
    assert any("streaming_llm" in w for w in r.warnings)


def test_no_metal_hardware_selects_pure_mlx_packing():
    r = select_kv_config(
        WorkloadProfile(seq_len=2048, head_dim=128),
        HardwareProfile(ram_gb=32, metal_available=False),
    )
    assert r.packing_strategy == "pure_mlx"


def test_metal_available_or_unknown_selects_metal_auto():
    for metal_available in (True, None):
        r = select_kv_config(
            WorkloadProfile(seq_len=2048, head_dim=128),
            HardwareProfile(ram_gb=32, metal_available=metal_available),
        )
        assert r.packing_strategy == "metal_auto"


def test_invalid_seq_len_raises():
    with pytest.raises(ValueError):
        select_kv_config(WorkloadProfile(seq_len=0, head_dim=128))


def test_invalid_head_dim_raises():
    with pytest.raises(ValueError):
        select_kv_config(WorkloadProfile(seq_len=2048, head_dim=0))


def test_result_to_dict_roundtrip():
    r = select_kv_config(WorkloadProfile(seq_len=2048, head_dim=128))
    d = r.to_dict()
    assert d["method"] == r.method
    assert d["knobs"] == r.knobs


def test_to_kv_cache_config_maps_knobs_and_packing():
    workload = WorkloadProfile(seq_len=2048, head_dim=128)
    result = select_kv_config(workload, HardwareProfile(ram_gb=64, metal_available=False))

    class _FakeKVCacheConfig(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    fake_base_module = type(sys)("veloxquant_mlx.cache.base")
    fake_base_module.KVCacheConfig = _FakeKVCacheConfig
    fake_cache_module = type(sys)("veloxquant_mlx.cache")
    fake_pkg = type(sys)("veloxquant_mlx")
    sys.modules["veloxquant_mlx"] = fake_pkg
    sys.modules["veloxquant_mlx.cache"] = fake_cache_module
    sys.modules["veloxquant_mlx.cache.base"] = fake_base_module
    try:
        cfg = to_kv_cache_config(result, workload)
    finally:
        del sys.modules["veloxquant_mlx"]
        del sys.modules["veloxquant_mlx.cache"]
        del sys.modules["veloxquant_mlx.cache.base"]

    assert cfg["method"] == result.method
    assert cfg["head_dim"] == workload.head_dim
    assert cfg["use_metal_kernels"] is False
    for k, v in result.knobs.items():
        assert cfg[k] == v
