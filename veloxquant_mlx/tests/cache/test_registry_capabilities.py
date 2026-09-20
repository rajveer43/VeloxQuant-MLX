"""Tests for the strategy-capabilities registry metadata (RFC Phase 1)."""

from __future__ import annotations

from veloxquant_mlx.cache.registry import (
    MethodInfo,
    StrategyCapabilities,
    all_method_names,
    static_method_info,
)


def test_capabilities_defaults():
    caps = StrategyCapabilities()
    assert caps.supported_bits is None
    assert caps.requires_calibration is False
    assert caps.has_metal_kernel is False
    assert caps.tunable_parameters == {}


def test_capabilities_to_dict_round_visits_fields():
    a = StrategyCapabilities(has_metal_kernel=True, tunable_parameters={"alpha": "float"})
    b = StrategyCapabilities()
    assert a != b
    assert a.to_dict() != b.to_dict()


def test_capabilities_to_dict_shape():
    data = StrategyCapabilities(supported_bits=[2, 3, 4]).to_dict()
    assert data["supported_bits"] == [2, 3, 4]
    assert data["has_metal_kernel"] is False


def test_static_method_info_returns_real_info():
    info = static_method_info("kivi")
    assert isinstance(info, MethodInfo)
    assert info.capabilities.requires_calibration is False


def test_static_method_info_never_probes(monkeypatch):
    import veloxquant_mlx.cache.registry as reg

    calls = []

    def _fake_probe(method: str):
        calls.append(method)
        return reg.ServeTier.HONEST_ESTIMATE

    monkeypatch.setattr(reg, "probe_serve_tier", _fake_probe)
    info = static_method_info("turboquant_prod")
    assert info.capabilities.requires_calibration is True
    assert calls == []


def test_static_method_info_servable():
    assert static_method_info("kivi").serve_tier.is_servable


def test_quantization_capabilities_filled():
    info = static_method_info("turboquant_rvq")
    assert info.capabilities.compresses_keys is True
    assert info.capabilities.uses_eviction is False
    assert info.capabilities.supported_bits is not None
    assert 3 in info.capabilities.supported_bits


def test_kivi_has_metal_kernel():
    info = static_method_info("kivi")
    assert info.capabilities.has_metal_kernel is True


def test_kivi_tunable_params_exposed():
    info = static_method_info("kivi")
    params = info.capabilities.tunable_parameters
    assert isinstance(params, dict)
    assert all(isinstance(v, str) for v in params.values())
    assert "kivi_group_size" in params


def test_eviction_method_flag():
    info = static_method_info("h2o")
    assert info.capabilities.uses_eviction is True
    assert info.capabilities.uses_merging is False


def test_value_compression_flagged():
    # a full-float4-quant method compresses values
    info = static_method_info("kvtc")
    assert info.capabilities.compresses_values is True


def test_streaming_support():
    info = static_method_info("streaming_llm")
    assert info.capabilities.supports_streaming is True


def test_capabilities_cover_all_methods():
    for name in all_method_names():
        info = static_method_info(name)
        assert isinstance(info.capabilities, StrategyCapabilities)
        assert len(info.capabilities.tunable_parameters) >= 0


def test_toga_calibration_flagged():
    for name in ("polar", "qjl", "vecinfer", "spectral"):
        assert static_method_info(name).capabilities.requires_calibration is True


def test_attention_flags():
    info = static_method_info("kivi")
    caps = info.capabilities
    assert caps.supports_gqa and caps.supports_mha and caps.supports_mqa


def test_get_method_matches_static_capabilities(monkeypatch):
    import veloxquant_mlx.cache.registry as reg

    # stub the probe so the test is hermetic: capabilities come from the same
    # _capabilities_for table on both paths regardless of the live probe result
    monkeypatch.setattr(reg, "probe_serve_tier", lambda name: reg.ServeTier.HONEST_BYTES)
    assert reg.get_method("kivi").capabilities.to_dict() == static_method_info("kivi").capabilities.to_dict()


def test_get_method_uses_probed_tier(monkeypatch):
    import veloxquant_mlx.cache.registry as reg

    monkeypatch.setattr(reg, "probe_serve_tier", lambda name: reg.ServeTier.NOT_TRIMMABLE)
    assert reg.get_method("h2o").serve_tier is reg.ServeTier.NOT_TRIMMABLE


def test_unknown_method_raises():
    import pytest

    with pytest.raises(KeyError):
        static_method_info("not_a_method_xyz")
