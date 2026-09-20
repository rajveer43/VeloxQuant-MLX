"""Tests for candidate filtering (RFC Phase 3).

All tests inject a non-probing registry (``static_method_info``) or a tiny fake
so the suite stays instant and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

from veloxquant_mlx.cache.registry import (
    MethodFamily,
    MethodInfo,
    ServeTier,
    StrategyCapabilities,
    static_method_info,
)
from veloxquant_mlx.planning.candidate_filter import (
    CandidateFilterOptions,
    filter_candidates,
)
from veloxquant_mlx.planning.memory_estimator import MemoryEstimate
from veloxquant_mlx.planning.workload import WorkloadProfile
from veloxquant_mlx.profiling.hardware_profiler import HardwareProfile
from veloxquant_mlx.profiling.model_profiler import ModelProfile

_LONG_MEMORY = 256 * 1024**3
_HW = HardwareProfile(chip="M4", chip_generation=4, available_memory_bytes=_LONG_MEMORY)


def _model(attention: str = "gqa") -> ModelProfile:
    return ModelProfile(
        model_id="t", architecture="llama", num_layers=4, num_query_heads=8,
        num_kv_heads={"gqa": 4, "mha": 8, "mqa": 1}[attention],
        head_dim=128, attention_type=attention,
    )


def _workload(**kw) -> WorkloadProfile:
    defaults = {"context_length": 8192, "generation_length": 256}
    defaults.update(kw)
    return WorkloadProfile(**defaults)


def _estimate(method: str, compressed: int = 1 << 20) -> MemoryEstimate:
    return MemoryEstimate(
        method=method, baseline_bytes=1 << 30, compressed_bytes=compressed,
        resident_bytes=compressed, confidence="high",
    )


def _estimates(methods: list[str], compressed: int = 1 << 20) -> dict[str, MemoryEstimate]:
    return {m: _estimate(m, compressed) for m in methods}


@dataclass
class _FakeInfo:
    name: str
    serve_tier: ServeTier
    capabilities: StrategyCapabilities
    unsupported_reason: str | None = None


def _as_real(info: _FakeInfo) -> MethodInfo:
    return MethodInfo(
        name=info.name,
        family=MethodFamily.QUANTIZATION,
        serve_tier=info.serve_tier,
        blurb="",
        capabilities=info.capabilities,
        unsupported_reason=info.unsupported_reason,
    )


def _lookup_from(infos: dict[str, _FakeInfo]):
    def lookup(name: str) -> MethodInfo:
        return _as_real(infos[name])

    return lookup


def _filter(infos: dict[str, _FakeInfo], model=None, hw=_HW, workload=None, estimate_bytes: int = 1 << 18, **options_kw):
    names = list(infos)
    opts = CandidateFilterOptions(registry_lookup=_lookup_from(infos), **options_kw)
    return filter_candidates(
        model or _model(),
        hw,
        workload or _workload(),
        _estimates(names, compressed=estimate_bytes),
        options=opts,
        methods=names,
    )


def test_servable_real_methods_pass():
    from veloxquant_mlx.cache.registry import all_method_names

    result = filter_candidates(
        _model(), _HW, _workload(), _estimates(list(all_method_names())),
        options=CandidateFilterOptions(registry_lookup=static_method_info),
    )
    assert len(result.viable) > 10
    assert "turboquant_rvq" in result.viable
    assert "kivi" in result.viable


def test_unservable_method_excluded_with_reason():
    caps = StrategyCapabilities()
    result = _filter(
        {
            "a": _FakeInfo("a", ServeTier.HONEST_BYTES, caps),
            "crash": _FakeInfo("crash", ServeTier.CRASHES, caps, unsupported_reason="boom"),
        }
    )
    assert result.viable == ["a"]
    assert "boom" in result.excluded["crash"]


def test_attention_mismatch_excludes():
    caps = StrategyCapabilities(supports_mha=True, supports_gqa=False, supports_mqa=False)
    result = _filter(
        {"a": _FakeInfo("a", ServeTier.HONEST_BYTES, caps)}, model=_model("gqa")
    )
    assert result.viable == []
    assert "gqa" in result.excluded["a"]


def test_memory_budget_excludes_oversized():
    caps = StrategyCapabilities()
    result = _filter(
        {"a": _FakeInfo("a", ServeTier.HONEST_BYTES, caps)},
        estimate_bytes=1 << 21,  # 2 MiB footprint
        memory_budget_bytes=1 << 20,  # budget = 1 MiB
    )
    assert result.viable == []
    assert "budget" in result.excluded["a"]


def test_memory_budget_uses_hardware_when_unset():
    caps = StrategyCapabilities()
    tiny_hw = HardwareProfile(chip="M1", chip_generation=1, available_memory_bytes=1024)
    result = _filter(
        {"a": _FakeInfo("a", ServeTier.HONEST_BYTES, caps)}, hw=tiny_hw
    )
    assert result.viable == []
    assert "memory budget" in result.excluded["a"]


def test_calibration_excluded_when_prefer_no_calibration():
    result = _filter(
        {
            "cal": _FakeInfo("cal", ServeTier.HONEST_BYTES, StrategyCapabilities(requires_calibration=True)),
            "fast": _FakeInfo("fast", ServeTier.HONEST_BYTES, StrategyCapabilities()),
        },
        prefer_no_calibration=True,
    )
    assert "fast" in result.viable
    assert "cal" in result.excluded
    assert "calibration" in result.excluded["cal"]


def test_calibration_is_soft_warning_without_preference():
    result = _filter(
        {"cal": _FakeInfo("cal", ServeTier.HONEST_BYTES, StrategyCapabilities(requires_calibration=True))}
    )
    assert result.viable == ["cal"]
    assert any("calibration" in w for w in result.soft_warnings["cal"])


def test_require_metal_excludes_reference_only():
    result = _filter(
        {
            "has": _FakeInfo("has", ServeTier.HONEST_BYTES, StrategyCapabilities(has_metal_kernel=True)),
            "nope": _FakeInfo("nope", ServeTier.HONEST_BYTES, StrategyCapabilities(has_metal_kernel=False)),
        },
        require_metal=True,
    )
    assert result.viable == ["has"]
    assert "require-metal" in result.excluded["nope"]


def test_missing_metal_is_only_soft_warning_by_default():
    result = _filter(
        {"ref": _FakeInfo("ref", ServeTier.HONEST_BYTES, StrategyCapabilities(has_metal_kernel=False))}
    )
    assert result.viable == ["ref"]
    assert any("Metal" in w for w in result.soft_warnings["ref"])


def test_eviction_soft_warning():
    result = _filter(
        {"ev": _FakeInfo("ev", ServeTier.HONEST_BYTES, StrategyCapabilities(uses_eviction=True))}
    )
    assert any("drops old tokens" in w for w in result.soft_warnings["ev"])


def test_short_context_soft_warning_for_compressors():
    caps = StrategyCapabilities(compresses_keys=True)
    result = _filter(
        {"c": _FakeInfo("c", ServeTier.HONEST_BYTES, caps)},
        workload=_workload(context_length=500),
    )
    assert result.viable == ["c"]
    assert any("short" in w for w in result.soft_warnings["c"])


def test_long_context_no_short_context_warning():
    caps = StrategyCapabilities(compresses_keys=True)
    result = _filter(
        {"c": _FakeInfo("c", ServeTier.HONEST_BYTES, caps)},
        workload=_workload(context_length=20000),
    )
    assert all("short" not in w for w in result.soft_warnings.get("c", []))


def test_missing_estimate_excludes():
    caps = StrategyCapabilities()
    infos = {"a": _FakeInfo("a", ServeTier.HONEST_BYTES, caps)}
    opts = CandidateFilterOptions(registry_lookup=_lookup_from(infos))
    result = filter_candidates(_model(), _HW, _workload(), {}, options=opts, methods=["a"])
    assert result.viable == []
    assert "no memory estimate" in result.excluded["a"]


def test_result_to_dict_shape():
    result = _filter(
        {"a": _FakeInfo("a", ServeTier.HONEST_BYTES, StrategyCapabilities())}
    )
    data = result.to_dict()
    assert data["viable"] == ["a"]
    assert data["excluded"] == {}
    assert "a" in data["methods"]
    assert "soft_warnings" in data


def test_defaults_object():
    opts = CandidateFilterOptions()
    assert opts.memory_budget_bytes is None
    assert opts.prefer_no_calibration is False
    assert opts.require_metal is False
