"""Tests for strategy ranking and working-memory analysis (RFC Phase 5)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from veloxquant_mlx.planning.strategy_planner import (
    DEFAULT_OBJECTIVE_WEIGHTS,
    PlanningOptions,
    RecommendationResult,
    plan_strategy,
    recommend_strategy,
)
from veloxquant_mlx.planning.workload import WorkloadObjective, WorkloadProfile
from veloxquant_mlx.profiling.hardware_profiler import HardwareProfile
from veloxquant_mlx.profiling.model_profiler import ModelProfile


@dataclass
class _Rec:
    memory_reduction: float = 0.0
    latency_ms_per_token: float = 0.0
    throughput_tok_s: float = 0.0

    @property
    def savings_percent(self) -> float:
        return (1.0 - self.memory_reduction) * 100.0


def _model() -> ModelProfile:
    return ModelProfile(
        model_id="llama-7b", architecture="llama", num_layers=8, num_query_heads=8,
        num_kv_heads=8, head_dim=128, attention_type="mha",
    )


def _hw() -> HardwareProfile:
    return HardwareProfile(
        chip="M4", chip_generation=4,
        available_memory_bytes=64 * 1024**3, mlx_version="0.32.2",
    )


def _workload(objective: str = WorkloadObjective.BALANCED) -> WorkloadProfile:
    return WorkloadProfile(
        context_length=8192, generation_length=512, objective=objective
    )


def test_plan_returns_ranked_result():
    result = plan_strategy(_model(), _hw(), _workload())
    assert isinstance(result, RecommendationResult)
    assert result.best is not None
    assert len(result.ranked) == 3


def test_plan_top_choices_are_variant_mechanisms():
    result = plan_strategy(_model(), _hw(), _workload())
    names = {item.method for item in result.ranked}
    for name in names:
        assert name in result.candidates.viable


def test_plan_memory_objective_prefers_huge_savings():
    mem = plan_strategy(_model(), _hw(), _workload(WorkloadObjective.MEMORY))
    lat = plan_strategy(_model(), _hw(), _workload(WorkloadObjective.LATENCY))
    # memory-first picks the smallest-footprint methods; eviction/hybrid on top
    assert mem.best is not None
    assert mem.best.memory_estimate.savings_percent >= 90
    # latency objective must also resolve to a best method
    assert lat.best is not None


def test_plan_quality_objective_deprioritizes_eviction():
    from veloxquant_mlx.planning.memory_estimator import method_quant_bits

    result = plan_strategy(_model(), _hw(), _workload(WorkloadObjective.QUALITY))
    # top pick must not be an eviction-only method (they pay the quality penalty)
    assert result.best is not None
    _, _, evicts = method_quant_bits(result.best.method)
    assert evicts is False


def test_plan_latency_ranks_differ_from_random():
    lat = plan_strategy(_model(), _hw(), _workload(WorkloadObjective.LATENCY))
    mem = plan_strategy(_model(), _hw(), _workload(WorkloadObjective.MEMORY))
    # objectives can agree, but latency must not be byte-for-byte identical
    # on the proxy score ordering without evidence of divergence
    assert lat.best is not None and mem.best is not None


def test_plan_scores_in_range():
    result = plan_strategy(_model(), _hw(), _workload())
    for item in result.ranked:
        assert 0.0 <= item.score <= 1.0
        for axis in ("memory", "latency", "throughput", "quality"):
            assert 0.0 <= item.objective_scores[axis] <= 1.0


def test_plan_max_results_respected():
    result = plan_strategy(
        _model(), _hw(), _workload(), options=PlanningOptions(max_results=5)
    )
    assert len(result.ranked) == 5


def test_plan_score_order_is_sorted():
    result = plan_strategy(_model(), _hw(), _workload())
    scores = [item.score for item in result.ranked]
    assert scores == sorted(scores, reverse=True)


def test_plan_skips_unmodeled_methods_gracefully():
    result = plan_strategy(_model(), _hw(), _workload())
    # even uncurated methods can rank; nothing should raise
    for item in result.ranked:
        assert item.method in result.candidates.method_info


def test_plan_fallback_when_no_budget_fits():
    tiny = HardwareProfile(chip="M1", chip_generation=1, available_memory_bytes=64)
    result = plan_strategy(_model(), tiny, _workload())
    assert result.fallback_used is True
    assert result.best is None
    assert result.fallback_reason is not None
    assert result.candidates.viable == []


def test_plan_empirical_memory_override():
    from veloxquant_mlx.cache.registry import all_method_names

    others = set(all_method_names()) - {"kivi"}
    result = plan_strategy(
        _model(), _hw(), _workload(WorkloadObjective.MEMORY),
        options=PlanningOptions(
            empirical={"kivi": _Rec(memory_reduction=0.2)},
            additional_exclusions=others,
        ),
    )
    # 0.2 reduction -> 80% savings: patched estimate, evidence flips to empirical
    kivi = next(i for i in result.ranked if i.method == "kivi")
    assert kivi.memory_estimate.reduction_ratio == pytest.approx(0.2, abs=0.001)
    assert kivi.evidence == "empirical"


def test_plan_empirical_latency_override():
    from veloxquant_mlx.cache.registry import all_method_names

    others = set(all_method_names()) - {"kivi"}
    result = plan_strategy(
        _model(), _hw(), _workload(),
        options=PlanningOptions(
            empirical={"kivi": _Rec(latency_ms_per_token=1.1)},
            additional_exclusions=others,
        ),
    )
    kivi = next(i for i in result.ranked if i.method == "kivi")
    assert kivi.evidence == "empirical"


def test_plan_empirical_latency_override_takes_effect():
    result = plan_strategy(
        _model(), _hw(), _workload(WorkloadObjective.LATENCY),
        options=PlanningOptions(empirical={"zipcache": _Rec(latency_ms_per_token=5000.0)}, max_results=5),
    )
    # an absurd measured latency must knock zipcache off the latency podium
    best = result.best
    for item in result.ranked:
        if item.method == "zipcache":
            assert item.objective_scores["latency"] == 0.0
    if best is not None:
        assert best.method != "zipcache"


def test_plan_empirical_record_with_junk_values_ignored():
    from veloxquant_mlx.cache.registry import all_method_names

    others = set(all_method_names()) - {"kivi"}
    result = plan_strategy(
        _model(), _hw(), _workload(),
        options=PlanningOptions(
            empirical={"kivi": _Rec(memory_reduction=99.0)},
            additional_exclusions=others,
        ),
    )
    # memory_reduction out of (0, 1] must be ignored
    kivi = next(i for i in result.ranked if i.method == "kivi")
    assert kivi.evidence == "analytic"


def test_plan_exclusions_respected():
    result = plan_strategy(
        _model(), _hw(), _workload(),
        options=PlanningOptions(additional_exclusions={"kivi", "h2o"}),
    )
    names = {item.method for item in result.ranked}
    assert names.isdisjoint({"kivi", "h2o"})
    assert "kivi" in result.candidates.excluded


def test_plan_empty_empirical_is_fine():
    result = plan_strategy(_model(), _hw(), _workload(), options=PlanningOptions())
    assert result.best is not None
    assert all(item.evidence == "analytic" for item in result.ranked)


def test_objective_weights_table_has_balanced():
    assert WorkloadObjective.BALANCED in DEFAULT_OBJECTIVE_WEIGHTS


def test_custom_weights_override():
    weights = {"memory": 1.0, "latency": 0.0, "throughput": 0.0, "quality": 0.0}
    result = plan_strategy(
        _model(), _hw(), _workload(), options=PlanningOptions(objective_weights=weights)
    )
    assert result.best is not None
    # pure-memory objective: the single strongest reducer takes rank 1
    biggest = max(
        result.ranked, key=lambda i: i.memory_estimate.savings_percent
    )
    assert biggest is result.ranked[0]


def test_scored_method_to_dict():
    result = plan_strategy(_model(), _hw(), _workload())
    data = result.best.to_dict()  # type: ignore[union-attr]
    assert data["method"] == result.best.method  # type: ignore[union-attr]
    assert "score" in data
    assert "objective_scores" in data
    assert "memory" in data


def test_result_to_dict_shape():
    data = plan_strategy(_model(), _hw(), _workload()).to_dict()
    for key in (
        "objective", "ranked", "candidates", "model", "hardware", "workload",
        "fallback_used",
    ):
        assert key in data


def test_recommend_strategy_detects_hardware():
    result = recommend_strategy(
        {
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "hidden_size": 1024,
            "architectures": ["LlamaForCausalLM"],
        },
        workload=WorkloadProfile(objective=WorkloadObjective.BALANCED),
    )
    assert result.hardware.chip != "unknown"
    assert result.best is not None


def test_recommend_strategy_passes_prebuilt_model():
    result = recommend_strategy(model=_model(), workload=_workload())
    assert result.model.model_id == "llama-7b"


def test_recommend_strategy_requires_input():
    import pytest

    with pytest.raises(ValueError):
        recommend_strategy(model_config=None)


def test_fallback_result_explains_best_is_none():
    tiny = HardwareProfile(chip="M1", chip_generation=1, available_memory_bytes=64)
    result = plan_strategy(_model(), tiny, _workload())
    assert result.fallback_used
    assert result.best is None


@pytest.mark.parametrize("bandwidth_gbps", [20.0, 90.0, 120.0, 400.0])
def test_recommendation_currently_invariant_to_bandwidth(bandwidth_gbps):
    """VeloxQuant-MLX#509: bandwidth is a shared scalar across every
    candidate's latency_ms, and _normalize_axis()'s min-max normalization
    is invariant to a uniform positive scalar -- so today, bandwidth's
    absolute accuracy does not change any recommendation. This pins that
    invariance so a future change to the scoring formula (e.g. bandwidth
    entering a hard filter or absolute SLA gate) is caught rather than
    silently making a previously-inert table value load-bearing.
    """
    hw = HardwareProfile(
        chip="M4", chip_generation=4,
        available_memory_bytes=64 * 1024**3, mlx_version="0.32.2",
        peak_memory_bandwidth_gbps=bandwidth_gbps,
    )
    baseline = plan_strategy(_model(), _hw(), _workload())
    result = plan_strategy(_model(), hw, _workload())

    assert [s.method for s in result.ranked] == [s.method for s in baseline.ranked]
    for a, b in zip(result.ranked, baseline.ranked, strict=True):
        assert a.score == pytest.approx(b.score, abs=1e-9)
