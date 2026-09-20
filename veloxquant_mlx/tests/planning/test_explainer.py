"""Tests for the plain-text explainer (RFC Phase 5)."""

from __future__ import annotations

from veloxquant_mlx.planning.explainer import (
    _fmt_bytes,
    explain,
    explain_filtering,
    explain_hardware_model,
)
from veloxquant_mlx.planning.strategy_planner import plan_strategy
from veloxquant_mlx.planning.workload import WorkloadObjective, WorkloadProfile
from veloxquant_mlx.profiling.hardware_profiler import HardwareProfile
from veloxquant_mlx.profiling.model_profiler import ModelProfile


def _model() -> ModelProfile:
    return ModelProfile(
        model_id="tiny", architecture="llama", num_layers=4, num_query_heads=8,
        num_kv_heads=8, head_dim=128, attention_type="mha",
    )


def _hw() -> HardwareProfile:
    return HardwareProfile(chip="M4", chip_generation=4, available_memory_bytes=16 * 1024**3)


def _result(objective: str = WorkloadObjective.BALANCED):
    return plan_strategy(
        _model(), _hw(),
        WorkloadProfile(context_length=4096, objective=objective),
    )


def test_explain_contains_hardware_model_workload():
    text = explain(_result())
    assert "Hardware:" in text and "M4" in text
    assert "Model:" in text and "llama" in text
    assert "Workload:" in text and "4096" in text


def test_explain_contains_recommendation_and_rank():
    text = explain(_result())
    assert "RECOMMENDATION" in text
    assert "turboquant_rvq" in text or "1." in text


def test_explain_mentions_evidence():
    text = explain(_result())
    assert "evidence" in text


def test_explain_can_disable_filtering_section():
    text = explain(_result(), include_filtering=False)
    assert "Filtered" not in text
    assert "RECOMMENDATION" in text


def test_explain_warning_cap():
    text = explain(_result(), warning_limit=1)
    assert "Caveats" in text or "more caveats" in text


def test_explain_fallback_mode():
    tiny = HardwareProfile(chip="M1", chip_generation=1, available_memory_bytes=64)

    result = plan_strategy(_model(), tiny, WorkloadProfile())
    assert result.fallback_used
    text = explain(result)
    assert "fall back" in text
    assert "RECOMMENDATION" in text


def test_explain_hardware_model_no_duplicate_chip():
    text = explain_hardware_model(_result())
    # chip should appear once in the hardware sentence
    assert text.count("M4") == 1


def test_explain_filtering_reports_counts():
    text = explain_filtering(_result())
    assert "viable" in text


def test_fmt_bytes_human_readable():
    assert _fmt_bytes(0) == "0 B"
    assert _fmt_bytes(512) == "512 B"
    assert _fmt_bytes(2048) == "2.00 KiB"
    assert _fmt_bytes(5 * 1024**3) == "5.00 GiB"
    assert _fmt_bytes(3 * 1024**4) == "3.00 TiB"


def test_explain_objective_label():
    text = explain(_result(WorkloadObjective.MEMORY))
    assert "memory" in text.lower()
