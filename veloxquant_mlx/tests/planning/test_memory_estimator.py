"""Tests for the analytical memory estimator (RFC Phase 3)."""

from __future__ import annotations

import pytest

from veloxquant_mlx.planning.memory_estimator import (
    MemoryEstimate,
    MemoryEstimateError,
    estimate_candidate_memory,
    estimate_memory,
    method_quant_bits,
)
from veloxquant_mlx.planning.workload import WorkloadProfile
from veloxquant_mlx.profiling.model_profiler import ModelProfile


def _model() -> ModelProfile:
    return ModelProfile(
        model_id="t", architecture="llama", num_layers=4, num_query_heads=8,
        num_kv_heads=4, head_dim=128, attention_type="gqa",
    )


def _workload(**kw) -> WorkloadProfile:
    defaults = {"context_length": 4096, "generation_length": 512}
    defaults.update(kw)
    return WorkloadProfile(**defaults)


def test_qkv_gqa_baseline() -> None:
    est = estimate_memory("turboquant_rvq", _model(), _workload())
    per_token_layer = 2 * 4 * 128 * 2  # K+V * kv_heads * head_dim * fp16
    expected = per_token_layer * 4 * 1 * (4096 + 512)
    assert est.baseline_bytes == expected


def test_compressed_less_than_baseline_for_quant_method() -> None:
    est = estimate_memory("turboquant_rvq", _model(), _workload())
    assert 0 < est.compressed_bytes < est.baseline_bytes
    assert est.reduction_ratio < 1.0
    assert est.savings_percent > 0


def test_eviction_bounded_at_budget() -> None:
    est = estimate_memory("h2o", _model(), _workload())
    per_token_layer = 2 * 4 * 128 * 2
    budget_baseline = per_token_layer * 4 * 512  # budget=512, batch=1
    assert est.compressed_bytes == budget_baseline
    assert est.savings_percent > 85


def test_eviction_never_exceeds_baseline_for_short_context() -> None:
    est = estimate_memory("h2o", _model(), _workload(context_length=64, generation_length=16))
    # generated tokens (80) < budget (512) -> full fp16 footprint dominates
    assert est.compressed_bytes <= est.baseline_bytes


def test_streaming_llm_budget_is_sink_plus_window() -> None:
    est = estimate_memory("streaming_llm", _model(), _workload())
    per_token_layer = 2 * 4 * 128 * 2
    assert est.compressed_bytes == per_token_layer * 4 * 516


def test_batch_scales_footprint() -> None:
    single = estimate_memory("kivi", _model(), _workload())
    batched = estimate_memory("kivi", _model(), _workload(batch_size=2))
    assert batched.baseline_bytes == 2 * single.baseline_bytes
    assert batched.compressed_bytes == 2 * single.compressed_bytes


def test_concurrent_requests_scale_too() -> None:
    one = estimate_memory("kivi", _model(), _workload(num_concurrent_requests=1))
    three = estimate_memory("kivi", _model(), _workload(num_concurrent_requests=3))
    assert three.baseline_bytes == 3 * one.baseline_bytes


def test_method_quant_bits_reports_bits() -> None:
    kb, vb, ev = method_quant_bits("turboquant_rvq")
    assert (kb, vb, ev) == (3.0, 16.0, False)
    kb, vb, ev = method_quant_bits("h2o")
    assert ev is True
    kb, vb, ev = method_quant_bits("vecinfer")
    assert kb == 4.0

def test_unknown_method_defaults_to_fp16_low_confidence() -> None:
    est = estimate_memory("nonexistent_method", _model(), _workload())
    assert est.compressed_bytes == est.baseline_bytes
    assert est.confidence == "low"
    assert any("no curated memory model" in a for a in est.assumptions)


def test_high_confidence_for_curated_method() -> None:
    assert estimate_memory("kivi", _model(), _workload()).confidence == "high"


def test_gear_residual_adds_back() -> None:
    plain = estimate_memory("kivi", _model(), _workload())
    gear = estimate_memory("gear", _model(), _workload())
    # gear adds a low-rank residual on top of 2-bit keys+values
    assert gear.compressed_bytes > 0
    assert gear.compressed_bytes < plain.compressed_bytes


def test_workspace_scales_with_layers() -> None:
    est = estimate_memory("kivi", _model(), _workload())
    assert est.workspace_bytes == 4 * 16 * 1024


def test_peak_is_at_least_baseline() -> None:
    est = estimate_memory("polar", _model(), _workload())
    assert est.peak_bytes >= est.baseline_bytes


def test_resident_never_negative() -> None:
    est = estimate_memory("zipcache", _model(), _workload())
    assert est.resident_bytes >= 0


def test_invalid_model_geometry_raises() -> None:
    bad = ModelProfile(num_layers=0, num_kv_heads=0, head_dim=0)
    with pytest.raises(MemoryEstimateError):
        estimate_memory("kivi", bad, _workload())


def test_estimate_candidate_memory_batch() -> None:
    estimates = estimate_candidate_memory(["kivi", "h2o"], _model(), _workload())
    assert set(estimates) == {"kivi", "h2o"}
    assert isinstance(estimates["kivi"], MemoryEstimate)


def test_to_dict_shape() -> None:
    data = estimate_memory("kivi", _model(), _workload()).to_dict()
    for key in (
        "method", "baseline_bytes", "compressed_bytes", "resident_bytes",
        "reduction_ratio", "savings_percent", "confidence", "assumptions",
    ):
        assert key in data
    assert data["method"] == "kivi"


def test_reduction_ratio_bounds() -> None:
    est = estimate_memory("turboquant_prod", _model(), _workload())
    assert 0.0 <= est.reduction_ratio <= 1.0


def test_reduction_ratio_zero_baseline_guard() -> None:
    est = MemoryEstimate(method="x", baseline_bytes=0, compressed_bytes=1 << 20)
    assert est.reduction_ratio == 1.0


def test_savings_percent_matches_ratio() -> None:
    est = MemoryEstimate(method="x", baseline_bytes=1000, compressed_bytes=250)
    assert est.savings_percent == 75.0
