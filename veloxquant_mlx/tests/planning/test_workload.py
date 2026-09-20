"""Tests for the workload profile (RFC Phase 3)."""

from __future__ import annotations

from veloxquant_mlx.planning.workload import (
    WorkloadObjective,
    WorkloadProfile,
    workload_from_dict,
)


def test_defaults():
    w = WorkloadProfile()
    assert w.context_length == 4096
    assert w.generation_length == 512
    assert w.batch_size == 1
    assert w.num_concurrent_requests == 1
    assert w.objective == WorkloadObjective.BALANCED


def test_total_tokens_per_request():
    w = WorkloadProfile(context_length=8000, generation_length=2000)
    assert w.total_tokens_per_request == 10000


def test_effective_batch():
    w = WorkloadProfile(batch_size=4, num_concurrent_requests=3)
    assert w.effective_batch == 12
    assert WorkloadProfile().effective_batch == 1


def test_properties_enabled():
    assert WorkloadProfile(context_length=123).context_length == 123


def test_to_dict_round_trip():
    w = WorkloadProfile(
        context_length=1000,
        generation_length=2000,
        batch_size=2,
        num_concurrent_requests=5,
        objective=WorkloadObjective.MEMORY,
        max_latency_ms=250.0,
    )
    restored = workload_from_dict(w.to_dict())
    assert restored == w


def test_from_dict_missing_keys_use_defaults():
    restored = workload_from_dict({})
    assert restored.context_length == 4096
    assert restored.objective == WorkloadObjective.BALANCED


def test_from_dict_ignores_extra_keys():
    restored = workload_from_dict({"context_length": 10, "bogus": "x"})
    assert restored.context_length == 10
    assert not hasattr(restored, "bogus")


def test_objective_constants_distinct():
    names = {
        WorkloadObjective.MEMORY,
        WorkloadObjective.LATENCY,
        WorkloadObjective.THROUGHPUT,
        WorkloadObjective.QUALITY,
        WorkloadObjective.BALANCED,
    }
    assert len(names) == 5


def test_dataclass_fields_match_to_dict():
    w = WorkloadProfile()
    data = w.to_dict()
    assert set(data) == {
        "context_length",
        "generation_length",
        "batch_size",
        "num_concurrent_requests",
        "objective",
        "max_latency_ms",
    }
