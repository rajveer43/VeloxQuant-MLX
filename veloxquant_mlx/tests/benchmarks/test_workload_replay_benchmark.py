"""Tests for the KV-cache workload replay benchmark (issue #258).

Uses small, fast scenarios (short prompts, few decode steps) so the whole
suite stays cheap on CI — the point here is to validate the replay engine's
bookkeeping (TTFT/decode/memory/compression aggregation), not to produce
meaningful performance numbers.
"""

from __future__ import annotations

from veloxquant_mlx.benchmarks.workload_replay_benchmark import (
    STANDARD_WORKLOADS,
    WorkloadResult,
    WorkloadScenario,
    format_summary_table,
    results_to_json,
    run_suite,
    run_workload,
)

HEAD_DIM = 64


def _small_scenario(**overrides) -> WorkloadScenario:
    base = dict(
        name="tiny",
        description="tiny test scenario",
        prompt_lens=[4],
        n_new_tokens=3,
        n_streams=1,
        repeat=1,
        reuse_cache=False,
        sliding_window=None,
    )
    base.update(overrides)
    return WorkloadScenario(**base)


def test_standard_workloads_are_well_formed():
    assert set(STANDARD_WORKLOADS) == {
        "single_request",
        "long_context",
        "batch_generation",
        "variable_seq_lengths",
        "repeated_requests",
        "cache_growth",
        "cache_eviction_reuse",
    }
    for name, w in STANDARD_WORKLOADS.items():
        assert w.name == name
        assert w.prompt_lens
        assert w.n_new_tokens > 0
        assert w.n_streams >= 1
        assert w.repeat >= 1


def test_run_workload_single_stream_basic_metrics():
    workload = _small_scenario()
    result = run_workload("turboquant_prod", workload, head_dim=HEAD_DIM, bits=2, seed=1)

    assert isinstance(result, WorkloadResult)
    assert result.method == "turboquant_prod"
    assert result.workload == "tiny"
    # 1 stream * (4 prefill + 3 decode) tokens.
    assert result.total_tokens == 7
    assert result.wall_time_s > 0.0
    assert result.ttft_ms_mean >= 0.0
    assert result.decode_ms_mean >= 0.0
    assert result.decode_ms_p50 >= 0.0
    assert result.decode_ms_p95 >= result.decode_ms_p50
    assert result.tokens_per_sec > 0.0
    assert result.peak_memory_bytes > 0
    assert result.compression_ratio > 1.0  # 2-bit quantization should beat fp16
    assert result.quantize_overhead_ms >= 0.0
    assert result.dequantize_overhead_ms >= 0.0
    assert result.memory_snapshots == []  # reuse_cache=False -> no growth checkpoints


def test_run_workload_multi_stream_scales_tokens():
    workload = _small_scenario(n_streams=3)
    result = run_workload("turboquant_prod", workload, head_dim=HEAD_DIM, bits=2, seed=2)
    assert result.n_streams == 3
    assert result.total_tokens == 3 * 7


def test_run_workload_repeat_without_reuse_is_cold_start():
    workload = _small_scenario(repeat=4, reuse_cache=False)
    result = run_workload("turboquant_prod", workload, head_dim=HEAD_DIM, bits=2, seed=3)
    assert result.total_tokens == 4 * 7
    assert result.memory_snapshots == []


def test_run_workload_reuse_cache_grows_and_snapshots():
    workload = _small_scenario(prompt_lens=[1], n_new_tokens=5, repeat=3, reuse_cache=True)
    result = run_workload("turboquant_prod", workload, head_dim=HEAD_DIM, bits=2, seed=4)

    assert len(result.memory_snapshots) == 3
    # Cache grows monotonically across checkpoints since it's never reset.
    token_counts = [s.n_tokens for s in result.memory_snapshots]
    assert token_counts == sorted(token_counts)
    assert token_counts[-1] == 3 * (1 + 5)
    mem_bytes = [s.memory_bytes for s in result.memory_snapshots]
    assert mem_bytes == sorted(mem_bytes)


def test_run_workload_sliding_window_caps_growth():
    grown = _small_scenario(prompt_lens=[1], n_new_tokens=20, repeat=1, reuse_cache=True)
    capped = _small_scenario(
        prompt_lens=[1], n_new_tokens=20, repeat=1, reuse_cache=True, sliding_window=4
    )

    uncapped_result = run_workload("turboquant_prod", grown, head_dim=HEAD_DIM, bits=2, seed=5)
    capped_result = run_workload("turboquant_prod", capped, head_dim=HEAD_DIM, bits=2, seed=5)

    assert capped_result.memory_snapshots[-1].memory_bytes <= (
        uncapped_result.memory_snapshots[-1].memory_bytes
    )


def test_run_suite_builds_nested_results():
    workloads = {"tiny": _small_scenario()}
    results = run_suite(
        ["turboquant_prod", "turboquant_mse"], workloads, head_dim=HEAD_DIM, bits=2, seed=6
    )
    assert set(results) == {"turboquant_prod", "turboquant_mse"}
    for per_workload in results.values():
        assert set(per_workload) == {"tiny"}
        assert isinstance(per_workload["tiny"], WorkloadResult)


def test_format_summary_table_contains_expected_columns():
    workloads = {"tiny": _small_scenario()}
    results = run_suite(["turboquant_prod"], workloads, head_dim=HEAD_DIM, bits=2, seed=7)
    table = format_summary_table(results)
    assert "turboquant_prod" in table
    assert "tiny" in table
    assert "TTFT(ms)" in table
    assert "Tok/s" in table
    assert "Compr." in table


def test_results_to_json_is_serializable_and_round_trips():
    import json

    workloads = {"tiny": _small_scenario(reuse_cache=True, repeat=2, prompt_lens=[1])}
    results = run_suite(["turboquant_prod"], workloads, head_dim=HEAD_DIM, bits=2, seed=8)
    payload = results_to_json(results)
    dumped = json.dumps(payload)
    reloaded = json.loads(dumped)

    r = results["turboquant_prod"]["tiny"]
    assert reloaded["turboquant_prod"]["tiny"]["total_tokens"] == r.total_tokens
    assert reloaded["turboquant_prod"]["tiny"]["compression_ratio"] == r.compression_ratio
    assert len(reloaded["turboquant_prod"]["tiny"]["memory_snapshots"]) == len(
        r.memory_snapshots
    )
