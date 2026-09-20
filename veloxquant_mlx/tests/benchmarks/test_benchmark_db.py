"""Tests for the persistent benchmark store (RFC Phase 4)."""

from __future__ import annotations

import json

from veloxquant_mlx.benchmarks.benchmark_db import (
    BenchmarkDatabase,
    BenchmarkMatchQuery,
    BenchmarkRecord,
    benchmark_fingerprint,
    default_benchmark_dir,
)
from veloxquant_mlx.planning.workload import WorkloadProfile
from veloxquant_mlx.profiling.hardware_profiler import HardwareProfile
from veloxquant_mlx.profiling.model_profiler import ModelProfile


def _record(**kw) -> BenchmarkRecord:
    defaults = {
        "id": "r1",
        "method": "kivi",
        "model_id": "llama-7b",
        "architecture": "llama",
        "context_length": 8192,
        "batch_size": 1,
        "memory_reduction": 0.6,
        "throughput_tok_s": 50.0,
        "latency_ms_per_token": 20.0,
    }
    defaults.update(kw)
    return BenchmarkRecord(**defaults)


def _model(model_id: str = "llama-7b", architecture: str = "llama") -> ModelProfile:
    return ModelProfile(
        model_id=model_id, architecture=architecture, num_layers=32,
        num_query_heads=32, num_kv_heads=8, head_dim=128,
    )


def _wl(context: int = 8192, batch: int = 1) -> WorkloadProfile:
    return WorkloadProfile(context_length=context, batch_size=batch)


def test_fingerprint_stable_and_input_sensitive():
    a = benchmark_fingerprint("kivi", "llama-7b", 8192, 1)
    assert a == benchmark_fingerprint("kivi", "llama-7b", 8192, 1)
    assert a != benchmark_fingerprint("h2o", "llama-7b", 8192, 1)
    assert a != benchmark_fingerprint("kivi", "llama-7b", 4096, 1)
    assert len(a) == 16


def test_default_dir_under_cache():
    assert str(default_benchmark_dir()).endswith("veloxquant/benchmarks")


def test_empty_db(tmp_path):
    db = BenchmarkDatabase(tmp_path)
    assert db.find("nope") is None
    assert db.records == {}


def test_upsert_and_reload(tmp_path):
    db = BenchmarkDatabase(tmp_path)
    rec = _record()
    db.upsert(rec)

    db2 = BenchmarkDatabase(tmp_path)
    assert db2.find("r1") == rec


def test_add_keys_by_fingerprint(tmp_path):
    db = BenchmarkDatabase(tmp_path)
    rec = db.add("kivi", "llama-7b", context_length=8192, memory_reduction=0.7)
    assert rec.id == benchmark_fingerprint("kivi", "llama-7b", 8192, 1)
    # same key -> second add overwrites
    db.add("kivi", "llama-7b", context_length=8192, memory_reduction=0.5)
    assert db.find(rec.id).memory_reduction == 0.5  # type: ignore[union-attr]


def test_remove_and_clear(tmp_path):
    db = BenchmarkDatabase(tmp_path)
    rec = db.add("h2o", "llama-7b", context_length=8192)
    assert db.remove(rec.id) is True
    assert db.remove(rec.id) is False
    db.add("h2o", "llama-7b", context_length=8192)
    db.clear()
    assert db.records == {}


def test_persistence_files_created_even_when_empty(tmp_path):
    db = BenchmarkDatabase(tmp_path)
    db.clear()
    assert db.index_path.exists()


def test_corrupt_record_falls_back_to_placeholder(tmp_path):
    db = BenchmarkDatabase(tmp_path)
    rec = db.add("kivi", "llama-7b", context_length=8192)
    (db.records_dir / f"{rec.id}.json").write_text(json.dumps({"id": rec.id}))
    db2 = BenchmarkDatabase(tmp_path)
    loaded = db2.find(rec.id)
    assert loaded is not None and loaded.id == rec.id


def test_to_dict_round_trip():
    rec = _record(workload=WorkloadProfile(context_length=8192))
    assert BenchmarkRecord.from_dict(rec.to_dict()) == rec


def test_match_score_perfect_match():
    rec = _record(chip="M4", mlx_version="0.32.2")
    query = BenchmarkMatchQuery(
        method="kivi", model_id="llama-7b", chip="M4", mlx_version="0.32.2"
    )
    assert rec.match_score(query) == 1.0


def test_match_score_unknown_axes_not_penalized():
    rec = _record(chip="unknown", mlx_version="unknown")
    query = BenchmarkMatchQuery(method="kivi", model_id="llama-7b")
    assert rec.match_score(query) == 1.0


def test_match_score_context_and_chip_penalty():
    rec = _record(context_length=8192, chip="M4")
    query = BenchmarkMatchQuery(method="kivi", model_id="llama-7b", context_length=4096, chip="M1")
    # 0.35 * 1.0 (gap 8192 vs 4096) + 0.25 * (4-1)/3 = 0.35 + 0.25
    assert abs(rec.match_score(query) - 0.40) < 1e-4


def test_match_score_batch_penalty():
    rec = _record(batch_size=4)
    query = BenchmarkMatchQuery(method="kivi", model_id="llama-7b", batch_size=1)
    score = rec.match_score(query)
    # gap = min(1, |4-1|/1) = 1.0 -> 0.25 penalty
    assert score == 0.75


def test_match_score_clamped_below_zero():
    rec = _record(context_length=1_000_000, chip="M4", mlx_version="0.1.0")
    query = BenchmarkMatchQuery(
        method="kivi", model_id="llama-7b", context_length=1, chip="M4", mlx_version="0.32.2"
    )
    assert rec.match_score(query) >= 0.0


def test_find_best_match_happy_path(tmp_path):
    db = BenchmarkDatabase(tmp_path)
    rec = db.add(
        "kivi", "llama-7b", context_length=8192, batch_size=1,
        memory_reduction=0.6, chip="M4", mlx_version="0.32.2",
        architecture="llama", tags=["metal-kernel"],
    )
    best = db.find_best_match(
        _model(), _wl(),
        methods=["kivi"],
        hardware=HardwareProfile(chip="M4", mlx_version="0.32.2"),
    )
    assert best["kivi"] == rec


def test_find_best_match_known_chip_penalizes_mismatch(tmp_path):
    db = BenchmarkDatabase(tmp_path)
    db.add("kivi", "llama-7b", context_length=8192, chip="M1", mlx_version="0.32.2")
    best = db.find_best_match(
        _model(), _wl(),
        methods=["kivi"],
        hardware=HardwareProfile(chip="M4", mlx_version="0.32.2"),
        threshold=0.8,
    )
    assert best == {}  # 0.25 chip-gap penalty drops the match below 0.8
    # without hardware info, chip unknown -> not penalized -> matches
    best_unknown = db.find_best_match(_model(), _wl(), methods=["kivi"])
    assert "kivi" in best_unknown


def test_find_best_match_wrong_model_returned_empty(tmp_path):
    db = BenchmarkDatabase(tmp_path)
    db.add("kivi", "other-model", context_length=8192)
    assert db.find_best_match(_model(), _wl(), methods=["kivi"]) == {}


def test_threshold_low_allows_fuzzy_chip_match(tmp_path):
    db = BenchmarkDatabase(tmp_path)
    db.add("kivi", "llama-7b", context_length=1_000_000, chip="M4")
    best = db.find_best_match(
        _model(), _wl(), methods=["kivi"], threshold=0.0,
        hardware=HardwareProfile(chip="M4"),
    )
    assert "kivi" in best


def test_multiple_records_picks_best_per_method(tmp_path):
    db = BenchmarkDatabase(tmp_path)
    db.add("kivi", "llama-7b", context_length=8192)
    db.add("kivi", "llama-7b", context_length=8191)
    best = db.find_best_match(_model(), _wl(8192), methods=["kivi"])
    assert best["kivi"].context_length == 8192


def test_savings_percent_clamped():
    assert _record(memory_reduction=0.0).savings_percent == 100.0
    assert _record(memory_reduction=0.2).savings_percent == 80.0
    assert _record(memory_reduction=5.0).savings_percent == 0.0
