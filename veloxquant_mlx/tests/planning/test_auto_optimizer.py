"""AutoOptimizer's live serve-tier validation of the ranked set."""

from __future__ import annotations

from types import SimpleNamespace

from veloxquant_mlx.planning import AutoOptimizer, AutoOptimizerOptions
from veloxquant_mlx.profiling.hardware_profiler import HardwareProfile
from veloxquant_mlx.profiling.model_profiler import ModelProfile


def _optimizer(probe_top_n: int) -> AutoOptimizer:
    opt = AutoOptimizer(AutoOptimizerOptions(probe_top_n=probe_top_n))
    opt._hardware = HardwareProfile(
        chip="Apple M4", chip_generation=4, available_memory_bytes=16 * 1024**3
    )
    return opt


def test_methods_promoted_after_a_drop_are_probed_too(monkeypatch):
    model = ModelProfile(num_layers=32, num_query_heads=32, num_kv_heads=8, head_dim=128)
    base = [s.method for s in _optimizer(0).recommend_strategy(model=model, max_results=6).ranked]
    assert len(base) == 6
    # The top three crash, and so does the method promoted into the top three.
    crashing = set(base[:4])

    monkeypatch.setattr(
        "veloxquant_mlx.cache.registry.probe_serve_tier",
        lambda method: SimpleNamespace(is_servable=method not in crashing),
    )
    result = _optimizer(3).recommend_strategy(model=model)

    assert result.ranked
    assert not crashing & {s.method for s in result.ranked}
