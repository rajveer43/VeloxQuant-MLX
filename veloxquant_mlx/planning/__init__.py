"""Automatic KV-cache strategy selection (RFC ``method="auto"``).

The public surface is :class:`AutoOptimizer`:

    from veloxquant_mlx.planning import AutoOptimizer
    result = AutoOptimizer().recommend_strategy(model_config={...})

It profiles hardware + model, filters the registry by capability and memory
budget, ranks survivors under the workload's objective, cross-checks the
top choices against the live (probed) serve tier, and can fall back to the
library default when nothing fits — exactly the RFC's ``method="auto"`` path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "AutoOptimizer",
    "AutoOptimizerOptions",
    # planning submodules re-exported for convenience
    "plan_strategy",
    "recommend_strategy",
    "explain",
    "WorkloadProfile",
    "WorkloadObjective",
    "ModelProfile",
    "HardwareProfile",
    "RecommendationResult",
    "MemoryEstimate",
    "CandidateFilterResult",
]

from veloxquant_mlx.planning.candidate_filter import CandidateFilterResult
from veloxquant_mlx.planning.explainer import explain
from veloxquant_mlx.planning.memory_estimator import (
    MemoryEstimate,
    estimate_candidate_memory,
)
from veloxquant_mlx.planning.strategy_planner import (
    PlanningOptions,
    RecommendationResult,
    plan_strategy,
    recommend_strategy,
)
from veloxquant_mlx.planning.workload import WorkloadObjective, WorkloadProfile
from veloxquant_mlx.profiling.hardware_profiler import HardwareProfile
from veloxquant_mlx.profiling.model_profiler import (
    ModelProfile,
    profile_model_from_config,
)


@dataclass
class AutoOptimizerOptions:
    """Constructor options for :class:`AutoOptimizer`.

    Attributes:
        benchmark_db_dir: Path to a :class:`BenchmarkDatabase` storage
            directory; enabled only when explicitly given, so a stray dir
            never changes recommendations behind the user's back.
        probe_top_n: How many of the ranked candidates are validated against
            the live serve-tier probe before committing. ``0`` skips probing
            (faster, but a crash-prone standalone method could be recommended).
    """

    benchmark_db_dir: str | None = None
    probe_top_n: int = 3


#: Fallback method when nothing survives filtering (library default).
FALLBACK_METHOD = "turboquant_rvq"


class AutoOptimizer:
    """One-stop automatic strategy selector with real-time serve-tier guard.

    Deliberately stateful: ``detect_hardware`` / ``profile_model`` results and
    the benchmark database are cached on the instance so repeated
    ``recommend_strategy`` calls do not re-detect or re-probe.
    """

    def __init__(self, options: AutoOptimizerOptions | None = None) -> None:
        self.options = options or AutoOptimizerOptions()
        self._hardware: HardwareProfile | None = None
        self._benchmarks: Any | None = None

    # -- profiling -----------------------------------------------------------

    def detect_hardware(self) -> HardwareProfile:
        if self._hardware is None:
            self._hardware = HardwareProfile.detect()
        return self._hardware

    def profile_model(
        self, config: dict[str, Any] | None = None, **overrides: Any
    ) -> ModelProfile:
        """Profile a model from a HF-style config dict + optional overrides."""
        return profile_model_from_config(config, **overrides)

    def estimate_memory(
        self, workload: WorkloadProfile, model: ModelProfile | None = None
    ) -> dict[str, MemoryEstimate]:
        """Analytical per-method memory estimates for a workload."""
        if model is None:
            raise ValueError(
                "estimate_memory needs a `model`; the pipeline cannot invent "
                "attention geometry without one"
            )
        from veloxquant_mlx.cache.registry import all_method_names

        return estimate_candidate_memory(list(all_method_names()), model, workload)

    @property
    def _db(self) -> Any | None:
        if self._benchmarks is None and self.options.benchmark_db_dir:
            from veloxquant_mlx.benchmarks.benchmark_db import BenchmarkDatabase

            self._benchmarks = BenchmarkDatabase(self.options.benchmark_db_dir)
        return self._benchmarks

    # -- recommendation --------------------------------------------------------

    def recommend_strategy(
        self,
        model_config: dict[str, Any] | None = None,
        *,
        model: ModelProfile | None = None,
        workload: WorkloadProfile | None = None,
        objective: str | None = None,
        **kwargs: Any,
    ) -> RecommendationResult:
        """Recommend + validate the best KV-cache strategy for a model.

        :param model_config: HF-style config.json for architecture extraction.
        :param model: Prebuilt :class:`ModelProfile` (skips config parsing).
        :param workload: :class:`WorkloadProfile`; defaults to balanced.
        :param objective: Overrides ``workload.objective`` convenience.
        :param kwargs: Forwarded to :class:`PlanningOptions`
            (``memory_budget_bytes``, ``prefer_no_calibration``, ...).
        """
        if model is None:
            if model_config is None:
                raise ValueError("model_config required unless a model= is given")
            model = self.profile_model(model_config)

        workload = workload or WorkloadProfile()
        if objective is not None:
            from dataclasses import replace

            workload = replace(workload, objective=objective)

        hardware = self.detect_hardware()
        empirical = self._empirical_lookup(model, workload)
        excluded: set[str] = set()

        planning = PlanningOptions(
            empirical=empirical,
            additional_exclusions=excluded,
            **{
                key: value
                for key, value in kwargs.items()
                if key in PlanningOptions.__dataclass_fields__
            },
        )
        result = plan_strategy(
            model, hardware, workload, options=planning
        )

        # Validate the top picks with the real serve-tier probe and re-run
        # excluding any that crash at request time.
        if not result.fallback_used and self.options.probe_top_n:
            dropped = self._probe_and_drop(result)
            if dropped:
                excluded.update(dropped)
                planning = PlanningOptions(
                    empirical=empirical,
                    additional_exclusions=excluded,
                    **{
                        key: value
                        for key, value in kwargs.items()
                        if key in PlanningOptions.__dataclass_fields__
                    },
                )
                result = plan_strategy(model, hardware, workload, options=planning)

        return result

    def _empirical_lookup(self, model: ModelProfile, workload: WorkloadProfile) -> dict[str, Any]:
        db = self._db
        if db is None:
            return {}
        try:
            return dict(db.find_best_match(model, workload, hardware=self.detect_hardware()))
        except Exception:
            return {}  # corrupt/unsupported store must not break recommendation

    def _probe_and_drop(self, result: RecommendationResult) -> set[str]:
        """Serve-tier probe for the ranked set; returns crash-tier methods."""
        from veloxquant_mlx.cache.registry import probe_serve_tier

        dropped: set[str] = set()
        for item in result.ranked:
            try:
                tier = probe_serve_tier(item.method)
                if not tier.is_servable:
                    dropped.add(item.method)
            except Exception:
                dropped.add(item.method)
        return dropped

    def explain(self, result: RecommendationResult, **kwargs: Any) -> str:
        """Render :func:`~veloxquant_mlx.planning.explainer.explain` text."""
        return explain(result, **kwargs)

    def fallback_method(self) -> str:
        return FALLBACK_METHOD
