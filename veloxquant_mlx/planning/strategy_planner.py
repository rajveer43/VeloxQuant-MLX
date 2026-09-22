"""Strategy planner: rank viable KV-cache methods under an objective.

Takes the filtered candidate set from :mod:`candidate_filter`, scores each on
four normalized axes (memory, latency, throughput, quality), blends them with
the workload's objective weights, and optionally mixes in *measured* numbers
from the benchmark database. Returns a ranked
:class:`RecommendationResult` with exactly the facts the explainer needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from veloxquant_mlx.cache.registry import static_method_info
from veloxquant_mlx.planning.candidate_filter import (
    CandidateFilterOptions,
    CandidateFilterResult,
    filter_candidates,
)
from veloxquant_mlx.planning.memory_estimator import (
    MemoryEstimate,
    estimate_candidate_memory,
    method_quant_bits,
)
from veloxquant_mlx.planning.workload import WorkloadObjective, WorkloadProfile
from veloxquant_mlx.profiling.hardware_profiler import HardwareProfile
from veloxquant_mlx.profiling.model_profiler import ModelProfile

__all__ = [
    "ScoredMethod",
    "RecommendationResult",
    "PlanningOptions",
    "DEFAULT_OBJECTIVE_WEIGHTS",
    "plan_strategy",
    "recommend_strategy",
]

EvidenceSource = Literal["analytic", "empirical", "hybrid", "fallback"]

#: Objective -> per-axis weight. Keys mirror :class:`WorkloadObjective`.
#: Weights are normalized (divided by their sum) before use, so the exact
#: magnitudes only encode *relative* priorities.
DEFAULT_OBJECTIVE_WEIGHTS: dict[str, dict[str, float]] = {
    WorkloadObjective.MEMORY: {"memory": 0.8, "latency": 0.1, "throughput": 0.1, "quality": 0.2},
    WorkloadObjective.LATENCY: {"memory": 0.1, "latency": 0.6, "throughput": 0.3, "quality": 0.0},
    WorkloadObjective.THROUGHPUT: {
        "memory": 0.05,
        "latency": 0.15,
        "throughput": 0.8,
        "quality": 0.0,
    },
    WorkloadObjective.QUALITY: {"memory": 0.15, "latency": 0.1, "throughput": 0.0, "quality": 0.75},
    WorkloadObjective.BALANCED: {
        "memory": 0.4,
        "latency": 0.25,
        "throughput": 0.25,
        "quality": 0.1,
    },
}

#: Class-based latency multiplier: quantized caches pay per-byte decode+dequant,
#: eviction caches pay a merge/revive path. Applied to the analytical
#: bytes-per-token proxy, never to measured numbers.
_METHOD_CLASS_FACTOR: dict[str, float] = {
    "plain": 1.0,
    "quant": 1.08,
    "eviction": 1.05,
    "hybrid": 1.1,
}


def _class_for(info: Any) -> str:
    caps = info.capabilities
    if caps.uses_eviction and not caps.compresses_keys:
        return "eviction"
    if caps.uses_eviction and caps.compresses_keys:
        return "hybrid"
    if caps.compresses_keys:
        return "quant"
    return "plain"


def _normalize_axis(values: dict[str, float], lower_is_better: bool = True) -> dict[str, float]:
    """Min-max normalize a per-method axis to 0..1; flat input -> all 1.0."""
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return dict.fromkeys(values, 1.0)
    if lower_is_better:
        return {k: 1.0 - (v - lo) / (hi - lo) for k, v in values.items()}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _retention(
    method: str,
    estimate: MemoryEstimate,
    model: ModelProfile,
    workload: WorkloadProfile,
) -> float:
    """Fraction of generated tokens the cache actually retains (0..1).

    Eviction caches pin their footprint at a fixed budget, so over a long
    context they drop most tokens — that is a quality loss *by definition*,
    not a caveat. Quantization methods retain everything (retention 1.0).
    """
    _, _, evicts = method_quant_bits(method)
    if not evicts:
        return 1.0
    per_token = model.baseline_kv_bytes_per_token * model.num_layers * workload.effective_batch
    budget_tokens = max(1, estimate.compressed_bytes // max(1, per_token))
    return min(1.0, budget_tokens / max(1, workload.total_tokens_per_request))


def _quality_score(
    method: str,
    estimate: MemoryEstimate,
    info: Any,
    retention: float,
) -> float:
    key_bits, value_bits, _ = method_quant_bits(method)
    eff_bits = (key_bits + value_bits) / 2.0
    bit_fidelity = 0.55 + 0.45 * (eff_bits / 16.0)
    q = retention * bit_fidelity
    if info.capabilities.requires_calibration:
        q += 0.05
    return round(min(1.0, max(0.05, q)), 4)


@dataclass
class PlanningOptions:
    """Tunables for :func:`plan_strategy`; mirrors
    :class:`CandidateFilterOptions` plus ranking inputs."""

    max_results: int = 3
    objective_weights: dict[str, float] | None = None
    memory_budget_bytes: int | None = None
    prefer_no_calibration: bool = False
    require_metal: bool = False
    empirical: dict[str, Any] = field(default_factory=dict)
    additional_exclusions: set[str] = field(default_factory=set)
    seed: int | None = None


@dataclass
class ScoredMethod:
    """One ranked recommendation candidate with its scores and evidence."""

    method: str
    score: float
    objective_scores: dict[str, float]
    memory_estimate: MemoryEstimate
    evidence: EvidenceSource
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize score, objective_scores, memory, evidence, and warnings to a plain dict."""
        return {
            "method": self.method,
            "score": round(self.score, 4),
            "objective_scores": {k: round(v, 4) for k, v in self.objective_scores.items()},
            "memory": self.memory_estimate.to_dict(),
            "evidence": self.evidence,
            "warnings": list(self.warnings),
        }


@dataclass
class RecommendationResult:
    """Full outcome of a planning run, ready for the explainer / CLI."""

    objective: str
    ranked: list[ScoredMethod]
    candidates: CandidateFilterResult
    model: ModelProfile
    hardware: HardwareProfile
    workload: WorkloadProfile
    fallback_used: bool = False
    fallback_reason: str | None = None

    @property
    def best(self) -> ScoredMethod | None:
        """Top-ranked method, or None if nothing survived filtering."""
        return self.ranked[0] if self.ranked else None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full result (ranked list, candidates, inputs) to a plain dict."""
        return {
            "objective": self.objective,
            "ranked": [item.to_dict() for item in self.ranked],
            "candidates": self.candidates.to_dict(),
            "model": self.model.to_dict(),
            "hardware": self.hardware.to_dict(),
            "workload": self.workload.to_dict(),
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


def _weights_for(objective: str, override: dict[str, float] | None) -> dict[str, float]:
    if override is not None and override:
        return {k: float(v) for k, v in override.items() if float(v)}
    base = DEFAULT_OBJECTIVE_WEIGHTS.get(
        objective, DEFAULT_OBJECTIVE_WEIGHTS[WorkloadObjective.BALANCED]
    )
    total = sum(base.values())
    return {k: v / total for k, v in base.items()}


def plan_strategy(
    model: ModelProfile,
    hardware: HardwareProfile,
    workload: WorkloadProfile,
    *,
    options: PlanningOptions | None = None,
    filter_options: CandidateFilterOptions | None = None,
) -> RecommendationResult:
    """Rank the viable strategies for ``model``/``hardware``/``workload``.

    **Process**:
    1. Estimate memory for all registered methods (cheap, no weights loaded)
    2. Filter by attention type, Metal availability, memory budget
    3. Score survivors on 4 normalized axes: memory, latency, throughput, quality
    4. Blend scores by objective weights (memory vs latency vs quality trade-offs)
    5. Mix in measured benchmarks if available (overrides analytical estimates)
    6. Rank by composite score (deterministic for identical inputs)

    **Fallback**: If nothing survives filtering, returns empty ranked list with
    ``fallback_used=True``; the caller (``AutoOptimizer``) then falls back to
    the library default method (e.g., ``turboquant_rvq``).

    **Arguments**:
        model: Extracted model architecture (layers, heads, dtype)
        hardware: Detected Apple Silicon environment (chip, memory, bandwidth)
        workload: User intent (context length, objective)
        options: Ranking tunables (objective weights, max results to return)
        filter_options: Filtering overrides (memory budget, calibration preference)

    **Returns**:
        RecommendationResult with:
        - ranked: List of ScoredMethod objects (best first)
        - candidates: Filter result with viable/excluded methods
        - fallback_used: True if no viable candidates survived
        - evidence: Per-method data (memory, latency, confidence)

    **Determinism**: Identical inputs always produce identical output. Candidates
    are sorted before scoring to ensure consistent ranking even in ties.
    """
    opts = options or PlanningOptions()
    fopts = filter_options or CandidateFilterOptions(
        memory_budget_bytes=opts.memory_budget_bytes,
        prefer_no_calibration=opts.prefer_no_calibration,
        require_metal=opts.require_metal,
        registry_lookup=static_method_info,
    )
    # Estimates come first so the filter can apply its memory-budget check.
    # Estimate every registered method (cheap), not just the empirical set —
    # the filter needs an estimate to vet the whole registry.
    from veloxquant_mlx.cache.registry import all_method_names

    estimates = estimate_candidate_memory(list(all_method_names()), model, workload)
    candidates = filter_candidates(model, hardware, workload, estimates, options=fopts)
    if opts.additional_exclusions:
        for name in opts.additional_exclusions:
            if name in candidates.viable:
                candidates.viable.remove(name)
                candidates.excluded[name] = (
                    "excluded by previous serve-tier probe (crashes at request time)"
                )
    if not candidates.viable:
        return RecommendationResult(
            objective=workload.objective,
            ranked=[],
            candidates=candidates,
            model=model,
            hardware=hardware,
            workload=workload,
            fallback_used=True,
            fallback_reason=(
                "no viable candidate (see candidates.excluded); falling back "
                "to the library default method"
            ),
        )

    weights = _weights_for(workload.objective, opts.objective_weights)
    # NOTE (VeloxQuant-MLX#509): bandwidth is a single scalar shared by every
    # candidate below, so it divides every candidate's latency_ms by the
    # identical constant. _normalize_axis()'s min-max normalization is
    # algebraically invariant to a uniform positive scalar applied to every
    # input, so bandwidth currently cancels out of the ranking entirely --
    # verified: substituting values from 20 to 400 GB/s produces zero
    # recommendation flips. This means hardware.bandwidth_gbps's accuracy is
    # NOT currently load-bearing. If this formula ever changes to use
    # bandwidth non-relatively (e.g. an absolute ms/token SLA gate, or
    # bandwidth entering a hard filter instead of this shared normalized
    # scalar), that inertness breaks immediately -- re-verify the invariance
    # assumption before relying on bandwidth's absolute accuracy elsewhere.
    bandwidth = hardware.bandwidth_gbps or 100.0

    # --- Analytical per-axis values, pre-normalization --------------------
    latency_ms: dict[str, float] = {}
    throughput_s: dict[str, float] = {}
    for name in candidates.viable:
        per_token = max(
            1, estimates[name].compressed_bytes // max(1, workload.total_tokens_per_request)
        )
        factor = _METHOD_CLASS_FACTOR[_class_for(candidates.method_info[name])]
        latency_ms[name] = (per_token / bandwidth) * factor
        throughput_s[name] = 1.0 / latency_ms[name]

    # --- Benchmark overrides (measured beats proxy, same units) -----------
    evidence_notes: dict[str, list[str]] = {n: [] for n in candidates.viable}
    evidence_kind: dict[str, EvidenceSource] = dict.fromkeys(candidates.viable, "analytic")
    for name in candidates.viable:
        record = opts.empirical.get(name)
        if record is None:
            continue
        notes = evidence_notes[name]
        if record.memory_reduction and 0.0 < record.memory_reduction <= 1.0:
            estimates[name] = _patched_estimate(estimates[name], record)
            notes.append(f"measured {record.savings_percent:.1f}% memory savings")
            evidence_kind[name] = "empirical"
        if record.latency_ms_per_token and record.latency_ms_per_token > 0:
            latency_ms[name] = float(record.latency_ms_per_token)
            throughput_s[name] = 1.0 / max(latency_ms[name], 1e-9)
            notes.append(f"measured {record.latency_ms_per_token:.1f} ms/token")
            evidence_kind[name] = "empirical"
        if record.throughput_tok_s and record.throughput_tok_s > 0:
            throughput_s[name] = float(record.throughput_tok_s)
            notes.append(f"measured {record.throughput_tok_s:.1f} tok/s")
            evidence_kind[name] = "empirical"

    # --- Normalized axis scores (0 = worst, 1 = best) ----------------------
    latency_norm = _normalize_axis(latency_ms, lower_is_better=True)
    throughput_norm = _normalize_axis(throughput_s, lower_is_better=False)
    memory_norm = {n: 1.0 - estimates[n].reduction_ratio for n in candidates.viable}
    quality_norm = {
        n: _quality_score(
            n,
            estimates[n],
            candidates.method_info[n],
            _retention(n, estimates[n], model, workload),
        )
        for n in candidates.viable
    }

    scores: dict[str, float] = {
        n: round(
            sum(
                weights.get(axis, 0.0) * score
                for axis, score in {
                    "memory": memory_norm[n],
                    "latency": latency_norm[n],
                    "throughput": throughput_norm[n],
                    "quality": quality_norm[n],
                }.items()
            ),
            4,
        )
        for n in candidates.viable
    }

    ranked = [
        ScoredMethod(
            method=name,
            score=scores[name],
            objective_scores={
                "memory": memory_norm[name],
                "latency": latency_norm[name],
                "throughput": throughput_norm[name],
                "quality": quality_norm[name],
            },
            memory_estimate=estimates[name],
            evidence=evidence_kind[name],
            warnings=candidates.soft_warnings.get(name, []) + evidence_notes[name],
        )
        for name in sorted(candidates.viable, key=lambda n: (scores[n], n), reverse=True)[
            : opts.max_results
        ]
    ]
    return RecommendationResult(
        objective=workload.objective,
        ranked=ranked,
        candidates=candidates,
        model=model,
        hardware=hardware,
        workload=workload,
    )


def _patched_estimate(estimate: MemoryEstimate, record: Any) -> MemoryEstimate:
    """Rebuild an estimate whose compressed footprint came from a benchmark."""
    compressed = max(1, int(estimate.baseline_bytes * record.memory_reduction))
    return MemoryEstimate(
        method=estimate.method,
        baseline_bytes=estimate.baseline_bytes,
        compressed_bytes=compressed,
        workspace_bytes=estimate.workspace_bytes,
        peak_bytes=max(estimate.peak_bytes, compressed),
        resident_bytes=compressed,
        confidence=estimate.confidence,
        assumptions=list(estimate.assumptions)
        + ["compressed bytes overridden by benchmark record"],
    )


def recommend_strategy(
    model_config: dict[str, Any] | None = None,
    *,
    model: ModelProfile | None = None,
    hardware: HardwareProfile | None = None,
    workload: WorkloadProfile | None = None,
    **plan_kwargs: Any,
) -> RecommendationResult:
    """One-call convenience: detect hardware, profile model, then plan.

    **Summary**: Wraps the full pipeline for stateless, single-use recommendation.
    ``AutoOptimizer.recommend_strategy`` adds caching on top of this.

    **Arguments**:
        model_config: HuggingFace-style config dict for architecture extraction
            (num_layers, hidden_size, num_attention_heads, etc.). One of
            ``model_config`` or ``model`` must be given.
        model: Prebuilt ModelProfile; skips config parsing if provided.
        hardware: Prebuilt HardwareProfile; if None, auto-detects (one-time cost).
        workload: WorkloadProfile; defaults to balanced 4K context.
        **plan_kwargs: Forwarded to :func:`plan_strategy` as PlanningOptions
            (objective_weights, memory_budget_bytes, prefer_no_calibration, etc.)

    **Returns**:
        RecommendationResult with recommendation.method, ranked alternatives,
        and evidence for explainability.

    **Example**:
        >>> result = recommend_strategy(
        ...     model_config={"num_layers": 32, "hidden_size": 4096},
        ...     workload=WorkloadProfile(context_length=32768, objective="latency")
        ... )
        >>> print(result.ranked[0].method)  # Top recommendation
    """
    if hardware is None:
        hardware = HardwareProfile.detect()
    if model is None:
        from veloxquant_mlx.profiling.model_profiler import profile_model_from_config

        if model_config is None:
            raise ValueError(
                "recommend_strategy needs a `model_config` dict (HF config.json) "
                "or a prebuilt `model=ModelProfile`"
            )
        model = profile_model_from_config(model_config)
    if workload is None:
        workload = WorkloadProfile()
    return plan_strategy(model, hardware, workload, **plan_kwargs)
