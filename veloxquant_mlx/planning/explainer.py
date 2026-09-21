"""Human-readable explanations of auto-selection outcomes.

Turns a :class:`RecommendationResult` into the plain-text breakdown the RFC
requires: how candidates were filtered, why each thing warnings/exclusions,
which figures were measured vs. estimated, and where the numbers came from.
"""

from __future__ import annotations

from veloxquant_mlx.planning.strategy_planner import RecommendationResult

__all__ = [
    "explain",
    "explain_filtering",
    "explain_hardware_model",
]

_MEMORY_UNITS = (
    (1024**4, "TiB"),
    (1024**3, "GiB"),
    (1024**2, "MiB"),
    (1024, "KiB"),
)


def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return "0 B"
    for factor, unit in _MEMORY_UNITS:
        if n >= factor:
            return f"{n / factor:.2f} {unit}"
    return f"{n} B"


def explain_hardware_model(result: RecommendationResult) -> str:
    """Paragraph describing what the planner knew about hardware and model."""
    h = result.hardware
    m = result.model
    lines = [
        f"Hardware: {h.chip}"
        f"{f' (generation {h.chip_generation})' if h.chip_generation else ''}"
        f"{f', {_fmt_bytes(h.available_memory_bytes)} available' if h.available_memory_bytes else ''}"
        f"{f', MLX {h.mlx_version}' if h.mlx_version else ''}"
        f"{f', macOS {h.macos_version}' if h.macos_version else ''}.",
        f"Model: {m.model_id} ({m.architecture}), {m.num_layers} layers, "
        f"{m.num_query_heads}q/{m.num_kv_heads}kv heads, head_dim {m.head_dim}, "
        f"{m.attention_type.upper()} attention, compute dtype {m.dtype}.",
        f"Workload: {result.workload.context_length}-token context, "
        f"{result.workload.generation_length}-token generation, "
        f"batch {result.workload.batch_size} x {result.workload.num_concurrent_requests} "
        f"concurrent; objective '{result.objective}'.",
    ]
    return "\n".join(line for line in lines if line)


def explain_filtering(result: RecommendationResult) -> str:
    """Paragraph about what passed and what was filtered out."""
    cand = result.candidates
    if result.fallback_used:
        reasons = list(cand.excluded.values())
        preview = "; ".join(reasons[:3])
        extra = f" ({len(reasons)} total)" if len(reasons) > 3 else ""
        return (
            "No strategy survived filtering — the recommendation falls back to "
            f"the library default. Reasons: {preview}{extra}."
        )
    n_viable = len(cand.viable)
    n_excluded = len(cand.excluded)
    total_considered = n_viable + n_excluded
    lines = [
        f"Filtered {total_considered} methods down to {n_viable} viable "
        f"({n_excluded} excluded below)."
    ]
    if n_excluded:
        top = sorted(cand.excluded.items(), key=lambda kv: kv[0])[:3]
        lines.append("Sample exclusions: " + "; ".join(f"{name}: {reason}" for name, reason in top))
    warned = {name for name, warns in cand.soft_warnings.items() if warns}
    if warned:
        shown = ", ".join(sorted(warned)[:5])
        lines.append(f"Caveats (non-blocking) on: {shown}.")
    return " ".join(lines)


def _evidence_line(item) -> str:
    if item.evidence == "empirical":
        return "evidence: measured benchmark record"
    if item.evidence == "analytic":
        return "evidence: analytical estimate (no benchmark record)"
    return f"evidence: {item.evidence}"


def explain(
    result: RecommendationResult,
    *,
    include_filtering: bool = True,
    warning_limit: int = 2,
) -> str:
    """Full plain-text explanation of a recommendation.

    **Output** (sections separated by blank lines):
    1. **Hardware & Model**: Chip, memory, MLX/macOS versions, model
       architecture (layers, heads, attention type)
    2. **Filtering Summary** (optional): How many methods passed/failed, why
    3. **Recommendation**: Top 1–3 methods with:
       - Overall score (0–1)
       - Memory savings % vs fp16
       - Per-axis scores (memory, latency, throughput, quality)
       - Evidence type (measured vs analytical)
       - Warnings/caveats

    **Arguments**:
        result: RecommendationResult from plan_strategy() or AutoOptimizer
        include_filtering: Include filtering summary section (default True)
        warning_limit: Max caveats shown per method (default 2; extras omitted)

    **Example output**:
        Hardware: Apple M4 (generation 4), 18 GiB available, MLX 0.19.0, macOS 15.1.
        Model: Qwen/Qwen2.5-7B (qwen), 32 layers, 28q/4kv heads, head_dim 128, GQA
        attention, compute dtype float16.
        Workload: 32768-token context, 1024-token generation, batch 1 x 1 concurrent;
        objective 'latency'.

        Filtered 43 methods down to 18 viable (25 excluded below).
        Sample exclusions: ... (reasons)

        RECOMMENDATION
        1. kivi (overall 0.856) — ~81% memory vs fp16; memory 0.92, latency 0.85,
           throughput 0.80, quality 0.95; evidence: analytical estimate.
        2. polar (overall 0.823) — ~88% memory vs fp16; ...
        3. adakv (overall 0.801) — ~79% memory vs fp16; ...
           ! adakv: quality caveat for long generations (eviction method)
    """
    sections = [explain_hardware_model(result)]
    if include_filtering:
        sections.append(explain_filtering(result))

    if result.fallback_used:
        sections.append(
            "RECOMMENDATION: fall back to the library default "
            "(turboquant_rvq). Verify the excluded reasons before choosing."
        )
        return "\n\n".join(sections)

    lines = ["RECOMMENDATION"]
    for idx, item in enumerate(result.ranked, 1):
        est = item.memory_estimate
        score = item.score
        savings = est.savings_percent
        lines.append(
            f"{idx}. {item.method} (overall {score:.3f}) — "
            f"~{savings:.0f}% memory vs fp16; "
            f"memory {item.objective_scores.get('memory', 0.0):.2f}, "
            f"latency {item.objective_scores.get('latency', 0.0):.2f}, "
            f"throughput {item.objective_scores.get('throughput', 0.0):.2f}, "
            f"quality {item.objective_scores.get('quality', 0.0):.2f}; "
            f"{_evidence_line(item)}."
        )
        if item.warnings:
            for warning in item.warnings[:warning_limit]:
                lines.append(f"   ! {item.method}: {warning}")
        if len(item.warnings) > warning_limit:
            lines.append(f"   ... and {len(item.warnings) - warning_limit} more caveats")
    sections.append("\n".join(lines))
    return "\n\n".join(sections)
