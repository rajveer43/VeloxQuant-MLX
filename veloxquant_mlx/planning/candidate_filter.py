"""Candidate filtering for automatic strategy selection.

Filters the full method registry down to the set that *can* serve the given
model + workload on the given hardware, before any scoring happens. Each
rejection is recorded with a human-readable reason, and borderline issues
(needs calibration, no Metal kernel, quality caveat) surface as soft warnings
instead of silent drops — so the explainer can say *why* and the CLI can offer
``--include-calibration`` to relax the hard rule.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from veloxquant_mlx.cache.registry import MethodInfo, get_method
from veloxquant_mlx.planning.memory_estimator import MemoryEstimate
from veloxquant_mlx.planning.workload import WorkloadObjective, WorkloadProfile
from veloxquant_mlx.profiling.hardware_profiler import HardwareProfile
from veloxquant_mlx.profiling.model_profiler import ModelProfile

__all__ = [
    "CandidateFilterResult",
    "CandidateFilterOptions",
    "filter_candidates",
]

#: Methods users should never be told to batch-on unless they asked; eviction
#: drops historically-important (just) tokens, which degrades quality under
#: long generations. Used only for warning text, never for exclusion.
_EVICTION_QUALITY_WARNING = (
    "drops old tokens from cache when the budget fills; quality degrades on "
    "long generations faster than compression-only methods"
)


@dataclass
class CandidateFilterOptions:
    """Tunables for :func:`filter_candidates`.

    Attributes:
        memory_budget_bytes: Hard cap on compressed KV bytes; candidates whose
            estimated footprint exceeds it are excluded. ``None`` uses the
            hardware's available memory.
        prefer_no_calibration: Exclude methods that need an offline
            calibration pass (they add setup latency and a golden-dataset
            dependency before the cache can serve).
        require_metal: Exclude methods with no Metal kernel (they exercise a
            slower reference path on Apple Silicon).
        minimum_context_length: Workloads shorter than this (in tokens) don't
            benefit from compression; excludes compression-only methods as a
            soft warning rather than a hard rule.
        registry_lookup: How to fetch a registry entry; overridable for tests.
    """

    memory_budget_bytes: int | None = None
    prefer_no_calibration: bool = False
    require_metal: bool = False
    minimum_context_length: int = 2048
    registry_lookup: Callable[[str], MethodInfo] = get_method


@dataclass
class CandidateFilterResult:
    """Outcome of one filtering pass.

    Attributes:
        viable: Method names that passed all hard checks, insertion-ordered.
        excluded: Method name -> human-readable reason for each hard rejection.
        soft_warnings: Method name -> human-readable caveats (non-blocking).
        method_info: MethodInfo for every method considered, for the explainer.
    """

    viable: list[str] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)
    soft_warnings: dict[str, list[str]] = field(default_factory=dict)
    method_info: dict[str, MethodInfo] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "viable": list(self.viable),
            "excluded": dict(self.excluded),
            "soft_warnings": {
                name: list(reasons) for name, reasons in self.soft_warnings.items()
            },
            "methods": {
                name: info.to_dict() for name, info in self.method_info.items()
            },
        }


def _memory_budget(
    options: CandidateFilterOptions, hardware: HardwareProfile
) -> int | None:
    if options.memory_budget_bytes is not None:
        return options.memory_budget_bytes
    avail = hardware.available_memory_bytes
    if avail > 0:
        return avail
    return None


def filter_candidates(
    model: ModelProfile,
    hardware: HardwareProfile,
    workload: WorkloadProfile,
    estimates: Mapping[str, MemoryEstimate],
    *,
    options: CandidateFilterOptions | None = None,
    methods: Sequence[str] | None = None,
) -> CandidateFilterResult:
    """Return every method capable of serving ``model`` under ``workload``.

    :param estimates: Per-method :class:`MemoryEstimate` from the memory
        estimator; only its ``compressed_bytes`` is consulted for the budget
        check.
    :param methods: Restrict filtering to these method names (defaults to the
        whole registry). Useful for tests and for post-probe re-ranking, where
        a probe already dropped the crashed set.
    """
    opts = options or CandidateFilterOptions()
    lookup = opts.registry_lookup
    budget = _memory_budget(opts, hardware)
    result = CandidateFilterResult()

    from veloxquant_mlx.cache.registry import all_method_names

    considered = methods if methods is not None else all_method_names()
    for name in considered:
        info = lookup(name)
        result.method_info[name] = info
        caps = info.capabilities
        warnings: list[str] = []

        if not info.serve_tier.is_servable:
            result.excluded[name] = (
                f"serve tier {info.serve_tier.label} (not servable): "
                f"{info.unsupported_reason or 'no serving probe result'}"
            )
            continue

        # Attention-geometry match: MHA/GQA/MQA methods are often GQA-only.
        attn = model.attention_type
        supports_key = {
            "mha": caps.supports_mha,
            "gqa": caps.supports_gqa,
            "mqa": caps.supports_mqa,
        }
        if not supports_key.get(attn, True):
            result.excluded[name] = f"does not support the {attn} attention layout"
            continue

        # Hard memory budget check (compressed KV footprint must fit).
        estimate = estimates.get(name)
        if estimate is None:
            result.excluded[name] = "no memory estimate available"
            continue
        if budget is not None and estimate.compressed_bytes > budget:
            result.excluded[name] = (
                f"estimated KV footprint {estimate.compressed_bytes} bytes exceeds "
                f"the {budget}-byte memory budget"
            )
            continue

        # Hard calibration rule (optionally relaxed).
        if caps.requires_calibration:
            note = "requires an offline calibration pass before serving"
            if opts.prefer_no_calibration:
                result.excluded[name] = f"{note}; excluded by prefer-no-calibration"
                continue
            warnings.append(note)

        if opts.require_metal and not caps.has_metal_kernel:
            result.excluded[name] = (
                "no Metal kernel and require-metal is set (reference path only)"
            )
            continue

        # Soft warnings (never excluding).
        if not caps.has_metal_kernel:
            warnings.append(
                "no Metal kernel; uses the reference/fallback implementation path"
            )
        if caps.requires_calibration and model.dtype == "float16":
            warnings.append(
                "calibration is against fp16 activations; low-bit gains may be "
                "smaller than the benchmark table implies"
            )
        if caps.uses_eviction:
            warnings.append(_EVICTION_QUALITY_WARNING)
        if workload.context_length < opts.minimum_context_length and caps.compresses_keys:
            warnings.append(
                f"context ({workload.context_length}) is short; compression saves "
                "little and adds quantize/dequantize overhead"
            )
        if caps.supported_bits is None and workload.objective in (
            WorkloadObjective.MEMORY,
            WorkloadObjective.THROUGHPUT,
        ):
            warnings.append(
                "no supported-bit-width declared; memory gains are hard to predict"
            )

        result.viable.append(name)
        if warnings:
            result.soft_warnings[name] = warnings

    return result
