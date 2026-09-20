"""Workload description for the auto-selection pipeline.

A :class:`WorkloadProfile` captures everything about *how* the model will be
used that changes which KV-cache strategy fits (context length, generation
length, batch size/parallelism, raw compute vs. response-time goal). It is pure
data — no methods, just fields — so it can be round-tripped through JSON, the
benchmark database, and the CLI without impedance mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["WorkloadProfile", "WorkloadObjective", "workload_from_dict"]


class WorkloadObjective:
    """Identifier constants for the planner's supported objectives.

    Kept as a plain object (not an Enum) so downstream code can pass arbitrary
    objective names; the planner's ``DEFAULT_OBJECTIVE_WEIGHTS`` only assigns
    weights to the canonical ones below and falls back to ``balanced`` for
    anything else.
    """

    MEMORY = "memory"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    QUALITY = "quality"
    BALANCED = "balanced"


@dataclass
class WorkloadProfile:
    """Normalized description of a serving workload.

    Attributes:
        context_length: Maximum prompt context the served cache must hold.
        generation_length: Expected number of decode tokens per request.
        batch_size: Parallel requests sharing the KV cache.
        num_concurrent_requests: Requests served concurrently (drives how many
            caches the memory budget must cover).
        objective: One of the :class:`WorkloadObjective` names; the planner
            uses this to weight the memory/latency/throughput/quality axes.
        max_latency_ms: Hard latency ceiling, when the caller knows one
            (used to exclude strategies whose analytical proxy exceeds it).
    """

    context_length: int = 4096
    generation_length: int = 512
    batch_size: int = 1
    num_concurrent_requests: int = 1
    objective: str = WorkloadObjective.BALANCED
    max_latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_length": self.context_length,
            "generation_length": self.generation_length,
            "batch_size": self.batch_size,
            "num_concurrent_requests": self.num_concurrent_requests,
            "objective": self.objective,
            "max_latency_ms": self.max_latency_ms,
        }

    @property
    def total_tokens_per_request(self) -> int:
        """Context + generation, the cache lifetime of one request."""
        return self.context_length + self.generation_length

    @property
    def effective_batch(self) -> int:
        """Attention batch the memory estimator must budget for."""
        return self.batch_size * self.num_concurrent_requests


def workload_from_dict(data: dict[str, Any]) -> WorkloadProfile:
    """Rehydrate a :class:`WorkloadProfile` from ``to_dict`` output.

    Missing keys take their dataclass defaults; extra keys are ignored, so the
    schema can grow without breaking loading of older records.
    """
    return WorkloadProfile(
        context_length=int(data.get("context_length", 4096)),
        generation_length=int(data.get("generation_length", 512)),
        batch_size=int(data.get("batch_size", 1)),
        num_concurrent_requests=int(data.get("num_concurrent_requests", 1)),
        objective=str(data.get("objective", WorkloadObjective.BALANCED)),
        max_latency_ms=data.get("max_latency_ms"),
    )
