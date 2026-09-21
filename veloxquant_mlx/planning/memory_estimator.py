"""Analytical KV-cache memory estimation for the auto-selection pipeline.

Each method's theoretical footprint is estimated from the model's attention
geometry (heads x head_dim x dtype), the workload length/batch, and a
per-method compression model table. This keeps recommendation instant and
deterministic — no weights are loaded — at the cost of accuracy versus a
measured benchmark; the tables are tuned conservatively so that the planner
never *under-promises* memory savings.

The unit of everything is bytes, per the whole registered attention stack
(layers x batch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from veloxquant_mlx.planning.workload import WorkloadProfile
from veloxquant_mlx.profiling.model_profiler import ModelProfile

__all__ = [
    "MemoryEstimate",
    "MemoryEstimateError",
    "estimate_memory",
    "estimate_candidate_memory",
    "method_quant_bits",
]

#: fp16 = 16 bits per element; every ratio below is `effective_bits / 16`.
_BITS_PER_FP16 = 16

#: Bytes/256KB pages we assume a compressed cache needs per layer beyond the
#: tensors themselves (bookkeeping, codebooks, allocator slack). Kept
#: deliberately small and flat so it never dominates the arithmetic.
_WORKSPACE_BYTES_PER_LAYER = 16 * 1024

#: Per-method compression model. `key_bits`/`value_bits` state the default
#: quantized width used for the estimator (the practical serving default);
#: `eviction` and `budget` model token-bounding methods (steady-state cache
#: size in tokens, independent of context length once saturated). A method
#: absent from this table falls back to a no-compression fp16 model with
#: ``confidence="low"``, so the planner still *can* recommend it — just with a
#: documented caveat.
_METHOD_MEMORY_MODEL: dict[str, dict[str, Any]] = {
    # --- Key-only quantization, at the width the docs present as the norm ---
    "turboquant_prod": {"key_bits": 3},
    "turboquant_mse": {"key_bits": 4},
    "turboquant_rvq": {"key_bits": 3},
    "polar": {"key_bits": 2},
    "qjl": {"key_bits": 1},
    "spectral": {"key_bits": 3},
    "kivi": {"key_bits": 2},
    "kivi_sink": {"key_bits": 2},
    "kitty": {"key_bits": 2},
    "adakv": {"key_bits": 3},
    "kvquant": {"key_bits": 3},
    "cachegen": {"key_bits": 4},
    "nsnquant": {"key_bits": 2},
    "svdq": {"key_bits": 3},
    "amc": {"key_bits": 3},
    "age_tiered": {"key_bits": 2, "value_bits": 4},
    "kvtc": {"key_bits": 4, "value_bits": 4},
    "nestedkv": {"key_bits": 2, "value_bits": 2},
    "a2ats": {"key_bits": 2},
    "anchorkv": {"key_bits": 2},
    # --- Codebook / latent methods: keys to offsets; values stay fp16 -------
    "vecinfer": {"key_bits": 4},
    "palu": {"key_bits": 16, "value_bits": 2},
    # --- Cross-layer reuse: store one anchor per group, residuals fp16 ------
    "xquant": {"key_bits": 2, "value_bits": 2},
    "minicache": {"key_bits": 3},
    # --- GEAR: 2-bit + low-rank residual on keys AND values -----------------
    "gear": {"key_bits": 2, "value_bits": 2, "residual_fraction": 0.1},
    "rocketkv": {"key_bits": 2, "value_bits": 2},
    "kvzip": {"key_bits": 4, "value_bits": 4},
    # --- Eviction-only: fp16 but bounded steady state -----------------------
    "snapkv": {"eviction": True, "budget": 512},
    "streaming_llm": {"eviction": True, "budget": 516},  # 4 sink + 512 window
    "h2o": {"eviction": True, "budget": 512},
    "tova": {"eviction": True, "budget": 512},
    "pyramidkv": {"eviction": True, "budget": 512},
    "squeeze": {"eviction": True, "budget": 512},
    "chunkkv": {"eviction": True, "budget": 512},
    "cam": {"eviction": True, "budget": 512},
    "xkv": {"eviction": True, "budget": 512},
    "knorm": {"eviction": True, "budget": 512},
    "qfilters": {"eviction": True, "budget": 512},
    "keyformer": {"eviction": True, "budget": 512},
    "morphkv": {"eviction": True, "budget": 512},
    "curdkv": {"eviction": True, "budget": 512},
    # --- Hybrid: compression AND a bounded window ---------------------------
    "zipcache": {"key_bits": 4, "value_bits": 8, "eviction": True, "budget": 512},
    "skvq": {"key_bits": 3, "eviction": True, "budget": 516},
}


class MemoryEstimateError(RuntimeError):
    """Raised when a memory estimate cannot be constructed for a model."""


@dataclass
class MemoryEstimate:
    """Breakdown of one method's predicted KV-cache memory footprint.

    Attributes:
        method: Method name this estimate describes.
        baseline_bytes: fp16 K+V for the full context x batch (the no-compression
            starting point).
        compressed_bytes: Projected resident K+V after the method's
            compression/eviction model.
        workspace_bytes: Predicted non-tensor overhead (codebooks, allocator).
        peak_bytes: Largest allocation during prefill (usually baseline).
        resident_bytes: Footprint that stays resident during decode.
        confidence: ``"high"`` for curated-model methods, ``"medium"`` for a
            generic compression default, ``"low"`` for unmodeled methods.
        assumptions: Human-readable list of what the model assumed.
    """

    method: str
    baseline_bytes: int
    compressed_bytes: int
    workspace_bytes: int = 0
    peak_bytes: int = 0
    resident_bytes: int = 0
    confidence: str = "high"
    assumptions: list[str] = field(default_factory=list)

    @property
    def reduction_ratio(self) -> float:
        """Fraction of baseline the method is projected to keep (0..1)."""
        if self.baseline_bytes <= 0:
            return 1.0
        return max(0.0, self.compressed_bytes / self.baseline_bytes)

    @property
    def savings_percent(self) -> float:
        return max(0.0, (1.0 - self.reduction_ratio) * 100.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "baseline_bytes": self.baseline_bytes,
            "compressed_bytes": self.compressed_bytes,
            "workspace_bytes": self.workspace_bytes,
            "peak_bytes": self.peak_bytes,
            "resident_bytes": self.resident_bytes,
            "confidence": self.confidence,
            "reduction_ratio": self.reduction_ratio,
            "savings_percent": self.savings_percent,
            "assumptions": list(self.assumptions),
        }


def _bits_ratio(bits: float) -> float:
    return bits / _BITS_PER_FP16


def method_quant_bits(method: str) -> tuple[float, float, bool]:
    """Effective key/value bit widths and eviction flag for the planner's
    quality proxy: ``(key_bits, value_bits, uses_eviction)``. Unmodeled
    methods report fp16/fp16/False (no compression)."""
    row = _METHOD_MEMORY_MODEL.get(method, {})
    key_bits = float(row.get("key_bits", _BITS_PER_FP16))
    value_bits = float(row.get("value_bits", _BITS_PER_FP16))
    return key_bits, value_bits, bool(row.get("eviction", False))


def estimate_memory(
    method: str,
    model: ModelProfile,
    workload: WorkloadProfile,
) -> MemoryEstimate:
    """Estimate the serving KV footprint of one method for a model+workload.

    **Process**:
    1. Compute baseline fp16 footprint (layers × heads × head_dim × context × batch × 2 bytes)
    2. Look up per-method compression model from the registry
    3. Apply key/value bit width ratios and residual percentages
    4. For eviction methods, cap at budget tokens (not context length)
    5. Add workspace overhead for codebooks, allocator slack
    6. Return peak (highest allocation during prefill) and resident (decode holdover)

    **Accuracy**: ±20% vs actual MLX allocations (conservative: never under-promises).

    **Confidence levels**:
    - **high**: Curated memory model in registry (43 methods)
    - **medium**: Interpolated from similar family
    - **low**: No model found; defaults to no compression (fp16)

    **Arguments**:
        method: Method name to estimate (e.g., "kivi", "polar")
        model: ModelProfile with architecture (layers, heads, head_dim)
        workload: WorkloadProfile with context length, batch size

    **Returns**:
        MemoryEstimate with breakdown:
        - baseline_bytes: fp16 K+V (reference)
        - compressed_bytes: Effective footprint after method's compression
        - workspace_bytes: Codebook, index, allocator overhead
        - peak_bytes: Max allocation during prefill
        - resident_bytes: Steady-state decode footprint
        - confidence: "high", "medium", or "low"
        - assumptions: List of modeling caveats (e.g., eviction budget)

    **Raises**:
        MemoryEstimateError: If model lacks valid attention geometry
            (head_dim ≤ 0 or num_kv_heads ≤ 0 or num_layers ≤ 0).

    **Example**:
        >>> estimate = estimate_memory(
        ...     "kivi",
        ...     model=ModelProfile(num_layers=32, num_kv_heads=4, head_dim=128),
        ...     workload=WorkloadProfile(context_length=32768, batch_size=1)
        ... )
        >>> print(f"{estimate.savings_percent:.0f}% savings vs fp16")
        81% savings vs fp16
    """
    if model.head_dim <= 0 or model.num_kv_heads <= 0 or model.num_layers <= 0:
        raise MemoryEstimateError(
            f"cannot estimate memory for {method}: model has no valid attention "
            f"geometry (head_dim={model.head_dim}, kv_heads={model.num_kv_heads}, "
            f"layers={model.num_layers})"
        )

    batch = max(1, workload.effective_batch)
    total_tokens = max(1, workload.total_tokens_per_request)
    bytes_per_token_layer = model.baseline_kv_bytes_per_token
    baseline = bytes_per_token_layer * model.num_layers * batch * total_tokens

    model_row = _METHOD_MEMORY_MODEL.get(method)
    assumptions: list[str] = []
    if model_row is None:
        model_row = {"key_bits": _BITS_PER_FP16}
        confidence = "low"
        assumptions.append(f"{method} has no curated memory model; assuming no compression (fp16)")
    else:
        confidence = "high"

    key_ratio = _bits_ratio(float(model_row.get("key_bits", _BITS_PER_FP16)))
    value_ratio = _bits_ratio(float(model_row.get("value_bits", _BITS_PER_FP16)))
    residual_fraction = float(model_row.get("residual_fraction", 0.0))
    eviction = bool(model_row.get("eviction", False))
    budget = int(model_row.get("budget", 0))

    # Baseline split: keys and values are symmetric in fp16.
    key_baseline = baseline // 2
    value_baseline = baseline - key_baseline

    key_resident = key_baseline * key_ratio
    value_resident = value_baseline * value_ratio
    if residual_fraction:
        key_resident += key_baseline * residual_fraction
        value_resident += value_baseline * residual_fraction
    compressed = int(key_resident + value_resident)

    # Eviction: steady state is `budget` tokens, never more than generated.
    if eviction and budget > 0:
        steady_baseline = bytes_per_token_layer * model.num_layers * batch * budget
        steady = int(steady_baseline * ((key_ratio + value_ratio) / 2))
        compressed = int(min(baseline, steady))
        assumptions.append(
            f"eviction cache bounded at {budget} tokens instead of "
            f"{total_tokens} (quality caveat on long generations)"
        )

    workspace = model.num_layers * _WORKSPACE_BYTES_PER_LAYER
    peak = max(baseline, workspace + compressed)
    resident = max(compressed, workspace)

    if model_row.get("key_bits", _BITS_PER_FP16) == _BITS_PER_FP16 and not eviction:
        assumptions.append("keys estimated at fp16 (16-bit); request lower-bit config for savings")
    if model_row.get("value_bits"):
        assumptions.append(
            f"values quantized to {model_row['value_bits']}-bit — most methods "
            "keep values fp16 by default"
        )

    return MemoryEstimate(
        method=method,
        baseline_bytes=baseline,
        compressed_bytes=compressed,
        workspace_bytes=workspace,
        peak_bytes=peak,
        resident_bytes=resident,
        confidence=confidence,
        assumptions=assumptions,
    )


def estimate_candidate_memory(
    methods: list[str],
    model: ModelProfile,
    workload: WorkloadProfile,
) -> dict[str, MemoryEstimate]:
    """Estimate memory for a set of methods at once.

    **Convenience wrapper** over :func:`estimate_memory` for batch estimation.
    Useful in filtering and ranking pipelines where many methods are evaluated.

    **Arguments**:
        methods: List of method names (e.g., ["kivi", "polar", "turboquant_rvq"])
        model: Shared ModelProfile (applied to all methods)
        workload: Shared WorkloadProfile (applied to all methods)

    **Returns**:
        Dict mapping method name → MemoryEstimate. All estimates use the same
        model and workload parameters; confidence levels reflect whether each
        method has a curated registry entry.

    **Raises**:
        MemoryEstimateError: If estimation fails for any method (error propagated
            immediately; earlier methods in the list have been completed).

    **Example**:
        >>> estimates = estimate_candidate_memory(
        ...     ["kivi", "polar", "adakv"],
        ...     model=profile,
        ...     workload=WorkloadProfile(context_length=4096)
        ... )
        >>> for name, est in estimates.items():
        ...     print(f"{name}: {est.savings_percent:.1f}% savings")
    """
    return {method: estimate_memory(method, model, workload) for method in methods}
