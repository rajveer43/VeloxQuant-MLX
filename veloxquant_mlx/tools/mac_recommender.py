"""Mac chip + RAM method recommender (pure heuristics, no MLX required)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

ChipFamily = Literal["M1", "M2", "M3", "M4"]
ModelClass = Literal["1B", "3B", "7B", "14B", "32B"]
Goal = Literal[
    "everyday",
    "max_key_accounting",
    "max_context",
    "best_quality",
    "constant_memory",
]

ALLOWED_RAM_GB = (8, 16, 24, 32, 36, 48, 64, 128)
MODEL_WEIGHT_GB_4BIT = {
    "1B": 0.8,
    "3B": 2.0,
    "7B": 4.5,
    "14B": 8.0,
    "32B": 18.0,
}


@dataclass(frozen=True)
class RecommendRequest:
    chip: ChipFamily
    ram_gb: int
    model_class: ModelClass
    goal: Goal
    seq_len: int = 4096
    n_layers: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128


@dataclass(frozen=True)
class RecommendResult:
    method: str
    knobs: dict[str, Any]
    key_accounting_ratio: float
    resident_savings_likely: bool
    kv_fp16_mb: float
    kv_compressed_mb_estimate: float
    warnings: list[str]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_kv_fp16_mb(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
) -> float:
    """Full K+V fp16 cache size in megabytes."""
    bytes_ = 2 * n_layers * n_kv_heads * head_dim * seq_len * 2
    return bytes_ / (1024**2)


def recommend(req: RecommendRequest) -> RecommendResult:
    """Return a transparent method recommendation for Apple Silicon."""
    if req.ram_gb not in ALLOWED_RAM_GB:
        raise ValueError(f"ram_gb must be one of {ALLOWED_RAM_GB}, got {req.ram_gb}")
    if req.seq_len < 1:
        raise ValueError("seq_len must be >= 1")

    warnings: list[str] = []
    weight_gb = MODEL_WEIGHT_GB_4BIT[req.model_class]
    # Leave ~4 GB for OS + apps; activations need headroom too
    headroom_gb = req.ram_gb - weight_gb - 4.0
    if headroom_gb < 3.0:
        warnings.append(
            f"{req.model_class} 4-bit weights (~{weight_gb} GB) leave little "
            f"headroom on {req.ram_gb} GB (est. headroom {headroom_gb:.1f} GB). "
            "Prefer a smaller model, eviction, or full-KV compression."
        )

    kv_fp16 = estimate_kv_fp16_mb(req.n_layers, req.n_kv_heads, req.head_dim, req.seq_len)

    # Tiny-model Metal overhead warning (chip generation does not remove this)
    if req.model_class == "1B":
        warnings.append(
            "Metal kernel launch overhead can dominate on tiny models; "
            "prefer RVQ or disable Metal if tok/s drops."
        )

    tight = req.ram_gb <= 16 or headroom_gb < 3.0
    if req.goal == "everyday":
        method = "turboquant_rvq"
        knobs = {"bit_width_inlier": 1, "seed": 42}
        ratio = 7.5
        resident = False
        rationale = (
            "Zero-calibration default. Key accounting ~7.5x at head_dim=128. "
            "Default path dequantizes into parent fp16 cache."
        )
        if tight and req.model_class in ("7B", "14B", "32B"):
            warnings.append(
                "Tight RAM with a mid/large model: consider goal=max_context "
                "(rabitq) or goal=constant_memory (eviction) for long prompts."
            )

    elif req.goal == "max_key_accounting":
        method = "vecinfer"
        knobs = {
            "key_codebook_bits": 8,
            "value_codebook_bits": 8,
            "key_sub_dim": 8,
            "value_sub_dim": 8,
            "use_metal_kernels": None,
            "note": "Requires one-time codebook calibration",
        }
        ratio = 16.0
        resident = False
        rationale = (
            "Product VQ 1-bit path targets ~16x key accounting when "
            "head_dim is divisible by sub_dim=8. Needs calibration."
        )
        if req.head_dim % 8 != 0:
            warnings.append(
                f"head_dim={req.head_dim} is not divisible by 8; "
                "VecInfer sub_dim must divide head_dim."
            )

    elif req.goal == "best_quality":
        method = "spectral"
        knobs = {"bit_width_inlier": 3, "note": "Requires spectral rotation calibration"}
        ratio = 5.3
        resident = False
        rationale = (
            "SpectralQuant targets better reconstruction at moderate "
            "compression via eigenbasis rotation (calibration required)."
        )

    elif req.goal == "max_context":
        if tight:
            method = "rabitq"
            knobs = {"note": "1-bit keys + MSE-b4 values; prefer fused Metal path when available"}
            ratio = 6.0
            resident = True
            rationale = (
                "Full-KV compression is more likely to free resident memory "
                "than key-only accounting methods on tight RAM."
            )
        else:
            method = "rabitq"
            knobs = {"note": "Full KV compression for longer context in fixed RAM"}
            ratio = 6.0
            resident = True
            rationale = (
                "RaBitQ compresses keys and values. Better candidate for "
                "real context capacity gains than key-only RVQ accounting."
            )

    elif req.goal == "constant_memory":
        method = "streaming_llm"
        knobs = {"stream_n_sink": 4, "stream_window_size": 512}
        ratio = 1.0
        resident = True
        rationale = (
            "Structural eviction keeps a fixed sink + window. Cache token "
            "count stays bounded regardless of generation length."
        )
        warnings.append(
            "Eviction drops tokens; quality depends on the task. "
            "For importance-based eviction try method=h2o instead."
        )

    else:
        raise ValueError(f"Unknown goal: {req.goal}")

    # Chip note: bandwidth/generation matters less than RAM for method pick
    if req.chip in ("M1", "M2") and req.model_class in ("14B", "32B"):
        warnings.append(
            f"{req.chip} with {req.model_class}: expect lower tok/s; "
            "memory fit still depends mainly on unified RAM."
        )

    compressed_mb = kv_fp16 / ratio if ratio > 0 else kv_fp16
    # Resident estimate is only meaningful when resident_savings_likely
    if not resident:
        warnings.append(
            "Resident RSS savings are unlikely at short context for this "
            "method's default path (accounting ratio still valid)."
        )

    return RecommendResult(
        method=method,
        knobs=knobs,
        key_accounting_ratio=ratio,
        resident_savings_likely=resident,
        kv_fp16_mb=round(kv_fp16, 2),
        kv_compressed_mb_estimate=round(compressed_mb, 2),
        warnings=warnings,
        rationale=rationale,
    )


def ruleset_dict() -> dict[str, Any]:
    """Export static metadata for docs / JS widgets."""
    return {
        "version": 1,
        "allowed_ram_gb": list(ALLOWED_RAM_GB),
        "model_classes": list(MODEL_WEIGHT_GB_4BIT.keys()),
        "goals": [
            "everyday",
            "max_key_accounting",
            "max_context",
            "best_quality",
            "constant_memory",
        ],
        "chips": ["M1", "M2", "M3", "M4"],
        "weight_gb_4bit_estimate": dict(MODEL_WEIGHT_GB_4BIT),
        "defaults": {
            "everyday": {"method": "turboquant_rvq", "bit_width_inlier": 1, "ratio": 7.5},
            "max_key_accounting": {"method": "vecinfer", "ratio": 16.0},
            "best_quality": {"method": "spectral", "ratio": 5.3},
            "max_context": {"method": "rabitq", "ratio": 6.0, "resident_likely": True},
            "constant_memory": {
                "method": "streaming_llm",
                "ratio": 1.0,
                "resident_likely": True,
            },
        },
    }
