"""Mac chip + RAM method recommender (pure heuristics, no MLX required)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

ChipFamily = Literal["M1", "M2", "M3", "M4"]
ModelClass = Literal["1B", "3B", "7B", "14B", "32B", "70B", "120B", "235B", "671B"]
Goal = Literal[
    "everyday",
    "max_key_accounting",
    "max_context",
    "best_quality",
    "constant_memory",
]

# Mac Studio (M2/M3 Ultra) tops out at 192 GB; some configs go to 512 GB.
ALLOWED_RAM_GB = (8, 16, 24, 32, 36, 48, 64, 96, 128, 192, 512)
# Weight-only estimate at 4-bit quantization (Parameters x 4 bits / 8, plus
# a small tokenizer/overhead margin). 70B+ classes reflect what mlx-community
# actually ships (Llama 3.1 70B, Mixtral-8x22B/120B-class, Qwen 235B MoE,
# DeepSeek V3/R1 671B) rather than being purely theoretical.
MODEL_WEIGHT_GB_4BIT = {
    "1B": 0.8,
    "3B": 2.0,
    "7B": 4.5,
    "14B": 8.0,
    "32B": 18.0,
    "70B": 40.0,
    "120B": 68.0,
    "235B": 132.0,
    "671B": 380.0,
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
        # Negative headroom means the weights alone overrun the machine, so say
        # "will not fit" rather than "barely fits" — the softer phrasing would
        # read as a caution about a setup that is in fact impossible.
        if headroom_gb < 0:
            warnings.append(
                f"A {req.model_class} model will not fit in {req.ram_gb} GB. Its "
                f"weights alone need ~{weight_gb} GB, which is more than this Mac "
                f"has once macOS takes its share — you are about "
                f"{abs(headroom_gb):.1f} GB short of any headroom. Pick a smaller "
                "model."
            )
        else:
            warnings.append(
                f"A {req.model_class} model barely fits in {req.ram_gb} GB. Its "
                f"weights alone take ~{weight_gb} GB, leaving about "
                f"{headroom_gb:.1f} GB of headroom for everything else. Try a "
                "smaller model, or pick a goal that caps memory."
            )

    kv_fp16 = estimate_kv_fp16_mb(req.n_layers, req.n_kv_heads, req.head_dim, req.seq_len)

    # Tiny-model Metal overhead warning (chip generation does not remove this)
    if req.model_class == "1B":
        warnings.append(
            "On a model this small the GPU spends more time starting work than "
            "doing it, so compression can actually slow generation down. If speed "
            "drops, switch to turboquant_rvq or turn the Metal kernels off."
        )

    tight = req.ram_gb <= 16 or headroom_gb < 3.0
    if req.goal == "everyday":
        method = "turboquant_rvq"
        knobs = {"bit_width_inlier": 1, "seed": 42}
        ratio = 7.5
        resident = False
        rationale = (
            "The safe everyday pick: it works out of the box with no setup step, "
            "and shrinks the key half of the cache by about 7.5x. It unpacks each "
            "value back to full precision as it is read, so this is a size "
            "measurement rather than a drop in live memory use."
        )
        if tight and req.model_class in ("7B", "14B", "32B", "70B", "120B", "235B", "671B"):
            warnings.append(
                "RAM is tight for a model this size. For long prompts you will get "
                "more out of 'Fit the longest conversation' (rabitq), which "
                "compresses the whole cache, or 'Never grow past a fixed memory "
                "limit' (streaming_llm), which caps it outright."
            )

    elif req.goal == "max_key_accounting":
        method = "vecinfer"
        knobs = {
            "key_codebook_bits": 8,
            "value_codebook_bits": 8,
            "key_sub_dim": 8,
            "value_sub_dim": 8,
            "use_metal_kernels": None,
            "note": "Needs a one-time setup pass over sample data before first use",
        }
        ratio = 16.0
        resident = False
        rationale = (
            "The smallest key cache on offer, around 16x. The trade is a one-time "
            "setup pass over sample data before you can use it, and the same "
            "unpack-on-read caveat as the everyday option."
        )
        if req.head_dim % 8 != 0:
            warnings.append(
                f"This model will not work with vecinfer: it splits each vector "
                f"into groups of 8, and this model's head dimension "
                f"({req.head_dim}) does not divide evenly by 8."
            )

    elif req.goal == "best_quality":
        method = "spectral"
        knobs = {
            "bit_width_inlier": 3,
            "note": "Needs a one-time setup pass over sample data before first use",
        }
        ratio = 5.3
        resident = False
        rationale = (
            "Compresses less (about 5.3x) but reconstructs the cache more "
            "faithfully, so answers stay closest to the uncompressed model. "
            "Needs a one-time setup pass over sample data."
        )

    elif req.goal == "max_context":
        if tight:
            method = "rabitq"
            knobs = {"note": "Compresses the whole cache; turn on the Metal kernels if available"}
            ratio = 6.0
            resident = True
            rationale = (
                "Compresses both halves of the cache, so it genuinely gives RAM "
                "back rather than only measuring smaller. That matters most on a "
                "machine as tight as this one."
            )
        else:
            method = "rabitq"
            knobs = {"note": "Compresses the whole cache, for longer chats in the same memory"}
            ratio = 6.0
            resident = True
            rationale = (
                "Compresses both halves of the cache, not just the keys, so the "
                "space it frees is real. That makes it the better choice when you "
                "want the longest possible conversation in the RAM you have."
            )

    elif req.goal == "constant_memory":
        method = "streaming_llm"
        knobs = {"stream_n_sink": 4, "stream_window_size": 512}
        ratio = 1.0
        resident = True
        rationale = (
            "Keeps the first few tokens plus a sliding window of recent ones and "
            "discards the rest, so memory use stops growing no matter how long "
            "the conversation runs."
        )
        warnings.append(
            "This works by forgetting older tokens, so the model can lose track of "
            "things said early in a long conversation. How much that hurts depends "
            "on the task. To drop the least-used tokens instead of simply the "
            "oldest, try method=h2o."
        )

    else:
        raise ValueError(f"Unknown goal: {req.goal}")

    # Chip note: bandwidth/generation matters less than RAM for method pick
    if req.chip in ("M1", "M2") and req.model_class in ("14B", "32B", "70B", "120B", "235B", "671B"):
        warnings.append(
            f"A {req.model_class} model on an {req.chip} will generate text more "
            "slowly than on a newer chip. Whether it fits at all, though, comes "
            "down to how much RAM you have rather than which chip it is."
        )

    # Extreme-scale MoE models (120B+) need Ultra-class unified memory; call
    # this out explicitly since the "will not fit" warning above only fires
    # when headroom is negative, not when the fit is technically possible but
    # leaves the machine unusable for anything else.
    if req.model_class in ("120B", "235B", "671B") and req.ram_gb < 192:
        warnings.append(
            f"A {req.model_class} model is only practical on Mac Studio (M2/M3 "
            "Ultra) configurations with 192 GB or more of unified memory. Smaller "
            "machines that technically fit the weights will have little to no "
            "headroom for the KV cache or macOS itself."
        )

    compressed_mb = kv_fp16 / ratio if ratio > 0 else kv_fp16
    # Resident estimate is only meaningful when resident_savings_likely
    if not resident:
        warnings.append(
            "This method measures smaller but may not free much actual RAM on "
            "short prompts, because its default path unpacks values back to full "
            "precision as it reads them. The size figure is real; treat it as a "
            "measure of how well the data compresses, not as RAM you get back."
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
