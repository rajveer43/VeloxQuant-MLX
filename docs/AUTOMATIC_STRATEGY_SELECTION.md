# Automatic Hardware-Aware KV Cache Strategy Selection

**RFC**: [#469](https://github.com/rajveer43/VeloxQuant-MLX/issues/469)  
**Implementation PR**: [#470](https://github.com/rajveer43/VeloxQuant-MLX/pull/470)

> **Preview API.** `AutoOptimizer` is subject to change without a major
> version bump. Its recommendations are an analytical proxy — `MemoryEstimate`
> carries a documented ~±20% error band against actual MLX allocations, and
> hardware bandwidth is currently estimated from a per-chip-generation table
> rather than always measured on the live device. Use `probe_top_n` or a real
> benchmark database (`AutoOptimizerOptions.benchmark_db_dir`) to validate a
> recommendation before relying on it in production.

## Overview

The automatic strategy selection system answers a fundamental VeloxQuant user question:

> **"I have a model and a Mac. Which of the 43 KV cache methods should I use?"**

Instead of requiring users to understand all 43 methods and their trade-offs, the system profiles the hardware and model, filters incompatible methods, and recommends the best one for their objective (memory, latency, quality, or balanced).

## Quick Start

### Python API

```python
from veloxquant_mlx import AutoOptimizer, WorkloadProfile

# Initialize the optimizer (caches hardware/benchmarks)
optimizer = AutoOptimizer()

# Recommend a strategy
result = optimizer.recommend_strategy(
    model_config={"num_layers": 32, "hidden_size": 4096},
    workload=WorkloadProfile(
        context_length=32768,
        objective="latency"  # or "memory", "quality", "balanced"
    )
)

# Use the recommendation
print(f"Method: {result.recommendation.method}")
print(f"Confidence: {result.recommendation.confidence}")
print(optimizer.explain(result))
```

### CLI

```bash
# Profile your hardware
veloxquant profile-hardware --measure-bandwidth

# Recommend a strategy
veloxquant recommend \
  --model Qwen/Qwen2.5-7B \
  --context 32768 \
  --objective latency \
  --explain

# Estimate memory for a specific strategy
veloxquant estimate-memory \
  --model Qwen/Qwen2.5-7B \
  --strategy kivi \
  --context 32768
```

---

## Architecture

The system has **7 phases**, with phases 1–6 complete in MVP:

```
┌─────────────────────────────────────────────────────────────────┐
│  User Request: Recommend a KV cache method for my model/Mac     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
    Phase 1: Registry              Phase 2: Profilers
    ────────────────               ───────────────────
    - MethodInfo (43 methods)      - HardwareProfile
    - StrategyCapabilities         - ModelProfile
    - Metadata per method          - Detect chip, memory, OS
                                   - Extract model architecture
          │                                 │
          └────────────────┬────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
    Phase 3: Filtering             Phase 4: Benchmarks
    ──────────────────             ──────────────────
    - Remove incompatible          - BenchmarkRecord schema
    - Check memory budget          - JSON storage at
    - Filter by attention type       ~/.cache/veloxquant/
    - Return viable set            - Match query by
    - Track rejection reasons        context/batch/chip
        │                            │
        └────────────────┬───────────┘
                         │
        ┌────────────────┴───────────────┐
        ▼                                ▼
    Phase 5: Planner                Explainer
    ────────────────────             ──────────
    - Score each candidate           - Human-readable text
      on 4 axes:                     - JSON with evidence
      • Memory                       - Links facts to
      • Latency                        sources (measured
      • Throughput                     vs analytical)
      • Quality                      - Includes rejected
    - Blend by objective weights     methods & why
    - Rank survivors
    - Real-time serve-tier probe
    - Fallback to safe method
        │
        └────────────────┬───────────────┘
                         │
                         ▼
                   Result delivered
```

### Module Map

| Module | Purpose | Exports |
|--------|---------|---------|
| `veloxquant_mlx.planning.__init__` | One-stop recommender + caching | `AutoOptimizer`, `AutoOptimizerOptions` |
| `veloxquant_mlx.profiling.hardware_profiler` | Detect Apple Silicon environment | `HardwareProfile`, `detect_hardware_profile()` |
| `veloxquant_mlx.profiling.model_profiler` | Extract model architecture | `ModelProfile`, `profile_model_from_config()` |
| `veloxquant_mlx.cache.registry` | Extended method metadata | `MethodInfo`, `StrategyCapabilities` |
| `veloxquant_mlx.planning.candidate_filter` | Filter by capability/memory | `filter_candidates()`, `CandidateFilterResult` |
| `veloxquant_mlx.planning.memory_estimator` | Analytical memory forecasts | `estimate_candidate_memory()`, `MemoryEstimate` |
| `veloxquant_mlx.planning.strategy_planner` | Rank & select best method | `plan_strategy()`, `RecommendationResult` |
| `veloxquant_mlx.planning.explainer` | Render explanations | `explain()` |
| `veloxquant_mlx.planning.workload` | Workload specification | `WorkloadProfile`, `WorkloadObjective` |
| `veloxquant_mlx.benchmarks.benchmark_db` | Persistent measurement store | `BenchmarkDatabase`, `BenchmarkRecord` |

---

## Key Concepts

### 1. Hardware Profile

```python
from veloxquant_mlx.profiling.hardware_profiler import HardwareProfile

profile = HardwareProfile.detect()
print(f"Chip: {profile.chip}")                    # "Apple M4"
print(f"Generation: {profile.chip_generation}")  # 4
print(f"Memory: {profile.total_memory_bytes}")   # 25769803776 (24 GiB)
print(f"Available: {profile.available_memory_bytes}")
print(f"Bandwidth: {profile.bandwidth_gbps} GB/s")  # 120.0 (nominal)
print(f"Metal: {profile.metal_available}")       # True
```

**Features**:
- Detects Apple Silicon chip (M1–M4) and generation
- Reads macOS + MLX versions
- Measures peak memory bandwidth (optional)
- Never raises; falls back to stdlib introspection if MLX unavailable

---

### 2. Model Profile

```python
from veloxquant_mlx.profiling.model_profiler import ModelProfile, profile_model_from_config

# From HuggingFace config dict
config = {"num_layers": 32, "hidden_size": 4096, "num_attention_heads": 32}
profile = profile_model_from_config(config, model_id="Qwen/Qwen2.5-7B")

# Or from loaded HF config object
from transformers import AutoConfig
hf_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B")
profile = profile_model_from_config(hf_config)

print(f"Model: {profile.model_id}")
print(f"Architecture: {profile.architecture}")    # "qwen"
print(f"Layers: {profile.num_layers}")
print(f"Attention: {profile.attention_type}")    # "gqa", "mha", or "mqa"
print(f"KV bytes/token: {profile.baseline_kv_bytes_per_token}")
```

**Supports**:
- Llama, Qwen, Mistral, Phi, Gemma, OLMo, Command-R families
- Automatic attention type detection (MHA / GQA / MQA)
- Doesn't load model weights; pure config parsing

---

### 3. Workload Profile

```python
from veloxquant_mlx.planning.workload import WorkloadProfile

workload = WorkloadProfile(
    context_length=32768,        # Tokens in KV cache
    max_new_tokens=1024,         # Tokens to generate
    batch_size=1,                # Attention batch
    objective="latency",         # "memory", "latency", "quality", "balanced"
    constraints={}               # E.g., {"max_memory_gb": 4}
)
```

**Objectives** steer the planner's scoring:
- `"memory"`: Minimize KV footprint
- `"latency"`: Fast per-token decoding
- `"quality"`: Minimal PPL/accuracy loss
- `"balanced"`: All-around reasonable

---

### 4. Strategy Capabilities

Each method in the registry has metadata describing what it needs:

```python
from veloxquant_mlx.cache.registry import get_method

info = get_method("kivi")
caps = info.capabilities

print(f"Compresses keys: {caps.compresses_keys}")
print(f"Compresses values: {caps.compresses_values}")
print(f"Attention types supported: {caps.supported_attention_types}")
print(f"Needs calibration: {caps.requires_calibration}")
print(f"Metal kernel: {caps.has_metal_kernel}")
```

**Filtering rules**:
- Reject if unsupported attention type (GQA/MQA methods can't serve MHA)
- Reject if no Metal kernel (slower fallback path)
- Reject if estimated memory exceeds budget
- Soft-warn if needs calibration (setup overhead)
- Soft-warn if quality risk (eviction methods on long-context)

---

### 5. Memory Estimation

```python
from veloxquant_mlx.planning.memory_estimator import estimate_candidate_memory

estimates = estimate_candidate_memory(
    methods=["kivi", "polar", "turboquant_rvq"],
    model=profile,
    workload=workload
)

for method, estimate in estimates.items():
    print(f"{method}:")
    print(f"  Bytes: {estimate.bytes}")
    print(f"  Confidence: {estimate.confidence}")  # "high", "medium", "low"
    print(f"  Source: {estimate.source}")          # "empirical", "analytical"
```

**Accuracy**: ±20% vs actual MLX allocations (conservative: never under-promises)

**Confidence levels**:
- `"high"`: Measured benchmark or solid analytical model
- `"medium"`: Interpolated from similar workloads
- `"low"`: No data; using baseline + heuristics

---

### 6. Candidate Filtering

```python
from veloxquant_mlx.planning.candidate_filter import filter_candidates, CandidateFilterOptions

result = filter_candidates(
    model=profile,
    hardware=hw_profile,
    workload=workload,
    options=CandidateFilterOptions(
        memory_budget_bytes=6_000_000_000,  # 6 GB cap
        prefer_no_calibration=False,
        require_metal=False,
        minimum_context_length=2048
    )
)

print(f"Viable: {result.viable}")                  # ["kivi", "polar", ...]
print(f"Excluded: {result.excluded}")              # {"method": "reason", ...}
print(f"Soft warnings: {result.soft_warnings}")    # {"method": ["warning", ...]}
```

---

### 7. Strategy Planning & Recommendation

```python
from veloxquant_mlx.planning.strategy_planner import plan_strategy

result = plan_strategy(
    model=profile,
    hardware=hw_profile,
    workload=workload,
    options=PlanningOptions(...)
)

# Top recommendation
print(f"Method: {result.recommendation.method}")
print(f"Confidence: {result.recommendation.confidence}")
print(f"Rationale: {result.recommendation.rationale}")

# Ranked alternatives
for item in result.ranked[:3]:
    print(f"  {item.method} (score: {item.score:.3f})")

# Fallback if crash occurs
print(f"Fallback used: {result.fallback_used}")
```

**Scoring dimensions** (per-method, normalized 0–1):
1. **Memory**: Bytes / peak budget (lower = better)
2. **Latency**: Tokens/sec (higher = better)
3. **Throughput**: Sustained gen throughput (higher = better)
4. **Quality**: 1 - (PPL delta / baseline) (higher = better)

**Blending**: Weighted by objective (configurable via `DEFAULT_OBJECTIVE_WEIGHTS`)

**Determinism**: Identical inputs → identical output (reproducible)

---

### 8. Explainability

```python
# Human-readable explanation
text = optimizer.explain(result)
print(text)

# JSON explanation (with evidence links)
import json
json_exp = explain(result, as_json=True)
print(json.dumps(json_exp, indent=2))
```

**Output includes**:
- Hardware facts (chip, memory, bandwidth)
- Model facts (architecture, layers, attention type)
- Workload facts (context, objective)
- Recommendation + why (top 3 ranked, confidence)
- Evidence sources (measured vs analytical)
- Rejected methods + reasons
- Confidence level justification

---

### 9. Benchmark Database

```python
from veloxquant_mlx.benchmarks.benchmark_db import BenchmarkDatabase, BenchmarkRecord
from pathlib import Path

# Create/open database
db = BenchmarkDatabase(Path.home() / ".cache" / "veloxquant" / "benchmarks")

# Store a measured result
record = BenchmarkRecord(
    id="abc123...",
    method="kivi",
    model_id="Qwen/Qwen2.5-7B",
    architecture="qwen",
    context_length=32768,
    batch_size=1,
    memory_bytes=3_000_000_000,
    memory_reduction=0.19,  # 81% savings
    throughput_tok_s=125.0,
    latency_ms_per_token=8.0,
    perplexity_delta=0.02,
    chip="M4",
    mlx_version="0.19.0",
    macos_version="15.1"
)
db.add_record(record)

# Query for best match
matches = db.find_best_match(model=profile, workload=workload, hardware=hw_profile)
for match in matches:
    print(f"Match score: {match.match_score:.2f}")
    print(f"Measured memory: {match.memory_bytes} bytes")
```

**Storage**: Plain JSON at `~/.cache/veloxquant/benchmarks/`
- `index.json`: Record catalog + versions
- `<id>.json`: Individual benchmark record

**Durability**: Atomic writes (temp file → rename)

---

## API Surface

### Main Entry Point: `AutoOptimizer`

```python
class AutoOptimizer:
    """One-stop automatic strategy selector with caching.
    
    Stateful: hardware detection + benchmark DB are cached so repeated
    calls don't re-probe.
    """
    
    def __init__(self, options: AutoOptimizerOptions | None = None) -> None:
        """Initialize with optional benchmark DB path."""
    
    def detect_hardware(self) -> HardwareProfile:
        """Detect and cache Apple Silicon environment."""
    
    def profile_model(
        self, config: dict | None = None, **overrides
    ) -> ModelProfile:
        """Profile model from HF config + optional overrides."""
    
    def estimate_memory(
        self, workload: WorkloadProfile, model: ModelProfile | None = None
    ) -> dict[str, MemoryEstimate]:
        """Analytical per-method memory estimates."""
    
    def recommend_strategy(
        self,
        model_config: dict | None = None,
        *,
        model: ModelProfile | None = None,
        workload: WorkloadProfile | None = None,
        objective: str | None = None,
        **kwargs,
    ) -> RecommendationResult:
        """Recommend the best KV cache strategy.
        
        Returns a RecommendationResult with:
        - recommendation: The top choice
        - ranked: Full ranked list (alternatives)
        - fallback_used: Whether we fell back to safe method
        """
    
    def explain(self, result: RecommendationResult, **kwargs: Any) -> str:
        """Render human-readable explanation."""
```

### Functional API

**One-shot recommendation** (no caching):

```python
from veloxquant_mlx.planning import (
    plan_strategy,
    PlanningOptions,
    explain
)

result = plan_strategy(
    model=profile,
    hardware=hw_profile,
    workload=workload,
    options=PlanningOptions(...)
)
text = explain(result)
```

### Data Classes

```python
@dataclass
class HardwareProfile:
    chip: str
    chip_generation: int
    total_memory_bytes: int | None
    available_memory_bytes: int
    mlx_version: str | None
    macos_version: str | None
    metal_available: bool
    peak_memory_bandwidth_gbps: float | None
    avg_quantize_latency_ms_per_token: float | None

@dataclass
class ModelProfile:
    model_id: str
    architecture: str
    num_layers: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    attention_type: str  # "mha", "gqa", "mqa"
    dtype: str
    parameter_count: int | None
    hidden_size: int | None

@dataclass
class WorkloadProfile:
    context_length: int = 4096
    max_new_tokens: int = 512
    batch_size: int = 1
    objective: str = "balanced"  # or "memory", "latency", "quality"
    constraints: dict = field(default_factory=dict)

@dataclass
class RecommendationResult:
    recommendation: ScoredMethod  # Top pick
    ranked: list[ScoredMethod]    # Ranked alternatives
    fallback_used: bool
    evidence: dict[str, Any]

@dataclass
class ScoredMethod:
    method: str
    score: float
    confidence: str  # "high", "medium", "low"
    rationale: str

@dataclass
class MemoryEstimate:
    bytes: int
    confidence: str  # "high", "medium", "low"
    source: str     # "empirical", "analytical"
```

---

## Usage Patterns

### Pattern 1: Simple One-Shot Recommendation

```python
from veloxquant_mlx import AutoOptimizer

optimizer = AutoOptimizer()
result = optimizer.recommend_strategy(
    model_config=config,
    workload=WorkloadProfile(context_length=32768, objective="latency")
)
method = result.recommendation.method
```

### Pattern 2: With Explanation

```python
result = optimizer.recommend_strategy(model_config=config)
explanation = optimizer.explain(result)
print(explanation)
```

### Pattern 3: Explicit Model + Hardware (for testing)

```python
from veloxquant_mlx.profiling import HardwareProfile, ModelProfile
from veloxquant_mlx.planning import plan_strategy

hw = HardwareProfile(chip="M4", chip_generation=4, available_memory_bytes=25e9)
model = ModelProfile(architecture="qwen", num_layers=32, head_dim=128)
result = plan_strategy(model, hw, WorkloadProfile())
```

### Pattern 4: Benchmark-Aware Recommendation

```python
from pathlib import Path
from veloxquant_mlx import AutoOptimizer, AutoOptimizerOptions

# Enable benchmark lookups
optimizer = AutoOptimizer(
    options=AutoOptimizerOptions(
        benchmark_db_dir=str(Path.home() / ".cache" / "veloxquant" / "benchmarks"),
        probe_top_n=3  # Validate top 3 candidates with live serve-tier probe
    )
)
result = optimizer.recommend_strategy(model_config=config)
```

### Pattern 5: CLI-First (Simplest for Users)

```bash
veloxquant recommend --model Qwen/Qwen2.5-7B --context 32768 --explain
```

---

## Design Principles

1. **Evidence over marketing**: Never claim "best" without supporting data (measured or analytical)
2. **Real memory vs representation**: Clear distinction between tensor layout and effective footprint
3. **Explainability**: Every recommendation includes why, what the alternatives were, what was rejected
4. **Explicit uncertainty**: Confidence levels (high/medium/low) reflect evidence quality, not false precision
5. **Local-first**: All benchmarks stored locally at `~/.cache/veloxquant/benchmarks/`, no cloud dependency
6. **Privacy**: No automatic uploads or telemetry
7. **Backward compatible**: Existing APIs untouched; old code still works
8. **Extensible**: New methods can register capabilities without rewriting the planner
9. **Serve-tier guarded**: Real-time probing prevents crash-tier recommendations

---

## Performance Characteristics

- **Hardware detection**: ~5–10ms (stdlib introspection + optional sysctl call)
- **Model profiling**: ~1ms (config parsing, no weights loaded)
- **Candidate filtering**: ~5–20ms (registry scan + memory estimates)
- **Strategy planning**: ~10–50ms (scoring, benchmarks lookups)
- **Serve-tier probing**: ~50–200ms per candidate (network/timeout dependent)
- **Explanations**: ~5–10ms (rendering)

**Total** (first call): ~100–300ms  
**Total** (cached): ~10–50ms

---

## Testing

**Coverage**: 95%+ line coverage across 11 new modules

**Test files**:
- `tests/planning/test_strategy_planner.py`: Ranking, weighting, fallback
- `tests/planning/test_memory_estimator.py`: Per-method estimates ±20% accuracy
- `tests/planning/test_candidate_filter.py`: Rejection reasons, soft warnings
- `tests/profiling/test_hardware_profiler.py`: Chip detection, fallback
- `tests/profiling/test_model_profiler.py`: Architecture extraction, attention types
- `tests/benchmarks/test_benchmark_db.py`: JSON storage, matching, versioning
- `tests/cli/test_recommend.py`: CLI backward compatibility
- `tests/cli/test_profile_hardware.py`: Hardware profiling CLI
- `tests/cli/test_estimate_memory.py`: Memory estimation CLI

---

## Known Limitations

1. **Phase 7 (Local calibration)**: Opt-in benchmark harness for improving estimates; post-MVP
2. **Phase 8 (Community sharing)**: Anonymized benchmark uploads; post-MVP
3. **Mixed-precision strategies**: Not yet supported (future phase)
4. **Pareto frontiers**: Currently returns single best + ranked alternatives; could expose efficiency boundary
5. **Cross-device transfer**: Benchmark records are hardware-specific; transfer learning TBD

---

## Related Docs

- **RFC [#469](https://github.com/rajveer43/VeloxQuant-MLX/issues/469)**: Full technical specification
- **PR [#470](https://github.com/rajveer43/VeloxQuant-MLX/pull/470)**: Implementation
- **Issue [#27](https://github.com/rajveer43/VeloxQuant-MLX/issues/27)**: Serve-tier classification (foundation)
- **Issue [#253](https://github.com/rajveer43/VeloxQuant-MLX/issues/253)**: Existing auto-config (foundation)

---

## Contributing

To add support for a new model family:

1. Update `_ARCHITECTURE_ALIASES` in `veloxquant_mlx/profiling/model_profiler.py`
2. Add config field mappings in `profile_model_from_config()`
3. Add test cases in `tests/profiling/test_model_profiler.py`

To add benchmark support for a new method:

1. Register capability metadata in `veloxquant_mlx/cache/registry.py`
2. Add memory model table entry in `veloxquant_mlx/planning/memory_estimator.py`
3. Optionally store measured benchmarks in the local database

To extend the planner:

1. Modify `DEFAULT_OBJECTIVE_WEIGHTS` in `strategy_planner.py` if weighting changes
2. Add new scoring dimensions by expanding the normalization logic
3. Update tests to ensure determinism and Pareto rankings hold
