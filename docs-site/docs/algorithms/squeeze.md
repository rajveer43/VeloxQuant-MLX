# SqueezeAttention — 2D Layer×Token Data-Driven Budget Eviction

**Method id:** `squeeze` · **New in 0.24.0** · *Inspired by* [SqueezeAttention (arXiv:2404.04793)](https://arxiv.org/abs/2404.04793)
(Wang et al., 2024) — **SqueezeAttention-adapted (VeloxQuant-MLX implementation)**,
not a faithful port.

SqueezeAttention-adapted is the library's **sixth eviction configuration**, the
first **2D (layer × token)** budget method, and the first with a **data-driven**
per-layer budget. It is H2O-adapted's cumulative-attention-mass eviction with a
per-layer budget that is *measured* rather than assumed: each layer reports its
attention **concentration** during prefill, and a fixed total budget is
reallocated toward broad (low-concentration) layers and away from concentrated
ones. When `squeeze_strength = 0.0` it reduces exactly to uniform H2O-adapted.

## Why measure instead of assume

[PyramidKV](./pyramidkv.md) also gives early layers more budget than deep layers,
but it does so with a *fixed* positional taper — calibration-free, yet blind to
the actual prompt. SqueezeAttention reads the **geometry of each layer's key set**
and puts budget where *this* prompt's attention actually spreads. It is the
data-driven sibling of PyramidKV: same H2O scorer, same per-layer-budget idea, but
the schedule is computed from observed concentration rather than assumed from
depth.

| Eviction axis | When it fires | Score signal | Budget |
|---|---|---|---|
| SnapKV-adapted | Once at prefill end | Key-as-query attention proxy | Uniform |
| StreamingLLM-adapted | Every token | Position (recency + sink) | Uniform |
| H2O-adapted | Every token (over budget) | Cumulative attention mass | Uniform |
| TOVA-adapted | Every token (over budget) | Current-step attention weight | Uniform |
| PyramidKV-adapted | Every token (over budget) | Cumulative attention mass | Per-layer **fixed** pyramid |
| **SqueezeAttention-adapted** | Every token (over budget) | Cumulative attention mass | **Per-layer data-driven** |

## The concentration proxy

At the cache-wrapper level the true attention distribution is not visible, so
SqueezeAttention estimates each layer's concentration from the **geometry** of its
key set — the mean pairwise cosine similarity of the (direction-normalised) keys:

```
concentration = mean off-diagonal cosine( K_norm, K_norm )
```

- **High** concentration (keys cluster in direction) → a query would attend to a
  few similar tokens → the layer needs a **smaller** budget.
- **Low** concentration (keys spread in direction) → broad attention → the layer
  needs a **larger** budget.

Identical keys score `1.0`; mutually orthogonal keys score `0.0`; the measure is
scale-invariant (direction only).

## Usage

SqueezeAttention's reallocation only takes effect through
`KVCacheBuilder.for_model`, which builds a shared coordinator across layers:

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="squeeze",
    head_dim=128,
    squeeze_budget=512,  # AVERAGE budget across layers (uniform-H2O baseline)
    squeeze_n_sink=4,  # initial positions never evicted (attention sinks)
    squeeze_strength=1.0,  # 0.0 = uniform (== H2O), 1.0 = full inverse-concentration
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

Building a single cache via `KVCacheFactory.create` (no coordinator) falls back to
`squeeze_budget` and behaves as one uniform-budget H2O layer.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `squeeze_budget` | `512` | **Average** per-layer budget. The reallocation is centred on this value, so total memory matches uniform H2O at the same number. |
| `squeeze_n_sink` | `4` | Initial positions always retained (attention sinks). Also sets the minimum-budget floor (`n_sink + 1`). |
| `squeeze_strength` | `1.0` | Reallocation strength. `0.0` = uniform (identical to H2O); `1.0` = full inverse-concentration split. Interpolates linearly. |
| `squeeze_resolved_budget` | `None` | Explicit per-layer budget override (mainly for single-cache/testing). Normally `None` — the coordinator supplies it after prefill. |

## The budget schedule

`squeeze_budgets(concentrations, avg_budget, n_sink, strength)` returns the
per-layer budgets. Each layer's raw weight is `(1 - concentration)`; weights are
normalised so the mean budget equals `avg_budget`, then blended toward uniform by
`strength` and floored at `n_sink + 1`:

```
budget[i] = avg_budget * ((1 - strength) + strength * weight[i])
```

Example — 3 layers with concentrations `[0.1, 0.5, 0.9]`, `avg_budget=100`,
`strength=1.0`:

```
[180, 100, 20]   mean = 100.0
```

The broad layer (concentration 0.1) gets ~1.8× the average; the concentrated layer
(0.9) gets ~0.2×. `strength=0.0` makes every layer `100`.

## How it works — the one-shot re-budget

The repo's contract is one cache per layer, iterated independently by
`mlx_lm.generate`. SqueezeAttention needs a global view to reallocate, so a single
shared **`SqueezeCoordinator`** is injected at `for_model` build time:

1. **Prefill.** On its first `update_and_fetch`, each layer measures its
   concentration over the incoming keys and reports it to the coordinator. Until
   the coordinator finalises, the layer evicts against the average fallback
   budget.
2. **Finalise (once).** When every attention layer has reported, the coordinator
   computes the schedule with `squeeze_budgets(...)` and publishes each layer's
   resolved budget.
3. **Adopt + trim.** Each layer pulls its resolved budget and re-stamps it onto
   every head's state; any head now over budget is trimmed by H2O cumulative
   score (lowest-score non-sink tokens dropped, sinks always kept).
4. **Decode.** Runs against the frozen schedule — no further re-budgeting.

This is the **first eviction method with a runtime re-budgeting step**. Unlike the
XQuant / MiniCache coordinators (which exchange *tensors* every step), this one
exchanges only per-layer scalars and runs its allocation exactly once.

Within a layer, eviction is identical to H2O: the incoming key is a proxy query,
`scores += softmax(K_stored @ k_i / sqrt(D))` accumulates importance, and the
lowest-score non-sink token is evicted when over budget. No `.bits` attribute —
stored K/V remain fp16. Each cache exposes `layer_budget`, `concentration`,
`compression_ratio`, and `tokens_kept`.

## Relationship to H2O and PyramidKV

SqueezeAttention **is** H2O with a per-layer budget — the eviction scorer, sink
protection, and byte accounting are shared. The only additions are the
`concentration_score` proxy, the `squeeze_budgets` allocator, and the coordinator
that re-budgets after prefill. Set `squeeze_strength=0.0` and SqueezeAttention and
H2O are bit-for-bit identical. Against PyramidKV it differs in *how* the per-layer
budget is chosen: PyramidKV assumes a fixed positional taper at build time;
SqueezeAttention measures concentration from the prompt and reallocates at the
prefill boundary.

## Proxy limitation

The paper derives each layer's budget from the **observed prefill attention maps**.
We use the cosine-dispersion of the key set as an attention-free stand-in — the
"broad vs concentrated" *shape* is captured, but the exact per-layer values are a
geometric proxy, not read from real attention. The re-budget is one-shot at the
prefill boundary (the paper also re-budgets once). Eviction within a layer uses the
same key-as-query proxy as H2O-adapted.

Documented as "SqueezeAttention-adapted (cosine-dispersion proxy, key-as-query
proxy, one-shot re-budget)" throughout — never claimed as a faithful port.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_squeeze_cache.py` (19 tests) and
`veloxquant_mlx/tests/quantizers/test_squeeze.py` (28 tests):

- **Concentration proxy:** `1.0` for identical keys, `0.0` for orthogonal keys,
  scale-invariant, neutral (`0.0`) for fewer than two rows, always in `[-1, 1]`
- **Allocator:** `strength=0` gives uniform budgets (== H2O) regardless of
  concentration; mean within 5% of `avg_budget`; budgets monotone in
  concentration; broad layer gets more than concentrated; floored at `n_sink+1`;
  all-concentrated falls back to uniform; negative concentration clamps; `strength`
  interpolates between uniform and full; single-layer, empty, and out-of-range
  `strength` edge cases
- **Eviction:** single-token bootstrap; budget never exceeded across a 40-step
  stress test; `budget + 1` → exactly `budget`; sinks always present after
  evictions; `n_sink=0` edge case; score non-negativity; byte accounting
- **Coordinator:** not finalised until all layers report; resolves broad→more /
  concentrated→less; report is idempotent per layer; `strength=0` uniform; reset
- **for_model:** returns `SqueezeAttentionCache` per layer; one shared coordinator;
  data-driven budgets vary; broad early layer keeps more than deep concentrated
  layer; mean ≈ `avg_budget`; `strength=0` uniform; budget enforced after re-budget
- Determinism: identical inputs produce identical outputs

The offline harness in `benchmark_scripts/benchmark_squeeze.py` sweeps
`(n_layers, seq_len, avg_budget, strength)`, building a shared-coordinator per-layer
cache set against a mock model whose layers grow more concentrated with depth, and
reporting measured concentration, the resolved schedule, per-layer kept tokens, and
compression ratio — **synthetic, not model-level.** Results are committed in
`figures/squeeze/results.json` (run on Apple Silicon). They
confirm the design end-to-end: `strength=0.0` produces uniform budgets (== H2O);
`strength>0` reallocates so the broad early layer retains more tokens than the
concentrated deep layer; and the schedule mean matches `avg_budget`. The wall-clock
numbers are dominated by the O(S²) pure-Python eviction loop run across all layers —
a **prefill-batch worst case**, not a per-decode-step cost.

**No model-level (perplexity/throughput) benchmark has been run.** The committed
numbers are the synthetic harness only; no quality figures are claimed.

## When to use it

SqueezeAttention-adapted is best when you want H2O-style importance eviction with a
depth-adaptive budget **and** you would rather the split be inferred from the actual
prompt than fixed a priori. It generalises PyramidKV: identical machinery, but the
budget follows the measured attention geometry. At equal total memory it should
retain more of what matters for *this* input than either uniform H2O or a fixed
pyramid.

| Scenario | Recommended method |
|----------|-------------------|
| Compress all tokens uniformly | KIVI-2bit |
| Hard cap on tokens, evict at prefill only | SnapKV-adapted |
| Constant-memory, position-based eviction | StreamingLLM-adapted |
| Constant-memory, cumulative-importance eviction, uniform budget | H2O-adapted |
| Constant-memory, current-step-importance eviction (reactive) | TOVA-adapted |
| Constant-memory, importance eviction with a fixed depth-adaptive budget | PyramidKV-adapted |
| **Constant-memory, importance eviction with a data-driven depth-adaptive budget** | **SqueezeAttention-adapted** |

**See also:** [ChunkKV-adapted](./chunkkv) shares the same H2O importance scorer but
varies a different axis — the *granularity* of eviction (whole chunks vs single
tokens) rather than the per-layer budget. The two are orthogonal knobs on the same
eviction core; `chunk_size=1` and `strength=0` both recover plain H2O.
