# PyramidKV — Layer-Adaptive Budget Attention-Mass Eviction

**Method id:** `pyramidkv` · **New in 0.23.0** · *Inspired by* [PyramidKV (arXiv:2406.02069)](https://arxiv.org/abs/2406.02069)
(Cai et al., 2024) — **PyramidKV-adapted (VeloxQuant-MLX implementation)**,
not a faithful port.

PyramidKV-adapted is the library's **fifth eviction configuration** and the first
with a **per-layer budget**. It is H2O-adapted's cumulative-attention-mass eviction
wearing a *pyramid* of budgets instead of a single global one: early layers get a
large budget, deep layers get a small one, and the **average is held fixed** so the
total cache size matches a uniform-budget baseline. When the pyramid is flat
(`pyramid_beta = 1.0`) it reduces exactly to H2O-adapted.

## Why a pyramid

The paper's observation is **pyramidal information funneling**: attention in early
transformer layers is broad and near-uniform (many tokens matter), while in deep
layers it concentrates on a few tokens. A uniform KV budget across all layers is
therefore mis-allocated — early layers are starved while deep layers are
over-provisioned. Redistributing the *same total budget* into a pyramid puts memory
where the attention actually spreads.

| Eviction axis | When it fires | Score signal | Budget |
|---|---|---|---|
| SnapKV-adapted | Once at prefill end | Key-as-query attention proxy | Uniform |
| StreamingLLM-adapted | Every token | Position (recency + sink) | Uniform |
| H2O-adapted | Every token (over budget) | Cumulative attention mass | **Uniform** |
| TOVA-adapted | Every token (over budget) | Current-step attention weight | Uniform |
| **PyramidKV-adapted** | Every token (over budget) | Cumulative attention mass | **Per-layer pyramid** |

## Usage

PyramidKV's pyramid only takes effect through `KVCacheBuilder.for_model`, which
knows each layer's index and can compute the schedule:

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="pyramidkv",
    head_dim=128,
    pyramid_budget=512,   # AVERAGE budget across layers (uniform-H2O baseline)
    pyramid_n_sink=4,     # initial positions never evicted (attention sinks)
    pyramid_beta=1.5,     # pyramid steepness: 1.0 = flat (== H2O), larger = steeper
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

Building a single cache via `KVCacheFactory.create` (no layer context) falls back to
`pyramid_budget` and behaves as one uniform-budget H2O layer.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pyramid_budget` | `512` | **Average** per-layer budget. The pyramid is centred on this value, so total memory matches uniform H2O at the same number. |
| `pyramid_n_sink` | `4` | Initial positions always retained (attention sinks). Also sets the minimum-budget floor (`n_sink + 1`). |
| `pyramid_beta` | `2.0` | Pyramid steepness. `1.0` = flat (identical to H2O); larger = steeper taper (early layers keep more, deep layers keep fewer). |

## The budget schedule

`pyramid_budgets(n_layers, avg_budget, n_sink, beta)` returns the per-layer budgets:
a linear taper from a maximum at layer 0 to a minimum at the last layer, centred so
the mean equals `avg_budget`, floored at `n_sink + 1`.

Example — 12 layers, `avg_budget=512`, `beta=2.0`:

```
[1019, 927, 835, 742, 650, 558, 466, 374, 282, 189, 97, 5]   mean = 512.0
```

Early layers hold ~2× the average; the deepest layer holds only its sinks plus a few
tokens. A gentler `beta=1.5` narrows this spread; `beta=1.0` makes every layer 512.

## How it works

At `for_model` build time:

1. **Count** the attention-bearing layers.
2. **Allocate** the pyramid with `pyramid_budgets(...)` and inject each layer's value
   into that layer's config as `pyramid_resolved_budget`.
3. Each `PyramidKVCache` is constructed with its own budget — **no runtime
   coordinator** is needed (unlike XQuant / MiniCache); layers never exchange data
   during generation.

Then at every step, per head, within each layer (identical to H2O):

1. The new key vector `k_i` is a proxy query and attends softmax over stored keys:
   `attn = softmax(K_stored @ k_i / sqrt(D))`.
2. `scores += attn` accumulates cumulative importance; new tokens start at 0.
3. If the layer exceeds **its own budget**, the lowest-score non-sink token is
   evicted (first `pyramid_n_sink` protected with `+inf`).

No `.bits` attribute — stored K/V remain fp16. Each cache exposes `layer_budget` (its
resolved budget), `compression_ratio`, and `tokens_kept`.

## Relationship to H2O

PyramidKV **is** H2O with a per-layer budget. The eviction scorer, sink protection,
and byte accounting are shared. The only addition is the `pyramid_budgets` allocator
and the `for_model` wiring that gives each layer a different budget. Set
`pyramid_beta=1.0` and PyramidKV and H2O are bit-for-bit identical.

## Proxy limitation

The paper derives each layer's budget from the **observed prefill attention entropy**.
We use a fixed monotone (linear) schedule as a deterministic, calibration-free
stand-in — the funneling *shape* (early-broad, deep-narrow) is preserved, but the
exact per-layer values are not data-driven. Eviction within a layer uses the same
key-as-query proxy as H2O-adapted (true queries are not visible at the cache-wrapper
level).

Documented as "PyramidKV-adapted (fixed schedule, key-as-query proxy)" throughout —
never claimed as a faithful port.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_pyramidkv_cache.py` (19 tests) and
`veloxquant_mlx/tests/quantizers/test_pyramidkv.py` (24 tests):

- **Allocator:** schedule length matches layers; monotonically decreasing; mean within
  5% of `avg_budget`; `beta=1.0` gives a flat schedule (== H2O); larger `beta` widens
  the first-to-last spread; every budget floored at `n_sink + 1`; single-layer and
  empty-model edge cases; `beta < 1.0` raises
- **Eviction:** single-token bootstrap; budget never exceeded across a 30-step stress
  test; `budget + 1` → exactly `budget`; sinks always present after evictions;
  `n_sink=0` edge case; score non-negativity; byte accounting formula
- **for_model:** returns `PyramidKVCache` per layer; budgets form a decreasing
  pyramid; mean ≈ `avg_budget`; `beta=1.0` gives uniform budgets; an early-layer cache
  retains more tokens than a deep-layer cache on the same sequence
- Determinism: identical inputs produce identical outputs

The offline harness in `benchmark_scripts/benchmark_pyramidkv.py` sweeps
`(n_layers, seq_len, avg_budget, beta)`, building a full per-layer pyramid against a
mock model and reporting the schedule, per-layer kept tokens, and compression ratio —
**synthetic, not model-level.** Results are committed in
`figures/pyramidkv/results.json` (24 configurations, run on Apple
Silicon). They confirm the design end-to-end: `beta=1.0` produces a flat schedule
(every layer identical, == uniform H2O); `beta>1.0` produces a strictly decreasing
budget with the early-layer cache retaining more tokens than the deep-layer cache; and
the schedule mean matches `avg_budget` in every configuration. The wall-clock numbers
are dominated by the O(S²) pure-Python eviction loop run across all layers — a
**prefill-batch worst case**, not a per-decode-step cost.

**No model-level (perplexity/throughput) benchmark has been run.** The committed
numbers are the synthetic harness only; no quality figures are claimed.

## When to use it

PyramidKV-adapted is best when you already want H2O-style importance eviction but
suspect a **uniform budget is mis-allocated across depth** — which the funneling
literature argues is the common case. At equal total memory it should retain more of
what matters (broad early context, concentrated deep context) than uniform H2O.

| Scenario | Recommended method |
|----------|-------------------|
| Compress all tokens uniformly | KIVI-2bit |
| Hard cap on tokens, evict at prefill only | SnapKV-adapted |
| Constant-memory, position-based eviction | StreamingLLM-adapted |
| Constant-memory, cumulative-importance eviction, uniform budget | H2O-adapted |
| Constant-memory, current-step-importance eviction (reactive) | TOVA-adapted |
| **Constant-memory, importance eviction with depth-adaptive budget** | **PyramidKV-adapted** |
