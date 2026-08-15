# TOVA — Current-Step Attention-Weight Eviction (Memoryless)

**Method id:** `tova` · **New in 0.22.0** · *Inspired by* [TOVA / "Transformers are Multi-State RNNs" (arXiv:2401.06104)](https://arxiv.org/abs/2401.06104)
(Oren et al., 2024) — **TOVA-adapted (VeloxQuant-MLX implementation)**,
not a faithful port.

TOVA-adapted is the library's **fourth eviction axis** and the first
**memoryless** one. At every step it drops the token receiving the lowest
attention weight in the **current** step — no running score is carried across
steps. The cache is bounded to `tova_budget` positions at all times.

The distinction from H2O-adapted is the whole point of adding TOVA:

| | H2O-adapted | TOVA-adapted |
|---|---|---|
| Score signal | **Cumulative** attention mass (sum over all steps) | **Current-step** attention weight only |
| Memory of past | Inertial — a past heavy hitter resists eviction | Memoryless — history discarded every step |
| Reaction | Conservative | Reactive — a token that stops being attended is dropped |
| State field | `scores` (per-token running sum) | none |

## Eviction axes at a glance

| Eviction axis | When it fires | Score signal | Memory shape |
|---|---|---|---|
| SnapKV-adapted | Once at prefill end | Key-as-query attention proxy | Grows during decode |
| StreamingLLM-adapted | Every token | Position (recency + sink) | Constant |
| H2O-adapted | Every token (budget exceeded) | Cumulative attention mass | Constant (≤ budget) |
| **TOVA-adapted** | Every token (budget exceeded) | Current-step attention weight | Constant (≤ budget) |

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="tova",
    head_dim=128,
    tova_budget=512,  # max tokens retained at any time (sinks + non-sinks)
    tova_n_sink=4,  # initial positions never evicted (attention sinks)
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tova_budget` | `512` | Maximum token positions retained at any time. When the cache exceeds this count, the non-sink token with the lowest current-step attention weight is permanently evicted. |
| `tova_n_sink` | `4` | Number of initial token positions always retained (attention-sink tokens never eligible for eviction). |

## How it works

For every incoming token (both prefill and decode), per head:

1. **Append.** The new key/value pair is appended to the cache.
2. **Score the current step (only if over budget).** The new key vector `k_i` is
   used as a proxy query and attends to all rows including itself via scaled
   dot-product softmax: `weights = softmax(K_cache @ k_i / sqrt(D))`. This gives
   `[n_total]` softmax weights for the current step.
3. **Eviction (if over budget).** A protected weight view is constructed: the
   first `tova_n_sink` positions receive `+inf` (never evicted). The token with
   the minimum protected weight is permanently removed. The weights are then
   **discarded** — nothing is carried to the next step.
4. **Guarantee.** After every step, the cache holds at most `tova_budget` tokens.

No `.bits` attribute — stored K/V remain in fp16. The `compression_ratio` and
`tokens_kept` properties report the storage accounting.

## Why memoryless matters

H2O accumulates attention mass, so a token that was heavily attended early in the
sequence keeps a high score and resists eviction long after it stops being
relevant. TOVA scores by the present step alone: if a token is no longer attended
to *right now*, it is a candidate for eviction regardless of its past. This makes
TOVA more responsive to topic shifts within a long context, at the cost of
possibly dropping a token that will be attended to again later.

Neither policy dominates — they have different failure modes, which is exactly why
both are provided.

## Proxy limitation

The paper reads the **true attention distribution** of the most recent query row
from the forward pass. At the cache-wrapper level, queries are not visible — only
K and V arrive at `update_and_fetch`. We substitute the incoming **key vector** as
a proxy query, computing an approximation of the current-step attention
distribution over stored keys.

This is the same key-as-query approximation used by SnapKV-adapted and H2O-adapted.
Keys and queries are both projected from the same residual stream and are
correlated, but the proxy is still an approximation. In particular, it may
over-weight tokens that are geometrically similar to the incoming key rather than
those the actual query would attend to.

Documented as "TOVA-adapted (key-as-query proxy)" throughout — never claimed as a
faithful port.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_tova_cache.py` (18 tests) and
`veloxquant_mlx/tests/quantizers/test_tova.py` (19 tests):

- `init_tova_state` fields correct; state carries **no `scores` field** (memoryless)
- Empty state returns zero-row K/V placeholder
- Single token bootstraps state; multi-token absorption below budget keeps all tokens
- Budget never exceeded across a 30-step decode stress test
- `budget + 1` tokens → exactly `budget` tokens remain after eviction
- Keys and values row counts stay equal after evictions
- First `tova_n_sink` tokens survive all evictions (verified with known-value sink tokens)
- `n_sink=0` edge case: all tokens eligible for eviction, budget still respected
- **Memorylessness:** no scores are carried across updates (asserted every step)
- **Current-step eviction correctness:** a token orthogonal to the incoming key is
  evicted over one aligned with it (constructed axis-aligned test vectors)
- Byte accounting formula: `n_kept * D * 4` (K + V, fp16)
- `compression_ratio == 1` below budget; `> 1` after evictions
- `tokens_seen` accumulates by `B * H * S` per call
- Factory dispatch (`KVCacheFactory.create`) returns `TOVAKVCache`
- `for_model` propagates `tova_budget` and `tova_n_sink` to all layer caches
- Determinism: identical inputs produce identical outputs
- `cache.offset` tracks the true absolute token position (not the retained
  row count) through sustained eviction, across prefill block size, and
  across a prefill-then-decode mix

#### RoPE positions had to be fixed too (#175)

`mlx_lm` rotates both the query and the incoming key at `offset=cache.offset`
*before* `update_and_fetch` runs. Before #175, `TOVAKVCache` reset
`self.offset = 0` on every call and let the base `KVCache.update_and_fetch`
set it back to the retained-row count, so once eviction pinned the kept set
at `tova_budget`, `offset` stalled there while the true position kept
climbing — the same defect fixed for Q-Filters (#171) and L2Norm (#174).
`tova_update` drops exactly the evicted row and keeps the rest in temporal
order (never renumbers survivors), so the same `_true_offset` counter that
sufficed for those caches applies directly here — no `offset` property
split like [H2O's](../algorithms/h2o) was needed, because every
`update_and_fetch` call fully resets the base class's buffers regardless of
prefill or decode.

Note: H2O was audited in the same pass and does **not** share this defect —
it already tracks the true absolute step count directly (`self.offset =
self._states[0].next_pos`) and additionally re-rotates surviving keys when
eviction opens an interior gap, since H2O (unlike TOVA) can renumber
positions when `h2o_n_sink > 0` protects sinks while interior rows are
still evicted. See [H2O's docs](../algorithms/h2o) for that fix (shipped in
v0.44.4, predating this audit).

The offline harness in `benchmark_scripts/benchmark_tova.py` sweeps
`(seq_len, budget, n_sink)` and reports latency and compression ratio —
**synthetic, not model-level.** Results are committed in
`figures/tova/results.json` (28 configurations, run on Apple
Silicon). Across every configuration the measured compression ratio equals
`seq_len / budget` exactly (e.g. 2048 tokens at budget 64 → 32×), confirming the
eviction logic end-to-end. The latencies reflect the O(S²) cost of the
per-token Python eviction loop and are a **prefill-batch worst case** — per-token
decode cost is small.

**No model-level (perplexity/throughput) benchmark has been run.** The committed
numbers are the synthetic harness only; no quality figures are claimed.

## When to use it

TOVA-adapted is best when you want a **budget-bounded, reactive cache** that
prioritizes tokens relevant to the *current* context over ones that were important
in the past. It complements H2O: pick TOVA when context shifts within a long
sequence and stale heavy hitters should be released; pick H2O when consistently
attended tokens should be protected against transient dips in attention.

See also [MorphKV](../algorithms/morphkv) — it generalizes TOVA's single-latest-token
signal to a **sliding window** of recent tokens, so retention averages over recent
context instead of reacting to one (possibly noisy) latest query. Setting
`morphkv_window=1` collapses MorphKV back onto this TOVA cache, bit-for-bit.

See also [KVzip](../algorithms/kvzip) — it replaces TOVA's latest-**query** signal
with a **query-agnostic reconstruction probe**: rank a stored token by how much the
model relies on it to reconstruct its own context, so one compressed cache serves
diverse future queries. Setting `kvzip_probe="latest"` collapses KVzip back onto
this TOVA cache, bit-for-bit.

| Scenario | Recommended method |
|----------|-------------------|
| Compress all tokens uniformly | KIVI-2bit |
| Hard cap on tokens, evict at prefill only | SnapKV-adapted |
| Constant-memory, position-based eviction | StreamingLLM-adapted |
| Constant-memory, cumulative-importance eviction (inertial) | H2O-adapted |
| **Constant-memory, current-step-importance eviction (reactive)** | **TOVA-adapted** |
| Recover quality from aggressive quantization | GEAR |
