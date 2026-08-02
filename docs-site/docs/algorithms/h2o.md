# H2O — Cumulative Attention-Mass Heavy-Hitter Oracle Eviction

**Method id:** `h2o` · **New in 0.21.0** · *Inspired by* [H2O (arXiv:2306.14048)](https://arxiv.org/abs/2306.14048)
(Zhang et al., ICLR 2024) — **H2O-adapted (VeloxQuant-MLX implementation)**,
not a faithful port.

H2O-adapted is the library's **third eviction axis** and the first based on
**cumulative per-token attention mass**. Unlike SnapKV-adapted (which fires once
at prefill end) and StreamingLLM-adapted (which evicts by position), H2O runs
continuously at every step and uses the running sum of attention weights as its
importance signal. The cache is bounded to `h2o_budget` positions at all times.

| Eviction axis | When it fires | Score signal | Memory shape |
|---|---|---|---|
| SnapKV-adapted | Once at prefill end | Key-as-query attention proxy | Grows during decode |
| StreamingLLM-adapted | Every token | Position (recency + sink) | Constant |
| **H2O-adapted** | Every token (budget exceeded) | Cumulative attention mass | Constant (≤ budget) |

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="h2o",
    head_dim=128,
    h2o_budget=512,  # max tokens retained at any time (sinks + non-sinks)
    h2o_n_sink=4,  # initial positions never evicted (attention sinks)
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `h2o_budget` | `512` | Maximum token positions retained at any time. When the cache exceeds this count, the lowest-score non-sink token is permanently evicted. |
| `h2o_n_sink` | `4` | Number of initial token positions always retained (attention-sink tokens never eligible for eviction). |

## How it works

For every incoming token (both prefill and decode), per head:

1. **Approximate attention distribution.** The new key vector `k_i` is used as a
   proxy query and attends to all currently stored key rows via scaled dot-product
   softmax: `attn = softmax(K_stored @ k_i / sqrt(D))`. This gives `[n_kept]`
   softmax weights for the existing cache entries.
2. **Score accumulation.** The weights are added to the existing per-token cumulative
   score vector: `scores += attn`. New tokens start with score 0 and begin
   accumulating on subsequent steps.
3. **Eviction (if over budget).** If the total token count exceeds `h2o_budget`, a
   protected score view is constructed: the first `h2o_n_sink` positions receive
   `+inf` (they are never evicted). The token with the minimum protected score is
   permanently removed.
4. **Guarantee.** After every step, the cache holds at most `h2o_budget` tokens.

No `.bits` attribute — stored K/V remain in fp16. The `compression_ratio` and
`tokens_kept` properties report the storage accounting.

## Proxy limitation

The paper accumulates attention weights from the **true query** vectors at each
decode step. At the cache-wrapper level, queries are not visible — only K and V
arrive at `update_and_fetch`. We substitute the incoming **key vector** as a proxy
query, computing an approximation of the attention distribution over stored keys.

This is the same key-as-query approximation used by SnapKV-adapted. Keys and queries
are both projected from the same residual stream and are correlated, but the proxy is
still an approximation. In particular, it may over-weight tokens that are geometrically
similar to recent keys rather than those that answer the actual query.

Documented as "H2O-adapted (key-as-query proxy)" throughout — never claimed as a
faithful port.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_h2o_cache.py` (15 tests) and
`veloxquant_mlx/tests/quantizers/test_h2o.py` (18 tests):

- `init_h2o_state` fields correct; empty state returns zero-row K/V placeholder
- Single token bootstraps state; multi-token absorption below budget keeps all tokens
- Budget never exceeded across a 30-step decode stress test
- `budget + 1` tokens → exactly `budget` tokens remain after eviction
- `scores` array length always equals number of kept tokens
- First `h2o_n_sink` tokens survive all evictions (verified with known-value sink tokens)
- `n_sink=0` edge case: all tokens eligible for eviction, budget still respected
- Scores are non-negative (sums of softmax weights)
- Scores accumulate across steps (total mass grows monotonically)
- Byte accounting formula: `n_kept * D * 4` (K + V, fp16)
- `compression_ratio == 1` below budget; `> 1` after evictions
- `tokens_seen` accumulates by `B * H * S` per call
- Factory dispatch (`KVCacheFactory.create`) returns `H2OKVCache`
- `for_model` propagates `h2o_budget` and `h2o_n_sink` to all layer caches
- Determinism: identical inputs produce identical outputs

The offline harness in `benchmark_scripts/benchmark_h2o.py` sweeps
`(seq_len, budget, n_sink)` and reports latency and compression ratio —
**synthetic, not model-level.**

**No model-level benchmark has been run.** Until `results_h2o.json` is committed
with hardware numbers, no throughput or perplexity figures are claimed.

## When to use it

H2O-adapted is best when you want a **budget-bounded cache that improves over
recency-only eviction** (StreamingLLM) by using attention signal rather than position.
Heavy-hitter tokens (those consistently attended to) survive; recency is not the only
criterion for retention.

| Scenario | Recommended method |
|----------|-------------------|
| Compress all tokens uniformly | KIVI-2bit |
| Hard cap on tokens, evict at prefill only | SnapKV-adapted |
| Constant-memory, position-based eviction | StreamingLLM-adapted |
| **Constant-memory, importance-based eviction (continuous)** | **H2O-adapted** |
| Recover quality from aggressive quantization | GEAR |

**See also:** [CaM-adapted](./cam) makes the same eviction choice as H2O but
**merges** the loser into a similar survivor instead of dropping it, recovering a
share of the lost mass at high compression. `cam_merge="drop"` is exactly H2O.

See also: [L2Norm](../algorithms/knorm) — the same keep/evict machinery with an *intrinsic* scorer (key L2 norm, computed once at insertion): no per-step softmax over the cache, and path-independent kept sets, at the price of trusting the paper's low-norm ⇒ high-attention finding rather than reacting to the query stream. And [Q-Filters](../algorithms/qfilters) — the same machinery again with a *projection* scorer (a frozen per-head key-SVD direction): also no per-step softmax, but path-dependent and sign-ambiguous, since the direction is estimated from keys rather than the paper's queries. And [Keyformer](../algorithms/keyformer) — H2O's *exact* accumulator plus one ingredient: **Gumbel noise** on the eviction logits that rescues "late riser" tokens greedy accumulation would prune too early. `keyformer_tau=0` collapses back onto this H2O cache bit-for-bit. And [MorphKV](../algorithms/morphkv) — the antidote to this cache's *early-token bias*: rather than accumulating attention forever, it ranks stored tokens by a **sliding window of recent** attention, so a constant-size cache re-targets toward the current topic instead of clinging to stale early heavy hitters. And [KVzip](../algorithms/kvzip) — a different antidote to the same bias: rather than accumulating query attention, it ranks stored tokens by **reconstruction reliance** (how much the model relies on a KV pair to reconstruct its own context), a *query-agnostic* profile that keeps what any future query is likely to need instead of what happened to be a heavy hitter early. And [CurDKV](../algorithms/curdkv) — the antidote to this cache's *key-only blindness*: every scorer above (this one included) ranks a token using only its key side, so a token whose key looks important but whose value contributes nothing to the output is indistinguishable from one that truly matters. CurDKV derives a **leverage score** from the joint key-and-value structure instead, so two tokens with identical keys but divergent values receive different scores — a distinction this H2O cache structurally cannot make.
