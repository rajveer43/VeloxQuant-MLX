# H2O — Cumulative Attention-Mass Heavy-Hitter Oracle Eviction

**Method id:** `h2o` · **New in 0.21.0** · *Inspired by* [H2O (arXiv:2306.14048)](https://arxiv.org/abs/2306.14048)
(Zhang et al., ICLR 2024) — **H2O-adapted (VeloxQuant-MLX implementation)**,
not a faithful port.

:::danger[Not safe for real generation yet — cache freezes on early tokens]
Real-model testing (below) confirms eviction corrupts output — as few as one
or two evictions can send generation into repetition loops, and heavier
eviction causes total collapse. Root cause: the scoring formula gives every
newly-arrived token a starting score of exactly `0`, and the eviction rule
removes the *global minimum* score — so once the cache is full, the brand-new
token is (almost always) its own eviction target, before it ever gets a
chance to accumulate attention mass. In practice this means **the kept set
freezes on whichever tokens filled the budget first** (typically the prompt)
and never admits anything generated afterward — the model ends up generating
forever while attending only to the original prompt, with zero visibility
into its own recent output. This is a real limitation of the paper's own
cumulative-sum formulation at low budgets, not an implementation bug in the
eviction *rule* itself (which correctly implements Algorithm 1 — see
[fidelity check](#fidelity-to-algorithm-1)). See
[the finding](#real-end-to-end-generation-eviction-breaks-coherence) before
using `method="h2o"` for anything beyond studying the mechanism. A separate
scalability bug that used to crash long prefills outright (regardless of
eviction settings) has since been fixed — see
[long-context testing](#long-context-testing-the-freeze-holds-at-scale-plus-a-scalability-bug-now-fixed).
:::

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

## Fidelity to Algorithm 1

Checked directly against the paper's Algorithm 1 / Definition 4.3 (local
greedy heavy-hitter eviction):

| Paper | This implementation | Match |
|---|---|---|
| `Si ← (Si−1 ∪ {i}) \ {u}`, `u ← argmax_v Fscore(Si−1∪{i}\{v})` | Evict the single lowest-scoring element (`argmin`) — equivalent to maximizing the *remaining* set's score | ✅ |
| `Fscore(T) = Σ_{s∈T} o_s`, accumulated per token | Running sum of softmax weights per token | ✅ |
| §4.1: local (no-lookahead) H2 is "equally effective" as the global variant — the paper's own recommended, deployable version | Implements exactly the local variant | ✅ (correct choice) |
| `\|Si\| = k`, evict at most one per step | `budget` field, one eviction per over-budget call | ✅ |
| §5.3 sink/StreamingLLM extension | `n_sink` protected leading positions | ✅ |

**Proxy limitation:** the paper accumulates attention weights from the
**true query** vectors at each decode step. At the cache-wrapper level,
queries are not visible — only K and V arrive at `update_and_fetch`, and (as
documented in [How it works](#how-it-works) above) `mlx_lm`'s attention
module consumes the true query internally before the cache is ever called.
We substitute the incoming **key vector** as a proxy query, computing an
approximation of the attention distribution over stored keys.

This is the same key-as-query approximation used by SnapKV-adapted. Keys and queries
are both projected from the same residual stream and are correlated, but the proxy is
still an approximation. In particular, it may over-weight tokens that are geometrically
similar to recent keys rather than those that answer the actual query.

Documented as "H2O-adapted (key-as-query proxy)" throughout — never claimed as a
faithful port.

**Not a fidelity gap, but a consequence of the formula itself:** the paper's
own scoring rule (new tokens start at score 0, minimum gets evicted) is what
produces the early-token freeze described below — implementing it exactly as
specified is what causes the practical problem, not a deviation from it.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_h2o_cache.py` (18 tests) and
`veloxquant_mlx/tests/quantizers/test_h2o.py` (28 tests):

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
- Position tracking stays gap-free and contiguous under a 60-step stress test; `n_sink` positions never move
- Interior-eviction RoPE remap recovers each surviving key's exact original pre-rotation value (fingerprinted-token test)
- `cache.offset` tracks the true absolute step count, not the kept-row count, once eviction has occurred
- The vectorized below-budget batch path matches the sequential per-token loop it replaces bit-for-bit within fp16 rounding, including for a single call whose batch straddles the below-budget/over-budget boundary
- A 4,000-token single-call absorption (no eviction) completes without error — the regression test for the prefill scalability crash

The offline harness in `benchmark_scripts/benchmark_h2o.py` sweeps
`(seq_len, budget, n_sink)` and reports latency and compression ratio —
**synthetic, not model-level.**

### Real end-to-end generation: eviction breaks coherence

Model-level testing has now been run — `mlx_lm.generate` on
`mlx-community/Llama-3.2-1B-Instruct-4bit` (`head_dim=64`, `rope_theta=500000`)
and `mlx-community/Mistral-7B-Instruct-v0.3-4bit` (`head_dim=128`,
`rope_theta=1000000`) — and the result is a real, reproducible problem, not
just a theoretical caveat.

:::danger[Eviction corrupts generation quality — this is not tuning, it collapses almost immediately]
The moment eviction actually fires (as little as **one or two evictions**,
tested with `h2o_n_sink=0` and `h2o_budget` set to only 1–10 tokens above the
prompt length), output degrades into repetition loops. Severity scales with
how many evictions occur — a handful causes phrase-level repetition, dozens
causes single-token loops:

| Model | Budget vs prompt | Evictions | Output |
|---|---|---|---|
| Llama-3.2-1B | prompt + 1 | many (40 generated tokens, budget 14) | `Paris.\nThe\nThe\nThe\nThe\nThe\nThe\nThe...` (collapses to one repeated token) |
| Llama-3.2-1B | prompt + 3 | many | `Paris.\nWhat is the capital is the capital is the capital...` (phrase-loop) |
| Llama-3.2-1B | prompt + 10 | many | `Paris.\nWhat is the largest city in France? Answer in one word. Answer in one word...` (longer phrase-loop, closer to baseline) |
| Llama-3.2-1B | no eviction (budget=10000) | 0 | `Paris.\nWhat is the largest city in France? Answer in one word. Paris.\nWhat is the largest city in France? Lyon...` (baseline repetition is a model quirk on this short prompt, not a cache defect) |
| Llama-3.2-1B | 32 (long 64-token prompt) | many | `- - - - - - - - - - - -...` (total collapse) |
| Mistral-7B | 64 (long 72-token prompt) | many | blank lines only (total collapse) |

**Root cause — verified precisely, and it is NOT a RoPE/position bug:**
the scoring formula gives every newly-arrived token a starting score of
exactly `0.0` (it hasn't accumulated any attention mass yet), and the
eviction rule removes whichever token holds the **global minimum** score. On
essentially any real (non-degenerate) score distribution, the brand-new
token's `0.0` *is* that global minimum — so **the most recently generated
token is evicted on almost every step once the cache is full**, before it
ever gets a chance to accumulate score. Traced directly on the Llama-3.2-1B
run above: after 43 true decode steps, the cache's kept positions were still
exactly `[0..13]` — the prompt, unchanged, with **zero** generated tokens
ever admitted. The model spent the entire generation attending only to the
original prompt, with no visibility into anything it had itself generated —
which is exactly the repetition-loop failure pattern observed. This is a
genuine limitation of the paper's cumulative-sum scoring at low/tight
budgets, not a bug in this implementation's eviction *rule*, which correctly
implements the paper's Algorithm 1 (verified line-by-line — see
[Evidence](#evidence) above and the module docstring).

A second, narrower issue was found and fixed during this investigation:
K/V arriving at `update_and_fetch` are already RoPE-rotated by the attention
layer upstream, and `mlx_lm` also rotates the *next* query/key using
`cache.offset` — both assume a contiguous, gap-free cache. Once *any* row is
evicted from the interior of the retained set (which the early-token freeze
above makes rare, but not impossible — e.g. with `h2o_n_sink > 0`, sink rows
are permanently protected while later rows can still eventually be evicted,
opening a real interior gap), position bookkeeping desyncs and corrupts
attention math for every step afterward. Both are now fixed: stored keys are
de-rotated and re-rotated to a gap-free layout on every eviction
(`rope_remap_positions`), and `cache.offset` is tracked as the true absolute
step count rather than the kept-row count, so the model's own query rotation
stays correct without this cache needing to intercept it. Verified via a
synthetic fingerprinted-token test that forces an interior eviction and
confirms every surviving key de-rotates back to its exact original
unrotated value. This fix is real and independently worth having, but by
itself it does **not** fix the collapse demonstrated above — the freeze
happens before interior eviction geometry ever becomes relevant.
:::

**Practical takeaway:** do not use `h2o` in this library for real generation
today. It is validated at the unit/synthetic level (the mechanism —
scoring, sink protection, budget enforcement, and now RoPE position
consistency under interior eviction — is implemented correctly per the
paper's Algorithm 1) but **not yet safe for actual model output**, because
the paper's own cumulative-sum scoring formula starves newly-generated
tokens of any chance to survive once the budget is full. Fixing this needs a
change to the eviction/scoring policy itself (e.g. a grace period before a
token becomes eviction-eligible, or age-normalized scoring) — a bigger,
more opinionated algorithmic change than a bugfix, and not undertaken here.
Tracked as follow-up work, not silently patched over.

### Long-context testing: the freeze holds at scale, plus a scalability bug (now fixed)

The same methodology was repeated on a genuinely long prompt (~3,238 tokens)
to check whether the early-token freeze above behaves differently at scale,
and to stress-test the implementation on long context in general.

| Budget | Prefill | Decode | Kept (final) | Evicted | Compression | Generated tokens survived? | Output |
|---|---|---|---|---|---|---|---|
| 256 | 3,238 | 300 | 256 | 3,282 | 13.82× | **0** | `"of of of of of..."` — total collapse |
| 800 | 3,238 | 300 | 800 | 2,738 | 4.42× | **0** | Garbled/gibberish tokens |
| 1,200 | 3,238 | 400 | 1,200 | 2,438 | 3.03× | **0** | Degenerates into `"I am I am I am..."` loop |

In every case the kept set is exactly `[0, budget-1]` — the earliest slice of
the **prompt** — confirming the freeze is budget- and prompt-length-independent:
bigger budgets don't fix it, they just delay how quickly the output degrades
from coherent-looking phrases into pure repetition.

:::warning[A second, independent bug was found and fixed here: H2O couldn't even complete a long prefill]
Testing the no-eviction baseline (`h2o_budget` larger than the prompt, so no
eviction should ever occur) crashed **during prefill alone**, before a single
decode step:

```
RuntimeError: [metal::malloc] Resource limit (499000) exceeded.
```

Bisected the breaking point: prefill succeeded at 500 tokens, failed
somewhere before 1,000 — independent of `h2o_budget`, `h2o_n_sink`, or
`max_tokens` (reproduced even at `max_tokens=1`).

**Root cause:** `h2o_update` processed every incoming token, including the
entire prefill batch, with a Python `for` loop issuing 4 separate
`mx.concatenate` calls per token (keys, values, scores, positions). At
~3,238 tokens this builds an unfused lazy-evaluation graph large enough to
exceed MLX's Metal resource/command-buffer tracking limit — a pure
implementation scalability bug, completely independent of the RoPE fixes and
the scoring-freeze issue above.

**Fixed:** whichever leading portion of an incoming batch is guaranteed not
to trigger eviction (because the cache hasn't yet reached `h2o_budget`) is
now absorbed via a single batched masked-attention matmul instead of a
per-token loop — mathematically the same score-accumulation formula, just
computed all at once. Verified numerically equivalent to the sequential loop
it replaces (exact match to fp16 rounding, including for calls that straddle
the below-budget/over-budget boundary). The same 3,238-token no-eviction
prefill that previously crashed now completes in ~2.4–3.2 seconds.

Only the genuinely sequential part — the eviction decision itself, where
each eviction depends on the previous step's result — still runs a per-token
loop, and only once the cache is actually full. This means the fix has no
effect on the freeze-heavy scenarios in the table above (their timing and
output are byte-for-byte identical before and after this fix); it only
unblocks the case that used to crash outright: long context with a budget
large enough to avoid heavy early eviction.
:::

## When to use it

**Today: nowhere in production.** The [confirmed eviction-corruption finding
above](#real-end-to-end-generation-eviction-breaks-coherence) means this cache
should not be pointed at real generation until position remapping is
implemented — treat it as a reference implementation of the paper's scoring
mechanism, validated at the unit/synthetic level only.

Once that's fixed, the intended niche is a **budget-bounded cache that
improves over recency-only eviction** (StreamingLLM) by using attention
signal rather than position — heavy-hitter tokens (those consistently
attended to) survive; recency is not the only criterion for retention.

| Scenario | Recommended method |
|----------|-------------------|
| Compress all tokens uniformly | KIVI-2bit |
| Hard cap on tokens, evict at prefill only | SnapKV-adapted |
| Constant-memory, position-based eviction | StreamingLLM-adapted |
| Constant-memory, importance-based eviction (continuous) | H2O-adapted — **not yet safe for real generation, see above** |
| Recover quality from aggressive quantization | GEAR |

**See also:** [CaM-adapted](./cam) makes the same eviction choice as H2O but
**merges** the loser into a similar survivor instead of dropping it, recovering a
share of the lost mass at high compression. `cam_merge="drop"` is exactly H2O.

See also: [L2Norm](../algorithms/knorm) — the same keep/evict machinery with an *intrinsic* scorer (key L2 norm, computed once at insertion): no per-step softmax over the cache, and path-independent kept sets, at the price of trusting the paper's low-norm ⇒ high-attention finding rather than reacting to the query stream. And [Q-Filters](../algorithms/qfilters) — the same machinery again with a *projection* scorer (a frozen per-head key-SVD direction): also no per-step softmax, but path-dependent and sign-ambiguous, since the direction is estimated from keys rather than the paper's queries. And [Keyformer](../algorithms/keyformer) — H2O's *exact* accumulator plus one ingredient: **Gumbel noise** on the eviction logits that rescues "late riser" tokens greedy accumulation would prune too early. `keyformer_tau=0` collapses back onto this H2O cache bit-for-bit. And [MorphKV](../algorithms/morphkv) — the antidote to this cache's *early-token bias*: rather than accumulating attention forever, it ranks stored tokens by a **sliding window of recent** attention, so a constant-size cache re-targets toward the current topic instead of clinging to stale early heavy hitters. And [KVzip](../algorithms/kvzip) — a different antidote to the same bias: rather than accumulating query attention, it ranks stored tokens by **reconstruction reliance** (how much the model relies on a KV pair to reconstruct its own context), a *query-agnostic* profile that keeps what any future query is likely to need instead of what happened to be a heavy hitter early. And [CurDKV](../algorithms/curdkv) — the antidote to this cache's *key-only blindness*: every scorer above (this one included) ranks a token using only its key side, so a token whose key looks important but whose value contributes nothing to the output is indistinguishable from one that truly matters. CurDKV derives a **leverage score** from the joint key-and-value structure instead, so two tokens with identical keys but divergent values receive different scores — a distinction this H2O cache structurally cannot make.
