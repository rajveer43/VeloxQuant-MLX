# H2O — Cumulative Attention-Mass Heavy-Hitter Oracle Eviction

**Method id:** `h2o` · **New in 0.21.0** · *Inspired by* [H2O (arXiv:2306.14048)](https://arxiv.org/abs/2306.14048)
(Zhang et al., ICLR 2024) — **H2O-adapted (VeloxQuant-MLX implementation)**,
not a faithful port.

:::warning[Freeze mechanism fixed via `h2o_grace` — but tight budgets still degrade output]
Real-model testing (below) originally found that eviction corrupted output —
as few as one or two evictions could send generation into repetition loops,
and heavier eviction caused total collapse. Root cause: the scoring formula
gives every newly-arrived token a starting score of exactly `0`, and the
eviction rule removes the *global minimum* score — so the brand-new token
was (almost always) its own eviction target the instant the cache filled,
before it ever accumulated attention mass. The kept set froze on whichever
tokens filled the budget first (typically the prompt) and never admitted
anything generated afterward.

**Fixed** via `h2o_grace` (default `16`): the most-recently-arrived `grace`
tokens are protected from eviction the same way sink tokens are, giving
every new token `grace` update steps to actually compete before it can be
evicted. Verified structurally on real models: the kept-position window now
genuinely advances with generation instead of freezing at `[0, budget-1]`
forever — see [grace-period testing](#grace-period-testing-fixes-the-freeze-mechanism-coherence-still-needs-budget-headroom).

**Not fully solved:** fixing the freeze mechanism is necessary but not
sufficient for *coherent* output. When `h2o_budget` is tight relative to
`h2o_n_sink + h2o_grace` plus how much of the prompt needs to survive,
eviction is forced to thin out old tokens into a sparse, gapped window
(e.g. keeping positions `..., 37, 39, 41, 43, ...` instead of a contiguous
range) to make room — and generation still degrades under that regime, just
via a different mechanism (broken local coherence from the gaps) rather than
the original freeze. Give `h2o_budget` real headroom over `h2o_n_sink +
h2o_grace` for coherent output; see the table below for what "enough" looks
like in practice.
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
    h2o_budget=512,  # max tokens retained at any time (sinks + grace + non-sinks)
    h2o_n_sink=4,  # initial positions never evicted (attention sinks)
    h2o_grace=16,  # most-recent tokens never evicted (fixes the early-token freeze)
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `h2o_budget` | `512` | Maximum token positions retained at any time. When the cache exceeds this count, the lowest-score non-sink, non-grace token is permanently evicted. |
| `h2o_n_sink` | `4` | Number of initial token positions always retained (attention-sink tokens never eligible for eviction). |
| `h2o_grace` | `16` | Number of most-recently-arrived tokens always retained, giving each new token this many update steps to accumulate real attention mass before it becomes eviction-eligible. `0` reproduces the original paper-faithful (but freeze-prone) behavior. `h2o_n_sink + h2o_grace` must be `< h2o_budget`. |

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
   protected score view is constructed: the first `h2o_n_sink` positions
   receive `+inf` (attention sinks), and the last `h2o_grace` positions — the
   most recently arrived tokens, since positions are always kept sorted
   ascending after eviction — also receive `+inf`. The token with the
   minimum protected score among the remainder is permanently removed.
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
`h2o_grace` (default `16`, see [How it works](#how-it-works)) is this
library's own addition on top of Algorithm 1 to make the method usable in
practice; the paper itself does not specify a grace period, so `h2o_grace=0`
is the fidelity-preserving setting if you want to study the paper's
mechanism exactly as published.

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
- `h2o_grace` defaults to a nonzero value in `KVCacheConfig`; grace-protected tokens survive even when they hold the lowest score, verified against a synthetic case where `h2o_grace=0` would evict the newest tokens
- The fused Metal eviction kernel (`veloxquant_mlx/tests/metal/test_h2o_evict.py`, 18 tests) matches the pure-MLX eviction path bit-for-bit, including sink+grace protection, interior eviction, tie-break-matches-`mx.argmin`, and bit-identical untouched rows

The offline harness in `benchmark_scripts/benchmark_h2o.py` sweeps
`(seq_len, budget, n_sink)` and reports latency and compression ratio —
**synthetic, not model-level.**

### Real end-to-end generation: the `h2o_grace=0` baseline (historical)

Model-level testing was run — `mlx_lm.generate` on
`mlx-community/Llama-3.2-1B-Instruct-4bit` (`head_dim=64`, `rope_theta=500000`)
and `mlx-community/Mistral-7B-Instruct-v0.3-4bit` (`head_dim=128`,
`rope_theta=1000000`) — and found a real, reproducible problem, not just a
theoretical caveat. This section documents the **`h2o_grace=0` behavior**
(the library's default before the grace-period fix, and still what you get
if you explicitly set `h2o_grace=0` to study the paper's mechanism exactly
as published). See [grace-period testing](#grace-period-testing-fixes-the-freeze-mechanism-coherence-still-needs-budget-headroom)
below for the current default (`h2o_grace=16`) behavior.

:::danger[grace=0: eviction corrupts generation quality — this is not tuning, it collapses almost immediately]
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

**This `h2o_grace=0` behavior is no longer the default.** It's preserved
here as a historical record and as an explicit opt-in
(`h2o_grace=0`) for anyone studying the paper's Algorithm 1 exactly as
published, including its practical failure mode at tight budgets. For real
usage, see the grace-period results next.

### Grace-period testing: fixes the freeze mechanism, coherence still needs budget headroom

Re-ran the same real-model methodology with `h2o_grace > 0` to check whether
protecting recently-arrived tokens actually fixes the freeze — and to be
honest about what it does and doesn't fix.

**The freeze mechanism itself is fixed, structurally verified.** With
`h2o_grace=8` and a short prompt (13 tokens, `h2o_budget=14`), the kept
position set — which with `h2o_grace=0` stayed frozen at `[0..13]` forever —
now genuinely advances: `[0, 1, 2, 3, 4, 5, 27, 29, 31, 33, 35, 37, 39, 41]`.
Generated tokens are finally entering and surviving in the cache, which
never happened at `h2o_grace=0`.

**But structural advancement alone doesn't guarantee coherent output.** At
that same tight budget, the kept window has gaps every other position
(`27, 29, 31, ...`) to make room for both `h2o_grace` and ongoing eviction —
and generation degrades anyway, just via a different mechanism (broken local
context from the gaps) rather than the original freeze:

| `h2o_grace` | `h2o_budget` | Kept positions (tail) | Output |
|---|---|---|---|
| `0` | 14 | `[0..13]` (frozen) | `Paris.\nThe\nThe answer in French...` |
| `8` | 14 | `[..., 27, 29, 31, 33, 35, 37, 39, 41]` (gapped, but advancing) | `Paris.\nTheTheTheThe...` (still degenerates) |
| `0` | 32 | `[0..31]` (frozen) | Long but eventually repetitive |
| `16` | 32 | `[..., 21, 23, 25, ..., 51]` (gapped) | `...the largest largest largest...` (different repetition, still broken) |
| `32` | 128 | `[..., 110, 111, ..., 119]` (**contiguous** — enough headroom that eviction barely fired) | **Coherent, on-topic** photosynthesis explanation |

The pattern: **give `h2o_budget` real headroom over `h2o_n_sink + h2o_grace`
plus how much of the prompt/generation actually needs to be live at once.**
When there's enough room that eviction rarely or never fires (the
`grace=32, budget=128` row), output is coherent. When budget is so tight
that eviction must skip every other position just to keep the grace window
open, local coherence breaks down regardless of whether the freeze mechanism
itself is fixed — a token's neighbors matter for coherent local generation,
not just "is it in the cache at all."

**Practical guidance:** `h2o_grace` (default `16`) fixes the specific,
provable bug (permanent freeze on the first `budget` tokens). It does not
turn H2O into a cache that works well at very tight budgets — that was
never really about the freeze alone. Size `h2o_budget` generously if you
want coherent long-form generation; treat H2O at very tight budgets as still
experimental.

### Long-context testing: the freeze holds at scale, plus a scalability bug (now fixed)

*(Historical: results below predate the `h2o_grace` fix and use `h2o_grace=0`
implicitly, since the field didn't exist yet when this section was written.
Kept for the scalability-bug findings, which are independent of grace.)*

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

**With `h2o_grace` at its default (`16`) and `h2o_budget` sized with real
headroom** (see [grace-period testing](#grace-period-testing-fixes-the-freeze-mechanism-coherence-still-needs-budget-headroom) —
budget comfortably above `h2o_n_sink + h2o_grace` plus expected live context),
H2O-adapted is a **budget-bounded cache that improves over recency-only
eviction** (StreamingLLM) by using attention signal rather than position —
heavy-hitter tokens (those consistently attended to) survive; recency is not
the only criterion for retention.

**At very tight budgets**, treat it as still experimental: the permanent
freeze is fixed, but coherent output at aggressive compression ratios is not
guaranteed — see the gapped-window failure mode above.

| Scenario | Recommended method |
|----------|-------------------|
| Compress all tokens uniformly | KIVI-2bit |
| Hard cap on tokens, evict at prefill only | SnapKV-adapted |
| Constant-memory, position-based eviction | StreamingLLM-adapted |
| Constant-memory, importance-based eviction (continuous), budget has headroom | H2O-adapted |
| Constant-memory, importance-based eviction at very tight budgets | Still experimental — see grace-period findings above |
| Recover quality from aggressive quantization | GEAR |

**See also:** [CaM-adapted](./cam) makes the same eviction choice as H2O but
**merges** the loser into a similar survivor instead of dropping it, recovering a
share of the lost mass at high compression. `cam_merge="drop"` is exactly H2O.

See also: [L2Norm](../algorithms/knorm) — the same keep/evict machinery with an *intrinsic* scorer (key L2 norm, computed once at insertion): no per-step softmax over the cache, and path-independent kept sets, at the price of trusting the paper's low-norm ⇒ high-attention finding rather than reacting to the query stream. And [Q-Filters](../algorithms/qfilters) — the same machinery again with a *projection* scorer (a frozen per-head key-SVD direction): also no per-step softmax, but path-dependent and sign-ambiguous, since the direction is estimated from keys rather than the paper's queries. And [Keyformer](../algorithms/keyformer) — H2O's *exact* accumulator plus one ingredient: **Gumbel noise** on the eviction logits that rescues "late riser" tokens greedy accumulation would prune too early. `keyformer_tau=0` collapses back onto this H2O cache bit-for-bit. And [MorphKV](../algorithms/morphkv) — the antidote to this cache's *early-token bias*: rather than accumulating attention forever, it ranks stored tokens by a **sliding window of recent** attention, so a constant-size cache re-targets toward the current topic instead of clinging to stale early heavy hitters. And [KVzip](../algorithms/kvzip) — a different antidote to the same bias: rather than accumulating query attention, it ranks stored tokens by **reconstruction reliance** (how much the model relies on a KV pair to reconstruct its own context), a *query-agnostic* profile that keeps what any future query is likely to need instead of what happened to be a heavy hitter early. And [CurDKV](../algorithms/curdkv) — the antidote to this cache's *key-only blindness*: every scorer above (this one included) ranks a token using only its key side, so a token whose key looks important but whose value contributes nothing to the output is indistinguishable from one that truly matters. CurDKV derives a **leverage score** from the joint key-and-value structure instead, so two tokens with identical keys but divergent values receive different scores — a distinction this H2O cache structurally cannot make.
