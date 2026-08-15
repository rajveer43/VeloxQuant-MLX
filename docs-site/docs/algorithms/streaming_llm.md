# StreamingLLM — Sink + Recency-Window Token Eviction

**Method id:** `streaming_llm` · **New in 0.20.0** · *Inspired by* [StreamingLLM (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453)
(Xiao et al., ICLR 2024) — **StreamingLLM-adapted (VeloxQuant-MLX implementation)**,
not a faithful port.

StreamingLLM-adapted is the repo's **structural positional eviction** method — tokens
are kept or dropped purely by position (first N sinks + last W recent), with no scoring,
no calibration, and no proxy signals. This is orthogonal to SnapKV-adapted (which evicts
by attention score) and to all quantization methods (which compress all tokens to fewer bits).

The cache never grows beyond `stream_n_sink + stream_window_size` positions, making
decode memory **constant** regardless of generation length.

| Method | Eviction signal | Constant memory? | Calibration |
|--------|----------------|-----------------|-------------|
| SnapKV-adapted | Prefill attention score | No (evicts once at prefill; decode grows) | None |
| StreamingLLM-adapted | Token position (sink + recency) | ✅ Yes — always bounded | None |
| KIVI-2bit | (no eviction) | No | None |

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="streaming_llm",
    head_dim=128,
    stream_n_sink=4,  # initial token positions always kept (attention sinks)
    stream_window_size=512,  # FIFO capacity for most-recent tokens
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stream_n_sink` | `4` | Number of initial token positions frozen as attention sinks. These are never evicted regardless of sequence length. |
| `stream_window_size` | `512` | FIFO capacity for recent tokens. Once the window fills, the oldest recent token is evicted for each new token. |

**Total tokens in cache at any time:** `stream_n_sink + min(n_decoded, stream_window_size)`.

## How it works

Each call to `update_and_fetch(keys, values)` — whether prefill or decode:

1. **Sink accumulation.** The first `stream_n_sink` token positions seen are frozen
   into a sink buffer. Sink tokens are never evicted.
2. **Recent FIFO.** All subsequent tokens enter a FIFO queue of capacity
   `stream_window_size`. When the FIFO exceeds capacity, the oldest token is dropped
   from the front.
3. **Concatenation.** The returned K/V tensors are `[sink_rows || recent_rows]` in
   original token order (sinks first, then recent). Shape: `[B, H, n_sink + n_recent, D]`.

Both prefill (large S) and decode (S=1) tokens are processed identically through this
logic. The constant-memory property follows: the cache size is always bounded by
`stream_n_sink + stream_window_size` regardless of how many tokens have been generated.

No `.bits` attribute — stored K/V remain in fp16. The `streaming_ratio` and
`tokens_in_window` properties report the storage accounting.

## Adaptation limitations

**No attention mask adjustment.** The paper also adjusts the attention mask so tokens
beyond the window are invisible to the query. A cache wrapper cannot inject attention
masks — all returned K/V positions will be attended to by the model. The functional
*memory budget* is still bounded; the limitation is that the model may attend to old
sink tokens at positions not matching their original RoPE embeddings in some architectures.

**No RoPE position-ID remapping.** The paper's original implementation remaps RoPE
position IDs when using positional encodings. We preserve original token positions inside
the returned K/V rows. Position-ID remapping requires model-level patching.

**Fixed sink count.** `stream_n_sink` is a fixed hyperparameter. The paper's original
finding is empirical — any token in the first ~4 positions consistently acts as a sink.

Documented as "StreamingLLM-adapted" throughout — never claimed as a faithful port.

### RoPE position semantics vs. the paper (#189)

The bullet above states the divergence; this section reviews it, as requested by
[#189](https://github.com/rajveer43/VeloxQuant-MLX/issues/189).

**What the paper does.** StreamingLLM's reference implementation assigns RoPE
position IDs *by slot in the kept cache* — the sink tokens keep positions
`0..n_sink-1`, and the recent window is renumbered `n_sink..n_sink+n_recent-1`
every time the window slides, regardless of the tokens' original position in
the sequence. Every surviving key is re-rotated each step the window advances,
since a token's slot (and therefore its assigned position) shifts as older
recent tokens fall off the front.

**What this implementation does instead.** `StreamingLLMKVCache` preserves each
surviving token's **original absolute position** and reports the true position
count via `self.offset` (`_true_offset` in `streaming_llm_cache.py`, fixed for
[#171](https://github.com/rajveer43/VeloxQuant-MLX/issues/171)). Survivors are
never re-rotated; positions become non-contiguous across the sink/recent gap
instead of being renumbered.

**Why this is not a correctness bug.** RoPE is a relative encoding:
`⟨rope(q, i), rope(k, j)⟩` depends only on `i - j`, not on `i` and `j`
individually. mlx_lm rotates the incoming query and key at `offset=cache.offset`
*before* `update_and_fetch` runs, so as long as the offset reported here is the
token's **true** absolute position, every stored key, every new key, and every
new query sit on one consistent absolute axis — the relative-distance property
that RoPE actually relies on is preserved with no re-rotation needed. This is
what `#171` fixed: before it, `self.offset` reported the *retained row count*
instead of the true position, which was a real bug (positions silently drifted
once the window saturated). Choosing to preserve absolute positions rather than
renumber is a separate, deliberate design choice from that bug fix.

**Why we chose absolute-position preservation over the paper's renumbering:**
- **Consistency.** Every eviction cache in this library (SnapKV-adapted,
  Q-Filters-adapted, KNorm/L2Norm-adapted, TOVA-adapted, H2O-adapted) preserves
  original absolute positions and reports them via the same `_true_offset`
  pattern. Renumbering StreamingLLM alone would make it the one method with
  different position semantics, complicating any cross-method comparison.
- **No re-rotation cost.** Renumbering requires re-rotating every recent-window
  key on every step the window slides (i.e., on every decode token once the
  window is full) — extra compute on the hot path this implementation avoids.
- **Cache-wrapper boundary.** `update_and_fetch` only sees K/V tensors, not the
  model's rotary embedding layer, so implementing the paper's renumbering
  faithfully would need model-level patching regardless (same limitation noted
  above for attention-mask adjustment) — it can't be done cleanly as a pure
  cache-wrapper change.

**What is NOT validated here.** Whether the paper's renumbering scheme produces
measurably different generation quality, perplexity, or long-context retrieval
behavior compared to absolute-position preservation has **not been measured**.
Doing so needs a real model forward pass (perplexity/NIAH/generation runs on
actual weights), which this development environment cannot run — see
[#187](https://github.com/rajveer43/VeloxQuant-MLX/issues/187) and
[#198](https://github.com/rajveer43/VeloxQuant-MLX/issues/198) for the general
GPU-blocked-measurement pattern this repo follows. If that gap is ever closed,
comparing both schemes empirically (not just arguing from the RoPE-relativity
identity) would be the way to fully settle this.

**Decision:** keep absolute-position preservation as the implementation's
default and documented behavior. It is mathematically sound (RoPE relativity),
consistent with every sibling cache in the library, and avoids both a
per-step re-rotation cost and a cache-wrapper-boundary limitation the paper's
scheme would hit regardless. The divergence from the paper is intentional and
stated plainly, not an oversight.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_streaming_llm_cache.py` (18 tests) and
`veloxquant_mlx/tests/quantizers/test_streaming_llm.py` (17 tests):

- `init_streaming_window` creates empty buffers with correct shapes
- Sink tokens absorb first `n_sink` positions correctly
- Sink buffer is frozen after `n_sink` fill — additional tokens go to recent window
- `tokens_seen` accumulates correctly over multiple calls
- Recent FIFO trims to `window_size` when exceeded
- `n_recent <= window_size` holds across 20 single-token decode steps
- `n_sink + n_recent <= n_sink + window_size` at all times (30-step stress test)
- `stream_get_kv` returns `[n_sink + n_recent, D]` combined tensor, dtype fp16
- Sink rows appear first in output (verified against known token values)
- `stream_fp16_bytes = (n_sink + n_recent) × D × 4` (K + V, fp16)
- `streaming_ratio == 1.0` when all tokens fit; `> 1.0` after overflow
- Large prefill (S=1000 with n_sink=4, window=8) trims to exactly 12 output positions
- `n_sink=0` edge case: all tokens go to recent window
- Determinism; `for_model` config propagation (`_n_sink`, `_window_size`)
- `cache.offset` tracks the true absolute token position (not the retained
  row count) through sustained eviction, across a single large prefill
  block, and across a prefill-then-decode mix (#189 — this cache previously
  had no regression coverage for the #171 offset fix, unlike every sibling
  eviction cache)

The offline harness in `benchmark_scripts/benchmark_streaming_llm.py` measures
streaming_ratio and ms/head across `(seq_len, window_size)` sweep on synthetic data —
**synthetic, not model-level.**

**No model-level benchmark has been run.** Until `results_streaming_llm.json` is
committed with hardware numbers, no throughput or perplexity figures are claimed.

## When to use it

StreamingLLM-adapted is a **constant-memory streaming** method: use it when you need
to generate arbitrarily long sequences at a fixed memory footprint — real-time assistants,
streaming summarization, long-running agents. The eviction is purely positional, so it
pairs naturally with quantization on the kept tokens.

| Scenario | Recommended method |
|----------|-------------------|
| Compress all tokens uniformly | KIVI-2bit |
| Hard cap on token count — evict by importance score | SnapKV-adapted |
| Hard cap on token count — constant memory, streaming generation | StreamingLLM-adapted |
| Protect high-attention tokens while compressing rest | ZipCache-adapted |
| Recover quality from aggressive quantization | GEAR |

**SnapKV-adapted vs StreamingLLM-adapted:** SnapKV evicts once at prefill by attention
score and then grows during decode. StreamingLLM evicts continuously by position and stays
constant-memory forever. If you need constant decode-phase memory, use StreamingLLM-adapted.
