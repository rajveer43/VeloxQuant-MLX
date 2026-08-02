# ChunkKV — Chunk-Level (Semantic-Block) Eviction

**Method id:** `chunkkv` · **New in 0.25.0** · *Inspired by* [ChunkKV (arXiv:2502.00299)](https://arxiv.org/abs/2502.00299)
(Liu et al., 2025) — **ChunkKV-adapted (VeloxQuant-MLX implementation)**, not a
faithful port.

ChunkKV-adapted is the library's **seventh eviction configuration** and the first
that evicts at **chunk** rather than **token** granularity. Every other eviction
method — [SnapKV](./snapkv.md), [StreamingLLM](./streaming_llm.md), [H2O](./h2o.md),
[TOVA](./tova.md), [PyramidKV](./pyramidkv.md), [SqueezeAttention](./squeeze.md) —
scores and drops **individual tokens**. ChunkKV partitions the sequence into
contiguous **chunks** of `chunk_size` tokens and keeps or drops each chunk *as a
whole*, so surviving context stays locally coherent. When `chunk_size = 1` it
reduces **bit-for-bit** to H2O-adapted.

## Why chunks instead of tokens

A token is not a self-contained unit of meaning. Token-level eviction ranks every
position independently and keeps the top-scoring ones, which can punch holes
through a clause, a variable definition, or a table row whose value is
*collective* — the pieces matter together or not at all. ChunkKV keeps
contiguous spans intact: it ranks **chunks** by a pooled importance signal and
retains whole chunks, trading a little scoring granularity for local coherence.

| Eviction axis | Granularity | When it fires | Score signal | Budget |
|---|---|---|---|---|
| SnapKV-adapted | Token | Once at prefill end | Key-as-query attention proxy | Uniform |
| StreamingLLM-adapted | Token | Every token | Position (recency + sink) | Uniform |
| H2O-adapted | Token | Every token (over budget) | Cumulative attention mass | Uniform |
| TOVA-adapted | Token | Every token (over budget) | Current-step attention weight | Uniform |
| PyramidKV-adapted | Token | Every token (over budget) | Cumulative attention mass | Per-layer fixed pyramid |
| SqueezeAttention-adapted | Token | Every token (over budget) | Cumulative attention mass | Per-layer data-driven |
| **ChunkKV-adapted** | **Chunk** | Every token (over budget) | Pooled attention-mass / key-norm | Uniform |

## The chunk-importance proxy

At the cache-wrapper level the true attention distribution is not visible, so
ChunkKV pools an existing **per-token** proxy into a **per-chunk** score (the mean
over the chunk's tokens):

- `chunkkv_score="attn_mass"` (default) — each token's cumulative attention mass
  under H2O's key-as-query scorer, mean-pooled over the chunk. This is the same
  signal H2O ranks tokens by; ChunkKV just ranks *chunks* of it.
- `chunkkv_score="key_norm"` — each token's key L2 norm (a magnitude-outlier
  proxy for salient tokens), mean-pooled over the chunk. Calibration-free and
  cheaper (no accumulation), but a coarser importance signal.

Sinks (the first `chunkkv_n_sink` positions) are always kept and never grouped
into an evictable chunk. Because chunks are kept whole, the number of retained
tokens is the largest chunk-aligned count that does not exceed the budget — it may
land a few tokens *below* budget when `budget − n_sink` is not a multiple of
`chunk_size`, which is why compression can edge slightly above the token-level
baseline at the same budget.

## Usage

ChunkKV needs **no coordinator** — every layer resolves its own chunks
independently — so the standard single-config path works, whether you build one
cache or one per layer via `for_model`:

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="chunkkv",
    head_dim=128,
    chunkkv_budget=512,  # max tokens kept per layer (sinks included)
    chunkkv_chunk_size=8,  # eviction granularity C; 1 == H2O bit-for-bit
    chunkkv_n_sink=4,  # initial positions never evicted (attention sinks)
    chunkkv_score="attn_mass",  # "attn_mass" (H2O scorer) | "key_norm"
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunkkv_budget` | `512` | Maximum tokens kept per layer (sinks included). Whole-chunk retention may land a few tokens below this. |
| `chunkkv_chunk_size` | `8` | Eviction granularity `C`. `1` reduces bit-for-bit to H2O-adapted; larger keeps bigger contiguous blocks and runs fewer eviction passes. |
| `chunkkv_n_sink` | `4` | Initial positions always retained (attention sinks); never grouped into an evictable chunk. |
| `chunkkv_score` | `"attn_mass"` | Chunk-importance proxy. `"attn_mass"` = mean-pooled H2O cumulative attention mass; `"key_norm"` = mean-pooled key L2 norm. |

## How it works

Eviction reuses H2O's per-token machinery and adds one thing: the **unit of
eviction** is a chunk, not a token.

1. **Score update.** In `attn_mass` mode the incoming key is a proxy query and
   `scores += softmax(K_stored @ k_i / sqrt(D))` accumulates per-token importance,
   exactly as in H2O. In `key_norm` mode each token's score is fixed at its key
   norm on insertion.
2. **Append + evict.** While the cache exceeds `budget`, the non-sink tail is
   partitioned into contiguous chunks of `chunk_size`, each chunk is scored by the
   **mean** of its tokens, and the lowest-scoring whole chunk is dropped. Dropping
   a whole chunk can take the count below budget; the loop stops as soon as the
   cache fits.
3. **Head alignment.** Different heads can settle at slightly different
   chunk-aligned counts; the wrapper trims every head to the common minimum
   (`chunkkv_trim_to`, keeping sinks + the most recent tail) so the emitted tensor
   is rectangular `[B, H, n_kept, D]`. At `chunk_size = 1` all heads already hold
   exactly `budget`, so no trimming occurs.

No `.bits` attribute — stored K/V remain fp16. Each cache exposes `layer_budget`,
`chunk_size`, `compression_ratio`, `tokens_seen`, and `tokens_kept`.

## Relationship to H2O

ChunkKV **is** H2O with a chunk-granular eviction unit. At `chunk_size = 1` every
chunk is a single token, mean-pooling is the identity, and "evict the lowest-mean
chunk once over budget" is exactly "evict the lowest-score token once over budget"
— so the two are bit-for-bit identical, asserted by a dedicated equivalence test.
This is the analogue of "`strength = 0` == H2O" ([SqueezeAttention](./squeeze.md))
and "flat pyramid == H2O" ([PyramidKV](./pyramidkv.md)): a token-granularity knob
whose zero setting recovers the baseline.

## Proxy limitation

The paper ranks chunks by **observed attention over the chunk** and adds a
layer-wise index-reuse trick (one layer's kept-chunk indices seed the next). We
use a mean-pooled per-token proxy for chunk importance and resolve each layer
independently (no index reuse). The "keep whole coherent spans" mechanism is
preserved; the exact importance signal is a cache-observable proxy, not read from
real attention. Eviction within a chunk uses the same key-as-query proxy as
H2O-adapted, there is no RoPE position-ID remapping after eviction, and the budget
is uniform across heads within a layer.

Documented as "ChunkKV-adapted (pooled-score proxy, key-as-query proxy, no index
reuse)" throughout — never claimed as a faithful port.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/quantizers/test_chunkkv.py` (19 tests) and
`veloxquant_mlx/tests/cache/test_chunkkv_cache.py` (14 tests):

- **Partitioning:** contiguous, gap-free coverage of the non-sink tail; ragged
  final chunk; `chunk_size=1` is per-token; sinks-exceed-length edge case;
  rejects `chunk_size < 1`
- **Pooling + keep-mask:** chunk scores are per-chunk means; the keep-mask is
  chunk-aligned, always keeps sinks, and never exceeds budget
- **Eviction:** budget never exceeded across a stress test; survivors are whole
  chunks (no partial chunk retained); sinks always present; both score modes run;
  byte accounting; determinism (no RNG)
- **Cache:** budget enforced; chunk-aligned survivors; sink preservation; correct
  output shapes across batch/heads; `key_norm` mode; prefill-then-decode;
  factory + `for_model` return `ChunkKVCache` per layer
- **`chunk_size=1` == H2O:** identical kept keys **and** values versus
  `H2OKVCache` at the same budget, at both the primitive and cache level

The offline harness in `benchmark_scripts/benchmark_chunkkv.py` sweeps
`(seq_len, budget, chunk_size, score_mode)` on synthetic fp16 K/V and compares
each config to a token-level H2O baseline at the same budget. Results are committed
in `figures/chunkkv/results.json` (run on Apple Silicon). The
**measured** facts:

- **`chunk_size=1` reproduces H2O exactly** — identical compression and survivors.
- **Larger chunks cut eviction cost sharply** while holding compression. At
  `seq_len=1024, budget=128, attn_mass`, the pure-Python eviction pass drops from
  **~5.9 s** at `C=1` to **~0.46 s** at `C=16` (~12.7× fewer/faster passes) — a
  prefill-batch worst case, not a per-decode-step cost.
- Compression can edge slightly **above** the token baseline at the same budget
  because whole-chunk retention lands a few tokens below budget.

**No model-level (perplexity/throughput) benchmark has been run.** The harness is
model-free: it measures compression, kept-token count, and eviction latency, plus
a survivor-contiguity diagnostic. On a proxy scorer the token-level baseline
already tends to keep contiguous survivors, so the contiguity *gain* is near zero
here — **ChunkKV's real semantic-coherence advantage is a property of true
attention on real prompts and is not claimed from this synthetic harness.**

## When to use it

ChunkKV-adapted is best when you want H2O-style importance eviction but care about
keeping **contiguous spans** intact — long-context tasks where local structure
(code blocks, list items, retrieved passages) is worth more whole than shredded.
Set `chunk_size=1` to fall back to plain H2O; raise it to trade scoring
granularity for coherence and cheaper eviction.

| Scenario | Recommended method |
|----------|-------------------|
| Compress all tokens uniformly | KIVI-2bit |
| Hard cap on tokens, evict at prefill only | SnapKV-adapted |
| Constant-memory, position-based eviction | StreamingLLM-adapted |
| Constant-memory, cumulative-importance eviction, uniform budget | H2O-adapted |
| Constant-memory, current-step-importance eviction (reactive) | TOVA-adapted |
| Constant-memory, importance eviction with a fixed depth-adaptive budget | PyramidKV-adapted |
| Constant-memory, importance eviction with a data-driven depth-adaptive budget | SqueezeAttention-adapted |
| **Constant-memory, importance eviction that keeps whole contiguous chunks** | **ChunkKV-adapted** |

**See also:** [CaM-adapted](./cam) varies the other end of eviction — instead of
changing *what* is evicted (a chunk vs a token), it changes what *happens* to the
loser: CaM merges it into a survivor rather than dropping it.

See also: [L2Norm](../algorithms/knorm) — note the sign inversion: ChunkKV's `key_norm` scoring treats *high*-norm chunks as important, while L2Norm implements the EMNLP 2024 finding that *low*-norm keys attract high attention.
