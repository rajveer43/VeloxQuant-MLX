# ChunkKV — Chunk-Level (Semantic-Block) Eviction

**Method id:** `chunkkv` · **New in 0.25.0** (index reuse in a later release) · *Inspired by* [ChunkKV (arXiv:2502.00299)](https://arxiv.org/abs/2502.00299)
(Liu et al., NeurIPS 2025) — **ChunkKV-adapted (VeloxQuant-MLX implementation)**, not a
faithful port.

## Should I use this?

Use ChunkKV when you're evicting KV cache under memory pressure and care about
keeping **contiguous spans of text intact** — code blocks, table rows, list
items, retrieved passages — rather than scattering survivors across the
sequence. Every other eviction method in this library scores and drops
**individual tokens**, which can leave a subject without its object or a
function signature without its body. ChunkKV keeps or drops whole **chunks** of
`chunk_size` tokens instead, trading a little scoring precision for local
coherence.

If you don't need chunk-level coherence, plain [H2O](./h2o.md) is simpler and
`chunk_size=1` makes ChunkKV reduce to it bit-for-bit anyway — so there's no
downside to trying ChunkKV first and dialing `chunk_size` down if it doesn't
help.

## Quick start

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

No coordinator is required for the default path — every layer resolves its own
chunks independently. Set `chunkkv_reuse_layers` (see below) if you also want
the paper's cross-layer speed-up.

## Checking what ChunkKV kept

Every `ChunkKVCache` exposes:

- `layer_budget` — this layer's configured token budget
- `chunk_size` — the eviction granularity in effect
- `tokens_kept` / `tokens_seen` — current retained count vs. total seen
- `compression_ratio` — `full_seq_bytes / chunkkv_kept_bytes`
- `is_index_reuse_leader` — `True` unless this layer is a follower reusing another layer's indices

```python
cache = caches[0]
print(cache.tokens_kept, "/", cache.tokens_seen, "kept")
print(f"{cache.compression_ratio:.1f}x compression")
```

## Tuning

| Parameter | Default | What it controls |
|-----------|---------|-------------------|
| `chunkkv_budget` | `512` | Max tokens kept per layer (sinks included). Whole-chunk retention may land a few tokens below this. |
| `chunkkv_chunk_size` | `8` | Span width kept/dropped as a unit. `1` = plain H2O; larger = more coherence per surviving span, coarser scoring, fewer eviction passes. Paper's sweet spot is `5–20`; default `8` sits in that range. |
| `chunkkv_n_sink` | `4` | Leading positions always retained (attention sinks); never grouped into an evictable chunk. |
| `chunkkv_score` | `"attn_mass"` | Chunk-importance proxy. `"attn_mass"` = pooled H2O cumulative attention mass; `"key_norm"` = pooled key L2 norm (cheaper, coarser). |
| `chunkkv_reuse_layers` | `1` | Layer-wise index reuse block size (paper's Algorithm 2). `1` disables it (every layer evicts independently); `N > 1` groups layers into blocks of `N` where only the first layer evicts and the rest reuse its kept-token positions, trading a small accuracy cost for less eviction work. Requires `KVCacheBuilder.for_model`. |

**Choosing `chunk_size`:** start at `8`; go lower (`3-5`) if compression must
stay tight even at small budgets, higher (`16-20`) if you want cheaper eviction
passes and can tolerate coarser importance ranking. The paper found `5-20`
roughly equally good and performance dropping off outside that range.

**Choosing `chunkkv_reuse_layers`:** start at `1` (off). If you're prefill-bound
and layer count is a meaningful chunk of latency, try `2`-`4` — the paper
reports up to 20.7% latency reduction / 26.5% throughput improvement at
`reuse_layers=2`, with under 0.6% LongBench score loss. It only helps because
ChunkKV's kept-chunk indices are unusually similar between adjacent layers
(measured, not assumed — see "How it works" below); applying the same idea to a
token-level method wouldn't transfer, since token-level indices vary far more
layer to layer.

## How it works

**Chunk eviction (every layer, every step).** The proxy score
(`attn_mass`: same key-as-query cumulative attention mass H2O uses;
`key_norm`: the token's key L2 norm, fixed on insertion) accumulates per token as
usual. Sinks aside, whenever the cache is over budget, the non-sink tail is
partitioned into contiguous chunks of `chunk_size`, each chunk's score is the
**mean** of its tokens' scores, and the lowest-scoring whole chunk is dropped —
not the lowest-scoring token. Dropping a whole chunk can take the count below
budget; eviction stops as soon as the cache fits. Different heads can settle at
slightly different chunk-aligned lengths, so the cache wrapper trims every head
down to the shortest one (keeping sinks + the most recent tail) before returning
a rectangular tensor.

**Layer-wise index reuse (`chunkkv_reuse_layers > 1`).** Layers are grouped into
consecutive blocks. The first ("leader") layer in each block runs the eviction
above as normal and publishes exactly which token positions it kept, every
step. The other ("follower") layers in the block skip scoring and eviction
entirely — they just apply the leader's kept-position list to their own K/V.
This only works because ChunkKV's chunk-level survivors are unusually stable
across depth (see below); reusing indices is essentially free extra compression
throughput, not an approximation layered on top of a different signal.

## Why layer-wise reuse works for chunks but not tokens

The paper measured Jaccard similarity of kept indices between adjacent layers
and found ChunkKV's chunks agree far more than token-level methods' individual
tokens do:

| Model | H2O | SnapKV | ChunkKV |
|---|---|---|---|
| LLaMA-3-8B | 25.3% | 28.0% | **57.7%** |
| Qwen2-7B | 14.9% | 16.5% | **44.3%** |
| Mistral-7B | 15.2% | 15.8% | **52.2%** |

Chunk-level decisions are smoother across depth because a chunk's mean score
averages out per-token noise that makes individual token rankings volatile
layer to layer. That stability is what makes reuse a safe trade — a follower
layer reusing its leader's indices is reusing a decision that would likely have
come out nearly the same anyway.

## Relationship to H2O

ChunkKV **is** H2O with a chunk-granular eviction unit. At `chunk_size = 1`
every chunk is a single token, mean-pooling is the identity, and "evict the
lowest-mean chunk once over budget" is exactly "evict the lowest-score token
once over budget" — so the two are bit-for-bit identical, asserted by a
dedicated equivalence test. This is the analogue of "`strength = 0` == H2O"
([SqueezeAttention](./squeeze.md)) and "flat pyramid == H2O"
([PyramidKV](./pyramidkv.md)): a token-granularity knob whose zero setting
recovers the baseline.

| Eviction axis | Granularity | Score signal | Budget |
|---|---|---|---|
| SnapKV-adapted | Token | Key-as-query attention proxy | Uniform |
| StreamingLLM-adapted | Token | Position (recency + sink) | Uniform |
| H2O-adapted | Token | Cumulative attention mass | Uniform |
| TOVA-adapted | Token | Current-step attention weight | Uniform |
| PyramidKV-adapted | Token | Cumulative attention mass | Per-layer fixed pyramid |
| SqueezeAttention-adapted | Token | Cumulative attention mass | Per-layer data-driven |
| **ChunkKV-adapted** | **Chunk** | Pooled attention-mass / key-norm | Uniform |

## When to use it

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
| **Same, but also want cross-layer eviction-cost savings** | **ChunkKV-adapted with `chunkkv_reuse_layers > 1`** |

**See also:** [CaM-adapted](./cam) varies the other end of eviction — instead of
changing *what* is evicted (a chunk vs a token), it changes what *happens* to the
loser: CaM merges it into a survivor rather than dropping it.

See also: [L2Norm](../algorithms/knorm) — note the sign inversion: ChunkKV's `key_norm` scoring treats *high*-norm chunks as important, while L2Norm implements the EMNLP 2024 finding that *low*-norm keys attract high attention.

## Fidelity to the paper

ChunkKV-adapted preserves the paper's core mechanisms — chunk-level scoring and
eviction, plus (now) layer-wise index reuse — but takes several documented
shortcuts to work as a cache-level wrapper without visibility into real
attention or the model's forward pass:

- **Streaming, not one-shot prefill compression.** The paper computes chunk
  importance once, from a fixed observation window at the end of prefill, and
  compresses in a single pass. This implementation evicts continuously —
  whenever the cache exceeds budget, at any step — because the wrapper has no
  hook into a distinct "prefill boundary."
- **Pooled per-token proxy, not real attention-over-chunk.** The paper ranks
  chunks by attention scores actually observed over each chunk. Cache wrappers
  don't see the true attention matrix, so this uses a mean-pooled per-token
  proxy (H2O's cumulative attention mass, or key norm) instead — same
  chunk-granular decision, different signal.
- **Key-as-query proxy**, shared with H2O-adapted / SnapKV-adapted: the
  incoming key stands in for the true query when scoring, since the query
  vector isn't visible at the cache level either.
- **No RoPE position-ID remapping** after eviction, and **uniform budget across
  heads** within a layer.
- **Index reuse is exact reuse, not approximate.** The implementation matches
  the paper's Algorithm 2 mechanism directly (leader publishes indices,
  followers apply them) — this is the one piece of the paper implemented
  without adaptation, verified by a bit-for-bit equivalence test between a
  leader's and its follower's output.

None of this is claimed as a faithful port — see the module docstrings in
`quantizers/chunkkv.py` and `cache/chunkkv_cache.py` for the complete list.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/quantizers/test_chunkkv.py` (25 tests),
`veloxquant_mlx/tests/cache/test_chunkkv_cache.py` (21 tests), and
`veloxquant_mlx/tests/cache/test_chunkkv_coordinator.py` (7 tests):

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
- **Index reuse:** a follower fed the leader's recorded kept-positions ends up
  bit-identical to the leader (primitive level); a live follower cache mirrors
  its leader cache's K/V exactly across prefill **and** multiple decode steps
  (cache level); `for_model` assigns leader/follower roles in contiguous
  `chunkkv_reuse_layers`-sized blocks and every follower's output matches its
  leader's; `chunkkv_reuse_layers=1` makes every layer a leader (reuse off)

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

**No model-level (perplexity/throughput) benchmark has been run** for either the
base chunking mechanism or `chunkkv_reuse_layers`. The harness is model-free: it
measures compression, kept-token count, and eviction latency, plus a
survivor-contiguity diagnostic. On a proxy scorer the token-level baseline
already tends to keep contiguous survivors, so the contiguity *gain* is near zero
here — **ChunkKV's real semantic-coherence advantage is a property of true
attention on real prompts and is not claimed from this synthetic harness.** The
paper's 20.7%/26.5% latency/throughput numbers for index reuse are the paper's
own measurements on real models, not reproduced here; what's verified here is
that the reuse mechanism itself is correct (bit-identical to the leader), not
its speed-up on this codebase's synthetic harness.
