# SnapKV — Prefill Observation-Window Token Eviction

**Method id:** `snapkv` · **New in 0.19.0** · *Inspired by* [SnapKV (arXiv:2404.14469)](https://arxiv.org/abs/2404.14469)
(Yuan et al., ICLR 2025) — **SnapKV-adapted (VeloxQuant-MLX implementation)**,
not a faithful port.

SnapKV-adapted is the repo's first **token eviction** method and the first where
the paper's actual signal — attention scores from an observation window — is
computable at the cache-wrapper level without model interception. Every other
method in the suite compresses all tokens to fewer bits; SnapKV-adapted instead
drops low-importance token positions entirely and retains only a `snap_budget`-sized
subset in fp16.

| Method | What's stored | Mechanism | Composable? |
|--------|--------------|-----------|-------------|
| KIVI-2bit | All tokens, 2 bits/element | Quantization | — |
| ZipCache-adapted | All tokens, mixed bits | Per-token bit routing | — |
| SnapKV-adapted | Budget tokens, fp16 | Token eviction | ✅ stack any quantizer on kept tokens |

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="snapkv",
    head_dim=128,
    snap_budget=512,  # max tokens retained after prefill eviction
    snap_obs_window=32,  # trailing key rows used as proxy queries
    snap_n_sink=4,  # initial positions always kept (attention sinks)
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `snap_budget` | `512` | Max token positions retained after prefill. Positions beyond this budget are dropped permanently. |
| `snap_obs_window` | `32` | Number of trailing key rows used as proxy queries for scoring all prefix positions. |
| `snap_n_sink` | `4` | Number of initial token positions always retained (attention-sink tokens). |

## How it works

Per head, per prefill call (`S > 1` tokens in one `update_and_fetch`):

1. **Observation-window scoring.** The last `snap_obs_window` key rows act as proxy
   queries and attend to all `S` prefix key rows via scaled dot-product softmax:
   `attn = softmax(K[-w:] @ K^T / sqrt(D))`. Mean-pooling over the window gives
   `[S]` importance scores per token.
2. **Sink guarantee.** The first `snap_n_sink` positions are always retained
   (attention-sink tokens consistently receive high attention weight).
3. **Top-k selection.** From the remaining positions, the top `(snap_budget −
   snap_n_sink)` by score are selected. The union is sorted to preserve original
   order.
4. **Eviction.** Only the selected `[n_kept, D]` fp16 K/V rows are stored.
   All evicted positions are permanently dropped.

During **decode** (`S == 1` per call): new tokens are always appended — never
evicted. The budget applies to the prefill history only.

No `.bits` attribute — stored K/V remain in fp16. The `eviction_ratio` and
`keep_rate` properties report the storage accounting.

## Proxy limitation

The paper uses the final prompt *query* vectors (from the model's attention layer)
for the observation window. A cache wrapper only sees K and V at `update_and_fetch`
time — the Q matrix is not available. We substitute the last `snap_obs_window` *key*
vectors as proxy queries.

This is a stronger proxy than key-norm-only methods (KIVI-Sink, AdaKV-proxy, Kitty,
ZipCache-adapted): we compute the actual attention distribution, just from K rather
than true Q. Key and query spaces are correlated (both projected from the same residual
stream), but the proxy is still an approximation.

The paper's max-pool smoothing step (`kernel_size > 1`) is not implemented. Documented
as "SnapKV-adapted (key-as-query proxy)" throughout — never claimed as a faithful port.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_snapkv_cache.py` (12 tests) and
`veloxquant_mlx/tests/quantizers/test_snapkv.py` (18 tests):

- `obs_window_attention_scores` output shape `[S]`, dtype fp32, values ∈ [0, 1]
- `obs_window` clamped to `S` — no index error when `obs_window > S`
- `snap_select_indices` returns exactly `min(budget, S)` indices, sorted ascending
- First `n_sink` positions always in kept set (verified with zero scores — sinks win)
- High-score tokens are preferentially selected (verified on synthetic known-top data)
- `snapkv_compress` output shape `[min(budget, S), D]`, dtype fp16
- `budget >= S` keeps all tokens (no eviction on short sequences)
- `eviction_ratio > 1` after prefill with budget < S
- `keep_rate` in `(0, 1]` after realistic-budget prefill
- Decode accumulation: seq dim grows by 1 per single-token call
- Determinism; `for_model` config propagation (`_budget`, `_obs_window`, `_n_sink`)

The offline harness in `benchmark_scripts/benchmark_snapkv.py` measures attention
coverage (fraction of total obs-window attention mass in the kept set) vs a
random-budget baseline — **synthetic, not model-level.**

**No model-level benchmark has been run.** Until `results_snapkv.json` is committed
with hardware numbers, no throughput or perplexity figures are claimed.

## When to use it

SnapKV-adapted is a **token-count budget** method: use it when you need a hard cap
on the number of token positions in the cache (e.g., fixed RAM budget for very long
prompts where quantization alone is not sufficient). The eviction and compression axes
are orthogonal — stack a quantizer cache on the kept tokens for combined eviction +
compression.

| Scenario | Recommended method |
|----------|-------------------|
| Compress all tokens uniformly | KIVI-2bit |
| Protect important tokens at full precision | KIVI-Sink |
| Per-token mixed bit-width | ZipCache-adapted |
| Hard cap on token count (very long contexts) | SnapKV-adapted |
| Recover quality from aggressive quantization | GEAR |

**See also:** [ChunkKV-adapted](./chunkkv) evicts on the same importance idea but at
**chunk** rather than token granularity — it keeps whole contiguous spans instead of
the top individual positions, trading scoring resolution for local coherence.
