# Keyformer — Gumbel-Regularized Heavy-Hitter Eviction

**Method id:** `keyformer` · **New in 0.32.0** · *Inspired by* ["Keyformer: KV
Cache Reduction through Key Tokens Selection for Efficient Generative
Inference" (Adnan et al., MLSys 2024,
arXiv:2403.09054)](https://arxiv.org/abs/2403.09054) — **Keyformer-adapted
(VeloxQuant-MLX implementation)**. The estimator is deliberately changed from
the paper's (see [Adaptation notes](#adaptation-notes)).

The paper's contribution is a **regularizer, not a new importance signal**.
Naively evicting by an accumulated attention score is unstable: a token that
reads low *early* — before the queries that will attend to it arrive — gets
pruned and can never recover, even if it would have become a heavy hitter.
Keyformer adds **Gumbel noise** to the eviction logits so borderline tokens are
not deterministically doomed on a single low reading (a "late riser").

## Where it sits — the proxy-attention scorer family

Keyformer joins the repo's largest eviction family. Structurally it *is* the
[H2O](../algorithms/h2o) pair — additive proxy-attention accumulation with a
protected-sink top-budget eviction — with **one** new ingredient: the Gumbel
term on the eviction ranking.

| Scorer class | Signal | Methods |
|---|---|---|
| Attention / proxy | softmax weights (true or key-as-query proxy) | [SnapKV](../algorithms/snapkv) · [H2O](../algorithms/h2o) · [TOVA](../algorithms/tova) · [PyramidKV](../algorithms/pyramidkv) · [SqueezeAttention](../algorithms/squeeze) · [ChunkKV](../algorithms/chunkkv) · [CaM](../algorithms/cam) · **Keyformer** |
| Structural | position only (sinks, recency) | [StreamingLLM](../algorithms/streaming_llm) · sink · sliding-window |
| Intrinsic | the stored key itself (L2 norm) | [L2Norm](../algorithms/knorm) |
| Projection | key's projection onto a frozen per-head direction | [Q-Filters](../algorithms/qfilters) |

### `keyformer_tau = 0` **is** H2O-adapted

Setting the temperature to zero removes the noise and this cache collapses,
bit-for-bit, onto [H2O](../algorithms/h2o). That is the honest ablation: the
*only* thing Keyformer adds over H2O is the Gumbel regularizer, and you can
turn it off to see exactly what it buys. A dedicated test asserts the
`tau=0` kept set equals H2O's, and the benchmark prints an `h2o` column as a
cross-check.

## :warning: The honesty crux — read this first

1. **Proxy query.** Like [H2O](../algorithms/h2o)/[SnapKV](../algorithms/snapkv),
   a cache never sees the true query vector, so the incoming **key** is used as
   a proxy query to estimate the attention each stored key receives. The paper
   accumulates the model's real attention logits. This is a documented
   substitution, not the paper's math.
2. **Frozen per-position noise, not annealed sampling.** The paper redraws
   Gumbel noise and anneals a temperature across the full generation. A cache
   processes blocks with no global step counter it can trust, so we draw **one
   deterministic Gumbel value per token position** (seeded from a fixed base
   seed + a per-head running position) and freeze it. `tau` scales that frozen
   noise. This preserves the mechanism's intent — a borderline token is not
   doomed by one low reading — while staying reproducible and order-diagnosable.
   It is **not** the paper's annealing schedule, and we do not claim it is.
3. **Not validated on a trained model.** The regularizer's benefit is measured
   only under constructed "late-riser" geometry, with a stable-importance
   control where it has nothing to rescue.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="keyformer",
    head_dim=128,
    keyformer_budget=512,  # max tokens kept (incl. sinks)
    keyformer_n_sink=4,  # leading positions never evicted
    keyformer_recent=0,  # trailing protected window (extension, off)
    keyformer_tau=1.0,  # Gumbel temperature; 0 = H2O-adapted (ablation)
    keyformer_seed=0,  # base seed for the frozen per-position noise
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

Single-layer, no coordinator — the default `for_model` path returns one
`KeyformerKVCache` per attention layer.

## How it works

Per incoming token (prefill and decode alike):

1. Accumulate the new key's proxy-attention mass (softmax of key-as-query over
   the stored keys) into the per-token cumulative score — [H2O](../algorithms/h2o)'s
   additive rule.
2. Append the new token with cumulative score `0` and a **frozen per-position
   Gumbel draw** seeded by the head's running position.
3. If over `keyformer_budget`: evict the non-protected token with the lowest
   `score + tau · gumbel`. The Gumbel term is the whole mechanism; at `tau = 0`
   this is exactly H2O's argmin on the raw score. Sinks (first
   `keyformer_n_sink`) and the optional trailing `keyformer_recent` window are
   forced to survive.

The Gumbel noise perturbs only the **eviction decision** — the stored
cumulative mass itself stays clean, so the noise never compounds across steps.

Byte accounting mirrors H2O's — `keyformer_kept_bytes`, `full_seq_bytes`,
`compression_ratio`, `tokens_seen`, `tokens_kept`. The transient float32
score/gumbel bookkeeping (one value per kept token) is not counted as cache
payload, same as H2O's scores.

## Adaptation notes

**What we do NOT implement:**
- **The paper's annealed, redrawn Gumbel schedule** — replaced by frozen
  per-position noise (crux 2). This is the mechanism deviation, not a footnote.
- The model's real attention logits — replaced by the key-as-query proxy
  (crux 1), same approximation as H2O/SnapKV-adapted.
- RoPE position-ID remapping after eviction (same as every eviction method here).
- Per-head budgets / temperatures (uniform across heads).

**Extensions beyond the paper (off by default):**
- `keyformer_recent` — protects the most recent tokens StreamingLLM-style.
- `keyformer_seed` — makes the frozen noise reproducible and per-head
  independent.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/quantizers/test_keyformer.py` (17 tests) and
`veloxquant_mlx/tests/cache/test_keyformer_cache.py` (12 tests):

- **`tau = 0` collapses onto H2O-adapted** — the kept set equals H2O's,
  bit-for-bit, over an identical stream; and with no noise the kept set is
  **seed-invariant**.
- Budget is never exceeded, token-by-token or in a prefill block, across
  batch/head shapes.
- Sinks and the `recent` window survive heavy eviction; `n_sink + recent >=
  budget` and negative `tau` raise at build time.
- The frozen Gumbel draw is deterministic per `(seed, position)` and the full
  run is reproducible.
- **Late-riser mechanism:** with a planted token that reads low early but
  aligns with a *later* burst, the token's survival rate across noise-seeds is
  **higher with the Gumbel term on than off** — a statistical mechanism claim,
  not a per-seed guarantee.

The offline harness in `benchmark_scripts/benchmark_keyformer.py` (results in
`figures/keyformer/results.json`) sweeps sequence length
(256/512) and budget (32/64) across `tau ∈ {0, 2, 6}`, an H2O cross-check, and
random eviction, under two data regimes:

- **`late_riser` geometry:** greedy `tau=0` (== H2O-adapted) evicts the planted
  late-riser **100% of the time** — exactly the failure the paper describes —
  while the Gumbel term (`tau=6`) rescues it a **large fraction** of the time
  (survival 0.00 → ~0.75). This survival rate is the mechanism's clean, direct
  observable.
- **`stable` geometry** (heavy hitters are heavy from token 0): greedy already
  keeps them, so the noise has nothing to rescue and is neutral-to-slightly
  worse. Reporting this control is the point — the regularizer is not a free
  win.

The downstream probe-attention **perturbation** is a noisier, regime-dependent
secondary effect that does **not** uniformly improve; it is reported as-is
rather than cherry-picked. **No model-level benchmark has been run** — these
are offline-synthetic survival-rate, output-perturbation and byte-accounting
numbers, not perplexity or throughput on a real model.

## When to use it

Keyformer is [H2O](../algorithms/h2o) with a safety net for late-rising tokens.
If your workload has tokens that only become important well after they enter
the cache — retrieval-style prompts where a late query re-activates early
context — the Gumbel regularizer can keep them alive where greedy accumulation
would have dropped them. If importance is stable (heavy hitters are heavy from
the start), plain [H2O](../algorithms/h2o) is simpler and the noise buys
nothing — or just set `keyformer_tau=0` and you are running H2O.

| Method | Score | Late-riser protection | Path-independent |
|--------|-------|-----------------------|------------------|
| [H2O](../algorithms/h2o) | cumulative proxy-attention mass | none (greedy) | no |
| **Keyformer** | proxy-attention mass **+ Gumbel noise** | **yes (regularizer)** | no |
| [Q-Filters](../algorithms/qfilters) | projection onto frozen key-SVD direction | n/a | no |
