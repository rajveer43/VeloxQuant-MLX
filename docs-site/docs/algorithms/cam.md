# CaM — Cache Merging (Merge Evicted Tokens Instead of Dropping)

**Method id:** `cam` · **New in 0.26.0**

Every other eviction method in this library — [SnapKV](./snapkv.md),
[StreamingLLM](./streaming_llm.md), [H2O](./h2o.md), [TOVA](./tova.md),
[PyramidKV](./pyramidkv.md), [SqueezeAttention](./squeeze.md),
[ChunkKV](./chunkkv.md) — permanently throws away the tokens it evicts. CaM
instead **merges** each evicted token into the surviving token it most
resembles, so the information is folded into a neighbour rather than lost
outright. It picks the same loser H2O would; only what happens to the loser
differs.

## Should I use this?

Use CaM when:

- You're already running (or considering) **H2O-style importance eviction**
  and want a drop-in upgrade — CaM reuses H2O's scoring and budget, adding
  only the merge step.
- You're compressing **aggressively** (small budget relative to context
  length). Merging's benefit over plain dropping grows with compression
  ratio; at low compression, dropping barely hurts and merging has little to
  gain.
- You can tolerate a small amount of extra per-token compute (a cosine
  similarity search + a blend) in exchange for better retained quality at the
  same memory budget.

Reach for something else when:

- You want the simplest possible eviction with no merge machinery — use
  [H2O](./h2o.md) directly (`cam_merge="drop"` makes CaM identical to it
  anyway).
- You need a compression ratio guarantee from a fixed schedule rather than
  importance-based eviction — see [PyramidKV](./pyramidkv.md) or
  [SqueezeAttention](./squeeze.md).

## Quick start

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="cam",
    head_dim=128,
    cam_budget=512,  # max tokens kept per layer (sinks included)
    cam_n_sink=4,  # initial positions never evicted (attention sinks)
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

CaM needs no coordinator — every layer merges independently — so this is the
whole setup. Defaults (`cam_merge="sim_weighted"`, `cam_merge_gate=True`)
match the paper's recommended configuration.

## Checking what CaM is doing

```python
cache = caches[0]
cache.merge_mode        # "sim_weighted" | "mean" | "drop"
cache.merge_gate        # whether the Eq. 14 gate is active
cache.compression_ratio  # full_seq_bytes / cam_kept_bytes, > 1 = savings
cache.tokens_kept        # tokens currently retained (diagnostic, head 0)
```

## Tuning

| Parameter | Default | What it controls |
|-----------|---------|-------------------|
| `cam_budget` | `512` | Maximum tokens kept per layer (sinks included). CaM trims to exactly this once past budget. |
| `cam_n_sink` | `4` | Initial positions always retained (attention sinks); never evicted or used as a merge target. |
| `cam_merge` | `"sim_weighted"` | Merge rule. `"sim_weighted"` = cosine-weighted blend (recommended); `"mean"` = unweighted average (ablation baseline); `"drop"` = no blend, bit-for-bit identical to H2O. |
| `cam_merge_gate` | `True` | Whether to probabilistically decide *if* a merge happens at all (paper's Eq. 14), or always merge every over-budget loser. Leave on — see [The merge gate](#the-merge-gate). |
| `cam_merge_keys` | `False` | Merge keys as well as values. Values are always merged; merging keys shifts the attention geometry (the paper treats this as optional). |

**If quality is worse than expected at a given budget:** check
`cam_merge_gate` is `True` (the default). Turning it off unconditionally
merges every loser, which the paper's own ablation shows can perform *worse*
than plain dropping — a low-signal loser merged into an unrelated survivor
perturbs that survivor more than simply evicting the loser would have.

**To fall back to plain H2O** at any point (e.g. for a baseline comparison),
set `cam_merge="drop"` — this reduces bit-for-bit to
[H2O](./h2o.md) regardless of the gate setting.

## How it works

When the cache exceeds budget, CaM picks the same loser H2O would (lowest
cumulative attention mass, sinks protected), then:

1. **The merge gate** — decide *whether* to merge at all (see below). If the
   gate says no, the loser is simply dropped, same as H2O.
2. **Find the target** — the surviving non-sink token whose key is most
   similar (cosine) to the loser's key.
3. **Blend** — `x_new = (1 - w)·x_survivor + w·x_evicted`, where `w` is a
   cosine-similarity weight (see [Fidelity to the paper](#fidelity-to-the-paper)
   for why cosine rather than the paper's attention weight). A loser that
   closely resembles its survivor is absorbed strongly; a dissimilar one
   barely perturbs it. Values are always merged; keys only if
   `cam_merge_keys=True`.
4. **Transfer mass + remove** — the survivor inherits the loser's cumulative
   score, and the loser's slot is removed, keeping the cache at exactly
   `cam_budget`.

### The merge gate

The paper does not merge every over-budget loser unconditionally — first it
samples a Bernoulli gate,

```
p = clamp(score(loser) / score(target), 0, 1)
merge ~ Bernoulli(p)
```

so the probability of merging scales with how much accumulated importance
the loser carries relative to its target. A loser with little relative mass
(the common case for a token that overflows the budget right after being
appended, before it has accumulated any attention) is usually *not* merged —
it's simply dropped, same as plain eviction. A loser with comparable or
greater mass is merged with high probability.

The paper's own ablation (its Table 2, "w.o. Merge Mask") shows this gate is
load-bearing: removing it and merging unconditionally performs *worse* than
plain dropping, because folding a low-signal loser into an unrelated
survivor can perturb that survivor more than evicting the loser cleanly
would have. `cam_merge_gate` defaults to `True` for this reason; set it to
`False` only to reproduce the ablated, non-recommended configuration.

The gate's draws are deterministic and reproducible (seeded from
`KVCacheConfig.seed`), so a run with the same config and inputs always
produces the same result.

| Eviction axis | Disposition | Score signal | Budget |
|---|---|---|---|
| SnapKV-adapted | Drop | Key-as-query attention proxy | Uniform |
| H2O-adapted | Drop | Cumulative attention mass | Uniform |
| TOVA-adapted | Drop | Current-step attention weight | Uniform |
| PyramidKV-adapted | Drop | Cumulative attention mass | Per-layer pyramid |
| SqueezeAttention-adapted | Drop | Cumulative attention mass | Per-layer data-driven |
| ChunkKV-adapted | Drop (chunk) | Pooled attention-mass / key-norm | Uniform |
| **CaM-adapted** | **Merge (gated)** | Cumulative attention mass | Uniform |

## When to use it

| Scenario | Recommended method |
|----------|-------------------|
| Compress all tokens uniformly | KIVI-2bit |
| Hard cap on tokens, evict at prefill only | SnapKV-adapted |
| Constant-memory, cumulative-importance eviction, uniform budget | H2O-adapted |
| Constant-memory, importance eviction with a fixed depth-adaptive budget | PyramidKV-adapted |
| Constant-memory, importance eviction with a data-driven depth-adaptive budget | SqueezeAttention-adapted |
| Constant-memory, importance eviction that keeps whole contiguous chunks | ChunkKV-adapted |
| **Aggressive eviction that merges (not drops) evicted tokens to recover quality** | **CaM-adapted** |

## Fidelity to the paper

*Inspired by* [CaM: Cache Merging for Memory-efficient LLMs Inference](https://proceedings.mlr.press/v235/zhang24n.html)
(Zhang et al., ICML 2024, PMLR 235:58840-58850) — documented here as
**CaM-adapted (VeloxQuant-MLX implementation)**, not a faithful port.
Differences from the paper, stated plainly:

- **Cosine-similarity blend weight, not the paper's attention-prominence
  weight.** At the streaming eviction boundary, the evicted token is
  frequently the token just appended (score 0, before it accumulates any
  mass), so an attention-mass *blend* weight would make the merge a no-op.
  Cosine similarity between the loser's and survivor's keys is always
  meaningful and cache-observable. The merge *gate* (above) still uses
  attention mass, per the paper — this substitution is specific to the blend
  step only.
- **Single nearest-survivor merge target**, not the paper's local window of
  `m` contiguous tokens (`j:j+m`). The gate's target score is this one
  neighbour's score, standing in for the paper's `avg(Ā_j:j+m)`.
- **Key-as-query proxy** (same as [H2O-adapted](./h2o.md)): both the
  importance score and the merge-similarity are computed from the key
  vectors the cache holds, not the true query / attention maps the paper
  reads.
- **No RoPE position-ID remapping** after a merge.
- **Uniform budget across heads** within a layer.

## Relationship to H2O

CaM **is** H2O with a gated merge step spliced into the eviction. The scorer
(cumulative attention mass, key-as-query proxy), sink protection, eviction
choice, and byte accounting are all H2O's. Set `cam_merge="drop"` and CaM and
H2O are bit-for-bit identical — the analogue of "`chunk_size=1` == H2O"
([ChunkKV](./chunkkv.md)) and "`strength=0` == H2O"
([SqueezeAttention](./squeeze.md)).

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/quantizers/test_cam.py` and
`veloxquant_mlx/tests/cache/test_cam_cache.py`, covering:

- **Merge target:** picks the closest non-sink survivor by cosine; excludes
  sinks and the evicted slot; returns `-1` when only sinks remain
- **Merge gate:** probability bounds and clamping; deterministic and
  reproducible sampling by `(seed, draw_id)`; a low-relative-score loser is
  blocked (dropped, not merged) the large majority of the time; disabling
  the gate reproduces unconditional merging
- **Blend:** `drop` returns the survivor unchanged; `mean` averages values;
  `sim_weighted` folds a similar loser in strongly; `merge_keys` toggles key
  blending; values-only merge leaves surviving keys identical to the drop path
- **Eviction:** budget held at exactly `cam_budget`; sinks always retained;
  deterministic (seeded, no unseeded RNG); byte accounting
- **Cache:** budget enforced; sink preservation; correct output shapes across
  batch/heads; all three modes + `merge_keys` run; prefill-then-decode;
  factory + `for_model` return `CaMKVCache` per layer
- **`drop` == H2O:** identical kept keys **and** values versus `H2OKVCache`
  at the same budget, at both the primitive and cache level, regardless of
  gate setting

The offline harness in `benchmark_scripts/benchmark_cam.py` sweeps
`(seq_len, budget, merge_mode)` on synthetic fp16 K/V, measuring each
config's **output perturbation** — the cosine distance between the
compressed cache's attention output and the full-cache output over random
probe queries (lower is better) — against the token-level H2O baseline
(`drop`). Results are committed in `figures/cam/results.json` (Apple
Silicon). The measured finding: **`sim_weighted` merging reduces
perturbation versus dropping, and the gain grows with compression ratio.** At
the most aggressive setting (`seq=1024, budget=64`, 16×) it cuts perturbation
from **0.955 → 0.708** (gain **+0.247**); gains shrink toward zero at low
compression (2×), where dropping barely hurts — exactly the regime where CaM
claims no benefit. This benchmark predates the merge-gate fix and was run
with unconditional merging; it is being re-run with the gate enabled.

**No model-level (perplexity/throughput) benchmark has been run.** The
harness is model-free (synthetic K/V + probe queries); it measures the
output-perturbation proxy CaM targets, not end-to-end task quality.
