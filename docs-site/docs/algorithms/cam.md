# CaM — Cache Merging (Merge Evicted Tokens Instead of Dropping)

**Method id:** `cam` · **New in 0.26.0** · *Inspired by* [CaM: Cache Merging for Memory-efficient LLMs Inference](https://proceedings.mlr.press/v235/zhang24n.html)
(Zhang et al., ICML 2024, PMLR 235:58840-58850) — **CaM-adapted (VeloxQuant-MLX implementation)**,
not a faithful port.

CaM-adapted is the library's **eighth eviction configuration** and the first on the
**merge-vs-drop** axis. Every other eviction method — [SnapKV](./snapkv.md),
[StreamingLLM](./streaming_llm.md), [H2O](./h2o.md), [TOVA](./tova.md),
[PyramidKV](./pyramidkv.md), [SqueezeAttention](./squeeze.md), [ChunkKV](./chunkkv.md)
— **permanently discards** the tokens it evicts. CaM instead **merges** each evicted
token into the surviving token it most resembles (a cosine-weighted blend), then
removes only the redundant slot. The eviction *choice* is H2O's; only the
disposition differs. With `cam_merge="drop"` it reduces **bit-for-bit** to H2O.

## Why merge instead of drop

Cache eviction *always* perturbs the output — the dropped token still carried some
attention mass, and that mass is simply lost. CaM's observation is that this
perturbation is what degrades quality as the compression ratio climbs. Instead of
discarding the loser, it folds the loser's value (and optionally key) into a
retained neighbour, so the information is compressed rather than deleted. At high
compression — where dropping hurts most — merging recovers a measurable share of
the lost signal.

| Eviction axis | Disposition | Score signal | Budget |
|---|---|---|---|
| SnapKV-adapted | Drop | Key-as-query attention proxy | Uniform |
| H2O-adapted | Drop | Cumulative attention mass | Uniform |
| TOVA-adapted | Drop | Current-step attention weight | Uniform |
| PyramidKV-adapted | Drop | Cumulative attention mass | Per-layer pyramid |
| SqueezeAttention-adapted | Drop | Cumulative attention mass | Per-layer data-driven |
| ChunkKV-adapted | Drop (chunk) | Pooled attention-mass / key-norm | Uniform |
| **CaM-adapted** | **Merge** | Cumulative attention mass | Uniform |

## The merge

When the cache exceeds budget, CaM picks the same loser H2O would (lowest
cumulative attention mass, sinks protected), then:

1. **Find the target** — the surviving non-sink token whose key is most similar
   (cosine) to the loser's key (`most_similar_survivor`).
2. **Blend** — `x_new = (1 - w)·x_survivor + w·x_evicted`, where the weight `w`
   depends on `cam_merge`:
   - `"sim_weighted"` (default) — `w = clip(cos(k_evicted, k_survivor), 0, 1)`.
     A loser that closely resembles its survivor is absorbed strongly; a
     dissimilar one barely perturbs it.
   - `"mean"` — `w = 0.5` (unweighted ablation baseline).
   - `"drop"` — `w = 0`; the survivor is untouched and the loser is dropped
     (== H2O).
   Values are always merged; keys only when `cam_merge_keys=True`.
3. **Transfer mass + remove** — the survivor inherits the loser's cumulative
   score, and the loser's slot is removed, keeping the cache at exactly `budget`.

### Why not the paper's attention-mass weight

CaM's paper weights the merge by the discarded token's attention prominence. At
the streaming eviction boundary, the evicted token is frequently the token *just
appended* — its cumulative score is still 0, so an attention-mass weight would
make the merge a **no-op**. We therefore weight by key cosine similarity: always
meaningful, cache-observable, and faithful to CaM's intent (fold a token into the
neighbour it most resembles). This is documented, not claimed as a faithful port.

## Usage

CaM needs **no coordinator** — every layer merges independently — so the standard
single-config path works:

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="cam",
    head_dim=128,
    cam_budget=512,  # max tokens kept per layer (sinks included)
    cam_n_sink=4,  # initial positions never evicted (attention sinks)
    cam_merge="sim_weighted",  # "sim_weighted" | "mean" | "drop" (drop == H2O)
    cam_merge_keys=False,  # merge keys too (values are always merged)
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cam_budget` | `512` | Maximum tokens kept per layer (sinks included). CaM trims to exactly this once past budget. |
| `cam_n_sink` | `4` | Initial positions always retained (attention sinks); never evicted or used as a merge target. |
| `cam_merge` | `"sim_weighted"` | Merge rule. `"sim_weighted"` = cosine-weighted blend; `"mean"` = unweighted; `"drop"` = no blend (bit-for-bit H2O). |
| `cam_merge_keys` | `False` | Merge keys as well as values. Values are always merged; merging keys shifts the attention geometry (the paper treats this as optional). |

## Relationship to H2O

CaM **is** H2O with a merge step spliced into the eviction. The scorer
(cumulative attention mass, key-as-query proxy), sink protection, eviction choice,
and byte accounting are all H2O's. The only addition is `most_similar_survivor` +
`merge_pair`, which blend the loser into a survivor before its slot is removed.
Set `cam_merge="drop"` and CaM and H2O are bit-for-bit identical — the analogue of
"`chunk_size=1` == H2O" ([ChunkKV](./chunkkv.md)) and "`strength=0` == H2O"
([SqueezeAttention](./squeeze.md)).

## Proxy limitation

Documented as "CaM-adapted (cosine-similarity merge weight, key-as-query proxy,
single nearest-survivor merge)" — never claimed as a faithful port. Specifically:
cosine-similarity merge weight rather than the paper's attention-prominence weight
(which is ~0 for a just-appended token); single nearest-survivor merge (no
multi-target soft assignment or the paper's sampling over discarded locations);
key-as-query proxy for the importance score; no RoPE position-ID remapping after
merge; uniform budget across heads within a layer.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/quantizers/test_cam.py` (18 tests) and
`veloxquant_mlx/tests/cache/test_cam_cache.py` (14 tests):

- **Merge target:** picks the closest non-sink survivor by cosine; excludes sinks
  and the evicted slot; returns `-1` when only sinks remain
- **Blend:** `drop` returns the survivor unchanged; `mean` averages values;
  `sim_weighted` folds a similar loser in strongly; `merge_keys` toggles key
  blending; values-only merge leaves surviving keys identical to the drop path
- **Eviction:** budget held at exactly `budget`; sinks always retained; merged
  values differ from the dropped ones; deterministic (no RNG); byte accounting
- **Cache:** budget enforced; sink preservation; correct output shapes across
  batch/heads; all three modes + `merge_keys` run; prefill-then-decode; factory +
  `for_model` return `CaMKVCache` per layer
- **`drop` == H2O:** identical kept keys **and** values versus `H2OKVCache` at the
  same budget, at both the primitive and cache level

The offline harness in `benchmark_scripts/benchmark_cam.py` sweeps
`(seq_len, budget, merge_mode)` on synthetic fp16 K/V, measuring each config's
**output perturbation** — the cosine distance between the compressed cache's
attention output and the full-cache output over random probe queries (lower is
better) — against the token-level H2O baseline (`drop`). Results are committed in
`figures/cam/results.json` (Apple Silicon). The measured
finding: **`sim_weighted` merging reduces perturbation versus dropping, and the
gain grows with compression ratio.** At the most aggressive setting (`seq=1024,
budget=64`, 16×) it cuts perturbation from **0.955 → 0.708** (gain **+0.247**);
gains shrink toward zero at low compression (2×), where dropping barely hurts —
exactly the regime where CaM claims no benefit.

**No model-level (perplexity/throughput) benchmark has been run.** The harness is
model-free (synthetic K/V + probe queries); it measures the output-perturbation
proxy CaM targets, not end-to-end task quality.

## When to use it

CaM-adapted is best when you want H2O-style importance eviction but at an
**aggressive** compression ratio where dropping tokens visibly degrades output —
merging recovers a share of the lost mass at no extra memory. Set
`cam_merge="drop"` to fall back to plain H2O; use `"sim_weighted"` for the merge.
It composes with the same use cases as H2O.

| Scenario | Recommended method |
|----------|-------------------|
| Compress all tokens uniformly | KIVI-2bit |
| Hard cap on tokens, evict at prefill only | SnapKV-adapted |
| Constant-memory, cumulative-importance eviction, uniform budget | H2O-adapted |
| Constant-memory, importance eviction with a fixed depth-adaptive budget | PyramidKV-adapted |
| Constant-memory, importance eviction with a data-driven depth-adaptive budget | SqueezeAttention-adapted |
| Constant-memory, importance eviction that keeps whole contiguous chunks | ChunkKV-adapted |
| **Aggressive eviction that merges (not drops) evicted tokens to recover quality** | **CaM-adapted** |
