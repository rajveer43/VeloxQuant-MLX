---
id: cross-model-transfer
title: Cross-Model KV Transfer
sidebar_label: Cross-Model KV Transfer
slug: /algorithms/cross-model-transfer
---

# Cross-Model KV Transfer — Reuse One Model's Prefill in Another

**Module:** `veloxquant_mlx.transfer` · **Not a `method=` cache** · *Inspired by*
["Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping
for Prefill Reuse" (Heo, Shafipour, Zhao, Golub, Kamani, Borkar, Chandran,
Zardoshti, Darvish Rouhani; **NVIDIA**,
arXiv:2608.03893)](https://arxiv.org/abs/2608.03893) —
**CrossKV-adapted (VeloxQuant-MLX implementation)**, not a faithful port.

## This is not a compression method

Every other entry in this documentation shrinks **one** model's KV cache and
plugs into `KVCacheConfig(method="...")`. This one does something structurally
different, and the distinction matters before you read any further:

| | Every other method here | Cross-Model KV Transfer |
|---|---|---|
| Models involved | 1 | **2** (source + target) |
| What it reduces | cache **bytes** | receiver's **prefill compute** |
| Cache size after | smaller | **unchanged** — same as the target's own |
| Setup cost | none, or a short calibration | **offline fit per model pair** |
| API | `KVCacheConfig(method=...)` | `veloxquant_mlx.transfer` |

If you want a smaller cache, this is the wrong page — start at the
[method library](../algorithms/overview). This page is for a different
situation: you are **switching between two models in the same family**, and
you do not want the receiving model to re-read the whole prompt.

## The problem

Production serving swaps models mid-session for cost-quality cascading,
routing, or mid-conversation upgrades. Every swap makes the receiving model
re-run prefill over the entire accumulated context, and prefill cost scales
with both model size and prompt length. Prefix caching does not help — it
only reuses KV **within** a single model.

The paper's observation is that the two models' caches are related closely
enough to be **mapped** rather than recomputed. On Qwen3 14B→32B, one source
layer linearly explains 56% of the variance in the target's keys and 32% in
its values; combining several source layers raises that to 79% and 65%.

## The mapping

For each target `(layer, head)`, a ridge regression is fit offline:

```
K̂_t = (strip_rope(K_s) · W_K + b_K) · target_rope
V̂_t =            V_s    · W_V + b_V
```

Three components, in the paper's order of importance:

1. **Top-`k` cross-layer source selection** (§3.2) — each target layer is fit
   from the `k` source layers that predict it best, concatenated. This is the
   single largest contributor: dropping to `k=1` takes key `R²` from 0.79 to
   0.56 and collapses downstream accuracy.
2. **Per-head ridge** (§3.1) — closed form `W = (XᵀX + λI)⁻¹ XᵀY`, λ=0.01,
   solved on centered data so the intercept escapes the penalty. **No gradient
   training.**
3. **Content-space mapping** (§3.3) — keys are un-rotated before fitting and
   re-rotated with the *target's* RoPE at apply time, so the fit is
   position-free and transfers to context lengths it never saw.

## :warning: The honesty crux — read this first

1. **The paper's retention numbers are not ours.** The paper reports 73–98%
   accuracy retention on four of six pairs, and 2.7–25× speedup versus
   re-prefill — measured on 14B→32B and 8B→70B pairs on 8×H100 nodes. **We
   have not reproduced those numbers on Apple Silicon**, and this page does
   not claim them. What is verified here is that the implementation recovers a
   known linear relationship exactly and that the Metal kernel matches its
   reference; downstream retention on real model pairs is unmeasured.

2. **Two of the paper's own six pairs failed badly.** Ministral 3B→14B and
   8B→14B fell to 42–44% average retention (11–15% floor-normalized). Matched
   KV shapes correlate with success but **do not guarantee it**. Expect to
   evaluate each pair rather than assuming it will work.

3. **The fit is expensive and the artifact is large.** The paper reports
   ~47–87 min per pair on an 8×H100 node, producing 1.01–3.36 B mapper
   parameters (4–12 GB). This is an offline, once-per-pair job. On Apple
   Silicon the artifact is loaded lazily per layer and released after use, so
   peak memory is one layer's weights — but the disk cost is real.

4. **Scale limits on Apple Silicon.** Holding source + target + mapper
   resident is realistic only at the small end of a family (e.g. Qwen3
   0.6B→1.7B). The paper's pairs assume datacenter memory.

5. **Ridge only — no MLP fallback.** The paper's nonlinear MLP variant (§4.4),
   which recovers up to +37 pp on the pairs where ridge fails, is **not
   implemented**. Pairs whose relationship is not close to linear degrade with
   no recourse.

6. **`R²` will mislead you.** The paper's §4.5 finding is that calibration
   `R²` does *not* predict downstream retention across pairs (r = −0.20);
   attention-output cosine does (r = +0.57). We record `R²` per layer for
   source selection and debugging. **Do not read a high `R²` as a quality
   guarantee** — the attention-output cosine diagnostic is not yet implemented.

7. **Refused configurations.** Mismatched KV shapes, `rope_scaling`
   (Llama-3.1-style), and `traditional=True` RoPE all raise `MatchedKVError`
   rather than being approximated. The paper only evaluates matched-KV pairs;
   silently mapping outside that regime would produce plausible, wrong keys.

## Usage

Fitting takes KV activations you have already collected from both models over
the same tokens — capturing them requires hooking each architecture's
attention modules, which belongs in your code rather than here.

```python
import mlx.core as mx
from veloxquant_mlx.transfer import fit_mapper, load_mapper, transfer_cache
from veloxquant_mlx.transfer.mapper import MapperConfig, ModelKVSpec

src_spec = ModelKVSpec.from_model(source_model)
tgt_spec = ModelKVSpec.from_model(target_model)

# source_kv / target_kv: {layer: (keys, values)} of [n_kv_heads, N, head_dim],
# keys post-RoPE as the model stores them, over the same calibration tokens.
mapper = fit_mapper(
    source_kv, target_kv, src_spec, tgt_spec,
    positions=mx.arange(n_tokens),
    config=MapperConfig(k=8, ridge_lambda=0.01),
)
mapper.save("mappers/qwen3-0.6b-to-1.7b")
```

At serving time the fit is not repeated:

```python
mapper = load_mapper("mappers/qwen3-0.6b-to-1.7b")

# source_kv is the cache the small model just prefilled.
target_kv = transfer_cache(source_kv, mapper, positions=mx.arange(n_tokens))
# → {target_layer: (keys, values)}, ready to install into the target model.
```

`transfer_cache` releases each layer's mapper weights as it goes
(`unload_after=True`, the default), so peak memory stays at one layer rather
than the whole artifact. Set it `False` when mapping many caches back to back.

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `k` | `8` | Source layers concatenated per target layer. The paper sweeps this per pair; `k=1` is shown to be uniformly insufficient. |
| `ridge_lambda` | `0.01` | Tikhonov term. The paper reports a wide flat region, collapsing only at `λ=1`. |
| `n_calib_sequences` | `500` | Calibration sequences. The paper's sweep flattens after ~200; `N=50` is still within ~1.6 pp. |
| `calib_seq_len` | `1024` | Tokens per calibration sequence. |
| `stride` | `4` | Token subsampling stride. |
| `content_space` | `True` | Strip/re-apply RoPE around the map. Disabling ties the fit to the calibration context length. |
| `dtype` | `"float16"` | Storage dtype for fitted weights. |

Calibration **domain** is the one axis the paper found to carry a real cost:
substituting CodeAlpaca for FineWeb-Edu cost 5.24 pp on HellaSwag, while
Wikipedia stayed within noise. Calibrate on text resembling your workload.

## Metal kernel

The apply path's RoPE re-encode — converting keys from the source model's
rotation to the target's — is fused into a single Metal kernel
(`crosskv_rope_recode`). Because both rotations act on the same element pair at
the same position, they compose into one rotation by the per-dimension angle
*difference*:

```
angle = p · (θ_t^(-d/half) − θ_s^(-d/half))
```

so the kernel does one `sincos` and one 2×2 rotation per pair, with no
intermediate array and no materialized cos/sin tables.

**Measured: 12.9× faster than the two-pass MLX path** at 8K context
(8 groups × 8192 tokens × 128 dims, fp16: 8.76 ms → 0.68 ms on M-series),
agreeing with the reference to fp16 resolution.

This differs from the RoPE remap in the [H2O](../algorithms/h2o) eviction
kernel: there the base is *shared*, so the angles collapse to
`(new_pos − old_pos) · inv_freq`. Here the bases differ between models, so the
difference must be taken per dimension after exponentiation, and no position
delta exists at all.

The kernel is used automatically for 3-D `[BH, N, D]` inputs when Metal is
available, falling back to the MLX path otherwise.

## What is not implemented

Tracked as follow-ups rather than silently missing:

- **Nonlinear MLP mapper** (§4.4) — the fallback for pairs where ridge fails.
- **Attention-output cosine diagnostic** (§4.5) — the metric that actually
  predicts retention, and the right way to screen a pair before deploying it.
- **Evaluation harness** — the five-benchmark retention suite and the
  multi-turn handoff drift measurement (§4.6).
- **Cross-family transfer** — the paper leaves Qwen3→Llama open; so do we.
- **Mismatched-KV pairs** — refused, per the paper's own scope.
