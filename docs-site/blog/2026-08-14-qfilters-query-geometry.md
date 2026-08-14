---
slug: qfilters-query-geometry
title: "The Sign Was the Whole Paper: Debugging a KV Cache Compressor on Real Models"
date: 2026-08-14
authors: rajveer
tags: [kv-cache, q-filters, metal, apple-silicon, mlx, benchmarks]
---
# The Sign Was the Whole Paper

## How one arbitrary minus sign turned a KV cache compressor from useful into noise — and what it took to prove it on real models

---

I had a KV cache compression method in my library called Q-Filters. It had tests. It had docs. It had a benchmark harness with committed results. It shipped in version 0.31.0.

It also didn't work.

Not "worked slightly worse than the paper." Not "worked on some heads." On real trained weights, its scoring signal correlated with true attention at **−0.032** — statistically indistinguishable from a coin flip, and pointing the wrong way about half the time.

Fixing it meant implementing the paper properly, writing two Metal kernels, finding a position-encoding bug that made end-to-end measurement impossible, and throwing away two benchmark harnesses that produced confident, meaningless numbers. Along the way the calibrated version went from **ppl 598 to 16.3** on the same workload.

This is the whole run, including the parts where I was wrong.

{/* truncate */}

---

## What Q-Filters is supposed to do

Every token a language model generates writes a **Key** and a **Value** vector into the KV cache. At 32K context these outweigh the model itself. So you evict: keep the important entries, drop the rest.

The hard part is deciding what's important. The obvious answer — look at the attention weights — is expensive and, worse, incompatible with FlashAttention, which never materializes the attention matrix you'd need to inspect.

[Q-Filters](https://arxiv.org/abs/2503.02812) (Godey et al., 2025) makes a sharp observation. For a trained attention head, the Query and Key distributions are *jointly anisotropic*: they drift away from the origin along a shared direction. Call that direction `uʰ`. Then the paper's Theorem 3.3 says:

```
E_Q [ <Q_i, K_j> ]  ≈  κʰ · <K_j, uʰ>

with   κʰ = E_Q [ <Q_i, uʰ> ]  >  0
```

Read that carefully, because the entire method lives inside it. The expected attention logit for a cached key is proportional to that key's **projection onto a single fixed direction**. One dot product per key. No attention matrix. No query needed at eviction time.

You compute `uʰ` once, offline, per model. Then eviction is: project, rank, drop the bottom.

---

## The bug I had shipped

The paper gets `uʰ` from the **SVD of query activations**, collected offline (§3.2, Eq. 1):

```
Qʰ = U Σ Vᵀ,   V = (v₁, v₂, …, v_dH)
```

My implementation got it from the SVD of the **keys** it happened to observe at runtime.

The reasoning behind that shortcut wasn't crazy. A KV cache never sees query vectors — it only receives the K and V passed to `update_and_fetch`. Queries live upstream in the attention module. So I'd substituted "a different estimator of the same head-geometry direction" and documented it honestly. The module docstring literally had a section header reading `THE HONESTY CRUX`, and the docs said the substitution was "a genuine deviation, not a shortcut."

Documented dishonesty is still dishonesty when the thing you documented is *broken*.

Here's what I'd missed. Look at `κʰ > 0` again. That positivity is what makes "higher projection means more attention" a valid ranking rule — flip its sign and you're ranking by *least* important. And `κʰ` is defined as an expectation over the **query** distribution. It is not a property of the keys. It is not recoverable from the keys.

Estimating from keys throws away exactly the quantity that tells you which end of the axis matters.

The symptom was visible in my own committed benchmark and I'd rationalized it: the key-SVD recovered the planted axis with `filter_cosine ≈ 0.97`, but whether `sign=+1` or `sign=-1` was the good arm "flips from row to row." I'd shipped the sign as a *config knob* — an ablation the user could try both ways. That's not a knob. That's a coin flip wearing a parameter name.

There was a second, worse problem I only found later. My key-side estimator **mean-centered** the data before the SVD:

```python
x = keys.astype(mx.float32)
x = x - mx.mean(x, axis=0, keepdims=True)  # center
cov = (x.T @ x) / max(int(x.shape[0]) - 1, 1)
```

Centering is standard PCA hygiene. It is also precisely wrong here. The paper's Observation 3.1 is about the cloud's **drift away from the origin** — the mean offset *is* the signal. Subtracting it leaves you measuring variance, which is a different direction entirely.

I verified this directly:

```
drift along u = 6.0, competing spread along w = 2.0

uncentered SVD  ·  u = 1.000   (drift recovered)
centered   SVD  ·  u = 0.001   (drift destroyed)
centered   SVD  ·  w = 1.000   (returns the variance axis instead)
```

So the old implementation was doing two things wrong at once: measuring the wrong matrix, and then destroying the drift signal even within that matrix.

---

## Implementing the paper

The fix is a separate offline calibration module — which turned out to be well-precedented in my own codebase. I already had `amc_calibration.py` doing offline SVD calibration, and, more usefully, `a2ats.py` was already computing `H = E[qᵀq]` from **query** states. So query access was a solved problem; I'd just never connected it to Q-Filters.

`qfilters_calibration.py` implements §3.2 step 1 as written:

```python
def compute_qfilters(queries, max_svd_samples=3000):
    """[H, N, D] query activations -> [H, D] unit-norm Q-Filters."""
    for head in range(h):
        mat = q[head].astype(np.float64)  # NOT centered
        u, _s, vt = np.linalg.svd(mat, full_matrices=False)
        v1, u1 = vt[0], u[:, 0]

        # Paper §3.2 step 1c: v_1^+ = sgn(1^T u_1) v_1
        s = float(np.sum(u1))
        v1 = v1 if s >= 0.0 else -v1
        out[head] = v1 / max(np.linalg.norm(v1), 1e-12)
```

Two details carry all the weight:

**No mean-centering.** Commented, tested, and justified — because it looks like an omission and a future reader will "fix" it.

**Sign anchoring on `sgn(1ᵀu₁)`.** An SVD's signs are arbitrary: `(u₁, v₁)` and `(−u₁, −v₁)` are both valid factorizations. Anchoring on the left singular vector's sum orients the filter along the direction the queries actually drift, which is what makes `κʰ > 0` hold.

Plus GQA handling — Llama-3.2-1B has 32 query heads feeding 8 KV heads, and the paper says to average each group's filters onto its KV head, renormalizing so projections stay comparable across heads.

The immediate check, on planted geometry:

```
head 0  query-SVD cos vs +u = 0.9999
head 1  query-SVD cos vs +u = 1.0000
key-SVD             cos vs +u = +0.3649   (axis only, sign arbitrary)
```

Sign recovered. Now: does any of this hold on a real model?

---

## Run 1: does the anisotropy actually exist?

Everything above assumes trained attention heads really are anisotropic in the way the paper claims. That's the paper's empirical finding, and I'd never checked it.

I wrote a script to measure both observations on real query activations. Observation 3.1 is `E<Q,uʰ> > 0`. Observation 3.2 is that projections onto every *other* SVD component have near-zero mean — the anisotropy is one-directional, not diffuse.

Three cached models, 1968 attention heads total:

| Model | Obs 3.1: `E<Q,uʰ> > 0` | Obs 3.2 ratio | Top-component energy |
|---|---|---|---|
| Llama-3.2-1B-Instruct-4bit | **100%** of 512 heads | 44.4× | 90.6% |
| Llama-3.2-3B-Instruct-4bit | **100%** of 672 heads | 52.3× | 84.7% |
| Qwen2.5-7B-Instruct-4bit | **100%** of 784 heads | 47.9× | 81.3% |

Observation 3.1 held in **every single head measured**. Not 95%, not "most heads" — 1968 for 1968, with median projection +11.9 to +15.5.

That universal positivity *is* `κʰ > 0`. The quantity my key-side estimator had been throwing away is, empirically, one of the most reliable properties of a trained transformer.

Observation 3.2 held too: the leading component's mean projection runs ~50× the others, which sit near zero. That's the paper's Figure 2c, reproduced.

The Qwen result was a small surprise. The paper's §5 lists Qwen-2.5 as a *limitation* — its QKV projection bias was expected to break the geometric assumptions. The anisotropy shows up anyway, at 47.9×.

---

## Run 2: does the filter predict real attention?

Anisotropy existing is necessary but not sufficient. The claim that matters is that projecting onto `v₁⁺` predicts *attention*. So I built the paper's Figure 4: compute the true attention map, measure the actual mean attention each position receives,

```
Sʰ_t = (1 / (L - t + 1)) · Σ_{i=t..L} Aʰ_it
```

and rank-correlate each scoring method against it. Calibration on one corpus, evaluation on held-out text.

| Scorer | Llama-3.2-1B (128 KV heads) | Llama-3.2-3B (224 KV heads) |
|---|---|---|
| **Q-Filters, calibrated (query-SVD)** | **+0.783** (100% sign-correct) | **+0.863** (100%) |
| K-norm (Devoto et al.) | +0.460 (94.5%) | +0.410 (94.6%) |
| Q-Filters, key-SVD (what I'd shipped) | **−0.032** (46.1%) | **−0.008** (49.1%) |

This is the result that made the whole exercise worth it.

The calibrated filter correlates strongly with true attention and **beats K-norm**, reproducing the paper's Figure 4 ordering. Meanwhile the thing I had shipped, tested, documented and released sits at −0.032 with its sign correct **46% of the time**. Worse than guessing.

And the failure was even more complete than "ambiguous sign." With isotropic key noise, the mean-centering estimator has no drift left to lock onto, so it returns a nearly unrelated direction:

```
pure-drift key geometry:
  key-SVD   |cos| vs planted direction = 0.02
  query-SVD  cos  vs planted direction = 0.99
```

Two independent measurements, same conclusion: query-SVD and key-SVD aren't two estimators of one thing. One is a signal; the other is noise.

---

## Metal kernels, and one trap

With the math right, the hot path deserved GPU treatment. My library already had fused eviction kernels for H2O and Keyformer, so the pattern existed — but Q-Filters differs structurally in a way that changes the design.

H2O and Keyformer evict **exactly one row per token**. So dispatch 1 is an argmin reduction to a single index, and dispatch 2 shifts rows around it in closed form: `src = j + (j >= evict_idx)`.

Q-Filters evicts a **whole block down to budget at once**. There's no closed form — a thread writing output row `j` can't know its source without knowing how many survivors precede it. So:

1. **`qfilters_score.metal`** — full `[BH, n_total]` projection scores, fp32 accumulation over fp16 keys, with sink and recent rows forced to `+INFINITY` so protection is baked into the values the threshold later sees.
2. **`qfilters_evict_apply.metal`** — one threadgroup per `(batch, head)`, cooperative scan to build a survivor index list in threadgroup memory, then a fully parallel gather.

The threshold itself stays in MLX via `mx.sort`. A top-k is a primitive MLX already implements well; reimplementing it in Metal would be slower and harder to trust.

**The trap.** The obvious way to apply a threshold is `score >= thresh`. That silently overflows the budget whenever the threshold value repeats — and here duplicates aren't an edge case, they're guaranteed, because every protected row shares `+INFINITY`. Overflow would break the cache's size guarantee, which is the one thing a compressor must never break.

So admission runs in two tiers: strictly-greater always survives; equal-to-threshold survives only while quota remains, scanning low index to high. That reproduces `mx.argsort`'s lowest-index-first tie-break, so kernel and MLX paths agree exactly.

```
tied scores (all keys identical, all scores equal):  kept == budget exactly
all-protected kept set (10 sinks + 10 recent == 20): kept == budget exactly
```

Twenty kernel tests, mostly parity checks against a numpy argsort reference for keys *and* values. A kernel that's merely plausible but disagrees with the reference is a silent correctness bug, which is the worst kind.

---

## Harness bug 1: the numbers that lied to me

Then I tried to measure end-to-end perplexity, and got this:

```
fp16 full cache                    ppl     1.240
Q-Filters CALIBRATED (query-SVD)   ppl  4624.480
Q-Filters fallback (key-SVD)       ppl  6650.481
```

A method correlating +0.78 with true attention does not produce ppl 4624. When results are absurd, suspect the harness.

**Harness bug 1: single-pass prefill.** I'd fed the whole sequence as one block. The cache evicts *during* that block, so most tokens were being predicted from a cache already mutilated in a way real generation never does. The paper is explicit about the setup (§4): let the cache grow to a threshold, then evict as you go, scoring each next-token prediction. Token-by-token, not one shot.

Rewriting it that way helped — and still gave ppl 598. Which meant the harness wasn't the only problem.

---

## The bug underneath: RoPE positions

Something structural remained, and the *tell* was that the calibrated arm was doing worse than the fallback — contradicting two independent correlation measurements. That ordering couldn't be a scoring problem. It had to be downstream of scoring.

I read how `mlx_lm` actually calls the cache:

```python
queries = self.rope(queries, offset=cache.offset)
keys = self.rope(keys, offset=cache.offset)
keys, values = cache.update_and_fetch(keys, values)
```

Rotary position encoding is applied to both queries and keys, at `cache.offset`, **before** the cache is updated. So `cache.offset` isn't bookkeeping — it's the position signal the model rotates by.

And my cache was reporting the number of *retained* rows:

```
true pos -> cache.offset used for RoPE:
  t=  63  offset=  63  drift=   +0
  t=  64  offset=  64  drift=   +0
  t=  70  offset=  64  drift=   +6
  t= 100  offset=  64  drift=  +36
  t= 199  offset=  64  drift= +135
```

Correct until the budget fills, then frozen forever while true position climbs. Every token after eviction was rotated at the wrong position, and the error grew without bound.

I checked whether I'd caused it. `git stash`, rerun on `master`: identical `offset: 128`. **Pre-existing**, and exactly the limitation both docstrings had disclosed as "No RoPE position-ID remapping after eviction" — a line I'd written myself without understanding that it meant end-to-end generation was broken.

The fix is smaller than I expected, and the reason is worth stating. RoPE is *relative*: `<rope(q,i), rope(k,j)>` depends only on `i − j`. Q-Filters **preserves** original positions — it drops rows but never renumbers the survivors. So every stored key already carries the rotation for its true absolute position. Reporting the true position puts queries, new keys, and survivors back on one consistent axis, and no re-rotation is needed.

That's precisely why H2O and Keyformer need a delta-rotation pass in their apply kernels and this cache doesn't: they renumber positions on eviction, and Q-Filters doesn't.

```python
self._true_offset += S
self.offset = self._true_offset
```

Effect: **ppl 598.5 → 17.6.**

---

## Harness bug 2: the baseline that should have stopped me

With RoPE fixed the numbers were finally sane — and now the *other* problem became visible. My fp16 baseline was **1.240**.

That number should have stopped me on day one. A real 1B model on ordinary prose scores somewhere around 4–15. I'd built the eval text as `'...short paragraph...' * 30`, which is trivially predictable: the model just learns the loop and echoes it. With the baseline flattened to 1.24 the dynamic range was gone, and every eviction policy looked equally catastrophic against it.

Swapping in continuous, non-repetitive prose:

```
fp16 full cache (no eviction)      ppl     4.050    <- realistic at last
```

Both harness mistakes are now documented in the benchmark script's docstring, specifically so nobody repeats them — including me.

---

## Run 3: the measurement that was blocked

Now the comparison means something. 1024 tokens, generation mode, eviction active:

**Llama-3.2-1B** — fp16 baseline **4.050**

| Budget | Calibrated | Key-SVD fallback | Gap to fp16 closed |
|---|---|---|---|
| 256 (~4×) | **8.476** | 13.358 | **52%** |
| 128 (~8×) | **16.307** | 23.645 | **37%** |
| 64 (~16×) | **25.933** | 31.274 | **20%** |

**Llama-3.2-3B** — fp16 baseline **3.305**

| Budget | Calibrated | Key-SVD fallback | Gap to fp16 closed |
|---|---|---|---|
| 256 (~4×) | **5.076** | 7.264 | **55%** |
| 128 (~8×) | **10.046** | 14.909 | **42%** |

Calibrated wins at every budget on both models, in the same order the correlation numbers predicted. Three independent measurements — anisotropy, attention correlation, generation perplexity — agreeing.

---

## The finding I wasn't looking for

While debugging, I tested one hypothesis that turned out to matter more than anything else in the perplexity table. Q-Filters as specified has no recency protection. But next-token prediction depends enormously on the *immediately preceding* tokens — and a long-range importance score will happily evict them.

Llama-3.2-1B, budget 128, calibrated, sweeping the trailing protected window:

| `qfilters_recent` | 0 | 32 | 64 | 96 |
|---|---|---|---|---|
| perplexity | **263.2** | **16.3** | 20.3 | 24.9 |

A **16× swing** from a parameter that defaults to zero. Everything in the table above depends on it being set.

This doesn't contradict the paper. Q-Filters is evaluated on Ruler and needle-in-a-haystack — *retrieval* tasks, where the answer sits somewhere in the middle of a long context and recency is not what you need. In open-ended generation the priority inverts. Projection ranking finds what matters globally; a recency window covers what matters locally; you need both.

I left the default at 0 so the out-of-box configuration stays paper-faithful, and documented ≈budget/4 for generation.

---

## What I'm not claiming

The honest boundary, because it's the part most easily overstated:

**These are not a reproduction of the paper's Figure 5.** Perplexity is still well above fp16 at every ratio. My setup has uniform per-head budgets, no RoPE renumbering, and a much smaller calibration set than the paper's §4.2 (which uses 20 samples × 2048 tokens). These numbers compare *policies* against each other; they don't reproduce the paper's headline results.

**Not measured at all:** TTFT and throughput (paper Figure 10), Ruler, needle-in-a-haystack, and any comparison against SnapKV, Expected Attention, or StreamingLLM.

**L2Norm is excluded from the tables** — deliberately, not by oversight. It carries the identical un-fixed `offset` defect (`knorm_cache.py:165`), so its numbers would be dominated by position drift rather than eviction quality. Including it would have been an unfair fight in my favor. Generalizing the RoPE fix across the other evicting caches is the obvious next task, and probably affects H2O and TOVA too.

---

## What I'd take away from this

**A documented limitation is not a discharged one.** I had written "no RoPE position-ID remapping after eviction" in two docstrings. I'd written `THE HONESTY CRUX` as a section header. Both were true, both were prominent, and neither told me the method was broken — because I'd never run it on a real model. Documentation records a decision; it doesn't validate one.

**Absurd numbers are information.** Every genuinely useful step here came from refusing to accept an implausible result: ppl 4624 found the prefill bug, a baseline of 1.24 found the repetitive-text bug, and the calibrated arm underperforming found the RoPE bug. The temptation each time was to report the number with a caveat.

**Test the paper's premise, not just your code.** My original test suite was thorough about mechanics — 27 tests, all passing, on a method that scored −0.032 against real attention. Synthetic tests confirm you implemented what you intended. They cannot tell you whether what you intended is what the paper meant.

**Sometimes the fix is one line.** `self.offset = self._true_offset`, once I understood that RoPE is relative and Q-Filters preserves positions. Understanding the geometry made the code trivial; guessing at it would have produced a delta-rotation pass I didn't need.

---

## TL;DR

- My Q-Filters implementation derived its filter from **key** SVD instead of the paper's **query** SVD, discarding the `κʰ > 0` term that fixes the sign
- On real weights it scored **−0.032** Spearman against true attention, sign correct **46%** of the time — noise
- The paper's anisotropy is real: Observation 3.1 held in **1968 of 1968 heads** across Llama-3.2-1B/3B and Qwen2.5-7B, including the paper's own stated limitation case
- Properly calibrated: **+0.783 / +0.863** Spearman, 100% sign-correct, beating K-norm's +0.460 / +0.410
- Generation perplexity, calibrated vs fallback: **8.48 vs 13.36** at 4×, **16.31 vs 23.65** at 8×, **25.93 vs 31.27** at 16× (1B); **5.08 vs 7.26** and **10.05 vs 14.91** (3B)
- Found and fixed a pre-existing RoPE bug where `cache.offset` froze at the budget — position drift **+135 by token 199**, and **ppl 598.5 → 17.6** once fixed
- `qfilters_recent` causes a **16× perplexity swing** in generation (263 → 16.3) and defaults to off; set ≈budget/4 if you generate text
- Two Metal kernels for the eviction hot path, with a tie-handling rule that keeps the budget exact when protected rows all score `+inf`
- Two benchmark harnesses were discarded for producing confident, meaningless numbers; both pitfalls are documented in the script
- Not claimed: Figure 5 reproduction, TTFT, Ruler, NIAH, or baselines against SnapKV/StreamingLLM

Code, scripts, and raw numbers:
[github.com/rajveer43/VeloxQuant-MLX](https://github.com/rajveer43/VeloxQuant-MLX) — see `benchmark_scripts/qfilters_real_model_*.py` and `figures/qfilters/real_model_results.json`.

The paper: [Q-Filters: Leveraging Query-Key Geometry for Efficient KV Cache Compression](https://arxiv.org/abs/2503.02812), Godey, Devoto, Zhao, Scardapane, Minervini, de la Clergerie, Sagot.

If you maintain a KV cache implementation with a documented limitation you've never load-bearing-tested, this is your sign to go run it on a real model.
