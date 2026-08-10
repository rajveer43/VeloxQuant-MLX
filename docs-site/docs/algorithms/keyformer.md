# Keyformer — Gumbel-Regularized Heavy-Hitter Eviction

**Method id:** `keyformer` · **New in 0.32.0**, annealing/RoPE-remap/Metal
kernel added later · *Inspired by* ["Keyformer: KV Cache Reduction through Key
Tokens Selection for Efficient Generative Inference" (Adnan et al., MLSys
2024, arXiv:2403.09054)](https://arxiv.org/abs/2403.09054) —
**Keyformer-adapted (VeloxQuant-MLX implementation)**. The estimator is
deliberately changed from the paper's (see [Adaptation notes](#adaptation-notes)).

The paper's contribution is a **regularizer, not a new importance signal**.
Naively evicting by an accumulated attention score is unstable: a token that
reads low *early* — before the queries that will attend to it arrive — gets
pruned and can never recover, even if it would have become a heavy hitter.
Keyformer adds **Gumbel noise** to the eviction logits so borderline tokens are
not deterministically doomed on a single low reading (a "late riser").

:::info Two paper-fidelity gaps fixed, a fused Metal kernel added, and real-model validation completed
A comparison against the paper's Algorithm 1 / Section 3.3.1 found two real
gaps beyond the already-documented proxy-query and frozen-noise deviations:
**(1) no temperature annealing** — `tau` was a single constant for the whole
run, but the paper's actual mechanism (Equation 10) anneals it from `tau_init`
(≈1, prompt phase) to `tau_end` (≈2, as decoding discards more tokens); and
**(2) no RoPE position tracking** — this cache never tracked which absolute
position each kept key was rotated at, so an interior eviction silently
desynced survivors' rotation from their storage index, the same bug class
[H2O](../algorithms/h2o) had and fixed. Both are now fixed — see
[Annealing and RoPE-remap testing](#annealing-and-rope-remap-testing-fixes-two-real-gaps)
below — and a fused Metal eviction kernel (mirroring H2O's, ~2-3x measured
end-to-end) has been added on top of the corrected semantics. Real-model
testing (`mlx_lm.generate()`, not just synthetic state) then found and fixed
a genuine prefill-scalability crash the Metal-kernel change introduced, and
separately found — but did **not** fix — that eviction during a long prefill
(not just tight-budget decode) can still degrade output, on both Keyformer
and H2O. See [Real-model validation](#real-model-validation) below for the
full, honest breakdown.
:::

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
bit-for-bit, onto [H2O](../algorithms/h2o) — including positions, `next_pos`,
and (on Metal) the fused kernel's own reduction, all now verified to match
H2O's exactly. That is the honest ablation: the *only* thing Keyformer adds
over H2O is the Gumbel regularizer, and you can turn it off to see exactly
what it buys. `keyformer_tau=0` also disables annealing (equivalent to
`tau_init=tau_end=0`), so this collapse holds regardless of any
`keyformer_anneal_steps` setting. A dedicated test asserts the `tau=0` kept
set equals H2O's, and the benchmark prints an `h2o` column as a cross-check.

## :warning: The honesty crux — read this first

1. **Proxy query.** Like [H2O](../algorithms/h2o)/[SnapKV](../algorithms/snapkv),
   a cache never sees the true query vector, so the incoming **key** is used as
   a proxy query to estimate the attention each stored key receives. The paper
   accumulates the model's real attention logits. This is a documented
   substitution, not the paper's math.
2. **Frozen per-position noise, not redrawn sampling.** The paper redraws
   Gumbel noise fresh every decoding step. A cache processes blocks with no
   global step counter it can trust to redraw against, so we draw **one
   deterministic Gumbel value per token position** (seeded from a fixed base
   seed + a per-head running position) and freeze it. The (now-annealed — see
   below) temperature scales that frozen noise. This preserves the mechanism's
   intent — a borderline token is not doomed by one low reading — while
   staying reproducible and order-diagnosable. It is **not** the paper's
   redraw-every-step sampling, and we do not claim it is.
3. **Not validated on a trained model.** The regularizer's benefit is measured
   only under constructed "late-riser" geometry, with a stable-importance
   control where it has nothing to rescue.

**Two further gaps found and fixed** (previously undocumented, beyond the
three crux points above — see [Annealing and RoPE-remap testing](#annealing-and-rope-remap-testing-fixes-two-real-gaps)):

4. **Temperature annealing was entirely missing.** A single constant `tau` for
   the whole run cannot represent the paper's actual mechanism (Equation 10):
   `tau = tau_init + t · delta_tau`, ramping from `tau_init` (paper default 1,
   during the prompt phase when nothing has been discarded yet) to `tau_end`
   (paper default 2, as decoding discards more tokens). Fixed via
   `keyformer_tau_init` / `keyformer_tau_end` / `keyformer_anneal_steps`.
   `keyformer_tau` remains as a backward-compatible constant-temperature alias.
5. **No RoPE position tracking after eviction.** This cache never tracked
   which absolute position each kept key was rotated at, so an interior
   eviction silently desynced survivors' storage index from the rotation baked
   into their keys — the exact bug class [H2O](../algorithms/h2o) had and
   fixed. Fixed the same way: positions are now tracked and re-rotated on
   eviction via `rope_remap_positions`.

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
    keyformer_tau_init=1.0,  # Gumbel temperature at pos=0 (paper default 1)
    keyformer_tau_end=2.0,  # Gumbel temperature once annealed (paper default 2)
    keyformer_anneal_steps=256,  # steps to ramp tau_init -> tau_end
    keyformer_rope_base=10000.0,  # RoPE base for post-eviction position remap
    keyformer_seed=0,  # base seed for the frozen per-position noise
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

For a constant (non-annealed) temperature, matching the previous behavior,
pass `keyformer_tau=<value>` instead — it overrides `tau_init`/`tau_end` and
disables annealing.

Single-layer, no coordinator — the default `for_model` path returns one
`KeyformerKVCache` per attention layer. When Metal is available
(`veloxquant_mlx.metal.metal_available()`), the eviction step runs on a fused
Metal kernel automatically; it falls back to a pure-MLX loop otherwise.

## How it works

Per incoming token (prefill and decode alike):

1. Accumulate the new key's proxy-attention mass (softmax of key-as-query over
   the stored keys) into the per-token cumulative score — [H2O](../algorithms/h2o)'s
   additive rule.
2. Append the new token with cumulative score `0`, its true absolute position,
   and a **frozen per-position Gumbel draw** seeded by the head's running
   position.
3. If over `keyformer_budget`: evict the non-protected token with the lowest
   `score + tau · gumbel`, where `tau` is the **current annealed temperature**
   (`tau_init + min(pos, anneal_steps) · (tau_end - tau_init) / anneal_steps`,
   Equation 10). The Gumbel term is the whole mechanism; at `tau = 0` this is
   exactly H2O's argmin on the raw score. Sinks (first `keyformer_n_sink`) and
   the optional trailing `keyformer_recent` window are forced to survive.
   Survivors positioned after the evicted token's position shift down by one
   and are **re-rotated (RoPE remap)**, exactly like [H2O](../algorithms/h2o).

`keyformer_tau_init == keyformer_tau_end` (or `keyformer_anneal_steps = 0`,
or the `keyformer_tau` alias) reproduces a constant temperature — no
annealing — for backward compatibility with earlier configs.

The Gumbel noise perturbs only the **eviction decision** — the stored
cumulative mass itself stays clean, so the noise never compounds across steps.

When Metal is available, the per-token eviction step (append,
Gumbel-regularized argmin, evict, RoPE-remap) runs as two fused kernel
dispatches instead of a Python loop — structurally identical to
[H2O's kernel](../algorithms/h2o), with one addition: a per-row frozen Gumbel
value is threaded through both dispatches and folded into the reduction as
`score + tau · gumbel`. At `tau = 0` the kernel's reduction is bit-for-bit
identical to H2O's kernel (verified directly, at the kernel level). Falls
back to the pure-MLX loop when Metal is unavailable.

Byte accounting mirrors H2O's — `keyformer_kept_bytes`, `full_seq_bytes`,
`compression_ratio`, `tokens_seen`, `tokens_kept`. The transient float32
score/gumbel bookkeeping (one value per kept token) is not counted as cache
payload, same as H2O's scores.

## Adaptation notes

**What we do NOT implement:**
- **The paper's redrawn-every-step Gumbel sampling** — replaced by frozen
  per-position noise (crux 2). This is the mechanism deviation, not a footnote.
  Temperature *annealing* (crux/gap the paper does specify, Equation 10) IS
  now implemented — see [above](#how-it-works).
- The model's real attention logits — replaced by the key-as-query proxy
  (crux 1), same approximation as H2O/SnapKV-adapted.
- Per-head budgets / temperature schedules (uniform across heads).

**Extensions beyond the paper (off by default):**
- `keyformer_recent` — protects the most recent tokens StreamingLLM-style.
- `keyformer_seed` — makes the frozen noise reproducible and per-head
  independent.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/quantizers/test_keyformer.py` (29 tests),
`veloxquant_mlx/tests/cache/test_keyformer_cache.py` (17 tests), and
`veloxquant_mlx/tests/metal/test_keyformer_evict.py` (19 tests):

- **`tau = 0` collapses onto H2O-adapted** — the kept set, positions, and
  `next_pos` all equal H2O's, bit-for-bit, over an identical stream; and with
  no noise the kept set is **seed-invariant**. The Metal kernel's reduction
  independently matches H2O's kernel bit-for-bit at `tau=0` too.
- Budget is never exceeded, token-by-token or in a prefill block, across
  batch/head shapes.
- Sinks and the `recent` window survive heavy eviction; `n_sink + recent >=
  budget` and negative `tau`/`tau_end` raise at build time.
- The frozen Gumbel draw is deterministic per `(seed, position)` and the full
  run is reproducible, including with annealing enabled.
- **Temperature annealing**: `tau` ramps linearly from `tau_init` to `tau_end`
  over `anneal_steps` and holds at `tau_end` afterward; `anneal_steps=0` or
  `tau_init==tau_end` reproduces a constant temperature exactly.
- **RoPE remap correctness**: an interior eviction's survivors, de-rotated at
  their new position, recover their exact original pre-rotation value
  (fingerprinted-token test, same construction as H2O's equivalent test); kept
  positions stay sorted and unique under stress, and sinks never move.
- **Metal kernel parity**: the fused kernel matches the pure-MLX reference
  bit-for-bit (interior eviction, sink+recent protection, tie-break behavior,
  large-`n_total` stress up to 2048 rows) and is ~2.8x faster than the Python
  loop at `n_kept=512, D=128`.
- **Late-riser mechanism:** with a planted token that reads low early but
  aligns with a *later* burst, the token's survival rate across noise-seeds is
  **higher with the Gumbel term on than off** — a statistical mechanism claim,
  not a per-seed guarantee. (Measured on **values**, not keys, since keys can
  now be legitimately re-rotated by the RoPE-remap fix — see the test's
  comments for why a raw key comparison stopped being valid once positions
  were tracked correctly.)

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
rather than cherry-picked. This harness is offline-synthetic — survival-rate,
output-perturbation and byte-accounting numbers, not perplexity or throughput
on a real model — and predates the annealing/RoPE-remap fixes; it has not
been re-run against them. **A separate, model-level validation pass now
exists** (real `mlx_lm.generate()` calls, not synthetic state) — see
[Real-model validation](#real-model-validation) below for prefill/decode
correctness, Metal-kernel speedup, and an honestly-reported real-model
limitation this offline harness could not have surfaced.

## Annealing and RoPE-remap testing: fixes two real gaps

Comparing the implementation against the paper's Algorithm 1 and Section
3.3.1 directly (not just checking it "resembles" H2O with noise) surfaced two
concrete, fixable gaps, independent of the three honesty-crux approximations
that were already documented:

1. **No temperature annealing existed at all.** `tau` was a single float,
   constant for the entire run. The paper's Equation 10 is not a refinement —
   it is *why* the score function behaves like a near-standard softmax during
   the prompt phase (nothing discarded yet, `tau≈1`) and grows more
   randomized as decoding discards more tokens (`tau→2`). A constant `tau` is
   either always too sharp (no protection when most needed, late in
   generation) or always too soft (needless randomization during the prompt).
   Fixed via `keyformer_tau_init`/`keyformer_tau_end`/`keyformer_anneal_steps`,
   verified to ramp linearly and hold at `tau_end` past `anneal_steps`.
2. **No RoPE position tracking, at all.** Unlike [H2O](../algorithms/h2o),
   this cache never tracked absolute positions, so any interior eviction (not
   evicting the newest arrival) silently corrupted the rotation-vs-storage-index
   invariant every subsequent proxy-attention computation and the model's own
   attention math depend on. This is not a theoretical concern: once
   sink/recent protection is combined with *any* non-recency-based selection
   (which is the entire point of the Gumbel regularizer), interior eviction
   becomes routine, not rare. Fixed the same way H2O fixed it: track
   positions, re-rotate shifted survivors via `rope_remap_positions`, verified
   via the same fingerprinted-interior-eviction construction H2O's test uses.

A byproduct worth flagging honestly: fixing RoPE tracking changes eviction
outcomes in existing tests that happened to encode assumptions the freeze bug
made trivially true (e.g. "the newest token is always protected" was only
ever true because the newest token was always the eviction target absent
noise — the freeze, not a designed invariant). Those tests were corrected to
check the actual invariant (position sorted/unique, sinks fixed, values —
which are never rotated — used as the fingerprint) rather than the
accidental one.

## Real-model validation

Everything above (through "Evidence") was synthetic-only until this pass —
the docs previously said so explicitly. `benchmark_scripts/benchmark_keyformer_real_model.py`
runs actual `mlx_lm.generate()` on **Llama-3.2-1B-Instruct-4bit** and
**Llama-3.2-3B-Instruct-4bit**, covering prefill, decode, and the fused Metal
kernel — with both a genuine bug found and fixed during this pass, and a
genuine limitation found and left honestly open.

### Bug found and fixed: long-prefill crash (same class as H2O's original fix)

Running a ~3200-token prompt through `mlx_lm.generate()` with the fused Metal
kernel active raised `RuntimeError: [metal::malloc] Resource limit (499000)
exceeded` — the exact failure class H2O's own prefill-scalability fix
addressed (see [H2O](../algorithms/h2o)'s docs), now reproduced here. Root
cause: `keyformer_update`'s per-token eviction loop queues one eviction's
worth of unevaluated graph nodes (concatenations, or two Metal kernel
dispatches) per token, with no periodic evaluation forcing the graph to
materialize — and the RoPE-remap/Metal-kernel changes in this pass added more
per-eviction graph nodes (a `positions` concat, a `gumbel` concat/compaction)
than the pre-fix code had, which was enough to tip a long, heavily-evicting
prefill over the resource ceiling that a shorter graph did not hit. **Fixed**
via the same `_EVAL_FLUSH_INTERVAL = 32` periodic `mx.eval()` flush H2O uses
— confirmed via git bisection against the pre-Metal-kernel code that this
crash did not exist before this pass's changes, and confirmed fixed by
re-running the identical failing prompt after adding the flush.

### Bug NOT found here: `tau=0` real-model parity holds exactly

`keyformer_tau=0` produced **byte-for-byte identical generated text** to
running H2O-adapted directly (`h2o_grace=0, h2o_decay=1.0`) on the same
prompt, same budget, same model — not just matching synthetic state as the
unit tests already covered, but the actual decoded string from a real
`mlx_lm.generate()` call. This is the strongest form of evidence for the
"`tau=0` is H2O-adapted" claim available without literally diffing weights.

### Limitation found, NOT Keyformer-specific, left honestly open: eviction during a long prefill degrades output

A prompt long enough to force eviction *during prefill itself* (not just
during decode) — even at a moderate ~2-3x compression ratio, well outside the
"tight budget" zone the existing "Grace-and-decay testing" section on the H2O
page already documents — produces garbled, incoherent output on both models
tested. This is **not new to Keyformer and not introduced by this pass**:
running H2O directly (`h2o_grace=32, h2o_decay=0.98`, i.e. the *fixed*
defaults) on the byte-identical prompt and eviction pressure degrades just as
badly (to a single-character response in one control run). A no-eviction
control on the identical prompt (budget larger than prompt length) stays
fully coherent on both methods, isolating the cause to eviction-during-prefill
specifically, not the prompt content, model size, or absolute RoPE position
(both conditions reach the same final position count).

This is a **broader manifestation** of the already-documented "a single
eviction can drop context the model needs mid-generation" issue on the H2O
page — that page's evidence was decode-only; this is the first confirmation
it also applies to prefill, and that it is not resolved by grace, decay,
annealing, or the RoPE-remap fix, all of which are correctness/behavior fixes
for *how eviction selects and remaps*, not for *what happens when the wrong
thing gets evicted*. No fix is attempted here — reporting it, like the
existing open issue, rather than either hiding it or overclaiming a fix.

### Metal kernel speedup, verified end-to-end (not just the isolated kernel)

`tests/metal/test_keyformer_evict.py`'s benchmark measures the kernel in
isolation (~2.8x). End-to-end through `mlx_lm.generate()` — prompt + decode,
with an assertion that eviction actually triggered on **both** the Metal and
pure-MLX runs being compared (an earlier draft of this benchmark silently
measured two no-eviction code paths at a budget the test never actually
exceeded, yielding a bogus ~1.02x "speedup" — caught and fixed before being
reported here):

| Model | Budget | Pure-MLX | Metal | Speedup |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct-4bit | 128 | 12.9 tok/s | 38.7 tok/s | **2.99x** |
| Llama-3.2-1B-Instruct-4bit | 192 | 14.8 tok/s | 32.4 tok/s | **2.19x** |
| Llama-3.2-3B-Instruct-4bit | 128 | 7.7 tok/s | 20.3 tok/s | **2.64x** |
| Llama-3.2-3B-Instruct-4bit | 192 | 8.4 tok/s | 15.9 tok/s | **1.88x** |

Speedup shrinks as budget grows relative to a fixed decode length, since a
larger budget means a smaller fraction of decode steps actually trigger
eviction (the only branch the kernel accelerates) — consistent with the
isolated-kernel benchmark's fixed-`n_kept` measurement being an upper bound,
not a guaranteed per-token multiplier.

### Annealing vs. constant tau: no measurable coherence advantage found

The hypothesis that annealing might soften tight-budget degradation was
tested directly: at a deliberately tight budget on the 3B model, constant and
annealed tau were run across 4 different noise seeds. At one seed they
diverged sharply (constant collapsed into a hard `"a a a a..."` loop, annealed
did not); at the other three seeds both degraded almost identically. This is
**seed noise, not a reproducible annealing benefit** — consistent with the
"Annealing and RoPE-remap testing" section above, which fixes score-function
fidelity to the paper, not the separate eviction-quality gap described above.
Annealing is not a fix for tight-budget degradation and is not claimed to be
one.

## When to use it

Keyformer is [H2O](../algorithms/h2o) with a safety net for late-rising tokens.
If your workload has tokens that only become important well after they enter
the cache — retrieval-style prompts where a late query re-activates early
context — the Gumbel regularizer can keep them alive where greedy accumulation
would have dropped them. If importance is stable (heavy hitters are heavy from
the start), plain [H2O](../algorithms/h2o) is simpler and the noise buys
nothing — or just set `keyformer_tau=0` and you are running H2O.

For long generations, prefer `keyformer_tau_init`/`keyformer_tau_end`/
`keyformer_anneal_steps` over a constant `keyformer_tau`: the paper's
rationale is that a near-softmax temperature during the prompt (nothing
discarded yet) and a higher, more randomized temperature later (as more
tokens compete for eviction) is a better default than either extreme held
constant. RoPE-position correctness (interior eviction no longer desyncs
rotation) and the fused Metal kernel apply automatically regardless of which
temperature mode you use — there is no separate opt-in.

| Method | Score | Late-riser protection | Path-independent |
|--------|-------|-----------------------|------------------|
| [H2O](../algorithms/h2o) | cumulative proxy-attention mass | none (greedy) | no |
| **Keyformer** | proxy-attention mass **+ Gumbel noise** | **yes (regularizer)** | no |
| [Q-Filters](../algorithms/qfilters) | projection onto frozen key-SVD direction | n/a | no |
