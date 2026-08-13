---
id: a2ats
title: A2ATS-adapted
sidebar_label: A2ATS-adapted
slug: /algorithms/a2ats
---

# A2ATS-adapted

## The problem this solves

As a language model writes, it keeps notes on every word it has already seen so
it doesn't have to reread them. Those notes are the **KV cache**, and they grow
with the conversation. A long chat can spend more memory on the cache than on
the model itself.

There are two ways to shrink it. You can throw old notes away, or you can write
every note in shorthand. A2ATS-adapted does the second: it keeps every token,
stored more cheaply.

It saves in two places:

- **Shorthand instead of full notes.** Rather than storing each token's numbers
  exactly, it stores the nearest entry from a shared lookup table of common
  patterns — like writing "see pattern #57" instead of copying out 128 numbers.
  That table is the **codebook**, and you have to build it in advance from your
  own data. This is the tradeoff: the shorthand is close to the original, not
  identical.
- **Less bookkeeping for old tokens.** Models track *where* each word sits in
  the sentence. A2ATS does that exactly for recent tokens and approximates it
  for older ones, on the bet that what matters most is usually what was said
  recently.

That bet is the whole method. It pays off for chat histories, code completion,
and running summaries. It works against you when the answer is buried far back
in a long document.

## Before you start

Three things are worth knowing up front, because they decide whether this is
the right pick.

**You have to run a calibration step first.** The codebook has to be built from
a sample of your own model's data. It's a one-time offline job covered in
[Calibration](#calibration-one-time-setup), but you can't skip it. Without it
the code falls back to a random table that exists only so the tests can run,
and your output will be visibly bad. If you want something that works with no
setup, use [TurboQuant RVQ](../algorithms/rvq) instead.

**It won't make generation faster.** The speedups in the original paper come
from a custom GPU kernel written for NVIDIA hardware that this project doesn't
have. On Apple Silicon you're trading a little accuracy for a smaller cache,
not for speed. Expect decode to get slightly *slower* as context grows.

:::danger[And right now it won't shrink your actual memory use either]
This is the one people get caught by. The page reports a compression ratio, and
the number is real arithmetic — but it describes how small the data *could* be,
not how much memory your process actually uses.

The reason: after compressing each token, the code immediately expands it back
to full size before storing it, because the "recent vs. old" bookkeeping has to
be redone every single step against your current position. So the cache in
memory still holds full-size data.

That makes this useful for **measuring** what the technique would cost you in
accuracy and what footprint it could reach — real numbers you can plan
with — but not yet for saving RAM today. If you need your memory use to drop
now, use [VecInfer](../algorithms/vecinfer) or
[TurboQuant RVQ](../algorithms/rvq).
:::

## Quick start

Work through [Calibration](#calibration-one-time-setup) first — without a
trained codebook this produces near-random quantization. Once you have
`a2ats_codebook.npz`:

```python
import mlx_lm
import numpy as np
import mlx.core as mx
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# head_dim must match the model. Llama-3.2-3B uses 128; read it off the config
# rather than guessing, since a mismatch raises at construction time.
head_dim = model.args.head_dim or (model.args.hidden_size // model.args.num_attention_heads)

data = np.load("a2ats_codebook.npz")

config = KVCacheConfig(
    method="a2ats",
    head_dim=head_dim,
    a2ats_window=128,  # recent tokens get exact position encoding
    a2ats_codebook=mx.array(data["codebook"]),
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches

response = mlx_lm.generate(
    model,
    tokenizer,
    prompt="Summarise this conversation so far.",
    max_tokens=200,
)
```

You only need to set two things yourself: `head_dim`, which comes from the
model, and `a2ats_codebook`, which comes from calibration. Everything else
defaults to the values the original paper used, and those are sensible to start
with.

One rule about `head_dim`: it has to be an even number, and divisible by
`a2ats_sub_dim` (which defaults to `8`). `128` and `64` both work. If you get
it wrong the code raises an error immediately instead of quietly producing bad
output, so you'll know.

## Should you use this?

**It's a good fit if** your model mostly needs recent context — conversations,
iterative editing, working on the file that's currently open — and you're able
to run that one-time calibration job.

**Look elsewhere if:**

- **You need to find things buried far back**, like "what did the contract say
  on page 3". This is exactly where the approximation hurts most. In our
  testing, that kind of workload had roughly **8x** more error than a version
  doing all the work exactly, versus about **3x** when the useful information
  was nearby.
- **You can't run calibration** → [TurboQuant RVQ](../algorithms/rvq) needs
  none.
- **You want accuracy that doesn't depend on distance** →
  [VecInfer](../algorithms/vecinfer) and
  [CommVQ-adapted](../algorithms/commvq) treat every position the same way.

### The one setting worth tuning: `a2ats_window`

This is how many recent tokens get exact treatment. Tokens inside the window
are handled with *no* approximation at all — bit-for-bit identical to not using
this method. Everything older gets the cheap path.

So the window is the slice of your conversation you're paying full price for.

| `a2ats_window` | What happens |
|---|---|
| Larger (say 512) | Better accuracy, smaller savings |
| Smaller (say 64) | Bigger savings, more error on older tokens |
| Bigger than your context | Nothing is approximated at all |
| `0` or less | Everything is approximated (don't do this) |

Rule of thumb: if you know roughly how far back your model needs to look, set
the window to cover that.

## Calibration (one-time setup)

This is the step you can't skip. You're building the lookup table of common
patterns — the codebook — that the shorthand refers to.

The idea is simple: run a batch of text through your model that looks like what
you'll actually use it for, watch the numbers it produces internally, and find
the few hundred most representative patterns. Save those. From then on, every
token gets stored as "the closest one of these."

You do this once and reuse the saved file forever, as long as you keep using
the same model.

:::warning[The example below uses random numbers as a placeholder]
`keys_calib` here is filled with random data so the snippet runs on its own. A
codebook built from random numbers is worthless. Replace it with real values
collected from your model on text that resembles your actual workload.
:::

```python
import numpy as np
import mlx.core as mx
from veloxquant_mlx.allocators.vecinfer import train_codebook

sub_dim = 8  # must match a2ats_sub_dim
bits = 8  # must match a2ats_codebook_bits

# Collect real key activations from a calibration prompt set —
# shape [n_tokens, n_heads, head_dim].
keys_calib = mx.array(np.random.default_rng(0).standard_normal((4096, 8, 128)).astype(np.float32))

codebook = train_codebook(keys_calib.reshape(-1, sub_dim), n_centroids=2**bits, seed=42)
np.savez("a2ats_codebook.npz", codebook=np.asarray(codebook))
```

Then load it in your config:

```python
data = np.load("a2ats_codebook.npz")
config = KVCacheConfig(
    method="a2ats",
    head_dim=128,
    a2ats_codebook=mx.array(data["codebook"]),
)
```

### Optional: query-aware calibration

A2ATS can also bias codebook selection toward the directions your queries
actually look in. This needs a second calibration artifact, the query
second-moment matrix:

```python
from veloxquant_mlx.quantizers.a2ats import a2ats_query_second_moment

# collected_queries: [n_tokens, a2ats_sub_dim] query sub-vectors
h = a2ats_query_second_moment(collected_queries)

config = KVCacheConfig(
    method="a2ats",
    head_dim=128,
    a2ats_sub_dim=8,
    a2ats_codebook=...,
    a2ats_query_h=h,
)
```

With `a2ats_query_h` supplied you get the paper's actual query-aware objective.
Without it, a simpler cosine-similarity approximation runs instead; see
[Limitations](#limitations) for what differs.

To turn query-awareness off entirely and use plain nearest-centroid
quantization:

```python
config = KVCacheConfig(method="a2ats", head_dim=128, a2ats_use_query_aware=False)
```

## Configuration reference

Most people set the first three and leave the rest alone.

**The ones you'll actually touch:**

| Setting | What it does |
|---|---|
| `head_dim` | **Required.** Comes from your model. Must be even, and divisible by `a2ats_sub_dim` |
| `a2ats_codebook` | Your calibrated lookup table. Leave it out and you get a random one that produces garbage |
| `a2ats_window` | How many recent tokens get exact treatment. Default `128` |

**The ones you can safely ignore at first:**

| Setting | What it does |
|---|---|
| `a2ats_sub_dim` | How many numbers get bundled into one lookup. Default `8` |
| `a2ats_codebook_bits` | Lookup table size, as `2**bits` entries. Default `8` |
| `a2ats_use_query_aware` | Whether to favor tokens the model is likely hunting for. Default `True` |
| `a2ats_query_h` | Optional extra calibration that upgrades the above from approximate to exact. Must be `sub_dim × sub_dim` |
| `a2ats_beta` | Balance between accuracy and relevance, `0` to `1`. Only used when `a2ats_query_h` is absent. Default `0.5` |
| `a2ats_retrieval_fraction` | What share of tokens get the special treatment, `0` to `1`. Default `0.20` |
| `a2ats_b` | Stand-in distance used for old tokens. Default `2048` |
| `a2ats_rope_base` | Position-tracking frequency. Match your model's if you change it. Default `10000.0` |

You get one cache per layer of the model, and `KVCacheBuilder.for_model` builds
them all for you.

## Checking how much you saved

The cache keeps a running tally you can print at any time:

```python
cache = caches[0]
print(f"compression: {cache.compression_ratio:.1f}x")
print(f"effective bits/element: {cache.assigned_avg_bits:.2f}")
print(f"codebook overhead: {cache.codebook_bytes / 1024:.1f} KB")
print(f"tokens seen: {cache.tokens_seen}")
```

Remember these describe how small the data *could* be, not your actual memory
use — see the [warning at the top](#before-you-start).

| What you can read | What it tells you |
|---|---|
| `compression_ratio` | How many times smaller the compressed form is |
| `compressed_key_bytes` / `compressed_value_bytes` | Size in compressed form |
| `fp16_key_bytes` / `fp16_value_bytes` | Size without any compression |
| `codebook_bytes` | Size of the lookup table itself |
| `assigned_avg_bits` | Bits spent per number stored |
| `tokens_seen` / `tokens_retrieved` | How many tokens have gone through |

### What you get with the defaults

The compression is decided entirely by your settings, not by your data, so you
can work it out before running anything. Two settings control it: `sub_dim`
(how many numbers get bundled into one lookup) and `bits` (how big the lookup
table is).

| `a2ats_sub_dim` | `a2ats_codebook_bits` | Bits per number | Smaller by | Table size |
|---|---|---:|---:|---:|
| `8` (default) | `8` (default) | 1.00 | 16x | 4 KB |
| `4` | `8` | 2.00 | 8x | 2 KB |
| `8` | `4` | 0.50 | 32x | 0.25 KB |

Out of the box you're storing about one bit per number instead of sixteen.

The lookup table itself is a one-off cost that doesn't grow with your
conversation, so on any real workload it rounds to nothing. And the pattern in
that table is the usual one: bigger compression means rougher approximation, so
pushing past the defaults buys space with quality.

## Troubleshooting

**"head_dim must be even"** — `head_dim` has to come from your model, and the
method needs an even number. Don't hardcode it; read it from the model config
like the [Quick start](#quick-start) does.

**"head_dim not divisible by a2ats_sub_dim"** — these two numbers have to
divide evenly. `a2ats_sub_dim` defaults to `8`, so `head_dim` needs to be a
multiple of 8. Either you've got the wrong `head_dim`, or you changed `sub_dim`
to something that doesn't fit.

**The output is gibberish, or a codebook shape error** — almost always a
calibration mismatch. The `sub_dim` and `bits` you used when *building* the
codebook must be the same ones you use when *running*. A codebook built with
`sub_dim=8` won't work with `sub_dim=4`. And if you didn't pass
`a2ats_codebook` at all, you're on the random fallback, which produces
nonsense by design.

**"a2ats_query_h must have shape..."** — this optional matrix has to be square,
sized to match your `a2ats_sub_dim`. With the default of 8, it needs to be 8×8.

**Everything works but memory use didn't drop** — expected. See the
[warning at the top](#before-you-start).

**It's slower than not using it** — also expected, and it gets more noticeable
as context grows. See [Limitations](#limitations).

## Limitations

Read this before you rely on it for anything real.

**The approximation genuinely costs you accuracy.** This isn't free. In our
tests, older tokens came out meaningfully less accurate — roughly 3x more error
when the useful context was nearby, and 8x when it was far away. Recent tokens
inside the window are perfect; everything older pays. The
[Benchmark](#benchmark) has the exact numbers.

**"Query-aware" here is weaker than it sounds.** The method is supposed to pay
extra attention to tokens your model is actually looking for. But the plumbing
in mlx_lm never tells the cache what the model is looking for — it only hands
over the tokens being stored. So the code substitutes a rough stand-in. Several
other methods here ([AMC-adapted](../algorithms/amc),
[H2O](../algorithms/h2o), [SnapKV](../algorithms/snapkv)) have the same
limitation, so it's a known gap rather than a flaw unique to this one.

**There's a better and a worse version of that feature, and you get the worse
one by default.** If you supply the optional `a2ats_query_h` (see
[query-aware calibration](#optional-query-aware-calibration)) you get the exact
method from the paper. Without it you get a rough approximation, and one of its
knobs, `a2ats_beta`, becomes sensitive to the scale of your data — meaning a
value that works on one model may not transfer to another.

**Don't panic if query-awareness looks "worse" in the numbers.** It deliberately
picks entries that aren't the closest match, in exchange for being more useful
for the tokens the model actually cares about. So a pure accuracy-of-reconstruction
measurement will always make it look slightly worse. That's the trade working as
designed, not a bug — but it does mean our offline benchmark can't show you the
upside.

**It never throws anything away.** Every token is kept. Some other methods on
this site work by deleting old tokens; this one doesn't. If you were hoping for
eviction, that's [H2O](../algorithms/h2o) or [SnapKV](../algorithms/snapkv).

**It gets slower as your conversation gets longer.** Whether a token counts as
"recent" depends on where you are right now, which changes with every word
generated. So the work can't be done once and cached — each step redoes it
across the entire history. The per-token cost is small, but it scales with total
conversation length, not with what you just added.

**None of this has been tested on a real model.** Our benchmark is synthetic —
it runs the math on generated data without loading a language model. The claims
about real-world quality come from the original paper, on hardware and workloads
this project hasn't reproduced. Treat the accuracy numbers as directional.

## Benchmark

We tested two situations: one where the useful information is recent
(`local_recency`), and one where it's far back (`long_range_dependent`). Lower
numbers are better, and the "Ratio" column is what matters — how many times
worse this method is than doing everything exactly.

The window was deliberately set very small (16 tokens) here to make the effect
visible.

| Situation | Which tokens | This method | Exact | How much worse |
|---|---|---:|---:|---:|
| Useful info nearby | all of them | 3.929 | 1.347 | **2.9x** |
| | recent ones | 8.248850 | 8.248850 | **identical** |
| | older ones | 3.830 | 1.080 | 3.5x |
| Useful info far back | all of them | 8.384 | 1.014 | **8.3x** |
| | recent ones | 1.090521 | 1.090522 | **identical** |
| | older ones | 8.762 | 1.006 | 8.7x |

Two things to take from this.

**Recent tokens really are untouched.** Those rows match the exact version to
seven decimal places. The promise that the window is lossless holds up.

**The catch is that "recent" is a small slice.** With a 16-token window, about
92% of a 200-token conversation counts as old, and 96% at 400 tokens. All the
error lives there. This is why the window setting matters so much: it's
literally the fraction of your conversation that stays perfect.

Also worth knowing: this gap isn't something you can tune away. We swept the
`a2ats_b` setting across its whole useful range and it never closed. Squashing
every old token's real position into one shared stand-in loses information, and
no amount of knob-turning gets it back.

The benchmark produces identical results across runs, and it doesn't load a
language model — so it isn't a reproduction of the original paper's real-world
quality or speed claims.

## How it works

If you want to know what happens under the hood, each time a batch of tokens
comes in:

1. **Pick out the important ones.** A fraction of the tokens (20% by default)
   are flagged as likely to matter, and get more careful treatment in the next
   step.
2. **Look everything up in the table.** Each token is matched to the closest
   entry in your codebook. The flagged ones from step 1 use a smarter matching
   rule that weighs what the model tends to search for; the rest just take the
   nearest match.
3. **Store the looked-up version.** From here on, the cache holds the
   approximation rather than the original.
4. **Redo the position bookkeeping, every step.** Recent tokens get their real
   positions. Older ones get left alone, with the correction applied elsewhere.
   This has to happen fresh each time because "recent" keeps changing as you
   generate.
5. **Values take the simple path.** The second half of each cache entry skips
   the position handling entirely — it never needed it — and skips the
   importance sorting too.

## How it compares

| Method | How it handles positions | Focuses on likely-needed tokens? |
|---|---|:---:|
| [VecInfer](../algorithms/vecinfer) | Doesn't touch them | No |
| [CommVQ-adapted](../algorithms/commvq) | Builds the handling into the lookup table, same for every token | No |
| **A2ATS-adapted** | **Exact for recent, approximate for old** | **Yes, for a subset** |

[CommVQ-adapted](../algorithms/commvq) attacks the same problem from a
different angle: it constrains what the lookup table can represent so positions
work out automatically, treating every token the same. A2ATS instead varies the
effort by how old the token is. In principle you could combine the two ideas,
though nobody here has tried.

## For contributors — paper fidelity

<details>
<summary>Correspondence to the source paper, and where this port deviates</summary>

*Inspired by* ["A2ATS: Retrieval-Based KV Cache Reduction via Windowed Rotary
Position Embedding and Query-Aware Vector Quantization" (He, Xing, Wang, Xu, Wu,
Zhou, Liu, Xue, Li — **ACL 2025 Findings**)](https://aclanthology.org/2025.findings-acl.644/).
This is A2ATS-adapted (VeloxQuant-MLX implementation), not a faithful port. It's
a normal-track method — a live-verified peer-reviewed venue, no exception needed
(unlike [AMC-adapted](../algorithms/amc) or
[NestedKV-adapted](../algorithms/nestedkv)).

**Windowed RoPE (Eq. 11–12).** Far keys are returned unrotated (`k̃_i = k_i`,
Eq. 12); the constant `R_b` encoding "far" relative position rides on the query
(Eq. 11, `u_ij = q_i R_b k_j^T`). This decoupling is what makes a shared
codebook viable across inputs (§3.1, Observation 2). `a2ats_window` is `w` and
`a2ats_b` is `b`, independent knobs as in the paper's §5.1 (`w=64`, `b=2048`).

**Query-aware VQ (Eq. 13–18).** `a2ats_h_weighted_assignment` minimizes
`(k̃ − c) H (k̃ − c)ᵀ` where `H = E[q̃ᵀq̃]`, computed exactly via the
Eq. (15)–(18) Cholesky identity rather than approximated. Requires
`a2ats_query_h`. The cosine blend used otherwise is a substitute: its cosine
term is a constant per-centroid bias rather than a per-token coupling, and
`beta` is scale-dependent.

**Per-step re-rotation.** Eq. (11) makes near/far a function of the advancing
decode query, so the split can't be baked into stored keys — doing so freezes
each token's class at write time. This port stores pre-RoPE keys and re-applies
windowed RoPE each step, costing an `O(total_tokens)` pass.

**Not implemented:**

- No CUDA kernel fusion; pure MLX throughout.
- No automatic codebook or `H` calibration. `a2ats_query_second_moment` computes
  `H` from queries you collect, but nothing collects them for you.
- No query-side `R_b` inside the cache. `far_query_rope` is exposed, but
  composing Eq. (11)'s two halves is left to callers with real query access.
- No composition with [CommVQ-adapted](../algorithms/commvq)'s RoPE-commuting
  codebook constraint.
- No trained-model perplexity/throughput/retrieval-accuracy benchmark.

**Evidence.** 67 tests across `tests/quantizers/test_a2ats_rope.py` (17),
`tests/quantizers/test_a2ats.py` (19), and `tests/cache/test_a2ats_cache.py`
(31):

- `test_windowed_rope_outside_window_returns_key_unrotated` — far keys equal the
  *input*, not merely differ from exact RoPE. The weaker form let a real bug
  pass ([#29](https://github.com/rajveer43/VeloxQuant-MLX/issues/29)).
- `test_windowed_rope_far_tokens_are_position_independent` — far keys carry no
  positional information, the §3.1 shared-codebook premise.
- `test_far_query_rope_b_is_independent_of_window` /
  `test_far_query_rope_reconstructs_paper_attention_score` — `b` is a real
  separate knob, and the halves compose back into Eq. (11).
- `test_token_rotation_updates_as_decode_position_advances` — near/far tracks the
  advancing query rather than freezing at write time.
- `test_h_weighted_assignment_matches_bruteforce_eq14` /
  `test_h_identity_reduces_to_plain_nearest_centroid` /
  `test_h_weighted_differs_from_plain_vq_under_anisotropic_h` — the Cholesky
  route computes Eq. (14) exactly, reduces to plain VQ when `H ∝ I` (§3.2's
  premise), and diverges under anisotropic `H`.
- `test_window_zero_always_unrotated` /
  `test_window_exceeds_seqlen_always_exact` — both degradation boundaries.
- Config validation, byte accounting, determinism across mixed prefill+decode,
  `for_model` propagation, factory dispatch.

</details>

## See also

- [Calibration guide](../guides/calibration)
- [VecInfer](../algorithms/vecinfer) — uniform exact position encoding
- [CommVQ-adapted](../algorithms/commvq) — codebook-constrained alternative
- [Algorithm overview](../algorithms/overview)
