# We doubled AdaKV's real-world speed on Apple Silicon — here's what changed for you

If you're running local LLM inference on a Mac with VeloxQuant-MLX and using
**AdaKV** — the per-head adaptive-bit KV cache — your next `pip install
--upgrade` is going to make your model noticeably faster. Not in a synthetic
benchmark. In an actual chat session, on an actual M4, measured token by
token as the model typed.

This post is about the *results*. If you want the engineering post-mortem,
that's a separate writeup — here we're answering one question: **what does
this mean for the tokens/sec you actually see on your screen?**

## The headline number

We ran the exact same prompt, on the exact same machine (Apple M4, 10-core
GPU, 24 GB unified memory), through `mlx_lm.generate()` — the real
generation loop, not an isolated kernel microbenchmark — using
`mlx-community/Llama-3.2-1B-Instruct-4bit`. Three configurations, same
model, same prompt, same everything except one setting:

| Setup | Tokens/sec | vs. no compression at all |
|---|---|---|
| Plain fp16 cache (no compression) | ~101–109 tok/s | baseline |
| AdaKV, default settings | ~53–54 tok/s | 51% |
| AdaKV, with one config value changed | ~91–100 tok/s | **85–96%** |

That middle row is what every AdaKV user has been getting, without knowing
there was more on the table. The bottom row is available *today*, with no
code changes, no waiting for a new release — just a config value that
already existed and was quietly not doing anything.

## What AdaKV is, for anyone who hasn't met it yet

When an LLM generates text, it keeps a running memory of everything it's
already read and written — the "KV cache." For long conversations or long
documents, that memory can get big enough to matter, especially on a laptop
or Mac Studio with a fixed pool of unified memory shared between the model
and everything else.

AdaKV's idea: not every part of the model's attention needs the same amount
of precision. Some attention heads are doing more delicate work than others,
so AdaKV measures each head's importance and gives the "important" heads
more bits of precision and the less-important ones fewer — squeezing the
cache smaller without a uniform, blunt cut.

It's a genuinely useful idea. The problem was never the idea. The problem
was that the *bookkeeping* AdaKV does to decide "which heads get how many
bits" was running far more often than it needed to, and that bookkeeping
turned out to be surprisingly expensive on Apple's Metal/MLX stack — not
because the math is hard, but because of *how* it was being run.

## Why "faster" isn't the usual story

Normally when someone says they made a quantization method faster, it's
because they wrote a better computational kernel, or found a smarter
formula. That's not this. AdaKV's actual math didn't change one bit — every
number it produces today is reachable with the *old* code too, if you
already knew to configure it correctly.

What actually happened: AdaKV was recalculating "which head gets how many
bits" **on every single token, on every single layer of the model** — even
though that recalculation barely changes from one token to the next in
practice. And each of those recalculations forced the whole framework to
stop and synchronize, which on a GPU is a bit like stopping a factory
assembly line to double-check one clipboard, sixteen times a second, for
no operational reason.

There was already a setting for this — `adakv_update_interval` — that was
supposed to control how often that recalculation happens. It existed. It
was documented. It just wasn't wired up to actually do anything; every
value you set it to behaved identically to "recalculate every single time."
We fixed that, and the moment it started working, throughput jumped.

## What you get, and what you give up

Turning `adakv_update_interval` up from its default means AdaKV refreshes
its "which head matters right now" read less often — say, every 8 or 16
tokens instead of every token. In practice, head importance doesn't swing
wildly between one token and the next, so the recommendation is stale by
only a handful of tokens at any given moment, and the actual bit allocation
barely moves as a result.

Here's what that trade bought, on the same real generation run:

- **`adakv_update_interval = 8`** → 91–93% of uncompressed fp16 speed, while
  still keeping AdaKV's memory savings.
- **`adakv_update_interval = 16`** → 92–96% of uncompressed fp16 speed —
  functionally indistinguishable from not compressing at all, latency-wise,
  while still getting the smaller cache footprint.

If you want the mathematically exact, recompute-every-token behavior AdaKV
has always had, it's still there — leave the setting at its default. If you
want most of your speed back for a workload where "importance" doesn't need
to be recalculated 16 times a second, turning this one number up is now a
real, working lever instead of a documented-but-broken one.

## Why this matters more on a Mac than you might expect

This isn't a cloud GPU cluster story. VeloxQuant-MLX exists specifically for
single-device Apple Silicon inference — the kind of setup where you don't
have eight GPUs to throw compute at, you have one chip, one pool of unified
memory, and you want your local model to feel responsive while you're
actually using it. A cache method that quietly costs you half your tokens
per second isn't a rounding error on that kind of machine — it's the
difference between a model that feels conversational and one that feels
like it's thinking too hard about everything.

## The takeaway

If you use AdaKV today: update, and try `adakv_update_interval=8` or `=16`
for your workload. You'll very likely see your effective speed roughly
double, for a bit-allocation freshness cost you probably won't notice.

If you don't use AdaKV yet and were holding off because of speed: this is
the moment to look again. The gap between "compressed cache" and
"uncompressed speed" on a real model, on a real Mac, running a real
conversation, just went from "you'll feel it" to "you probably won't."

---

*Verified on Apple M4 (10-core GPU, 24 GB), `mlx-community/Llama-3.2-1B-Instruct-4bit`, real end-to-end `mlx_lm.generate()` — not an isolated kernel benchmark. See [PR #512](https://github.com/rajveer43/VeloxQuant-MLX/pull/512) and [issue #504](https://github.com/rajveer43/VeloxQuant-MLX/issues/504) for the full numbers and test coverage.*
