# I Built a Harness to Measure Whether KV Compression Saves Energy. It Found Three Bugs Before It Found an Answer.

*What happened when I tried to get a real joules-per-token number for compressed KV cache on an Apple M4 — and why every one of my first three results was plausible, quotable, and wrong.*

---

## The Question

VeloxQuant-MLX compresses the KV cache. That much is measured: 41 methods, real perplexity numbers, real compression ratios. The pitch has always had a second half though, and it was never measured — **compressed KV should use less energy**. Fewer bytes moved from DRAM means less work for the memory controller, and on Apple Silicon the memory controller is a meaningful slice of the power budget.

That's a reasonable story. It's also just a story. There was no joules-per-token figure anywhere in the project, and an energy-aware inference mode was starting to get designed on top of an assumption nobody had tested.

So the task was narrow: build a measurement harness. Not a kernel, not an optimization, not an "energy mode." Just get a number. FP16 KV versus compressed KV, joules per token, on real hardware.

I got a number. Then I got a different number. Then a third. The interesting part of this post is not the answer — the energy answer is still open — it's the three ways I nearly published something false.

---

## Ground Rule: Never Invent an Energy Number

I want to be upfront about the constraint I set before writing any code, because it ended up doing all the work.

**A fabricated energy number is worse than no harness at all.** If the harness reports 0.32 J/token and that figure is wrong, it doesn't stay in the benchmark script — it goes into a design doc, then a README, then someone's decision about what to build next. A missing number blocks one thing. A wrong number corrupts everything downstream of it.

Concretely, that meant:

- Energy that can't be measured returns `None`, never `0.0`. A silent zero reads downstream as "inference is free."
- Anything derived analytically gets labelled **DERIVED**, never mixed into a table beside measured values without a marker.
- A null or negative result gets reported as-is, not retried until it looks better.

This sounds like boilerplate ethics. It isn't. Every bug below was caught by one of these rules firing, not by me being clever.

---

## The Setup

Apple M4, 25.77 GB unified memory, MLX 0.32.0, `Qwen3-8B-4bit` (36 layers, 8 KV heads, head_dim 128). Three arms:

| Arm | Cache | Mechanism |
|---|---|---|
| **A** | stock `mlx_lm` `KVCache` | FP16 baseline |
| **B1** | KIVI, 4-bit | **Quantization** — scales bytes by the bit ratio |
| **B2** | Q-Filters, budget 512 | **Eviction** — caps bytes at the budget |

B1 and B2 both reduce KV traffic, but by *different mechanisms*, and that distinction turned out to be the most useful thing the harness produces. Quantization narrows every token: traffic still grows with sequence length, just more slowly. Eviction throws tokens away: past the budget, traffic **stops growing entirely**.

There's no arm C. A fused Metal kernel was the obvious next step, and I deliberately didn't build it — its precondition is a profiled bottleneck from A and B, and optimizing a bottleneck nobody has demonstrated is how you spend a week making something faster that wasn't slow.

---

## Constraint 1: MLX Cannot Tell You About Bandwidth

The original plan called for recording "memory bandwidth" alongside GPU utilization. It turns out you can't.

The entire public surface of `mx.metal` is:

```
clear_cache, device_info, get_active_memory, get_cache_memory,
get_peak_memory, is_available, reset_peak_memory, set_cache_limit,
set_memory_limit, set_wired_limit, start_capture, stop_capture
```

Every one of those is **allocation-side**. How much memory is reserved, cached, peaked. Not one of them is **traffic-side** — bytes actually moved. And `powermetrics`, the macOS tool that reports power, gives you watts and residency, not DRAM bytes per second.

So there is no measured bandwidth number available on this platform, from either tool.

The honest response is to compute it analytically and label it loudly:

```python
def kv_bytes_per_token(config, n_layers, n_kv_heads, head_dim, seq_len) -> int:
    """Bytes of KV cache read per decode step. **DERIVED, NOT MEASURED.**

    This is an analytical model of cache geometry, not an observation of DRAM
    traffic -- MLX exposes no bytes-moved counter. It assumes each decode step
    reads the whole resident cache once, which is what dense attention does,
    and ignores tiling and cache-hierarchy effects.
    """
```

The dishonest response — the one I was one careless table away from — is to put a derived figure in a column next to measured throughput and let the reader assume both came from the same instrument. There's a real escalation path for anyone who genuinely needs measured bandwidth (`mx.metal.start_capture()` writes a GPU trace that Xcode Instruments can read, and Instruments *does* have bandwidth counters), but it's interactive and unscriptable, so it's documented rather than built on.

---

## Bug 1: The Residual Window That Wasn't There

First derived model, straightforward: 4-bit KV should read a quarter of what fp16 reads, plus a little overhead for group scales and zero-points.

Test asserted the ratio landed in `[0.25, 0.32)`. It passed. Ship it.

Then I read `KIVIKVCache._account_bytes` more carefully:

> The most recent `residual_length` tokens are kept in **fp16** (KIVI's "residual"): newly generated tokens dominate attention and are cheap to keep exact; they are quantized only once they age out of the residual window.

Default `residual_length` is **128 tokens**. My model quantized *every resident token*, so it was undercounting KIVI's real traffic — the most recent 128 tokens are fp16, at 8× the bytes I was charging them.

How much does it matter? At 36 layers, 8 KV heads, head_dim 128:

| Sequence length | Naive ratio | Correct ratio |
|---|---|---|
| 512 | 0.28 | **0.48** |
| 4,296 | 0.28 | **0.33** |
| 32,768 | 0.28 | **0.315** |

At short contexts the residual window *dominates* — the naive model reported KIVI reading 28% of baseline when it actually reads 48%. The error shrinks as the residual amortises, but the benchmark runs at 4,296 tokens, right where it's still substantial.

What makes this a good example of the failure mode: **the number was plausible**. 0.28 for 4-bit-versus-16-bit is exactly what you'd expect if you did the arithmetic on a napkin. There's nothing to notice. It's only wrong if you know KIVI keeps a residual window, and the only way to know that is to read the cache implementation instead of trusting your mental model of what "4-bit" means.

The fix bills the residual at fp16:

```python
residual = min(resident, int(getattr(config, "residual_length", 0) or 0))
n_quantized = max(0, resident - residual)
```

And the test now asserts the thing that actually characterizes the behavior — below the residual length, a 4-bit arm reads *exactly* what fp16 reads:

```python
def test_kv_bytes_per_token_residual_window_is_billed_at_fp16():
    cfg = KVCacheConfig(method="kivi", bit_width_inlier=4, residual_length=128)
    short = 64  # entirely inside the fp16 residual window
    assert kv_bytes_per_token(cfg, ...) == kv_bytes_per_token(None, ...)
```

---

## Bug 2: I Benchmarked the Fallback Path

First full run on Qwen3-8B. The results:

| Arm | tok/s | Peak MB | KV B/tok (derived) |
|---|---|---|---|
| A — fp16 | 16.36 | 6846.6 | 633,470,976 |
| B1 — KIVI 4-bit | 16.46 | 6701.6 | 197,959,680 |
| B2 — Q-Filters 512 | **8.67** | **8058.8** | 75,497,472 |

Q-Filters was **47% slower than baseline** while using **more peak memory** — despite reading an eighth of the bytes. Consistent across all three reps, so not noise.

You could write that up. "Eviction overhead exceeds its bandwidth savings on Apple Silicon" is a coherent finding, and the memory inversion even has a plausible story — scoring state and retained-window buffers on top of the cache.

It's also entirely an artifact of how I configured it.

`QFiltersKVCache` has two paths. The fast one does eviction as a single vectorized selection across all `(B, H)` groups. The fallback loops over every group **in Python**, on every decode step, across all 36 layers. Which one runs is gated by:

```python
def _can_batch(self) -> bool:
    """True when every group can be evicted in one vectorized selection.

    Requires a calibrated filter: only then is every group's filter frozen
    before token 0, so all B*H groups hold the same row count.
    """
    return self._filters is not None
```

I built the arm with `KVCacheConfig(method="qfilters", qfilters_budget=512)` and no filters. `self._filters` is `None`. Every decode step took the Python loop.

I wasn't measuring Q-Filters. I was measuring an uncalibrated fallback path that no real deployment would use — the repo's own perplexity script calibrates before running, and filters are a *constructor argument*, invisible if you only wire configs.

Calibrated, on the same hardware:

| | Uncalibrated | Calibrated |
|---|---|---|
| Throughput vs fp16 | **−47%** | **−37%** |
| Peak memory | 8058 MB (highest) | 7981 MB |
| On Llama-3.2-1B | −47% | **−10%** |

The direction of the finding survives — Q-Filters is still slower — but the magnitude was inflated, and on the smaller model the "eviction is catastrophically slow" reading collapses to "eviction costs 10%."

The lesson I'd generalize: **a benchmark arm that silently degrades is worse than one that crashes.** If `QFiltersKVCache` had raised "no filters supplied," I'd have fixed it in thirty seconds. Instead it quietly did something correct-but-different, and produced numbers with no visible defect. The harness now calibrates before timing, and if calibration fails it prints a warning that the fallback path is *not representative* rather than reporting the slow number as a result.

---

## Bug 3: The One That Nearly Got Published

With throughput solid, the last piece was energy. `powermetrics` requires root, so the harness is built to degrade: `available()` checks `os.geteuid() == 0` first, unprivileged runs return `None` everywhere, and the whole test suite (2,275 tests) stays green as a normal user.

Getting `powermetrics` to talk to Python was its own small adventure. `-f plist` couldn't be verified without root — `powermetrics` enforces its superuser check *before* validating arguments, so an unprivileged probe returns the same error for every format value including nonsense ones. When I finally captured real output under `sudo`, the parser found **1 sample out of 3**.

The reason, visible only in a hex dump:

```
b'ger>\n</dict>\n</dict>\n</plist>\n\x00<?xml version="1.0" encoding="UTF-8"?>\n'
```

`powermetrics` **NUL-terminates** each emitted plist document. Splitting on the XML header leaves a leading `\x00` on every document after the first, and `plistlib` rejects them. My parser silently kept the first and dropped the rest — no exception, no warning, just two-thirds of the data gone.

Fixed. Then the privileged run:

| Arm | tok/s | J | J/token |
|---|---|---|---|
| A — fp16 | 18.35 | 399.32 | 1.997 ±0.837 |
| B1 — KIVI 4-bit | 18.75 | 441.54 | 2.208 ±0.158 |
| B2 — Q-Filters 512 | 12.31 | 269.20 | **1.346** ±0.586 |

**There it is.** Q-Filters at 1.35 J/token against fp16's 2.00 — a 33% energy reduction. Eviction pays for itself on energy even though it costs throughput. That's a genuinely interesting result, it's directionally what the compression thesis predicted, and it was sitting in a table ready to be written up.

Look at the spreads: ±0.837 on a median of 1.997 is **±42%**. Per-rep:

| Arm | rep | decode_s | GPU mW | J | J/token |
|---|---|---|---|---|---|
| fp16 | 1 | 10.68 | 11,124 | 399.3 | 2.00 |
| fp16 | 2 | 10.90 | 10,901 | 500.8 | 2.50 |
| **fp16** | **3** | **11.44** | **3,214** | **165.9** | **0.83** |
| Q-Filters | 1 | 16.52 | 5,325 | 269.2 | 1.35 |
| Q-Filters | 2 | 16.24 | 4,891 | 226.3 | 1.13 |
| **Q-Filters** | **3** | **16.25** | **9,323** | **460.8** | **2.30** |

fp16 rep 3 did the **same work** as reps 1 and 2 — decode time 11.44s versus 10.68s and 10.90s — and reported **one-third the energy**, with GPU power at 3,214 mW against ~11,000 mW. Q-Filters rep 3 did work identical to its own reps 1 and 2 and reported **double**.

Identical compute cannot cost a third of the energy. The sampler was lying.

Two causes:

**The teardown discarded buffered output.** `__exit__` called `terminate()` and *then* joined the reader thread. `powermetrics` writes a full plist document per interval into a pipe the reader drains in 64 KiB chunks; at 100 ms over a ~40-second run that's megabytes, always with some in flight. Killing the process first threw away whatever hadn't been read, and *how much* depended on thread scheduling — which is exactly why it varied run to run.

**The integration invented energy for unobserved time.** This is the worse one:

```python
mean_w = (sum(readings) / len(readings)) / 1000.0
return mean_w * self.elapsed_s   # full wall time
```

The mean is over *surviving* samples. `elapsed_s` is the *full* run. If you collect 4 seconds of samples during a 40-second run, you compute the mean power over those 4 seconds and bill it across all 40 — inventing energy for 36 seconds nobody measured. The result is a number that looks precise and tracks nothing.

The fix integrates over the window actually covered, and exposes that coverage:

```python
@property
def coverage(self) -> float | None:
    """Fraction of wall time the samples cover, or None if unmeasured.

    1.0 means every interval was captured. Substantially less means output
    was dropped, and the energy figure is correspondingly less trustworthy.
    """
    if not self._samples or self.elapsed_s <= 0:
        return None
    return self.sampled_s / self.elapsed_s
```

Now under-sampling **under-reports** instead of fabricating, the shortfall is auditable, and the benchmark prints a warning below 90% coverage.

Why this was the dangerous one: the 33% energy win was **the result I expected**. Compression reduces bytes, bytes cost energy, so compression should save energy. Confirmation makes you stop looking. The only reason I looked was the ±42% spread — and the harness reports spread because thermal drift was a known concern, not because I anticipated this. **A confound control written for one reason caught a bug of a completely different kind.**

---

## What's Actually Established

Throughput and derived traffic, three interleaved reps, medians:

| Arm | tok/s | Peak MB | KV B/tok (derived) | J/token |
|---|---|---|---|---|
| A — fp16 | 16.77 | 6846.7 | 633,470,976 | *not yet measured* |
| B1 — KIVI 4-bit | 15.22 | 6701.7 | 210,935,808 | *not yet measured* |
| B2 — Q-Filters 512 | 10.60 | 7981.7 | 75,497,472 | *not yet measured* |

**The derived model behaves as designed.** KIVI reads 33% of baseline bytes; Q-Filters reads 12%, and unlike quantization its figure doesn't grow with sequence length. The two mechanisms are cleanly distinguishable, which was the main design goal.

**Neither compressed arm is faster. Both are slower.** KIVI costs ~9% throughput while reading a third of the bytes. Q-Filters costs ~37% while reading an eighth, and uses *more* peak memory than the baseline.

At this scale, on this hardware, **reducing KV bytes did not buy speed.** Decode at 4K context on an 8B model isn't bandwidth-bound enough for traffic reduction to dominate the per-step dequantization and eviction-scoring cost.

That's a negative result and it stays in the table. It bears directly on whether an energy-aware inference mode is worth building, and burying it would defeat the point of measuring in the first place.

**The energy question is open.** Throughput is not energy — a slower arm at lower power can genuinely win on joules, which is exactly why the column exists. The "dequantization costs more than the traffic it saves" hypothesis is now *supported on the throughput axis*. On the energy axis it is untested, because the only data I have is data I've shown you not to trust.

---

## What I'd Tell Someone Building One of These

**The rule that saved me was "never report a number you can't defend," not any particular piece of engineering.** All three bugs produced output that looked fine. None threw an exception. Two of them agreed with my priors.

**Confound controls catch things they weren't designed for.** Interleaving arms and reporting spread was for thermal drift on a fan-cooled M4. Blocked arms would have hidden the sampler bug completely — every fp16 rep would have run together, the dropout would have been absorbed into one arm's median, and the spread that made me look would never have appeared.

**Distinguish derived from measured, in the type system if you can.** `kv_bytes_per_token` says DERIVED in its docstring, in the column header, and in the footer under every table. It would be so easy to let it sit next to `tokens_per_s` and become "the bandwidth number."

**`None` is a real value; `0.0` is a lie.** Four separate tests exist solely to assert that missing energy surfaces as `None`:

```python
def test_j_per_token_is_none_not_zero_when_energy_missing():
    """A silent 0.0 would propagate downstream as 'inference is free'."""
    assert compute_j_per_token(None, 128) is None
    assert compute_j_per_token(None, 128) != 0.0
```

That looks like paranoia until you notice that a zero in an energy column is indistinguishable from a spectacular result.

**Silent degradation is worse than failure.** The Q-Filters fallback and the plist parser both did something *reasonable* with bad input instead of complaining. Both cost hours. If a code path is meaningfully slower or lossier than its fast path, it should be loud about it.

---

## Where This Goes

The harness is committed and the tests pass unprivileged, which is what matters for CI. Once a clean privileged run confirms coverage near 100%, the energy column gets filled with whatever comes out — including, quite possibly, "compression costs more energy than it saves on this hardware." That would be the outcome that most changes the plan, and it's the one I'd most want to know.

The Metal kernel still isn't built. Its precondition hasn't been met, because the profiled bottleneck hasn't been identified. That's what the harness is for.

---

*Harness: `veloxquant_mlx/profiling/`, `benchmark_scripts/benchmark_energy.py`. Guide: [Energy Profiling](https://veloxquant-mlx.netlify.app/docs/guides/energy-profiling). All numbers above are from Apple M4 / MLX 0.32.0 / Qwen3-8B-4bit, 4096-token prompt, 200 decode tokens, 3 interleaved reps.*
