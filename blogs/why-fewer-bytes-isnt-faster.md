# I Cut KV Cache Traffic by 88% and Inference Got *Slower*. Here's the Arithmetic I Should Have Done First.

*Why "compress the KV cache, go faster" is a reasonable intuition that fails on an 8B model at 4K context — and the two-line calculation that predicts it before you run anything.*

---

## The Intuition

It goes like this, and it's not stupid:

1. The KV cache is big.
2. Reading it from memory is slow.
3. So read less of it.
4. Go faster.

I built a [profiling harness](./energy-profiling-harness.md) to test exactly this on an Apple M4. Three arms on `Qwen3-8B-4bit` at 4,096 tokens of context: FP16 baseline, KIVI 4-bit quantization, and Q-Filters eviction at a 512-token budget.

The compression worked. Per decode step, the derived KV traffic:

| Arm | KV bytes read/step | vs. baseline |
|---|---|---|
| FP16 baseline | 633.5 MB | 100% |
| KIVI 4-bit | 210.9 MB | 33% |
| Q-Filters, budget 512 | 75.5 MB | **12%** |

An 88% traffic reduction for Q-Filters. Step 3 of the intuition, delivered.

Then step 4:

| Arm | Throughput | vs. baseline |
|---|---|---|
| FP16 baseline | 16.77 tok/s | — |
| KIVI 4-bit | 15.22 tok/s | **−9%** |
| Q-Filters, budget 512 | 10.60 tok/s | **−37%** |

Both compressed arms are **slower**. The one that cut traffic the most is slower by the widest margin.

---

## The Two-Line Calculation

Here's what I should have run before writing any benchmark code.

An M4 has roughly **120 GB/s** of memory bandwidth. At 16.77 tok/s, one decode step takes **59.6 ms**. So how long does reading the KV cache actually take?

```
633.5 MB / 120 GB/s = 5.28 ms
```

**5.28 ms out of a 59.6 ms step. The entire KV cache read is 8.9% of decode time.**

The other 91% is the 8B model's weights streaming through the same memory bus, plus the attention math itself. Which means:

| Arm | KV read time | Share of step |
|---|---|---|
| FP16 | 5.28 ms | 8.9% |
| KIVI 4-bit | 1.76 ms | 2.9% |
| Q-Filters 512 | 0.63 ms | 1.1% |

Cutting 633 MB to 75 MB saves **4.65 ms**. That's the entire prize. Even a hypothetical compressor with *zero* decode cost — free dequantization, free eviction scoring, no overhead at all — could speed this workload up by at most **7.8%**.

This is Amdahl's law wearing a different hat. You cannot optimize your way past a component that was only 9% of the runtime. The upper bound was fixed before any compression algorithm entered the picture.

---

## Where the Time Actually Went

If the ceiling is a 4.65 ms saving and KIVI lost 6.1 ms, the overhead is knowable by subtraction:

| Arm | Step time | Traffic saved | Net change | **Implied overhead** |
|---|---|---|---|---|
| FP16 | 59.6 ms | — | — | — |
| KIVI 4-bit | 65.7 ms | −3.52 ms | +6.1 ms | **~9.6 ms** |
| Q-Filters 512 | 94.3 ms | −4.65 ms | +34.7 ms | **~39.4 ms** |

KIVI spends about **9.6 ms of extra compute to save 3.5 ms of memory traffic**. It's a losing trade by roughly 3:1, and no amount of tuning the bit-width fixes it, because the prize on the other side is capped at 5.28 ms.

The work is real and unavoidable. KIVI stores 4-bit codes with per-group scales and zero-points; before attention can run, every one of those has to be dequantized back to fp16 — across 36 layers, on every single step. Q-Filters has to score every resident token against its projection filter and select which to evict, also per layer, per step.

That work scales with the *same* tokens the compression is shrinking. You're paying compute proportional to cache size in order to reduce memory traffic proportional to cache size. Whether that trade wins depends entirely on the ratio of compute throughput to memory bandwidth on your hardware — and on Apple Silicon, with unified memory and a lot of GPU compute, it doesn't win here.

---

## When the Intuition *Does* Hold

The 8.9% figure is not a law of nature. It's specific to this configuration, and the thing that moves it most is context length.

KV traffic grows linearly with sequence length. Model weights don't — they're the same 8B parameters whether you're at 4K or 64K. So the KV share of each step climbs as context grows:

| Context | KV bytes/step (fp16) | Direction |
|---|---|---|
| 4,296 | 633 MB | 8.9% of step |
| 32,768 | ~4.8 GB | substantially higher |

At long enough context, KV traffic *does* become the dominant term, and compression's arithmetic inverts. This is not a coincidence — it's what these methods were designed for. The Q-Filters paper targets long-context workloads specifically. Benchmarking it at 4K is arguably testing it outside its intended regime.

**I have not run that experiment.** The harness supports it (`--context-tokens 32768`), and the honest position is that long-context behavior is an open question here, not a claim I get to make because the trend line points the right way.

---

## What Compression Actually Bought

Throughput was never the primary claim. Memory was:

| Arm | Peak memory |
|---|---|
| FP16 baseline | 6,846.7 MB |
| KIVI 4-bit | **6,701.7 MB** |

At 4K the gap is modest, because model weights dominate the footprint. But the KV portion is the part that grows — at 32K the difference is measured in gigabytes, and on a 16 GB Mac that is precisely the difference between a working long-context session and an out-of-memory crash.

That's the trade this configuration actually offers: **9% throughput for headroom that scales with context.** For a lot of local-inference use cases that's a good deal. It just isn't the "compress and go faster" story.

(Q-Filters is the odd one out — 7,981 MB, *more* than the baseline, because its scoring state and retained-window buffers outweigh what a 512-token budget gives back against a 4,296-token sequence. Eviction at a budget that aggressive, on a context that short, isn't buying anything on either axis.)

---

## The Takeaway

**Do the bandwidth arithmetic before you build the benchmark.** Two numbers — bytes moved per step, and memory bandwidth — give you the ceiling on what any compression method can possibly deliver. If that ceiling is 8%, no algorithm is going to hand you 2×, and you've learned it in thirty seconds instead of a week.

It's also worth being precise about what "compression is slower" means, because it's easy to over-generalize. It does **not** mean the caches are broken — they compress exactly as designed, verified by both the derived model and peak memory. It does **not** mean KV compression is useless — the memory headroom is real and it's the actual product. And it does **not** mean the result generalizes to long context, which is where these methods live and where I haven't measured.

It means one specific thing: **at 4K context on an 8B model on an M4, KV traffic is not the bottleneck, so removing it doesn't help.**

There's a related question this doesn't answer at all. A slower arm can still draw less power per unit time and win on total joules — throughput and energy are different measurements. That's the one I built the harness for in the first place, and [it's still open](./energy-profiling-harness.md).

---

*All figures: Apple M4 (120 GB/s), MLX 0.32.0, `mlx-community/Qwen3-8B-4bit` (36 layers, 8 KV heads, head_dim 128), 4,096-token prompt, 200 decode tokens, 3 interleaved reps, medians. KV bytes/step are **derived** from cache geometry — MLX exposes no bytes-moved counter. Harness: `benchmark_scripts/benchmark_energy.py`.*
