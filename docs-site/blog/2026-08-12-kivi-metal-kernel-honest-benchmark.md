---
slug: kivi-metal-kernel-honest-benchmark
title: "A 5.65× Metal Kernel"
description: "A 5.65x Metal kernel for KIVI KV cache quantization made zero measurable difference end-to-end, and four separate benchmarking mistakes explain why."
date: 2026-08-12
authors: rajveer
tags: [metal, apple-silicon, mlx, gpu, performance, kivi, benchmarking]
---

# A 5.65× Metal Kernel That Made My LLM Exactly 0% Faster

*How I fused KIVI's KV-cache quantization into two bit-exact Metal kernels, measured a 1.40×–5.65× op-level speedup, watched it vanish completely end-to-end — and the four separate times my own benchmarks gave me confidently wrong answers along the way.*

---

There's a particular satisfaction in watching a GPU kernel you wrote beat the framework's version by 5×. There's a different feeling entirely when you plug it into the actual model and the tokens-per-second doesn't move at all.

This post is about both, and about the part in between: **four occasions where my benchmarks confidently told me something false.** One said the kernel was 11× *slower* than baseline. One said it was 28× faster. One said quantization consumed 97.83% of prefill time, when the real figure is about 1–2%. One showed a clean 127 MB memory saving that turned out to be nothing at all.

All four were my fault. All four produced a clean-looking number with a plausible story attached, which is exactly what made them dangerous.

The kernel is real, it's bit-exact, it ships, and I'd merge it again. But the honest headline is the one above, and the useful content is why both halves of it are true at the same time.

This is the long version, with the complete experimental record — every measurement generation, including the ones that were wrong.

---

## What we're actually optimizing

If you run a language model locally, the thing that eventually stops you isn't compute. It's the KV cache.

Every token the model has seen leaves behind a key and a value vector in every attention layer. The model needs them to attend to the past, so they stay resident for the whole generation. The cache grows linearly with context — and unlike model weights, which you load once, it grows *while you're using it*.

On a 7B model at 32k context that's several gigabytes, comparable to the quantized weights themselves. On a Mac with unified memory, it's the difference between a long conversation working and your machine swapping itself to death.

[KIVI](/algorithms/kivi) (Liu et al., ICML 2024) is one answer, and it's the baseline every other algorithm in this library gets measured against. The insight is that keys and values want to be quantized along *different axes*:

- **Keys** are quantized **per-channel** — each channel gets its own scale, computed across a group of tokens.
- **Values** are quantized **per-token** — each token gets its own scale, computed across a group of channels.

Why asymmetric? Key tensors have a few channels with consistently huge magnitudes. Quantize per-token and those outliers blow up the scale for every other channel sharing the group. Value tensors lack that structure, and per-token suits them better.

A third piece matters for everything that follows: the most recent `residual_length` tokens stay in fp16. They're what attention weights most heavily, and they're also the tokens whose group isn't full yet. Once enough fresh tokens accumulate, they get quantized as a batch and folded into the compressed store. **That batching event is a flush**, and it is the operation this entire post is about.

The quantization itself is textbook asymmetric min/max:

```
zero  = min(group)
scale = (max(group) - min(group)) / (2^bits - 1)
q     = round((x - zero) / scale)
recon = q * scale + zero
```

In MLX this is roughly eight array operations: reshape to expose the group axis, `min`, `max`, subtract, divide, `round`, `clip`, multiply, add, reshape back.

Eight operations means eight kernel launches and — the expensive part — **eight round trips to memory**. Every intermediate is materialized. The quantized codes, which never needed to exist as a full-size tensor, get written to RAM and read straight back.

That's the target. One fused kernel, one pass, no intermediates.

---

## The kernel

### The layout problem that shapes everything

KV tensors are `[batch, heads, seq, head_dim]`, row-contiguous. Flatten batch and heads and you get `[BH, S, D]`, where element `(bh, s, d)` sits at `bh*S*D + s*D + d`.

Which means the two modes face opposite problems:

- **Values (per-token)** — the group runs along `D`, the *contiguous* axis. Adjacent elements in a group are adjacent in memory.
- **Keys (per-channel)** — the group runs along `S`, the *strided* axis. Adjacent elements are `D` floats apart. Typically 128.

My first version was one kernel handling both, which meant transposing the key tensor so the token axis became contiguous, then reusing the same code path.

That transpose was the whole problem. It's a full-size materializing copy — exactly the memory traffic the kernel exists to eliminate. I'd removed eight round trips and added one large one back. On the key path, the "optimized" kernel was a net loss.

So I threw it out and wrote two kernels, one per layout.

### Kernel A — per-channel keys: one thread, one whole group

The trick is almost aggressively simple: **give each thread an entire group, and don't reduce at all.**

```cpp
uint tid = thread_position_in_grid.x;

const uint BH = x_shape[0];
const uint S  = x_shape[1];
const uint NG = (S + GROUP_SIZE - 1u) / GROUP_SIZE;   // token groups

if (tid >= BH * NG * DHEAD) { return; }

const uint d   = tid % DHEAD;
const uint r   = tid / DHEAD;
const uint grp = r % NG;
const uint bh  = r / NG;

const uint base = bh * S * DHEAD + d;
const uint s0   = grp * GROUP_SIZE;
const uint s1   = min(s0 + GROUP_SIZE, S);

float gmin =  INFINITY;
float gmax = -INFINITY;
for (uint s = s0; s < s1; ++s) {
    float v = float(x[base + s * DHEAD]);
    gmin = min(gmin, v);
    gmax = max(gmax, v);
}
```

Each thread strides by `DHEAD` — 128 floats between consecutive reads. In isolation that looks like the worst access pattern available.

But look at the indexing. `d = tid % DHEAD` means **consecutive threads take consecutive channels**. At any step of that loop, the 32 threads in a SIMD group read 32 *adjacent* addresses. The warp's access is fully coalesced. The stride is per-thread; the warp moves through memory as a solid block.

And because a thread owns its group outright, there is **no cross-thread reduction**. No threadgroup memory, no barriers, no butterfly, no transpose. The strided-looking layout turned out to need the least machinery.

### Kernel B — per-token values: one SIMD group, one quantization group

The contiguous axis wants the mirror image. Lanes split a single group and cooperate:

```cpp
uint lane = thread_position_in_threadgroup.x;
uint gid  = threadgroup_position_in_grid.x;

// Whole threadgroups exit together, so every lane still reaches the
// butterfly below — a divergent return would deadlock the shuffle.
if (gid >= x_shape[0] * S * NGD) { return; }

float gmin =  INFINITY;
float gmax = -INFINITY;
for (uint i = lane; i < GROUP_SIZE; i += 32u) {
    uint d = d0 + i;
    if (d < d1) {
        float v = float(x[row_base + d]);
        gmin = min(gmin, v);
        gmax = max(gmax, v);
    }
}

// Butterfly: after 5 XOR shuffles every lane holds the group-wide min/max.
for (uint off = 16u; off > 0u; off >>= 1u) {
    gmin = min(gmin, simd_shuffle_xor(gmin, off));
    gmax = max(gmax, simd_shuffle_xor(gmax, off));
}
```

The `simd_shuffle_xor` butterfly is the nice part. Five shuffles reduce 32 lanes to a min/max that *every lane already holds* — no broadcast step. And because lanes advance in lockstep, it needs **no threadgroup memory and no barriers**, unlike a tree reduction.

Note the comment on the bounds check. That `return` must be uniform across the threadgroup. If individual lanes bailed early, the survivors would shuffle against threads that no longer exist and the reduction would hang or return garbage. Whole threadgroups exit together, so every lane reaching the butterfly reaches it with all 32 partners intact.

At KIVI's default `GROUP_SIZE=32` this is an exact fit: one lane per element, one butterfly, done.

### Two optimizations I was sure would work, and didn't

I assumed caching each group in registers would help — read once, use twice, skip the second global load. Controlled same-process A/B said otherwise:

| kernel | register caching | result |
|---|---|---|
| per-channel (keys) | group in registers | **0.76×** at S=2048 — actively harmful |
| per-token (values) | group in registers | **1.00×–1.02×** — exactly neutral |

In the channel kernel, a whole group is `GROUP_SIZE` floats *per thread*. The occupancy that costs outweighs the saved loads, which were hitting cache anyway. In the token kernel it's split across 32 lanes, so it's cheap — and buys nothing, for the same cache reason.

Both reverted, and the `REG_SLOTS` machinery deleted. I also swept threadgroup width and found the defaults (256 for channel, 32 for token) already optimal.

:::tip[The pattern]
Two hypotheses, both plausible, both wrong, both settled in about twenty minutes by a controlled A/B rather than by argument. The measurement was cheaper than the reasoning.
:::

---

## Three ways to be off by one bit

My acceptance criterion was bit-exactness against the MLX reference — not "close enough," but *identical output*. That turned out to be the most instructive constraint in the project, because it surfaced three failure modes a tolerance test sails straight past.

### 1. FMA contraction

My first parity run failed on **192 of 300 configurations** — and every failure was off by exactly 1 ULP. That uniformity is a fingerprint: not an algorithm bug, a rounding difference.

The culprit was `q * scale + gmin`. Metal's compiler sees a multiply feeding an add and contracts it into a fused multiply-add: one instruction, one rounding. MLX does them separately, with two roundings. Same math, different result on ~0.02% of elements.

The fix is to break the pattern the optimizer looks for:

```cpp
float prod = q * scale;      // NOT an fma
out[base + s * DHEAD] = T(prod + gmin);
```

The irony is that the fused version is *more* accurate — it carries more intermediate precision. But the contract is parity with the reference, not maximum accuracy, so the less accurate version is the correct one. Uncomfortable sentence; right call. 300/300 after the fix.

### 2. Rounding mode

Metal's `round()` is half-away-from-zero. `mx.round` is half-to-even. They agree on everything except exact `.5` codes — rare in random data, and *systematically common* in real quantization, because uniform grids produce exact midpoints. The fix is `rint()`. There's a test pinning it with inputs hand-built to land on `.5`.

### 3. Padding semantics

When a group doesn't divide evenly, MLX pads the tail by **replicating the edge value**, `x[..., -1:]`, not with zeros. Pad with zeros and you've silently dragged `gmin` to 0 for every ragged group, corrupting the scale. Since every pad slot holds that same value, folding it in once is equivalent to looping:

```cpp
if (s1 < s0 + GROUP_SIZE) {
    float pad_val = float(x[base + (S - 1u) * DHEAD]);
    gmin = min(gmin, pad_val);
    gmax = max(gmax, pad_val);
}
```

All three have dedicated regression tests now. They're the kind of bug that produces *plausible* output — slightly different, never obviously broken. Without bit-exactness as the bar, all three would have shipped.

---

## The bug that made generation hang forever

My favourite failure of the project, because the fix made the code simpler and the symptom was so much worse than the cause.

`mx.fast.metal_kernel` JIT-compiles from source. I was passing shape constants through the header as `#define`s, which lets the compiler turn `tid % DHEAD` into a shift-and-mask instead of integer division. Good idea for `DHEAD` — it's the model's head dimension, fixed for the life of a cache.

I did the same for the sequence length.

The sequence length **grows by one every decode step**. So every token triggered a fresh shader compilation. Generation didn't crash and didn't error — it just stopped. I killed the process after several minutes with no output.

The fix was to pass shape as a runtime buffer. MLX provides `x_shape` for free on every input, so this meant *deleting* code, not adding it.

:::danger[Before → after]
**Before:** hung indefinitely (killed after minutes)
**After:** 1.0 second

Guarded now by a test that runs 55 sequence lengths through both kernels and asserts the dispatch cache holds exactly **2** entries — not 110. That test isn't checking performance. It's checking that one specific catastrophic bug can't come back.
:::

---

## Four benchmarks that lied

Here the post stops being about GPU programming and starts being about measurement, which is the part I'd actually want to read.

### Lie #1 — "Quantization is 97.83% of prefill"

I wanted to know how much runtime quantization accounted for, so I instrumented the call: timer before, timer after, `mx.eval()` in between to force the computation. Here's the raw output:

```
# mlx-community/Llama-3.2-3B-Instruct-4bit  layers=28

PREFILL-dominated (8k prompt, 4 new tokens):
  prefill  metal=False  wall= 20.754s  quant= 20303.3ms (97.83% of wall)  calls=224  [keys 19483.2ms / values  820.0ms]
  prefill  metal=True   wall= 20.940s  quant= 20489.1ms (97.85% of wall)  calls=224  [keys 19805.5ms / values  683.6ms]

DECODE-dominated (2k prompt, 240 new tokens):
  decode   metal=False  wall= 10.114s  quant=  4829.6ms (47.75% of wall)  calls=504  [keys  4550.0ms / values  279.6ms]
  decode   metal=True   wall=  9.952s  quant=  4682.5ms (47.05% of wall)  calls=504  [keys  4457.8ms / values  224.7ms]
```

Quantization was apparently the entire bottleneck. It was nonsense, and the mistake is in the description above: **`mx.eval()` inside the measured region.**

MLX is lazily evaluated. Operations build a graph; nothing computes until something forces it. By calling `eval()` inside my timer I wasn't measuring quantization — I was measuring *every pending operation in the graph at that moment*: attention, the MLP, the whole layer stack, all attributed to the one function that happened to trigger the flush.

:::warning[Claimed vs actual]
**Claimed:** 97.83% of prefill
**Actual:** ~1–2% of prefill

Off by roughly **fifty times**, in the flattering direction, and it looked entirely plausible.
:::

Notice too that the numbers are self-refuting if you read them properly: `metal=False` and `metal=True` report *the same* 97.8% share. A measurement that can't distinguish the two arms is measuring something other than the thing you changed.

The one genuinely useful output was incidental: `calls=224` across 28 layers on an 8k prompt means 8 flushes per layer — which revealed that **mlx_lm chunks prefill at 2048 tokens.** That number matters later.

:::tip[Lesson]
In a lazy framework, a synchronization point inside your timer measures everything the framework was putting off. Force the graph to a known state *before* you start the clock.
:::

### Lie #2 — "0.09× and 28.08×"

Fresh microbenchmark, both flush sizes:

```
per-flush cost (keys+values, H=8 D=128, 28 layers)

     S    off ms     on ms  speedup |  x28 layers off        on     saved
    32    1.1334   12.4505    0.09x |           31.7ms    348.6ms   -316.9ms
  2048   19.8718    0.7076   28.08x |          556.4ms     19.8ms    536.6ms
```

Eleven times *slower* at small sizes, twenty-eight times *faster* at large ones. I nearly wrote a whole section theorizing about launch-overhead crossovers — there's a tidy story available where fixed dispatch cost dominates at S=32 and bandwidth savings dominate at S=2048.

The story was fiction. **I had left an LLM benchmark running in the background on the same GPU.**

Same benchmark, idle GPU:

```
     S    off ms     on ms  speedup |  x28 layers off        on     saved
    32    0.5050    0.3601    1.40x |           14.1ms     10.1ms      4.1ms
  2048    3.8337    0.6785    5.65x |          107.3ms     19.0ms     88.3ms
```

Both processes were fighting for the same hardware, and contention landed unevenly across runs. These weren't noisy-around-the-truth — off by **15× in one direction and 5× in the other**, *and they looked like a coherent narrative.* That's the dangerous part. Random noise looks random. Contention produces confident, structured, wrong answers.

The downstream damage is worth showing, because the wrong numbers propagated into a wrong *prediction*:

```
--- predicted end-to-end (from the CONTENDED numbers) ---
PREFILL 8k prompt: 4 chunks x 537ms saved = 2146ms of ~21000ms  ->  10.2% faster
DECODE 240 tokens: 7 flushes x -316.9ms   = -2218ms of ~10000ms -> -22.18% "faster"

--- predicted end-to-end (from the IDLE numbers) ---
PREFILL 8k prompt: 4 chunks x 88ms saved  =  353ms of ~21000ms  ->   1.7% faster
DECODE 240 tokens: 7 flushes x 4.1ms      =   28ms of ~10000ms  ->   0.28% faster
```

A predicted 10.2% prefill win and a 22% decode *regression*, versus the truth of +1.7% and +0.28%. Had I stopped there, I'd have gone hunting for a decode regression that never existed.

:::tip[Lesson]
A GPU is one resource. Check what else is running — and be most suspicious when a surprising result arrives with a satisfying explanation already attached.
:::

### Lie #3 — the benchmark that gave four different answers

Subtler, and I think the most broadly applicable. For the same kernel and the same configuration, my attempts produced **0.31×, 1.0×, 2.16×, and 3.4×**. Not scatter around a value — four different conclusions, each internally consistent.

The root cause: I was calling the function repeatedly **on the same input tensor**.

MLX can recognize it has already computed something and reuse the result. The reference path — eight standard, individually cacheable array ops — benefits enormously. My custom kernel benefits far less. So the "baseline" was quietly handed a shortcut the kernel couldn't take, and every extra repetition widened the gap. Layering best-of-N on top amplified it further, because best-of-N systematically selects the run where caching helped most.

I got this badly wrong. **I concluded the kernel was slower, set the feature flag off, and wrote that conclusion into the code and the tests.** It was only when two independently-designed methods disagreed with me that I went back:

1. **Interleaved A/B** — alternate on/off inside one process, so drift and thermal state hit both arms equally.
2. **Rotating input pool** — cycle 20 distinct tensors so nothing can be reused.

Both landed at **1.36×–2.14×**, agreeing with each other and disagreeing with me. I reverted the flag and corrected the tests.

The benchmark that ships in `test_kivi_quant.py` now does both, and its docstring explains both traps so the next person doesn't re-derive them.

:::tip[Lesson]
If your benchmark reuses inputs, you're partly measuring your framework's cache. And when two well-designed methods agree against your conclusion, the conclusion is what's wrong.
:::

### Lie #4 — the 127 MB memory saving that wasn't

This one I caught only because I ran a third model.

Llama's peak memory, from a clean interleaved run:

```
  peak memory (GB): fp16=2.604  off=2.726  on=2.599
```

A **127 MB reduction** with the kernel on. And there's a beautiful explanation sitting right there: the fused kernel eliminates MLX's intermediate tensors, so of course the high-water mark drops. Mechanistically plausible, exactly the result I wanted, and it would have made this post better.

Then the other two models came back:

| model | kernel off | kernel on | delta |
|---|---|---|---|
| Llama-3.2-3B | 2.726 GB | 2.599 GB | **−127 MB** |
| Qwen2.5-7B | 5.080 GB | 5.130 GB | **+50 MB** |
| Mistral-7B | 4.921 GB | 4.941 GB | **+20 MB** |

Two of three moved the *opposite* direction. It's allocator noise — MLX's memory pool responds to allocation ordering in ways unrelated to which kernel ran, and 127 MB out of 2.7 GB sits well inside that.

:::warning[The near-miss]
If I'd only run the model named in the original issue, I'd have shipped a false claim with a compelling mechanism attached. **The third model is what turns a result into a finding** — and the strongest argument for running it is precisely when the first one already told you what you hoped to hear.
:::

There's a deeper reason this had to be noise, which I'll come back to at the end: **quantize-then-dequantize cannot reduce peak memory, by construction.**

---

## The full end-to-end record

Four generations of end-to-end measurement, in the order I ran them. I'm including the early ones because their disagreement is the point.

### Generation 1 — single-shot, three prompt lengths

First real run. Single-shot timings, no repeats, `Llama-3.2-3B-Instruct-4bit`, 28 layers, 8 KV heads, `head_dim=128`:

```
## PREFILL (prompt tokens/sec)
prompt tok       fp16  kernel off  kernel on  on vs off  flush/layer
       559      488.2       481.6      486.0      1.01x          512
      2059      463.4       451.7      465.7      1.03x         2016
      8209      386.4       355.9      351.6      0.99x         8160

## DECODE (generation tokens/sec, 120 tokens)
config             tok/s  vs fp16   peak MB  KV comp
fp16               46.06     100%       0.0        -
off                46.00     100%       0.0    4.99x
on                 44.24      96%       0.0    4.99x

decode kernel on vs off: 0.962x
```

Two problems. First, `peak MB` reads `0.0` — a unit bug on my side: `mlx_lm.stream_generate` reports `peak_memory` in **GB**, and I was dividing by `1024**2` as though it were bytes. The raw JSON shows the real values hiding at `2.48e-06`.

Second, and more importantly: **decode at 0.962× looks like a 4% regression.** Single-shot numbers on one prompt, with no repeat structure — nowhere near enough to distinguish a real regression from thermal drift. Generation 3 is what settles it.

### Generation 2 — the sync-instrumented run

Lie #1 above. Produced the 97.83% figure, which was wrong, and the 2048-token prefill chunking discovery, which was right and load-bearing.

### Generation 3 — repeated runs with medians

Same model, but now multiple repeats per configuration reporting median/min/max, so spread is visible:

```
## PREFILL (prompt tok/s, max_tokens=4)
  prompt=2065 tok  (~2 chunks of 2048)
    fp16  median=  468.49  min=  448.16  max=  473.22   vs fp16 100.0%
    off   median=  451.76  min=  420.91  max=  464.51   vs fp16  96.4%
    on    median=  460.48  min=  424.83  max=  470.01   vs fp16  98.3%
    -> kernel on vs off: 1.019x

  prompt=8213 tok  (~5 chunks of 2048)
    fp16  median=  300.09  min=  280.98  max=  352.22   vs fp16 100.0%
    off   median=  297.13  min=  291.43  max=  320.70   vs fp16  99.0%
    on    median=  296.32  min=  293.69  max=  309.83   vs fp16  98.7%
    -> kernel on vs off: 0.997x

## DECODE (generation tok/s, 240 tokens, 2k prompt)
  240 new tokens
    fp16  median=   38.48  min=   38.01  max=   40.82   vs fp16 100.0%
    off   median=   40.21  min=   39.72  max=   41.10   vs fp16 104.5%
    on    median=   39.78  min=   37.22  max=   41.05   vs fp16 103.4%
    -> kernel on vs off: 0.989x

  peak memory (GB): fp16=2.604  off=2.726  on=2.599
  KV compression:   off=5.02x  on=5.02x
```

**This table is the most useful thing I measured**, and not because of the ratios. Look at the `min`/`max` columns.

At 8k prompt, the **unchanged fp16 baseline** — same code, same model, nothing swapped — ranges from **280.98 to 352.22 tok/s**. That's a **±25% spread** from thermal state and system scheduling alone, on a configuration where nothing about the code changed between runs.

Now recall the prediction: +1.7% on prefill. **Looking for a 1.7% effect through ±25% run-to-run variance is like weighing a signature on a bathroom scale.** No amount of care in the on/off comparison fixes that; the instrument simply doesn't resolve the quantity.

Note also that KIVI *itself* (off, 96.4%) is slightly slower than fp16 at 2k — the quantization work is real, it's just small. And the decode ordering (off at 104.5% of fp16, i.e. *faster* than no quantization at all) is a tell that we're deep inside noise, since compressing the cache cannot make decode faster than not compressing it.

Generation 1's apparent 0.962× decode regression shows up here as 0.989×, with overlapping min/max ranges. It was drift.

### Generation 4 — three models, interleaved, single process

The final protocol: one process per model, configurations interleaved rather than run in blocks, output text compared byte-for-byte between arms.

```
### mlx-community/Llama-3.2-3B-Instruct-4bit
  layers=28 kv_heads=8 prompt=2065 tok
  -> kernel on vs off:  prefill 1.019x (2k) / 0.997x (8k)   decode 0.989x
  -> identical text on/off: True

### mlx-community/Qwen2.5-7B-Instruct-4bit
  layers=28 kv_heads=4 prompt=2064 tok
  config  prefill tok/s  decode tok/s   peak GB   KV comp
  fp16            206.0         23.61     5.105         -
  off             203.7         23.40     5.080     4.94x
  on              208.3         23.26     5.130     4.94x
  -> kernel on vs off:  prefill 1.023x   decode 0.994x
  -> identical text on/off: True

### mlx-community/Mistral-7B-Instruct-v0.3-4bit
  layers=32 kv_heads=8 prompt=2054 tok
  config  prefill tok/s  decode tok/s   peak GB   KV comp
  fp16            143.4         20.86     4.891         -
  off             140.0         20.34     4.921     4.75x
  on              140.2         20.26     4.941     4.75x
  -> kernel on vs off:  prefill 1.001x   decode 0.996x
  -> identical text on/off: True
```

Consolidated:

| model | layers | KV heads | prefill | decode | identical text | KV compression |
|---|---|---|---|---|---|---|
| Llama-3.2-3B-4bit | 28 | 8 | 1.019× / 0.997× | 0.989× | ✅ | 5.02× |
| Qwen2.5-7B-4bit | 28 | **4** | 1.023× | 0.994× | ✅ | 4.94× |
| Mistral-7B-v0.3-4bit | 32 | 8 | 1.001× | 0.996× | ✅ | 4.75× |

Everything within ±2%, which given ±25% baseline variance is indistinguishable from nothing.

**Qwen is the most valuable row.** Four KV heads instead of eight means a completely different flush geometry, exercising different bounds-check and ragged-tail paths. It still produces byte-identical output — which is the strongest evidence that the bit-exactness work held up outside the unit tests.

---

## Why it's invisible, and why that was predictable

The true per-flush picture, idle GPU, Apple M4, 8 KV heads × 128 head dim, keys and values together:

| flush size | kernel off | kernel on | speedup |
|---|---|---|---|
| 32 *(decode)* | 0.5050 ms | 0.3601 ms | **1.40×** |
| 2048 *(prefill chunk)* | 3.8337 ms | 0.6785 ms | **5.65×** |

Recall that mlx_lm chunks prefill at 2048 tokens. That's a happy accident: prefill flushes land exactly on the kernel's strongest case, where there's enough work to amortize dispatch and memory-traffic savings dominate. Decode flushes are always small — `residual_length` tokens, 32 here — the weak case.

Scale it to 28 layers, an 8k prompt (4 prefill chunks), 240 decode tokens (7 flushes):

| phase | saved | of wall | predicted | measured |
|---|---|---|---|---|
| prefill (8k) | 353 ms | ~21,000 ms | **+1.7%** | 0.997× |
| decode (240 tok) | 28 ms | ~10,000 ms | **+0.28%** | 0.989× |

There's the whole story, available before running a single model.

> **A 5.65× speedup on 1.7% of the work is a 1.7% speedup.** Amdahl's law doesn't care how good the kernel is.

And 1.7% is four times smaller than the noise floor of the measurement. The end-to-end result wasn't a disappointment — it was arithmetic, and I could have computed it in ten minutes before writing any Metal at all.

---

## So why does the kernel ship?

Given that it's invisible end-to-end, why merge it?

- **It's free.** Bit-exact output, no regression on any model, 84 dedicated tests, byte-identical generations across three architectures. It never makes anything slower.
- **It removes a floor.** Quantization is ~1–2% of runtime *now*. If the surrounding work gets faster — better attention kernels, better matmuls — that share grows. Fixed costs matter more as everything else shrinks.
- **Op-level wins are real even when invisible.** 1.40× and 5.65× are honest measurements of the operation. That the operation is a small slice of the whole is a separate fact, and both belong in the report.

But the real reason to stay clear-eyed: **the kernel was never where the memory win lived.**

KIVI as implemented does quantize-then-*dequantize* — it computes the compressed representation and immediately expands it back to fp16 for attention. The 4.75×–5.02× compression is real *arithmetic*, but it is an **accounting result, not a storage result.** The tensor sitting in memory is still fp16.

This is also the structural reason Lie #4 had to be noise: if nothing is stored in compressed form, no kernel that computes the compression faster can reduce the high-water mark. I should have known the 127 MB was suspect on those grounds alone, before the other two models contradicted it.

To actually reduce memory you need two more things:

1. **Packed storage** — keep quantized codes as `uint8`, never materialize the fp16 reconstruction.
2. **Dequant-in-SDPA** — teach attention to read packed codes directly, so expansion happens in registers and never in RAM.

That's where both the memory win and the *real* speedup live, because it doesn't fuse 1–2% of the work — it shrinks the tensors every other operation has to move.

> The kernel was step one. It was worth doing. It just isn't the point.

---

## What I'd take away

If you're writing GPU kernels against a framework like MLX or PyTorch:

**Match the thread mapping to the memory layout, not to intuition.** The strided access pattern needed *less* machinery than the contiguous one — no reduction, no barriers, no transpose. My first instinct, transposing to make the layout "nice," was the version that lost.

**Bit-exactness is a debugging tool, not just a correctness bar.** FMA contraction, rounding mode, and padding semantics all produce *plausible* output. A tolerance test passes all three. Demanding identical output turned three silent behavioral differences into three failing tests with obvious causes.

**Never specialize on a value that grows.** Baking sequence length into a JIT header compiles one shader per token. The symptom was an indefinite hang; the fix deleted code.

**Estimate the ceiling before you optimize.** Ten minutes with Amdahl's law would have predicted the end-to-end result up front. It wouldn't have changed the decision to build it — but it would have set the expectation correctly, and I'd have spent my time on the parts that were load-bearing.

**Measure your noise floor before your effect.** The single most useful number in this entire project was ±25% — the run-to-run spread of an *unchanged* baseline. Without it, every ratio in every table is unfalsifiable.

**Be most suspicious when the number is good.** All four bad measurements came with satisfying stories. 97.83% "proved" the work mattered. 28.08× had a tidy overhead-crossover explanation. The 127 MB saving had a clean mechanism. Every one was wrong, and the plausible explanation is what let each survive as long as it did.

And: run the third model.

---

*Kernels live in `veloxquant_mlx/metal/src/`, tests in `veloxquant_mlx/tests/metal/test_kivi_quant.py`. See the [KIVI algorithm reference](/algorithms/kivi) for the method itself, and [Metal kernels](/guides/metal-kernels) for how kernels are dispatched library-wide. KIVI: [Liu et al., ICML 2024](https://arxiv.org/abs/2402.02750). All measurements on an Apple M4 with MLX.*
