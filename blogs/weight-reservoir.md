# The ROM Chip That Wasn't, and the 230× Speedup That Was

*What happens when you take a thought experiment about burning LLM weights into silicon, try to build the closest real thing in software, and let the machine tell you which parts of the idea survive contact with MLX's actual copy semantics.*

---

The idea started as the kind of thing you think about in the shower: what if Apple burned a frozen LLM's weights into a ROM chip sitting right next to RAM? Read-only. Never re-derived. Never fully materialized in general-purpose memory. A model that's just *there*, the way a calculator's multiplication table is just there.

You can't ship that. This is a software library, not a silicon fab. But strip the hardware away and look at what's actually being asked for — weights that are read-only, computed once, and shared across processes without each one paying full price — and it stops being science fiction. It starts looking like a real, specific gap in how VeloxQuant-MLX loads models today.

So we built the closest real thing, measured it against the actual machine it would run on, and let two of our own assumptions get overturned by the data along the way. This is the record of that — including the part where a "fix" I proposed made things worse, and the part where I had to walk back a claim about compression that turned out to be mathematically impossible.

---

## What already existed

VeloxQuant-MLX already had most of the ingredients, just not assembled this way.

[`QuantizedLinear`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/veloxquant_mlx/weight/quantized_linear.py) compresses a weight matrix by normalizing each row, rotating it (Hadamard transform or a QR-derived rotation, depending on the dimension), and mapping it onto a small Lloyd-Max codebook — 2 to 4 bits per weight instead of 16. `quantize_model()` walks an entire `mlx-lm` model and replaces every `nn.Linear` with one of these. It works, and the compression ratios are real.

But the compressed result only ever lives as an `mx.array` in one process's memory. Every time you load the model — a new server worker, a restarted process, a second experiment running alongside the first — you pay the full cost again: dequantize the source weights, rotate them, run nearest-centroid search against the codebook, for every layer. Nothing from the last time you did this is reused.

That's the gap. Not "we lack compression" — we had that. "We recompute the compression from scratch on every single load."

## Two questions that had to be answered before writing any code

The ROM framing implies something specific: that multiple processes could share one physical copy of the weights, the way the OS page cache lets multiple processes reading the same file share pages in RAM. Before designing a file format around that idea, it needed to actually be true for MLX. So, two direct tests against the library itself rather than its documentation.

**Does `mx.load()` already give us this for free?** I saved a 500 MB tensor to `.safetensors`, then launched two independent processes that both called `mx.load()` on it, and read each process's physical footprint with `vmmap`:

```
PID 15970: Physical footprint 514.3M
PID 15971: Physical footprint 514.3M
```

Each process paid the full 514 MB independently — 1.03 GB combined for one 500 MB file. `vmmap` showed no mapped-file region for the safetensors path at all. The loader reads and copies into a private heap allocation. No sharing, no `mmap()`, nothing.

**Fine — what if I build the mmap myself?** I `mmap`'d a raw binary file with `np.memmap` (confirmed lazy — RSS didn't move) and wrapped it with `mx.array()`:

```
RSS after np.memmap (lazy, no touch):        1534.5 MB   (unchanged)
RSS after mx.array() wrap (no eval):         2542.3 MB   (+1008 MB — 2× the file size)
RSS after mx.eval(x):                        2542.3 MB   (no further change)
```

`mx.array()` copies the entire buffer immediately, before `mx.eval()` is even called, and the jump is roughly double the source size — there's a staging copy in there somewhere before the data lands in MLX's own unified-memory arena. There is no zero-copy constructor from a buffer in this version of MLX's Python API.

That killed the strongest version of the pitch. Processes on this machine are not going to share physical pages of model weights through anything MLX gives you today. If a future MLX version adds a real zero-copy buffer path, this is the test to rerun. Until then, the honest framing is narrower: **skip the recomputation, not the RAM.**

## Building the thing anyway

What survives the two negative findings above is still worth having: a file format that holds a model's already-quantized weights, persisted once, loadable without repeating the compression work. Call it a reservoir instead of a ROM — closer to what it actually is.

The first version was almost embarrassingly simple. Serialize each `QuantizedLinear` layer's compressed indices and per-row norms into a flat, page-aligned binary file — one blob for the 2–4 bit indices, one for the fp32 norms, a small JSON header recording each layer's shape and the seed used to derive its rotation and codebook. On load, reconstruct each layer from the header and drop the persisted indices straight in, skipping the whole dequantize-rotate-quantize pipeline.

I benchmarked it against `quantize_model()` on `Qwen2.5-0.5B-Instruct-4bit` — 168 linear layers, a small enough model to iterate on quickly. The baseline took 81.6 seconds. My new reservoir loader took 66.6 seconds.

That's not a win. That's barely a rounding error.

## Where the time actually went

I profiled it instead of guessing.

```
ncalls  tottime  cumtime  function
   168    0.126   66.587  QuantizedLinear.__init__
    24    2.079   57.442  make_rotation_matrix
    24   54.693   55.147  numpy.linalg.qr
   168    0.001    8.907  CodebookFactory.create
   168    4.016    8.899  lloyd_max
```

There it was. 55 of the 66.6 seconds were spent inside `np.linalg.qr` — the routine that derives a rotation matrix for any layer whose input dimension doesn't cleanly fit MLX's Hadamard transform (a dimension like 4864, which is neither a clean power of two nor a multiple of the special constants MLX's fast transform supports). Another 8.9 seconds went to fitting Lloyd-Max codebooks.

The reservoir file *looked* like it had skipped requantization. What it actually skipped was the nearest-centroid search — the smallest part of the cost. Every layer's rotation matrix and codebook were being silently recomputed from the stored seed, on every load, exactly as expensive as before. The format was a placebo.

The fix was to stop being clever about "everything is derivable from the seed" and just persist the actual rotation matrices and codebooks, then reconstruct each layer by bypassing `QuantizedLinear.__init__`'s expensive branches entirely — direct field assignment onto a bare module instead of calling a constructor that recomputes things you already have on disk.

That dropped the load time to somewhere between 0.36 and 9.2 seconds, against the same 81.6-second baseline. Even the conservative end of that range is a 9× speedup; the isolated, repeatable measurement (0.36s, matching `cProfile`'s CPU-time reading almost exactly) is closer to 230×. The gap between those two numbers is real and I haven't root-caused it — plausibly Metal shader warm-up, plausibly GPU dispatch contention from running back-to-back with the baseline in the same process. I'm reporting the range rather than picking the number that looks better.

## The part where fixing one problem created a worse one

Feeling good about the speedup, I checked the file size. The source model is 265 MB on disk. My reservoir file was 2.5 gigabytes.

Nine and a half times larger than the model it was supposedly compressing.

I broke down where the bytes went:

| Blob | Size |
|---|---|
| index blob (the actual 4-bit weights) | 341 MB |
| norms blob | 3.4 MB |
| **rotation blob** | **2168 MB** |
| centroids blob | 2.6 MB |

Eighty-six percent of the file was rotation matrices. Specifically, the 24 layers that fall back to QR rotation instead of the fast Hadamard path — each one has `in_features = 4864`, and a `4864 × 4864` matrix at 4 bytes per float is about 95 MB. Twenty-four of those is 2.17 GB, and that's before touching a single actual weight.

The instinct here is "surely you can store that more compactly." I had the same instinct, and I want to walk through why it doesn't work, because the wrong answer is genuinely tempting.

`np.linalg.qr` has a `mode='raw'` option that returns the underlying Householder reflectors instead of the assembled orthogonal matrix — sounds exactly like what you'd want, a compact representation of the same rotation. I checked what shape it actually returns:

```python
h, tau = np.linalg.qr(G, mode="raw")
# h.shape == (4864, 4864)   — 180.5 MB, even bigger than the dense matrix
# tau.shape == (4864,)      — 0.037 MB
```

Still a full `d × d` array. The reason isn't an API limitation — it's that a Haar-random orthogonal `d × d` matrix genuinely contains `d(d-1)/2` degrees of freedom. That's Θ(d²) information, full stop. There is no encoding, clever or otherwise, that gets a truly random rotation below quadratic storage, because the matrix doesn't have any structure to exploit. It's not compressible the way the *weights* are compressible — the weights have statistical structure a codebook can exploit; a random rotation, by construction, doesn't.

So the earlier plan — "store the rotation as reflectors, get a smaller file" — was wrong. Not underexplored. Wrong, provably, in about ten minutes of checking.

## Making the tradeoff a choice instead of a default

Once "compress the rotation matrix" was off the table, what was left was a genuine, irreducible tradeoff: you can have a fast load (persist the rotation matrix, pay the disk space) or a small file (don't persist it, pay the QR cost again on every load). Not both, for any layer that needs QR fallback.

The fix was to stop pretending there was a single right answer and expose the choice:

```python
save_reservoir(model, path, persist_rotation=False)  # default: small file
save_reservoir(model, path, persist_rotation=True)  # fast load, large file
```

With the default off, the reservoir file for the same model comes out to 349.7 MB — 1.3× the source model, essentially just the compressed weights plus some structural overhead — and load time goes back up to about 59 seconds, since QR-fallback layers regenerate their rotation from the stored seed exactly as the original `quantize_model()` path does. With it on, you're back to sub-second loads and a 2.5 GB file. Hadamard-compatible layers — 144 of the 168 in this model — are unaffected either way, since their rotation is just a `d`-length sign vector, cheap to store regardless.

Neither setting is wrong. What was wrong was shipping one of them silently as the only option and calling the result "a smaller reservoir."

## What didn't get tested, and why

The original plan called for benchmarking `Qwen3-4B-4bit` — a more realistic model size, and specifically what the ideation phase had proposed. `quantize_model()` on that model reliably killed the process with SIGKILL on this 24 GB machine. I checked the peak memory footprint before the kill: 21.2 GB, against roughly 14–15 GB of actually available memory at the time.

This isn't a bug I introduced. It's the same headroom constraint documented elsewhere in this repo (`docs/MEMORY_CONSTRAINT_FINDINGS.md`) for a 32B model on the same hardware — `quantize_model()`'s dequantize-rotate-quantize pipeline makes several full-size intermediate copies rather than working in place, and a 4B model's pipeline apparently needs more headroom than this particular machine has free right now. It's a real, separate problem, and it blocked the concurrent-process memory benchmark I'd wanted to run — the one that would have measured whether four processes loading the reservoir simultaneously actually show smaller RSS than four processes each running `quantize_model()` from scratch. That benchmark still needs to happen, on a model this machine can actually load, or on a machine with more headroom.

## What actually shipped

- `save_reservoir()` / `load_reservoir()` / `graft_reservoir()` in `weight/reservoir.py` — a flat, page-aligned binary format, with `persist_rotation` as an explicit, documented tradeoff rather than a hidden default.
- A `_fast_quantized_linear()` construction path that builds a `QuantizedLinear` from persisted state without touching `__init__`'s QR or Lloyd-Max branches.
- Eleven tests, including a deliberately non-Hadamard-compatible layer (`in_features=50`) to exercise the QR fallback path directly, and bit-exact round-trip checks in both `persist_rotation` modes — the loaded model's forward pass output is byte-identical to the freshly quantized one, not just "close."
- A results file with the actual numbers, not just the ones that made the feature look good.

## What this experiment is actually about

None of the interesting findings here came from writing code that worked on the first try. They came from measuring something, getting an answer that contradicted an assumption, and following that instead of the original plan:

- The "shared ROM" framing died the moment two processes independently paid full memory cost for the same file — a five-minute test with `vmmap`, run before any format design.
- The first reservoir format's near-zero speedup came from profiling instead of trusting that "we skip requantization" was true because the code was structured to look like it should be.
- The Householder-reflector idea died in about ten minutes of checking `mode='raw'`'s actual return shape — a claim that sounded plausible enough to write into a planning doc, and would have stayed there if nobody had gone and checked.

The honest version of this project isn't "we built a ROM chip in software." It's: cross-process sharing doesn't work with MLX's current copy semantics, skipping recomputation is a real and large win once you persist the *right* things, and file size versus load time is a fundamental tradeoff for anything that needs a truly random rotation — not a bug to be engineered away, just a dial to expose honestly instead of hiding.

That's a smaller claim than the one I started with. It's also the one that's actually true.
