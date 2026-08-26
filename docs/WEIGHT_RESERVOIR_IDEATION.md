# Weight Reservoir — ideation & go/no-go findings

## Origin

Thought experiment: what if Apple burned a frozen LLM's weights into a ROM
chip next to RAM — read-only, never re-derived, never fully materialized in
general-purpose RAM? Literal ROM silicon is out of scope for a software
library. But the underlying properties — **read-only, persisted once,
shared across processes without each paying full RAM cost** — map onto a
real, buildable gap in VeloxQuant-MLX: today `quantize_model()` produces
`QuantizedLinear` layers whose `_w_indices` / `_w_norms` live only as
in-process `mx.array`s, rebuilt from scratch (rotation + Lloyd-Max
assignment) on every load, in every process.

**Scope for this doc:** one already-quantized model's weights, persisted
read-only on local disk, loaded by multiple processes on **one machine**.
Not hardware, not third-party weight bundling, not network sharing.

## What already exists to build on

- [`weight/quantized_linear.py`](../veloxquant_mlx/weight/quantized_linear.py) —
  `QuantizedLinear.quantize_weights()` already produces the compact
  representation (`_w_indices`: uint8 per-element codebook index, `_w_norms`:
  fp32 per-row scale). This is the artifact the reservoir format needs to
  serialize — no new compression scheme required.
- [`weight/model_quantizer.py`](../veloxquant_mlx/weight/model_quantizer.py) —
  `quantize_model()` walks an `nn.Module` tree and replaces `nn.Linear` /
  mlx-lm `QuantizedLinear` layers in place. A reservoir loader needs the
  mirror operation: walk the tree and *attach* pre-quantized state instead
  of computing it.
- [`codebooks/base.py`](../veloxquant_mlx/codebooks/base.py) — `CodebookFactory`
  already produces small, deterministic centroid tables from `(distribution,
  bits, dim)`. Centroids are cheap to regenerate (not worth persisting) as
  long as the reservoir header records the parameters used to derive them.
- [`docs/PACKED_STORAGE_ROADMAP.md`](PACKED_STORAGE_ROADMAP.md) established the
  project's existing discipline for this exact class of claim: distinguish
  **accounting bytes** (theoretical compression ratio) from **resident
  bytes** (what Activity Monitor / RSS actually shows). The reservoir idea
  must be held to the same standard — see Finding 2 below.

## Findings (measured, not assumed)

Two questions gate whether "Weight Reservoir" is worth building at all:
does MLX's own loader already get this for free, and can `mx.array` be
constructed as a zero-copy view over an mmap'd buffer? Both were tested
directly against the installed environment (MLX 0.32.0 core /
mlx-python 0.6.5, macOS, this repo's `.venv`) rather than inferred from
docs.

### Finding 1 — `mx.load()` on `.safetensors` is not shared across processes

Saved a 500 MB float32 tensor to `.safetensors`, then launched two
independent `mx.load()` processes concurrently and read `vmmap --summary`
physical footprint for each:

```
PID 15970: Physical footprint 514.3M
PID 15971: Physical footprint 514.3M
```

Each process paid the full 514 MB independently — 1.03 GB combined for one
500 MB file. `vmmap` also showed **no mapped-file region** corresponding to
the safetensors path in the loading process; the loader reads and copies
into a private heap allocation rather than `mmap()`-ing the file. So the OS
page cache is not currently doing the "reservoir" job for us via the
existing loader — this part of the original idea does not come for free.

### Finding 2 — `mx.array()` has no zero-copy path from a buffer/mmap

Tried the more direct route: `np.memmap()` a raw binary file (lazy, no
copy — confirmed by RSS staying flat after `memmap()`), then wrap it with
`mx.array(memmap_view)`:

```
RSS after np.memmap (lazy, no touch):        1534.5 MB   (unchanged)
RSS after mx.array() wrap (no eval):         2542.3 MB   (+1008 MB, 2x file size)
RSS after mx.eval(x):                        2542.3 MB   (no further change)
```

`mx.array()` eagerly copies the full buffer at construction time — before
`mx.eval()` is even called — and the jump is roughly **2x** the source
size, suggesting an intermediate copy inside the constructor path (likely
buffer → numpy-owned staging → MLX unified-memory allocation). There is no
documented or observed zero-copy constructor in this MLX version's public
Python API.

### Conclusion: reframe from "zero-copy sharing" to "smaller copy"

The literal ROM analogy — pages shared for free across processes via OS
mmap — **is not achievable through MLX's current Python API**. Any
reservoir loader will pay at least one copy into MLX's Metal-backed
unified-memory arena, same as today's `quantize_model()` path. That kills
the strongest version of the pitch (N processes sharing one physical
copy).

What survives, and is still worth building:

1. **Skip re-quantization on every load.** Today, loading a quantized model
   means: load fp16/affine-quantized weights, dequantize, rotate, Lloyd-Max
   assign — a real compute cost paid per process, per load. A reservoir
   file with pre-computed `_w_indices` / `_w_norms` turns that into
   deserialize + one copy. This is a **load-time / CPU-time** win, not a
   RAM-sharing win.
2. **Smaller copy, not a shared copy.** Even without OS-level page sharing,
   copying a 2–4 bit reservoir blob into unified memory is still 4–8x less
   data moved and held than copying fp16 weights, which is what
   `quantize_model()`'s current input path does before it even starts
   compressing. Concurrent processes each still pay their own compressed
   footprint, but that footprint is much smaller than today's
   dequantize-then-compress pipeline's peak.
3. **Deferred/partial loading becomes possible.** Because indices are
   fixed-width uint8 per element with a page-aligned flat layout, a loader
   could `mmap()` the file and only copy (fault in) the layers actually
   used in a given forward pass — relevant for MoE or layer-skipping
   experiments — even though full zero-copy sharing across processes isn't
   available. This is a real but secondary benefit, not validated here.

This mirrors the exact lesson already written up in
[`docs/PACKED_STORAGE_ROADMAP.md`](PACKED_STORAGE_ROADMAP.md): a compression
ratio computed from byte-accounting does not automatically translate into a
resident-memory or cross-process win. Any reservoir PoC must report
**measured RSS**, not `memory_bytes` accounting, exactly as that roadmap
mandates for KV-cache quantizers.

## On-disk format

Flat, page-aligned binary, one file per model:

- **Header** (JSON): magic (`VQRS`), version, and per-layer `(name,
  out_features, in_features, bits, seed, use_hadamard, has_bias, bias,
  index_offset/nbytes, norms_offset/nbytes, rotation_offset/nbytes,
  centroids_offset/nbytes)`.
- **Index blob**: concatenated `_w_indices` (uint8, `out_features ×
  in_features` per layer), each layer's segment padded to a page boundary
  (16 KB on Apple Silicon) so per-layer `mmap()` slicing is possible later.
- **Norms blob**: concatenated `_w_norms` (fp32, `out_features` per layer).
- **Rotation blob**: Hadamard-compatible layers always persist their `(d,)`
  ±1 diagonal here (cheap). QR-fallback layers persist their full `d × d`
  matrix here **only if `persist_rotation=True`** at save time — see
  Finding 4.
- **Centroids blob**: each layer's Lloyd-Max codebook centroids (≤256 fp32
  values per layer), always persisted — cheap regardless of bit-width.

## PoC scope

Given Finding 1/2, the PoC's job is to **measure the load-time and
compressed-copy-size wins**, not to claim cross-process RAM sharing:

1. `weight/reservoir.py`:
   - `save_reservoir(model, path, persist_rotation=False)` — serialize an
     already-`quantize_model()`-processed model's `QuantizedLinear` layers
     to the format above.
   - `load_reservoir(path) -> dict[str, QuantizedLinear]` — deserialize
     header, reconstruct each layer via a fast path that bypasses
     `QuantizedLinear.__init__`'s expensive rotation/codebook setup, and
     assign `_w_indices` / `_w_norms` directly from the blob.
   - `graft_reservoir(model, path) -> nn.Module` — load a reservoir and
     replace matching named modules in an existing (possibly raw,
     never-quantized) module tree.
2. Benchmark A (single process): wall-clock load time for
   `quantize_model()` (dequantize + rotate + Lloyd-Max from fp16/affine
   source) vs. `load_reservoir()`, on one small model.
3. Benchmark B (N concurrent processes): RSS per process for
   `quantize_model()` vs. `load_reservoir()`. Per Finding 1/2, **do not
   expect shared physical pages** — report honestly if each process still
   pays its own footprint.
4. Correctness: `load_reservoir()`-produced model generates bit-identical
   output to the freshly `quantize_model()`-processed model.
5. Write measured results back into this doc before proposing promotion to
   a real feature.

## PoC results (measured)

Implementation: [`veloxquant_mlx/weight/reservoir.py`](../veloxquant_mlx/weight/reservoir.py)
(`save_reservoir` / `load_reservoir` / `graft_reservoir`), tests in
[`veloxquant_mlx/tests/weight/test_reservoir.py`](../veloxquant_mlx/tests/weight/test_reservoir.py)
(11/11 passing, bit-exact round trip including bias, grafting onto a raw
never-quantized module tree, and both `persist_rotation` modes on a real
QR-fallback layer). Full numbers in
[`figures/validation/weight_reservoir_results.json`](../figures/validation/weight_reservoir_results.json).

### Finding 3 — the format's first draft only persisted indices/norms, and that barely helped

The initial format persisted just `_w_indices` / `_w_norms` and
reconstructed each layer's rotation matrix and codebook from its seed via
`QuantizedLinear.__init__` on every load — same as `quantize_model()`
does. Profiling `load_reservoir()` on `Qwen2.5-0.5B-Instruct-4bit` (168
Linear layers, 24 of them wider than Hadamard-compatible dimensions and
falling back to QR rotation) showed **55 of 66.6 seconds** spent in
`np.linalg.qr` alone, plus another 8.9s in Lloyd-Max codebook fitting —
work the reservoir was supposed to skip, not repeat. This is the same
"accounting vs. resident/actual cost" trap
[`docs/PACKED_STORAGE_ROADMAP.md`](PACKED_STORAGE_ROADMAP.md) warns about,
just for compute instead of memory: the file format *looked* like it
skipped requantization, but the loader silently redid the expensive part
anyway.

**Fix:** persist the rotation matrix/diagonal and codebook centroids
themselves in the reservoir file, and reconstruct `QuantizedLinear` via a
`_fast_quantized_linear()` helper that bypasses `__init__`'s QR/Lloyd-Max
branches entirely (`nn.Module.__init__` + direct field assignment, mirroring
exactly what the real constructor sets up). After this fix, `load_reservoir()`
on the same model measured **0.35s–0.36s** in isolated runs (cProfile CPU
time: 0.346s) against a `quantize_model()` baseline of **81.6s** — a real,
reproducible **~230x** speedup on the setup/load side. (One measurement
taken back-to-back with the `quantize_model()` baseline in the same process
showed 9.2s instead of 0.36s; this variance is unresolved — plausibly
Metal shader warm-up or GPU dispatch contention — and is reported rather
than smoothed over. Even the conservative reading, 81.6s → 9.2s, is a real
~9x win.)

### Finding 4 — persisting full rotation matrices makes the file *larger* than the source model, not smaller (resolved)

`RotationPreconditioner` (the QR fallback, used whenever `in_features`
isn't Hadamard-compatible) stores a full `d × d` fp32 orthogonal matrix.
For `Qwen2.5-0.5B-Instruct-4bit`, 24 layers have `in_features = 4864`, so
each persisted rotation matrix is `4864² × 4 bytes ≈ 94.8 MB` — 24 of them
alone total **2.17 GB**, which is **86% of the resulting 2.52 GB reservoir
file**. The *source* model on disk is 265 MB. The first-draft reservoir
file was **9.5x larger than the model it compresses.**

Breakdown:

| Blob | Size |
| --- | --- |
| index blob (the actual 4-bit weights) | 341 MB |
| norms blob | 3.4 MB |
| **rotation blob** | **2168 MB** |
| centroids blob | 2.6 MB |

Hadamard-compatible layers (144 of 168 here) are unaffected — they persist
only a `(d,)` sign vector, effectively free. The problem is specific to the
QR-fallback path.

**Investigated and ruled out:** storing the QR rotation "compactly" via
Householder reflectors. `np.linalg.qr(G, mode='raw')` returns `(h, tau)`
where `h` is still a full `d × d` array (180.5 MB for `d=4864` at fp64,
even larger than the fp32 dense `Q` matrix) — a Haar-random `d × d`
orthogonal matrix has `d(d-1)/2` degrees of freedom, which is `Θ(d²)`
information with no compact encoding. There is no format trick that
shrinks this.

**Fix shipped:** `save_reservoir()` gained a `persist_rotation: bool =
False` parameter (default `False`). When `False`, QR-fallback layers store
nothing for their rotation matrix — `load_reservoir()` re-derives it from
the persisted seed via `np.linalg.qr`, paying the same cost
`quantize_model()` already pays, but keeping the file close to the size of
the compressed weights. When `True`, the matrix is persisted for a fast
load at the cost of file size. Hadamard-compatible layers are unaffected
either way (their `(d,)` diagonal is always cheap). Measured on
`Qwen2.5-0.5B-Instruct-4bit`:

| Mode | File size | Load time |
| --- | --- | --- |
| `persist_rotation=True` | 2515.7 MB (9.5x source) | 0.36s–9.2s |
| `persist_rotation=False` (default) | 349.7 MB (1.3x source) | 59.3s |

This makes the tradeoff explicit and caller-controlled instead of silently
shipping the large-file behavior as the only option. `TestQRFallbackRotationPersistence`
in [`test_reservoir.py`](../veloxquant_mlx/tests/weight/test_reservoir.py)
confirms bit-exact round trips in both modes on a real QR-fallback layer
(`in_features=50`, deliberately non-Hadamard-compatible) and confirms the
file-size difference is real, not just theoretical.

### Finding 5 — could not run the planned N=4 concurrent-process RSS benchmark on the target model

The original PoC plan called for benchmarking `mlx-community/Qwen3-4B-4bit`
(matching the issue's suggested model). `quantize_model()` on that model
reliably triggered **SIGKILL (exit 137)** on this 24 GB machine — measured
peak footprint 21.2 GB against ~14–15 GB free+inactive, consistent with
[`docs/MEMORY_CONSTRAINT_FINDINGS.md`](MEMORY_CONSTRAINT_FINDINGS.md)'s
documented headroom constraints on this exact machine. `mlx_lm.load()`
alone is fast and lightweight (0.5s for the 0.5B model); the memory
pressure comes entirely from `quantize_model()`'s dequantize → rotate →
argmin pipeline, which per Finding 1/2 makes multiple full-size intermediate
copies rather than working in place.

The concurrent-process benchmark was not run this pass. It remains valid
future work, but on a smaller model than originally planned, or after
addressing `quantize_model()`'s own peak-memory pipeline (a separate
problem from the reservoir format, out of scope here). Per Findings 1–2,
no result from that benchmark would show shared physical pages regardless
of model size — the question it would answer is only "how much smaller is
each process's own peak," not "do processes share memory."

## Go/no-go read after ideation and PoC

**Conditional go, narrower than the original pitch.** What's real and
measured:

- Cross-process RAM sharing via mmap does not happen with MLX 0.32.0's
  Python API (Findings 1–2) — the strongest version of the original "ROM"
  framing does not hold up.
- Skipping requantization on load is real and large — ~9x–230x depending
  on measurement variance not yet root-caused (Finding 3).
- The file-size side of the pitch is now caller-controlled rather than
  silently broken: `persist_rotation=False` (default) keeps the reservoir
  within ~1.3x of the source model's size for any architecture, at the
  cost of paying `quantize_model()`'s QR cost on every load for
  QR-fallback layers; `persist_rotation=True` trades that away for a much
  faster load at the cost of a file that can be 9x+ larger than the source
  model. There is no way to have both simultaneously — a dense random
  `d × d` rotation matrix is inherently `Θ(d²)` information (Finding 4,
  resolved).
- The originally planned concurrent-process memory benchmark could not run
  on the target model due to an unrelated `quantize_model()` OOM issue on
  this machine (Finding 5) — still open, not required to ship the
  save/load path itself.

Promote to a real feature after: (a) the 9.2s vs 0.36s variance in Finding
3 is root-caused so the `persist_rotation=True` load-time claim has one
number, not a range spanning 25x, and (b) Finding 5's concurrent-process
benchmark runs on a model this machine can actually quantize without OOM
(or on a machine with more headroom), to get the originally-planned N=4
RSS numbers. If a future MLX version adds a real zero-copy buffer-protocol
constructor, Finding 2 should be re-run before resurrecting the
cross-process sharing pitch.

## Explicitly out of scope

- Any actual hardware/silicon proposal.
- Bundling or licensing third-party model weights into the library/repo.
- Cross-machine / network sharing of the reservoir — single-machine,
  multi-process only.
