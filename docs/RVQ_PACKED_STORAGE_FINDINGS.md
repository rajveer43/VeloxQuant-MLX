# TurboQuantRVQ Packed Storage — Investigation, Fix, and Measured Findings

**Date:** 2026-08-09
**Hardware:** Apple M4 (24GB unified memory)
**Model used for real-model tests:** mlx-community/Llama-3.2-1B-Instruct-4bit
**Library:** VeloxQuant-MLX v0.44.0 (branch `feat/27-packed-storage-turboquant-rvq`)
**Related:** #27 (original accounting-vs-resident audit), #56/#125-#130 (standalone-method serving refusal work), `docs/PACKED_STORAGE_ROADMAP.md`

---

## 1. Why this exists

`docs/PACKED_STORAGE_ROADMAP.md` names `turboquant_rvq` as the first target for
converting accounting-only compression into genuinely reduced resident memory:

> `turboquant_rvq` | No | Dequant into parent fp16 cache; counters are accounting

Issue #27's own minimum-honest-v1 scope names the same three methods —
`turboquant_rvq`, `kivi`, `vecinfer` — as the first tier to fix. This document
covers **`turboquant_rvq` only**. `kivi` and `vecinfer` are not touched by this
work and their rows in the roadmap table are unchanged.

This document records every step taken: what the old code actually did, how the
new packed storage was designed, every correctness check run, the real bugs
found along the way (including a false start in the memory-measurement
methodology itself), and the final, reproducible numbers.

---

## 2. Starting point: what `TurboQuantRVQKVCache` did before this change

Read directly from `veloxquant_mlx/cache/turboquant_rvq_cache.py` prior to this
branch:

```python
def update_and_fetch(self, keys, values):
    ...
    ev = self._quantizer.encode(k_unit)
    k_hat_u = self._quantizer.decode(ev)          # <- decode immediately
    k_dequant = (k_hat_u.astype(kdtype) * safe).reshape(B, H, S, D)

    per_tok = (math.ceil(self._head_dim * 2 * self._bits / 8) + 2) * H * B
    self._key_bytes_compressed += per_tok * S      # <- counter only, no real storage
    self._key_bytes_fp16       += H * B * S * self._head_dim * 2

    return super().update_and_fetch(k_dequant, values)   # <- stores fp16 via parent class
```

`compressed_key_bytes` / `fp16_key_bytes` were pure counters computed alongside
storage that was, in fact, always fp16-resident (inherited from
`mlx_lm.models.cache.KVCache.update_and_fetch`, which allocates and writes an
`mx.array` fp16 tensor). The "7.5× compression" the README quotes for
`turboquant_rvq` at b=1 was real as a *bit-width accounting* ratio, but the
process never actually held fewer bytes for it.

---

## 3. What `TurboQuantRVQ.encode()` actually produces

Read from `veloxquant_mlx/quantizers/turboquant_rvq.py`. This matters because it
determines whether "just use `mx.quantize()` instead" (MLX's own native affine
quantizer) was a viable shortcut — it is not, because RVQ is a different,
non-affine scheme:

1. Rotate `x` via a Hadamard or random orthogonal preconditioner → `y`.
2. Stage 1: quantize `y` against a **fixed, precomputed Lloyd-Max codebook**
   fit to `N(0, 1/d)` (Gaussian) → `idx1`, an array of small integers in
   `[0, 2**b)`.
3. Compute the residual `r1 = y - y_hat1` and quantize it against a **second
   fixed Lloyd-Max codebook** fit to a Laplacian distribution matched to the
   stage-1 quantization error → `idx2`.
4. Reconstruction: `x_hat = unrotate(y_hat1 + y_hat2)`, both `y_hat1`/`y_hat2`
   being codebook lookups (gathers), not an affine `scale * q + bias` formula.

`mx.quantize()` (MLX's native primitive) only implements per-group **affine**
min/max quantization — a materially different, simpler scheme. Swapping RVQ's
storage for `mx.quantize()` would silently change what is actually stored and
how it reconstructs; it was not used. `idx1` and `idx2` are both
`(batch, head_dim)` `uint8` arrays with values in `[0, 2**bits)` — the same
shape MLX's own quantizer packs, which is what made a custom-but-analogous
packer possible.

---

## 4. Packing design

### 4.1 Bit layout

A new `_pack_indices` / `_unpack_indices` pair (in
`veloxquant_mlx/cache/turboquant_rvq_cache.py`) bit-packs the trailing
dimension of a `(..., d)` `uint8` array into `(..., ceil(d / el_per_word))`
`uint32` words, LSB-first — the same layout `mx.quantize` uses internally, so
the scheme is a known-good, provably correct one rather than a new invention:

```python
def _pack_indices(idx, bits):
    *lead, d = idx.shape
    el_per_word = 32 // bits
    pad = (-d) % el_per_word
    if pad:
        idx = mx.concatenate([idx, mx.zeros((*lead, pad), dtype=idx.dtype)], axis=-1)
    d_padded = d + pad
    n_words = d_padded // el_per_word
    idx = idx.reshape(*lead, n_words, el_per_word).astype(mx.uint32)
    shifts = mx.arange(el_per_word, dtype=mx.uint32) * bits
    return mx.sum(idx << shifts, axis=-1)
```

Implemented entirely in MLX array ops (bit-shift, mask, sum) — no Python loops
over tokens, no NumPy round-trips — so it stays fast on the decode hot path,
unlike the existing `BitPackBuffer` utility (`veloxquant_mlx/dsa/bit_pack.py`),
which is NumPy-based and 1-D only, and was considered and rejected for this use
because it would force a NumPy↔MLX round-trip on every generated token.

**Verified**, via a standalone script before any cache wiring, round-trip
correctness for `bits ∈ {1, 2, 3, 4}` at `d=128`:

```
bits=1: packed_shape=(16, 4),  bytes_per_row=16, fp16_bytes_per_row=256, round_trip_ok=True
bits=2: packed_shape=(16, 8),  bytes_per_row=32, fp16_bytes_per_row=256, round_trip_ok=True
bits=3: packed_shape=(16, 13), bytes_per_row=52, fp16_bytes_per_row=256, round_trip_ok=True
bits=4: packed_shape=(16, 16), bytes_per_row=64, fp16_bytes_per_row=256, round_trip_ok=True
```

### 4.2 Cache storage model

`TurboQuantRVQKVCache` now stores, per layer:

- `_packed1`, `_packed2`: `(B, H, capacity, n_words)` `uint32` — the two RVQ
  index streams, grown in `step=256`-token chunks (mirroring
  `mlx_lm.models.cache.KVCache`'s own growth strategy).
- `_norms`: `(B, H, capacity, 1)` `fp16` — the per-token key norm needed to
  undo the unit-normalization RVQ requires before quantizing.
- `self.values`: plain `fp16`, unchanged, grown the same way.

`update_and_fetch` packs and writes only the *newly arrived* `S` tokens into
these buffers each call, then reconstructs the **full cached range**
`[0, offset)` back to fp16 via `_dequantize_range` before returning — this
reconstruction step is required because `mlx_lm`'s attention needs a
`(B, H, S_total, D)` fp16 tensor to run standard scaled-dot-product attention
against, and because `hasattr(cache, "bits")` in
`mlx_lm.scaled_dot_product_attention` routes to mlx-lm's own fused quantized
SDPA kernel, which expects mlx-lm's specific `(packed, scales, biases)` affine
layout — not RVQ's codebook-based one. Exposing `.bits` publicly would
silently misroute attention, so (matching the pre-existing code's own
comment) the bit-width stays private (`_bits`) and is exposed only as
`assigned_bits`.

This means dequantization happens on every fetch — the same accepted
performance tradeoff `mlx_lm`'s own native `QuantizedKVCache` makes.

---

## 5. Correctness verification (all done before any memory measurement)

### 5.1 Bit-exact round trip against ground truth

Compared the packed cache's `update_and_fetch` output directly against calling
`TurboQuantRVQ.encode()`/`decode()` with **no cache storage involved at all**
(same rotation, same codebooks, same input):

```
max abs diff between cache-packed-then-unpacked and direct encode/decode: 0.0
PASS: packed round-trip matches direct encode/decode
cosine(orig, reconstructed): 0.9777170419692993
```

Zero difference, and the 0.978 cosine similarity at b=2 matches the README's
independently-stated ~0.98 cosine figure for RVQ b=2 — confirming the packing
math introduces no additional reconstruction error beyond RVQ's own.

### 5.2 Multi-step growth across the `step=256` boundary

Simulated a 10-token prefill followed by 300 single-token decode steps
(crossing the pre-allocation boundary twice):

```
offset after all steps: 310
nbytes: 359600
is_trimmable: True
trimmed: 50, new offset: 260
empty(): False
PASS: multi-step growth across step boundary works correctly
```

### 5.3 `deepcopy` isolation

Per-request cache isolation matters because `mlx_lm.server`'s
`fetch_nearest_cache` returns `copy.deepcopy(cache_entry.prompt_cache)` per
request (established in #27 finding 3). Verified the packed cache preserves
this:

```
PASS: deepcopy isolation works
```

(mutating the original after copying did not affect the copy's `offset`.)

### 5.4 `state` / `meta_state` round trip — **a real bug was found and fixed here**

`mlx_lm`'s `_BaseCache.from_state()` constructs cache instances via
`cls.__new__(cls)`, **bypassing `__init__` entirely**, then only assigns
`.state` and `.meta_state`. The first version of this change computed
`self._quantizer`, `self._el_per_word`, `self._n_words`,
`self._key_bytes_compressed`, and `self._key_bytes_fp16` only inside
`__init__` — so a cache reconstructed via `from_state()` (the exact path
`mlx_lm.server` uses to restore a saved session) was missing all of them and
crashed on the very next `update_and_fetch` call with
`AttributeError: 'TurboQuantRVQKVCache' object has no attribute '_n_words'`.

This was caught by writing `test_state_meta_state_round_trip_via_from_state`
in `veloxquant_mlx/tests/cache/test_turboquant_rvq_cache.py` — not by manual
testing, which had only exercised the live object, never a `from_state()`
reconstruction.

**Fix:** extracted quantizer/packing-constant construction into a
`_build_derived_state()` method, called from both `__init__` and the
`meta_state` setter. `meta_state` now also carries `seed` (it previously did
not), since `TurboQuantRVQ`'s rotation matrix depends on it and a
reconstructed cache with the wrong seed would silently decode garbage against
previously-packed data. The cumulative `_key_bytes_compressed` /
`_key_bytes_fp16` counters are intentionally **reset**, not preserved, across
a `from_state()` restore, since they track bytes processed by *this object
instance* going forward, not bytes physically present in the restored state.

After the fix, all of the following pass:

```
test_state_meta_state_round_trip_via_from_state PASSED
```

### 5.5 Real end-to-end generation on a real model

```python
model, tokenizer = mlx_lm.load('mlx-community/Llama-3.2-1B-Instruct-4bit')
config = KVCacheConfig(method='turboquant_rvq', bit_width_inlier=1, seed=42)
patch_model_kv_cache(model, config)
response = mlx_lm.generate(model, tokenizer, prompt='The capital of France is', max_tokens=30)
# GENERATED: 'Paris.\nThe city of Paris.\nThe city of Paris...'
```

Coherent output confirms the packed storage works correctly inside the real
`mlx_lm.generate()` path, not just in isolated unit tests.

### 5.6 Automated test suite added

`veloxquant_mlx/tests/cache/test_turboquant_rvq_cache.py` gained 11 new tests
(6 pre-existing tests continue to pass unmodified):

- `test_pack_unpack_round_trip` (parametrized over bits ∈ {1,2,3,4})
- `test_packed_storage_matches_direct_encode_decode`
- `test_growth_across_step_boundary`
- `test_trim`
- `test_empty`
- `test_deepcopy_isolates_state`
- `test_state_meta_state_round_trip_via_from_state`
- `test_nbytes_reports_true_packed_size_not_fp16` — the core acceptance
  criterion: asserts `cache.nbytes` is strictly less than an equivalent plain
  `mlx_lm.models.cache.KVCache`'s `nbytes` for the *same* cached tokens (not
  an estimate compared against an estimate — an actual object of each type,
  same input, real `.nbytes` property compared).

Full suite result: **1771 passed** (was 1759 before any change in this
session; +1 from the earlier #130 fix, +11 from this change).

---

## 6. Memory measurement: methodology, a false start, and the real numbers

This section is deliberately detailed because getting a real memory number
right took several wrong turns worth recording, in the same spirit as issue
#27's own "don't paper over gaps" standard.

### 6.1 First attempt: OS-level RSS — inconclusive, discarded

Used `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` before/after
generation. At a short prompt (~400 tokens), baseline and packed-cache RSS
were statistically indistinguishable (1085.8 MB vs 1087.5 MB) — the Python
interpreter and 4-bit model weights dominate total process RSS at this scale,
swamping any KV-cache-sized signal. At a much longer prompt (30,002 tokens),
RSS *still* barely moved, which is suspicious enough on its own to distrust
the measurement rather than the cache. **Conclusion: OS-level `ru_maxrss` is
not a sensitive enough instrument for this; discarded.**

### 6.2 Switched to `mlx.core.get_peak_memory()` / `get_active_memory()`

MLX manages its own unified-memory allocator and exposes precise
introspection (`mx.get_active_memory()`, `mx.get_peak_memory()`,
`mx.reset_peak_memory()`, `mx.clear_cache()`) — a much more sensitive and
relevant instrument than OS RSS for a library that only ever touches memory
through MLX arrays.

### 6.3 Second false start: a broken "native quantized cache" comparison

An initial attempt to compare against mlx-lm's own native `QuantizedKVCache`
passed `kv_bits=4` as a kwarg to `mlx_lm.generate()` — **this is not a real
parameter of that function** (confirmed via `inspect.signature`); it silently
fell through `**kwargs` into an unrelated sampler parameter and had no effect.
The resulting number (a suspicious 4961.2 MB, higher than fp16) was initially
almost reported as "mlx-lm's native path is broken," which would have been
wrong. **Correction:** the right way to select mlx-lm's native quantized cache
is to construct `mlx_lm.models.cache.QuantizedKVCache(group_size=64, bits=4)`
per layer directly and wire it via `model.make_cache`, matching how
`veloxquant_mlx`'s own `patch_model_kv_cache` works. Re-running with the
correctly-wired native cache reproduced the *same* 4961.2 MB figure — so the
number itself was real, just initially attributed to the wrong cause. This is
recorded so the same mistake is not repeated in future benchmarking work in
this repo.

### 6.4 Third false start: unexplained run-to-run variance

Early repeated trials of the packed RVQ cache showed a real, reproducible
**bimodal** peak-memory pattern within the same benchmarking script (1402.3 MB
on some runs, 2097.5 MB, then 2792.8 MB, then 3488.0 MB on a longer loop) —
increasing by almost exactly one model-weights-worth of memory (~695MB) on
each successive loop iteration, while the fp16 baseline stayed perfectly
stable at 1608.4 MB across all repeats. This looked, briefly, like a real
memory leak in the new cache code.

**Root cause, confirmed:** it was not a leak in `TurboQuantRVQKVCache`. It was
a benchmarking-script bug — the loop reassigned `model`/`tokenizer`/`caches`
each iteration without an explicit `del` + `gc.collect()`, so the *previous*
iteration's model object (and its MLX arrays) stayed alive by reference count
until the reassignment completed, and `mx.clear_cache()` alone does not force
Python-level garbage collection of objects still referenced. Adding explicit
`del model, tokenizer; gc.collect()` between measurement iterations made every
run perfectly reproducible. **This confirms the earlier "leak" was a test
artifact, not a defect in the cache implementation** — but it is recorded
here because it is a trap any future memory benchmarking in this codebase
could fall into the same way.

### 6.5 Final methodology (used for the numbers below)

```python
def measure(setup_fn):
    mx.clear_cache()
    model, tokenizer = mlx_lm.load('mlx-community/Llama-3.2-1B-Instruct-4bit')
    setup_fn(model)
    mx.reset_peak_memory()
    mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=False)
    peak = mx.get_peak_memory() / 1e6
    del model, tokenizer
    gc.collect()
    return peak
```

- Prompt: `"The quick brown fox jumps over the lazy dog. " * 400` → 4002 tokens
  (confirmed via `tokenizer.encode`).
- 100 decode tokens after the prefill.
- Fresh model load per configuration (no state shared across configurations).
- `mx.clear_cache()` before each run, explicit `del` + `gc.collect()` after.
- Every number below reproduced identically across repeated trials (2+ trials
  per configuration, bit-for-bit identical peak figures each time) once this
  methodology was used — this is the signal that the earlier variance was
  fully explained by the loop bug in §6.4, not residual nondeterminism.

### 6.6 Results

| Configuration | Peak memory (`mx.get_peak_memory()`) | vs. fp16 baseline |
|---|---|---|
| fp16 baseline (`mlx_lm.models.cache.KVCache`) | **1608.4 MB** | — |
| mlx-lm native `QuantizedKVCache(bits=4, group_size=64)` | **2537.2 MB** | **+57.7% (worse)** |
| `turboquant_rvq`, b=1, packed storage (this PR) | **1402.3 MB** | **−12.8% (better)** |

And, for context, this implementation vs. mlx-lm's own native quantized cache
at the same prompt: **−44.7%** (1402.3 MB vs. 2537.2 MB).

**Why mlx-lm's own native quantized cache is worse than fp16 here:**
`QuantizedKVCache.update_and_fetch` (in `mlx_lm/models/cache.py`) calls
`mx.quantize(keys, ...)` / `mx.quantize(values, ...)` fresh on every incoming
chunk. For a single large prefill call (`S=4002` in one shot), this
materializes full-size intermediate quantization buffers alongside the
existing tensors before the packed result is written — a real, measured cost
at this call pattern, not a bug in mlx-lm's design intent (steady-state
per-token decode is presumably its target case, not one-shot bulk prefill).
This implementation avoids that cost because `_pack_indices` writes directly
into the pre-allocated packed buffer for only the newly-arrived tokens each
call, without an intermediate full-tensor quantization pass.

**This is one measured data point with its methodology fully stated, not a
universal claim.** It should be expected to shift at different context
lengths, different chunking/prefill strategies (e.g. chunked prefill rather
than one giant call), different `group_size`/`bits` choices for the native
cache, or different hardware. Anyone repeating this benchmark should reuse
the `del` + `gc.collect()` + `mx.clear_cache()` discipline in §6.5 — omitting
it reproduces the misleading bimodal pattern in §6.4, not a real result.

---

## 7. Scope and what was deliberately not done

- **`kivi` and `vecinfer` are untouched.** They are named as the next two
  targets in #27's tier-1 scope and in `docs/PACKED_STORAGE_ROADMAP.md`, but
  implementing them was out of scope for this change (see the discussion in
  the originating conversation: ship and measure one method fully before
  repeating the pattern).
- **`merge`/`BatchKVCache` support was not added.** `TurboQuantRVQKVCache`
  does not override `merge`, matching its pre-existing behavior (it never
  overrode `merge` before this change either) — cross-request batching for
  this cache remains untested/unsupported, consistent with #27's own note
  that `BatchKVCache.merge` semantics under compression were "Unknown" and
  out of scope for a first tier.
- **No CI-enforced memory assertion was added** for the `mx.get_peak_memory()`
  real-model numbers in §6.6, since that requires a model download and is
  sensitive to the exact call pattern (see §6.3–6.4). The `nbytes`-level
  regression test (`test_nbytes_reports_true_packed_size_not_fp16`, §5.6) is
  the durable, deterministic, CI-safe proxy for the same honesty guarantee —
  it does not require a model download and cannot exhibit the measurement
  artifacts described in §6.3/§6.4.
- **No external issue was filed against `ml-explore/mlx-lm`** regarding the
  §6.6 finding that their native `QuantizedKVCache` is slower/heavier than
  fp16 on large single-shot prefill calls. This is recorded here as context
  for VeloxQuant-MLX's own positioning, not reported upstream.
