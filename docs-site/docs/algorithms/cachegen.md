# CacheGen — Entropy-Coded KV Cache Storage

**Method id:** `cachegen` · **New in 0.16.0**

CacheGen shrinks how much space your KV cache takes up on disk or in
transit, without changing a single value the model sees. It works by
noticing that neighboring tokens' KV values are similar, so it stores the
*difference* between them instead of the raw values — and that difference
compresses much further than the raw values do.

## Should I use this?

Use CacheGen when:

- You're **storing or transmitting** KV caches (offload to CPU/disk,
  streaming a cache over a network, checkpointing a long conversation) and
  want a smaller footprint for that.
- Your workload has **long, coherent contexts** — documents, chat history,
  code — where nearby tokens are semantically related. That's what the
  compression exploits.
- You want compression with **zero extra reconstruction error** on top of
  whatever base quantization you choose — CacheGen never makes the stored
  values worse, only smaller.

Reach for something else when:

- You're bottlenecked on **decode-time GPU/unified-memory bandwidth on
  Apple Silicon**, not storage size — CacheGen decodes back to fp16 before
  attention runs, so the working set at attend time is unchanged. Prefer
  [PALU](../algorithms/palu) or [SVDq](../algorithms/svdq), which keep the
  cache compressed through the attend step.
- Your data is **not token-correlated** (e.g., shuffled or synthetic KV) —
  compression gracefully falls back to zero savings rather than backfiring,
  but there's no upside either.

## Quick start

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="cachegen",
    head_dim=128,
    cachegen_bits=4,  # bit-width for the shallowest layers
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

That's it — `for_model` automatically assigns a layer-wise bit schedule
(shallow layers get more bits, deep layers get fewer — see [Tuning](#tuning)),
groups the entropy estimate by channel, and every layer reports how many
bytes it's actually costing you.

## Checking how much you saved

```python
key_bytes = sum(c.compressed_key_bytes for c in caches)
fp16_bytes = sum(c.fp16_key_bytes for c in caches)
print(f"Key cache: {key_bytes / fp16_bytes:.1%} of fp16 size")
```

Each cache also exposes `entropy_savings` (fraction saved vs. fixed-width
packing of the same codes) and `assigned_avg_bits` (effective bits/element
after entropy coding, for comparison against `cachegen_bits`).

## Tuning

| Parameter | Default | What it controls |
|-----------|---------|-------------------|
| `cachegen_bits` | `4` | Base bit-width, used for the shallowest layer group. Lower = smaller but lossier. |
| `cachegen_layer_groups` | `3` | How many depth-based groups to split layers into for the bit schedule (paper default: 3 — early/middle/late). |
| `cachegen_group_size` | `32` | Tokens per quantization group. Smaller groups track local statistics more closely but add per-group overhead. |
| `cachegen_use_delta` | `True` | Encode the delta between neighboring tokens' codes instead of raw codes. This is the main lever for compression — leave it on unless you're debugging. |
| `cachegen_per_channel` | `True` | Fit a separate probability distribution per channel before measuring entropy, instead of pooling all channels together. Leave it on; it's a strictly better estimate. |

**If you want a specific quality/size trade-off:** start at `cachegen_bits=4`
and check `assigned_avg_bits` — if it's close to 4, your data isn't very
token-correlated and lower bits won't help much. If it's noticeably lower
than 4, the entropy coder is finding real structure and you can likely drop
`cachegen_bits` further without hurting quality much.

**Layer-wise scheduling only activates through `KVCacheBuilder.for_model`.**
If you build a single cache directly via `KVCacheFactory.create` (no model
context), every layer uses the uniform `cachegen_bits` value — there's no
layer index to build a schedule from.

## How it works

CacheGen builds on three observations from the paper about how KV cache
values are distributed:

1. **Token-wise locality** — adjacent tokens' KV vectors are similar, so the
   per-token *delta* of the quantized codes concentrates near zero and
   compresses far better than the raw codes.
2. **Layer-wise sensitivity** — output quality is more sensitive to precision
   loss in shallow layers than in deep ones, so shallow layers keep more bits
   and deep layers get progressively coarser quantization
   (`cachegen_layer_groups`, applied automatically by `for_model`).
3. **Channel/layer grouping** — a probability distribution fit separately per
   channel compresses much better than one pooled distribution across all
   channels (`cachegen_per_channel`).

The pipeline per head:

1. Asymmetric min/max group-quantize keys/values to integer codes (the same
   scheme as KIVI).
2. Apply the reversible **token-delta** transform along the sequence axis.
3. Measure the **Shannon entropy** of the delta symbol stream, grouped by
   channel, and report the compressed size from it.
4. Reconstruct fp16 from the codes (identical to plain group quant — no
   extra loss from this layer).

| Method | Reconstruction | Compresses via | Win |
|--------|----------------|-----------------|-----|
| KIVI-2bit | group quant | fixed 2-bit packing | bandwidth |
| CacheGen | identical to group quant | entropy coding of deltas | storage |

## Fidelity to the paper

*Inspired by* [CacheGen (arXiv:2310.07240, SIGCOMM 2024)](https://arxiv.org/abs/2310.07240)
— documented here as **CacheGen-adapted (VeloxQuant-MLX implementation)**,
not a faithful port. Differences from the paper, stated plainly:

- **No serial range codec.** The paper ships a real arithmetic coder. A true
  per-step arithmetic coder is sequential and would bottleneck MLX's
  parallel decode. Instead, the entropy-coded byte size is **modelled from
  the measured Shannon entropy** of the delta stream — an estimate of what
  an ideal coder achieves, reported through `compressed_*_bytes`. The
  reconstructed tensors are exact regardless (this layer is lossless over
  the codes).
- **Never-worse-than-fixed-width cap.** A real arithmetic coder falls back
  to raw packing when the stream is incompressible, so the estimate is
  capped at the fixed-width packed size. On uncorrelated data the savings
  are exactly 0%, never negative.
- **No network streaming / bandwidth adaptation.** The paper's other major
  contribution — adapting per-chunk compression level to live network
  bandwidth during transmission — is out of scope here; this implementation
  covers the KV cache encoder only, not the streaming layer.
- **No model-level benchmark yet.** `benchmark_scripts/benchmark_cachegen.py`
  reports synthetic entropy savings (~17% on correlated 3-bit data, 0% on iid
  data); no end-to-end throughput or perplexity numbers have been run.

**Known limitation:** the win is a **storage** win. It does not reduce the
working-set memory at attend time (codes are dequantized to fp16 for SDPA),
so on Apple Silicon's bandwidth-bound decode it is lower-leverage than the
low-rank/cross-layer methods for raw inference speed.

## Evidence

All claims trace to passing tests in
`veloxquant_mlx/tests/cache/test_cachegen_cache.py` and
`veloxquant_mlx/tests/quantizers/test_cachegen.py`, covering:

- Reconstruction matches plain group quant exactly (lossless over codes)
- Token-delta transform is reversible (prefix-sum recovers the codes)
- Delta entropy < raw entropy on token-correlated data
- Per-channel entropy grouping is never worse than pooled entropy
- `entropy_savings > 0` on correlated data; `compressed < fixed_width`
- Savings never negative on iid data (the cap); `compressed <= fixed_width`
- `for_model` produces a non-increasing per-layer bit-width schedule
- Byte-accounting ordering: `compressed <= fixed_width < fp16`
