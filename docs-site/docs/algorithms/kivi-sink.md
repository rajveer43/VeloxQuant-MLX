# KVSink-Adapted Sink Protection

**Method id:** `kivi_sink` · **New in 0.9.0** · *Inspired by* [KVSink (Su & Yuan,
COLM 2025, arXiv:2508.04257)](https://arxiv.org/abs/2508.04257) — **adapted, not a
faithful port** (see "Fidelity to the paper" below).

Attention-sink tokens — positions that absorb disproportionate attention mass —
are where low-bit KV quantization breaks first. `kivi_sink` layers dynamic sink
protection on top of [KIVI](../algorithms/kivi)'s deterministic group
quantization: the top-k highest-key-norm token positions are kept in fp16 and
**excluded from quantization-parameter calibration**, while everything else is
quantized as usual.

## Usage

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="kivi_sink",
    bit_width_inlier=2,
    kivi_group_size=32,
    residual_length=32,
    n_sink_tokens=5,  # top-k high-key-norm tokens kept fp16
)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches
```

## How it works

1. On every `update_and_fetch`, the per-token key L2-norm (mean over KV heads)
   is folded into a running top-k of the highest-norm absolute positions.
2. Tokens in that sink set are stored fp16, never quantized.
3. **Calibration exclusion** — the detail that matters: before computing each
   group's min/max quantization parameters, sink rows are replaced by the
   nearest non-sink row. Without this, a large-magnitude sink inflates its
   group's scale and ruins every neighbor in the group even though the sink
   itself is restored. (The KVSink paper calls this out explicitly; our unit
   tests reproduce the failure when it is omitted.)
4. KIVI's fp16 residual window still applies; the two mechanisms compose, and
   byte accounting tracks `sink_fp16_bytes` separately from
   `residual_fp16_bytes` with no double counting.

Deterministic end to end: top-k on key norm + min/max group quantization. No
codebook training, no RNG.

## Fidelity to the paper

KVSink's true mechanism detects sinks via extreme-magnitude outlier channels in
the **hidden state** at a model-specific "emergence layer". VeloxQuant-MLX's
cache wrappers never see the hidden state — by design, they receive only
per-layer K/V tensors, which is what keeps the three-line integration free of
model surgery. This implementation uses the cache-observable proxy: sink tokens
also exhibit anomalously **large key L2-norm** (the same outlier phenomenon; the
library's `KeyNormObserver` is built on this signal).

Known v1 limitation: sink selection is **prefill-dominant**. A token quantized
in an earlier call is not retroactively restored if it later qualifies as a
sink. In practice attention sinks emerge among early tokens, which arrive in
the prefill block where protection is fully effective.

## Evidence (unit tests; end-to-end benchmark not yet run)

All claims below trace to passing tests in
`veloxquant_mlx/tests/cache/test_sink_cache.py` on synthetic data with planted
high-norm sink tokens (25× magnitude, positions {0, 7, 20, 41, 90}, S=128,
b=2):

- Planted sinks are detected and preserved **bit-exact fp16**; neighbors are
  quantized.
- Sink-protected KIVI achieves **lower key reconstruction MSE than plain
  KIVI** at the same bit-width.
- Dynamic selection achieves **lower MSE than Preserve-First-N at equal fp16
  budget** when sinks are not all at the front (the KVSink paper's central
  claim, reproduced at cache level).
- `n_sink_tokens=0` reproduces plain KIVI bit-for-bit.
- Byte accounting partitions tokens exactly across compressed / sink /
  residual pools.

**No model-level benchmark has been run yet.** `benchmark_scripts/benchmark_sink.py`
is ready (fp16 / KIVI-2bit / +sink k=5 / +sink k=20 on the long-prompt
protocol); until its `results.json` is committed, no throughput or compression
figures are claimed for this method.

## When to use it

Reach for `kivi_sink` over plain `kivi` when running at aggressive bit-widths
(b=2) where sink-token quantization error is the dominant failure mode. The
cost is k tokens of fp16 storage per layer — negligible at k=5.
