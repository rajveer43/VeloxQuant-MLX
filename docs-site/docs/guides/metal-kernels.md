---
id: metal-kernels
title: Metal GPU Kernels
sidebar_label: Metal Kernels
slug: /guides/metal-kernels
---

# Metal GPU Kernels

VeloxQuant-MLX compiles Metal kernel modules at runtime using `mx.fast.metal_kernel`, spanning quantization, fused attention, KV-cache eviction, and cross-model KV transfer. This guide explains what each kernel does, how they are loaded, performance characteristics, and fallback behaviour.

:::warning[Apple Silicon required]
All Metal kernels require macOS on an M-series chip. On unsupported hardware, VeloxQuant-MLX falls back to MLX Python ops automatically.
:::

## Available kernels

| Kernel module | Functions | Algorithm |
|---|---|---|
| `metal/_vecinfer.py` | `vecinfer_quantize_metal`, `vecinfer_dequant_metal`, `vecinfer_encode_decode_metal` | VecInfer PVQ |
| `metal/_rabitq.py` | `rabitq_hamming_score` | RaBitQ 1-bit |
| `metal/_rabitq_attend.py` | `rabitq_fused_attend` | RaBitQ asymmetric attention (1-bit keys + 4-bit values) |
| `metal/_rabitq_encode.py` | `rabitq_encode` | RaBitQ encode (rotate + binarize + pack + magnitude) |
| `metal/_rabitq_values.py` | `rabitq_pack_values` | Nibble packing for 4-bit value indices |
| `metal/_rabitq_prefill.py` | `rabitq_prefill_attend` | RaBitQ prefill/cross-attention on `simdgroup_matrix` tiles |
| `metal/_scalar_attend.py` | `scalar_fused_decode_attend` | Group-affine (KIVI / SKVQ / Kitty) decode + attention |
| `metal/_kivi_quant.py` | `kivi_group_quant_dequant` | KIVI asymmetric group quantize+dequantize round-trip |
| `metal/_comm_vq.py` | `comm_vq_decode_metal` | CommVQ RoPE |
| `metal/_crosskv_rope.py` | `crosskv_rope_recode` | Cross-model KV transfer (fused RoPE re-encode) |
| `metal/_scalar_quant.py` | `turboquant_scalar_quantize`, `turboquant_scalar_dequantize`, `turboquant_hadamard_quantize` | TurboQuant RVQ |
| `metal/_rvq_attend.py` | `turboquant_fused_rvq_decode_attend` | RVQ + attention fusion |
| `metal/_rvq_quant_pack.py` | `rvq_quant_pack` | Fused two-stage RVQ quantize, packed to uint32 |
| `metal/_qjl.py` | `qjl_encode`, `qjl_inner_product` | QJL |
| `metal/_bit_packing.py` | `turboquant_bit_pack`, `turboquant_bit_unpack` | All algorithms |
| `metal/fused_sdpa.py` | `metal_fused_sdpa` | All (fused attention) |
| `metal/_flash_prefill.py` | `flash_prefill_attend` | Plain-fp16 causal flash attention (from-scratch prefill) |
| `metal/_experimental_streaming_prefill.py` | `streaming_prefill_attend` | Row-owned streaming causal attention (experimental) |
| `metal/_h2o_evict.py` | `h2o_fused_evict` | H2O fused eviction: sink-protected argmin + evict + RoPE-remap |
| `metal/_keyformer_evict.py` | `keyformer_fused_evict` | Keyformer fused eviction: Gumbel-regularized argmin + evict + RoPE-remap |
| `metal/_qfilters_evict.py` | `qfilters_fused_evict`, `qfilters_score` | Q-Filters fused eviction: projection scoring + block top-k compaction |

## How kernels are loaded

Kernels are compiled **lazily on first use** via `mx.fast.metal_kernel`. The first call to any function in a kernel module triggers JIT compilation:

```python
import mlx.core as mx

# This triggers compilation on first call (~200-800ms)
from veloxquant_mlx.metal._scalar_quant import turboquant_scalar_quantize

keys = mx.random.normal(shape=(1, 8, 512, 128))
quantized = turboquant_scalar_quantize(keys, bits=1)  # compilation happens here

# Subsequent calls use the cached compiled kernel
quantized2 = turboquant_scalar_quantize(keys, bits=1)  # fast
```

Compiled kernels are cached in memory for the process lifetime. There is no persistent disk cache — each Python process recompiles on first use.

## Performance characteristics

All numbers below are from this repo's own benchmark scripts on an Apple M4 MacBook; each row states its exact configuration.

| Operation | Baseline | Metal kernel | Speedup | Configuration |
|---|---|---|---|---|
| VecInfer `quantize_vq` | 228 ms | 15.6 ms | **14.7×** | S=8192 (range 6.9–14.7× over S=128–8192; see `figures/metal/summary.png`) |
| RaBitQ fused attend (nibble-packed V) | 2.492 ms | 1.404 ms | **1.78×** | vs dequantize+SDPA, B=1 H=8 S_q=1 D=128 S_kv=8192 (`scripts/metal_rabitq_attend_bench.py`) |
| RaBitQ fused attend (nibble-packed V) | 0.681 ms | 0.481 ms | **1.42×** | same shape, S_kv=2048 |
| RaBitQ fused attend (nibble-packed V) | 0.309 ms | 0.281 ms | **1.10×** | same shape, S_kv=512 |
| RaBitQ encode | 4.511 ms | 0.752 ms | **6.0×** | vs numpy round-trip, N=32768 D=128 (`scripts/metal_rabitq_encode_bench.py`); 2.88× vs pure MLX ops |

Honest caveats: with *unpacked* (one byte per index) values the fused attend loses at short contexts (0.65× at S_kv=512) — nibble-packing the value indices (two per byte, `rabitq_pack_values`) halves value bandwidth and flips that to a small win. The encoder is a wash below N≈1024. All kernels are built for the long-context / large-batch regime.

## Fallback behaviour

VeloxQuant-MLX detects Metal availability at import time:

```python
from veloxquant_mlx.metal import metal_available

if metal_available():
    print("Metal kernels active")
else:
    print("Falling back to MLX Python ops")
```

When Metal is unavailable:
- All quantization and dequantization use equivalent pure MLX operations
- Attention scores use standard `mx.matmul`
- Fused SDPA reverts to the unfused path
- Performance is lower but results are numerically identical

## Fused SDPA kernel

The fused scaled dot-product attention kernel (`metal_fused_sdpa`) is the highest-impact optimisation. It combines:

1. Key dequantization
2. Scaled dot-product attention (`Q @ Kᵀ / √d`)
3. Softmax
4. Weighted sum of values

into a single Metal dispatch, avoiding materialising the full dequantized key matrix.

```python
from veloxquant_mlx.metal.fused_sdpa import metal_fused_sdpa, supports_shape

# Check compatibility
ok = supports_shape(batch=1, heads=8, seq_len=4096, head_dim=128)

if ok:
    attn_output = metal_fused_sdpa(
        queries=q,
        encoded_keys=encoded_k,  # compressed format from VecInfer
        values=v,
        scale=1.0 / (head_dim**0.5),
    )
```

## Fused RaBitQ asymmetric pipeline

Two kernels form a fully GPU-resident pipeline for an asymmetric-precision cache — **1-bit packed keys + 4-bit codebook values**, a combination that fused attention kernels normally can't express because keys and values use different formats:

- `rabitq_encode` — one dispatch turns raw fp16 keys into the cache representation: randomized Hadamard rotation (threadgroup butterfly), sign binarization via `simd_ballot` (each SIMD-group's 32 sign bits land in one vote mask = 4 packed bytes), and the per-vector L1/D magnitude.
- `rabitq_fused_attend` — one dispatch scores every cached slot directly from the packed bits (XOR + popcount), runs an online softmax, and accumulates values from the 4-bit codebook. No dequantized K or V matrix is ever materialized. The kv axis is split across 8 SIMD-groups flash-decoding style so decode-shaped calls still fill the GPU.
- `rabitq_pack_values` — packs two 4-bit value indices per byte (low nibble = even dim). The attend kernel detects the packed shape (`[.., D//2]`) automatically and reads nibbles directly — half the value-cache memory and bandwidth, bit-identical outputs to the unpacked path.

```python
import mlx.core as mx
from veloxquant_mlx.metal.kernels import rabitq_encode, rabitq_fused_attend

# Encode: [N, D] fp16 keys -> packed bits + per-vector magnitude
k_bits_flat, k_mag_flat = rabitq_encode(keys, diag)  # [N, D//8] uint8, [N] fp32

# Attend: score packed keys, gather 4-bit values — single dispatch
out = rabitq_fused_attend(
    q,  # [B, H, S_q, D]    fp16, pre-rotated
    q_scale,  # [B, H, S_q]       fp32, e.g. L1(q)/D (fold in 1/sqrt(D))
    k_bits,  # [B, H, S_kv, D/8] uint8 packed sign bits
    k_mag,  # [B, H, S_kv]      fp32 per-key magnitude
    k_const,  # [B, H, S_kv]      fp32 additive bias (zeros for centroid-free)
    v_idx,  # [B, H, S_kv, D]   uint8 value codebook indices
    v_cents,  # [16]              fp32 scalar value codebook
)  # -> [B, H, S_q, D] fp16
```

The score per slot is `(D − 2·ham) · q_scale · k_mag + k_const`, the sign-bit estimate of `⟨q, k⟩`. Parity is verified against a numpy reference in `veloxquant_mlx/tests/metal/test_rabitq_attend.py` and `test_rabitq_encode.py`, including an end-to-end encode→attend test.

### Prefill / cross-attention

`rabitq_fused_attend` is decode-shaped: one query per threadgroup, scalar dot products. When `S_q` is large — the multi-turn VLM case, where a new turn attends over a long compressed image-token history — that layout leaves the matrix pipeline idle. `rabitq_prefill_attend` is the matmul-shaped companion: both `Q·K̂ᵀ` and `W·V̂` run on 8×8 `simdgroup_matrix` tiles, with keys sign-decoded and values nibble-decoded inside the tile loop.

Two differences from the decode kernel matter in practice:

- Scores are **exact dots** against sign-decoded keys (`(q · signs·k_mag)·scale + k_const`), not the Hamming estimate.
- It is **cross-attention only** — every query row attends over all `S_kv` slots with no causal mask. New-token self-attention belongs on the fp16 path.

Values must be nibble-packed (the `rabitq_pack_values` format).

## Fused group-affine (KIVI-style) attention

`scalar_fused_decode_attend` is the scalar/group-quant analogue of the codebook fused attends above — it serves the **KIVI / SKVQ / Kitty / group-quant family**, where keys and values are stored as `uint8` codes plus a per-group `(scale, zero)` pair rather than a codebook.

The pure-MLX path for these methods reconstructs `code * scale + zero` into a full fp16 tensor and then calls `scaled_dot_product_attention`, paying a `dequantize → DRAM → SDPA` round-trip every decode step. This kernel reconstructs `x_hat` in-register inside a FlashAttention-style online softmax, so no dequantized `K_hat`/`V_hat` ever reaches DRAM. The win compounds with context: the fp16 `K_hat` grows linearly with `S_kv` while the packed codes stay `16/b` times smaller.

Note the two grouping axes are different — keys group along tokens (per-channel), values along channels (per-token), matching KIVI's layout:

```python
import mlx.core as mx
from veloxquant_mlx.metal.kernels import scalar_fused_decode_attend

out = scalar_fused_decode_attend(
    q,  # [B, H, S_q, D]    fp16 queries (pre-rotated)
    k_codes,  # [B, H, S_kv, D]   uint8  key codes
    k_scale,  # [B, H, GK, D]     fp32   GK = ceil(S_kv / group_size)
    k_zero,  # [B, H, GK, D]     fp32
    v_codes,  # [B, H, S_kv, D]   uint8  value codes
    v_scale,  # [B, H, S_kv, GV]  fp32   GV = ceil(D / group_size)
    v_zero,  # [B, H, S_kv, GV]  fp32
    group_size=32,
    scale=1.0 / (D**0.5),
    # nsg omitted -> autotuned from the dispatch shape (recommended)
)  # -> [B, H, S_q, D] fp16
```

One compiled kernel serves any `(S_kv, D, g)` — the group counts are read from the passed shapes. `D` must be ≤ 256 and `nsg` in `1..32`.

### Tuning `nsg` (autotuned by default)

`nsg` sets how many SIMD-groups split the kv axis *inside* each threadgroup. It matters more than it looks: at a real single-token decode step the kernel dispatches only `n_tg = B * H_kv * S_q` threadgroups — often just 4–8 on a 10-core GPU — and total concurrency is roughly `n_tg * nsg`. When `n_tg` alone can't fill the machine, widening each threadgroup is the only way to add work **without giving up threadgroup count** (which is why this beats GQA head-packing, measured 2.7–4.7× *slower* for trading count away).

Leaving `nsg` unset picks the value from the dispatch shape: the largest the threadgroup-memory budget admits when under-dispatched, backing off once `n_tg ≥ 32`. That is **1.2–4.2× faster** than the previous fixed default of 4:

| shape | `n_tg` | `S_kv` | chosen `nsg` | vs `nsg=4` |
|---|---:|---:|---:|---:|
| `H_q=8 H_kv=8` (MHA) | 8 | 16 384 | 32 | **4.19×** |
| `H_q=32 H_kv=8` (GQA 4) | 8 | 16 384 | 16 | **2.97×** |
| `H_q=28 H_kv=4` (GQA 7) | 4 | 16 384 | 16 | **2.64×** |
| `H_q=32 H_kv=4` (GQA 8) | 4 | 16 384 | 8 | **1.84×** |
| `H_q=32 H_kv=8`, B=8 | 64 | 16 384 | 8 | 1.19× |

The win is largest where `n_tg` is smallest and grows with `S_kv` — the signature of an occupancy-limited kernel. Once dispatch alone saturates the GPU (`n_tg ≥ 32`, e.g. batched serving) it flattens to ~1.0–1.2×.

The budget bounds `nsg * heads_per_kv * ceil(D/32)`; at `D=128` that means up to `nsg=32` for MHA, 16 at `heads_per_kv=4`, and 8 at `heads_per_kv=8`. Pass an explicit `nsg` to pin it for benchmarking or an unusual shape.

Full sweep, real-model throughput, and limitations: [`docs/NSG_AUTOTUNE_BENCHMARK_REPORT.md`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/docs/NSG_AUTOTUNE_BENCHMARK_REPORT.md).

Measured on Apple M4 (10-core GPU), `B=1 H=32 D=128 b=2 g=32 S_q=1`, versus dequantize → MLX SDPA:

| `S_kv` | Speedup |
|---|---|
| 512 | 6.4× |
| 65536 | 12.2× |

Parity max abs error is `1.2e-4` against the reference — the kernel accumulates its softmax in fp32, so it is *more* accurate than the fp16 baseline it replaces. See `veloxquant_mlx/tests/metal/test_scalar_attend.py`.

## KIVI group quantize+dequantize round-trip

`kivi_group_quant_dequant` fuses the whole `KIVIKVCache._quant_dequant_along` round-trip — `moveaxis → pad → reshape → min/max → round/clip → reconstruct → moveaxis back` — into a single dispatch, instead of materializing each intermediate as a full-size MLX array.

Two kernel variants exist because KIVI's two schemes have opposite memory layouts against the cache's `[..., S, D]` row-contiguous storage:

- **Per-token** (`axis=-1`, values) groups along `D`, the contiguous axis — one SIMD-group per quantization group, lanes split the group, and a `simd_shuffle_xor` butterfly reduces without barriers.
- **Per-channel** (`axis=-2`, keys) groups along `S`, the strided axis — one thread owns a whole group outright, removing the cross-thread reduction entirely. (Transposing to reuse the per-token kernel was tried and rejected: the transpose's own memory traffic outweighed the fusion's savings.)

```python
from veloxquant_mlx.metal.kernels import kivi_group_quant_dequant

out = kivi_group_quant_dequant(
    x,  # [..., S, D] fp16/fp32/bf16
    axis=-2,  # -2 = per-channel (keys), -1 = per-token (values)
    group_size=32,
    levels=3,  # 2**b - 1
    eps=1e-8,  # floors the scale for degenerate (min == max) groups
)
```

Bit-exactness with the MLX path depends on three details the parity tests pin: padding replicates the last live element (not zero), rounding is half-to-even (`rint`, not `metal::round`), and `eps` floors the scale for single-element groups. See `veloxquant_mlx/metal/_kivi_quant.py` for the full rationale.

## Fused RVQ quantize + pack

`rvq_quant_pack` fuses `TurboQuantRVQKVCache.update_and_fetch`'s two-stage codebook quantize (stage-1 nearest-centroid, dequantize, residual, stage-2 nearest-centroid) and both bit-packing passes into one dispatch — the MLX path otherwise materializes two full `(N, D)` uint8 index buffers before packing each with a separate kernel.

```python
from veloxquant_mlx.metal.kernels import rvq_quant_pack

packed1, packed2 = rvq_quant_pack(
    rotated,  # [N, D] fp16/fp32 — post-Hadamard/QR rotated vectors, D a power of two <= 1024
    centroids1,  # [2**bits] stage-1 sorted centroids
    boundaries1,  # [2**bits - 1] stage-1 Voronoi boundaries
    boundaries2,  # [2**bits - 1] stage-2 (residual) Voronoi boundaries
    bits=2,  # 1-4
)  # -> (packed1, packed2): [N, ceil(D / (32 // bits))] uint32 each
```

Matches `ScalarCodebook.quantize`'s boundary-count comparisons (not a naive argmin) and `_pack_indices`'s LSB-first packing exactly, so ties and partial trailing words resolve identically to the pure-MLX path.

## Cross-model KV transfer (CrossKV RoPE recode)

`crosskv_rope_recode` fuses the `strip_rope → apply_rope` pair used when transplanting a KV cache from one model to another with a different `rope_theta`. Both rotations act on the same `(d, d + D/2)` element pair at the same position, so they compose into a single rotation by the per-dimension angle *difference* — one dispatch, no intermediate `[N, D]` array, no materialized cos/sin tables.

```python
from veloxquant_mlx.metal.kernels import crosskv_rope_recode

out = crosskv_rope_recode(
    keys,  # [BH, N, D] fp16/fp32, rotated under source_base, D even
    positions,  # [N] absolute positions, shared across BH groups
    source_base,  # source model's rope_theta
    target_base,  # target model's rope_theta
)  # -> [BH, N, D], same dtype as keys
```

Numerically equivalent to `apply_rope(strip_rope(keys, positions, source_base), positions, target_base)`. Unlike the eviction kernels' RoPE remap (same base, position-delta only), here the *bases* differ between the two rotations, so the angle difference is taken per-dimension after exponentiation rather than collapsing to a position delta.

## Plain-fp16 prefill attention (from-scratch)

For a fresh conversation there is no compressed cache to exploit yet — K/V exist at full precision for the first time. `flash_prefill_attend` targets exactly that case: standard causal self-attention over plain fp16 Q/K/V on `simdgroup_matrix` tiles, with no compression, no cross-attention mode, no mask tensor, and no attention sinks — every one of those is baked out at compile time rather than handled with runtime branches.

```python
from veloxquant_mlx.metal.kernels import flash_prefill_attend

out = flash_prefill_attend(
    q,  # [B, H, S_q, D]  fp16 — queries
    k,  # [B, H, S_kv, D] fp16 — plain (uncompressed) keys
    v,  # [B, H, S_kv, D] fp16 — plain (uncompressed) values
    scale,  # [1] fp32 — 1/sqrt(D)
)  # -> [B, H, S_q, D] fp16
```

Always causal, aligning queries to the tail of the KV cache (`q_abs = (S_kv - S_q) + q_pos`) — the plain-fp16 counterpart to `rabitq_prefill_attend(causal=True)`, tuned for `S_q ≈ S_kv` with no pre-existing cache. Tile sizes (`BQ`, `BK`, the W·V depth-tile batch `PDT`) are chosen per head-dim from a measured lookup table (see `blogs/prefill-roofline.md`), not a generic memory-budget heuristic — e.g. `BK=32` wins at `D=32` but is structurally infeasible at `D=128` (would blow the 32 KB threadgroup budget). `D` must be divisible by 8 and ≤ 128.

### Experimental: row-owned streaming prefill

`streaming_prefill_attend` is a from-scratch alternative decomposition to `flash_prefill_attend`, built to benchmark a genuinely different design against the tiled `simdgroup_matrix` approach — **not** a replacement; `flash_prefill_attend` remains the production kernel.

Instead of matrix tiles staged through threadgroup memory, ownership is **by query row**: one SIMD-group (32 lanes) owns one query row for the entire kernel, each lane owns a fixed stride-32 slab of head-dims, and K/V stream directly from device memory one (or a small block of) token(s) at a time — relying on the GPU's L2 cache rather than explicit threadgroup reuse. The online-softmax state is redundantly (but bit-identically) computed in every lane, trading a few scalar FLOPs for removing a cross-lane broadcast. Zero threadgroup memory, zero barriers, anywhere in the kernel family.

```python
from veloxquant_mlx.metal.kernels import streaming_prefill_attend

out = streaming_prefill_attend(
    q,
    k,
    v,  # same shapes/dtypes as flash_prefill_attend; D must be a multiple of 32, <= 128
    scale,
    implementation="streaming",  # or streaming_block{2,4,8}, streaming_multirow
)
```

`implementation` selects the variant: `"streaming"` is the block=1 baseline; `"streaming_block2/4/8"` unroll the KV loop to amortize the softmax update over more tokens per step; `"streaming_multirow"` dispatches 4 independent SIMD-groups per threadgroup purely for occupancy (no data sharing between them). See `veloxquant_mlx/metal/src/experimental_streaming_prefill_ARCHITECTURE.md` for the full design rationale and the hypotheses the benchmarking step was meant to confirm or refute — matrix-unit throughput is expected to win at large `D`/compute-bound `S`, while the streaming kernel's zero fixed overhead is expected to win at small `S`.

## Fused KV-cache eviction kernels

Three quantizer families — **H2O**, **Keyformer**, and **Q-Filters** — evict cache rows once a budget is exceeded. Each was originally a per-`(batch, head)` Python loop; these kernels batch the eviction decision and the row compaction across every `(batch, head)` group in two GPU dispatches.

All three require callers to only invoke them when every group is already over budget — the below-budget case has no eviction step and is handled entirely by the existing vectorized MLX path.

### H2O fused evict

`h2o_fused_evict` replaces `h2o_update`'s over-budget branch: sink-protected argmin, evict, and RoPE-remap the rows whose position shifted.

```python
from veloxquant_mlx.metal.kernels import h2o_fused_evict

keys_out, values_out, scores_out, positions_out = h2o_fused_evict(
    keys_mid,  # [BH, n_total, D] fp16 — n_kept stored + 1 newly appended
    values_mid,  # [BH, n_total, D] fp16
    scores_mid,  # [BH, n_total] fp32 — cumulative scores, appended row = 0.0
    positions_mid,  # [BH, n_total] int32 — absolute positions
    n_sink=4,  # leading positions protected from eviction
    rope_base=10000.0,  # must match the model's rope_theta
    grace=0,  # trailing (most-recent) rows also protected
    nsg=4,  # SIMD-groups per threadgroup for the reduction dispatch
)  # each output has n_total - 1 rows (one evicted)
```

Two dispatches, not one barrier-fused kernel: dispatch 1 is a sink/grace-protected argmin reduction (SIMD butterfly + threadgroup merge, ties resolve to the lowest index like `mx.argmin`) that every thread needs before dispatch 2 can compact the surviving rows and re-rotate (NeoX-style RoPE) only the rows whose position shifted.

### Keyformer fused evict

`keyformer_fused_evict` is structurally the H2O kernel with one addition threaded through both dispatches: a per-row frozen Gumbel value, folded into the selection score as `score + tau * gumbel`. At `tau == 0` the reduction is bit-for-bit identical to H2O's.

```python
from veloxquant_mlx.metal.kernels import keyformer_fused_evict

keys_out, values_out, scores_out, gumbel_out, positions_out = keyformer_fused_evict(
    keys_mid,
    values_mid,
    scores_mid,
    gumbel_mid,  # [BH, n_total] fp32 — frozen per-position Gumbel noise
    positions_mid,
    n_sink=4,
    rope_base=10000.0,
    tau=0.5,  # annealed Gumbel temperature; 0.0 collapses to H2O's raw-score argmin
    recent=0,  # trailing rows protected from eviction
    nsg=4,
)
```

### Q-Filters fused evict

Q-Filters evicts a whole block down to `budget` in one shot rather than one row per token, so its two dispatches split differently: `qfilters_score` computes a full `[BH, n_total]` projection score array (paper Theorem 3.3) with sink/recent rows forced to `+inf` so they always survive; the keep-threshold is then chosen on the MLX side via `mx.sort` (reusing MLX's own top-k rather than re-deriving one in Metal), and `qfilters_fused_evict`'s second dispatch compacts against that threshold.

```python
from veloxquant_mlx.metal.kernels import qfilters_fused_evict, qfilters_score

keys_out, values_out, scores_out = qfilters_fused_evict(
    keys_mid,  # [BH, n_total, D] fp16
    values_mid,  # [BH, n_total, D] fp16
    filter_dir,  # [BH, D] fp32 — frozen per-group unit-norm Q-Filter
    budget=2048,  # rows to retain, <= QFILTERS_MAX_BUDGET (4096)
    n_sink=4,
    recent=64,
    sign=1,  # +1 = paper direction (keep highest projections), -1 = inverted ablation
)  # -> (keys_out, values_out, scores_out), each [BH, budget, ...], temporal order preserved
```

Unlike H2O/Keyformer, Q-Filters does **not** remap position ids after eviction (a documented limitation), so keys are copied bit-identically. `budget` is capped at `QFILTERS_MAX_BUDGET = 4096` — the apply kernel stages the survivor index list in threadgroup memory, which must be a compile-time size (16 KB at the cap, within Metal's 32 KB threadgroup budget); larger budgets should use the pure-MLX path.

## Bit packing

Sub-byte indices (1-bit, 2-bit) are packed into uint32 words to minimise memory bandwidth:

```python
from veloxquant_mlx.metal._bit_packing import turboquant_bit_pack, turboquant_bit_unpack
import mlx.core as mx

# indices: int32 in range [0, 2^bits)
indices = mx.array([[0, 1, 0, 1, 1, 0, 0, 1, ...]], dtype=mx.int32)

packed = turboquant_bit_pack(indices, bits=1)
# packed: uint32, 32× smaller than indices

recovered = turboquant_bit_unpack(packed, bits=1, original_length=indices.shape[-1])
```

## Debugging kernel issues

If you see Metal errors, enable verbose kernel output:

```bash
MLX_METAL_DEBUG=1 python your_script.py
```

Common issues:

| Error | Cause | Fix |
|---|---|---|
| `Metal kernel compilation failed` | Xcode CLI tools missing | `xcode-select --install` |
| `Kernel shape mismatch` | head_dim not a multiple of 32 | Use `supports_shape()` to check |
| `Metal device not found` | Running in VM or Rosetta | Run natively on Apple Silicon |

## See also

- [mlx_lm integration](../guides/mlx-lm-integration)
- [API — Metal functions](../api/metal-api)
- [Installation troubleshooting](../getting-started/installation)
