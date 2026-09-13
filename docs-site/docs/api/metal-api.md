---
id: metal-api
title: Metal Kernels API
sidebar_label: Metal Kernels
slug: /api/metal-api
description: Python API reference for veloxquant_mlx.metal, covering low-level Apple Silicon Metal kernels for VecInfer, RaBitQ, KIVI-style group-affine attention, CommVQ, cross-model RoPE recoding, scalar quantization, RVQ fusion, prefill attention, KV-cache eviction, fused SDPA, bit packing, and QJL.
keywords: [metal kernels, Metal, "API reference", "python api", Apple Silicon, fused SDPA, GPU kernels]
---

# Metal Kernels API

`veloxquant_mlx.metal`

All Metal kernels are compiled lazily on first call via `mx.fast.metal_kernel`. These are low-level functions — most users should interact with them indirectly through quantizer and cache classes.

:::warning[Apple Silicon only]
All functions in this module require macOS on an M-series chip (Metal GPU access via `mx.fast.metal_kernel`). There is no dedicated `MetalUnavailableError` class — check `metal_available()` before calling these directly, or let the higher-level quantizer/cache classes fall back to their pure-MLX implementations.
:::

---

## Availability check

```python
from veloxquant_mlx.metal import metal_available

if not metal_available():
    raise RuntimeError("Metal not available on this device")
```

---

## VecInfer kernels

`veloxquant_mlx.metal._vecinfer`

### `vecinfer_quantize_metal`

```python
def vecinfer_quantize_metal(
    x: mx.array,
    codebook: mx.array,
    sub_dim: int,
) -> mx.array
```

Drop-in Metal replacement for `veloxquant_mlx.allocators.vecinfer.quantize_vq`. Computes squared distances in thread-local registers (peak memory O(N) instead of O(N · n_centroids · sub_dim)).

| Parameter | Type | Description |
|---|---|---|
| `x` | `mx.array` | `[..., D]` input, `D` divisible by `sub_dim` |
| `codebook` | `mx.array` | `[n_centroids, sub_dim]` |
| `sub_dim` | `int` | Sub-vector dimension |

**Returns:** `[..., D // sub_dim]` int32 codebook indices.

**Raises:** `ValueError` if `D` is not divisible by `sub_dim`, or `codebook` isn't shaped `[n_centroids, sub_dim]`.

---

### `vecinfer_dequant_metal`

```python
def vecinfer_dequant_metal(
    indices: mx.array,
    codebook: mx.array,
    out_dtype: mx.Dtype | None = None,
) -> mx.array
```

Drop-in Metal replacement for `veloxquant_mlx.allocators.vecinfer.dequantize_vq` — a codebook gather.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `indices` | `mx.array` | Required | `[..., n_sub]` codebook indices, promoted to uint32 |
| `codebook` | `mx.array` | Required | `[n_centroids, sub_dim]` centroid table |
| `out_dtype` | `mx.Dtype \| None` | `None` | Output dtype; defaults to `codebook.dtype` |

**Returns:** `[..., n_sub * sub_dim]` reconstruction.

**Raises:** `ValueError` if `codebook` is not 2D.

---

### `vecinfer_encode_decode_metal`

```python
def vecinfer_encode_decode_metal(
    keys: mx.array,
    k_codebook: mx.array,
    sub_dim: int,
    H_mat: mx.array,
    smooth: mx.array | None = None,
) -> tuple[mx.array, mx.array]
```

Fused key encode+decode in a single Metal dispatch: smooth → Walsh-Hadamard transform → VQ encode → dequant → inverse-WHT → inverse-smooth. Replaces 7 MLX graph nodes with one dispatch.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `keys` | `mx.array` | Required | `[B, H, S, D]` fp16 or fp32 |
| `k_codebook` | `mx.array` | Required | `[n_centroids, sub_dim]` fp32 centroids |
| `sub_dim` | `int` | Required | Sub-vector size; must divide `D` |
| `H_mat` | `mx.array` | Required | `[D, D]` Walsh-Hadamard matrix (fp32) |
| `smooth` | `mx.array \| None` | `None` | `[H, D]` or `[D]` smooth factors, or `None` to skip smoothing |

**Returns:** `(k_hat, k_indices)` — `k_hat` is `[B, H, S, D]` fp16 (the reconstructed, smoothed+rotated-then-inverted keys); `k_indices` is `[B, H, S, n_sub]` int32.

**Raises:** `ValueError` if `keys` is not 4D, `D` is not divisible by `sub_dim`, or `D > 512` (threadgroup limit).

There is a similarly-named `vecinfer_encode_decode_simple_metal(values, v_codebook, ...)` for the value-side path (no smooth/Hadamard step) — see `veloxquant_mlx/metal/_vecinfer.py` for its signature.

---

Note: `compute_query_lut` (asymmetric-MIPS query-codebook LUT precompute) is a plain-Python/MLX function, not a Metal kernel — it lives in `veloxquant_mlx.allocators.vecinfer` and is documented on the [Allocators API](./allocators#compute_query_lut) page.

---

## RaBitQ kernels

`veloxquant_mlx.metal._rabitq`

### `rabitq_hamming_score`

```python
def rabitq_hamming_score(
    qbits: mx.array,   # [D//8] uint8   — packed query sign bits
    bits: mx.array,    # [N, D//8] uint8 — packed candidate sign bits
    Cx: mx.array,      # [N] float32    — per-candidate constant
    scale: mx.array,   # [1] float32    — ||qhat - c||_1 / D
) -> mx.array
```

XOR + popcount Hamming scoring for N candidates against one query:
`score[i] = popcount(XOR(qbits, bits[i])) * scale + Cx[i]`.

- Returns: `[N]` float32 approximate distances (lower = closer)

### `rabitq_fused_attend`

`veloxquant_mlx.metal._rabitq_attend`

```python
def rabitq_fused_attend(
    q: mx.array,        # [B, H, S_q, D]    fp16  — queries (pre-rotated)
    q_scale: mx.array,  # [B, H, S_q]       fp32  — per-query score scale
    k_bits: mx.array,   # [B, H, S_kv, D/8] uint8 — packed 1-bit key signs
    k_mag: mx.array,    # [B, H, S_kv]      fp32  — per-key magnitude
    k_const: mx.array,  # [B, H, S_kv]      fp32  — additive score bias
    v_idx: mx.array,    # [B, H, S_kv, D] or [B, H, S_kv, D//2] uint8
    v_cents: mx.array,  # [n_cents]         fp32  — scalar value codebook
) -> mx.array
```

Single-dispatch attention over an asymmetric cache (1-bit keys + codebook values). Scores each slot from packed bits via `(D - 2*ham) * q_scale * k_mag + k_const`, runs an online softmax split across 8 SIMD-groups (flash-decoding), and accumulates codebook values. Fold any `1/sqrt(D)` scaling into `q_scale`/`k_const`. Requires D divisible by 8, D ≤ 256.

`v_idx` may be one index per element (`[.., D]`) or nibble-packed (`[.., D//2]`, from `rabitq_pack_values`) — the format is detected from the shape; packed requires ≤ 16 codebook entries and produces bit-identical outputs.

- Returns: `[B, H, S_q, D]` fp16 attention output

### `rabitq_pack_values`

`veloxquant_mlx.metal._rabitq_values`

```python
def rabitq_pack_values(v_idx: mx.array) -> mx.array
```

Packs 4-bit value indices two-per-byte along the last axis (low nibble = even element; values masked to 4 bits). Any shape with an even last dimension.

- Returns: uint8 array with the last dimension halved — feed directly to `rabitq_fused_attend`

### `rabitq_encode`

`veloxquant_mlx.metal._rabitq_encode`

```python
def rabitq_encode(
    keys: mx.array,  # [N, D] fp16/fp32 — raw (pre-rotation) key vectors
    diag: mx.array,  # [D] fp32 — +-1 Hadamard diagonal
) -> tuple[mx.array, mx.array]
```

Fused rotate + binarize + bit-pack + L1-magnitude in one dispatch; sign packing uses `simd_ballot`. Outputs plug into `rabitq_fused_attend` as `k_bits`/`k_mag` (with `k_const = 0`). Requires D a power of two, divisible by 8, ≤ 1024.

- Returns: `(k_bits [N, D//8] uint8, k_mag [N] fp32)`

### `rabitq_prefill_attend`

`veloxquant_mlx.metal._rabitq_prefill`

```python
def rabitq_prefill_attend(
    q: mx.array,        # [B, H, S_q, D]    fp16  — new-turn queries
    scale: mx.array,    # [1]               fp32  — softmax scale (1/sqrt(D))
    k_bits: mx.array,   # [B, H, S_kv, D/8] uint8 — packed 1-bit key signs
    k_mag: mx.array,    # [B, H, S_kv]      fp32  — per-key magnitude
    k_const: mx.array,  # [B, H, S_kv]      fp32  — additive score bias
    v_idx: mx.array,    # [B, H, S_kv, D/2] uint8 — nibble-packed value indices
    v_cents: mx.array,  # [n_cents <= 16]   fp32  — scalar value codebook
    *,
    causal: bool = False,
) -> mx.array
```

Prefill-shaped companion to `rabitq_fused_attend`, for large `S_q` (multi-turn VLM: a new turn attending over compressed image-token history). Both `Q·K̂ᵀ` and `W·V̂` run on 8×8 `simdgroup_matrix` tiles; K is sign-decoded and V nibble-decoded inside the tile loop, so no dequantized K/V is materialized.

Scores are exact dots — `(q · signs·k_mag) * scale + k_const` — not the Hamming estimate the decode kernel uses. By default this is cross-attention: every query row attends over all `S_kv` slots with no mask. Pass `causal=True` for autoregressive self-attention prefill — queries then align to the tail of the KV cache (`q_abs = (S_kv - S_q) + q_pos`, matching `fused_sdpa`'s convention), and slot `j` is masked whenever `j > q_abs`. Values must be nibble-packed (`rabitq_pack_values` format). Requires `D % 8 == 0`, `D <= 128` (threadgroup memory budget).

- Returns: `[B, H, S_q, D]` fp16 attention output

---

## Group-affine (KIVI-style) attention

`veloxquant_mlx.metal._scalar_attend`

### `scalar_fused_decode_attend`

```python
def scalar_fused_decode_attend(
    q: mx.array,        # [B, H, S_q, D]   fp16/fp32 — queries (pre-rotated)
    k_codes: mx.array,  # [B, H, S_kv, D]  uint8 — key codes
    k_scale: mx.array,  # [B, H, GK, D]    fp32  — GK = ceil(S_kv/group_size)
    k_zero: mx.array,   # [B, H, GK, D]    fp32
    v_codes: mx.array,  # [B, H, S_kv, D]  uint8 — value codes
    v_scale: mx.array,  # [B, H, S_kv, GV] fp32  — GV = ceil(D/group_size)
    v_zero: mx.array,   # [B, H, S_kv, GV] fp32
    group_size: int,
    scale: float,
    nsg: int | None = None,   # None -> autotuned from the dispatch shape
) -> mx.array
```

Single-dispatch SDP attention directly over an asymmetric group-min/max ("affine") quantized cache — the KIVI / SKVQ / Kitty / group-quant family. Reconstructs `k_hat = k_codes*k_scale + k_zero` (per-channel groups) and `v_hat = v_codes*v_scale + v_zero` (per-token groups) in-register inside a FlashAttention-style online softmax; no fp16 `K_hat`/`V_hat` is written to DRAM.

The kv axis is split across `nsg` SIMD-groups flash-decoding style so single-query decode shapes still fill the GPU. **`nsg` defaults to `None`, which autotunes it from the dispatch shape** — total GPU concurrency is roughly `n_tg * nsg` (where `n_tg = B * H_kv * S_q`), so under-dispatched decode shapes get a wider threadgroup and already-saturated ones back off. Measured **1.2–4.2× faster** than the previous fixed default of `nsg=4`. Pass an explicit int to pin it.

Constraints: `q` must be 4-D, `D ≤ 256`, `1 ≤ nsg ≤ 32`. The threadgroup-memory budget bounds `nsg * heads_per_kv * ceil(D/32)`; at `D=128` that admits `nsg` up to 32 (MHA), 16 (`heads_per_kv=4`), or 8 (`heads_per_kv=8`).

Measured on Apple M4 (B=1, H=32, D=128, b=2, g=32, S_q=1) vs. dequantize → MLX SDPA: **6.4× at S_kv=512, rising to 12.2× at S_kv=65536**. Softmax accumulates in fp32, so parity error (`1.2e-4` max abs) is better than the fp16 baseline.

- Returns: `[B, H, S_q, D]` fp16 attention output

---

## KIVI group quantization

`veloxquant_mlx.metal._kivi_quant`

### `kivi_group_quant_dequant`

```python
def kivi_group_quant_dequant(
    x: mx.array,
    axis: int,
    group_size: int,
    levels: int,
    eps: float = 1e-8,
) -> mx.array
```

Fuses `KIVIKVCache._quant_dequant_along`'s full round-trip (moveaxis → pad → group min/max → round/clip → reconstruct → moveaxis back) into one dispatch. `axis=-2` (per-channel, keys) uses one thread per group; `axis=-1` (per-token, values) uses one SIMD-group per group with a `simd_shuffle_xor` reduction. Padding replicates the last live element, rounding is half-to-even, `eps` floors degenerate group scales — all pinned by parity tests against the MLX path.

- Returns: array of `x`'s shape and dtype, quantized and reconstructed

---

## CommVQ kernels

`veloxquant_mlx.metal._comm_vq`

### `comm_vq_decode_metal`

```python
def comm_vq_decode_metal(
    indices: mx.array,
    codebook: mx.array,
    positions: mx.array,
    inv_freq: mx.array,
    n_cb: int,
    sub_dim: int,
    cb_size: int,
) -> mx.array
```

Fused CommVQ centroid gather + RoPE decode in a single Metal pass.

| Parameter | Type | Description |
|---|---|---|
| `indices` | `mx.array` | `[N, n_cb]` uint8 sub-codebook indices |
| `codebook` | `mx.array` | `[n_cb, cb_size, sub_dim]` fp16 centroid table |
| `positions` | `mx.array` | `[N]` int32 token positions for RoPE |
| `inv_freq` | `mx.array` | `[D//2]` fp32 RoPE inverse-frequency table |
| `n_cb` | `int` | Number of sub-codebooks |
| `sub_dim` | `int` | Sub-dimension per codebook (`D // n_cb`) |
| `cb_size` | `int` | Codebook size (`2^b`) |

**Returns:** `[N, D]` fp16 decoded keys with RoPE applied.

---

## Cross-model KV transfer

`veloxquant_mlx.metal._crosskv_rope`

### `crosskv_rope_recode`

```python
def crosskv_rope_recode(
    keys: mx.array,        # [BH, N, D] fp16/fp32 — rotated under source_base, D even
    positions: mx.array,   # [N] absolute positions, shared across BH groups
    source_base: float,    # source model's rope_theta
    target_base: float,    # target model's rope_theta
) -> mx.array
```

Fuses `strip_rope` → `apply_rope` into a single dispatch when transplanting a KV cache between models with different `rope_theta`: the two rotations on the same `(d, d + D/2)` pair compose into one rotation by the per-dimension angle difference. Numerically equivalent to `veloxquant_mlx.transfer.rope.recode_rope`.

- Returns: `[BH, N, D]` keys rotated as though produced by the target model, same dtype as `keys`

### `is_available`

```python
def is_available() -> bool
```

True when a Metal GPU is present to dispatch to.

---

## Scalar quantization kernels

`veloxquant_mlx.metal._scalar_quant`

### `turboquant_scalar_quantize`

```python
def turboquant_scalar_quantize(x: mx.array, centroids: mx.array, b: int) -> mx.array
```

Nearest-centroid Lloyd-Max scalar quantization on GPU.

| Parameter | Type | Description |
|---|---|---|
| `x` | `mx.array` | `[..., d]` float input (any float dtype) |
| `centroids` | `mx.array` | `[2^b]` fp32 Lloyd-Max centroids |
| `b` | `int` | Bits per index, 1-4 |

**Returns:** `[..., d]` uint8 indices.

**Raises:** `ValueError` if `b` is outside `1..4`, or `centroids.size != 2**b`.

### `turboquant_scalar_dequantize`

```python
def turboquant_scalar_dequantize(indices: mx.array, centroids: mx.array) -> mx.array
```

Decodes b-bit indices to fp16 via a centroid gather. `centroids` is `[2^b]` fp32; `indices` is `[..., d]` uint8. Returns `[..., d]` fp16 reconstructed values.

### `turboquant_hadamard_quantize`

```python
def turboquant_hadamard_quantize(
    x: mx.array,
    diag: mx.array,
    centroids: mx.array,
    b: int,
) -> mx.array
```

Fused randomized-Hadamard preconditioner + scalar quantize in one Metal dispatch: computes `y = diag * H * x / sqrt(D)` and nearest-centroid quantizes `y`, no intermediate allocation.

| Parameter | Type | Description |
|---|---|---|
| `x` | `mx.array` | `[B, D]` fp16 input. `D` must be a power of 2, `<= 1024` |
| `diag` | `mx.array` | `[D]` float ±1 diagonal signs |
| `centroids` | `mx.array` | `[2^b]` fp32 Lloyd-Max centroids |
| `b` | `int` | Bits per index, 1-4 |

**Returns:** `[B, D]` uint8 indices (not a `(indices, scale_factors)` tuple — the scale is folded into `centroids`).

---

## RVQ + Attention fusion

`veloxquant_mlx.metal._rvq_attend`

### `turboquant_fused_rvq_decode_attend`

```python
def turboquant_fused_rvq_decode_attend(
    q: mx.array,
    k_indices1: mx.array,
    k_indices2: mx.array,
    centroids1: mx.array,
    centroids2: mx.array,
    v_indices: mx.array,
    v_codebook: mx.array,
    b1: int,
    b2: int,
    bv: int,
) -> mx.array
```

Fused two-stage RVQ key decode + scaled-dot-product attention: decodes keys on the fly from two-stage RVQ indices inside an online-softmax loop — no intermediate `K_hat` tensor is materialized. Does **not** take an `EncodedVector` directly — indices and centroids are passed as separate raw arrays.

| Parameter | Type | Description |
|---|---|---|
| `q` | `mx.array` | `[B, H, S_q, D]` fp16 queries (pre-rotated) |
| `k_indices1` | `mx.array` | `[B, H, S_kv, D]` uint8 first-stage key indices |
| `k_indices2` | `mx.array` | `[B, H, S_kv, D]` uint8 second-stage (residual) key indices |
| `centroids1` | `mx.array` | `[2^b1]` fp32 Gaussian centroids (stage 1) |
| `centroids2` | `mx.array` | `[2^b2]` fp32 Laplacian centroids (stage 2) |
| `v_indices` | `mx.array` | `[B, H, S_kv, D // sub_dim_v]` uint8 value indices |
| `v_codebook` | `mx.array` | `[2^bv, sub_dim_v]` fp16 value codebook |
| `b1`, `b2`, `bv` | `int` | Bit-widths for key stage 1, key stage 2, and values |

**Returns:** `[B, H, S_q, D]` fp16 attention output.

---

## Fused RVQ quantize + pack

`veloxquant_mlx.metal._rvq_quant_pack`

### `rvq_quant_pack`

```python
def rvq_quant_pack(
    rotated: mx.array,      # [N, D] fp16/fp32 — post-rotation vectors, D power of two <= 1024
    centroids1: mx.array,   # [2**bits] stage-1 sorted centroids
    boundaries1: mx.array,  # [2**bits - 1] stage-1 Voronoi boundaries
    boundaries2: mx.array,  # [2**bits - 1] stage-2 (residual) Voronoi boundaries
    bits: int,               # 1-4
) -> tuple[mx.array, mx.array]
```

Fuses stage-1 quantize, stage-2 (residual) quantize, and both bit-packs into one dispatch — bit-identical to `ScalarCodebook.quantize` + `_pack_indices` run twice.

- Returns: `(packed1, packed2)`, each `[N, ceil(D / (32 // bits))]` uint32

---

## Prefill attention kernels

`veloxquant_mlx.metal._flash_prefill` / `veloxquant_mlx.metal._experimental_streaming_prefill`

### `flash_prefill_attend`

```python
def flash_prefill_attend(
    q: mx.array,      # [B, H, S_q, D]  fp16 — queries
    k: mx.array,      # [B, H, S_kv, D] fp16 — plain (uncompressed) keys
    v: mx.array,      # [B, H, S_kv, D] fp16 — plain (uncompressed) values
    scale: mx.array,  # [1] fp32 — 1/sqrt(D)
) -> mx.array
```

Causal flash attention over plain fp16 K/V for from-scratch prefill (no existing compressed cache) — `simdgroup_matrix` tiles, `exp2` softmax with pre-folded scale, and a causal block-skip that drops fully-future KV chunks before loading. Always causal (`q_abs = (S_kv - S_q) + q_pos`). Requires `D % 8 == 0`, `D <= 128`.

- Returns: `[B, H, S_q, D]` fp16 attention output

### `streaming_prefill_attend`

```python
def streaming_prefill_attend(
    q: mx.array, k: mx.array, v: mx.array,  # same shapes/dtypes as flash_prefill_attend
    scale: mx.array,
    implementation: str = "streaming",
    # one of: "streaming", "streaming_block2", "streaming_block4",
    #         "streaming_block8", "streaming_multirow"
) -> mx.array
```

Experimental row-owned alternative to `flash_prefill_attend`: one SIMD-group owns one query row for the whole kernel, K/V stream from device memory with no threadgroup memory and no barriers. Built to benchmark against the tiled approach, not to replace it — `flash_prefill_attend` remains the production kernel. Requires `D % 32 == 0`, `D <= 128`.

- Returns: `[B, H, S_q, D]` fp16 attention output

---

## KV-cache eviction kernels

`veloxquant_mlx.metal._h2o_evict` / `_keyformer_evict` / `_qfilters_evict`

Callers must only invoke these when every `(batch, head)` group is already over budget — the below-budget case is handled by the existing vectorized MLX path.

### `h2o_fused_evict`

```python
def h2o_fused_evict(
    keys_mid: mx.array,      # [BH, n_total, D] fp16
    values_mid: mx.array,    # [BH, n_total, D] fp16
    scores_mid: mx.array,    # [BH, n_total] fp32 — appended row's score is 0.0
    positions_mid: mx.array, # [BH, n_total] int32
    n_sink: int,
    rope_base: float,
    grace: int = 0,
    nsg: int = 4,
) -> tuple[mx.array, mx.array, mx.array, mx.array]
```

Two dispatches: a sink/grace-protected argmin reduction, then a compaction that re-rotates (NeoX-style RoPE) exactly the rows whose position shifted. Matches `h2o_update`'s eviction branch bit-for-bit.

- Returns: `(keys_out, values_out, scores_out, positions_out)`, each with `n_total - 1` rows

### `keyformer_fused_evict`

```python
def keyformer_fused_evict(
    keys_mid: mx.array, values_mid: mx.array, scores_mid: mx.array,
    gumbel_mid: mx.array,     # [BH, n_total] fp32 — frozen per-position Gumbel noise
    positions_mid: mx.array,
    n_sink: int,
    rope_base: float,
    tau: float = 0.0,   # 0.0 collapses to h2o_fused_evict's raw-score argmin
    recent: int = 0,
    nsg: int = 4,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]
```

Structurally `h2o_fused_evict` with the selection value replaced by `score + tau * gumbel`.

- Returns: `(keys_out, values_out, scores_out, gumbel_out, positions_out)`, each with `n_total - 1` rows

### `qfilters_score`

```python
def qfilters_score(
    keys: mx.array,        # [BH, n_total, D] fp16
    filter_dir: mx.array,  # [BH, D] fp32 — per-group unit-norm Q-Filter
    n_sink: int = 0,
    recent: int = 0,
    sign: int = 1,          # +1 keeps highest projections, -1 inverts
) -> mx.array
```

Projection scores `sign * <k_i, filter_dir>` (paper Theorem 3.3); sink/recent rows are forced to `+inf` so they always survive downstream selection.

- Returns: `[BH, n_total]` fp32 scores

### `qfilters_fused_evict`

```python
def qfilters_fused_evict(
    keys_mid: mx.array, values_mid: mx.array,
    filter_dir: mx.array,
    budget: int,             # <= QFILTERS_MAX_BUDGET (4096)
    n_sink: int = 0,
    recent: int = 0,
    sign: int = 1,
) -> tuple[mx.array, mx.array, mx.array]
```

Scores every row via `qfilters_score`, picks the keep-threshold with `mx.sort` on the MLX side, then compacts the surviving `budget` rows in temporal order. No RoPE remap (documented limitation) — keys are copied bit-identically.

- Returns: `(keys_out, values_out, scores_out)`, each `[BH, budget, ...]`

---

## Fused SDPA

`veloxquant_mlx.metal.fused_sdpa`

### `metal_fused_sdpa`

```python
from veloxquant_mlx.metal.fused_sdpa import metal_fused_sdpa

def metal_fused_sdpa(
    q_tilde: mx.array,       # [B, H_q, S_q, D]   fp32 — already smooth+Hadamard transformed
    k_indices: mx.array,     # [B, H_kv, S_kv, n_sub] codebook indices for keys (transformed space)
    k_codebook: mx.array,    # [n_centroids, sub_dim]
    v_indices: mx.array,     # [B, H_kv, S_kv, n_sub_v] value indices
    v_codebook: mx.array,    # [n_centroids_v, sub_dim_v]
    scale: float,
    *,
    causal: bool = True,
    sliding_window: int = 0,
    out_dtype: mx.Dtype | None = None,
) -> mx.array
```

Fused SDPA specifically for VecInfer's compressed K/V representation (codebook indices + codebook, not a generic `EncodedVector` and not "all VeloxQuant-MLX key formats"). `q_tilde` must already be transformed via `apply_dual_transform_queries` so that `q_tilde @ K_tilde.T == q @ K_hat.T`; the result is mathematically identical to running standard SDPA on the dequantized fp16 `K_hat`, without ever materializing it.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `q_tilde` | `mx.array` | Required | `[B, H_q, S_q, D]`, already transformed; cast to fp32 internally |
| `k_indices` | `mx.array` | Required | `[B, H_kv, S_kv, n_sub]` key codebook indices |
| `k_codebook` | `mx.array` | Required | `[n_centroids, sub_dim]` key centroid table |
| `v_indices` | `mx.array` | Required | `[B, H_kv, S_kv, n_sub_v]` value indices |
| `v_codebook` | `mx.array` | Required | `[n_centroids_v, sub_dim_v]` value centroid table |
| `scale` | `float` | Required | Attention scale, usually `1/sqrt(head_dim)` |
| `causal` | `bool` | `True` | Apply causal mask (queries align to the tail of `S_kv`) |
| `sliding_window` | `int` | `0` | If `> 0`, only attend to the last `sliding_window` keys before each query position |
| `out_dtype` | `mx.Dtype \| None` | `None` | Output dtype; defaults to `q_tilde.dtype` |

**Returns:** `[B, H_q, S_q, D]` attention output.

**Raises:** `ValueError` if `n_sub*sub_dim != D`, key/value codebooks disagree on `n_centroids`, or `H_q` is not a multiple of `H_kv`; also if `n_centroids`/`n_sub`/`D` exceed the kernel's compile-time caps (see `supports_shape`).

### `supports_shape`

```python
def supports_shape(n_centroids: int, n_sub: int, head_dim: int) -> bool
```

Quick check for whether a `(n_centroids, n_sub, head_dim)` configuration fits the fused kernel's compile-time caps (`MAX_N_CENTROIDS`, `MAX_N_SUB`, `MAX_HEAD_DIM` in `fused_sdpa.py`) — **not** a generic `(batch, heads, seq_len, head_dim)` shape check. `head_dim` must also be a multiple of 32 (checked separately inside `metal_fused_sdpa`/`_get_kernel`, not by this function).

### `patch_mlx_lm_for_fused_sdpa`

```python
from veloxquant_mlx.metal.fused_sdpa import patch_mlx_lm_for_fused_sdpa

def patch_mlx_lm_for_fused_sdpa() -> None
```

Monkey-patches `mlx_lm.models.base.scaled_dot_product_attention`, plus the same-named reference already bound into every currently-imported `mlx_lm.models.*` / `mlx_vlm.models.*` submodule (each does `from .base import scaled_dot_product_attention`, a value-import that a later reassignment of `base`'s own attribute would not otherwise reach). The patched function only routes to a cache's own `fused_sdpa(...)` method when that cache is running in the memory-bound VecInfer configuration (`fused_sdpa=True` and `fused_sdpa_memory_bound=True` in `KVCacheConfig`) with a plain causal-or-none mask and no attention sinks; otherwise it falls through to the original implementation. Takes **no `model` argument** — call it *after* `mlx_lm.load(...)` so the target model's modules are already in `sys.modules` to patch; calling it before load silently fails to intercept that model's attention calls. Idempotent (safe to call multiple times); `unpatch_mlx_lm()` reverses it, and `is_patched()` reports current state.

---

## Bit packing

`veloxquant_mlx.metal._bit_packing`

### `turboquant_bit_pack`

```python
def turboquant_bit_pack(indices: mx.array, b: int) -> mx.array
```

Packs uint8 indices into tightly bit-packed uint8 storage (not uint32 words).

| Parameter | Type | Description |
|---|---|---|
| `indices` | `mx.array` | `[N]` uint8 with values in `[0, 2^b)`. `N` must be divisible by `8 // b` |
| `b` | `int` | Bits per index. Must be 1, 2, or 4 |

**Returns:** `[N * b // 8]` uint8 packed buffer.

**Raises:** `ValueError` if `b` is not 1, 2, or 4, or `N` is not divisible by `8 // b`.

### `turboquant_bit_unpack`

```python
def turboquant_bit_unpack(packed: mx.array, N: int, b: int) -> mx.array
```

Unpacks bit-packed uint8 storage back into uint8 indices (not uint32 → int32).

| Parameter | Type | Description |
|---|---|---|
| `packed` | `mx.array` | `[N * b // 8]` uint8 packed buffer |
| `N` | `int` | Number of original indices to recover |
| `b` | `int` | Bits per index. Must be 1, 2, or 4 |

**Returns:** `[N]` uint8 indices.

---

## QJL kernels

`veloxquant_mlx.metal._qjl`

### `qjl_encode`

```python
def qjl_encode(x: mx.array, S: mx.array) -> tuple[mx.array, mx.array]
```

Computes 1-bit QJL encoding: `sign(S @ x)` bit-packed, plus `‖x‖`.

| Parameter | Type | Description |
|---|---|---|
| `x` | `mx.array` | `[B, d]` fp16 input vectors |
| `S` | `mx.array` | `[m, d]` fp16 JL projection matrix; `m` must be divisible by 8 |

**Returns:** `(packed_signs, norms)` — `packed_signs` is `[B, m//8]` uint8 (LSB-first bit order); `norms` is `[B]` fp16 Euclidean norms. Not a single packed-bits array — always a 2-tuple including the norms.

**Raises:** `ValueError` if `x`/`S` are not 2D, dimensions disagree, or `m % 8 != 0`.

### `qjl_inner_product`

```python
def qjl_inner_product(
    q_proj: mx.array,
    packed_signs: mx.array,
    norms: mx.array,
) -> mx.array
```

Computes unbiased QJL attention scores: `√(π/2)/m · norms[s,h] · ⟨q_proj[h,:], (2·signs−1)⟩` for every (head, kv-slot) pair.

| Parameter | Type | Description |
|---|---|---|
| `q_proj` | `mx.array` | `[H, m]` fp16 pre-projected queries (`S @ q`) |
| `packed_signs` | `mx.array` | `[S_kv, H, m//8]` uint8 bit-packed key signs from `qjl_encode` |
| `norms` | `mx.array` | `[S_kv, H]` fp16 key norms |

**Returns:** `[H, S_kv]` fp16 attention scores. (Not parameterized by bare `head_dim`/`sketch_dim` ints — dimensions are read from the array shapes.)

---

## See also

- [Metal kernels guide](../guides/metal-kernels)
- [VecInfer algorithm](../algorithms/vecinfer)
- [TurboQuant RVQ algorithm](../algorithms/rvq)
