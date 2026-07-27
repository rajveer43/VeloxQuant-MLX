---
id: metal-api
title: Metal Kernels API
sidebar_label: Metal Kernels
slug: /api/metal-api
---

# Metal Kernels API

`veloxquant_mlx.metal`

All Metal kernels are compiled lazily on first call via `mx.fast.metal_kernel`. These are low-level functions — most users should interact with them indirectly through quantizer and cache classes.

:::warning[Apple Silicon only]
All functions in this module require macOS on an M-series chip. On unsupported hardware they raise `MetalUnavailableError`.
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
    keys: mx.array,
    codebook: mx.array,
    smooth_factors: mx.array,
    num_subspaces: int,
) -> mx.array
```

Product VQ encoding on GPU. Returns integer indices of shape `[batch, heads, seq, num_subspaces]`. **13× faster** than equivalent Python ops.

---

### `vecinfer_dequant_metal`

```python
def vecinfer_dequant_metal(
    indices: mx.array,
    codebook: mx.array,
    smooth_factors: mx.array,
) -> mx.array
```

Codebook gather + smooth-factor inverse. Returns reconstructed keys of shape `[batch, heads, seq, head_dim]`.

---

### `vecinfer_encode_decode_metal`

```python
def vecinfer_encode_decode_metal(
    keys: mx.array,
    codebook: mx.array,
    smooth_factors: mx.array,
    num_subspaces: int,
) -> tuple[mx.array, mx.array]
```

Fused encode then decode in one kernel dispatch. Returns `(indices, reconstructed_keys)`.

---

### `compute_query_lut`

```python
from veloxquant_mlx.allocators.vecinfer import compute_query_lut

def compute_query_lut(
    queries: mx.array,
    codebook: mx.array,
    smooth_factors: mx.array,
) -> mx.array
```

Precomputes a query-codebook distance look-up table for asymmetric MIPS (Maximum Inner Product Search). Returns `[batch, heads, num_subspaces, num_centroids]`.

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
) -> mx.array
```

Prefill-shaped companion to `rabitq_fused_attend`, for large `S_q` (multi-turn VLM: a new turn attending over compressed image-token history). Both `Q·K̂ᵀ` and `W·V̂` run on 8×8 `simdgroup_matrix` tiles; K is sign-decoded and V nibble-decoded inside the tile loop, so no dequantized K/V is materialized.

Scores are exact dots — `(q · signs·k_mag) * scale + k_const` — not the Hamming estimate the decode kernel uses. **Cross-attention only:** every query row attends over all `S_kv` slots with no causal mask. Values must be nibble-packed (`rabitq_pack_values` format).

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
    nsg: int = 4,
) -> mx.array
```

Single-dispatch SDP attention directly over an asymmetric group-min/max ("affine") quantized cache — the KIVI / SKVQ / Kitty / group-quant family. Reconstructs `k_hat = k_codes*k_scale + k_zero` (per-channel groups) and `v_hat = v_codes*v_scale + v_zero` (per-token groups) in-register inside a FlashAttention-style online softmax; no fp16 `K_hat`/`V_hat` is written to DRAM.

The kv axis is split across `nsg` SIMD-groups flash-decoding style so single-query decode shapes still fill the GPU (`nsg=8` is tuned on M4). One compiled kernel serves any `(S_kv, D, g)`.

Constraints: `q` must be 4-D, `D ≤ 256`, `1 ≤ nsg ≤ 32`.

Measured on Apple M4 (B=1, H=32, D=128, b=2, g=32, S_q=1) vs. dequantize → MLX SDPA: **6.4× at S_kv=512, rising to 12.2× at S_kv=65536**. Softmax accumulates in fp32, so parity error (`1.2e-4` max abs) is better than the fp16 baseline.

- Returns: `[B, H, S_q, D]` fp16 attention output

---

## CommVQ kernels

`veloxquant_mlx.metal._comm_vq`

### `comm_vq_decode_metal`

```python
def comm_vq_decode_metal(
    indices: mx.array,
    codebook: mx.array,
    cos_freqs: mx.array,
    sin_freqs: mx.array,
    positions: mx.array,
) -> mx.array
```

Fused centroid gather + RoPE application in a single Metal pass. Returns decoded+position-embedded keys.

---

## Scalar quantization kernels

`veloxquant_mlx.metal._scalar_quant`

### `turboquant_scalar_quantize`

```python
def turboquant_scalar_quantize(x: mx.array, bits: int) -> mx.array
```

Lloyd-Max scalar quantization on GPU.

### `turboquant_scalar_dequantize`

```python
def turboquant_scalar_dequantize(indices: mx.array, bits: int, scale: float) -> mx.array
```

### `turboquant_hadamard_quantize`

```python
def turboquant_hadamard_quantize(x: mx.array, bits: int) -> tuple[mx.array, mx.array]
```

Fused WHT rotation + scalar quantization in one pass. Returns `(indices, scale_factors)`.

---

## RVQ + Attention fusion

`veloxquant_mlx.metal._rvq_attend`

### `turboquant_fused_rvq_decode_attend`

```python
def turboquant_fused_rvq_decode_attend(
    queries: mx.array,
    encoded_keys: EncodedVector,
    values: mx.array,
    scale: float,
) -> mx.array
```

Two-stage RVQ decode + scaled dot-product attention in a single kernel. Most efficient path for TurboQuant RVQ inference.

---

## Fused SDPA

`veloxquant_mlx.metal.fused_sdpa`

### `metal_fused_sdpa`

```python
from veloxquant_mlx.metal.fused_sdpa import metal_fused_sdpa

def metal_fused_sdpa(
    queries: mx.array,
    encoded_keys: EncodedVector,
    values: mx.array,
    scale: float,
    mask: mx.array | None = None,
) -> mx.array
```

Fused dequantize + scaled dot-product attention. Supports all VeloxQuant-MLX key formats.

### `supports_shape`

```python
def supports_shape(batch: int, heads: int, seq_len: int, head_dim: int) -> bool
```

Returns `True` if the fused kernel supports this attention shape. Requires `head_dim` to be a multiple of 32.

### `patch_mlx_lm_for_fused_sdpa`

```python
from veloxquant_mlx.metal.fused_sdpa import patch_mlx_lm_for_fused_sdpa

def patch_mlx_lm_for_fused_sdpa(model) -> None
```

Monkey-patches each attention layer to use `metal_fused_sdpa` instead of standard `mx.matmul`. Call once after model load.

---

## Bit packing

`veloxquant_mlx.metal._bit_packing`

### `turboquant_bit_pack`

```python
def turboquant_bit_pack(indices: mx.array, bits: int) -> mx.array
```

Packs `bits`-bit indices into uint32 words. Input shape `[..., N]`, output shape `[..., ceil(N*bits/32)]`.

### `turboquant_bit_unpack`

```python
def turboquant_bit_unpack(
    packed: mx.array,
    bits: int,
    original_length: int,
) -> mx.array
```

Unpacks uint32 words back to int32 indices.

---

## QJL kernels

`veloxquant_mlx.metal._qjl`

### `qjl_encode`

```python
def qjl_encode(keys: mx.array, projection: mx.array) -> mx.array
```

Project + sign in one Metal pass. Returns packed uint32 bit strings.

### `qjl_inner_product`

```python
def qjl_inner_product(
    query_bits: mx.array,
    key_bits: mx.array,
    head_dim: int,
    sketch_dim: int,
) -> mx.array
```

Approximates `⟨q, k⟩` via bit string inner product.

---

## See also

- [Metal kernels guide](../guides/metal-kernels)
- [VecInfer algorithm](../algorithms/vecinfer)
- [TurboQuant RVQ algorithm](../algorithms/rvq)
