---
id: metal-kernels
title: Metal GPU Kernels
sidebar_label: Metal Kernels
slug: /guides/metal-kernels
---

# Metal GPU Kernels

VeloxQuant-MLX compiles eleven Metal kernel modules at runtime using `mx.fast.metal_kernel`. This guide explains what each kernel does, how they are loaded, performance characteristics, and fallback behaviour.

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
| `metal/_comm_vq.py` | `comm_vq_decode_metal` | CommVQ RoPE |
| `metal/_scalar_quant.py` | `turboquant_scalar_quantize`, `turboquant_scalar_dequantize`, `turboquant_hadamard_quantize` | TurboQuant RVQ |
| `metal/_rvq_attend.py` | `turboquant_fused_rvq_decode_attend` | RVQ + attention fusion |
| `metal/_qjl.py` | `qjl_encode`, `qjl_inner_product` | QJL |
| `metal/_bit_packing.py` | `turboquant_bit_pack`, `turboquant_bit_unpack` | All algorithms |
| `metal/fused_sdpa.py` | `metal_fused_sdpa` | All (fused attention) |

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
    nsg=8,  # SIMD-groups splitting the kv axis; 8 is tuned for M4
)  # -> [B, H, S_q, D] fp16
```

One compiled kernel serves any `(S_kv, D, g)` — the group counts are read from the passed shapes. `D` must be ≤ 256 and `nsg` in `1..32`.

Measured on Apple M4 (10-core GPU), `B=1 H=32 D=128 b=2 g=32 S_q=1`, versus dequantize → MLX SDPA:

| `S_kv` | Speedup |
|---|---|
| 512 | 6.4× |
| 65536 | 12.2× |

Parity max abs error is `1.2e-4` against the reference — the kernel accumulates its softmax in fp32, so it is *more* accurate than the fp16 baseline it replaces. See `veloxquant_mlx/tests/metal/test_scalar_attend.py`.

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
