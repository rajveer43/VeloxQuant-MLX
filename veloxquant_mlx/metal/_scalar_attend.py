"""Universal fused group-affine decode + attention Metal kernel.

Single-dispatch scaled-dot-product attention directly over an
asymmetric group-min/max ("affine") quantized KV cache — the KIVI /
SKVQ / Kitty / group-quant family. Keys and values are stored as
uint8 codes plus per-group ``(scale, zero)``; the kernel reconstructs
each element in-register as ``x_hat = code * scale + zero`` and runs a
FlashAttention-style online softmax. No dequantized ``K_hat`` / ``V_hat``
tensor is ever materialized in DRAM.

This is the scalar/group-quant analogue of the codebook fused attends
(:mod:`_rvq_attend`, :mod:`fused_sdpa`, :mod:`_rabitq_attend`). It kills
the ``dequantize -> DRAM -> SDPA`` round-trip that the pure-MLX path
(reconstruct ``code*scale+zero`` into a full fp16 tensor, then call
``scaled_dot_product_attention``) pays every decode step. At long
context that round-trip dominates: the fp16 K_hat grows linearly with
S_kv while the packed codes are ``16/b`` times smaller.

Quantization layout (matches KIVI's ``_quant_dequant_along``)
------------------------------------------------------------
Keys — per-CHANNEL groups (group along the token axis):
  * ``k_codes [B, H, S_kv, D]``  uint8 codes in ``[0, 2^b - 1]``
  * ``k_scale [B, H, GK,   D]``  fp32, GK = ceil(S_kv / g)
  * ``k_zero  [B, H, GK,   D]``  fp32
  reconstruction: ``k_hat[.., sk, d] = k_codes[..,sk,d] * k_scale[..,sk/g,d] + k_zero[..,sk/g,d]``

Values — per-TOKEN groups (group along the channel axis):
  * ``v_codes [B, H, S_kv, D]``  uint8
  * ``v_scale [B, H, S_kv, GV]`` fp32, GV = ceil(D / g)
  * ``v_zero  [B, H, S_kv, GV]`` fp32
  reconstruction: ``v_hat[.., sk, d] = v_codes[..,sk,d] * v_scale[..,sk,d/g] + v_zero[..,sk,d/g]``

Score model (per kv slot sk, query (b, h, sq)):

    score_sk = (sum_d q[..,sq,d] * k_hat[..,sk,d]) * scale

``scale`` (typically ``1/sqrt(D)``) is passed in and applied by the
kernel; no other scaling is folded in.

Public API:
  - :func:`scalar_fused_decode_attend`
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return (Path(__file__).parent / "src" / filename).read_text()


_cache: dict = {}


# ===========================================================================
# Metal source — fused affine decode + flash-decoding attend
# ===========================================================================
# Grid:        (B * H * S_q * 32, NSG_C, 1) — MLX grid = total threads.
# Threadgroup: (32, NSG_C, 1)               — NSG_C SIMD-groups of 32 lanes.
#
# Each threadgroup handles one query position (b, h, sq). A single
# 32-lane pass over S_kv under-fills the GPU for decode shapes
# (B*H*S_q small), so the kv axis is split flash-decoding style:
# SIMD-group sg processes slots sk = sg, sg + NSG_C, ... with its own
# online softmax (running_m, running_d, my_out[]). The NSG_C partial
# results are merged through threadgroup memory at the end (identical
# merge to _rabitq_attend).
#
# Within one SIMD-group the 32 lanes stripe the D-dim vectors in steps
# of 32; simd_sum reduces the partial dot product, so the per-slot loop
# needs no barriers. Only the final cross-SIMD-group merge barriers.
#
# GK (key groups along tokens) and GV (value groups along channels) and
# the group size G are read from the passed shapes / a param, so one
# compiled kernel serves any (S_kv, D, g).

_SCALAR_AFFINE_ATTEND_SRC = _read_kernel_source("scalar_affine_attend.metal")


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------


def _scalar_affine_attend_kernel(D: int, nsg: int):
    key = ("scalar_affine_attend", D, nsg)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"scalar_affine_attend_d{D}_nsg{nsg}",
            input_names=[
                "q",
                "k_codes",
                "k_scale",
                "k_zero",
                "v_codes",
                "v_scale",
                "v_zero",
                "gsize",
                "scale_arr",
            ],
            output_names=["out"],
            header=f"#define NSG_C {nsg}\n",
            source=_SCALAR_AFFINE_ATTEND_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scalar_fused_decode_attend(
    q: mx.array,
    k_codes: mx.array,
    k_scale: mx.array,
    k_zero: mx.array,
    v_codes: mx.array,
    v_scale: mx.array,
    v_zero: mx.array,
    group_size: int,
    scale: float,
    nsg: int = 4,
) -> mx.array:
    """Fused group-affine (KIVI-style) key/value decode + SDP attention.

    Reconstructs ``k_hat = k_codes*k_scale + k_zero`` (per-channel groups)
    and ``v_hat = v_codes*v_scale + v_zero`` (per-token groups) on-the-fly
    inside an online-softmax loop — no intermediate fp16 K_hat/V_hat.

    Args:
        q:        ``[B, H, S_q, D]`` fp16/fp32 queries (pre-rotated).
        k_codes:  ``[B, H, S_kv, D]`` uint8 key codes.
        k_scale:  ``[B, H, GK, D]``   fp32, GK = ceil(S_kv/group_size).
        k_zero:   ``[B, H, GK, D]``   fp32.
        v_codes:  ``[B, H, S_kv, D]`` uint8 value codes.
        v_scale:  ``[B, H, S_kv, GV]`` fp32, GV = ceil(D/group_size).
        v_zero:   ``[B, H, S_kv, GV]`` fp32.
        group_size: quantization group size g.
        scale:    attention scale applied to the raw dot (e.g. 1/sqrt(D)).
        nsg:      SIMD-groups per threadgroup splitting the kv axis.

    Returns:
        ``[B, H, S_q, D]`` fp16 attention output.
    """
    if q.ndim != 4:
        raise ValueError(f"scalar_fused_decode_attend: q must be 4D, got {q.shape}")
    B, H, S_q, D = q.shape
    if D > 256:
        raise ValueError(f"scalar_fused_decode_attend: D={D} must be <= 256")
    if not (1 <= nsg <= 32):
        raise ValueError(f"scalar_fused_decode_attend: nsg={nsg} must be in 1..32")

    n_tg = B * H * S_q
    gsize = mx.array([group_size], dtype=mx.uint32)
    scale_arr = mx.array([scale], dtype=mx.float32)

    outputs = _scalar_affine_attend_kernel(D, nsg)(
        inputs=[
            q.astype(mx.float16),
            k_codes.astype(mx.uint8),
            k_scale.astype(mx.float32),
            k_zero.astype(mx.float32),
            v_codes.astype(mx.uint8),
            v_scale.astype(mx.float32),
            v_zero.astype(mx.float32),
            gsize,
            scale_arr,
        ],
        # MLX grid = total threads; NSG_C SIMD-groups of 32 lanes each
        grid=(n_tg * 32, nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, H, S_q, D)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]


__all__ = ["scalar_fused_decode_attend"]
