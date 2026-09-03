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
  * ``k_codes [B, H_kv, S_kv, D]``  uint8 codes in ``[0, 2^b - 1]``
  * ``k_scale [B, H_kv, GK,   D]``  fp32, GK = ceil(S_kv / g)
  * ``k_zero  [B, H_kv, GK,   D]``  fp32
  reconstruction: ``k_hat[.., sk, d] = k_codes[..,sk,d] * k_scale[..,sk/g,d] + k_zero[..,sk/g,d]``

Values — per-TOKEN groups (group along the channel axis):
  * ``v_codes [B, H_kv, S_kv, D]``  uint8
  * ``v_scale [B, H_kv, S_kv, GV]`` fp32, GV = ceil(D / g)
  * ``v_zero  [B, H_kv, S_kv, GV]`` fp32
  reconstruction: ``v_hat[.., sk, d] = v_codes[..,sk,d] * v_scale[..,sk,d/g] + v_zero[..,sk,d/g]``

Score model (per kv slot sk, query (b, h, sq)):

    score_sk = (sum_d q[..,sq,d] * k_hat[..,sk,d]) * scale

``scale`` (typically ``1/sqrt(D)``) is passed in and applied by the
kernel; no other scaling is folded in.

GQA head-packing
-----------------
``q`` carries ``H_q`` query heads while ``k_codes``/``v_codes`` (and their
scale/zero arrays) carry ``H_kv`` key/value heads, with ``H_q`` required to
be an exact multiple of ``H_kv``. ``H_kv`` is inferred from
``k_codes.shape[1]`` — no separate parameter is needed. When ``H_q >
H_kv``, the ``heads_per_kv = H_q // H_kv`` query heads that share one kv
head (contiguous blocks: query heads ``[h_kv*heads_per_kv,
(h_kv+1)*heads_per_kv)`` map to kv head ``h_kv``, i.e. ``repeat_interleave``
order) are packed into one threadgroup, so each K/V code is decoded once
and reused across all heads that share it, instead of being redundantly
redecoded per query head. ``H_q == H_kv`` (plain MHA, ``heads_per_kv=1``)
degenerates to the original per-head dispatch with bit-identical output.

Two-pass decode-once + predecoded-attend (issue #308 spike)
-------------------------------------------------------------
Issue #307's GQA head-packing (above) trades away threadgroup count
(``B*H_q*S_q`` -> ``B*H_kv*S_q``) to save K/V decode work, and was
measured 2.7-4.7x *slower* because occupancy, not bandwidth or decode
compute, is the binding constraint at realistic decode shapes. Issue #308
investigated a SIMD-shuffle-based alternative and found it inapplicable:
``simd_shuffle`` only exchanges data between lanes already co-resident in
one threadgroup, so it cannot share decoded state across query heads that
live in *different*, independently-dispatched threadgroups without first
sacrificing the same threadgroup count as head-packing already does.

This module instead offers a genuinely different two-pass experiment:
:func:`scalar_decode_once` decodes every K/V code exactly once into a
``[B, H_kv, S_kv, D]`` fp16 device buffer (dispatched flat over all
elements -- large and occupancy-friendly by construction, since it scales
with ``S_kv`` rather than ``S_q``), and :func:`scalar_predecoded_attend`
then runs the *original* full-occupancy ``B*H_q*S_q`` dispatch against
that already-decoded buffer -- heads sharing a kv head still redundantly
*re-read* the same decoded fp16 rows from DRAM, but no longer redundantly
*re-decode* them. This trades one extra DRAM round-trip (the decode
pass's own read-then-write) for eliminating the redundant per-head decode
arithmetic, without touching threadgroup count on the attend dispatch.
Whether that round-trip is cheaper than redundant on-the-fly decode is
exactly the open, unverified question this spike measures.

Public API:
  - :func:`scalar_fused_decode_attend`
  - :func:`scalar_decode_once`
  - :func:`scalar_predecoded_attend`
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
# Grid:        (B * H_kv * S_q * 32, NSG_C, 1) — MLX grid = total threads.
# Threadgroup: (32, NSG_C, 1)                  — NSG_C SIMD-groups of 32 lanes.
#
# Each threadgroup handles one (b, h_kv, sq) slot and ALL HEADS_PER_KV_C
# query heads sharing that kv head — K/V codes are decoded once per
# (sk, d) and the dot-product/weighted-accumulate is repeated per packed
# head against the already-decoded value, so K/V DRAM reads don't scale
# with heads_per_kv. A single 32-lane pass over S_kv under-fills the GPU
# for decode shapes (B*H_kv*S_q small), so the kv axis is also split
# flash-decoding style: SIMD-group sg processes slots sk = sg, sg +
# NSG_C, ... with its own per-packed-head online softmax (running_m[],
# running_d[], my_out[][]). The NSG_C partial results are merged through
# threadgroup memory at the end (identical merge to _rabitq_attend, just
# repeated per packed head).
#
# Within one SIMD-group the 32 lanes stripe the D-dim vectors in steps
# of 32; simd_sum reduces the partial dot product, so the per-slot loop
# needs no barriers. Only the final cross-SIMD-group merge barriers.
#
# GK (key groups along tokens) and GV (value groups along channels) and
# the group size G are read from the passed shapes / a param, so one
# compiled kernel serves any (S_kv, D, g). heads_per_kv=1 (H_q == H_kv)
# degenerates every per-head loop to a single trip, compiling down to
# the original non-GQA code path.

_SCALAR_AFFINE_ATTEND_SRC = _read_kernel_source("scalar_affine_attend.metal")
_SCALAR_AFFINE_DECODE_ONCE_SRC = _read_kernel_source("scalar_affine_decode_once.metal")
_SCALAR_PREDECODED_ATTEND_SRC = _read_kernel_source("scalar_predecoded_attend.metal")


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------


def _scalar_affine_attend_kernel(D: int, nsg: int, heads_per_kv: int):
    key = ("scalar_affine_attend", D, nsg, heads_per_kv)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"scalar_affine_attend_d{D}_nsg{nsg}_hpk{heads_per_kv}",
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
            header=f"#define NSG_C {nsg}\n#define HEADS_PER_KV_C {heads_per_kv}\n",
            source=_SCALAR_AFFINE_ATTEND_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


def _scalar_affine_decode_kernel(mode: str):
    assert mode in ("K", "V")
    key = ("scalar_affine_decode_once", mode)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"scalar_affine_decode_once_{mode.lower()}",
            input_names=["codes", "scale", "zero", "gsize"],
            output_names=["out"],
            header=f"#define DECODE_MODE_K {1 if mode == 'K' else 0}\n",
            source=_SCALAR_AFFINE_DECODE_ONCE_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


def _scalar_predecoded_attend_kernel(nsg: int):
    key = ("scalar_predecoded_attend", nsg)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"scalar_predecoded_attend_nsg{nsg}",
            input_names=["q", "k_hat", "v_hat", "scale_arr"],
            output_names=["out"],
            header=f"#define NSG_C {nsg}\n",
            source=_SCALAR_PREDECODED_ATTEND_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# Threadgroup-memory budget (bytes), matching the 32KB ceiling
# _flash_prefill.py's _PDT_HARD_BUDGET already assumes for this GPU family.
_TG_MEM_BUDGET_BYTES = 32768

# Defensive ceiling on heads_per_kv — 2x the largest ratio in any current
# open-weights model (Qwen2's 7x), bounding the compile-time per-lane
# register-array sizes (running_m/running_d/my_out) against pathological inputs.
_MAX_HEADS_PER_KV = 16


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

    Supports grouped-query attention: ``k_codes``/``v_codes`` (and their
    scale/zero arrays) may carry fewer heads than ``q``. ``H_kv`` is
    inferred from ``k_codes.shape[1]`` and must evenly divide ``q``'s head
    count ``H_q``; the ``heads_per_kv = H_q // H_kv`` query heads sharing
    a kv head are packed into one threadgroup so each K/V code is decoded
    once and reused across them. ``H_q == H_kv`` (plain MHA) is unaffected.

    Args:
        q:        ``[B, H_q, S_q, D]`` fp16/fp32 queries (pre-rotated).
        k_codes:  ``[B, H_kv, S_kv, D]`` uint8 key codes.
        k_scale:  ``[B, H_kv, GK, D]``   fp32, GK = ceil(S_kv/group_size).
        k_zero:   ``[B, H_kv, GK, D]``   fp32.
        v_codes:  ``[B, H_kv, S_kv, D]`` uint8 value codes.
        v_scale:  ``[B, H_kv, S_kv, GV]`` fp32, GV = ceil(D/group_size).
        v_zero:   ``[B, H_kv, S_kv, GV]`` fp32.
        group_size: quantization group size g.
        scale:    attention scale applied to the raw dot (e.g. 1/sqrt(D)).
        nsg:      SIMD-groups per threadgroup splitting the kv axis.

    Returns:
        ``[B, H_q, S_q, D]`` fp16 attention output.
    """
    if q.ndim != 4:
        raise ValueError(f"scalar_fused_decode_attend: q must be 4D, got {q.shape}")
    if k_codes.ndim != 4 or v_codes.ndim != 4:
        raise ValueError(
            "scalar_fused_decode_attend: k_codes and v_codes must be 4D, "
            f"got k_codes={k_codes.shape}, v_codes={v_codes.shape}"
        )
    B, H, S_q, D = q.shape
    if D > 256:
        raise ValueError(f"scalar_fused_decode_attend: D={D} must be <= 256")
    if not (1 <= nsg <= 32):
        raise ValueError(f"scalar_fused_decode_attend: nsg={nsg} must be in 1..32")

    Bk, H_kv, _, Dk = k_codes.shape
    if Bk != B:
        raise ValueError(f"scalar_fused_decode_attend: batch mismatch: q B={B} vs k_codes B={Bk}")
    if Dk != D:
        raise ValueError(
            f"scalar_fused_decode_attend: head_dim mismatch: q D={D} vs k_codes D={Dk}"
        )
    if v_codes.shape[1] != H_kv:
        raise ValueError(
            f"scalar_fused_decode_attend: k_codes H_kv={H_kv} vs "
            f"v_codes H_kv={v_codes.shape[1]} mismatch"
        )
    if H % H_kv != 0:
        raise ValueError(
            f"scalar_fused_decode_attend: H_q={H} must be a multiple of "
            f"H_kv={H_kv} (k_codes.shape[1]) for GQA head-packing"
        )
    heads_per_kv = H // H_kv
    if heads_per_kv > _MAX_HEADS_PER_KV:
        raise ValueError(
            f"scalar_fused_decode_attend: heads_per_kv={heads_per_kv} exceeds "
            f"the supported maximum of {_MAX_HEADS_PER_KV}"
        )
    # sh_o[NSG_C*HEADS_PER_KV_C*8*32] + sh_m[NSG_C*HEADS_PER_KV_C] + sh_d[NSG_C*HEADS_PER_KV_C],
    # all float32 — must match scalar_affine_attend.metal's threadgroup array declarations.
    n_slots = nsg * heads_per_kv
    tg_mem_bytes = (n_slots * 8 * 32 + n_slots + n_slots) * 4
    if tg_mem_bytes > _TG_MEM_BUDGET_BYTES:
        raise ValueError(
            f"scalar_fused_decode_attend: nsg={nsg} * heads_per_kv={heads_per_kv} "
            f"exceeds the {_TG_MEM_BUDGET_BYTES}B threadgroup-memory budget for "
            f"the softmax merge buffers ({tg_mem_bytes}B); reduce nsg or use a "
            f"smaller H_q/H_kv ratio"
        )

    n_tg = B * H_kv * S_q
    gsize = mx.array([group_size], dtype=mx.uint32)
    scale_arr = mx.array([scale], dtype=mx.float32)

    outputs = _scalar_affine_attend_kernel(D, nsg, heads_per_kv)(
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


def scalar_decode_once(
    codes: mx.array,
    scale: mx.array,
    zero: mx.array,
    group_size: int,
    mode: str,
) -> mx.array:
    """Decode a group-affine quantized K or V tensor to fp16, once.

    Part of the issue #308 two-pass spike: decodes every code exactly
    once into a full ``[B, H_kv, S, D]`` fp16 buffer, dispatched flat
    over all elements (occupancy scales with ``S``, not ``S_q`` -- large
    and GPU-friendly by construction). Feeds
    :func:`scalar_predecoded_attend`.

    Args:
        codes: ``[B, H_kv, S, D]`` uint8 codes (K or V).
        scale: K -> ``[B, H_kv, GK, D]``; V -> ``[B, H_kv, S, GV]``, fp32.
        zero:  same shape as ``scale``, fp32.
        group_size: quantization group size g.
        mode: ``"K"`` (per-channel groups along tokens) or ``"V"``
            (per-token groups along channels) -- selects the group-index
            arithmetic matching the two layouts documented on
            :func:`scalar_fused_decode_attend`.

    Returns:
        ``[B, H_kv, S, D]`` fp16 decoded tensor.
    """
    if mode not in ("K", "V"):
        raise ValueError(f"scalar_decode_once: mode must be 'K' or 'V', got {mode!r}")
    if codes.ndim != 4:
        raise ValueError(f"scalar_decode_once: codes must be 4D, got {codes.shape}")

    B, H_kv, S, D = codes.shape
    N = B * H_kv * S * D
    gsize = mx.array([group_size], dtype=mx.uint32)

    outputs = _scalar_affine_decode_kernel(mode)(
        inputs=[
            codes.astype(mx.uint8),
            scale.astype(mx.float32),
            zero.astype(mx.float32),
            gsize,
        ],
        grid=(N, 1, 1),
        threadgroup=(min(256, N), 1, 1),
        output_shapes=[(B, H_kv, S, D)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]


def scalar_predecoded_attend(
    q: mx.array,
    k_hat: mx.array,
    v_hat: mx.array,
    scale: float,
    nsg: int = 4,
) -> mx.array:
    """Flash-decoding SDPA over already-decoded fp16 k_hat/v_hat.

    Part of the issue #308 two-pass spike: pairs with
    :func:`scalar_decode_once`. Dispatches the full
    ``B*H_q*S_q`` threadgroups (one per query head), matching the
    high-occupancy baseline from issue #307's addendum -- heads sharing a
    kv head redundantly re-read the same decoded fp16 rows, but no
    on-the-fly dequantization happens in this kernel.

    Args:
        q:     ``[B, H_q, S_q, D]`` fp16/fp32 queries (pre-rotated).
        k_hat: ``[B, H_kv, S_kv, D]`` fp16 decoded keys (from
            ``scalar_decode_once(..., mode="K")``).
        v_hat: ``[B, H_kv, S_kv, D]`` fp16 decoded values (from
            ``scalar_decode_once(..., mode="V")``).
        scale: attention scale applied to the raw dot (e.g. 1/sqrt(D)).
        nsg:   SIMD-groups per threadgroup splitting the kv axis.

    Returns:
        ``[B, H_q, S_q, D]`` fp16 attention output.
    """
    if q.ndim != 4:
        raise ValueError(f"scalar_predecoded_attend: q must be 4D, got {q.shape}")
    if k_hat.ndim != 4 or v_hat.ndim != 4:
        raise ValueError(
            "scalar_predecoded_attend: k_hat and v_hat must be 4D, "
            f"got k_hat={k_hat.shape}, v_hat={v_hat.shape}"
        )
    B, H, S_q, D = q.shape
    if D > 256:
        raise ValueError(f"scalar_predecoded_attend: D={D} must be <= 256")
    if not (1 <= nsg <= 32):
        raise ValueError(f"scalar_predecoded_attend: nsg={nsg} must be in 1..32")

    Bk, H_kv, _, Dk = k_hat.shape
    if Bk != B:
        raise ValueError(f"scalar_predecoded_attend: batch mismatch: q B={B} vs k_hat B={Bk}")
    if Dk != D:
        raise ValueError(f"scalar_predecoded_attend: head_dim mismatch: q D={D} vs k_hat D={Dk}")
    if v_hat.shape != k_hat.shape:
        raise ValueError(
            f"scalar_predecoded_attend: k_hat shape={k_hat.shape} vs "
            f"v_hat shape={v_hat.shape} mismatch"
        )
    if H % H_kv != 0:
        raise ValueError(
            f"scalar_predecoded_attend: H_q={H} must be a multiple of H_kv={H_kv} (k_hat.shape[1])"
        )

    n_tg = B * H * S_q
    scale_arr = mx.array([scale], dtype=mx.float32)

    outputs = _scalar_predecoded_attend_kernel(nsg)(
        inputs=[
            q.astype(mx.float16),
            k_hat.astype(mx.float16),
            v_hat.astype(mx.float16),
            scale_arr,
        ],
        grid=(n_tg * 32, nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(B, H, S_q, D)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]


__all__ = [
    "scalar_fused_decode_attend",
    "scalar_decode_once",
    "scalar_predecoded_attend",
]
