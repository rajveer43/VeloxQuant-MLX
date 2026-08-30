"""Tiled prefill attention over the asymmetric RaBitQ cache — simdgroup_matrix.

The prefill-shaped companion to :func:`rabitq_fused_attend`. Targets any
S_q new-turn tokens attending over S_kv compressed history slots — the
multi-turn VLM workload (cross-attention, image-token history) as well
as standard autoregressive self-attention prefill (``causal=True``) — a
matmul-shaped problem where the decode kernel's one-query-per-threadgroup
scalar dots waste the hardware matrix pipeline. Both matmuls (Q·K̂ᵀ and
W·V̂) run on 8×8 ``simdgroup_matrix`` tiles (MSL spec §2.4, §6.7); K is
decoded from 1-bit packed signs and V from nibble-packed 4-bit codebook
indices on the fly inside the tile loop — no dequantized K or V matrix
is ever materialized.

Score model (differs from the decode kernel — exact dot, not Hamming):

    k_hat[j]    = signs(k_bits[j]) * k_mag[j]      (+-1 per dim)
    score[i][j] = (q[i] . k_hat[j]) * scale + k_const[j]

``scale`` is a plain scalar (fold 1/sqrt(D) here); it is multiplied
into Q once at staging. Values are nibble-packed only (the format
:func:`rabitq_pack_values` produces). Supports both cross-attention
(every query row attends over all S_kv slots) and causal self-attention
(``causal=True``) for autoregressive prefill. Causal alignment follows
the same convention as ``fused_sdpa.metal``: queries align to the tail
of the KV cache, i.e. ``q_abs = (S_kv - S_q) + q_pos``, and slot ``j``
is masked when ``j > q_abs``.

Public API:
  - :func:`rabitq_prefill_attend`
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return (Path(__file__).parent / "src" / filename).read_text()


_cache: dict = {}


# ===========================================================================
# Metal source — FlashAttention-style tiles on simdgroup_matrix
# ===========================================================================
# Grid:        (B * H * ceil(S_q/32) * 32, NSG_C, 1) — MLX grid = threads.
# Threadgroup: (32, NSG_C=4, 1) — 4 SIMD-groups of 32 lanes.
#
# Each threadgroup owns one (b, h, 32-row query block); SIMD-group sg
# owns rows [sg*8, sg*8+8) of that block. All 4 SIMD-groups walk the kv
# chunks TOGETHER: each 8-slot chunk of K (then V) is decoded ONCE into
# a shared staging tile and consumed by all four 8-row Q blocks — this
# is what makes fused prefill viable, since decode work per threadgroup
# is S_kv*D regardless of how many query rows it serves. (v1 gave each
# SIMD-group its own chunks and re-decoded K/V once per 8 query rows —
# 16x more decode traffic; it lost to dequantize+SDPA by 5-10x.)
#
# Decode is byte-wise: one k_bits byte yields 8 dims, one v_idx byte
# yields 2 dims — lanes iterate bytes, not elements.
#
# simdgroup_matrix constraints honoured (spec §2.4): matrix ops run
# under uniform SIMD-group control flow, and since the element-to-thread
# mapping is unspecified, every elementwise step (softmax, rescale,
# accumulate) round-trips through threadgroup memory via
# simdgroup_store.
#
# Precision: Q/K̂/V̂ tiles and 8×8 MAC fragments are half (a QK̂ᵀ dot
# spans D<=128 scale-folded terms; a W·V̂ partial spans 8), the running
# output accumulator and softmax state are float. ~28 KB threadgroup
# memory at D=128; all-float tiles would exceed the 32 KB budget.

_RABITQ_PREFILL_SRC = _read_kernel_source("rabitq_prefill.metal")


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

_N_SIMDGROUPS = 4


def _prefill_kernel(n_bytes: int, d: int):
    key = ("rabitq_prefill_attend", n_bytes, d)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"rabitq_prefill_attend_nb{n_bytes}_d{d}",
            input_names=[
                "q",
                "scale",
                "k_bits",
                "k_mag",
                "k_const",
                "v_idx",
                "v_cents",
                "flags",
            ],
            output_names=["out"],
            source=_RABITQ_PREFILL_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rabitq_prefill_attend(
    q: mx.array,  # [B, H, S_q, D]      fp16 — new-turn queries
    scale: mx.array,  # [1]                 fp32 — softmax scale (1/sqrt(D))
    k_bits: mx.array,  # [B, H, S_kv, D/8]   uint8 — packed 1-bit key signs
    k_mag: mx.array,  # [B, H, S_kv]        fp32 — per-key magnitude
    k_const: mx.array,  # [B, H, S_kv]        fp32 — additive score bias
    v_idx: mx.array,  # [B, H, S_kv, D/2]   uint8 — nibble-packed value indices
    v_cents: mx.array,  # [n_cents <= 16]     fp32 — scalar value codebook
    *,
    causal: bool = False,
) -> mx.array:
    """Tiled prefill attention over the compressed asymmetric cache.

    By default this is cross-attention: every query row attends over all
    ``S_kv`` cached slots. Pass ``causal=True`` for autoregressive
    self-attention prefill — queries then align to the tail of the KV
    cache (``q_abs = (S_kv - S_q) + q_pos``, matching the convention in
    ``fused_sdpa.metal``), and slot ``j`` is masked whenever
    ``j > q_abs``. Scores are exact dots with sign-decoded keys —
    ``(q . signs*k_mag) * scale + k_const`` — computed on
    simdgroup_matrix 8x8 tiles, as is the weight-value product. Values
    must be nibble-packed (see :func:`rabitq_pack_values`).

    Returns:
        ``[B, H, S_q, D]`` fp16 attention output.
    """
    if q.ndim != 4:
        raise ValueError(f"rabitq_prefill_attend: q must be 4D, got {q.shape}")
    B, H, S_q, D = q.shape
    if D % 8 != 0:
        raise ValueError(f"rabitq_prefill_attend: D={D} must be divisible by 8")
    if D > 128:
        raise ValueError(
            f"rabitq_prefill_attend: D={D} exceeds the 128 limit (threadgroup memory budget)"
        )
    n_bytes = D // 8

    if k_bits.ndim != 4 or k_bits.shape[:2] != (B, H) or k_bits.shape[3] != n_bytes:
        raise ValueError(
            f"rabitq_prefill_attend: k_bits must be [B, H, S_kv, {n_bytes}], got {k_bits.shape}"
        )
    S_kv = k_bits.shape[2]
    if k_mag.shape != (B, H, S_kv):
        raise ValueError(f"rabitq_prefill_attend: k_mag must be {(B, H, S_kv)}, got {k_mag.shape}")
    if k_const.shape != (B, H, S_kv):
        raise ValueError(
            f"rabitq_prefill_attend: k_const must be {(B, H, S_kv)}, got {k_const.shape}"
        )
    if v_idx.shape != (B, H, S_kv, D // 2):
        raise ValueError(
            f"rabitq_prefill_attend: v_idx must be nibble-packed "
            f"{(B, H, S_kv, D // 2)} (see rabitq_pack_values), got {v_idx.shape}"
        )
    if v_cents.ndim != 1 or v_cents.shape[0] > 16:
        raise ValueError(
            f"rabitq_prefill_attend: v_cents must be 1D with <= 16 entries, got {v_cents.shape}"
        )

    n_qblk = (S_q + 31) // 32
    n_tg = B * H * n_qblk
    flags = mx.array([1 if causal else 0], dtype=mx.uint32)

    outputs = _prefill_kernel(n_bytes, D)(
        inputs=[
            q.astype(mx.float16),
            scale.reshape(1).astype(mx.float32),
            k_bits.astype(mx.uint8),
            k_mag.astype(mx.float32),
            k_const.astype(mx.float32),
            v_idx.astype(mx.uint8),
            v_cents.astype(mx.float32),
            flags,
        ],
        template=[("N_BYTES", n_bytes), ("MAX_D", D), ("NSG_C", _N_SIMDGROUPS)],
        # MLX grid = total threads; one (32 x NSG) threadgroup per q-block
        grid=(n_tg * 32, _N_SIMDGROUPS, 1),
        threadgroup=(32, _N_SIMDGROUPS, 1),
        output_shapes=[(B, H, S_q, D)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]


__all__ = ["rabitq_prefill_attend"]
