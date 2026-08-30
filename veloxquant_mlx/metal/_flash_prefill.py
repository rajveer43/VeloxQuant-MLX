"""Plain-fp16 causal flash attention for from-scratch prefill — simdgroup_matrix.

Issue #277 part 2: for a fresh conversation there is no compressed cache
to exploit yet — K/V are produced at full precision for the first time.
This kernel targets exactly that case: standard causal self-attention
over plain fp16 Q/K/V, no compression, no cross-attention mode, no mask
tensor, no attention sinks. Every one of those is baked out at compile
time rather than handled with runtime branches, which is the concrete
edge this kernel has over a fully general SDPA implementation.

Built from the same online-softmax / simdgroup_matrix tiling as
:func:`rabitq_prefill_attend`, with two deltas aimed specifically at
closing the gap to MLX's own tuned SDPA (see ``blogs/prefill-roofline.md``
for the roofline measurements that motivated them):

  1. ``BK=16`` instead of 8 — half as many threadgroup-barrier round
     trips per unit of K/V processed.
  2. ``exp2`` softmax with the scale pre-folded by ``log2(e)``, and a
     causal block-skip that drops fully-future KV chunks before any
     load/matmul work (not just after, via masking) — both mirror what
     MLX's own steel attention kernel already does.

Public API:
  - :func:`flash_prefill_attend`
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
# Grid:        (B * H * ceil(S_q/(NSG_C*8)) * 32, NSG_C, 1) — MLX grid = threads.
# Threadgroup: (32, NSG_C, 1) — NSG_C SIMD-groups of 32 lanes each.
#
# Each threadgroup owns one (b, h, NSG_C*8-row query block); SIMD-group
# sg owns rows [sg*8, sg*8+8) of that block. All NSG_C SIMD-groups walk the kv
# chunks TOGETHER: each 16-slot chunk of K (then V) is loaded ONCE into
# a shared staging tile and consumed by all four 8-row Q blocks.
#
# Causal block-skip: a threadgroup only visits KV chunks that can
# contain at least one unmasked slot for its query rows — chunks
# entirely in the future are never loaded, matmul'd, or softmaxed.
#
# Precision: Q/K/V tiles and 8x8 MAC fragments are half, the running
# output accumulator and softmax state are float. ~31 KB threadgroup
# memory at D=128 (BQ=8, BK=16, NSG_C=4) — see the module docstring's
# tile-size search; BQ=16 doesn't fit within the 32 KB budget at NSG_C=4.

_FLASH_PREFILL_SRC = _read_kernel_source("flash_prefill.metal")


# ---------------------------------------------------------------------------
# Kernel factory
# ---------------------------------------------------------------------------

_N_SIMDGROUPS = 4

# W.V depth-tile batch (PDT, see PHASE 7 in flash_prefill.metal): larger
# batches amortize more simdgroup_barrier round trips per chunk but grow
# p_tile by PDT*8*2 bytes/SIMD-group. A "largest PDT that fits 32KB"
# heuristic is NOT used here because it measures worse than a smaller
# PDT at D=64: PDT=8 there (20.86KB/threadgroup) was 24-27% SLOWER than
# PDT=4 (18.38KB) across S in {2048, 8192} — direct measurement
# (scripts/flash_prefill_harness.py bench, PDT swept over {1,2,4,8} at
# each D, see blogs/prefill-roofline.md) rather than a memory-budget
# proxy. Measured winners, held fixed as a lookup table:
#   D=32:  PDT=2  (flat across PDT at this D; 2 picked as a
#          middle-of-the-road value — no measured value stood out)
#   D=64:  PDT=4  (measured best at S=2048 and S=8192; PDT=8 regresses)
#   D=128: PDT=2  (only PDT<=2 fits the 32KB hard limit at this D;
#          PDT=2 beats PDT=1 at both S=512 and S=2048)
# Any D not in the table falls back to the largest PDT dividing D/8
# that fits the hard 32KB budget — unmeasured, but a safe default until
# benchmarked.
_MEASURED_PDT = {32: 2, 64: 4, 128: 2}
_PDT_HARD_BUDGET = 32768


# KV chunk width (BK, see PHASE 6 in flash_prefill.metal): wider chunks
# mean fewer barrier round trips per unit of K/V, at the cost of more
# kv_tile/s_tile/w_tile threadgroup memory (which shrinks occupancy).
# BK=32 doesn't fit the 32KB budget at all at D=128 (kv_tile alone would
# be 8KB), so it's only measured at D=32/64. Measured
# (scripts/flash_prefill_harness.py bench + a standalone sweep, BK in
# {16,32} at each valid D — see blogs/prefill-roofline.md):
#   D=32:  BK=32 wins clearly (0.42-0.50ms vs 0.67-1.38ms at S=512,
#          2.39ms vs 2.60-2.77ms at S=2048) — D=32 has the most
#          threadgroup-memory headroom (9.5-14KB even at BK=32), so the
#          wider chunk's barrier savings aren't offset by occupancy loss.
#   D=64:  BK=16 wins (marginally but consistently across two repeated
#          runs at S=2048 and S=8192) — D=64 is already tighter on
#          memory (17-22KB), so BK=32's growth costs more than it saves.
#   D=128: BK=32 is structurally infeasible (kv_tile alone would be
#          8KB, blowing the 32KB budget even at PDT=1) — kept at BK=16.
_MEASURED_BK = {32: 32, 64: 16, 128: 16}


# Rows per SIMD-group (BQ, see PHASE 6 in flash_prefill.metal): BQ=16
# doubles the QK^T/W.V matmul tile (a BQT=2 x BKT grid of 8x8 fragments
# instead of BQT=1) and halves n_qblk (half as many threadgroups
# dispatched, each doing twice the row-work). Only feasible within the
# 32KB budget at D=32 (many BK/PDT combos) and D=64 (BK=16 only, and
# only near the hard ceiling — 31-32KB, i.e. losing whatever occupancy
# margin D=64 had left); infeasible at D=128 at any BK/PDT. Measured
# (standalone sweep, see blogs/prefill-roofline.md): BQ=16 was NOT kept
# — see that section for the specific numbers and why.
_MEASURED_BQ = {32: 8, 64: 8, 128: 8}


def _tile_mem(d: int, bq: int, bk: int, pdt: int) -> int:
    return (
        _N_SIMDGROUPS * bq * d * 2  # q_tile
        + bk * d * 2  # kv_tile
        + _N_SIMDGROUPS * bq * bk * 2  # s_tile
        + _N_SIMDGROUPS * bq * bk * 2  # w_tile
        + _N_SIMDGROUPS * bq * pdt * 8 * 2  # p_tile
        + _N_SIMDGROUPS * bq * d * 4  # out_tile
    )


def _pick_bq(d: int) -> int:
    return _MEASURED_BQ.get(d, 8)


def _pick_bk(d: int) -> int:
    return _MEASURED_BK.get(d, 16)


def _pick_pdt(d: int, bq: int, bk: int) -> int:
    if d in _MEASURED_PDT:
        return _MEASURED_PDT[d]
    n_depth_tiles = d // 8
    for pdt in (8, 4, 2, 1):
        if n_depth_tiles % pdt == 0 and _tile_mem(d, bq, bk, pdt) <= _PDT_HARD_BUDGET:
            return pdt
    return 1


def _flash_kernel(d: int, bq: int, bk: int, pdt: int):
    key = ("flash_prefill_attend", d, bq, bk, pdt)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"flash_prefill_attend_d{d}_bq{bq}_bk{bk}_pdt{pdt}",
            input_names=["q", "k", "v", "scale"],
            output_names=["out"],
            source=_FLASH_PREFILL_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def flash_prefill_attend(
    q: mx.array,  # [B, H, S_q, D]   fp16 — queries
    k: mx.array,  # [B, H, S_kv, D]  fp16 — keys, plain (not compressed)
    v: mx.array,  # [B, H, S_kv, D]  fp16 — values, plain (not compressed)
    scale: mx.array,  # [1]              fp32 — softmax scale (1/sqrt(D))
) -> mx.array:
    """Causal flash attention over plain fp16 K/V — from-scratch prefill.

    Always causal: queries align to the tail of the KV cache
    (``q_abs = (S_kv - S_q) + q_pos``, matching the convention in
    ``fused_sdpa.metal`` / ``rabitq_prefill_attend``), and KV slot ``j``
    is masked whenever ``j > q_abs``. This is the plain-fp16 counterpart
    to ``rabitq_prefill_attend(causal=True)`` — no compression, tuned
    for the shape where ``S_q ≈ S_kv`` and there is no pre-existing
    cache to exploit.

    Returns:
        ``[B, H, S_q, D]`` fp16 attention output.
    """
    if q.ndim != 4:
        raise ValueError(f"flash_prefill_attend: q must be 4D, got {q.shape}")
    B, H, S_q, D = q.shape
    if D % 8 != 0:
        raise ValueError(f"flash_prefill_attend: D={D} must be divisible by 8")
    if D > 128:
        raise ValueError(
            f"flash_prefill_attend: D={D} exceeds the 128 limit (threadgroup memory budget)"
        )
    if k.shape[:2] != (B, H) or k.shape[3] != D:
        raise ValueError(f"flash_prefill_attend: k must be [B, H, S_kv, {D}], got {k.shape}")
    S_kv = k.shape[2]
    if v.shape != (B, H, S_kv, D):
        raise ValueError(f"flash_prefill_attend: v must be {(B, H, S_kv, D)}, got {v.shape}")

    bq = _pick_bq(D)
    bq_tg = _N_SIMDGROUPS * bq
    n_qblk = (S_q + bq_tg - 1) // bq_tg
    n_tg = B * H * n_qblk
    bk = _pick_bk(D)
    pdt = _pick_pdt(D, bq, bk)

    outputs = _flash_kernel(D, bq, bk, pdt)(
        inputs=[
            q.astype(mx.float16),
            k.astype(mx.float16),
            v.astype(mx.float16),
            scale.reshape(1).astype(mx.float32),
        ],
        template=[
            ("MAX_D", D),
            ("NSG_C", _N_SIMDGROUPS),
            ("BQ_ROWS", bq),
            ("KV_CHUNK", bk),
            ("P_DEPTH_TILES", pdt),
        ],
        # MLX grid = total threads; one (32 x NSG) threadgroup per q-block
        grid=(n_tg * 32, _N_SIMDGROUPS, 1),
        threadgroup=(32, _N_SIMDGROUPS, 1),
        output_shapes=[(B, H, S_q, D)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]


__all__ = ["flash_prefill_attend"]
