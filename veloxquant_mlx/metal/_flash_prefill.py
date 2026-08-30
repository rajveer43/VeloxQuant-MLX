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


def _flash_kernel(d: int):
    key = ("flash_prefill_attend", d)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"flash_prefill_attend_d{d}",
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

    bq_tg = _N_SIMDGROUPS * 8  # BQ=8 rows per SIMD-group (see flash_prefill.metal)
    n_qblk = (S_q + bq_tg - 1) // bq_tg
    n_tg = B * H * n_qblk

    outputs = _flash_kernel(D)(
        inputs=[
            q.astype(mx.float16),
            k.astype(mx.float16),
            v.astype(mx.float16),
            scale.reshape(1).astype(mx.float32),
        ],
        template=[("MAX_D", D), ("NSG_C", _N_SIMDGROUPS)],
        # MLX grid = total threads; one (32 x NSG) threadgroup per q-block
        grid=(n_tg * 32, _N_SIMDGROUPS, 1),
        threadgroup=(32, _N_SIMDGROUPS, 1),
        output_shapes=[(B, H, S_q, D)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]


__all__ = ["flash_prefill_attend"]
