"""Row-owned streaming causal prefill attention — experimental alternative.

From-scratch alternative architecture to ``flash_prefill_attend``
(``_flash_prefill.py`` / ``flash_prefill.metal``), built to compare a
genuinely different computational decomposition against the conventional
tiled ``simdgroup_matrix`` FlashAttention-style kernel. See
``metal/src/experimental_streaming_prefill_ARCHITECTURE.md`` for the full
design rationale — in short: ownership is by QUERY ROW (one SIMD-group
owns one query row for the whole kernel), each lane owns a fixed
stride-32 slab of head-dims, K/V stream directly from device memory one
(or a small block of) token(s) at a time, and there is zero threadgroup
memory / zero barriers anywhere in this kernel family.

This module is purely additive: it does not modify ``_flash_prefill.py``,
``flash_prefill.metal``, or any existing export. ``flash_prefill_attend``
remains the production kernel used elsewhere in the codebase.

Public API:
  - :func:`streaming_prefill_attend`
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return (Path(__file__).parent / "src" / filename).read_text()


_STREAMING_PREFILL_SRC = _read_kernel_source("experimental_streaming_prefill.metal")

_cache: dict = {}

# implementation name -> (kv_block, rows_per_threadgroup)
_IMPLEMENTATIONS = {
    "streaming": (1, 1),
    "streaming_block2": (2, 1),
    "streaming_block4": (4, 1),
    "streaming_block8": (8, 1),
    # multirow uses the block=1 single-row algorithm (per the architecture
    # note's Step 8/kernel-family description) but dispatches 4 independent
    # SIMD-groups per threadgroup purely for occupancy/dispatch granularity.
    "streaming_multirow": (1, 4),
}

_MULTIROW_ROWS_PER_TG = 4


def _stream_kernel(d: int, kv_block: int, rows_per_tg: int):
    key = ("streaming_prefill_attend", d, kv_block, rows_per_tg)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"stream_prefill_attend_d{d}_kb{kv_block}_rtg{rows_per_tg}",
            input_names=["q", "k", "v", "scale"],
            output_names=["out"],
            source=_STREAMING_PREFILL_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def streaming_prefill_attend(
    q: mx.array,  # [B, H, S_q, D]   fp16 — queries
    k: mx.array,  # [B, H, S_kv, D]  fp16 — keys, plain (not compressed)
    v: mx.array,  # [B, H, S_kv, D]  fp16 — values, plain (not compressed)
    scale: mx.array,  # [1]              fp32 — softmax scale (1/sqrt(D))
    implementation: str = "streaming",
) -> mx.array:
    """Row-owned streaming causal attention over plain fp16 K/V.

    An experimental, from-scratch alternative to ``flash_prefill_attend``:
    instead of tiled ``simdgroup_matrix`` matmuls staged through
    threadgroup memory, one SIMD-group (32 lanes) owns one query row for
    the whole kernel and streams K/V directly from device memory, with
    the online-softmax update redundantly (but bit-identically) computed
    in every lane. Zero threadgroup memory, zero barriers. See
    ``metal/src/experimental_streaming_prefill_ARCHITECTURE.md``.

    Always causal: queries align to the tail of the KV cache
    (``q_abs = (S_kv - S_q) + q_pos``), matching ``flash_prefill_attend``'s
    convention exactly, so the two kernels are directly comparable.

    Args:
        q, k, v: fp16 tensors, shapes as above.
        scale: fp32 ``[1]`` softmax scale.
        implementation: one of ``"streaming"`` (block=1 baseline),
            ``"streaming_block2"``, ``"streaming_block4"``,
            ``"streaming_block8"`` (KV-blocked variants — 2/4/8 KV tokens
            per online-softmax update), or ``"streaming_multirow"`` (4
            independent SIMD-groups per threadgroup, block=1 algorithm,
            zero cross-SIMD-group communication).

    Returns:
        ``[B, H, S_q, D]`` fp16 attention output.
    """
    if implementation not in _IMPLEMENTATIONS:
        raise ValueError(
            f"streaming_prefill_attend: unknown implementation={implementation!r}; "
            f"must be one of {sorted(_IMPLEMENTATIONS)}"
        )
    if q.ndim != 4:
        raise ValueError(f"streaming_prefill_attend: q must be 4D, got {q.shape}")
    B, H, S_q, D = q.shape
    if D % 32 != 0:
        raise ValueError(
            f"streaming_prefill_attend: D={D} must be a multiple of 32 "
            "(one head-dim slab per lane in the owning SIMD-group)"
        )
    if D > 128:
        raise ValueError(
            f"streaming_prefill_attend: D={D} exceeds the 128 limit "
            "(only D in {32, 64, 128} is specialized)"
        )
    if k.shape[:2] != (B, H) or k.shape[3] != D:
        raise ValueError(f"streaming_prefill_attend: k must be [B, H, S_kv, {D}], got {k.shape}")
    S_kv = k.shape[2]
    if v.shape != (B, H, S_kv, D):
        raise ValueError(f"streaming_prefill_attend: v must be {(B, H, S_kv, D)}, got {v.shape}")

    kv_block, rows_per_tg = _IMPLEMENTATIONS[implementation]

    n_qblk = (S_q + rows_per_tg - 1) // rows_per_tg
    n_tg = B * H * n_qblk

    outputs = _stream_kernel(D, kv_block, rows_per_tg)(
        inputs=[
            q.astype(mx.float16),
            k.astype(mx.float16),
            v.astype(mx.float16),
            scale.reshape(1).astype(mx.float32),
        ],
        template=[
            ("MAX_D", D),
            ("KV_BLOCK", kv_block),
            ("ROWS_PER_TG", rows_per_tg),
        ],
        # MLX grid = total threads; one (32 x ROWS_PER_TG) threadgroup per
        # (b, h, query-row-block).
        grid=(n_tg * 32, rows_per_tg, 1),
        threadgroup=(32, rows_per_tg, 1),
        output_shapes=[(B, H, S_q, D)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]


__all__ = ["streaming_prefill_attend"]
