"""Keyformer-adapted fused eviction Metal kernels — Gumbel-argmin reduction + apply.

Replaces the Python per-token loop in ``keyformer_update``'s over-budget
branch (``veloxquant_mlx/quantizers/keyformer.py``) — append, sink/recent-
protected Gumbel-regularized argmin, evict, RoPE-remap the shifted rows —
with two GPU dispatches per incoming token, batched across all
``(batch, head)`` pairs at once. Structurally this is
:mod:`veloxquant_mlx.metal._h2o_evict` with ONE addition threaded through
both dispatches: a per-row frozen Gumbel value, compacted alongside
keys/values/scores/positions, and folded into the reduction's selection
value as ``score + tau * gumbel`` instead of the raw score. At ``tau == 0``
the reduction is bit-for-bit identical to H2O-adapted's kernel, matching the
``tau=0 == H2O-adapted`` collapse the pure-MLX path guarantees.

Two dispatches, not fused into one barrier-synchronized kernel — same
rationale as H2O's kernel (spec decision D4 in
``paper/research/H2O_METAL_KERNEL_TECH_SPEC.md``): the eviction decision
(``evict_idx``) must be known by every thread before the compaction/rotation
phase runs.

  1. :func:`_keyformer_evict_reduce` — sink/recent-protected Gumbel-regularized
     argmin over ``n_total`` candidate rows per ``(batch, head)`` group, one
     threadgroup per group.
  2. :func:`_keyformer_evict_apply` — given ``evict_idx``, compacts the
     surviving ``n_kept`` rows (including gumbel) and re-rotates (NeoX-style
     RoPE) exactly the rows whose position shifted, matching
     ``keyformer_update``'s per-token eviction branch bit-for-bit.

Precondition (mirrors H2O's D3): callers MUST only invoke
:func:`keyformer_fused_evict` when every ``(batch, head)`` group is already
over budget (``n_kept + 1 > budget``) — the below-budget case has no eviction
step at all.

Public API:
  - :func:`keyformer_fused_evict`
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return (Path(__file__).parent / "src" / filename).read_text()


_cache: dict = {}


# ===========================================================================
# Metal source — dispatch 1: sink/recent-protected Gumbel-regularized argmin
# ===========================================================================
# Grid:        (BH, 1, 1) threadgroups — one per (batch*head) group.
# Threadgroup: (32, NSG_C, 1) — NSG_C SIMD-groups of 32 lanes.
#
# Each threadgroup grid-strides its NSG_C*32 threads over the n_total
# candidate rows (n_kept stored + 1 newly appended), tracking a running
# (min value, min index) pair over "score + tau * gumbel". A SIMD-group
# butterfly reduction (simd_shuffle_xor) merges the 32 lanes of each
# SIMD-group; a final threadgroup-memory merge combines the NSG_C SIMD-group
# results. Ties resolve toward the lowest index, matching mx.argmin.

_KEYFORMER_EVICT_REDUCE_SRC = _read_kernel_source("keyformer_evict_reduce.metal")


# ===========================================================================
# Metal source — dispatch 2: compact (incl. gumbel) + conditionally re-rotate
# ===========================================================================
# Grid:        (BH * n_kept, 1, 1) — one thread per output row.
# Threadgroup: (min(n_kept, 256), 1, 1).

_KEYFORMER_EVICT_APPLY_SRC = _read_kernel_source("keyformer_evict_apply.metal")


# ---------------------------------------------------------------------------
# Kernel factories
# ---------------------------------------------------------------------------


def _evict_reduce_kernel(nsg: int):
    key = ("keyformer_evict_reduce", nsg)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"keyformer_evict_reduce_nsg{nsg}",
            input_names=["scores_mid", "gumbel_mid", "n_sink_arr", "n_recent_arr", "tau_arr"],
            output_names=["evict_idx"],
            header=f"#define NSG_C {nsg}\n",
            source=_KEYFORMER_EVICT_REDUCE_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


def _evict_apply_kernel():
    key = ("keyformer_evict_apply",)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name="keyformer_evict_apply",
            input_names=[
                "keys_mid",
                "values_mid",
                "scores_mid",
                "gumbel_mid",
                "positions_mid",
                "evict_idx",
                "rope_base_arr",
            ],
            output_names=["keys_out", "values_out", "scores_out", "gumbel_out", "positions_out"],
            source=_KEYFORMER_EVICT_APPLY_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def keyformer_fused_evict(
    keys_mid: mx.array,
    values_mid: mx.array,
    scores_mid: mx.array,
    gumbel_mid: mx.array,
    positions_mid: mx.array,
    n_sink: int,
    rope_base: float,
    tau: float = 0.0,
    recent: int = 0,
    nsg: int = 4,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Fused Gumbel-regularized argmin + evict + RoPE-remap, batched over (batch*head).

    Matches ``keyformer_update``'s per-token eviction branch bit-for-bit.
    Caller must have already computed the "mid" state — the ``n_kept``
    currently-stored rows with the one new token conceptually appended
    (score 0, its own frozen Gumbel draw, its own position) — and must only
    call this when every group is over budget (``n_total = n_kept + 1 >
    budget``).

    Args:
        keys_mid:      ``[BH, n_total, D]`` fp16 — stored + appended keys.
        values_mid:    ``[BH, n_total, D]`` fp16.
        scores_mid:    ``[BH, n_total]`` fp32 cumulative proxy-attention scores
                       (appended row's score is 0.0, per ``keyformer_update``).
        gumbel_mid:    ``[BH, n_total]`` fp32 frozen per-position Gumbel noise.
        positions_mid: ``[BH, n_total]`` int32 absolute positions (appended
                       row's position is ``next_pos``).
        n_sink:        Number of leading positions protected from eviction
                       (uniform across all BH groups).
        rope_base:     RoPE frequency base — must match the model's own.
        tau:           Current annealed Gumbel temperature (see
                       ``quantizers/keyformer.py``'s ``_tau_at``) — the
                       caller computes the schedule; this kernel only applies
                       the resulting scalar. ``0.0`` collapses the selection
                       to the raw score (H2O-adapted's argmin).
        recent:        Number of most-recently-arrived rows (trailing array
                       index) protected from eviction, uniform across all BH
                       groups.
        nsg:           SIMD-groups per threadgroup for the reduction kernel.

    Returns:
        ``(keys_out, values_out, scores_out, gumbel_out, positions_out)``,
        each with ``n_total - 1`` rows (one row evicted) — ``keys_out``/
        ``values_out`` fp16, ``scores_out``/``gumbel_out`` fp32,
        ``positions_out`` int32.
    """
    if keys_mid.ndim != 3:
        raise ValueError(
            f"keyformer_fused_evict: keys_mid must be 3D [BH, n_total, D], got {keys_mid.shape}"
        )
    BH, n_total, D = keys_mid.shape
    n_kept = n_total - 1
    if n_kept <= 0:
        raise ValueError(
            f"keyformer_fused_evict: n_total={n_total} implies n_kept<=0 — nothing to evict from"
        )
    if D % 2 != 0:
        raise ValueError(f"keyformer_fused_evict: D={D} must be even (NeoX-style RoPE pairs)")
    if not (1 <= nsg <= 32):
        raise ValueError(f"keyformer_fused_evict: nsg={nsg} must be in 1..32")

    n_sink_arr = mx.array([n_sink], dtype=mx.uint32)
    n_recent_arr = mx.array([recent], dtype=mx.uint32)
    tau_arr = mx.array([tau], dtype=mx.float32)

    (evict_idx,) = _evict_reduce_kernel(nsg)(
        inputs=[
            scores_mid.astype(mx.float32),
            gumbel_mid.astype(mx.float32),
            n_sink_arr,
            n_recent_arr,
            tau_arr,
        ],
        grid=(BH * 32, nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(BH,)],
        output_dtypes=[mx.int32],
    )

    rope_base_arr = mx.array([rope_base], dtype=mx.float32)
    tg = min(n_kept, 256)
    n_tg = (BH * n_kept + tg - 1) // tg

    keys_out, values_out, scores_out, gumbel_out, positions_out = _evict_apply_kernel()(
        inputs=[
            keys_mid.astype(mx.float16),
            values_mid.astype(mx.float16),
            scores_mid.astype(mx.float32),
            gumbel_mid.astype(mx.float32),
            positions_mid.astype(mx.int32),
            evict_idx,
            rope_base_arr,
        ],
        grid=(n_tg * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(BH, n_kept, D), (BH, n_kept, D), (BH, n_kept), (BH, n_kept), (BH, n_kept)],
        output_dtypes=[mx.float16, mx.float16, mx.float32, mx.float32, mx.int32],
    )
    return keys_out, values_out, scores_out, gumbel_out, positions_out


__all__ = ["keyformer_fused_evict"]
