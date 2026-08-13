"""H2O-adapted fused eviction Metal kernels — argmin reduction + apply.

Replaces the Python per-token loop in ``h2o_update``'s over-budget branch
(``veloxquant_mlx/quantizers/h2o.py``) — append, sink-protected argmin,
evict, RoPE-remap the shifted rows — with two GPU dispatches per incoming
token, batched across all ``(batch, head)`` pairs at once, instead of one
Python-level iteration per ``(batch, head)`` per token plus ~15 small MLX
ops each. Design: see ``paper/research/H2O_METAL_KERNEL_TECH_SPEC.md``.

Two dispatches, not fused into one barrier-synchronized kernel (spec
decision D4): the eviction decision (``evict_idx``) must be known by every
thread before the compaction/rotation phase runs, and a single-threadgroup
barrier-fused design would cap the largest ``h2o_budget`` this kernel could
serve to whatever fits in one threadgroup's grid-stride reduction. Two
dispatches avoid that ceiling entirely, at the cost of one extra dispatch
per token — a real but modest cost against the ~15+ ops it replaces.

  1. :func:`_h2o_evict_reduce` — sink-protected argmin over ``n_total``
     candidate rows per ``(batch, head)`` group, one threadgroup per group.
  2. :func:`_h2o_evict_apply` — given ``evict_idx``, compacts the surviving
     ``n_kept`` rows and re-rotates (NeoX-style RoPE, ``rope_remap_positions``
     equivalent) exactly the rows whose position shifted, matching
     ``h2o_update``'s per-token eviction branch bit-for-bit.

Precondition (spec decision D3): callers MUST only invoke
:func:`h2o_fused_evict` when every ``(batch, head)`` group is already over
budget (``n_kept + 1 > budget``) — the below-budget case is handled entirely
by the existing vectorized ``_batch_absorb_no_eviction`` MLX path and is not
implemented here.

Public API:
  - :func:`h2o_fused_evict`
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return (Path(__file__).parent / "src" / filename).read_text()


_cache: dict = {}


# ===========================================================================
# Metal source — dispatch 1: sink-protected argmin reduction
# ===========================================================================
# Grid:        (BH, 1, 1) threadgroups — one per (batch*head) group.
# Threadgroup: (32, NSG_C, 1) — NSG_C SIMD-groups of 32 lanes.
#
# Each threadgroup grid-strides its NSG_C*32 threads over the n_total
# candidate rows (n_kept stored + 1 newly appended), tracking a running
# (min value, min index) pair. A SIMD-group butterfly reduction
# (simd_shuffle_xor) merges the 32 lanes of each SIMD-group; a final
# threadgroup-memory merge combines the NSG_C SIMD-group results. Ties
# resolve toward the lowest index, matching mx.argmin.

_H2O_EVICT_REDUCE_SRC = _read_kernel_source("h2o_evict_reduce.metal")


# ===========================================================================
# Metal source — dispatch 2: compact + conditionally re-rotate
# ===========================================================================
# Grid:        (BH * n_kept, 1, 1) — one thread per output row.
# Threadgroup: (min(n_kept, 256), 1, 1).
#
# Each thread computes its source row (skipping evict_idx), copies
# values/scores/positions verbatim, and either copies keys bit-identically
# (source position <= evicted position) or applies a delta-rotation by -1
# position (source position > evicted position) — see
# H2O_METAL_KERNEL_TECH_SPEC.md section 3, step 7.

_H2O_EVICT_APPLY_SRC = _read_kernel_source("h2o_evict_apply.metal")


# ---------------------------------------------------------------------------
# Kernel factories
# ---------------------------------------------------------------------------


def _evict_reduce_kernel(nsg: int):
    key = ("h2o_evict_reduce", nsg)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"h2o_evict_reduce_nsg{nsg}",
            input_names=["scores_mid", "n_sink_arr", "n_grace_arr"],
            output_names=["evict_idx"],
            header=f"#define NSG_C {nsg}\n",
            source=_H2O_EVICT_REDUCE_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


def _evict_apply_kernel():
    key = ("h2o_evict_apply",)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name="h2o_evict_apply",
            input_names=[
                "keys_mid",
                "values_mid",
                "scores_mid",
                "positions_mid",
                "evict_idx",
                "rope_base_arr",
            ],
            output_names=["keys_out", "values_out", "scores_out", "positions_out"],
            source=_H2O_EVICT_APPLY_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def h2o_fused_evict(
    keys_mid: mx.array,
    values_mid: mx.array,
    scores_mid: mx.array,
    positions_mid: mx.array,
    n_sink: int,
    rope_base: float,
    grace: int = 0,
    nsg: int = 4,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Fused sink+grace-protected argmin + evict + RoPE-remap, batched over (batch*head).

    Matches ``h2o_update``'s per-token eviction branch bit-for-bit (see
    ``H2O_METAL_KERNEL_TECH_SPEC.md`` section 3). Caller must have already
    computed the "mid" state — the ``n_kept`` currently-stored rows with the
    one new token conceptually appended (score 0, its own position) — and
    must only call this when every group is over budget
    (``n_total = n_kept + 1 > budget``); the below-budget case is handled by
    the existing ``_batch_absorb_no_eviction`` MLX path.

    Args:
        keys_mid:      ``[BH, n_total, D]`` fp16 — stored + appended keys.
        values_mid:    ``[BH, n_total, D]`` fp16.
        scores_mid:    ``[BH, n_total]`` fp32 cumulative scores (appended
                       row's score is 0.0, per ``h2o_update``).
        positions_mid: ``[BH, n_total]`` int32 absolute positions (appended
                       row's position is ``next_pos``).
        n_sink:        Number of leading positions protected from eviction
                       (uniform across all BH groups, matching
                       ``H2OState.n_sink``).
        rope_base:     RoPE frequency base — must match the model's own, or
                       the re-rotation of shifted rows will not cancel out
                       the original rotation correctly (same requirement as
                       ``rope_remap_positions``).
        grace:         Number of most-recently-arrived rows (trailing array
                       index — positions are always sorted ascending, see
                       ``H2OState.grace``) protected from eviction, uniform
                       across all BH groups.
        nsg:           SIMD-groups per threadgroup for the reduction kernel.

    Returns:
        ``(keys_out, values_out, scores_out, positions_out)``, each with
        ``n_total - 1`` rows (one row evicted) — ``keys_out``/``values_out``
        fp16, ``scores_out`` fp32, ``positions_out`` int32.
    """
    if keys_mid.ndim != 3:
        raise ValueError(
            f"h2o_fused_evict: keys_mid must be 3D [BH, n_total, D], got {keys_mid.shape}"
        )
    BH, n_total, D = keys_mid.shape
    n_kept = n_total - 1
    if n_kept <= 0:
        raise ValueError(
            f"h2o_fused_evict: n_total={n_total} implies n_kept<=0 — nothing to evict from"
        )
    if D % 2 != 0:
        raise ValueError(f"h2o_fused_evict: D={D} must be even (NeoX-style RoPE pairs)")
    if not (1 <= nsg <= 32):
        raise ValueError(f"h2o_fused_evict: nsg={nsg} must be in 1..32")

    n_sink_arr = mx.array([n_sink], dtype=mx.uint32)
    n_grace_arr = mx.array([grace], dtype=mx.uint32)

    (evict_idx,) = _evict_reduce_kernel(nsg)(
        inputs=[scores_mid.astype(mx.float32), n_sink_arr, n_grace_arr],
        grid=(BH * 32, nsg, 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(BH,)],
        output_dtypes=[mx.int32],
    )

    rope_base_arr = mx.array([rope_base], dtype=mx.float32)
    tg = min(n_kept, 256)
    n_tg = (BH * n_kept + tg - 1) // tg

    keys_out, values_out, scores_out, positions_out = _evict_apply_kernel()(
        inputs=[
            keys_mid.astype(mx.float16),
            values_mid.astype(mx.float16),
            scores_mid.astype(mx.float32),
            positions_mid.astype(mx.int32),
            evict_idx,
            rope_base_arr,
        ],
        grid=(n_tg * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(BH, n_kept, D), (BH, n_kept, D), (BH, n_kept), (BH, n_kept)],
        output_dtypes=[mx.float16, mx.float16, mx.float32, mx.int32],
    )
    return keys_out, values_out, scores_out, positions_out


__all__ = ["h2o_fused_evict"]
