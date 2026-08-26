"""Q-Filters fused eviction Metal kernels — projection scoring + block compaction.

Replaces the generic-MLX scoring and protected top-k in ``qfilters_update``'s
over-budget branch (``veloxquant_mlx/quantizers/qfilters.py``) with two GPU
dispatches, batched across all ``(batch, head)`` pairs at once rather than the
per-head Python iteration the cache does today.

How this differs from the h2o/keyformer kernels it is modelled on
-----------------------------------------------------------------
Those methods evict exactly **one** row per token, so dispatch 1 is an argmin
reduction to a single index and dispatch 2 shifts rows around it in closed
form. Q-Filters instead evicts a whole block down to ``budget`` in one shot,
so:

  1. :func:`_qfilters_score` computes the full ``[BH, n_total]`` projection
     score array ``sign * <k_i, filter_dir>`` (paper Theorem 3.3), with sink
     and recent rows forced to ``+inf`` so they always survive.
  2. The keep-threshold is chosen on the MLX side with ``mx.sort`` — a top-k
     is exactly the primitive MLX already implements well, and re-deriving one
     in Metal would be slower and harder to trust than reusing it.
  3. :func:`_qfilters_evict_apply` compacts the surviving rows against that
     threshold, one threadgroup per group, preserving temporal order.

Ties at the threshold are resolved lowest-index-first inside the apply kernel
(see the .metal file's tie-handling note) so the kept set never exceeds
``budget`` even when scores are duplicated — protected rows all share
``+inf``, so duplicates are the norm here, not an edge case.

Unlike h2o/keyformer, there is **no RoPE remapping**: Q-Filters does not remap
position ids after eviction (a documented limitation of the method as
implemented here), so keys are copied bit-identically.

Public API:
  - :func:`qfilters_fused_evict`
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

# Upper bound on `budget` this kernel serves. The apply kernel stages the
# survivor index list in threadgroup memory, which must be a compile-time
# size; 4096 uint indices = 16 KB, within Metal's 32 KB threadgroup budget
# while covering every budget the cache realistically uses. Callers above this
# fall back to the pure-MLX path rather than silently truncating.
QFILTERS_MAX_BUDGET = 4096


def _read_kernel_source(filename: str) -> str:
    """Read a standalone .metal kernel source file from metal/src/."""
    return (Path(__file__).parent / "src" / filename).read_text()


_cache: dict = {}


# ===========================================================================
# Metal source — dispatch 1: protected projection scoring
# ===========================================================================
# Grid:        (BH * n_total, 1, 1) — one thread per (group, row) pair.
# Threadgroup: (min(n_total, 256), 1, 1).
_QFILTERS_SCORE_SRC = _read_kernel_source("qfilters_score.metal")


# ===========================================================================
# Metal source — dispatch 2: threshold compaction
# ===========================================================================
# Grid:        (BH * TG, 1, 1) — one threadgroup per (batch*head) group.
# Threadgroup: (TG, 1, 1).
_QFILTERS_EVICT_APPLY_SRC = _read_kernel_source("qfilters_evict_apply.metal")


# ---------------------------------------------------------------------------
# Kernel factories
# ---------------------------------------------------------------------------


def _score_kernel():
    key = ("qfilters_score",)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name="qfilters_score",
            input_names=["keys", "filter_dir", "n_sink_arr", "n_recent_arr", "sign_arr"],
            output_names=["scores_out"],
            source=_QFILTERS_SCORE_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


def _evict_apply_kernel(max_budget: int):
    key = ("qfilters_evict_apply", max_budget)
    if key not in _cache:
        _cache[key] = mx.fast.metal_kernel(
            name=f"qfilters_evict_apply_b{max_budget}",
            input_names=["keys_mid", "values_mid", "scores", "thresh_arr", "budget_arr"],
            output_names=["keys_out", "values_out", "scores_out"],
            header=f"#define QF_MAX_BUDGET {max_budget}\n",
            source=_QFILTERS_EVICT_APPLY_SRC,
            ensure_row_contiguous=True,
        )
    return _cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def qfilters_score(
    keys: mx.array,
    filter_dir: mx.array,
    n_sink: int = 0,
    recent: int = 0,
    sign: int = 1,
) -> mx.array:
    """Projection scores ``sign * <k_i, filter_dir>``, sink/recent rows at ``+inf``.

    Args:
        keys:       ``[BH, n_total, D]`` fp16 cached keys.
        filter_dir: ``[BH, D]`` fp32 per-group unit-norm Q-Filter.
        n_sink:     Leading rows forced to survive (scored ``+inf``).
        recent:     Trailing rows forced to survive (scored ``+inf``).
        sign:       ``+1`` keeps the highest projections (the paper's
                    direction); ``-1`` inverts the ranking.

    Returns:
        ``[BH, n_total]`` fp32 scores.

    Raises:
        ValueError: On rank/shape mismatches or a ``sign`` outside ``{-1, +1}``.
    """
    if keys.ndim != 3:
        raise ValueError(f"qfilters_score: keys must be 3D [BH, n_total, D], got {keys.shape}")
    if filter_dir.ndim != 2:
        raise ValueError(f"qfilters_score: filter_dir must be 2D [BH, D], got {filter_dir.shape}")
    bh, n_total, d = (int(x) for x in keys.shape)
    if int(filter_dir.shape[0]) != bh or int(filter_dir.shape[1]) != d:
        raise ValueError(
            f"qfilters_score: filter_dir {tuple(filter_dir.shape)} does not match "
            f"keys [BH={bh}, ..., D={d}]"
        )
    if sign not in (1, -1):
        raise ValueError(f"qfilters_score: sign must be +1 or -1, got {sign!r}")

    tg = min(n_total, 256)
    n_threads = bh * n_total
    n_tg = (n_threads + tg - 1) // tg

    (scores,) = _score_kernel()(
        inputs=[
            keys.astype(mx.float16),
            filter_dir.astype(mx.float32),
            mx.array([n_sink], dtype=mx.uint32),
            mx.array([recent], dtype=mx.uint32),
            mx.array([float(sign)], dtype=mx.float32),
        ],
        grid=(n_tg * tg, 1, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(bh, n_total)],
        output_dtypes=[mx.float32],
    )
    return scores


def qfilters_fused_evict(
    keys_mid: mx.array,
    values_mid: mx.array,
    filter_dir: mx.array,
    budget: int,
    n_sink: int = 0,
    recent: int = 0,
    sign: int = 1,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fused projection scoring + block eviction, batched over ``(batch*head)``.

    Matches ``qfilters_update``'s over-budget branch: score every stored row
    against the frozen filter, then keep the ``budget`` highest-scoring rows
    with sinks and the trailing ``recent`` window forced to survive, in
    original temporal order.

    Callers must only invoke this when the group is over budget
    (``n_total > budget``) — the under-budget case has no eviction step.

    Args:
        keys_mid:   ``[BH, n_total, D]`` fp16 stored + newly appended keys.
        values_mid: ``[BH, n_total, D]`` fp16, same layout.
        filter_dir: ``[BH, D]`` fp32 frozen per-group unit-norm Q-Filter.
        budget:     Rows to retain per group (including protected rows).
        n_sink:     Leading rows never evicted.
        recent:     Trailing rows never evicted.
        sign:       ``+1`` = paper direction; ``-1`` = inverted ablation.

    Returns:
        ``(keys_out, values_out, scores_out)`` — keys/values ``[BH, budget, D]``
        fp16, scores ``[BH, budget]`` fp32, all in temporal order.

    Raises:
        ValueError: If shapes disagree, ``budget`` is out of range, or the
            group is not actually over budget.
    """
    if keys_mid.ndim != 3:
        raise ValueError(
            f"qfilters_fused_evict: keys_mid must be 3D [BH, n_total, D], got {keys_mid.shape}"
        )
    if values_mid.shape != keys_mid.shape:
        raise ValueError(
            f"qfilters_fused_evict: values_mid {tuple(values_mid.shape)} must match "
            f"keys_mid {tuple(keys_mid.shape)}"
        )
    bh, n_total, d = (int(x) for x in keys_mid.shape)
    if budget <= 0:
        raise ValueError(f"qfilters_fused_evict: budget must be > 0, got {budget}")
    if budget > QFILTERS_MAX_BUDGET:
        raise ValueError(
            f"qfilters_fused_evict: budget={budget} exceeds QFILTERS_MAX_BUDGET="
            f"{QFILTERS_MAX_BUDGET} (threadgroup index-list capacity) — use the "
            "pure-MLX path for budgets this large"
        )
    if n_total <= budget:
        raise ValueError(
            f"qfilters_fused_evict: n_total={n_total} <= budget={budget} — caller must "
            "only invoke this when the group is over budget"
        )
    if n_sink + recent > budget:
        raise ValueError(
            f"qfilters_fused_evict: n_sink ({n_sink}) + recent ({recent}) exceeds "
            f"budget ({budget}) — protected rows alone would overflow the kept set"
        )

    scores = qfilters_score(keys_mid, filter_dir, n_sink=n_sink, recent=recent, sign=sign)

    # Keep-threshold = the budget-th largest score per group. Sorting ascending
    # and indexing from the end mirrors qfilters_update's argsort-based
    # selection, so both paths break ties toward the same rows.
    ordered = mx.sort(scores, axis=-1)  # ascending, [BH, n_total]
    thresh = ordered[:, n_total - budget]  # [BH]

    keys_out, values_out, scores_out = _evict_apply_kernel(QFILTERS_MAX_BUDGET)(
        inputs=[
            keys_mid.astype(mx.float16),
            values_mid.astype(mx.float16),
            scores,
            thresh.astype(mx.float32),
            mx.array([budget], dtype=mx.uint32),
        ],
        grid=(bh * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(bh, budget, d), (bh, budget, d), (bh, budget)],
        output_dtypes=[mx.float16, mx.float16, mx.float32],
    )
    return keys_out, values_out, scores_out


__all__ = ["QFILTERS_MAX_BUDGET", "qfilters_fused_evict", "qfilters_score"]
