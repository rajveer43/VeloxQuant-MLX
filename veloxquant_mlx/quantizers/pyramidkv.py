"""PyramidKV budget-schedule primitives + attention-mass eviction state.

Inspired by "PyramidKV: Dynamic KV Cache Compression based on Pyramidal
Information Funneling" (Cai et al., 2024, arXiv:2406.02069). Documented as
"PyramidKV-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

The core PyramidKV observation is *pyramidal information funneling*: attention in
early transformer layers is broad and near-uniform, while in deep layers it
concentrates on a few tokens. A single uniform KV budget across all layers is
therefore wasteful — early layers are starved and deep layers over-provisioned.
PyramidKV instead allocates a **pyramid**: large budgets in early layers tapering
to small budgets in deep layers, holding the *average* budget fixed so the total
cache size matches a uniform baseline.

This module holds two things:
  1. ``pyramid_budgets`` — the pure allocator: given ``n_layers`` and an average
     budget, returns the per-layer budget schedule (large → small).
  2. ``PyramidState`` + ``pyramid_update`` — the per-head eviction, which reuses
     the H2O key-as-query cumulative-attention-mass scorer but with the *layer's
     own* budget rather than a global one.

Relationship to H2O-adapted:
  H2O gives every layer the same ``h2o_budget``. PyramidKV is H2O's eviction
  scorer wearing a per-layer budget computed by ``pyramid_budgets``. When the
  schedule is flat (max == min), PyramidKV reduces exactly to H2O.

Adaptation limitations (stated plainly):
  - Key-as-query proxy: attention weights are computed using the incoming key
    vector in place of the true query (not visible at cache level). Same
    approximation as H2O-adapted / SnapKV-adapted.
  - The paper derives the per-layer budget from the *observed* prefill attention
    entropy; we use a fixed monotone schedule (linear taper) as a deterministic,
    calibration-free stand-in. The funneling shape is preserved; the exact
    per-layer values are not data-driven.
  - No RoPE position-ID remapping after eviction.
  - Uniform budget across heads *within* a layer (the pyramid is across layers).

Public API
----------
pyramid_budgets     — per-layer budget schedule (the allocator)
PyramidState        — immutable per-head eviction state
init_pyramid_state  — construct empty state for a layer's budget
pyramid_update      — absorb S new tokens, evict lowest-score token if over budget
pyramid_get_kv      — extract current (keys, values) arrays
pyramid_fp16_bytes  — bytes stored in current state
full_pyramid_fp16_bytes — hypothetical cost without eviction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


def pyramid_budgets(
    n_layers: int,
    avg_budget: int,
    n_sink: int,
    beta: float = 2.0,
) -> list[int]:
    """Per-layer KV budget schedule — large in early layers, small in deep layers.

    The schedule tapers linearly from a maximum at layer 0 to a minimum at the
    last layer, centred so the mean budget equals ``avg_budget``. ``beta`` sets
    how steep the pyramid is: ``beta = 2`` means the first layer gets ~2× the
    average and the last layer ~0× above the sink floor; ``beta = 1`` is flat
    (every layer gets ``avg_budget`` → reduces to uniform H2O).

    The minimum budget is floored at ``n_sink + 1`` so every layer can always
    hold its sinks plus at least one token. For ``beta > 2.0`` the raw linear
    taper's low end would fall below this floor; naively clamping only the
    low layers up (with no compensating reduction elsewhere) would silently
    inflate the realized mean above ``avg_budget`` (#76). Instead, any excess
    from floor-clamping is redistributed away from the *unclamped* layers —
    proportional to their slack above the floor, iterated until stable — so
    the taper shape (steeper for larger ``beta``) is preserved while the mean
    is pulled back to (approximately, subject to integer rounding)
    ``avg_budget`` for every ``beta >= 1.0``, not just ``beta <= 2.0``.

    Args:
        n_layers:   Number of attention-bearing layers.
        avg_budget: Target mean budget across layers (the uniform-H2O baseline).
        n_sink:     Sink tokens each layer protects (sets the min-budget floor).
        beta:       Pyramid steepness. 1.0 = flat; larger = steeper taper.

    Returns:
        Length-``n_layers`` list of per-layer budgets (ints), decreasing, whose
        mean is approximately ``avg_budget`` regardless of ``beta``.
    """
    if n_layers <= 0:
        return []
    if n_layers == 1:
        return [max(avg_budget, n_sink + 1)]
    if beta < 1.0:
        raise ValueError(f"pyramid_budgets: beta must be >= 1.0, got {beta}.")

    floor = n_sink + 1
    # Half-width of the taper around the mean: at beta=2 the top layer gets
    # 2*avg above floor-adjusted centre; symmetric linear ramp keeps the mean
    # (before floor-clamping is applied below).
    span = (avg_budget - floor) * (beta - 1.0)
    hi = avg_budget + span
    lo = avg_budget - span

    values = []
    for i in range(n_layers):
        frac = i / (n_layers - 1)  # 0.0 at layer 0 → 1.0 at last layer
        b = hi + (lo - hi) * frac  # linear taper hi → lo
        values.append(b)

    # Water-filling renormalization: clamp sub-floor layers up to floor, then
    # remove the resulting total excess (above avg_budget * n_layers) from
    # the still-unclamped layers, proportional to their slack above floor.
    # Reclamping can push additional layers to the floor, so repeat until no
    # layer's value moves below floor between iterations (at most one newly
    # clamped layer per pass, hence n_layers passes always suffice).
    target_total = float(avg_budget) * n_layers
    clamped = [False] * n_layers
    for _ in range(n_layers):
        for i in range(n_layers):
            if values[i] < floor:
                values[i] = float(floor)
                clamped[i] = True
        excess = sum(values) - target_total
        if excess <= 1e-9:
            break
        free_idx = [i for i in range(n_layers) if not clamped[i]]
        free_slack_total = sum(values[i] - floor for i in free_idx)
        if not free_idx or free_slack_total <= 1e-9:
            break
        for i in free_idx:
            values[i] -= excess * (values[i] - floor) / free_slack_total

    return [max(int(round(v)), floor) for v in values]


@dataclass
class PyramidState:
    """Per-head PyramidKV eviction state for one layer.

    Identical in shape to H2OState — PyramidKV reuses H2O's cumulative-mass
    scorer; only the ``budget`` differs per layer (set by ``pyramid_budgets``).

    Attributes:
        keys:   [n_kept, D] fp16 stored key rows, or None before first update.
        values: [n_kept, D] fp16 stored value rows, or None before first update.
        scores: [n_kept] cumulative softmax attention mass (float32), or None.
        n_sink: Number of leading sink positions — never evicted.
        budget: This layer's maximum tokens to keep (from the pyramid schedule).
    """

    keys: mx.array | None
    values: mx.array | None
    scores: mx.array | None
    n_sink: int
    budget: int


def init_pyramid_state(n_sink: int, budget: int, head_dim: int) -> PyramidState:  # noqa: ARG001
    """Create an empty PyramidState for a layer with the given per-layer budget.

    Args:
        n_sink:   Number of initial sink positions to protect from eviction.
        budget:   This layer's budget (already resolved from the pyramid schedule).
        head_dim: Head dimension D (unused here; accepted for API symmetry).

    Raises:
        ValueError: if there are sink positions to protect but they leave no
            evictable room within ``budget`` (``n_sink=0, budget=0`` remains
            a valid "disabled cache" configuration).
    """
    if n_sink > 0 and n_sink >= budget:
        raise ValueError(
            f"pyramidkv: n_sink ({n_sink}) must be < budget ({budget}) — no "
            "evictable positions remain, so sinks would be evicted once "
            "the cache fills"
        )
    return PyramidState(keys=None, values=None, scores=None, n_sink=n_sink, budget=budget)


def _attention_scores(query_proxy: mx.array, keys: mx.array) -> mx.array:
    """Softmax attention weights of query_proxy against each key row.

    Args:
        query_proxy: [D] — used as a stand-in for the true query.
        keys:        [n, D] — existing key rows.

    Returns:
        [n] softmax weights summing to ~1.
    """
    scale = 1.0 / math.sqrt(float(query_proxy.shape[-1]))
    logits = (keys @ query_proxy) * scale  # [n]
    return mx.softmax(logits, axis=-1)


def pyramid_update(
    state: PyramidState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
) -> PyramidState:
    """Absorb S new tokens, evicting the lowest-score non-sink token if over this layer's budget.

    Identical mechanics to ``h2o_update`` — cumulative attention-mass scoring with
    sink protection — but bounded by ``state.budget`` (the per-layer pyramid value)
    rather than a global budget.

    Args:
        state:      Current PyramidState for this head.
        new_keys:   [S, D] fp16 new key rows.
        new_values: [S, D] fp16 new value rows.

    Returns:
        Updated PyramidState with at most ``state.budget`` tokens.
    """
    S = new_keys.shape[0]

    for i in range(S):
        k_i = new_keys[i]  # [D]
        v_i = new_values[i]  # [D]

        if state.keys is None:
            # Bootstrap: first token ever — no eviction needed.
            state = PyramidState(
                keys=k_i[None].astype(mx.float16),
                values=v_i[None].astype(mx.float16),
                scores=mx.ones((1,), dtype=mx.float32),
                n_sink=state.n_sink,
                budget=state.budget,
            )
            continue

        # --- score update --------------------------------------------------
        attn = _attention_scores(k_i.astype(mx.float32), state.keys.astype(mx.float32))
        updated_scores = state.scores + attn  # [n_kept]

        # --- append new token (score = 0; begins accumulating next step) ---
        keys_cat = mx.concatenate([state.keys, k_i[None].astype(mx.float16)], axis=0)
        values_cat = mx.concatenate([state.values, v_i[None].astype(mx.float16)], axis=0)
        scores_cat = mx.concatenate([updated_scores, mx.zeros((1,), dtype=mx.float32)], axis=0)

        n_total = keys_cat.shape[0]

        if n_total > state.budget:
            # Build eviction-protected score view: sinks get +inf.
            n_sink_eff = min(state.n_sink, n_total)
            if n_sink_eff > 0:
                inf_block = mx.full((n_sink_eff,), float("inf"), dtype=mx.float32)
                protected = mx.concatenate([inf_block, scores_cat[n_sink_eff:]], axis=0)
            else:
                protected = scores_cat

            evict_idx = int(mx.argmin(protected).item())
            keep_indices = [j for j in range(n_total) if j != evict_idx]
            keys_cat = keys_cat[keep_indices]
            values_cat = values_cat[keep_indices]
            scores_cat = scores_cat[keep_indices]

        state = PyramidState(
            keys=keys_cat,
            values=values_cat,
            scores=scores_cat,
            n_sink=state.n_sink,
            budget=state.budget,
        )

    return state


def pyramid_get_kv(state: PyramidState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update.
    """
    if state.keys is None:
        dummy = mx.zeros((0, 1), dtype=mx.float16)
        return dummy, dummy
    return state.keys, state.values


def pyramid_fp16_bytes(state: PyramidState) -> int:
    """Bytes currently stored for K + V in fp16."""
    if state.keys is None:
        return 0
    n, D = state.keys.shape
    return n * D * 2 * 2  # K + V, 2 bytes each


def full_pyramid_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return tokens_seen * head_dim * 2 * 2  # K + V, 2 bytes each


__all__ = [
    "pyramid_budgets",
    "PyramidState",
    "init_pyramid_state",
    "pyramid_update",
    "pyramid_get_kv",
    "pyramid_fp16_bytes",
    "full_pyramid_fp16_bytes",
]
