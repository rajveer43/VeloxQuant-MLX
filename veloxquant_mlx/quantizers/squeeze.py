"""SqueezeAttention budget-reallocation primitives + attention-mass eviction state.

Inspired by "SqueezeAttention: 2D Management of KV-Cache in LLM Inference via
Layer-wise Optimal Budget" (Wang et al., 2024, arXiv:2404.04793). Documented as
"SqueezeAttention-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

Where H2O / TOVA / SnapKV budget along one axis (tokens, uniform across layers)
and PyramidKV budgets along the layer axis with a *fixed* positional taper,
SqueezeAttention budgets along **both** the layer and token axes and does so
**data-drivenly**: it measures each layer's attention *concentration* from the
prefill keys and reallocates a fixed total budget toward layers whose attention
is broad (they need more tokens) and away from layers whose attention is
concentrated (they need fewer). Within each layer it then evicts by H2O
cumulative attention mass.

This module holds three things:
  1. ``concentration_score`` — the pure, attention-free concentration proxy:
     mean pairwise cosine similarity of a layer's key set. High → keys cluster
     → attention concentrated → the layer needs *less* budget.
  2. ``squeeze_budgets`` — the pure allocator: given a per-layer concentration
     vector and an average budget, reallocates the total by inverse-concentration
     (mean held fixed, floored at ``n_sink + 1``). ``strength=0`` → uniform.
  3. ``SqueezeState`` + ``squeeze_update`` — the per-head eviction, which reuses
     the H2O key-as-query cumulative-attention-mass scorer but with the *layer's
     resolved* budget (set by the coordinator after prefill) rather than a global
     one.

Relationship to PyramidKV / H2O:
  - PyramidKV uses a *fixed* linear taper resolved at build time (no data).
    SqueezeAttention *measures* concentration from prefill and reallocates —
    the layer budgets are data-driven, resolved once at the prefill boundary by
    ``SqueezeCoordinator``. When ``strength = 0`` every layer gets ``avg_budget``
    and SqueezeAttention reduces exactly to uniform H2O.

Adaptation limitations (stated plainly):
  - Key-as-query proxy: both the concentration measurement and the within-layer
    eviction use the incoming key vector in place of the true query (not visible
    at cache level). Same approximation as H2O-adapted / PyramidKV-adapted.
  - Cosine-dispersion proxy for attention entropy: the paper reads actual
    attention maps; we approximate concentration by the geometry of the key set.
  - One-shot re-budget at the prefill boundary — budgets are resolved once from
    the prompt and then frozen for decode (the paper also re-budgets once).
  - No RoPE position-ID remapping after eviction.
  - Uniform budget across heads *within* a layer (the 2D grid is layer × token).

Public API
----------
concentration_score  — attention-free per-layer concentration proxy
squeeze_budgets       — reallocate a fixed total budget by inverse-concentration
SqueezeState          — immutable per-head eviction state
init_squeeze_state    — construct empty state for a layer's budget
squeeze_update        — absorb S new tokens, evict lowest-score token if over budget
squeeze_get_kv        — extract current (keys, values) arrays
squeeze_fp16_bytes    — bytes stored in current state
full_squeeze_fp16_bytes — hypothetical cost without eviction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


def concentration_score(keys: mx.array) -> float:
    """Attention-free concentration proxy for one layer's key set.

    Computes the mean pairwise cosine similarity of the (row-normalised) key
    rows. A **high** value means the keys cluster tightly in direction — a proxy
    for concentrated attention (a query would attend to a few similar tokens), so
    the layer needs a *smaller* budget. A **low** value means the keys spread out
    in direction — a proxy for broad attention, so the layer needs a *larger*
    budget.

    The diagonal (self-similarity == 1) is excluded so a single-token or
    identical-key set does not trivially report perfect concentration.

    Args:
        keys: ``[n, D]`` key rows for one head/layer (any dtype). ``n`` may be 0
            or 1, in which case there are no off-diagonal pairs.

    Returns:
        Mean off-diagonal cosine similarity in ``[-1, 1]`` (float). Returns 0.0
        when fewer than two key rows are available (no pairs to compare) — a
        neutral concentration that yields the average budget.
    """
    if keys is None:
        return 0.0
    n = keys.shape[0]
    if n < 2:
        return 0.0

    k = keys.astype(mx.float32)
    norms = mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True)) + 1e-8  # [n, 1]
    k_norm = k / norms  # [n, D]
    gram = k_norm @ k_norm.T  # [n, n] cosine

    # Sum off-diagonal entries: total - trace, over n*(n-1) unordered*2 pairs.
    total = float(mx.sum(gram).item())
    diag = float(mx.sum(mx.diagonal(gram)).item())
    off = total - diag
    denom = n * (n - 1)
    return off / denom


def squeeze_budgets(
    concentrations: list[float],
    avg_budget: int,
    n_sink: int,
    strength: float = 1.0,
) -> list[int]:
    """Reallocate a fixed total budget across layers by inverse-concentration.

    Each layer's raw weight is ``(1 - concentration)`` — broad (low-concentration)
    layers weigh more, concentrated layers weigh less. Weights are normalised to
    sum to ``n_layers`` and multiplied by ``avg_budget``, so the *mean* budget is
    exactly ``avg_budget`` (total memory matches a uniform baseline). ``strength``
    linearly interpolates between uniform (``0.0``) and the full inverse-
    concentration allocation (``1.0``):

        budget[i] = avg_budget * ((1 - strength) + strength * weight[i])

    Every budget is floored at ``n_sink + 1`` so a layer can always hold its
    sinks plus one token; the floor may push the realised mean slightly above
    ``avg_budget`` in extreme allocations.

    Args:
        concentrations: Per-layer concentration scores (from
            ``concentration_score``), one per attention-bearing layer. Values are
            clamped to ``[0, 1]`` before use (negative cosine means broad, so it
            clamps to the max-budget end).
        avg_budget: Target mean budget across layers (the uniform baseline).
        n_sink: Sink tokens each layer protects (sets the min-budget floor).
        strength: Reallocation strength. ``0.0`` = uniform (reduces to H2O);
            ``1.0`` = full inverse-concentration split. Must be in ``[0, 1]``.

    Returns:
        Length-``len(concentrations)`` list of per-layer budgets (ints) whose mean
        is approximately ``avg_budget``.

    Raises:
        ValueError: if ``strength`` is outside ``[0, 1]``.
    """
    n_layers = len(concentrations)
    if n_layers == 0:
        return []
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"squeeze_budgets: strength must be in [0, 1], got {strength}.")

    floor = n_sink + 1
    if n_layers == 1:
        return [max(avg_budget, floor)]

    # Clamp concentrations to [0, 1]; raw weight favours broad (low) layers.
    clamped = [min(max(c, 0.0), 1.0) for c in concentrations]
    raw = [1.0 - c for c in clamped]
    total_raw = sum(raw)
    if total_raw <= 0.0:
        # All layers maximally concentrated → fall back to uniform.
        weights = [1.0] * n_layers
    else:
        # Normalise so the weights average 1.0 (sum == n_layers).
        weights = [w * n_layers / total_raw for w in raw]

    budgets: list[int] = []
    for w in weights:
        blended = (1.0 - strength) + strength * w  # 1.0 at strength=0
        b = avg_budget * blended
        budgets.append(max(int(round(b)), floor))
    return budgets


@dataclass
class SqueezeState:
    """Per-head SqueezeAttention eviction state for one layer.

    Identical in shape to H2OState / PyramidState — SqueezeAttention reuses H2O's
    cumulative-mass scorer; only the ``budget`` differs per layer (resolved by the
    coordinator from measured concentration).

    Attributes:
        keys:   [n_kept, D] fp16 stored key rows, or None before first update.
        values: [n_kept, D] fp16 stored value rows, or None before first update.
        scores: [n_kept] cumulative softmax attention mass (float32), or None.
        n_sink: Number of leading sink positions — never evicted.
        budget: This layer's maximum tokens to keep (resolved by squeeze_budgets).
    """

    keys: mx.array | None
    values: mx.array | None
    scores: mx.array | None
    n_sink: int
    budget: int


def init_squeeze_state(n_sink: int, budget: int, head_dim: int) -> SqueezeState:  # noqa: ARG001
    """Create an empty SqueezeState for a layer with the given resolved budget.

    Args:
        n_sink:   Number of initial sink positions to protect from eviction.
        budget:   This layer's budget (already resolved by squeeze_budgets, or the
                  average fallback before the coordinator re-budgets).
        head_dim: Head dimension D (unused here; accepted for API symmetry).

    Raises:
        ValueError: if there are sink positions to protect but they leave no
            evictable room within ``budget`` (``n_sink=0, budget=0`` remains
            a valid "disabled cache" configuration).
    """
    if n_sink > 0 and n_sink >= budget:
        raise ValueError(
            f"squeeze: n_sink ({n_sink}) must be < budget ({budget}) — no "
            "evictable positions remain, so sinks would be evicted once "
            "the cache fills"
        )
    return SqueezeState(keys=None, values=None, scores=None, n_sink=n_sink, budget=budget)


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


def squeeze_update(
    state: SqueezeState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
) -> SqueezeState:
    """Absorb S new tokens, evicting the lowest-score non-sink token if over this layer's budget.

    Identical mechanics to ``h2o_update`` — cumulative attention-mass scoring with
    sink protection — but bounded by ``state.budget`` (the layer's resolved budget
    from ``squeeze_budgets``) rather than a global budget.

    Args:
        state:      Current SqueezeState for this head.
        new_keys:   [S, D] fp16 new key rows.
        new_values: [S, D] fp16 new value rows.

    Returns:
        Updated SqueezeState with at most ``state.budget`` tokens.
    """
    S = new_keys.shape[0]

    for i in range(S):
        k_i = new_keys[i]  # [D]
        v_i = new_values[i]  # [D]

        if state.keys is None:
            # Bootstrap: first token ever — no eviction needed.
            state = SqueezeState(
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

        state = SqueezeState(
            keys=keys_cat,
            values=values_cat,
            scores=scores_cat,
            n_sink=state.n_sink,
            budget=state.budget,
        )

    return state


def squeeze_get_kv(state: SqueezeState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update.
    """
    if state.keys is None:
        dummy = mx.zeros((0, 1), dtype=mx.float16)
        return dummy, dummy
    return state.keys, state.values


def squeeze_fp16_bytes(state: SqueezeState) -> int:
    """Bytes currently stored for K + V in fp16."""
    if state.keys is None:
        return 0
    n, D = state.keys.shape
    return n * D * 2 * 2  # K + V, 2 bytes each


def full_squeeze_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return tokens_seen * head_dim * 2 * 2  # K + V, 2 bytes each


__all__ = [
    "concentration_score",
    "squeeze_budgets",
    "SqueezeState",
    "init_squeeze_state",
    "squeeze_update",
    "squeeze_get_kv",
    "squeeze_fp16_bytes",
    "full_squeeze_fp16_bytes",
]
