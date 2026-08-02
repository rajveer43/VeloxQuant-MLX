"""TOVA (Token Omission Via Attention) KV eviction primitives.

Inspired by "Transformers are Multi-State RNNs" (Oren et al., 2024,
arXiv:2401.06104), whose TOVA policy retains a fixed-size cache by dropping, at
each step, the single token with the lowest attention weight in the *current*
step. Documented as "TOVA-adapted (VeloxQuant-MLX implementation)" — not a
faithful port. See adaptation limitations below and in cache/tova_cache.py.

The distinguishing feature versus H2O-adapted: TOVA is *memoryless*. It scores
tokens by the attention weight received at the **current** step only, with no
running accumulation. H2O keeps a cumulative sum (inertial — a token that was a
heavy hitter long ago survives); TOVA reacts instantly to the present context (a
token that stops being attended to is evicted even if it dominated earlier).

Adaptation limitations (stated plainly):
  - Key-as-query proxy: the paper reads the actual attention distribution of the
    most recent query row from the forward pass. At cache level the query is not
    visible, so we use the incoming key vector as a proxy query to approximate
    the current-step attention distribution. Same approximation as SnapKV-adapted
    and H2O-adapted.
  - No RoPE position-ID remapping after eviction.
  - Uniform budget across all heads.
  - When a multi-token chunk (S > 1) arrives, tokens are absorbed one at a time
    and the last incoming key of each step is the proxy query. This differs from
    the paper's single-step decode framing but keeps budget semantics identical.

Public API
----------
TovaState          — immutable per-head state dataclass
init_tova_state    — construct empty state
tova_update        — absorb S new tokens, evict lowest current-step-weight token
tova_get_kv        — extract current (keys, values) arrays
tova_fp16_bytes    — bytes stored in current state
full_tova_fp16_bytes — hypothetical cost without eviction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class TovaState:
    """Per-head sliding TOVA state.

    Attributes:
        keys:   [n_kept, D] fp16 stored key rows, or None before first update.
        values: [n_kept, D] fp16 stored value rows, or None before first update.
        n_sink: Number of leading sink positions — never evicted.
        budget: Maximum tokens to keep at any time (including sinks).

    Note:
        Unlike H2OState there is no ``scores`` field — TOVA is memoryless and
        recomputes the current-step attention weights fresh on every update,
        discarding them after the eviction decision.
    """

    keys: mx.array | None
    values: mx.array | None
    n_sink: int
    budget: int


def init_tova_state(n_sink: int, budget: int, head_dim: int) -> TovaState:  # noqa: ARG001
    """Create an empty TovaState before any tokens arrive.

    Args:
        n_sink:   Number of initial sink positions to protect from eviction.
        budget:   Maximum total tokens kept (sinks + non-sinks).
        head_dim: Head dimension D (unused here; accepted for API symmetry
                  with H2O's init_h2o_state and StreamingLLM's init).

    Raises:
        ValueError: if there are sink positions to protect but they leave no
            evictable room within ``budget`` (``n_sink=0, budget=0`` remains
            a valid "disabled cache" configuration).
    """
    if n_sink > 0 and n_sink >= budget:
        raise ValueError(
            f"tova: n_sink ({n_sink}) must be < budget ({budget}) — no "
            "evictable positions remain, so sinks would be evicted once "
            "the cache fills"
        )
    return TovaState(keys=None, values=None, n_sink=n_sink, budget=budget)


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


def tova_update(
    state: TovaState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
) -> TovaState:
    """Absorb S new tokens into state, evicting the lowest current-step-weight token if over budget.

    For each of the S incoming tokens:
      1. Append the new token to the cache.
      2. If total tokens > budget: compute the current-step attention weights of
         the newly appended key (as proxy query) over *all* rows including
         itself, protect the first ``n_sink`` positions with +inf, and evict the
         non-sink token with the lowest current-step weight.

    Unlike H2O, no per-token score is carried across steps — the weight vector is
    computed fresh each step for the eviction decision and then discarded.

    Args:
        state:      Current TovaState for this head.
        new_keys:   [S, D] fp16 new key rows.
        new_values: [S, D] fp16 new value rows.

    Returns:
        Updated TovaState with at most ``state.budget`` tokens.
    """
    S = new_keys.shape[0]

    for i in range(S):
        k_i = new_keys[i]  # [D]
        v_i = new_values[i]  # [D]

        if state.keys is None:
            # Bootstrap: first token ever — no eviction needed.
            state = TovaState(
                keys=k_i[None].astype(mx.float16),
                values=v_i[None].astype(mx.float16),
                n_sink=state.n_sink,
                budget=state.budget,
            )
            continue

        # --- append new token ----------------------------------------------
        keys_cat = mx.concatenate([state.keys, k_i[None].astype(mx.float16)], axis=0)
        values_cat = mx.concatenate([state.values, v_i[None].astype(mx.float16)], axis=0)

        n_total = keys_cat.shape[0]

        if n_total > state.budget:
            # Current-step attention weights of the new key over ALL rows.
            weights = _attention_scores(
                k_i.astype(mx.float32), keys_cat.astype(mx.float32)
            )  # [n_total]

            # Build eviction-protected weight view: sinks get +inf.
            n_sink_eff = min(state.n_sink, n_total)
            if n_sink_eff > 0:
                inf_block = mx.full((n_sink_eff,), float("inf"), dtype=mx.float32)
                protected = mx.concatenate([inf_block, weights[n_sink_eff:]], axis=0)
            else:
                protected = weights

            evict_idx = int(mx.argmin(protected).item())
            keep_indices = [j for j in range(n_total) if j != evict_idx]
            keys_cat = keys_cat[keep_indices]
            values_cat = values_cat[keep_indices]

        state = TovaState(
            keys=keys_cat,
            values=values_cat,
            n_sink=state.n_sink,
            budget=state.budget,
        )

    return state


def tova_get_kv(state: TovaState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update.
    """
    if state.keys is None:
        dummy = mx.zeros((0, 1), dtype=mx.float16)
        return dummy, dummy
    return state.keys, state.values


def tova_fp16_bytes(state: TovaState) -> int:
    """Bytes currently stored for K + V in fp16."""
    if state.keys is None:
        return 0
    n, D = state.keys.shape
    return n * D * 2 * 2  # K + V, 2 bytes each


def full_tova_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return tokens_seen * head_dim * 2 * 2  # K + V, 2 bytes each


__all__ = [
    "TovaState",
    "init_tova_state",
    "tova_update",
    "tova_get_kv",
    "tova_fp16_bytes",
    "full_tova_fp16_bytes",
]
