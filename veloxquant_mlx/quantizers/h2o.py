"""H2O (Heavy Hitter Oracle) KV eviction primitives.

Inspired by "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
Large Language Models" (Zhang et al., ICLR 2024, arXiv:2306.14048).
Documented as "H2O-adapted (VeloxQuant-MLX implementation)" — not a faithful
port. See adaptation limitations below and in cache/h2o_cache.py.

Adaptation limitations (stated plainly):
  - Key-as-query proxy: the paper accumulates attention weights from the actual
    query vectors at each decode step. At cache level the query is not visible,
    so we use the incoming key vector as a proxy query to approximate the
    attention distribution. This is the same approximation used by SnapKV-adapted.
  - No RoPE position-ID remapping after eviction.
  - Uniform budget across all heads.
  - Scores are accumulated additively (sum of softmax weights) rather than
    the paper's exact formulation, which may differ in low-budget regimes.

Public API
----------
H2OState          — immutable per-head state dataclass
init_h2o_state    — construct empty state
h2o_update        — absorb S new tokens, evict if over budget
h2o_get_kv        — extract current (keys, values) arrays
h2o_fp16_bytes    — bytes stored in current state
full_h2o_fp16_bytes — hypothetical cost without eviction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class H2OState:
    """Per-head sliding H2O state.

    Attributes:
        keys:   [n_kept, D] fp16 stored key rows, or None before first update.
        values: [n_kept, D] fp16 stored value rows, or None before first update.
        scores: [n_kept] cumulative softmax attention mass (float32), or None.
        n_sink: Number of leading sink positions — never evicted.
        budget: Maximum tokens to keep at any time (including sinks).
    """

    keys: mx.array | None
    values: mx.array | None
    scores: mx.array | None
    n_sink: int
    budget: int


def init_h2o_state(n_sink: int, budget: int, head_dim: int) -> H2OState:  # noqa: ARG001
    """Create an empty H2OState before any tokens arrive.

    Args:
        n_sink:   Number of initial sink positions to protect from eviction.
        budget:   Maximum total tokens kept (sinks + non-sinks).
        head_dim: Head dimension D (unused here; accepted for API symmetry
                  with StreamingLLM's init_streaming_window).

    Raises:
        ValueError: if there are sink positions to protect but they leave no
            evictable room within ``budget`` (``n_sink=0, budget=0`` remains
            a valid "disabled cache" configuration).
    """
    if n_sink > 0 and n_sink >= budget:
        raise ValueError(
            f"h2o: n_sink ({n_sink}) must be < budget ({budget}) — no "
            "evictable positions remain, so sinks would be evicted once "
            "the cache fills"
        )
    return H2OState(keys=None, values=None, scores=None, n_sink=n_sink, budget=budget)


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


def h2o_update(
    state: H2OState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
) -> H2OState:
    """Absorb S new tokens into state, evicting the lowest-score non-sink token if over budget.

    For each of the S incoming tokens:
      1. Compute approximate attention weights of the new key (as proxy query)
         over all currently stored keys.
      2. Accumulate those weights into existing per-token scores.
      3. Append the new token with score 0 (it starts accumulating next step).
      4. If total tokens > budget: permanently evict the non-sink token with the
         lowest cumulative score.

    Args:
        state:      Current H2OState for this head.
        new_keys:   [S, D] fp16 new key rows.
        new_values: [S, D] fp16 new value rows.

    Returns:
        Updated H2OState with at most ``state.budget`` tokens.
    """
    S = new_keys.shape[0]

    for i in range(S):
        k_i = new_keys[i]  # [D]
        v_i = new_values[i]  # [D]

        if state.keys is None:
            # Bootstrap: first token ever — no eviction needed.
            state = H2OState(
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
            # Build eviction-protected score view: sinks get +inf
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

        state = H2OState(
            keys=keys_cat,
            values=values_cat,
            scores=scores_cat,
            n_sink=state.n_sink,
            budget=state.budget,
        )

    return state


def h2o_get_kv(state: H2OState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update.
    """
    if state.keys is None:
        dummy = mx.zeros((0, 1), dtype=mx.float16)
        return dummy, dummy
    return state.keys, state.values


def h2o_fp16_bytes(state: H2OState) -> int:
    """Bytes currently stored for K + V in fp16."""
    if state.keys is None:
        return 0
    n, D = state.keys.shape
    return n * D * 2 * 2  # K + V, 2 bytes each


def full_h2o_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return tokens_seen * head_dim * 2 * 2  # K + V, 2 bytes each


__all__ = [
    "H2OState",
    "init_h2o_state",
    "h2o_update",
    "h2o_get_kv",
    "h2o_fp16_bytes",
    "full_h2o_fp16_bytes",
]
