"""CurDKV (value-guided leverage-score) KV eviction primitives.

Inspired by "Value-Guided KV Compression for LLMs via Approximated CUR
Decomposition" (Sengupta, Chaudhary, Chakraborty; NeurIPS 2025,
arXiv:2509.15038). Documented as "CurDKV-adapted (VeloxQuant-MLX
implementation)" — not a faithful port. See adaptation limitations below and
in cache/curdkv_cache.py.

Adaptation limitations (stated plainly):
  - Key-as-query proxy: the paper's leverage scores are derived from the true
    attention-output matrix softmax(QK^T)V, built from the model's real query
    vectors. At cache level the query is not visible, so we use the incoming
    key vector as a proxy query, exactly the approximation H2O-adapted and
    SnapKV-adapted already use.
  - Approximated leverage scores via a small-rank SVD of the proxy
    attention-weighted value block, not the paper's own CUR sampling
    algorithm. This is a standard, generically-cited leverage-score estimator
    (Mahoney & Drineas-style), not a reproduction of the paper's specific
    sketching routine.
  - Uniform budget across all heads (same convention as H2O-adapted).
  - No RoPE position-ID remapping after eviction.

The mechanism gap this closes: every other eviction method in this repo
(H2O, SnapKV, TOVA, PyramidKV, Keyformer, MorphKV, KVzip, ...) scores a token
using only its *key* side (attention-mass, norm, key-SVD projection,
reconstruction reliance). None of them fold the *value* vector's own
contribution into the retention decision. CurDKV-adapted derives leverage
scores from the joint (K, V) structure: two tokens with identical keys but
different values receive different scores here — see
`test_curdkv.py::test_identical_keys_different_values_diverge`.

Public API
----------
CurDKVState          — immutable per-head state dataclass
init_curdkv_state    — construct empty state
curdkv_update        — absorb S new tokens, evict if over budget
curdkv_get_kv        — extract current (keys, values) arrays
curdkv_fp16_bytes    — bytes stored in current state
full_curdkv_fp16_bytes — hypothetical cost without eviction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class CurDKVState:
    """Per-head sliding CurDKV state.

    Attributes:
        keys:   [n_kept, D] fp16 stored key rows, or None before first update.
        values: [n_kept, D] fp16 stored value rows, or None before first update.
        leverage_scores: [n_kept] cumulative leverage-score estimate
                         (float32), or None.
        n_updates: [n_kept] number of accumulation steps each token's score
                   has been through so far (float32), including its own
                   self-leverage seed. Used to compare tokens of different
                   ages on a fair per-step-average basis at eviction time —
                   see honesty crux point 4 in the docs for why a raw
                   cumulative-sum comparison is scale-biased against
                   newcomers. `None` before the first update.
        n_sink: Number of leading sink positions — never evicted.
        budget: Maximum tokens to keep at any time (including sinks).
        rank_cap: SVD rank cap used for leverage-score estimation.
    """

    keys: mx.array | None
    values: mx.array | None
    leverage_scores: mx.array | None
    n_updates: mx.array | None
    n_sink: int
    budget: int
    rank_cap: int


def init_curdkv_state(n_sink: int, budget: int, head_dim: int, rank_cap: int = 16) -> CurDKVState:  # noqa: ARG001
    """Create an empty CurDKVState before any tokens arrive.

    Args:
        n_sink:   Number of initial sink positions to protect from eviction.
        budget:   Maximum total tokens kept (sinks + non-sinks).
        head_dim: Head dimension D (unused here; accepted for API symmetry
                  with H2O's init_h2o_state).
        rank_cap: Maximum SVD rank used when estimating leverage scores.

    Raises:
        ValueError: if there are sink positions to protect but they leave no
            evictable room within ``budget`` (``n_sink=0, budget=0`` remains
            a valid "disabled cache" configuration).
    """
    if n_sink > 0 and n_sink >= budget:
        raise ValueError(
            f"curdkv: n_sink ({n_sink}) must be < budget ({budget}) — no "
            "evictable positions remain, so sinks would be evicted once "
            "the cache fills"
        )
    return CurDKVState(
        keys=None,
        values=None,
        leverage_scores=None,
        n_updates=None,
        n_sink=n_sink,
        budget=budget,
        rank_cap=rank_cap,
    )


def _attention_weights(query_proxy: mx.array, keys: mx.array) -> mx.array:
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


def _leverage_scores(
    query_proxy: mx.array, keys: mx.array, values: mx.array, rank_cap: int
) -> mx.array:
    """Approximated leverage scores of the proxy attention-output contribution.

    Builds the proxy attention-weighted value block
    (``weighted_values[i] = attn[i] * values[i]``) — each row is that
    token's weighted contribution to the (proxy) attention output — then
    estimates row-leverage scores via an ENERGY-WEIGHTED sum over the top-
    ``rank_cap`` left singular vectors of that block:
    ``l_i = sum_j (s_j^2 / sum(s^2)) * U[i, j]^2``, normalized to sum to 1.
    This is the "approximated CUR" leverage-score stand-in described in the
    module docstring; it is not a full CUR row/column sampling routine.

    Weighting each singular direction's contribution by its own energy
    (``s_j^2``) — rather than a hard top-k/bottom-(n-k) split — is
    deliberate: a hard split degenerates whenever ``k`` reaches the block's
    rank, because the left singular vectors of a full-rank ``[n, k]`` block
    with ``k >= n`` form a complete orthogonal basis, and every row of an
    orthogonal matrix has unit norm by construction — erasing the very
    magnitude signal this estimator exists to capture, regardless of how
    small the tail singular values actually are. Energy-weighting keeps
    directions with near-zero singular value from contributing meaningfully
    to any row's score, without relying on a brittle rank cutoff.

    Args:
        query_proxy: [D] — proxy query (incoming key vector).
        keys:        [n, D] — existing key rows.
        values:      [n, D] — existing value rows.
        rank_cap:    Maximum number of leading singular directions considered.

    Returns:
        [n] non-negative leverage scores, summing to ~1 (or all-zero if the
        weighted-value block is degenerately all-zero).
    """
    attn = _attention_weights(query_proxy, keys)  # [n]
    weighted_values = values * attn[:, None]  # [n, D]

    n, d = weighted_values.shape

    wv_np = np.array(weighted_values.astype(mx.float32).tolist(), dtype=np.float64)
    if not np.any(wv_np):
        return mx.zeros((n,), dtype=mx.float32)

    # Full SVD is fine at these sizes (n, d are small per-head cache blocks).
    u, s, _ = np.linalg.svd(wv_np, full_matrices=False)
    k = max(1, min(rank_cap, n, d))
    u_k = u[:, :k]  # [n, k]
    s_k = s[:k]  # [k]

    energy = s_k * s_k
    energy_total = energy.sum()
    if energy_total <= 0:
        return mx.zeros((n,), dtype=mx.float32)
    weights = energy / energy_total  # [k], sums to 1

    scores = np.sum((u_k * u_k) * weights[None, :], axis=1)  # [n]

    total = scores.sum()
    if total > 0:
        scores = scores / total
    return mx.array(scores.astype(np.float32))


def curdkv_update(
    state: CurDKVState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
) -> CurDKVState:
    """Absorb S new tokens into state, evicting the lowest-leverage non-sink token if over budget.

    For each of the S incoming tokens:
      1. Compute approximated leverage scores of the new key (as proxy query)
         over all currently stored (key, value) rows — a joint key+value
         importance signal, unlike H2O's key-only attention-mass score.
      2. Accumulate those leverage scores into existing per-token scores.
      3. Append the new token, seeded with its OWN leverage score within the
         resulting (existing + new) block — not a flat 0. A flat-0 seed
         would let a token with a genuinely negligible value contribution
         tie forever with already-negligible survivors (leverage scores can
         legitimately be exactly 0, unlike H2O's softmax scores, which are
         never exactly 0), and index-based tie-breaking would then protect
         whichever of the two happens to sit at the lower index rather than
         evicting by actual value-relevance. Seeding with self-leverage lets
         a negligible-value newcomer be evicted immediately instead of
         parking at a permanent tie.
      4. If total tokens > budget: permanently evict the non-sink token with the
         lowest MEAN per-step leverage score (cumulative score / number of
         accumulation steps), not the raw cumulative sum. A raw-sum
         comparison is scale-biased against newcomers: an old survivor's
         score is a sum over every step it has been in the cache, while a
         brand-new token's score is a single self-leverage sample — so a
         newcomer with a far larger, more output-relevant value than
         anything currently cached could otherwise be evicted on the very
         step it arrives, purely because it hasn't had time to accumulate,
         defeating the point of value-aware retention (see GH issue for the
         regression case).

    Args:
        state:      Current CurDKVState for this head.
        new_keys:   [S, D] fp16 new key rows.
        new_values: [S, D] fp16 new value rows.

    Returns:
        Updated CurDKVState with at most ``state.budget`` tokens.
    """
    S = new_keys.shape[0]

    for i in range(S):
        k_i = new_keys[i]  # [D]
        v_i = new_values[i]  # [D]

        if state.keys is None:
            # Bootstrap: first token ever — no eviction needed.
            state = CurDKVState(
                keys=k_i[None].astype(mx.float16),
                values=v_i[None].astype(mx.float16),
                leverage_scores=mx.ones((1,), dtype=mx.float32),
                n_updates=mx.ones((1,), dtype=mx.float32),
                n_sink=state.n_sink,
                budget=state.budget,
                rank_cap=state.rank_cap,
            )
            continue

        # --- leverage-score update ------------------------------------
        lev = _leverage_scores(
            k_i.astype(mx.float32),
            state.keys.astype(mx.float32),
            state.values.astype(mx.float32),
            state.rank_cap,
        )
        updated_scores = state.leverage_scores + lev  # [n_kept]
        updated_n_updates = state.n_updates + 1.0  # [n_kept]

        # --- append new token, seeded with its own leverage within the
        # resulting (existing + new) block (see docstring: not a flat 0) ---
        keys_cat = mx.concatenate([state.keys, k_i[None].astype(mx.float16)], axis=0)
        values_cat = mx.concatenate([state.values, v_i[None].astype(mx.float16)], axis=0)
        self_lev = _leverage_scores(
            k_i.astype(mx.float32),
            keys_cat.astype(mx.float32),
            values_cat.astype(mx.float32),
            state.rank_cap,
        )[-1:]
        scores_cat = mx.concatenate([updated_scores, self_lev], axis=0)
        n_updates_cat = mx.concatenate([updated_n_updates, mx.ones((1,), dtype=mx.float32)], axis=0)

        n_total = keys_cat.shape[0]

        if n_total > state.budget:
            # Compare MEAN per-step leverage, not the raw cumulative sum —
            # an old survivor's sum grows with every step it has survived,
            # so comparing raw sums is biased against newcomers regardless
            # of their actual value (see docstring point 4 above).
            mean_scores = scores_cat / n_updates_cat

            # Build eviction-protected score view: sinks get +inf
            n_sink_eff = min(state.n_sink, n_total)
            if n_sink_eff > 0:
                inf_block = mx.full((n_sink_eff,), float("inf"), dtype=mx.float32)
                protected = mx.concatenate([inf_block, mean_scores[n_sink_eff:]], axis=0)
            else:
                protected = mean_scores

            evict_idx = int(mx.argmin(protected).item())
            keep_indices = [j for j in range(n_total) if j != evict_idx]
            keys_cat = keys_cat[keep_indices]
            values_cat = values_cat[keep_indices]
            scores_cat = scores_cat[keep_indices]
            n_updates_cat = n_updates_cat[keep_indices]

        state = CurDKVState(
            keys=keys_cat,
            values=values_cat,
            leverage_scores=scores_cat,
            n_updates=n_updates_cat,
            n_sink=state.n_sink,
            budget=state.budget,
            rank_cap=state.rank_cap,
        )

    return state


def curdkv_get_kv(state: CurDKVState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update.
    """
    if state.keys is None:
        dummy = mx.zeros((0, 1), dtype=mx.float16)
        return dummy, dummy
    return state.keys, state.values


def curdkv_fp16_bytes(state: CurDKVState) -> int:
    """Bytes currently stored for K + V in fp16."""
    if state.keys is None:
        return 0
    n, D = state.keys.shape
    return n * D * 2 * 2  # K + V, 2 bytes each


def full_curdkv_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return tokens_seen * head_dim * 2 * 2  # K + V, 2 bytes each


__all__ = [
    "CurDKVState",
    "init_curdkv_state",
    "curdkv_update",
    "curdkv_get_kv",
    "curdkv_fp16_bytes",
    "full_curdkv_fp16_bytes",
]
