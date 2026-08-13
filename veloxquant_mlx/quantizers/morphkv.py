"""MorphKV KV eviction primitives — recent-window correlation retention.

Inspired by "Dialogue Without Limits: Constant-Sized KV Caches for Extended
Responses in LLMs" (Ghadia, Kumar, Jain, Nair, Das, ICML 2025,
arXiv:2503.00979). Documented as "MorphKV-adapted (VeloxQuant-MLX
implementation)" — not a faithful port. See adaptation limitations below and in
cache/morphkv_cache.py.

The paper's finding: retaining KV by a *cumulative* attention score (H2O-style)
suffers "early-token bias" — tokens that were heavy hitters early dominate the
keep set and crowd out context the model is *currently* attending to. MorphKV
keeps a constant-size cache by ranking stored tokens according to their
correlation with the attention pattern of a **sliding window of recent tokens**,
so retention tracks what the recent context actually reads and older-but-stale
tokens are dropped.

WHERE THIS SITS IN THE REPO
---------------------------
This is the repo's proxy-attention scorer family (SnapKV / H2O / TOVA /
PyramidKV / SqueezeAttention / ChunkKV / CaM / Keyformer). The distinguishing
axis: every existing scorer ranks a stored token against **either** cumulative
history (H2O accumulates mass forever) **or** a single most-recent query (TOVA /
SnapKV use the latest position). MorphKV-adapted is the first to rank by
correlation with a *window* of the last ``window`` tokens. Two honest reference
behaviors bracket it:

  - ``window = 1`` collapses onto a **latest-token (TOVA-adapted-style)**
    eviction: the recent-relevance signal is just the newest key's attention
    over the keep set. This is the clean, assertable reduction (exercised by a
    dedicated test), the analogue of Keyformer's ``tau = 0`` == H2O collapse.
  - a large window approaches averaging over recent context; it never becomes
    H2O's *cumulative-forever* rule (MorphKV recomputes from the live window,
    it does not accumulate), so we do NOT claim an H2O collapse — only the
    ``window = 1`` reduction is pinned exactly.

THE HONESTY CRUX (read before trusting any number)
--------------------------------------------------
1. **Proxy query.** Like H2O / TOVA / SnapKV / Keyformer-adapted, a cache never
   sees the true query vector, so incoming KEYS are used as proxy queries to
   estimate the attention each stored key receives. The paper uses the model's
   real attention patterns. Documented substitution, not the paper's math.
2. **Constant-size, recomputed — not accumulated.** We keep no cumulative score
   array. Each step, retention is recomputed from the current keep set and a
   ring buffer of the last ``window`` key rows. That is the mechanism: the cache
   is a fixed budget refreshed against recent context, not a growing accumulator.
3. Nothing here is validated on a trained model. The paper's headline numbers
   (accuracy / memory savings) are the paper's, on trained models — NEVER quoted
   as ours. The mechanism's benefit is measured only under a constructed
   "topic-shift" geometry in the benchmark, with a null control where it shows
   no advantage.

Adaptation limitations (stated plainly):
  - Key-as-query proxy (crux 1).
  - No RoPE position-ID remapping after eviction.
  - Uniform budget / n_sink / window across all heads.
  - The trailing ``window`` tokens are protected from eviction (they are the
    recency context that drives the ranking); leading ``n_sink`` tokens are
    protected as sinks.

Public API (mirrors quantizers/keyformer.py)
--------------------------------------------
MorphKVState        — immutable per-head state dataclass
init_morphkv_state  — construct empty state (validates guards)
morphkv_update      — absorb S new tokens, evict least recent-relevant if over budget
morphkv_get_kv      — extract current (keys, values) arrays
morphkv_fp16_bytes  — bytes stored in current state
full_morphkv_fp16_bytes — hypothetical cost without eviction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class MorphKVState:
    """Per-head MorphKV eviction state.

    Attributes:
        keys:   [n_kept, D] fp16 stored key rows, or None before first update.
        values: [n_kept, D] fp16 stored value rows, or None before first update.
        pos:    Running count of token positions this head has ever inserted
                (diagnostic; retention itself is stateless-of-history — it is
                recomputed from ``keys`` and the recent window each step).
        n_sink: Number of leading sink positions — never evicted.
        budget: Maximum tokens kept at any time (including sinks).
        window: Size of the trailing recent-token window whose aggregate
                proxy-attention drives retention (>= 1). The last ``window``
                stored tokens are themselves protected from eviction.
        head_dim: Head dimension D (for byte accounting before first insert).
    """

    keys: mx.array | None
    values: mx.array | None
    pos: int
    n_sink: int
    budget: int
    window: int
    head_dim: int


def init_morphkv_state(
    n_sink: int,
    budget: int,
    head_dim: int,
    window: int = 8,
) -> MorphKVState:
    """Create an empty MorphKVState before any tokens arrive.

    Raises:
        ValueError: if ``budget < 1``, ``window < 1``, ``n_sink >= budget``, or
            ``window > budget`` (the recent window cannot exceed the cache), or
            if sinks + window leave no evictable room.
    """
    if budget < 1:
        raise ValueError(f"morphkv: budget must be >= 1, got {budget!r}")
    if window < 1:
        raise ValueError(f"morphkv: window must be >= 1, got {window!r}")
    if n_sink >= budget:
        raise ValueError(f"morphkv: n_sink ({n_sink}) must be < budget ({budget})")
    if window > budget:
        raise ValueError(f"morphkv: window ({window}) must be <= budget ({budget})")
    if n_sink + window >= budget:
        raise ValueError(
            f"morphkv: n_sink ({n_sink}) + window ({window}) must be < "
            f"budget ({budget}) — no evictable positions remain"
        )
    return MorphKVState(
        keys=None,
        values=None,
        pos=0,
        n_sink=n_sink,
        budget=budget,
        window=window,
        head_dim=int(head_dim),
    )


def _attention_scores(query_proxy: mx.array, keys: mx.array) -> mx.array:
    """Softmax proxy-attention weights of ``query_proxy`` against each key row.

    Identical formula to quantizers/keyformer.py / tova.py — softmax of the
    key-as-query dot products, scaled by 1/sqrt(D).

    Args:
        query_proxy: [D] incoming key used as a stand-in for the true query.
        keys:        [n, D] existing key rows.

    Returns:
        [n] softmax weights summing to ~1.
    """
    scale = 1.0 / math.sqrt(float(query_proxy.shape[-1]))
    logits = (keys @ query_proxy) * scale  # [n]
    return mx.softmax(logits, axis=-1)


def _recent_relevance(keys: mx.array, recent_keys: mx.array) -> mx.array:
    """Aggregate proxy-attention each stored key receives from the recent window.

    For each of the last ``window`` key rows (used as proxy queries), compute the
    softmax attention it places over ALL stored ``keys``, then average across the
    window. This is the MorphKV signal: "how much does the recent context attend
    to this stored token." Higher = more recent-relevant = keep.

    With a single recent key (``window == 1``) this reduces to that one key's
    attention distribution over the keep set — the TOVA-adapted-style latest-token
    ranking, which is the pinned reduction.

    Args:
        keys:        [n, D] stored key rows (the keep-set candidates).
        recent_keys: [w, D] the last ``w`` (<= window) key rows.

    Returns:
        [n] mean recent-window attention mass per stored key.
    """
    keys_f = keys.astype(mx.float32)
    acc = mx.zeros((keys_f.shape[0],), dtype=mx.float32)
    w = int(recent_keys.shape[0])
    for j in range(w):
        acc = acc + _attention_scores(recent_keys[j].astype(mx.float32), keys_f)
    return acc / float(w)


def morphkv_update(
    state: MorphKVState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
) -> MorphKVState:
    """Absorb S new tokens, evicting the least recent-relevant token if over budget.

    For each of the S incoming tokens:
      1. Append the new token to the cache.
      2. If over budget: rank the current keep set by recent-window relevance
         (:func:`_recent_relevance` over the last ``window`` stored keys), force
         the leading ``n_sink`` sinks and the trailing ``window`` recent tokens
         to survive (+inf), and evict the non-protected token with the LOWEST
         recent-relevance. Constant-size: ``n_kept <= budget`` after every token.

    No cumulative score is carried across steps — the ranking is recomputed fresh
    each step from the live keep set and recent window (that is the mechanism).
    Kept tokens are returned in original temporal order.
    """
    S = int(new_keys.shape[0])

    for i in range(S):
        k_i = new_keys[i].astype(mx.float16)  # [D]
        v_i = new_values[i].astype(mx.float16)  # [D]

        if state.keys is None:
            state = MorphKVState(
                keys=k_i[None],
                values=v_i[None],
                pos=state.pos + 1,
                n_sink=state.n_sink,
                budget=state.budget,
                window=state.window,
                head_dim=state.head_dim,
            )
            continue

        # --- append new token ---------------------------------------------
        keys_cat = mx.concatenate([state.keys, k_i[None]], axis=0)
        values_cat = mx.concatenate([state.values, v_i[None]], axis=0)

        n_total = int(keys_cat.shape[0])

        if n_total > state.budget:
            # Recent window = last min(window, n_total) key rows.
            w_eff = min(state.window, n_total)
            recent = keys_cat[n_total - w_eff :]
            relevance = _recent_relevance(keys_cat, recent)  # [n_total]

            # Protect sinks (leading) and the recent window (trailing) with +inf.
            n_sink_eff = min(state.n_sink, n_total)
            protect = mx.zeros((n_total,), dtype=mx.float32)
            if n_sink_eff > 0:
                protect[:n_sink_eff] = float("inf")
            # Trailing recent window always protected (it drives the ranking).
            protect[n_total - w_eff :] = float("inf")
            sel = relevance + protect

            evict_idx = int(mx.argmin(sel).item())
            keep = [j for j in range(n_total) if j != evict_idx]
            keys_cat = keys_cat[keep]
            values_cat = values_cat[keep]

        state = MorphKVState(
            keys=keys_cat,
            values=values_cat,
            pos=state.pos + 1,
            n_sink=state.n_sink,
            budget=state.budget,
            window=state.window,
            head_dim=state.head_dim,
        )

    return state


def morphkv_get_kv(state: MorphKVState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update
    (same contract as ``keyformer_get_kv`` / ``tova_get_kv``).
    """
    if state.keys is None:
        dummy = mx.zeros((0, 1), dtype=mx.float16)
        return dummy, dummy
    return state.keys, state.values


def morphkv_fp16_bytes(state: MorphKVState) -> int:
    """Bytes currently stored for K + V in fp16.

    The recent-window ring is a view into ``keys`` (not extra payload), so only
    K + V are counted — same accounting as H2O / TOVA / Keyformer.
    """
    if state.keys is None:
        return 0
    n, D = state.keys.shape
    return n * D * 2 * 2  # K + V, 2 bytes each


def full_morphkv_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return tokens_seen * head_dim * 2 * 2


__all__ = [
    "MorphKVState",
    "init_morphkv_state",
    "morphkv_update",
    "morphkv_get_kv",
    "morphkv_fp16_bytes",
    "full_morphkv_fp16_bytes",
]
