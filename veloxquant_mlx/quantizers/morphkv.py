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

from veloxquant_mlx.quantizers._eviction_common import (
    attention_scores,
    fp16_kv_bytes,
    full_fp16_kv_bytes,
    get_kv,
)


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
        acc = acc + attention_scores(recent_keys[j].astype(mx.float32), keys_f)
    return acc / float(w)


def _recent_relevance_batched(keys: mx.array, w_eff: int) -> mx.array:
    """Batched-``[BH,n,D]`` equivalent of :func:`_recent_relevance`.

    Replaces the ``for j in range(w): acc += attention_scores(...)`` window
    loop (one dot-product-based score per recent-window position, computed
    sequentially against the same ``keys``) with a single batched matmul
    over the window axis — same recipe as PyramidKV's ``pyramid_update_heads``
    fix (#549): ``keys_f @ recent_keys.T`` computes every window position's
    logits against every stored key in one call, softmax per probe row, then
    mean-reduce over the window axis instead of accumulating ``w`` separate
    calls.

    Args:
        keys:  ``[BH, n, D]`` stored key rows (the keep-set candidates). The
               trailing ``w_eff`` rows of each are also this step's recent
               window (mirrors ``morphkv_update``'s ``recent = keys_cat[n -
               w_eff:]``).
        w_eff: Window size (``min(window, n)``), shared across all ``BH``
               rows (true for every real caller: one uniform
               ``morphkv_window`` per layer).

    Returns:
        ``[BH, n]`` mean recent-window attention mass per stored key.
    """
    keys_f = keys.astype(mx.float32)
    n = int(keys_f.shape[1])
    recent = keys_f[:, n - w_eff :]  # [BH, w_eff, D]
    scale = 1.0 / math.sqrt(float(keys_f.shape[-1]))
    logits = (recent @ mx.swapaxes(keys_f, -1, -2)) * scale  # [BH, w_eff, n]
    attn = mx.softmax(logits, axis=-1)  # each recent-window row sums to ~1
    return mx.mean(attn, axis=1)  # [BH, n]


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


_EVAL_FLUSH_INTERVAL = 32


def morphkv_update_batched(
    keys: mx.array | None,  # [BH, n, D] fp16 or None
    values: mx.array | None,  # [BH, n, D] fp16 or None
    new_keys: mx.array,  # [BH, S, D]
    new_values: mx.array,  # [BH, S, D]
    n_sink: int,
    budget: int,
    window: int,
    *,
    return_indices: bool = False,
) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, mx.array]:
    """Vectorized-over-``BH`` equivalent of calling :func:`morphkv_update`
    once per ``(batch, head)`` pair with identical per-row ``n_sink``/
    ``budget``/``window`` (true for every real caller: :class:`MorphKVKVCache`
    applies one uniform config to every head), so the per-token append/rank/
    evict math — otherwise identical for every row — can run as one batched
    MLX call per step instead of ``BH`` separate Python-level calls into
    :func:`morphkv_update`. Same fix, same template, as
    :func:`veloxquant_mlx.quantizers.h2o.h2o_update_batched`. Uses
    :func:`_recent_relevance_batched` internally, which also removes the
    inner ``for j in range(window):`` loop :func:`_recent_relevance` runs
    per call — issue #560's second, compounding finding.

    Numerically identical to the per-head loop it replaces: every op below
    is the same formula as :func:`morphkv_update`'s bootstrap/append/evict
    branches, applied over a leading ``BH`` axis instead of a Python loop.

    With ``return_indices=True``, also return indices into prior + new rows.

    Returns:
        ``(keys, values)`` — ``[BH, n_kept, D]`` each.
    """
    bh, s, d = new_keys.shape
    n_prior = 0 if keys is None else keys.shape[1]
    indices = mx.broadcast_to(mx.arange(n_prior)[None], (bh, n_prior))
    if s == 0:
        return (keys, values, indices) if return_indices else (keys, values)
    if n_sink + window >= budget:
        raise ValueError(
            f"morphkv: n_sink ({n_sink}) + window ({window}) must be < "
            f"budget ({budget}) — no evictable positions remain"
        )

    for i in range(s):
        if return_indices:
            indices = mx.concatenate([indices, mx.full((bh, 1), n_prior + i)], axis=1)
        if keys is None:
            keys = new_keys[:, i : i + 1].astype(mx.float16)  # [BH, 1, D]
            values = new_values[:, i : i + 1].astype(mx.float16)
            continue

        keys_cat = mx.concatenate([keys, new_keys[:, i : i + 1].astype(mx.float16)], axis=1)
        values_cat = mx.concatenate([values, new_values[:, i : i + 1].astype(mx.float16)], axis=1)

        n_total = keys_cat.shape[1]
        if n_total > budget:
            w_eff = min(window, n_total)
            relevance = _recent_relevance_batched(keys_cat, w_eff)  # [BH, n_total]

            n_sink_eff = min(n_sink, n_total)
            protect = mx.zeros((bh, n_total), dtype=mx.float32)
            if n_sink_eff > 0:
                sink_inf = mx.full((bh, n_sink_eff), float("inf"), dtype=mx.float32)
                protect = mx.concatenate([sink_inf, protect[:, n_sink_eff:]], axis=1)
            # Trailing recent window always protected (it drives the ranking).
            window_inf = mx.full((bh, w_eff), float("inf"), dtype=mx.float32)
            protect = mx.concatenate([protect[:, : n_total - w_eff], window_inf], axis=1)
            sel = relevance + protect

            evict_idx = mx.argmin(sel, axis=-1, keepdims=True)  # [BH, 1]
            rows = mx.arange(n_total - 1)[None]  # [1, n_total-1]
            source = rows + (rows >= evict_idx)  # [BH, n_total-1]

            keys_cat = mx.take_along_axis(keys_cat, source[..., None], axis=1)
            values_cat = mx.take_along_axis(values_cat, source[..., None], axis=1)
            if return_indices:
                indices = mx.take_along_axis(indices, source, axis=1)

        keys, values = keys_cat, values_cat

        if (i + 1) % _EVAL_FLUSH_INTERVAL == 0:
            mx.eval(keys, values, indices)

    return (keys, values, indices) if return_indices else (keys, values)


def morphkv_get_kv(state: MorphKVState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update
    (same contract as ``keyformer_get_kv`` / ``tova_get_kv``).
    """
    return get_kv(state.keys, state.values)


def morphkv_fp16_bytes(state: MorphKVState) -> int:
    """Bytes currently stored for K + V in fp16.

    The recent-window ring is a view into ``keys`` (not extra payload), so only
    K + V are counted — same accounting as H2O / TOVA / Keyformer.
    """
    return fp16_kv_bytes(state.keys)


def full_morphkv_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return full_fp16_kv_bytes(tokens_seen, head_dim)


__all__ = [
    "MorphKVState",
    "init_morphkv_state",
    "morphkv_update",
    "morphkv_update_batched",
    "morphkv_get_kv",
    "morphkv_fp16_bytes",
    "full_morphkv_fp16_bytes",
]
