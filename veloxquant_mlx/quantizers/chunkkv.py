"""ChunkKV-adapted eviction primitives — chunk-level (semantic-block) KV eviction.

Inspired by "ChunkKV: Semantic-Preserving KV Cache Compression for Efficient
Long-Context LLM Inference" (Liu et al., 2025, arXiv:2502.00299). Documented as
"ChunkKV-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

Every other eviction configuration in the repo (SnapKV, StreamingLLM, H2O, TOVA,
PyramidKV, SqueezeAttention) scores and evicts **individual tokens**. ChunkKV's
insight is that a token is not a self-contained unit of meaning: dropping the
lowest-scoring tokens shreds contiguous spans (a clause, a variable definition, a
table row) whose value is collective. ChunkKV instead partitions the sequence
into contiguous **chunks** of size ``C`` and evicts at chunk granularity — a chunk
is kept or dropped as a whole — so surviving context stays locally coherent.

This module holds two things:
  1. ``chunk_partition`` / ``chunk_scores`` / ``chunkkv_keep_mask`` — the pure,
     stateless chunk machinery: split a length into sink + body chunks, pool a
     per-token score vector into a per-chunk score, and turn a budget into a
     chunk-aligned boolean keep-mask over tokens.
  2. ``ChunkKVState`` + ``chunkkv_update`` — the per-head eviction. It reuses
     H2O's key-as-query cumulative-attention-mass scorer (the ``"attn_mass"``
     score mode) or a pooled key-L2-norm proxy (the ``"key_norm"`` mode), but
     when the cache exceeds the budget it evicts the lowest-scoring **chunk** of
     ``C`` contiguous non-sink tokens rather than a single token.

Relationship to H2O:
  When ``chunk_size == 1`` every chunk is a single token, chunk-pooling is the
  identity, and "evict the lowest-scoring chunk once over budget" is exactly
  "evict the lowest-scoring token once over budget" — so ChunkKV-adapted reduces
  **bit-for-bit** to H2O-adapted at ``C = 1``. This is the analogue of
  "``strength = 0`` == H2O" (SqueezeAttention) and "flat pyramid == H2O"
  (PyramidKV), and is asserted by a dedicated equivalence test.

Adaptation limitations (stated plainly):
  - Key-as-query proxy: like H2O-adapted / SnapKV-adapted, the incoming key
    vector stands in for the true query (not visible at cache level) when scoring.
  - Pooled-score proxy for the paper's chunk importance: the paper ranks chunks
    by observed attention over the chunk; we pool a per-token proxy score (mean)
    into a per-chunk score. Different signal, same chunk-granular decision.
  - No layer-wise index-reuse optimisation from the paper (a decode-speed trick
    that reuses one layer's kept-chunk indices at the next); each layer/head here
    resolves its own chunks independently.
  - Streaming eviction (a chunk is dropped as soon as the cache exceeds budget by
    a chunk) rather than a single one-shot prefill compression.
  - No RoPE position-ID remapping after eviction.
  - Uniform budget across heads within a layer.

Public API
----------
chunk_partition       — split (seq_len, chunk_size, n_sink) into sink + body chunk ranges
chunk_scores          — pool a per-token score vector into per-chunk scores (mean)
chunkkv_keep_mask     — chunk-aligned boolean keep-mask over tokens for a budget
ChunkKVState          — immutable per-head eviction state
init_chunkkv_state    — construct empty state for a layer's budget
chunkkv_update        — absorb S new tokens, evict lowest-score chunk if over budget
chunkkv_trim_to       — trim a state to a common length (keeps sinks + recent tail)
chunkkv_get_kv        — extract current (keys, values) arrays
chunkkv_fp16_bytes    — bytes stored in current state
full_chunkkv_fp16_bytes — hypothetical cost without eviction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


def chunk_partition(
    seq_len: int, chunk_size: int, n_sink: int
) -> tuple[list[int], list[tuple[int, int]]]:
    """Split ``[0, seq_len)`` into protected sink positions and body chunks.

    The first ``min(n_sink, seq_len)`` positions are sink positions (always kept,
    never grouped into an evictable chunk). The remaining positions are grouped
    into contiguous chunks of ``chunk_size``; the final chunk may be shorter (a
    "ragged tail").

    Args:
        seq_len:    Total number of token positions.
        chunk_size: Chunk width ``C`` (>= 1).
        n_sink:     Number of leading sink positions to protect.

    Returns:
        ``(sink_indices, body_chunks)`` where ``sink_indices`` is a list of the
        protected leading positions and ``body_chunks`` is a list of
        ``(start, stop)`` half-open ranges partitioning the non-sink tail. When
        ``chunk_size == 1`` every body chunk is a single position.

    Raises:
        ValueError: if ``chunk_size < 1``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_partition: chunk_size must be >= 1, got {chunk_size}.")

    n_sink_eff = min(max(n_sink, 0), seq_len)
    sink_indices = list(range(n_sink_eff))

    body_chunks: list[tuple[int, int]] = []
    start = n_sink_eff
    while start < seq_len:
        stop = min(start + chunk_size, seq_len)
        body_chunks.append((start, stop))
        start = stop
    return sink_indices, body_chunks


def chunk_scores(token_scores: mx.array, body_chunks: list[tuple[int, int]]) -> mx.array:
    """Pool a per-token score vector into one score per body chunk (mean).

    A chunk's score is the mean of its tokens' scores. Mean (not sum) is used so
    the ragged final chunk is not penalised for holding fewer tokens.

    Args:
        token_scores: ``[seq_len]`` per-token proxy scores (float).
        body_chunks:  ``(start, stop)`` ranges from :func:`chunk_partition`.

    Returns:
        ``[len(body_chunks)]`` float32 per-chunk scores. Empty if there are no
        body chunks.
    """
    if not body_chunks:
        return mx.zeros((0,), dtype=mx.float32)
    s = token_scores.astype(mx.float32)
    means = [mx.mean(s[a:b]) for (a, b) in body_chunks]
    return mx.stack(means, axis=0)


def chunkkv_keep_mask(
    token_scores: mx.array, seq_len: int, chunk_size: int, n_sink: int, budget: int
) -> mx.array:
    """Chunk-aligned boolean keep-mask over ``seq_len`` tokens for a budget.

    Sink positions are always kept. Body chunks are ranked by pooled score
    (:func:`chunk_scores`) and kept — whole — from highest score down until adding
    the next chunk would exceed ``budget``. Because chunks are kept whole, the
    number of retained tokens is the largest chunk-aligned count that does not
    exceed ``budget`` (sinks included); it may be strictly below ``budget`` when
    ``budget - n_sink`` is not a multiple of ``chunk_size``.

    Ties in chunk score are broken toward the **more recent** chunk (higher start
    index), matching the recency bias of the token-level methods.

    Args:
        token_scores: ``[seq_len]`` per-token proxy scores (higher = keep).
        seq_len:      Number of tokens the mask covers.
        chunk_size:   Chunk width ``C``.
        n_sink:       Leading sink positions (always kept).
        budget:       Maximum tokens to keep (sinks included).

    Returns:
        ``[seq_len]`` boolean ``mx.array``; ``True`` at kept positions.
    """
    keep = [False] * seq_len
    sink_indices, body_chunks = chunk_partition(seq_len, chunk_size, n_sink)
    for i in sink_indices:
        keep[i] = True

    remaining = budget - len(sink_indices)
    if remaining > 0 and body_chunks:
        scores = chunk_scores(token_scores, body_chunks)
        order = list(range(len(body_chunks)))
        # Highest score first; ties → later (higher start) chunk first (recency).
        order.sort(key=lambda c: (float(scores[c].item()), body_chunks[c][0]), reverse=True)
        for c in order:
            a, b = body_chunks[c]
            width = b - a
            if width <= remaining:
                for i in range(a, b):
                    keep[i] = True
                remaining -= width
    return mx.array(keep)


@dataclass
class ChunkKVState:
    """Per-head ChunkKV-adapted eviction state for one layer.

    Identical fields to H2OState — ChunkKV reuses H2O's cumulative-mass scorer;
    only the *unit of eviction* differs (a chunk of ``chunk_size`` tokens rather
    than one token) and, in ``"key_norm"`` mode, the score signal.

    Attributes:
        keys:       [n_kept, D] fp16 stored key rows, or None before first update.
        values:     [n_kept, D] fp16 stored value rows, or None before first update.
        scores:     [n_kept] proxy score per token (float32), or None. In
                    ``"attn_mass"`` mode this is cumulative softmax attention mass
                    (like H2O); in ``"key_norm"`` mode it is the token's key L2 norm.
        n_sink:     Number of leading sink positions — never evicted.
        budget:     Maximum tokens to keep at any time (including sinks).
        chunk_size: Eviction granularity ``C``. ``1`` reduces to H2O exactly.
        score_mode: ``"attn_mass"`` (default) or ``"key_norm"``.
    """

    keys: mx.array | None
    values: mx.array | None
    scores: mx.array | None
    n_sink: int
    budget: int
    chunk_size: int
    score_mode: str


def init_chunkkv_state(
    n_sink: int,
    budget: int,
    head_dim: int,  # noqa: ARG001
    chunk_size: int = 8,
    score_mode: str = "attn_mass",
) -> ChunkKVState:
    """Create an empty ChunkKVState before any tokens arrive.

    Args:
        n_sink:     Number of initial sink positions to protect from eviction.
        budget:     Maximum total tokens kept (sinks + non-sinks).
        head_dim:   Head dimension D (unused here; accepted for API symmetry).
        chunk_size: Eviction granularity ``C`` (>= 1). ``1`` reduces to H2O.
        score_mode: ``"attn_mass"`` (cumulative attention-mass proxy, like H2O) or
                    ``"key_norm"`` (mean key-L2-norm proxy).

    Raises:
        ValueError: if ``chunk_size < 1``, ``score_mode`` is unknown, or there
            are sink positions to protect but they leave no evictable room
            within ``budget`` (``n_sink=0, budget=0`` remains a valid
            "disabled cache" configuration).
    """
    if chunk_size < 1:
        raise ValueError(f"init_chunkkv_state: chunk_size must be >= 1, got {chunk_size}.")
    if score_mode not in ("attn_mass", "key_norm"):
        raise ValueError(
            f"init_chunkkv_state: score_mode must be 'attn_mass' or 'key_norm', got {score_mode!r}."
        )
    if n_sink > 0 and n_sink >= budget:
        raise ValueError(
            f"chunkkv: n_sink ({n_sink}) must be < budget ({budget}) — no "
            "evictable positions remain, so sinks would be evicted once "
            "the cache fills"
        )
    return ChunkKVState(
        keys=None,
        values=None,
        scores=None,
        n_sink=n_sink,
        budget=budget,
        chunk_size=int(chunk_size),
        score_mode=score_mode,
    )


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


def _lowest_scoring_chunk(scores: mx.array, n_sink_eff: int, chunk_size: int) -> list[int]:
    """Indices of the lowest-scoring evictable chunk of ``chunk_size`` tokens.

    The non-sink tail ``[n_sink_eff, n_total)`` is partitioned into contiguous
    chunks (the newest chunk may be ragged). The chunk with the lowest **mean**
    score is selected for eviction; ties break toward the *older* chunk (lower
    start index) so recent context is preferred. Sinks are never returned.

    Returns:
        Sorted list of token indices to evict (one chunk). Empty if the tail is
        empty.
    """
    n_total = int(scores.shape[0])
    _, body_chunks = chunk_partition(n_total, chunk_size, n_sink_eff)
    if not body_chunks:
        return []
    pooled = chunk_scores(scores, body_chunks)
    # argmin with older-chunk tie-break: lists are already start-ascending, and
    # mx.argmin returns the first minimum → the oldest lowest-scoring chunk.
    evict_chunk = int(mx.argmin(pooled).item())
    a, b = body_chunks[evict_chunk]
    return list(range(a, b))


def chunkkv_update(
    state: ChunkKVState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
) -> ChunkKVState:
    """Absorb S new tokens, evicting the lowest-score chunk if over budget.

    For each of the S incoming tokens:
      1. Update the per-token proxy scores (``"attn_mass"``: accumulate the new
         key's attention weight over stored keys, exactly like H2O; ``"key_norm"``:
         the score is simply the token's key L2 norm, fixed at insertion).
      2. Append the new token.
      3. While the cache exceeds ``budget``, evict the lowest-scoring **chunk** of
         up to ``chunk_size`` contiguous non-sink tokens (a single token when
         ``chunk_size == 1``). Evicting a whole chunk can drop the count below
         ``budget``; the loop stops as soon as the cache fits.

    At ``chunk_size == 1`` and ``score_mode == "attn_mass"`` this is identical to
    ``h2o_update`` (single-token eviction, cumulative-mass scoring, sink
    protection).

    Args:
        state:      Current ChunkKVState for this head.
        new_keys:   [S, D] fp16 new key rows.
        new_values: [S, D] fp16 new value rows.

    Returns:
        Updated ChunkKVState with at most ``state.budget`` tokens.
    """
    S = new_keys.shape[0]

    for i in range(S):
        k_i = new_keys[i]  # [D]
        v_i = new_values[i]  # [D]

        if state.keys is None:
            # Bootstrap: first token ever — no eviction needed.
            if state.score_mode == "key_norm":
                first_score = mx.sqrt(mx.sum(k_i.astype(mx.float32) ** 2))[None]
            else:
                first_score = mx.ones((1,), dtype=mx.float32)
            state = ChunkKVState(
                keys=k_i[None].astype(mx.float16),
                values=v_i[None].astype(mx.float16),
                scores=first_score.astype(mx.float32),
                n_sink=state.n_sink,
                budget=state.budget,
                chunk_size=state.chunk_size,
                score_mode=state.score_mode,
            )
            continue

        # --- score update --------------------------------------------------
        if state.score_mode == "key_norm":
            # Existing scores are fixed norms; the new token gets its own norm.
            updated_scores = state.scores
            new_score = mx.sqrt(mx.sum(k_i.astype(mx.float32) ** 2))[None]
        else:
            attn = _attention_scores(k_i.astype(mx.float32), state.keys.astype(mx.float32))
            updated_scores = state.scores + attn  # [n_kept]
            new_score = mx.zeros((1,), dtype=mx.float32)

        # --- append new token ---------------------------------------------
        keys_cat = mx.concatenate([state.keys, k_i[None].astype(mx.float16)], axis=0)
        values_cat = mx.concatenate([state.values, v_i[None].astype(mx.float16)], axis=0)
        scores_cat = mx.concatenate([updated_scores, new_score], axis=0)

        n_total = keys_cat.shape[0]

        # --- chunk-aligned eviction while over budget ----------------------
        while keys_cat.shape[0] > state.budget:
            n_now = keys_cat.shape[0]
            n_sink_eff = min(state.n_sink, n_now)
            evict = _lowest_scoring_chunk(scores_cat, n_sink_eff, state.chunk_size)
            if not evict:
                break  # nothing evictable (all sinks) — cannot shrink further
            evict_set = set(evict)
            keep_indices = [j for j in range(n_now) if j not in evict_set]
            keys_cat = keys_cat[keep_indices]
            values_cat = values_cat[keep_indices]
            scores_cat = scores_cat[keep_indices]

        state = ChunkKVState(
            keys=keys_cat,
            values=values_cat,
            scores=scores_cat,
            n_sink=state.n_sink,
            budget=state.budget,
            chunk_size=state.chunk_size,
            score_mode=state.score_mode,
        )

    return state


def chunkkv_trim_to(state: ChunkKVState, n: int) -> ChunkKVState:
    """Trim a state to at most ``n`` tokens, keeping sinks + the most recent tail.

    Whole-chunk retention lets different heads settle at slightly different token
    counts; the cache wrapper trims every head to the common minimum so the
    emitted tensor is rectangular. Sinks are always retained; beyond them the
    **most recent** ``n - n_sink`` non-sink tokens are kept (recency preference,
    consistent with the eviction tie-break). A no-op when the state already holds
    ``<= n`` tokens — so at ``chunk_size == 1`` (all heads at exactly ``budget``)
    nothing is trimmed and the H2O equivalence is preserved.

    Args:
        state: State to trim.
        n:     Target maximum token count (>= 0).

    Returns:
        A trimmed ChunkKVState (or ``state`` unchanged if already within ``n``).
    """
    if state.keys is None:
        return state
    n_total = int(state.keys.shape[0])
    if n_total <= n:
        return state

    n_sink_eff = min(state.n_sink, n_total, n)
    n_recent = n - n_sink_eff
    tail_start = n_total - n_recent if n_recent > 0 else n_total
    keep_indices = list(range(n_sink_eff)) + list(range(tail_start, n_total))
    return ChunkKVState(
        keys=state.keys[keep_indices],
        values=state.values[keep_indices],
        scores=state.scores[keep_indices],
        n_sink=state.n_sink,
        budget=state.budget,
        chunk_size=state.chunk_size,
        score_mode=state.score_mode,
    )


def chunkkv_get_kv(state: ChunkKVState) -> tuple[mx.array, mx.array]:
    """Return ``(keys, values)`` arrays from state.

    Returns ``([0, 1], [0, 1])`` zero-row placeholders before the first update.
    """
    if state.keys is None:
        dummy = mx.zeros((0, 1), dtype=mx.float16)
        return dummy, dummy
    return state.keys, state.values


def chunkkv_fp16_bytes(state: ChunkKVState) -> int:
    """Bytes currently stored for K + V in fp16."""
    if state.keys is None:
        return 0
    n, D = state.keys.shape
    return n * D * 2 * 2  # K + V, 2 bytes each


def full_chunkkv_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return tokens_seen * head_dim * 2 * 2  # K + V, 2 bytes each


__all__ = [
    "chunk_partition",
    "chunk_scores",
    "chunkkv_keep_mask",
    "ChunkKVState",
    "init_chunkkv_state",
    "chunkkv_update",
    "chunkkv_trim_to",
    "chunkkv_get_kv",
    "chunkkv_fp16_bytes",
    "full_chunkkv_fp16_bytes",
]
