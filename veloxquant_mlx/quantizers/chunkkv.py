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
  - Streaming eviction (a chunk is dropped as soon as the cache exceeds budget by
    a chunk) rather than a single one-shot prefill compression.
  - No RoPE position-ID remapping after eviction.
  - Uniform budget across heads within a layer.

Layer-wise index reuse (Algorithm 2 of the paper) IS implemented — see
``chunkkv_apply_reuse_indices`` below and ``ChunkKVIndexReuseCoordinator`` in
``cache/chunkkv_cache.py``. The paper observes that ChunkKV's kept-chunk indices
are far more similar between adjacent layers than token-level methods' (Table 2:
44-58% Jaccard similarity vs 15-28%), so a "leader" layer's indices can be reused
by the next ``Nreuse - 1`` layers instead of each running its own eviction,
cutting compression overhead (the paper reports 20.7% latency / 26.5% throughput
improvement) at a small (<0.6%) task-performance cost.

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
chunkkv_apply_reuse_indices — absorb S new tokens using a leader layer's kept-token
                              positions instead of running independent eviction
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, overload

import mlx.core as mx

from veloxquant_mlx.quantizers._eviction_common import (
    attention_scores,
    fp16_kv_bytes,
    full_fp16_kv_bytes,
    get_kv,
)


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

    # chunk_partition guarantees every body chunk is exactly `chunk_size`
    # wide except possibly the last (ragged tail). Exploit that: reshape the
    # uniform-width prefix into [n_full, chunk_size] and mean-reduce it in
    # one vectorized op instead of one Python-level mx.mean() call per
    # chunk — this is the pooling step inside the per-token eviction hot
    # loop (_lowest_scoring_chunk), so cutting it from O(n_chunks) graph
    # nodes to O(1) matters at small chunk_size / long context.
    widths = [b - a for (a, b) in body_chunks]
    chunk_size = widths[0] if widths else 0
    n_full = len(body_chunks) - (1 if widths and widths[-1] != chunk_size else 0)

    if n_full == len(body_chunks):
        # No ragged tail — the whole thing reshapes cleanly.
        start = body_chunks[0][0]
        stop = body_chunks[-1][1]
        return mx.mean(s[start:stop].reshape(len(body_chunks), chunk_size), axis=1)

    if n_full > 0:
        start = body_chunks[0][0]
        full_stop = body_chunks[n_full - 1][1]
        full_means = mx.mean(s[start:full_stop].reshape(n_full, chunk_size), axis=1)
        tail_a, tail_b = body_chunks[-1]
        tail_mean = mx.mean(s[tail_a:tail_b])[None]
        return mx.concatenate([full_means, tail_mean], axis=0)

    # Single (ragged) chunk only.
    a, b = body_chunks[0]
    return mx.mean(s[a:b])[None]


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
        # Materialize all chunk scores in one sync (`.tolist()`) instead of
        # calling `.item()` once per chunk inside the sort key below — same
        # values, one GPU->CPU round trip instead of len(body_chunks).
        scores_list = scores.tolist()
        order = list(range(len(body_chunks)))
        # Highest score first; ties → later (higher start) chunk first (recency).
        order.sort(key=lambda c: (scores_list[c], body_chunks[c][0]), reverse=True)
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
            within ``budget``, or ``n_sink=0`` with ``0 < budget <
            chunk_size`` (eviction would drop every row);
            ``n_sink=0, budget=0`` remains a valid "disabled cache"
            configuration).
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
    if n_sink == 0 and 0 < budget < chunk_size:
        raise ValueError(
            f"chunkkv: chunk_size ({chunk_size}) must be <= budget ({budget}) when "
            "n_sink=0 — an eviction would drop every row and empty the cache"
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


@overload
def chunkkv_update(
    state: ChunkKVState,
    new_keys: mx.array,
    new_values: mx.array,
    record_kept_positions: Literal[False] = False,
) -> ChunkKVState:
    """Overload signature: without position recording, returns only the updated state."""
    ...


@overload
def chunkkv_update(
    state: ChunkKVState,
    new_keys: mx.array,
    new_values: mx.array,
    record_kept_positions: Literal[True],
) -> tuple[ChunkKVState, list[list[int]]]:
    """Overload signature: with position recording, also returns kept-position lists."""
    ...


def chunkkv_update(
    state: ChunkKVState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
    record_kept_positions: bool = False,
) -> ChunkKVState | tuple[ChunkKVState, list[list[int]]]:
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
        record_kept_positions: If True, also return the list of kept-index sets
            (one per absorbed token, relative to that token's pre-eviction
            concatenated sequence) — used by the layer-wise index-reuse
            coordinator to publish a leader layer's eviction decisions.

    Returns:
        Updated ``ChunkKVState`` with at most ``state.budget`` tokens, or, when
        ``record_kept_positions=True``, ``(state, kept_positions)`` where
        ``kept_positions`` has one entry per input token: the sorted list of
        indices retained after that token's append-and-evict step.
    """
    S = new_keys.shape[0]
    kept_positions: list[list[int]] = []

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
            kept_positions.append([0])
            continue

        # --- score update --------------------------------------------------
        if state.score_mode == "key_norm":
            # Existing scores are fixed norms; the new token gets its own norm.
            updated_scores = state.scores
            new_score = mx.sqrt(mx.sum(k_i.astype(mx.float32) ** 2))[None]
        else:
            attn = attention_scores(k_i.astype(mx.float32), state.keys.astype(mx.float32))
            updated_scores = state.scores + attn  # [n_kept]
            new_score = mx.zeros((1,), dtype=mx.float32)

        # --- append new token ---------------------------------------------
        keys_cat = mx.concatenate([state.keys, k_i[None].astype(mx.float16)], axis=0)
        values_cat = mx.concatenate([state.values, v_i[None].astype(mx.float16)], axis=0)
        scores_cat = mx.concatenate([updated_scores, new_score], axis=0)

        n_total = keys_cat.shape[0]
        surviving = list(range(n_total))  # tracks positions in the pre-evict frame

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
            surviving = [surviving[j] for j in keep_indices]

        kept_positions.append(surviving)
        state = ChunkKVState(
            keys=keys_cat,
            values=values_cat,
            scores=scores_cat,
            n_sink=state.n_sink,
            budget=state.budget,
            chunk_size=state.chunk_size,
            score_mode=state.score_mode,
        )

    if record_kept_positions:
        return state, kept_positions
    return state


def chunkkv_apply_reuse_indices(
    state: ChunkKVState,
    new_keys: mx.array,  # [S, D] fp16
    new_values: mx.array,  # [S, D] fp16
    kept_positions: list[list[int]],
) -> ChunkKVState:
    """Absorb S new tokens by reusing a leader layer's kept-index decisions.

    Implements the follower side of the paper's Algorithm 2 (layer-wise index
    reuse): rather than scoring and evicting independently, each incoming token
    is appended and then the exact index set the leader layer kept at that step
    (``kept_positions[i]``, indices into the pre-eviction concatenated sequence)
    is applied here too. This assumes the follower's pre-step token count matches
    the leader's — true when both run the same absorption schedule under the same
    ``chunkkv_reuse_layers`` grouping.

    No score bookkeeping is needed: only the leader's scores drive eviction
    decisions, so this state's ``scores`` field is not meaningfully maintained
    (kept at a length-matching placeholder) — followers never evict on their own.

    Args:
        state:          Current ChunkKVState for this (follower) head.
        new_keys:       [S, D] fp16 new key rows.
        new_values:     [S, D] fp16 new value rows.
        kept_positions: One entry per input token — the leader's surviving-index
            list for that step, as returned by
            ``chunkkv_update(..., record_kept_positions=True)``.

    Returns:
        Updated ChunkKVState mirroring the leader's kept-token positions.

    Raises:
        ValueError: if ``len(kept_positions) != new_keys.shape[0]``.
    """
    S = new_keys.shape[0]
    if len(kept_positions) != S:
        raise ValueError(
            f"chunkkv_apply_reuse_indices: kept_positions has {len(kept_positions)} "
            f"entries but {S} tokens were supplied."
        )

    for i in range(S):
        k_i = new_keys[i]
        v_i = new_values[i]

        if state.keys is None:
            state = ChunkKVState(
                keys=k_i[None].astype(mx.float16),
                values=v_i[None].astype(mx.float16),
                scores=mx.ones((1,), dtype=mx.float32),
                n_sink=state.n_sink,
                budget=state.budget,
                chunk_size=state.chunk_size,
                score_mode=state.score_mode,
            )
            continue

        keys_cat = mx.concatenate([state.keys, k_i[None].astype(mx.float16)], axis=0)
        values_cat = mx.concatenate([state.values, v_i[None].astype(mx.float16)], axis=0)
        scores_cat = mx.concatenate([state.scores, mx.zeros((1,), dtype=mx.float32)], axis=0)

        keep = kept_positions[i]
        state = ChunkKVState(
            keys=keys_cat[keep],
            values=values_cat[keep],
            scores=scores_cat[keep],
            n_sink=state.n_sink,
            budget=state.budget,
            chunk_size=state.chunk_size,
            score_mode=state.score_mode,
        )

    return state


# How often (in loop iterations) the batched update forces graph
# materialization. Same rationale as h2o.py's identically-named constant:
# without this, a long prefill queues one eviction's worth of unevaluated
# graph nodes per token, risking MLX's Metal resource/command-buffer
# tracking limit before generation finishes.
_EVAL_FLUSH_INTERVAL = 32


def _attention_scores_batched_masked(
    query_proxy: mx.array, keys: mx.array, valid_mask: mx.array
) -> mx.array:
    """Softmax attention weights, batched over ``[BH]``, over a padded ``keys``
    buffer where only ``valid_mask`` columns hold real (non-padding) rows.

    Args:
        query_proxy: ``[BH, D]``.
        keys:        ``[BH, n, D]`` (columns beyond a row's true length are
            padding garbage).
        valid_mask:  ``[BH, n]`` bool — True at real (non-padding) columns.

    Returns:
        ``[BH, n]`` softmax weights; padding columns get exactly 0 (masked
        to ``-inf`` before the softmax).
    """
    scale = 1.0 / math.sqrt(float(query_proxy.shape[-1]))
    logits = (keys @ query_proxy[..., None])[..., 0] * scale  # [BH, n]
    logits = mx.where(valid_mask, logits, mx.array(float("-inf"), dtype=logits.dtype))
    return mx.softmax(logits, axis=-1)


def _evict_lowest_chunk_batched(
    scores: mx.array,  # [BH, n] fp32, padding entries anything (masked out below)
    lengths: mx.array,  # [BH] int32 -- valid length per row, post-append
    n_sink: int,
    chunk_size: int,
    over_budget: mx.array,  # [BH] bool -- rows that actually need eviction this step
) -> mx.array:
    """Per-row lowest-scoring-chunk eviction mask, batched over ``[BH]``.

    Batched-``[BH]`` equivalent of calling :func:`_lowest_scoring_chunk` once
    per row and turning its returned index list into a boolean mask — rows
    with different ``lengths`` (already-ragged from earlier steps within the
    same absorption loop) get independently correct chunk boundaries via
    length-relative masking, since ``n_sink``/``chunk_size`` are uniform
    across rows (true for every real caller — one cache config per layer)
    but each row's own valid length is not.

    Chunk pooling uses a ``[BH, n, n_chunks_max]`` one-hot membership matmul
    instead of Python-level per-chunk ``mx.mean`` calls (mirrors
    :func:`chunk_scores`'s existing full-batch-of-one vectorization, extended
    over the leading ``BH`` axis). ``mx.argmin`` returns the first minimum
    along an axis, i.e. the lowest chunk id — the oldest tied chunk — matching
    :func:`_lowest_scoring_chunk`'s documented older-chunk tie-break exactly.

    Returns:
        ``[BH, n]`` bool mask — True at positions to evict this step (empty,
        all-False, for rows where ``over_budget`` is False).
    """
    bh, n = scores.shape
    pos = mx.broadcast_to(mx.arange(n)[None, :], (bh, n))
    valid = pos < lengths[:, None]
    is_sink = pos < n_sink
    in_body = valid & ~is_sink
    chunk_id = mx.where(in_body, (pos - n_sink) // chunk_size, -1)

    n_chunks_max = max((n - n_sink + chunk_size - 1) // chunk_size, 1) if n > n_sink else 1
    onehot = (chunk_id[..., None] == mx.arange(n_chunks_max)[None, None, :]).astype(mx.float32)
    counts = mx.sum(onehot, axis=1)  # [BH, C]
    sums = mx.sum(onehot * scores.astype(mx.float32)[..., None], axis=1)  # [BH, C]
    means = mx.where(counts > 0, sums / mx.maximum(counts, 1), mx.array(float("inf")))

    evict_chunk = mx.argmin(means, axis=-1)  # [BH]
    return (chunk_id == evict_chunk[:, None]) & in_body & over_budget[:, None]


def _compact_ragged_batched(
    keys: mx.array,
    values: mx.array,
    scores: mx.array,
    positions: mx.array,
    remove_mask: mx.array,  # [BH, n] bool -- True at rows to drop
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Push ``remove_mask`` positions to the tail of each row, preserving the
    relative order of survivors, via one stable-sort-free ``argsort``.

    Generalizes the single-eviction ``rows + (rows >= evict_idx)``
    ``take_along_axis`` trick (:func:`h2o._evict_via_mlx_batched`,
    :func:`cam._merge_pair_batched`'s compaction) to an arbitrary,
    per-row-varying number of removed positions — needed here because a
    chunk eviction can remove ``chunk_size`` rows, or fewer for the ragged
    tail chunk, and a row can also remove zero (not over budget this step).
    No stability requirement: shifting removed positions by ``+n`` keeps
    every row's sort key unique (survivors keep their original ``pos``,
    removed rows get ``pos + n``, and ``pos`` is already unique per row), so
    ``mx.argsort`` can never reorder two positions ambiguously.

    Returns:
        ``(keys, values, scores, positions, n_removed)`` — same shapes as
        the inputs (still padded/rectangular; only trailing columns become
        stale/unused garbage), plus ``[BH]`` int32 count of removed
        positions per row.
    """
    bh, n = remove_mask.shape
    pos = mx.broadcast_to(mx.arange(n)[None, :], (bh, n))
    sort_key = mx.where(remove_mask, pos + n, pos)
    order = mx.argsort(sort_key, axis=-1)  # [BH, n]
    keys = mx.take_along_axis(keys, order[..., None], axis=1)
    values = mx.take_along_axis(values, order[..., None], axis=1)
    scores = mx.take_along_axis(scores, order, axis=1)
    positions = mx.take_along_axis(positions, order, axis=1)
    n_removed = mx.sum(remove_mask.astype(mx.int32), axis=1)
    return keys, values, scores, positions, n_removed


def chunkkv_update_batched(
    keys: mx.array | None,  # [BH, n_max, D] fp16 or None, padding garbage beyond `lengths`
    values: mx.array | None,  # [BH, n_max, D] fp16 or None
    scores: mx.array | None,  # [BH, n_max] fp32 or None
    positions: mx.array | None,  # [BH, n_max] int32 or None -- true absolute positions
    lengths: mx.array | None,  # [BH] int32 or None -- valid prefix length per row
    new_keys: mx.array,  # [BH, S, D]
    new_values: mx.array,  # [BH, S, D]
    new_positions: mx.array,  # [BH, S] int32 -- true absolute positions of new_keys/new_values
    n_sink: int,
    budget: int,
    chunk_size: int,
    score_mode: str,
    record_kept_positions: bool = False,
) -> (
    tuple[mx.array, mx.array, mx.array, mx.array, mx.array]
    | tuple[mx.array, mx.array, mx.array, mx.array, mx.array, list[list[list[int]]]]
):
    """Vectorized-over-``BH`` equivalent of calling :func:`chunkkv_update`
    once per ``(batch, head)`` pair with identical per-row config.

    All ``BH`` rows share ``n_sink``/``budget``/``chunk_size``/``score_mode``
    (true for every real caller: :class:`ChunkKVCache` applies one uniform
    config to every head), so the per-token score/append/evict math —
    otherwise identical for every row — can run as one batched MLX call per
    step instead of ``BH`` separate Python-level calls into
    :func:`chunkkv_update`. Mirrors
    :func:`veloxquant_mlx.quantizers.h2o.h2o_update_batched`'s design, but
    unlike H2O/CaM/Squeeze (always exactly one row evicted, so state stays
    perfectly rectangular every step), a chunk eviction here removes
    ``chunk_size`` rows (or fewer, for the ragged tail chunk) — or zero, for
    a row not yet over budget — so different ``BH`` rows can genuinely hold
    different valid lengths *within* one multi-token absorption. State is
    therefore carried as a padded ``[BH, n_max, D]``/``[BH, n_max]`` buffer
    plus an explicit ``[BH]`` ``lengths`` array (the standard batched-ragged-
    sequence pattern); columns at/beyond ``lengths[i]`` are unused padding,
    never read by anything downstream of this function. The caller
    (:class:`ChunkKVCache`) already re-aligns every row to a common length
    once per whole call via a batched trim (see ``ChunkKVCache``'s own
    docstring) — exactly mirroring what the per-head loop + the existing
    scalar :func:`chunkkv_trim_to` already did, just now also batched.

    The per-token loop over ``S`` itself remains a genuine recurrence (each
    token's score update and chunk-eviction decision depends on the previous
    token's state) and is untouched, exactly as in the function it replaces
    — only the outer per-``(b,h)`` Python dispatch is removed.

    Numerically identical to the per-head loop it replaces: every op is the
    same formula as :func:`chunkkv_update`'s bootstrap/score-update/evict
    branches, applied over a leading ``BH`` axis (and, for the chunk-eviction
    step, using each row's own ``lengths`` for chunk boundaries) instead of a
    Python loop — verified bit-for-bit equivalent in
    ``veloxquant_mlx/tests/quantizers/test_chunkkv_batched.py``.

    Args:
        score_mode: ``"attn_mass"`` or ``"key_norm"`` — uniform across rows.

    Args:
        record_kept_positions: If True, also return a ``[BH]``-major list of
            per-step kept-index lists (one list of ``S`` entries per row,
            mirroring :func:`chunkkv_update`'s own ``record_kept_positions``)
            — used by :class:`ChunkKVCache` to publish a leader layer's
            eviction decisions to the index-reuse coordinator. Materializing
            these as Python lists is paid only when reuse is enabled (the
            leader's hot path when reuse is *not* configured never touches
            this, matching the scalar function's own opt-in cost).

    Returns:
        ``(keys, values, scores, positions, lengths)``, or, when
        ``record_kept_positions=True``, that 5-tuple plus a ``kept_positions``
        list-of-``BH``-lists-of-``S``-lists. ``keys``/``values`` are
        ``[BH, n_max, D]``, ``scores``/``positions`` are ``[BH, n_max]`` (all
        padded; only the first ``lengths[i]`` columns of row ``i`` are
        meaningful), and ``lengths`` is ``[BH]`` int32.
    """
    bh, s, d = new_keys.shape
    if s == 0:
        if record_kept_positions:
            return keys, values, scores, positions, lengths, [[] for _ in range(bh)]
        return keys, values, scores, positions, lengths
    if n_sink >= budget:
        raise ValueError("chunkkv: n_sink must be < budget — no evictable positions remain")

    kept_positions: list[list[list[int]]] = [[] for _ in range(bh)] if record_kept_positions else []

    # Required capacity: eviction fires every step once a row is over
    # budget (at most one chunk removed per absorbed token -- see this
    # function's docstring), so no row's physical width ever needs to
    # exceed `budget + 1` once it has been over budget at least once; a row
    # still bootstrapping (never yet over budget) only needs width up to its
    # own `lengths + s`. `budget + s` upper-bounds both cases and is safe
    # across repeated calls too (existing buffer width only ever needs to be
    # `>= budget + 1`, which holds once any call has had `s >= 1`) -- grown
    # defensively anyway rather than assumed, so a future change to the
    # eviction step-count invariant fails safe instead of overflowing a
    # `put_along_axis` write out of bounds.
    existing_width = 0 if keys is None else keys.shape[1]
    existing_max_len = 0 if lengths is None else int(mx.max(lengths).item())
    required = max(existing_max_len + s, budget + 1)
    if required > existing_width:
        pad = required - existing_width
        if keys is None:
            keys = mx.zeros((bh, required, d), dtype=mx.float16)
            values = mx.zeros((bh, required, d), dtype=mx.float16)
            scores = mx.zeros((bh, required), dtype=mx.float32)
            positions = mx.zeros((bh, required), dtype=mx.int32)
            lengths = mx.zeros((bh,), dtype=mx.int32)
        else:
            keys = mx.concatenate([keys, mx.zeros((bh, pad, d), dtype=mx.float16)], axis=1)
            values = mx.concatenate([values, mx.zeros((bh, pad, d), dtype=mx.float16)], axis=1)
            scores = mx.concatenate([scores, mx.zeros((bh, pad), dtype=mx.float32)], axis=1)
            positions = mx.concatenate([positions, mx.zeros((bh, pad), dtype=mx.int32)], axis=1)

    n_max = keys.shape[1]

    for i in range(s):
        k_i = new_keys[:, i]  # [BH, D]
        v_i = new_values[:, i].astype(mx.float16)  # [BH, D]
        p_i = new_positions[:, i]  # [BH]

        is_bootstrap = lengths == 0  # [BH] rows absorbing their very first token
        pos = mx.broadcast_to(mx.arange(n_max)[None, :], (bh, n_max))
        valid_mask = pos < lengths[:, None]

        if score_mode == "key_norm":
            new_score_col = mx.sqrt(mx.sum(k_i.astype(mx.float32) ** 2, axis=-1))  # [BH]
            scores_after_update = scores  # fixed norms; nothing to decay/accumulate
        else:
            attn = _attention_scores_batched_masked(
                k_i.astype(mx.float32), keys.astype(mx.float32), valid_mask
            )
            scores_after_update = mx.where(valid_mask, scores + attn, scores)
            new_score_col = mx.where(is_bootstrap, mx.array(1.0), mx.array(0.0))

        write_pos = lengths[:, None]  # [BH, 1] -- each row's own append slot
        keys = mx.put_along_axis(
            keys, write_pos[..., None], k_i.astype(mx.float16)[:, None, :], axis=1
        )
        values = mx.put_along_axis(values, write_pos[..., None], v_i[:, None, :], axis=1)
        scores = mx.put_along_axis(scores_after_update, write_pos, new_score_col[:, None], axis=1)
        positions = mx.put_along_axis(positions, write_pos, p_i[:, None], axis=1)

        new_lengths = lengths + 1
        over_budget = new_lengths > budget

        evict_mask = _evict_lowest_chunk_batched(
            scores, new_lengths, n_sink, chunk_size, over_budget
        )

        if record_kept_positions:
            keep_mask_np = (~evict_mask).tolist()
            new_lengths_list = new_lengths.tolist()
            for row in range(bh):
                nl = new_lengths_list[row]
                kept_positions[row].append([j for j in range(nl) if keep_mask_np[row][j]])

        keys, values, scores, positions, n_removed = _compact_ragged_batched(
            keys, values, scores, positions, evict_mask
        )
        lengths = new_lengths - n_removed

        if (i + 1) % _EVAL_FLUSH_INTERVAL == 0:
            mx.eval(keys, values, scores, positions, lengths)

    if record_kept_positions:
        return keys, values, scores, positions, lengths, kept_positions
    return keys, values, scores, positions, lengths


def chunkkv_apply_reuse_indices_batched(
    keys: mx.array | None,  # [BH, n_max, D] fp16 or None
    values: mx.array | None,  # [BH, n_max, D] fp16 or None
    positions: mx.array | None,  # [BH, n_max] int32 or None -- true absolute positions
    lengths: mx.array | None,  # [BH] int32 or None
    new_keys: mx.array,  # [BH, S, D]
    new_values: mx.array,  # [BH, S, D]
    new_positions: mx.array,  # [BH, S] int32
    kept_positions_per_row: list[
        list[list[int]]
    ],  # [BH][S] -- indices into that row's pre-evict frame
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Vectorized-over-``BH`` equivalent of calling
    :func:`chunkkv_apply_reuse_indices` once per ``(batch, head)`` pair.

    Unlike the leader path, there is no independent scoring/eviction
    decision here — ``kept_positions_per_row[bh][i]`` is externally supplied
    (the leader's own decision for this row's head, identical across every
    batch element sharing that head — see
    ``ChunkKVIndexReuseCoordinator.fetch``'s per-``(layer, head)`` keying),
    so this function only needs to batch the append-then-select mechanics,
    not any scoring math. Since a follower's own K/V are its own real inputs
    (only the kept-index *decision* is reused), this stays index-list-driven
    rather than trying to re-derive a batched score.

    Because ``kept_positions_per_row[bh]`` are plain Python lists (not MLX
    arrays — the same representation :func:`chunkkv_update`'s
    ``record_kept_positions=True`` already returns and
    :func:`chunkkv_apply_reuse_indices` already consumes), they are turned
    into one ``[BH, n_max]`` boolean keep-mask per step via a Python-level
    scatter (cheap: at most ``budget`` entries per row, no Metal dispatch),
    then applied with the same padded/ragged-lengths machinery as
    :func:`chunkkv_update_batched`.

    Returns:
        ``(keys, values, positions, lengths)`` — no ``scores`` output:
        followers never evict on their own, so (mirroring
        :func:`chunkkv_apply_reuse_indices`, whose own docstring notes its
        ``scores`` field is an unmaintained placeholder) there is nothing
        meaningful to track.
    """
    bh, s, d = new_keys.shape
    if s == 0:
        return keys, values, positions, lengths

    # Required capacity: unlike the leader path (where eviction fires every
    # step once over budget, so the physical row count never exceeds
    # `budget + 1` regardless of `s` -- see chunkkv_update_batched), a
    # follower's per-step kept-index lists are externally supplied and can
    # be arbitrarily wide (bounded only by the leader's own budget, which
    # this function has no visibility into), so a buffer sized once on the
    # first call is not guaranteed sufficient for every later call -- grow
    # (zero-pad) whenever this call's own peak requirement exceeds the
    # existing buffer, instead of assuming monotonic sufficiency.
    max_kept_width = max(
        (max((len(kp) for kp in row), default=0) for row in kept_positions_per_row), default=0
    )
    existing_width = 0 if keys is None else keys.shape[1]
    existing_max_len = 0 if lengths is None else int(mx.max(lengths).item())
    required = max(existing_max_len + s, max_kept_width, 1)
    if required > existing_width:
        pad = required - existing_width
        if keys is None:
            keys = mx.zeros((bh, required, d), dtype=mx.float16)
            values = mx.zeros((bh, required, d), dtype=mx.float16)
            positions = mx.zeros((bh, required), dtype=mx.int32)
            lengths = mx.zeros((bh,), dtype=mx.int32)
        else:
            keys = mx.concatenate([keys, mx.zeros((bh, pad, d), dtype=mx.float16)], axis=1)
            values = mx.concatenate([values, mx.zeros((bh, pad, d), dtype=mx.float16)], axis=1)
            positions = mx.concatenate([positions, mx.zeros((bh, pad), dtype=mx.int32)], axis=1)

    n_max = keys.shape[1]

    for i in range(s):
        k_i = new_keys[:, i].astype(mx.float16)  # [BH, D]
        v_i = new_values[:, i].astype(mx.float16)
        p_i = new_positions[:, i]  # [BH]

        write_pos = lengths[:, None]  # [BH, 1]
        keys = mx.put_along_axis(keys, write_pos[..., None], k_i[:, None, :], axis=1)
        values = mx.put_along_axis(values, write_pos[..., None], v_i[:, None, :], axis=1)
        positions = mx.put_along_axis(positions, write_pos, p_i[:, None], axis=1)
        new_lengths = lengths + 1

        # Build this step's [BH, n_max] keep mask from the externally given
        # per-row index lists (relative to that row's own pre-evict frame,
        # i.e. columns [0, new_lengths[row])).
        keep_np = [[False] * n_max for _ in range(bh)]
        for row in range(bh):
            for j in kept_positions_per_row[row][i]:
                keep_np[row][j] = True
        keep_mask = mx.array(keep_np)
        remove_mask = (mx.arange(n_max)[None, :] < new_lengths[:, None]) & ~keep_mask

        pos = mx.broadcast_to(mx.arange(n_max)[None, :], (bh, n_max))
        sort_key = mx.where(remove_mask, pos + n_max, pos)
        order = mx.argsort(sort_key, axis=-1)
        keys = mx.take_along_axis(keys, order[..., None], axis=1)
        values = mx.take_along_axis(values, order[..., None], axis=1)
        positions = mx.take_along_axis(positions, order, axis=1)
        n_removed = mx.sum(remove_mask.astype(mx.int32), axis=1)
        lengths = new_lengths - n_removed

        if (i + 1) % _EVAL_FLUSH_INTERVAL == 0:
            mx.eval(keys, values, positions, lengths)

    return keys, values, positions, lengths


def chunkkv_trim_batched(
    keys: mx.array,
    values: mx.array,
    scores: mx.array | None,
    positions: mx.array,
    lengths: mx.array,
    n_sink: int,
    target: int,
) -> tuple[mx.array, mx.array, mx.array | None, mx.array, mx.array]:
    """Vectorized-over-``BH`` equivalent of calling :func:`chunkkv_trim_to`
    once per row with a shared ``target`` — the batched form of
    :class:`ChunkKVCache`'s own post-loop cross-head min-length alignment.

    Keeps each row's sinks plus its most recent ``target - n_sink`` non-sink
    tokens, exactly like the scalar version — a no-op for rows already at or
    under ``target`` (so at ``chunk_size == 1``, where every row already
    holds exactly ``budget``, nothing is trimmed and the H2O equivalence is
    preserved, same as the scalar path).

    Args:
        scores: Pass ``None`` for the follower path (which has no
            meaningfully-maintained scores to trim in lockstep — see
            :func:`chunkkv_apply_reuse_indices_batched`).

    Returns:
        ``(keys, values, scores, positions, lengths)`` — same shapes as
        input; ``scores`` is ``None`` iff the input was ``None``.
    """
    bh, n_max = keys.shape[0], keys.shape[1]
    pos = mx.broadcast_to(mx.arange(n_max)[None, :], (bh, n_max))

    target_arr = mx.minimum(mx.array(target, dtype=mx.int32), lengths)
    n_sink_eff = mx.minimum(mx.minimum(mx.array(n_sink, dtype=mx.int32), lengths), target_arr)
    n_recent = target_arr - n_sink_eff
    tail_start = mx.where(n_recent > 0, lengths - n_recent, lengths)

    is_sink_kept = pos < n_sink_eff[:, None]
    is_recent_kept = (pos >= tail_start[:, None]) & (pos < lengths[:, None])
    keep = is_sink_kept | is_recent_kept

    # Relative order within survivors: sinks first (original position),
    # then the recent tail (original position, already > any sink position)
    # -- both already monotonic in `pos`, so `pos` itself is a valid sort key
    # for survivors; non-kept columns (recency gap + padding) sort last.
    sort_key = mx.where(keep, pos, pos + n_max)
    order = mx.argsort(sort_key, axis=-1)
    keys = mx.take_along_axis(keys, order[..., None], axis=1)
    values = mx.take_along_axis(values, order[..., None], axis=1)
    positions = mx.take_along_axis(positions, order, axis=1)
    if scores is not None:
        scores = mx.take_along_axis(scores, order, axis=1)
    return keys, values, scores, positions, target_arr


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
    if state.keys is None or state.values is None or state.scores is None:
        # keys/values/scores are all-or-nothing: None only before first update.
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
    return get_kv(state.keys, state.values)


def chunkkv_fp16_bytes(state: ChunkKVState) -> int:
    """Bytes currently stored for K + V in fp16."""
    return fp16_kv_bytes(state.keys)


def full_chunkkv_fp16_bytes(tokens_seen: int, head_dim: int) -> int:
    """Hypothetical fp16 K + V bytes if all ``tokens_seen`` were stored."""
    return full_fp16_kv_bytes(tokens_seen, head_dim)


__all__ = [
    "chunk_partition",
    "chunk_scores",
    "chunkkv_keep_mask",
    "ChunkKVState",
    "init_chunkkv_state",
    "chunkkv_update",
    "chunkkv_update_batched",
    "chunkkv_apply_reuse_indices",
    "chunkkv_apply_reuse_indices_batched",
    "chunkkv_trim_to",
    "chunkkv_trim_batched",
    "chunkkv_get_kv",
    "chunkkv_fp16_bytes",
    "full_chunkkv_fp16_bytes",
]
