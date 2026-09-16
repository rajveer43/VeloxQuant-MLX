"""Shared explicit-mask construction for prefill-eviction KV caches.

Fixes VeloxQuant-MLX#370: every eviction-family cache (SnapKV, StreamingLLM,
H2O, TOVA, ChunkKV, CaM) inherits ``mlx_lm.models.cache.KVCache.make_mask``
unchanged, which is exactly::

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

``create_attention_mask`` returns the *string* ``"causal"`` whenever
``return_array`` is False and the query length is > 1 (true for every
model's prefill call unless it opts into ``return_array=True``, which none
of the eviction methods' target models do). ``mx.fast.scaled_dot_product_
attention`` accepts ``mask="causal"`` even when the key count differs from
the query count, silently applying **lower-right-aligned** causal masking:
query ``i`` is treated as attending to a *contiguous trailing window* of
keys ending at key index ``i`` (see ``mx.fast.scaled_dot_product_attention``
docs: "the last query aligns with the last key"). Eviction breaks that
assumption the moment it drops a non-trailing token — the kept keys are a
sparse, non-contiguous subset of original positions, not a trailing window
— so every query ends up attending to the wrong keys relative to its true
position, silently corrupting attention with no error raised.

This module builds an **explicit boolean mask** from each kept key row's
true original absolute position instead, whenever eviction has made the
"causal" string's trailing-window assumption unsafe. Position ``j`` (a
stored row) is visible to query position ``i`` iff ``j <= i`` — ordinary
causal validity expressed on true positions rather than row indices.

Known residual limitation — head-uniform masking
--------------------------------------------------
``mx.fast.scaled_dot_product_attention``'s mask must broadcast to
``[B, N, T_q, T_kv]``, and mlx_lm's own mask machinery (see
``BatchKVCache.make_mask`` in ``mlx_lm.models.cache``, and
``create_causal_mask``'s ``left_padding`` handling) only ever varies a mask
by batch element and query/key position — **never by attention head**: a
per-head mask does not broadcast against GQA's query-head count (which the
cache cannot even observe — only sequence length reaches ``make_mask``, not
head counts), since MLX broadcasting requires the head dimension to be
either ``1`` or exactly the query head count. Every one of the six caches
this module serves evicts independently per (batch, KV-head), with
genuinely data-dependent kept positions that can differ head to head.

Building a true per-head mask is therefore not representable in mlx_lm's
mask contract at all — a structural ceiling, not a gap this module leaves
unaddressed. The mask built here uses **head 0's** kept positions (per
batch element) and broadcasts it over every head. This is exactly correct
whenever a layer's heads happen to agree on which positions survived
(common: same sink/budget configuration, often-similar attention
concentration across heads of one layer), and for the mask's shape/pattern
purposes is never worse than today's ``"causal"`` string, which is wrong
for every head, unconditionally, the moment eviction triggers.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import create_causal_mask


def eviction_make_mask(
    query_positions: mx.array,
    key_positions: mx.array,
    N: int,
    return_array: bool = False,
    window_size: int | None = None,
) -> str | mx.array | None:
    """Build an explicit causal mask from true absolute positions.

    Args:
        query_positions: ``[B, N]`` int32 — true absolute position of each
            of this call's ``N`` query rows.
        key_positions: ``[B, T_kv]`` int32 — true absolute position of each
            currently-stored key row (head 0, per batch element — see the
            module docstring's "head-uniform masking" limitation).
        N: Number of query rows this step (``queries.shape[2]``).
        return_array: Passed through from ``create_attention_mask``; forces
            the explicit-array path regardless (this function already only
            runs when an explicit array is required).
        window_size: Passed through for sliding-window composition; unused
            by any of this module's callers today (none wrap
            ``sliding_window``), accepted for signature parity with
            ``create_attention_mask``.

    Returns:
        ``None`` when ``N == 1`` (mlx_lm's own no-mask-needed fast path for
        single-token decode — safe here too, since a lone query with no
        earlier same-step queries needs no intra-step masking and the KV
        cache itself already excludes evicted/future rows). Otherwise a
        ``[B, 1, N, T_kv]`` boolean array: ``True`` where key row ``j`` is
        causally visible to query row ``i`` (``key_positions[b, j] <=
        query_positions[b, i]``).
    """
    if N == 1:
        return None
    # [B, N, 1] >= [B, 1, T_kv] -> [B, N, T_kv]
    visible = query_positions[:, :, None] >= key_positions[:, None, :]
    if window_size is not None:
        visible = visible & (query_positions[:, :, None] < key_positions[:, None, :] + window_size)
    return visible[:, None, :, :]


def uniform_kept_positions(offset: int, n_kept: int) -> mx.array:
    """``[n_kept]`` positions for a cache whose kept rows are already a
    contiguous trailing window ``[offset - n_kept, offset)`` — the one case
    where the inherited ``"causal"`` string shortcut was already correct.
    Provided so callers can cheaply special-case "no eviction has happened
    yet this call" without hand-rolling ``mx.arange``.
    """
    return mx.arange(offset - n_kept, offset, dtype=mx.int32)


__all__ = ["eviction_make_mask", "uniform_kept_positions", "create_causal_mask"]
