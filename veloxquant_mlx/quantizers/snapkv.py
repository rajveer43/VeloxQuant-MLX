"""SnapKV-adapted quantizer primitives — prefill observation-window token eviction.

Inspired by "SnapKV: LLM Knows What You are Looking for Before Generation"
(Yuan et al., ICLR 2025, arXiv:2404.14469). Documented as "SnapKV-adapted
(VeloxQuant-MLX implementation)" — not a faithful port.

What SnapKV adds that the repo did not have: **token eviction** — the cache
stores only a *budget* number of token positions instead of compressing all
tokens to fewer bits. This is the first method on the eviction axis and the
first where the paper's actual signal (attention scores over the observation
window) is computable at the cache-wrapper level without model interception.

Adaptation decisions (documented, never hidden):
  1. **Key-as-query proxy.** The paper uses the final ``obs_window`` *query*
     vectors from the prompt — not observable by a cache wrapper (only K/V are
     visible at ``update_and_fetch`` time). We substitute the last
     ``obs_window`` *key* vectors as proxy queries. Key and query spaces are
     correlated (both projected from the same residual stream), making this a
     stronger proxy than key-norm-only methods (KIVI-Sink, AdaKV-proxy,
     ZipCache-adapted). Still an approximation — stated plainly, never hidden.
  2. **Mean-pool only.** The paper applies a 1-D max-pool of width
     ``kernel_size`` to the pooled attention vector before ranking. We use
     mean-pooling only (no sliding-window kernel).
  3. **Stored tokens remain fp16.** This is pure eviction — no further
     quantization of the kept tokens. Composable with any quantizer cache
     wrapping the kept subset.
  4. **Uniform budget across heads.** All heads use ``snap_budget`` tokens.

The eviction happens **at every multi-token prefill chunk** (``S > 1``). Decode tokens
(``S == 1``) are always appended to the kept set — they are never evicted.

This module holds the pure, side-effect-free numerics: observation-window
attention scoring, top-k selection, and byte accounting.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import mlx.core as mx


class SnapKVState(NamedTuple):
    """Indices and fp16 slice for a SnapKV-evicted KV matrix.

    Attributes:
        kept_keys:    [n_kept, D] fp16 — the selected key rows.
        kept_values:  [n_kept, D] fp16 — the matching value rows.
        kept_indices: [n_kept] int32 — indices into the supplied candidate matrix.
        n_original:   int — total prefill token count before eviction.
        n_kept:       int — number of retained tokens (≤ n_original).
    """

    kept_keys: mx.array
    kept_values: mx.array
    kept_indices: mx.array
    n_original: int
    n_kept: int


def obs_window_attention_scores(
    keys: mx.array,
    obs_window: int,
) -> mx.array:
    """Compute per-token importance scores from the observation window.

    Uses the last ``obs_window`` key rows as proxy queries and computes
    their softmax attention distribution over all ``S`` key rows.
    Mean-pooling collapses the observation window into a single ``[S]``
    importance vector.

    Args:
        keys: ``[S, D]`` fp32/fp16 key matrix for one head.
        obs_window: Number of trailing tokens used as proxy queries.
            Clamped to ``min(obs_window, S)``.

    Returns:
        ``[S]`` fp32 importance score per token. Values are in [0, 1] and
        reflect the mean attention weight each prefix token receives from
        the observation window.
    """
    S, D = keys.shape
    w = min(max(obs_window, 1), S)
    k32 = keys.astype(mx.float32)
    q_proxy = k32[-w:]  # [w, D]
    scale = math.sqrt(D)
    logits = (q_proxy @ k32.T) / scale  # [w, S]
    attn = mx.softmax(logits, axis=-1)  # [w, S] — each row sums to 1
    scores = mx.mean(attn, axis=0)  # [S]
    return scores.astype(mx.float32)


def snap_select_indices(
    scores: mx.array,
    budget: int,
    n_sink: int,
    *,
    backend: str = "auto",
) -> mx.array:
    """Select the top-``budget`` token indices by importance score.

    Always includes the first ``n_sink`` positions (attention sinks), then
    fills the remaining budget with the highest-scored non-sink tokens.
    The union is returned sorted in ascending order (preserving original
    token order for sequential access).

    Args:
        scores: ``[S]`` fp32 importance scores. NaNs rank as negative infinity.
            Equal scores prefer the earliest candidate index.
        backend: ``auto``/``mlx`` use device selection; ``metal`` uses experimental
            prefix compaction; ``reference`` uses Python stable sorting.
        budget: Total number of tokens to keep (including sinks).
            Clamped to ``min(budget, S)``.
        n_sink: Number of initial positions always kept.
            Clamped to ``min(n_sink, budget)``.

    Returns:
        ``[n_kept]`` int32 indices in ascending order,
        where ``n_kept = min(budget, S)``.
    """
    if scores.ndim != 1:
        raise ValueError("scores must have shape [S]")
    return _snap_select_batched(scores[None], budget, n_sink, backend=backend)[0]


def _snap_select_batched(scores, budget, n_sink, *, backend="auto"):
    """Fixed-cardinality selection. NaNs rank as negative infinity."""
    if backend not in ("auto", "mlx", "metal", "reference"):
        raise ValueError(f"Unsupported SnapKV backend: {backend}")
    G, S = scores.shape
    budget = min(max(budget, 1), S)
    n_sink = min(max(n_sink, 0), budget)
    if budget >= S or budget == n_sink:
        return mx.broadcast_to(mx.arange(budget, dtype=mx.int32), (G, budget))
    if backend == "reference":
        result = []
        for values in scores.tolist():
            ranked = sorted(
                range(n_sink, S),
                key=lambda i: -math.inf if math.isnan(values[i]) else values[i],
                reverse=True,
            )
            result.append(sorted(list(range(n_sink)) + ranked[: budget - n_sink]))
        return mx.array(result, mx.int32)
    dynamic = scores[:, n_sink:]
    dynamic = mx.where(mx.isnan(dynamic), -float("inf"), dynamic)
    k = budget - n_sink
    threshold = mx.sort(dynamic, axis=-1)[:, dynamic.shape[-1] - k : dynamic.shape[-1] - k + 1]
    if backend == "metal":
        from veloxquant_mlx.metal._snapkv_select import select_from_threshold

        return select_from_threshold(dynamic, threshold, n_sink, k)
    above = dynamic > threshold
    equal = dynamic == threshold
    remaining = k - mx.sum(above.astype(mx.int32), axis=-1, keepdims=True)
    selected = above | (equal & (mx.cumsum(equal.astype(mx.int32), axis=-1) <= remaining))
    candidates = mx.arange(n_sink, S, dtype=mx.int32)
    ordered = mx.sort(mx.where(selected, candidates, S), axis=-1)[:, :k]
    return mx.concatenate(
        [mx.broadcast_to(mx.arange(n_sink, dtype=mx.int32), (G, n_sink)), ordered], axis=-1
    )


def snapkv_compress(
    keys: mx.array,
    values: mx.array,
    budget: int,
    obs_window: int = 32,
    n_sink: int = 4,
    *,
    backend: str = "auto",
) -> SnapKVState:
    """Compress ``[S, D]`` K and V to a budget-token subset via obs-window scoring.

    Args:
        keys: ``[S, D]`` fp16/fp32 key matrix for one head.
        values: ``[S, D]`` fp16/fp32 value matrix for one head.
        budget: Maximum number of tokens to retain.
        obs_window: Number of trailing key rows used as proxy queries.
        n_sink: Number of initial positions always kept.

    Returns:
        :class:`SnapKVState` with the selected fp16 key/value rows and metadata.
    """
    if keys.ndim != 2 or keys.shape != values.shape or keys.shape[-1] == 0:
        raise ValueError("K/V must have matching [S,D] shapes with D > 0")
    S, D = keys.shape
    scores = obs_window_attention_scores(keys, obs_window) if S else mx.zeros((0,))
    indices = snap_select_indices(scores, budget, n_sink, backend=backend)
    kept_k = mx.take(keys, indices, axis=0).astype(mx.float16)
    kept_v = mx.take(values, indices, axis=0).astype(mx.float16)

    return SnapKVState(
        kept_keys=kept_k,
        kept_values=kept_v,
        kept_indices=indices,
        n_original=S,
        n_kept=indices.shape[0],
    )


def _snapkv_compress_batched(
    keys,
    values,
    budget,
    obs_window,
    n_sink,
    *,
    backend="auto",
    output_dtype=mx.float16,
    batched_scoring=False,
):
    """Batch selection/gather while preserving the original per-head scorer."""
    B, H, S, D = keys.shape
    if values.shape != keys.shape or min(B, H, D) <= 0:
        raise ValueError("K/V must have matching nonzero batch/head/dimension shapes")
    flat_k = keys.reshape(B * H, S, D)
    flat_v = values.reshape(B * H, S, D)
    count = min(max(budget, 1), S)
    sinks = min(max(n_sink, 0), count)
    # Shape-only no-selection paths avoid constructing the scorer entirely.
    if count == S or sinks == count:
        return keys[:, :, :count].astype(output_dtype), values[:, :, :count].astype(output_dtype)
    if batched_scoring:
        w = min(max(obs_window, 1), S)
        k32 = flat_k.astype(mx.float32)
        logits = (k32[:, -w:] @ mx.swapaxes(k32, -1, -2)) / math.sqrt(D)
        scores = mx.mean(mx.softmax(logits, axis=-1), axis=-2)
    else:
        scores = mx.stack(
            [obs_window_attention_scores(flat_k[g], obs_window) for g in range(B * H)]
        )
    indices = _snap_select_batched(scores, budget, n_sink, backend=backend)
    if backend == "metal":
        from veloxquant_mlx.metal._snapkv_select import gather_kv

        k, v = gather_kv(flat_k, flat_v, indices, output_dtype)
    else:
        k = mx.take_along_axis(flat_k, indices[..., None], axis=1).astype(output_dtype)
        v = mx.take_along_axis(flat_v, indices[..., None], axis=1).astype(output_dtype)
    return k.reshape(B, H, count, D), v.reshape(B, H, count, D)


def snapkv_fp16_bytes(state: SnapKVState) -> int:
    """Bytes stored for a SnapKVState (kept fp16 K + V rows).

    Both K and V are fp16 (2 bytes/element). Only the kept rows are stored.
    """
    D = int(state.kept_keys.shape[1])
    return int(state.n_kept * D * 2 * 2)  # K + V, fp16


def full_fp16_bytes(n: int, d: int) -> int:
    """Bytes for uncompressed fp16 K + V (both tensors, ``n`` tokens, dim ``d``)."""
    return int(n * d * 2 * 2)


__all__ = [
    "SnapKVState",
    "obs_window_attention_scores",
    "snap_select_indices",
    "snapkv_compress",
    "snapkv_fp16_bytes",
    "full_fp16_bytes",
]
