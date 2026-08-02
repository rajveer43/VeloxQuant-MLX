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

The eviction happens **once at prefill** (``S > 1``). Decode tokens
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
        kept_indices: [n_kept] int32 — original token positions.
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
) -> mx.array:
    """Select the top-``budget`` token indices by importance score.

    Always includes the first ``n_sink`` positions (attention sinks), then
    fills the remaining budget with the highest-scored non-sink tokens.
    The union is returned sorted in ascending order (preserving original
    token order for sequential access).

    Args:
        scores: ``[S]`` fp32 importance scores.
        budget: Total number of tokens to keep (including sinks).
            Clamped to ``min(budget, S)``.
        n_sink: Number of initial positions always kept.
            Clamped to ``min(n_sink, budget)``.

    Returns:
        ``[n_kept]`` int32 indices in ascending order,
        where ``n_kept = min(budget, S)``.
    """
    S = int(scores.shape[0])
    budget = min(max(budget, 1), S)
    n_sink = min(max(n_sink, 0), budget)

    if budget >= S:
        return mx.arange(S, dtype=mx.int32)

    sink_set = set(range(n_sink))
    n_dynamic = budget - n_sink

    if n_dynamic <= 0:
        return mx.array(sorted(sink_set), dtype=mx.int32)

    score_list = scores.tolist()
    ranked = sorted(
        [i for i in range(S) if i not in sink_set],
        key=lambda i: score_list[i],
        reverse=True,
    )
    top_dynamic = set(ranked[:n_dynamic])
    kept = sorted(sink_set | top_dynamic)
    return mx.array(kept, dtype=mx.int32)


def snapkv_compress(
    keys: mx.array,
    values: mx.array,
    budget: int,
    obs_window: int = 32,
    n_sink: int = 4,
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
    S, D = keys.shape
    scores = obs_window_attention_scores(keys, obs_window)
    indices = snap_select_indices(scores, budget, n_sink)
    idx_list = [int(i) for i in indices.tolist()]

    kept_k = mx.stack([keys[i].astype(mx.float16) for i in idx_list], axis=0)
    kept_v = mx.stack([values[i].astype(mx.float16) for i in idx_list], axis=0)

    return SnapKVState(
        kept_keys=kept_k,
        kept_values=kept_v,
        kept_indices=indices,
        n_original=S,
        n_kept=len(idx_list),
    )


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
