"""A2ATS-adapted query-aware VQ assignment + retrieval-set selection.

Inspired by "A2ATS: Retrieval-Based KV Cache Reduction via Windowed Rotary
Position Embedding and Query-Aware Vector Quantization" (He, Xing, Wang, Xu,
Wu, Zhou, Liu, Xue, Li — ACL 2025 Findings,
aclanthology.org/2025.findings-acl.644). Documented as "A2ATS-adapted
(VeloxQuant-MLX implementation)" — not a faithful port.

**Retrieval-selection signal is approximated.** The paper selects its
high-fidelity "retrieval set" using real attention-relevance signals that are
not observable inside a cache wrapper (the same category of gap as every
other saliency/attention proxy method in this repo: H2O-adapted's key-as-query
proxy, SnapKV-adapted's prefill window, ZipCache-adapted's key-norm proxy).
This port substitutes a **query-similarity proxy**: the top
``retrieval_fraction`` of tokens by cosine similarity between their key vector
and the current query are treated as the retrieval set (kept at a finer
codebook / lower compression than the remainder). Every token is still
quantized — nothing is evicted; the retrieval set only gets preferential
codebook assignment (see :func:`a2ats_query_aware_assignment`). Whether the
paper's own retrieval step additionally drops non-retrieved tokens outright is
a question for the cache wrapper layer, not this module — this module exposes
the *split*, the cache wrapper decides what to do with each half.

**Query-aware codebook assignment** (this module's other primitive) reuses
the same blend shape as
:func:`veloxquant_mlx.quantizers.amc.amc_query_aware_saliency` — a weighted
sum of a magnitude/reconstruction-quality term and a query-cosine-similarity
term, including the same zero-norm-guard style (also shared with NestedKV-
adapted's ``_cosine_anomaly`` and CurDKV-adapted's leverage-score guards) —
applied here to *codebook centroid selection* rather than *tier assignment*:
instead of picking the nearest centroid by reconstruction error alone (plain
nearest-centroid VQ, e.g. :func:`veloxquant_mlx.allocators.vecinfer.quantize_vq`),
A2ATS-adapted also rewards centroids whose direction is more aligned with the
current query, so retrieval-relevant tokens are quantized against centroids
that (all else equal) yield an inner product estimate closer to the query
direction.

Public API:
  a2ats_query_aware_assignment — query-aware nearest-centroid selection
  a2ats_select_retrieval_set   — split token indices into retrieval / bulk sets
"""

from __future__ import annotations

import math
from typing import Tuple

import mlx.core as mx

from veloxquant_mlx.dsa.heap import MaxHeap


def a2ats_query_aware_assignment(
    x: mx.array,
    codebook: mx.array,
    query: mx.array,
    beta: float = 0.5,
) -> mx.array:
    """Query-aware nearest-centroid assignment.

    ``score(c) = beta * (-||x - c||^2 normalized) + (1 - beta) * cosine_similarity(query, c)``

    Picks, for each row of ``x``, the centroid maximizing this blended score
    rather than the plain nearest centroid by reconstruction error alone.
    ``beta=1.0`` reduces to plain nearest-centroid VQ (equivalent to
    :func:`veloxquant_mlx.allocators.vecinfer.quantize_vq`); ``beta=0.0`` is
    pure query-direction alignment, ignoring reconstruction quality entirely.

    Args:
        x: ``[N, d]`` sub-vectors to quantize (``d`` == ``codebook.shape[-1]``).
        codebook: ``[K, d]`` candidate centroids.
        query: ``[d]`` current query sub-vector (same sub-dim slice as ``x``
            and ``codebook``).
        beta: Blend coefficient in ``[0, 1]``.

    Returns:
        ``[N]`` int32 assigned centroid indices.
    """
    x32 = x.astype(mx.float32)
    cb32 = codebook.astype(mx.float32)
    q32 = query.astype(mx.float32)

    diff = x32[:, None, :] - cb32[None, :, :]  # [N, K, d]
    sq_err = mx.sum(diff * diff, axis=-1)  # [N, K]
    max_err = mx.max(sq_err, axis=-1, keepdims=True)
    max_err = mx.maximum(max_err, 1e-8)
    err_term = 1.0 - sq_err / max_err  # [N, K], higher = better, in [0, 1]

    cb_norm = mx.sqrt(mx.sum(cb32 * cb32, axis=-1))  # [K]
    q_norm = mx.sqrt(mx.sum(q32 * q32))  # scalar
    eps = 1e-8
    denom = mx.maximum(cb_norm * q_norm, eps)
    cos_sim = (cb32 @ q32) / denom  # [K]
    cos_sim = mx.clip((cos_sim + 1.0) * 0.5, 0.0, 1.0)  # [-1,1] -> [0,1]

    score = beta * err_term + (1.0 - beta) * cos_sim[None, :]  # [N, K]
    return mx.argmax(score, axis=-1).astype(mx.int32)


def a2ats_select_retrieval_set(
    keys: mx.array,
    query: mx.array,
    retrieval_fraction: float = 0.20,
) -> Tuple[mx.array, mx.array]:
    """Split token indices into a high-fidelity retrieval set and the bulk.

    Uses :class:`veloxquant_mlx.dsa.heap.MaxHeap` to select the top
    ``ceil(retrieval_fraction * N)`` tokens by query-cosine-similarity in
    better-than-full-sort fashion, consistent with this repo's DSA-first
    convention (the same heap-based top-k pattern used by
    :func:`veloxquant_mlx.quantizers.amc.amc_assign_tiers`).

    Args:
        keys: ``[N, D]`` key vectors.
        query: ``[D]`` current query vector.
        retrieval_fraction: Fraction of tokens routed to the retrieval set,
            in ``[0, 1]``.

    Returns:
        ``(retrieval_idx, bulk_idx)`` — int32 index arrays, disjoint,
        covering every position ``0..N-1`` exactly once.
    """
    n = int(keys.shape[0])
    if n == 0:
        empty = mx.array([], dtype=mx.int32)
        return empty, empty

    k32 = keys.astype(mx.float32)
    q32 = query.astype(mx.float32)
    k_norm = mx.sqrt(mx.sum(k32 * k32, axis=-1))
    q_norm = mx.sqrt(mx.sum(q32 * q32))
    eps = 1e-8
    denom = mx.maximum(k_norm * q_norm, eps)
    sim = (k32 @ q32) / denom  # [N]
    mx.eval(sim)

    n_retrieve = max(1, math.ceil(retrieval_fraction * n))
    n_retrieve = min(n_retrieve, n)

    heap: MaxHeap = MaxHeap()
    sim_list = sim.tolist()
    for i, s in enumerate(sim_list):
        heap.push(float(s), i)

    retrieval_ids = []
    for _ in range(n_retrieve):
        if len(heap) == 0:
            break
        _, idx = heap.pop()
        retrieval_ids.append(idx)
    retrieval_ids.sort()

    retrieval_set = set(retrieval_ids)
    bulk_ids = [i for i in range(n) if i not in retrieval_set]

    return (
        mx.array(retrieval_ids, dtype=mx.int32),
        mx.array(bulk_ids, dtype=mx.int32),
    )


__all__ = [
    "a2ats_query_aware_assignment",
    "a2ats_select_retrieval_set",
]
