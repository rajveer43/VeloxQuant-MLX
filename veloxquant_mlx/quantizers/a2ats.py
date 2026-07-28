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

**Query-aware codebook assignment** comes in two forms here:

1. :func:`a2ats_h_weighted_assignment` — **the paper's actual objective**
   (Eq. 13/14): minimize ``(k̃ − c) H (k̃ − c)ᵀ`` where ``H = E[q̃ᵀq̃]`` is the
   query second-moment matrix. Computed exactly via the Eq. (15)-(18)
   Cholesky identity, which turns the ``H``-weighted form into a plain
   Euclidean distance in the space ``z = k̃L``. Requires a calibrated ``H``
   (see :func:`a2ats_query_second_moment`), so it is available on paths with
   offline calibration.
2. :func:`a2ats_query_aware_assignment` — a cosine blend between
   reconstruction error and query-centroid alignment, reusing the shape of
   :func:`veloxquant_mlx.quantizers.amc.amc_query_aware_saliency`. This is a
   **substitute** for Eq. (14), not an implementation of it: the cosine term
   is a constant per-centroid bias rather than a per-token coupling. Retained
   for the decode path, where only a single proxy query is available and no
   calibrated ``H`` exists. See its docstring for the precise differences
   (:issue:`29`, finding 4).

Public API:
  a2ats_query_second_moment    — calibrate H = E[qᵀq] (Eq. 10)
  a2ats_cholesky_factor        — L with H = L Lᵀ (Eq. 15)
  a2ats_h_weighted_assignment  — the paper's Eq. (14) assignment
  a2ats_query_aware_assignment — cosine-blend substitute (decode path)
  a2ats_select_retrieval_set   — split token indices into retrieval / bulk sets
"""
from __future__ import annotations

import math
from typing import Tuple

import mlx.core as mx
import numpy as np

from veloxquant_mlx.dsa.heap import MaxHeap


def a2ats_query_second_moment(queries: mx.array, ridge: float = 1e-4) -> mx.array:
    """Second-moment matrix ``H = E[q^T q]`` of a set of query states (Eq. 10).

    This is the object the paper's §3.2 is about: the query-aware and plain-VQ
    objectives coincide **only if ``H ∝ I``**, and Figure 2 shows it is not —
    that non-diagonal structure is the paper's contribution.

    Args:
        queries: ``[M, d]`` post-PE query states, collected offline on a
            representative dataset (paper §4.2's calibration step).
        ridge: Added to the diagonal so ``H`` is positive *definite* rather
            than merely semi-definite, which Cholesky (Eq. 15) requires.
            Needed whenever ``M < d`` or the queries are rank-deficient.

    Returns:
        ``[d, d]`` float32 positive-definite second-moment matrix.
    """
    q32 = queries.astype(mx.float32)
    if q32.ndim == 1:
        q32 = q32[None, :]
    m, d = q32.shape
    h = (q32.T @ q32) / float(max(m, 1))
    return h + ridge * mx.eye(d, dtype=mx.float32)


def a2ats_cholesky_factor(h: mx.array) -> mx.array:
    """Lower-triangular Cholesky factor ``L`` with ``H = L Lᵀ`` (Eq. 15).

    Eq. (16)-(18) use ``L`` to map the ``H``-weighted objective into a plain
    Euclidean one on ``z = k̃L``, which is what lets conventional k-means++
    build the codebook. Implemented directly (no ``mx.linalg.cholesky``
    dependency) for the small ``d`` values this repo's sub-vector VQ uses.

    Args:
        h: ``[d, d]`` symmetric positive-definite matrix.

    Returns:
        ``[d, d]`` float32 lower-triangular ``L``.
    """
    a = np.array(h.astype(mx.float32), copy=True).astype(np.float64)
    d = a.shape[0]
    ell = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(i + 1):
            s = a[i, j] - float(ell[i, :j] @ ell[j, :j])
            if i == j:
                ell[i, j] = math.sqrt(max(s, 1e-12))
            else:
                ell[i, j] = s / ell[j, j]
    return mx.array(ell.astype(np.float32))


def a2ats_h_weighted_assignment(
    x: mx.array,
    codebook: mx.array,
    cholesky_factor: mx.array,
) -> mx.array:
    """The paper's query-aware assignment, Eq. (14)/(17).

    ``f'(k̃; C) = argmin_j (k̃ − c_j) H (k̃ − c_j)ᵀ``

    Implemented via the Eq. (17) identity — with ``H = L Lᵀ``, the
    ``H``-weighted quadratic form is a plain squared Euclidean distance in
    the transformed space ``z = k̃L``:

        ``(k̃ − c_j) H (k̃ − c_j)ᵀ = (k̃L − c_jL)(k̃L − c_jL)ᵀ``

    So this is exact, not an approximation: it computes the paper's
    objective, just in the coordinates where it is cheap.

    Unlike the cosine blend in :func:`a2ats_query_aware_assignment`, the
    ``H`` term couples ``k̃`` and the query distribution **per token** —
    it is not a constant per-centroid bias added to every row.

    Args:
        x: ``[N, d]`` sub-vectors to quantize.
        codebook: ``[K, d]`` centroids, in the original (untransformed) space.
        cholesky_factor: ``[d, d]`` ``L`` from :func:`a2ats_cholesky_factor`.

    Returns:
        ``[N]`` int32 assigned centroid indices.
    """
    if x.shape[0] == 0:
        return mx.zeros((0,), dtype=mx.int32)
    ell = cholesky_factor.astype(mx.float32)
    z = x.astype(mx.float32) @ ell            # [N, d]
    cz = codebook.astype(mx.float32) @ ell    # [K, d]
    diff = z[:, None, :] - cz[None, :, :]     # [N, K, d]
    return mx.argmin(mx.sum(diff * diff, axis=-1), axis=-1).astype(mx.int32)


def a2ats_query_aware_assignment(
    x: mx.array,
    codebook: mx.array,
    query: mx.array,
    beta: float = 0.5,
) -> mx.array:
    """Cosine-blend query-aware assignment — a *substitute* for Eq. (14),
    not an implementation of it.

    ``score(c) = beta * (-||x - c||^2 normalized) + (1 - beta) * cosine_similarity(query, c)``

    Picks, for each row of ``x``, the centroid maximizing this blended score
    rather than the plain nearest centroid by reconstruction error alone.
    ``beta=1.0`` reduces to plain nearest-centroid VQ (equivalent to
    :func:`veloxquant_mlx.allocators.vecinfer.quantize_vq`); ``beta=0.0`` is
    pure query-direction alignment, ignoring reconstruction quality entirely.

    .. warning::
       **This is not the paper's estimator.** Paper Eq. (13)/(14) minimizes an
       ``H``-weighted quadratic form, where ``H = E[q̃ᵀq̃]`` is the query
       second-moment matrix; §3.2's entire argument is that ``H ∦ I``. Two
       concrete differences:

       - ``cos_sim`` here depends only on ``(codebook, query)``, so it adds an
         **identical per-centroid bias to every row of ``x``**. Eq. (14)
         couples ``k̃`` and ``H`` per token.
       - ``err_term`` is normalized per-row by ``max_err``, making ``beta``
         scale-dependent: the same ``beta`` means different things for
         different key magnitudes. ``H``-weighting has no such sensitivity.

       Use :func:`a2ats_h_weighted_assignment` for the paper's actual
       objective. This function is retained for the single-query decode path,
       where no calibrated ``H`` is available (see :issue:`29`, finding 4).

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

    diff = x32[:, None, :] - cb32[None, :, :]         # [N, K, d]
    sq_err = mx.sum(diff * diff, axis=-1)              # [N, K]
    max_err = mx.max(sq_err, axis=-1, keepdims=True)
    max_err = mx.maximum(max_err, 1e-8)
    err_term = 1.0 - sq_err / max_err                  # [N, K], higher = better, in [0, 1]

    cb_norm = mx.sqrt(mx.sum(cb32 * cb32, axis=-1))     # [K]
    q_norm = mx.sqrt(mx.sum(q32 * q32))                 # scalar
    eps = 1e-8
    denom = mx.maximum(cb_norm * q_norm, eps)
    cos_sim = (cb32 @ q32) / denom                      # [K]
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
    sim = (k32 @ q32) / denom   # [N]
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
    "a2ats_query_second_moment",
    "a2ats_cholesky_factor",
    "a2ats_h_weighted_assignment",
    "a2ats_query_aware_assignment",
    "a2ats_select_retrieval_set",
]
