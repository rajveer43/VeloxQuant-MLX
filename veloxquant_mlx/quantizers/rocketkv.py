"""RocketKV-adapted quantizer primitives — two-stage KV cache compression.

Inspired by "RocketKV: Accelerating Long-Context LLM Inference via Two-Stage
KV Cache Compression" (Behnam, Fu, Zhao, Tsai, Yu, Tumanov; NVIDIA/Georgia
Tech; ICML 2025, arXiv:2502.14051). Documented as "RocketKV-adapted
(VeloxQuant-MLX implementation)" — not a faithful port.

RocketKV composes two stages:
  1. Coarse-grain PERMANENT eviction at prefill (the paper adopts SnapKV
     directly; this repo reuses its own ``snapkv_compress`` — see
     ``veloxquant_mlx.quantizers.snapkv``).
  2. Fine-grain DYNAMIC top-k selection at every decode step, over the
     surviving stage-1 subset, via Hybrid Sparse Attention (HSA) — this
     module's original contribution.

HSA approximates per-step top-k attention scores with a two-dimensional
reduction instead of the single-dimension reductions used by Quest
(sequence-only, page min/max) or SparQ/Loki (head-only, top-magnitude
channels):

  Step 1 (paged summaries, updated incrementally as decode tokens arrive):
      Group keys into pages of ``page_size`` along the sequence dimension;
      store the element-wise max/min over each page (paper's ``K_max``,
      ``K_min``).
  Step 2 (per decode step, given query ``q``):
      Select the top-``head_topk1`` head-dimension channels by |q| magnitude
      (SparQ-style). Restricted to those channels, compute, per page, the
      per-channel upper bound ``sum_c max(q[c] * K_max[c], q[c] * K_min[c])``
      — equivalently ``q_pos . K_max + q_neg . K_min`` — as an approximate
      attention score, then pick the top-``k2`` pages (page-granularity here,
      not per-token — see limitation below) as the sparse-attention index set.
  Step 3:
      Fetch the exact K/V rows for the selected pages and run standard
      (dense, MLX-native) attention over just that subset.

Adaptation decisions (documented, never hidden):
  1. **Page-granularity selection, not token-granularity.** The paper's
     Step 3 gathers the *union of pages* whose approximate score ranks in
     the top-k2, i.e. the sparse-attention set size is
     ``k2 * page_size`` tokens, not exactly ``k2`` tokens. This module keeps
     that page-granularity behavior (matches the paper's own Algorithm 1,
     which operates on pages throughout).
  2. **No fused kernel.** Reconstruction/gathering happens eagerly in MLX
     ops each decode step — no FlashAttention-style tiled kernel. Same
     limitation class as every other non-fused method in this repo
     (AnchorKV-adapted, NestedKV-adapted).
  3. **Single (batch, head) at a time in the pure-numerics layer** — the
     cache wrapper (``rocketkv_cache.py``) loops over batch/head, matching
     the convention of ``snapkv.py`` and ``anchorkv.py``.
  4. **Adaptive compression decomposition (paper §3.6)** is implemented
     faithfully: ``r = clip(0.2 + 0.06 * log2(c), 0.2, 0.8)`` splits an
     overall compression ratio ``c`` into a stage-1 (eviction) ratio
     ``c ** r`` and a stage-2 (HSA) ratio ``c ** (1 - r)``, and the stage-2
     ratio is split evenly across the sequence (page size) and head
     (``head_topk1``) dimensions.

This module holds the pure, side-effect-free numerics: paged key summaries,
head-dimension top-k selection, the two-dimensional HSA score approximation,
page-level top-k index selection, and the adaptive compression-ratio split.
Stage-1 eviction itself is NOT reimplemented here — see
``veloxquant_mlx.quantizers.snapkv.snapkv_compress``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import mlx.core as mx


class PagedKeySummary(NamedTuple):
    """Per-page element-wise max/min key summaries.

    Attributes:
        page_max: ``[n_pages, D]`` fp32 — element-wise max over each page.
        page_min: ``[n_pages, D]`` fp32 — element-wise min over each page.
        page_size: Tokens per page (the last page may be partial).
        n_tokens: Total token count summarized.
    """

    page_max: mx.array
    page_min: mx.array
    page_size: int
    n_tokens: int


def build_paged_summary(keys: mx.array, page_size: int) -> PagedKeySummary:
    """Summarize ``[S, D]`` keys into per-page element-wise max/min.

    Args:
        keys: ``[S, D]`` key matrix for one head.
        page_size: Tokens per page. Clamped to ``>= 1``.

    Returns:
        :class:`PagedKeySummary` with ``n_pages = ceil(S / page_size)``.
    """
    S, D = keys.shape
    page_size = max(1, page_size)
    k32 = keys.astype(mx.float32)

    n_pages = math.ceil(S / page_size)
    pad = n_pages * page_size - S
    if pad > 0:
        pad_max = mx.full((pad, D), -mx.inf)
        pad_min = mx.full((pad, D), mx.inf)
        k_for_max = mx.concatenate([k32, pad_max], axis=0)
        k_for_min = mx.concatenate([k32, pad_min], axis=0)
    else:
        k_for_max = k32
        k_for_min = k32

    page_max = mx.max(k_for_max.reshape(n_pages, page_size, D), axis=1)
    page_min = mx.min(k_for_min.reshape(n_pages, page_size, D), axis=1)
    return PagedKeySummary(page_max=page_max, page_min=page_min, page_size=page_size, n_tokens=S)


def append_paged_summary(summary: PagedKeySummary, new_keys: mx.array) -> PagedKeySummary:
    """Incorporate newly appended ``[S_new, D]`` keys into an existing summary.

    Decode tokens arrive one (or a few) at a time; rebuilding the whole paged
    summary from scratch each step would be O(S) per step. Instead, the
    trailing partial page (if any) is recomputed together with the new
    tokens, and only the delta of full pages beyond it is added — cheap
    relative to a full rebuild, and exact (matches ``build_paged_summary``
    applied to the concatenation).

    Args:
        summary: Prior :class:`PagedKeySummary`.
        new_keys: ``[S_new, D]`` newly appended key rows.

    Returns:
        Updated :class:`PagedKeySummary` covering ``n_tokens + S_new`` tokens.
    """
    page_size = summary.page_size
    n_prior = summary.n_tokens
    n_full_pages_prior = n_prior // page_size
    trailing = n_prior - n_full_pages_prior * page_size  # tokens in the partial last page

    out_max_pages = [summary.page_max[i] for i in range(n_full_pages_prior)]
    out_min_pages = [summary.page_min[i] for i in range(n_full_pages_prior)]

    # Fold new tokens into the (possibly partial) trailing page first, then
    # into fresh full pages, one page at a time.
    new32 = new_keys.astype(mx.float32)
    n_new = new32.shape[0]
    cur_max = summary.page_max[n_full_pages_prior] if trailing else None
    cur_min = summary.page_min[n_full_pages_prior] if trailing else None
    cur_count = trailing

    idx = 0
    while idx < n_new:
        take = min(page_size - cur_count, n_new - idx)
        chunk = new32[idx : idx + take]
        chunk_max = mx.max(chunk, axis=0)
        chunk_min = mx.min(chunk, axis=0)
        if cur_count > 0:
            cur_max = mx.maximum(cur_max, chunk_max)
            cur_min = mx.minimum(cur_min, chunk_min)
        else:
            cur_max = chunk_max
            cur_min = chunk_min
        cur_count += take
        idx += take
        if cur_count >= page_size:
            out_max_pages.append(cur_max)
            out_min_pages.append(cur_min)
            cur_max, cur_min, cur_count = None, None, 0

    if cur_count > 0:
        out_max_pages.append(cur_max)
        out_min_pages.append(cur_min)

    page_max = mx.stack(out_max_pages, axis=0)
    page_min = mx.stack(out_min_pages, axis=0)
    return PagedKeySummary(
        page_max=page_max, page_min=page_min, page_size=page_size, n_tokens=n_prior + n_new
    )


def head_dim_topk_mask(query: mx.array, k1: int) -> mx.array:
    """Select the top-``k1`` head-dimension channels by ``|query|`` magnitude.

    SparQ-style channel selection: restricts the HSA approximation to the
    ``k1`` channels the query weighs most heavily.

    Args:
        query: ``[D]`` fp32 query vector for one head, one decode step.
        k1: Number of channels to keep. Clamped to ``[1, D]``.

    Returns:
        ``[D]`` bool mask, True at the ``k1`` selected channel positions.
    """
    D = int(query.shape[0])
    k1 = min(max(k1, 1), D)
    abs_q = mx.abs(query.astype(mx.float32))
    if k1 >= D:
        return mx.ones((D,), dtype=mx.bool_)
    threshold_idx = D - k1
    sorted_vals = mx.sort(abs_q)
    threshold = sorted_vals[threshold_idx]
    return abs_q >= threshold


def hsa_approx_scores(query: mx.array, summary: PagedKeySummary, k1: int) -> mx.array:
    """Approximate per-page attention scores for one query via HSA (paper Step 2).

    Restricted to the top-``k1`` head-dim channels by ``|query|``, computes a
    per-channel upper bound on ``q . k`` for any token whose per-channel
    value lies within ``[K_min, K_max]``: channel ``c`` contributes
    ``q[c] * K_max[c]`` when ``q[c] >= 0`` (maximizing that term) and
    ``q[c] * K_min[c]`` when ``q[c] < 0``. This is the paper's Algorithm 1
    Step 2 (``P <- Kmax[i: g_i>=0], Kmin[i: g_i<0]`` selected *per channel* by
    the sign of ``q``, then dotted with ``q``) — NOT a max of two whole-vector
    dot products, which is not a valid bound when ``q`` has mixed-sign
    entries (a per-page max over ``q.K_max`` and ``q.K_min`` can UNDER-shoot
    the true per-token dot product on channels where the two disagree in
    sign-optimality).

    Args:
        query: ``[D]`` fp32 query vector for one head, one decode step.
        summary: :class:`PagedKeySummary` for the same head.
        k1: Head-dimension channels to keep (see :func:`head_dim_topk_mask`).

    Returns:
        ``[n_pages]`` fp32 approximate attention scores (pre-softmax, unscaled
        dot-product bound — sufficient for ranking, not for exact softmax).
    """
    D = int(query.shape[0])
    mask = head_dim_topk_mask(query, k1)
    q32 = query.astype(mx.float32) * mask.astype(mx.float32)
    scale = 1.0 / math.sqrt(D)
    pos = mx.maximum(q32, 0.0)  # contributes via K_max
    neg = mx.minimum(q32, 0.0)  # contributes via K_min
    scores = summary.page_max.astype(mx.float32) @ pos + summary.page_min.astype(mx.float32) @ neg
    return scores * scale


def select_topk_pages(scores: mx.array, k2: int) -> mx.array:
    """Select the ``k2`` highest-scoring page indices, ascending order.

    Args:
        scores: ``[n_pages]`` fp32 approximate scores.
        k2: Number of pages to select. Clamped to ``[1, n_pages]``.

    Returns:
        ``[n_selected]`` int32 page indices, ``n_selected = min(k2, n_pages)``,
        sorted ascending.
    """
    n_pages = int(scores.shape[0])
    k2 = min(max(k2, 1), n_pages)
    if k2 >= n_pages:
        return mx.arange(n_pages, dtype=mx.int32)
    score_list = scores.tolist()
    ranked = sorted(range(n_pages), key=lambda i: score_list[i], reverse=True)
    chosen = sorted(ranked[:k2])
    return mx.array(chosen, dtype=mx.int32)


def gather_page_tokens(page_indices: mx.array, page_size: int, n_tokens: int) -> mx.array:
    """Expand selected page indices into the corresponding token indices.

    Args:
        page_indices: ``[n_selected_pages]`` int32 page indices, ascending.
        page_size: Tokens per page.
        n_tokens: Total token count (clamps the final partial page).

    Returns:
        ``[n_selected_tokens]`` int32 token indices, ascending, deduplicated.
    """
    out: list[int] = []
    for p in page_indices.tolist():
        start = p * page_size
        end = min(start + page_size, n_tokens)
        out.extend(range(start, end))
    if not out:
        return mx.array([], dtype=mx.int32)
    return mx.array(out, dtype=mx.int32)


def split_compression_ratio(c: float) -> tuple[float, float]:
    """Adaptive decomposition of an overall ratio ``c`` into (stage1, stage2).

    Implements the paper's §3.6 formula: ``r = clip(0.2 + 0.06*log2(c), 0.2, 0.8)``,
    stage-1 ratio ``c^r``, stage-2 ratio ``c^(1-r)``. ``c <= 1`` returns
    ``(1.0, 1.0)`` (no compression to split).

    Args:
        c: Overall target compression ratio (``c = S / token_budget``).

    Returns:
        ``(stage1_ratio, stage2_ratio)``, both ``>= 1.0``.
    """
    if c <= 1.0:
        return 1.0, 1.0
    r = min(0.2 + 0.06 * math.log2(c), 0.8)
    r = max(r, 0.2)
    stage1_ratio = c**r
    stage2_ratio = c ** (1 - r)
    return stage1_ratio, stage2_ratio


def split_hsa_dims(stage2_ratio: float) -> tuple[int, float]:
    """Split the stage-2 HSA ratio evenly across sequence and head dimensions.

    The sequence-dimension split becomes the page size (rounded up to an
    integer, since a page must contain a whole number of tokens); the
    head-dimension split absorbs the rounding remainder so the product still
    equals (approximately) ``stage2_ratio``.

    Args:
        stage2_ratio: Compression ratio assigned to stage 2 (HSA), ``>= 1.0``.

    Returns:
        ``(page_size, head_dim_ratio)``. ``page_size >= 1``.
    """
    if stage2_ratio <= 1.0:
        return 1, 1.0
    half = stage2_ratio**0.5
    page_size = max(1, math.ceil(half))
    head_dim_ratio = stage2_ratio / page_size
    return page_size, max(head_dim_ratio, 1.0)


def hsa_fp32_bytes(page_max: mx.array, page_min: mx.array) -> int:
    """Auxiliary storage bytes for one head's paged max/min summary (fp16-packed)."""
    n_pages, D = page_max.shape
    return int(n_pages * D * 2 * 2)  # max + min, fp16


__all__ = [
    "PagedKeySummary",
    "build_paged_summary",
    "append_paged_summary",
    "head_dim_topk_mask",
    "hsa_approx_scores",
    "select_topk_pages",
    "gather_page_tokens",
    "split_compression_ratio",
    "split_hsa_dims",
    "hsa_fp32_bytes",
]
