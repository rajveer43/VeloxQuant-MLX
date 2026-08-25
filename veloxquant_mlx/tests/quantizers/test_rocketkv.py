"""Tests for RocketKV-adapted quantizer primitives — two-stage KV compression.

RocketKV-adapted (arXiv:2502.14051, ICML 2025) composes stage-1 permanent
eviction (reused from ``snapkv.py``, tested there) with stage-2 Hybrid Sparse
Attention (HSA) — this module's original contribution: paged element-wise
max/min key summaries, head-dimension top-k channel selection, a
two-dimensional approximate-attention-score step, page-level top-k
selection, and the adaptive compression-ratio decomposition (paper §3.6).
All data is synthetic — no model loading.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.rocketkv import (
    append_paged_summary,
    build_paged_summary,
    gather_page_tokens,
    head_dim_topk_mask,
    hsa_approx_scores,
    hsa_fp32_bytes,
    select_topk_pages,
    split_compression_ratio,
    split_hsa_dims,
)


def _rand_keys(S: int, D: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((S, D)).astype(np.float32))


# ---------------------------------------------------------------------------
# build_paged_summary
# ---------------------------------------------------------------------------


def test_paged_summary_shapes() -> None:
    k = _rand_keys(S=17, D=16)
    summary = build_paged_summary(k, page_size=4)
    assert summary.page_max.shape == (5, 16)  # ceil(17/4) = 5
    assert summary.page_min.shape == (5, 16)
    assert summary.n_tokens == 17


def test_paged_summary_max_min_correct_per_page() -> None:
    k = _rand_keys(S=8, D=4, seed=1)
    summary = build_paged_summary(k, page_size=4)
    expected_max0 = mx.max(k[0:4], axis=0)
    expected_min0 = mx.min(k[0:4], axis=0)
    assert np.allclose(np.array(summary.page_max[0]), np.array(expected_max0))
    assert np.allclose(np.array(summary.page_min[0]), np.array(expected_min0))


def test_paged_summary_single_page_when_page_size_ge_seq_len() -> None:
    k = _rand_keys(S=5, D=8)
    summary = build_paged_summary(k, page_size=100)
    assert summary.page_max.shape == (1, 8)


def test_paged_summary_page_size_clamped_to_at_least_1() -> None:
    k = _rand_keys(S=4, D=8)
    summary = build_paged_summary(k, page_size=0)
    assert summary.page_size == 1
    assert summary.page_max.shape == (4, 8)


# ---------------------------------------------------------------------------
# append_paged_summary — must exactly match a from-scratch rebuild
# ---------------------------------------------------------------------------


def test_append_matches_full_rebuild_exact_pages() -> None:
    k1 = _rand_keys(S=12, D=8, seed=2)
    k2 = _rand_keys(S=8, D=8, seed=3)
    direct = build_paged_summary(mx.concatenate([k1, k2], axis=0), page_size=4)
    incremental = append_paged_summary(build_paged_summary(k1, page_size=4), k2)
    assert np.allclose(np.array(direct.page_max), np.array(incremental.page_max))
    assert np.allclose(np.array(direct.page_min), np.array(incremental.page_min))
    assert direct.n_tokens == incremental.n_tokens == 20


def test_append_matches_full_rebuild_partial_pages() -> None:
    """Prior summary ends mid-page; new tokens straddle the page boundary."""
    k1 = _rand_keys(S=13, D=8, seed=4)  # 13 = 3 full pages of 4 + 1 trailing
    k2 = _rand_keys(S=7, D=8, seed=5)
    direct = build_paged_summary(mx.concatenate([k1, k2], axis=0), page_size=4)
    incremental = append_paged_summary(build_paged_summary(k1, page_size=4), k2)
    assert np.allclose(np.array(direct.page_max), np.array(incremental.page_max))
    assert np.allclose(np.array(direct.page_min), np.array(incremental.page_min))


def test_append_single_token_at_a_time_matches_rebuild() -> None:
    """Decode calls arrive one token at a time — the realistic RocketKV path."""
    k_all = _rand_keys(S=15, D=8, seed=6)
    summary = build_paged_summary(k_all[:9], page_size=4)
    for i in range(9, 15):
        summary = append_paged_summary(summary, k_all[i : i + 1])
    direct = build_paged_summary(k_all, page_size=4)
    assert np.allclose(np.array(direct.page_max), np.array(summary.page_max))
    assert np.allclose(np.array(direct.page_min), np.array(summary.page_min))
    assert summary.n_tokens == 15


# ---------------------------------------------------------------------------
# head_dim_topk_mask
# ---------------------------------------------------------------------------


def test_head_dim_topk_mask_selects_exactly_k1() -> None:
    q = mx.array([0.1, -5.0, 2.0, 0.3, -0.2], dtype=mx.float32)
    mask = head_dim_topk_mask(q, k1=2)
    assert int(mx.sum(mask.astype(mx.int32)).item()) == 2
    idx = [i for i in range(5) if bool(mask[i].item())]
    assert set(idx) == {1, 2}  # |−5.0| and |2.0| are the two largest magnitudes


def test_head_dim_topk_mask_k1_ge_d_selects_all() -> None:
    q = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
    mask = head_dim_topk_mask(q, k1=10)
    assert bool(mx.all(mask).item())


# ---------------------------------------------------------------------------
# hsa_approx_scores / select_topk_pages / gather_page_tokens
# ---------------------------------------------------------------------------


def test_hsa_scores_shape_matches_n_pages() -> None:
    k = _rand_keys(S=20, D=16, seed=7)
    summary = build_paged_summary(k, page_size=5)
    q = _rand_keys(S=1, D=16, seed=8)[0]
    scores = hsa_approx_scores(q, summary, k1=8)
    assert scores.shape == (4,)


def test_hsa_scores_bound_true_attention_within_page() -> None:
    """The HSA approx score for a page must upper-bound every token's true
    (full-head-dim, unscaled) dot product in that page — the paper's Step 2
    invariant (max(q.Kmax, q.Kmin) bounds q.k for k in [Kmin, Kmax])."""
    k = _rand_keys(S=16, D=8, seed=9)
    summary = build_paged_summary(k, page_size=4)
    q = _rand_keys(S=1, D=8, seed=10)[0]
    # k1 = D (no head-dim restriction) isolates the sequence-dim bound alone.
    scores = hsa_approx_scores(q, summary, k1=8)
    scale = 1.0 / math.sqrt(8)
    for p in range(4):
        page_tokens = k[p * 4 : (p + 1) * 4]
        true_dots = (page_tokens @ q) * scale
        assert float(scores[p].item()) >= float(mx.max(true_dots).item()) - 1e-4


def test_select_topk_pages_picks_highest_scores() -> None:
    scores = mx.array([0.1, 0.9, 0.3, 0.05, 0.7], dtype=mx.float32)
    pages = select_topk_pages(scores, k2=2)
    assert pages.tolist() == [1, 4]


def test_select_topk_pages_k2_ge_n_pages_selects_all() -> None:
    scores = mx.array([0.1, 0.2, 0.3], dtype=mx.float32)
    pages = select_topk_pages(scores, k2=100)
    assert pages.tolist() == [0, 1, 2]


def test_gather_page_tokens_expands_and_clamps_final_page() -> None:
    pages = mx.array([0, 2], dtype=mx.int32)
    toks = gather_page_tokens(pages, page_size=4, n_tokens=10)
    # page 0 -> [0,1,2,3]; page 2 -> [8,9] (clamped, n_tokens=10)
    assert toks.tolist() == [0, 1, 2, 3, 8, 9]


def test_gather_page_tokens_empty_input() -> None:
    toks = gather_page_tokens(mx.array([], dtype=mx.int32), page_size=4, n_tokens=10)
    assert toks.tolist() == []


# ---------------------------------------------------------------------------
# split_compression_ratio / split_hsa_dims — adaptive decomposition (§3.6)
# ---------------------------------------------------------------------------


def test_split_compression_ratio_no_compression_below_1() -> None:
    assert split_compression_ratio(1.0) == (1.0, 1.0)
    assert split_compression_ratio(0.5) == (1.0, 1.0)


def test_split_compression_ratio_product_equals_c() -> None:
    for c in (2.0, 8.0, 64.0, 400.0):
        s1, s2 = split_compression_ratio(c)
        assert s1 * s2 == pytest.approx(c, rel=1e-6)


def test_split_compression_ratio_matches_paper_worked_example() -> None:
    """Paper §3.6: c=64 -> r=0.56 -> stage1=10.3x, stage2=6.2x (approximately)."""
    s1, s2 = split_compression_ratio(64.0)
    assert s1 == pytest.approx(10.3, abs=0.1)
    assert s2 == pytest.approx(6.2, abs=0.1)


def test_split_compression_ratio_r_clamped_to_0_2_0_8() -> None:
    """Very small c should not push r below 0.2; very large c should not push it above 0.8."""
    s1_small, s2_small = split_compression_ratio(1.01)
    r_small = math.log(s1_small, 1.01)
    assert r_small >= 0.2 - 1e-6

    s1_large, s2_large = split_compression_ratio(1e12)
    r_large = math.log(s1_large, 1e12)
    assert r_large <= 0.8 + 1e-6


def test_split_hsa_dims_page_size_at_least_1() -> None:
    page_size, head_ratio = split_hsa_dims(1.0)
    assert page_size == 1
    assert head_ratio == 1.0


def test_split_hsa_dims_product_approx_ratio() -> None:
    _, stage2 = split_compression_ratio(64.0)
    page_size, head_ratio = split_hsa_dims(stage2)
    assert page_size * head_ratio == pytest.approx(stage2, rel=0.5)


# ---------------------------------------------------------------------------
# hsa_fp32_bytes
# ---------------------------------------------------------------------------


def test_hsa_fp32_bytes_scales_with_pages_and_dim() -> None:
    k = _rand_keys(S=20, D=16, seed=11)
    summary = build_paged_summary(k, page_size=5)
    n_bytes = hsa_fp32_bytes(summary.page_max, summary.page_min)
    assert n_bytes == 4 * 16 * 2 * 2  # n_pages * D * (max+min) * fp16
