"""Tests for A2ATS-adapted query-aware VQ assignment + retrieval selection.

A2ATS-adapted (He et al., ACL 2025 Findings, aclanthology.org/2025.findings-acl.644)
blends reconstruction quality with query-cosine-similarity when assigning
codebook centroids, and splits tokens into a high-fidelity retrieval set vs.
the bulk via the same query-similarity signal. All data is synthetic.
"""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.a2ats import (
    a2ats_query_aware_assignment,
    a2ats_select_retrieval_set,
)


def _mat(n, d, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((n, d)).astype(np.float32))


# ---------------------------------------------------------------------------
# a2ats_query_aware_assignment
# ---------------------------------------------------------------------------

def test_assignment_output_shape_and_dtype() -> None:
    x = _mat(5, 4)
    cb = _mat(8, 4, seed=1)
    q = _mat(1, 4, seed=2)[0]
    idx = a2ats_query_aware_assignment(x, cb, q, beta=0.5)
    assert idx.shape == (5,)
    assert idx.dtype == mx.int32


def test_assignment_indices_in_range() -> None:
    x = _mat(10, 4, seed=3)
    cb = _mat(6, 4, seed=4)
    q = _mat(1, 4, seed=5)[0]
    idx = a2ats_query_aware_assignment(x, cb, q, beta=0.3)
    idx_np = np.array(idx)
    assert idx_np.min() >= 0
    assert idx_np.max() < 6


def test_beta_one_reduces_to_nearest_centroid() -> None:
    """beta=1.0 -> pure reconstruction-error term -> must pick the same
    centroid as plain nearest-centroid VQ, regardless of query."""
    x = mx.array([[1.0, 0.0, 0.0, 0.0]], dtype=mx.float32)
    cb = mx.array([
        [1.0, 0.0, 0.0, 0.0],   # nearest to x
        [0.0, 0.0, 0.0, 1.0],   # far from x, but aligned with an adversarial query
    ], dtype=mx.float32)
    adversarial_query = mx.array([0.0, 0.0, 0.0, 1.0], dtype=mx.float32)
    idx = a2ats_query_aware_assignment(x, cb, adversarial_query, beta=1.0)
    assert int(idx[0].item()) == 0


def test_query_aware_prefers_relevant_centroid_over_nearest() -> None:
    """Construct a case where the nearest centroid by reconstruction error is
    NOT the most query-relevant one; confirm beta < 1.0 shifts the
    assignment toward the query-relevant centroid — the direct proof this
    mode does something (mirrors AMC's
    test_query_aware_saliency_downweights_high_magnitude_irrelevant_tokens)."""
    x = mx.array([[1.0, 0.1, 0.0, 0.0]], dtype=mx.float32)
    cb = mx.array([
        [1.0, 0.0, 0.0, 0.0],    # nearest to x by L2, but orthogonal-ish to query
        [0.0, 0.0, 0.0, 1.0],    # far from x, but exactly aligned with query
    ], dtype=mx.float32)
    query = mx.array([0.0, 0.0, 0.0, 1.0], dtype=mx.float32)

    idx_nearest_only = a2ats_query_aware_assignment(x, cb, query, beta=1.0)
    idx_query_dominant = a2ats_query_aware_assignment(x, cb, query, beta=0.0)

    assert int(idx_nearest_only[0].item()) == 0     # pure reconstruction picks centroid 0
    assert int(idx_query_dominant[0].item()) == 1    # pure query-alignment picks centroid 1


def test_assignment_deterministic() -> None:
    x = _mat(6, 4, seed=7)
    cb = _mat(4, 4, seed=8)
    q = _mat(1, 4, seed=9)[0]
    idx1 = a2ats_query_aware_assignment(x, cb, q, beta=0.5)
    idx2 = a2ats_query_aware_assignment(x, cb, q, beta=0.5)
    assert np.array_equal(np.array(idx1), np.array(idx2))


# ---------------------------------------------------------------------------
# a2ats_select_retrieval_set
# ---------------------------------------------------------------------------

def test_retrieval_set_respects_fraction() -> None:
    keys = _mat(100, 8, seed=10)
    query = _mat(1, 8, seed=11)[0]
    ret_idx, bulk_idx = a2ats_select_retrieval_set(keys, query, retrieval_fraction=0.2)
    assert int(ret_idx.shape[0]) == pytest.approx(20, abs=2)
    assert int(ret_idx.shape[0]) + int(bulk_idx.shape[0]) == 100


def test_retrieval_and_bulk_disjoint_and_cover_all() -> None:
    keys = _mat(30, 8, seed=12)
    query = _mat(1, 8, seed=13)[0]
    ret_idx, bulk_idx = a2ats_select_retrieval_set(keys, query, retrieval_fraction=0.3)
    ret_set = set(np.array(ret_idx).tolist())
    bulk_set = set(np.array(bulk_idx).tolist())
    assert ret_set.isdisjoint(bulk_set)
    assert ret_set | bulk_set == set(range(30))


def test_retrieval_set_picks_most_similar_to_query() -> None:
    """Construct keys with a clear query-similarity ranking; confirm the
    retrieval set contains the most-similar tokens, not arbitrary ones."""
    d = 4
    query = mx.array([1.0, 0.0, 0.0, 0.0], dtype=mx.float32)
    keys = mx.array([
        [1.0, 0.0, 0.0, 0.0],    # most similar (parallel)
        [0.9, 0.1, 0.0, 0.0],    # second most similar
        [0.0, 1.0, 0.0, 0.0],    # orthogonal
        [-1.0, 0.0, 0.0, 0.0],   # anti-parallel (least similar)
    ], dtype=mx.float32)
    ret_idx, bulk_idx = a2ats_select_retrieval_set(keys, query, retrieval_fraction=0.5)
    ret_set = set(np.array(ret_idx).tolist())
    assert ret_set == {0, 1}


def test_retrieval_set_empty_input() -> None:
    keys = mx.zeros((0, 8))
    query = mx.zeros(8)
    ret_idx, bulk_idx = a2ats_select_retrieval_set(keys, query)
    assert ret_idx.shape[0] == 0
    assert bulk_idx.shape[0] == 0


def test_retrieval_set_single_token() -> None:
    keys = _mat(1, 8, seed=14)
    query = _mat(1, 8, seed=15)[0]
    ret_idx, bulk_idx = a2ats_select_retrieval_set(keys, query, retrieval_fraction=0.2)
    assert int(ret_idx.shape[0]) + int(bulk_idx.shape[0]) == 1


def test_retrieval_set_fraction_zero_still_returns_at_least_one() -> None:
    """max(1, ceil(...)) guard — retrieval_fraction=0.0 still routes one
    token, avoiding a degenerate always-empty retrieval set."""
    keys = _mat(10, 8, seed=16)
    query = _mat(1, 8, seed=17)[0]
    ret_idx, bulk_idx = a2ats_select_retrieval_set(keys, query, retrieval_fraction=0.0)
    assert int(ret_idx.shape[0]) == 1
    assert int(bulk_idx.shape[0]) == 9


def test_retrieval_set_fraction_one_returns_all() -> None:
    keys = _mat(10, 8, seed=18)
    query = _mat(1, 8, seed=19)[0]
    ret_idx, bulk_idx = a2ats_select_retrieval_set(keys, query, retrieval_fraction=1.0)
    assert int(ret_idx.shape[0]) == 10
    assert int(bulk_idx.shape[0]) == 0


def test_retrieval_set_deterministic() -> None:
    keys = _mat(15, 8, seed=20)
    query = _mat(1, 8, seed=21)[0]
    r1, b1 = a2ats_select_retrieval_set(keys, query, retrieval_fraction=0.25)
    r2, b2 = a2ats_select_retrieval_set(keys, query, retrieval_fraction=0.25)
    assert np.array_equal(np.array(r1), np.array(r2))
    assert np.array_equal(np.array(b1), np.array(b2))


# ---------------------------------------------------------------------------
# H-weighted assignment — the paper's actual Eq. (13)/(14) objective
# (:issue:`29`, finding 4)
# ---------------------------------------------------------------------------

def test_second_moment_is_symmetric_positive_definite() -> None:
    """``H = E[qᵀq]`` plus a ridge must be SPD — Cholesky (Eq. 15) requires
    definite, not merely semi-definite, which matters when M < d."""
    from veloxquant_mlx.quantizers.a2ats import a2ats_query_second_moment

    q = mx.array(np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32))
    h = np.array(a2ats_query_second_moment(q))   # M=3 < d=8: rank-deficient
    assert np.allclose(h, h.T, atol=1e-5)
    assert np.all(np.linalg.eigvalsh(h) > 0)


def test_cholesky_factor_reconstructs_h() -> None:
    """Eq. (15): ``H = L Lᵀ``, with ``L`` lower-triangular."""
    from veloxquant_mlx.quantizers.a2ats import (
        a2ats_cholesky_factor, a2ats_query_second_moment,
    )

    q = mx.array(np.random.default_rng(1).standard_normal((50, 6)).astype(np.float32))
    h = a2ats_query_second_moment(q)
    ell = a2ats_cholesky_factor(h)
    assert np.allclose(np.array(ell), np.tril(np.array(ell)), atol=1e-6)
    assert np.allclose(np.array(ell @ ell.T), np.array(h), atol=1e-4)


def test_h_weighted_assignment_matches_bruteforce_eq14() -> None:
    """The Eq. (17) transform is an identity, not an approximation: assigning
    in ``z = k̃L`` space must equal minimizing ``(k̃ − c) H (k̃ − c)ᵀ``
    directly."""
    from veloxquant_mlx.quantizers.a2ats import (
        a2ats_cholesky_factor, a2ats_h_weighted_assignment,
        a2ats_query_second_moment,
    )

    rng = np.random.default_rng(2)
    q = mx.array(rng.standard_normal((60, 4)).astype(np.float32))
    h = a2ats_query_second_moment(q)
    ell = a2ats_cholesky_factor(h)
    x = mx.array(rng.standard_normal((25, 4)).astype(np.float32))
    cb = mx.array(rng.standard_normal((16, 4)).astype(np.float32))

    got = np.array(a2ats_h_weighted_assignment(x, cb, ell))

    xn, cbn, hn = np.array(x), np.array(cb), np.array(h)
    want = [
        int(np.argmin(np.einsum("kd,de,ke->k", xn[i] - cbn, hn, xn[i] - cbn)))
        for i in range(xn.shape[0])
    ]
    assert np.array_equal(got, np.array(want))


def test_h_identity_reduces_to_plain_nearest_centroid() -> None:
    """§3.2's premise: the query-aware and plain-VQ objectives coincide
    exactly when ``H ∝ I``. This pins that reduction."""
    from veloxquant_mlx.quantizers.a2ats import (
        a2ats_cholesky_factor, a2ats_h_weighted_assignment,
    )

    rng = np.random.default_rng(3)
    x = mx.array(rng.standard_normal((20, 4)).astype(np.float32))
    cb = mx.array(rng.standard_normal((16, 4)).astype(np.float32))
    ell = a2ats_cholesky_factor(mx.eye(4))

    got = np.array(a2ats_h_weighted_assignment(x, cb, ell))
    xn, cbn = np.array(x), np.array(cb)
    want = np.argmin(((xn[:, None, :] - cbn[None, :, :]) ** 2).sum(-1), axis=-1)
    assert np.array_equal(got, want)


def test_h_weighted_assignment_empty_input() -> None:
    from veloxquant_mlx.quantizers.a2ats import (
        a2ats_cholesky_factor, a2ats_h_weighted_assignment,
    )

    out = a2ats_h_weighted_assignment(
        mx.zeros((0, 4)), mx.ones((8, 4)), a2ats_cholesky_factor(mx.eye(4))
    )
    assert out.shape == (0,)
    assert out.dtype == mx.int32


def test_h_weighted_differs_from_plain_vq_under_anisotropic_h() -> None:
    """The whole point of Eq. (14): when ``H`` is strongly non-diagonal, the
    paper's assignment must actually diverge from plain nearest-centroid —
    otherwise §3.2's contribution would be vacuous."""
    from veloxquant_mlx.quantizers.a2ats import (
        a2ats_cholesky_factor, a2ats_h_weighted_assignment,
    )

    # Heavily anisotropic: dimension 0 matters ~100x more than dimension 1.
    h = mx.array(np.array([[100.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    ell = a2ats_cholesky_factor(h)
    x = mx.array(np.array([[1.0, 0.0]], dtype=np.float32))
    # c0 is closer in plain L2 (0.16 vs 0.5625); c1 is closer once dim 0 is
    # weighted 100x (16.0 vs 0.5625) — the assignment flips.
    cb = mx.array(np.array([[0.6, 0.0], [1.0, 0.75]], dtype=np.float32))

    plain = int(np.argmin(((np.array(x)[:, None, :] - np.array(cb)[None, :, :]) ** 2).sum(-1)[0]))
    weighted = int(np.array(a2ats_h_weighted_assignment(x, cb, ell))[0])
    assert plain == 0
    assert weighted == 1
