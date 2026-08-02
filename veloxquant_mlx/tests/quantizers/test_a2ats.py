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
    cb = mx.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # nearest to x
            [0.0, 0.0, 0.0, 1.0],  # far from x, but aligned with an adversarial query
        ],
        dtype=mx.float32,
    )
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
    cb = mx.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # nearest to x by L2, but orthogonal-ish to query
            [0.0, 0.0, 0.0, 1.0],  # far from x, but exactly aligned with query
        ],
        dtype=mx.float32,
    )
    query = mx.array([0.0, 0.0, 0.0, 1.0], dtype=mx.float32)

    idx_nearest_only = a2ats_query_aware_assignment(x, cb, query, beta=1.0)
    idx_query_dominant = a2ats_query_aware_assignment(x, cb, query, beta=0.0)

    assert int(idx_nearest_only[0].item()) == 0  # pure reconstruction picks centroid 0
    assert int(idx_query_dominant[0].item()) == 1  # pure query-alignment picks centroid 1


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
    keys = mx.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # most similar (parallel)
            [0.9, 0.1, 0.0, 0.0],  # second most similar
            [0.0, 1.0, 0.0, 0.0],  # orthogonal
            [-1.0, 0.0, 0.0, 0.0],  # anti-parallel (least similar)
        ],
        dtype=mx.float32,
    )
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
