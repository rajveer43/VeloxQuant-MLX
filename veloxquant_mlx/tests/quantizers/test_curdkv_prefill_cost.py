"""CurDKV prefill cost: leverage scores must not need a per-token SVD.

Prefill scores every token twice, each time with an SVD of an [n, D] block per
head, which was ~85% of the 16 ms/token prefill cost. The score only needs the
top right singular directions, so it is computed from the [D, D] Gram matrix.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers import curdkv as cd


def _inputs(bh=3, n=40, d=16, seed=0):
    rng = np.random.default_rng(seed)
    q = mx.array(rng.standard_normal((bh, d)).astype(np.float32))
    k = mx.array(rng.standard_normal((bh, n, d)).astype(np.float32))
    v = mx.array(rng.standard_normal((bh, n, d)).astype(np.float32))
    return q, k, v


@pytest.mark.parametrize("rank_cap", [4, 16, 64])
def test_batched_scores_match_per_head_svd_reference(rank_cap) -> None:
    q, k, v = _inputs()
    got = np.array(cd._leverage_scores_batched(q, k, v, rank_cap))
    want = np.stack(
        [np.array(cd._leverage_scores(q[i], k[i], v[i], rank_cap)) for i in range(q.shape[0])]
    )
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_batched_scores_do_not_run_an_svd(monkeypatch) -> None:
    q, k, v = _inputs()

    def boom(*a, **kw):
        raise AssertionError("per-token leverage scores must not run an SVD")

    monkeypatch.setattr(np.linalg, "svd", boom)
    scores = np.array(cd._leverage_scores_batched(q, k, v, 16))
    assert scores.shape == (3, 40) and np.all(scores >= 0)
