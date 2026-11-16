"""#969: per-head smooth factors must cancel between keys and queries under GQA.

``apply_dual_transform_keys`` averaged an ``[H_q, D]`` smooth (H_q != H_kv)
while ``apply_dual_transform_queries`` kept it per head, so
``q~ @ k~^T != q @ k^T``. The cache now reconciles smooth against H_kv on the
first update_and_fetch.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import mlx.core as mx
import pytest

from veloxquant_mlx.allocators.vecinfer import (
    apply_dual_transform_keys,
    apply_dual_transform_queries,
    walsh_hadamard_matrix,
)
from veloxquant_mlx.cache.vecinfer_cache import VecInferKVCache
from veloxquant_mlx.core.exceptions import QuantizerConfigError

D = 16
H_KV = 2
H_Q = 8


def _cache(smooth, **kw):
    cfg = SimpleNamespace(
        head_dim=D,
        key_sub_dim=4,
        value_sub_dim=4,
        key_codebook_bits=5,
        value_codebook_bits=5,
        smooth_factors=smooth,
        use_metal_kernels=False,
        **kw,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VecInferKVCache(cfg)


def _first_update(c):
    mx.random.seed(3)
    k = mx.random.normal((1, H_KV, 4, D))
    v = mx.random.normal((1, H_KV, 4, D))
    c.update_and_fetch(k, v)


def _score_err(smooth) -> float:
    """max |q.k - q~.k~| for H_q query heads sharing H_kv key heads."""
    mx.random.seed(1)
    H = walsh_hadamard_matrix(D)
    K = mx.random.normal((1, H_KV, 5, D))
    Q = mx.random.normal((1, H_Q, 1, D))
    Kt = apply_dual_transform_keys(K, smooth, H)
    Qt = apply_dual_transform_queries(Q, smooth, H)
    n_rep = H_Q // H_KV
    ref = Q @ mx.repeat(K, n_rep, axis=1).transpose(0, 1, 3, 2)
    got = Qt @ mx.repeat(Kt, n_rep, axis=1).transpose(0, 1, 3, 2)
    return float(mx.max(mx.abs(ref - got)))


def test_h_q_row_smooth_breaks_identity_without_reconciliation() -> None:
    """Documents the underlying mismatch the cache has to guard against."""
    mx.random.seed(0)
    assert _score_err(mx.random.uniform(0.5, 3.0, (H_Q, D))) > 1e-2


def test_h_q_row_smooth_is_reduced_to_kv_heads() -> None:
    mx.random.seed(0)
    sm = mx.random.uniform(0.5, 3.0, (H_Q, D))
    c = _cache(sm)
    _first_update(c)
    assert c._smooth.shape == (H_KV, D)
    n_rep = H_Q // H_KV
    expected = mx.mean(sm.reshape(H_KV, n_rep, D), axis=1)
    assert mx.allclose(c._smooth, expected).item()
    # With the reconciled smooth, keys and queries cancel again.
    assert _score_err(c._smooth) < 1e-4


def test_kv_head_rows_are_left_untouched() -> None:
    mx.random.seed(0)
    sm = mx.random.uniform(0.5, 3.0, (H_KV, D))
    c = _cache(sm)
    _first_update(c)
    assert mx.array_equal(c._smooth, sm)


@pytest.mark.parametrize("shape", [(D,), (1, D)])
def test_broadcast_smooth_is_left_untouched(shape) -> None:
    sm = mx.random.uniform(0.5, 3.0, shape)
    c = _cache(sm)
    _first_update(c)
    assert c._smooth.shape == shape


@pytest.mark.parametrize("rows", [3, 5])
def test_non_divisible_row_count_is_rejected(rows) -> None:
    c = _cache(mx.random.uniform(0.5, 3.0, (rows, D)))
    with pytest.raises(QuantizerConfigError, match="KV heads"):
        _first_update(c)


def test_fewer_rows_than_kv_heads_is_rejected() -> None:
    """Fewer smooth rows than KV heads (not 1): keys average, queries repeat, so reject."""
    c = _cache(mx.random.uniform(0.5, 3.0, (2, D)))
    mx.random.seed(0)
    k = mx.random.normal((1, 4, 3, D))
    with pytest.raises(QuantizerConfigError):
        c.update_and_fetch(k, k)
