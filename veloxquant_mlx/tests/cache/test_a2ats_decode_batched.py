"""a2ats decode assigns every head's token in one batched call (#846).

A single decode token is always its head's whole retrieval set, so the heap
selection, host sync and per-head loop in ``_quantize_head`` are pure
overhead there (9 ms/step at 8 heads vs 0.2 for kivi). The batched path must
pick the same centroids as the per-head path, for both the cosine-blend and the
calibrated H-weighted assignment.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache import a2ats_cache
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

D, H, B = 64, 8, 2


def _make(**extra):
    return KVCacheFactory.create(KVCacheConfig(method="a2ats", head_dim=D, **extra))


def _variants():
    sub_dim = _make()._sub_dim
    rng = np.random.default_rng(1)
    a = rng.standard_normal((sub_dim, sub_dim))
    h = (a @ a.T / sub_dim + np.eye(sub_dim)).astype(np.float32)
    return {"cosine": {}, "h_weighted": {"a2ats_query_h": mx.array(h)}}


@pytest.mark.parametrize("variant", ["cosine", "h_weighted"])
def test_batched_decode_matches_per_head_path(variant):
    cache = _make(**_variants()[variant])
    rng = np.random.default_rng(0)
    for _ in range(5):
        k = mx.array(rng.standard_normal((B, H, 1, D)).astype(np.float16))
        flat = k.astype(mx.float32).reshape(B * H, D)
        batched = cache._quantize_decode_batched(flat, mx.float16)
        pos = mx.zeros((1,), dtype=mx.int32)
        per_head = mx.stack(
            [cache._quantize_head(flat[i : i + 1], pos, out_dtype=mx.float16) for i in range(B * H)]
        ).reshape(B * H, D)
        np.testing.assert_array_equal(np.array(batched), np.array(per_head))


def test_decode_does_not_run_the_retrieval_heap(monkeypatch):
    cache = _make()
    k = mx.random.normal((1, H, 64, D)).astype(mx.float16)
    cache.update_and_fetch(k, k)  # prefill legitimately selects a retrieval set

    def boom(*a, **kw):
        raise AssertionError("single-token decode must not run the per-head selection")

    monkeypatch.setattr(a2ats_cache, "a2ats_select_retrieval_set", boom)
    t = mx.random.normal((1, H, 1, D)).astype(mx.float16)
    k_out, _ = cache.update_and_fetch(t, t)
    assert k_out.shape == (1, H, 65, D)
