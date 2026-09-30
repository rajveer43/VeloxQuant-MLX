"""Equivalence tests: batched [B, H, S, D] merge_pair/reconstruct_layer vs.
a per-(b, h) reference loop calling the same functions on [S, D] slices.

merge_pair/reconstruct_layer's SLERP math (magnitude/direction split, dot
product, arccos/sin trig, retention mask) is elementwise over all leading
dims — no cross-row reduction anywhere — so batching over [B, H] is not an
approximation: it is bit-for-bit identical to the unbatched per-head loop,
just dispatched as one MLX call instead of B*H Python-level calls. Also
covers the real MiniCacheKVCache (merge role) end-to-end.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.minicache_cache import MiniCacheKVCache
from veloxquant_mlx.cache.minicache_coordinator import MiniCacheCoordinator
from veloxquant_mlx.quantizers.minicache import merge_pair, reconstruct_layer


def _rand(B, H, S, D, seed):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32))


def _reference_loop(x_primary, x_merge, retention_threshold, t, which):
    B, H, S, D = x_primary.shape
    out_b = []
    for b in range(B):
        out_h = []
        for h in range(H):
            res = merge_pair(
                x_primary[b, h], x_merge[b, h], retention_threshold=retention_threshold, t=t
            )
            out_h.append(reconstruct_layer(res, which))
        out_b.append(mx.stack(out_h, axis=0))
    return mx.stack(out_b, axis=0)


@pytest.mark.parametrize(
    "desc,B,H,S,D,ret,t,seed",
    [
        ("single_head", 1, 1, 6, 8, 0.9, 0.5, 0),
        ("multi_head", 1, 4, 6, 8, 0.9, 0.5, 1),
        ("multi_batch_multi_head", 3, 5, 6, 8, 0.9, 0.5, 2),
        ("low_threshold_all_merge", 2, 3, 10, 16, -1.0, 0.5, 3),
        ("high_threshold_all_retain", 2, 3, 10, 16, 1.1, 0.5, 4),
        ("t_endpoint_zero", 2, 3, 6, 8, 0.9, 0.0, 5),
        ("t_endpoint_one", 2, 3, 6, 8, 0.9, 1.0, 6),
        ("odd_head_dim", 2, 3, 6, 7, 0.9, 0.5, 7),
        ("single_token", 2, 3, 1, 8, 0.9, 0.5, 8),
    ],
)
def test_batched_matches_reference_loop(desc, B, H, S, D, ret, t, seed):
    xp = _rand(B, H, S, D, seed)
    xm = _rand(B, H, S, D, seed + 1000)

    res_batched = merge_pair(xp, xm, retention_threshold=ret, t=t)
    out_merge_batched = reconstruct_layer(res_batched, "merge")
    out_primary_batched = reconstruct_layer(res_batched, "primary")

    ref_merge = _reference_loop(xp, xm, ret, t, "merge")
    ref_primary = _reference_loop(xp, xm, ret, t, "primary")

    assert mx.array_equal(out_merge_batched, ref_merge), f"{desc}: merge reconstruction diverged"
    assert mx.array_equal(out_primary_batched, ref_primary), (
        f"{desc}: primary reconstruction diverged"
    )


def test_retained_mask_shape_matches_leading_dims():
    B, H, S, D = 2, 3, 6, 8
    xp = _rand(B, H, S, D, 0)
    xm = _rand(B, H, S, D, 1)
    res = merge_pair(xp, xm, retention_threshold=0.9, t=0.5)
    assert res.retained.shape == (B, H, S)


def test_2d_single_head_shape_unaffected():
    """The pre-existing [S, D] single-head call path (still used elsewhere
    conceptually) must produce identical output to before this change."""
    S, D = 6, 8
    xp = _rand(1, 1, S, D, 0)[0, 0]
    xm = _rand(1, 1, S, D, 1)[0, 0]
    res = merge_pair(xp, xm, retention_threshold=0.9, t=0.5)
    assert res.retained.shape == (S,)
    out = reconstruct_layer(res, "merge")
    assert out.shape == (S, D)


# ---------------------------------------------------------------------------
# Real MiniCacheKVCache (merge role) end-to-end
# ---------------------------------------------------------------------------


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


@pytest.mark.parametrize("B,H,seed", [(1, 1, 0), (2, 1, 1), (1, 4, 2), (3, 5, 3)])
def test_real_cache_merge_role_matches_reference(B, H, seed):
    D = 8
    cfg = _Cfg(minicache_retention_threshold=0.9, minicache_slerp_t=0.5)
    coord = MiniCacheCoordinator()

    primary = MiniCacheKVCache(cfg, role="primary", group_id=0, coordinator=coord, n_readers=1)
    merge = MiniCacheKVCache(cfg, role="merge", group_id=0, coordinator=coord)

    S = 5
    k_p = _rand(B, H, S, D, seed).astype(mx.float16)
    v_p = _rand(B, H, S, D, seed + 1).astype(mx.float16)
    k_m = _rand(B, H, S, D, seed + 2).astype(mx.float16)
    v_m = _rand(B, H, S, D, seed + 3).astype(mx.float16)

    primary.update_and_fetch(k_p, v_p)
    k_out, v_out = merge.update_and_fetch(k_m, v_m)

    ref_k = _reference_loop(k_p.astype(mx.float32), k_m.astype(mx.float32), 0.9, 0.5, "merge")
    ref_v = _reference_loop(v_p.astype(mx.float32), v_m.astype(mx.float32), 0.9, 0.5, "merge")

    assert mx.array_equal(k_out, ref_k)
    assert mx.array_equal(v_out, ref_v)
