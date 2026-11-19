"""Input validation / robustness of the GEAR numerics (``quantizers/gear.py``).

Before this, bad arguments were accepted silently (``sparse_frac=1.5`` kept the
wrong number of entries, ``rank=-1`` became rank 1) or died deep inside the SVD
with an unrelated message (``SVD failed with error code -4``), a ``GEARState``
without ``d_cols`` billed 0 bytes, and the batched sparse scatter computed its
per-row offsets in int32.
"""

from __future__ import annotations

import math
import os
from fractions import Fraction

import mlx.core as mx
import pytest

from veloxquant_mlx import KVCacheConfig
from veloxquant_mlx.cache.gear_cache import GEARKVCache
from veloxquant_mlx.cache.options import GEAROptions
from veloxquant_mlx.quantizers import gear as g


def _x(n=16, d=32, seed=0):
    mx.random.seed(seed)
    return mx.random.normal((n, d))


# ---------------------------------------------------------------- sparse frac
@pytest.mark.parametrize("frac", [1.5, 2.5, -0.1, float("nan"), float("inf")])
def test_sparse_outliers_rejects_bad_frac(frac) -> None:
    with pytest.raises(ValueError, match="sparse"):
        g.sparse_outliers(_x(), frac)


@pytest.mark.parametrize("frac", [1.5, -0.1, float("nan")])
def test_sparse_outliers_batched_rejects_bad_frac(frac) -> None:
    with pytest.raises(ValueError, match="sparse"):
        g.sparse_outliers_batched(mx.zeros((2, 16, 32)), frac)


def test_sparse_frac_one_keeps_every_entry() -> None:
    idx, val = g.sparse_outliers(_x(), 1.0)
    assert idx.shape[0] == 16 * 32 == val.shape[0]
    assert len(set(idx.tolist())) == 16 * 32


def test_sparse_frac_zero_is_still_no_term() -> None:
    assert g.sparse_outliers(_x(), 0.0) == (None, None)


def test_nnz_is_not_truncated_by_float_error() -> None:
    # int(100 * 0.29) == 28 in IEEE doubles; the decimal intent is 29.
    idx, _ = g.sparse_outliers(mx.random.normal((10, 10)), 0.29)
    assert idx.shape[0] == 29


def test_nnz_unchanged_for_default_fractions() -> None:
    for frac in (0.01, 0.02, 0.05, 0.1):
        exact = Fraction(str(frac))
        for n in range(1, 4097, 7):
            for d in (64, 128):
                assert g._nnz_for(n * d, frac) == math.floor(n * d * exact)
                # The old formula agrees on these, so defaults keep behaving.
                assert g._nnz_for(n * d, frac) == int(n * d * frac)


# ------------------------------------------------------------ rank/bits/group
def test_lowrank_error_rejects_negative_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        g.lowrank_error(_x(), -1)


def test_gear_compress_rejects_negative_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        g.gear_compress(_x(), bits=2, rank=-1)


@pytest.mark.parametrize("bits", [0, -1, 9, 2.5])
def test_gear_compress_rejects_bad_bits(bits) -> None:
    with pytest.raises(ValueError, match="bits"):
        g.gear_compress(_x(), bits=bits)


@pytest.mark.parametrize("gs", [0, -4])
def test_gear_compress_rejects_bad_group_size(gs) -> None:
    with pytest.raises(ValueError, match="group_size"):
        g.gear_compress(_x(), bits=2, group_size=gs)


def test_quantize_base_rejects_bad_bits_and_group_size() -> None:
    with pytest.raises(ValueError, match="bits"):
        g.quantize_base(_x(), 0)
    with pytest.raises(ValueError, match="group_size"):
        g.quantize_base(_x(), 2, group_size=0)


def test_gear_compress_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        g.gear_compress(mx.zeros((0, 32)), bits=2)


def test_gear_compress_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        g.gear_compress(mx.zeros((2, 4, 32)), bits=2)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_gear_compress_rejects_non_finite_input(bad) -> None:
    x = _x()
    x = mx.where(mx.arange(x.size).reshape(x.shape) == 5, bad, x)
    with pytest.raises(ValueError, match="finite"):
        g.gear_compress(x, bits=2)


def test_valid_arguments_still_work() -> None:
    x = _x(64, 32)
    for axis in ("channel", "token"):
        for bits in (1, 2, 8):
            s = g.gear_compress(x, bits=bits, rank=2, base_axis=axis)
            assert g.gear_reconstruct(s).shape == x.shape


# --------------------------------------------------------------- state d_cols
def test_state_without_d_cols_fails_loudly() -> None:
    s = g.gear_compress(_x(64, 32), bits=2, rank=2, base_axis="token")
    bad = s._replace(d_cols=0)
    with pytest.raises(ValueError, match="d_cols"):
        g.gear_reconstruct(bad)
    with pytest.raises(ValueError, match="d_cols"):
        g.gear_bytes(bad)
    with pytest.raises(ValueError, match="d_cols"):
        g.base_only_bytes(bad)


# ---------------------------------------------- batched scatter past 2**31 elements
def test_batched_reconstruct_builds_past_int32_elements() -> None:
    """Lazy graph only (nothing is evaluated, so no memory): 3 * 2**30 > 2**31 - 1.

    The old ``out.reshape(-1)`` raised ``OverflowError`` while building this.
    """
    bh, n, d = 3, 2**15, 2**15
    assert bh * n * d > 2**31 - 1
    base = mx.broadcast_to(mx.zeros(()), (bh, n, d))
    idx = mx.zeros((bh, 4), dtype=mx.int32)
    val = mx.ones((bh, 4), dtype=mx.float16)
    out = g.gear_reconstruct_batched(base, None, None, idx, val)
    assert out.shape == (bh, n, d)


@pytest.mark.skipif(
    os.environ.get("VELOX_BIG_MEMORY_TESTS") != "1",
    reason="needs ~11 GB of GPU memory; opt in with VELOX_BIG_MEMORY_TESTS=1",
)
def test_scatter_add_rows_correct_past_int32_elements() -> None:
    bh, n, d = 2, 8, 2**27 + 1  # per head 2**30 + 8 elements; total 2.15e9 > 2**31 - 1
    assert bh * n * d > 2**31 - 1
    out = mx.zeros((bh, n, d), dtype=mx.float16)
    per_head = n * d
    idx = mx.array([[3, per_head - 1], [7, per_head - 1]], dtype=mx.int32)
    val = mx.array([[1, 2], [3, 4]], dtype=mx.float16)
    res = g._scatter_add_rows(out, idx, val).reshape(bh, per_head)
    mx.eval(res)
    assert [res[0, 3].item(), res[0, per_head - 1].item()] == [1.0, 2.0]
    assert [res[1, 7].item(), res[1, per_head - 1].item()] == [3.0, 4.0]
    assert int(mx.sum(res != 0).item()) == 4


def test_batched_reconstruct_matches_per_row() -> None:
    bh, n, d = 3, 16, 32
    mx.random.seed(1)
    x = mx.random.normal((bh, n, d))
    resid = x - mx.round(x)
    idx, val = g.sparse_outliers_batched(resid, 0.05)
    base = mx.zeros((bh, n, d))
    out = g.gear_reconstruct_batched(base, None, None, idx, val, out_dtype=mx.float32)
    for i in range(bh):
        flat = mx.zeros((n * d,)).at[idx[i]].add(val[i].astype(mx.float32))
        assert mx.array_equal(out[i], flat.reshape(n, d))


# ------------------------------------------------------- config at construct
@pytest.mark.parametrize(
    "kw,match",
    [
        ({"gear_sparse_fraction": 1.5}, "sparse"),
        ({"gear_sparse_fraction": -0.1}, "sparse"),
        ({"gear_bits": 0}, "bits"),
        ({"gear_rank": -1}, "rank"),
        ({"gear_group_size": 0}, "group_size"),
    ],
)
def test_cache_rejects_bad_config_at_construction(kw, match) -> None:
    with pytest.raises(ValueError, match=match):
        GEARKVCache(KVCacheConfig(method="gear", head_dim=32, **kw))


def test_cache_default_config_still_constructs() -> None:
    GEARKVCache(KVCacheConfig(method="gear", head_dim=32))


@pytest.mark.parametrize("t", [0.0, -0.5, 1.5, float("nan"), float("inf")])
def test_energy_threshold_rejected_everywhere(t) -> None:
    with pytest.raises(ValueError, match="energy_threshold"):
        g.lowrank_error(_x(), None, t)
    with pytest.raises(ValueError, match="energy_threshold"):
        g.gear_compress(_x(), bits=2, energy_threshold=t)
    with pytest.raises(ValueError, match="gear_energy_threshold"):
        GEARKVCache(KVCacheConfig(method="gear", head_dim=32, gear_energy_threshold=t))


def test_energy_threshold_one_is_allowed() -> None:
    L, R = g.lowrank_error(_x(), None, 1.0)
    assert L.shape[1] == 16  # full rank of a 16x32 residual


# ------------------------------------------------ KVCacheConfig scope
def test_config_rejects_bad_gear_fields_at_construction() -> None:
    with pytest.raises(ValueError, match="gear_sparse_fraction"):
        KVCacheConfig(method="gear", head_dim=32, gear_sparse_fraction=1.5)
    with pytest.raises(ValueError, match="gear_bits"):
        KVCacheConfig(method="gear", head_dim=32, options=GEAROptions(gear_bits=0))


def test_config_for_other_methods_ignores_gear_fields() -> None:
    # Every config carries gear_* fields; only method="gear" reads them.
    KVCacheConfig(method="kivi", head_dim=32, gear_bits=0, gear_sparse_fraction=9.0)


def test_cache_still_validates_after_config_mutation() -> None:
    cfg = KVCacheConfig(method="gear", head_dim=32)
    with pytest.warns(DeprecationWarning):
        cfg.gear_bits = 0
    with pytest.raises(ValueError, match="gear_bits"):
        GEARKVCache(cfg)


# ------------------------------------------------ non-finite KV in the cache
def _kv(s, d=32):
    mx.random.seed(3)
    return mx.random.normal((1, 2, s, d)), mx.random.normal((1, 2, s, d))


def _poison(a, bad):
    flat = a.reshape(-1)
    flat = mx.where(mx.arange(flat.size) == 7, bad, flat)
    return flat.reshape(a.shape)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("which", ["keys", "values"])
def test_cache_rejects_non_finite_kv_before_svd(which, bad) -> None:
    c = GEARKVCache(KVCacheConfig(method="gear", head_dim=32))
    k, v = _kv(64)
    if which == "keys":
        k = _poison(k, bad)
    else:
        v = _poison(v, bad)
    with pytest.raises(ValueError, match=f"non-finite.*{which}"):
        c.update_and_fetch(k, v)


def test_cache_finite_prefill_and_decode_unaffected() -> None:
    c = GEARKVCache(KVCacheConfig(method="gear", head_dim=32))
    k, v = _kv(64)
    out_k, out_v = c.update_and_fetch(k, v)
    assert out_k.shape == k.shape and out_v.shape == v.shape
    k1, v1 = _kv(1)
    c.update_and_fetch(k1, v1)
    assert c.offset == 65


# ------------------------------------------------- whole-number floats keep working
def test_integral_floats_are_accepted_and_coerced() -> None:
    assert g._validate_rank(8.0) == 8 and isinstance(g._validate_rank(8.0), int)
    assert g._validate_bits(2.0) == 2 and isinstance(g._validate_bits(2.0), int)
    assert g._validate_group_size(32.0) == 32
    assert g._validate_rank(0.0) == 0


@pytest.mark.parametrize("bad", [8.5, float("nan"), float("inf"), True, "8"])
def test_fractional_or_non_numeric_ints_rejected(bad) -> None:
    with pytest.raises(ValueError, match="rank"):
        g._validate_rank(bad)


def test_cache_accepts_float_valued_integer_config() -> None:
    cfg = KVCacheConfig(
        method="gear", head_dim=32, gear_rank=8.0, gear_bits=2.0, gear_group_size=32.0
    )
    c = GEARKVCache(cfg)
    assert (c._rank, c._bits, c._gs) == (8, 2, 32)
    assert all(isinstance(v, int) for v in (c._rank, c._bits, c._gs))
    k, v = _kv(64)
    out_k, _ = c.update_and_fetch(k, v)
    assert out_k.shape == k.shape


def test_gear_compress_accepts_integral_floats() -> None:
    s = g.gear_compress(_x(64, 32), bits=2.0, rank=4.0, group_size=32.0)
    assert s.rank == 4 and s.bits == 2
