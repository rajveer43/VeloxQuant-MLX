"""SVDq input validation: bad arguments fail loudly and early, valid ones are unchanged.

Before this, ``svdq_rank=-3`` silently sliced the basis from the end, ``svdq_rank=0``
blamed the bit schedule, ``energy_threshold<=0`` collapsed every head to rank 1 and
``>1``/NaN silently disabled compression, ``svdq_group_size=1.5`` was truncated to 1,
``svdq_group_size=0`` was a ``ZeroDivisionError``, bit-schedule entries above 8, NaN or
strings failed late with opaque ``TypeError`` (or were accepted), an all-zero schedule
returned only the mean key, and one NaN in a prefill chunk was folded into the running
Gram matrix, so that head stayed NaN for every later chunk, clean or not. All of that is
rejected with a ``QuantizerConfigError`` (a ``ValueError``) that names the argument.
"""

from __future__ import annotations

import warnings

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.cache.svdq_cache import SVDqKVCache
from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.quantizers.svdq import (
    DEFAULT_BIT_SCHEDULE,
    basis_from_gram_batched,
    equivalent_bit_width,
    latent_group_slices,
    project_quantize_reconstruct_batched,
    quantize_latents_mixed,
    svd_compress_keys,
    svd_compress_keys_batched,
    validate_svdq_config,
)

H, S, D = 4, 128, 64


def _keys(seed: int = 0, h: int = H, s: int = S) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((h, s, D)) @ np.diag(np.linspace(3, 0.05, D))).astype(np.float16)


def _cfg(**kw) -> KVCacheConfig:
    return KVCacheConfig(method="svdq", head_dim=D, **kw)


def _cache(**kw) -> SVDqKVCache:
    return SVDqKVCache(_cfg(**kw))


def _feed(cache: SVDqKVCache, arr: np.ndarray) -> np.ndarray:
    k = mx.array(arr[None])
    out, _ = cache.update_and_fetch(k, k)
    mx.eval(out)
    return np.asarray(out[0].astype(mx.float32))


BAD_RANKS = [0, -1, -3, True, 2.5, "8", float("nan"), float("inf")]
BAD_ENERGY = [0.0, -1.0, 1.5, float("nan"), float("inf"), True, "0.9"]
BAD_GROUP_SIZES = [0, -1, 1.5, True, "32", float("nan")]
BAD_SCHEDULES = [
    (9, 4, 2, 1, 1, 0, 0, 0),
    (16,) * 8,
    (8, 4, 2.5, 1, 1, 0, 0, 0),
    (True, 4, 2, 1, 1, 0, 0, 0),
    ("4",) * 8,
    (float("nan"),) * 8,
    (8, 4, 2, -1, 1, 0, 0, 0),
    (),
    [],
    4,
    "84211000",
    None,
    (0,) * 8,  # truncates every group: stores nothing
]


# --------------------------------------------------------------------------- config / cache
@pytest.mark.parametrize("rank", BAD_RANKS)
def test_config_rejects_bad_rank(rank) -> None:
    with pytest.raises(ValueError, match="svdq_rank"):
        _cfg(svdq_rank=rank)


@pytest.mark.parametrize("t", BAD_ENERGY)
def test_config_rejects_bad_energy(t) -> None:
    with pytest.raises(ValueError, match="svdq_energy_threshold"):
        _cfg(svdq_energy_threshold=t)


@pytest.mark.parametrize("g", BAD_GROUP_SIZES)
def test_config_rejects_bad_group_size(g) -> None:
    with pytest.raises(ValueError, match="svdq_group_size"):
        _cfg(svdq_group_size=g)


@pytest.mark.parametrize("sched", BAD_SCHEDULES, ids=repr)
def test_config_rejects_bad_schedule(sched) -> None:
    with pytest.raises(ValueError, match="svdq_bit_schedule"):
        _cfg(svdq_bit_schedule=sched)


def test_errors_are_quantizer_config_errors() -> None:
    with pytest.raises(QuantizerConfigError):
        _cfg(svdq_rank=0)


def test_cache_rechecks_after_config_mutation() -> None:
    """The cache validates for itself: a config edited after construction is not trusted."""
    cfg = _cfg()
    cfg.svdq_group_size = 0
    with pytest.raises(ValueError, match="svdq_group_size"):
        SVDqKVCache(cfg)
    cfg = _cfg()
    cfg.svdq_bit_schedule = (9,) * 8
    with pytest.raises(ValueError, match="svdq_bit_schedule"):
        SVDqKVCache(cfg)


def test_other_methods_ignore_svdq_fields() -> None:
    """Every config carries the svdq_* fields; only method='svdq' reads them."""
    KVCacheConfig(method="kivi", head_dim=D, svdq_rank=-3, svdq_group_size=0)


def test_valid_extremes_accepted() -> None:
    _cache(svdq_rank=1)
    _cache(svdq_rank=10**6)  # above the available rank: clamped, not an error
    _cache(svdq_energy_threshold=1.0)
    _cache(svdq_energy_threshold=1e-6)
    _cache(svdq_group_size=1)
    _cache(svdq_bit_schedule=(8,) * 8)
    _cache(svdq_bit_schedule=(4,), svdq_rank=4)
    _cache(svdq_bit_schedule=(8, 0))
    _cache(svdq_bit_schedule=(8, 4, 2, 1, 1) + (0,) * 20)


def test_validate_config_defaults_and_coercion() -> None:
    assert validate_svdq_config(object()) == {
        "rank": None,
        "energy": 0.95,
        "bit_schedule": DEFAULT_BIT_SCHEDULE,
        "group_size": 32,
    }
    cfg = _cfg(
        svdq_rank=np.int64(32),
        svdq_group_size=np.int32(16),
        svdq_bit_schedule=[8.0, 4, np.int64(2), 1, 1, 0, 0, 0],
        svdq_energy_threshold=np.float32(0.5),
    )
    out = validate_svdq_config(cfg)
    assert out["rank"] == 32 and type(out["rank"]) is int
    assert out["group_size"] == 16 and type(out["group_size"]) is int
    assert out["bit_schedule"] == DEFAULT_BIT_SCHEDULE
    assert all(type(b) is int for b in out["bit_schedule"])
    assert out["energy"] == pytest.approx(0.5)
    out = validate_svdq_config(_cfg(svdq_bit_schedule=np.array([8, 4, 2, 1, 1, 0, 0, 0])))
    assert out["bit_schedule"] == DEFAULT_BIT_SCHEDULE


def test_integral_floats_behave_exactly_like_ints() -> None:
    """``32.0`` / ``8.0`` (as a JSON config carries them) keep working and change nothing."""
    keys = _keys()
    ref = _feed(_cache(svdq_rank=32, svdq_group_size=16), keys)
    got = _feed(
        _cache(
            svdq_rank=32.0,
            svdq_group_size=16.0,
            svdq_bit_schedule=(8.0, 4.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        ),
        keys,
    )
    assert np.array_equal(ref, got)


def test_valid_default_behaviour_is_pinned() -> None:
    """Characterisation measured on the pre-change tree: same rank and bit accounting."""
    c = _cache()
    out = _feed(c, _keys())
    assert c.rank == 48
    assert c.assigned_avg_bits == pytest.approx(1.5)
    assert not np.isnan(out).any()


# --------------------------------------------------------------------------- non-finite prefill
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_prefill_rejects_non_finite_keys(bad) -> None:
    arr = _keys()
    arr[1, 5, 7] = bad
    with pytest.raises(QuantizerConfigError, match="finite"):
        _feed(_cache(), arr)


def test_rejected_chunk_leaves_cache_untouched() -> None:
    """The old failure: one NaN chunk poisoned the running Gram, so head 1 stayed NaN."""
    clean = _keys(seed=1)
    bad = _keys(seed=2)
    bad[1, 5, 7] = float("nan")

    victim = _cache()
    with pytest.raises(QuantizerConfigError):
        _feed(victim, bad)
    assert victim._gram is None and victim._row_sum is None and victim._n_rows == 0
    got = _feed(victim, clean)

    fresh = _cache()
    assert np.array_equal(got, _feed(fresh, clean))
    assert not np.isnan(got).any()


def test_rejected_later_chunk_keeps_earlier_state() -> None:
    first, second = _keys(seed=1), _keys(seed=2)
    bad = _keys(seed=3)
    bad[0, 0, 0] = float("inf")

    victim = _cache()
    _feed(victim, first)
    n_rows, gram = victim._n_rows, victim._gram.copy()
    with pytest.raises(QuantizerConfigError):
        _feed(victim, bad)
    assert victim._n_rows == n_rows and np.array_equal(victim._gram, gram)

    clean = _cache()
    _feed(clean, first)
    assert np.array_equal(_feed(victim, second), _feed(clean, second))


def test_decode_step_is_not_scanned_and_does_not_poison() -> None:
    """Decode rows are not checked (that would be a host sync per token), and a bad
    decode row never reaches the fitted basis, so it damages only its own token."""
    c = _cache()
    _feed(c, _keys())
    row = _keys(seed=9, s=1)
    nan_row = row.copy()
    nan_row[2, 0, 3] = float("nan")
    out = _feed(c, nan_row)  # must not raise; returns the whole cache, newest token last
    assert np.isnan(out[2, -1]).any() and not np.isnan(out[[0, 1, 3]]).any()
    after = _feed(c, _keys(seed=10, s=1))
    assert not np.isnan(after[:, -1]).any()  # the next token is clean in every head
    assert not np.isnan(after[[0, 1, 3]]).any()
    assert np.isnan(after[2]).any(axis=-1).sum() == 1  # head 2: only the one bad token


def test_single_row_first_call_still_passes_through() -> None:
    """S == 1 before any fit is held raw (no basis is fitted), as before."""
    c = _cache()
    out = _feed(c, _keys(s=1))
    assert out.shape == (H, 1, D)


# --------------------------------------------------------------------------- standalone functions
@pytest.mark.parametrize("rank", BAD_RANKS)
def test_svd_functions_reject_bad_rank(rank) -> None:
    keys = mx.array(_keys())
    with pytest.raises(ValueError, match="rank"):
        svd_compress_keys(keys[0], rank=rank)
    with pytest.raises(ValueError, match="rank"):
        svd_compress_keys_batched(keys, rank=rank)


@pytest.mark.parametrize("t", BAD_ENERGY)
def test_svd_functions_reject_bad_energy(t) -> None:
    keys = mx.array(_keys())
    with pytest.raises(ValueError, match="energy_threshold"):
        svd_compress_keys(keys[0], energy_threshold=t)
    with pytest.raises(ValueError, match="energy_threshold"):
        svd_compress_keys_batched(keys, energy_threshold=t)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_svd_functions_reject_non_finite(bad) -> None:
    arr = _keys().astype(np.float32)
    arr[0, 3, 2] = bad
    with pytest.raises(QuantizerConfigError, match="finite"):
        svd_compress_keys(mx.array(arr[0]))
    with pytest.raises(QuantizerConfigError, match="finite"):
        svd_compress_keys_batched(mx.array(arr))


def test_svd_functions_reject_wrong_shape_and_empty() -> None:
    with pytest.raises(ValueError, match="2-D"):
        svd_compress_keys(mx.zeros((H, S, D)))
    with pytest.raises(ValueError, match="3-D"):
        svd_compress_keys_batched(mx.zeros((S, D)))
    with pytest.raises(ValueError, match="non-empty"):
        svd_compress_keys(mx.zeros((0, D)))
    with pytest.raises(ValueError, match="non-empty"):
        svd_compress_keys_batched(mx.zeros((H, 0, D)))


def test_basis_from_gram_rejects_bad_arguments() -> None:
    x = _keys().astype(np.float64)
    gram = np.swapaxes(x, 1, 2) @ x
    row_sum = x.sum(axis=1)
    basis_from_gram_batched(gram, row_sum, S)  # valid
    with pytest.raises(ValueError, match="rank"):
        basis_from_gram_batched(gram, row_sum, S, rank=0)
    with pytest.raises(ValueError, match="energy_threshold"):
        basis_from_gram_batched(gram, row_sum, S, energy_threshold=0.0)
    with pytest.raises(ValueError, match="n_rows"):
        basis_from_gram_batched(gram, row_sum, 0)
    poisoned = gram.copy()
    poisoned[1, 2, 2] = np.nan
    with pytest.raises(QuantizerConfigError, match="finite"):
        basis_from_gram_batched(poisoned, row_sum, S)


@pytest.mark.parametrize("n", [0, -2, 1.5, True, "8"])
def test_latent_group_slices_rejects_bad_n_groups(n) -> None:
    with pytest.raises(ValueError, match="n_groups"):
        latent_group_slices(16, n)


def test_latent_group_slices_valid_unchanged() -> None:
    assert latent_group_slices(16, 8) == [(2 * i, 2 * i + 2) for i in range(8)]
    assert latent_group_slices(0, 8) == []
    assert latent_group_slices(3, 8) == [(0, 1), (1, 2), (2, 3)]
    assert latent_group_slices(16, 4.0) == [(0, 4), (4, 8), (8, 12), (12, 16)]


def test_latent_group_slices_cache_is_typed_and_safe() -> None:
    """Results are cached per (r, n_groups); a cached ``1`` must not answer for ``True``,
    unhashable arguments are still rejected, and callers cannot corrupt the cache."""
    assert latent_group_slices(16, 1) == [(0, 16)]  # caches the valid pair
    with pytest.raises(ValueError, match="n_groups"):
        latent_group_slices(16, True)
    with pytest.raises(ValueError, match="n_groups"):
        latent_group_slices(16, [8])  # unhashable
    first = latent_group_slices(16, 8)
    first.append((99, 99))
    assert latent_group_slices(16, 8) == [(2 * i, 2 * i + 2) for i in range(8)]


@pytest.mark.parametrize("g", BAD_GROUP_SIZES)
def test_quantize_latents_rejects_bad_group_size(g) -> None:
    with pytest.raises(ValueError, match="group_size"):
        quantize_latents_mixed(mx.ones((8, 16)), mx.ones(16), group_size=g)


@pytest.mark.parametrize("sched", [s for s in BAD_SCHEDULES if s != (0,) * 8], ids=repr)
def test_quantize_latents_rejects_bad_schedule(sched) -> None:
    with pytest.raises(ValueError, match="bit_schedule"):
        quantize_latents_mixed(mx.ones((8, 16)), mx.ones(16), bit_schedule=sched)


def test_low_level_functions_allow_an_all_zero_schedule() -> None:
    """Only the config rejects it; the low-level quantizer just truncates every group."""
    out = quantize_latents_mixed(mx.ones((8, 16)), mx.ones(16), bit_schedule=(0,) * 8)
    assert float(mx.abs(out).max()) == 0.0
    assert equivalent_bit_width(16, (0,) * 8) == 0.0


def test_typed_schedule_cache_does_not_let_bool_through() -> None:
    """``(True, ...) == (1, ...)`` for hashing: a cached valid schedule must not admit it."""
    L, sv = mx.ones((8, 16)), mx.ones(16)
    quantize_latents_mixed(L, sv, bit_schedule=(1, 4, 2, 1, 1, 0, 0, 0))  # caches the valid one
    with pytest.raises(ValueError, match="bit_schedule"):
        quantize_latents_mixed(L, sv, bit_schedule=(True, 4, 2, 1, 1, 0, 0, 0))
    equivalent_bit_width(16, (1, 1))
    with pytest.raises(ValueError, match="bit_schedule"):
        equivalent_bit_width(16, (True, True))
    with pytest.raises(ValueError, match="bit_schedule"):
        equivalent_bit_width(16, [[1], [1]])  # unhashable entries take the uncached path


@pytest.mark.parametrize("sched", [(), [], (9,), (2.5,)], ids=repr)
def test_equivalent_bit_width_rejects_bad_schedule(sched) -> None:
    with pytest.raises(ValueError, match="bit_schedule"):
        equivalent_bit_width(16, sched)


def test_equivalent_bit_width_valid_unchanged() -> None:
    assert equivalent_bit_width(64) == 2.0
    assert equivalent_bit_width(3, DEFAULT_BIT_SCHEDULE) == pytest.approx((8 + 4 + 2) / 3)
    assert equivalent_bit_width(16, [8.0, 4.0]) == 6.0


def _basis(h: int = H):
    keys = mx.array(_keys(h=h))
    _, V, mu, _ = svd_compress_keys_batched(keys, rank=32)
    return keys, V, mu


def test_project_quantize_reconstruct_validates() -> None:
    keys, V, mu = _basis()
    sched = [DEFAULT_BIT_SCHEDULE] * H
    ok = project_quantize_reconstruct_batched(keys, V, mu, sched, 16)
    assert ok.shape == (H, S, D)
    with pytest.raises(ValueError, match="group_size"):
        project_quantize_reconstruct_batched(keys, V, mu, sched, 0)
    with pytest.raises(ValueError, match="group_size"):
        project_quantize_reconstruct_batched(keys, V, mu, sched, 2.5)
    bad = list(sched)
    bad[3] = (9, 4, 2, 1, 1, 0, 0, 0)  # only the last head is bad
    with pytest.raises(ValueError, match="bit_schedule"):
        project_quantize_reconstruct_batched(keys, V, mu, bad, 16)
    with pytest.raises(ValueError, match="one entry per head"):
        project_quantize_reconstruct_batched(keys, V, mu, sched[:-1], 16)
    with pytest.raises(ValueError, match="one entry per head"):
        project_quantize_reconstruct_batched(keys, V[:-1], mu, sched, 16)


def test_project_quantize_reconstruct_mixed_valid_schedules_unchanged() -> None:
    """Heads with different valid schedules (list or tuple) still group and reconstruct."""
    keys, V, mu = _basis()
    scheds = [DEFAULT_BIT_SCHEDULE, [4, 4, 4, 4, 4, 4, 4, 4], DEFAULT_BIT_SCHEDULE, (8.0,) * 8]
    mixed = project_quantize_reconstruct_batched(keys, V, mu, scheds, 16)
    for h in range(H):
        solo = project_quantize_reconstruct_batched(
            keys[h : h + 1], [V[h]], [mu[h]], [tuple(int(b) for b in scheds[h])], 16
        )
        assert np.array_equal(np.asarray(mixed[h]), np.asarray(solo[0]))


def test_project_quantize_reconstruct_validate_flag_only_skips_checks() -> None:
    """The cache's decode path passes validate=False: same output on valid input."""
    keys, V, mu = _basis()
    sched = [DEFAULT_BIT_SCHEDULE] * H
    checked = project_quantize_reconstruct_batched(keys, V, mu, sched, 16)
    unchecked = project_quantize_reconstruct_batched(keys, V, mu, sched, 16, validate=False)
    assert np.array_equal(np.asarray(checked), np.asarray(unchecked))


# --------------------------------------------------------------------------- no stray noise
def test_valid_config_emits_no_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _feed(_cache(svdq_rank=32, svdq_group_size=16), _keys())
