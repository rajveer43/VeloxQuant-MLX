"""PALU input validation: bad arguments fail loudly and early, valid ones are unchanged.

Before this, ``rank=0``/``-1``/``True`` silently became rank 1, ``hi_bit=0`` produced an
all-NaN reconstruction with ``assigned_avg_bits == 0``, one NaN in the prefill keys
turned a whole head group's reconstruction into NaN, and ``energy_threshold=nan``
silently selected full rank (no compression). All of that is rejected with a
``QuantizerConfigError`` (a ``ValueError``) that names the argument.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.cache.palu_cache import PALUKVCache, _TensorLowRank
from veloxquant_mlx.quantizers.palu import (
    group_head_svd,
    n_hi_channels,
    quantize_latent,
    quantize_latent_batched,
    validate_palu_config,
)

D = 32


def _rnd(*shape: int, seed: int = 0) -> mx.array:
    mx.random.seed(seed)
    return mx.random.normal(shape).astype(mx.float16)


def _cache(**kw) -> PALUKVCache:
    return PALUKVCache(KVCacheConfig(method="palu", head_dim=D, **kw))


def _run(cache: PALUKVCache, H: int = 8, S: int = 64) -> tuple[mx.array, mx.array]:
    k, v = _rnd(1, H, S, D, seed=1), _rnd(1, H, S, D, seed=2)
    ko, vo = cache.update_and_fetch(k, v)
    mx.eval(ko, vo)
    return ko, vo


BAD_RANKS = [0, -1, -5, True, 2.5, "4", float("nan"), float("inf")]
BAD_ENERGY = [0.0, -0.5, 1.5, float("nan"), float("inf"), True, "0.9"]
BAD_BITS = [0, -1, 9, 4.5, True, "4", float("nan")]
BAD_GROUP_SIZES = [0, -4, 2.5, True, "32", float("nan")]
BAD_FRACS = [-0.1, 1.5, float("nan"), float("inf"), True, "0.25"]


# --------------------------------------------------------------------------- group_head_svd
@pytest.mark.parametrize("rank", BAD_RANKS)
def test_svd_rejects_bad_rank(rank) -> None:
    with pytest.raises(ValueError, match="rank"):
        group_head_svd(_rnd(4, 64, D), rank=rank)


@pytest.mark.parametrize("et", BAD_ENERGY)
def test_svd_rejects_bad_energy_threshold(et) -> None:
    with pytest.raises(ValueError, match="energy_threshold"):
        group_head_svd(_rnd(4, 64, D), energy_threshold=et)


def test_svd_integral_float_rank_equals_int_rank() -> None:
    x = _rnd(4, 64, D)
    a = group_head_svd(x, rank=8)
    b = group_head_svd(x, rank=8.0)
    assert a[0].shape == b[0].shape == (D, 8)
    assert bool(mx.array_equal(a[0], b[0]))


def test_svd_rank_above_available_is_still_clamped() -> None:
    # Only rank >= 1 is required; asking for more than D just gives full rank.
    V, _, sv = group_head_svd(_rnd(4, 64, D), rank=1000)
    assert V.shape == (D, D) and sv.shape == (D,)


def test_svd_energy_threshold_one_is_allowed() -> None:
    V, _, _ = group_head_svd(_rnd(4, 64, D), energy_threshold=1.0)
    assert V.shape == (D, D)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_svd_rejects_non_finite_input(bad) -> None:
    x = np.array(_rnd(4, 64, D)).astype(np.float32)
    x[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        group_head_svd(mx.array(x))


@pytest.mark.parametrize("shape", [(4, 0, D), (0, 8, D), (4, 8, 0), (8, D), (D,)])
def test_svd_rejects_empty_or_wrong_rank_input(shape) -> None:
    with pytest.raises(ValueError, match=r"\[G, S, D\]"):
        group_head_svd(mx.zeros(shape))


# --------------------------------------------------------------------------- quantize_latent
def _args(**over):
    return {"hi_bit": 4, "lo_bit": 2, "hi_fraction": 0.25, "group_size": 16, **over}


@pytest.mark.parametrize(
    "fn,shape", [(quantize_latent, (64, 16)), (quantize_latent_batched, (2, 64, 16))]
)
class TestQuantizeLatentArgs:
    def _call(self, fn, shape, **over):
        sv = mx.arange(16, 0, -1).astype(mx.float32)
        return fn(_rnd(*shape), sv, **_args(**over))

    @pytest.mark.parametrize("bits", BAD_BITS)
    def test_bad_hi_bit(self, fn, shape, bits) -> None:
        with pytest.raises(ValueError, match="hi_bit"):
            self._call(fn, shape, hi_bit=bits)

    @pytest.mark.parametrize("bits", BAD_BITS)
    def test_bad_lo_bit(self, fn, shape, bits) -> None:
        with pytest.raises(ValueError, match="lo_bit"):
            self._call(fn, shape, lo_bit=bits)

    @pytest.mark.parametrize("gs", BAD_GROUP_SIZES)
    def test_bad_group_size(self, fn, shape, gs) -> None:
        with pytest.raises(ValueError, match="group_size"):
            self._call(fn, shape, group_size=gs)

    @pytest.mark.parametrize("frac", BAD_FRACS)
    def test_bad_hi_fraction(self, fn, shape, frac) -> None:
        with pytest.raises(ValueError, match="hi_fraction"):
            self._call(fn, shape, hi_fraction=frac)

    def test_singular_values_length_mismatch(self, fn, shape) -> None:
        with pytest.raises(ValueError, match="singular_values"):
            fn(_rnd(*shape), mx.ones((5,)), **_args())

    @pytest.mark.parametrize("bits", [1, 8, 4.0])
    def test_boundary_and_integral_float_bits_work(self, fn, shape, bits) -> None:
        out = self._call(fn, shape, hi_bit=bits, lo_bit=bits)
        assert out.shape == shape and bool(mx.all(mx.isfinite(out)).item())


def test_validation_cache_does_not_let_bool_pass_as_int() -> None:
    """The constant-argument validation is cached; True == 1 must not share a cache entry."""
    sv = mx.arange(16, 0, -1).astype(mx.float32)
    quantize_latent_batched(
        _rnd(2, 64, 16), sv, **_args(hi_bit=1, lo_bit=1, group_size=1)
    )  # primes the cache
    with pytest.raises(ValueError, match="hi_bit"):
        quantize_latent_batched(_rnd(2, 64, 16), sv, **_args(hi_bit=True, lo_bit=1, group_size=1))
    with pytest.raises(ValueError, match="group_size"):
        quantize_latent_batched(_rnd(2, 64, 16), sv, **_args(hi_bit=1, lo_bit=1, group_size=True))


def test_quantize_latent_unhashable_argument_is_a_clear_error() -> None:
    sv = mx.arange(16, 0, -1).astype(mx.float32)
    with pytest.raises(ValueError, match="hi_bit"):
        quantize_latent_batched(_rnd(2, 64, 16), sv, **_args(hi_bit=[4]))


def test_quantize_latent_wrong_ndim() -> None:
    sv = mx.ones((16,))
    with pytest.raises(ValueError, match=r"\[S, r\]"):
        quantize_latent(_rnd(2, 64, 16), sv, **_args())
    with pytest.raises(ValueError, match=r"\[G, S, r\]"):
        quantize_latent_batched(_rnd(64, 16), sv, **_args())


def test_hi_bit_zero_used_to_give_all_nan() -> None:
    """Regression for the reason hi_bit/lo_bit must be >= 1: 0 bits means 0 levels."""
    # Calls the underlying quantizer directly (it has no validation of its own).
    from veloxquant_mlx.quantizers._quant_utils import _group_quant_dequant

    out = _group_quant_dequant(_rnd(32, 4), 0, 32)
    assert bool(mx.any(mx.isnan(out)).item())


# --------------------------------------------------------------------------- n_hi_channels
@pytest.mark.parametrize(
    "r,frac,expected",
    [
        (50, 0.58, 29),
        (100, 0.29, 29),
        (16, 0.25, 4),
        (10, 0.0, 1),
        (7, 0.01, 1),
        (8, 1.0, 8),
        (32, 0.5, 16),
    ],
)
def test_n_hi_channels_is_exact_decimal_floor(r, frac, expected) -> None:
    assert n_hi_channels(r, frac) == expected


def test_n_hi_channels_old_truncation_was_off_by_one() -> None:
    # int(50 * 0.58) is 28: binary 0.58 * 50 == 28.999999999999996.
    assert int(50 * 0.58) == 28
    assert n_hi_channels(50, 0.58) == 29


@pytest.mark.parametrize("frac", [0.05, 0.1, 0.125, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75])
def test_n_hi_channels_unchanged_for_common_fractions(frac) -> None:
    for r in range(1, 1025):
        assert n_hi_channels(r, frac) == max(1, int(r * frac)), (r, frac)


def test_n_hi_channels_default_fraction_unchanged_to_4096() -> None:
    assert all(n_hi_channels(r, 0.25) == max(1, int(r * 0.25)) for r in range(1, 4097))


def test_byte_accounting_uses_same_split_as_quantizer() -> None:
    """latent_bytes / assigned_avg_bits must bill the exact hi/lo split (29 of 50 at 0.58).

    The expectation is spelled out literally (29, not recomputed from the helper) so
    this fails if the accounting ever goes back to ``int(r * frac)`` == 28.
    """
    r, frac, n_hi = 50, 0.58, 29
    lr = _TensorLowRank(1, None, 0.9, 4, 2, frac, 32, quantize=True)
    lr._r = r
    lr._mu = [mx.zeros((D,))]
    assert math.isclose(lr.assigned_avg_bits, (n_hi * 4 + (r - n_hi) * 2) / D)
    code = math.ceil(64 * n_hi * 4 / 8) + math.ceil(
        64 * (r - n_hi) * 2 / 8
    )  # 29*4 + 21*2 bits/token
    assert lr.latent_bytes(64, 1) == code + math.ceil(64 / 32) * r * 4


# --------------------------------------------------------------------------- cache / config
@pytest.mark.parametrize(
    "field,bad_values",
    [
        ("palu_rank", BAD_RANKS),
        ("palu_energy_threshold", BAD_ENERGY),
        ("palu_n_head_groups", [0, -2, 2.5, True, "4", float("nan")]),
        ("palu_hi_bit", BAD_BITS),
        ("palu_lo_bit", BAD_BITS),
        ("palu_hi_fraction", BAD_FRACS),
        ("palu_group_size", BAD_GROUP_SIZES),
        ("palu_min_fit_rows", [-5, 2.5, True, "8"]),
        ("palu_quantize_values", ["False", "no", 0, 1, None]),
    ],
)
def test_cache_and_config_reject_bad_values(field, bad_values) -> None:
    for bad in bad_values:
        with pytest.raises(ValueError, match=field):
            _cache(**{field: bad})


def test_config_rejects_at_construction_not_at_create() -> None:
    with pytest.raises(ValueError, match="palu_rank"):
        KVCacheConfig(method="palu", head_dim=D, palu_rank=0)


def test_other_methods_ignore_palu_fields() -> None:
    # Every KVCacheConfig carries palu_*; only method="palu" may validate them.
    KVCacheConfig(method="kivi", head_dim=D, palu_rank=0, palu_hi_bit=99)


def test_cache_constructor_revalidates_after_mutation() -> None:
    cfg = KVCacheConfig(method="palu", head_dim=D)
    object.__setattr__(cfg, "palu_hi_bit", 0)
    with pytest.raises(ValueError, match="palu_hi_bit"):
        PALUKVCache(cfg)


def test_hi_fraction_message_is_stable() -> None:
    with pytest.raises(ValueError, match="palu_hi_fraction"):
        _cache(palu_hi_fraction=1.5)


def test_integral_floats_behave_like_ints() -> None:
    a = _run(
        _cache(palu_rank=8, palu_n_head_groups=2, palu_hi_bit=4, palu_lo_bit=2, palu_group_size=16)
    )
    b = _run(
        _cache(
            palu_rank=8.0,
            palu_n_head_groups=2.0,
            palu_hi_bit=4.0,
            palu_lo_bit=2.0,
            palu_group_size=16.0,
        )
    )
    assert all(bool(mx.array_equal(x, y)) for x, y in zip(a, b, strict=True))


def test_validate_palu_config_defaults() -> None:
    cfg = validate_palu_config(KVCacheConfig(method="palu", head_dim=D))
    assert cfg == {
        "rank": None,
        "energy": 0.90,
        "n_groups": 4,
        "hi_bit": 4,
        "lo_bit": 2,
        "hi_fraction": 0.25,
        "group_size": 32,
        "quantize_values": True,
        "min_fit_rows": 0,
    }


def test_numpy_scalars_are_accepted() -> None:
    cfg = validate_palu_config(
        KVCacheConfig(
            method="palu", head_dim=D, palu_rank=np.int64(8), palu_quantize_values=np.bool_(False)
        )
    )
    assert cfg["rank"] == 8 and cfg["quantize_values"] is False


# --------------------------------------------------------------------------- non-finite KV in the cache
def test_cache_prefill_rejects_nan_keys_instead_of_poisoning_the_group() -> None:
    k = np.array(_rnd(1, 8, 64, D, seed=1)).astype(np.float32)
    k[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _cache().update_and_fetch(mx.array(k).astype(mx.float16), _rnd(1, 8, 64, D, seed=2))


def test_cache_prefill_rejects_inf_values() -> None:
    v = np.array(_rnd(1, 8, 64, D, seed=2)).astype(np.float32)
    v[0, 3, 5, 7] = np.inf
    with pytest.raises(ValueError, match="finite"):
        _cache().update_and_fetch(_rnd(1, 8, 64, D, seed=1), mx.array(v).astype(mx.float16))


def test_cache_decode_nan_is_not_checked_and_stays_local() -> None:
    """Pins the scope: the decode path has no host sync, so it is deliberately unchecked.

    A NaN in one decoded key only damages that token (and its quantization group).
    """
    c = _cache()
    _run(c)
    kd = np.array(_rnd(1, 8, 1, D, seed=3)).astype(np.float32)
    kd[0, 0, 0, 0] = np.nan
    ko, _ = c.update_and_fetch(mx.array(kd).astype(mx.float16), _rnd(1, 8, 1, D, seed=4))
    ko = np.array(ko.astype(mx.float32))
    assert np.isnan(ko[0, :, 64]).any()  # the new token is bad ...
    assert not np.isnan(ko[0, :, :64]).any()  # ... the 64 prefill tokens are not


def test_valid_prefill_still_finite_and_close() -> None:
    ko, vo = _run(_cache(palu_rank=16))
    assert bool(mx.all(mx.isfinite(ko)).item()) and bool(mx.all(mx.isfinite(vo)).item())
    assert ko.shape == (1, 8, 64, D)
