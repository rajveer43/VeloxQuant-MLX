"""bit_width_inlier is read by 7 methods; naming it for the other 36 must warn (#901)."""

from __future__ import annotations

import dataclasses
import sys
import warnings

import mlx.core as mx
import pytest

from veloxquant_mlx.cache import base
from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig, KVCacheFactory

ALL_METHODS = sorted(base._CACHE_CLASS_BY_METHOD)
READERS = sorted(base._INLIER_BITS_METHODS)
IGNORERS = [m for m in ALL_METHODS if m not in base._INLIER_BITS_METHODS]


def _caught(**kw) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        KVCacheConfig(**kw)
    return [w for w in caught if "bit_width_inlier" in str(w.message)]


def test_there_are_36_ignoring_methods() -> None:
    assert len(ALL_METHODS) == 43
    assert len(IGNORERS) == 36


def test_reader_set_matches_what_the_caches_actually_read() -> None:
    """Instrument attribute access so the hard-coded set cannot drift from the code."""
    reads: set[str] = set()
    orig = KVCacheConfig.__getattribute__

    def spy(self, name):
        if name == "bit_width_inlier":
            mod = sys._getframe(1).f_globals.get("__name__", "")
            if mod not in ("veloxquant_mlx.cache.base", "dataclasses", "copy"):
                reads.add(orig(self, "method"))
        return orig(self, name)

    KVCacheConfig.__getattribute__ = spy  # type: ignore[method-assign]
    try:
        for m in ALL_METHODS:
            KVCacheFactory.create(KVCacheConfig(method=m, head_dim=64))
    finally:
        KVCacheConfig.__getattribute__ = orig  # type: ignore[method-assign]
    assert reads == set(base._INLIER_BITS_METHODS)


@pytest.mark.parametrize("method", IGNORERS)
def test_warns_when_ignored(method: str) -> None:
    got = _caught(method=method, head_dim=64, bit_width_inlier=4)
    assert len(got) == 1, method
    assert got[0].category is UserWarning
    assert repr(method) in str(got[0].message)
    assert got[0].filename == __file__  # points at the caller, not base.py


@pytest.mark.parametrize("method", READERS)
def test_no_warning_for_methods_that_read_it(method: str) -> None:
    assert _caught(method=method, head_dim=64, bit_width_inlier=4) == []


@pytest.mark.parametrize("method", ALL_METHODS)
def test_no_warning_when_not_named(method: str) -> None:
    assert _caught(method=method, head_dim=64) == []


def test_message_points_at_the_methods_own_field() -> None:
    (w,) = _caught(method="kvquant", head_dim=64, bit_width_inlier=4)
    assert "kvquant_bits" in str(w.message)
    (w,) = _caught(method="gear", head_dim=64, bit_width_inlier=4)
    assert "gear_bits" in str(w.message)


def test_dataclasses_replace_does_not_warn() -> None:
    cfg = KVCacheConfig(method="kvquant", head_dim=64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dataclasses.replace(cfg, head_dim=128)
    assert not [w for w in caught if "bit_width_inlier" in str(w.message)]


def test_builder_for_model_does_not_warn() -> None:
    """for_model copies the config with dataclasses.replace, naming every field."""
    from veloxquant_mlx.tests.cache.test_base_for_model import _make_fake_model

    cfg = KVCacheConfig(method="kvquant", head_dim=32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        caches = KVCacheBuilder.for_model(_make_fake_model(), cfg)
    assert len(caches) == 2
    assert not [w for w in caught if "bit_width_inlier" in str(w.message)]


@pytest.mark.parametrize("method", IGNORERS)
def test_synthetic_run_is_independent_of_bit_width_inlier(method: str) -> None:
    """The 36 really ignore it: same cache, same output at 2 and 4, warning aside."""

    def run(bits: int):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cache = KVCacheFactory.create(
                KVCacheConfig(method=method, head_dim=64, bit_width_inlier=bits, seed=1)
            )
        mx.random.seed(3)
        if method in base.STANDALONE_METHODS:  # VeloxQuant's own KVCache ABC (qjl)
            for _ in range(8):
                cache.append(mx.random.normal((64,)), mx.random.normal((64,)))
            out = cache.attend(mx.random.normal((64,)))
            mx.eval(out)
            return out, out, cache.memory_bytes()
        k = mx.random.normal((1, 2, 40, 64))
        v = mx.random.normal((1, 2, 40, 64))
        ko, vo = cache.update_and_fetch(k, v)
        mx.eval(ko, vo)
        return ko, vo, getattr(cache, "compressed_key_bytes", None)

    ko2, vo2, b2 = run(2)
    ko4, vo4, b4 = run(4)
    assert mx.array_equal(ko2, ko4).item()
    assert mx.array_equal(vo2, vo4).item()
    assert b2 == b4
