"""Tests for CodebookFactory's memoization (VeloxQuant-MLX#508).

Codebook centroid construction is a pure, deterministic function of
(distribution, b, d, polar_level) -- no seed or other hidden state -- so
repeated construction for the same args used to redo real Lloyd-Max work
from scratch on every call (e.g. every layer of a model, every cache-miss
request). CodebookFactory.create now memoizes on those args.
"""

from __future__ import annotations

import numpy as np
import pytest

from veloxquant_mlx.codebooks.base import CodebookFactory
from veloxquant_mlx.core.exceptions import QuantizerConfigError


@pytest.fixture(autouse=True)
def _clear_codebook_cache():
    CodebookFactory._create_cached.cache_clear()
    yield
    CodebookFactory._create_cached.cache_clear()


def test_same_args_return_identical_object():
    cb1 = CodebookFactory.create("gaussian", b=2, d=128)
    cb2 = CodebookFactory.create("gaussian", b=2, d=128)
    assert cb1 is cb2


def test_different_bits_are_not_shared():
    cb1 = CodebookFactory.create("gaussian", b=2, d=128)
    cb2 = CodebookFactory.create("gaussian", b=3, d=128)
    assert cb1 is not cb2
    assert cb1.k != cb2.k


def test_different_dim_are_not_shared():
    cb1 = CodebookFactory.create("gaussian", b=2, d=128)
    cb2 = CodebookFactory.create("gaussian", b=2, d=64)
    assert cb1 is not cb2


def test_different_distribution_are_not_shared():
    cb1 = CodebookFactory.create("gaussian", b=2, d=128)
    cb2 = CodebookFactory.create("beta", b=2, d=128)
    assert cb1 is not cb2


def test_cached_result_matches_uncached_centroids():
    """Memoized path must return bit-identical centroids to a fresh build."""
    cb_cached = CodebookFactory.create("gaussian", b=3, d=64)
    CodebookFactory._create_cached.cache_clear()
    cb_fresh = CodebookFactory.create("gaussian", b=3, d=64)
    np.testing.assert_array_equal(cb_cached.centroids_numpy(), cb_fresh.centroids_numpy())


def test_invalid_args_still_raise_and_do_not_poison_cache():
    with pytest.raises(QuantizerConfigError):
        CodebookFactory.create("gaussian", b=0, d=128)
    # A valid call afterward must still succeed (error wasn't cached as a result).
    cb = CodebookFactory.create("gaussian", b=2, d=128)
    assert cb.k == 4
