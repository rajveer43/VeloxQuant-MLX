"""Tests for KVCacheConfig.options (#420).

KVCacheConfig grew one flat dataclass field per method's hyperparameter
(207 fields across 37 method families). This adds a `options` field that
takes a small per-method dataclass from veloxquant_mlx.cache.options
(H2OOptions, SnapKVOptions, ...) and reads its values through to the
matching flat fields, so KVCacheFactory / every cache class -- which read
flat fields via getattr(config, "h2o_budget", ...) -- keep working
unchanged. The flat fields still work directly too, but now emit a
DeprecationWarning naming their options replacement.
"""

from __future__ import annotations

import warnings

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.options import H2OOptions


def test_flat_field_construction_still_works_no_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = KVCacheConfig(method="h2o", h2o_budget=256)
    assert config.h2o_budget == 256
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_options_populates_matching_flat_fields() -> None:
    config = KVCacheConfig(method="h2o", options=H2OOptions(h2o_budget=999, h2o_n_sink=8))
    assert config.h2o_budget == 999
    assert config.h2o_n_sink == 8
    # Fields not set on the options instance keep their own dataclass default.
    assert config.h2o_grace == 16


def test_explicit_flat_kwarg_takes_priority_over_options() -> None:
    config = KVCacheConfig(method="h2o", h2o_budget=111, options=H2OOptions(h2o_budget=999))
    assert config.h2o_budget == 111


def test_post_construction_flat_field_set_emits_deprecation_warning() -> None:
    config = KVCacheConfig(method="snapkv")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.snap_budget = 2048
    assert config.snap_budget == 2048
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert "SnapKVOptions" in str(deprecations[0].message)


def test_setting_unrelated_field_does_not_warn() -> None:
    config = KVCacheConfig(method="h2o")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config.seed = 7
    assert config.seed == 7
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_kv_cache_factory_create_reads_options_through_to_cache() -> None:
    config = KVCacheConfig(
        method="h2o", head_dim=64, options=H2OOptions(h2o_budget=128, h2o_n_sink=2)
    )
    cache = KVCacheFactory.create(config)
    assert cache._budget == 128
    assert cache._n_sink == 2
