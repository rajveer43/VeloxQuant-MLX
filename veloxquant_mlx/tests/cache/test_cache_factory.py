"""Regression for #405: KVCacheFactory.create() dispatches via one registry table.

create() used to be a 40-branch if/elif chain with all ~40 cache classes
imported eagerly at the top of the function, plus a hand-duplicated method
list in the error message -- three places that had to be kept in sync by
hand for every new method. It's now a single ``_CACHE_CLASS_BY_METHOD`` dict
resolved lazily via importlib, with the error message's choices derived from
the same dict. These tests catch that dict drifting from ``MethodName`` (the
Literal every other call site treats as the authoritative method list) and
catch any individual entry pointing at the wrong module/class.
"""

from __future__ import annotations

import typing

import pytest

from veloxquant_mlx.cache.base import (
    _CACHE_CLASS_BY_METHOD,
    KVCacheConfig,
    KVCacheFactory,
    MethodName,
)
from veloxquant_mlx.core.exceptions import QuantizerConfigError


def test_registry_keys_match_method_name_literal():
    literal_names = set(typing.get_args(MethodName))
    assert set(_CACHE_CLASS_BY_METHOD) == literal_names


@pytest.mark.parametrize("method", sorted(_CACHE_CLASS_BY_METHOD))
def test_every_registered_method_constructs_its_mapped_class(method):
    module_name, class_name = _CACHE_CLASS_BY_METHOD[method]
    import importlib

    expected_cls = getattr(importlib.import_module(module_name), class_name)

    config = KVCacheConfig(method=method, head_dim=64, bit_width_inlier=2, seed=0)
    cache = KVCacheFactory.create(config)

    assert type(cache) is expected_cls


def test_unknown_method_error_lists_every_registered_choice():
    with pytest.raises(QuantizerConfigError) as exc_info:
        KVCacheFactory.create(KVCacheConfig(method="not_a_real_method"))

    message = str(exc_info.value)
    for method in _CACHE_CLASS_BY_METHOD:
        assert method in message, f"{method} missing from unknown-method error"


def test_unknown_method_error_does_not_hand_duplicate_the_list():
    """The choices string must come from the registry, not a separate literal."""
    with pytest.raises(QuantizerConfigError) as exc_info:
        KVCacheFactory.create(KVCacheConfig(method="not_a_real_method"))

    message = str(exc_info.value)
    choices = message.split("Choices: ", 1)[1].rstrip(".")
    assert set(choices.split(", ")) == set(_CACHE_CLASS_BY_METHOD)


def test_registering_a_new_method_only_requires_one_dict_entry(monkeypatch):
    """Adding a method should not require touching create()'s body at all."""
    from veloxquant_mlx.cache import base as cache_base

    fake_registry = dict(cache_base._CACHE_CLASS_BY_METHOD)
    fake_registry["h2o_fake_alias"] = ("veloxquant_mlx.cache.h2o_cache", "H2OKVCache")
    monkeypatch.setattr(cache_base, "_CACHE_CLASS_BY_METHOD", fake_registry)

    config = KVCacheConfig(method="h2o_fake_alias", head_dim=64, bit_width_inlier=2, seed=0)
    cache = cache_base.KVCacheFactory.create(config)

    from veloxquant_mlx.cache.h2o_cache import H2OKVCache

    assert type(cache) is H2OKVCache
