"""Tests for KVCacheBuilder.for_model's standalone-method refusal.

Regression coverage for issue #56: for_model() (and, transitively,
patch_model_kv_cache / patch_vlm_kv_cache) used to build/wire a cache for
any config.method string with no check that the method actually satisfies
the mlx_lm KVCache serving contract. Methods like "spectral" or "qjl"
implement VeloxQuant's own KVCache ABC (append_key/append_value/attend/
memory_bytes), not mlx_lm.models.cache.KVCache (update_and_fetch/nbytes/
state/trim/merge/meta_state) -- see issue #27. Wiring one of these into a
live mlx_lm.generate() call used to fail deep inside generation instead of
at config/patch time.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from veloxquant_mlx.cache.base import STANDALONE_METHODS, KVCacheBuilder, KVCacheConfig
from veloxquant_mlx.core.exceptions import QuantizerConfigError


def _make_fake_model(n_layers: int = 2, n_heads: int = 2, head_dim: int = 32) -> SimpleNamespace:
    hidden_size = n_heads * head_dim
    layers = [
        SimpleNamespace(self_attn=SimpleNamespace(head_dim=head_dim)) for _ in range(n_layers)
    ]
    args = SimpleNamespace(hidden_size=hidden_size, num_attention_heads=n_heads)
    return SimpleNamespace(layers=layers, args=args)


@pytest.mark.parametrize("method", sorted(STANDALONE_METHODS))
def test_for_model_refuses_standalone_methods(method: str) -> None:
    model = _make_fake_model()
    config = KVCacheConfig(method=method, bit_width_inlier=1, seed=42)

    with pytest.raises(QuantizerConfigError, match=method):
        KVCacheBuilder.for_model(model, config)


@pytest.mark.parametrize("method", sorted(STANDALONE_METHODS))
def test_for_model_error_message_names_the_exact_method(method: str) -> None:
    """The error must literally contain the method string so a user can act
    on it without re-reading the source -- a bare regex-match on the class
    of error is not enough of a guarantee on its own.
    """
    model = _make_fake_model()
    config = KVCacheConfig(method=method, bit_width_inlier=1, seed=42)

    with pytest.raises(QuantizerConfigError) as exc_info:
        KVCacheBuilder.for_model(model, config)

    assert repr(method) in str(exc_info.value)


def test_standalone_methods_is_exactly_the_five_known_methods() -> None:
    """Pins the registry itself. If a future PR adds/removes an entry, this
    test forces a conscious update instead of silently widening or
    narrowing which methods get refused.
    """
    assert STANDALONE_METHODS == {
        "turboquant_prod",
        "turboquant_mse",
        "polar",
        "qjl",
        "spectral",
    }


def test_for_model_refuses_before_touching_model_attributes() -> None:
    """The refusal must fire before for_model reads model.layers /
    model.model.layers -- otherwise a malformed/incomplete fake or partially
    loaded model could raise a confusing AttributeError instead of the
    intended QuantizerConfigError.
    """
    model = object()  # has no .layers, .model, or anything else
    config = KVCacheConfig(method="spectral", bit_width_inlier=1, seed=42)

    with pytest.raises(QuantizerConfigError, match="spectral"):
        KVCacheBuilder.for_model(model, config)


def test_for_model_refusal_wins_over_invalid_bit_width_list() -> None:
    """Standalone-method refusal must be checked before the per-layer
    bit_width_inlier list-length validation, so a caller sees the more
    fundamental "wrong method" error rather than an unrelated
    "wrong list length" error that would vanish once they fixed the method.
    """
    model = _make_fake_model(n_layers=2)
    # Deliberately wrong length (3 vs. 2 attention layers) -- if refusal
    # didn't run first, this would raise a *different* QuantizerConfigError.
    config = KVCacheConfig(method="qjl", bit_width_inlier=[1, 1, 1], seed=42)

    with pytest.raises(QuantizerConfigError, match="qjl"):
        KVCacheBuilder.for_model(model, config)


def test_for_model_accepts_serving_compatible_method() -> None:
    model = _make_fake_model()
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)

    caches = KVCacheBuilder.for_model(model, config)

    assert len(caches) == 2


def test_default_method_is_serving_compatible() -> None:
    """Regression test for issue #130: KVCacheConfig()'s default method used
    to be "turboquant_prod", a STANDALONE_METHODS entry -- so a user who
    built a config with no explicit `method=` and then called
    patch_model_kv_cache()/for_model() on a live model got refused
    immediately, even though nothing about their code named a standalone
    method. The out-of-the-box default must always be one for_model()
    actually accepts.
    """
    default_method = KVCacheConfig().method
    assert default_method not in STANDALONE_METHODS

    model = _make_fake_model()
    caches = KVCacheBuilder.for_model(model, KVCacheConfig())
    assert len(caches) == 2


def test_for_model_accepts_all_non_standalone_methods() -> None:
    """Every method NOT in STANDALONE_METHODS must still be accepted by the
    refusal check itself (build may still fail later for unrelated
    per-method config reasons, but never with the standalone-refusal error).
    """
    import typing

    hints = typing.get_type_hints(KVCacheConfig)
    method_literal_args = typing.get_args(hints["method"])
    non_standalone = [m for m in method_literal_args if m not in STANDALONE_METHODS]
    assert non_standalone, "expected at least one non-standalone method in the registry"

    for method in non_standalone:
        model = _make_fake_model()
        config = KVCacheConfig(method=method, bit_width_inlier=1, seed=42)
        try:
            KVCacheBuilder.for_model(model, config)
        except QuantizerConfigError as exc:
            assert "standalone" not in str(exc), (
                f"method {method!r} is not in STANDALONE_METHODS but was refused "
                f"as if it were: {exc}"
            )
