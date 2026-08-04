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


def test_for_model_accepts_serving_compatible_method() -> None:
    model = _make_fake_model()
    config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)

    caches = KVCacheBuilder.for_model(model, config)

    assert len(caches) == 2
