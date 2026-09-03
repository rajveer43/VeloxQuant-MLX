"""Tests for PrefixCache (#310, Phase 2): library-level prefix-KV reuse for
programmatic mlx_lm.generate()/stream_generate() callers.

Mirrors test_mlx_lm_patch.py's _make_fake_model helper -- no real downloaded
model is needed since fetch()/insert() operate purely on List[int] tokens
and cache objects, matching this repo's existing per-method test convention.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.integration.prefix_cache import PrefixCache


def _make_fake_model(n_layers: int = 4, n_heads: int = 4, head_dim: int = 32) -> SimpleNamespace:
    hidden_size = n_heads * head_dim
    layers = [
        SimpleNamespace(self_attn=SimpleNamespace(head_dim=head_dim)) for _ in range(n_layers)
    ]
    args = SimpleNamespace(hidden_size=hidden_size, num_attention_heads=n_heads)
    return SimpleNamespace(layers=layers, args=args)


_TRIMMABLE_CONFIG = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
_NOT_TRIMMABLE_CONFIG = KVCacheConfig(
    method="h2o", h2o_budget=8, h2o_n_sink=1, h2o_grace=0, h2o_decay=1.0
)


def test_fetch_miss_builds_fresh_cache_via_for_model() -> None:
    model = _make_fake_model(n_layers=3)
    pc = PrefixCache(_TRIMMABLE_CONFIG)

    cache, rest = pc.fetch(model, [1, 2, 3])

    assert len(cache) == 3
    assert rest == [1, 2, 3]


def test_fetch_hit_after_insert_returns_trimmed_rest() -> None:
    model = _make_fake_model(n_layers=2)
    pc = PrefixCache(_TRIMMABLE_CONFIG)

    stored_tokens = list(range(1, 9))
    stored_cache, _ = pc.fetch(model, stored_tokens)
    for c in stored_cache:
        c.update_and_fetch  # sanity: real cache objects, not stubs

    import mlx.core as mx
    import numpy as np

    for i, tok in enumerate(stored_tokens):
        rng = np.random.default_rng(hash((tok, i)) & 0xFFFF)
        k = mx.array(rng.standard_normal((1, 4, 1, 32)).astype(np.float16))
        v = mx.array(rng.standard_normal((1, 4, 1, 32)).astype(np.float16))
        for c in stored_cache:
            c.update_and_fetch(k, v)

    pc.insert(model, stored_tokens, stored_cache)

    query = [1, 2, 3, 4, 5, 6, 9, 10]
    cache, rest = pc.fetch(model, query)

    assert rest == [9, 10]
    assert cache[0].offset == 6


def test_model_key_none_falls_back_to_id_and_distinguishes_models() -> None:
    model_a = _make_fake_model(n_layers=2)
    model_b = _make_fake_model(n_layers=2)
    pc = PrefixCache(_TRIMMABLE_CONFIG)

    tokens = [1, 2, 3]
    cache_a, _ = pc.fetch(model_a, tokens)
    pc.insert(model_a, tokens, cache_a)

    # Same token list, different model object -> must not hit model_a's entry.
    cache_b, rest_b = pc.fetch(model_b, tokens)
    assert rest_b == tokens


def test_explicit_model_key_reused_across_new_model_object() -> None:
    model_1 = _make_fake_model(n_layers=2)
    pc = PrefixCache(_TRIMMABLE_CONFIG)

    tokens = [1, 2, 3, 4]
    cache, _ = pc.fetch(model_1, tokens, model_key="shared-key")
    pc.insert(model_1, tokens, cache, model_key="shared-key")

    del model_1
    model_2 = _make_fake_model(n_layers=2)

    hit_cache, rest = pc.fetch(model_2, tokens, model_key="shared-key")
    assert rest == []
    assert hit_cache[0].offset == cache[0].offset


def test_fetch_refuses_standalone_method_before_generation() -> None:
    model = _make_fake_model(n_layers=2)
    config = KVCacheConfig(method="spectral", bit_width_inlier=1, seed=42)
    pc = PrefixCache(config)

    with pytest.raises(QuantizerConfigError, match="spectral"):
        pc.fetch(model, [1, 2, 3])


def test_not_trimmable_method_fetch_hit_requires_exact_prefix() -> None:
    model = _make_fake_model(n_layers=2)
    pc = PrefixCache(_NOT_TRIMMABLE_CONFIG)

    import mlx.core as mx
    import numpy as np

    stored_tokens = list(range(1, 9))
    stored_cache, _ = pc.fetch(model, stored_tokens)
    for i, tok in enumerate(stored_tokens):
        rng = np.random.default_rng(hash((tok, i)) & 0xFFFF)
        k = mx.array(rng.standard_normal((1, 4, 1, 32)).astype(np.float16))
        v = mx.array(rng.standard_normal((1, 4, 1, 32)).astype(np.float16))
        for c in stored_cache:
            c.update_and_fetch(k, v)
    pc.insert(model, stored_tokens, stored_cache)

    partial_query = [1, 2, 3, 4, 5, 6, 9, 10]
    cache, rest = pc.fetch(model, partial_query)
    assert rest == partial_query  # no partial reuse

    exact_cache, exact_rest = pc.fetch(model, list(stored_tokens))
    assert exact_rest == []


def test_not_trimmable_config_warns_at_construction(capsys) -> None:
    PrefixCache(_NOT_TRIMMABLE_CONFIG)
    out = capsys.readouterr().out
    assert "does not support prefix-cache trimming" in out


def test_trimmable_config_does_not_warn_at_construction(capsys) -> None:
    PrefixCache(_TRIMMABLE_CONFIG)
    out = capsys.readouterr().out
    assert "does not support prefix-cache trimming" not in out


def test_generate_convenience_wrapper_round_trips_cache_key(monkeypatch) -> None:
    """stream_generate can't run against a bare SimpleNamespace model (needs
    a real nn.Module __call__), so this stubs mlx_lm.generate.stream_generate
    and asserts insert() receives a correctly-accumulated cache_key -- same
    "no real downloaded model" tier as the rest of this repo's tests."""
    model = _make_fake_model(n_layers=2)
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]

    fake_responses = [
        SimpleNamespace(text="hel", token=10),
        SimpleNamespace(text="lo", token=11),
    ]

    def _fake_stream_generate(*, model, tokenizer, prompt, prompt_cache, **kwargs):
        assert prompt == [1, 2, 3]  # full miss -> rest == whole prompt
        for r in fake_responses:
            yield r

    import sys

    monkeypatch.setattr(sys.modules["mlx_lm.generate"], "stream_generate", _fake_stream_generate)

    pc = PrefixCache(_TRIMMABLE_CONFIG)
    inserted = {}
    original_insert = pc.insert

    def _spy_insert(model, prompt, cache, **kwargs):
        inserted["prompt"] = prompt
        return original_insert(model, prompt, cache, **kwargs)

    pc.insert = _spy_insert

    result = pc.generate(model, tokenizer, "hello", max_tokens=2)

    assert result == "hello"
    assert inserted["prompt"] == [1, 2, 3, 10, 11]
