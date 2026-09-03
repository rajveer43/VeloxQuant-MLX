"""Phase 0 of issue #310: verify mlx_lm's LRUPromptCache prefix-reuse
mechanism (fetch_nearest_cache / trim_prompt_cache / insert_cache) actually
works correctly against VeloxQuant's own compressed cache classes.

veloxquant serve already inherits this mechanism for free by handing off to
unmodified mlx_lm.server.run() -- nothing here is new production code, this
is the missing proof that the mechanism is trustworthy for our cache
classes before Phase 1 (CLI tuning) and Phase 2 (library-level PrefixCache)
build on top of it.

TurboQuantRVQKVCache stands in for the ~35/40 trimmable methods (its
trim()/state() shapes are representative of the base _MLXKVCache contract);
H2OKVCache stands in for the 18 NOT_TRIMMABLE eviction/hybrid methods, per
this repo's established convention (see test_issue_83_state_writethrough.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import LRUPromptCache

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

_TRIMMABLE_CONFIG = dict(method="turboquant_rvq", head_dim=32, bit_width_inlier=1, seed=42)
_NOT_TRIMMABLE_CONFIG = dict(
    method="h2o", head_dim=32, h2o_budget=8, h2o_n_sink=1, h2o_grace=0, h2o_decay=1.0
)


def _make_layer_cache(method_config: dict) -> list:
    cfg = KVCacheConfig(**method_config)
    return [KVCacheFactory.create(cfg)]


def _feed_tokens(cache_list: list, token_ids: list, H: int = 2, D: int = 32) -> None:
    """Feed one synthetic (K, V) pair per token, seeded off (token_id,
    position) so two calls with the same tokens at the same position
    produce identical inputs -- needed for the bit-exact coherence test."""
    for i, tok in enumerate(token_ids):
        rng = np.random.default_rng(hash((tok, i)) & 0xFFFF)
        k = mx.array(rng.standard_normal((1, H, 1, D)).astype(np.float16))
        v = mx.array(rng.standard_normal((1, H, 1, D)).astype(np.float16))
        for c in cache_list:
            c.update_and_fetch(k, v)


# ---------------------------------------------------------------------------
# Trimmable method (turboquant_rvq): partial-prefix reuse must actually work.
# ---------------------------------------------------------------------------


def test_trimmable_method_returns_partial_hit_with_correct_rest() -> None:
    lru = LRUPromptCache(max_size=10)
    model = "fake-model"

    stored_tokens = list(range(1, 9))  # [1..8]
    stored_cache = _make_layer_cache(_TRIMMABLE_CONFIG)
    _feed_tokens(stored_cache, stored_tokens)
    lru.insert_cache(model, stored_tokens, stored_cache)

    query_tokens = [1, 2, 3, 4, 5, 6, 9, 10]  # shares first 6 tokens
    cache, rest = lru.fetch_nearest_cache(model, query_tokens)

    assert rest == [9, 10]
    assert cache is not None
    assert cache[0].offset == 6
    assert cache[0] is not stored_cache[0]  # deepcopy isolation


def test_trimmable_method_exact_match_returns_empty_rest() -> None:
    lru = LRUPromptCache(max_size=10)
    model = "fake-model"

    stored_tokens = list(range(1, 9))  # [1..8]
    stored_cache = _make_layer_cache(_TRIMMABLE_CONFIG)
    _feed_tokens(stored_cache, stored_tokens)
    lru.insert_cache(model, stored_tokens, stored_cache)

    cache, rest = lru.fetch_nearest_cache(model, list(stored_tokens))

    assert rest == []
    assert cache is not None
    assert cache[0].offset == 8


def test_trimmable_generation_after_trim_matches_fresh_cache_fed_same_prefix() -> None:
    """Bit-exact coherence check: continuing generation on a reused
    (deepcopy'd + trimmed) cache must produce identical state to a fresh
    cache fed only the shared prefix followed by the same new tokens."""
    lru = LRUPromptCache(max_size=10)
    model = "fake-model"

    full_tokens = list(range(1, 11))  # [1..10]
    stored_cache = _make_layer_cache(_TRIMMABLE_CONFIG)
    _feed_tokens(stored_cache, full_tokens)
    lru.insert_cache(model, full_tokens, stored_cache)

    shared_prefix = [1, 2, 3, 4, 5, 6]
    new_suffix = [99, 100]
    query_tokens = shared_prefix + new_suffix

    reused_cache, rest = lru.fetch_nearest_cache(model, query_tokens)
    assert rest == new_suffix
    _feed_tokens(reused_cache, new_suffix)

    fresh_cache = _make_layer_cache(_TRIMMABLE_CONFIG)
    _feed_tokens(fresh_cache, shared_prefix)
    _feed_tokens(fresh_cache, new_suffix)

    assert reused_cache[0].offset == fresh_cache[0].offset == 8

    reused_state = reused_cache[0].state
    fresh_state = fresh_cache[0].state
    for reused_elem, fresh_elem in zip(reused_state, fresh_state):
        mx.eval(reused_elem, fresh_elem)
        assert mx.array_equal(reused_elem, fresh_elem)


# ---------------------------------------------------------------------------
# NOT_TRIMMABLE method (h2o): only exact-prefix hits, never partial reuse.
# ---------------------------------------------------------------------------


def test_not_trimmable_method_never_receives_partial_overlap_hit() -> None:
    lru = LRUPromptCache(max_size=10)
    model = "fake-model"

    stored_tokens = list(range(1, 9))  # [1..8]
    stored_cache = _make_layer_cache(_NOT_TRIMMABLE_CONFIG)
    _feed_tokens(stored_cache, stored_tokens)
    lru.insert_cache(model, stored_tokens, stored_cache)

    assert stored_cache[0].is_trimmable() is False

    query_tokens = [1, 2, 3, 4, 5, 6, 9, 10]  # shares first 6 tokens
    cache, rest = lru.fetch_nearest_cache(model, query_tokens)

    assert cache is None
    assert rest == query_tokens  # full original query, unchanged

    # Exact-match on the same method must still hit -- only the trim branch
    # is blocked, not the whole method.
    exact_cache, exact_rest = lru.fetch_nearest_cache(model, list(stored_tokens))
    assert exact_rest == []
    assert exact_cache is not None


def test_not_trimmable_trim_is_never_called() -> None:
    lru = LRUPromptCache(max_size=10)
    model = "fake-model"

    stored_tokens = list(range(1, 9))  # [1..8]
    stored_cache = _make_layer_cache(_NOT_TRIMMABLE_CONFIG)
    _feed_tokens(stored_cache, stored_tokens)
    stored_cache[0].trim = MagicMock(wraps=stored_cache[0].trim)
    lru.insert_cache(model, stored_tokens, stored_cache)

    query_tokens = [1, 2, 3, 4, 5, 6, 9, 10]  # partial overlap
    lru.fetch_nearest_cache(model, query_tokens)

    stored_cache[0].trim.assert_not_called()
