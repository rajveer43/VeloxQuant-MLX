"""Regression for #83: eviction/compression caches must populate mlx_lm's
base ``.keys``/``.values``/``.offset`` so ``.state`` (read unconditionally by
``mlx_lm.generate()`` during chunked prefill and the speculative-decoding
rewind path) never crashes with ``AttributeError: 'NoneType' object has no
attribute 'shape'``.

Covers all 17 affected classes: 15 that now write through to the base class
via ``super().update_and_fetch(...)`` on every call (and correctly report
``is_trimmable() == False``, since their internal per-token eviction state
isn't rolled back by a base-class ``trim()``), plus ``PALUKVCache`` and
``KVTCKVCache`` which bypass the base fp16 buffer entirely (true latent
storage) and instead override ``.state``/``.trim()``/``.is_trimmable()``
directly against their own storage.

Config per method mirrors each method's own dedicated test file's ``_make``
helper, kept minimal here since only the base-class bookkeeping contract is
under test — not each method's compression/eviction quality.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

# method -> extra config kwargs beyond method/head_dim
_CONFIGS = {
    "amc": {},
    "cam": {"cam_budget": 8, "cam_n_sink": 1},
    "chunkkv": {"chunkkv_budget": 8, "chunkkv_n_sink": 1, "chunkkv_chunk_size": 2},
    "curdkv": {"curdkv_budget": 8, "curdkv_n_sink": 1},
    "h2o": {"h2o_budget": 8, "h2o_n_sink": 1},
    "keyformer": {"keyformer_budget": 8, "keyformer_n_sink": 1},
    "knorm": {"knorm_budget": 8, "knorm_n_sink": 1},
    "kvzip": {"kvzip_budget": 8, "kvzip_n_sink": 1},
    "morphkv": {"morphkv_budget": 8, "morphkv_n_sink": 1, "morphkv_window": 4},
    "nestedkv": {"nestedkv_budget": 8, "nestedkv_n_sink": 1},
    "pyramidkv": {"pyramid_budget": 8, "pyramid_n_sink": 1},
    "qfilters": {"qfilters_budget": 8, "qfilters_n_sink": 1, "qfilters_calib_tokens": 4},
    "squeeze": {"squeeze_budget": 8, "squeeze_n_sink": 1},
    "streaming_llm": {"stream_n_sink": 1, "stream_window_size": 8},
    "tova": {"tova_budget": 8, "tova_n_sink": 1},
    "palu": {"palu_rank": 4, "palu_n_head_groups": 1},
    "kvtc": {"kvtc_bit_budget": 32},
}

# The 15 "case A" methods: reset-then-write-through, no real trim support.
_CASE_A_METHODS = [m for m in _CONFIGS if m not in ("palu", "kvtc")]
# The 2 "case C" methods: bypass the base buffer, own state/trim.
_CASE_C_METHODS = ["palu", "kvtc"]

ALL_METHODS = list(_CONFIGS)


def _make(method: str, head_dim: int = 8, **extra):
    cfg = dict(method=method, head_dim=head_dim)
    cfg.update(_CONFIGS[method])
    cfg.update(extra)
    return KVCacheFactory.create(KVCacheConfig(**cfg))


def _rand_kv(S: int, H: int = 1, D: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    V = mx.array(rng.standard_normal((1, H, S, D)).astype(np.float16))
    return K, V


# ---------------------------------------------------------------------------
# Core crash repro from the issue: .state must not raise after prefill.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ALL_METHODS)
def test_state_accessible_after_single_prefill(method: str) -> None:
    """Regression for #83's exact repro: cache.state must not raise
    AttributeError after a single update_and_fetch call."""
    cache = _make(method)
    k, v = _rand_kv(S=4)
    cache.update_and_fetch(k, v)

    k_state, v_state = cache.state  # must not raise
    mx.eval(k_state, v_state)
    assert k_state.shape[0] == 1
    assert k_state.shape[-1] == 8
    assert cache.offset > 0


@pytest.mark.parametrize("method", ALL_METHODS)
def test_state_shape_matches_offset(method: str) -> None:
    """cache.state's seq-length dim must equal cache.offset, mirroring the
    base KVCache contract mlx_lm relies on elsewhere (e.g. make_mask)."""
    cache = _make(method)
    k, v = _rand_kv(S=6)
    cache.update_and_fetch(k, v)

    k_state, v_state = cache.state
    mx.eval(k_state, v_state)
    assert k_state.shape[2] == cache.offset
    assert v_state.shape[2] == cache.offset


# ---------------------------------------------------------------------------
# mlx_lm's chunked prefill: update_and_fetch called multiple times with S>1.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ALL_METHODS)
def test_state_accessible_across_chunked_prefill(method: str) -> None:
    """Simulates mlx_lm's chunked prefill: several S>1 calls in a row (one
    per prefill_step_size chunk), each followed by a .state access (mirrors
    generate.py's `mx.eval([c.state for c in cache])` after every chunk)."""
    cache = _make(method)
    for seed in range(3):
        k, v = _rand_kv(S=5, seed=seed)
        cache.update_and_fetch(k, v)
        k_state, v_state = cache.state
        mx.eval(k_state, v_state)
        assert k_state.shape[2] == cache.offset


@pytest.mark.parametrize("method", ALL_METHODS)
def test_state_accessible_after_decode_steps(method: str) -> None:
    """Prefill followed by several S==1 decode steps; .state must stay
    accessible and consistent with offset throughout."""
    cache = _make(method)
    k, v = _rand_kv(S=5)
    cache.update_and_fetch(k, v)

    for i in range(3):
        k1, v1 = _rand_kv(S=1, seed=100 + i)
        cache.update_and_fetch(k1, v1)
        k_state, v_state = cache.state
        mx.eval(k_state, v_state)
        assert k_state.shape[2] == cache.offset


# ---------------------------------------------------------------------------
# is_trimmable() / trim() honesty.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", _CASE_A_METHODS)
def test_case_a_reports_not_trimmable(method: str) -> None:
    """These 15 caches' internal per-token eviction/compression state can't
    be rolled back by the base class's offset-only trim() — is_trimmable()
    must report False so mlx_lm's can_trim_prompt_cache() correctly skips
    them instead of silently corrupting state via a no-op-on-content trim."""
    cache = _make(method)
    k, v = _rand_kv(S=4)
    cache.update_and_fetch(k, v)
    assert cache.is_trimmable() is False


@pytest.mark.parametrize("method", _CASE_C_METHODS)
def test_case_c_trim_rolls_back_state_and_offset(method: str) -> None:
    """PALU/KVTC own true latent storage end-to-end, so trim() can and must
    correctly shrink both offset and the reconstructed state, and further
    update_and_fetch calls after a trim must keep working."""
    cache = _make(method)
    k, v = _rand_kv(S=10)
    cache.update_and_fetch(k, v)
    assert cache.is_trimmable() is True

    n_trimmed = cache.trim(3)
    assert n_trimmed == 3
    assert cache.offset == 7

    k_state, v_state = cache.state
    mx.eval(k_state, v_state)
    assert k_state.shape[2] == 7
    assert v_state.shape[2] == 7

    # Cache must still work after a trim — this is exactly what mlx_lm does
    # after trimming for prefix-cache reuse / speculative-decoding rewind.
    k1, v1 = _rand_kv(S=1, seed=99)
    cache.update_and_fetch(k1, v1)
    assert cache.offset == 8
    k_state2, v_state2 = cache.state
    mx.eval(k_state2, v_state2)
    assert k_state2.shape[2] == 8


# ---------------------------------------------------------------------------
# No double counting: repeated update_and_fetch calls must not make offset
# grow faster than the true retained/seen token count.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ALL_METHODS)
def test_offset_does_not_exceed_tokens_seen(method: str) -> None:
    """A buggy write-through (e.g. appending instead of replacing on a
    full-state-every-call cache) would make offset grow unboundedly across
    repeated prefill-shaped calls even though the true retained count is
    capped by the method's budget."""
    cache = _make(method)
    total_seen = 0
    for seed in range(4):
        k, v = _rand_kv(S=5, seed=seed)
        cache.update_and_fetch(k, v)
        total_seen += 5
        assert cache.offset <= total_seen


# ---------------------------------------------------------------------------
# The issue's own scenario: a multi-layer cache list, exactly what mlx_lm's
# generate.py does after every prefill chunk (``mx.eval([c.state for c in
# cache])``) and in the speculative-decoding rewind path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ALL_METHODS)
def test_multi_layer_cache_list_state_eval(method: str) -> None:
    """Regression for #83's exact crash site: mlx_lm.generate's _prefill does
    `mx.eval([c.state for c in cache])` on a *list* of per-layer caches after
    every chunk. Simulate 3 layers x 2 chunks."""
    caches = [_make(method) for _ in range(3)]
    for seed in range(2):
        k, v = _rand_kv(S=4, seed=seed)
        for cache in caches:
            cache.update_and_fetch(k, v)
        mx.eval([c.state for c in caches])  # must not raise
        for cache in caches:
            k_state, v_state = cache.state
            assert k_state.shape[2] == cache.offset
