"""Tests for A2ATSKVCache — windowed RoPE + query-aware retrieval VQ.

A2ATS-adapted (He et al., ACL 2025 Findings, aclanthology.org/2025.findings-acl.644)
compresses keys via query-aware VQ (retrieval-set tokens get query-aware
centroid assignment, the bulk gets plain nearest-centroid) and reconstructs
with distance-gated windowed RoPE. Values follow a plain nearest-centroid VQ
path (no RoPE, no retrieval-set preference). All data is synthetic — no model
loading.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory, KVCacheBuilder
from veloxquant_mlx.cache.a2ats_cache import A2ATSKVCache


def _make(**cfg):
    base = dict(
        method="a2ats",
        head_dim=32,
        a2ats_sub_dim=8,
        a2ats_codebook_bits=6,
        a2ats_window=4,
    )
    base.update(cfg)
    return KVCacheFactory.create(KVCacheConfig(**base))


def _rand_kv(B=1, H=2, S=10, D=32, seed=0):
    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32))
    return k, v


# ---------------------------------------------------------------------------
# Config validation — write this FIRST, before wiring tests, per this
# session's own lesson: 5 sibling methods shipped without this exact check.
# ---------------------------------------------------------------------------

def test_beta_above_one_rejected() -> None:
    with pytest.raises(ValueError, match="a2ats_beta"):
        _make(a2ats_beta=1.5)


def test_beta_negative_rejected() -> None:
    with pytest.raises(ValueError, match="a2ats_beta"):
        _make(a2ats_beta=-0.1)


def test_retrieval_fraction_above_one_rejected() -> None:
    with pytest.raises(ValueError, match="a2ats_retrieval_fraction"):
        _make(a2ats_retrieval_fraction=1.2)


def test_retrieval_fraction_negative_rejected() -> None:
    with pytest.raises(ValueError, match="a2ats_retrieval_fraction"):
        _make(a2ats_retrieval_fraction=-0.1)


def test_odd_head_dim_rejected() -> None:
    with pytest.raises(ValueError, match="must be even"):
        _make(head_dim=33)


def test_head_dim_not_divisible_by_sub_dim_rejected() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        _make(head_dim=30, a2ats_sub_dim=8)


# ---------------------------------------------------------------------------
# Factory dispatch + shape
# ---------------------------------------------------------------------------

def test_factory_dispatch() -> None:
    c = _make()
    assert isinstance(c, A2ATSKVCache)


def test_prefill_output_shape_preserved() -> None:
    c = _make()
    k, v = _rand_kv(S=12)
    ko, vo = c.update_and_fetch(k, v)
    assert ko.shape == (1, 2, 12, 32)
    assert vo.shape == (1, 2, 12, 32)


def test_decode_after_prefill_accumulates() -> None:
    c = _make()
    k1, v1 = _rand_kv(S=8, seed=1)
    c.update_and_fetch(k1, v1)
    k2, v2 = _rand_kv(S=1, seed=2)
    ko, vo = c.update_and_fetch(k2, v2)
    assert ko.shape == (1, 2, 9, 32)
    assert c.tokens_seen == 9


def test_no_nan_in_output() -> None:
    c = _make()
    k, v = _rand_kv(S=16, seed=3)
    ko, vo = c.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert not bool(mx.any(mx.isnan(ko)).item())
    assert not bool(mx.any(mx.isnan(vo)).item())


# ---------------------------------------------------------------------------
# use_query_aware toggle
# ---------------------------------------------------------------------------

def test_query_aware_off_runs_without_crash() -> None:
    c = _make(a2ats_use_query_aware=False)
    k, v = _rand_kv(S=10, seed=4)
    ko, vo = c.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert ko.shape == (1, 2, 10, 32)
    assert c.tokens_retrieved == 0


def test_query_aware_on_tracks_retrieved_tokens() -> None:
    c = _make(a2ats_use_query_aware=True, a2ats_retrieval_fraction=0.3)
    k, v = _rand_kv(S=20, H=1, seed=5)
    c.update_and_fetch(k, v)
    assert c.tokens_retrieved > 0


# ---------------------------------------------------------------------------
# Windowed RoPE integration — near tokens should reconstruct with lower
# error than far tokens would under an equivalent all-approximate baseline
# ---------------------------------------------------------------------------

def test_large_window_reduces_to_always_exact_no_crash() -> None:
    c = _make(a2ats_window=10_000)
    k, v = _rand_kv(S=10, seed=6)
    ko, vo = c.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert not bool(mx.any(mx.isnan(ko)).item())


def test_zero_window_no_crash() -> None:
    c = _make(a2ats_window=0)
    k, v = _rand_kv(S=10, seed=7)
    ko, vo = c.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert not bool(mx.any(mx.isnan(ko)).item())


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------

def test_compression_ratio_greater_than_one() -> None:
    c = _make()
    k, v = _rand_kv(S=20, seed=8)
    c.update_and_fetch(k, v)
    assert c.compression_ratio > 1.0


def test_compressed_bytes_less_than_fp16() -> None:
    c = _make()
    k, v = _rand_kv(S=20, seed=9)
    c.update_and_fetch(k, v)
    assert c.compressed_key_bytes < c.fp16_key_bytes
    assert c.compressed_value_bytes < c.fp16_value_bytes


def test_assigned_avg_bits_matches_formula() -> None:
    c = _make(head_dim=32, a2ats_sub_dim=8, a2ats_codebook_bits=6)
    expected = (32 // 8) * 6 / 32
    assert c.assigned_avg_bits == pytest.approx(expected)


def test_codebook_bytes_static() -> None:
    c = _make(a2ats_codebook_bits=6, a2ats_sub_dim=8)
    assert c.codebook_bytes == (2 ** 6) * 8 * 2


def test_compression_ratio_before_any_update_is_one() -> None:
    c = _make()
    assert c.compression_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# for_model construction
# ---------------------------------------------------------------------------

def test_build_via_for_model_propagates_config() -> None:
    class _Attn:
        head_dim = 32

    class _Layer:
        self_attn = _Attn()

    class _Model:
        layers = [_Layer(), _Layer(), _Layer()]

    cfg = KVCacheConfig(
        method="a2ats", head_dim=32,
        a2ats_window=64, a2ats_beta=0.7, a2ats_retrieval_fraction=0.35,
    )
    caches = KVCacheBuilder.for_model(_Model(), cfg)
    assert all(isinstance(c, A2ATSKVCache) for c in caches)
    assert caches[0]._window == 64
    assert caches[0]._beta == pytest.approx(0.7)
    assert caches[0]._retrieval_fraction == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism() -> None:
    k, v = _rand_kv(S=15, seed=10)
    c1 = _make()
    c2 = _make()
    ko1, vo1 = c1.update_and_fetch(k, v)
    ko2, vo2 = c2.update_and_fetch(k, v)
    mx.eval(ko1, ko2, vo1, vo2)
    assert np.allclose(np.array(ko1), np.array(ko2))
    assert np.allclose(np.array(vo1), np.array(vo2))


def test_determinism_across_prefill_and_decode() -> None:
    def _run():
        c = _make(seed=42)
        k1, v1 = _rand_kv(S=8, seed=11)
        c.update_and_fetch(k1, v1)
        k2, v2 = _rand_kv(S=1, seed=12)
        return c.update_and_fetch(k2, v2)

    ko1, vo1 = _run()
    ko2, vo2 = _run()
    mx.eval(ko1, ko2, vo1, vo2)
    assert np.allclose(np.array(ko1), np.array(ko2))
    assert np.allclose(np.array(vo1), np.array(vo2))


# ---------------------------------------------------------------------------
# Calibrated vs. random-init codebook
# ---------------------------------------------------------------------------

def test_explicit_codebook_used_when_provided() -> None:
    codebook = mx.zeros((2 ** 6, 8), dtype=mx.float32)
    c = _make(a2ats_codebook=codebook)
    assert np.allclose(np.array(c._codebook), np.array(codebook))


def test_random_init_codebook_when_absent() -> None:
    c = _make()
    assert c._codebook.shape == (2 ** 6, 8)


# ---------------------------------------------------------------------------
# Empty / degenerate sequence
# ---------------------------------------------------------------------------

def test_single_token_prefill() -> None:
    c = _make()
    k, v = _rand_kv(S=1, seed=13)
    ko, vo = c.update_and_fetch(k, v)
    mx.eval(ko, vo)
    assert ko.shape == (1, 2, 1, 32)
    assert not bool(mx.any(mx.isnan(ko)).item())


# ---------------------------------------------------------------------------
# Distance gating tracks the advancing decode position (:issue:`29`, finding 3)
#
# Coverage gap: nothing asserted that a token's near/far class follows the
# decode position. That gap is why the frozen-rotation bug went unnoticed —
# the parent KVCache concatenates whatever it is handed and never revisits it,
# so writing rotated keys froze each token's class at write time.
# ---------------------------------------------------------------------------

def test_token_rotation_updates_as_decode_position_advances() -> None:
    """A token classified "near" at prefill must lose its exact rotation once
    the decode position moves past the window.

    Paper Eq. (11) makes near/far a function of the *current* decoding query
    ``i``, which advances every step.
    """
    cache = _make(a2ats_window=4, a2ats_use_query_aware=False)
    k, v = _rand_kv(B=1, H=1, S=6, seed=5)
    k_prefill, _ = cache.update_and_fetch(k, v)
    tok5_near = np.array(k_prefill[0, 0, 5]).astype(np.float32)

    for i in range(10):
        k1, v1 = _rand_kv(B=1, H=1, S=1, seed=100 + i)
        k_now, _ = cache.update_and_fetch(k1, v1)

    tok5_far = np.array(k_now[0, 0, 5]).astype(np.float32)
    # Position 5 was distance 0 at prefill (near/exact); it is now 10 steps
    # back, well outside window=4, so it must have become far/unrotated.
    assert not np.allclose(tok5_near, tok5_far, atol=1e-2)


def test_far_tokens_in_cache_are_unrotated() -> None:
    """Eq. (12) end-to-end through the cache: once a token is outside the
    window, its stored key is returned in the pre-RoPE frame — so two tokens
    with the same underlying key content are position-independent."""
    cache = _make(a2ats_window=2, a2ats_use_query_aware=False)
    k, v = _rand_kv(B=1, H=1, S=8, seed=6)
    k_out, _ = cache.update_and_fetch(k, v)

    # Far rows (distance >= 2 from query_position=7) must be position-free:
    # re-fetching after more decode steps leaves them untouched.
    far_before = np.array(k_out[0, 0, :5]).astype(np.float32)
    for i in range(3):
        k1, v1 = _rand_kv(B=1, H=1, S=1, seed=200 + i)
        k_now, _ = cache.update_and_fetch(k1, v1)
    far_after = np.array(k_now[0, 0, :5]).astype(np.float32)
    assert np.allclose(far_before, far_after, atol=1e-2)


def test_cache_length_grows_across_prefill_and_decode() -> None:
    """The per-step re-rotation must not disturb normal cache accumulation."""
    cache = _make(a2ats_window=4)
    k, v = _rand_kv(B=1, H=2, S=6, seed=7)
    k_out, v_out = cache.update_and_fetch(k, v)
    assert k_out.shape == (1, 2, 6, 32)
    for step in range(1, 4):
        k1, v1 = _rand_kv(B=1, H=2, S=1, seed=300 + step)
        k_out, v_out = cache.update_and_fetch(k1, v1)
        assert k_out.shape == (1, 2, 6 + step, 32)
        assert v_out.shape == (1, 2, 6 + step, 32)


# ---------------------------------------------------------------------------
# H-weighted assignment wiring (:issue:`29`, finding 4)
# ---------------------------------------------------------------------------

def test_query_h_enables_paper_assignment_path() -> None:
    """Supplying a calibrated ``a2ats_query_h`` switches the retrieval set to
    the paper's Eq. (14) objective instead of the cosine-blend substitute."""
    from veloxquant_mlx.quantizers.a2ats import a2ats_query_second_moment

    h = a2ats_query_second_moment(mx.array(
        np.random.default_rng(0).standard_normal((40, 8)).astype(np.float32)
    ))
    cache = _make(a2ats_query_h=h)
    assert cache._cholesky_factor is not None
    k, v = _rand_kv(B=1, H=2, S=8, seed=8)
    k_out, v_out = cache.update_and_fetch(k, v)
    assert k_out.shape == (1, 2, 8, 32)
    assert not bool(mx.any(mx.isnan(k_out)).item())


def test_query_h_absent_falls_back_to_cosine_blend() -> None:
    cache = _make()
    assert cache._cholesky_factor is None


def test_query_h_shape_is_validated() -> None:
    """``H`` is a per-sub-vector matrix — a head_dim-sized one is a config
    error, not something to silently broadcast."""
    with pytest.raises(ValueError, match="a2ats_query_h must have shape"):
        _make(a2ats_query_h=np.eye(32, dtype=np.float32))
