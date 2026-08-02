"""Tests for A2ATS-adapted windowed RoPE — distance-gated exact/approximate
rotation.

A2ATS-adapted (He et al., ACL 2025 Findings, aclanthology.org/2025.findings-acl.644)
applies exact RoPE to tokens within a trailing window of the current decode
position, and a single fixed-offset approximate rotation to tokens outside it.
All data is synthetic.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.a2ats_rope import (
    a2ats_apply_exact_rope,
    a2ats_apply_windowed_rope,
)


def _mat(n, d, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((n, d)).astype(np.float32))


# ---------------------------------------------------------------------------
# a2ats_apply_exact_rope
# ---------------------------------------------------------------------------


def test_exact_rope_preserves_shape_and_dtype() -> None:
    x = _mat(6, 16)
    out = a2ats_apply_exact_rope(x, mx.arange(6))
    assert out.shape == (6, 16)
    assert out.dtype == mx.float16


def test_exact_rope_position_zero_is_near_identity() -> None:
    """At position 0, cos=1, sin=0 for every frequency -> rotation is identity."""
    x = _mat(1, 8, seed=3)
    out = a2ats_apply_exact_rope(x, mx.array([0]))
    assert np.allclose(np.array(out), np.array(x.astype(mx.float16)), atol=1e-2)


def test_exact_rope_different_positions_differ() -> None:
    x = mx.array(np.ones((2, 8), dtype=np.float32))
    out = a2ats_apply_exact_rope(x, mx.array([0, 50]))
    assert not np.allclose(np.array(out[0]), np.array(out[1]), atol=1e-3)


def test_exact_rope_empty_input() -> None:
    x = mx.zeros((0, 8))
    out = a2ats_apply_exact_rope(x, mx.array([], dtype=mx.int32))
    assert out.shape == (0, 8)


def test_exact_rope_matches_hand_computed_rotation() -> None:
    """Direct numeric check against the standard RoPE formula for one token."""
    d = 4
    x = mx.array([[1.0, 0.0, 1.0, 0.0]], dtype=mx.float32)
    pos = mx.array([2])
    base = 10000.0
    out = a2ats_apply_exact_rope(x, pos, base=base)

    half = d // 2
    inv_freq = 1.0 / (base ** (np.arange(half) / half))
    angle = 2 * inv_freq
    cos, sin = np.cos(angle), np.sin(angle)
    x1, x2 = np.array([1.0, 0.0]), np.array([1.0, 0.0])
    expected = np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos])
    assert np.allclose(np.array(out[0]), expected, atol=1e-2)


# ---------------------------------------------------------------------------
# a2ats_apply_windowed_rope — window boundary behavior
# ---------------------------------------------------------------------------


def test_windowed_rope_within_window_matches_exact_rope() -> None:
    """Tokens inside the trailing window get RoPE identical to the exact path."""
    x = _mat(10, 16, seed=1)
    positions = mx.arange(10)
    exact = a2ats_apply_exact_rope(x, positions)
    windowed = a2ats_apply_windowed_rope(x, positions, query_position=9, window=3)

    # distance = 9 - position; near iff distance < 3 -> positions 7, 8, 9
    for i in (7, 8, 9):
        assert np.allclose(np.array(windowed[i]), np.array(exact[i]), atol=1e-2), i


def test_windowed_rope_outside_window_uses_fixed_offset() -> None:
    """Tokens outside the window must NOT match exact RoPE — the approximation
    must actually differ, not accidentally coincide (mirrors the AMC
    saturated-clamp bug this repo already caught once: an "approximation"
    that silently equals the exact path would hide a real bug).

    Note: the far-bucket rotation uses angle ``window * inv_freq``, which is
    mathematically identical to a token's *exact* rotation at absolute
    position == window (a genuine RoPE periodicity property, not a logic
    bug). This test's positions are offset from 0 so that no far token's
    absolute position coincides with ``window`` itself.
    """
    x = _mat(10, 16, seed=1)
    positions = mx.arange(100, 110)  # offset so none equals window=3
    exact = a2ats_apply_exact_rope(x, positions)
    windowed = a2ats_apply_windowed_rope(x, positions, query_position=109, window=3)

    for i in (0, 1, 2, 3, 4, 5):
        assert not np.allclose(np.array(windowed[i]), np.array(exact[i]), atol=1e-2), i


def test_windowed_rope_far_tokens_share_single_rotation() -> None:
    """All far tokens get the SAME rotation (the fixed-offset approximation),
    not each their own distinct exact rotation — the defining property of
    the approximation bucket."""
    x = mx.array(np.ones((5, 8), dtype=np.float32))  # identical inputs
    positions = mx.array([0, 1, 2, 3, 4])
    windowed = a2ats_apply_windowed_rope(x, positions, query_position=100, window=3)
    # All 5 tokens are far (distance >= 97 >> window=3); since inputs are
    # identical and rotation is shared, outputs must be identical too.
    for i in range(1, 5):
        assert np.allclose(np.array(windowed[0]), np.array(windowed[i]), atol=1e-4)


def test_window_zero_always_approximate() -> None:
    """window<=0 -> distance < window is never true -> every token is "far".

    Positions are offset from 0 so no token's absolute position coincides
    with the far-bucket's offset (0 for window<=0, a RoPE-identity angle —
    see test_windowed_rope_outside_window_uses_fixed_offset's note).
    """
    x = _mat(6, 8, seed=2)
    positions = mx.arange(100, 106)
    exact = a2ats_apply_exact_rope(x, positions)
    windowed = a2ats_apply_windowed_rope(x, positions, query_position=105, window=0)
    for i in range(6):
        assert not np.allclose(np.array(windowed[i]), np.array(exact[i]), atol=1e-2), i


def test_window_exceeds_seqlen_always_exact() -> None:
    x = _mat(6, 8, seed=2)
    positions = mx.arange(6)
    exact = a2ats_apply_exact_rope(x, positions)
    windowed = a2ats_apply_windowed_rope(x, positions, query_position=5, window=10_000)
    assert np.allclose(np.array(windowed), np.array(exact), atol=1e-2)


def test_windowed_rope_empty_input() -> None:
    x = mx.zeros((0, 8))
    out = a2ats_apply_windowed_rope(x, mx.array([], dtype=mx.int32), query_position=0, window=4)
    assert out.shape == (0, 8)


def test_windowed_rope_shape_dtype_preserved() -> None:
    x = _mat(4, 12, seed=9)
    out = a2ats_apply_windowed_rope(x, mx.arange(4), query_position=3, window=2)
    assert out.shape == (4, 12)
    assert out.dtype == mx.float16


def test_windowed_rope_no_nan_at_boundaries() -> None:
    x = _mat(20, 16, seed=4)
    positions = mx.arange(20)
    for window in (0, 1, 20, 10_000):
        out = a2ats_apply_windowed_rope(x, positions, query_position=19, window=window)
        assert not bool(mx.any(mx.isnan(out)).item()), window
