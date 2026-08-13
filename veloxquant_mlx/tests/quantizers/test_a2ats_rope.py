"""Tests for A2ATS-adapted windowed RoPE — distance-gated exact/approximate
rotation.

A2ATS-adapted (He et al., ACL 2025 Findings, aclanthology.org/2025.findings-acl.644)
applies exact RoPE to tokens within a trailing window of the current decode
position; tokens outside it are left **unrotated** on the key side (Eq. 12,
``k̃_i = k_i``), with the constant ``R_b`` applied to the query instead
(Eq. 11). All data is synthetic.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.a2ats_rope import (
    a2ats_apply_exact_rope,
    a2ats_apply_far_query_rope,
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


def test_windowed_rope_outside_window_returns_key_unrotated() -> None:
    """Paper Eq. (12): far keys are returned in their **pre-RoPE** frame,
    ``k̃_i = k_i`` — identical to the input, not merely different from exact
    RoPE.

    This is the strong form of the assertion. The pre-:issue:`29` version of
    this test only checked far tokens *differed* from exact RoPE, which the
    buggy ``R_window`` rotation also satisfied — so the bug passed. Asserting
    exact equality with the input is what actually pins Eq. (12).
    """
    x = _mat(10, 16, seed=1)
    positions = mx.arange(100, 110)
    windowed = a2ats_apply_windowed_rope(x, positions, query_position=109, window=3)

    for i in (0, 1, 2, 3, 4, 5):
        assert np.allclose(
            np.array(windowed[i]).astype(np.float32),
            np.array(x[i]).astype(np.float32),
            atol=1e-2,
        ), i


def test_windowed_rope_far_tokens_are_position_independent() -> None:
    """The defining property of Eq. (12): far keys carry **no** positional
    information, so identical inputs at different positions give identical
    outputs.

    This is what makes a single shared codebook viable across inputs
    (paper §3.1, Observation 2) — the entire reason WRoPE exists.
    """
    x = mx.array(np.ones((5, 8), dtype=np.float32))  # identical inputs
    positions = mx.array([0, 17, 133, 2040, 9001])  # wildly different positions
    windowed = a2ats_apply_windowed_rope(x, positions, query_position=100_000, window=3)
    for i in range(1, 5):
        assert np.allclose(np.array(windowed[0]), np.array(windowed[i]), atol=1e-4)


def test_window_zero_always_unrotated() -> None:
    """window<=0 -> distance < window is never true -> every token is "far",
    so every key comes back in the pure pre-RoPE frame (Eq. 12)."""
    x = _mat(6, 8, seed=2)
    positions = mx.arange(100, 106)
    windowed = a2ats_apply_windowed_rope(x, positions, query_position=105, window=0)
    assert np.allclose(
        np.array(windowed).astype(np.float32),
        np.array(x).astype(np.float32),
        atol=1e-2,
    )


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


# ---------------------------------------------------------------------------
# a2ats_apply_far_query_rope — the query-side half of WRoPE (Eq. 11)
#
# Coverage gap identified in :issue:`29`: nothing pinned `b` as a knob
# independent of the window size `w`.
# ---------------------------------------------------------------------------


def test_far_query_rope_b_is_independent_of_window() -> None:
    """Paper §5.1 sets ``w=64`` and ``b=2048`` — a factor of 32 apart, so they
    are plainly not one value.

    Before :issue:`29` the far offset was derived from ``window``, hardcoding
    ``b == w`` and making the paper's own configuration inexpressible. This
    pins that ``b`` is a real, separate knob.
    """
    q = _mat(3, 16, seed=11)
    at_2048 = a2ats_apply_far_query_rope(q, b=2048)
    at_64 = a2ats_apply_far_query_rope(q, b=64)
    assert not np.allclose(np.array(at_2048), np.array(at_64), atol=1e-2)


def test_far_query_rope_matches_exact_rope_at_position_b() -> None:
    """``R_b`` is just the RoPE rotation at relative position ``b`` — so it
    must equal exact RoPE applied at absolute position ``b``."""
    q = _mat(4, 16, seed=12)
    b = 2048
    via_far = a2ats_apply_far_query_rope(q, b=b)
    via_exact = a2ats_apply_exact_rope(q, mx.array([b] * 4))
    assert np.allclose(np.array(via_far), np.array(via_exact), atol=1e-2)


def test_far_query_rope_accepts_1d_and_2d() -> None:
    """A single decode query is ``[D]``; a batch is ``[N, D]``. Both work,
    and the 1-D path is the 2-D path's first row."""
    q1 = _mat(1, 8, seed=13)[0]
    out1 = a2ats_apply_far_query_rope(q1, b=512)
    assert out1.shape == (8,)
    out2 = a2ats_apply_far_query_rope(q1[None, :], b=512)
    assert out2.shape == (1, 8)
    assert np.allclose(np.array(out1), np.array(out2[0]), atol=1e-3)


def test_far_query_rope_reconstructs_paper_attention_score() -> None:
    """End-to-end Eq. (11) for a far token: ``u_ij = q_i R_b k_j^T``.

    Composing the two halves — unrotated far key (Eq. 12) and ``R_b``-rotated
    query — must reproduce the paper's far-token score. This is the property
    the pre-:issue:`29` key-side ``R_window`` rotation broke: it computed
    ``q_i R_w^T k_j^T``, rotating the wrong operand in the wrong direction.
    """
    b, d = 2048, 16
    q = _mat(1, d, seed=14)[0]
    k = _mat(1, d, seed=15)

    far_key = a2ats_apply_windowed_rope(k, mx.array([0]), query_position=100_000, window=8)
    rotated_q = a2ats_apply_far_query_rope(q, b=b)
    got = float((rotated_q.astype(mx.float32) @ far_key.astype(mx.float32).T).item())

    # Reference: rotate the query by R_b, leave the key alone, take the dot.
    q_ref = np.array(a2ats_apply_exact_rope(q[None, :], mx.array([b])).astype(mx.float32))[0]
    want = float(q_ref @ np.array(k.astype(mx.float32))[0])

    assert abs(got - want) < 1e-1, (got, want)
