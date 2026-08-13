"""Content-space RoPE tests for cross-model KV transfer.

The mapping's correctness rests on RoPE being exactly invertible: if
``strip_rope`` does not perfectly undo the source model's rotation, the ridge
fit is done on contaminated features and every downstream number is wrong for
a reason no benchmark would localize. These tests pin that property, the
two-base composition (the case that distinguishes this module from
``rope_remap_positions``), and the shapes the apply path actually feeds it.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx.transfer.rope import (
    apply_rope,
    recode_rope,
    rope_angles,
    strip_rope,
)

SRC_BASE = 10000.0
TGT_BASE = 1000000.0


@pytest.fixture
def keys() -> mx.array:
    mx.random.seed(0)
    return mx.random.normal((3, 32, 64)).astype(mx.float32)


@pytest.fixture
def positions() -> mx.array:
    return mx.arange(32)


def test_strip_then_apply_is_identity(keys, positions):
    """Round-tripping at one base must return the original keys."""
    out = apply_rope(strip_rope(keys, positions, SRC_BASE), positions, SRC_BASE)
    assert float(mx.max(mx.abs(out - keys))) < 1e-5


def test_rotation_preserves_norm(keys, positions):
    """RoPE is orthogonal, so re-encoding must not change per-token norms."""
    out = recode_rope(keys, positions, SRC_BASE, TGT_BASE)
    before = mx.linalg.norm(keys, axis=-1)
    after = mx.linalg.norm(out, axis=-1)
    assert float(mx.max(mx.abs(after - before))) < 1e-4


def test_recode_matches_strip_then_apply(keys, positions):
    """The fused single-rotation form must equal the explicit two-step form.

    This is the property that lets the Metal kernel exist at all.
    """
    two_step = apply_rope(strip_rope(keys, positions, SRC_BASE), positions, TGT_BASE)
    fused = recode_rope(keys, positions, SRC_BASE, TGT_BASE, use_metal=False)
    assert float(mx.max(mx.abs(two_step - fused))) < 1e-5


def test_recode_with_equal_bases_is_identity(keys, positions):
    """Same source and target base means no net rotation.

    Guards the sign convention: an inverted subtraction would double the
    rotation here instead of cancelling it.
    """
    out = recode_rope(keys, positions, SRC_BASE, SRC_BASE, use_metal=False)
    assert float(mx.max(mx.abs(out - keys))) < 1e-5


def test_position_zero_is_unrotated(keys):
    """RoPE at position 0 is the identity rotation for any base."""
    pos = mx.zeros((keys.shape[1],))
    assert float(mx.max(mx.abs(strip_rope(keys, pos, SRC_BASE) - keys))) < 1e-5


def test_noncontiguous_positions_supported(keys):
    """Positions need not be contiguous — post-eviction caches have gaps."""
    pos = mx.array([0, 5, 11, 12, 40] + list(range(100, 127)))
    out = apply_rope(strip_rope(keys, pos, SRC_BASE), pos, SRC_BASE)
    assert float(mx.max(mx.abs(out - keys))) < 1e-4


def test_empty_sequence_is_passthrough():
    """A zero-token cache must not crash the rotation path."""
    empty = mx.zeros((2, 0, 64))
    assert strip_rope(empty, mx.zeros((0,)), SRC_BASE).shape == (2, 0, 64)
    assert recode_rope(empty, mx.zeros((0,)), SRC_BASE, TGT_BASE).shape == (2, 0, 64)


@pytest.mark.parametrize("shape", [(16, 64), (4, 16, 64), (2, 3, 16, 64)])
def test_rank_agnostic(shape, positions):
    """Rotation applies over the last two axes for 2-D, 3-D and 4-D inputs."""
    mx.random.seed(1)
    x = mx.random.normal(shape).astype(mx.float32)
    pos = mx.arange(shape[-2])
    out = apply_rope(strip_rope(x, pos, SRC_BASE), pos, SRC_BASE)
    assert out.shape == shape
    assert float(mx.max(mx.abs(out - x))) < 1e-5


def test_rope_angles_match_reference_formula():
    """Angles must be ``p * base^(-d/half)`` — the convention the fit assumes.

    Pinned against an explicit recomputation so a refactor cannot silently
    change the frequency schedule and invalidate every saved mapper.
    """
    pos = mx.array([0.0, 1.0, 7.0])
    angles = rope_angles(pos, 8, SRC_BASE)
    half = 4
    expected = mx.array(
        [[float(p) * (SRC_BASE ** (-d / half)) for d in range(half)] for p in [0.0, 1.0, 7.0]]
    )
    assert float(mx.max(mx.abs(angles - expected))) < 1e-6


def test_long_context_precision():
    """Angle differencing must stay accurate at large positions.

    ``recode_rope`` subtracts inverse frequencies *before* multiplying by the
    position; doing it the other way loses significant bits once ``p`` reaches
    the thousands, which is exactly the long-context regime this is for.
    """
    mx.random.seed(2)
    n = 4096
    x = mx.random.normal((2, n, 64)).astype(mx.float32)
    pos = mx.arange(n)
    two_step = apply_rope(strip_rope(x, pos, SRC_BASE), pos, TGT_BASE)
    fused = recode_rope(x, pos, SRC_BASE, TGT_BASE, use_metal=False)
    assert float(mx.max(mx.abs(two_step - fused))) < 1e-3
