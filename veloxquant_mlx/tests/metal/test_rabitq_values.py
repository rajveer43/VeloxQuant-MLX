"""Tests for nibble-packed 4-bit values: pack kernel + packed attend path.

The strongest guarantee here is bit-exactness: the packed and unpacked
attend paths read identical index values and perform identical
arithmetic, so their fp16 outputs must be *equal*, not just close.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import pytest

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import rabitq_fused_attend, rabitq_pack_values
from veloxquant_mlx.tests.metal.test_rabitq_attend import (
    _make_inputs,
    _reference_attend,
)

pytestmark = pytest.mark.skipif(
    not metal_available(),
    reason="Metal compute kernels not available on this build of mlx.",
)


def _pack_np(v_idx: np.ndarray) -> np.ndarray:
    """Reference numpy nibble packing (low nibble = even element)."""
    lo = v_idx[..., 0::2] & 0xF
    hi = v_idx[..., 1::2] & 0xF
    return (lo | (hi << 4)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Pack kernel parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(6,), (4, 64), (1, 2, 33, 128), (2, 2, 7, 256)])
def test_pack_values_parity(shape):
    rng = np.random.default_rng(sum(shape))
    v_idx = rng.integers(0, 16, shape, dtype=np.uint8)

    packed = rabitq_pack_values(mx.array(v_idx))
    mx.eval(packed)

    assert packed.shape == (*shape[:-1], shape[-1] // 2)
    assert packed.dtype == mx.uint8
    np.testing.assert_array_equal(np.array(packed), _pack_np(v_idx))


def test_pack_values_masks_out_of_range():
    """Indices >= 16 must be masked to 4 bits, never corrupt a neighbour."""
    v_idx = np.array([[0xFF, 0x01, 0x12, 0x03]], dtype=np.uint8)
    packed = rabitq_pack_values(mx.array(v_idx))
    mx.eval(packed)
    np.testing.assert_array_equal(np.array(packed), _pack_np(v_idx))


def test_pack_values_rejects_odd_last_dim():
    with pytest.raises(ValueError, match="even"):
        rabitq_pack_values(mx.array(np.zeros((4, 7), np.uint8)))


# ---------------------------------------------------------------------------
# Packed attend path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("S_kv", [1, 33, 256])
def test_attend_packed_matches_unpacked_exactly(D, S_kv):
    """Packed and unpacked v_idx must give bit-identical fp16 outputs."""
    B, H, S_q = 1, 2, 4
    q, q_scale, k_bits, k_mag, k_const, v_idx, v_cents = _make_inputs(
        B, H, S_q, S_kv, D, seed=D + S_kv
    )

    args = (
        mx.array(q),
        mx.array(q_scale),
        mx.array(k_bits),
        mx.array(k_mag),
        mx.array(k_const),
    )
    out_unpacked = rabitq_fused_attend(*args, mx.array(v_idx), mx.array(v_cents))
    out_packed = rabitq_fused_attend(*args, rabitq_pack_values(mx.array(v_idx)), mx.array(v_cents))
    mx.eval(out_unpacked, out_packed)

    np.testing.assert_array_equal(np.array(out_unpacked), np.array(out_packed))


def test_attend_packed_matches_reference():
    B, H, S_q, S_kv, D = 2, 2, 8, 33, 128
    q, q_scale, k_bits, k_mag, k_const, v_idx, v_cents = _make_inputs(B, H, S_q, S_kv, D, seed=11)
    expected = _reference_attend(q, q_scale, k_bits, k_mag, k_const, v_idx, v_cents)

    out = rabitq_fused_attend(
        mx.array(q),
        mx.array(q_scale),
        mx.array(k_bits),
        mx.array(k_mag),
        mx.array(k_const),
        rabitq_pack_values(mx.array(v_idx)),
        mx.array(v_cents),
    )
    mx.eval(out)
    np.testing.assert_allclose(np.array(out, dtype=np.float32), expected, atol=1e-2, rtol=1e-2)


def test_attend_packed_rejects_large_codebook():
    B, H, S_q, S_kv, D = 1, 1, 1, 4, 64
    q, q_scale, k_bits, k_mag, k_const, v_idx, _ = _make_inputs(B, H, S_q, S_kv, D)
    v_cents_32 = np.linspace(-1, 1, 32).astype(np.float32)
    packed = rabitq_pack_values(mx.array(v_idx))
    with pytest.raises(ValueError, match="16"):
        rabitq_fused_attend(
            mx.array(q),
            mx.array(q_scale),
            mx.array(k_bits),
            mx.array(k_mag),
            mx.array(k_const),
            packed,
            mx.array(v_cents_32),
        )


def test_attend_rejects_wrong_v_idx_width():
    B, H, S_q, S_kv, D = 1, 1, 1, 4, 64
    q, q_scale, k_bits, k_mag, k_const, v_idx, v_cents = _make_inputs(B, H, S_q, S_kv, D)
    with pytest.raises(ValueError, match="nibble-packed"):
        rabitq_fused_attend(
            mx.array(q),
            mx.array(q_scale),
            mx.array(k_bits),
            mx.array(k_mag),
            mx.array(k_const),
            mx.array(v_idx[..., : D // 4]),
            mx.array(v_cents),
        )
