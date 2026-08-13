"""Tests for BitPackBuffer — roundtrip packing for b = 1, 2, 3, 4."""

from __future__ import annotations

import numpy as np
import pytest

from veloxquant_mlx.dsa.bit_pack import BitPackBuffer


@pytest.mark.parametrize("b", [1, 2, 3, 4])
@pytest.mark.parametrize("n", [1, 7, 8, 15, 16, 100, 128, 255, 256, 1000])
def test_bit_pack_roundtrip(b: int, n: int) -> None:
    buf = BitPackBuffer(b=b)
    rng = np.random.default_rng(b * 1000 + n)
    original = rng.integers(0, 2**b, size=n, dtype=np.uint8)
    packed = buf.pack(original)
    recovered = buf.unpack(packed, n)
    np.testing.assert_array_equal(original, recovered)


def test_invalid_b() -> None:
    with pytest.raises(ValueError):
        BitPackBuffer(b=5)
    with pytest.raises(ValueError):
        BitPackBuffer(b=0)


def test_value_out_of_range() -> None:
    buf = BitPackBuffer(b=2)
    bad = np.array([0, 1, 4], dtype=np.uint8)  # 4 >= 2^2
    with pytest.raises(ValueError):
        buf.pack(bad)


def test_packed_size_b2() -> None:
    buf = BitPackBuffer(b=2)
    idx = np.zeros(8, dtype=np.uint8)
    packed = buf.pack(idx)
    assert len(packed) == 2  # 8 values / 4 per byte = 2 bytes


def test_packed_size_b3() -> None:
    buf = BitPackBuffer(b=3)
    idx = np.zeros(8, dtype=np.uint8)
    packed = buf.pack(idx)
    assert len(packed) == 3  # 8 values × 3 bits = 24 bits = 3 bytes


def test_packed_size_b4() -> None:
    buf = BitPackBuffer(b=4)
    idx = np.zeros(16, dtype=np.uint8)
    packed = buf.pack(idx)
    assert len(packed) == 8  # 16 values / 2 per byte


def test_all_ones_b1() -> None:
    buf = BitPackBuffer(b=1)
    original = np.ones(8, dtype=np.uint8)
    packed = buf.pack(original)
    recovered = buf.unpack(packed, 8)
    np.testing.assert_array_equal(original, recovered)


@pytest.mark.parametrize("b", [1, 2, 3, 4])
def test_asymmetric_lengths(b: int) -> None:
    buf = BitPackBuffer(b=b)
    for n in [1, 3, 5, 7, 9, 11, 13]:
        if n == 0:
            continue
        original = np.zeros(n, dtype=np.uint8)
        original[0] = 2**b - 1
        packed = buf.pack(original)
        recovered = buf.unpack(packed, n)
        np.testing.assert_array_equal(original, recovered, err_msg=f"b={b}, n={n}")
