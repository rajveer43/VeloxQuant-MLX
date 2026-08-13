"""Tests for the order-0 Huffman entropy coder (quantizers/_entropy_coding.py).

Covers: lossless round-trip on random integer code arrays across several
alphabet sizes/lengths, the code table's byte cost being counted (not
hidden), and degenerate single-symbol / empty inputs not crashing.
"""

from __future__ import annotations

import numpy as np
import pytest

from veloxquant_mlx.quantizers._entropy_coding import (
    entropy_decode,
    entropy_encode,
    table_nbytes,
)


# ---------------------------------------------------------------------------
# round-trip losslessness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "seed,n,alphabet",
    [
        (0, 1, 2),
        (1, 10, 3),
        (2, 200, 16),
        (3, 1000, 256),
        (4, 500, 5),
        (5, 37, 64),
    ],
)
def test_round_trip_lossless(seed, n, alphabet):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, alphabet, size=n)
    payload, table = entropy_encode(codes)
    decoded = entropy_decode(payload, table, n)
    assert np.array_equal(codes, decoded)


def test_round_trip_skewed_distribution():
    """Highly skewed frequency (mostly one symbol) — the case where Huffman
    coding should meaningfully compress, and a good round-trip stress test
    for the variable-length codes."""
    rng = np.random.default_rng(9)
    n = 2000
    codes = np.where(rng.random(n) < 0.9, 0, rng.integers(1, 10, size=n))
    payload, table = entropy_encode(codes)
    decoded = entropy_decode(payload, table, n)
    assert np.array_equal(codes, decoded)


# ---------------------------------------------------------------------------
# table overhead counted, not hidden
# ---------------------------------------------------------------------------
def test_table_nbytes_positive_for_nonempty_alphabet():
    codes = np.array([1, 2, 3, 1, 2, 1])
    _, table = entropy_encode(codes)
    assert table_nbytes(table) > 0
    assert len(table) == 3


def test_table_nbytes_scales_with_alphabet_size():
    rng = np.random.default_rng(0)
    small = rng.integers(0, 4, size=500)
    large = rng.integers(0, 200, size=500)
    _, table_small = entropy_encode(small)
    _, table_large = entropy_encode(large)
    assert table_nbytes(table_large) > table_nbytes(table_small)


def test_table_nbytes_zero_for_empty_input():
    _, table = entropy_encode(np.array([], dtype=np.int64))
    assert table_nbytes(table) == 0


# ---------------------------------------------------------------------------
# degenerate inputs don't crash
# ---------------------------------------------------------------------------
def test_single_symbol_input_does_not_crash():
    codes = np.full(50, 7)
    payload, table = entropy_encode(codes)
    assert table == {7: "0"}
    decoded = entropy_decode(payload, table, 50)
    assert np.array_equal(decoded, codes)


def test_single_element_array():
    codes = np.array([3])
    payload, table = entropy_encode(codes)
    decoded = entropy_decode(payload, table, 1)
    assert np.array_equal(decoded, codes)


def test_empty_array_does_not_crash():
    payload, table = entropy_encode(np.array([], dtype=np.int64))
    assert payload == b""
    assert table == {}
    decoded = entropy_decode(payload, table, 0)
    assert decoded.shape == (0,)


# ---------------------------------------------------------------------------
# realized size behaves sanely
# ---------------------------------------------------------------------------
def test_payload_nonempty_for_nontrivial_input():
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 8, size=100)
    payload, _ = entropy_encode(codes)
    assert len(payload) > 0


def test_decode_mismatched_n_raises():
    codes = np.array([1, 2, 3, 1, 2])
    payload, table = entropy_encode(codes)
    with pytest.raises(ValueError):
        entropy_decode(payload, table, n=100)
