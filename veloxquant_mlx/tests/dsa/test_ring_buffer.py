"""Tests for RingBuffer."""

from __future__ import annotations

import pytest

from veloxquant_mlx.dsa.ring_buffer import RingBuffer


def test_basic_append_and_len() -> None:
    buf: RingBuffer[int] = RingBuffer(5)
    assert len(buf) == 0
    buf.append(1)
    buf.append(2)
    assert len(buf) == 2


def test_wraparound() -> None:
    buf: RingBuffer[int] = RingBuffer(3)
    for i in range(5):
        buf.append(i)
    assert len(buf) == 3
    assert list(buf) == [2, 3, 4]
    assert buf[0] == 2
    assert buf[-1] == 4


def test_evicted_item_returned() -> None:
    buf: RingBuffer[int] = RingBuffer(3)
    buf.append(0)
    buf.append(1)
    buf.append(2)
    evicted = buf.append(3)
    assert evicted == 0


def test_negative_indexing() -> None:
    buf: RingBuffer[int] = RingBuffer(5)
    for i in range(4):
        buf.append(i)
    assert buf[-1] == 3
    assert buf[-2] == 2
    assert buf[-4] == 0


def test_empty_raises() -> None:
    buf: RingBuffer[int] = RingBuffer(3)
    with pytest.raises(IndexError):
        _ = buf[0]


def test_is_full() -> None:
    buf: RingBuffer[int] = RingBuffer(2)
    assert not buf.is_full()
    buf.append(1)
    buf.append(2)
    assert buf.is_full()


def test_to_list() -> None:
    buf: RingBuffer[int] = RingBuffer(4)
    buf.append(10)
    buf.append(20)
    assert buf.to_list() == [10, 20]


def test_to_stacked_mlx() -> None:
    import mlx.core as mx
    import numpy as np

    buf: RingBuffer = RingBuffer(4)
    for i in range(3):
        buf.append(mx.array([float(i), float(i + 1)], dtype=mx.float16))
    stacked = buf.to_stacked()
    assert stacked.shape == (3, 2)
    np.testing.assert_allclose(np.array(stacked), [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], atol=1e-3)


def test_capacity_one() -> None:
    buf: RingBuffer[int] = RingBuffer(1)
    buf.append(99)
    evicted = buf.append(100)
    assert evicted == 99
    assert list(buf) == [100]


def test_invalid_capacity() -> None:
    with pytest.raises(ValueError):
        RingBuffer(0)


def test_iterator() -> None:
    buf: RingBuffer[int] = RingBuffer(5)
    for i in range(5):
        buf.append(i)
    assert list(buf) == list(range(5))
