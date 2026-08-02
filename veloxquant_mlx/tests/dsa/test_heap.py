"""Tests for MaxHeap and SortedChannelIndex."""

from __future__ import annotations

import pytest

from veloxquant_mlx.dsa.heap import MaxHeap, SortedChannelIndex


class TestMaxHeap:
    def test_push_pop_order(self) -> None:
        heap = MaxHeap()
        heap.push(3.0, 0)
        heap.push(1.0, 1)
        heap.push(5.0, 2)
        heap.push(2.0, 3)

        p0, _ = heap.pop()
        p1, _ = heap.pop()
        assert p0 >= p1

    def test_pop_gives_max(self) -> None:
        heap = MaxHeap()
        for v in [4.0, 1.0, 9.0, 2.0, 7.0]:
            heap.push(v, int(v))
        p, _ = heap.pop()
        assert p == 9.0

    def test_empty_pop_raises(self) -> None:
        heap = MaxHeap()
        with pytest.raises(IndexError):
            heap.pop()

    def test_len(self) -> None:
        heap = MaxHeap()
        assert len(heap) == 0
        heap.push(1.0, 0)
        heap.push(2.0, 1)
        assert len(heap) == 2

    def test_peek(self) -> None:
        heap = MaxHeap()
        heap.push(3.0, 0)
        heap.push(7.0, 1)
        p, _ = heap.peek()
        assert p == 7.0
        assert len(heap) == 2  # peek does not remove

    def test_single_element(self) -> None:
        heap = MaxHeap()
        heap.push(42.0, 0)
        p, v = heap.pop()
        assert p == 42.0
        assert v == 0


class TestSortedChannelIndex:
    def test_insert_and_top_k(self) -> None:
        idx = SortedChannelIndex()
        for ch, mag in enumerate([0.1, 0.5, 0.3, 0.9, 0.2]):
            idx.insert(ch, mag)
        top = idx.top_k(2)
        assert 3 in top  # channel 3 has magnitude 0.9
        assert len(top) == 2

    def test_update(self) -> None:
        idx = SortedChannelIndex()
        idx.insert(0, 0.1)
        idx.insert(1, 0.9)
        idx.update(0, 1.5)
        top = idx.top_k(1)
        assert top[0] == 0  # channel 0 is now highest

    def test_len(self) -> None:
        idx = SortedChannelIndex()
        assert len(idx) == 0
        idx.insert(0, 1.0)
        idx.insert(1, 2.0)
        assert len(idx) == 2

    def test_top_k_zero(self) -> None:
        idx = SortedChannelIndex()
        idx.insert(0, 1.0)
        assert idx.top_k(0) == []

    def test_large_index(self) -> None:
        idx = SortedChannelIndex()
        for ch in range(128):
            idx.insert(ch, float(ch))
        top4 = idx.top_k(4)
        assert set(top4) == {124, 125, 126, 127}
