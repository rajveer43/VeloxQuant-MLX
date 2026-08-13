from __future__ import annotations

from typing import List, Optional, Tuple


class MaxHeap:
    """Binary max-heap backed by an internal array.

    Nodes are (priority, value) pairs. The heap property is
    ``priority[parent] >= priority[child]`` for all nodes.

    All heap operations are implemented manually without the ``heapq`` module.

    Args:
        None. The heap starts empty.
    """

    def __init__(self) -> None:
        self._data: List[Tuple[float, int]] = []  # (priority, value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parent(i: int) -> int:
        return (i - 1) >> 1

    @staticmethod
    def _left(i: int) -> int:
        return 2 * i + 1

    @staticmethod
    def _right(i: int) -> int:
        return 2 * i + 2

    def _sift_up(self, i: int) -> None:
        """Restore heap property upward from index i."""
        while i > 0:
            p = self._parent(i)
            if self._data[i][0] > self._data[p][0]:
                self._data[i], self._data[p] = self._data[p], self._data[i]
                i = p
            else:
                break

    def _sift_down(self, i: int) -> None:
        """Restore heap property downward from index i."""
        n = len(self._data)
        while True:
            largest = i
            l = self._left(i)
            r = self._right(i)
            if l < n and self._data[l][0] > self._data[largest][0]:
                largest = l
            if r < n and self._data[r][0] > self._data[largest][0]:
                largest = r
            if largest == i:
                break
            self._data[i], self._data[largest] = self._data[largest], self._data[i]
            i = largest

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def push(self, priority: float, value: int) -> None:
        """Insert a (priority, value) pair.

        Args:
            priority: Comparison key (higher = higher priority).
            value: Associated integer payload.
        """
        self._data.append((priority, value))
        self._sift_up(len(self._data) - 1)

    def pop(self) -> Tuple[float, int]:
        """Remove and return the maximum (priority, value) pair.

        Returns:
            Tuple of (priority, value).

        Raises:
            IndexError: If the heap is empty.
        """
        if not self._data:
            raise IndexError("pop from empty MaxHeap")
        top = self._data[0]
        last = self._data.pop()
        if self._data:
            self._data[0] = last
            self._sift_down(0)
        return top

    def peek(self) -> Tuple[float, int]:
        """Return but do not remove the maximum pair.

        Raises:
            IndexError: If the heap is empty.
        """
        if not self._data:
            raise IndexError("peek on empty MaxHeap")
        return self._data[0]

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"MaxHeap(size={len(self._data)})"


class SortedChannelIndex:
    """Max-heap–based structure for tracking top-k high-magnitude channels.

    Used by OutlierDetector to maintain a running top-k across streaming
    key vectors during the prefill phase.

    Each entry maps a channel index to a magnitude estimate. The structure
    supports O(log n) insert, O(k log n) top-k query, and O(log n) update.

    Args:
        None. The index starts empty.
    """

    def __init__(self) -> None:
        # Heap entries are (magnitude, (channel_idx, version)) so that
        # re-insertions with an unchanged magnitude can still be told apart
        # from the live entry by version rather than by value equality.
        self._heap: MaxHeap = MaxHeap()
        # channel_idx -> (latest magnitude, latest version) — used to
        # recognize which heap entry for a channel is still live (lazy
        # deletion) and to skip redundant re-inserts of unchanged values.
        self._valid: dict[int, tuple[float, int]] = {}
        self._next_version: int = 0

    def insert(self, channel_idx: int, magnitude: float) -> None:
        """Insert or update a channel's magnitude.

        If the channel already exists, the magnitude is updated (old entry
        is lazily invalidated). Re-inserting the same magnitude for a
        channel that already holds it is a no-op — this keeps the heap
        bounded at O(number of distinct channels observed) instead of
        growing by one entry per call.

        Args:
            channel_idx: Index of the channel (0-indexed).
            magnitude: Non-negative magnitude value.
        """
        current = self._valid.get(channel_idx)
        if current is not None and current[0] == magnitude:
            return
        version = self._next_version
        self._next_version += 1
        self._valid[channel_idx] = (magnitude, version)
        self._heap.push(magnitude, (channel_idx, version))

    def update(self, channel_idx: int, new_magnitude: float) -> None:
        """Update the magnitude of an existing channel.

        Internally uses lazy deletion — the old entry remains in the heap
        but is skipped when popped because the stored magnitude no longer
        matches ``_valid``.

        Args:
            channel_idx: Channel whose magnitude should be updated.
            new_magnitude: Replacement magnitude value.
        """
        self.insert(channel_idx, new_magnitude)

    def top_k(self, k: int) -> List[int]:
        """Return indices of the k channels with the highest magnitudes.

        This operation does *not* remove elements from the heap.

        Args:
            k: Number of top channels to return.

        Returns:
            List of up to k channel indices, ordered from highest to lowest magnitude.
        """
        if k <= 0:
            return []
        result: List[int] = []
        seen: set[int] = set()

        # Copy the heap data to a temporary structure to avoid mutation
        heap_copy = MaxHeap()
        heap_copy._data = list(self._heap._data)

        while len(heap_copy) > 0 and len(result) < k:
            mag, (ch, version) = heap_copy.pop()
            # Lazy deletion: skip if this exact (magnitude, version) is no
            # longer the live entry for the channel. Comparing by version
            # (not just magnitude) means a re-insert with an unchanged
            # value can never masquerade as a second, distinct live entry.
            if ch in seen:
                continue
            if self._valid.get(ch) == (mag, version):
                seen.add(ch)
                result.append(ch)

        return result

    def __len__(self) -> int:
        """Return the number of unique channels tracked."""
        return len(self._valid)

    def __repr__(self) -> str:
        return f"SortedChannelIndex(n_channels={len(self._valid)}, heap_size={len(self._heap)})"
