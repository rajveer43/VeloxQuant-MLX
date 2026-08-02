from __future__ import annotations

from typing import Generic, Iterator, List, Optional, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """Fixed-capacity circular buffer with O(1) append and index access.

    When full, the oldest element is silently overwritten (FIFO eviction).
    Supports negative indexing (``buf[-1]`` is the most recently appended item).

    Internal layout:
        ``_data``: list of length ``_capacity``, pre-filled with ``None``.
        ``_head``: index in ``_data`` of the *oldest* element.
        ``_size``: number of currently live elements.

    Args:
        capacity: Maximum number of elements the buffer can hold. Must be >= 1.

    Raises:
        ValueError: If capacity < 1.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"RingBuffer capacity must be >= 1, got {capacity}")
        self._capacity: int = capacity
        self._data: List[Optional[T]] = [None] * capacity
        self._head: int = 0  # index of oldest element
        self._size: int = 0  # current number of elements

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, item: T) -> Optional[T]:
        """Append an item to the buffer.

        If the buffer is full the oldest item is evicted and returned.

        Args:
            item: Element to append.

        Returns:
            The evicted element if the buffer was full, otherwise ``None``.
        """
        evicted: Optional[T] = None
        if self._size == self._capacity:
            # Overwrite oldest slot; advance head
            evicted = self._data[self._head]
            self._data[self._head] = item
            self._head = (self._head + 1) % self._capacity
        else:
            write_pos = (self._head + self._size) % self._capacity
            self._data[write_pos] = item
            self._size += 1
        return evicted

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> T:
        """Return the element at logical position idx (0 = oldest).

        Supports negative indexing: -1 is the most recently appended.

        Args:
            idx: Logical index in [-size, size).

        Returns:
            The element at the given position.

        Raises:
            IndexError: If idx is out of range.
        """
        if self._size == 0:
            raise IndexError("RingBuffer is empty")
        if idx < 0:
            idx = self._size + idx
        if idx < 0 or idx >= self._size:
            raise IndexError(f"RingBuffer index {idx} out of range for size {self._size}")
        physical = (self._head + idx) % self._capacity
        return self._data[physical]  # type: ignore[return-value]

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[T]:
        """Iterate from oldest to newest element."""
        for i in range(self._size):
            yield self[i]

    # ------------------------------------------------------------------
    # Predicates / helpers
    # ------------------------------------------------------------------

    def is_full(self) -> bool:
        """Return True if the buffer is at capacity."""
        return self._size == self._capacity

    def to_list(self) -> List[T]:
        """Return all elements as a plain list (oldest first).

        Returns:
            List of length len(self).
        """
        return [self[i] for i in range(self._size)]

    def to_stacked(self) -> "import mlx.core as mx; mx.array":  # type: ignore[return]
        """Stack all stored MLX arrays along axis 0.

        Only valid when ``T = mx.array``. Uses ``mx.stack`` for efficiency.

        Returns:
            Single MLX array of shape (size, *element_shape).

        Raises:
            RuntimeError: If the buffer is empty.
        """
        import mlx.core as mx  # lazy import — keeps math/ MLX-free

        if self._size == 0:
            raise RuntimeError("Cannot stack an empty RingBuffer")
        elements = [self[i] for i in range(self._size)]
        return mx.stack(elements, axis=0)

    def __repr__(self) -> str:
        return f"RingBuffer(capacity={self._capacity}, size={self._size}, head={self._head})"
