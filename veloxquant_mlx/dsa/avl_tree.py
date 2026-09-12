from __future__ import annotations

import numpy as np

from veloxquant_mlx.core.constants import VORONOI_LINEAR_THRESHOLD


class AVLNode:
    """Single node of an AVL tree.

    Attributes:
        key: Float key (centroid value).
        value: Integer payload (centroid index in the codebook).
        left: Left child.
        right: Right child.
        height: Cached subtree height (1 for a leaf).
    """

    __slots__ = ("key", "value", "left", "right", "height")

    def __init__(self, key: float, value: int) -> None:
        self.key: float = key
        self.value: int = value
        self.left: AVLNode | None = None
        self.right: AVLNode | None = None
        self.height: int = 1

    def __repr__(self) -> str:
        return f"AVLNode(key={self.key:.4f}, value={self.value}, h={self.height})"


class AVLTree:
    """Self-balancing AVL binary search tree.

    Keys are float centroid values; values are integer centroid indices.
    All rotations are implemented manually to satisfy the balance invariant:
    |balance_factor(node)| <= 1 for every node.

    Args:
        None. Tree starts empty.
    """

    def __init__(self) -> None:
        self.root: AVLNode | None = None
        self._size: int = 0

    # ------------------------------------------------------------------
    # Height and balance
    # ------------------------------------------------------------------

    def _height(self, node: AVLNode | None) -> int:
        return 0 if node is None else node.height

    def _balance_factor(self, node: AVLNode | None) -> int:
        if node is None:
            return 0
        return self._height(node.left) - self._height(node.right)

    def _update_height(self, node: AVLNode) -> None:
        node.height = 1 + max(self._height(node.left), self._height(node.right))

    # ------------------------------------------------------------------
    # Rotations
    # ------------------------------------------------------------------

    def _rotate_right(self, y: AVLNode) -> AVLNode:
        """Right rotation around node y.

        Before:        After:
             y              x
            / \\            / \\
           x   T3         T1   y
          / \\                 / \\
         T1  T2              T2  T3
        """
        x = y.left
        assert x is not None
        T2 = x.right

        x.right = y
        y.left = T2

        self._update_height(y)
        self._update_height(x)
        return x

    def _rotate_left(self, x: AVLNode) -> AVLNode:
        """Left rotation around node x.

        Before:        After:
           x                y
          / \\             / \\
         T1   y           x   T3
             / \\         / \\
            T2  T3       T1  T2
        """
        y = x.right
        assert y is not None
        T2 = y.left

        y.left = x
        x.right = T2

        self._update_height(x)
        self._update_height(y)
        return y

    def _rebalance(self, node: AVLNode) -> AVLNode:
        """Rebalance a node if its balance factor is out of range [-1, 1]."""
        self._update_height(node)
        bf = self._balance_factor(node)

        # Left heavy
        if bf > 1:
            assert node.left is not None
            if self._balance_factor(node.left) < 0:
                # Left-Right case
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)

        # Right heavy
        if bf < -1:
            assert node.right is not None
            if self._balance_factor(node.right) > 0:
                # Right-Left case
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, key: float, value: int) -> None:
        """Insert a (key, value) pair into the tree.

        Args:
            key: Float key.
            value: Integer index payload.
        """
        self.root = self._insert(self.root, key, value)
        self._size += 1

    def _insert(self, node: AVLNode | None, key: float, value: int) -> AVLNode:
        if node is None:
            return AVLNode(key, value)
        if key < node.key:
            node.left = self._insert(node.left, key, value)
        elif key > node.key:
            node.right = self._insert(node.right, key, value)
        else:
            # Duplicate key: update value in place
            node.value = value
            self._size -= 1  # undo the +1 from insert()
            return node
        return self._rebalance(node)

    # ------------------------------------------------------------------
    # Nearest-centroid search
    # ------------------------------------------------------------------

    def search_nearest(self, query: float) -> tuple[float, int]:
        """Find the (key, value) pair with key closest to query.

        Args:
            query: Target value.

        Returns:
            Tuple (nearest_key, nearest_value).

        Raises:
            RuntimeError: If the tree is empty.
        """
        if self.root is None:
            raise RuntimeError("AVLTree is empty — cannot search")
        best_key, best_val = self.root.key, self.root.value
        node = self.root
        while node is not None:
            if abs(node.key - query) < abs(best_key - query):
                best_key, best_val = node.key, node.value
            if query < node.key:
                node = node.left
            elif query > node.key:
                node = node.right
            else:
                return node.key, node.value
        return best_key, best_val

    # ------------------------------------------------------------------
    # Range query
    # ------------------------------------------------------------------

    def range_query(self, lo: float, hi: float) -> list[int]:
        """Return all values whose keys fall in [lo, hi].

        Args:
            lo: Inclusive lower bound.
            hi: Inclusive upper bound.

        Returns:
            List of integer values with keys in [lo, hi].
        """
        result: list[int] = []
        self._range_query(self.root, lo, hi, result)
        return result

    def _range_query(
        self,
        node: AVLNode | None,
        lo: float,
        hi: float,
        result: list[int],
    ) -> None:
        if node is None:
            return
        if lo <= node.key <= hi:
            result.append(node.value)
        if lo < node.key:
            self._range_query(node.left, lo, hi, result)
        if hi > node.key:
            self._range_query(node.right, lo, hi, result)

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"AVLTree(size={self._size}, root_key={self.root.key if self.root else None})"


class VoronoiTree:
    """Nearest-centroid search structure wrapping AVLTree.

    Falls back to linear scan (``numpy.argmin``) for small codebooks
    (``n_centroids <= LINEAR_THRESHOLD``).

    Args:
        None. Call build() before calling nearest().
    """

    LINEAR_THRESHOLD: int = VORONOI_LINEAR_THRESHOLD

    def __init__(self) -> None:
        self._tree: AVLTree = AVLTree()
        self._centroids: np.ndarray | None = None
        self._use_linear: bool = True

    def build(self, centroids: np.ndarray) -> None:
        """Populate the search structure from a 1-D centroid array.

        Args:
            centroids: 1-D float array of centroid values, shape (k,).
        """
        centroids = np.asarray(centroids, dtype=np.float64)
        self._centroids = centroids
        k = len(centroids)
        self._use_linear = k <= self.LINEAR_THRESHOLD

        if not self._use_linear:
            self._tree = AVLTree()
            for idx, c in enumerate(centroids):
                self._tree.insert(float(c), idx)

    def nearest(self, value: float) -> int:
        """Return the index of the centroid nearest to value.

        Args:
            value: Query float.

        Returns:
            Index of the nearest centroid.

        Raises:
            RuntimeError: If build() has not been called.
        """
        if self._centroids is None:
            raise RuntimeError("VoronoiTree: call build() before nearest()")

        if self._use_linear:
            return int(np.argmin(np.abs(self._centroids - value)))

        _, idx = self._tree.search_nearest(value)
        return idx

    def __repr__(self) -> str:
        k = len(self._centroids) if self._centroids is not None else 0
        mode = "linear" if self._use_linear else "avl"
        return f"VoronoiTree(k={k}, mode={mode!r})"
