"""Tests for AVLTree and VoronoiTree."""

from __future__ import annotations

import numpy as np
import pytest

from veloxquant_mlx.dsa.avl_tree import AVLTree, VoronoiTree


def check_balance_invariant(tree: AVLTree) -> None:
    """Assert |balance_factor| <= 1 for every node."""

    def _check(node):
        if node is None:
            return
        bf = tree._balance_factor(node)
        assert abs(bf) <= 1, f"Balance factor {bf} at node {node.key:.4f} violates AVL invariant"
        _check(node.left)
        _check(node.right)

    _check(tree.root)


def test_balance_after_sorted_inserts() -> None:
    tree = AVLTree()
    for i in range(100):
        tree.insert(float(i), i)
    check_balance_invariant(tree)


def test_balance_after_reverse_inserts() -> None:
    tree = AVLTree()
    for i in range(100, 0, -1):
        tree.insert(float(i), i)
    check_balance_invariant(tree)


def test_balance_after_random_inserts() -> None:
    rng = np.random.default_rng(0)
    tree = AVLTree()
    keys = rng.uniform(-10, 10, 200)
    for i, k in enumerate(keys):
        tree.insert(float(k), i)
    check_balance_invariant(tree)


def test_len_after_inserts() -> None:
    tree = AVLTree()
    for i in range(50):
        tree.insert(float(i), i)
    assert len(tree) == 50


def test_duplicate_key_update() -> None:
    tree = AVLTree()
    tree.insert(1.0, 10)
    tree.insert(1.0, 20)  # same key, updates value
    _, v = tree.search_nearest(1.0)
    assert v == 20
    assert len(tree) == 1  # no new node


def test_nearest_exact() -> None:
    tree = AVLTree()
    for i in range(10):
        tree.insert(float(i), i)
    key, value = tree.search_nearest(5.0)
    assert value == 5


def test_nearest_approximate() -> None:
    tree = AVLTree()
    centroids = [0.0, 1.0, 2.0, 5.0, 10.0]
    for i, c in enumerate(centroids):
        tree.insert(c, i)
    _, idx = tree.search_nearest(4.2)
    assert idx == 3  # 5.0 is closest


def test_empty_tree_search_raises() -> None:
    tree = AVLTree()
    with pytest.raises(RuntimeError):
        tree.search_nearest(0.0)


def test_range_query() -> None:
    tree = AVLTree()
    for i in range(20):
        tree.insert(float(i), i)
    result = tree.range_query(5.0, 10.0)
    assert set(result) == {5, 6, 7, 8, 9, 10}


class TestVoronoiTree:
    def test_build_and_nearest_small(self) -> None:
        vt = VoronoiTree()
        centroids = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        vt.build(centroids)
        assert vt.nearest(0.4) == 0
        assert vt.nearest(0.6) == 1
        assert vt.nearest(2.9) == 3

    def test_build_and_nearest_large(self) -> None:
        vt = VoronoiTree()
        centroids = np.linspace(-5, 5, 32).astype(np.float32)
        vt.build(centroids)
        # Should use AVL tree (k=32 > LINEAR_THRESHOLD=16)
        for q in [-4.9, 0.0, 4.9]:
            idx = vt.nearest(q)
            assert abs(centroids[idx] - q) <= (centroids[1] - centroids[0])

    def test_not_built_raises(self) -> None:
        vt = VoronoiTree()
        with pytest.raises(RuntimeError):
            vt.nearest(0.0)
