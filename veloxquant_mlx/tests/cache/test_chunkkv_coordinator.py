"""Tests for ChunkKVIndexReuseCoordinator — layer-wise kept-index reuse bookkeeping."""

from __future__ import annotations

from veloxquant_mlx.cache.chunkkv_coordinator import ChunkKVIndexReuseCoordinator


def test_leader_for_groups_contiguous_blocks():
    coord = ChunkKVIndexReuseCoordinator(n_layers=6, reuse_layers=3)
    assert [coord.leader_for(i) for i in range(6)] == [0, 0, 0, 3, 3, 3]


def test_is_leader_matches_leader_for():
    coord = ChunkKVIndexReuseCoordinator(n_layers=4, reuse_layers=2)
    assert [coord.is_leader(i) for i in range(4)] == [True, False, True, False]


def test_reuse_layers_of_one_is_all_leaders():
    coord = ChunkKVIndexReuseCoordinator(n_layers=5, reuse_layers=1)
    assert all(coord.is_leader(i) for i in range(5))


def test_reuse_layers_clamped_to_at_least_one():
    coord = ChunkKVIndexReuseCoordinator(n_layers=5, reuse_layers=0)
    assert coord.reuse_layers == 1


def test_publish_and_fetch_roundtrip_per_head():
    coord = ChunkKVIndexReuseCoordinator(n_layers=4, reuse_layers=2)
    coord.publish(layer_id=0, head_idx=0, kept_positions=[[0, 1], [0, 1, 2]])
    coord.publish(layer_id=0, head_idx=1, kept_positions=[[0], [0, 1]])

    assert coord.fetch(layer_id=1, head_idx=0) == [[0, 1], [0, 1, 2]]
    assert coord.fetch(layer_id=1, head_idx=1) == [[0], [0, 1]]


def test_fetch_before_publish_returns_none():
    coord = ChunkKVIndexReuseCoordinator(n_layers=2, reuse_layers=2)
    assert coord.fetch(layer_id=1, head_idx=0) is None


def test_fetch_uses_own_leader_only():
    coord = ChunkKVIndexReuseCoordinator(n_layers=4, reuse_layers=2)
    coord.publish(layer_id=0, head_idx=0, kept_positions=[[0]])
    coord.publish(layer_id=2, head_idx=0, kept_positions=[[0, 1]])
    assert coord.fetch(layer_id=1, head_idx=0) == [[0]]
    assert coord.fetch(layer_id=3, head_idx=0) == [[0, 1]]
