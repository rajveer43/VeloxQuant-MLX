"""Tests for QuantizationGraph (DAG + topological sort)."""

from __future__ import annotations

import pytest

from veloxquant_mlx.core.exceptions import CyclicPipelineError
from veloxquant_mlx.dsa.dag import QuantizationGraph
from veloxquant_mlx.handlers.normalization import NormalizationHandler
from veloxquant_mlx.handlers.rotation_handler import RotationHandler


def _make_rotation_handler():
    import mlx.core as mx
    import numpy as np
    from veloxquant_mlx.math.rotation import make_rotation_matrix
    from veloxquant_mlx.preconditioners.rotation import RotationPreconditioner

    Pi = mx.array(make_rotation_matrix(64, seed=0).astype(np.float16))
    return RotationHandler(RotationPreconditioner(Pi))


def test_no_cycle_simple() -> None:
    g = QuantizationGraph()
    h1 = NormalizationHandler()
    h2 = _make_rotation_handler()
    g.add_node(h1)
    g.add_node(h2)
    g.add_edge(h1, h2)
    assert not g.has_cycle()


def test_cycle_detected() -> None:
    g = QuantizationGraph()
    h1 = NormalizationHandler()
    h2 = _make_rotation_handler()
    g.add_node(h1)
    g.add_node(h2)
    g.add_edge(h1, h2)
    g.add_edge(h2, h1)
    assert g.has_cycle()


def test_topological_sort_raises_on_cycle() -> None:
    g = QuantizationGraph()
    h1 = NormalizationHandler()
    h2 = _make_rotation_handler()
    g.add_node(h1)
    g.add_node(h2)
    g.add_edge(h1, h2)
    g.add_edge(h2, h1)
    with pytest.raises(CyclicPipelineError):
        g.topological_sort()


def test_topological_sort_linear_chain() -> None:
    g = QuantizationGraph()
    handlers = [NormalizationHandler() for _ in range(4)]
    for h in handlers:
        g.add_node(h)
    for i in range(3):
        g.add_edge(handlers[i], handlers[i + 1])
    order = g.topological_sort()
    # h0 must come before h1, h1 before h2, h2 before h3
    positions = {id(h): pos for pos, h in enumerate(order)}
    for i in range(3):
        assert positions[id(handlers[i])] < positions[id(handlers[i + 1])]


def test_empty_graph() -> None:
    g = QuantizationGraph()
    assert not g.has_cycle()
    assert g.topological_sort() == []


def test_single_node() -> None:
    g = QuantizationGraph()
    h = NormalizationHandler()
    g.add_node(h)
    order = g.topological_sort()
    assert len(order) == 1


def test_critical_path() -> None:
    g = QuantizationGraph()
    handlers = [NormalizationHandler() for _ in range(5)]
    for h in handlers:
        g.add_node(h)
    # Create a diamond: 0 -> 1 -> 3, 0 -> 2 -> 3, 3 -> 4
    g.add_edge(handlers[0], handlers[1])
    g.add_edge(handlers[0], handlers[2])
    g.add_edge(handlers[1], handlers[3])
    g.add_edge(handlers[2], handlers[3])
    g.add_edge(handlers[3], handlers[4])
    cp = g.critical_path()
    assert len(cp) >= 3  # at least 0 -> {1 or 2} -> 3 -> 4
    assert handlers[0] in cp
    assert handlers[4] in cp


def test_add_node_not_in_graph_raises() -> None:
    g = QuantizationGraph()
    h1 = NormalizationHandler()
    h2 = NormalizationHandler()
    g.add_node(h1)
    with pytest.raises(KeyError):
        g.add_edge(h1, h2)  # h2 not added
