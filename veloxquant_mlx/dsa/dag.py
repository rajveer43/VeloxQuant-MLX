"""Directed acyclic graph utilities for validating quantization handler pipelines.

Implements ``QuantizationGraph``, a DAG over ``QuantizationHandler`` nodes
(identified by ``id()``) supporting cycle detection and topological ordering
via Kahn's algorithm, plus a longest-path (critical path) query. Used to
verify that user-assembled handler chains are acyclic and to derive a valid
execution order before running them.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from veloxquant_mlx.core.exceptions import CyclicPipelineError


class QuantizationGraph:
    """Directed Acyclic Graph of QuantizationHandler nodes.

    Used to validate user-assembled pipelines (detect cycles) and to
    determine a correct execution order via topological sort (Kahn's algorithm).

    Nodes are QuantizationHandler instances identified by their ``id()``.
    Edges represent data-flow dependencies (src must execute before dst).

    Args:
        None. Graph starts empty.
    """

    def __init__(self) -> None:
        # handler id -> handler object
        self._nodes: dict[int, Any] = {}
        # adjacency: src_id -> set of dst_ids
        self._adj: dict[int, set[int]] = defaultdict(set)
        # in-degree count for Kahn's algorithm
        self._in_degree: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(self, handler: Any) -> None:
        """Register a handler node.

        Args:
            handler: A QuantizationHandler instance.
        """
        nid = id(handler)
        if nid not in self._nodes:
            self._nodes[nid] = handler
            self._in_degree[nid] = 0

    def add_edge(self, src: Any, dst: Any) -> None:
        """Add a directed edge src → dst.

        Both nodes must have been added with add_node() first.

        Args:
            src: Source handler (must execute before dst).
            dst: Destination handler.

        Raises:
            KeyError: If either node is not registered.
        """
        sid, did = id(src), id(dst)
        if sid not in self._nodes:
            raise KeyError(f"Source node not in graph: {src!r}")
        if did not in self._nodes:
            raise KeyError(f"Destination node not in graph: {dst!r}")
        if did not in self._adj[sid]:
            self._adj[sid].add(did)
            self._in_degree[did] += 1

    # ------------------------------------------------------------------
    # Cycle detection and topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    def has_cycle(self) -> bool:
        """Return True if the graph contains at least one directed cycle.

        Uses Kahn's algorithm: if not all nodes can be processed, a cycle exists.

        Returns:
            True if a cycle is present, False if the graph is a DAG.
        """
        try:
            self.topological_sort()
            return False
        except CyclicPipelineError:
            return True

    def topological_sort(self) -> list[Any]:
        """Return nodes in topological order using Kahn's algorithm.

        Kahn's algorithm:
            1. Compute in-degree for each node.
            2. Enqueue all nodes with in-degree 0.
            3. Repeatedly dequeue a node, append to result, and decrement
               in-degrees of its successors. Enqueue successors that reach 0.
            4. If |result| < |nodes|, a cycle exists.

        Returns:
            List of handler objects in valid execution order.

        Raises:
            CyclicPipelineError: If the graph contains a cycle.
        """
        # Work on a copy of in-degrees
        in_deg = dict(self._in_degree)
        queue: deque[int] = deque()

        for nid, deg in in_deg.items():
            if deg == 0:
                queue.append(nid)

        result: list[Any] = []

        while queue:
            nid = queue.popleft()
            result.append(self._nodes[nid])
            for neighbor_id in sorted(self._adj[nid]):  # sorted for determinism
                in_deg[neighbor_id] -= 1
                if in_deg[neighbor_id] == 0:
                    queue.append(neighbor_id)

        if len(result) != len(self._nodes):
            raise CyclicPipelineError(
                "QuantizationGraph contains a cycle — cannot produce a valid "
                "topological order. Check your set_next() / add_edge() calls."
            )
        return result

    # ------------------------------------------------------------------
    # Critical path (longest path in DAG, assuming unit edge weight)
    # ------------------------------------------------------------------

    def critical_path(self) -> list[Any]:
        """Return the longest path through the DAG (by number of nodes).

        Uses the topological order computed by Kahn's algorithm and a
        dynamic programming pass: dist[v] = max(dist[u] + 1) for all u→v.

        Returns:
            List of handler objects forming the critical (longest) path.

        Raises:
            CyclicPipelineError: If the graph contains a cycle.
        """
        order = self.topological_sort()
        order_ids = [id(h) for h in order]
        dist: dict[int, int] = dict.fromkeys(order_ids, 0)
        prev: dict[int, int | None] = dict.fromkeys(order_ids)

        for nid in order_ids:
            for neighbor_id in self._adj[nid]:
                if dist[nid] + 1 > dist[neighbor_id]:
                    dist[neighbor_id] = dist[nid] + 1
                    prev[neighbor_id] = nid

        # Find the end of the longest path
        end_id = max(order_ids, key=lambda nid: dist[nid])

        # Reconstruct path backward
        path_ids: list[int] = []
        cur: int | None = end_id
        while cur is not None:
            path_ids.append(cur)
            cur = prev[cur]
        path_ids.reverse()

        return [self._nodes[nid] for nid in path_ids]

    def __repr__(self) -> str:
        return (
            f"QuantizationGraph(nodes={len(self._nodes)}, "
            f"edges={sum(len(v) for v in self._adj.values())})"
        )
