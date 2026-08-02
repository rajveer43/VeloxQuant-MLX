"""ChunkKV cross-layer coordinator — layer-wise kept-index reuse (Algorithm 2).

The paper observes (Table 2) that ChunkKV's kept-chunk indices are far more
similar between adjacent layers than token-level methods' (44-58% Jaccard
similarity vs 15-28% for H2O/SnapKV on the models tested). Layer-wise index reuse
exploits this: layers are grouped into consecutive blocks of ``Nreuse``; only the
first ("leader") layer in each block runs full eviction, and the remaining
("follower") layers reuse the leader's exact kept-token positions instead of
scoring and evicting on their own. The paper reports up to 20.7% latency
reduction and 26.5% throughput improvement from this, at <0.6% LongBench /
neutral-to-positive GSM8K performance cost.

Unlike ``SqueezeCoordinator`` (a one-shot prefill-boundary re-budget), this
coordinator exchanges the leader's kept-index list **every step** — prefill and
decode alike — since eviction (and therefore the reusable index set) can change
at any step once the cache is over budget. Single-threaded by construction
(``mlx_lm.generate`` is sequential), so plain dicts need no locking.
"""

from __future__ import annotations

from typing import Optional


class ChunkKVIndexReuseCoordinator:
    """Shared per-model layer-wise index-reuse state for one generation.

    Args:
        n_layers:     Number of attention-bearing layers that will participate.
        reuse_layers: Block size ``Nreuse``. Layers ``[0, Nreuse)`` share leader
            layer 0's indices, ``[Nreuse, 2*Nreuse)`` share leader layer
            ``Nreuse``, and so on. ``1`` disables reuse (every layer is its own
            leader).
    """

    def __init__(self, n_layers: int, reuse_layers: int) -> None:
        self._n_layers = int(n_layers)
        self._reuse_layers = max(1, int(reuse_layers))
        # (leader_layer_id, head_idx) -> most recently published per-step kept_positions
        self._published: dict[tuple[int, int], list[list[int]]] = {}

    def leader_for(self, layer_id: int) -> int:
        """Return the leader layer id for ``layer_id``'s reuse block."""
        return (layer_id // self._reuse_layers) * self._reuse_layers

    def is_leader(self, layer_id: int) -> bool:
        """True if ``layer_id`` is the leader of its reuse block."""
        return layer_id == self.leader_for(layer_id)

    def publish(self, layer_id: int, head_idx: int, kept_positions: list[list[int]]) -> None:
        """Leader layer publishes one head's this-step per-token kept-index lists."""
        self._published[(layer_id, head_idx)] = kept_positions

    def fetch(self, layer_id: int, head_idx: int) -> Optional[list[list[int]]]:
        """Follower layer fetches its leader's most recently published indices.

        Returns:
            The leader's kept-index lists for ``head_idx``, or ``None`` if the
            leader hasn't published for this step yet (should not happen under
            the standard ``for_model`` build, where layers are updated
            leader-first within a block by construction order).
        """
        return self._published.get((self.leader_for(layer_id), head_idx))

    @property
    def reuse_layers(self) -> int:
        return self._reuse_layers

    @property
    def n_layers(self) -> int:
        return self._n_layers


__all__ = ["ChunkKVIndexReuseCoordinator"]
