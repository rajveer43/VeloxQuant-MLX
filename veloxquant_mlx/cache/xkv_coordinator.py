"""xKV cross-layer coordinator — shared-basis statistics for a group of layers.

The repo's contract is one cache object per layer; ``mlx_lm.generate`` iterates
them independently. xKV needs a *group* of layers to share one SVD basis, so
the group's members pool their keys here and read the joint basis back. Unlike:

  * ``XQuantCoordinator`` — a single anchor publishes; reusers read (no
    joint computation).
  * ``MiniCacheCoordinator`` — one primary publishes; exactly one merge
    partner consumes.

All ``XKVCache`` instances of a model hold a reference to one shared
``XKVCoordinator`` (injected at ``KVCacheBuilder.for_model`` build time).

Per group the coordinator keeps running Gram statistics
(:class:`~veloxquant_mlx.quantizers.xkv.JointGramStats`) of every key row any
member has added. Each member adds its prefill chunk and reads the basis fit
on everything pooled so far, so the basis is refit on every chunk instead of
being frozen from the first one (#892). A member iterated earlier in a forward
pass has not yet seen its peers' keys for the current chunk, so its basis
lags theirs by one chunk. ``version`` lets a member tell, on a later call,
that the pooled statistics have moved on.

Single-threaded by construction (mlx generate is sequential), so no locking.
"""

from __future__ import annotations

import mlx.core as mx

from veloxquant_mlx.quantizers.xkv import JointGramStats


class XKVCoordinator:
    """Shared cross-layer basis store for one model instance / one generation.

    Args:
        max_ctx: Largest single prefill call (rows) a group member may add.
            The pooled statistics are O(D^2), so this no longer bounds memory;
            it still rejects a single oversized call.

    Note: the shared rank / energy-threshold are NOT stored on the
    coordinator — every member of a group is built from the same
    ``KVCacheConfig`` (see ``KVCacheBuilder._build_xkv``), so the calling
    ``XKVCache`` passes its own ``xkv_rank``/``xkv_energy_threshold`` on every
    call. This avoids a second place where those two values could silently
    drift out of sync with the config.
    """

    def __init__(self, max_ctx: int = 8192) -> None:
        self._max_ctx = int(max_ctx)
        self._stats: dict[int, JointGramStats] = {}
        self._published_tokens: dict[int, int] = {}

    def reset(self) -> None:
        """Clear all pooled statistics and published-token counts (start of a fresh generation)."""
        self._stats.clear()
        self._published_tokens.clear()

    def add_member_keys(self, group_id: int, member_idx: int, keys: mx.array) -> None:
        """A group member adds its own raw keys ``[S, D]`` for this prefill call."""
        n_tokens = int(keys.shape[0])
        if member_idx == 0:
            if n_tokens > self._max_ctx:
                raise RuntimeError(
                    f"XKVCoordinator: group {group_id} call of {n_tokens} rows exceeds "
                    f"max_ctx={self._max_ctx}. Increase xkv_max_ctx."
                )
            self._published_tokens[group_id] = self._published_tokens.get(group_id, 0) + n_tokens
        self._stats.setdefault(group_id, JointGramStats()).add(keys)

    def shared_basis(
        self,
        group_id: int,
        rank: int | None = None,
        energy_threshold: float = 0.95,
    ) -> tuple[tuple[mx.array, mx.array, mx.array], int] | None:
        """The basis fit on every row pooled for ``group_id`` so far, with its version; ``None`` if nothing was added."""
        stats = self._stats.get(group_id)
        if stats is None or stats.n_rows == 0:
            return None
        basis = stats.basis(rank=rank, energy_threshold=energy_threshold)
        mx.eval(*basis)
        return basis, stats.version

    def version(self, group_id: int) -> int:
        """Number of adds so far for ``group_id`` (0 if none)."""
        stats = self._stats.get(group_id)
        return 0 if stats is None else stats.version

    def published_tokens(self, group_id: int) -> int:
        """Total tokens the leader member of ``group_id`` has added."""
        return self._published_tokens.get(group_id, 0)

    @property
    def max_ctx(self) -> int:
        """Configured per-call row limit."""
        return self._max_ctx


__all__ = ["XKVCoordinator"]
