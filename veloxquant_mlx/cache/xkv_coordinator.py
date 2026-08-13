"""xKV cross-layer coordinator — fan-in/fan-out shared-basis broadcast.

The repo's contract is one cache object per layer; ``mlx_lm.generate`` iterates
them independently. xKV needs a *group* of layers to jointly compute one
shared SVD basis before any of them can compress — this is a fan-in (collect
every member's raw prefill keys) *then* fan-out (broadcast the resulting
shared basis back to every member, including the one that triggered the
computation) pattern, unlike:

  * ``XQuantCoordinator`` — a single anchor publishes; reusers read (no
    fan-in, no joint computation).
  * ``MiniCacheCoordinator`` — one primary publishes; exactly one merge
    partner consumes (pairwise fan-in of two, not N).

All ``XKVCache`` instances of a model hold a reference to one shared
``XKVCoordinator`` (injected at ``KVCacheBuilder.for_model`` build time).

State is keyed by ``(group_id, token_start)`` so every member of a group
publishes for the *same* token range before the joint SVD runs. The basis is
computed lazily on the Nth (last) publish for that key and memoized — every
member's ``get_shared_basis`` call for the same key, whenever it arrives,
returns the identical basis. Each ``XKVCache`` caches its own copy locally
after the first successful fetch and never calls the coordinator again for
that generation, so the fan-in/fan-out round-trip happens exactly once per
group per generation (at prefill).

Single-threaded by construction (mlx generate is sequential), so a plain
dict-of-pending-publishes needs no locking.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from veloxquant_mlx.quantizers.xkv import joint_svd_compress


class _PendingGroup:
    """Raw keys published so far for one (group_id, token_start), awaiting
    the remaining members before the joint SVD can run."""

    __slots__ = ("token_start", "n_tokens", "member_keys", "basis")

    def __init__(self, token_start: int, n_tokens: int):
        self.token_start = token_start
        self.n_tokens = n_tokens
        self.member_keys: dict[int, mx.array] = {}  # member_idx -> [S, D]
        self.basis: Optional[tuple[mx.array, mx.array, mx.array]] = None


class XKVCoordinator:
    """Shared cross-layer basis store for one model instance / one generation.

    Args:
        max_ctx: Per-group token budget; publishing beyond this raises.

    Note: the shared rank / energy-threshold are NOT stored on the
    coordinator — every member of a group is built from the same
    ``KVCacheConfig`` (see ``KVCacheBuilder._build_xkv``), so the calling
    ``XKVCache`` passes its own ``xkv_rank``/``xkv_energy_threshold`` into
    ``get_shared_basis`` on every call. This avoids a second place where
    those two values could silently drift out of sync with the config.
    """

    def __init__(self, max_ctx: int = 8192) -> None:
        self._max_ctx = int(max_ctx)
        # group_id -> {token_start -> _PendingGroup}
        self._store: dict[int, dict[int, _PendingGroup]] = {}
        self._published_tokens: dict[int, int] = {}

    def reset(self) -> None:
        self._store.clear()
        self._published_tokens.clear()

    def publish_member_keys(
        self,
        group_id: int,
        member_idx: int,
        token_start: int,
        n_tokens: int,
        keys: mx.array,
    ) -> None:
        """A group member publishes its own raw keys for this token range.

        Args:
            group_id: The group this layer belongs to.
            member_idx: This layer's 0-indexed position within the group.
            token_start: Offset of the first token in this write.
            n_tokens: Number of tokens in this write.
            keys: This layer's own keys for the range, shape ``[S, D]``.
        """
        published = self._published_tokens.get(group_id, 0)
        if member_idx == 0 and published + n_tokens > self._max_ctx:
            raise RuntimeError(
                f"XKVCoordinator: group {group_id} exceeds max_ctx="
                f"{self._max_ctx} (have {published}, +{n_tokens}). "
                f"Increase xkv_max_ctx."
            )
        groups = self._store.setdefault(group_id, {})
        pending = groups.get(token_start)
        if pending is None:
            pending = _PendingGroup(token_start, n_tokens)
            groups[token_start] = pending
        pending.member_keys[member_idx] = keys
        if member_idx == 0:
            self._published_tokens[group_id] = published + n_tokens

    def get_shared_basis(
        self,
        group_id: int,
        token_start: int,
        expected_members: int,
        rank: Optional[int] = None,
        energy_threshold: float = 0.95,
    ) -> Optional[tuple[mx.array, mx.array, mx.array]]:
        """Return the shared basis for this group/token range once every
        expected member has published; ``None`` if still waiting.

        Computes the joint SVD lazily on the call that observes the last
        of ``expected_members`` publishes, and memoizes the result so every
        subsequent (or concurrent, in iteration order) caller for the same
        key gets back the identical arrays. ``rank``/``energy_threshold`` are
        only consulted on that triggering call (every group member shares the
        same ``KVCacheConfig`` values, so it does not matter which caller's
        values happen to trigger the computation).
        """
        pending = self._store.get(group_id, {}).get(token_start)
        if pending is None:
            return None
        if pending.basis is not None:
            return pending.basis
        if len(pending.member_keys) < expected_members:
            return None
        ordered_keys = [pending.member_keys[i] for i in sorted(pending.member_keys)]
        V_g, K_mean_g, s_g = joint_svd_compress(
            ordered_keys, rank=rank, energy_threshold=energy_threshold
        )
        mx.eval(V_g, K_mean_g, s_g)
        pending.basis = (V_g, K_mean_g, s_g)
        return pending.basis

    def published_tokens(self, group_id: int) -> int:
        return self._published_tokens.get(group_id, 0)

    @property
    def max_ctx(self) -> int:
        return self._max_ctx


__all__ = ["XKVCoordinator"]
