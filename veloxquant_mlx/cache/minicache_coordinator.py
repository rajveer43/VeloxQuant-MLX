"""MiniCache cross-layer coordinator — shared merged-direction state.

Like the XQuant coordinator, this gives per-layer cache objects a place to
coordinate without modifying the model forward pass. Each ``MiniCacheKVCache``
of a model holds a reference to one shared ``MiniCacheCoordinator`` (injected at
``KVCacheBuilder.for_model`` build time).

A **primary** layer merges its KV with its paired **merge** layer's KV — but a
cache only sees its *own* layer at ``update_and_fetch`` time. So the protocol is:

  * The primary layer, when it cannot yet see the merge layer's tensor, stores
    its own KV for this step and reconstructs itself losslessly (no merge yet).
  * The merge layer, arriving later in the same forward pass, fetches the
    primary's stored KV for the same token range, performs the SLERP merge, and
    publishes the shared direction + both magnitudes back. It reconstructs
    *itself* from the merge result.
  * Because both layers' true tensors were available at merge time, both
    reconstructions use the shared direction — the storage charged to the pair
    is one direction tensor + two magnitude scalars per token (the win).

State is keyed by ``(group_id, token_start)`` so a merge layer fetches exactly
the segment its paired primary wrote for the same step. Single-threaded by
construction (mlx generate is sequential).

Each published write is consumed by a fixed number of merge layers (one per
non-primary member of the group — usually 1, but ``minicache_group_size > 2``
means one primary feeds multiple mergers). ``fetch_primary`` counts reads per
entry and only reclaims it (dropping it from ``_store`` and crediting its
tokens back against ``_published_tokens``) once every expected reader has
consumed it — so ``_published_tokens`` tracks live, unconsumed tokens rather
than a lifetime total, and a long generation no longer exhausts ``max_ctx`` (#79).
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx


class _PrimaryWrite:
    """A primary layer's stored KV for a token range, awaiting its merge partner."""

    __slots__ = ("token_start", "n_tokens", "keys", "values", "reads_remaining")

    def __init__(
        self,
        token_start: int,
        n_tokens: int,
        keys: mx.array,
        values: mx.array,
        n_readers: int,
    ):
        self.token_start = token_start
        self.n_tokens = n_tokens
        self.keys = keys  # [B, H, S, D] fp16 — the primary layer's true KV
        self.values = values
        self.reads_remaining = n_readers


class MiniCacheCoordinator:
    """Shared cross-layer store for one model instance / one generation.

    Args:
        max_ctx: Per-group token budget; primary writes beyond this raise.
    """

    def __init__(self, max_ctx: int = 8192) -> None:
        self._max_ctx = int(max_ctx)
        # group_id -> {token_start -> _PrimaryWrite}
        self._store: dict[int, dict[int, _PrimaryWrite]] = {}
        self._published_tokens: dict[int, int] = {}

    def reset(self) -> None:
        self._store.clear()
        self._published_tokens.clear()

    def publish_primary(
        self,
        group_id: int,
        token_start: int,
        n_tokens: int,
        keys: mx.array,
        values: mx.array,
        n_readers: int = 1,
    ) -> None:
        """Primary layer stores its true KV for a token range.

        Args:
            group_id: Group this primary owns.
            token_start: Offset of the first token in this write.
            n_tokens: Number of tokens in this write.
            keys: Primary layer's true keys.
            values: Primary layer's true values.
            n_readers: Number of merge layers expected to fetch this entry
                before it is reclaimed (group_size - 1; default 1).
        """
        published = self._published_tokens.get(group_id, 0)
        if published + n_tokens > self._max_ctx:
            raise RuntimeError(
                f"MiniCacheCoordinator: group {group_id} exceeds max_ctx="
                f"{self._max_ctx} (have {published}, +{n_tokens}). "
                f"Increase minicache_max_ctx."
            )
        if n_readers <= 0:
            # Degenerate primary (no merge partners in its group, e.g. an
            # early standalone layer below minicache_start_frac): nothing
            # will ever call fetch_primary for this entry, so storing it
            # would leak forever. No-op — the caller doesn't need it back.
            return
        self._store.setdefault(group_id, {})[token_start] = _PrimaryWrite(
            token_start, n_tokens, keys, values, n_readers
        )
        self._published_tokens[group_id] = published + n_tokens

    def fetch_primary(self, group_id: int, token_start: int) -> Optional[_PrimaryWrite]:
        """Merge layer fetches its paired primary's KV for this token range.

        Once every expected reader (``n_readers`` passed to ``publish_primary``)
        has fetched an entry, it is reclaimed: dropped from the store and its
        tokens credited back against ``_published_tokens``, so a long-running
        generation never exhausts ``max_ctx`` on live-but-consumed entries (#79).
        """
        group_store = self._store.get(group_id, {})
        entry = group_store.get(token_start)
        if entry is None:
            return None
        entry.reads_remaining -= 1
        if entry.reads_remaining <= 0:
            del group_store[token_start]
            self._published_tokens[group_id] = (
                self._published_tokens.get(group_id, 0) - entry.n_tokens
            )
        return entry

    def published_tokens(self, group_id: int) -> int:
        return self._published_tokens.get(group_id, 0)

    @property
    def max_ctx(self) -> int:
        return self._max_ctx


__all__ = ["MiniCacheCoordinator"]
