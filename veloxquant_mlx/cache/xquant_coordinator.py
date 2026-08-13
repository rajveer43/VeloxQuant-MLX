"""XQuant cross-layer coordinator — shared anchor-code state across layers.

The repo's contract is one cache object per layer; ``mlx_lm.generate`` iterates
them independently. XQuant needs layers to coordinate (reuse layers borrow the
anchor layer's quantized codes). Rather than modify the model forward pass, all
``XQuantKVCache`` instances of a model hold a reference to a single shared
``XQuantCoordinator`` (injected at ``KVCacheBuilder.for_model`` build time).
Anchors publish their codes here; reusers read them back.

State is keyed by ``(group_id, token_start)`` so a reuse layer fetches exactly
the segment its paired anchor wrote for the same step. Prefill writes one large
segment (token_start=0); each decode step writes a one-token segment at the next
offset. A per-group token budget (``max_ctx``) bounds memory; exceeding it raises
``RuntimeError`` (mirrors the ``fused_sdpa_max_ctx`` guard in ``base.py``).

Each published segment is consumed by a fixed number of reuse layers (one per
non-anchor member of the group — usually 1, but ``xquant_group_size > 2`` means
one anchor feeds multiple reusers). ``fetch_anchor`` counts reads per segment
and only reclaims it (dropping it from ``_store`` and crediting its tokens back
against ``_published_tokens``) once every expected reader has consumed it —
so ``_published_tokens`` tracks live, unconsumed tokens rather than a lifetime
total, and a long generation no longer exhausts ``max_ctx`` (#80).

Single-threaded by construction (mlx generate is sequential), so a plain
dict-of-segments needs no locking.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from veloxquant_mlx.quantizers.xquant import GroupParams


class _Segment:
    """One published anchor write: codes for a contiguous token range."""

    __slots__ = ("token_start", "n_tokens", "codes", "params", "reads_remaining")

    def __init__(
        self,
        token_start: int,
        n_tokens: int,
        codes: mx.array,
        params: GroupParams,
        n_readers: int,
    ):
        self.token_start = token_start
        self.n_tokens = n_tokens
        self.codes = codes  # [B, H, n_groups, group_size, D] fp32 codes
        self.params = params  # anchor GroupParams (per (B,H) not stored here; see cache)
        self.reads_remaining = n_readers


class XQuantCoordinator:
    """Shared cross-layer code store for one model instance / one generation.

    Args:
        max_ctx: Per-group token budget. Anchor writes beyond this raise.
    """

    def __init__(self, max_ctx: int = 8192) -> None:
        self._max_ctx = int(max_ctx)
        # group_id -> {token_start -> _Segment}
        self._store: dict[int, dict[int, _Segment]] = {}
        # group_id -> running token count published by the anchor
        self._published_tokens: dict[int, int] = {}

    def reset(self) -> None:
        """Clear all published state (e.g. between generations)."""
        self._store.clear()
        self._published_tokens.clear()

    def register_anchor(
        self,
        group_id: int,
        token_start: int,
        n_tokens: int,
        codes: mx.array,
        params: GroupParams,
        n_readers: int = 1,
    ) -> None:
        """Publish anchor codes for a token range.

        Args:
            group_id: Group this anchor owns.
            token_start: Offset of the first token in this write.
            n_tokens: Number of tokens in this write.
            codes: Anchor integer codes (shape owned by the cache).
            params: Anchor GroupParams (for shape/bits reference).
            n_readers: Number of reuse layers expected to fetch this segment
                before it is reclaimed (group_size - 1; default 1).
        """
        published = self._published_tokens.get(group_id, 0)
        if published + n_tokens > self._max_ctx:
            raise RuntimeError(
                f"XQuantCoordinator: group {group_id} exceeds max_ctx={self._max_ctx} "
                f"(have {published}, +{n_tokens}). Increase xquant_max_ctx."
            )
        if n_readers <= 0:
            # Degenerate anchor (no reusers in its group, e.g. a trailing
            # partial group): nothing will ever call fetch_anchor for this
            # segment, so storing it would leak forever. No-op — the caller
            # doesn't need it back.
            return
        self._store.setdefault(group_id, {})[token_start] = _Segment(
            token_start, n_tokens, codes, params, n_readers
        )
        self._published_tokens[group_id] = published + n_tokens

    def fetch_anchor(self, group_id: int, token_start: int) -> Optional[_Segment]:
        """Fetch the anchor segment a reuse layer needs for this step.

        Once every expected reader (``n_readers`` passed to ``register_anchor``)
        has fetched a segment, it is reclaimed: dropped from the store and its
        tokens credited back against ``_published_tokens``, so a long-running
        generation never exhausts ``max_ctx`` on live-but-consumed segments (#80).

        Args:
            group_id: The reuse layer's group.
            token_start: The token offset of this step (matches the anchor write).

        Returns:
            The matching ``_Segment``, or ``None`` if the anchor has not
            published it yet (e.g. mis-ordered iteration — caller must handle).
        """
        group_store = self._store.get(group_id, {})
        seg = group_store.get(token_start)
        if seg is None:
            return None
        seg.reads_remaining -= 1
        if seg.reads_remaining <= 0:
            del group_store[token_start]
            self._published_tokens[group_id] = (
                self._published_tokens.get(group_id, 0) - seg.n_tokens
            )
        return seg

    def published_tokens(self, group_id: int) -> int:
        """Total tokens the anchor of ``group_id`` has published."""
        return self._published_tokens.get(group_id, 0)

    @property
    def max_ctx(self) -> int:
        return self._max_ctx


__all__ = ["XQuantCoordinator"]
