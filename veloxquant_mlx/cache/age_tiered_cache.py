"""AgeTieredKV KV cache — position/age-driven three-tier precision (issue #256).

Every token is retained (compression-only, like AMC — never eviction). Each
token's bit-width is set purely by how long ago it was written:
``age = current_position - token_position``. As a token ages past a tier
boundary, its stored slice is re-quantized at the coarser tier's bit-width;
this mirrors KIVI's "flush on boundary crossing" model rather than
quantizing once at write time and leaving the choice frozen — a token
written under this scheme is never *coarser* than its current age entitles,
which is what issue #256's "recent -> high precision, older -> low
precision" framing calls for.

See :mod:`veloxquant_mlx.quantizers.age_tiered` for the tiering/quantize
primitives this class wraps around a per-(batch, head) token buffer, the
same storage shape AMC and KIVI both use.
"""

from __future__ import annotations

from typing import Any, Dict, List

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.age_tiered import (
    MID,
    OLD,
    RECENT,
    age_tier_quantize,
    age_tiered_bytes,
    assign_age_tiers,
    default_age_tiers,
    full_fp16_bytes,
)


class AgeTieredKVCache(_MLXKVCache):
    """KV cache implementing position/age-gated three-tier quantization.

    Args:
        config: :class:`~veloxquant_mlx.cache.base.KVCacheConfig`. Fields
            consumed:
                ``age_recent_boundary`` (int, default 128) — tokens with
                    age < this stay at ``age_bits_recent``.
                ``age_mid_boundary`` (int, default 1024) — tokens with age
                    in ``[age_recent_boundary, age_mid_boundary)`` use
                    ``age_bits_mid``; ``age >= age_mid_boundary`` uses
                    ``age_bits_old``.
                ``age_bits_recent`` (int, default 8)
                ``age_bits_mid`` (int, default 4)
                ``age_bits_old`` (int, default 2)
                ``age_group_size`` (int, default 32) — token-axis group size
                    for the shared min/max quantizer.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly (same
        quantize-then-dequantize convention as every other method here).
        Both prefill (S > 1) and decode (S == 1) go through the identical
        per-step re-tiering pass. Writes through to the base ``mlx_lm``
        ``KVCache``'s ``self.keys`` / ``self.values`` / ``self.offset`` on
        every call, same as AMC (see #83) — without this, ``.state`` stays
        at its ``__init__`` default and mlx_lm's chunked prefill crashes.
        ``is_trimmable()`` reports ``False`` for the same reason AMC's does:
        the internal per-token tier state can't be rolled back by a
        base-class ``trim()``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._age_recent_boundary = int(getattr(config, "age_recent_boundary", 128))
        self._age_mid_boundary = int(getattr(config, "age_mid_boundary", 1024))
        if self._age_recent_boundary <= 0:
            raise ValueError(
                f"AgeTieredKVCache: age_recent_boundary must be > 0, got "
                f"{self._age_recent_boundary}."
            )
        if self._age_mid_boundary < self._age_recent_boundary:
            raise ValueError(
                f"AgeTieredKVCache: age_mid_boundary ({self._age_mid_boundary}) must be "
                f">= age_recent_boundary ({self._age_recent_boundary})."
            )

        bits_recent = int(getattr(config, "age_bits_recent", 8))
        bits_mid = int(getattr(config, "age_bits_mid", 4))
        bits_old = int(getattr(config, "age_bits_old", 2))
        for name, b in (("age_bits_recent", bits_recent), ("age_bits_mid", bits_mid), ("age_bits_old", bits_old)):
            if not 1 <= b <= 16:
                raise ValueError(f"AgeTieredKVCache: {name} must be in [1, 16], got {b}.")
        self._tiers = default_age_tiers(bits_recent, bits_mid, bits_old)

        self._group_size = int(getattr(config, "age_group_size", 32))
        self._head_dim: int = int(getattr(config, "head_dim", 128))

        self._B: int = 0
        self._H: int = 0
        self._keys: List[mx.array] = []  # per (b,h): [n_seen, D] fp16
        self._values: List[mx.array] = []  # per (b,h): [n_seen, D] fp16

        self._tokens_seen_total: int = 0
        self._current_position: int = 0
        self._age_tiered_bytes: int = 0
        self._full_seq_bytes: int = 0

    # ------------------------------------------------------------------
    def _ensure_state(self, B: int, H: int) -> None:
        if not self._keys:
            self._B = B
            self._H = H
            self._keys = [None] * (B * H)
            self._values = [None] * (B * H)

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, re-tier the whole buffer by age, return all tokens.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_seen, D]`` fp16 — AgeTieredKV
            never evicts, so ``n_seen`` equals the total tokens passed so far.
        """
        B, H, S, D = keys.shape
        self._ensure_state(B, H)

        self._full_seq_bytes += full_fp16_bytes(B * H * S, D)
        self._tokens_seen_total += B * H * S
        self._current_position += S

        k_out_b, v_out_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                idx = self._head_idx(b, h)
                k_step = keys[b, h].astype(mx.float16)  # [S, D]
                v_step = values[b, h].astype(mx.float16)  # [S, D]

                prev_k = self._keys[idx]
                prev_v = self._values[idx]
                new_k = k_step if prev_k is None else mx.concatenate([prev_k, k_step], axis=0)
                new_v = v_step if prev_v is None else mx.concatenate([prev_v, v_step], axis=0)
                n = int(new_k.shape[0])

                # age[i] = current_position - (i + 1); the token written this
                # step (i == n - 1) has age 0.
                ages = [self._current_position - (i + 1) for i in range(n)]
                tiers = assign_age_tiers(ages, self._age_recent_boundary, self._age_mid_boundary)

                new_k = self._requantize(new_k, tiers)
                new_v = self._requantize(new_v, tiers)

                self._keys[idx] = new_k
                self._values[idx] = new_v
                k_out_h.append(new_k)
                v_out_h.append(new_v)
            k_out_b.append(mx.stack(k_out_h, axis=0))
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)
        V_out = mx.stack(v_out_b, axis=0)

        self._age_tiered_bytes = age_tiered_bytes(
            self._accumulate_tier_counts(), self._tiers, self._head_dim
        )

        self.keys = None
        self.values = None
        self.offset = 0
        return super().update_and_fetch(K_out, V_out)

    def _accumulate_tier_counts(self) -> Dict[int, int]:
        """Recompute cumulative tier counts across all (b,h) from current state.

        Re-derives from ``self._keys`` (whose length per head always equals
        the total tokens seen by that head) rather than trying to track a
        running delta, since a token's tier can change between calls as it
        ages — a delta would double count or drift.
        """
        if not self._keys or self._keys[0] is None:
            return {RECENT: 0, MID: 0, OLD: 0}
        n_per_head = int(self._keys[0].shape[0])
        ages = [self._current_position - (i + 1) for i in range(n_per_head)]
        tiers = assign_age_tiers(ages, self._age_recent_boundary, self._age_mid_boundary)
        counts = {RECENT: 0, MID: 0, OLD: 0}
        for t in tiers:
            counts[t] += 1
        n_heads = len(self._keys)
        return {k: v * n_heads for k, v in counts.items()}

    def _requantize(self, x: mx.array, tiers: List[int]) -> mx.array:
        """Re-quantize each contiguous same-tier run of ``x`` at its tier's bit-width."""
        n = x.shape[0]
        if n == 0:
            return x
        by_tier = {cfg.tier: cfg.bits for cfg in self._tiers}
        out_chunks = []
        start = 0
        for i in range(1, n + 1):
            if i == n or tiers[i] != tiers[start]:
                bits = by_tier[tiers[start]]
                out_chunks.append(age_tier_quantize(x[start:i], bits, self._group_size))
                start = i
        return mx.concatenate(out_chunks, axis=0)

    # ------------------------------------------------------------------
    def is_trimmable(self) -> bool:
        """False: trim() would only roll back base-class offset bookkeeping,
        not the internal per-token tier state, silently corrupting future
        calls (same rationale as AMC's, see #83).
        """
        return False

    # ------------------------------------------------------------------
    @property
    def age_tiered_bytes(self) -> int:
        """Actual bytes stored across all heads (fp16-equivalent K + V, tiered)."""
        return self._age_tiered_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if AgeTieredKV were never applied."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / age_tiered_bytes; > 1 means savings over fp16."""
        if self._age_tiered_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._age_tiered_bytes

    @property
    def tokens_seen(self) -> int:
        """Total token positions ever passed to update_and_fetch (all heads summed)."""
        return self._tokens_seen_total

    @property
    def tokens_kept(self) -> int:
        """Tokens currently in the (B=0, H=0) head's cache — always == tokens per head seen."""
        if not self._keys or self._keys[0] is None:
            return 0
        return int(self._keys[0].shape[0])

    @property
    def tokens_recent(self) -> int:
        """Tokens currently in the RECENT (highest-precision) tier, across all heads."""
        return self._accumulate_tier_counts()[RECENT]

    @property
    def tokens_mid(self) -> int:
        """Tokens currently in the MID tier, across all heads."""
        return self._accumulate_tier_counts()[MID]

    @property
    def tokens_old(self) -> int:
        """Tokens currently in the OLD (lowest-precision) tier, across all heads."""
        return self._accumulate_tier_counts()[OLD]


__all__ = ["AgeTieredKVCache"]
