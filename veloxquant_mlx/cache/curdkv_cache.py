"""CurDKV-adapted KV cache — value-aware leverage-score heavy-hitter eviction.

Inspired by "Value-Guided KV Compression for LLMs via Approximated CUR
Decomposition" (Sengupta, Chaudhary, Chakraborty; NeurIPS 2025,
arXiv:2509.15038). Documented as "CurDKV-adapted (VeloxQuant-MLX
implementation)" — not a faithful port.

Value-aware eviction: each incoming token's approximate leverage score over
the existing cache is estimated from the joint (key, value) structure — the
proxy attention-weighted value block's dominant singular directions — using
the new key vector as a proxy query (true query not visible at the cache
wrapper level). The resulting scores are accumulated into a per-token
cumulative importance score. Whenever the cache exceeds ``curdkv_budget``
tokens, the lowest-score non-sink token is permanently dropped. The cache
never exceeds ``curdkv_budget`` positions.

This is the fourteenth eviction-family method in VeloxQuant-MLX, and the
first whose retention score is value-aware rather than key-only:
  - H2O-adapted    : cumulative attention mass over keys only.
  - KNorm-adapted  : intrinsic key-vector norm only.
  - Q-Filters      : frozen per-head key-SVD projection direction.
  - CurDKV-adapted : leverage scores over the joint (key, value) block — a
                     token with a "important-looking" key but a
                     near-zero/orthogonal value contribution is correctly
                     deprioritized here, unlike the key-only methods above.

Adaptation limitations (stated plainly):
  - Key-as-query proxy: leverage scores are computed using the new key vector
    in place of the true query (not visible at cache level). Same
    approximation as H2O-adapted/SnapKV-adapted.
  - Approximated leverage scores via a small-rank SVD of the proxy
    attention-weighted value block, not the paper's own CUR sampling
    algorithm — a standard leverage-score estimator, not a reproduction of
    the paper's specific sketching routine.
  - Uniform budget and n_sink across all heads.

FIXED, CONFIRMED-ON-REAL-MODELS PROBLEM, WITH A KNOWN REMAINING LIMITATION:
see :mod:`veloxquant_mlx.quantizers.curdkv` module docstring for the full
investigation. In short — this cache used to reset ``self.offset`` to the
post-eviction kept-row count every call, but ``mlx_lm``'s attention module
rotates the next query/key using ``self.rope(x, offset=cache.offset)``
*before* ``update_and_fetch`` is ever called, so that offset must equal the
true absolute step count, not the kept-row count. Fixed the same way
H2O-adapted already fixes it: ``self.offset`` now tracks the true absolute
step count directly, and surviving keys are re-rotated on eviction (see
``curdkv_update`` / ``rope_remap_positions``) so their baked-in rotation
stays consistent with their shifted storage index — verified via a
synthetic reproduction of ``mlx_lm.generate()``'s exact bulk-then-final-
token call split.

That fix is necessary but NOT sufficient for correct output on Llama-3-family
models: they use ``rope_scaling={"rope_type": "llama3", ...}``, a piecewise
per-frequency rescaling that ``rope_remap_positions``'s plain-RoPE math
doesn't reproduce (regardless of which ``curdkv_rope_base`` is configured).
Confirmed on a real Llama-3.2-1B run: every row an eviction re-rotated came
out numerically wrong, and generation was still degraded even with the
offset-desync bug fixed. This is a real, disclosed gap, not swept under the
"FIXED" heading above — see the quantizer module docstring and
https://github.com/rajveer43/VeloxQuant-MLX/issues/148 for the full detail
and reproduction. Affects H2O-adapted identically (same underlying
primitive).

Byte accounting:
    curdkv_kept_bytes  — fp16 bytes for currently retained K + V tokens
    full_seq_bytes     — hypothetical fp16 cost if all tokens were kept
    compression_ratio  — full_seq_bytes / curdkv_kept_bytes (> 1 = savings)
    tokens_seen        — total token positions ever passed to update_and_fetch
    tokens_kept        — tokens currently in the first (B=0, H=0) head's cache
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.a2ats_rope import rope_remap_positions
from veloxquant_mlx.quantizers.curdkv import (
    CurDKVState,
    curdkv_fp16_bytes,
    curdkv_get_kv,
    curdkv_update,
    full_curdkv_fp16_bytes,
    init_curdkv_state,
)


class CurDKVKVCache(_MLXKVCache):
    """KV cache implementing CurDKV-adapted value-aware leverage-score eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``curdkv_budget`` (int, default 512)   — maximum tokens retained at any time,
            ``curdkv_n_sink`` (int, default 4)      — leading positions never evicted,
            ``curdkv_rank_cap`` (int, default 16)   — SVD rank cap for leverage-score estimation,
            ``curdkv_rope_base`` (float, default 10000.0) — RoPE frequency base,
            must match the model's own attention RoPE base for post-eviction
            position remapping to cancel out the original rotation correctly.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Both prefill (S > 1) and decode (S == 1) tokens go through the same
        CurDKV update loop — unlike SnapKV-adapted, there is no prefill-only
        phase.
        Single-layer (no coordinator); ``KVCacheBuilder.for_model()`` propagates
        all ``curdkv_*`` fields automatically via ``dataclasses.replace``.
        The per-head state is lazily initialised on the first call to
        ``update_and_fetch`` when shapes are known.
        Does NOT delegate to the base ``mlx_lm`` ``KVCache.update_and_fetch``:
        that implementation assumes ``self.offset`` always equals the number
        of rows physically stored, which eviction breaks. Instead
        ``self.keys``/``self.values`` hold exactly the ``n_kept`` retained
        rows and ``self.offset`` is kept equal to the *true absolute step
        count*, so ``mlx_lm``'s ``self.rope(x, offset=cache.offset)`` call —
        made on both the query and the next incoming key, entirely inside the
        model's attention module, before this cache ever sees them — rotates
        at the correct position without this cache needing to intercept the
        query at all (see module docstring's FIXED, CONFIRMED-ON-REAL-MODELS
        PROBLEM). ``is_trimmable()`` reports ``False`` since the internal
        per-token state can't be rolled back by a base-class ``trim()``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "curdkv_budget", 512))
        self._n_sink = int(getattr(config, "curdkv_n_sink", 4))
        self._rank_cap = int(getattr(config, "curdkv_rank_cap", 16))
        self._rope_base = float(getattr(config, "curdkv_rope_base", 10000.0))

        self._head_dim: int = 0
        self._states: list[CurDKVState] = []
        self._B: int = 0
        self._H: int = 0

        self._curdkv_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        """Lazily initialise per-head CurDKVState list on first call."""
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            self._states = [
                init_curdkv_state(
                    self._n_sink, self._budget, D, self._rank_cap, rope_base=self._rope_base
                )
                for _ in range(B * H)
            ]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    def _fix_incoming_rope(self, keys: mx.array, offset_before: int, next_pos: int) -> mx.array:
        """Re-rotate incoming keys if the model rotated them at the wrong
        absolute position.

        Only matters as a defensive fallback: with ``self.offset`` now always
        equal to the true absolute step count (see the class docstring),
        ``offset_before`` and ``next_pos`` should already agree, and this is a
        no-op. Kept so that a future change which reintroduces an
        offset/position desync fails safe (corrects it) rather than silently
        corrupting generation again like the original bug this cache had.
        """
        if offset_before == next_pos:
            return keys
        B, H, S, D = keys.shape
        old_positions = mx.arange(offset_before, offset_before + S, dtype=mx.int32)
        new_positions = mx.arange(next_pos, next_pos + S, dtype=mx.int32)
        out_b = []
        for b in range(B):
            out_h = []
            for h in range(H):
                base = self._states[self._head_idx(b, h)].rope_base
                out_h.append(
                    rope_remap_positions(keys[b, h], old_positions, new_positions, base=base)
                )
            out_b.append(mx.stack(out_h, axis=0))
        return mx.stack(out_b, axis=0)

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply CurDKV eviction, return retained window.

        Manages ``self.keys`` / ``self.values`` / ``self.offset`` directly
        instead of delegating to the base class's append-only buffer, because
        that buffer assumes ``offset`` always equals the stored row count —
        false here the moment eviction drops a row. Keeping ``self.offset``
        equal to the *true absolute step count* (not the kept-row count) is
        what keeps ``mlx_lm``'s attention module rotating the next query and
        incoming key at the correct position — see the class/module
        docstrings for why getting this wrong silently corrupted generation.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_kept, D]`` fp16, where
            ``n_kept <= curdkv_budget`` for all heads.
        """
        B, H, S, D = keys.shape
        self._ensure_states(B, H, D)

        offset_before = self.offset
        next_pos = self._states[0].next_pos  # identical across heads (see class docstring)

        self._full_seq_bytes += B * H * S * D * 2 * 2  # K + V, fp16
        self._tokens_seen_total += B * H * S

        keys_fixed = self._fix_incoming_rope(keys.astype(mx.float16), offset_before, next_pos)

        k_out_b, v_out_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                idx = self._head_idx(b, h)
                st = self._states[idx]
                st = curdkv_update(
                    st,
                    keys_fixed[b, h],
                    values[b, h].astype(mx.float16),
                )
                self._states[idx] = st
                k_h, v_h = curdkv_get_kv(st)
                k_out_h.append(k_h)  # [n_kept, D]
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))  # [H, n_kept, D]
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)  # [B, H, n_kept, D]
        V_out = mx.stack(v_out_b, axis=0)

        # Byte accounting: sum across all head states
        self._curdkv_kept_bytes = sum(curdkv_fp16_bytes(st) for st in self._states)

        # Store exactly the n_kept retained rows — NOT delegated to the base
        # class's update_and_fetch, whose growing-buffer bookkeeping assumes
        # offset == stored row count. self.offset instead tracks the true
        # absolute step count, so the model's NEXT rope(..., offset=self.offset)
        # call rotates correctly even though fewer than `offset` rows are
        # physically stored.
        self.keys = K_out
        self.values = V_out
        self.offset = self._states[0].next_pos
        return K_out, V_out

    # ------------------------------------------------------------------
    def is_trimmable(self) -> bool:
        """False: trim() would only roll back base-class offset bookkeeping,
        not the internal per-token eviction/compression state that actually
        determines what gets returned, silently corrupting future calls.
        """
        return False

    def size(self) -> int:
        """Rows actually stored (``n_kept``), NOT ``self.offset`` (the true
        absolute step count) — the base class conflates the two, which is
        exactly the assumption this cache must not make. Overridden
        explicitly rather than relying on the base class's coincidentally
        harmless behavior here."""
        return 0 if self.keys is None else self.keys.shape[2]

    @property
    def state(self):
        """``(keys, values)`` — always exactly the ``n_kept`` stored rows.

        Not delegated to the base class's getter: it slices
        ``self.keys[..., :self.offset, :]``, and since ``self.offset`` is the
        true absolute step count (usually > n_kept once eviction has
        happened), that would silently clamp to all stored rows anyway via
        MLX's slice semantics — correct by coincidence, not by contract. Made
        explicit here instead.
        """
        return self.keys, self.values

    @state.setter
    def state(self, v):
        """Restoring from a saved state cannot recover the true step count
        that produced it (CurDKV's own eviction history is not persisted), so
        ``self.offset`` is set to the stored row count as the least-wrong
        available estimate. Loading a saved CurDKV cache mid-eviction-history
        is not a supported/tested path.
        """
        self.keys, self.values = v
        self.offset = 0 if self.keys is None else self.keys.shape[2]

    # ------------------------------------------------------------------
    @property
    def curdkv_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._curdkv_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / curdkv_kept_bytes; > 1 means memory savings over fp16."""
        if self._curdkv_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._curdkv_kept_bytes

    @property
    def tokens_seen(self) -> int:
        """Total token positions ever passed to update_and_fetch (all heads summed)."""
        return self._tokens_seen_total

    @property
    def tokens_kept(self) -> int:
        """Tokens currently in the (B=0, H=0) head's cache (diagnostic)."""
        if not self._states or self._states[0].keys is None:
            return 0
        return int(self._states[0].keys.shape[0])


__all__ = ["CurDKVKVCache"]
