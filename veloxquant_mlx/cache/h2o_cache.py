"""H2O-adapted KV cache — cumulative attention-mass heavy-hitter oracle eviction.

Inspired by "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of
Large Language Models" (Zhang et al., ICLR 2024, arXiv:2306.14048).
Documented as "H2O-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

Heavy-hitter eviction: each incoming token's approximate attention distribution
over the existing cache is computed using the new key vector as a proxy query
(true query not visible at the cache wrapper level). The resulting softmax
weights are accumulated into a per-token cumulative importance score. Whenever
the cache exceeds ``h2o_budget`` tokens, the lowest-score non-sink token is
permanently dropped. The cache never exceeds ``h2o_budget`` positions.

This is the third distinct eviction axis in VeloxQuant-MLX:
  - SnapKV-adapted : score-based, fires once at prefill end only.
  - StreamingLLM-adapted : positional (recency + sink), fires every step.
  - H2O-adapted    : cumulative attention mass, fires every step when over budget.

Adaptation limitations (stated plainly):
  - Key-as-query proxy: attention weights are computed using the new key vector
    in place of the true query (not visible at cache level). Same approximation
    as SnapKV-adapted.
  - Uniform budget and n_sink across all heads.
  - Scores accumulate as a running sum of softmax weights; the paper accumulates
    unnormalised attention logits in some variants — this may diverge at very
    low budgets.
  - RoPE remapping assumes ``traditional=False`` (NeoX-style) rotation, MLX's
    default and what Llama/Mistral/Qwen use — see
    :func:`veloxquant_mlx.quantizers.a2ats_rope.rope_remap_positions`.

KNOWN, CONFIRMED-ON-REAL-MODELS PROBLEM (not fixed here — needs a different
eviction/scoring policy, see docs-site/docs/algorithms/h2o.md for the full
writeup): real end-to-end generation testing on Llama-3.2-1B and Mistral-7B
showed generation degrading into repetition loops (mild) or total collapse
(severe) once the cache fills. Root cause, verified precisely: every
newly-arrived token starts at score 0.0, and eviction removes the
global-minimum-score token, so the brand-new token is (almost always) its own
eviction target the moment the budget is full — before it ever accumulates
attention mass. Traced directly: after 43 true decode steps on Llama-3.2-1B,
the kept set was still exactly the original prompt's positions; the model
generated the entire time with zero visibility into its own output. This is
what implementing the paper's own scoring formula correctly produces at tight
budgets — not a deviation from the paper, and not fixed by anything in this
module. A real fix needs a different eviction/scoring rule (e.g. a grace
period before a token becomes eviction-eligible).

RoPE position remapping (a separate, narrower, and now-fixed correctness
issue found during the same investigation — real, but NOT the cause of the
freeze above, since interior eviction rarely happens once the freeze sets
in). Two distinct causes, both fixed, for the case interior eviction *does*
occur (e.g. with ``h2o_n_sink > 0``, where sink rows are permanently
protected while later rows can still eventually be evicted, opening a real
gap):

  1. Stored keys arrive already RoPE-rotated for their true absolute
     position. Dropping a middle row leaves survivors' storage index out of
     sync with that baked-in rotation. Fixed by de-rotating and re-rotating
     kept keys to a gap-free layout (``rope_remap_positions``) whenever
     eviction changes the kept set — see ``h2o_update`` in
     :mod:`veloxquant_mlx.quantizers.h2o`.
  2. ``mlx_lm``'s attention module rotates BOTH the query and the incoming
     key using ``self.rope(x, offset=cache.offset)``, entirely inside the
     model's forward pass, before ``update_and_fetch`` is ever called — this
     cache cannot intercept or correct the query's rotation after the fact.
     The only way to make that rotation correct is for ``self.offset`` itself
     to always equal the true absolute step count rather than the number of
     rows this cache has kept. Fixed by managing ``self.keys``/``self.values``/
     ``self.offset`` directly (not delegating to the base class's
     append-only buffer, which assumes ``offset == stored row count``) — see
     ``update_and_fetch`` below.

Verified via a synthetic fingerprinted-token test that forces an interior
eviction and confirms every surviving key de-rotates back to its exact
original unrotated value. ``h2o_rope_base`` must match the model's own RoPE
base for cause (1)'s correction to cancel out the original rotation
correctly.

Byte accounting:
    h2o_kept_bytes    — fp16 bytes for currently retained K + V tokens
    full_seq_bytes    — hypothetical fp16 cost if all tokens were kept
    compression_ratio — full_seq_bytes / h2o_kept_bytes (> 1 = savings)
    tokens_seen       — total token positions ever passed to update_and_fetch
    tokens_kept       — tokens currently in the first (B=0, H=0) head's cache
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.a2ats_rope import rope_remap_positions
from veloxquant_mlx.quantizers.h2o import (
    H2OState,
    full_h2o_fp16_bytes,
    h2o_fp16_bytes,
    h2o_get_kv,
    h2o_update,
    init_h2o_state,
)


class H2OKVCache(_MLXKVCache):
    """KV cache implementing H2O-adapted heavy-hitter oracle eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``h2o_budget`` (int, default 512) — maximum tokens retained at any time,
            ``h2o_n_sink`` (int, default 4)   — leading positions never evicted,
            ``h2o_rope_base`` (float, default 10000.0) — RoPE frequency base,
            must match the model's own attention RoPE base for post-eviction
            position remapping to cancel out the original rotation correctly.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Both prefill (S > 1) and decode (S == 1) tokens go through the same
        H2O update loop — unlike SnapKV-adapted, there is no prefill-only phase.
        Single-layer (no coordinator); ``KVCacheBuilder.for_model()`` propagates
        all ``h2o_*`` fields automatically via ``dataclasses.replace``.
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
        query at all. ``.state``/``.size()``/``is_trimmable()`` are overridden
        to keep this contract explicit rather than relying on the base
        class's coincidentally-compatible slicing behavior (see #83 for the
        historical reason ``.state`` must stay valid across chunked prefill).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "h2o_budget", 512))
        self._n_sink = int(getattr(config, "h2o_n_sink", 4))
        self._rope_base = float(getattr(config, "h2o_rope_base", 10000.0))

        self._head_dim: int = 0
        self._states: list[H2OState] = []
        self._B: int = 0
        self._H: int = 0

        self._h2o_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        """Lazily initialise per-head H2OState list on first call."""
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            self._states = [
                init_h2o_state(self._n_sink, self._budget, D, rope_base=self._rope_base)
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
        """Absorb new K/V tokens, apply H2O eviction, return retained window.

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
            ``n_kept <= h2o_budget`` for all heads.
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
                st = h2o_update(
                    st,
                    keys_fixed[b, h],
                    values[b, h].astype(mx.float16),
                )
                self._states[idx] = st
                k_h, v_h = h2o_get_kv(st)
                k_out_h.append(k_h)  # [n_kept, D]
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))  # [H, n_kept, D]
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)  # [B, H, n_kept, D]
        V_out = mx.stack(v_out_b, axis=0)

        # Byte accounting: sum across all head states
        self._h2o_kept_bytes = sum(h2o_fp16_bytes(st) for st in self._states)

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
        that produced it (H2O's own eviction history is not persisted), so
        ``self.offset`` is set to the stored row count as the least-wrong
        available estimate. Loading a saved H2O cache mid-eviction-history is
        not a supported/tested path.
        """
        self.keys, self.values = v
        self.offset = 0 if self.keys is None else self.keys.shape[2]

    # ------------------------------------------------------------------
    @property
    def h2o_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._h2o_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / h2o_kept_bytes; > 1 means memory savings over fp16."""
        if self._h2o_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._h2o_kept_bytes

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


__all__ = ["H2OKVCache"]
