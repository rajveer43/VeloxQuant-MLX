"""NestedKV-adapted KV cache — multi-scale ensembled prefill eviction.

Inspired by "NestedKV: Nested Memory Routing for Long-Context KV Cache
Compression" (Chen, Liu, Gao, Fan, Wang, Chu, Lin, Hu; arXiv:2605.26678).
Documented as "NestedKV-adapted (VeloxQuant-MLX implementation)" — not a
faithful port. **No verified peer-reviewed venue as of 2026-07-14** — a
one-time, user-directed exception to this repo's venue-verification rule.
See ``paper/research/surveys/NEW_METHOD_SURVEY_V21.md``.

Multi-scale ensembled eviction: at the end of prefill, each head's tokens are
scored by three parallel continuum-memory anomaly signals (stable/global,
episodic/block-local, current/recent-window), combined via a head-adaptive
blend and a per-token surprise gate (see ``quantizers/nestedkv.py``). The
resulting per-token scores compete for a shared layer budget ACROSS heads
(not independently per head, unlike H2O/CurDKV) — a head whose tokens are
collectively more anomalous is allocated a larger share of the layer's total
budget. Decode tokens are appended unscored, never evicted — this is a
one-shot prefill compressor, not a per-step recurring eviction loop.

This is the 15th eviction-family method in VeloxQuant-MLX and the first that
ensembles multiple independent importance signals rather than committing to
one:
  - H2O / CurDKV / KVzip / Keyformer / MorphKV : one signal, scored every step.
  - SnapKV : one signal (obs-window attention), scored once at prefill.
  - NestedKV : THREE signals, scored once at prefill, combined by a
    head-adaptive blend + per-token surprise gate, with cross-head budget
    competition.

Adaptation limitations (stated plainly — see quantizers/nestedkv.py for the
full crux):
  - Unpublished preprint, no verified venue.
  - One-shot prefill compression; cache is NOT bounded during decode (grows
    with every decoded token, same as SnapKV-adapted's decode-phase design).
  - Key-only scoring, no query/attention access at all (not even a proxy).
  - Gate/blend constants (beta=3.0, tau=0.60, kappa=10.0, prior=(0.4,0.4,0.2),
    safeguard_alpha=0.20) taken directly from the paper's Appendix A.
  - **Ragged per-head budgets, zero-padded for stacking.** Every other
    eviction method in this repo (H2O, CurDKV, PyramidKV) uses a UNIFORM
    per-head budget, so every head's kept tensor is always the same length
    and heads stack trivially. NestedKV's cross-head competition (component
    5) can legitimately give different heads different token counts. This
    wrapper zero-pads shorter heads (at the front) up to the layer's max
    kept length purely so ``mx.stack`` can combine heads into one tensor;
    the padding is a tensor-shape accommodation only — byte accounting
    (``nestedkv_kept_bytes``) is computed from each head's true, unpadded
    state, so reported compression numbers are unaffected.

Byte accounting:
    nestedkv_kept_bytes — fp16 bytes for currently retained K + V tokens
    full_seq_bytes      — hypothetical fp16 cost if all tokens were kept
    compression_ratio   — full_seq_bytes / nestedkv_kept_bytes (> 1 = savings)
    tokens_seen         — total token positions ever passed to update_and_fetch
    tokens_kept         — tokens currently in the first (B=0, H=0) head's cache
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.nestedkv import (
    NestedKVState,
    full_nestedkv_fp16_bytes,
    init_nestedkv_state,
    nestedkv_allocate_head_budgets,
    nestedkv_append_decode,
    nestedkv_compress_prefill,
    nestedkv_fp16_bytes,
    nestedkv_get_kv,
    nestedkv_score,
)


class NestedKVKVCache(_MLXKVCache):
    """KV cache implementing NestedKV-adapted multi-scale ensembled eviction.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``nestedkv_budget``    (int, default 512)   — per-head-equivalent
                budget; total layer budget = this * n_heads,
            ``nestedkv_n_sink``    (int, default 4)     — leading sink positions,
            ``nestedkv_window``    (int, default 64)    — current-memory window W,
            ``nestedkv_beta``      (float, default 3.0) — head-adaptive blend temperature,
            ``nestedkv_tau``       (float, default 0.60)— surprise gate threshold,
            ``nestedkv_kappa``     (float, default 10.0)— surprise gate sharpness,
            ``nestedkv_safeguard_alpha`` (float, default 0.20) — per-head budget floor.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Eviction happens ONCE at prefill (S > 1), across all heads jointly
        (cross-head budget competition). Decode tokens (S == 1) are always
        appended, never rescored or evicted — same convention as
        SnapKV-adapted, NOT H2O's/CurDKV's per-step loop.
        Single-layer (no coordinator); ``KVCacheBuilder.for_model()``
        propagates all ``nestedkv_*`` fields automatically via
        ``dataclasses.replace``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "nestedkv_budget", 512))
        self._n_sink = int(getattr(config, "nestedkv_n_sink", 4))
        self._window = int(getattr(config, "nestedkv_window", 64))
        self._beta = float(getattr(config, "nestedkv_beta", 3.0))
        self._tau = float(getattr(config, "nestedkv_tau", 0.60))
        self._kappa = float(getattr(config, "nestedkv_kappa", 10.0))
        self._safeguard_alpha = float(getattr(config, "nestedkv_safeguard_alpha", 0.20))

        self._head_dim: int = 0
        self._states: list[NestedKVState] = []
        self._B: int = 0
        self._H: int = 0

        self._nestedkv_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        """Lazily initialise per-head NestedKVState list on first call."""
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            self._states = [init_nestedkv_state(self._n_sink) for _ in range(B * H)]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    @staticmethod
    def _pad_heads_to_common_length(rows: list[mx.array]) -> mx.array:
        """Stack a list of ``[n_h, D]`` arrays (possibly different ``n_h``)
        into ``[H, max_n, D]`` by zero-padding shorter heads at the front.

        NestedKV's cross-head budget competition (paper Section 2.6) can
        legitimately allocate different token counts to different heads —
        unlike every other eviction method in this repo (H2O, CurDKV,
        PyramidKV), which use a uniform per-head budget and never hit this.
        Padding is a pure tensor-shape accommodation for stacking; the
        library's own byte accounting (``nestedkv_fp16_bytes``) is computed
        from each head's true (unpadded) state, so compression numbers are
        unaffected by this padding.
        """
        max_n = max(int(r.shape[0]) for r in rows)
        D = rows[0].shape[1]
        padded = []
        for r in rows:
            n = int(r.shape[0])
            if n < max_n:
                pad = mx.zeros((max_n - n, D), dtype=r.dtype)
                r = mx.concatenate([pad, r], axis=0)
            padded.append(r)
        return mx.stack(padded, axis=0)

    def _process_prefill(self, keys: mx.array, values: mx.array):
        """One-shot prefill compression: score every head, allocate cross-head
        budgets, evict down to each head's allocated budget."""
        B, H, S, D = keys.shape
        k_out_b, v_out_b = [], []

        for b in range(B):
            head_scores = [
                nestedkv_score(
                    keys[b, h].astype(mx.float32),
                    window=self._window,
                    beta=self._beta,
                    tau=self._tau,
                    kappa=self._kappa,
                )
                for h in range(H)
            ]
            total_budget = self._budget * H
            head_budgets = nestedkv_allocate_head_budgets(
                head_scores, total_budget, safeguard_alpha=self._safeguard_alpha
            )

            k_out_h, v_out_h = [], []
            for h in range(H):
                idx = self._head_idx(b, h)
                st = self._states[idx]
                st = nestedkv_compress_prefill(
                    st,
                    keys[b, h],
                    values[b, h],
                    budget=head_budgets[h],
                    window=self._window,
                    beta=self._beta,
                    tau=self._tau,
                    kappa=self._kappa,
                )
                self._states[idx] = st
                k_h, v_h = nestedkv_get_kv(st)
                k_out_h.append(k_h)
                v_out_h.append(v_h)
            k_out_b.append(self._pad_heads_to_common_length(k_out_h))
            v_out_b.append(self._pad_heads_to_common_length(v_out_h))

        return mx.stack(k_out_b, axis=0), mx.stack(v_out_b, axis=0)

    def _process_decode(self, keys: mx.array, values: mx.array):
        """Plain unscored append for decode tokens — never evicted."""
        B, H, S, D = keys.shape
        k_out_b, v_out_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                idx = self._head_idx(b, h)
                st = self._states[idx]
                st = nestedkv_append_decode(st, keys[b, h], values[b, h])
                self._states[idx] = st
                k_h, v_h = nestedkv_get_kv(st)
                k_out_h.append(k_h)
                v_out_h.append(v_h)
            k_out_b.append(self._pad_heads_to_common_length(k_out_h))
            v_out_b.append(self._pad_heads_to_common_length(v_out_h))
        return mx.stack(k_out_b, axis=0), mx.stack(v_out_b, axis=0)

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens; prefill compresses once, decode always appends.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16 internally).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_kept, D]`` fp16.
        """
        B, H, S, D = keys.shape
        self._ensure_states(B, H, D)

        self._full_seq_bytes += B * H * S * D * 2 * 2  # K + V, fp16
        self._tokens_seen_total += B * H * S

        is_prefill = S > 1 and not self._states[0].compressed
        if is_prefill:
            K_out, V_out = self._process_prefill(keys, values)
        else:
            K_out, V_out = self._process_decode(keys, values)

        self._nestedkv_kept_bytes = sum(nestedkv_fp16_bytes(st) for st in self._states)

        return K_out, V_out

    # ------------------------------------------------------------------
    @property
    def nestedkv_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._nestedkv_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / nestedkv_kept_bytes; > 1 means memory savings over fp16."""
        if self._nestedkv_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._nestedkv_kept_bytes

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


__all__ = ["NestedKVKVCache"]
