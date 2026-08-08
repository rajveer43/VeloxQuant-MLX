"""SqueezeAttention-adapted KV cache — 2D layer×token budget attention-mass eviction.

Inspired by "SqueezeAttention: 2D Management of KV-Cache in LLM Inference via
Layer-wise Optimal Budget" (Wang et al., 2024, arXiv:2404.04793). Documented as
"SqueezeAttention-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

SqueezeAttention is H2O's cumulative-attention-mass eviction with a **data-driven
per-layer budget**. Where PyramidKV assumes a fixed positional taper resolved at
build time, SqueezeAttention *measures* each layer's attention concentration from
the prefill keys and reallocates a fixed total budget toward broad layers and
away from concentrated ones — a 2D (layer × token) budget grid.

The re-budget is **one-shot at the prefill boundary**, mediated by a shared
``SqueezeCoordinator`` (injected at ``KVCacheBuilder.for_model`` build time):
  1. On the first ``update_and_fetch`` (the prompt), the layer computes its
     concentration score over the incoming keys and reports it to the coordinator.
  2. Once every attention layer has reported, the coordinator computes the
     per-layer budget schedule (``squeeze_budgets``) and publishes it.
  3. The layer pulls its resolved budget and stamps it onto every head's eviction
     state; decode steps then run against the frozen budget.

Until the coordinator finalises (or when constructed with no coordinator, e.g.
``KVCacheFactory.create``), the layer uses ``squeeze_budget`` (the average) as a
fallback and behaves as one uniform-budget H2O layer. The 2D reallocation only
takes effect via ``for_model``.

This is the sixth distinct eviction configuration in VeloxQuant-MLX:
  - SnapKV-adapted     : score-based, once at prefill end.
  - StreamingLLM-adapted : positional (recency + sink), every step.
  - H2O-adapted        : cumulative attention mass, uniform budget, every step.
  - TOVA-adapted       : current-step attention weight (memoryless), every step.
  - PyramidKV-adapted  : H2O scoring with a fixed per-layer pyramid budget.
  - SqueezeAttention-adapted : H2O scoring with a DATA-DRIVEN per-layer budget
    (measured concentration, one-shot re-budget at prefill).

Adaptation limitations (stated plainly):
  - Key-as-query proxy for both concentration and eviction (same as H2O-adapted /
    PyramidKV-adapted).
  - Cosine-dispersion proxy for attention entropy (paper uses actual attention
    maps, not visible at cache level).
  - One-shot re-budget at the prefill boundary; frozen for decode.
  - No RoPE position-ID remapping after eviction.
  - Uniform budget across heads within a layer.

Byte accounting:
    squeeze_kept_bytes — fp16 bytes for currently retained K + V tokens
    full_seq_bytes     — hypothetical fp16 cost if all tokens were kept
    compression_ratio  — full_seq_bytes / squeeze_kept_bytes (> 1 = savings)
    tokens_seen        — total token positions ever passed to update_and_fetch
    tokens_kept        — tokens currently in the first (B=0, H=0) head's cache
    layer_budget       — this layer's resolved (or fallback) budget (diagnostic)
    concentration      — this layer's measured concentration score (diagnostic)
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.cache.squeeze_coordinator import SqueezeCoordinator
from veloxquant_mlx.quantizers.squeeze import (
    SqueezeState,
    concentration_score,
    init_squeeze_state,
    squeeze_fp16_bytes,
    squeeze_get_kv,
    squeeze_update,
)


class SqueezeAttentionCache(_MLXKVCache):
    """KV cache implementing SqueezeAttention-adapted 2D eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``squeeze_budget`` (int, default 512) — average/fallback budget.
            ``squeeze_n_sink`` (int, default 4)   — leading positions never evicted.
            ``squeeze_resolved_budget`` (int or None) — an explicit budget override
                (mainly for single-cache / testing); normally None and the
                coordinator supplies the budget after prefill.
        layer_id: This layer's index (used to report/query the coordinator). None
            for single-cache construction (behaves as uniform H2O).
        coordinator: Shared :class:`SqueezeCoordinator`, or None. When present, the
            layer reports concentration at prefill and pulls a resolved budget.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly. Per-head
        state is lazily initialised on the first call. The one-shot re-budget
        happens on the first ``update_and_fetch`` once the coordinator finalises.
        Writes through to the base ``mlx_lm`` ``KVCache``'s ``self.keys`` /
        ``self.values`` / ``self.offset`` on every call so ``.state`` stays
        valid (mlx_lm's ``generate()`` reads it unconditionally during
        chunked prefill); ``is_trimmable()`` reports ``False`` since the
        internal per-token state can't be rolled back by a base-class
        ``trim()`` (see #83).
    """

    def __init__(
        self,
        config: Any,
        layer_id: Optional[int] = None,
        coordinator: Optional[SqueezeCoordinator] = None,
    ) -> None:
        super().__init__()
        self._avg_budget = int(getattr(config, "squeeze_budget", 512))
        override = getattr(config, "squeeze_resolved_budget", None)
        self._budget = int(override) if override is not None else self._avg_budget
        self._n_sink = int(getattr(config, "squeeze_n_sink", 4))

        self._layer_id = layer_id
        self._coordinator = coordinator
        self._rebudgeted = False  # has the resolved budget been applied?
        self._reported = False  # has concentration been reported?
        self._concentration: float = 0.0

        self._head_dim: int = 0
        self._states: list[SqueezeState] = []
        self._B: int = 0
        self._H: int = 0

        self._squeeze_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        """Lazily initialise per-head SqueezeState list on first call."""
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            self._states = [init_squeeze_state(self._n_sink, self._budget, D) for _ in range(B * H)]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    def _apply_budget(self, budget: int) -> None:
        """Re-stamp ``budget`` onto every head's state, trimming any over-budget head.

        The coordinator can only finalise the schedule after *every* layer has
        reported, so a layer's prefill eviction may have run against the average
        fallback budget. When the resolved budget is smaller, each head is trimmed
        to it here by dropping the lowest-cumulative-score non-sink tokens (the
        same H2O ranking used during streaming), so the final state honours the
        data-driven budget exactly.
        """
        self._budget = int(budget)
        for idx, st in enumerate(self._states):
            self._states[idx] = self._trim_state(st, self._budget)

    @staticmethod
    def _trim_state(st: SqueezeState, budget: int) -> SqueezeState:
        """Return ``st`` re-stamped with ``budget``, trimmed to it if over budget.

        Drops the lowest-score non-sink tokens (sinks always kept) until the head
        holds at most ``budget`` tokens. A no-op when the head is already within
        budget or empty.
        """
        if st.keys is None:
            return SqueezeState(None, None, None, st.n_sink, budget)

        n = st.keys.shape[0]
        if n <= budget:
            return SqueezeState(st.keys, st.values, st.scores, st.n_sink, budget)

        # Protect sinks with +inf, then keep the top-`budget` by score.
        n_sink_eff = min(st.n_sink, n)
        if n_sink_eff > 0:
            inf_block = mx.full((n_sink_eff,), float("inf"), dtype=mx.float32)
            protected = mx.concatenate([inf_block, st.scores[n_sink_eff:]], axis=0)
        else:
            protected = st.scores

        order = mx.argsort(protected)  # ascending: lowest score first
        keep = order[n - budget :]  # top-`budget` scores
        keep_sorted = mx.sort(keep)  # preserve positional order
        keep_idx = [int(x.item()) for x in keep_sorted]
        return SqueezeState(
            keys=st.keys[keep_idx],
            values=st.values[keep_idx],
            scores=st.scores[keep_idx],
            n_sink=st.n_sink,
            budget=budget,
        )

    def _report_and_rebudget(self, keys: mx.array) -> None:
        """Report prefill concentration and pull a resolved budget if ready.

        Called once, on the first ``update_and_fetch``. Measures concentration on
        the (B=0, H=0) head's incoming keys as a representative sample, reports it
        to the coordinator, and — if the coordinator has finalised — applies this
        layer's resolved budget to all head states.
        """
        if self._coordinator is None or self._layer_id is None:
            self._reported = True
            self._rebudgeted = True
            return

        if not self._reported:
            self._concentration = concentration_score(keys[0, 0])
            self._coordinator.report_concentration(self._layer_id, self._concentration)
            self._reported = True

        if not self._rebudgeted:
            resolved = self._coordinator.resolved_budget(self._layer_id)
            if resolved is not None:
                self._apply_budget(resolved)
                self._rebudgeted = True

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply SqueezeAttention eviction, return retained window.

        On the first call the layer reports its concentration and (once the
        coordinator finalises) adopts its resolved budget before eviction runs.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_kept, D]`` fp16, where
            ``n_kept <= layer_budget`` for all heads.
        """
        B, H, S, D = keys.shape
        self._ensure_states(B, H, D)

        # One-shot re-budget at the prefill boundary (before eviction).
        if not self._rebudgeted:
            self._report_and_rebudget(keys)

        self._full_seq_bytes += B * H * S * D * 2 * 2  # K + V, fp16
        self._tokens_seen_total += B * H * S

        k_out_b, v_out_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                idx = self._head_idx(b, h)
                st = self._states[idx]
                st = squeeze_update(
                    st,
                    keys[b, h].astype(mx.float16),
                    values[b, h].astype(mx.float16),
                )
                self._states[idx] = st
                k_h, v_h = squeeze_get_kv(st)
                k_out_h.append(k_h)  # [n_kept, D]
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))  # [H, n_kept, D]
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)  # [B, H, n_kept, D]
        V_out = mx.stack(v_out_b, axis=0)

        # Byte accounting: sum across all head states.
        self._squeeze_kept_bytes = sum(squeeze_fp16_bytes(st) for st in self._states)

        # K_out/V_out is the full retained state every call, not a delta —
        # reset so the base class's append-only buffer starts fresh instead
        # of stacking on top of the previous call's rows. Without this,
        # self.keys/self.values/self.offset stay at __init__ defaults
        # forever, and mlx_lm's generate() crashes on `cache.state` during
        # chunked prefill (see #83).
        self.keys = None
        self.values = None
        self.offset = 0
        return super().update_and_fetch(K_out, V_out)

    # ------------------------------------------------------------------
    def is_trimmable(self) -> bool:
        """False: trim() would only roll back base-class offset bookkeeping,
        not the internal per-token eviction/compression state that actually
        determines what gets returned, silently corrupting future calls.
        """
        return False

    # ------------------------------------------------------------------
    @property
    def layer_budget(self) -> int:
        """This layer's resolved (or fallback) per-layer budget."""
        return self._budget

    @property
    def concentration(self) -> float:
        """This layer's measured concentration score (0.0 before prefill)."""
        return self._concentration

    @property
    def squeeze_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._squeeze_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / squeeze_kept_bytes; > 1 means memory savings over fp16."""
        if self._squeeze_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._squeeze_kept_bytes

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


__all__ = ["SqueezeAttentionCache"]
