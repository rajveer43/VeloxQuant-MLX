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
blend and a per-token surprise gate (see ``quantizers/nestedkv.py``). Each
head then independently keeps its own top-``nestedkv_budget`` scoring tokens
— UNIFORM per-head budget, like H2O/CurDKV (see "Uniform per-head budget"
below, issue #21, for why this class does not use the paper's cross-head
budget competition even though ``quantizers/nestedkv.py`` still implements
it as a tested primitive). Decode tokens are appended unscored, never
evicted — this is a one-shot prefill compressor, not a per-step recurring
eviction loop.

This is the 15th eviction-family method in VeloxQuant-MLX and the first that
ensembles multiple independent importance signals rather than committing to
one:
  - H2O / CurDKV / KVzip / Keyformer / MorphKV : one signal, scored every step.
  - SnapKV : one signal (obs-window attention), scored once at prefill.
  - NestedKV : THREE signals, scored once at prefill, combined by a
    head-adaptive blend + per-token surprise gate.

Adaptation limitations (stated plainly — see quantizers/nestedkv.py for the
full crux):
  - Unpublished preprint, no verified venue.
  - One-shot prefill compression; cache is NOT bounded during decode (grows
    with every decoded token, same as SnapKV-adapted's decode-phase design).
  - Key-only scoring, no query/attention access at all (not even a proxy).
  - Gate/blend constants (beta=3.0, tau=0.60, kappa=10.0, prior=(0.4,0.4,0.2),
    safeguard_alpha=0.20) taken directly from the paper's Appendix A.
  - **Uniform per-head budget (issue #21).** The paper's cross-head budget
    competition (component 5, ``nestedkv_allocate_head_budgets``) can
    legitimately give different heads different token counts. An earlier
    version of this wrapper let each head keep its ragged, cross-head-
    competed count and zero-padded shorter heads (at the front) to stack
    them into one tensor. That was a real correctness bug, not just a
    shape accommodation: the padded rows became real (unmasked) cache
    entries that the downstream attention computation attended to — and
    ``mlx_lm``'s attention forward computes ONE shared mask from the
    model's first layer's cache and reuses it for every layer, so a
    per-cache mask override cannot correctly express each NestedKV layer's
    independent padding pattern. Rather than build mask machinery this
    architecture cannot actually support end-to-end, this class now keeps
    every head's kept length uniform at ``nestedkv_budget`` — same
    convention as every other eviction method here (H2O, CurDKV,
    PyramidKV). ``nestedkv_score`` (the cross-scale anomaly ranking) still
    runs per head exactly as before; only the cross-head *reallocation of
    how many tokens each head keeps* is dropped, in favor of each head
    independently keeping its own top-``nestedkv_budget`` scoring tokens
    (sinks always included) — the same per-head-independent budget model
    H2O and CurDKV already use. No padding is ever needed, so there is
    nothing left to mask.

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
    init_nestedkv_state,
    nestedkv_append_decode,
    nestedkv_compress_prefill,
    nestedkv_fp16_bytes,
    nestedkv_get_kv,
)


class NestedKVKVCache(_MLXKVCache):
    """KV cache implementing NestedKV-adapted multi-scale ensembled eviction.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``nestedkv_budget``    (int, default 512)   — per-head budget
                (uniform across heads, see #21 — NOT per-head-equivalent
                summed into a cross-head-competed total as in the paper),
            ``nestedkv_n_sink``    (int, default 4)     — leading sink positions,
            ``nestedkv_window``    (int, default 64)    — current-memory window W,
            ``nestedkv_beta``      (float, default 3.0) — head-adaptive blend temperature,
            ``nestedkv_tau``       (float, default 0.60)— surprise gate threshold,
            ``nestedkv_kappa``     (float, default 10.0)— surprise gate sharpness,
            ``nestedkv_safeguard_alpha`` (float, default 0.20) — parsed and
                stored for API stability but no longer consumed by this
                class (see #21): it configured the cross-head budget floor
                in ``nestedkv_allocate_head_budgets``, which this class no
                longer calls. Still a real, tested parameter of that
                quantizer-level function for direct callers.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Eviction happens ONCE at prefill (S > 1), independently per head at
        a UNIFORM budget (see #21 — NOT the paper's cross-head budget
        competition, which produced ragged per-head lengths that could not
        be safely stacked into one tensor without either corrupting
        attention with fake padding rows or requiring per-layer attention
        masks this architecture doesn't support). Decode tokens (S == 1)
        are always appended, never rescored or evicted — same convention as
        SnapKV-adapted, NOT H2O's/CurDKV's per-step loop.
        Single-layer (no coordinator); ``KVCacheBuilder.for_model()``
        propagates all ``nestedkv_*`` fields automatically via
        ``dataclasses.replace``.
        Writes through to the base ``mlx_lm`` ``KVCache``'s ``self.keys`` /
        ``self.values`` / ``self.offset`` on every call so ``.state`` stays
        valid (mlx_lm's ``generate()`` reads it unconditionally during
        chunked prefill); ``is_trimmable()`` reports ``False`` since the
        internal per-token state can't be rolled back by a base-class
        ``trim()`` (see #83).
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
    # `mlx_lm.server`'s `ModelProvider.load()` decides whether to route
    # requests through `BatchGenerator` (continuous batching) purely by
    # `hasattr(c, "merge")` on a probe instance. The base `KVCache` this
    # inherits from defines `merge()` as a classmethod that returns a plain
    # `mlx_lm.models.cache.BatchKVCache`, oblivious to the multi-scale
    # ensembled eviction state this class needs. Left inherited, every
    # request — even a lone one, since `BatchGenerator` merges a batch of 1
    # too, for uniform batch-shape handling — silently replaces this cache
    # with that generic one: no eviction, no sink protection, unlimited
    # growth, while the server believes it is still running `nestedkv`.
    # This hides `merge` from `hasattr` instead (a bare classmethod
    # override wouldn't: `hasattr` would still see it as present and
    # callable). That makes `is_batchable` correctly report `False`,
    # routing `nestedkv` through `mlx_lm.server`'s sequential
    # `_serve_single` path instead, where this class already runs
    # correctly. See VeloxQuant-MLX#358 for the full 37-method scope of
    # this defect.
    merge = property(
        lambda self: (_ for _ in ()).throw(
            AttributeError(
                "NestedKVKVCache does not support batched merging; use it via "
                "the sequential serving path (see class docstring)."
            )
        )
    )

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
    def _process_prefill(self, keys: mx.array, values: mx.array):
        """One-shot prefill compression: score every head, evict down to a
        uniform per-head budget (see #21 and the module docstring for why
        this is uniform rather than the paper's cross-head-competed split).
        """
        B, H, S, D = keys.shape
        k_out_b, v_out_b = [], []

        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                idx = self._head_idx(b, h)
                st = self._states[idx]
                st = nestedkv_compress_prefill(
                    st,
                    keys[b, h],
                    values[b, h],
                    budget=self._budget,
                    window=self._window,
                    beta=self._beta,
                    tau=self._tau,
                    kappa=self._kappa,
                )
                self._states[idx] = st
                k_h, v_h = nestedkv_get_kv(st)
                k_out_h.append(k_h)
                v_out_h.append(v_h)
            # Every head kept exactly the same length: nestedkv_compress_prefill's
            # budget_eff = max(n_sink_eff, min(budget, S)) depends only on
            # budget (now uniform) and S (identical across heads in one
            # call), so this stacks safely with no padding.
            k_out_b.append(mx.stack(k_out_h, axis=0))
            v_out_b.append(mx.stack(v_out_h, axis=0))

        return mx.stack(k_out_b, axis=0), mx.stack(v_out_b, axis=0)

    def _process_decode(self, keys: mx.array, values: mx.array):
        """Plain unscored append for decode tokens — never evicted.

        Every head entered decode at the same uniform prefill length (#21)
        and grows by the same S every call, so heads stay uniform-length
        here too — no padding needed.
        """
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
            k_out_b.append(mx.stack(k_out_h, axis=0))
            v_out_b.append(mx.stack(v_out_h, axis=0))
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

        # K_out/V_out is the full retained state every call (uniform length
        # across heads, see #21), not a delta — reset so the base class's
        # append-only buffer starts fresh instead of stacking on top of the
        # previous call's rows. Without this, self.keys/self.values/self.offset
        # stay at __init__ defaults forever, and mlx_lm's generate() crashes
        # on `cache.state` during chunked prefill (see #83).
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
