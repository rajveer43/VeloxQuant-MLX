"""Keyformer-adapted KV cache — Gumbel-regularized heavy-hitter eviction.

Inspired by "Keyformer: KV Cache Reduction through Key Tokens Selection for
Efficient Generative Inference" (Adnan et al., MLSys 2024, arXiv:2403.09054).
Documented as "Keyformer-adapted (VeloxQuant-MLX implementation)" — not a
faithful port.

Accumulates each token's proxy-attention importance (H2O-adapted's rule) but
adds **Gumbel noise** to the score logits before the keep/evict decision. The
noise is the paper's contribution: it stops a token that reads low early —
before the queries that would attend to it arrive — from being deterministically
pruned and unable to recover ("late risers"). Setting ``keyformer_tau = 0``
removes the noise and this cache collapses exactly onto H2O-adapted behavior —
the honest ablation, checked by a dedicated test and the benchmark.

Where it sits: the repo's proxy-attention scorer family (SnapKV / H2O / TOVA /
PyramidKV / SqueezeAttention / ChunkKV / CaM). Structurally the H2O pair with a
Gumbel-noise regularizer layered on the eviction ranking.

FIXED, CONFIRMED-BY-PAPER-COMPARISON GAP #1 — no temperature annealing. The
paper's Section 3.3.1/Equation 10 anneals the Gumbel temperature from
``tau_init`` (paper default 1, prompt phase) to ``tau_end`` (paper default 2,
as decoding discards more tokens) over ``anneal_steps`` update steps. A single
constant ``keyformer_tau`` cannot represent this schedule. Fixed via
``keyformer_tau_init`` / ``keyformer_tau_end`` / ``keyformer_anneal_steps`` —
see ``quantizers/keyformer.py``'s module docstring. ``keyformer_tau`` is kept
as a backward-compatible alias for a constant (non-annealed) temperature.

FIXED, CONFIRMED-BY-PAPER-COMPARISON GAP #2 — no RoPE position tracking. This
cache never tracked which absolute position each kept key was rotated at, so
an interior eviction silently desynced survivors' storage index from the
rotation baked into their keys — the same bug class H2O-adapted had and fixed
(see ``cache/h2o_cache.py``). Fixed the same way: positions are now tracked
and re-rotated on eviction via ``rope_remap_positions``. See
``keyformer_rope_base`` below.

THE HONESTY CRUX:
  1. Proxy query — the incoming KEY stands in for the unseen query (as H2O /
     SnapKV-adapted).
  2. Frozen deterministic per-position Gumbel, seeded by a per-head running
     position — NOT the paper's redrawn-every-step sampling. Preserves the
     "don't doom a borderline token on one low reading" intent while staying
     reproducible; not claimed equivalent to the paper's redraw.
  3. Not validated on a trained model; the regularizer's benefit is measured
     only under constructed late-riser geometry, with a null control.

Adaptation limitations (stated plainly):
  - Key-as-query proxy (crux 1).
  - Frozen per-position Gumbel, not redrawn each step (crux 2).
  - Uniform budget / n_sink / tau schedule across all heads.
  - ``keyformer_recent`` (trailing protected window) is an extension, off by
    default.

Byte accounting (same names as H2OKVCache):
    keyformer_kept_bytes — fp16 bytes for retained K + V tokens
    full_seq_bytes       — hypothetical fp16 cost if all tokens were kept
    compression_ratio    — full_seq_bytes / keyformer_kept_bytes (> 1 = savings)
    tokens_seen          — total token positions ever passed to update_and_fetch
    tokens_kept          — tokens currently in the (B=0, H=0) head's cache
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.cache._eviction_mask import eviction_make_mask
from veloxquant_mlx.quantizers.keyformer import (
    init_keyformer_state,
    keyformer_update_batched,
)


class KeyformerKVCache(_MLXKVCache):
    """KV cache implementing Keyformer-adapted Gumbel-regularized eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``keyformer_budget`` (int, default 512)      — max tokens kept (incl. sinks),
            ``keyformer_n_sink`` (int, default 4)        — leading positions never evicted,
            ``keyformer_recent`` (int, default 0)        — trailing protected window (extension),
            ``keyformer_tau`` (float, default 1.0)       — constant-temperature alias;
                sets both ``tau_init``/``tau_end`` and disables annealing when set,
            ``keyformer_tau_init`` (float, default 1.0)  — Gumbel temperature at
                ``pos == 0`` (paper default 1); ignored if ``keyformer_tau`` is set,
            ``keyformer_tau_end`` (float, default 1.0)   — Gumbel temperature once
                annealing completes (paper default 2); ignored if ``keyformer_tau`` is set,
            ``keyformer_anneal_steps`` (int, default 0)  — steps to ramp ``tau_init`` ->
                ``tau_end`` over (Equation 10); 0 = constant temperature,
            ``keyformer_rope_base`` (float, default 10000.0) — RoPE base for
                post-eviction position remap; must match the model's own,
            ``keyformer_seed`` (int, default 0)          — base seed for the frozen noise.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Both prefill (S > 1) and decode (S == 1) go through the same update
        loop. Single-layer (no coordinator); ``KVCacheBuilder.for_model()``
        propagates all ``keyformer_*`` fields via ``dataclasses.replace``. The
        per-head state is lazily initialised on the first ``update_and_fetch``.
        Validation (tau >= 0, sink/recent-vs-budget) happens at construction.
        Writes through to the base ``mlx_lm`` ``KVCache``'s ``self.keys`` /
        ``self.values`` / ``self.offset`` on every call so ``.state`` stays
        valid (mlx_lm's ``generate()`` reads it unconditionally during
        chunked prefill); ``is_trimmable()`` reports ``False`` since the
        internal per-token state can't be rolled back by a base-class
        ``trim()`` (see #83).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "keyformer_budget", 512))
        self._n_sink = int(getattr(config, "keyformer_n_sink", 4))
        self._recent = int(getattr(config, "keyformer_recent", 0))
        _tau_const = getattr(config, "keyformer_tau", None)
        self._tau_init = float(getattr(config, "keyformer_tau_init", 1.0))
        self._tau_end = float(getattr(config, "keyformer_tau_end", 1.0))
        if _tau_const is not None:
            self._tau_init = float(_tau_const)
            self._tau_end = float(_tau_const)
        self._anneal_steps = int(getattr(config, "keyformer_anneal_steps", 0))
        self._rope_base = float(getattr(config, "keyformer_rope_base", 10000.0))
        self._seed = int(getattr(config, "keyformer_seed", 0))

        # Fail at build time with clear messages (delegates the guards).
        init_keyformer_state(
            self._n_sink,
            self._budget,
            1,
            recent=self._recent,
            tau_init=self._tau_init,
            tau_end=self._tau_end,
            anneal_steps=self._anneal_steps,
            rope_base=self._rope_base,
            seed=self._seed,
        )

        self._head_dim: int = 0
        self._B: int = 0
        self._H: int = 0
        self._initialised: bool = False

        # Flat [BH, n, D] / [BH, n] state — replaces the old per-(b,h)
        # KeyformerState list. Batching every head into one call (instead of
        # a Python loop calling keyformer_update once per (b,h) pair)
        # removes the O(B*H) Python-dispatch bottleneck on the decode hot
        # path — same fix, same template, as H2OKVCache's _bh_* state (see
        # h2o_cache.py); Keyformer is documented as H2O-adapted's
        # "Metal-fused sibling". Only the Gumbel noise stream differs per
        # row (self._seeds[hh] = self._seed + hh), threaded through
        # keyformer_update_batched's `seeds` argument.
        self._bh_keys: mx.array | None = None
        self._bh_values: mx.array | None = None
        self._bh_scores: mx.array | None = None
        self._bh_gumbel: mx.array | None = None
        self._bh_positions: mx.array | None = None
        self._seeds: list[int] = []
        self._next_pos: int = 0
        self._pos: int = 0

        self._keyformer_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

        # Chronological positions for masking, independent of RoPE renumbering.
        self._kept_positions: mx.array | None = None

        # Stored as a name, not an mx.Dtype, because mlx_lm.server deepcopies
        # cache entries per request and mx.Dtype objects raise TypeError from
        # copy.deepcopy (same convention as SnapKVCache._storage_dtype_name).
        self._storage_dtype_name: str | None = None

    @property
    def _storage_dtype(self) -> mx.Dtype:
        return mx.bfloat16 if self._storage_dtype_name == "bfloat16" else mx.float16

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        if not self._initialised:
            self._B = B
            self._H = H
            self._head_dim = D
            # Per-head seed offset keeps heads' frozen noise independent
            # while remaining fully deterministic — validated once here via
            # init_keyformer_state's guards (see __init__), one call per
            # head is unnecessary since every head shares budget/n_sink/tau.
            self._seeds = [self._seed + hh for hh in range(B * H)]
            self._initialised = True

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply Gumbel-regularized eviction, return window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            Prior retained rows plus all new rows, before eviction. mlx_lm
            builds the current attention mask before this call (#610), so
            only the state stored for the next call may shrink.
        """
        B, H, S, D = keys.shape
        self._ensure_states(B, H, D)
        if self._storage_dtype_name is None:
            self._storage_dtype_name = (
                "bfloat16" if keys.dtype == values.dtype == mx.bfloat16 else "float16"
            )

        self._full_seq_bytes += B * H * S * D * 2 * 2  # K + V, fp16-equivalent accounting
        self._tokens_seen_total += B * H * S

        new_keys_flat = keys.astype(self._storage_dtype).reshape(B * H, S, D)
        new_values_flat = values.astype(self._storage_dtype).reshape(B * H, S, D)

        # Preserve current attention inputs before updating retained state.
        prev_keys_flat = self._bh_keys
        prev_values_flat = self._bh_values
        new_positions = mx.broadcast_to(
            mx.arange(self._pos, self._pos + S, dtype=mx.int32)[None, None], (B, H, S)
        )
        positions = (
            new_positions
            if self._kept_positions is None
            else mx.concatenate([self._kept_positions, new_positions], axis=2)
        )

        (
            self._bh_keys,
            self._bh_values,
            self._bh_scores,
            self._bh_gumbel,
            self._bh_positions,
            self._next_pos,
            self._pos,
            indices,
        ) = keyformer_update_batched(
            self._bh_keys,
            self._bh_values,
            self._bh_scores,
            self._bh_gumbel,
            self._bh_positions,
            new_keys_flat,
            new_values_flat,
            self._n_sink,
            self._budget,
            self._recent,
            self._tau_init,
            self._tau_end,
            self._anneal_steps,
            self._rope_base,
            self._next_pos,
            self._pos,
            self._seeds,
            return_indices=True,
        )

        n_kept = self._bh_keys.shape[1]
        K_out = self._bh_keys.reshape(B, H, n_kept, D)
        V_out = self._bh_values.reshape(B, H, n_kept, D)

        self._keyformer_kept_bytes = B * H * n_kept * D * 2 * 2

        # Preserve Keyformer's existing RoPE offset contract.
        self.keys = K_out
        self.values = V_out
        self.offset = n_kept

        # Track chronology separately from the quantizer's shifted RoPE positions.
        self._kept_positions = mx.take_along_axis(positions, indices.reshape(B, H, n_kept), axis=2)

        if prev_keys_flat is None:
            return keys.astype(self._storage_dtype), values.astype(self._storage_dtype)
        full_keys_flat = mx.concatenate([prev_keys_flat, new_keys_flat], axis=1)
        full_values_flat = mx.concatenate([prev_values_flat, new_values_flat], axis=1)
        n_full = full_keys_flat.shape[1]
        return full_keys_flat.reshape(B, H, n_full, D), full_values_flat.reshape(B, H, n_full, D)

    # ------------------------------------------------------------------
    def make_mask(self, N: int, return_array: bool = False, window_size: int | None = None, **_):
        """Mask using chronological positions, independent of RoPE renumbering."""
        if self._kept_positions is None:
            return super().make_mask(N, return_array=return_array, window_size=window_size)
        B = self._kept_positions.shape[0]
        prev_positions = self._kept_positions[:, 0]
        new_positions = mx.arange(self._pos, self._pos + N, dtype=mx.int32)
        new_positions = mx.broadcast_to(new_positions[None, :], (B, N))
        key_positions = mx.concatenate([prev_positions, new_positions], axis=1)
        query_positions = new_positions
        return eviction_make_mask(
            query_positions, key_positions, N, return_array=return_array, window_size=window_size
        )

    # ------------------------------------------------------------------
    def is_trimmable(self) -> bool:
        """False: trim() would only roll back base-class offset bookkeeping,
        not the internal per-token eviction/compression state that actually
        determines what gets returned, silently corrupting future calls.
        """
        return False

    # ------------------------------------------------------------------
    @property
    def keyformer_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._keyformer_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / keyformer_kept_bytes; > 1 means memory savings over fp16."""
        if self._keyformer_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._keyformer_kept_bytes

    @property
    def tokens_seen(self) -> int:
        """Total token positions ever passed to update_and_fetch (all heads summed)."""
        return self._tokens_seen_total

    @property
    def tokens_kept(self) -> int:
        """Tokens currently in the (B=0, H=0) head's cache (diagnostic)."""
        if self._bh_keys is None:
            return 0
        return int(self._bh_keys.shape[1])


__all__ = ["KeyformerKVCache"]
