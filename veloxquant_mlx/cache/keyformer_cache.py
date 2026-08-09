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

THE HONESTY CRUX:
  1. Proxy query — the incoming KEY stands in for the unseen query (as H2O /
     SnapKV-adapted).
  2. Frozen deterministic per-position Gumbel, seeded by a per-head running
     position — NOT the paper's redrawn-and-annealed schedule. Preserves the
     "don't doom a borderline token on one low reading" intent while staying
     reproducible; not claimed equivalent to the paper's annealing.
  3. Not validated on a trained model; the regularizer's benefit is measured
     only under constructed late-riser geometry, with a null control.

Adaptation limitations (stated plainly):
  - Key-as-query proxy (crux 1).
  - Frozen per-position Gumbel, no annealing (crux 2).
  - No RoPE position-ID remapping after eviction.
  - Uniform budget / n_sink / tau across all heads.
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

from veloxquant_mlx.quantizers.keyformer import (
    KeyformerState,
    full_keyformer_fp16_bytes,
    init_keyformer_state,
    keyformer_fp16_bytes,
    keyformer_get_kv,
    keyformer_update,
)


class KeyformerKVCache(_MLXKVCache):
    """KV cache implementing Keyformer-adapted Gumbel-regularized eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``keyformer_budget`` (int, default 512) — max tokens kept (incl. sinks),
            ``keyformer_n_sink`` (int, default 4)   — leading positions never evicted,
            ``keyformer_recent`` (int, default 0)   — trailing protected window (extension),
            ``keyformer_tau`` (float, default 1.0)  — Gumbel temperature; 0 = H2O-adapted,
            ``keyformer_seed`` (int, default 0)     — base seed for the frozen noise.

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
        self._tau = float(getattr(config, "keyformer_tau", 1.0))
        self._seed = int(getattr(config, "keyformer_seed", 0))

        # Fail at build time with clear messages (delegates the guards).
        init_keyformer_state(
            self._n_sink, self._budget, 1, recent=self._recent, tau=self._tau, seed=self._seed
        )

        self._head_dim: int = 0
        self._states: list[KeyformerState] = []
        self._B: int = 0
        self._H: int = 0

        self._keyformer_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        if not self._states:
            self._B = B
            self._H = H
            self._head_dim = D
            # Per-head seed offset keeps heads' frozen noise independent while
            # remaining fully deterministic.
            self._states = [
                init_keyformer_state(
                    self._n_sink,
                    self._budget,
                    D,
                    recent=self._recent,
                    tau=self._tau,
                    seed=self._seed + hh,
                )
                for hh in range(B * H)
            ]

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply Gumbel-regularized eviction, return window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_kept, D]`` fp16, where
            ``n_kept <= keyformer_budget`` for all heads.
        """
        B, H, S, D = keys.shape
        self._ensure_states(B, H, D)

        self._full_seq_bytes += B * H * S * D * 2 * 2  # K + V, fp16
        self._tokens_seen_total += B * H * S

        k_out_b, v_out_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                idx = self._head_idx(b, h)
                st = keyformer_update(
                    self._states[idx],
                    keys[b, h].astype(mx.float16),
                    values[b, h].astype(mx.float16),
                )
                self._states[idx] = st
                k_h, v_h = keyformer_get_kv(st)
                k_out_h.append(k_h)
                v_out_h.append(v_h)
            k_out_b.append(mx.stack(k_out_h, axis=0))
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)
        V_out = mx.stack(v_out_b, axis=0)

        self._keyformer_kept_bytes = sum(keyformer_fp16_bytes(st) for st in self._states)

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
        if not self._states or self._states[0].keys is None:
            return 0
        return int(self._states[0].keys.shape[0])


__all__ = ["KeyformerKVCache"]
