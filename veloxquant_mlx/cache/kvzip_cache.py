"""KVzip-adapted KV cache — context-reconstruction reliance retention.

Inspired by "KVzip: Query-Agnostic KV Cache Compression with Context
Reconstruction" (Kim, Kim, Kwon, Lee, Yun, Song, NeurIPS 2025 (Oral),
arXiv:2505.23416, github.com/snu-mllab/KVzip). Documented as "KVzip-adapted
(VeloxQuant-MLX implementation)" — not a faithful port.

Keeps a constant-size cache by ranking stored tokens according to their
**context-reconstruction reliance** (paper's contribution: score a KV pair by
how much the model relies on it to reconstruct its own context, query-agnostic,
then evict the least-relied-upon pairs). Setting ``kvzip_probe = "latest"``
reduces this to a latest-token (TOVA-adapted-style) eviction — the honest
reference behavior, checked by a dedicated test.

Where it sits: the repo's proxy-attention scorer family (SnapKV / H2O / TOVA /
PyramidKV / SqueezeAttention / ChunkKV / CaM / Keyformer / MorphKV). The
distinguishing axis is reconstruction reliance (attention from a fixed
reconstruction probe), versus cumulative (H2O), latest-only (TOVA), or
recent-window (MorphKV) query attention.

THE HONESTY CRUX:
  1. Proxy reconstruction — the stored/incoming KEYS stand in for the unseen
     reconstruction queries (as H2O / TOVA / MorphKV-adapted).
  2. Query-agnostic, recomputed each step from the live keep set against the
     probe — NOT a cumulative accumulator. Only ``probe = "latest"`` collapse is
     pinned exactly (to the latest-token ranking); no H2O collapse is claimed.
  3. Not validated on a trained model; the paper's accuracy/memory numbers are
     the paper's, on trained models — never reproduced or claimed here. The
     mechanism's benefit is measured only under a constructed
     reconstruction-shift geometry, with a null control.

Adaptation limitations (stated plainly):
  - Key-as-reconstruction-probe proxy (crux 1).
  - No RoPE position-ID remapping after eviction.
  - Uniform budget / n_sink / probe across all heads.
  - Leading ``kvzip_n_sink`` tokens protected as sinks; no trailing window is
    force-protected (a token survives only if the reconstruction probe relies
    on it).

Byte accounting (same names as H2OKVCache / MorphKVKVCache):
    kvzip_kept_bytes — fp16 bytes for retained K + V tokens
    full_seq_bytes   — hypothetical fp16 cost if all tokens were kept
    compression_ratio — full_seq_bytes / kvzip_kept_bytes (> 1 = savings)
    tokens_seen      — total token positions ever passed to update_and_fetch
    tokens_kept      — tokens currently in the (B=0, H=0) head's cache
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.cache._deferred_eviction import DeferredEvictionMixin
from veloxquant_mlx.quantizers.kvzip import (
    init_kvzip_state,
    kvzip_update_batched,
)


class KVzipKVCache(DeferredEvictionMixin, _MLXKVCache):
    """KV cache implementing KVzip-adapted reconstruction-reliance retention for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``kvzip_budget`` (int, default 512) — max tokens kept (incl. sinks),
            ``kvzip_n_sink`` (int, default 4)   — leading positions never evicted,
            ``kvzip_probe``  (str, default "context") — reconstruction probe;
                "latest" collapses onto the TOVA-adapted latest-token ranking.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Both prefill (S > 1) and decode (S == 1) go through the same update
        loop. Single-layer (no coordinator); ``KVCacheBuilder.for_model()``
        propagates all ``kvzip_*`` fields via ``dataclasses.replace``. Per-head
        state is lazily initialised on the first ``update_and_fetch``. KVzip is
        deterministic (no RNG). Validation (budget/sink bounds, probe value)
        happens at construction.
        Writes through to the base ``mlx_lm`` ``KVCache``'s ``self.keys`` /
        ``self.values`` / ``self.offset`` on every call so ``.state`` stays
        valid (mlx_lm's ``generate()`` reads it unconditionally during
        chunked prefill); ``is_trimmable()`` reports ``False`` since the
        internal per-token state can't be rolled back by a base-class
        ``trim()`` (see #83).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "kvzip_budget", 512))
        self._n_sink = int(getattr(config, "kvzip_n_sink", 4))
        self._probe = str(getattr(config, "kvzip_probe", "context"))

        # Fail at build time with clear messages (delegates the guards).
        init_kvzip_state(self._n_sink, self._budget, 1, probe=self._probe)

        self._head_dim: int = 0
        self._B: int = 0
        self._H: int = 0
        self._initialised: bool = False

        # Flat [BH, n, D] state — replaces the old per-(b,h) KVzipState
        # list. Batching every head into one call (instead of a Python loop
        # calling kvzip_update once per (b,h) pair) removes the O(B*H)
        # Python-dispatch bottleneck on the decode hot path — same fix,
        # same template, as H2OKVCache's _bh_* state.
        self._bh_keys: mx.array | None = None
        self._bh_values: mx.array | None = None

        self._kvzip_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

        # Stored as a name, not an mx.Dtype, because mlx_lm.server deepcopies
        # cache entries per request and mx.Dtype objects raise TypeError from
        # copy.deepcopy (same convention as SnapKVCache._storage_dtype_name).
        self._storage_dtype_name: str | None = None

    @property
    def _storage_dtype(self) -> mx.Dtype:
        return mx.bfloat16 if self._storage_dtype_name == "bfloat16" else mx.float16

    # ------------------------------------------------------------------
    # `mlx_lm.server`'s `ModelProvider.load()` decides whether to route
    # requests through `BatchGenerator` (continuous batching) purely by
    # `hasattr(c, "merge")` on a probe instance. The base `KVCache` this
    # inherits from defines `merge()` as a classmethod that returns a plain
    # `mlx_lm.models.cache.BatchKVCache`, oblivious to the reconstruction-
    # reliance eviction state this class needs. Left inherited, every
    # request — even a lone one, since `BatchGenerator` merges a batch of 1
    # too, for uniform batch-shape handling — silently replaces this cache
    # with that generic one: no eviction, no sink protection, unlimited
    # growth, while the server believes it is still running `kvzip`. This
    # hides `merge` from `hasattr` instead (a bare classmethod override
    # wouldn't: `hasattr` would still see it as present and callable). That
    # makes `is_batchable` correctly report `False`, routing `kvzip` through
    # `mlx_lm.server`'s sequential `_serve_single` path instead, where this
    # class already runs correctly. See VeloxQuant-MLX#358 for the full
    # 37-method scope of this defect.
    merge = property(
        lambda self: (_ for _ in ()).throw(
            AttributeError(
                "KVzipKVCache does not support batched merging; use it via "
                "the sequential serving path (see class docstring)."
            )
        )
    )

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        if not self._initialised:
            self._B = B
            self._H = H
            self._head_dim = D
            self._initialised = True

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply reconstruction-reliance eviction, return window.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            Full prior kept rows plus all new rows, before eviction (#610).
            Only the state stored for the next call is compressed.
        """
        B, H, S, D = keys.shape
        if self._storage_dtype_name is None:
            self._storage_dtype_name = (
                "bfloat16" if keys.dtype == values.dtype == mx.bfloat16 else "float16"
            )
        self._ensure_states(B, H, D)
        full_k, full_v, positions = self._prepare_attention(
            keys, values, key_dtype=self._storage_dtype, value_dtype=self._storage_dtype
        )

        self._full_seq_bytes += B * H * S * D * 2 * 2  # K + V, fp16-equivalent accounting
        self._tokens_seen_total += B * H * S

        new_keys_flat = keys.astype(self._storage_dtype).reshape(B * H, S, D)
        new_values_flat = values.astype(self._storage_dtype).reshape(B * H, S, D)

        self._bh_keys, self._bh_values, indices = kvzip_update_batched(
            self._bh_keys,
            self._bh_values,
            new_keys_flat,
            new_values_flat,
            self._n_sink,
            self._budget,
            self._probe,
            return_indices=True,
        )

        self._positions = mx.take_along_axis(positions, indices.reshape(B, H, -1), axis=2)
        n_kept = self._bh_keys.shape[1]
        K_out = self._bh_keys.reshape(B, H, n_kept, D)
        V_out = self._bh_values.reshape(B, H, n_kept, D)

        self._kvzip_kept_bytes = B * H * n_kept * D * 2 * 2

        # Only persisted state is evicted; current attention uses every new row.
        self.keys, self.values = K_out, V_out
        self.offset += S
        return full_k, full_v

    # ------------------------------------------------------------------
    def is_trimmable(self) -> bool:
        """False: trim() would only roll back base-class offset bookkeeping,
        not the internal per-token eviction/compression state that actually
        determines what gets returned, silently corrupting future calls.
        """
        return False

    # ------------------------------------------------------------------
    @property
    def kvzip_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._kvzip_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / kvzip_kept_bytes; > 1 means memory savings over fp16."""
        if self._kvzip_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._kvzip_kept_bytes

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


__all__ = ["KVzipKVCache"]
