"""L2Norm-adapted KV cache — intrinsic key-norm eviction.

Inspired by "A Simple and Effective L2 Norm-Based Strategy for KV Cache
Compression" (Devoto, Zhao, Scardapane, Minervini — EMNLP 2024;
arXiv:2406.11430). Documented as "L2Norm-adapted (VeloxQuant-MLX
implementation)" — not a faithful port.

The repo's first **intrinsic-signal** eviction cache: token importance is
read directly off the stored key vector's L2 norm — the paper's finding is
that in trained decoder LMs a *low* key norm predicts *high* future
attention, so the cache keeps the lowest-norm tokens and evicts the
highest-norm ones. No attention scores, no key-as-query proxy (the
approximation H2O/SnapKV/TOVA need), no structure-only recency rule
(StreamingLLM): the paper's actual signal is fully observable at the cache
level, making this the cleanest adaptation in the eviction family.

Because the score is intrinsic (computed once at insertion, never updated):
  - eviction vectorizes as one protected top-k per incoming block — no
    per-token softmax-over-cache loop like H2O;
  - with ``knorm_recent=0`` the kept set is **path-independent**: prefill in
    one block and token-by-token decode yield bit-for-bit identical caches
    (the "keep k best with a heap" invariant — see quantizers/knorm.py).

``update_and_fetch`` batches every head's update in one shot via
``knorm_update_batched`` (reshape to ``[B*H, ...]``, one batched argsort,
reshape back) instead of a Python ``for b: for h:`` loop — safe because
budget/n_sink/recent/keep are uniform across all heads in one cache
instance, so every head's state has identical shape at every step (same
precondition RocketKV's decode-summary batching relied on). Measured on M4:
H=8 1.53ms -> 0.76ms/step (2.0x), H=32 4.07ms -> 0.77ms/step (5.3x).

Adaptation limitations (stated plainly):
  - The low-norm ⇒ high-attention correlation is the paper's empirical claim
    about trained models — not validated here on synthetic data (the
    benchmark's isotropic control shows no advantage, honestly reported).
  - No RoPE position-ID *renumbering* after eviction. Surviving tokens keep
    their original absolute positions, so ``self.offset`` reports the true
    token position and RoPE stays correct without re-rotating survivors
    (see ``update_and_fetch`` and :issue:`171`, :issue:`174`). Positions do
    become non-contiguous where tokens were dropped.
  - Uniform budget and n_sink across all heads.
  - ``knorm_recent`` (trailing protected window) is an extension, off by
    default; enabling it breaks the path-independence property.

Byte accounting (same names as H2OKVCache):
    knorm_kept_bytes  — fp16 bytes for currently retained K + V tokens
    full_seq_bytes    — hypothetical fp16 cost if all tokens were kept
    compression_ratio — full_seq_bytes / knorm_kept_bytes (> 1 = savings)
    tokens_seen       — total token positions ever passed to update_and_fetch
    tokens_kept       — tokens currently in the first (B=0, H=0) head's cache
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.cache._deferred_eviction import DeferredEvictionMixin
from veloxquant_mlx.quantizers.knorm import (
    init_knorm_state,
    knorm_update_batched,
)


class L2NormKVCache(DeferredEvictionMixin, _MLXKVCache):
    """KV cache implementing L2Norm-adapted intrinsic key-norm eviction for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``knorm_budget`` (int, default 512) — max tokens retained (incl. sinks),
            ``knorm_n_sink`` (int, default 4)   — leading positions never evicted,
            ``knorm_recent`` (int, default 0)   — trailing protected window (extension),
            ``knorm_keep``  (str, default "low") — "low" = paper finding; "high" = inverted.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly.
        Single-layer (no coordinator); the default ``KVCacheBuilder.for_model()``
        path returns one ``L2NormKVCache`` per attention layer. Per-head state
        is lazily initialised on the first ``update_and_fetch``. Validation
        (keep mode, sink/recent-vs-budget guard) happens at construction.
        Writes through to the base ``mlx_lm`` ``KVCache``'s ``self.keys`` /
        ``self.values`` / ``self.offset`` on every call so ``.state`` stays
        valid (mlx_lm's ``generate()`` reads it unconditionally during
        chunked prefill); ``is_trimmable()`` reports ``False`` since the
        internal per-token state can't be rolled back by a base-class
        ``trim()`` (see #83).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._budget = int(getattr(config, "knorm_budget", 512))
        self._n_sink = int(getattr(config, "knorm_n_sink", 4))
        self._recent = int(getattr(config, "knorm_recent", 0))
        self._keep = str(getattr(config, "knorm_keep", "low"))

        # Fail at build time with clear messages (delegates the guards).
        init_knorm_state(self._n_sink, self._budget, 1, recent=self._recent, keep=self._keep)

        self._head_dim: int = 0
        # Flat batched state across N = B*H heads, shared uniform budget/
        # n_sink/recent/keep — see knorm_update_batched's docstring for why
        # this batches cleanly (RocketKV PR #537 precedent).
        self._keys: mx.array | None = None  # [N, n_kept, D] fp16
        self._values: mx.array | None = None  # [N, n_kept, D] fp16
        self._norms: mx.array | None = None  # [N, n_kept] float32
        self._B: int = 0
        self._H: int = 0

        self._knorm_kept_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_seen_total: int = 0

        # True absolute token position, independent of how many rows survive
        # eviction. Reported as ``self.offset`` so mlx_lm's RoPE stays correct
        # after tokens are dropped (see #171 and update_and_fetch).
        self._true_offset: int = 0

        # Stored as a name, not an mx.Dtype, because mlx_lm.server deepcopies
        # cache entries per request and mx.Dtype objects raise TypeError from
        # copy.deepcopy (same convention as SnapKVCache._storage_dtype_name).
        self._storage_dtype_name: str | None = None

    @property
    def _storage_dtype(self) -> mx.Dtype:
        return mx.bfloat16 if self._storage_dtype_name == "bfloat16" else mx.float16

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int, D: int) -> None:
        if self._keys is None:
            self._B = B
            self._H = H
            self._head_dim = D
            N = B * H
            self._keys = mx.zeros((N, 0, D), dtype=self._storage_dtype)
            self._values = mx.zeros((N, 0, D), dtype=self._storage_dtype)
            self._norms = mx.zeros((N, 0), dtype=mx.float32)

    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Absorb new K/V tokens, apply key-norm eviction, return retained window.

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

        keys_flat = keys.astype(self._storage_dtype).reshape(B * H, S, D)
        values_flat = values.astype(self._storage_dtype).reshape(B * H, S, D)
        self._keys, self._values, self._norms, indices = knorm_update_batched(
            self._keys,
            self._values,
            self._norms,
            keys_flat,
            values_flat,
            self._budget,
            self._n_sink,
            self._recent,
            self._keep,
            return_indices=True,
        )

        self._positions = mx.take_along_axis(positions, indices.reshape(B, H, -1), axis=2)
        n_kept = self._keys.shape[1]
        K_out = self._keys.reshape(B, H, n_kept, D)
        V_out = self._values.reshape(B, H, n_kept, D)

        self._knorm_kept_bytes = B * H * n_kept * D * 2 * 2  # K + V, fp16

        # Only persisted state is evicted; current attention uses every new row.
        self.keys, self.values = K_out, V_out
        self._true_offset += S
        self.offset = self._true_offset
        return full_k, full_v

    # ------------------------------------------------------------------
    def is_trimmable(self) -> bool:
        """False: trim() would only roll back base-class offset bookkeeping,
        not the internal per-token eviction/compression state that actually
        determines what gets returned, silently corrupting future calls.
        """
        return False

    # ------------------------------------------------------------------
    # `mlx_lm.server`'s `ModelProvider.load()` decides whether to route
    # requests through `BatchGenerator` (continuous batching) purely by
    # `hasattr(c, "merge")` on a probe cache — see #15. The base `KVCache`
    # class this inherits from defines `merge()` as a classmethod that
    # returns a plain `mlx_lm.models.cache.BatchKVCache`, oblivious to
    # `_keys`/`_values`/`_norms`/`_true_offset`/the key-norm scoring this class actually
    # needs. Left inherited, every request (even a lone one — a batch of
    # size 1 is still merged for uniform batch-shape handling, see
    # `BatchGenerator.insert_segments` -> `_merge_caches`) silently replaces
    # this cache with that generic one: no budget cap, no eviction, no
    # `tokens_seen`/`tokens_kept`, while the server believes it is still
    # running `knorm`. A real batch-aware merge would need to interleave
    # per-sequence eviction state and per-head budgets across the batch — a
    # much larger undertaking than restoring this method's actual behavior
    # — so this hides `merge` from `hasattr` instead (a bare classmethod
    # override wouldn't: `hasattr` would still see it as present and
    # callable). That makes `is_batchable` correctly report `False`,
    # routing `knorm` through `mlx_lm.server`'s sequential `_serve_single`
    # path instead, where this class already runs correctly. See
    # VeloxQuant-MLX#357 for the same defect in ~10 other eviction/hybrid
    # cache classes that also never override `merge`.
    merge = property(
        lambda self: (_ for _ in ()).throw(
            AttributeError(
                "L2NormKVCache does not support batched merging; use it via "
                "the sequential serving path (see class docstring)."
            )
        )
    )

    # ------------------------------------------------------------------
    @property
    def knorm_kept_bytes(self) -> int:
        """Bytes currently stored across all heads (fp16 K + V, kept tokens only)."""
        return self._knorm_kept_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if all tokens were kept."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / knorm_kept_bytes; > 1 means memory savings over fp16."""
        if self._knorm_kept_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._knorm_kept_bytes

    @property
    def tokens_seen(self) -> int:
        """Total token positions ever passed to update_and_fetch (all heads summed)."""
        return self._tokens_seen_total

    @property
    def tokens_kept(self) -> int:
        """Tokens currently in the (B=0, H=0) head's cache (diagnostic)."""
        if self._keys is None:
            return 0
        return int(self._keys.shape[1])


__all__ = ["L2NormKVCache"]
