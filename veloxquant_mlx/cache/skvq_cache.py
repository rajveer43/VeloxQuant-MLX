"""SKVQ KV cache wrapper for mlx_lm integration.

Implements "SKVQ: Sliding-window Key and Value Cache Quantization for Large
Language Models" (Duanmu, Yuan, Li, Duan, Zhang, Lin — COLM 2024;
arXiv:2405.06219) on top of the standard mlx_lm ``update_and_fetch``
protocol. Documented as **"SKVQ-adapted (VeloxQuant-MLX implementation)"** —
not a faithful port (no offline calibration, no weight-fused permutation,
no 1.5-bit values, no FP8 metadata; see the module docstring of
``veloxquant_mlx/quantizers/skvq.py`` and ``paper/NEW_METHOD_SURVEY_V13.md``).

Three mechanisms compose:

  1. **Sliding fp16 window** — the NSNQuant chunk-flush idiom: tokens
     accumulate at fp16; every time ``skvq_window`` tokens age past the
     quantized frontier, that chunk is round-tripped through
     reorder → clipped group quant → dequant → inverse reorder **once and
     frozen**. The frontier only advances in whole chunks, so prefill and
     token-by-token decode produce identical quantized state by
     construction (path independence, pinned by test).
  2. **Channel reordering** — per-head permutations for K and V computed
     from the **first flushed chunk** (the paper computes them offline from
     a calibration corpus and fuses them into projection weights — our
     documented deviation), then frozen for the cache's lifetime.
     ``skvq_reorder=False`` is the identity-permutation ablation.
  3. **Clipped dynamic quantization** — per-token, per-group asymmetric
     min/max quant whose window is shrunk by a per-group grid-searched
     clip factor α (``skvq_clip_search=True``) or a fixed
     ``skvq_clip_alpha``. Quant groups run along the channel axis for both
     K and V; reordering is what makes that viable for keys.

The paper's attention-sink filter is kept: the first ``skvq_n_sink`` tokens
are restored to fp16 after chunk 0's flush (and accounted as fp16, not
compressed).

Like every method in this repo, the quantize→dequantize round-trip happens
inside ``update_and_fetch`` so the downstream SDPA call sees standard fp16
tensors. The paper's throughput gains come from fused CUDA kernels that do
not port to Metal — on Apple Silicon the win is *memory*, measured honestly
by the byte accounting below. Fully deterministic: no RNG anywhere.

Per-token storage at bits ``b``, head_dim D, group size g (per tensor):
``D * b / 8`` bytes of codes + ``ceil(D/g) * 2 * 2`` bytes fp16 (lo, scale)
per token. The searched α adds nothing (folded into lo/scale); the frozen
permutations add ``2 * H * D * 4`` bytes per layer total, counted once in
``perm_bytes``.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.skvq import (
    DEFAULT_ALPHA_GRID,
    clipped_group_dequant,
    clipped_group_quant,
    skvq_compressed_bytes,
    skvq_fp16_bytes,
)


class SKVQKVCache(_MLXKVCache):
    """KV cache implementing SKVQ sliding-window quantization for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed: ``head_dim``,
            ``skvq_bits_key``, ``skvq_bits_value``, ``skvq_group_size``,
            ``skvq_window`` (fp16 sliding window == flush chunk size),
            ``skvq_n_sink`` (leading tokens restored to fp16, the paper's
            filter), ``skvq_reorder``, ``skvq_clip_search``,
            ``skvq_clip_alpha`` (used when search is off), ``skvq_max_ctx``.

    Notes:
        Never exposes ``.bits`` — mlx_lm's SDPA checks
        ``hasattr(cache, "bits")`` to route to a quantized kernel path.
        We expose ``.assigned_avg_bits`` instead.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._D = int(config.head_dim)
        self._bits_k = int(getattr(config, "skvq_bits_key", 2))
        self._bits_v = int(getattr(config, "skvq_bits_value", 2))
        self._group_size = int(getattr(config, "skvq_group_size", 32))
        self._window = int(getattr(config, "skvq_window", 128))
        self._n_sink = int(getattr(config, "skvq_n_sink", 5))
        self._reorder = bool(getattr(config, "skvq_reorder", True))
        self._clip_search = bool(getattr(config, "skvq_clip_search", True))
        self._clip_alpha = float(getattr(config, "skvq_clip_alpha", 1.0))
        self._max_ctx = int(getattr(config, "skvq_max_ctx", 8192))

        # Fail at build time, not on the first update (clear messages).
        for name, b in (("skvq_bits_key", self._bits_k), ("skvq_bits_value", self._bits_v)):
            if not (1 <= b <= 8):
                raise ValueError(f"SKVQKVCache: {name}={b} must be in [1, 8] (uint8 codes)")
        if self._group_size < 1:
            raise ValueError(f"SKVQKVCache: skvq_group_size={self._group_size} must be >= 1")
        if self._window < 2:
            raise ValueError(
                f"SKVQKVCache: skvq_window={self._window} must be >= 2 (the "
                f"first chunk supplies the channel statistics)"
            )
        if not (0 <= self._n_sink < self._window):
            raise ValueError(
                f"SKVQKVCache: skvq_n_sink={self._n_sink} must be in "
                f"[0, skvq_window) so sinks live entirely inside chunk 0"
            )
        if not (0.0 < self._clip_alpha <= 1.0):
            raise ValueError(f"SKVQKVCache: skvq_clip_alpha={self._clip_alpha} must be in (0, 1]")

        self._alphas = DEFAULT_ALPHA_GRID if self._clip_search else (self._clip_alpha,)

        # Per-head channel permutations, frozen from the first flushed
        # chunk. None until then; identity is represented by None when
        # skvq_reorder=False. Shapes [H, D] int32.
        self._perm_k: Optional[mx.array] = None
        self._inv_k: Optional[mx.array] = None
        self._perm_v: Optional[mx.array] = None
        self._inv_v: Optional[mx.array] = None

        # Quantized frontier: tokens [0, _q_end) have been chunk-flushed.
        # Always a multiple of _window.
        self._q_end = 0

        # Byte accounting (cumulative unless noted)
        self._compressed_key_bytes = 0
        self._compressed_value_bytes = 0
        self._fp16_key_bytes = 0
        self._fp16_value_bytes = 0
        self._tokens_seen = 0
        self._B = 1
        self._H = 1

    # ------------------------------------------------------------------
    # Permutations (per-head, frozen at first flush)
    # ------------------------------------------------------------------
    @staticmethod
    def _per_head_perms(chunk: mx.array) -> tuple:
        """Sorted-by-dynamic-range channel permutation per head.

        Args:
            chunk: ``[B, H, r, D]`` fp16/fp32 rows of the first chunk.

        Returns:
            ``(perm, inv)`` int32 ``[H, D]``.
        """
        B, H, r, D = chunk.shape
        x = chunk.astype(mx.float32).transpose(1, 0, 2, 3).reshape(H, B * r, D)
        rng = mx.max(x, axis=1) - mx.min(x, axis=1)  # [H, D]
        perm = mx.argsort(rng, axis=-1).astype(mx.int32)
        inv = mx.argsort(perm, axis=-1).astype(mx.int32)
        return perm, inv

    @staticmethod
    def _gather_channels(x: mx.array, perm: mx.array) -> mx.array:
        """Apply per-head channel permutation to ``[B, H, r, D]``."""
        B, H, r, D = x.shape
        idx = mx.broadcast_to(perm[None, :, None, :], (B, H, r, D))
        return mx.take_along_axis(x, idx, axis=-1)

    # ------------------------------------------------------------------
    # Reorder → clipped quant → dequant → inverse reorder for one chunk
    # ------------------------------------------------------------------
    def _round_trip(self, x: mx.array, perm, inv, bits: int) -> mx.array:
        B, H, r, D = x.shape
        x32 = x.astype(mx.float32)
        if perm is not None:
            x32 = self._gather_channels(x32, perm)
        flat = x32.reshape(B * H * r, D)
        codes, lo, scale = clipped_group_quant(flat, bits, self._group_size, self._alphas)
        recon = clipped_group_dequant(codes, lo, scale, self._group_size, D)
        recon = recon.reshape(B, H, r, D)
        if inv is not None:
            recon = self._gather_channels(recon, inv)
        return recon.astype(x.dtype)

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys, values):
        """Append the incoming block, then flush every completed chunk.

        The flush overwrites the aged-out tokens' fp16 storage in place with
        their dequantized round-trip (quantize once, frozen), so decode-time
        tokens are quantized when they age past the window — the paper's
        sliding-window semantics, unlike KIVI's incoming-block-only
        simplification. Chunk 0 additionally restores the first
        ``skvq_n_sink`` rows to their exact fp16 values (the sink filter)
        and computes the frozen channel permutations.
        """
        B, H, S, D = keys.shape
        if self.offset + S > self._max_ctx:
            raise ValueError(
                f"SKVQKVCache: context {self.offset + S} exceeds skvq_max_ctx={self._max_ctx}"
            )
        self._B, self._H = B, H

        super().update_and_fetch(keys, values)

        r = self._window
        while self.offset - self._q_end >= r:
            s, e = self._q_end, self._q_end + r
            k_chunk = self.keys[..., s:e, :]
            v_chunk = self.values[..., s:e, :]

            if s == 0 and self._reorder:
                self._perm_k, self._inv_k = self._per_head_perms(k_chunk)
                self._perm_v, self._inv_v = self._per_head_perms(v_chunk)

            k_q = self._round_trip(k_chunk, self._perm_k, self._inv_k, self._bits_k)
            v_q = self._round_trip(v_chunk, self._perm_v, self._inv_v, self._bits_v)
            if s == 0 and self._n_sink > 0:
                # Sink filter: leading tokens stay fp16-exact.
                k_q = mx.concatenate(
                    [k_chunk[..., : self._n_sink, :], k_q[..., self._n_sink :, :]], axis=2
                )
                v_q = mx.concatenate(
                    [v_chunk[..., : self._n_sink, :], v_q[..., self._n_sink :, :]], axis=2
                )
            self.keys[..., s:e, :] = k_q
            self.values[..., s:e, :] = v_q
            n_q = r - (self._n_sink if s == 0 else 0)
            self._account_chunk_bytes(B, H, n_q, D)
            self._q_end = e

        self._fp16_key_bytes += B * H * S * D * 2
        self._fp16_value_bytes += B * H * S * D * 2
        self._tokens_seen += S
        return (
            self.keys[..., : self.offset, :],
            self.values[..., : self.offset, :],
        )

    # ------------------------------------------------------------------
    # Byte accounting
    # ------------------------------------------------------------------
    def _account_chunk_bytes(self, B: int, H: int, n_q: int, D: int) -> None:
        self._compressed_key_bytes += (
            skvq_compressed_bytes(n_q, D, self._bits_k, self._group_size) * B * H
        )
        self._compressed_value_bytes += (
            skvq_compressed_bytes(n_q, D, self._bits_v, self._group_size) * B * H
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @property
    def compressed_key_bytes(self) -> int:
        return self._compressed_key_bytes

    @property
    def compressed_value_bytes(self) -> int:
        return self._compressed_value_bytes

    @property
    def fp16_key_bytes(self) -> int:
        return self._fp16_key_bytes

    @property
    def fp16_value_bytes(self) -> int:
        return self._fp16_value_bytes

    @property
    def residual_fp16_bytes(self) -> int:
        """Bytes currently held at fp16 (keys + values): the un-flushed
        sliding-window tail plus the sink rows that flushing skipped — a
        snapshot, not a cumulative counter."""
        n_res = self.offset - self._q_end
        n_sink_kept = min(self._n_sink, self._q_end)
        return skvq_fp16_bytes(n_res + n_sink_kept, self._D) * 2 * self._B * self._H

    @property
    def perm_bytes(self) -> int:
        """One-off storage for the frozen per-head permutations (int32)."""
        if self._perm_k is None and self._perm_v is None:
            return 0
        return 2 * self._H * self._D * 4

    @property
    def quantized_tokens(self) -> int:
        """Tokens behind the flush frontier (multiple of the window),
        including the fp16-restored sink rows."""
        return self._q_end

    @property
    def tokens_seen(self) -> int:
        return self._tokens_seen

    @property
    def key_perms(self) -> Optional[mx.array]:
        """Frozen per-head key channel permutations ``[H, D]`` (None before
        the first flush or when ``skvq_reorder=False``)."""
        return self._perm_k

    @property
    def value_perms(self) -> Optional[mx.array]:
        return self._perm_v

    @property
    def assigned_avg_bits(self) -> float:
        """Effective bits/element over the quantized key region, including
        the fp16 (lo, scale) metadata (excludes the fp16 window and sinks;
        for an end-to-end ratio use ``(compressed_*_bytes +
        residual_fp16_bytes) / fp16_*_bytes``)."""
        n_q = self._q_end - min(self._n_sink, self._q_end)
        if n_q == 0:
            return 16.0
        elems = n_q * self._D * self._B * self._H
        return 8.0 * self._compressed_key_bytes / elems


__all__ = ["SKVQKVCache"]
