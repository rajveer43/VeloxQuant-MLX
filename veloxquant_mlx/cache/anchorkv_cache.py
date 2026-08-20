"""AnchorKV-adapted KV cache — anchor-residual compression, no eviction.

Inspired by "AnchorKV: Anchor-Residual KV Cache Compression" (Khalaf,
Shamshoum, Hodos, Sieradzki, Schuster; Technion; arXiv:2608.02901v1,
2026-08-03). Documented as "AnchorKV-adapted (VeloxQuant-MLX
implementation)" — not a faithful port. **No verified peer-reviewed venue
as of 2026-08-20** — a one-time, user-directed exception to this repo's
venue-verification rule (the same exception previously granted to
NestedKV). See ``paper/research/surveys/NEW_METHOD_SURVEY_V22.md`` and
``quantizers/anchorkv.py`` for the full list of adaptation decisions.

Unlike every eviction method in this repo (H2O, SnapKV, PyramidKV, AdaKV,
NestedKV, ...), AnchorKV never drops a token: every position stays inside
the softmax. At the end of prefill, each head selects a small set of anchor
positions (stored exactly), assigns every other token to its nearest anchor
(one index + one scalar coefficient), and spends a byte budget derived from
``anchorkv_theta`` on quantized residuals for whichever tokens' approximation
error costs the most attention-output error (a first-order utility estimate,
paper Eq. 6). Decode tokens are appended exactly at fp16, never retroactively
anchored — the same one-shot-prefill convention as SnapKV-adapted /
NestedKV-adapted.

This module reconstructs a DENSE fp16 K/V tensor on every call (prefill and
decode alike) and hands it to the base ``mlx_lm`` ``KVCache`` — the paper's
fused tiled-reconstruction kernel (never materializing the dense cache) is
NOT implemented here; this is a correctness-first adaptation, and the byte
accounting below reflects the COMPRESSED representation's true storage cost,
not the transient dense tensor this wrapper materializes to satisfy the
``update_and_fetch`` contract.

Byte accounting:
    anchorkv_bytes    — true compressed storage: anchors (fp16) + per-token
                         metadata (index + coefficient) + packed residuals,
                         summed over all heads
    full_seq_bytes    — hypothetical fp16 K + V cost for the same tokens
    compression_ratio — full_seq_bytes / anchorkv_bytes (> 1 = savings)
    tokens_kept       — always == tokens_total (no eviction, ever)
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.anchorkv import (
    ResidualCodec,
    allocate_residual_budget,
    anchorkv_budget_slots,
    assign_and_project,
    key_value_utility,
    select_anchors,
)


class AnchorKVKVCache(_MLXKVCache):
    """KV cache implementing AnchorKV-adapted anchor-residual compression for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``anchorkv_theta``         (float, default 0.05) — fraction of the
                uncompressed fp16 cache to retain; the single user-facing
                compression knob (paper §3.4).
            ``anchorkv_window``        (int, default 32)   — trailing positions
                always kept as anchors and used as proxy observation queries.
            ``anchorkv_rho``           (float, default 0.7) — fraction of the
                non-window anchor budget filled by attention score (the rest
                sampled uniformly).
            ``anchorkv_anchor_frac``   (float, default 1/128) — anchor budget
                k as a fraction of context length S (paper: k = S/128).
            ``anchorkv_residual_bits`` (int, default 2)    — bits/coordinate
                for stored residuals.
            ``anchorkv_seed``          (int, default 42)   — RNG seed for the
                uniform anchor share and the residual codec's rotation.

    Notes:
        No ``.bits`` attribute — stores and returns fp16 K/V directly (the
        anchor-residual representation is internal; ``update_and_fetch``
        always returns a dense reconstruction).
        Compression happens ONCE at prefill (S > 1), across all heads
        independently for anchor selection/assignment but with the residual
        budget allocated by pooling utilities ACROSS heads (paper §3.4).
        Decode tokens (S == 1) are always appended exactly — never dropped,
        never retroactively anchored.
        Single-layer (no coordinator); ``KVCacheBuilder.for_model()``
        propagates all ``anchorkv_*`` fields automatically via
        ``dataclasses.replace``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._theta = float(getattr(config, "anchorkv_theta", 0.05))
        self._window = int(getattr(config, "anchorkv_window", 32))
        self._rho = float(getattr(config, "anchorkv_rho", 0.7))
        self._anchor_frac = float(getattr(config, "anchorkv_anchor_frac", 1.0 / 128.0))
        self._residual_bits = int(getattr(config, "anchorkv_residual_bits", 2))
        self._seed = int(getattr(config, "anchorkv_seed", 42))

        self._head_dim: int = 0
        self._compressed: bool = False

        # Per-(batch*head) compressed state, populated once at prefill.
        self._anchor_positions: list[Any] = []
        self._key_assign: list[Any] = []
        self._value_assign: list[Any] = []
        self._key_residual_mask: list[Any] = []
        self._value_residual_mask: list[Any] = []
        self._key_codec: ResidualCodec | None = None
        self._value_codec: ResidualCodec | None = None
        # Exact anchor K/V and the decoded (anchor-projection + residual) fp16
        # reconstruction are recomputed once per prefill call and then reused;
        # decode tokens are concatenated onto the reconstruction directly.
        self._reconstructed_keys: list[Any] = []
        self._reconstructed_values: list[Any] = []

        self._B: int = 0
        self._H: int = 0

        self._anchorkv_bytes: int = 0
        self._full_seq_bytes: int = 0
        self._tokens_total: int = 0
        self._n_anchor_total: int = 0
        self._n_residual_total: int = 0

    # ------------------------------------------------------------------
    def _head_idx(self, b: int, h: int) -> int:
        return b * self._H + h

    def _compress_head(self, keys_h: Any, values_h: Any) -> tuple[Any, Any, int, int]:
        """Compress one head's ``[S, D]`` K/V; returns reconstructed fp16 (K, V)
        plus (n_anchor, n_residual) for byte accounting."""
        S = int(keys_h.shape[0])
        D = int(keys_h.shape[1])
        k_budget = max(1, int(round(S * self._anchor_frac)))

        anchors = select_anchors(
            keys_h.astype(mx.float32),
            k=k_budget,
            window=self._window,
            rho=self._rho,
            seed=self._seed,
        )
        n_anchor = int(anchors.shape[0])

        key_assign = assign_and_project(keys_h, anchors)
        value_assign = assign_and_project(values_h, anchors)

        m = min(self._window, S)
        proxy_q = keys_h.astype(mx.float32)[-m:]
        u_key, u_value = key_value_utility(
            proxy_q,
            keys_h.astype(mx.float32),
            values_h.astype(mx.float32),
            key_assign.residual,
            value_assign.residual,
        )

        anchor_set = set(int(a) for a in anchors.tolist())
        non_anchor_mask_np = [i not in anchor_set for i in range(S)]

        non_anchor_mask = mx.array(non_anchor_mask_np)
        neg_inf_on_anchor = mx.where(
            non_anchor_mask, mx.zeros((S,), dtype=mx.float32), mx.full((S,), -1e30)
        )
        u_key = u_key + neg_inf_on_anchor
        u_value = u_value + neg_inf_on_anchor

        codec = ResidualCodec(head_dim=D, seed=self._seed, bits=self._residual_bits)
        n_slots = anchorkv_budget_slots(
            seq_len=S,
            head_dim=D,
            n_anchor=n_anchor,
            theta=self._theta,
            residual_codec_bytes=codec.bytes_per_residual,
        )
        n_key_slots = n_slots // 2
        n_value_slots = n_slots - n_key_slots

        key_mask = allocate_residual_budget([u_key], n_key_slots)[0]
        value_mask = allocate_residual_budget([u_value], n_value_slots)[0]

        key_recon = self._reconstruct_side(keys_h, key_assign, key_mask, codec)
        value_recon = self._reconstruct_side(values_h, value_assign, value_mask, codec)

        n_residual = int(mx.sum(key_mask.astype(mx.int32)).item()) + int(
            mx.sum(value_mask.astype(mx.int32)).item()
        )
        anchor_bytes = n_anchor * D * 2 * 2
        metadata_bytes = (S - n_anchor) * 2 * (4 + 4)
        residual_bytes = n_residual * codec.bytes_per_residual
        self._anchorkv_bytes += anchor_bytes + metadata_bytes + residual_bytes

        return key_recon.astype(mx.float16), value_recon.astype(mx.float16), n_anchor, n_residual

    @staticmethod
    def _reconstruct_side(x: Any, assign, mask: Any, codec: ResidualCodec) -> Any:
        """``x_hat = gamma * anchor + (residual if mask else 0)`` (paper Eq. 3)."""
        chosen_anchor = x.astype(mx.float32)[assign.anchor_positions][assign.assign_idx]
        x_tilde = assign.gamma[:, None] * chosen_anchor

        codes, scale = codec.encode(assign.residual)
        decoded_residual = codec.decode(codes, scale)
        residual_term = mx.where(mask[:, None], decoded_residual, mx.zeros_like(decoded_residual))

        return x_tilde + residual_term

    def _process_prefill(self, keys: Any, values: Any):
        B, H, S, D = keys.shape
        self._B, self._H, self._head_dim = B, H, D

        k_out_b, v_out_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                k_recon, v_recon, n_anchor, n_residual = self._compress_head(
                    keys[b, h], values[b, h]
                )
                k_out_h.append(k_recon)
                v_out_h.append(v_recon)
                self._n_anchor_total += n_anchor
                self._n_residual_total += n_residual
            k_out_b.append(mx.stack(k_out_h, axis=0))
            v_out_b.append(mx.stack(v_out_h, axis=0))

        self._reconstructed_keys = mx.stack(k_out_b, axis=0)
        self._reconstructed_values = mx.stack(v_out_b, axis=0)
        self._compressed = True
        return self._reconstructed_keys, self._reconstructed_values

    def _process_decode(self, keys: Any, values: Any):
        """Append decode tokens exactly (fp16) — never anchored, never dropped."""
        k_new = keys.astype(mx.float16)
        v_new = values.astype(mx.float16)
        self._reconstructed_keys = mx.concatenate([self._reconstructed_keys, k_new], axis=2)
        self._reconstructed_values = mx.concatenate([self._reconstructed_values, v_new], axis=2)

        B, H, S, D = keys.shape
        self._anchorkv_bytes += B * H * S * D * 2 * 2  # fp16 K + V, exact
        return self._reconstructed_keys, self._reconstructed_values

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: Any, values: Any):
        """Absorb new K/V tokens; prefill compresses once, decode always appends exactly.

        Args:
            keys:   ``[B, H, S, D]`` new key tokens (any dtype; cast to fp16/fp32 internally).
            values: ``[B, H, S, D]`` new value tokens.

        Returns:
            ``(K_out, V_out)`` both ``[B, H, n_total, D]`` fp16 — ``n_total`` is
            always the full token count seen so far, never reduced.
        """
        B, H, S, D = keys.shape
        self._full_seq_bytes += B * H * S * D * 2 * 2
        self._tokens_total += B * H * S

        is_prefill = S > 1 and not self._compressed
        if is_prefill:
            K_out, V_out = self._process_prefill(keys, values)
        else:
            K_out, V_out = self._process_decode(keys, values)

        # K_out/V_out is the full retained state every call, not a delta —
        # reset so the base class's append-only buffer starts fresh instead
        # of stacking on top of the previous call's rows (same convention as
        # NestedKVKVCache / SnapKVKVCache's chunked-prefill re-enforcement).
        self.keys = None
        self.values = None
        self.offset = 0
        return super().update_and_fetch(K_out, V_out)

    # ------------------------------------------------------------------
    def is_trimmable(self) -> bool:
        """False: trim() would only roll back base-class offset bookkeeping,
        not the internal anchor/residual state that determines what gets
        returned, silently corrupting future calls.
        """
        return False

    # ------------------------------------------------------------------
    @property
    def anchorkv_bytes(self) -> int:
        """True compressed storage: anchors + metadata + packed residuals, all heads."""
        return self._anchorkv_bytes

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost for the same tokens, uncompressed."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """full_seq_bytes / anchorkv_bytes; > 1 means storage savings."""
        if self._anchorkv_bytes == 0:
            return 1.0
        return self._full_seq_bytes / self._anchorkv_bytes

    @property
    def tokens_kept(self) -> int:
        """Always equals tokens_total — AnchorKV never evicts a token."""
        return self._tokens_total

    @property
    def tokens_total(self) -> int:
        """Total token positions ever passed to update_and_fetch (all heads summed)."""
        return self._tokens_total

    @property
    def n_anchor_total(self) -> int:
        """Total anchor positions selected across all (batch, head) pairs at prefill."""
        return self._n_anchor_total

    @property
    def n_residual_total(self) -> int:
        """Total residual slots spent (K + V) across all (batch, head) pairs at prefill."""
        return self._n_residual_total


__all__ = ["AnchorKVKVCache"]
