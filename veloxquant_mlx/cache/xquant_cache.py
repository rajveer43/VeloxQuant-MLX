"""XQuant KV cache wrapper — cross-layer KV cache reuse.

Inspired by "XQuant: Achieving Ultra-Low Bit KV Cache Quantization with
Cross-Layer Compression" (arXiv:2510.11236, EMNLP 2025). Documented as
"XQuant-adapted (VeloxQuant-MLX implementation)" — faithful to the cross-layer
reuse core, adapted at the integration boundary via a shared
:class:`XQuantCoordinator` rather than a modified attention forward pass.

Per-layer roles (assigned at build time by ``pair_layers``):
    Anchor layer — quantizes K/V with asymmetric min/max group quant, publishes
        the integer codes to the coordinator, returns the fp16 reconstruction.
    Reuse layer — fetches the paired anchor's codes for the same token range,
        fits its *own* per-group scale/zero to those codes (correcting the small
        cross-layer drift), and reconstructs. Stores only its params (+ optional
        low-bit residual) — never a full code tensor. That is the byte win.

Both keys and values are compressed (XQuant is a both-tensor method; values are
typically quite correlated across layers too). Set ``xquant_base_bits`` for the
anchor; reuse layers inherit the same bit-width plus an optional residual.

Byte accounting:
    Anchor: full ``compressed_key_bytes`` / ``compressed_value_bytes`` (codes + params).
    Reuse:  only ``reuse_param_bytes`` (scale+zero per group) + optional residual.
    ``fp16_*`` always reflects the uncompressed cost for the ratio.

Degenerate case: with no coordinator (single isolated layer) the cache behaves
as a plain anchor — useful for unit-testing the anchor path in isolation.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.cache.xquant_coordinator import XQuantCoordinator
from veloxquant_mlx.quantizers.xquant import (
    GroupParams,
    compute_reuse_params,
    dequant_with_params,
    quantize_codes,
    quantize_residual,
)


class XQuantKVCache(_MLXKVCache):
    """KV cache implementing XQuant cross-layer reuse for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``xquant_base_bits``       (int, default 2),
            ``xquant_residual_bits``   (int, default 0),
            ``xquant_group_quant_size``(int, default 32).
        role: ``"anchor"`` or ``"reuse"`` (default ``"anchor"``).
        group_id: Cross-layer group this layer belongs to.
        coordinator: Shared :class:`XQuantCoordinator` (None → degenerate anchor).
    """

    def __init__(
        self,
        config: Any,
        role: str = "anchor",
        group_id: int = 0,
        coordinator: Optional[XQuantCoordinator] = None,
    ) -> None:
        super().__init__()
        self._role: str = role if coordinator is not None else "anchor"
        self._group_id: int = int(group_id)
        self._coord: Optional[XQuantCoordinator] = coordinator

        self._base_bits: int = int(getattr(config, "xquant_base_bits", 2))
        self._residual_bits: int = int(getattr(config, "xquant_residual_bits", 0))
        self._gqs: int = int(getattr(config, "xquant_group_quant_size", 32))

        self._token_offset: int = 0  # this layer's running token count

        # Byte accounting
        self._compressed_key_bytes: int = 0
        self._compressed_value_bytes: int = 0
        self._reuse_param_bytes: int = 0
        self._fp16_key_bytes: int = 0
        self._fp16_value_bytes: int = 0

    # ------------------------------------------------------------------
    # Anchor / reuse quantization (per (B, H) head)
    # ------------------------------------------------------------------
    def _quantize_anchor(self, t: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize a [B, H, S, D] tensor. Returns (recon_fp16, codes_stacked).

        codes_stacked: [B, H, n_groups, gs, D] fp32 codes for coordinator storage.
        params are recomputed deterministically by reusers, so only codes travel.
        """
        B, H, S, D = t.shape
        recon_b, codes_b = [], []
        for b in range(B):
            recon_h, codes_h = [], []
            for h in range(H):
                codes, params = quantize_codes(t[b, h], self._base_bits, self._gqs)
                recon_h.append(dequant_with_params(codes, params))
                codes_h.append(codes)
            recon_b.append(mx.stack(recon_h, axis=0))
            codes_b.append(mx.stack(codes_h, axis=0))
        return mx.stack(recon_b, axis=0), mx.stack(codes_b, axis=0)

    def _reconstruct_reuse(self, t: mx.array, codes_stacked: mx.array) -> mx.array:
        """Reconstruct a [B, H, S, D] tensor from shared anchor codes.

        Fits this layer's own params to the codes; optionally adds a residual.
        """
        B, H, S, D = t.shape
        recon_b = []
        for b in range(B):
            recon_h = []
            for h in range(H):
                codes = codes_stacked[b, h]
                params = compute_reuse_params(t[b, h], codes, self._base_bits, self._gqs)
                recon = dequant_with_params(codes, params)
                if self._residual_bits > 0:
                    recon = (
                        recon.astype(mx.float32)
                        + quantize_residual(t[b, h], recon, self._residual_bits, self._gqs).astype(
                            mx.float32
                        )
                    ).astype(mx.float16)
                recon_h.append(recon)
            recon_b.append(mx.stack(recon_h, axis=0))
        return mx.stack(recon_b, axis=0)

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, S, D = keys.shape
        tok_start = self._token_offset

        if self._role == "anchor":
            k_out, k_codes = self._quantize_anchor(keys)
            v_out, v_codes = self._quantize_anchor(values)
            if self._coord is not None:
                # Store keys+values codes together (tuple in .codes slot).
                self._coord.register_anchor(
                    self._group_id,
                    tok_start,
                    S,
                    codes=(k_codes, v_codes),
                    params=GroupParams(scale=None, zero=None, n_rows=S, bits=self._base_bits),
                )
            self._account_anchor(B, H, S, D)
        else:
            seg = self._coord.fetch_anchor(self._group_id, tok_start)
            if seg is None:
                # Anchor hasn't published this step (mis-ordered) — fall back to
                # self-quantization so correctness never depends on iteration order.
                k_out, _ = self._quantize_anchor(keys)
                v_out, _ = self._quantize_anchor(values)
                self._account_anchor(B, H, S, D)
            else:
                k_codes, v_codes = seg.codes
                k_out = self._reconstruct_reuse(keys, k_codes)
                v_out = self._reconstruct_reuse(values, v_codes)
                self._account_reuse(B, H, S, D)

        self._token_offset += S
        return super().update_and_fetch(k_out, v_out)

    # ------------------------------------------------------------------
    # Byte accounting
    # ------------------------------------------------------------------
    def _code_param_bytes(self, S: int, D: int, bits: int, B: int, H: int) -> int:
        code_bytes = math.ceil(S * D * bits / 8)
        n_groups = math.ceil(S / self._gqs)
        param_bytes = n_groups * D * 2 * 2  # scale + zero, fp16
        return (code_bytes + param_bytes) * B * H

    def _param_only_bytes(self, S: int, D: int, B: int, H: int) -> int:
        n_groups = math.ceil(S / self._gqs)
        param_bytes = n_groups * D * 2 * 2
        res_bytes = 0
        if self._residual_bits > 0:
            res_bytes = math.ceil(S * D * self._residual_bits / 8) + param_bytes
        return (param_bytes + res_bytes) * B * H

    def _account_anchor(self, B: int, H: int, S: int, D: int) -> None:
        self._compressed_key_bytes += self._code_param_bytes(S, D, self._base_bits, B, H)
        self._compressed_value_bytes += self._code_param_bytes(S, D, self._base_bits, B, H)
        self._fp16_key_bytes += B * H * S * D * 2
        self._fp16_value_bytes += B * H * S * D * 2

    def _account_reuse(self, B: int, H: int, S: int, D: int) -> None:
        pb = self._param_only_bytes(S, D, B, H)
        self._reuse_param_bytes += pb
        self._compressed_key_bytes += pb  # reuse stores only params (+residual)
        self._fp16_key_bytes += B * H * S * D * 2
        self._fp16_value_bytes += B * H * S * D * 2

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def role(self) -> str:
        return self._role

    @property
    def group_id(self) -> int:
        return self._group_id

    @property
    def compressed_key_bytes(self) -> int:
        return self._compressed_key_bytes

    @property
    def compressed_value_bytes(self) -> int:
        return self._compressed_value_bytes

    @property
    def reuse_param_bytes(self) -> int:
        return self._reuse_param_bytes

    @property
    def fp16_key_bytes(self) -> int:
        return self._fp16_key_bytes

    @property
    def fp16_value_bytes(self) -> int:
        return self._fp16_value_bytes

    @property
    def effective_pair_bits(self) -> float:
        """Effective per-element bits charged to *this* layer (key side)."""
        if self._fp16_key_bytes == 0:
            return float(self._base_bits)
        # fp16 = 2 bytes = 16 bits per element; ratio scales to bits.
        return 16.0 * self._compressed_key_bytes / self._fp16_key_bytes


__all__ = ["XQuantKVCache"]
