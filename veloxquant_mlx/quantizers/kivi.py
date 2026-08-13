"""KIVI — tuning-free asymmetric group quantization for KV caches.

Based on: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
Liu, Yuan et al., ICML 2024 (arXiv:2402.02750).
Reference implementation: https://github.com/jy-yuan/KIVI

Algorithm (this module implements the per-channel *group* quantizer that
KIVI applies to keys; the per-token variant for values shares the same
math along a different axis and is selected by the ``axis`` argument):

  Group quantization of a real-valued group g (a contiguous slice of size
  ``group_size`` along the quantization axis):

      zero = min(g)
      scale = (max(g) - min(g)) / (2**b - 1)
      q = round((g - zero) / scale)            # uint, in [0, 2**b - 1]
      g_hat = q * scale + zero                  # asymmetric dequant

  KIVI's asymmetry: **keys are quantized per channel** (the quantization
  group runs along the token axis, one (scale, zero) per channel-group) and
  **values per token** (group runs along the channel axis, one (scale, zero)
  per token-group).  The most recent ``residual_length`` tokens are kept in
  fp16 by the *cache* wrapper (:class:`KIVIKVCache`); this quantizer handles
  the quantized portion only.

This standalone ``Quantizer`` operates on a ``(batch, d)`` array and
quantizes per-channel by default (axis=0, the KIVI key scheme): each of the
``d`` channels is split into ceil(batch / group_size) groups along the batch
(token) axis and quantized independently.  It is fully deterministic — no
codebook training, no RNG — so it never introduces the run-to-run parity
flakiness seen in VQ-based methods.

Public API:
  KIVIQuantizer — encode / decode / estimate_inner_product
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import numpy as np

from veloxquant_mlx.core.abstractions import ArtifactStore, Quantizer
from veloxquant_mlx.core.context import EncodedVector
from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.core.registry import QuantizerRegistry


@QuantizerRegistry.register("kivi")
class KIVIQuantizer(Quantizer):
    """KIVI asymmetric group quantizer (per-channel keys by default).

    Args:
        d: Vector dimension (head_dim).
        b: Bit-width per element (KIVI default 2).
        group_size: Quantization group size along the quantization axis.
            KIVI's reference uses 32 (keys) / 32 (values).  Must divide the
            relevant axis length at encode time, or the final ragged group
            is padded by repeating its own min/max (lossless for that group).
        axis: ``"channel"`` (default, KIVI key scheme — group along the
            token/batch axis, one set of scales per channel) or ``"token"``
            (KIVI value scheme — group along the channel axis, one set of
            scales per token).
        seed: Unused (KIVI is deterministic); accepted for factory parity.
        store: Unused; accepted for factory parity.

    Notes:
        ``encode`` returns an :class:`EncodedVector` whose ``indices`` hold
        the uint8 quantized codes and whose ``norm``/``residual_norm`` fields
        are repurposed to carry the per-group ``scale`` and ``zero`` (KIVI
        stores no L2 norm).  ``decode`` reverses the asymmetric mapping.
    """

    def __init__(
        self,
        d: int,
        b: int = 2,
        group_size: int = 32,
        axis: str = "channel",
        m: Optional[int] = None,  # accepted for QuantizerFactory parity
        seed: int = 42,
        store: Optional[ArtifactStore] = None,
        **kwargs: Any,
    ) -> None:
        if b < 1 or b > 8:
            raise QuantizerConfigError(f"KIVIQuantizer: b={b} must be in [1, 8] (uint8 codes).")
        if group_size < 1:
            raise QuantizerConfigError(f"KIVIQuantizer: group_size={group_size} must be >= 1.")
        if axis not in ("channel", "token"):
            raise QuantizerConfigError(
                f"KIVIQuantizer: axis={axis!r} must be 'channel' or 'token'."
            )
        self._d = int(d)
        self._b = int(b)
        self._group_size = int(group_size)
        self._axis = axis
        self._levels = (1 << self._b) - 1  # 2**b - 1
        self._eps = 1e-8

    # ------------------------------------------------------------------
    # Core group quant/dequant (pure MLX, deterministic)
    # ------------------------------------------------------------------
    def _quantize_groups(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Asymmetric per-group quantization along axis 0 of a 2-D array.

        Groups partition the rows (axis 0) into blocks of ``group_size``.
        Returns ``(codes_uint8, scale, zero)`` where ``scale``/``zero`` have
        one row per group, broadcast back over the group's rows on decode.
        """
        n, d = x.shape
        gs = self._group_size
        n_groups = (n + gs - 1) // gs
        pad = n_groups * gs - n
        if pad:
            # Repeat the last row to fill the ragged final group.  This only
            # affects the padded rows, which are discarded on decode.
            x = mx.concatenate([x, mx.broadcast_to(x[-1:], (pad, d))], axis=0)
        xg = x.reshape(n_groups, gs, d)  # [G, gs, d]
        gmin = mx.min(xg, axis=1, keepdims=True)  # [G, 1, d]
        gmax = mx.max(xg, axis=1, keepdims=True)
        scale = (gmax - gmin) / self._levels
        scale = mx.maximum(scale, self._eps)  # avoid /0 on constant groups
        codes = mx.round((xg - gmin) / scale)
        codes = mx.clip(codes, 0, self._levels).astype(mx.uint8)
        codes = codes.reshape(n_groups * gs, d)[:n]  # drop padding
        scale = scale.reshape(n_groups, d)
        zero = gmin.reshape(n_groups, d)
        return codes, scale.astype(mx.float32), zero.astype(mx.float32)

    def _dequantize_groups(
        self, codes: mx.array, scale: mx.array, zero: mx.array, n: int
    ) -> mx.array:
        gs = self._group_size
        d = codes.shape[1]
        n_groups = scale.shape[0]
        pad = n_groups * gs - n
        c = codes
        if pad:
            c = mx.concatenate([codes, mx.broadcast_to(codes[-1:], (pad, d))], axis=0)
        cg = c.reshape(n_groups, gs, d).astype(mx.float32)
        recon = cg * scale[:, None, :] + zero[:, None, :]
        return recon.reshape(n_groups * gs, d)[:n]

    # ------------------------------------------------------------------
    # Quantizer interface
    # ------------------------------------------------------------------
    def encode(self, x: Any) -> EncodedVector:
        """Encode ``(batch, d)`` keys with KIVI group quantization.

        For ``axis="channel"`` (keys) the groups run along the batch/token
        axis directly.  For ``axis="token"`` (values) we transpose so each
        token's channels form the groups, then transpose scales back.
        """
        if not isinstance(x, mx.array):
            x = mx.array(x)
        n, d = x.shape
        if d != self._d:
            raise QuantizerConfigError(f"KIVIQuantizer.encode: expected dim {self._d}, got {d}.")
        if self._axis == "channel":
            codes, scale, zero = self._quantize_groups(x.astype(mx.float32))
        else:  # token: quantize along channel axis → operate on x.T
            codes_t, scale_t, zero_t = self._quantize_groups(x.astype(mx.float32).T)
            codes = codes_t.T
            scale = scale_t.T  # [d_groups, n] → [n, d_groups]
            zero = zero_t.T
        ev = EncodedVector(
            quantizer_type="kivi",
            batch_size=n,
            dim=d,
            indices=codes,
            norm=scale,  # repurposed: per-group scale
            residual_norm=zero,  # repurposed: per-group zero-point
        )
        return ev

    def decode(self, ev: EncodedVector) -> Any:
        """Reconstruct ``(batch, d)`` fp16 keys from an EncodedVector."""
        codes = ev.indices
        scale = ev.norm
        zero = ev.residual_norm
        n = ev.batch_size
        if self._axis == "channel":
            recon = self._dequantize_groups(codes, scale, zero, n)
        else:
            recon_t = self._dequantize_groups(codes.T, scale.T, zero.T, ev.dim)
            recon = recon_t.T
        return recon.astype(mx.float16)

    def estimate_inner_product(self, q: Any, ev: EncodedVector) -> Any:
        """Estimate ``⟨q, k⟩`` for each encoded key via full reconstruction.

        KIVI carries no sketch; the unbiased estimate is the exact dot with
        the dequantized keys.  Returns shape ``(batch,)`` fp16.
        """
        if not isinstance(q, mx.array):
            q = mx.array(q)
        k_hat = self.decode(ev).astype(mx.float32)
        q32 = q.reshape(-1).astype(mx.float32)
        return (k_hat @ q32).astype(mx.float16)

    def __repr__(self) -> str:
        return (
            f"KIVIQuantizer(d={self._d}, b={self._b}, "
            f"group_size={self._group_size}, axis={self._axis!r})"
        )


__all__ = ["KIVIQuantizer"]
