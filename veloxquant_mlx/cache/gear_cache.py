"""GEAR KV cache wrapper — error-feedback compression over a base group quant.

Inspired by "GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless
Generative Inference of LLM" (Kang et al., arXiv:2403.05527). Documented as
"GEAR-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

Unlike CacheGen (whose reconstruction is identical to plain group quant and whose
win is a storage-byte model), GEAR's reconstruction is a genuine lossy
reconstruction that **recovers quality** the base bit-width alone would lose:

    X  ~=  Quant_b(X)  +  L . R  +  S

The base layer follows the paper's **KCVT** backbone: keys quantized
per-channel, values quantized per-token (``gear_compress(..., base_axis=...)``)
— not a generic per-token quantizer applied to both, which was this module's
behavior before this backbone was wired in.

The wrapper compresses each ``[B, H, S, D]`` head matrix with
``gear_compress`` / ``gear_reconstruct`` and hands the reconstructed fp16 K/V to
the parent ``mlx_lm`` cache (so SDPA stays on the clean fp16 path — no ``.bits``
attribute). Byte accounting reports the GEAR stored size (base codes + low-rank
factors + sparse triples) against both fp16 and a base-only baseline, plus an
error-recovery ratio quantifying how much quantization error the feedback layers
removed.

Adaptation: the residual SVD is computed per ``update_and_fetch`` call on the
tensor the cache holds (prefill batch when ``S > 1``, single-token at decode).
GEAR's fused streaming-dequant CUDA kernel is not ported — we reconstruct fp16
then call MLX SDPA, so stored size shrinks but attend-time peak memory does not.

Overhead caveat: the low-rank factors cost ``(N + D) * r * 2`` bytes and the
sparse triples ``nnz * 6`` bytes. For these to stay below the fp16 budget the
rank must be genuinely *low* relative to ``D`` (the GEAR premise) — on tiny head
dims with a near-``D/2`` rank the error-feedback overhead can exceed fp16. Keep
``gear_rank`` small (or use ``gear_energy_threshold``) so ``compressed`` stays
between ``base_only`` and ``fp16``. This is the configured operating regime; it
is not enforced, so an unreasonable rank is reported honestly as overhead.

Byte accounting:
    compressed_key_bytes / compressed_value_bytes   — GEAR three-part stored size
    base_only_key_bytes  / base_only_value_bytes    — base codes alone (baseline)
    fp16_key_bytes       / fp16_value_bytes          — uncompressed cost for the ratio
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.gear import (
    base_only_bytes,
    gear_bytes,
    gear_compress,
    gear_reconstruct,
)


class GEARKVCache(_MLXKVCache):
    """KV cache implementing GEAR error-feedback compression for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``gear_bits``             (int, default 2),
            ``gear_rank``             (int | None, default None → energy threshold),
            ``gear_energy_threshold`` (float, default 0.90),
            ``gear_sparse_fraction``  (float, default 0.01),
            ``gear_group_size``       (int, default 32),
            ``gear_quantize_values``  (bool, default True — GEAR values too).

    Notes:
        No ``.bits`` attribute — keeps mlx_lm SDPA on the clean fp16 path.
        Single-layer (no coordinator); ``for_model`` propagates the ``gear_*``
        fields automatically via ``dataclasses.replace``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._bits = int(getattr(config, "gear_bits", 2))
        rank = getattr(config, "gear_rank", None)
        self._rank: Optional[int] = None if rank is None else int(rank)
        self._energy = float(getattr(config, "gear_energy_threshold", 0.90))
        self._sparse_frac = float(getattr(config, "gear_sparse_fraction", 0.01))
        self._gs = int(getattr(config, "gear_group_size", 32))
        self._quant_values = bool(getattr(config, "gear_quantize_values", True))

        self._compressed_key_bytes = 0
        self._compressed_value_bytes = 0
        self._base_only_key_bytes = 0
        self._base_only_value_bytes = 0
        self._fp16_key_bytes = 0
        self._fp16_value_bytes = 0
        # error-recovery accumulators (sum of squared residual, key side)
        self._err_base_sq = 0.0
        self._err_after_sq = 0.0

    # ------------------------------------------------------------------
    def _compress_and_account(self, t: mx.array, is_key: bool) -> mx.array:
        """Compress [B, H, S, D] per head with GEAR, accumulate accounting, return fp16."""
        B, H, S, D = t.shape
        base_axis = "channel" if is_key else "token"  # KCVT: keys per-channel, values per-token
        recon_b = []
        comp = 0
        base = 0
        for b in range(B):
            recon_h = []
            for h in range(H):
                mat = t[b, h]  # [S, D]
                state = gear_compress(
                    mat,
                    bits=self._bits,
                    rank=self._rank,
                    sparse_frac=self._sparse_frac,
                    group_size=self._gs,
                    energy_threshold=self._energy,
                    base_axis=base_axis,
                )
                rec = gear_reconstruct(state)
                recon_h.append(rec)
                comp += gear_bytes(state)
                base += base_only_bytes(state)
                if is_key:
                    self._accumulate_error(mat, state, rec, base_axis)
            recon_b.append(mx.stack(recon_h, axis=0))
        out = mx.stack(recon_b, axis=0)

        fp16 = B * H * S * D * 2
        if is_key:
            self._compressed_key_bytes += comp
            self._base_only_key_bytes += base
            self._fp16_key_bytes += fp16
        else:
            self._compressed_value_bytes += comp
            self._base_only_value_bytes += base
            self._fp16_value_bytes += fp16
        return out

    def _accumulate_error(self, mat: mx.array, state, rec: mx.array, base_axis: str) -> None:
        """Track base-vs-GEAR squared residual for the error-recovery ratio."""
        from veloxquant_mlx.quantizers.gear import quantize_base

        x32 = mat.astype(mx.float32)
        _, base_recon = quantize_base(x32, self._bits, self._gs, axis=base_axis)
        self._err_base_sq += float(mx.sum((x32 - base_recon) ** 2).item())
        self._err_after_sq += float(mx.sum((x32 - rec.astype(mx.float32)) ** 2).item())

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        k_out = self._compress_and_account(keys, is_key=True)
        if self._quant_values:
            v_out = self._compress_and_account(values, is_key=False)
        else:
            v_out = values
            B, H, S, D = values.shape
            self._fp16_value_bytes += B * H * S * D * 2
        return super().update_and_fetch(k_out, v_out)

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
    def base_only_key_bytes(self) -> int:
        return self._base_only_key_bytes

    @property
    def base_only_value_bytes(self) -> int:
        return self._base_only_value_bytes

    @property
    def fp16_key_bytes(self) -> int:
        return self._fp16_key_bytes

    @property
    def fp16_value_bytes(self) -> int:
        return self._fp16_value_bytes

    @property
    def assigned_avg_bits(self) -> float:
        """Effective key bit-width after error-feedback overhead (vs fp16=16)."""
        if self._fp16_key_bytes == 0:
            return float(self._bits)
        return 16.0 * self._compressed_key_bytes / self._fp16_key_bytes

    @property
    def error_recovery_ratio(self) -> float:
        """Fraction of the base quantization error removed by GEAR (key side).

        ``1 - ||X - GEAR(X)||^2 / ||X - base(X)||^2``. 0 = no recovery,
        →1 = near-lossless. The core GEAR claim, measured not asserted.
        """
        if self._err_base_sq <= 0.0:
            return 0.0
        return 1.0 - self._err_after_sq / self._err_base_sq


__all__ = ["GEARKVCache"]
