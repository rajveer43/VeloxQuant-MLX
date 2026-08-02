"""PALU KV cache — true low-rank latent storage for keys *and* values.

Inspired by "PALU: Compressing KV-Cache with Low-Rank Projection"
(arXiv:2407.21118, ICLR 2025).  Documented as "PALU-adapted (VeloxQuant-MLX
implementation)" — not a faithful port.

The defining property (and what separates this from the repo's SVDq cache):
**the cache stores the latent codes ``[B, H, S, r]`` directly and never
materialises full fp16 ``[B, H, S, D]`` keys/values for storage.**  SVDq
reconstructs full fp16 keys, hands them to the parent ``mlx_lm`` ``KVCache``,
and so wins only on byte-accounting/bandwidth.  PALU keeps the cache itself
low-rank and reconstructs to fp16 *only* at attend time, so its peak storage is
genuinely smaller.

Design:

  Prefill (first call, S > 1):
    1. Partition the H heads into ``palu_n_head_groups`` contiguous groups.
    2. For keys (and values, unless ``palu_quantize_values`` is False but still
       low-rank): fit one shared projection per group via group-head SVD.
    3. Project each head into its group's latent space, mixed-bit quantize the
       ``[S, r]`` latents, and seed the growing latent buffers.

  Decode (S == 1):
    1. Project the new key/value into the stored group projections.
    2. Mixed-bit quantize and append to the latent buffers.

  Every call returns reconstructed fp16 ``[B, H, S, D]`` (latent → dequant →
  reconstruct) for the downstream SDPA, but storage stays latent.

Because the parent's fp16 ``self.keys`` / ``self.values`` ring buffer is
bypassed, this class manages ``offset`` itself and stores latents in per-group
lists, then reconstructs the full sequence on each fetch.

Byte accounting:
  compressed_key_bytes / compressed_value_bytes — latent storage at the mixed-
    bit effective rate (this is the real stored size, not a reconstruction)
  fp16_key_bytes / fp16_value_bytes             — what full fp16 would cost
  projection_bytes                              — V_g + mu_g (fp32), amortised

  Effective bit-width ≈ (r / D) * weighted_avg(hi_bit, lo_bit).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.quantizers.palu import (
    group_head_svd,
    head_group_bounds,
    project_to_latent,
    quantize_latent,
    reconstruct_from_latent,
)


class _TensorLowRank:
    """Per-tensor (keys or values) group-head low-rank state + latent buffer.

    Holds, for one of the two tensors, the per-group projections fit at prefill
    and the growing list of reconstructed (dequantised) latents, plus the
    fp16 mean so reconstruction is ``L @ V.T + mu``.
    """

    def __init__(
        self,
        n_head_groups: int,
        rank: Optional[int],
        energy_threshold: float,
        hi_bit: int,
        lo_bit: int,
        hi_fraction: float,
        group_size: int,
        quantize: bool,
    ) -> None:
        self.n_head_groups = n_head_groups
        self.rank = rank
        self.energy_threshold = energy_threshold
        self.hi_bit = hi_bit
        self.lo_bit = lo_bit
        self.hi_fraction = hi_fraction
        self.group_size = group_size
        self.quantize = quantize

        self._bounds: list[tuple[int, int]] = []  # head-group ranges
        self._V: list[mx.array] = []  # per group [D, r]
        self._mu: list[mx.array] = []  # per group [D]
        self._sv: list[mx.array] = []  # per group [r]
        self._head_group: list[int] = []  # head -> group index
        self._r: int = 0  # rank (uniform across groups)
        # Latent buffer: list over heads, each a growing [S, r] fp16 array.
        self._latents: Optional[list[mx.array]] = None
        self._fitted = False

    # ------------------------------------------------------------------
    def fit_prefill(self, x: mx.array) -> None:
        """Fit group projections from the prefill batch ``x`` [B, H, S, D]."""
        B, H, S, D = x.shape
        self._bounds = head_group_bounds(H, self.n_head_groups)
        self._head_group = [0] * H
        ranks = []
        for g, (lo, hi) in enumerate(self._bounds):
            # Stack this group's heads (batch 0) → [G, S, D]
            x_g = x[0, lo:hi].astype(mx.float32)
            V, mu, sv = group_head_svd(x_g, rank=self.rank, energy_threshold=self.energy_threshold)
            mx.eval(V, mu, sv)
            self._V.append(V)
            self._mu.append(mu)
            self._sv.append(sv)
            ranks.append(int(V.shape[1]))
            for h in range(lo, hi):
                self._head_group[h] = g
        # Use a uniform rank across groups for clean buffer shapes: the min
        # retained rank (any group with a larger rank is truncated to match).
        self._r = min(ranks) if ranks else D
        for g in range(len(self._V)):
            if self._V[g].shape[1] > self._r:
                self._V[g] = self._V[g][:, : self._r]
                self._sv[g] = self._sv[g][: self._r]
        self._fitted = True

    # ------------------------------------------------------------------
    def _encode_head(self, x_hd: mx.array, h: int) -> mx.array:
        """Project + (optionally) quantize one head's [S, D] → [S, r] fp16."""
        g = self._head_group[h]
        L = project_to_latent(x_hd, self._V[g], self._mu[g])  # [S, r] fp32
        if self.quantize:
            L = quantize_latent(
                L,
                self._sv[g],
                hi_bit=self.hi_bit,
                lo_bit=self.lo_bit,
                hi_fraction=self.hi_fraction,
                group_size=self.group_size,
            )
        return L.astype(mx.float16)

    def append(self, x: mx.array) -> None:
        """Project + quantize ``x`` [B, H, S, D] and grow the latent buffers."""
        B, H, S, D = x.shape
        encoded = [self._encode_head(x[0, h].astype(mx.float32), h) for h in range(H)]
        if self._latents is None:
            self._latents = encoded
        else:
            self._latents = [
                mx.concatenate([self._latents[h], encoded[h]], axis=0) for h in range(H)
            ]

    def reconstruct(self) -> mx.array:
        """Reconstruct full fp16 keys/values [1, H, S, D] from latent buffers."""
        assert self._latents is not None
        heads = []
        for h, L in enumerate(self._latents):
            g = self._head_group[h]
            heads.append(reconstruct_from_latent(L, self._V[g], self._mu[g]))  # [S, D]
        return mx.stack(heads, axis=0)[None]  # [1, H, S, D]

    # ------------------------------------------------------------------
    @property
    def stored_rank(self) -> int:
        return self._r

    def latent_bytes(self, n_tokens: int, H: int) -> int:
        """Real stored size of the quantized latent buffer for ``n_tokens``."""
        r = self._r if self._r > 0 else 0
        if r == 0:
            return 0
        if not self.quantize:
            # fp16 latents — still a (D/r) win over full fp16.
            return n_tokens * r * 2 * H
        n_hi = max(1, int(r * self.hi_fraction))
        n_lo = r - n_hi
        code_bytes = math.ceil(n_tokens * n_hi * self.hi_bit / 8) + math.ceil(
            n_tokens * n_lo * self.lo_bit / 8
        )
        n_groups = math.ceil(n_tokens / self.group_size)
        param_bytes = n_groups * r * 2 * 2  # scale + zero, fp16, per channel
        return (code_bytes + param_bytes) * H

    def projection_bytes(self, D: int, H: int) -> int:
        """Per-group V [D, r] + mu [D] (fp32), shared across heads in a group."""
        n_groups = len(self._V)
        r = self._r if self._r > 0 else D
        return n_groups * (D * r + D) * 4

    @property
    def assigned_avg_bits(self) -> float:
        D = self._mu[0].shape[0] if self._mu else 0
        if self._r == 0 or D == 0:
            return 16.0
        if not self.quantize:
            return 16.0 * self._r / D
        n_hi = max(1, int(self._r * self.hi_fraction))
        n_lo = self._r - n_hi
        weighted = (n_hi * self.hi_bit + n_lo * self.lo_bit) / self._r
        return weighted * self._r / D


class PALUKVCache(_MLXKVCache):
    """KV cache implementing PALU true-latent low-rank K/V compression.

    Args:
        config: :class:`KVCacheConfig`.  Fields consumed:
            ``head_dim`` (D),
            ``palu_rank`` (int | None — explicit rank; None → energy threshold),
            ``palu_energy_threshold`` (float, default 0.90),
            ``palu_n_head_groups`` (int, default 4),
            ``palu_hi_bit`` / ``palu_lo_bit`` / ``palu_hi_fraction`` (mixed-bit),
            ``palu_group_size`` (int, default 32),
            ``palu_quantize_values`` (bool, default True).

    Notes:
        Does not expose ``.bits`` — ``mlx_lm``'s SDPA checks
        ``hasattr(cache, "bits")`` to route to a quantized kernel; we keep that
        path clean and hand it reconstructed fp16.
        Exposes ``.assigned_avg_bits`` (max of key/value effective bit-widths).
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._D = int(config.head_dim)
        rank = getattr(config, "palu_rank", None)
        et = float(getattr(config, "palu_energy_threshold", 0.90))
        n_groups = int(getattr(config, "palu_n_head_groups", 4))
        hi_bit = int(getattr(config, "palu_hi_bit", 4))
        lo_bit = int(getattr(config, "palu_lo_bit", 2))
        hi_frac = float(getattr(config, "palu_hi_fraction", 0.25))
        if not 0.0 <= hi_frac <= 1.0:
            raise ValueError(f"palu: palu_hi_fraction must be in [0, 1], got {hi_frac}")
        gsize = int(getattr(config, "palu_group_size", 32))
        quant_values = bool(getattr(config, "palu_quantize_values", True))

        self._keys_lr = _TensorLowRank(
            n_groups, rank, et, hi_bit, lo_bit, hi_frac, gsize, quantize=True
        )
        self._vals_lr = _TensorLowRank(
            n_groups, rank, et, hi_bit, lo_bit, hi_frac, gsize, quantize=quant_values
        )

        # We bypass the parent fp16 ring buffer; track offset ourselves.
        self._palu_offset = 0
        self._H = 0

        # Byte accounting
        self._compressed_key_bytes = 0
        self._compressed_value_bytes = 0
        self._fp16_key_bytes = 0
        self._fp16_value_bytes = 0
        self._projection_bytes = 0

    # ------------------------------------------------------------------
    # mlx_lm protocol — true latent storage (parent fp16 buffer bypassed)
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        B, H, S, D = keys.shape
        self._H = H

        if not self._keys_lr._fitted:
            self._keys_lr.fit_prefill(keys)
            self._vals_lr.fit_prefill(values)

        self._keys_lr.append(keys)
        self._vals_lr.append(values)
        self._palu_offset += S

        k_out = self._keys_lr.reconstruct()  # [1, H, total, D] fp16
        v_out = self._vals_lr.reconstruct()
        self._account_bytes(H, S, D)
        return k_out, v_out

    def _account_bytes(self, H: int, S: int, D: int) -> None:
        n = self._palu_offset
        # Recompute realised latent storage for the full sequence so the ratio
        # reflects the actual stored low-rank buffer, not a per-call delta.
        self._compressed_key_bytes = self._keys_lr.latent_bytes(
            n, H
        ) + self._keys_lr.projection_bytes(D, H)
        self._compressed_value_bytes = self._vals_lr.latent_bytes(
            n, H
        ) + self._vals_lr.projection_bytes(D, H)
        self._projection_bytes = self._keys_lr.projection_bytes(
            D, H
        ) + self._vals_lr.projection_bytes(D, H)
        self._fp16_key_bytes = n * H * D * 2
        self._fp16_value_bytes = n * H * D * 2

    # ------------------------------------------------------------------
    # mlx_lm KVCache surface — keep consistent with our own offset
    # ------------------------------------------------------------------
    @property
    def offset(self) -> int:  # type: ignore[override]
        return self._palu_offset

    @offset.setter
    def offset(self, v: int) -> None:
        self._palu_offset = int(v)

    def size(self) -> int:
        return self._palu_offset

    @property
    def nbytes(self) -> int:
        return self._compressed_key_bytes + self._compressed_value_bytes

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
    def projection_bytes(self) -> int:
        return self._projection_bytes

    @property
    def rank(self) -> int:
        """Latent rank actually stored (uniform across head groups)."""
        return self._keys_lr.stored_rank

    @property
    def assigned_avg_bits(self) -> float:
        """Effective bit-width — max of the key and value latent rates."""
        return max(self._keys_lr.assigned_avg_bits, self._vals_lr.assigned_avg_bits)


__all__ = ["PALUKVCache"]
