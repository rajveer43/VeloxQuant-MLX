"""NSNQuant KV cache wrapper for mlx_lm integration.

Implements "NSNQuant: A Double Normalization Approach for Calibration-Free
Low-Bit Vector Quantization of KV Cache" (Son, Choi, Yoo — NeurIPS 2025;
arXiv:2505.18231) on top of the standard mlx_lm ``update_and_fetch``
protocol. Documented as "NSNQuant-adapted (VeloxQuant-MLX implementation)" —
not a faithful port (post-RoPE keys, explicit value Hadamard, k-means-only
codebook, fp16 metadata; see the module docstring of
``veloxquant_mlx/quantizers/nsnquant.py`` and ``paper/NEW_METHOD_SURVEY_V11.md``).

Chunk-flush residual buffer (the paper's decode-time story, and the same
idiom as KIVI's fp16 residual window): tokens accumulate at fp16; every time
``nsn_residual_length`` tokens age past the quantized frontier, that chunk is
round-tripped through NSN + Hadamard + universal-codebook VQ **as one
self-contained unit** — the chunk computes its own channel mean ``o``, so no
statistics are frozen, no coordinator exists, and chunk *i* is forever
independent of anything that arrives later. Prefill and decode produce
identical chunk boundaries by construction (the flush frontier only ever
advances in whole chunks), so the quantized state is path-independent.

Like every method in this repo, the quantize→dequantize round-trip happens
inside ``update_and_fetch`` so the downstream SDPA call sees standard fp16
tensors. Both keys **and** values are quantized (mirroring the paper), unlike
the keys-only SVDq/xKV precedent. The paper's throughput gains come from
fused CUDA kernels that do not port to Metal — on Apple Silicon the win is
*memory*, measured honestly by the byte accounting below.

Per-token storage at ``nsn_bits`` b for head_dim D (per tensor):
``D * b / 8`` bytes of codes (2-bit: uint8 sign mask + uint8 index per 8-dim
subvector; 1-bit: index only) + 4 bytes fp16 (s1, s2) per token + ``2 * D``
bytes fp16 (channel mean o) amortized per chunk. All three metadata terms are
counted in ``compressed_*_bytes`` — not waved away (the paper 4-bit
double-quantizes them; we do not).
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.math.rotation import is_hadamard_compatible
from veloxquant_mlx.quantizers.nsnquant import (
    build_universal_codebook,
    hadamard_forward,
    hadamard_inverse,
    nsn_inverse,
    nsn_transform,
    vq_decode,
    vq_encode,
)


class NSNQuantKVCache(_MLXKVCache):
    """KV cache implementing NSNQuant universal-codebook VQ for one layer.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed: ``head_dim``,
            ``nsn_bits`` (2 = sign mask + index, 1 = index only),
            ``nsn_residual_length`` (chunk size / fp16 window; the paper
            recommends 128 for 1-bit), ``nsn_codebook_size``,
            ``nsn_subvector_dim``, ``nsn_seed``, ``nsn_max_ctx``.

    Notes:
        Never exposes ``.bits`` — mlx_lm's SDPA checks
        ``hasattr(cache, "bits")`` to route to a quantized kernel path.
        We expose ``.assigned_avg_bits`` instead.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._D = int(config.head_dim)
        self._bits = int(getattr(config, "nsn_bits", 2))
        self._residual_length = int(getattr(config, "nsn_residual_length", 64))
        self._codebook_size = int(getattr(config, "nsn_codebook_size", 256))
        self._sub_d = int(getattr(config, "nsn_subvector_dim", 8))
        self._seed = int(getattr(config, "nsn_seed", 1234))
        self._max_ctx = int(getattr(config, "nsn_max_ctx", 8192))

        # Fail at build time, not on the first update (clear messages).
        if self._bits not in (1, 2):
            raise ValueError(f"NSNQuantKVCache: nsn_bits must be 1 or 2, got {self._bits}")
        if self._D % self._sub_d != 0:
            raise ValueError(
                f"NSNQuantKVCache: head_dim {self._D} must be divisible by "
                f"nsn_subvector_dim {self._sub_d}"
            )
        if not is_hadamard_compatible(self._D):
            raise ValueError(
                f"NSNQuantKVCache: head_dim {self._D} unsupported by "
                f"mx.hadamard_transform (needs d = m * 2^k, m in "
                f"{{1, 12, 20, 28}})"
            )
        if self._residual_length < 2:
            raise ValueError(
                "NSNQuantKVCache: nsn_residual_length must be >= 2 (a chunk "
                "must contain enough tokens for a meaningful channel mean)"
            )

        # Universal codebook — model/data independent (synthetic Gaussian),
        # deterministic per (size, sub_d, seed, kind); cached module-wide.
        kind = "magnitude" if self._bits == 2 else "signed"
        self._cb = build_universal_codebook(
            codebook_size=self._codebook_size,
            subvector_dim=self._sub_d,
            seed=self._seed,
            kind=kind,
        )

        # Quantized frontier: tokens [0, _q_end) have been chunk-flushed
        # (their fp16 storage overwritten with the dequantized round-trip).
        # Always a multiple of _residual_length.
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
    # NSN + Hadamard + VQ round-trip for one self-contained chunk
    # ------------------------------------------------------------------
    def _round_trip(self, x: mx.array) -> mx.array:
        x_nsn, s1, o, s2 = nsn_transform(x)
        enc = vq_encode(hadamard_forward(x_nsn), self._cb, self._bits)
        dec = vq_decode(enc, self._cb)
        return nsn_inverse(hadamard_inverse(dec), s1, o, s2).astype(x.dtype)

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys, values):
        """Append the incoming block, then flush every completed chunk.

        The flush overwrites the aged-out tokens' fp16 storage in place with
        their dequantized round-trip, so decode-time tokens *do* get
        quantized once they age past the chunk frontier (unlike KIVI's
        incoming-block-only simplification).
        """
        B, H, S, D = keys.shape
        if self.offset + S > self._max_ctx:
            raise ValueError(
                f"NSNQuantKVCache: context {self.offset + S} exceeds nsn_max_ctx={self._max_ctx}"
            )
        self._B, self._H = B, H

        super().update_and_fetch(keys, values)

        r = self._residual_length
        while self.offset - self._q_end >= r:
            s, e = self._q_end, self._q_end + r
            k_chunk = self.keys[..., s:e, :]
            v_chunk = self.values[..., s:e, :]
            self.keys[..., s:e, :] = self._round_trip(k_chunk)
            self.values[..., s:e, :] = self._round_trip(v_chunk)
            self._q_end = e
            self._account_chunk_bytes(B, H, r, D)

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
    def _account_chunk_bytes(self, B: int, H: int, r: int, D: int) -> None:
        n_sub = D // self._sub_d
        # Payload: 2-bit = sign mask + index (2 uint8 per subvector);
        #          1-bit = index only (1 uint8 per subvector).
        payload = r * n_sub * (2 if self._bits == 2 else 1)
        # Metadata, fp16: s1 + s2 per token, o (channel mean) per chunk.
        metadata = r * 2 * 2 + D * 2
        per_tensor = (payload + metadata) * B * H
        self._compressed_key_bytes += per_tensor
        self._compressed_value_bytes += per_tensor

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
        """Bytes currently held in the fp16 residual window (keys + values) —
        a snapshot of the un-flushed tail, not a cumulative counter."""
        n_res = self.offset - self._q_end
        return n_res * self._D * 2 * 2 * self._B * self._H

    @property
    def quantized_tokens(self) -> int:
        """Tokens behind the chunk-flush frontier (multiple of the chunk size)."""
        return self._q_end

    @property
    def assigned_avg_bits(self) -> float:
        """Effective bits/element over the quantized region, including the
        fp16 s1/s2/o metadata (excludes the fp16 residual window; for an
        end-to-end ratio use ``(compressed_*_bytes + residual_fp16_bytes) /
        fp16_*_bytes``)."""
        if self._q_end == 0:
            return 16.0
        elems = self._q_end * self._D * self._B * self._H
        return 8.0 * self._compressed_key_bytes / elems


__all__ = ["NSNQuantKVCache"]
