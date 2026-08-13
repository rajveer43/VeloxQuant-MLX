"""KVTC quantizer — local PCA + DP-optimal per-component bit allocation + entropy coding.

Inspired by "KV Cache Transform Coding for Compact Storage in LLM Inference"
(NVIDIA, **ICLR 2026**, accepted poster, arXiv:2511.01815). Documented as
"KVTC-adapted (VeloxQuant-MLX implementation)" — not a faithful port. See
``cache/kvtc_cache.py`` for the cache wrapper and
``allocators/kvtc_dp.py`` / ``quantizers/_entropy_coding.py`` for the two new
building blocks this method introduces.

Modeled on the Palu / SVDq pair already shipped
(``quantizers/palu.py`` + ``cache/palu_cache.py``,
``quantizers/svdq.py`` + ``cache/svdq_cache.py``): local (per-sequence)
SVD/PCA fit at prefill, latent projection, per-component quantization of
latent channels. What is NEW here (not in Palu/SVDq/SpectralQuant) is *how*
the per-component bit-width is chosen (DP-optimal under a hard budget,
**may be exactly 0**, i.e. a component is dropped entirely) and an extra
lossless **entropy-coding** stage over the quantized codes.

THE HONESTY CRUX (read before trusting any number)
----------------------------------------------------
1. **Local (per-sequence) PCA, not the paper's pre-calibrated global basis.**
   The paper fits one PCA basis offline on a calibration corpus and reuses
   it for all future caches at inference. This library has no calibration
   pipeline wired into ``KVCacheBuilder.for_model``, so the basis is fit
   **online from the sequence's own prefill keys/values** — the same
   "fit-locally, no calibration set" limitation SVDq/Palu already document.
2. **The DP allocator optimizes an analytic distortion proxy, not a
   real-activation-fit rate-distortion model.** ``allocators/kvtc_dp.py``'s
   DP is exact and real (it correctly finds the budget-constrained minimum
   of the proxy it is given); what is a *proxy* is the objective itself —
   the reused ``D(v, b) = v * beta ** (-b)`` Gaussian-quantization
   distortion curve from ``allocators/ratequant.py``, not a curve fit on
   real LLM activation statistics as the paper does.
3. **Entropy coding is a real, measured, lossless order-0 Huffman coder**
   (``quantizers/_entropy_coding.py``), not the paper's (possibly more
   sophisticated) scheme, and not the theoretical Shannon-entropy bound. We
   report the **realized** post-entropy-coding byte count, including the
   code table's own storage cost.
4. **Both K and V**, mirroring Palu (not SVDq's keys-only scope) — the
   paper compresses both tensors.
5. **Not path-dependent** (unlike the eviction family
   H2O/TOVA/MorphKV/KVzip): the PCA basis and DP-derived bit allocation are
   fixed once at prefill and reused, unchanged, for every subsequent decode
   token — see ``cache/kvtc_cache.py``'s determinism test.
6. Nothing here is validated on a trained model. The paper's headline
   numbers (up to 20×, up to 40× in some regimes, <1pp accuracy loss on
   LLaMA 3 / Mistral NeMo / R1-Qwen2.5 1.5B-70B across AIME25, GSM8K,
   LiveCodeBench, LongBench, MATH-500, MMLU, Qasper, RULER) are the
   **paper's**, on trained models — never quoted as this repo's own.

Algorithm
---------
Compress (``kvtc_compress``):
  1. Center ``x [S, D]`` and run truncated SVD (reusing
     ``quantizers/_quant_utils.py::_truncated_svd``, the same shared helper
     SVDq/Palu/GEAR already use) with **no fixed-energy truncation** —
     ``r = min(S, D)``. The DP allocator itself decides how many components
     survive by assigning some of them exactly 0 bits.
  2. Project to latents ``L = (x - mean) @ V -> [S, r]``.
  3. Per-component variance ``v_i = s_i^2 / S`` (sample variance along
     principal axis ``i``, from the singular values).
  4. Call ``allocators.kvtc_dp.dp_allocate_bits(variances, total_bit_budget)``
     to get an integer bit-width per component (may be 0).
  5. Quantize each surviving (bits > 0) component independently with a
     min/max affine integer coder; 0-bit components are **dropped from
     storage entirely** — no zero-filled placeholder is stored.
  6. Entropy-code the concatenated integer codes of all surviving
     components (order-0 Huffman, ``quantizers/_entropy_coding.py``).

Decompress (``kvtc_decompress``): entropy-decode, dequantize each surviving
component, zero-fill dropped components, un-project
(``latents @ V.T + mean``), return ``[S, D]``.

Entropy-coding fallback (#77)
------------------------------
Order-0 Huffman only beats fixed-width packing when the symbol distribution
is meaningfully non-uniform. Quantized PCA-latent codes from Gaussian-ish KV
data land close to uniform across their ``2**bits`` levels (the point of
min/max affine quantization is to spread values evenly), so Huffman coding
routinely has *nothing* to gain — and its per-call code table is pure
overhead on top of that. ``kvtc_compress`` therefore compares the realized
Huffman-coded size (payload + table) against plain fixed-width packing and
keeps whichever is smaller, recording the choice on
``KVTCArtifact.is_entropy_coded`` so ``kvtc_decompress`` knows which path to
invert. This mirrors the "never worse than fixed-width" guarantee
``quantizers/cachegen.py::entropy_coded_bytes`` already provides for its own
(estimate-only) accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import numpy as np

from veloxquant_mlx.allocators.kvtc_dp import DEFAULT_BETA, DEFAULT_BIT_CHOICES, dp_allocate_bits
from veloxquant_mlx.quantizers._entropy_coding import (
    entropy_decode,
    entropy_encode,
    table_nbytes,
)
from veloxquant_mlx.quantizers._quant_utils import _truncated_svd

_EPS = 1e-8


@dataclass
class KVTCArtifact:
    """Everything needed to decompress a KVTC-compressed ``[S, D]`` tensor.

    Attributes:
        V: Projection basis ``[D, r]`` fp32 (right singular vectors).
        mean: Mean vector ``[D]`` fp32, subtracted before projection.
        bit_allocation: Per-component bit-width ``[r]`` int, **may contain
            zeros** (dropped components).
        S: Number of rows (tokens) in the original tensor — needed to
            un-flatten the entropy-decoded stream and to reconstruct
            dropped (all-zero) columns of the right shape.
        n_survived: Number of components with ``bit_allocation > 0``
            (``== (bit_allocation > 0).sum()``, cached for convenience).
        entropy_payload: The coded bitstream for all surviving components'
            codes, concatenated component-major (see ``kvtc_compress``) —
            **one combined blob**, not per-component, to amortize the
            (small) fixed per-call table overhead. Huffman-coded when
            ``is_entropy_coded`` is True, else fixed-width packed (each
            component's ``S`` codes packed at its own bit-width, byte-aligned
            per component — see :func:`_pack_fixed_width`).
        entropy_table: The Huffman code table returned alongside
            ``entropy_payload`` when ``is_entropy_coded`` is True
            (``dict[int, str]``); empty when fixed-width packing was used
            instead. Its storage cost is counted in ``kvtc_fp16_bytes`` —
            never hidden.
        is_entropy_coded: Whether ``entropy_payload`` is Huffman-coded
            (True) or fixed-width packed (False). ``kvtc_compress`` picks
            whichever is smaller — Huffman coding only wins on real,
            non-uniform code distributions; see the module docstring (#77).
        quant_min: Per-surviving-component min value used for dequant,
            shape ``[n_survived]`` fp32.
        quant_scale: Per-surviving-component quantization scale, shape
            ``[n_survived]`` fp32.
        survived_idx: Original component indices (into the ``r`` axis) of
            the surviving components, shape ``[n_survived]`` int.
    """

    V: mx.array
    mean: mx.array
    bit_allocation: np.ndarray
    S: int
    n_survived: int
    entropy_payload: bytes
    entropy_table: dict
    quant_min: np.ndarray
    quant_scale: np.ndarray
    survived_idx: np.ndarray
    is_entropy_coded: bool = True


def quantize_component(col: np.ndarray, bits: int) -> tuple[np.ndarray, float, float]:
    """Min/max affine-quantize one component column ``[S]`` at ``bits`` bits.

    Returns ``(codes, min_val, scale)`` where ``codes`` are non-negative
    integers in ``[0, 2**bits - 1]`` and
    ``dequantized ~= codes * scale + min_val``.
    """
    levels = (1 << bits) - 1
    lo = float(col.min())
    hi = float(col.max())
    scale = max((hi - lo) / levels, _EPS)
    codes = np.clip(np.round((col - lo) / scale), 0, levels).astype(np.int64)
    return codes, lo, scale


def _pack_fixed_width(codes: np.ndarray, bits: int) -> bytes:
    """Pack a ``[S]`` array of non-negative integer codes at ``bits`` bits each.

    Byte-aligned (padded with trailing zero bits to a whole byte) so a
    multi-component concatenation of these packs stays trivially seekable —
    matches the per-component ``ceil(S * bits / 8)`` accounting already used
    by :func:`kvtc_pre_entropy_bytes`.
    """
    if codes.shape[0] == 0 or bits == 0:
        return b""
    bitstring = "".join(format(int(c), f"0{bits}b") for c in codes.tolist())
    pad = (-len(bitstring)) % 8
    bitstring += "0" * pad
    return int(bitstring, 2).to_bytes(len(bitstring) // 8, byteorder="big")


def _unpack_fixed_width(payload: bytes, bits: int, n: int) -> np.ndarray:
    """Exact inverse of :func:`_pack_fixed_width` for ``n`` codes at ``bits`` bits."""
    if n == 0 or bits == 0:
        return np.zeros((n,), dtype=np.int64)
    n_bits = len(payload) * 8
    bitstring = bin(int.from_bytes(payload, byteorder="big"))[2:].zfill(n_bits)
    out = np.empty((n,), dtype=np.int64)
    for i in range(n):
        out[i] = int(bitstring[i * bits : (i + 1) * bits], 2)
    return out


def _encode_survived_codes(
    all_codes: list[np.ndarray], bits_per_component: list[int]
) -> tuple[bytes, dict, bool]:
    """Encode surviving components' codes, falling back to fixed-width packing
    when Huffman coding doesn't pay for itself (#77).

    Args:
        all_codes: Per-surviving-component ``[S]`` int64 code arrays.
        bits_per_component: The bit-width used to quantize each entry of
            ``all_codes`` (parallel list, same length).

    Returns:
        ``(payload, table, is_entropy_coded)``. ``table`` is ``{}`` when
        fixed-width packing was chosen.
    """
    flat_codes = np.concatenate(all_codes) if all_codes else np.zeros((0,), dtype=np.int64)

    entropy_payload, entropy_table = entropy_encode(flat_codes)
    entropy_total = len(entropy_payload) + table_nbytes(entropy_table)

    fixed_payload = b"".join(
        _pack_fixed_width(codes, bits) for codes, bits in zip(all_codes, bits_per_component)
    )
    fixed_total = len(fixed_payload)

    if fixed_total < entropy_total:
        return fixed_payload, {}, False
    return entropy_payload, entropy_table, True


def kvtc_compress(
    tensor: mx.array,
    total_bit_budget: int,
    bit_choices: tuple[int, ...] = DEFAULT_BIT_CHOICES,
    beta: float = DEFAULT_BETA,
) -> KVTCArtifact:
    """Local PCA + DP-optimal bit allocation + entropy coding for ``[S, D]``.

    Args:
        tensor: ``[S, D]`` fp16/fp32 keys or values for one (batch, head).
        total_bit_budget: Total integer bits across all
            ``r = min(S, D)`` principal components (per token — the same
            budget applies to every row of the ``[S, r]`` latent matrix).
        bit_choices: Allowed per-component bit-widths, passed through to
            :func:`veloxquant_mlx.allocators.kvtc_dp.dp_allocate_bits`.
        beta: Distortion decay constant, passed through to the DP allocator.

    Returns:
        :class:`KVTCArtifact`.

    Raises:
        ValueError: delegated from :func:`dp_allocate_bits` for a negative
            budget, or if ``tensor`` has fewer than 1 row.
    """
    if tensor.shape[0] < 1:
        raise ValueError(f"kvtc_compress: tensor must have S >= 1, got shape {tensor.shape!r}")

    x = tensor.astype(mx.float32)
    S, D = int(x.shape[0]), int(x.shape[1])
    mean = mx.mean(x, axis=0)  # [D]
    x_centered = x - mean[None, :]

    # No fixed-energy truncation: r = min(S, D). The DP allocator itself
    # decides how many components survive by assigning some of them 0 bits.
    r = min(S, D)
    U, s_vals, Vt = _truncated_svd(x_centered, rank=r)
    mx.eval(U, s_vals, Vt)
    V = Vt.T  # [D, r]

    L = x_centered @ V  # [S, r]
    mx.eval(L)

    s_np = np.asarray(s_vals.tolist(), dtype=np.float64)
    variances = (s_np**2) / max(S, 1)  # per-component sample variance

    bit_alloc = dp_allocate_bits(variances, total_bit_budget, bit_choices=bit_choices, beta=beta)

    L_np = np.asarray(L.tolist(), dtype=np.float64)
    survived_idx = np.where(bit_alloc > 0)[0]

    all_codes: list[np.ndarray] = []
    mins: list[float] = []
    scales: list[float] = []
    bits_per_component: list[int] = []
    for i in survived_idx:
        col = L_np[:, i]
        bits = int(bit_alloc[i])
        codes, lo, scale = quantize_component(col, bits)
        all_codes.append(codes)
        mins.append(lo)
        scales.append(scale)
        bits_per_component.append(bits)

    payload, table, is_entropy_coded = _encode_survived_codes(all_codes, bits_per_component)

    return KVTCArtifact(
        V=V,
        mean=mean,
        bit_allocation=bit_alloc,
        S=S,
        n_survived=int(survived_idx.shape[0]),
        entropy_payload=payload,
        entropy_table=table,
        quant_min=np.asarray(mins, dtype=np.float64),
        quant_scale=np.asarray(scales, dtype=np.float64),
        survived_idx=survived_idx,
        is_entropy_coded=is_entropy_coded,
    )


def kvtc_decompress(artifact: KVTCArtifact) -> mx.array:
    """Inverse of :func:`kvtc_compress`. Returns ``[S, D]`` fp16.

    Decodes the combined code stream (Huffman or fixed-width, per
    ``artifact.is_entropy_coded`` — see #77), dequantizes each surviving
    component, zero-fills dropped components, and un-projects:
    ``latents @ V.T + mean``.
    """
    S = artifact.S
    r = int(artifact.bit_allocation.shape[0])
    D = int(artifact.V.shape[0])

    n_survived = artifact.n_survived
    if artifact.is_entropy_coded:
        flat_codes = entropy_decode(
            artifact.entropy_payload, artifact.entropy_table, n_survived * S
        )
    else:
        payload = artifact.entropy_payload
        pieces = []
        offset = 0
        for i in artifact.survived_idx:
            bits = int(artifact.bit_allocation[i])
            nbytes = -(-(S * bits) // 8)  # ceil division, matches _pack_fixed_width's padding
            pieces.append(_unpack_fixed_width(payload[offset : offset + nbytes], bits, S))
            offset += nbytes
        flat_codes = np.concatenate(pieces) if pieces else np.zeros((0,), dtype=np.int64)

    L_np = np.zeros((S, r), dtype=np.float64)
    for k, i in enumerate(artifact.survived_idx):
        codes = flat_codes[k * S : (k + 1) * S]
        lo = artifact.quant_min[k]
        scale = artifact.quant_scale[k]
        L_np[:, i] = codes.astype(np.float64) * scale + lo

    L = mx.array(L_np.astype(np.float32))
    x_hat = L @ artifact.V.T + artifact.mean[None, :]
    return x_hat.astype(mx.float16)


def kvtc_pre_entropy_bytes(artifact: KVTCArtifact) -> int:
    """Bytes the quantized codes would cost at **fixed-width packing**,
    i.e. before entropy coding — the pre-entropy-coding size the benchmark
    diffs against ``kvtc_fp16_bytes`` to report entropy coding's realized
    delta.

    Sum over surviving components of ``ceil(S * bits / 8)`` (fixed-width
    packed codes), NOT counting the (small) per-component min/scale, which
    are counted identically in both accountings so they cancel in the
    entropy-coding-gain ratio.
    """
    S = artifact.S
    total = 0
    for i in artifact.survived_idx:
        bits = int(artifact.bit_allocation[i])
        total += -(-(S * bits) // 8)  # ceil division
    return total


def kvtc_fp16_bytes(artifact: KVTCArtifact) -> int:
    """Realized total stored bytes: projection ``V`` + ``mean`` (fp32) +
    per-surviving-component quant params (min/scale, fp32) +
    **realized entropy-coded payload** + **entropy code table** (never
    hidden). This is the actual stored size, NOT the pre-entropy-coding
    fixed-width size — see :func:`kvtc_pre_entropy_bytes` for that.
    """
    D, r = int(artifact.V.shape[0]), int(artifact.V.shape[1])
    projection_bytes = (D * r + D) * 4  # V + mean, fp32
    n_survived = artifact.n_survived
    quant_param_bytes = n_survived * (4 + 4)  # min + scale, fp32, per surviving component
    payload_bytes = len(artifact.entropy_payload)
    table_bytes = table_nbytes(artifact.entropy_table)
    return projection_bytes + quant_param_bytes + payload_bytes + table_bytes


__all__ = [
    "KVTCArtifact",
    "kvtc_compress",
    "kvtc_decompress",
    "kvtc_fp16_bytes",
    "kvtc_pre_entropy_bytes",
    "quantize_component",
]
