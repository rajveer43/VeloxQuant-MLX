"""CacheGen quantizer — delta-locality group quant + entropy-coded byte model.

Inspired by "CacheGen: KV Cache Compression and Streaming for Fast Large
Language Model Serving" (Liu et al., **SIGCOMM 2024**, arXiv:2310.07240).
Documented as "CacheGen-adapted (VeloxQuant-MLX implementation)" — not a
faithful port.

What CacheGen adds that the repo did not have: **entropy coding** of the
quantized KV. Every other method in the suite packs codes at a fixed bit-width;
CacheGen exploits the KV cache's *distributional* structure to encode the codes
into a smaller bitstream. Its three observations:

  1. **Token-wise locality** — adjacent tokens' KV vectors are similar, so the
     *delta* between consecutive tokens' quantized codes is concentrated near
     zero and is far more compressible than the raw codes.
  2. **Layer-wise sensitivity** — deeper layers tolerate coarser quantization;
     CacheGen spends fewer bits on later layers.
  3. **Arithmetic coding** — the delta symbol stream, being low-entropy, is
     compressed with an entropy coder down toward its Shannon entropy.
  4. **Channel/layer grouping** — fitting a separate symbol distribution per
     channel (§5.1.3) yields much lower entropy than pooling all channels
     together; layer separation is already implicit (each cache instance
     owns one layer).

Adaptation:
  * The reconstruction path is the existing asymmetric min/max group quant — the
    *values* the model sees are identical to KIVI-style quant (no extra loss
    from the entropy layer; entropy coding is lossless over the codes).
  * We do **not** ship a per-step arithmetic codec (a serial range coder would
    bottleneck MLX's parallel decode and add no quality). Instead we model the
    entropy-coded byte size from the **measured Shannon entropy** of the
    delta-coded symbol stream — an honest lower-ish bound on what a real
    arithmetic coder achieves, reported through ``compressed_*_bytes``.
  * Layer-wise bit selection is exposed via config (``cachegen_bits`` plus an
    optional per-depth schedule applied by the builder).

This module holds the pure numerics: group quant exposing integer codes,
token-delta transform, and the entropy/byte estimator.  The cache wrapper owns
the per-layer state and accounting.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import mlx.core as mx


class CodeStream(NamedTuple):
    """Quantized codes plus the params needed to dequantize them.

    Attributes:
        codes:  [n_groups, group_size, D] fp32 integer codes in [0, 2^bits-1].
        scale:  [n_groups, 1, D] fp32 per-group step.
        zero:   [n_groups, 1, D] fp32 per-group min.
        n_rows: int original (pre-pad) token count.
        bits:   int bit-width.
    """

    codes: mx.array
    scale: mx.array
    zero: mx.array
    n_rows: int
    bits: int


def _pad_to_groups(x32: mx.array, group_size: int) -> tuple[mx.array, int, int]:
    n, d = x32.shape
    n_groups = (n + group_size - 1) // group_size
    pad = n_groups * group_size - n
    if pad:
        x32 = mx.concatenate([x32, mx.broadcast_to(x32[-1:], (pad, d))], axis=0)
    return x32, n_groups, n


def quantize_to_codes(x: mx.array, bits: int, group_size: int = 32) -> CodeStream:
    """Asymmetric min/max group quant exposing the integer codes.

    Args:
        x: [N, D] fp16/fp32 (one head's keys or values).
        bits: bit-width.
        group_size: tokens per group along axis 0.

    Returns:
        CodeStream (codes + dequant params).
    """
    x32 = x.astype(mx.float32)
    x32, n_groups, n = _pad_to_groups(x32, group_size)
    d = x32.shape[-1]
    xg = x32.reshape(n_groups, group_size, d)
    gmin = mx.min(xg, axis=1, keepdims=True)
    gmax = mx.max(xg, axis=1, keepdims=True)
    levels = (1 << bits) - 1
    scale = mx.maximum((gmax - gmin) / levels, 1e-8)
    codes = mx.clip(mx.round((xg - gmin) / scale), 0, levels)
    return CodeStream(codes=codes, scale=scale, zero=gmin, n_rows=n, bits=bits)


def dequant_codes(stream: CodeStream) -> mx.array:
    """Reconstruct fp16 [n_rows, D] from a CodeStream."""
    recon = stream.codes * stream.scale + stream.zero
    n_groups, gs, d = recon.shape
    return recon.reshape(n_groups * gs, d)[: stream.n_rows].astype(mx.float16)


def token_delta(codes_flat: mx.array) -> mx.array:
    """Token-wise delta of a flattened code matrix [N, D].

    Row 0 is kept as-is; row i (i>0) becomes ``codes[i] - codes[i-1]``. Because
    adjacent tokens' KV are similar, the deltas concentrate near zero — the
    locality CacheGen exploits. Reversible: a prefix-sum recovers the codes.
    """
    if codes_flat.shape[0] <= 1:
        return codes_flat
    prev = codes_flat[:-1]
    rest = codes_flat[1:] - prev
    return mx.concatenate([codes_flat[:1], rest], axis=0)


def symbol_entropy_bits(symbols: mx.array) -> float:
    """Shannon entropy (bits/symbol) of an integer symbol array.

    This is the per-symbol size an ideal arithmetic coder approaches. Computed
    over the empirical distribution of the (delta) code values.
    """
    s = symbols.astype(mx.int32).reshape(-1)
    n = int(s.shape[0])
    if n == 0:
        return 0.0
    smin = int(mx.min(s).item())
    shifted = (s - smin).astype(mx.int32)
    nbins = int(mx.max(shifted).item()) + 1
    counts = mx.zeros((nbins,), dtype=mx.float32)
    ones = mx.ones((n,), dtype=mx.float32)
    counts = counts.at[shifted].add(ones)
    p = counts / float(n)
    nz = p > 0
    p_nz = mx.where(nz, p, mx.ones_like(p))  # avoid log(0); masked out below
    ent = -mx.sum(mx.where(nz, p * (mx.log(p_nz) / math.log(2.0)), mx.zeros_like(p)))
    return float(ent.item())


def entropy_coded_bytes(
    stream: CodeStream, use_delta: bool = True, per_channel: bool = True
) -> int:
    """Estimate the entropy-coded size (bytes) of a CodeStream's codes.

    Models a real arithmetic coder by measuring the Shannon entropy of the
    (optionally delta-transformed) code symbols and multiplying by the symbol
    count. Params (scale/zero) are added at fp16 like every other method.

    Args:
        stream: the quantized CodeStream.
        use_delta: apply the token-delta transform before measuring entropy.
        per_channel: fit a separate probability distribution per channel
            before measuring entropy (§5.1.3: grouping by channel/layer gives
            much lower entropy than pooling all channels together — the paper
            reports up to 53% smaller bitstreams from this). If False, all
            channels are pooled into one distribution (the coarser estimate).

    Returns:
        Estimated compressed size in bytes (codes via entropy + fp16 params).
        Capped at the fixed-width packed size: a real arithmetic coder never
        stores more than the raw codes (it falls back to raw packing when the
        symbol stream is incompressible), so neither does this estimate.
    """
    n_groups, gs, d = stream.codes.shape
    flat = stream.codes.reshape(n_groups * gs, d)[: stream.n_rows]  # [N, D]
    symbols = token_delta(flat) if use_delta else flat
    if per_channel:
        code_bits = 0.0
        for c in range(d):
            bits_per_sym = min(symbol_entropy_bits(symbols[:, c]), float(stream.bits))
            code_bits += bits_per_sym * stream.n_rows
    else:
        bits_per_sym = min(symbol_entropy_bits(symbols), float(stream.bits))
        code_bits = bits_per_sym * stream.n_rows * d
    code_bytes = math.ceil(code_bits / 8)
    param_bytes = n_groups * d * 2 * 2  # scale + zero, fp16
    return code_bytes + param_bytes


def fixed_width_bytes(stream: CodeStream) -> int:
    """Naive fixed-bit-width packed size (bytes) for the same codes — baseline."""
    n_groups, gs, d = stream.codes.shape
    code_bytes = math.ceil(stream.n_rows * d * stream.bits / 8)
    param_bytes = n_groups * d * 2 * 2
    return code_bytes + param_bytes


def _pad_to_groups_batched(x32: mx.array, group_size: int) -> tuple[mx.array, int, int]:
    g, n, d = x32.shape
    n_groups = (n + group_size - 1) // group_size
    pad = n_groups * group_size - n
    if pad:
        x32 = mx.concatenate([x32, mx.broadcast_to(x32[:, -1:], (g, pad, d))], axis=1)
    return x32, n_groups, n


def quantize_to_codes_batched(x: mx.array, bits: int, group_size: int = 32) -> CodeStream:
    """Batched ``quantize_to_codes`` over a leading ``G = B*H`` axis.

    Args:
        x: ``[G, N, D]`` fp16/fp32 — ``G`` independent heads' keys or values.
        bits: bit-width.
        group_size: tokens per group along axis 1.

    Returns:
        CodeStream whose fields carry the extra leading ``G`` axis:
        ``codes``/``scale``/``zero`` are ``[G, n_groups, group_size (or 1), D]``.
    """
    g = x.shape[0]
    x32 = x.astype(mx.float32)
    x32, n_groups, n = _pad_to_groups_batched(x32, group_size)
    d = x32.shape[-1]
    xg = x32.reshape(g, n_groups, group_size, d)
    gmin = mx.min(xg, axis=2, keepdims=True)
    gmax = mx.max(xg, axis=2, keepdims=True)
    levels = (1 << bits) - 1
    scale = mx.maximum((gmax - gmin) / levels, 1e-8)
    codes = mx.clip(mx.round((xg - gmin) / scale), 0, levels)
    return CodeStream(codes=codes, scale=scale, zero=gmin, n_rows=n, bits=bits)


def dequant_codes_batched(stream: CodeStream) -> mx.array:
    """Batched ``dequant_codes``: reconstruct fp16 ``[G, n_rows, D]``."""
    recon = stream.codes * stream.scale + stream.zero
    g, n_groups, gs, d = recon.shape
    return recon.reshape(g, n_groups * gs, d)[:, : stream.n_rows].astype(mx.float16)


def token_delta_batched(codes_flat: mx.array) -> mx.array:
    """Batched ``token_delta`` over ``[G, N, D]`` — delta along axis 1."""
    if codes_flat.shape[1] <= 1:
        return codes_flat
    prev = codes_flat[:, :-1]
    rest = codes_flat[:, 1:] - prev
    return mx.concatenate([codes_flat[:, :1], rest], axis=1)


def _batched_symbol_entropy_bits(
    symbols: mx.array, alphabet_min: int, alphabet_size: int
) -> mx.array:
    """Shannon entropy (bits/symbol) per leading-axis slice, no host sync.

    Args:
        symbols: ``[G, N]`` int-valued (stored as float32/int32) symbol stream.
        alphabet_min: known lower bound of ``symbols`` (e.g. ``-levels`` for a
            delta stream, ``0`` for a raw code stream) — data-independent, so
            no ``.item()`` read of the actual data is needed to size the
            histogram (codes are clipped to ``[0, levels]`` at quantization
            time, and a token-delta of two such codes is bounded by
            ``[-levels, levels]``; both bounds are static given ``bits``).
        alphabet_size: number of distinct symbol values in range.

    Returns:
        ``[G]`` fp32 entropy in bits/symbol — a lazy MLX array; the caller
        evaluates and reads it back with a single batched ``.tolist()``
        instead of one ``.item()`` per (head, channel).
    """
    g, n = symbols.shape
    shifted = (symbols - alphabet_min).astype(mx.int32)
    # One-hot histogram via matmul instead of per-slice scatter-add: the
    # alphabet is small (<=511 even at 8-bit codes — see quantize_to_codes's
    # clip range), so this stays cheap and needs no data-dependent shape.
    onehot = (shifted[:, :, None] == mx.arange(alphabet_size)[None, None, :]).astype(mx.float32)
    counts = mx.sum(onehot, axis=1)  # [G, alphabet_size]
    p = counts / float(n)
    nz = p > 0
    p_nz = mx.where(nz, p, mx.ones_like(p))
    ent = -mx.sum(mx.where(nz, p * (mx.log(p_nz) / math.log(2.0)), mx.zeros_like(p)), axis=1)
    return ent  # [G]


def entropy_coded_bytes_batched(
    stream: CodeStream, use_delta: bool = True, per_channel: bool = True
) -> list[int]:
    """Batched ``entropy_coded_bytes`` over the CodeStream's leading ``G`` axis.

    Same estimate as :func:`entropy_coded_bytes`, computed for all ``G``
    heads (and, if ``per_channel``, all ``D`` channels) in one vectorized
    pass with a single final host read, instead of ``entropy_coded_bytes``
    called once per head with its own internal per-channel Python loop and
    two ``.item()`` calls per channel (``G * D`` host syncs total there).

    Returns:
        Length-``G`` list of estimated compressed sizes in bytes (codes via
        entropy + fp16 params), each capped at that head's fixed-width size.
    """
    g, n_groups, gs, d = stream.codes.shape
    flat = stream.codes.reshape(g, n_groups * gs, d)[:, : stream.n_rows]  # [G, N, D]
    symbols = token_delta_batched(flat) if use_delta else flat
    levels = (1 << stream.bits) - 1
    alphabet_min = -levels if use_delta else 0
    alphabet_size = 2 * levels + 1 if use_delta else levels + 1

    if per_channel:
        # [G, D, N] so the batched entropy fn's leading axis covers every
        # (head, channel) pair in one pass — no Python loop over channels.
        per_ch = mx.moveaxis(symbols, 2, 1).reshape(g * d, stream.n_rows)
        ent = _batched_symbol_entropy_bits(per_ch, alphabet_min, alphabet_size)
        ent = mx.minimum(ent, float(stream.bits)).reshape(g, d)
        code_bits = mx.sum(ent, axis=1) * stream.n_rows  # [G]
    else:
        flat_gn = symbols.reshape(g, -1)
        ent = _batched_symbol_entropy_bits(flat_gn, alphabet_min, alphabet_size)
        ent = mx.minimum(ent, float(stream.bits))
        code_bits = ent * stream.n_rows * d  # [G]

    code_bytes = mx.ceil(code_bits / 8.0)
    param_bytes = n_groups * d * 2 * 2  # scale + zero, fp16 — same for every head
    total = code_bytes + float(param_bytes)
    return [int(v) for v in total.tolist()]


def fixed_width_bytes_batched(stream: CodeStream) -> list[int]:
    """Batched ``fixed_width_bytes`` over the CodeStream's leading ``G`` axis."""
    g, n_groups, gs, d = stream.codes.shape
    code_bytes = math.ceil(stream.n_rows * d * stream.bits / 8)
    param_bytes = n_groups * d * 2 * 2
    return [code_bytes + param_bytes] * g


def layer_group_bits(n_layers: int, base_bits: int, n_groups: int = 3) -> list[int]:
    """Per-layer bit-width schedule from the paper's layer-wise sensitivity insight (§5.1.2/§5.2).

    Splits the ``n_layers`` transformer layers into ``n_groups`` contiguous
    groups (earliest first) and assigns progressively *fewer* bits to deeper
    groups: ``base_bits`` to the first third, ``base_bits - 1`` to the middle
    third, ``base_bits - 2`` to the last third (floored at 2 bits). This
    mirrors the paper's finding that output quality is far more sensitive to
    precision loss in shallow layers than in deep ones (Figure 4), so shallow
    layers get conservative (higher-bit) quantization and deep layers get
    coarser quantization.

    Args:
        n_layers: number of attention-bearing layers.
        base_bits: bit-width assigned to the shallowest layer group (also the
            ceiling — no layer is assigned more than this).
        n_groups: number of contiguous layer groups (paper uses 3).

    Returns:
        Length-``n_layers`` list of per-layer bit-widths, non-increasing with depth.
    """
    if n_layers <= 0:
        return []
    n_groups = min(n_groups, n_layers)
    bounds = [round(n_layers * g / n_groups) for g in range(n_groups + 1)]
    bounds[-1] = n_layers
    schedule = []
    for i in range(n_layers):
        group = next(g for g in range(n_groups) if bounds[g] <= i < bounds[g + 1])
        schedule.append(max(2, base_bits - group))
    return schedule


def cachegen_quant_dequant(x: mx.array, bits: int, group_size: int = 32) -> mx.array:
    """Drop-in quant→dequant (values identical to plain group quant).

    The entropy layer is storage-only; the reconstructed tensor is exactly the
    group-quant reconstruction, so this is a drop-in for ``_group_quant_dequant``.
    """
    return dequant_codes(quantize_to_codes(x, bits, group_size))


__all__ = [
    "CodeStream",
    "quantize_to_codes",
    "dequant_codes",
    "token_delta",
    "symbol_entropy_bits",
    "entropy_coded_bytes",
    "fixed_width_bytes",
    "layer_group_bits",
    "cachegen_quant_dequant",
    "quantize_to_codes_batched",
    "dequant_codes_batched",
    "token_delta_batched",
    "entropy_coded_bytes_batched",
    "fixed_width_bytes_batched",
]
