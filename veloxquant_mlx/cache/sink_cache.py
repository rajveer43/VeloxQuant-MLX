"""KVSink-adapted sink protection on top of KIVI group quantization.

**Inspired by, not a faithful port of**, "KVSink: Understanding and Enhancing
the Preservation of Attention Sinks in KV Cache Quantization for LLMs"
(Su & Yuan, COLM 2025, arXiv:2508.04257).

The paper identifies attention-sink tokens via extreme-magnitude outlier
channels in the *hidden state* at a model-specific "emergence layer", then
excludes those tokens from quantization.  Our cache wrappers never see the
hidden state — by design, they receive only per-layer K/V tensors inside
``update_and_fetch`` — so a literal port is not possible at this level.

**Adaptation:** sink tokens also exhibit anomalously large key L2-norm (the
same outlier-magnitude phenomenon, observable in the K tensor the cache does
see; the repo's ``KeyNormObserver`` is built on this signal).  This cache
therefore maintains a running top-k of the highest-key-norm token positions
and stores those tokens' K/V in fp16, delegating everything else to KIVI's
deterministic asymmetric group quantization (per-channel keys, per-token
values, fp16 residual window).  All documentation must label this method
"KVSink-adapted" — never plain "KVSink".

Known v1 limitation (prefill-dominant selection): the base mlx_lm cache
stores a contiguous fp16 tensor after dequantization, so a token that was
already quantized in an earlier call is *not* retroactively restored if it
later qualifies as a sink.  Sinks identified within the current incoming
block are protected; in practice attention sinks emerge among early tokens,
which arrive in the prefill block where protection is fully effective.

Deterministic end to end: top-k on key norm + min/max group quantization.
No codebook training, no RNG.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from veloxquant_mlx.cache.kivi_cache import KIVIKVCache


class SinkProtectedKVCache(KIVIKVCache):
    """KIVI cache with KVSink-adapted high-key-norm token protection.

    Args:
        config: :class:`KVCacheConfig`.  Fields consumed beyond KIVI's
            (``head_dim``, ``bit_width_inlier``, ``kivi_group_size``,
            ``residual_length``): **``n_sink_tokens``** — the number of
            highest-key-norm token positions kept in fp16 (paper default 5).

    Notes:
        Selection signal is the per-token key L2-norm averaged over KV heads
        (mean, not max: a sink absorbs attention from *all* heads, and the
        mean is robust to a single head's scale spikes).  Positions are
        tracked in absolute sequence coordinates via ``self.offset``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._n_sink = int(getattr(config, "n_sink_tokens", 5))
        if self._n_sink < 0:
            raise ValueError(f"SinkProtectedKVCache: n_sink_tokens={self._n_sink} must be >= 0.")
        # Running candidates: absolute position -> key-norm (float).
        # Pruned to the top n_sink entries after every update.
        self._sink_norms: dict[int, float] = {}
        self._sink_fp16_bytes = 0

    # ------------------------------------------------------------------
    # Sink bookkeeping
    # ------------------------------------------------------------------
    def _update_sinks(self, keys: mx.array, start_pos: int) -> set:
        """Fold the incoming block's key norms into the running top-k.

        Returns the current sink-position set after the update.
        """
        if self._n_sink == 0:
            return set()
        # [B, H, S, D] -> per-token norm, mean over heads, batch 0.
        norms = mx.linalg.norm(keys.astype(mx.float32), axis=-1)  # [B, H, S]
        per_tok = mx.mean(norms, axis=1)[0]  # [S]
        per_tok_np = [float(v) for v in per_tok.tolist()]
        for i, v in enumerate(per_tok_np):
            self._sink_norms[start_pos + i] = v
        if len(self._sink_norms) > self._n_sink:
            keep = sorted(self._sink_norms.items(), key=lambda kv: -kv[1])
            self._sink_norms = dict(keep[: self._n_sink])
        return set(self._sink_norms.keys())

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys, values):
        """Quantize aged-out tokens except sinks; keep residual + sinks fp16.

        Ages tokens out based on **true cumulative position** (mirrors the
        fix in ``KIVIKVCache.update_and_fetch``): store the incoming block
        via the parent first, then quantize-in-place the newly-aged leading
        slice ``[self._n_quantized, self.offset - residual_length)`` of the
        accumulated buffer, skipping any position still marked a sink.
        """
        B, H, S, D = keys.shape
        r = self._residual_length
        start = self.offset  # absolute position of keys[:, :, 0, :]

        sinks = self._update_sinks(keys, start)
        k_all, v_all = super(KIVIKVCache, self).update_and_fetch(keys, values)

        new_boundary = max(0, self.offset - r)
        n_quant_now = new_boundary - self._n_quantized
        n_sink_in_block = 0
        if n_quant_now > 0:
            lo, hi = self._n_quantized, new_boundary
            sink_local = sorted(p - lo for p in sinks if lo <= p < hi)
            k_region = self.keys[:, :, lo:hi, :]
            v_region = self.values[:, :, lo:hi, :]
            if sink_local:
                # Per the KVSink paper, sinks must be excluded from
                # quantization *parameter calibration*, not just restored
                # afterwards: a 25x-magnitude sink inflates its group's
                # min/max scale and ruins every neighbor in the group.
                # Neutralize each sink row with the nearest non-sink row
                # before computing quant params, then restore fp16 below.
                sink_set = set(sink_local)
                k_region = mx.array(k_region)  # materialize a copy
                v_region = mx.array(v_region)
                for idx in sink_local:
                    src = idx - 1
                    while src in sink_set and src > 0:
                        src -= 1
                    if src < 0 or src in sink_set:
                        src = idx + 1
                        while src in sink_set and src < n_quant_now - 1:
                            src += 1
                    if 0 <= src < n_quant_now and src not in sink_set:
                        k_region[:, :, idx, :] = self.keys[:, :, lo + src, :]
                        v_region[:, :, idx, :] = self.values[:, :, lo + src, :]
            k_q = self._quant_dequant_along(k_region, axis=-2)
            v_q = self._quant_dequant_along(v_region, axis=-1)
            # Restore fp16 for sink positions inside the quantized region.
            for idx in sink_local:
                k_q[:, :, idx, :] = self.keys[:, :, lo + idx, :]
                v_q[:, :, idx, :] = self.values[:, :, lo + idx, :]
            self.keys[:, :, lo:hi, :] = k_q
            self.values[:, :, lo:hi, :] = v_q
            n_sink_in_block = len(sink_local)
            self._n_quantized = new_boundary
            k_all, v_all = self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

        self._account_bytes_with_sinks(B, H, S, D, n_quant_now, n_sink_in_block)
        return k_all, v_all

    def _account_bytes_with_sinks(
        self, B: int, H: int, S: int, D: int, n_quant_now: int, n_sink: int
    ) -> None:
        """KIVI accounting with sinks split out — no double counting.

        Of the tokens newly quantized this call: ``n_quant_now - n_sink``
        are compressed and ``n_sink`` go to the sink fp16 pool.  The
        residual window is reported as its *current* live size (the
        cumulative fp16 tail), not a per-call delta, so it plateaus at
        ``residual_length`` once the window fills.
        """
        import math

        n_compressed = n_quant_now - n_sink
        gs = self._group_size
        if n_compressed > 0:
            k_groups = math.ceil(n_compressed / gs)
            k_code_bytes = math.ceil(n_compressed * D * self._b / 8) * H * B
            k_param_bytes = k_groups * D * 2 * 2 * H * B
            v_groups = math.ceil(D / gs)
            v_code_bytes = math.ceil(n_compressed * D * self._b / 8) * H * B
            v_param_bytes = n_compressed * v_groups * 2 * 2 * H * B
            self._key_bytes_compressed += k_code_bytes + k_param_bytes
            self._value_bytes_compressed += v_code_bytes + v_param_bytes
        self._sink_fp16_bytes += n_sink * D * 2 * 2 * H * B  # K+V
        n_res = self.offset - self._n_quantized
        self._residual_fp16_bytes = n_res * D * 2 * 2 * H * B
        self._key_bytes_fp16 += H * B * S * D * 2
        self._value_bytes_fp16 += H * B * S * D * 2
        self._tokens_seen += S

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @property
    def sink_fp16_bytes(self) -> int:
        """Bytes held in fp16 for protected sink tokens (keys + values)."""
        return self._sink_fp16_bytes

    @property
    def sink_positions(self) -> list:
        """Current protected absolute token positions, highest-norm first."""
        return [p for p, _ in sorted(self._sink_norms.items(), key=lambda kv: -kv[1])]


__all__ = ["SinkProtectedKVCache"]
