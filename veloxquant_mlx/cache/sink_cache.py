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
        """Quantize aged-out tokens except sinks; keep residual + sinks fp16."""
        B, H, S, D = keys.shape
        r = self._residual_length
        start = self.offset  # absolute position of keys[:, :, 0, :]

        sinks = self._update_sinks(keys, start)

        if S <= r:
            k_out, v_out = keys, values
            n_quant = 0
            n_sink_in_block = 0
        else:
            n_quant = S - r
            sink_local = sorted(p - start for p in sinks if 0 <= p - start < n_quant)
            k_region = keys[:, :, :n_quant, :]
            v_region = values[:, :, :n_quant, :]
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
                        while src in sink_set and src < n_quant - 1:
                            src += 1
                    if 0 <= src < n_quant and src not in sink_set:
                        k_region[:, :, idx, :] = keys[:, :, src, :]
                        v_region[:, :, idx, :] = values[:, :, src, :]
            k_q = self._quant_dequant_along(k_region, axis=-2)
            v_q = self._quant_dequant_along(v_region, axis=-1)
            # Restore fp16 for sink positions inside the quantized region.
            for idx in sink_local:
                k_q[:, :, idx, :] = keys[:, :, idx, :]
                v_q[:, :, idx, :] = values[:, :, idx, :]
            n_sink_in_block = len(sink_local)
            k_out = mx.concatenate([k_q, keys[:, :, n_quant:, :]], axis=2)
            v_out = mx.concatenate([v_q, values[:, :, n_quant:, :]], axis=2)

        self._account_bytes_with_sinks(B, H, S, D, n_quant, n_sink_in_block)
        # offset is advanced by the parent call below.
        return super(KIVIKVCache, self).update_and_fetch(k_out, v_out)

    def _account_bytes_with_sinks(
        self, B: int, H: int, S: int, D: int, n_quant: int, n_sink: int
    ) -> None:
        """KIVI accounting with sinks split out — no double counting.

        Of the S incoming tokens: ``n_quant - n_sink`` are compressed,
        ``n_sink`` go to the sink fp16 pool, and ``S - n_quant`` go to the
        residual fp16 window.  Calling the parent ``_account_bytes`` with a
        reduced n_quant would inflate its residual term (it derives
        ``n_res = S - n_quant``), so we account the three pools explicitly.
        """
        import math

        n_compressed = n_quant - n_sink
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
        self._residual_fp16_bytes += (S - n_quant) * D * 2 * 2 * H * B
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
