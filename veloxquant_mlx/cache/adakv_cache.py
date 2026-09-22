"""AdaKV-proxy KV cache wrapper — per-head adaptive bit allocation over KIVI.

Inspired by "Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget
Allocation for Efficient LLM Inference" (arXiv:2407.11550, 2024). Documented
as "AdaKV-proxy (VeloxQuant-MLX implementation)" — a proxy adaptation, not a
faithful port. See :mod:`veloxquant_mlx.quantizers.adakv` for the algorithm.

Design:
    Prefill (first call, S > 1):
        1. Update running per-head norm accumulators from the incoming batch.
        2. Recompute the per-head bit assignment from current norm-variance
           estimates under the global average-bits budget.
        3. Quantize each head's keys at its assigned bit-width and forward the
           reconstructed fp16 keys to the underlying mlx_lm KVCache.

    Decode (subsequent calls, S == 1 per step):
        1. Update accumulators with the new key token.
        2. Recompute the per-head bit assignment (every ``adakv_update_interval``
           steps; default 1 = every step, matching the original behaviour).
        3. Quantize the new key at the per-head assignment, forward fp16.

    Values are left at fp16 throughout (AdaKV-proxy is a key-only method).

Performance fix (VeloxQuant-MLX#504): two real, measured host-sync costs on
the real `mlx_lm.generate()` decode path, both now gated to
``adakv_update_interval`` instead of firing every single step/layer:
  1. ``_quantize_per_head`` looped ``for b in range(B): for h in range(H):``
     calling one MLX op per head — the O(B*H) Python-dispatch pattern
     documented (and fixed once already) for H2O. Now batched via
     :func:`veloxquant_mlx.quantizers.adakv.quantize_heads_batched`, which
     groups heads by their (typically 2-3 distinct) assigned bit-widths.
  2. The larger cost: ``allocate_head_bits`` calls ``.tolist()`` on its
     importance input, and ``_update_norm_accumulators`` called
     ``mx.eval()`` unconditionally — both forced a full host sync of that
     step's pending MLX graph on *every* call (every layer, every decode
     step). Measured on Llama-3.2-1B (this M4): recomputing every step
     cost ~53% of end-to-end decode throughput vs. plain fp16; gating both
     to ``adakv_update_interval`` recovered the large majority of it.
     ``adakv_update_interval=1`` (the default) preserves the exact prior
     behaviour for anyone relying on per-step recomputation.

Byte accounting:
    compressed_key_bytes — weighted by each head's assigned bit-width
    fp16_key_bytes       — cost if stored as fp16 (for ratio)
    value_fp16_bytes     — values always fp16

What is NOT implemented (documented):
    - True Ada-KV head-adaptive *eviction* budget (needs softmax attention).
    - Cross-layer budget sharing.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.quantizers.adakv import (
    allocate_head_bits,
    compute_head_attention_entropy,
    quantize_heads_batched,
)


class AdaKVCache(_MLXKVCache):
    """KV cache implementing AdaKV-proxy per-head adaptive bit allocation.

    Args:
        config: :class:`KVCacheConfig`.  Fields consumed:
            ``adakv_target_avg_bits`` (float, default 2.5),
            ``adakv_lo_bit``          (int, default 2),
            ``adakv_mid_bit``         (int, default 3),
            ``adakv_hi_bit``          (int, default 4),
            ``adakv_group_size``      (int, default 32),
            ``adakv_update_interval`` (int, default 1),
            ``adakv_importance``      (str, default ``"norm_variance"``),
            ``adakv_obs_window``      (int, default 32).

    Importance signals:
        ``"norm_variance"`` — inter-token key-norm variance. Measures
        *quantization sensitivity*, and is anti-correlated with the paper's
        attention-dispersion criterion.

        ``"attention_entropy"`` — observation-window attention entropy, which
        carries Ada-KV's own sign (dispersed heads get more budget). Costs one
        ``[w, S]`` attention matrix per head at prefill.

        See :mod:`veloxquant_mlx.quantizers.adakv` for why these differ.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._target_avg_bits: float = float(getattr(config, "adakv_target_avg_bits", 2.5))
        self._lo_bit: int = int(getattr(config, "adakv_lo_bit", 2))
        self._mid_bit: int = int(getattr(config, "adakv_mid_bit", 3))
        self._hi_bit: int = int(getattr(config, "adakv_hi_bit", 4))
        self._group_size: int = int(getattr(config, "adakv_group_size", 32))
        self._update_interval: int = max(1, int(getattr(config, "adakv_update_interval", 1)))
        self._importance_mode: str = str(getattr(config, "adakv_importance", "norm_variance"))
        self._obs_window: int = int(getattr(config, "adakv_obs_window", 32))
        if self._importance_mode not in ("norm_variance", "attention_entropy"):
            raise QuantizerConfigError(
                f"AdaKVCache: adakv_importance must be 'norm_variance' or "
                f"'attention_entropy', got {self._importance_mode!r}."
            )

        # Allowed bit set (dedup + sort). mid==hi collapses to a 2-tier set.
        self._allowed_bits: list[int] = sorted({self._lo_bit, self._mid_bit, self._hi_bit})

        # Running per-head accumulators of the per-token key L2 norm.
        self._norm_sum: mx.array | None = None  # [H] fp32
        self._norm_sq_sum: mx.array | None = None  # [H] fp32
        self._n_tokens: int = 0  # total tokens seen

        # Last observed per-head attention entropy ([H] fp32), for the
        # "attention_entropy" mode. Unlike norm variance this cannot be folded
        # into a running scalar accumulator — entropy is not a mean of
        # per-token quantities — so it is recomputed from whatever key block
        # the current call carries. At decode (S == 1) a single row carries no
        # attention distribution, so the prefill estimate is retained.
        self._entropy: mx.array | None = None  # [H] fp32

        # Current per-head bit assignment ([H] ints). Set on first update.
        self._head_bits: list[int] | None = None

        # Steps since the bit assignment was last recomputed — gates
        # _recompute_head_bits to adakv_update_interval (see that method).
        self._steps_since_recompute: int = 0

        # Degenerate-target warning is emitted at most once per cache.
        self._warned_degenerate: bool = False

        # Byte accounting
        self._compressed_key_bytes: int = 0
        self._fp16_key_bytes: int = 0
        self._value_fp16_bytes: int = 0

    # ------------------------------------------------------------------
    # Running statistics
    # ------------------------------------------------------------------
    def _update_norm_accumulators(self, keys: mx.array) -> None:
        """Update running per-head sum/sum-of-squares of per-token ‖k_t‖₂.

        Args:
            keys: [B, H, S, D]. Per-token norms are averaged over the batch.
        """
        B, H, S, D = keys.shape
        k32 = keys.astype(mx.float32)
        norms = mx.sqrt(mx.sum(k32 * k32, axis=-1))  # [B, H, S]
        norms_b = mx.mean(norms, axis=0)  # [H, S] (avg over batch)
        new_sum = mx.sum(norms_b, axis=-1)  # [H]
        new_sq_sum = mx.sum(norms_b * norms_b, axis=-1)  # [H]

        if self._norm_sum is None:
            self._norm_sum = new_sum
            self._norm_sq_sum = new_sq_sum
        else:
            self._norm_sum = self._norm_sum + new_sum
            self._norm_sq_sum = self._norm_sq_sum + new_sq_sum
        # Evaluate periodically (every adakv_update_interval steps), not on
        # every call — an intermediate eval() on new_sum/new_sq_sum before
        # the add would be a wasted host sync (both are immediately consumed
        # by the addition either way), but evaluating the post-fold
        # accumulator every single call is itself the dominant real-model
        # cost this class had (see VeloxQuant-MLX#504: measured ~53% of
        # end-to-end decode throughput on Llama-3.2-1B). Still evaluated at
        # least every adakv_update_interval steps (matching
        # _recompute_head_bits's own cadence below) so the pending graph
        # cannot grow unboundedly across a long decode run — the same
        # graph-growth-crash guard H2O's _EVAL_FLUSH_INTERVAL documents,
        # just batched to the interval the caller already configured rather
        # than forced every step. Default adakv_update_interval=1 preserves
        # exact prior behaviour.
        if self._n_tokens % self._update_interval == 0:
            mx.eval(self._norm_sum, self._norm_sq_sum)

        self._n_tokens += S

    def _running_head_importance(self) -> mx.array:
        """Per-head norm variance from running accumulators ([H] fp32)."""
        if self._norm_sum is None or self._n_tokens < 2:
            H = 0 if self._norm_sum is None else self._norm_sum.shape[0]
            return mx.zeros((H,), dtype=mx.float32)
        n = self._n_tokens
        mean = self._norm_sum / n
        return mx.maximum(self._norm_sq_sum / n - mean * mean, 0.0)

    def _current_importance(self, n_heads: int) -> mx.array:
        """Per-head importance under the configured signal ([H] fp32)."""
        if self._importance_mode == "attention_entropy":
            if self._entropy is not None:
                return self._entropy
            return mx.zeros((n_heads,), dtype=mx.float32)
        return self._running_head_importance()

    def _recompute_head_bits(self, n_heads: int) -> None:
        """Recompute the per-head bit assignment from current statistics,
        gated to ``adakv_update_interval`` steps.

        ``allocate_head_bits`` calls ``.tolist()`` on its importance input —
        a host sync that, called unconditionally on every ``update_and_fetch``
        (every layer, every decode step), forces materialization of that
        step's entire pending MLX graph up to this point rather than letting
        work batch/pipeline across layers within one step. Measured on real
        `mlx_lm.generate()` (Llama-3.2-1B, this M4): recomputing every step
        cost ~53% of end-to-end decode throughput vs. plain fp16; gating to
        the already-documented (but previously unimplemented — see module
        docstring's former "NOT implemented" note) ``adakv_update_interval``
        recovers the large majority of it, since the bit assignment changes
        slowly relative to a single decode step for any real workload (see
        VeloxQuant-MLX#504). Default ``adakv_update_interval=1`` preserves
        exact prior behaviour (recompute every step) for anyone relying on it.
        """
        self._steps_since_recompute += 1
        if self._head_bits is not None and self._steps_since_recompute < self._update_interval:
            return
        self._steps_since_recompute = 0
        importance = self._current_importance(n_heads)
        self._head_bits = allocate_head_bits(
            importance,
            target_avg_bits=self._target_avg_bits,
            allowed_bits=self._allowed_bits,
            n_heads=n_heads,
            # The degenerate-target warning is a configuration diagnostic; the
            # cache emits it once at construction-time behaviour rather than on
            # every decode step.
            warn_degenerate=not self._warned_degenerate,
        )
        if not (self._allowed_bits[0] < self._target_avg_bits < self._allowed_bits[-1]):
            self._warned_degenerate = True

    # ------------------------------------------------------------------
    # Core quantization
    # ------------------------------------------------------------------
    def _quantize_per_head(self, keys: mx.array) -> mx.array:
        """Quantize keys [B, H, S, D] with each head at its assigned bit-width.

        Batched over heads-grouped-by-bit-width (see
        ``quantize_heads_batched``) instead of a Python ``for b: for h:``
        loop calling ``quantize_head`` per (b,h) pair — that loop shape was
        measured costing 61.6% of real decode throughput on this exact class
        (VeloxQuant-MLX#504: 132.1 -> 50.8 tok/s, real ``mlx_lm.generate()``).
        """
        assert self._head_bits is not None and len(self._head_bits) == keys.shape[1]
        return quantize_heads_batched(keys, self._head_bits, self._group_size)

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Re-derive per-head importance (norm or attention-entropy), recompute the per-head bit allocation, and quantize each head's keys at its assigned bit-width; values pass through fp16 unchanged."""
        B, H, S, D = keys.shape

        self._update_norm_accumulators(keys)
        if self._importance_mode == "attention_entropy" and S > 1:
            # Only a multi-token block carries an attention distribution; at
            # decode (S == 1) the prefill estimate is kept.
            ent = compute_head_attention_entropy(keys, self._obs_window)
            mx.eval(ent)
            self._entropy = ent
        self._recompute_head_bits(H)
        k_out = self._quantize_per_head(keys)

        self._account_bytes(B, H, S, D)
        return super().update_and_fetch(k_out, values)

    def _account_bytes(self, B: int, H: int, S: int, D: int) -> None:
        n_groups = math.ceil(S / self._group_size)
        assert self._head_bits is not None
        for h in range(H):
            b = self._head_bits[h]
            code_bytes = math.ceil(S * D * b / 8)
            param_bytes = n_groups * D * 2 * 2  # scale + zero, fp16
            self._compressed_key_bytes += (code_bytes + param_bytes) * B
        self._fp16_key_bytes += B * H * S * D * 2
        self._value_fp16_bytes += B * H * S * D * 2

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def head_bits(self) -> list[int]:
        """Current per-head bit assignment ([H] ints), or [] before first update."""
        return list(self._head_bits) if self._head_bits is not None else []

    @property
    def assigned_avg_bits(self) -> float:
        """Actual average bits/element across heads (0.0 before first update)."""
        if not self._head_bits:
            return 0.0
        return sum(self._head_bits) / len(self._head_bits)

    @property
    def head_importance(self) -> list[float]:
        """Current per-head importance under the configured signal ([H] floats)."""
        n = len(self._head_bits) if self._head_bits else 0
        return self._current_importance(n).tolist()

    @property
    def importance_mode(self) -> str:
        """Which importance signal drives allocation."""
        return self._importance_mode

    @property
    def compressed_key_bytes(self) -> int:
        """Realized stored bytes for the compressed key cache (per-head codes at their assigned bit-width + group params, all heads/batches)."""
        return self._compressed_key_bytes

    @property
    def fp16_key_bytes(self) -> int:
        """Hypothetical fp16 key cost if nothing were compressed."""
        return self._fp16_key_bytes

    @property
    def value_fp16_bytes(self) -> int:
        """Actual value cost — values are stored fp16 throughout (key-only method)."""
        return self._value_fp16_bytes

    @property
    def target_avg_bits(self) -> float:
        """Configured target average bit-width the per-head allocation aims for."""
        return self._target_avg_bits

    @property
    def allowed_bits(self) -> list[int]:
        """Configured discrete bit-widths a head's importance can be mapped to."""
        return list(self._allowed_bits)

    @property
    def group_size(self) -> int:
        """Group size used for asymmetric group quantization scale/zero fitting."""
        return self._group_size


__all__ = ["AdaKVCache"]
