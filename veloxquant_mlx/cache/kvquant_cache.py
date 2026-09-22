"""KVQuant-NUQ KV cache wrapper — non-uniform quantization + outlier isolation.

Inspired by "KVQuant: Towards 10 Million Context Length LLM Inference with KV
Cache Quantization" (arXiv:2401.18079, NeurIPS 2024). Documented as
"KVQuant-adapted (VeloxQuant-MLX implementation)" — implements the four
cache-observable pillars (per-channel keys / per-token values, NUQ datatype,
per-vector dense-and-sparse outlier isolation, and Attention Sink-Aware
quantization) and documents pre-RoPE key quantization as out of scope. See
:mod:`veloxquant_mlx.quantizers.kvquant` for the numerics.

Quantization axes (matching KVQuant's asymmetry, the same axes KIVI uses):
    Keys   — per-channel: each head-dim channel gets its own non-uniform levels
             (sample axis = tokens). Channels have stable, distinct distributions.
    Values — per-token: each token gets its own levels (sample axis = channels).

Level lifecycle:
    Prefill (S > 1) fits the NUQ levels from the incoming batch and freezes them
    (``kvquant_refit_interval == 0``, the default — mirrors SVDq's frozen-V).
    Decode tokens quantize against the frozen levels. With a positive refit
    interval, levels are re-fit every N decode steps from the most recent token.
    Value levels are per-token (one signpost table per token, sample axis =
    channels) and are, by design, always fit fresh every call — never frozen,
    regardless of ``kvquant_refit_interval`` (that field only gates the *key*
    path). This is inherent to the per-token scheme, not a bug.

Performance (VeloxQuant-MLX#504): the per-head application was originally a
Python ``for b in range(B): for h in range(H):`` loop, and the Lloyd-Max fit
inside it forced a host sync (``mx.eval``) every iteration — together
measured a real 59x `mlx_lm.generate()` decode slowdown (94.6 -> 2.8 tok/s,
Llama-3.2-1B, Apple M4). Both are fixed: application is now batched over the
flattened ``B*H`` axis (:func:`~veloxquant_mlx.quantizers.kvquant.split_dense_sparse_batched`,
:func:`~veloxquant_mlx.quantizers.kvquant.fit_nuq_levels_batched`, etc.),
and the Lloyd-Max loop no longer evaluates per iteration — measured 9.6x
real end-to-end recovery (2.8 -> 27 tok/s on the same benchmark). **This
does not close the full gap to fp16 speed.** Unlike AdaKV's equivalent fix
(VeloxQuant-MLX#512), the remaining cost here is not dispatch overhead —
it's the Lloyd-Max fit's own O(``n_iters`` x N x L x D) arithmetic, paid
fresh for values on *every* decode step by the algorithm's own per-token-
levels design. ``kvquant_lloyd_iters`` (default 8) is the existing,
already-correctly-wired lever for trading fit quality for speed — lower
values measured a further 27 -> 41 tok/s at ``lloyd_iters=1`` on the same
benchmark, still well short of fp16 parity. There is no further "fix" to
apply here without changing the algorithm's per-token-value-levels design.

Byte accounting:
    compressed_*  — NUQ codes + per-(channel/token) level table + fp16 outlier
                    side-channel (value + position index).
    fp16_*        — uncompressed cost for the ratio.

Attention Sink-Aware quantization (paper §3.5):
    The first ``kvquant_n_sink`` tokens of the sequence are restored to exact
    fp16 and excluded from both the level fit and the outlier thresholds — the
    model is disproportionately sensitive to quantization error at the sink
    positions it dumps excess attention onto.

What is NOT implemented (documented):
    - Pre-RoPE key quantization (needs a model-forward hook — outside the cache
      contract; we see post-RoPE keys only).
    - Offline calibration-set level fitting (we fit online; zero calibration).
    - Fisher-information sensitivity weighting of the Lloyd-Max objective
      (Eq. 1 weights each squared error by F_ii; needs gradients from a
      calibration pass, so our fit is the unweighted k-means special case).
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.quantizers.kvquant import (
    dequant_nuq_batched,
    fit_nuq_levels_batched,
    quantize_nuq_batched,
    split_dense_sparse_batched,
)


class KVQuantKVCache(_MLXKVCache):
    """KV cache implementing KVQuant-NUQ non-uniform quantization.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``kvquant_bits``             (int, default 3),
            ``kvquant_outlier_fraction`` (float, default 0.01),
            ``kvquant_group_size``       (int, default 32; reserved for grouped fits),
            ``kvquant_lloyd_iters``      (int, default 8),
            ``kvquant_refit_interval``   (int, default 0 = freeze prefill levels).

    No ``.bits`` attribute — mlx_lm's SDPA checks ``hasattr(cache, "bits")``
    to route to its quantized-matmul kernel path, which expects a different
    cache layout (mx.quantize's native tuple format) and a ``.group_size``
    attribute this cache doesn't have. We expose ``.nuq_bits`` instead.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._bits: int = int(getattr(config, "kvquant_bits", 3))
        self._outlier_fraction: float = float(getattr(config, "kvquant_outlier_fraction", 0.01))
        self._group_size: int = int(getattr(config, "kvquant_group_size", 32))
        self._lloyd_iters: int = int(getattr(config, "kvquant_lloyd_iters", 8))
        self._refit_interval: int = int(getattr(config, "kvquant_refit_interval", 0))
        self._n_sink: int = int(getattr(config, "kvquant_n_sink", 1))
        if self._n_sink < 0:
            raise QuantizerConfigError(
                f"KVQuantKVCache: kvquant_n_sink={self._n_sink} must be >= 0"
            )

        # Per-channel |key| outlier threshold frozen at prefill, reused at decode
        # where a single token cannot define its own per-channel top-k.
        self._key_outlier_thresh: mx.array | None = None
        self._sink_kept: int = 0

        # Frozen levels fit at prefill: keys per-channel [H, L, D],
        # values per-token use levels [H, L, D] in transposed space.
        # _key_levels is a stacked [H, L, D] array (not a Python list) since
        # #504's batching rewrite — indexing/iterating it behaves the same
        # as the old list-of-per-head-arrays for every existing caller
        # (cache.key_levels[0], `for level in cache.key_levels`), but it can
        # now be passed straight to mx.tile when sharing frozen levels
        # across the batch axis.
        self._key_levels: mx.array | None = None  # [H, L, D]
        self._value_levels: list | None = None  # list over heads of [L, D] (channel-as-sample)
        self._n_tokens: int = 0
        self._outlier_count: int = 0
        self._prev_outlier_count: int = 0

        # Byte accounting
        self._compressed_key_bytes: int = 0
        self._compressed_value_bytes: int = 0
        self._fp16_key_bytes: int = 0
        self._fp16_value_bytes: int = 0

    # ------------------------------------------------------------------
    # Batched (over B*H) NUQ application
    # ------------------------------------------------------------------
    def _quant_keys_batched(self, k_bh: mx.array, levels: mx.array | None):
        """Keys: per-channel NUQ, batched over BH. ``k_bh``: [BH, S, D].

        Returns (recon_fp16 [BH, S, D], levels_used [BH, L, D]).

        Batched over the flattened (batch, head) axis instead of a Python
        ``for b: for h:`` loop — the O(B*H) dispatch pattern measured
        costing the majority of a 59x real ``mlx_lm.generate()`` decode
        slowdown on this class (VeloxQuant-MLX#504). Numerically identical
        to calling the old per-head path once per row of that axis.

        At decode (S == 1) a per-channel column holds a single sample, so the
        rank-based outlier split degenerates (it cannot keep >=1 inlier and
        still flag anything). We instead apply a magnitude threshold carried
        over from the frozen prefill statistics, so decode keys keep the same
        outlier protection the prefill keys got.
        """
        ds = split_dense_sparse_batched(k_bh, self._outlier_fraction)
        if levels is None:
            levels = fit_nuq_levels_batched(ds.inliers, self._bits, self._lloyd_iters)
        codes = quantize_nuq_batched(ds.inliers, levels)
        recon = dequant_nuq_batched(codes, levels).astype(mx.float32)
        mask = ds.outlier_mask
        vals = ds.outlier_vals
        if (
            k_bh.shape[1] < 2
            and self._outlier_fraction > 0.0
            and self._key_outlier_thresh is not None
        ):
            # Decode: reuse the frozen per-channel threshold from prefill.
            # _key_outlier_thresh is [1, D]; broadcasts against [BH, 1, D].
            k32 = k_bh.astype(mx.float32)
            mask = mx.abs(k32) >= self._key_outlier_thresh[None, :, :]
            vals = mx.where(mask, k32, mx.zeros_like(k32))
        recon = mx.where(mask, vals, recon)
        self._outlier_count += int(mx.sum(mask).item())
        return recon.astype(mx.float16), levels

    def _quant_values_batched(self, v_bh: mx.array):
        """Values: per-token NUQ, batched over BH. ``v_bh``: [BH, S, D].

        Transposes S<->D per row so tokens are columns (per-token levels),
        the same axis swap the old per-head path did, just batched. Values
        are unaffected by the decode degeneracy: a per-token column has D
        samples regardless of S, so the rank-based split is always well
        posed — always re-fit fresh (never frozen), matching prior behaviour.

        Returns (recon_fp16 [BH, S, D], levels_used [BH, L, S] — for
        introspection only; not reused across calls).
        """
        v_ds = mx.swapaxes(v_bh, 1, 2)  # [BH, D, S]: each column is one token
        ds = split_dense_sparse_batched(v_ds, self._outlier_fraction)
        levels = fit_nuq_levels_batched(ds.inliers, self._bits, self._lloyd_iters)
        codes = quantize_nuq_batched(ds.inliers, levels)
        recon = dequant_nuq_batched(codes, levels).astype(mx.float32)
        recon = mx.where(ds.outlier_mask, ds.outlier_vals, recon)
        self._outlier_count += int(mx.sum(ds.outlier_mask).item())
        return mx.swapaxes(recon.astype(mx.float16), 1, 2), levels  # back to [BH, S, D]

    def _capture_key_thresholds(self, keys: mx.array) -> None:
        """Freeze the per-channel outlier threshold from the prefill keys.

        ``thresh[0, d]`` is the k-th largest |key| in channel ``d`` over the
        prefill tokens — the same cut the rank-based split makes. Decode steps
        (S == 1) reuse it because a single token cannot define its own top-k.
        """
        if self._outlier_fraction <= 0.0:
            self._key_outlier_thresh = None
            return
        # keys: [B, H, S, D] → pool tokens across batch/heads per channel.
        k32 = keys.astype(mx.float32)
        flat = k32.reshape(-1, k32.shape[-1])  # [B*H*S, D]
        n = flat.shape[0]
        if n < 2:
            self._key_outlier_thresh = None
            return
        k = max(1, int(round(n * self._outlier_fraction)))
        k = min(k, n - 1)
        sorted_desc = mx.sort(mx.abs(flat), axis=0)[::-1]
        self._key_outlier_thresh = sorted_desc[k - 1 : k, :]  # [1, D]
        mx.eval(self._key_outlier_thresh)

    def _apply(self, keys: mx.array, values: mx.array):
        B, H, S, D = keys.shape
        # Keys use per-channel levels (stable across tokens) → fit at prefill and
        # freeze. Values use per-token levels (one set per token) → inherently
        # re-fit every call; they are never frozen across steps.
        is_prefill = self._key_levels is None
        refit_keys = is_prefill or (
            self._refit_interval > 0
            and self._n_tokens > 0
            and (self._n_tokens % self._refit_interval == 0)
        )

        # Sink tokens are restored to fp16 below, so — following the paper — they
        # are also excluded from the level fit and from the outlier thresholds.
        # Otherwise the very tokens we keep exact would still skew the datatype
        # derived for every other token.
        n_sink = min(self._n_sink, S) if self._n_tokens == 0 else 0
        fit_slice = slice(n_sink, None) if n_sink > 0 and n_sink < S else slice(None)
        if refit_keys:
            self._capture_key_thresholds(keys[:, :, fit_slice, :])

        # Flatten (B, H) into one leading axis — every batched primitive
        # below operates on this axis instead of a Python for-b/for-h loop
        # (see VeloxQuant-MLX#504: that loop, plus a per-Lloyd-Max-iteration
        # forced host sync, measured a 59x real end-to-end decode slowdown).
        keys_bh = keys.reshape(B * H, S, D)
        values_bh = values.reshape(B * H, S, D)

        if refit_keys:
            # Matches prior behaviour exactly: key levels are fit ONLY from
            # batch element 0's data (keys[0]), then shared across every
            # batch element — never fit independently per b, even though
            # each (b, h) pair is otherwise processed independently. This
            # mirrors the old per-head loop, which fit `new_klev[h]` once
            # at b==0 and reused it (unchanged) for b==1..B-1 (see #504
            # investigation notes — verified against the pre-batching code).
            keys_h0 = keys[0].reshape(H, S, D)  # [H, S, D], batch element 0 only
            fit_inliers = split_dense_sparse_batched(
                keys_h0[:, fit_slice, :], self._outlier_fraction
            ).inliers
            key_levels_h = fit_nuq_levels_batched(
                fit_inliers, self._bits, self._lloyd_iters
            )  # [H, L, D]
            key_levels = mx.tile(key_levels_h, (B, 1, 1))  # [B*H, L, D]
        else:
            # Frozen levels are stored per-head [H, L, D]; tile across B to
            # match the flattened BH axis (shared across the batch, as
            # before — frozen keys are fit once and reused for every batch
            # element). `not refit_keys` implies `not is_prefill`, so
            # self._key_levels is guaranteed set here (assert makes that
            # explicit for readers and satisfies static typing).
            assert self._key_levels is not None
            key_levels = mx.tile(self._key_levels, (B, 1, 1))

        k_out_bh, klev_used = self._quant_keys_batched(keys_bh, key_levels)
        v_out_bh, vlev_used = self._quant_values_batched(values_bh)

        if refit_keys:
            # Store per-head levels (first B-tile is representative — frozen
            # levels are shared across the batch by construction above).
            self._key_levels = klev_used[:H]
        self._value_levels = [vlev_used[0]]  # most-recent per-token levels (introspection)

        k_out = k_out_bh.reshape(B, H, S, D)
        v_out = v_out_bh.reshape(B, H, S, D)

        # Attention Sink-Aware quantization (paper §3.5): the model allocates a
        # disproportionate attention score to the first few tokens and is
        # correspondingly sensitive to quantization error there, so those
        # positions are restored to their exact fp16 values. Only the leading
        # positions of the *whole sequence* are sinks, so this applies on the
        # prefill call (n_tokens == 0), never to mid-stream decode tokens.
        if n_sink > 0:
            k_out = mx.concatenate(
                [keys[:, :, :n_sink, :].astype(mx.float16), k_out[:, :, n_sink:, :]], axis=2
            )
            v_out = mx.concatenate(
                [values[:, :, :n_sink, :].astype(mx.float16), v_out[:, :, n_sink:, :]], axis=2
            )
            self._sink_kept = n_sink

        return k_out, v_out

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Fit (or reuse frozen) per-channel key / per-token value NUQ levels, isolate outliers, and preserve sink tokens in fp16; return dequantized K/V."""
        B, H, S, D = keys.shape
        n_sink = min(self._n_sink, S) if self._n_tokens == 0 else 0
        k_out, v_out = self._apply(keys, values)
        self._n_tokens += S
        self._account_bytes(B, H, S, D, n_sink)
        return super().update_and_fetch(k_out, v_out)

    def _account_bytes(self, B: int, H: int, S: int, D: int, n_sink: int = 0) -> None:
        L = 1 << self._bits
        # Sink tokens are stored as raw fp16 and are not coded at all.
        s_q = max(0, S - n_sink)
        # Codes: bits per element. Level table: L fp16 per channel (keys) / per
        # token (values). Outlier side-channel: fp16 value + ~index bits.
        code_bytes = math.ceil(s_q * D * self._bits / 8)
        key_table_bytes = L * D * 2  # per-channel table
        val_table_bytes = L * s_q * 2  # per-token table (one per coded token)
        idx_bits = max(1, math.ceil(math.log2(max(2, max(1, s_q) * D))))
        # Charge the outliers actually produced this call, not the nominal
        # fraction: the rank-based split rounds per column, and the decode path
        # uses a carried-over threshold, so realized counts differ from
        # S * D * outlier_fraction. Accounting must follow the real side-channel.
        n_out_per_head = self._outlier_count - self._prev_outlier_count
        self._prev_outlier_count = self._outlier_count
        # _outlier_count aggregates keys+values across the whole B*H loop.
        per_head_out = n_out_per_head / max(1, 2 * B * H)
        outlier_bytes = int(round(per_head_out * (2 + math.ceil(idx_bits / 8))))
        sink_bytes = n_sink * D * 2  # exact fp16 sink rows

        self._compressed_key_bytes += (
            (code_bytes + key_table_bytes + outlier_bytes + sink_bytes) * B * H
        )
        self._compressed_value_bytes += (
            (code_bytes + val_table_bytes + outlier_bytes + sink_bytes) * B * H
        )
        self._fp16_key_bytes += B * H * S * D * 2
        self._fp16_value_bytes += B * H * S * D * 2

    # ------------------------------------------------------------------
    # `mlx_lm.server`'s `ModelProvider.load()` decides whether to route
    # requests through `BatchGenerator` (continuous batching) purely by
    # `hasattr(c, "merge")` on a probe cache — see #15/#357. The base
    # `KVCache` class this inherits from defines `merge()` as a classmethod
    # that returns a plain `mlx_lm.models.cache.BatchKVCache`, oblivious to
    # the frozen NUQ levels, outlier thresholds, and sink bookkeeping this
    # class needs. Left inherited, every request (even a lone one — a batch
    # of size 1 is still merged for uniform batch-shape handling) silently
    # replaces this cache with that generic one: no quantization, no
    # outlier isolation, no sink protection, while the server believes it
    # is still running `kvquant`. This hides `merge` from `hasattr` instead
    # (a bare classmethod override wouldn't: `hasattr` would still see it as
    # present and callable). That makes `is_batchable` correctly report
    # `False`, routing `kvquant` through `mlx_lm.server`'s sequential
    # `_serve_single` path instead, where this class already runs
    # correctly. See VeloxQuant-MLX#358: this defect turned out to affect
    # essentially every custom cache class in the repo (37, not the ~11
    # originally scoped), not just eviction/hybrid methods.
    merge = property(
        lambda self: (_ for _ in ()).throw(
            AttributeError(
                "KVQuantKVCache does not support batched merging; use it via "
                "the sequential serving path (see class docstring)."
            )
        )
    )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def nuq_bits(self) -> int:
        """Configured non-uniform quantization bit-width."""
        # Deliberately not named `.bits` — mlx_lm's SDPA checks
        # `hasattr(cache, "bits")` to route to its quantized-matmul kernel,
        # which expects mx.quantize's native tuple layout and a `.group_size`
        # attribute this cache doesn't have. Exposing `.bits` here would
        # silently hijack attention dispatch. See other *_cache.py classes
        # (e.g. kivi_cache.py, turboquant_rvq_cache.py) for the same guard.
        return self._bits

    @property
    def outlier_fraction(self) -> float:
        """Configured nominal fraction of elements treated as dense-and-sparse outliers."""
        return self._outlier_fraction

    @property
    def outlier_count(self) -> int:
        """Realized count of elements isolated as outliers (K + V), across all calls so far."""
        return self._outlier_count

    @property
    def n_sink(self) -> int:
        """Configured number of leading (attention-sink) tokens kept in fp16."""
        return self._n_sink

    @property
    def sink_kept(self) -> int:
        """Sink tokens actually retained in fp16 (bounded by the prefill length)."""
        return self._sink_kept

    @property
    def key_outlier_thresh(self):
        """Frozen per-channel |key| outlier threshold reused during decode."""
        return self._key_outlier_thresh

    @property
    def key_levels(self):
        """Frozen per-channel non-uniform quantization levels fit at prefill."""
        return self._key_levels

    @property
    def value_levels(self):
        """Frozen per-token non-uniform quantization levels fit at prefill."""
        return self._value_levels

    @property
    def compressed_key_bytes(self) -> int:
        """Realized stored bytes for the compressed key cache (NUQ codes + per-channel level table + fp16 outlier/sink side-channel, all heads/batches)."""
        return self._compressed_key_bytes

    @property
    def compressed_value_bytes(self) -> int:
        """Realized stored bytes for the compressed value cache (NUQ codes + per-token level table + fp16 outlier/sink side-channel, all heads/batches)."""
        return self._compressed_value_bytes

    @property
    def fp16_key_bytes(self) -> int:
        """Hypothetical fp16 key cost if nothing were compressed."""
        return self._fp16_key_bytes

    @property
    def fp16_value_bytes(self) -> int:
        """Hypothetical fp16 value cost if nothing were compressed."""
        return self._fp16_value_bytes

    @property
    def effective_bits(self) -> float:
        """Effective per-element key bits (codes + table + outliers vs fp16)."""
        if self._fp16_key_bytes == 0:
            return float(self._bits)
        return 16.0 * self._compressed_key_bytes / self._fp16_key_bytes


__all__ = ["KVQuantKVCache"]
