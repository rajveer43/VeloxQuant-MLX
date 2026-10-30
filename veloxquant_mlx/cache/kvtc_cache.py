"""KVTC-adapted KV cache — local PCA + DP-optimal per-component bit allocation + entropy coding.

Inspired by "KV Cache Transform Coding for Compact Storage in LLM Inference"
(NVIDIA, **ICLR 2026**, accepted poster, arXiv:2511.01815). Documented as
"KVTC-adapted (VeloxQuant-MLX implementation)" — not a faithful port.

Modeled on :class:`~veloxquant_mlx.cache.palu_cache.PALUKVCache` (both K and
V compressed, true-latent storage — not SVDq's keys-only, reconstruct-fp16
scope). The mechanism this method adds over Palu/SVDq/SpectralQuant: a
**DP-optimal, discrete, per-component** bit allocation (may drop a
component entirely, i.e. assign it 0 bits) instead of a fixed hand-chosen
split, followed by a real, measured **entropy-coding** stage. See
``quantizers/kvtc.py`` and ``allocators/kvtc_dp.py`` for the mechanism and
the honesty crux (local per-sequence PCA vs the paper's pre-calibrated
global basis; analytic distortion proxy vs the paper's real-activation-fit
rate-distortion model; order-0 Huffman coder vs the paper's possibly more
sophisticated scheme; paper's headline numbers are the paper's, on trained
models, never reproduced here).

NOT PATH-DEPENDENT ON DECODE (contrast with the eviction family)
-----------------------------------------------------------------
Unlike H2O/TOVA/MorphKV/KVzip (eviction, path-dependent keep-sets), KVTC is
in the Palu/SVDq/SpectralQuant family: the local PCA basis (``V``, ``mean``)
and the DP-derived per-component bit allocation are refit on every
``update_and_fetch`` call with ``S > 1`` (each prefill chunk) over all rows
seen so far plus the chunk, from running Gram statistics, so they are not
frozen from a short first chunk (#893). Single-token decode calls reuse the
fitted basis **unchanged**: feeding the same decode tokens one at a time
after prefill produces the identical stored basis/allocation/codes/
decompressed output -- pinned by a determinism test in
``tests/cache/test_kvtc_cache.py``.

Design
------
Prefill (every call with ``S > 1``, refit on all rows so far):
  1. For keys and values independently: run local PCA (via
     ``quantizers/kvtc.py::kvtc_compress``) on the prefill batch, at
     ``kvtc_bit_budget`` total bits per token. Store the resulting
     projection ``V``, ``mean``, and per-component ``bit_allocation`` as
     fixed layer state.
  2. Store the (fixed-basis) latent codes for the prefill batch.

Decode (subsequent calls, any ``S``):
  1. Project the new key/value into the **already-fitted** basis
     (``(x - mean) @ V``), quantize each component at its **already-fixed**
     bit allocation, and grow the entropy-coded latent store.

Because true latent (entropy-coded) storage is not a fixed-shape tensor the
parent ``mlx_lm`` fp16 ring buffer cannot hold it — like
:class:`PALUKVCache`, this class bypasses the parent buffer entirely and
manages its own offset, re-running :func:`~veloxquant_mlx.quantizers.kvtc.kvtc_compress`
over the accumulated raw rows on every call (the DP allocation and PCA
basis are frozen after prefill; only the per-token codes grow). Every call
returns reconstructed fp16 ``[B, H, S_total, D]`` for the downstream SDPA
call; storage accounting is based on the realized entropy-coded bytes.

Byte accounting
----------------
``kvtc_bytes`` / ``full_seq_bytes`` / ``compression_ratio`` — key + value
combined, mirroring Palu's combined reporting style.
``pre_entropy_bytes`` — fixed-width (pre-entropy-coding) size, for
comparison.
``entropy_coding_gain`` — ``pre_entropy_bytes / kvtc_bytes``. Since #77,
``kvtc_compress`` guarantees the entropy-coded payload+table never exceeds
fixed-width packing (falling back to fixed-width when Huffman doesn't pay
for itself — see ``quantizers/kvtc.py``'s module docstring), but this ratio
divides by the FULL stored size (``kvtc_bytes``, which also carries the
fixed projection-basis overhead unrelated to entropy coding), so it can
still legitimately read < 1 when that basis dominates — it is not itself
guaranteed >= 1.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.allocators.kvtc_dp import (
    DEFAULT_BETA,
    DEFAULT_BIT_CHOICES,
    dp_allocate_bits,
)
from veloxquant_mlx.core.exceptions import QuantizerConfigError
from veloxquant_mlx.quantizers.kvtc import (
    KVTCArtifact,
    _encode_survived_codes,
    kvtc_fp16_bytes,
    kvtc_pca_from_gram,
    kvtc_pre_entropy_bytes,
    quantize_component,
)


class _TensorKVTC:
    """Per-tensor (keys or values), single (batch, head) KVTC state.

    Holds the raw fp32 rows accumulated so far (needed because the
    compressed artifact must cover the full accumulated sequence for
    correct reconstruction, even though the local PCA basis and DP bit
    allocation are fit ONCE at prefill and never re-fit) plus the frozen
    basis/allocation from prefill.
    """

    def __init__(self, bit_budget: int, bit_choices: tuple[int, ...], beta: float) -> None:
        self.bit_budget = bit_budget
        self.bit_choices = bit_choices
        self.beta = beta

        self._raw_rows: mx.array | None = None  # [S_total, D] fp32, accumulated
        self._artifact: KVTCArtifact | None = None
        self._fitted = False
        self._L_np: np.ndarray | None = None  # [S_total, r] float64 latents of _raw_rows
        self._pending: tuple | None = None  # codes not yet entropy-encoded (#841)
        # Running Gram matrix / row sum / row count of every raw row, for the
        # per-chunk refit (#893).
        self._gram: np.ndarray | None = None
        self._row_sum: np.ndarray | None = None
        self._n_rows = 0

    def fit_chunk(self, x_new: mx.array) -> mx.array:
        """Refit the PCA basis and DP bit allocation on every row seen so far plus ``x_new`` [S, D].

        Returns the reconstruction of all rows, [S_total, D] fp16. The basis
        comes from running Gram statistics and the codes stay unencoded until
        the byte accounting asks for them (as in :meth:`append`), so a refit
        costs O(D^3) plus a pass over the rows, not an SVD or an entropy
        encode of the whole sequence (#893).
        """
        x32 = x_new.astype(mx.float32)
        x64 = np.asarray(x32, dtype=np.float64)
        if self._gram is None:
            self._gram, self._row_sum = x64.T @ x64, x64.sum(axis=0)
            self._raw_rows = x32
        else:
            self._gram += x64.T @ x64
            self._row_sum += x64.sum(axis=0)
            self._raw_rows = mx.concatenate([self._raw_rows, x32], axis=0)
        self._n_rows += x64.shape[0]
        V, mean, variances = kvtc_pca_from_gram(self._gram, self._row_sum, self._n_rows)
        bit_alloc = dp_allocate_bits(
            variances, self.bit_budget, bit_choices=self.bit_choices, beta=self.beta
        )
        survived_idx = np.where(bit_alloc > 0)[0]
        self._artifact = KVTCArtifact(
            V=V,
            mean=mean,
            bit_allocation=bit_alloc,
            S=self._n_rows,
            n_survived=int(survived_idx.shape[0]),
            entropy_payload=b"",
            entropy_table={},
            quant_min=np.zeros(0),
            quant_scale=np.zeros(0),
            survived_idx=survived_idx,
            is_entropy_coded=False,
        )
        self._fitted = True
        self._L_np = None
        return self._requantize()

    def append(self, x: mx.array) -> mx.array:
        """Absorb new rows ``x`` [S, D] through the FIXED basis/allocation; return reconstructed [S_total, D] fp16.

        The projection basis (``V``, ``mean``) and the per-component bit
        allocation are frozen at prefill and never recomputed — only the
        quantization codes grow to cover the newly accumulated rows (each
        component's min/scale are also refit per call from the accumulated
        rows only insofar as any *new* row could, in principle, shift a
        component's observed min/max; the basis/allocation themselves never
        change, which is what the "not path-dependent" claim is about and
        what the determinism test pins).
        """
        assert self._fitted and self._artifact is not None and self._raw_rows is not None
        self._raw_rows = mx.concatenate([self._raw_rows, x.astype(mx.float32)], axis=0)
        return self._requantize(x.astype(mx.float32))

    def trim(self, n: int) -> None:
        """Drop the ``n`` most-recent rows and re-quantize through the frozen basis."""
        assert self._fitted and self._raw_rows is not None
        keep = max(0, self._raw_rows.shape[0] - n)
        self._raw_rows = self._raw_rows[:keep]
        if self._L_np is not None:
            self._L_np = self._L_np[:keep]
        if keep > 0:
            self._requantize()
            self._materialize()
        else:
            self._fitted = False
            self._artifact = None
            self._L_np = None
            self._pending = None
            self._gram = None
            self._row_sum = None
            self._n_rows = 0

    def _project(self, x: mx.array) -> np.ndarray:
        """Project fp32 rows ``x`` [S, D] onto the frozen basis; float64 [S, r]."""
        assert self._artifact is not None
        L = (x - self._artifact.mean[None, :]) @ self._artifact.V
        mx.eval(L)
        return np.asarray(L.tolist(), dtype=np.float64).reshape(x.shape[0], -1)

    def _requantize(self, new_rows: mx.array | None = None) -> mx.array:
        """Re-quantize the current rows through the frozen basis/allocation and return the reconstruction.

        Only ``new_rows`` are projected (the projection of older rows is
        cached in ``_L_np``). The per-component min/scale are still refit
        over all rows, exactly as before, but the entropy-coded artifact is
        NOT rebuilt here: Huffman/bit-packing is O(n) pure Python and made
        every decode step cost ~1 ms per cached token (#841). The codes are
        kept in ``_pending`` and encoded on demand by :meth:`_materialize`
        (byte accounting, trim). The reconstruction is the same values
        ``kvtc_decompress`` would return, because the entropy coding is
        lossless.
        """
        assert self._artifact is not None and self._raw_rows is not None
        art = self._artifact
        if (
            self._L_np is None
            or self._L_np.shape[0] + (0 if new_rows is None else new_rows.shape[0])
            != self._raw_rows.shape[0]
        ):
            self._L_np = self._project(self._raw_rows)
        elif new_rows is not None:
            self._L_np = np.concatenate([self._L_np, self._project(new_rows)])
        L_np = self._L_np
        n = L_np.shape[0]
        r = int(art.bit_allocation.shape[0])

        L_hat = np.zeros((n, r), dtype=np.float64)
        all_codes, mins, scales, bits_per_component = [], [], [], []
        for i in art.survived_idx:
            bits = int(art.bit_allocation[i])
            codes, lo, scale = quantize_component(L_np[:, i], bits)
            all_codes.append(codes)
            mins.append(lo)
            scales.append(scale)
            bits_per_component.append(bits)
            L_hat[:, i] = codes.astype(np.float64) * scale + lo

        self._pending = (all_codes, mins, scales, bits_per_component, n)
        x_hat = mx.array(L_hat.astype(np.float32)) @ art.V.T + art.mean[None, :]
        return x_hat.astype(mx.float16)

    def _materialize(self) -> None:
        """Entropy-encode the pending codes into ``_artifact`` (no-op when current)."""
        if self._pending is None or self._artifact is None:
            return
        all_codes, mins, scales, bits_per_component, n = self._pending
        payload, table, is_entropy_coded = _encode_survived_codes(all_codes, bits_per_component)
        old = self._artifact
        self._artifact = KVTCArtifact(
            V=old.V,
            mean=old.mean,
            bit_allocation=old.bit_allocation,
            S=int(n),
            n_survived=old.n_survived,
            entropy_payload=payload,
            entropy_table=table,
            quant_min=np.asarray(mins, dtype=np.float64),
            quant_scale=np.asarray(scales, dtype=np.float64),
            survived_idx=old.survived_idx,
            is_entropy_coded=is_entropy_coded,
        )
        self._pending = None

    # ------------------------------------------------------------------
    @property
    def stored_bytes(self) -> int:
        """Realized entropy-coded storage size for this tensor's fitted artifact."""
        if self._artifact is None:
            return 0
        self._materialize()
        return kvtc_fp16_bytes(self._artifact)

    @property
    def pre_entropy_bytes(self) -> int:
        """Fixed-width (pre-entropy-coding) storage size for this tensor's fitted artifact."""
        if self._artifact is None:
            return 0
        self._materialize()
        return kvtc_pre_entropy_bytes(self._artifact)

    @property
    def n_survived(self) -> int:
        """Number of principal components that survived DP allocation (nonzero bits assigned)."""
        return 0 if self._artifact is None else self._artifact.n_survived


class KVTCKVCache(_MLXKVCache):
    """KV cache implementing KVTC-adapted local-PCA + DP-optimal bit allocation + entropy coding.

    Args:
        config: :class:`KVCacheConfig`. Fields consumed:
            ``head_dim`` (D),
            ``kvtc_bit_budget`` (int, default ``4 * head_dim`` — total bits
                per token across all principal components, for K and V
                independently; "4 avg-bits-per-component equivalent scaled
                by head_dim"),
            ``kvtc_bit_choices`` (tuple[int, ...], default
                ``allocators.kvtc_dp.DEFAULT_BIT_CHOICES``),
            ``kvtc_beta`` (float, default
                ``allocators.kvtc_dp.DEFAULT_BETA``).

    Notes:
        Applies to **both K and V independently** (mirrors Palu's scope,
        not SVDq's keys-only scope) — see module docstring.
        No ``.bits`` attribute (this isn't a fixed-bit-width quantizer in
        that sense, and it isn't an eviction cache either).
        Single (batch, head) state per layer; ``B`` and ``H`` handled by
        looping (mirrors PALUKVCache's per-head loop).
        Bypasses the parent fp16 ring buffer (true latent storage), like
        PALUKVCache — manages its own ``offset``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._D = int(config.head_dim)
        self._bit_budget = int(getattr(config, "kvtc_bit_budget", 4 * self._D))
        self._bit_choices = tuple(getattr(config, "kvtc_bit_choices", DEFAULT_BIT_CHOICES))
        self._beta = float(getattr(config, "kvtc_beta", DEFAULT_BETA))

        if self._bit_budget < 0:
            raise QuantizerConfigError(
                f"KVTCKVCache: kvtc_bit_budget must be >= 0, got {self._bit_budget!r}"
            )

        self._keys_states: list[_TensorKVTC] = []
        self._vals_states: list[_TensorKVTC] = []
        self._kvtc_offset = 0
        self._B = 0
        self._H = 0

        self._full_seq_bytes = 0
        # Last (k_out, v_out) reconstructed by update_and_fetch, for .state —
        # mlx_lm's generate() reads cache.state directly (e.g. during chunked
        # prefill and the speculative-decoding rewind path); since self.keys/
        # self.values are never populated (true latent storage), .state must
        # be overridden to serve this instead (see #83).
        self._last_state: tuple | None = None

    # ------------------------------------------------------------------
    # `mlx_lm.server`'s `ModelProvider.load()` decides whether to route
    # requests through `BatchGenerator` (continuous batching) purely by
    # `hasattr(c, "merge")` on a probe cache — see #15/#357/#358. The base
    # `KVCache` class this inherits from defines `merge()` as a classmethod
    # that returns a plain `mlx_lm.models.cache.BatchKVCache`, oblivious to
    # the frozen PCA basis, DP bit allocation, and entropy-coded state this
    # class needs. Left inherited, every request (even a lone one — a batch
    # of size 1 is still merged for uniform batch-shape handling) silently
    # replaces this cache with that generic one: no PCA, no DP allocation,
    # no entropy coding, while the server believes it is still running
    # `kvtc`. This hides `merge` from `hasattr` instead (a bare classmethod
    # override wouldn't: `hasattr` would still see it as present and
    # callable). That makes `is_batchable` correctly report `False`, routing
    # `kvtc` through `mlx_lm.server`'s sequential `_serve_single` path
    # instead, where this class already runs correctly. See
    # VeloxQuant-MLX#358 for the full 37-method scope of this defect.
    merge = property(
        lambda self: (_ for _ in ()).throw(
            AttributeError(
                "KVTCKVCache does not support batched merging; use it via "
                "the sequential serving path (see class docstring)."
            )
        )
    )

    # ------------------------------------------------------------------
    def _ensure_states(self, B: int, H: int) -> None:
        if not self._keys_states:
            self._B, self._H = B, H
            self._keys_states = [
                _TensorKVTC(self._bit_budget, self._bit_choices, self._beta) for _ in range(B * H)
            ]
            self._vals_states = [
                _TensorKVTC(self._bit_budget, self._bit_choices, self._beta) for _ in range(B * H)
            ]

    def _idx(self, b: int, h: int) -> int:
        return b * self._H + h

    # ------------------------------------------------------------------
    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Refit the local PCA basis + DP bit allocation on every multi-row call (prefill chunk), else append through the fitted basis on decode; return reconstructed fp16 K/V."""
        B, H, S, D = keys.shape
        self._ensure_states(B, H)
        n_new = S

        # A multi-row call always (re)fits the basis and bit allocation on all
        # rows so far plus this chunk (#893); single-row decode calls append
        # through the fitted basis.
        is_prefill = S > 1 or not self._keys_states[0]._fitted

        if is_prefill:
            # Rows held so far (single-token calls before any fit) are raw and
            # live in ``_last_state``. The PCA basis and bit allocation are
            # frozen after the first fit, and one row gives a degenerate basis
            # (#840), so keep rows lossless until a call carries more than one,
            # then fit on everything held.
            held_k, held_v = self._last_state if self._last_state is not None else (None, None)
            if S == 1:
                k_out = (
                    keys
                    if held_k is None
                    else mx.concatenate([held_k, keys.astype(held_k.dtype)], axis=2)
                )
                v_out = (
                    values
                    if held_v is None
                    else mx.concatenate([held_v, values.astype(held_v.dtype)], axis=2)
                )
                self._kvtc_offset += S
                self._full_seq_bytes += B * H * S * D * 2 * 2  # K + V, fp16
                self._last_state = (k_out, v_out)
                return k_out, v_out
            if not self._keys_states[0]._fitted and held_k is not None and self._kvtc_offset > 0:
                keys = mx.concatenate([held_k.astype(keys.dtype), keys], axis=2)
                values = mx.concatenate([held_v.astype(values.dtype), values], axis=2)
                S = keys.shape[2]

        k_out_b, v_out_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                idx = self._idx(b, h)
                ks = self._keys_states[idx]
                vs = self._vals_states[idx]

                k_bh = keys[b, h]
                v_bh = values[b, h]

                if is_prefill:
                    k_rec = ks.fit_chunk(k_bh)
                    v_rec = vs.fit_chunk(v_bh)
                else:
                    k_rec = ks.append(k_bh)
                    v_rec = vs.append(v_bh)

                k_out_h.append(k_rec)
                v_out_h.append(v_rec)
            k_out_b.append(mx.stack(k_out_h, axis=0))
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0).astype(keys.dtype)
        V_out = mx.stack(v_out_b, axis=0).astype(values.dtype)

        self._kvtc_offset += n_new
        self._full_seq_bytes += B * H * n_new * D * 2 * 2  # K + V, fp16
        self._last_state = (K_out, V_out)
        return K_out, V_out

    # ------------------------------------------------------------------
    # mlx_lm KVCache surface — own offset (true latent storage, like Palu)
    # ------------------------------------------------------------------
    @property
    def offset(self) -> int:  # type: ignore[override]
        """Number of tokens accumulated so far (own counter — parent ring buffer is bypassed)."""
        return self._kvtc_offset

    @offset.setter
    def offset(self, v: int) -> None:
        """Restore the token offset (e.g. from a saved cache state)."""
        self._kvtc_offset = int(v)

    def size(self) -> int:
        """Number of tokens currently stored."""
        return self._kvtc_offset

    @property
    def state(self):  # type: ignore[override]
        """Last reconstructed fp16 (keys, values), since the true-latent buffers bypass the parent ring buffer ``mlx_lm`` reads directly."""
        if self._last_state is None:
            empty = mx.zeros((1, self._H, 0, self._D), dtype=mx.float16)
            return empty, empty
        return self._last_state

    @state.setter
    def state(self, v) -> None:
        """Restore ``(keys, values)`` as the last reconstructed state and resync the offset."""
        k, v_ = v
        self._last_state = (k, v_)
        self._kvtc_offset = int(k.shape[2])

    def is_trimmable(self) -> bool:
        """Whether this cache supports trimming."""
        return True

    def trim(self, n: int) -> int:
        """Drop the ``n`` most-recently-added tokens, re-quantizing through the frozen basis; returns the number actually dropped."""
        n = min(self._kvtc_offset, n)
        if n <= 0:
            return 0
        # Before the first fit the rows are raw in ``_last_state`` only (#840).
        if self._keys_states and self._keys_states[0]._fitted:
            for ks in self._keys_states:
                ks.trim(n)
            for vs in self._vals_states:
                vs.trim(n)
        self._kvtc_offset -= n
        if self._last_state is not None:
            k, v_ = self._last_state
            self._last_state = (k[:, :, : k.shape[2] - n, :], v_[:, :, : v_.shape[2] - n, :])
        return n

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @property
    def kvtc_bytes(self) -> int:
        """Realized total stored bytes (K + V, all heads/batches), including
        the entropy-coded payload, code table, projection basis, and quant
        params — never the pre-entropy-coding size."""
        return sum(s.stored_bytes for s in self._keys_states) + sum(
            s.stored_bytes for s in self._vals_states
        )

    # ``cli/telemetry.py``'s ``/v1/kv/stats`` (and, downstream, the macOS
    # app's live panel) discovers a method's byte coverage by probing for
    # exactly these four attribute names (``compressed_key_bytes`` /
    # ``compressed_value_bytes`` / ``fp16_key_bytes`` / ``fp16_value_bytes``)
    # — see ``registry.py::telemetry_coverage`` and every sibling latent-
    # storage cache this class is modeled on (``palu_cache.py``,
    # ``svdq_cache.py``). Without them, ``kvtc`` silently probed as
    # ``TelemetryCoverage.NONE``: `/v1/kv/stats` reported
    # ``"not_reported_reason": "this method does not report byte or token
    # counters"`` even while genuinely compressing, and the app's control
    # panel showed nothing at all for a servable, compressing method.
    # ``_keys_states``/``_vals_states`` already track K and V independently
    # (unlike ``kvtc_bytes``, which sums both), so no new accounting is
    # needed — these just expose the existing split.
    @property
    def nbytes(self) -> int:  # type: ignore[override]
        """Total realized stored bytes (K + V); the inherited base-buffer size is always 0 here (#841)."""
        return self.kvtc_bytes

    @property
    def compressed_key_bytes(self) -> int:
        """Realized stored bytes for the compressed key cache (entropy-coded payload + table + basis, all heads/batches)."""
        return sum(s.stored_bytes for s in self._keys_states)

    @property
    def compressed_value_bytes(self) -> int:
        """Realized stored bytes for the compressed value cache (entropy-coded payload + table + basis, all heads/batches)."""
        return sum(s.stored_bytes for s in self._vals_states)

    @property
    def fp16_key_bytes(self) -> int:
        """Hypothetical fp16 key cost if nothing were compressed."""
        return self._full_seq_bytes // 2

    @property
    def fp16_value_bytes(self) -> int:
        """Hypothetical fp16 value cost if nothing were compressed."""
        return self._full_seq_bytes // 2

    @property
    def pre_entropy_bytes(self) -> int:
        """Fixed-width (pre-entropy-coding) size, K + V, all heads/batches."""
        return sum(s.pre_entropy_bytes for s in self._keys_states) + sum(
            s.pre_entropy_bytes for s in self._vals_states
        )

    @property
    def entropy_coding_gain(self) -> float:
        """``pre_entropy_bytes / kvtc_bytes`` — realized entropy-coding gain.

        A modest, honestly-scoped secondary effect (see
        ``quantizers/_entropy_coding.py``): NOT the theoretical
        Shannon-entropy bound. Since #77, the entropy-coding stage itself
        (payload + table) is guaranteed to never exceed fixed-width packing
        — but this ratio's denominator (``kvtc_bytes``) also includes the
        fixed projection-basis overhead, which is unrelated to entropy
        coding and can dominate at small survived-component counts, so this
        property can still read < 1 for that reason.
        """
        kb = self.kvtc_bytes
        if kb == 0:
            return 1.0
        return self.pre_entropy_bytes / kb

    @property
    def full_seq_bytes(self) -> int:
        """Hypothetical fp16 K + V cost if nothing were compressed."""
        return self._full_seq_bytes

    @property
    def compression_ratio(self) -> float:
        """``full_seq_bytes / kvtc_bytes``; > 1 means memory savings over fp16."""
        kb = self.kvtc_bytes
        if kb == 0:
            return 1.0
        return self._full_seq_bytes / kb

    @property
    def bit_budget(self) -> int:
        """Configured total bits per token across all principal components (K and V independently)."""
        return self._bit_budget


__all__ = ["KVTCKVCache"]
