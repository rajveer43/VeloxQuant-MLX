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

NOT PATH-DEPENDENT (contrast with the eviction family)
--------------------------------------------------------
Unlike H2O/TOVA/MorphKV/KVzip (eviction, path-dependent keep-sets), KVTC is
in the Palu/SVDq/SpectralQuant family: the local PCA basis (``V``, ``mean``)
and the DP-derived per-component bit allocation are fit **once**, at the
first ``update_and_fetch`` call with ``S > 1`` (prefill), and reused
**unchanged** for every subsequent token (prefill continuation or decode).
Feeding the same sequence one token at a time after prefill vs. all at once
produces the identical stored basis/allocation/codes/decompressed output —
pinned by a determinism test in
``tests/cache/test_kvtc_cache.py``.

Design
------
Prefill (first call, ``S > 1``):
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
``entropy_coding_gain`` — ``pre_entropy_bytes / kvtc_bytes`` (>= 1 typically;
a modest, honestly-reported secondary effect on synthetic Gaussian-like
data — see the module docstring in ``quantizers/_entropy_coding.py``).
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.allocators.kvtc_dp import DEFAULT_BETA, DEFAULT_BIT_CHOICES
from veloxquant_mlx.quantizers._entropy_coding import entropy_encode
from veloxquant_mlx.quantizers.kvtc import (
    KVTCArtifact,
    kvtc_compress,
    kvtc_decompress,
    kvtc_fp16_bytes,
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

        self._raw_rows: Optional[mx.array] = None  # [S_total, D] fp32, accumulated
        self._artifact: Optional[KVTCArtifact] = None
        self._fitted = False

    def fit_prefill(self, x0: mx.array) -> mx.array:
        """Fit the basis + allocation from the prefill batch ``x0`` [S, D]; return reconstructed [S, D] fp16."""
        self._raw_rows = x0.astype(mx.float32)
        self._artifact = kvtc_compress(
            self._raw_rows, self.bit_budget, bit_choices=self.bit_choices, beta=self.beta
        )
        self._fitted = True
        return kvtc_decompress(self._artifact)

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
        return self._requantize()

    def trim(self, n: int) -> None:
        """Drop the ``n`` most-recent rows and re-quantize through the frozen basis."""
        assert self._fitted and self._raw_rows is not None
        keep = max(0, self._raw_rows.shape[0] - n)
        self._raw_rows = self._raw_rows[:keep]
        if keep > 0:
            self._requantize()
        else:
            self._fitted = False
            self._artifact = None

    def _requantize(self) -> mx.array:
        """Re-derive the entropy-coded artifact for the current ``_raw_rows`` through the frozen basis/allocation."""
        assert self._artifact is not None and self._raw_rows is not None
        V = self._artifact.V
        mean = self._artifact.mean
        bit_alloc = self._artifact.bit_allocation

        # Re-quantize the accumulated rows through the FROZEN basis and
        # FROZEN bit allocation (only the per-token codes/quant-params grow;
        # the DP is never re-invoked after prefill).
        x_centered = self._raw_rows - mean[None, :]
        L = x_centered @ V
        mx.eval(L)
        L_np = np.asarray(L.tolist(), dtype=np.float64)

        survived_idx = self._artifact.survived_idx

        all_codes = []
        mins, scales = [], []
        for i in survived_idx:
            bits = int(bit_alloc[i])
            codes, lo, scale = quantize_component(L_np[:, i], bits)
            all_codes.append(codes)
            mins.append(lo)
            scales.append(scale)
        flat_codes = np.concatenate(all_codes) if all_codes else np.zeros((0,), dtype=np.int64)
        payload, table = entropy_encode(flat_codes)

        self._artifact = KVTCArtifact(
            V=V,
            mean=mean,
            bit_allocation=bit_alloc,
            S=int(self._raw_rows.shape[0]),
            n_survived=self._artifact.n_survived,
            entropy_payload=payload,
            entropy_table=table,
            quant_min=np.asarray(mins, dtype=np.float64),
            quant_scale=np.asarray(scales, dtype=np.float64),
            survived_idx=survived_idx,
        )
        return kvtc_decompress(self._artifact)

    # ------------------------------------------------------------------
    @property
    def stored_bytes(self) -> int:
        if self._artifact is None:
            return 0
        return kvtc_fp16_bytes(self._artifact)

    @property
    def pre_entropy_bytes(self) -> int:
        if self._artifact is None:
            return 0
        return kvtc_pre_entropy_bytes(self._artifact)

    @property
    def n_survived(self) -> int:
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
            raise ValueError(f"KVTCKVCache: kvtc_bit_budget must be >= 0, got {self._bit_budget!r}")

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
        self._last_state: Optional[tuple] = None

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
        B, H, S, D = keys.shape
        self._ensure_states(B, H)

        k_out_b, v_out_b = [], []
        for b in range(B):
            k_out_h, v_out_h = [], []
            for h in range(H):
                idx = self._idx(b, h)
                ks = self._keys_states[idx]
                vs = self._vals_states[idx]

                k_bh = keys[b, h]
                v_bh = values[b, h]

                if not ks._fitted:
                    k_rec = ks.fit_prefill(k_bh)
                    v_rec = vs.fit_prefill(v_bh)
                else:
                    k_rec = ks.append(k_bh)
                    v_rec = vs.append(v_bh)

                k_out_h.append(k_rec)
                v_out_h.append(v_rec)
            k_out_b.append(mx.stack(k_out_h, axis=0))
            v_out_b.append(mx.stack(v_out_h, axis=0))

        K_out = mx.stack(k_out_b, axis=0)
        V_out = mx.stack(v_out_b, axis=0)

        self._kvtc_offset += S
        self._full_seq_bytes += B * H * S * D * 2 * 2  # K + V, fp16
        self._last_state = (K_out, V_out)
        return K_out, V_out

    # ------------------------------------------------------------------
    # mlx_lm KVCache surface — own offset (true latent storage, like Palu)
    # ------------------------------------------------------------------
    @property
    def offset(self) -> int:  # type: ignore[override]
        return self._kvtc_offset

    @offset.setter
    def offset(self, v: int) -> None:
        self._kvtc_offset = int(v)

    def size(self) -> int:
        return self._kvtc_offset

    @property
    def state(self):  # type: ignore[override]
        if self._last_state is None:
            empty = mx.zeros((1, self._H, 0, self._D), dtype=mx.float16)
            return empty, empty
        return self._last_state

    @state.setter
    def state(self, v) -> None:
        k, v_ = v
        self._last_state = (k, v_)
        self._kvtc_offset = int(k.shape[2])

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self._kvtc_offset, n)
        if n <= 0:
            return 0
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
        Shannon-entropy bound, and it can be < 1 when the code table
        overhead outweighs the achieved bitstream savings at short
        sequence lengths.
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
        return self._bit_budget


__all__ = ["KVTCKVCache"]
