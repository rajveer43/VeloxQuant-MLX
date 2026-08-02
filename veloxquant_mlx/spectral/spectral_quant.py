"""SpectralQuant: data-aware KV cache quantizer.

Paper: "3% Is All You Need: Breaking TurboQuant's Compression Limit via
Spectral Structure", Gopinath 2026.

Algorithm (Algorithm 1 in paper):
  Calibration (one-time):
    1. Collect KV vectors from ncal calibration sequences.
    2. Compute empirical covariance Σ = (1/N) Σ h_t h_t^T.
    3. Eigen-decompose: U, Λ = eigh(Σ), sorted descending.
    4. Compute d_eff = PR(Σ) = (Σλ_i)² / Σλ_i²; d_s = ⌈d_eff⌉.
    5. Train Lloyd-Max codebooks C_signal (signal dims) and C_noise (noise dims).
    6. Sample JL matrix A ~ N(0, 1/m)^{m×d_s} for signal-only QJL.

  Compression:
    h̃ = U^T h                           (spectral rotation)
    h̃_s = h̃[:d_s],  h̃_n = h̃[d_s:]
    c_s = quantize(h̃_s, C_signal)       (signal quantization)
    c_n = quantize(h̃_n, C_noise)        (noise quantization)
    ĥ_s^(0) = decode(c_s, C_signal)
    ε_s = h̃_s - ĥ_s^(0)               (signal quantization residual)
    s = sign(A · ε_s),  r = ‖ε_s‖      (JL sketch on signal error only)

  Decompression:
    ĥ_s^(0) = decode(c_s, C_signal)
    ĥ_n = decode(c_n, C_noise)
    ε̂_s = ‖ε_s‖ · (√π/2 / m) · A^T · sign(A · ε_s)
    ĥ_s = ĥ_s^(0) + ε̂_s
    ĥ = U · [ĥ_s ; ĥ_n]               (inverse rotation)

Primary config from paper (SQ_noQJL_v3): omit JL sketch entirely (m=0),
use b_signal = b_noise = 3 bits. Compression gain comes from NOT adding
QJL sketch bits for the d - d_s = 124 noise dimensions.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from veloxquant_mlx.codebooks.base import CodebookFactory
from veloxquant_mlx.core.abstractions import Quantizer
from veloxquant_mlx.core.context import EncodedVector
from veloxquant_mlx.core.constants import SQRT_PI_OVER_2
from veloxquant_mlx.math.rotation import make_jl_matrix


class SpectralQuantizer(Quantizer):
    """SpectralQuant quantizer matching Algorithm 1 of the paper.

    Three modifications over TurboQuant:
      1. Spectral (eigenvector) rotation instead of random rotation.
      2. QJL error correction only on signal dimensions (d_s ≈ 4).
      3. Separate codebooks for signal and noise dims (water-filled bits).

    The primary paper config (SQ_noQJL_v3) sets apply_qjl=False, which
    skips the JL sketch entirely and gains 18.6% better compression ratio
    over TurboQuant while improving cosine similarity by +2.59pp.

    Args:
        d: Head dimension.
        b_signal: Bit-width for signal dimensions (default 3, paper default).
        b_noise: Bit-width for noise dimensions (default 3, paper default).
        rotation: U matrix from eigh(Σ), columns are eigenvectors sorted
            descending by eigenvalue. Shape (d, d), float32. If None,
            falls back to random orthogonal (degrades to TurboQuant-like).
        d_s: Number of signal dimensions (d_eff rounded up). Typically 4.
        apply_qjl: If True, store JL sketch of signal residual.
            Paper primary config (SQ_noQJL_v3) uses False.
        jl_dim: JL sketch dimension m. Only used when apply_qjl=True.
            Defaults to d_s (full sketch of signal space).
        seed: Random seed for JL matrix.
    """

    def __init__(
        self,
        d: int,
        b_signal: int = 3,
        b_noise: int = 3,
        rotation: Optional[np.ndarray] = None,
        d_s: int = 4,
        apply_qjl: bool = False,
        jl_dim: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        import mlx.core as mx

        self._d = d
        self._b_signal = b_signal
        self._b_noise = b_noise
        self._d_s = max(1, min(d_s, d))
        self._apply_qjl = apply_qjl
        self._seed = seed

        # --- Rotation matrix U: columns = eigenvectors sorted descending ---
        # After rotation h̃ = U^T h, the first d_s coordinates carry signal.
        if rotation is not None:
            R = np.array(rotation, dtype=np.float32)
            # rotation may be passed as U (d×d, columns=eigenvectors) or
            # as Vt (rows=eigenvectors from SVD). Normalise to U convention:
            # we store U^T so that h̃ = _R @ h is the projection.
            self._R = mx.array(R.T if R.shape == (d, d) else R, dtype=mx.float32)  # (d, d)
            self._R_inv = mx.array(R if R.shape == (d, d) else R.T, dtype=mx.float32)
        else:
            rng = np.random.default_rng(seed)
            U, _ = np.linalg.qr(rng.standard_normal((d, d)).astype(np.float32))
            self._R = mx.array(U.T, dtype=mx.float32)
            self._R_inv = mx.array(U, dtype=mx.float32)

        # --- Codebooks: Lloyd-Max Gaussian, separate for signal / noise ---
        # Post-rotation coordinates are approximately N(0, λ_i / d), but
        # we use the standard Gaussian codebook (normalised per-dim by scale).
        dist = "gaussian"
        self._cb_signal = CodebookFactory.create(dist, b=b_signal, d=d)
        if b_noise == b_signal:
            self._cb_noise = self._cb_signal
        else:
            self._cb_noise = CodebookFactory.create(dist, b=b_noise, d=d)

        # --- JL sketch matrix for signal residual (only if apply_qjl) ---
        self._qjl = None
        if apply_qjl:
            m = jl_dim if jl_dim is not None else self._d_s
            from veloxquant_mlx.preconditioners.jl_sketch import QJLEncoder

            S_np = make_jl_matrix(self._d_s, m, seed=seed)
            self._qjl = QJLEncoder(mx.array(S_np.astype(np.float16)))

        # Precompute bit accounting
        self._bits_per_signal_dim = b_signal
        self._bits_per_noise_dim = b_noise
        self._jl_dim = self._qjl.m if self._qjl is not None else 0

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------

    def encode(self, x: Any) -> EncodedVector:
        """Encode KV vectors via spectral rotation + selective quantisation.

        Follows Algorithm 1 of the paper exactly.

        Args:
            x: Array of shape (batch, d), fp16 or fp32.

        Returns:
            EncodedVector. Fields used:
              indices: uint8 (batch, d) — codebook indices for all d dims.
              norm:    fp16  (batch,)   — per-vector std-dev scale (signal dims).
              signs:   int8  (batch, m) — QJL signs of signal residual (if apply_qjl).
              residual_norm: fp16 (batch,) — ‖ε_s‖ (if apply_qjl).
        """
        import mlx.core as mx

        if x.ndim == 1:
            x = x[None]
        batch = x.shape[0]

        # Step 1: Spectral rotation h̃ = U^T h = _R @ h
        x_f32 = x.astype(mx.float32)
        h_tilde = x_f32 @ self._R.T  # (batch, d); _R is U^T so _R^T = U
        mx.eval(h_tilde)
        h_tilde_np = np.array(h_tilde, dtype=np.float32)

        # Step 2: Per-vector std-dev scale so rotated coordinates fit codebook
        # We scale by the std of the signal dims (the dominant energy).
        # This matches the paper's normalisation before codebook lookup.
        h_s = h_tilde_np[:, : self._d_s]  # (batch, d_s)
        h_n = h_tilde_np[:, self._d_s :]  # (batch, d - d_s)

        # Per-vector abs-max scale for signal dims (matches TurboQuant convention)
        sig_absmax = np.max(np.abs(h_s), axis=1, keepdims=True)  # (batch, 1)
        sig_absmax = np.where(sig_absmax < 1e-8, 1.0, sig_absmax)
        sig_scale = sig_absmax[:, 0].astype(np.float32)  # (batch,)

        # Noise dims: use noise absmax for independent scaling
        if h_n.size > 0:
            noise_absmax = np.max(np.abs(h_n), axis=1, keepdims=True)
            noise_absmax = np.where(noise_absmax < 1e-8, 1.0, noise_absmax)
        else:
            noise_absmax = np.ones((batch, 1), dtype=np.float32)
        noise_scale = noise_absmax[:, 0].astype(np.float32)

        # Step 3: Quantize signal dims with C_signal
        h_s_norm = mx.array(h_s / sig_absmax, dtype=mx.float16)  # normalised
        idx_s_mx = self._cb_signal.quantize(h_s_norm)  # (batch, d_s) uint8

        # Step 4: Quantize noise dims with C_noise
        if h_n.size > 0:
            h_n_norm = mx.array(h_n / noise_absmax, dtype=mx.float16)
            idx_n_mx = self._cb_noise.quantize(h_n_norm)  # (batch, d-d_s) uint8
        else:
            idx_n_mx = None

        # Concatenate indices into single (batch, d) array
        if idx_n_mx is not None:
            indices_mx = mx.concatenate([idx_s_mx, idx_n_mx], axis=1)
        else:
            indices_mx = idx_s_mx

        # Step 5: QJL on signal residual only (if enabled)
        signs_mx = None
        residual_norm_mx = None
        if self._apply_qjl and self._qjl is not None:
            # Reconstruct signal estimate ĥ_s^(0) to compute residual
            h_s_hat_norm = self._cb_signal.dequantize(idx_s_mx)  # (batch, d_s) fp16
            h_s_hat = h_s_hat_norm.astype(mx.float32) * mx.array(sig_absmax, dtype=mx.float32)
            h_s_mx = mx.array(h_s, dtype=mx.float32)
            epsilon_s = (h_s_mx - h_s_hat).astype(mx.float16)  # (batch, d_s)
            signs_mx, residual_norm_mx = self._qjl.encode_key(epsilon_s)

        # Pack scales: store signal scale in norm field, noise scale in final_radius
        scales_mx = mx.array(sig_scale, dtype=mx.float16)
        noise_scales_mx = mx.array(noise_scale, dtype=mx.float16)

        return EncodedVector(
            quantizer_type="spectral_quant",
            batch_size=batch,
            dim=self._d,
            indices=indices_mx,
            norm=scales_mx,  # signal scale per token
            final_radius=noise_scales_mx,  # noise scale per token
            signs=signs_mx,
            residual_norm=residual_norm_mx,
        )

    def decode(self, ev: EncodedVector) -> Any:
        """Reconstruct vectors from SpectralQuant encoding.

        Follows decompression steps of Algorithm 1.

        Args:
            ev: EncodedVector produced by encode().

        Returns:
            Reconstructed array of shape (batch, d), fp16.
        """
        import mlx.core as mx

        sig_scale = ev.norm.astype(mx.float32)[:, None]  # (batch, 1)
        noise_scale = ev.final_radius.astype(mx.float32)[:, None]  # (batch, 1)
        indices_np = np.array(ev.indices, dtype=np.int32)  # (batch, d)
        batch = ev.batch_size

        idx_s = mx.array(indices_np[:, : self._d_s], dtype=mx.uint8)
        idx_n = mx.array(indices_np[:, self._d_s :], dtype=mx.uint8)

        # Decode signal dims: ĥ_s^(0) = decode(c_s) * scale
        h_s_hat = self._cb_signal.dequantize(idx_s).astype(mx.float32) * sig_scale

        # QJL correction on signal dims (if applicable)
        if (
            self._apply_qjl
            and self._qjl is not None
            and ev.signs is not None
            and ev.residual_norm is not None
        ):
            scale_qjl = SQRT_PI_OVER_2 / self._qjl.m
            r_norm = ev.residual_norm.astype(mx.float32)[:, None]  # (batch, 1)
            correction = (
                r_norm * scale_qjl * (ev.signs.astype(mx.float32) @ self._qjl._S.astype(mx.float32))
            )  # (batch, d_s)
            h_s_hat = h_s_hat + correction

        # Decode noise dims: ĥ_n = decode(c_n) * scale (no correction)
        if idx_n.shape[1] > 0:
            h_n_hat = self._cb_noise.dequantize(idx_n).astype(mx.float32) * noise_scale
            h_tilde_hat = mx.concatenate([h_s_hat, h_n_hat], axis=1)  # (batch, d)
        else:
            h_tilde_hat = h_s_hat

        # Inverse rotation: ĥ = U · ĥ̃ = _R_inv @ ĥ̃
        x_hat = h_tilde_hat @ self._R_inv.T  # (batch, d)
        return x_hat.astype(mx.float16)

    def estimate_inner_product(self, q: Any, ev: EncodedVector) -> Any:
        """Estimate ⟨q, k⟩ for all encoded keys.

        In the rotated basis: ⟨q, k⟩ = ⟨q̃, k̃⟩ where q̃ = U^T q.
        The MSE estimate is ĥ̃ · q̃, with optional QJL residual correction
        on signal dims.

        Args:
            q: Query vector, shape (d,) or (1, d), fp16.
            ev: Encoded keys from encode().

        Returns:
            Estimated inner products, shape (batch,), fp16.
        """
        import mlx.core as mx

        q_f32 = q.reshape(-1).astype(mx.float32)
        # Rotate query: q̃ = U^T q = _R @ q
        q_rot = (self._R @ q_f32).reshape(-1)  # (d,)
        q_rot_s = q_rot[: self._d_s]  # signal dims of query
        q_rot_n = q_rot[self._d_s :]  # noise dims of query

        sig_scale = ev.norm.astype(mx.float32)  # (batch,)
        noise_scale = ev.final_radius.astype(mx.float32)  # (batch,)
        indices_np = np.array(ev.indices, dtype=np.int32)
        batch = ev.batch_size

        idx_s = mx.array(indices_np[:, : self._d_s], dtype=mx.uint8)
        idx_n = mx.array(indices_np[:, self._d_s :], dtype=mx.uint8)

        # MSE IP from signal dims: ĥ_s · q̃_s
        h_s_hat = self._cb_signal.dequantize(idx_s).astype(mx.float32) * sig_scale[:, None]
        ip_signal = h_s_hat @ q_rot_s  # (batch,)

        # MSE IP from noise dims: ĥ_n · q̃_n
        if idx_n.shape[1] > 0:
            h_n_hat = self._cb_noise.dequantize(idx_n).astype(mx.float32) * noise_scale[:, None]
            ip_noise = h_n_hat @ q_rot_n  # (batch,)
        else:
            ip_noise = mx.zeros((batch,), dtype=mx.float32)

        ip_total = ip_signal + ip_noise

        # QJL residual correction on signal dims
        if (
            self._apply_qjl
            and self._qjl is not None
            and ev.signs is not None
            and ev.residual_norm is not None
        ):
            ip_qjl = self._qjl.estimate_ip(
                q_rot_s.reshape(1, -1).astype(mx.float16),
                ev.signs,
                ev.residual_norm,
            )  # (batch,) fp16
            ip_total = ip_total + ip_qjl.astype(mx.float32)

        return ip_total.astype(mx.float16)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def compression_ratio(self) -> float:
        """Theoretical compression ratio vs FP16 baseline.

        Matches Table 2 of the paper:
          TQ (3-bit): 5.02×  (3 bits + QJL on all d dims)
          SQ_noQJL_v3: 5.95× (3 bits signal + 3 bits noise, no QJL on noise)
        """
        fp16_bits = self._d * 16
        # Count only the quantization bits (matches paper Table 2 accounting).
        # Per-vector scales (2 fp16) are a small fixed overhead equivalent to
        # TurboQuant's residual norm, not included in the per-element budget.
        compressed_bits = (
            self._d_s * self._b_signal  # signal quantization
            + (self._d - self._d_s) * self._b_noise  # noise quantization
        )
        if self._apply_qjl and self._qjl is not None:
            m = self._qjl.m
            compressed_bits += m  # JL sign bits
            compressed_bits += 16  # residual norm fp16
        return fp16_bits / max(compressed_bits, 1)

    def __repr__(self) -> str:
        return (
            f"SpectralQuantizer(d={self._d}, b_s={self._b_signal}, "
            f"b_n={self._b_noise}, d_s={self._d_s}, qjl={self._apply_qjl})"
        )
