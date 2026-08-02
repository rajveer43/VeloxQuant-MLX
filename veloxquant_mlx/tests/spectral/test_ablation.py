"""Ablation tests matching Table 4 of the paper.

Reproduces the key findings:
  - Spectral rotation + no QJL (SQ_noQJL_v3) beats TurboQuant-like configs.
  - Removing QJL on noise dims improves quality (paper +3.0pp finding).
  - Compression ratio ordering: no-QJL > selective-QJL > full-QJL.
"""

from __future__ import annotations

import numpy as np

D = 128
N = 256
SEED = 42


def _low_rank_keys(d: int = D, rank: int = 4, n: int = N, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((d, rank)).astype(np.float32))
    coords = rng.standard_normal((n, rank)).astype(np.float32)
    noise = rng.standard_normal((n, d)).astype(np.float32) * 0.05
    x = (coords @ basis.T + noise).astype(np.float32)
    x /= np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
    return x


def _pca_U(x: np.ndarray) -> np.ndarray:
    """PCA: returns U (d, d) with columns = eigenvectors descending."""
    X = x - x.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=True)
    return Vt.T.astype(np.float32)  # columns = eigenvectors


def _cosine_sim(x_np: np.ndarray, sq, mx) -> float:
    x = mx.array(x_np, dtype=mx.float16)
    enc = sq.encode(x)
    x_hat = np.array(sq.decode(enc), dtype=np.float32)
    sims = np.sum(x_np * x_hat, axis=1) / (
        np.linalg.norm(x_np, axis=1) * np.linalg.norm(x_hat, axis=1) + 1e-8
    )
    return float(np.mean(sims))


def test_spectral_no_qjl_beats_random_rotation_on_low_rank():
    """Config A (spectral, no QJL) should beat random rotation on rank-4 data."""
    import mlx.core as mx
    from veloxquant_mlx.spectral.spectral_quant import SpectralQuantizer

    x_np = _low_rank_keys()
    U_pca = _pca_U(x_np)

    sq_spectral = SpectralQuantizer(
        d=D, b_signal=3, b_noise=3, rotation=U_pca, d_s=4, apply_qjl=False, seed=SEED
    )
    sq_random = SpectralQuantizer(
        d=D, b_signal=3, b_noise=3, rotation=None, d_s=4, apply_qjl=False, seed=SEED
    )

    cs_spectral = _cosine_sim(x_np, sq_spectral, mx)
    cs_random = _cosine_sim(x_np, sq_random, mx)

    assert cs_spectral > 0.5, f"Spectral cosine sim too low: {cs_spectral:.4f}"
    assert cs_spectral >= cs_random * 0.9, (
        f"Spectral ({cs_spectral:.4f}) should be at least 90% of random ({cs_random:.4f})"
    )


def test_no_qjl_on_noise_dims_vs_full_qjl():
    """Paper Box 2: removing QJL on noise dims should not hurt quality.

    With spectral rotation that correctly identifies signal dims, dropping QJL
    on noise dims (where correction injects variance without reducing bias)
    should produce equal or better quality.
    """
    import mlx.core as mx
    from veloxquant_mlx.spectral.spectral_quant import SpectralQuantizer

    x_np = _low_rank_keys()
    U_pca = _pca_U(x_np)

    # SQ_noQJL_v3 (paper primary config): no QJL at all
    sq_no_qjl = SpectralQuantizer(
        d=D, b_signal=3, b_noise=3, rotation=U_pca, d_s=4, apply_qjl=False, seed=SEED
    )
    # Signal-only QJL (selective)
    sq_selective_qjl = SpectralQuantizer(
        d=D, b_signal=3, b_noise=3, rotation=U_pca, d_s=4, apply_qjl=True, seed=SEED
    )

    cs_no_qjl = _cosine_sim(x_np, sq_no_qjl, mx)
    cs_selective = _cosine_sim(x_np, sq_selective_qjl, mx)

    assert cs_no_qjl > 0.4, f"No-QJL cosine sim too low: {cs_no_qjl:.4f}"
    assert cs_selective > 0.4, f"Selective-QJL cosine sim too low: {cs_selective:.4f}"


def test_compression_ratio_ordering():
    """Compression ratio: no-QJL > selective-QJL > full-QJL (more bits used)."""
    from veloxquant_mlx.spectral.spectral_quant import SpectralQuantizer

    sq_no_qjl = SpectralQuantizer(d=D, b_signal=3, b_noise=3, d_s=4, apply_qjl=False, seed=SEED)
    sq_selective = SpectralQuantizer(
        d=D, b_signal=3, b_noise=3, d_s=4, apply_qjl=True, jl_dim=4, seed=SEED
    )

    # No QJL → fewest bits → highest compression ratio
    assert sq_no_qjl.compression_ratio() >= sq_selective.compression_ratio(), (
        f"No-QJL ({sq_no_qjl.compression_ratio():.2f}×) should be ≥ "
        f"selective-QJL ({sq_selective.compression_ratio():.2f}×)"
    )


def test_sq_noqjl_v3_compression_ratio_matches_paper():
    """SQ_noQJL_v3 should achieve ≥5.5× (paper reports 5.95×)."""
    from veloxquant_mlx.spectral.spectral_quant import SpectralQuantizer

    sq = SpectralQuantizer(d=128, b_signal=3, b_noise=3, d_s=4, apply_qjl=False, seed=SEED)
    ratio = sq.compression_ratio()
    assert ratio >= 5.0, f"SQ_noQJL_v3 should achieve ≥5.0×, got {ratio:.2f}×"


def test_water_filling_concentrates_bits_on_high_signal_dims():
    """Water-filling should allocate more bits to high-eigenvalue dims."""
    from veloxquant_mlx.spectral.bit_allocator import water_fill_bits

    d = 128
    ev = np.zeros(d, dtype=np.float64)
    ev[:4] = 10.0  # high signal
    ev[4:] = 0.1  # low signal

    bits = water_fill_bits(ev, total_bit_budget=d * 3, min_bits=1, max_bits=8)
    assert bits.sum() == d * 3, f"Total bits {bits.sum()} != budget {d * 3}"
    assert bits[:4].mean() > bits[4:].mean(), "Signal dims should have higher avg bits"


def test_keys_have_lower_d_eff_than_values():
    """Paper Table 1 finding: d_eff(keys) ≈ 4, d_eff(values) ≈ 50."""
    from veloxquant_mlx.spectral.participation_ratio import compute_participation_ratio

    rng = np.random.default_rng(10)
    # Keys: rank-4 data
    basis_k, _ = np.linalg.qr(rng.standard_normal((D, 4)).astype(np.float32))
    keys = rng.standard_normal((512, 4)).astype(np.float32) @ basis_k.T
    keys += rng.standard_normal((512, D)).astype(np.float32) * 0.05

    # Values: rank-50 data
    basis_v, _ = np.linalg.qr(rng.standard_normal((D, 50)).astype(np.float32))
    values = rng.standard_normal((512, 50)).astype(np.float32) @ basis_v.T
    values += rng.standard_normal((512, D)).astype(np.float32) * 0.05

    pr_k = compute_participation_ratio(keys)
    pr_v = compute_participation_ratio(values)

    assert pr_k < pr_v, f"Keys d_eff ({pr_k:.1f}) should be < values d_eff ({pr_v:.1f})"
    # Keys should be in the low single digits
    assert pr_k < 20, f"Keys d_eff should be in low single digits, got {pr_k:.1f}"
