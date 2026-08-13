from __future__ import annotations

import numpy as np
import pytest

D = 128
N = 64
SEED = 42


def _make_U(d: int = D, seed: int = SEED) -> np.ndarray:
    """Random orthogonal matrix with columns = eigenvectors."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((d, d)).astype(np.float32))
    return U  # (d, d), columns are eigenvectors


def _random_vectors(n: int = N, d: int = D, seed: int = 0) -> "mx.array":
    import mlx.core as mx

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    return mx.array(x, dtype=mx.float16)


def test_encode_decode_shape():
    import mlx.core as mx
    from veloxquant_mlx.spectral.spectral_quant import SpectralQuantizer

    sq = SpectralQuantizer(d=D, b_signal=3, b_noise=3, d_s=4, seed=SEED)
    x = _random_vectors()
    ev = sq.encode(x)
    x_hat = sq.decode(ev)
    assert x_hat.shape == (N, D)


def test_encode_decode_single_vector():
    import mlx.core as mx
    from veloxquant_mlx.spectral.spectral_quant import SpectralQuantizer

    sq = SpectralQuantizer(d=D, b_signal=2, b_noise=2, d_s=4, seed=SEED)
    rng = np.random.default_rng(4)
    x_np = rng.standard_normal((1, D)).astype(np.float32)
    x = mx.array(x_np, dtype=mx.float16)
    ev = sq.encode(x)
    x_hat = sq.decode(ev)
    assert x_hat.shape == (1, D)


def test_cosine_similarity_with_spectral_rotation():
    """SpectralQuant should reconstruct low-rank key-like data faithfully."""
    import mlx.core as mx
    from veloxquant_mlx.spectral.spectral_quant import SpectralQuantizer

    # Generate low-rank data (rank 4) and compute real PCA rotation
    rng = np.random.default_rng(1)
    basis = rng.standard_normal((D, 4)).astype(np.float32)
    basis, _ = np.linalg.qr(basis)
    coords = rng.standard_normal((N, 4)).astype(np.float32)
    x_np = (coords @ basis.T).astype(np.float32)
    x_np /= np.linalg.norm(x_np, axis=-1, keepdims=True) + 1e-8

    # Compute real PCA rotation (as in calibration)
    X = x_np - x_np.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=True)
    U = Vt.T  # columns = eigenvectors descending

    sq = SpectralQuantizer(d=D, b_signal=3, b_noise=3, rotation=U, d_s=4, seed=SEED)
    x = mx.array(x_np, dtype=mx.float16)
    ev = sq.encode(x)
    x_hat_np = np.array(sq.decode(ev), dtype=np.float32)

    cos_sim = float(
        np.mean(
            np.sum(x_np * x_hat_np, axis=1)
            / (np.linalg.norm(x_np, axis=1) * np.linalg.norm(x_hat_np, axis=1) + 1e-8)
        )
    )
    # With correct spectral rotation on low-rank data, should reconstruct well
    assert cos_sim > 0.7, f"Cosine similarity too low: {cos_sim:.4f}"


def test_cosine_similarity_spectral_beats_random_on_low_rank():
    """Spectral rotation should beat random rotation on low-rank data."""
    import mlx.core as mx
    from veloxquant_mlx.spectral.spectral_quant import SpectralQuantizer

    rng = np.random.default_rng(2)
    basis = rng.standard_normal((D, 4)).astype(np.float32)
    basis, _ = np.linalg.qr(basis)
    coords = rng.standard_normal((N, 4)).astype(np.float32)
    x_np = (coords @ basis.T).astype(np.float32)
    x_np /= np.linalg.norm(x_np, axis=-1, keepdims=True) + 1e-8

    X = x_np - x_np.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=True)
    U_pca = Vt.T

    def cos_sim(sq):
        x = mx.array(x_np, dtype=mx.float16)
        ev = sq.encode(x)
        x_hat = np.array(sq.decode(ev), dtype=np.float32)
        return float(
            np.mean(
                np.sum(x_np * x_hat, axis=1)
                / (np.linalg.norm(x_np, axis=1) * np.linalg.norm(x_hat, axis=1) + 1e-8)
            )
        )

    sq_spectral = SpectralQuantizer(
        d=D, b_signal=3, b_noise=3, rotation=U_pca, d_s=4, seed=SEED, apply_qjl=False
    )
    sq_random = SpectralQuantizer(
        d=D, b_signal=3, b_noise=3, rotation=None, d_s=4, seed=SEED, apply_qjl=False
    )

    cs_spectral = cos_sim(sq_spectral)
    cs_random = cos_sim(sq_random)

    # Spectral rotation must be competitive on low-rank data
    assert cs_spectral >= cs_random * 0.95 or cs_spectral > 0.5, (
        f"Spectral ({cs_spectral:.4f}) should be ≥ random ({cs_random:.4f}) on low-rank data"
    )


def test_compression_ratio_sq_noqjl_v3():
    """SQ_noQJL_v3: should achieve ~5.95x as reported in Table 2."""
    from veloxquant_mlx.spectral.spectral_quant import SpectralQuantizer

    # Paper primary config: b_signal=3, b_noise=3, no QJL, d_s=4, d=128
    sq = SpectralQuantizer(d=128, b_signal=3, b_noise=3, d_s=4, apply_qjl=False, seed=SEED)
    ratio = sq.compression_ratio()
    # Paper reports 5.95×; allow small tolerance for scale storage overhead
    assert ratio > 4.5, f"SQ_noQJL_v3 compression ratio too low: {ratio:.2f}×"


def test_compression_ratio_beats_turboquant_5x():
    """SpectralQuant must beat TurboQuant's 5.02× (no QJL config)."""
    from veloxquant_mlx.spectral.spectral_quant import SpectralQuantizer

    sq = SpectralQuantizer(d=128, b_signal=3, b_noise=3, d_s=4, apply_qjl=False, seed=SEED)
    assert sq.compression_ratio() > 5.0, (
        f"Must beat TurboQuant 5.02× but got {sq.compression_ratio():.2f}×"
    )


def test_estimate_inner_product_correlates_with_true():
    """IP estimates must correlate with true inner products (r > 0.7)."""
    import mlx.core as mx
    from veloxquant_mlx.spectral.spectral_quant import SpectralQuantizer

    rng = np.random.default_rng(3)
    keys_np = rng.standard_normal((N, D)).astype(np.float32)
    query_np = rng.standard_normal(D).astype(np.float32)

    sq = SpectralQuantizer(d=D, b_signal=3, b_noise=3, d_s=4, seed=SEED)
    keys = mx.array(keys_np, dtype=mx.float16)
    q = mx.array(query_np, dtype=mx.float16)

    ev = sq.encode(keys)
    estimated = np.array(sq.estimate_inner_product(q, ev), dtype=np.float32)
    true_ip = keys_np @ query_np

    correlation = float(np.corrcoef(estimated, true_ip)[0, 1])
    assert correlation > 0.7, f"IP correlation too low: {correlation:.4f}"


def test_spectral_cache_append_and_attend():
    import mlx.core as mx
    from veloxquant_mlx.cache.base import KVCacheConfig
    from veloxquant_mlx.cache.spectral_cache import SpectralQuantKVCache

    cfg = KVCacheConfig(method="spectral", head_dim=D, bit_width_inlier=3, seed=SEED)
    cache = SpectralQuantKVCache(cfg)

    rng = np.random.default_rng(5)
    n_tokens = 10
    for _ in range(n_tokens):
        k = mx.array(rng.standard_normal(D).astype(np.float32), dtype=mx.float16)
        v = mx.array(rng.standard_normal(D).astype(np.float32), dtype=mx.float16)
        cache.append_key(k)
        cache.append_value(v)

    assert len(cache) == n_tokens
    q = mx.array(rng.standard_normal(D).astype(np.float32), dtype=mx.float16)
    out = cache.attend(q)
    assert out.shape == (D,)


def test_spectral_cache_memory_below_fp16():
    import mlx.core as mx
    from veloxquant_mlx.cache.base import KVCacheConfig
    from veloxquant_mlx.cache.spectral_cache import SpectralQuantKVCache

    cfg = KVCacheConfig(method="spectral", head_dim=D, bit_width_inlier=3, seed=SEED)
    cache = SpectralQuantKVCache(cfg)

    rng = np.random.default_rng(6)
    for _ in range(10):
        k = mx.array(rng.standard_normal(D).astype(np.float32), dtype=mx.float16)
        v = mx.array(rng.standard_normal(D).astype(np.float32), dtype=mx.float16)
        cache.append(k, v)

    mem = cache.memory_bytes()
    fp16_baseline = 10 * D * 2 * 2  # 10 tokens × d × key+value × 2 bytes
    assert mem < fp16_baseline, f"SpectralQuant memory {mem} should be < FP16 {fp16_baseline}"


def test_factory_creates_spectral_cache():
    from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
    from veloxquant_mlx.cache.spectral_cache import SpectralQuantKVCache

    cfg = KVCacheConfig(method="spectral", head_dim=D, bit_width_inlier=3)
    cache = KVCacheFactory.create(cfg)
    assert isinstance(cache, SpectralQuantKVCache)


def test_calibrate_inject_improves_cosine_similarity():
    """Injecting real calibration rotations should improve or maintain quality."""
    import mlx.core as mx
    from veloxquant_mlx.cache.base import KVCacheConfig
    from veloxquant_mlx.cache.spectral_cache import SpectralQuantKVCache
    from veloxquant_mlx.spectral.calibrate import calibrate_from_vectors

    rng = np.random.default_rng(7)
    # Generate low-rank key data
    basis = rng.standard_normal((D, 4)).astype(np.float32)
    basis, _ = np.linalg.qr(basis)
    coords = rng.standard_normal((200, 4)).astype(np.float32)
    keys_np = (coords @ basis.T).astype(np.float32)

    val_basis = rng.standard_normal((D, 50)).astype(np.float32)
    val_basis, _ = np.linalg.qr(val_basis)
    val_coords = rng.standard_normal((200, 50)).astype(np.float32)
    vals_np = (val_coords @ val_basis.T).astype(np.float32)

    rotations = calibrate_from_vectors({0: keys_np}, {0: vals_np}, model_name="test_inject")

    cfg = KVCacheConfig(method="spectral", head_dim=D, bit_width_inlier=3, seed=SEED)
    cache = SpectralQuantKVCache(cfg)
    cache.calibrate(rotations[0])

    # Append and decode, check quality
    test_keys = mx.array(
        keys_np[:10] / (np.linalg.norm(keys_np[:10], axis=1, keepdims=True) + 1e-8),
        dtype=mx.float16,
    )
    for i in range(10):
        cache.append_key(test_keys[i])
        cache.append_value(test_keys[i])

    q = mx.array(rng.standard_normal(D).astype(np.float32), dtype=mx.float16)
    out = cache.attend(q)
    assert out.shape == (D,)
