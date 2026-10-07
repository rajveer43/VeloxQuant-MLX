"""Tests for CommVQQuantizer — RoPE-commutative additive codebook VQ."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.comm_vq import CommVQQuantizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_quantizer(d: int = 64, b: int = 4, n_cb: int = 4) -> CommVQQuantizer:
    q = CommVQQuantizer(d=d, b=b, n_codebooks=n_cb, seed=42)
    rng = np.random.default_rng(0)
    keys = rng.standard_normal((512, d)).astype(np.float16)
    q.fit(mx.array(keys))
    return q


def _apply_rope_mlx_ref(
    x: np.ndarray, positions: np.ndarray, d: int, base: float = 10000.0
) -> np.ndarray:
    """Reference RoPE in NumPy."""
    half = d // 2
    inv_freq = 1.0 / (base ** (np.arange(half, dtype=np.float32) / half))
    x1, x2 = x[:, :half].astype(np.float32), x[:, half:].astype(np.float32)
    angles = positions[:, None].astype(np.float32) * inv_freq[None, :]
    cos_v = np.cos(angles)
    sin_v = np.sin(angles)
    out = np.concatenate([x1 * cos_v - x2 * sin_v, x1 * sin_v + x2 * cos_v], axis=1)
    return out.astype(np.float16)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip() -> None:
    """Encode pre-RoPE keys, decode gives post-RoPE reconstruction.

    We compare decoded output against the ground-truth post-RoPE keys.
    """
    d, b, n_cb = 64, 4, 4
    q = _make_quantizer(d=d, b=b, n_cb=n_cb)

    rng = np.random.default_rng(1)
    N = 128
    keys_np = rng.standard_normal((N, d)).astype(np.float16)
    pos_np = np.arange(N, dtype=np.int32)
    # Ground-truth post-RoPE keys
    keys_post_np = _apply_rope_mlx_ref(keys_np, pos_np, d)

    keys_mx = mx.array(keys_np)
    pos_mx = mx.array(pos_np)
    keys_post = mx.array(keys_post_np)

    ev = q.encode(keys_mx, positions=pos_mx)
    keys_hat = q.decode(ev)
    mx.eval(keys_hat)

    # Compare decoded (post-RoPE) against ground-truth post-RoPE
    mse = float(mx.mean(mx.sum((keys_post - keys_hat) ** 2, axis=-1)).item())

    # b=4, n_cb=4: 16 centroids for 16-dim sub-vectors.  Gaussian unit variance
    # gives typical per-dim MSE ~1; total MSE ~ 64.  Allow 2× slack for the
    # small (512-sample) training set and fp16 quantisation noise.
    assert mse < 130.0, f"MSE too high: {mse:.4f}"


def test_rope_commutativity() -> None:
    """CommVQ pre-RoPE encoding: decode(encode(x_pre)) + RoPE ≈ decode(encode(rotate(x_pre))).

    The commutativity guarantee is approximate (not exact due to quantization),
    but the error should be small relative to the reconstruction error.
    """
    d, b, n_cb = 64, 6, 4
    q = CommVQQuantizer(d=d, b=b, n_codebooks=n_cb, seed=7)
    rng = np.random.default_rng(2)
    keys = rng.standard_normal((1024, d)).astype(np.float16)
    q.fit(mx.array(keys))

    N = 64
    x_pre = rng.standard_normal((N, d)).astype(np.float16)
    pos_np = np.arange(N, dtype=np.int32)
    x_post = _apply_rope_mlx_ref(x_pre, pos_np, d)

    # Path A: encode pre-RoPE x → decode with RoPE
    ev_a = q.encode(mx.array(x_pre), positions=mx.array(pos_np))
    hat_a = q.decode(ev_a)
    mx.eval(hat_a)

    # Path B: encode post-RoPE x → decode with position=0 (no extra RoPE)
    ev_b = q.encode(mx.array(x_post), positions=mx.zeros((N,), dtype=mx.int32))
    hat_b = q.decode(ev_b)
    mx.eval(hat_b)

    # The two paths reconstruct from different representations; what we check
    # is that both reconstructions are close to the ground truth post-RoPE key.
    x_post_mx = mx.array(x_post)
    mse_a = float(mx.mean(mx.sum((x_post_mx - hat_a) ** 2, axis=-1)).item())
    mse_b = float(mx.mean(mx.sum((x_post_mx - hat_b) ** 2, axis=-1)).item())

    # Both paths should reconstruct the post-RoPE key with comparable MSE.
    # Generous thresholds — we're checking commutativity structure, not distortion.
    assert mse_a < 80.0, f"Path A MSE too high: {mse_a:.4f}"
    assert mse_b < 80.0, f"Path B MSE too high: {mse_b:.4f}"
    # The ratio between paths should not be extreme (commutativity check)
    ratio = max(mse_a, mse_b) / (min(mse_a, mse_b) + 1e-6)
    assert ratio < 8.0, (
        f"Commutativity mismatch: MSE_A={mse_a:.4f}, MSE_B={mse_b:.4f}, ratio={ratio:.2f}"
    )


def test_encode_decode_shapes() -> None:
    """Check that encode/decode produce correctly shaped outputs."""
    d, b, n_cb = 128, 4, 4
    q = _make_quantizer(d=d, b=b, n_cb=n_cb)

    N = 32
    keys = mx.array(np.random.randn(N, d).astype(np.float16))
    ev = q.encode(keys)

    assert ev.indices.shape == (N, n_cb), f"indices shape mismatch: {ev.indices.shape}"
    assert ev.indices.dtype == mx.uint8, f"indices dtype mismatch: {ev.indices.dtype}"

    decoded = q.decode(ev)
    mx.eval(decoded)
    assert decoded.shape == (N, d), f"decoded shape mismatch: {decoded.shape}"
    assert decoded.dtype == mx.float16, f"decoded dtype mismatch: {decoded.dtype}"


def test_compression_ratio() -> None:
    """Verify memory footprint vs fp16 baseline."""
    d, b, n_cb = 128, 8, 4
    q = CommVQQuantizer(d=d, b=b, n_codebooks=n_cb, seed=0)

    fp16_bytes = d * 2
    index_bytes = n_cb * 1  # n_cb uint8 indices
    ratio = fp16_bytes / index_bytes

    assert q.compression_ratio == ratio, f"Expected {ratio}x, got {q.compression_ratio}x"
    # With d=128, n_cb=4 → 128*2 / 4 = 64× compression
    assert q.compression_ratio == 64.0


def test_inner_product_shape() -> None:
    """estimate_inner_product returns shape [N]."""
    d, b, n_cb = 64, 4, 4
    q = _make_quantizer(d=d, b=b, n_cb=n_cb)

    N = 16
    keys = mx.array(np.random.randn(N, d).astype(np.float16))
    query = mx.array(np.random.randn(d).astype(np.float16))
    ev = q.encode(keys)
    ips = q.estimate_inner_product(query, ev)
    mx.eval(ips)

    assert ips.shape == (N,), f"IP shape mismatch: {ips.shape}"


def test_fit_required_before_encode() -> None:
    """encode() before fit() should raise RuntimeError."""
    q = CommVQQuantizer(d=64, b=4, n_codebooks=4)
    keys = mx.array(np.random.randn(10, 64).astype(np.float16))
    with pytest.raises(RuntimeError, match="not been trained"):
        q.encode(keys)


def test_b_over_8_rejected() -> None:
    """b > 8 implies cb_size > 256, which overflows the uint8 indices used
    by encode()/decode() (issue #621); must fail fast at construction."""
    with pytest.raises(ValueError, match="overflows the uint8 indices"):
        CommVQQuantizer(d=16, b=9, n_codebooks=4)


def test_b_equal_8_allowed() -> None:
    """b=8 -> cb_size=256, the max value representable by uint8 indices."""
    CommVQQuantizer(d=16, b=8, n_codebooks=4)


def test_trained_flag() -> None:
    q = CommVQQuantizer(d=64, b=4, n_codebooks=4)
    assert not q.trained
    q.fit(mx.array(np.random.randn(256, 64).astype(np.float16)))
    assert q.trained


@pytest.mark.parametrize("d,n_cb", [(64, 4), (128, 4), (128, 8)])
def test_various_configs(d: int, n_cb: int) -> None:
    """Smoke test across d/n_cb combinations."""
    q = CommVQQuantizer(d=d, b=4, n_codebooks=n_cb, seed=0)
    data = mx.array(np.random.randn(256, d).astype(np.float16))
    q.fit(data)

    x = mx.array(np.random.randn(8, d).astype(np.float16))
    ev = q.encode(x)
    out = q.decode(ev)
    mx.eval(out)
    assert out.shape == (8, d)


# ---------------------------------------------------------------------------
# #754 — encode() is chunked over rows
# ---------------------------------------------------------------------------


def _install_random_codebooks(q: CommVQQuantizer, seed: int = 0) -> CommVQQuantizer:
    """Skip EM training: a trained-looking random codebook is enough for encode()."""
    rng = np.random.default_rng(seed)
    q._codebooks = rng.standard_normal((q._n_cb, q._cb_size, q._sub_dim)).astype(np.float32)
    q._codebooks_mx = mx.array(q._codebooks).astype(mx.float16)
    q._trained = True
    return q


def test_encode_chunking_does_not_change_indices(monkeypatch) -> None:
    import veloxquant_mlx.quantizers.comm_vq as cv

    q = _install_random_codebooks(CommVQQuantizer(d=64, b=6, n_codebooks=4))
    x = mx.array(np.random.default_rng(1).standard_normal((300, 64)).astype(np.float16))
    reference = q._encode_batch(x)

    monkeypatch.setattr(cv, "_ENCODE_CHUNK_BYTES", 1)  # forces 1-row chunks
    chunked = q._encode_batch(x)
    assert chunked.shape == (300, 4)
    assert bool(mx.array_equal(reference, chunked).item())


def test_encode_empty_input() -> None:
    q = _install_random_codebooks(CommVQQuantizer(d=64, b=6, n_codebooks=4))
    out = q._encode_batch(mx.zeros((0, 64), dtype=mx.float16))
    assert out.shape == (0, 4) and out.dtype == mx.uint8


def test_encode_peak_memory_is_bounded() -> None:
    """The unchunked [N, K, sub_dim] difference tensor needed ~5.4 GB for these
    20k rows (~16 GB at 60k); chunked it stays a few hundred MB."""
    q = _install_random_codebooks(CommVQQuantizer(d=128, b=8, n_codebooks=4))
    x = mx.random.normal((20000, 128)).astype(mx.float16)
    mx.eval(x)
    mx.clear_cache()
    mx.reset_peak_memory()
    base = mx.get_peak_memory()
    mx.eval(q._encode_batch(x))
    assert (mx.get_peak_memory() - base) < 1.5e9


# ---------------------------------------------------------------------------
# #755 — fit() with fewer calibration vectors than codebook entries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [1, 10, 100, 255])
def test_fit_with_fewer_samples_than_codebook_size_warns_and_works(n_samples: int) -> None:
    q = CommVQQuantizer(d=64, b=8, n_codebooks=4, n_em_iters=3)
    keys = np.random.default_rng(0).standard_normal((n_samples, 64)).astype(np.float32)
    with pytest.warns(UserWarning, match="calibration vectors"):
        q.fit(keys)
    assert q.trained
    ev = q.encode(mx.array(keys).astype(mx.float16))
    assert ev.indices.shape == (n_samples, 4)
    assert q.decode(ev).shape == (n_samples, 64)


def test_fit_with_enough_samples_does_not_warn() -> None:
    import warnings

    q = CommVQQuantizer(d=64, b=4, n_codebooks=4, n_em_iters=3)
    keys = np.random.default_rng(0).standard_normal((16, 64)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        q.fit(keys)  # exactly 2**b vectors: still the unchanged, warning-free path


def test_fit_with_no_samples_raises_clear_error() -> None:
    from veloxquant_mlx.core.exceptions import QuantizerConfigError

    q = CommVQQuantizer(d=64, b=4, n_codebooks=4)
    with pytest.raises(QuantizerConfigError, match="no calibration vectors"):
        q.fit(np.zeros((0, 64), dtype=np.float32))
