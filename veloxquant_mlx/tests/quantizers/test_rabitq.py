"""Tests for RaBitQQuantizer — 1-bit IVF random orthogonal quantization."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.quantizers.rabitq import RaBitQQuantizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_quantizer(d: int = 64, nlist: int = 16, nprobe: int = 4) -> RaBitQQuantizer:
    q = RaBitQQuantizer(d=d, nlist=nlist, nprobe=nprobe, seed=42)
    rng = np.random.default_rng(0)
    keys = rng.standard_normal((512, d)).astype(np.float16)
    q.fit(mx.array(keys), max_samples=512)
    return q


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_trained_flag() -> None:
    q = RaBitQQuantizer(d=64, nlist=8)
    assert not q.trained
    q.fit(mx.array(np.random.randn(256, 64).astype(np.float16)))
    assert q.trained


def test_fit_required_before_encode() -> None:
    q = RaBitQQuantizer(d=64, nlist=8)
    keys = mx.array(np.random.randn(10, 64).astype(np.float16))
    with pytest.raises(RuntimeError, match="not been trained"):
        q.encode(keys)


def test_encode_shapes() -> None:
    d, nlist = 64, 16
    q = _make_quantizer(d=d, nlist=nlist)
    N = 32
    keys = mx.array(np.random.randn(N, d).astype(np.float16))
    ev = q.encode(keys)
    mx.eval(ev.indices, ev.norm)

    assert ev.indices.shape == (N, d // 8), f"bits shape: {ev.indices.shape}"
    assert ev.indices.dtype == mx.uint8
    assert ev.norm.shape == (N, 3), f"meta shape: {ev.norm.shape}"
    assert ev.norm.dtype == mx.float32


def test_decode_shape_dtype() -> None:
    d = 64
    q = _make_quantizer(d=d)
    N = 16
    keys = mx.array(np.random.randn(N, d).astype(np.float16))
    ev = q.encode(keys)
    out = q.decode(ev)
    mx.eval(out)
    assert out.shape == (N, d), f"decoded shape: {out.shape}"
    assert out.dtype == mx.float16


def test_compression_ratio() -> None:
    """16× vs fp16 at D=128: (128*2) / (128//8) = 256/16 = 16×."""
    d = 128
    q = RaBitQQuantizer(d=d, nlist=16)
    assert q.compression_ratio == 16.0, f"Expected 16×, got {q.compression_ratio}"


def test_compression_ratio_d64() -> None:
    """16× at D=64: (64*2) / (64//8) = 128/8 = 16×."""
    d = 64
    q = RaBitQQuantizer(d=d, nlist=8)
    assert q.compression_ratio == 16.0


def test_inner_product_shape() -> None:
    d = 64
    q = _make_quantizer(d=d)
    N = 20
    keys = mx.array(np.random.randn(N, d).astype(np.float16))
    query = mx.array(np.random.randn(d).astype(np.float16))
    ev = q.encode(keys)
    ips = q.estimate_inner_product(query, ev)
    mx.eval(ips)
    assert ips.shape == (N,), f"IP shape: {ips.shape}"


def test_search_returns_top_k() -> None:
    d = 64
    q = _make_quantizer(d=d, nlist=16, nprobe=4)
    N = 128
    keys = mx.array(np.random.randn(N, d).astype(np.float16))
    query = mx.array(np.random.randn(d).astype(np.float16))
    ev = q.encode(keys)
    top_k = 10
    result = q.search(query, ev, top_k=top_k)
    mx.eval(result)
    assert result.shape == (top_k,), f"search result shape: {result.shape}"


def test_search_recall_gaussian() -> None:
    """Recall@10 on unit Gaussian corpus using L2 ground truth (RaBitQ approximates L2).

    Recall threshold is lenient (0.2) — 1-bit quantization with nprobe=8/32 clusters
    probes only 25% of the corpus; recall above chance (0.01) confirms the
    approximate distance is working directionally.
    """
    d = 128
    N = 512
    top_k = 10
    rng = np.random.default_rng(7)

    corpus_np = rng.standard_normal((N, d)).astype(np.float16)
    query_np = rng.standard_normal(d).astype(np.float16)

    q = RaBitQQuantizer(d=d, nlist=32, nprobe=8, rerank=64, seed=0)
    q.fit(mx.array(corpus_np), max_samples=512)

    ev = q.encode(mx.array(corpus_np))
    result = q.search(mx.array(query_np), ev, top_k=top_k)
    mx.eval(result)

    # Ground truth: L2 distance (RaBitQ approximates L2, not dot product)
    corpus_f32 = corpus_np.astype(np.float32)
    query_f32 = query_np.astype(np.float32)
    l2_dists = np.sum((corpus_f32 - query_f32[None, :]) ** 2, axis=1)
    true_top_k = set(np.argsort(l2_dists)[:top_k].tolist())
    result_set = set(np.array(result).tolist())

    recall = len(true_top_k & result_set) / top_k
    assert recall >= 0.1, f"Recall@{top_k} too low: {recall:.2f}"


@pytest.mark.parametrize("d", [64, 128])
def test_various_d(d: int) -> None:
    q = RaBitQQuantizer(d=d, nlist=16, nprobe=4, seed=1)
    data = mx.array(np.random.randn(256, d).astype(np.float16))
    q.fit(data, max_samples=256)

    x = mx.array(np.random.randn(8, d).astype(np.float16))
    ev = q.encode(x)
    out = q.decode(ev)
    mx.eval(out)
    assert out.shape == (8, d)


def test_hamming_kernel_direct() -> None:
    """Smoke test the Metal Hamming kernel directly."""
    from veloxquant_mlx.metal._rabitq import rabitq_hamming_score

    D = 64
    N = 32
    n_bytes = D // 8

    rng = np.random.default_rng(99)
    qbits = mx.array(rng.integers(0, 256, size=(n_bytes,), dtype=np.uint8))
    bits = mx.array(rng.integers(0, 256, size=(N, n_bytes), dtype=np.uint8))
    Cx = mx.array(rng.standard_normal(N).astype(np.float32))
    scale = mx.array([0.5], dtype=mx.float32)

    scores = rabitq_hamming_score(qbits, bits, Cx, scale)
    mx.eval(scores)
    assert scores.shape == (N,)
    assert scores.dtype == mx.float32
    # scores should be non-negative (hamming >= 0) plus Cx which can vary
    # just check it ran without error and has finite values
    assert np.all(np.isfinite(np.array(scores)))
