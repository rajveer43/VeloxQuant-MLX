"""Parity tests for the fused KIVI group-quantization Metal kernels (#164).

There are **two** kernels, because KIVI's two schemes have opposite memory
layouts in a ``[..., S, D]`` row-contiguous cache: per-token values group along
the contiguous axis (SIMD group per group, butterfly reduction), per-channel
keys group along the strided axis (one thread owns a whole group, no reduction).
Both are exercised here through the same public entry point.

They must reproduce ``KIVIKVCache._quant_dequant_along`` **bit-for-bit**, not
merely within tolerance: KIVI is the repo's reference baseline, so a kernel that
shifted results by an ULP would silently move every published KIVI number and
make benchmarks depend on whether Metal happened to be enabled.

Three details make bit-exactness non-trivial, and each has a test below:
  * edge-replication padding for a ragged final group,
  * round-half-to-even (``rint``, not ``metal::round``),
  * an un-fused ``q*scale + gmin`` (an ``fma`` contraction changes ~0.02% of
    elements by 1 ULP).

Also guards the shape-specialization regression: the sequence length must stay a
runtime value, or the dispatch cache compiles one shader per sequence length and
decode grinds to a halt.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
import pytest

from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.cache.kivi_cache import KIVIKVCache
from veloxquant_mlx.metal import _kivi_quant, metal_available
from veloxquant_mlx.metal.kernels import kivi_group_quant_dequant

pytestmark = pytest.mark.skipif(
    not metal_available(),
    reason="Metal compute kernels not available on this build of mlx.",
)


def _cache(b: int = 2, g: int = 32) -> KIVIKVCache:
    """A KIVI cache used purely as the reference implementation."""

    class _Cfg:
        head_dim = 128
        bit_width_inlier = b
        kivi_group_size = g
        residual_length = 32
        use_metal_kernels = False  # reference must be the pure-MLX path

    return KIVIKVCache(_Cfg())


def _assert_bit_exact(x: mx.array, axis: int, b: int = 2, g: int = 32) -> None:
    ref = _cache(b, g)._quant_dequant_along(x, axis=axis)
    got = kivi_group_quant_dequant(x, axis=axis, group_size=g, levels=(1 << b) - 1, eps=1e-8)
    mx.eval(ref, got)
    assert bool(mx.all(ref == got).item()), (
        f"kernel != MLX for axis={axis} b={b} g={g} shape={x.shape}; "
        f"max|diff|={float(mx.max(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))).item())}"
    )


# ---------------------------------------------------------------------------
# Bit-exact parity across the configuration space
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [-2, -1])
@pytest.mark.parametrize("b", [1, 2, 4, 8])
@pytest.mark.parametrize("g", [4, 32, 128])
def test_bit_exact_across_bits_and_group_sizes(axis, b, g):
    """b=2/g=32 is KIVI's default; the rest guard the general path."""
    rng = np.random.default_rng(b * 100 + g)
    x = mx.array(rng.standard_normal((1, 4, 64, 128)).astype(np.float16))
    _assert_bit_exact(x, axis, b, g)


@pytest.mark.parametrize("axis", [-2, -1])
@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
def test_bit_exact_across_dtypes(axis, dtype):
    rng = np.random.default_rng(5)
    x = mx.array(rng.standard_normal((1, 4, 96, 128))).astype(dtype)
    _assert_bit_exact(x, axis)


@pytest.mark.parametrize("axis", [-2, -1])
@pytest.mark.parametrize(
    "shape", [(1, 8, 32, 128), (2, 2, 33, 64), (1, 1, 7, 16), (3, 1, 200, 128)]
)
def test_bit_exact_across_shapes(axis, shape):
    """Includes ragged lengths (33, 7, 200) that force a partially-padded
    final group — the edge-replication path."""
    rng = np.random.default_rng(hash(shape) % 2**31)
    x = mx.array(rng.standard_normal(shape).astype(np.float16))
    _assert_bit_exact(x, axis)


# ---------------------------------------------------------------------------
# The three bit-exactness hazards, pinned individually
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [-2, -1])
def test_ragged_group_uses_edge_replication_padding(axis):
    """L=50 with g=32 leaves an 18-element final group. The MLX path pads it
    with copies of the row's LAST element; zero-padding instead would change
    that group's min/max and shift every reconstruction inside it."""
    rng = np.random.default_rng(11)
    # Offset well away from zero so a zero-pad bug would visibly move min/max.
    x = mx.array((rng.standard_normal((1, 2, 50, 96)) + 8.0).astype(np.float16))
    _assert_bit_exact(x, axis)


@pytest.mark.parametrize("axis", [-2, -1])
def test_exact_half_codes_round_half_to_even(axis):
    """Integer-valued data makes (v-gmin)/scale land on exact .5 boundaries,
    where round-half-to-even (mx.round, rint) and round-half-away-from-zero
    (metal::round) disagree."""
    x = mx.array(np.tile(np.array([0, 1, 2, 3], np.float16), (1, 2, 64, 24)))
    _assert_bit_exact(x, axis)


@pytest.mark.parametrize("axis", [-2, -1])
def test_no_fma_contraction_in_reconstruction(axis):
    """At b=8 the codes are large enough that a fused q*scale+gmin differs
    from the separately-rounded form on a measurable fraction of elements.
    Pure-noise fp16 input over many elements is the case that caught it."""
    rng = np.random.default_rng(7)
    x = mx.array(rng.standard_normal((1, 4, 128, 128)).astype(np.float16))
    _assert_bit_exact(x, axis, b=8, g=32)


@pytest.mark.parametrize("axis", [-2, -1])
def test_degenerate_group_uses_eps_floor(axis):
    """min == max makes scale collapse; it must be floored at eps, not 0
    (which would produce inf/nan)."""
    x = mx.full((1, 2, 64, 32), 3.5, dtype=mx.float16)
    got = kivi_group_quant_dequant(x, axis=axis, group_size=32, levels=3, eps=1e-8)
    mx.eval(got)
    assert bool(mx.all(mx.isfinite(got)).item())
    _assert_bit_exact(x, axis)


def test_single_element_groups_are_lossless():
    """g=1 puts one element per group, so min==max==the value and the
    round-trip is exact — the degenerate case that motivated #162's
    group-aligned flushing."""
    rng = np.random.default_rng(3)
    x = mx.array(rng.standard_normal((1, 2, 16, 32)).astype(np.float16))
    got = kivi_group_quant_dequant(x, axis=-2, group_size=1, levels=3, eps=1e-8)
    mx.eval(got)
    assert bool(mx.all(got == x).item())


# ---------------------------------------------------------------------------
# Shape-specialization regression guard
# ---------------------------------------------------------------------------


def test_dispatch_cache_does_not_grow_with_sequence_length():
    """Regression: the sequence length must be a runtime value, not a #define.

    When the array shape was baked into the kernel header, every distinct
    sequence length compiled a new shader — so a decode loop compiled one shader
    per token and effectively hung. One config must map to exactly one variant
    per axis, no matter how many sequence lengths pass through it.

    ``head_dim`` *is* specialized, deliberately: it is fixed for the lifetime of
    a cache, so it contributes a constant, not growth.
    """
    saved = dict(_kivi_quant._cache)
    _kivi_quant._cache.clear()
    try:
        rng = np.random.default_rng(0)
        for L in range(16, 400, 7):  # 55 distinct sequence lengths
            x = mx.array(rng.standard_normal((1, 4, L, 128)).astype(np.float16))
            for axis in (-2, -1):
                mx.eval(kivi_group_quant_dequant(x, axis=axis, group_size=32, levels=3))
        assert len(_kivi_quant._cache) == 2, (
            f"expected 2 compiled variants (one per axis) for 55 sequence lengths, "
            f"got {len(_kivi_quant._cache)}: {list(_kivi_quant._cache)}"
        )
    finally:
        _kivi_quant._cache.clear()
        _kivi_quant._cache.update(saved)


# ---------------------------------------------------------------------------
# Cache-level integration: enabling the kernel must change nothing numerically
# ---------------------------------------------------------------------------


def _run_cache(method: str, use_metal: bool, seed: int = 3, S: int = 200, steps: int = 40):
    cfg = dict(
        method=method,
        head_dim=128,
        bit_width_inlier=2,
        kivi_group_size=32,
        residual_length=32,
        use_metal_kernels=use_metal,
    )
    if method == "kivi_sink":
        cfg["n_sink_tokens"] = 5
    cache = KVCacheFactory.create(KVCacheConfig(**cfg))
    rng = np.random.default_rng(seed)
    ko, vo = cache.update_and_fetch(
        mx.array(rng.standard_normal((1, 8, S, 128)).astype(np.float16)),
        mx.array(rng.standard_normal((1, 8, S, 128)).astype(np.float16)),
    )
    for _ in range(steps):
        ko, vo = cache.update_and_fetch(
            mx.array(rng.standard_normal((1, 8, 1, 128)).astype(np.float16)),
            mx.array(rng.standard_normal((1, 8, 1, 128)).astype(np.float16)),
        )
    mx.eval(ko, vo)
    return np.array(ko), np.array(vo), cache


@pytest.mark.parametrize("method", ["kivi", "kivi_sink"])
def test_cache_output_identical_with_and_without_kernel(method):
    """Prefill + 40 decode steps, kernel on vs off, must agree bit-for-bit —
    including SinkProtectedKVCache, which inherits _quant_dequant_along."""
    k_mlx, v_mlx, c_mlx = _run_cache(method, use_metal=False)
    k_mtl, v_mtl, c_mtl = _run_cache(method, use_metal=True)
    assert np.array_equal(k_mlx, k_mtl), f"{method}: keys differ with kernel enabled"
    assert np.array_equal(v_mlx, v_mtl), f"{method}: values differ with kernel enabled"
    assert c_mlx.compressed_key_bytes == c_mtl.compressed_key_bytes
    assert c_mlx.effective_compression_ratio == c_mtl.effective_compression_ratio


def test_kernel_auto_enables_on_metal_builds():
    """Unset use_metal_kernels auto-detects, matching VecInferKVCache. Safe
    because the kernel is bit-identical to the MLX path (tested above) and
    measured faster on both schemes."""
    cache = KVCacheFactory.create(
        KVCacheConfig(method="kivi", head_dim=128, bit_width_inlier=2, kivi_group_size=32)
    )
    assert cache._use_metal is True


def test_kernel_can_be_forced_off():
    """use_metal_kernels=False must pin the pure-MLX path, so a bisect can
    rule the kernel out without rebuilding."""
    cache = KVCacheFactory.create(
        KVCacheConfig(
            method="kivi",
            head_dim=128,
            bit_width_inlier=2,
            kivi_group_size=32,
            use_metal_kernels=False,
        )
    )
    assert cache._use_metal is False


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_invalid_axis():
    with pytest.raises(ValueError, match="axis must be -1 or -2"):
        kivi_group_quant_dequant(mx.zeros((4, 8)), axis=0, group_size=32, levels=3)


def test_rejects_1d_input():
    with pytest.raises(ValueError, match="at least 2D"):
        kivi_group_quant_dequant(mx.zeros((8,)), axis=-1, group_size=32, levels=3)


def test_rejects_bad_group_size_and_levels():
    x = mx.zeros((4, 8))
    with pytest.raises(ValueError, match="group_size"):
        kivi_group_quant_dequant(x, axis=-1, group_size=0, levels=3)
    with pytest.raises(ValueError, match="levels"):
        kivi_group_quant_dequant(x, axis=-1, group_size=32, levels=0)


def test_rejects_unsupported_dtype():
    with pytest.raises(ValueError, match="unsupported dtype"):
        kivi_group_quant_dequant(mx.zeros((4, 8), dtype=mx.int32), axis=-1, group_size=32, levels=3)


# ---------------------------------------------------------------------------
# Benchmark (printed, not asserted) — the evidence for #164's "is a kernel
# worth it?" question.
# ---------------------------------------------------------------------------


def test_kivi_quant_benchmark(capsys):
    """Timing methodology matters more than usual here.

    Two naive setups each gave a *different wrong answer* while developing
    this kernel, so both are deliberately avoided:

    * **Repeating one identical input** lets MLX reuse work across calls in a
      way the kernel path cannot, which flattered the MLX baseline enough to
      invert the result (it reported the kernel 3x *slower*). Fixed by
      rotating through a pool of distinct inputs.
    * **Timing A fully, then B** lets thermal drift and any load from earlier
      tests land unevenly on the two paths. Fixed by interleaving A/B samples
      and reporting the median.
    """
    cache = _cache()
    rng = np.random.default_rng(0)
    rows = []

    for S in (32, 512, 2048):
        pool = [mx.array(rng.standard_normal((1, 8, S, 128)).astype(np.float16)) for _ in range(20)]
        mx.eval(*pool)
        for axis in (-2, -1):
            # Bind pool/axis as defaults: without this the closures capture the
            # loop variables by reference and every iteration would time the
            # final (S, axis) pair.
            def _mlx(i, _pool=pool, _axis=axis):
                return cache._quant_dequant_along(_pool[i % len(_pool)], axis=_axis)

            def _ker(i, _pool=pool, _axis=axis):
                return kivi_group_quant_dequant(
                    _pool[i % len(_pool)], axis=_axis, group_size=32, levels=3
                )

            for i in range(100):  # warm + compile, outside all timing
                mx.eval(_mlx(i))
                mx.eval(_ker(i))
            mx.synchronize()

            def _sample(fn, off, iters=50):
                mx.synchronize()
                t0 = time.perf_counter()
                for i in range(iters):
                    mx.eval(fn(i + off))
                mx.synchronize()
                return (time.perf_counter() - t0) / iters * 1e3

            s_mlx, s_ker = [], []
            for rep in range(9):  # interleaved so drift hits both equally
                s_mlx.append(_sample(_mlx, rep * 7))
                s_ker.append(_sample(_ker, rep * 7))
            rows.append((S, axis, float(np.median(s_mlx)), float(np.median(s_ker))))

    with capsys.disabled():
        print(f"\n# KIVI group quant  |  B=1 H=8 D=128 b=2 g=32  |  MLX {mx.__version__}")
        print("| S | axis | scheme | MLX ms | Metal ms | speedup |")
        print("|---|------|--------|--------|----------|---------|")
        for S, axis, t_mlx, t_ker in rows:
            scheme = "keys (per-channel)" if axis == -2 else "values (per-token)"
            print(f"| {S} | {axis} | {scheme} | {t_mlx:.4f} | {t_ker:.4f} | {t_mlx / t_ker:.2f}x |")
        print(
            "\nMedians of interleaved samples over a rotating input pool. The win\n"
            "grows with block size: the kernels replace several full-size MLX\n"
            "intermediates with one pass, so they scale with the bytes moved.\n"
            "Prefill (one large group-aligned flush) is where this matters;\n"
            "decode flushes only group_size tokens at a time."
        )
