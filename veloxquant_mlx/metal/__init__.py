"""Optional Metal compute kernels for VeloxQuant-MLX.

These kernels replace hot-path pure-MLX operations with hand-written
Metal Shading Language shaders compiled at first use via
``mx.fast.metal_kernel``. The library still works without Metal (the
caches detect availability and fall back to the pure-MLX implementations
exported from :mod:`veloxquant_mlx.allocators.vecinfer`).

Phase 1 implements only the VecInfer codebook dequantization kernel
(the most expensive op on every ``update_and_fetch`` call when running
with low-bit codebooks). Fused dequant+SDPA is Phase 2.

Public surface:

* :func:`metal_available` — runtime capability check.
* :data:`USE_METAL` — module-level cached flag.
* :func:`vecinfer_dequant_metal` — Metal-backed
  ``dequantize_vq`` drop-in (re-exported from :mod:`.kernels`).
"""

from __future__ import annotations

import mlx.core as mx


def metal_available() -> bool:
    """True iff ``mx.fast.metal_kernel`` is usable on this build."""
    try:
        if not mx.metal.is_available():
            return False
        if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
            return False
        return True
    except Exception:
        return False


USE_METAL: bool = metal_available()


# Lazy re-export so importing the package doesn't compile the kernel.
def __getattr__(name: str):
    if name == "vecinfer_dequant_metal":
        from .kernels import vecinfer_dequant_metal as _fn

        return _fn
    if name == "vecinfer_quantize_metal":
        from .kernels import vecinfer_quantize_metal as _fn

        return _fn
    if name == "metal_fused_sdpa":
        from .fused_sdpa import metal_fused_sdpa as _fn

        return _fn
    if name == "fused_sdpa_supports_shape":
        from .fused_sdpa import supports_shape as _fn

        return _fn
    # TurboQuant kernels
    _turboquant_names = {
        "turboquant_bit_pack",
        "turboquant_bit_unpack",
        "turboquant_scalar_quantize",
        "turboquant_scalar_dequantize",
        "turboquant_hadamard_quantize",
        "qjl_encode",
        "qjl_inner_product",
        "turboquant_fused_rvq_decode_attend",
        "comm_vq_decode_metal",
        "rabitq_hamming_score",
    }
    if name in _turboquant_names:
        from . import kernels as _k

        return getattr(_k, name)
    raise AttributeError(f"module 'veloxquant_mlx.metal' has no attribute {name!r}")


__all__ = [
    "metal_available",
    "USE_METAL",
    # VecInfer
    "vecinfer_dequant_metal",
    "vecinfer_quantize_metal",
    "metal_fused_sdpa",
    "fused_sdpa_supports_shape",
    # TurboQuant
    "turboquant_bit_pack",
    "turboquant_bit_unpack",
    "turboquant_scalar_quantize",
    "turboquant_scalar_dequantize",
    "turboquant_hadamard_quantize",
    "qjl_encode",
    "qjl_inner_product",
    "turboquant_fused_rvq_decode_attend",
    "comm_vq_decode_metal",
    "rabitq_hamming_score",
]
