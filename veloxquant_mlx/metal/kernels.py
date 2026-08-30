"""Metal kernel wrappers for VeloxQuant-MLX — re-export facade.

Kernels are organized into focused submodules:
  _vecinfer      — VecInfer codebook dequantize, quantize, encode+decode
  _bit_packing   — TurboQuant b-bit index pack/unpack
  _scalar_quant  — TurboQuant scalar quantize/dequantize + fused Hadamard
  _scalar_attend — Fused group-affine (KIVI-style) decode + attention
  _kivi_quant    — Fused KIVI asymmetric group quantize+dequantize round-trip
  _qjl           — QJL encode and inner product
  _rvq_attend    — Fused RVQ decode + attention
  _rvq_quant_pack — Fused RVQ stage-1+stage-2 quantize, packed to uint32 (#251)
  _comm_vq       — CommVQ RoPE-commutative decode
  _rabitq        — RaBitQ packed Hamming scoring
  _rabitq_attend — Fused RaBitQ asymmetric attention (1-bit K + 4-bit V)
  _rabitq_encode — Fused RaBitQ key encode (rotate + pack + magnitude)
  _rabitq_values — Nibble packing for 4-bit value indices
  _h2o_evict     — Fused H2O eviction: sink-protected argmin + evict + RoPE-remap
  _keyformer_evict — Fused Keyformer eviction: Gumbel-regularized argmin + evict + RoPE-remap
  _qfilters_evict — Fused Q-Filters eviction: projection scoring + block top-k compaction
  _crosskv_rope    — Cross-model KV transfer: fused source→target RoPE re-encode
"""

from __future__ import annotations

from veloxquant_mlx.metal._bit_packing import (
    turboquant_bit_pack,
    turboquant_bit_unpack,
)
from veloxquant_mlx.metal._comm_vq import (
    comm_vq_decode_metal,
)
from veloxquant_mlx.metal._crosskv_rope import (
    crosskv_rope_recode,
)
from veloxquant_mlx.metal._flash_prefill import (
    flash_prefill_attend,
)
from veloxquant_mlx.metal._h2o_evict import (
    h2o_fused_evict,
)
from veloxquant_mlx.metal._keyformer_evict import (
    keyformer_fused_evict,
)
from veloxquant_mlx.metal._kivi_quant import (
    kivi_group_quant_dequant,
)
from veloxquant_mlx.metal._qfilters_evict import (
    qfilters_fused_evict,
    qfilters_score,
)
from veloxquant_mlx.metal._qjl import (
    qjl_encode,
    qjl_inner_product,
)
from veloxquant_mlx.metal._rabitq import (
    rabitq_hamming_score,
)
from veloxquant_mlx.metal._rabitq_attend import (
    rabitq_fused_attend,
)
from veloxquant_mlx.metal._rabitq_encode import (
    rabitq_encode,
)
from veloxquant_mlx.metal._rabitq_prefill import (
    rabitq_prefill_attend,
)
from veloxquant_mlx.metal._rabitq_values import (
    rabitq_pack_values,
)
from veloxquant_mlx.metal._rvq_attend import (
    turboquant_fused_rvq_decode_attend,
)
from veloxquant_mlx.metal._rvq_quant_pack import (
    rvq_quant_pack,
)
from veloxquant_mlx.metal._scalar_attend import (
    scalar_fused_decode_attend,
)
from veloxquant_mlx.metal._scalar_quant import (
    turboquant_hadamard_quantize,
    turboquant_scalar_dequantize,
    turboquant_scalar_quantize,
)
from veloxquant_mlx.metal._vecinfer import (
    vecinfer_dequant_metal,
    vecinfer_encode_decode_metal,
    vecinfer_encode_decode_simple_metal,
    vecinfer_quantize_metal,
)

__all__ = [
    "crosskv_rope_recode",
    "vecinfer_dequant_metal",
    "vecinfer_quantize_metal",
    "vecinfer_encode_decode_metal",
    "vecinfer_encode_decode_simple_metal",
    "turboquant_bit_pack",
    "turboquant_bit_unpack",
    "turboquant_scalar_quantize",
    "turboquant_scalar_dequantize",
    "turboquant_hadamard_quantize",
    "scalar_fused_decode_attend",
    "kivi_group_quant_dequant",
    "qjl_encode",
    "qjl_inner_product",
    "turboquant_fused_rvq_decode_attend",
    "rvq_quant_pack",
    "comm_vq_decode_metal",
    "rabitq_hamming_score",
    "rabitq_fused_attend",
    "rabitq_encode",
    "rabitq_pack_values",
    "rabitq_prefill_attend",
    "flash_prefill_attend",
    "h2o_fused_evict",
    "keyformer_fused_evict",
    "qfilters_fused_evict",
    "qfilters_score",
]
