"""KV-cache quantizer and eviction-policy implementations.

Groups the two families of compression strategy the rest of the package
builds on: fixed-bit-width **quantizers** (QJL, TurboQuant MSE/Prod/RVQ,
PolarQuant, CommVQ, RaBitQ, KIVI, ZipCache) constructed via
``QuantizerFactory``/``CompositeQuantizer``, and stateful **eviction
policies** (SnapKV, StreamingLLM, H2O, TOVA, PyramidKV, SqueezeAttention)
exposed as ``init_*``/``*_update``/``*_get_kv``/``*_fp16_bytes`` function
sets operating on a per-policy state object. See ``__all__`` for the full
re-exported surface.
"""

from __future__ import annotations

from veloxquant_mlx.quantizers.base import QuantizerFactory
from veloxquant_mlx.quantizers.comm_vq import CommVQQuantizer
from veloxquant_mlx.quantizers.composite import CompositeQuantizer
from veloxquant_mlx.quantizers.h2o import (
    H2OState,
    full_h2o_fp16_bytes,
    h2o_fp16_bytes,
    h2o_get_kv,
    h2o_update,
    init_h2o_state,
)
from veloxquant_mlx.quantizers.kivi import KIVIQuantizer
from veloxquant_mlx.quantizers.polarquant import PolarQuantizer
from veloxquant_mlx.quantizers.pyramidkv import (
    PyramidState,
    full_pyramid_fp16_bytes,
    init_pyramid_state,
    pyramid_budgets,
    pyramid_fp16_bytes,
    pyramid_get_kv,
    pyramid_update,
)
from veloxquant_mlx.quantizers.qjl import QJLQuantizer
from veloxquant_mlx.quantizers.rabitq import RaBitQQuantizer
from veloxquant_mlx.quantizers.snapkv import (
    SnapKVState,
    full_fp16_bytes,
    obs_window_attention_scores,
    snap_select_indices,
    snapkv_compress,
    snapkv_fp16_bytes,
)
from veloxquant_mlx.quantizers.squeeze import (
    SqueezeState,
    concentration_score,
    full_squeeze_fp16_bytes,
    init_squeeze_state,
    squeeze_budgets,
    squeeze_fp16_bytes,
    squeeze_get_kv,
    squeeze_update,
)
from veloxquant_mlx.quantizers.streaming_llm import (
    StreamingWindow,
    full_stream_fp16_bytes,
    init_streaming_window,
    stream_fp16_bytes,
    stream_get_kv,
    stream_update,
)
from veloxquant_mlx.quantizers.tova import (
    TovaState,
    full_tova_fp16_bytes,
    init_tova_state,
    tova_fp16_bytes,
    tova_get_kv,
    tova_update,
)
from veloxquant_mlx.quantizers.turboquant_mse import TurboQuantMSE
from veloxquant_mlx.quantizers.turboquant_prod import TurboQuantProd
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ
from veloxquant_mlx.quantizers.zipcache import (
    ZipCacheState,
    base_only_bytes,
    channel_dequant,
    channel_quant,
    saliency_mask,
    token_key_norms,
    zipcache_bytes,
    zipcache_compress,
    zipcache_quant_dequant,
    zipcache_reconstruct,
)

__all__ = [
    "QuantizerFactory",
    "CompositeQuantizer",
    "PolarQuantizer",
    "QJLQuantizer",
    "TurboQuantMSE",
    "TurboQuantProd",
    "TurboQuantRVQ",
    "CommVQQuantizer",
    "RaBitQQuantizer",
    "KIVIQuantizer",
    "ZipCacheState",
    "token_key_norms",
    "saliency_mask",
    "channel_quant",
    "channel_dequant",
    "zipcache_compress",
    "zipcache_reconstruct",
    "zipcache_bytes",
    "base_only_bytes",
    "zipcache_quant_dequant",
    "SnapKVState",
    "obs_window_attention_scores",
    "snap_select_indices",
    "snapkv_compress",
    "snapkv_fp16_bytes",
    "full_fp16_bytes",
    "StreamingWindow",
    "init_streaming_window",
    "stream_update",
    "stream_get_kv",
    "stream_fp16_bytes",
    "full_stream_fp16_bytes",
    "H2OState",
    "init_h2o_state",
    "h2o_update",
    "h2o_get_kv",
    "h2o_fp16_bytes",
    "full_h2o_fp16_bytes",
    "TovaState",
    "init_tova_state",
    "tova_update",
    "tova_get_kv",
    "tova_fp16_bytes",
    "full_tova_fp16_bytes",
    "pyramid_budgets",
    "PyramidState",
    "init_pyramid_state",
    "pyramid_update",
    "pyramid_get_kv",
    "pyramid_fp16_bytes",
    "full_pyramid_fp16_bytes",
    "concentration_score",
    "squeeze_budgets",
    "SqueezeState",
    "init_squeeze_state",
    "squeeze_update",
    "squeeze_get_kv",
    "squeeze_fp16_bytes",
    "full_squeeze_fp16_bytes",
]
