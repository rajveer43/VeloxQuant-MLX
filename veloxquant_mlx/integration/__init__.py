"""Integration shims that wire VeloxQuant-MLX caches into third-party inference loops.

Monkey-patches ``mlx_lm`` and ``mlx_vlm`` model instances so their normal
``generate()`` entry points build a VeloxQuant KV cache instead of the
stock ``mlx_lm.models.cache.KVCache``, without requiring callers to modify
the host library. Re-exports :func:`~veloxquant_mlx.integration.mlx_lm_patch.patch_model_kv_cache`
and :func:`~veloxquant_mlx.integration.mlx_vlm_patch.patch_vlm_kv_cache`.
"""

from __future__ import annotations

from veloxquant_mlx.integration.mlx_lm_patch import patch_model_kv_cache
from veloxquant_mlx.integration.mlx_vlm_patch import patch_vlm_kv_cache

__all__ = ["patch_model_kv_cache", "patch_vlm_kv_cache"]
