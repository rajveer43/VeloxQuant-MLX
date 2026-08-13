"""Cross-model KV cache transfer — reuse one model's prefilled KV in another.

Implements "Cross-Model KV Cache Transfer in LLM Families: A Closed-Form
Linear Mapping for Prefill Reuse" (Heo, Shafipour, Zhao, Golub, Kamani,
Borkar, Chandran, Zardoshti, Darvish Rouhani — NVIDIA, arXiv:2608.03893,
https://arxiv.org/abs/2608.03893). Documented as
"CrossKV-adapted (VeloxQuant-MLX implementation)" — see the deviations
listed below and on the docs page.

**This is not a cache-compression method.** Every other method in this
repository shrinks a single model's KV cache and plugs into
``KVCacheConfig.method``. This subsystem does something structurally
different: it maps a *source* model's already-prefilled KV cache into a
*target* model's expected format, so the target can decode without paying
prefill again. Nothing is compressed; prefill compute is traded for a
per-layer matmul. It therefore lives outside ``KVCacheConfig`` entirely and
has its own API (see :mod:`veloxquant_mlx.transfer.mapper`).

The mapping is a per-(target layer, head) ridge regression fit offline on a
small calibration set, with three components from the paper:

  1. **Per-head ridge** (§3.1) — closed form ``W = (XᵀX + λI)⁻¹ XᵀY``, no
     gradient training.
  2. **Top-k cross-layer source selection** (§3.2) — each target layer is fit
     from the ``k`` source layers that predict it best, concatenated. The
     paper's largest single contributor: ``k=1`` collapses downstream
     accuracy.
  3. **Content-space (RoPE-stripped) mapping** (§3.3) — keys are un-rotated
     before the fit and re-rotated with the *target's* RoPE at apply time, so
     the fit is position-free and transfers across context lengths.

Adaptation notes (stated plainly, per this repo's convention):

  - **Scale.** The paper evaluates 14B→32B, 8B→70B, and similar pairs on
    8×H100 nodes. Holding source + target + mapper resident on Apple Silicon
    is only realistic at the small end of a family (e.g. Qwen3 0.6B→1.7B).
    The paper's 73–98% retention figures are **theirs, on their pairs** — do
    not restate them as measured results for small pairs until measured.
  - **Matched-KV only.** Like the paper, this requires source and target to
    share KV head count and per-head dimension. :func:`fit_mapper` refuses
    mismatched pairs rather than silently producing a bad map — the paper
    explicitly leaves mismatched-KV to future work.
  - **Ridge only.** The nonlinear MLP variant (paper §4.4), which recovers up
    to +37 pp on the pairs where ridge fails, is not implemented here.
    Consequence: pairs where the cross-model relationship is not close to
    linear will degrade with no fallback.
  - **No RoPE scaling.** Models using ``rope_scaling`` (Llama-3.1-style) are
    refused, not approximated — the same gap tracked for the eviction caches
    in #148.

Public API::

    from veloxquant_mlx.transfer import fit_mapper, load_mapper, transfer_cache

    mapper = fit_mapper(source_model, target_model, calibration_texts, tokenizer)
    mapper.save("qwen3-0.6b-to-1.7b")

    # later, at serving time
    mapper = load_mapper("qwen3-0.6b-to-1.7b")
    target_cache = transfer_cache(source_cache, mapper)
"""

from __future__ import annotations

from veloxquant_mlx.transfer.apply import load_mapper, transfer_cache
from veloxquant_mlx.transfer.fit import (
    fit_mapper,
    fit_ridge,
    layer_r2,
    select_source_layers,
)
from veloxquant_mlx.transfer.mapper import (
    CrossModelMapper,
    LayerMap,
    MapperConfig,
    MatchedKVError,
    ModelKVSpec,
)
from veloxquant_mlx.transfer.rope import (
    apply_rope,
    recode_rope,
    strip_rope,
)

__all__ = [
    "strip_rope",
    "apply_rope",
    "recode_rope",
    "CrossModelMapper",
    "LayerMap",
    "MapperConfig",
    "ModelKVSpec",
    "MatchedKVError",
    "fit_mapper",
    "fit_ridge",
    "select_source_layers",
    "layer_r2",
    "transfer_cache",
    "load_mapper",
]
