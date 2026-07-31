"""Runtime registry of KV-cache methods, with serve-tier classification.

The control panel (#33) and ``veloxquant serve`` (#27) both need to answer
"which methods can actually be served, and what does each one do?" without
hardcoding a marketing list. This module derives that answer from the code:
method names come from ``KVCacheConfig``'s own ``Literal``, and serve tiers
come from *probing* a live cache instance rather than a hand-maintained table.

Probing matters. #27 found that 5 of 40 methods hard-crash under
``mlx_lm.server`` because they do not subclass ``mlx_lm.models.cache.KVCache``
and so never inherit ``update_and_fetch`` / ``is_trimmable``. A hand-written
list would drift from that reality the moment a cache changes base class; a
probe cannot.

Usage::

    from veloxquant_mlx.cache.registry import get_method, list_methods

    info = get_method("turboquant_rvq")
    if not info.serve_tier.is_servable:
        raise SystemExit(info.unsupported_reason)
"""
from __future__ import annotations

import copy
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

__all__ = [
    "ServeTier",
    "MethodFamily",
    "MethodInfo",
    "all_method_names",
    "list_methods",
    "get_method",
    "probe_serve_tier",
    "DEFAULT_SERVE_METHOD",
]

DOCS_BASE = "https://veloxquant-mlx.netlify.app/algorithms"

#: What ``veloxquant serve`` starts with when the user does not pick a method.
#: Deliberately *not* ``KVCacheConfig``'s default (``turboquant_prod``), which
#: is in the CRASHES tier — see #27 and the note in ``veloxquant_mlx/cli/serve.py``.
DEFAULT_SERVE_METHOD = "turboquant_rvq"


class ServeTier(str, Enum):
    """How well a method behaves under an ``mlx_lm.server`` process.

    Ordering reflects #27's support matrix. ``HONEST_BYTES`` is currently
    unreachable — it becomes reachable only when compressed storage is real
    (#27 option (d)). It exists so the UI has somewhere to put a method once
    that lands, instead of needing a new tier at that point.
    """

    HONEST_BYTES = "honest_bytes"
    ACCOUNTING_ONLY = "accounting_only"
    CRASHES = "crashes"

    @property
    def is_servable(self) -> bool:
        return self is not ServeTier.CRASHES

    @property
    def label(self) -> str:
        return {
            ServeTier.HONEST_BYTES: "serves",
            ServeTier.ACCOUNTING_ONLY: "accounting-only",
            ServeTier.CRASHES: "unsupported",
        }[self]


class MethodFamily(str, Enum):
    """What the method primarily does to the cache."""

    QUANTIZATION = "quantization"
    EVICTION = "eviction"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class MethodInfo:
    """Everything a UI needs to present one method."""

    name: str
    family: MethodFamily
    serve_tier: ServeTier
    blurb: str
    config_fields: List[str] = field(default_factory=list)
    paper_deviation: Optional[str] = None
    unsupported_reason: Optional[str] = None

    @property
    def docs_url(self) -> str:
        return f"{DOCS_BASE}/{self.name.replace('_', '-')}"

    @property
    def is_adapted(self) -> bool:
        """True when our implementation knowingly departs from the paper."""
        return self.paper_deviation is not None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form, consumed by ``veloxquant methods --json``."""
        return {
            "name": self.name,
            "family": self.family.value,
            "serve_tier": self.serve_tier.value,
            "serve_tier_label": self.serve_tier.label,
            "is_servable": self.serve_tier.is_servable,
            "blurb": self.blurb,
            "config_fields": list(self.config_fields),
            "paper_deviation": self.paper_deviation,
            "is_adapted": self.is_adapted,
            "unsupported_reason": self.unsupported_reason,
            "docs_url": self.docs_url,
        }


# --- Static descriptive metadata -------------------------------------------
# Family / blurb / knobs are editorial and cannot be derived from code.
# Serve tier is deliberately absent here: it is probed, never declared.
# A method missing from this table still appears in the registry with a
# placeholder blurb, so adding a cache never silently drops it from the UI.

_FAMILY: Dict[str, MethodFamily] = {
    "turboquant_prod": MethodFamily.QUANTIZATION,
    "turboquant_mse": MethodFamily.QUANTIZATION,
    "turboquant_rvq": MethodFamily.QUANTIZATION,
    "polar": MethodFamily.QUANTIZATION,
    "qjl": MethodFamily.QUANTIZATION,
    "vecinfer": MethodFamily.QUANTIZATION,
    "spectral": MethodFamily.QUANTIZATION,
    "kivi": MethodFamily.QUANTIZATION,
    "kivi_sink": MethodFamily.HYBRID,
    "svdq": MethodFamily.QUANTIZATION,
    "kitty": MethodFamily.QUANTIZATION,
    "adakv": MethodFamily.QUANTIZATION,
    "xquant": MethodFamily.QUANTIZATION,
    "kvquant": MethodFamily.QUANTIZATION,
    "palu": MethodFamily.QUANTIZATION,
    "cachegen": MethodFamily.QUANTIZATION,
    "minicache": MethodFamily.HYBRID,
    "gear": MethodFamily.QUANTIZATION,
    "zipcache": MethodFamily.HYBRID,
    "snapkv": MethodFamily.EVICTION,
    "streaming_llm": MethodFamily.EVICTION,
    "h2o": MethodFamily.EVICTION,
    "tova": MethodFamily.EVICTION,
    "pyramidkv": MethodFamily.EVICTION,
    "squeeze": MethodFamily.EVICTION,
    "chunkkv": MethodFamily.EVICTION,
    "cam": MethodFamily.EVICTION,
    "xkv": MethodFamily.EVICTION,
    "nsnquant": MethodFamily.QUANTIZATION,
    "knorm": MethodFamily.EVICTION,
    "skvq": MethodFamily.HYBRID,
    "qfilters": MethodFamily.EVICTION,
    "keyformer": MethodFamily.EVICTION,
    "morphkv": MethodFamily.EVICTION,
    "kvzip": MethodFamily.EVICTION,
    "kvtc": MethodFamily.QUANTIZATION,
    "curdkv": MethodFamily.EVICTION,
    "nestedkv": MethodFamily.QUANTIZATION,
    "amc": MethodFamily.EVICTION,
    "a2ats": MethodFamily.HYBRID,
}

_BLURB: Dict[str, str] = {
    "turboquant_prod": "TurboQuant product quantization; the library default for offline study.",
    "turboquant_mse": "TurboQuant variant fitted to minimize MSE rather than inner-product error.",
    "turboquant_rvq": "Residual vector quantization; the balanced default for serving.",
    "polar": "PolarQuant: polar-coordinate encoding of key vectors.",
    "qjl": "QJL: Johnson-Lindenstrauss sketch with 1-bit quantization.",
    "vecinfer": "VecInfer: codebook vector quantization for aggressive compression.",
    "spectral": "SpectralQuant: spectral-domain transform before quantization.",
    "kivi": "KIVI: asymmetric per-group min/max quantization, key-per-channel.",
    "kivi_sink": "KIVI with attention-sink protection for the first tokens.",
    "svdq": "SVDq: offline SVD to a latent basis, then mixed-precision on latents.",
    "kitty": "Kitty: dynamic channel-wise mixed precision by variance.",
    "adakv": "AdaKV-proxy: per-head adaptive bit allocation under a global budget.",
    "xquant": "XQuant: cross-layer KV reuse with anchor layers plus residuals.",
    "kvquant": "KVQuant-NUQ: non-uniform levels via Lloyd-Max with fp16 outliers.",
    "palu": "PALU: true low-rank latent projection of both keys and values.",
    "cachegen": "CacheGen: entropy-coded cache representation.",
    "minicache": "MiniCache: merges similar KV state across adjacent layers.",
    "gear": "GEAR: quantization plus a low-rank error-correction term.",
    "zipcache": "ZipCache: saliency-weighted mixed-precision compression.",
    "snapkv": "SnapKV: keeps tokens an observation window attends to most.",
    "streaming_llm": "StreamingLLM: attention sinks plus a sliding window.",
    "h2o": "H2O: keeps 'heavy hitter' tokens by accumulated attention.",
    "tova": "TOVA: single-token-per-step eviction by attention weight.",
    "pyramidkv": "PyramidKV: layer-varying budgets, wider at shallow layers.",
    "squeeze": "SqueezeAttention: reallocates budget across layers by importance.",
    "chunkkv": "ChunkKV: evicts at chunk granularity to preserve local semantics.",
    "cam": "CaM: merges evicted state back instead of discarding it.",
    "xkv": "XKV: cross-layer budget optimization.",
    "nsnquant": "NSNQuant: normalize-shift-normalize before quantization.",
    "knorm": "K-norm: evicts by key-norm as an attention proxy.",
    "skvq": "SKVQ: sliding-window quantization with clipped dynamic range.",
    "qfilters": "Q-Filters: projects keys onto learned filters to score them.",
    "keyformer": "Keyformer: Gumbel-softmax scoring for key-token selection.",
    "morphkv": "MorphKV: correlation-aware constant-size cache.",
    "kvzip": "KVzip: query-agnostic eviction via context reconstruction.",
    "kvtc": "KVTC: transform coding of the cache.",
    "curdkv": "CurDKV: CUR decomposition for value-aware selection.",
    "nestedkv": "NestedKV: hierarchical nested codebooks.",
    "amc": "AMC: adaptive memory compression.",
    "a2ats": "A2ATS-adapted: rotary-aware vector quantization with distance gating.",
}

#: Honest "-adapted" notes. Sourced from open issues that document the
#: deviation, so the UI cannot claim faithful reproduction where we know better.
_PAPER_DEVIATION: Dict[str, str] = {
    "adakv": (
        "Default config is non-adaptive (target equals lo_bit) and the importance "
        "proxy is sign-inverted vs the paper. See issue #31."
    ),
    "a2ats": (
        "Windowed RoPE rotates far keys that the paper leaves unrotated, and "
        "distance gating is frozen at write time. See issue #29."
    ),
}

#: Method-specific knobs, so #35 can show only the relevant fields.
_CONFIG_FIELDS: Dict[str, List[str]] = {
    "turboquant_rvq": ["bit_width_inlier", "seed"],
    "turboquant_prod": ["bit_width_inlier", "seed"],
    "turboquant_mse": ["bit_width_inlier", "seed"],
    "vecinfer": [
        "key_sub_dim", "value_sub_dim", "key_codebook_bits",
        "value_codebook_bits", "residual_length",
    ],
    "kivi": ["bit_width_inlier", "kivi_group_size"],
    "kivi_sink": ["bit_width_inlier", "kivi_group_size"],
    "svdq": [
        "svdq_rank", "svdq_energy_threshold", "svdq_hi_bit",
        "svdq_lo_bit", "svdq_hi_fraction", "svdq_group_size",
    ],
    "kitty": ["kitty_hi_fraction", "kitty_hi_bit", "kitty_lo_bit", "kitty_group_size"],
    "adakv": [
        "adakv_target_avg_bits", "adakv_lo_bit", "adakv_mid_bit",
        "adakv_hi_bit", "adakv_group_size", "adakv_update_interval",
    ],
    "xquant": [
        "xquant_group_size", "xquant_base_bits",
        "xquant_residual_bits", "xquant_group_quant_size", "xquant_max_ctx",
    ],
    "kvquant": [
        "kvquant_bits", "kvquant_outlier_fraction", "kvquant_group_size",
        "kvquant_lloyd_iters", "kvquant_refit_interval",
    ],
    "palu": ["palu_rank", "palu_energy_threshold"],
    "qjl": ["jl_dim", "seed"],
    "polar": ["bit_width_inlier", "seed"],
    "spectral": ["bit_width_inlier", "seed"],
}

_GENERIC_FIELDS = ["bit_width_inlier", "seed"]

_TIER_CACHE: Dict[str, "ServeTier"] = {}
_UNSUPPORTED_REASON: Dict[str, str] = {}


def all_method_names() -> List[str]:
    """Every method name ``KVCacheConfig`` accepts, straight from its ``Literal``.

    Read from the annotation rather than a copy of the list, so a new method
    is registered by the act of declaring it.
    """
    from veloxquant_mlx.cache import base as _base
    from veloxquant_mlx.cache.base import KVCacheConfig

    # ``from __future__ import annotations`` in base.py stringifies annotations,
    # so resolve against that module's globals rather than reading __annotations__.
    hints = typing.get_type_hints(KVCacheConfig, globalns=vars(_base))
    return list(typing.get_args(hints["method"]))


def probe_serve_tier(method: str) -> "ServeTier":
    """Classify one method by exercising a real cache instance.

    A method serves only if it inherits the ``mlx_lm`` serving contract: the
    server calls ``update_and_fetch`` per step, ``can_trim_prompt_cache`` for
    prefix reuse, and ``deepcopy`` to hand each request its own copy
    (``mlx_lm.models.cache`` line 1676). Any of those missing is a crash at
    request time, so all three are probed here.

    Results are memoized; the probe allocates small tensors and is cheap, but
    the UI calls this per method on every listing.
    """
    if method in _TIER_CACHE:
        return _TIER_CACHE[method]

    tier, reason = _run_probe(method)
    _TIER_CACHE[method] = tier
    if reason:
        _UNSUPPORTED_REASON[method] = reason
    return tier


def _run_probe(method: str) -> tuple["ServeTier", Optional[str]]:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache as _MLXKVCache
    from mlx_lm.models.cache import can_trim_prompt_cache

    from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

    try:
        cache = KVCacheFactory.create(
            KVCacheConfig(method=method, head_dim=128, bit_width_inlier=2, seed=42)
        )
    except Exception as exc:  # construction failure is itself disqualifying
        return ServeTier.CRASHES, f"cache construction failed: {type(exc).__name__}: {exc}"

    if not isinstance(cache, _MLXKVCache):
        return (
            ServeTier.CRASHES,
            f"{type(cache).__name__} does not subclass mlx_lm KVCache, so it "
            "inherits neither update_and_fetch nor is_trimmable; mlx_lm.server "
            "raises AttributeError on the first request. Needs an adapter (#27).",
        )

    try:
        keys = mx.random.normal((1, 8, 16, 128)).astype(mx.float16)
        values = mx.random.normal((1, 8, 16, 128)).astype(mx.float16)
        out_k, out_v = cache.update_and_fetch(keys, values)
        mx.eval(out_k, out_v)
        _ = cache.nbytes
    except Exception as exc:
        return ServeTier.CRASHES, f"update_and_fetch failed: {type(exc).__name__}: {exc}"

    try:
        if not can_trim_prompt_cache([cache]):
            return ServeTier.CRASHES, "cache reports itself as not trimmable"
    except Exception as exc:
        return ServeTier.CRASHES, f"trim probe failed: {type(exc).__name__}: {exc}"

    try:
        copy.deepcopy(cache)
    except Exception as exc:
        return (
            ServeTier.CRASHES,
            f"deepcopy failed ({type(exc).__name__}: {exc}); mlx_lm.server "
            "deepcopies cache entries per request.",
        )

    # Serves correctly, but stores dequantized fp16 and so over-reports nbytes.
    # Promoting to HONEST_BYTES requires #27 option (d); until then no method
    # may claim runtime memory reduction.
    return ServeTier.ACCOUNTING_ONLY, None


def get_method(name: str) -> MethodInfo:
    """Look up one method, probing its serve tier on first access."""
    if name not in all_method_names():
        raise KeyError(
            f"unknown method {name!r}. Known methods: {', '.join(all_method_names())}"
        )

    tier = probe_serve_tier(name)
    return MethodInfo(
        name=name,
        family=_FAMILY.get(name, MethodFamily.QUANTIZATION),
        serve_tier=tier,
        blurb=_BLURB.get(name, f"{name} KV-cache method."),
        config_fields=_CONFIG_FIELDS.get(name, list(_GENERIC_FIELDS)),
        paper_deviation=_PAPER_DEVIATION.get(name),
        unsupported_reason=_UNSUPPORTED_REASON.get(name),
    )


def list_methods(
    *,
    servable_only: bool = False,
    family: Optional[MethodFamily] = None,
) -> List[MethodInfo]:
    """All methods, optionally filtered. Sorted servable-first, then by name."""
    infos = [get_method(n) for n in all_method_names()]

    if servable_only:
        infos = [i for i in infos if i.serve_tier.is_servable]
    if family is not None:
        infos = [i for i in infos if i.family is family]

    return sorted(infos, key=lambda i: (not i.serve_tier.is_servable, i.name))
