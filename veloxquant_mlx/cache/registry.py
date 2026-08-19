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
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "ServeTier",
    "MethodFamily",
    "MethodInfo",
    "all_method_names",
    "list_methods",
    "get_method",
    "probe_serve_tier",
    "describe_field",
    "telemetry_coverage",
    "TelemetryCoverage",
    "DEFAULT_SERVE_METHOD",
]

DOCS_BASE = "https://veloxquant-mlx.netlify.app/algorithms"

#: What ``veloxquant serve`` starts with when the user does not pick a method.
#: This was originally chosen because ``KVCacheConfig`` defaulted to
#: ``turboquant_prod``, which is CRASHES-tier (#27). Since f6e9434 the library
#: default is ``turboquant_rvq`` — the same method — so the two now agree;
#: the constant is kept so the launcher default stays explicit rather than
#: silently tracking whatever the library default becomes next.
DEFAULT_SERVE_METHOD = "turboquant_rvq"


class ServeTier(str, Enum):
    """How well a method behaves under an ``mlx_lm.server`` process.

    Ordering reflects #27's support matrix. ``HONEST_BYTES`` is currently
    unreachable — it becomes reachable only when compressed storage is real
    (#27 option (d)). It exists so the UI has somewhere to put a method once
    that lands, instead of needing a new tier at that point.

    ``NOT_TRIMMABLE`` exists because "cannot be trimmed" and "crashes" are
    different facts and were previously conflated (#152). Every eviction cache
    deliberately returns ``is_trimmable() -> False`` — ``trim()`` would roll
    back only the base class's offset bookkeeping and not the internal
    per-token eviction state, silently corrupting later calls. Those caches
    serve correctly; only prompt-cache trimming is unavailable, so they must
    not be reported to users as unavailable methods.
    """

    HONEST_BYTES = "honest_bytes"
    ACCOUNTING_ONLY = "accounting_only"
    NOT_TRIMMABLE = "not_trimmable"
    CRASHES = "crashes"

    @property
    def is_servable(self) -> bool:
        return self is not ServeTier.CRASHES

    @property
    def is_trimmable(self) -> bool:
        """False when ``mlx_lm.server`` must not call ``trim()`` on this cache.

        Servable and trimmable are independent: a method can serve every
        request correctly while refusing to have its prompt cache trimmed.
        """
        return self is not ServeTier.NOT_TRIMMABLE

    @property
    def label(self) -> str:
        return {
            ServeTier.HONEST_BYTES: "available",
            ServeTier.ACCOUNTING_ONLY: "available",
            ServeTier.NOT_TRIMMABLE: "available (no prompt-cache trimming)",
            ServeTier.CRASHES: "not available yet",
        }[self]


class MethodFamily(str, Enum):
    """What the method primarily does to the cache."""

    QUANTIZATION = "quantization"
    EVICTION = "eviction"
    HYBRID = "hybrid"


class TelemetryCoverage(str, Enum):
    """Which byte counters a method actually reports.

    Coverage is uneven across the catalog: of 35 servable methods, 13 report
    keys and values, 5 report keys only (including the serve default
    ``turboquant_rvq``), and 17 — every eviction method — report none.

    A UI that assumes uniform coverage would render blanks or zeros for half
    the catalog and read as "no compression". This enum lets the UI say
    "not reported" instead, and lets a key-only ratio be labelled as such
    rather than passed off as whole-cache.
    """

    KEYS_AND_VALUES = "keys_and_values"
    KEYS_ONLY = "keys_only"
    NONE = "none"

    @property
    def label(self) -> str:
        return {
            TelemetryCoverage.KEYS_AND_VALUES: "full estimate",
            TelemetryCoverage.KEYS_ONLY: "partial estimate",
            TelemetryCoverage.NONE: "no estimate",
        }[self]


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
    coverage: TelemetryCoverage = TelemetryCoverage.NONE

    @property
    def docs_url(self) -> str:
        return f"{DOCS_BASE}/{self.name.replace('_', '-')}"

    @property
    def is_adapted(self) -> bool:
        """True when our implementation knowingly departs from the paper."""
        return self.paper_deviation is not None

    @property
    def field_schema(self) -> List[Dict[str, Any]]:
        """Type and default for each knob, derived from ``KVCacheConfig``.

        The UI renders inputs from this rather than from a hand-written table,
        so a field whose type or default changes cannot leave the panel
        submitting a value the config will reject.
        """
        return [describe_field(name) for name in self.config_fields]

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
            "field_schema": self.field_schema,
            "coverage": self.coverage.value,
            "coverage_label": self.coverage.label,
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
        "key_sub_dim",
        "value_sub_dim",
        "key_codebook_bits",
        "value_codebook_bits",
        "residual_length",
    ],
    "kivi": ["bit_width_inlier", "kivi_group_size"],
    "kivi_sink": ["bit_width_inlier", "kivi_group_size"],
    "svdq": [
        "svdq_rank",
        "svdq_energy_threshold",
        "svdq_bit_schedule",
        "svdq_group_size",
    ],
    "kitty": ["kitty_hi_fraction", "kitty_hi_bit", "kitty_lo_bit", "kitty_group_size"],
    "adakv": [
        "adakv_target_avg_bits",
        "adakv_lo_bit",
        "adakv_mid_bit",
        "adakv_hi_bit",
        "adakv_group_size",
        "adakv_update_interval",
    ],
    "xquant": [
        "xquant_group_size",
        "xquant_base_bits",
        "xquant_residual_bits",
        "xquant_group_quant_size",
        "xquant_max_ctx",
    ],
    "kvquant": [
        "kvquant_bits",
        "kvquant_outlier_fraction",
        "kvquant_group_size",
        "kvquant_lloyd_iters",
        "kvquant_refit_interval",
    ],
    "palu": ["palu_rank", "palu_energy_threshold"],
    "qjl": ["jl_dim", "seed"],
    "polar": ["bit_width_inlier", "seed"],
    "spectral": ["bit_width_inlier", "seed"],
}

_GENERIC_FIELDS = ["bit_width_inlier", "seed"]

_TIER_CACHE: Dict[str, "ServeTier"] = {}
_UNSUPPORTED_REASON: Dict[str, str] = {}


#: Human-readable hints for knobs whose names don't explain themselves.
_FIELD_HELP: Dict[str, str] = {
    "bit_width_inlier": "Bits per element for the main quantizer.",
    "seed": "Random seed for rotations / sketches.",
    "jl_dim": "Johnson-Lindenstrauss projection dimension.",
    "kivi_group_size": "Tokens per min/max quantization group.",
    "svdq_rank": "Latent rank; blank uses the energy threshold instead.",
    "svdq_energy_threshold": "Fraction of singular-value energy to retain.",
    "palu_rank": "Latent rank; blank uses the energy threshold instead.",
    "palu_energy_threshold": "Fraction of singular-value energy to retain.",
    "adakv_target_avg_bits": "Global average bits/element budget.",
    "kvquant_outlier_fraction": "Top-magnitude fraction kept in fp16.",
    "residual_length": "Recent tokens kept uncompressed.",
}


def describe_field(name: str) -> Dict[str, Any]:
    """Describe one ``KVCacheConfig`` field for form rendering.

    Reads the dataclass rather than a parallel table: types and defaults stay
    correct by construction. ``Optional[int]`` becomes a nullable int so the UI
    can offer a genuinely empty input, which is meaningful for fields like
    ``svdq_rank`` where blank selects a different code path.
    """
    import dataclasses

    from veloxquant_mlx.cache import base as _base
    from veloxquant_mlx.cache.base import KVCacheConfig

    hints = typing.get_type_hints(KVCacheConfig, globalns=vars(_base))
    fields = {f.name: f for f in dataclasses.fields(KVCacheConfig)}

    if name not in fields:
        return {"name": name, "type": "unknown", "default": None, "optional": True, "help": None}

    annotation = hints[name]
    optional = False
    origin = typing.get_origin(annotation)
    if origin is Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        optional = len(args) != len(typing.get_args(annotation))
        annotation = args[0] if args else annotation

    kind = {int: "int", float: "float", bool: "bool", str: "str"}.get(annotation, "unknown")

    default = fields[name].default
    if default is dataclasses.MISSING:
        default = None

    return {
        "name": name,
        "type": kind,
        "default": default,
        "optional": optional,
        "help": _FIELD_HELP.get(name),
    }


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

    # Not-trimmable is a *capability limit*, not a crash (#152). Every eviction
    # cache returns is_trimmable() -> False on purpose, because trim() would
    # roll back base-class offset bookkeeping without touching the internal
    # per-token eviction state. Such a cache still serves every request
    # correctly, so record the finding and keep probing rather than returning
    # early — a cache that is both not-trimmable *and* fails deepcopy is still
    # CRASHES, and returning here would hide that.
    trimmable = True
    try:
        trimmable = bool(can_trim_prompt_cache([cache]))
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

    if not trimmable:
        return (
            ServeTier.NOT_TRIMMABLE,
            "serves correctly, but reports is_trimmable() == False, so "
            "mlx_lm.server cannot trim its prompt cache — trim() would roll "
            "back offset bookkeeping without reverting internal eviction "
            "state. Expected for eviction/compression caches (#152).",
        )

    # Serves correctly, but stores dequantized fp16 and so over-reports nbytes.
    # Promoting to HONEST_BYTES requires #27 option (d); until then no method
    # may claim runtime memory reduction.
    return ServeTier.ACCOUNTING_ONLY, None


_COVERAGE_CACHE: Dict[str, "TelemetryCoverage"] = {}


def telemetry_coverage(method: str) -> "TelemetryCoverage":
    """Probe which byte counters a method exposes on a live cache.

    Probed rather than declared, for the same reason serve tier is: a counter
    added or removed from a cache must change what the UI promises, without
    anyone remembering to update a table.
    """
    if method in _COVERAGE_CACHE:
        return _COVERAGE_CACHE[method]

    coverage = TelemetryCoverage.NONE
    try:
        import mlx.core as mx

        from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory

        cache = KVCacheFactory.create(
            KVCacheConfig(method=method, head_dim=128, bit_width_inlier=2, seed=42)
        )
        keys = mx.random.normal((1, 8, 8, 128)).astype(mx.float16)
        cache.update_and_fetch(keys, keys)

        has_keys = hasattr(cache, "compressed_key_bytes")
        has_values = hasattr(cache, "compressed_value_bytes")
        if has_keys and has_values:
            coverage = TelemetryCoverage.KEYS_AND_VALUES
        elif has_keys:
            coverage = TelemetryCoverage.KEYS_ONLY
    except Exception:
        coverage = TelemetryCoverage.NONE

    _COVERAGE_CACHE[method] = coverage
    return coverage


def get_method(name: str) -> MethodInfo:
    """Look up one method, probing its serve tier on first access."""
    if name not in all_method_names():
        raise KeyError(f"unknown method {name!r}. Known methods: {', '.join(all_method_names())}")

    tier = probe_serve_tier(name)
    return MethodInfo(
        name=name,
        family=_FAMILY.get(name, MethodFamily.QUANTIZATION),
        serve_tier=tier,
        blurb=_BLURB.get(name, f"{name} KV-cache method."),
        config_fields=_CONFIG_FIELDS.get(name, list(_GENERIC_FIELDS)),
        paper_deviation=_PAPER_DEVIATION.get(name),
        unsupported_reason=_UNSUPPORTED_REASON.get(name),
        # Only meaningful for methods that actually run; a crash-tier cache
        # never reports anything.
        coverage=(telemetry_coverage(name) if tier.is_servable else TelemetryCoverage.NONE),
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
