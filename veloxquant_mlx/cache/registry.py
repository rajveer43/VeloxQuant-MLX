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
import types
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union, cast

__all__ = [
    "ServeTier",
    "MethodFamily",
    "MethodInfo",
    "StrategyCapabilities",
    "all_method_names",
    "list_methods",
    "get_method",
    "static_method_info",
    "probe_serve_tier",
    "describe_field",
    "field_is_relevant",
    "telemetry_coverage",
    "TelemetryCoverage",
    "DEFAULT_SERVE_METHOD",
]

DOCS_BASE = "https://veloxquant-mlx.netlify.app/docs/algorithms"

#: Registry method name -> published doc-site slug. The docs site (Docusaurus,
#: migrated from the old static-HTML `/algorithms/<slug>` layout) has its own
#: page-naming scheme that does not mechanically match `MethodInfo.name`
#: (e.g. `turboquant_rvq` -> `rvq`, `polar` -> `polarquant`), and it does not
#: yet have a page for most registry methods. A method absent from this map
#: has no docs page; `docs_url` is `None` rather than a synthesized dead link.
#: Sourced from https://veloxquant-mlx.netlify.app/sitemap.xml.
_DOCS_SLUG: dict[str, str] = {
    "turboquant_rvq": "rvq",
    "polar": "polarquant",
    "kivi": "kivi",
    "qjl": "qjl",
    "spectral": "spectral",
    "vecinfer": "vecinfer",
}

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
        """Whether ``mlx_lm.server`` can run this method at all (every tier except ``CRASHES``)."""
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
        """Human-readable UI label for this serve tier."""
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
        """Human-readable UI label for this telemetry coverage level."""
        return {
            TelemetryCoverage.KEYS_AND_VALUES: "full estimate",
            TelemetryCoverage.KEYS_ONLY: "partial estimate",
            TelemetryCoverage.NONE: "no estimate",
        }[self]


@dataclass(frozen=True)
class StrategyCapabilities:
    """What a strategy can and cannot do, for the auto-selection pipeline.

    Fed to :func:`veloxquant_mlx.planning.candidate_filter.filter_candidates`
    so incompatible methods can be ruled out before any scoring happens.
    Conservative defaults (``supports_*`` all True) mean a method whose
    capabilities are unknown is never *wrongly* excluded — the cost of being
    wrong that way is a recommendation that does not fit, whereas the cost of
    wrongly including a method is at most a deprioritized candidate.
    """

    supported_bits: list[int] | None = None
    requires_calibration: bool = False
    supports_gqa: bool = True
    supports_mqa: bool = True
    supports_mha: bool = True
    supports_prefill: bool = True
    supports_decode: bool = True
    supports_streaming: bool = False
    has_metal_kernel: bool = False
    compresses_keys: bool = True
    compresses_values: bool = False
    uses_eviction: bool = False
    uses_merging: bool = False
    tunable_parameters: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict serialization of this capability set (for JSON API responses)."""
        return {
            "supported_bits": self.supported_bits,
            "requires_calibration": self.requires_calibration,
            "supports_gqa": self.supports_gqa,
            "supports_mqa": self.supports_mqa,
            "supports_mha": self.supports_mha,
            "supports_prefill": self.supports_prefill,
            "supports_decode": self.supports_decode,
            "supports_streaming": self.supports_streaming,
            "has_metal_kernel": self.has_metal_kernel,
            "compresses_keys": self.compresses_keys,
            "compresses_values": self.compresses_values,
            "uses_eviction": self.uses_eviction,
            "uses_merging": self.uses_merging,
            "tunable_parameters": dict(self.tunable_parameters),
        }


@dataclass(frozen=True)
class MethodInfo:
    """Everything a UI needs to present one method."""

    name: str
    family: MethodFamily
    serve_tier: ServeTier
    blurb: str
    config_fields: list[str] = field(default_factory=list)
    paper_deviation: str | None = None
    unsupported_reason: str | None = None
    coverage: TelemetryCoverage = TelemetryCoverage.NONE
    capabilities: StrategyCapabilities = field(default_factory=StrategyCapabilities)

    @property
    def docs_url(self) -> str | None:
        """Full documentation URL for this method, or None if no docs slug is registered."""
        slug = _DOCS_SLUG.get(self.name)
        return f"{DOCS_BASE}/{slug}" if slug is not None else None

    @property
    def is_adapted(self) -> bool:
        """True when our implementation knowingly departs from the paper."""
        return self.paper_deviation is not None

    @property
    def field_schema(self) -> list[dict[str, Any]]:
        """Type and default for each knob, derived from ``KVCacheConfig``.

        The UI renders inputs from this rather than from a hand-written table,
        so a field whose type or default changes cannot leave the panel
        submitting a value the config will reject.
        """
        return [describe_field(name) for name in self.config_fields]

    def to_dict(self) -> dict[str, Any]:
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
            "capabilities": self.capabilities.to_dict(),
        }


# --- Static descriptive metadata -------------------------------------------
# Family / blurb / knobs are editorial and cannot be derived from code.
# Serve tier is deliberately absent here: it is probed, never declared.
# A method missing from this table still appears in the registry with a
# placeholder blurb, so adding a cache never silently drops it from the UI.

_FAMILY: dict[str, MethodFamily] = {
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
    "anchorkv": MethodFamily.HYBRID,
    "rocketkv": MethodFamily.HYBRID,
    "age_tiered": MethodFamily.QUANTIZATION,
}

_BLURB: dict[str, str] = {
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
    "anchorkv": "AnchorKV-adapted: anchor-residual compression, no eviction.",
    "rocketkv": "RocketKV-adapted: SnapKV eviction + hybrid sparse attention selection.",
    "age_tiered": "AgeTieredKV: position/age-gated 3-tier precision (recent/mid/old), no eviction.",
}

#: Honest "-adapted" notes. Sourced from open issues that document the
#: deviation, so the UI cannot claim faithful reproduction where we know better.
_PAPER_DEVIATION: dict[str, str] = {
    "adakv": (
        "Default importance proxy (norm_variance) is sign-inverted vs the paper's "
        "attention-entropy criterion; set adakv_importance=\"attention_entropy\" to "
        "match the paper's sign. (The non-adaptive target==lo_bit default this "
        "note used to describe was fixed by #31 — target_avg_bits now defaults "
        "to 2.5, above lo_bit.)"
    ),
    "a2ats": (
        "Windowed RoPE rotates far keys that the paper leaves unrotated, and "
        "distance gating is frozen at write time. See issue #29."
    ),
}

#: Method-specific knobs, so #35 can show only the relevant fields.
_CONFIG_FIELDS: dict[str, list[str]] = {
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
    # n_sink_tokens (SinkProtectedKVCache.__init__, default 5) was missing
    # here -- unlike a kivi_*-prefixed field, its generic-looking name never
    # matches field_is_relevant's own-name prefix fallback either, so it was
    # invisible to both `--set` validation and the app's parameter editor.
    # Found verifying VeloxQuant-Studio issue #14.
    "kivi_sink": ["bit_width_inlier", "kivi_group_size", "n_sink_tokens"],
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
        "kvquant_n_sink",
    ],
    "palu": [
        "palu_rank",
        "palu_energy_threshold",
        "palu_n_head_groups",
        "palu_hi_bit",
        "palu_lo_bit",
        "palu_hi_fraction",
        "palu_group_size",
        "palu_quantize_values",
    ],
    # pyramid_resolved_budget is deliberately excluded: it is an internal,
    # per-layer field written only by KVCacheBuilder._build_pyramidkv (via
    # dataclasses.replace), never a user-facing knob -- the uncurated
    # name-prefix fallback (field_is_relevant / _default_config_fields) can't
    # tell that apart from pyramid_budget/pyramid_n_sink/pyramid_beta, since
    # all four share the "pyramid_" prefix. Left exposed, `--set
    # pyramid_resolved_budget=N` or the app's parameter editor could silently
    # pin every layer to one fixed budget, bypassing the pyramid schedule
    # entirely with no indication anything unusual happened. Found verifying
    # VeloxQuant-Studio issue #24; cachegen_resolved_bits and
    # squeeze_resolved_budget are the same pattern on cachegen/squeeze.
    # squeeze fixed verifying issue #28; cachegen_resolved_bits remains open.
    "pyramidkv": ["pyramid_backend", "pyramid_beta", "pyramid_budget", "pyramid_n_sink"],
    # squeeze_resolved_budget is deliberately excluded: same class of bug as
    # pyramid_resolved_budget above, but resolved differently at runtime --
    # KVCacheBuilder._build_squeeze never writes it via dataclasses.replace;
    # instead it hands every layer a shared SqueezeCoordinator object, and
    # each layer pulls its resolved per-layer budget from the coordinator
    # once every layer has reported prefill concentration
    # (SqueezeAttentionCache._report_and_rebudget). The field only exists as
    # a manual override for single-cache/testing construction with no
    # coordinator. It still shares the "squeeze_" prefix with
    # squeeze_budget/squeeze_n_sink/squeeze_strength, so the uncurated
    # name-prefix fallback can't tell them apart. Left exposed, `--set
    # squeeze_resolved_budget=N` or the app's parameter editor could silently
    # pin every layer to one fixed budget, bypassing the 2D data-driven
    # reallocation entirely with no indication anything unusual happened.
    # Found verifying VeloxQuant-Studio issue #28.
    "squeeze": ["squeeze_budget", "squeeze_n_sink", "squeeze_strength"],
    "qjl": ["jl_dim", "seed"],
    "polar": ["bit_width_inlier", "seed"],
    "spectral": ["bit_width_inlier", "seed"],
    "rocketkv": [
        "rocketkv_compression_ratio",
        "rocketkv_page_size",
        "rocketkv_head_topk1",
        "rocketkv_obs_window",
        "rocketkv_n_sink",
    ],
    "age_tiered": [
        "age_recent_boundary",
        "age_mid_boundary",
        "age_bits_recent",
        "age_bits_mid",
        "age_bits_old",
        "age_group_size",
    ],
}

_GENERIC_FIELDS = ["bit_width_inlier", "seed"]

#: Per-method capability overrides on top of the family-derived defaults.
#: A method missing here keeps the family defaults (quantization -> key
#: compression; eviction -> bounded-memory token dropping), which is the
#: conservative choice for the auto-selection pipeline: better to include a
#: method that turns out weaker than to wrongly rule one out. `supported_bits`
#: and `has_metal_kernel` are always explicit because neither follows from the
#: family alone.
_CAPABILITIES: dict[str, dict[str, Any]] = {
    # --- TurboQuant family: vector rotations/codebooks precomputed offline ---
    "turboquant_prod": {"requires_calibration": True, "supported_bits": [1, 2, 3, 4]},
    "turboquant_mse": {"requires_calibration": True, "supported_bits": [1, 2, 3, 4]},
    "turboquant_rvq": {"supported_bits": [1, 2, 3, 4], "has_metal_kernel": True},
    # --- Standalone research methods: precomputed transform artifacts --------
    "polar": {"requires_calibration": True, "supported_bits": [1, 2, 3, 4]},
    "qjl": {"requires_calibration": True, "supported_bits": [1]},
    "spectral": {"requires_calibration": True, "supported_bits": [1, 2, 3, 4]},
    # --- VecInfer: trained codebook over keys and values ---------------------
    "vecinfer": {
        "requires_calibration": True,
        "has_metal_kernel": True,
        "compresses_keys": True,
        "compresses_values": True,
        "supported_bits": None,
    },
    # --- Group / channel quantization ----------------------------------------
    "kivi": {"supported_bits": [1, 2, 3, 4], "has_metal_kernel": True},
    "kivi_sink": {"supported_bits": [1, 2, 3, 4], "has_metal_kernel": True},
    "kitty": {"supported_bits": [2, 4]},
    "adakv": {"supported_bits": [2, 3, 4]},
    "kvquant": {"supported_bits": [2, 3, 4]},
    "cachegen": {"supported_bits": [2, 4, 8]},
    "nsnquant": {"supported_bits": [1, 2, 3, 4]},
    # --- Low-rank / latent methods: need offline fits on calibration tokens ---
    "svdq": {"requires_calibration": True, "supported_bits": [1, 2, 3, 4, 8]},
    "palu": {
        "requires_calibration": True,
        "compresses_values": True,
        "supported_bits": [2, 4],
    },
    # --- Cross-layer sharing -------------------------------------------------
    "xquant": {"compresses_values": True, "uses_merging": True, "supported_bits": [2, 4]},
    "minicache": {"uses_merging": True},
    # --- GEAR: quantization + low-rank residual over keys AND values ---------
    "gear": {"compresses_values": True, "supported_bits": [2, 4]},
    # --- Transform coding ----------------------------------------------------
    "kvtc": {"compresses_values": True},
    "nestedkv": {"compresses_values": True, "supported_bits": [1, 2, 3, 4]},
    # --- Hybrid: compression plus token selection ----------------------------
    "zipcache": {"uses_eviction": True, "supported_bits": [2, 4]},
    "skvq": {"supported_bits": [1, 2, 3, 4], "supports_streaming": True},
    "a2ats": {"supported_bits": [1, 2, 3, 4]},
    "anchorkv": {"supported_bits": [1, 2, 3, 4]},
    # --- Eviction methods with Metal backends --------------------------------
    "snapkv": {"has_metal_kernel": True},
    "h2o": {"has_metal_kernel": True},
    "tova": {"has_metal_kernel": True},
    "pyramidkv": {"has_metal_kernel": True},
    "qfilters": {"has_metal_kernel": True},
    # --- Eviction that merges state back instead of discarding it ------------
    "cam": {"uses_merging": True},
}

#: Methods whose config-field prefix doesn't match the method name itself
#: (e.g. ``snapkv``'s fields are ``snap_*``, not ``snapkv_*``). Every other
#: method's fields follow a plain ``{method}_*`` prefix — verified against
#: every field name in ``KVCacheConfig`` — so this map only needs the
#: exceptions, not a full method -> prefix table.
_FIELD_PREFIX_ALIAS: dict[str, str] = {
    "snapkv": "snap",
    "streaming_llm": "stream",
    "pyramidkv": "pyramid",
    "nsnquant": "nsn",
}

_TIER_CACHE: dict[str, ServeTier] = {}
_UNSUPPORTED_REASON: dict[str, str] = {}


#: Human-readable hints for knobs whose names don't explain themselves.
_FIELD_HELP: dict[str, str] = {
    "bit_width_inlier": "Bits per element for the main quantizer.",
    "seed": "Random seed for rotations / sketches.",
    "jl_dim": "Johnson-Lindenstrauss projection dimension.",
    "kivi_group_size": "Tokens per min/max quantization group.",
    "n_sink_tokens": "Number of early attention-sink tokens kept in fp16, never quantized.",
    "svdq_rank": "Latent rank; blank uses the energy threshold instead.",
    "svdq_energy_threshold": "Fraction of singular-value energy to retain.",
    "palu_rank": "Latent rank; blank uses the energy threshold instead.",
    "palu_energy_threshold": "Fraction of singular-value energy to retain.",
    "palu_n_head_groups": "Number of head groups sharing a low-rank projection.",
    "palu_hi_bit": "Mixed-bit: bits for the top latent channels.",
    "palu_lo_bit": "Mixed-bit: bits for the remaining latent channels.",
    "palu_hi_fraction": "Fraction of latent channels kept at the higher bit-width.",
    "palu_group_size": "Tokens per latent quantization group.",
    "palu_quantize_values": "Also mixed-bit quantize values (off keeps value latents at fp16).",
    "adakv_target_avg_bits": "Global average bits/element budget.",
    "kvquant_outlier_fraction": "Top-magnitude fraction kept in fp16.",
    "residual_length": "Recent tokens kept uncompressed.",
    "rocketkv_compression_ratio": "Overall target ratio; adaptively split across both stages.",
    "xquant_residual_bits": (
        "Bits for the reuse layer's residual vs. the anchor's codes. Default 4: "
        "0 (pure reuse) assumes adjacent layers are highly correlated, which often "
        "does not hold on real models and can produce incoherent output even at "
        "high base_bits — only drop below 4 if you've confirmed strong cross-layer "
        "correlation for your model (see VeloxQuant-MLX#380)."
    ),
}


def describe_field(name: str) -> dict[str, Any]:
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
    # KVCacheConfig fields use PEP 604 `int | None` syntax, which resolves to
    # types.UnionType, not typing.Union — both must be checked, or every
    # Optional field here (25 of them) silently falls through as "unknown".
    if origin is Union or origin is types.UnionType:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        optional = len(args) != len(typing.get_args(annotation))
        annotation = args[0] if args else annotation
        origin = typing.get_origin(annotation)

    # A parameterized tuple (`tuple[int, ...]`, e.g. svdq_bit_schedule)
    # reduces to `tuple` via get_origin; a bare `tuple` annotation (e.g.
    # kvtc_bit_choices) already *is* `tuple`, so origin is None and the
    # annotation itself is checked directly below.
    if origin is tuple:
        kind = "array"
    else:
        kind = {int: "int", float: "float", bool: "bool", str: "str", tuple: "array"}.get(
            annotation, "unknown"
        )

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


def field_is_relevant(method: str, name: str) -> bool:
    """Whether ``KVCacheConfig`` field ``name`` has any effect for ``method``.

    ``KVCacheConfig`` is one flat dataclass covering all methods, so nothing
    stops constructing e.g. ``KVCacheConfig(method="h2o", kivi_group_size=64)``
    — it succeeds and silently ignores the unrelated field. This gives callers
    (``veloxquant serve --set``, the control panel) a way to catch that before
    it looks like a no-op configuration change (issue #345).

    A field not tied to any single method (``_GENERIC_FIELDS``, plus any name
    with no ``_`` at all, e.g. ``capacity``) is always considered relevant.
    Otherwise: methods listed in ``_CONFIG_FIELDS`` use that explicit list: it
    is curated for the control panel (#35) and may be narrower than the
    prefix (e.g. a shared field intentionally left off one method's list).
    Methods not yet in ``_CONFIG_FIELDS`` fall back to a name-prefix check
    (via ``_FIELD_PREFIX_ALIAS`` for the few methods whose fields don't share
    the method's own name) rather than rejecting the field outright, since
    most methods have not had their exact field list curated yet.
    """
    if name in _GENERIC_FIELDS or "_" not in name:
        return True

    if method in _CONFIG_FIELDS:
        return name in _CONFIG_FIELDS[method]

    prefix = _FIELD_PREFIX_ALIAS.get(method, method)
    return name.startswith(prefix + "_")


def _default_config_fields(method: str) -> list[str]:
    """Fallback ``config_fields`` for a method not yet curated in
    ``_CONFIG_FIELDS``.

    Prior to this, an uncurated method fell back to ``_GENERIC_FIELDS`` only
    (``bit_width_inlier``, ``seed``) — silently hiding every method-specific
    knob from any UI built on ``field_schema`` (the macOS app's parameter
    editor), even though ``field_is_relevant`` already promises a name-prefix
    fallback for exactly this case. Scans ``KVCacheConfig``'s real fields by
    the same prefix rule ``field_is_relevant`` uses, so the two stay
    consistent: a field this accepts as relevant now also appears in the
    schema, and vice versa.

    Sorted for a deterministic, reviewable order — an explicit
    ``_CONFIG_FIELDS`` entry (hand-ordered by conceptual grouping) still wins
    once a method is curated, same as today.
    """
    from veloxquant_mlx.cache import base as _base
    from veloxquant_mlx.cache.base import KVCacheConfig

    hints = typing.get_type_hints(KVCacheConfig, globalns=vars(_base))
    prefix = _FIELD_PREFIX_ALIAS.get(method, method)
    method_fields = sorted(
        name for name in hints if name != "method" and name.startswith(prefix + "_")
    )
    return list(_GENERIC_FIELDS) + method_fields


def _capabilities_for(method: str) -> StrategyCapabilities:
    """Build the :class:`StrategyCapabilities` for one method.

    Family-derived defaults (quantization compresses keys, eviction bounds
    memory, hybrid merges the two) are overridden by the curated
    ``_CAPABILITIES`` table, and the tunable-parameter map is derived from the
    same ``KVCacheConfig`` field rules the rest of the registry uses
    (``_CONFIG_FIELDS``, then the name-prefix fallback), so knob lists cannot
    drift from what the config actually accepts.
    """
    family = _FAMILY.get(method, MethodFamily.QUANTIZATION)
    caps: dict[str, Any] = {
        "supported_bits": None,
        "requires_calibration": False,
        "supports_gqa": True,
        "supports_mqa": True,
        "supports_mha": True,
        "supports_prefill": True,
        "supports_decode": True,
        "supports_streaming": family is MethodFamily.EVICTION,
        "has_metal_kernel": False,
        "compresses_keys": family is not MethodFamily.EVICTION,
        "compresses_values": False,
        "uses_eviction": family is MethodFamily.EVICTION,
        "uses_merging": False,
    }
    caps.update(_CAPABILITIES.get(method, {}))

    tunable: dict[str, str] = {}
    for field_name in _CONFIG_FIELDS.get(method) or _default_config_fields(method):
        tunable[field_name] = describe_field(field_name)["type"]
    caps["tunable_parameters"] = tunable
    return StrategyCapabilities(**caps)


def all_method_names() -> list[str]:
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


def probe_serve_tier(method: str) -> ServeTier:
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


def _run_probe(method: str) -> tuple[ServeTier, str | None]:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache as _MLXKVCache
    from mlx_lm.models.cache import can_trim_prompt_cache

    from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory, MethodName

    try:
        # method always originates from all_method_names(), which reads this
        # same Literal back via reflection (see its docstring) — mypy can't
        # see that connection through the reflection, but the invariant holds.
        cache = KVCacheFactory.create(
            KVCacheConfig(
                method=cast(MethodName, method), head_dim=128, bit_width_inlier=2, seed=42
            )
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


_COVERAGE_CACHE: dict[str, TelemetryCoverage] = {}


def telemetry_coverage(method: str) -> TelemetryCoverage:
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
        from mlx_lm.models.cache import KVCache as _MLXKVCache

        from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheFactory, MethodName

        # method always originates from all_method_names() — see the
        # analogous comment in _run_probe.
        cache = KVCacheFactory.create(
            KVCacheConfig(
                method=cast(MethodName, method), head_dim=128, bit_width_inlier=2, seed=42
            )
        )
        # Callers only reach here when tier.is_servable, which _run_probe
        # already established means cache subclasses mlx_lm's KVCache (see its
        # isinstance check) — standalone methods report CRASHES there instead.
        if not isinstance(cache, _MLXKVCache):
            return TelemetryCoverage.NONE
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
        config_fields=_CONFIG_FIELDS.get(name) or _default_config_fields(name),
        paper_deviation=_PAPER_DEVIATION.get(name),
        unsupported_reason=_UNSUPPORTED_REASON.get(name),
        # Only meaningful for methods that actually run; a crash-tier cache
        # never reports anything.
        coverage=(telemetry_coverage(name) if tier.is_servable else TelemetryCoverage.NONE),
        capabilities=_capabilities_for(name),
    )


def static_method_info(name: str) -> MethodInfo:
    """Build a :class:`MethodInfo` without exercising a real cache.

    Used by the auto-selection pipeline's *ranking* pass: it needs all 43
    methods' capabilities immediately, and probing each one (exercising an
    ``mlx_lm`` cache with real tensors) costs on the order of a second per
    method. Serve tier is reported optimistically as ``HONEST_BYTES`` and
    probing is deferred until the caller *commits* to a method
    (``get_method`` / :func:`probe_serve_tier`) — so a stale tier can never
    silently disappear: the crash surface is real tensors, not this table.
    """
    if name not in all_method_names():
        raise KeyError(
            f"unknown method {name!r}. Known methods: {', '.join(all_method_names())}"
        )
    return MethodInfo(
        name=name,
        family=_FAMILY.get(name, MethodFamily.QUANTIZATION),
        serve_tier=ServeTier.HONEST_BYTES,
        blurb=_BLURB.get(name, f"{name} KV-cache method."),
        config_fields=_CONFIG_FIELDS.get(name) or _default_config_fields(name),
        paper_deviation=_PAPER_DEVIATION.get(name),
        unsupported_reason=None,
        coverage=TelemetryCoverage.NONE,
        capabilities=_capabilities_for(name),
    )


def list_methods(
    *,
    servable_only: bool = False,
    family: MethodFamily | None = None,
) -> list[MethodInfo]:
    """All methods, optionally filtered. Sorted servable-first, then by name."""
    infos = [get_method(n) for n in all_method_names()]

    if servable_only:
        infos = [i for i in infos if i.serve_tier.is_servable]
    if family is not None:
        infos = [i for i in infos if i.family is family]

    return sorted(infos, key=lambda i: (not i.serve_tier.is_servable, i.name))
