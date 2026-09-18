"""Per-method option dataclasses for :class:`~veloxquant_mlx.cache.base.KVCacheConfig`.

``KVCacheConfig`` grew one flat dataclass field per quantization/eviction
method's hyperparameter (207 fields across 37 method families as of this
module's introduction) rather than one option object per method (#420). This
module groups each method's already-adjacent fields into its own small
dataclass so a caller configuring, say, H2O only sees H2O's five fields
instead of all 207.

Usage::

    from veloxquant_mlx.cache.options import H2OOptions

    config = KVCacheConfig(method="h2o", options=H2OOptions(budget=1024))

Passing ``options`` populates the matching flat fields on ``KVCacheConfig``
(here, ``h2o_budget``) for any flat field still at its dataclass default, so
every existing cache class -- which reads ``getattr(config, "h2o_budget",
512)`` directly -- keeps working unchanged; see ``KVCacheConfig.__post_init__``.
Setting a flat per-method field directly still works too (backwards
compatible) but now emits a ``DeprecationWarning`` naming the ``options``
replacement.

These classes are pure data containers with no behavior of their own --
field names, types, defaults, and comments are carried over verbatim from
the original flat fields on ``KVCacheConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class VecInferOptions:
    """VecInfer-specific configuration."""

    key_sub_dim: int = 4
    value_sub_dim: int = 8
    key_codebook_bits: int = 12
    value_codebook_bits: int = 8
    residual_length: int = 128


@dataclass
class KIVIOptions:
    """KIVI configuration (asymmetric group quantization)."""

    kivi_group_size: int = 32  # min/max group size (KIVI default 32)


@dataclass
class SVDqOptions:
    """SVDq configuration (sub-2-bit key compression via offline SVD)."""

    svdq_rank: int | None = None  # explicit rank; None → energy threshold
    svdq_energy_threshold: float = 0.95  # fraction of singular value energy to retain
    # 8-group per-group bit schedule (paper Eq. 6), most-significant group
    # first; 0 truncates that group entirely. Default is the paper's own
    # worked example, mean bit-width b̄ = 2.
    svdq_bit_schedule: tuple[int, ...] = (8, 4, 2, 1, 1, 0, 0, 0)
    svdq_group_size: int = 32  # group size for latent quantization


@dataclass
class KittyOptions:
    """Kitty configuration (dynamic channel-wise mixed-precision)."""

    kitty_hi_fraction: float = 0.25  # fraction of channels routed to hi_bit
    kitty_hi_bit: int = 4  # bits for high-variance channels
    kitty_lo_bit: int = 2  # bits for low-variance channels
    kitty_group_size: int = 32  # group size for channel quantization


@dataclass
class AdaKVOptions:
    """AdaKV-proxy configuration (per-head adaptive bit allocation)."""

    # NOTE: target must lie strictly inside (lo_bit, hi_bit) for per-head
    # adaptation to be possible at all — at either endpoint the budget forces a
    # uniform allocation (equivalent to plain KIVI). Default 2.5, not 2.0.
    adakv_target_avg_bits: float = 2.5  # global average bits/element target
    adakv_lo_bit: int = 2  # minimum bits any head can get
    adakv_mid_bit: int = 3  # middle tier (set == hi for a 2-tier set)
    adakv_hi_bit: int = 4  # maximum bits any head can get
    adakv_group_size: int = 32  # group size for per-head quantization
    adakv_update_interval: int = 1  # recompute allocation every N tokens (1 = every step)
    adakv_importance: str = "norm_variance"  # "norm_variance" | "attention_entropy"
    adakv_obs_window: int = 32  # obs-window size for attention_entropy


@dataclass
class XQuantOptions:
    """XQuant configuration (cross-layer KV cache reuse)."""

    xquant_group_size: int = 2  # layers per anchor/reuse group (2 = pairs)
    xquant_base_bits: int = 2  # anchor quantizer bit-width
    # Reuse-layer correction residual. 0 (pure reuse) assumes adjacent layers'
    # K/V are highly correlated, which measured ~0 (sometimes negative) cosine
    # similarity on real models -- pure reuse then reproduces incoherent output
    # even at near-lossless xquant_base_bits (VeloxQuant-MLX#380). 4 is the
    # validated floor (test_uncorrelated_residual_recovers): sufficient to
    # recover coherent generation without correlation, at a modest byte cost.
    xquant_residual_bits: int = 4
    xquant_group_quant_size: int = 32  # token group size for quantization
    xquant_max_ctx: int = 8192  # coordinator per-group token budget


@dataclass
class KVQuantOptions:
    """KVQuant-NUQ configuration (non-uniform datatype + outlier isolation)."""

    kvquant_bits: int = 3  # base NUQ bit-width
    kvquant_outlier_fraction: float = 0.01  # top-magnitude fraction kept fp16 (0 = pure NUQ)
    kvquant_group_size: int = 32  # group size for per-channel/per-token fitting
    kvquant_lloyd_iters: int = 8  # Lloyd-Max iterations for level fitting
    kvquant_refit_interval: int = 0  # refit levels every N decode steps (0 = freeze prefill)
    kvquant_n_sink: int = 1  # leading attention-sink tokens kept fp16 (paper §3.5; 0 = off)


@dataclass
class PALUOptions:
    """PALU configuration (true-latent low-rank K *and* V)."""

    palu_rank: int | None = None  # explicit latent rank; None → energy threshold
    palu_energy_threshold: float = 0.90  # singular-value energy to retain
    palu_n_head_groups: int = 4  # group-head low-rank: heads share a projection
    palu_hi_bit: int = 4  # mixed-bit: top latent channels
    palu_lo_bit: int = 2  # mixed-bit: remaining latent channels
    palu_hi_fraction: float = 0.25  # fraction of latent channels at hi_bit
    palu_group_size: int = 32  # token group size for latent quantization
    palu_quantize_values: bool = True  # low-rank + mixed-bit values too (False = LR-only)


@dataclass
class CacheGenOptions:
    """CacheGen configuration (entropy-coded byte model over group quant)."""

    cachegen_bits: int = 4  # base group-quant bit-width (shallowest layer group)
    cachegen_group_size: int = 32  # token group size
    cachegen_use_delta: bool = True  # token-delta transform before entropy coding
    cachegen_per_channel: bool = True  # group entropy estimate by channel (§5.1.3)
    cachegen_layer_groups: int = 3  # number of layer-depth groups for the bit schedule
    cachegen_resolved_bits: int | None = (
        None  # per-layer bit-width injected by for_model (None → uniform cachegen_bits)
    )


@dataclass
class MiniCacheOptions:
    """MiniCache configuration (cross-layer depth-dimension SLERP merge)."""

    minicache_start_frac: float = 0.5  # depth fraction below which layers are never merged
    minicache_group_size: int = 2  # layers per merge group (2 = pairs)
    minicache_retention_threshold: float = 0.9  # cosine below which a token pair is kept unmerged
    minicache_slerp_t: float = 0.5  # SLERP interpolation factor
    minicache_max_ctx: int = 8192  # coordinator per-group token budget


@dataclass
class GEAROptions:
    """GEAR configuration (error-feedback: residual low-rank + sparse outliers)."""

    gear_bits: int = 2  # ultra-low base bit-width
    gear_rank: int | None = None  # residual low-rank; None → energy threshold
    gear_energy_threshold: float = 0.90  # residual singular-value energy to retain
    gear_sparse_fraction: float = 0.01  # top-|residual| fraction kept exact (0 = pure low-rank)
    gear_group_size: int = 32  # base group-quant token group size
    gear_quantize_values: bool = True  # apply GEAR to values too (False = keys only)


@dataclass
class ZipCacheOptions:
    """ZipCache-adapted configuration (saliency-adaptive per-token mixed-precision)."""

    zipcache_hi_bits: int = 4  # bit-width for salient (high-norm) tokens
    zipcache_lo_bits: int = 2  # bit-width for non-salient tokens
    zipcache_hi_fraction: float = 0.20  # fraction of tokens routed to hi_bits
    zipcache_group_size: int = 32  # token group size for min/max quantization
    zipcache_quantize_values: bool = True  # apply mixed-precision to values too


@dataclass
class SnapKVOptions:
    """SnapKV-adapted configuration (prefill observation-window token eviction)."""

    snap_dtype: str = "auto"  # auto preserves BF16; float16 forces legacy storage
    snap_batched_scoring: bool = False  # experimental: may change near-tie scores
    snap_backend: str = "auto"  # auto | mlx | metal (experimental) | reference
    snap_budget: int = 512  # max tokens retained after prefill eviction
    snap_obs_window: int = 32  # trailing key rows used as proxy queries
    snap_n_sink: int = 4  # initial positions always kept (attention sinks)


@dataclass
class StreamingLLMOptions:
    """StreamingLLM-adapted configuration (sink + recency-window structural eviction)."""

    stream_n_sink: int = 4  # initial token positions frozen as attention sinks
    stream_window_size: int = 512  # FIFO capacity for recent tokens


@dataclass
class H2OOptions:
    """H2O-adapted configuration (cumulative attention-mass heavy-hitter eviction)."""

    h2o_budget: int = 512  # max tokens kept at any time (sinks + non-sinks)
    h2o_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    h2o_rope_base: float = 10000.0  # RoPE base for post-eviction position remap; match the model
    h2o_grace: int = 16  # most-recent tokens protected from eviction; fixes the early-token freeze
    h2o_decay: float = (
        0.98  # per-step multiplicative decay on existing scores; fixes score staleness
    )


@dataclass
class TOVAOptions:
    """TOVA-adapted configuration (current-step attention-weight eviction, memoryless)."""

    tova_budget: int = 512  # max tokens kept at any time (sinks + non-sinks)
    tova_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    tova_backend: str = "auto"  # auto | mlx | metal | reference (parity/benchmarking)


@dataclass
class PyramidKVOptions:
    """PyramidKV-adapted configuration (layer-adaptive budget attention-mass eviction)."""

    pyramid_budget: int = 512  # AVERAGE per-layer budget (uniform-H2O baseline)
    pyramid_backend: str = "reference"  # reference | mlx | metal | auto (MLX)
    pyramid_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    pyramid_beta: float = 2.0  # pyramid steepness: 1.0 = flat (== H2O), larger = steeper taper
    pyramid_resolved_budget: int | None = (
        None  # per-layer budget injected by for_model (None → uniform)
    )


@dataclass
class SqueezeAttentionOptions:
    """SqueezeAttention-adapted configuration (2D layer×token data-driven budget eviction)."""

    squeeze_budget: int = 512  # AVERAGE per-layer budget (uniform-H2O baseline)
    squeeze_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    squeeze_strength: float = (
        1.0  # reallocation strength: 0.0 = uniform (== H2O), 1.0 = full inverse-concentration
    )
    squeeze_resolved_budget: int | None = (
        None  # explicit per-layer budget override (None → coordinator supplies it)
    )


@dataclass
class ChunkKVOptions:
    """ChunkKV-adapted configuration (chunk-level / semantic-block eviction)."""

    chunkkv_budget: int = 512  # max tokens kept per layer (sinks included)
    chunkkv_chunk_size: int = 8  # eviction granularity C; 1 == H2O bit-for-bit
    chunkkv_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    chunkkv_score: str = (
        "attn_mass"  # chunk-importance proxy: "attn_mass" (H2O scorer) | "key_norm"
    )
    chunkkv_reuse_layers: int = 1  # Algorithm 2: layers per reuse block; 1 = disabled


@dataclass
class CaMOptions:
    """CaM-adapted configuration (cache merging — merge evicted tokens, not drop)."""

    cam_budget: int = 512  # max tokens kept per layer (sinks included)
    cam_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    cam_merge: str = (
        "sim_weighted"  # merge rule: "sim_weighted" | "mean" | "drop" (drop == H2O bit-for-bit)
    )
    cam_merge_keys: bool = False  # merge keys too (values are always merged)
    cam_merge_gate: bool = True  # Eq. 14 Bernoulli merge gate (False = unconditional merge, the paper's ablated config)


@dataclass
class XKVOptions:
    """xKV configuration (cross-layer shared-subspace key compression)."""

    xkv_group_size: int = 2  # layers per shared-subspace group (2 = pairs)
    xkv_rank: int | None = None  # explicit shared rank; None → energy threshold
    xkv_energy_threshold: float = 0.95  # fraction of singular value energy to retain
    xkv_latent_bits: int = 4  # single-bit-width latent quantization
    xkv_group_quant_size: int = 32  # token group size for latent quantization
    xkv_max_ctx: int = 8192  # coordinator per-group token budget


@dataclass
class NSNQuantOptions:
    """NSNQuant configuration (calibration-free universal-codebook VQ)."""

    nsn_bits: int = 2  # 2 = sign mask + index, 1 = index only
    nsn_residual_length: int = 64  # fp16 chunk buffer; paper suggests 128 for 1-bit
    nsn_codebook_size: int = 256  # centroids (256 → uint8 indices)
    nsn_subvector_dim: int = 8  # VQ subvector dimension (paper: 8)
    nsn_seed: int = 1234  # codebook RNG seed (synthetic Gaussian)
    nsn_max_ctx: int = 8192  # per-layer token budget


@dataclass
class L2NormOptions:
    """L2Norm-adapted configuration (intrinsic key-norm eviction)."""

    knorm_budget: int = 512  # max tokens kept (incl. sinks)
    knorm_n_sink: int = 4  # leading positions never evicted
    knorm_recent: int = 0  # trailing protected window (0 = paper-faithful)
    knorm_keep: str = "low"  # "low" = paper finding; "high" = inverted ablation


@dataclass
class SKVQOptions:
    """SKVQ-adapted configuration (sliding-window reorder + clip quant)."""

    skvq_bits_key: int = 2  # key code bit-width (paper: 2)
    skvq_bits_value: int = 2  # value code bit-width (paper: 1.5; we ship integer bits)
    skvq_group_size: int = 32  # channels per quant group (per-token groups)
    skvq_window: int = 128  # fp16 sliding window == flush chunk size (paper: ~128)
    skvq_n_sink: int = 5  # leading tokens restored to fp16 (paper's filter: ~5)
    skvq_reorder: bool = True  # channel reordering (False = identity ablation)
    skvq_clip_search: bool = True  # per-group clip-factor grid search at flush time
    skvq_clip_alpha: float = 1.0  # fixed clip factor when search is off
    skvq_max_ctx: int = 8192  # per-layer token budget


@dataclass
class QFiltersOptions:
    """Q-Filters-adapted configuration (query-agnostic projection eviction)."""

    qfilters_budget: int = 512  # max tokens kept (incl. sinks)
    qfilters_n_sink: int = 4  # leading positions never evicted
    qfilters_recent: int = 0  # trailing protected window (extension, off)
    qfilters_calib_tokens: int = 128  # tokens observed before the filter freezes
    qfilters_sign: int = 1  # +1 = paper direction; -1 = inverted ablation
    # Retention floor: below qfilters_budget / tokens_seen < this, generation
    # quality is unverified for the caller's prompt shape. A benchmark swept
    # across qfilters_budget on a real long-context prompt (see the
    # "qwen3-8b-qfilters-budget-sweep" post in docs-site/blog) found a sharp
    # coherence cliff between ~80% and ~92% retention, not a gradual slope —
    # everything from 23% to 80% retained produced the same fully-repeated,
    # non-language output. 0.9 sits just above where that run recovered
    # coherence; it is a warning threshold from one measured prompt/model,
    # not a proven universal safe point (see qfilters_on_low_retention below).
    qfilters_min_retention: float | None = 0.9
    # What to do the first time retention drops below qfilters_min_retention:
    #   "warn"   (default) — emit one warnings.warn(), keep running.
    #   "raise"  — raise ValueError instead of silently returning degraded text.
    #   "ignore" — no check at all (also skipped when qfilters_min_retention=None).
    qfilters_on_low_retention: Literal["warn", "raise", "ignore"] = "warn"


@dataclass
class KeyformerOptions:
    """Keyformer-adapted configuration (Gumbel-regularized eviction)."""

    keyformer_budget: int = 512  # max tokens kept (incl. sinks)
    keyformer_n_sink: int = 4  # leading positions never evicted
    keyformer_recent: int = 0  # trailing protected window (extension, off)
    keyformer_tau: float | None = (
        None  # constant-temperature alias; overrides tau_init/tau_end (disables annealing) if set
    )
    keyformer_tau_init: float = 1.0  # Gumbel temperature at pos=0 (paper default 1)
    keyformer_tau_end: float = 1.0  # Gumbel temperature once annealed (paper default 2)
    keyformer_anneal_steps: int = 0  # steps to ramp tau_init -> tau_end; 0 = constant temperature
    keyformer_rope_base: float = 10000.0  # RoPE base for post-eviction position remap
    keyformer_seed: int = 0  # base seed for the frozen per-position noise


@dataclass
class MorphKVOptions:
    """MorphKV-adapted configuration (recent-window correlation retention)."""

    morphkv_budget: int = 512  # max tokens kept (incl. sinks)
    morphkv_n_sink: int = 4  # leading positions never evicted
    morphkv_window: int = 8  # trailing recent-attention window; 1 = latest-token


@dataclass
class KVzipOptions:
    """KVzip-adapted configuration (context-reconstruction reliance eviction)."""

    kvzip_budget: int = 512  # max tokens kept (incl. sinks)
    kvzip_n_sink: int = 4  # leading positions never evicted
    kvzip_probe: str = "context"  # reconstruction probe; "latest" == TOVA-adapted latest-token


@dataclass
class KVTCOptions:
    """KVTC-adapted configuration (local PCA + DP-optimal bit allocation + entropy coding)."""

    kvtc_bit_budget: int = 512  # total bits per token across all components (K, V independently); default = 4 * head_dim(128)
    kvtc_bit_choices: tuple = (0, 1, 2, 3, 4, 6, 8)  # allowed per-component bit-widths (0 = drop)
    kvtc_beta: float = (
        3.5  # distortion decay constant D(v,b) = v * beta**(-b), shared with ratequant.py
    )


@dataclass
class CurDKVOptions:
    """CurDKV-adapted configuration (value-aware leverage-score eviction)."""

    curdkv_budget: int = 512  # max tokens kept at any time (sinks + non-sinks)
    curdkv_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    curdkv_rank_cap: int = 16  # SVD rank cap for leverage-score estimation
    curdkv_rope_base: float = 10000.0  # RoPE base for post-eviction position remap; match the model


@dataclass
class NestedKVOptions:
    """NestedKV-adapted configuration (multi-scale ensembled prefill eviction; no verified venue)."""

    nestedkv_budget: int = 512  # per-head-equivalent budget (total layer budget = this * n_heads)
    nestedkv_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    nestedkv_window: int = 64  # W, current-memory trailing window
    nestedkv_beta: float = 3.0  # head-adaptive blend temperature (paper Appendix A default)
    nestedkv_tau: float = 0.60  # surprise gate threshold (paper Appendix A default)
    nestedkv_kappa: float = 10.0  # surprise gate sharpness (paper Appendix A default)
    nestedkv_safeguard_alpha: float = (
        0.20  # per-head guaranteed-floor fraction (paper Appendix A default)
    )


@dataclass
class AMCOptions:
    """AMC-adapted configuration (saliency-driven tiered rank+precision; no verified venue)."""

    amc_k_high: float = 0.20  # top percentile -> High tier (rank/bits per Algorithm 1)
    amc_k_mid: float = 0.30  # next percentile -> Mid tier
    amc_use_query_saliency: bool = False  # Eq. 3 query-aware blend (off = pure magnitude, Eq. 1-2)
    amc_query_alpha: float = 0.5  # Eq. 3 balance coefficient (magnitude vs. query cosine)
    amc_adaptive_thresholds: bool = (
        False  # Eq. 4-5 sequence-adaptive closed-loop threshold adjustment
    )
    amc_threshold_window: int = 64  # trailing window size for closed-loop variance tracking
    amc_gamma: float = 0.1  # threshold attenuation scaling factor (Eq. 4-5)
    amc_calib_variance: float | None = (
        None  # offline calibration variance; required if amc_adaptive_thresholds=True
    )
    amc_group_size: int = 32  # token-axis group size for tier quantization


@dataclass
class A2ATSOptions:
    """A2ATS-adapted configuration (windowed RoPE + query-aware VQ)."""

    a2ats_codebook_bits: int = 8  # codebook size 2^bits
    a2ats_sub_dim: int = 8  # VQ sub-vector width
    a2ats_window: int = 128  # trailing exact-RoPE window w (positions)
    # constant far-token relative position b (Eq. 11); independent of w
    a2ats_b: int = 2048
    a2ats_use_query_aware: bool = True  # paper's primary reported path (default ON)
    a2ats_beta: float = 0.5  # query/reconstruction blend, in [0, 1]
    a2ats_retrieval_fraction: float = 0.20  # fraction of tokens routed to query-aware assignment
    a2ats_rope_base: float = 10000.0  # RoPE frequency base
    # [sub_dim, sub_dim] query second-moment H (Eq. 10); enables the paper's Eq. 14 assignment
    a2ats_query_h: Any = None
    a2ats_codebook: Any = None  # mx.array | np.ndarray | None (random init if absent)


@dataclass
class AnchorKVOptions:
    """AnchorKV-adapted configuration (anchor-residual compression, no eviction; no verified venue)."""

    anchorkv_theta: float = (
        0.05  # fraction of the uncompressed fp16 cache to retain (paper's single knob)
    )
    anchorkv_window: int = 32  # trailing positions always anchors + proxy observation queries
    anchorkv_rho: float = 0.7  # fraction of non-window anchor budget filled by attention score
    anchorkv_anchor_frac: float = (
        1.0 / 128.0
    )  # anchor budget k as a fraction of context length (paper: S/128)
    anchorkv_residual_bits: int = 2  # bits/coordinate for stored residuals
    anchorkv_seed: int = 42  # RNG seed for uniform anchor sampling + residual codec rotation


@dataclass
class RocketKVOptions:
    """RocketKV-adapted configuration (two-stage: SnapKV eviction + hybrid sparse attention)."""

    rocketkv_compression_ratio: float = (
        8.0  # overall target ratio c; drives the adaptive split (paper §3.6)
    )
    rocketkv_page_size: int | None = None  # HSA page size; None derives it from the adaptive split
    rocketkv_head_topk1: int | None = (
        None  # HSA head-dim channels kept; None derives it from the split
    )
    rocketkv_obs_window: int = 32  # stage-1 SnapKV observation window
    rocketkv_n_sink: int = 4  # stage-1 SnapKV sink tokens always kept


@dataclass
class AgeTieredKVOptions:
    """AgeTieredKV configuration (position/age-gated 3-tier precision, issue #256)."""

    age_recent_boundary: int = 128  # age < this -> RECENT tier (age_bits_recent)
    age_mid_boundary: int = 1024  # age < this (and >= age_recent_boundary) -> MID tier
    age_bits_recent: int = 8  # bit-width for the newest tokens
    age_bits_mid: int = 4  # bit-width for mid-age tokens
    age_bits_old: int = 2  # bit-width for tokens older than age_mid_boundary
    age_group_size: int = 32  # token-axis group size for the shared min/max quantizer


@dataclass
class SpectralQuantOptions:
    """SpectralQuant configuration (data-aware, calibration-based)."""

    spectral_key_d_eff: int = 4  # signal dimensions for keys (paper: ~4)
    spectral_val_d_eff: int = 50  # signal dimensions for values (paper: ~50)
    spectral_apply_qjl: bool = True  # apply QJL on signal dims only
    spectral_model_name: str = "model"  # identifier for rotation cache on disk
