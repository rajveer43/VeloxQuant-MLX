from __future__ import annotations

import math
from dataclasses import dataclass, field
from dataclasses import replace as dataclasses_replace
from typing import Any, Literal, Optional, Union

from veloxquant_mlx.core.abstractions import ArtifactStore, KVCache, QuantizationObserver
from veloxquant_mlx.core.exceptions import QuantizerConfigError

# Methods whose cache class implements VeloxQuant's own KVCache ABC
# (append_key/append_value/attend/memory_bytes) instead of subclassing
# mlx_lm.models.cache.KVCache (update_and_fetch/nbytes/state/trim/merge/
# meta_state). mlx_lm.generate() drives caches purely through the latter
# interface via model.make_cache() -> cache.update_and_fetch(...), so an
# object built from one of these methods does not satisfy that contract and
# either crashes or silently produces wrong output deep inside generation
# instead of failing at config/patch time. Verified against the codebase in
# https://github.com/rajveer43/VeloxQuant-MLX/issues/27: these are the 5
# "standalone" methods among the 40 in the registry; all others subclass
# mlx_lm's KVCache and are safe to wire into patch_model_kv_cache /
# KVCacheBuilder.for_model for live mlx_lm serving.
STANDALONE_METHODS = frozenset(
    {
        "turboquant_prod",
        "turboquant_mse",
        "polar",
        "qjl",
        "spectral",
    }
)


@dataclass
class KVCacheConfig:
    """Configuration for a KVCache instance.

    Attributes:
        method: Quantisation algorithm.
        head_dim: Attention head dimension (d).
        bit_width_inlier: Bit-width for inlier channels. Either a single int
            applied uniformly across all layers, OR a ``list[int]`` of length
            ``n_layers`` for per-layer RateQuant-style allocation. When passed
            as a list, ``KVCacheBuilder.for_model()`` consumes element ``i``
            for layer ``i``; ``KVCacheFactory.create()`` requires an int.
        bit_width_outlier: Bit-width for outlier channels (None → same as inlier).
        jl_dim: JL projection dimension m.
        n_outlier_channels: Number of outlier channels to detect.
        seed: Random seed.
        dtype: MLX dtype for computations.
        capacity: Maximum tokens to store (None → unlimited).
        sliding_window: If set, wrap cache with sliding-window eviction.
        store: ArtifactStore to load precomputed artifacts from.
        observers: List of QuantizationObserver instances.
    """

    method: Literal[
        "turboquant_prod",
        "turboquant_mse",
        "turboquant_rvq",
        "polar",
        "qjl",
        "vecinfer",
        "spectral",
        "kivi",
        "kivi_sink",
        "svdq",
        "kitty",
        "adakv",
        "xquant",
        "kvquant",
        "palu",
        "cachegen",
        "minicache",
        "gear",
        "zipcache",
        "snapkv",
        "streaming_llm",
        "h2o",
        "tova",
        "pyramidkv",
        "squeeze",
        "chunkkv",
        "cam",
        "xkv",
        "nsnquant",
        "knorm",
        "skvq",
        "qfilters",
        "keyformer",
        "morphkv",
        "kvzip",
        "kvtc",
        "curdkv",
        "nestedkv",
        "amc",
        "a2ats",
        "anchorkv",
        "rocketkv",
        "age_tiered",
    ] = "turboquant_rvq"
    head_dim: int = 128
    bit_width_inlier: Union[int, list] = 2
    bit_width_outlier: Optional[int] = None
    jl_dim: Optional[int] = None
    n_outlier_channels: Optional[int] = None
    n_calib_tokens: Optional[int] = None
    enable_vectorized_attend: bool = True
    enable_outlier_two_stream: bool = False
    enable_fused_query_dot: bool = False
    seed: int = 42
    dtype: Any = None
    capacity: Optional[int] = None
    sliding_window: Optional[int] = None
    store: Optional[ArtifactStore] = None
    observers: list = field(default_factory=list)
    # --- VecInfer-specific configuration -------------------------------
    key_sub_dim: int = 4
    value_sub_dim: int = 8
    key_codebook_bits: int = 12
    value_codebook_bits: int = 8
    residual_length: int = 128
    # --- KIVI configuration (asymmetric group quantization) ------------
    kivi_group_size: int = 32  # min/max group size (KIVI default 32)
    # --- SVDq configuration (sub-2-bit key compression via offline SVD) --
    svdq_rank: Optional[int] = None  # explicit rank; None → energy threshold
    svdq_energy_threshold: float = 0.95  # fraction of singular value energy to retain
    # 8-group per-group bit schedule (paper Eq. 6), most-significant group
    # first; 0 truncates that group entirely. Default is the paper's own
    # worked example, mean bit-width b̄ = 2.
    svdq_bit_schedule: tuple[int, ...] = (8, 4, 2, 1, 1, 0, 0, 0)
    svdq_group_size: int = 32  # group size for latent quantization
    # --- Kitty configuration (dynamic channel-wise mixed-precision) ------
    kitty_hi_fraction: float = 0.25  # fraction of channels routed to hi_bit
    kitty_hi_bit: int = 4  # bits for high-variance channels
    kitty_lo_bit: int = 2  # bits for low-variance channels
    kitty_group_size: int = 32  # group size for channel quantization
    # --- AdaKV-proxy configuration (per-head adaptive bit allocation) ----
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
    # --- XQuant configuration (cross-layer KV cache reuse) ---------------
    xquant_group_size: int = 2  # layers per anchor/reuse group (2 = pairs)
    xquant_base_bits: int = 2  # anchor quantizer bit-width
    xquant_residual_bits: int = 0  # reuse-layer correction residual (0 = pure reuse)
    xquant_group_quant_size: int = 32  # token group size for quantization
    xquant_max_ctx: int = 8192  # coordinator per-group token budget
    # --- KVQuant-NUQ configuration (non-uniform datatype + outlier isolation) -
    kvquant_bits: int = 3  # base NUQ bit-width
    kvquant_outlier_fraction: float = 0.01  # top-magnitude fraction kept fp16 (0 = pure NUQ)
    kvquant_group_size: int = 32  # group size for per-channel/per-token fitting
    kvquant_lloyd_iters: int = 8  # Lloyd-Max iterations for level fitting
    kvquant_refit_interval: int = 0  # refit levels every N decode steps (0 = freeze prefill)
    kvquant_n_sink: int = 1  # leading attention-sink tokens kept fp16 (paper §3.5; 0 = off)
    # --- PALU configuration (true-latent low-rank K *and* V) -------------
    palu_rank: Optional[int] = None  # explicit latent rank; None → energy threshold
    palu_energy_threshold: float = 0.90  # singular-value energy to retain
    palu_n_head_groups: int = 4  # group-head low-rank: heads share a projection
    palu_hi_bit: int = 4  # mixed-bit: top latent channels
    palu_lo_bit: int = 2  # mixed-bit: remaining latent channels
    palu_hi_fraction: float = 0.25  # fraction of latent channels at hi_bit
    palu_group_size: int = 32  # token group size for latent quantization
    palu_quantize_values: bool = True  # low-rank + mixed-bit values too (False = LR-only)
    # --- CacheGen configuration (entropy-coded byte model over group quant) ----
    cachegen_bits: int = 4  # base group-quant bit-width (shallowest layer group)
    cachegen_group_size: int = 32  # token group size
    cachegen_use_delta: bool = True  # token-delta transform before entropy coding
    cachegen_per_channel: bool = True  # group entropy estimate by channel (§5.1.3)
    cachegen_layer_groups: int = 3  # number of layer-depth groups for the bit schedule
    cachegen_resolved_bits: Optional[int] = (
        None  # per-layer bit-width injected by for_model (None → uniform cachegen_bits)
    )
    # --- MiniCache configuration (cross-layer depth-dimension SLERP merge) -----
    minicache_start_frac: float = 0.5  # depth fraction below which layers are never merged
    minicache_group_size: int = 2  # layers per merge group (2 = pairs)
    minicache_retention_threshold: float = 0.9  # cosine below which a token pair is kept unmerged
    minicache_slerp_t: float = 0.5  # SLERP interpolation factor
    minicache_max_ctx: int = 8192  # coordinator per-group token budget
    # --- GEAR configuration (error-feedback: residual low-rank + sparse outliers) ---
    gear_bits: int = 2  # ultra-low base bit-width
    gear_rank: Optional[int] = None  # residual low-rank; None → energy threshold
    gear_energy_threshold: float = 0.90  # residual singular-value energy to retain
    gear_sparse_fraction: float = 0.01  # top-|residual| fraction kept exact (0 = pure low-rank)
    gear_group_size: int = 32  # base group-quant token group size
    gear_quantize_values: bool = True  # apply GEAR to values too (False = keys only)
    # --- ZipCache-adapted configuration (saliency-adaptive per-token mixed-precision) ---
    zipcache_hi_bits: int = 4  # bit-width for salient (high-norm) tokens
    zipcache_lo_bits: int = 2  # bit-width for non-salient tokens
    zipcache_hi_fraction: float = 0.20  # fraction of tokens routed to hi_bits
    zipcache_group_size: int = 32  # token group size for min/max quantization
    zipcache_quantize_values: bool = True  # apply mixed-precision to values too
    # --- SnapKV-adapted configuration (prefill observation-window token eviction) ---
    snap_budget: int = 512  # max tokens retained after prefill eviction
    snap_obs_window: int = 32  # trailing key rows used as proxy queries
    snap_n_sink: int = 4  # initial positions always kept (attention sinks)
    # --- StreamingLLM-adapted configuration (sink + recency-window structural eviction) ---
    stream_n_sink: int = 4  # initial token positions frozen as attention sinks
    stream_window_size: int = 512  # FIFO capacity for recent tokens
    # --- H2O-adapted configuration (cumulative attention-mass heavy-hitter eviction) ---
    h2o_budget: int = 512  # max tokens kept at any time (sinks + non-sinks)
    h2o_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    h2o_rope_base: float = 10000.0  # RoPE base for post-eviction position remap; match the model
    h2o_grace: int = 16  # most-recent tokens protected from eviction; fixes the early-token freeze
    h2o_decay: float = (
        0.98  # per-step multiplicative decay on existing scores; fixes score staleness
    )
    # --- TOVA-adapted configuration (current-step attention-weight eviction, memoryless) ---
    tova_budget: int = 512  # max tokens kept at any time (sinks + non-sinks)
    tova_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    # --- PyramidKV-adapted configuration (layer-adaptive budget attention-mass eviction) ---
    pyramid_budget: int = 512  # AVERAGE per-layer budget (uniform-H2O baseline)
    pyramid_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    pyramid_beta: float = 2.0  # pyramid steepness: 1.0 = flat (== H2O), larger = steeper taper
    pyramid_resolved_budget: Optional[int] = (
        None  # per-layer budget injected by for_model (None → uniform)
    )
    # --- SqueezeAttention-adapted configuration (2D layer×token data-driven budget eviction) ---
    squeeze_budget: int = 512  # AVERAGE per-layer budget (uniform-H2O baseline)
    squeeze_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    squeeze_strength: float = (
        1.0  # reallocation strength: 0.0 = uniform (== H2O), 1.0 = full inverse-concentration
    )
    squeeze_resolved_budget: Optional[int] = (
        None  # explicit per-layer budget override (None → coordinator supplies it)
    )
    # --- ChunkKV-adapted configuration (chunk-level / semantic-block eviction) ---
    chunkkv_budget: int = 512  # max tokens kept per layer (sinks included)
    chunkkv_chunk_size: int = 8  # eviction granularity C; 1 == H2O bit-for-bit
    chunkkv_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    chunkkv_score: str = (
        "attn_mass"  # chunk-importance proxy: "attn_mass" (H2O scorer) | "key_norm"
    )
    chunkkv_reuse_layers: int = 1  # Algorithm 2: layers per reuse block; 1 = disabled
    # --- CaM-adapted configuration (cache merging — merge evicted tokens, not drop) ---
    cam_budget: int = 512  # max tokens kept per layer (sinks included)
    cam_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    cam_merge: str = (
        "sim_weighted"  # merge rule: "sim_weighted" | "mean" | "drop" (drop == H2O bit-for-bit)
    )
    cam_merge_keys: bool = False  # merge keys too (values are always merged)
    cam_merge_gate: bool = True  # Eq. 14 Bernoulli merge gate (False = unconditional merge, the paper's ablated config)
    # --- xKV configuration (cross-layer shared-subspace key compression) -
    xkv_group_size: int = 2  # layers per shared-subspace group (2 = pairs)
    xkv_rank: Optional[int] = None  # explicit shared rank; None → energy threshold
    xkv_energy_threshold: float = 0.95  # fraction of singular value energy to retain
    xkv_latent_bits: int = 4  # single-bit-width latent quantization
    xkv_group_quant_size: int = 32  # token group size for latent quantization
    xkv_max_ctx: int = 8192  # coordinator per-group token budget
    # --- NSNQuant configuration (calibration-free universal-codebook VQ) -
    nsn_bits: int = 2  # 2 = sign mask + index, 1 = index only
    nsn_residual_length: int = 64  # fp16 chunk buffer; paper suggests 128 for 1-bit
    nsn_codebook_size: int = 256  # centroids (256 → uint8 indices)
    nsn_subvector_dim: int = 8  # VQ subvector dimension (paper: 8)
    nsn_seed: int = 1234  # codebook RNG seed (synthetic Gaussian)
    nsn_max_ctx: int = 8192  # per-layer token budget
    # --- L2Norm-adapted configuration (intrinsic key-norm eviction) ------
    knorm_budget: int = 512  # max tokens kept (incl. sinks)
    knorm_n_sink: int = 4  # leading positions never evicted
    knorm_recent: int = 0  # trailing protected window (0 = paper-faithful)
    knorm_keep: str = "low"  # "low" = paper finding; "high" = inverted ablation
    # --- SKVQ-adapted configuration (sliding-window reorder + clip quant) -
    skvq_bits_key: int = 2  # key code bit-width (paper: 2)
    skvq_bits_value: int = 2  # value code bit-width (paper: 1.5; we ship integer bits)
    skvq_group_size: int = 32  # channels per quant group (per-token groups)
    skvq_window: int = 128  # fp16 sliding window == flush chunk size (paper: ~128)
    skvq_n_sink: int = 5  # leading tokens restored to fp16 (paper's filter: ~5)
    skvq_reorder: bool = True  # channel reordering (False = identity ablation)
    skvq_clip_search: bool = True  # per-group clip-factor grid search at flush time
    skvq_clip_alpha: float = 1.0  # fixed clip factor when search is off
    skvq_max_ctx: int = 8192  # per-layer token budget
    # --- Q-Filters-adapted configuration (query-agnostic projection eviction) -
    qfilters_budget: int = 512  # max tokens kept (incl. sinks)
    qfilters_n_sink: int = 4  # leading positions never evicted
    qfilters_recent: int = 0  # trailing protected window (extension, off)
    qfilters_calib_tokens: int = 128  # tokens observed before the filter freezes
    qfilters_sign: int = 1  # +1 = paper direction; -1 = inverted ablation
    # --- Keyformer-adapted configuration (Gumbel-regularized eviction) --
    keyformer_budget: int = 512  # max tokens kept (incl. sinks)
    keyformer_n_sink: int = 4  # leading positions never evicted
    keyformer_recent: int = 0  # trailing protected window (extension, off)
    keyformer_tau: Optional[float] = (
        None  # constant-temperature alias; overrides tau_init/tau_end (disables annealing) if set
    )
    keyformer_tau_init: float = 1.0  # Gumbel temperature at pos=0 (paper default 1)
    keyformer_tau_end: float = 1.0  # Gumbel temperature once annealed (paper default 2)
    keyformer_anneal_steps: int = 0  # steps to ramp tau_init -> tau_end; 0 = constant temperature
    keyformer_rope_base: float = 10000.0  # RoPE base for post-eviction position remap
    keyformer_seed: int = 0  # base seed for the frozen per-position noise
    # --- MorphKV-adapted configuration (recent-window correlation retention) --
    morphkv_budget: int = 512  # max tokens kept (incl. sinks)
    morphkv_n_sink: int = 4  # leading positions never evicted
    morphkv_window: int = 8  # trailing recent-attention window; 1 = latest-token
    # --- KVzip-adapted configuration (context-reconstruction reliance eviction) --
    kvzip_budget: int = 512  # max tokens kept (incl. sinks)
    kvzip_n_sink: int = 4  # leading positions never evicted
    kvzip_probe: str = "context"  # reconstruction probe; "latest" == TOVA-adapted latest-token
    # --- KVTC-adapted configuration (local PCA + DP-optimal bit allocation + entropy coding) --
    kvtc_bit_budget: int = 512  # total bits per token across all components (K, V independently); default = 4 * head_dim(128)
    kvtc_bit_choices: tuple = (0, 1, 2, 3, 4, 6, 8)  # allowed per-component bit-widths (0 = drop)
    kvtc_beta: float = (
        3.5  # distortion decay constant D(v,b) = v * beta**(-b), shared with ratequant.py
    )
    # --- CurDKV-adapted configuration (value-aware leverage-score eviction) --
    curdkv_budget: int = 512  # max tokens kept at any time (sinks + non-sinks)
    curdkv_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    curdkv_rank_cap: int = 16  # SVD rank cap for leverage-score estimation
    curdkv_rope_base: float = 10000.0  # RoPE base for post-eviction position remap; match the model
    # --- NestedKV-adapted configuration (multi-scale ensembled prefill eviction; no verified venue) --
    nestedkv_budget: int = 512  # per-head-equivalent budget (total layer budget = this * n_heads)
    nestedkv_n_sink: int = 4  # initial positions protected from eviction (attention sinks)
    nestedkv_window: int = 64  # W, current-memory trailing window
    nestedkv_beta: float = 3.0  # head-adaptive blend temperature (paper Appendix A default)
    nestedkv_tau: float = 0.60  # surprise gate threshold (paper Appendix A default)
    nestedkv_kappa: float = 10.0  # surprise gate sharpness (paper Appendix A default)
    nestedkv_safeguard_alpha: float = (
        0.20  # per-head guaranteed-floor fraction (paper Appendix A default)
    )
    # --- AMC-adapted configuration (saliency-driven tiered rank+precision; no verified venue) --
    amc_k_high: float = 0.20  # top percentile -> High tier (rank/bits per Algorithm 1)
    amc_k_mid: float = 0.30  # next percentile -> Mid tier
    amc_use_query_saliency: bool = False  # Eq. 3 query-aware blend (off = pure magnitude, Eq. 1-2)
    amc_query_alpha: float = 0.5  # Eq. 3 balance coefficient (magnitude vs. query cosine)
    amc_adaptive_thresholds: bool = (
        False  # Eq. 4-5 sequence-adaptive closed-loop threshold adjustment
    )
    amc_threshold_window: int = 64  # trailing window size for closed-loop variance tracking
    amc_gamma: float = 0.1  # threshold attenuation scaling factor (Eq. 4-5)
    amc_calib_variance: Optional[float] = (
        None  # offline calibration variance; required if amc_adaptive_thresholds=True
    )
    amc_group_size: int = 32  # token-axis group size for tier quantization
    # --- A2ATS-adapted configuration (windowed RoPE + query-aware VQ) --
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
    # --- AnchorKV-adapted configuration (anchor-residual compression, no eviction; no verified venue) --
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
    # --- RocketKV-adapted configuration (two-stage: SnapKV eviction + hybrid sparse attention) --
    rocketkv_compression_ratio: float = (
        8.0  # overall target ratio c; drives the adaptive split (paper §3.6)
    )
    rocketkv_page_size: Optional[int] = (
        None  # HSA page size; None derives it from the adaptive split
    )
    rocketkv_head_topk1: Optional[int] = (
        None  # HSA head-dim channels kept; None derives it from the split
    )
    rocketkv_obs_window: int = 32  # stage-1 SnapKV observation window
    rocketkv_n_sink: int = 4  # stage-1 SnapKV sink tokens always kept
    # --- AgeTieredKV configuration (position/age-gated 3-tier precision, issue #256) --
    age_recent_boundary: int = 128  # age < this -> RECENT tier (age_bits_recent)
    age_mid_boundary: int = 1024  # age < this (and >= age_recent_boundary) -> MID tier
    age_bits_recent: int = 8  # bit-width for the newest tokens
    age_bits_mid: int = 4  # bit-width for mid-age tokens
    age_bits_old: int = 2  # bit-width for tokens older than age_mid_boundary
    age_group_size: int = 32  # token-axis group size for the shared min/max quantizer
    # --- KVSink-adapted sink protection (method="kivi_sink") -----------
    n_sink_tokens: int = 5  # top-k high-key-norm tokens kept fp16
    smooth_factors: Any = None  # mx.array | np.ndarray | None
    key_codebook: Any = None  # mx.array | np.ndarray | None
    value_codebook: Any = None  # mx.array | np.ndarray | None
    # --- SpectralQuant configuration (data-aware, calibration-based) ----
    spectral_key_d_eff: int = 4  # signal dimensions for keys (paper: ~4)
    spectral_val_d_eff: int = 50  # signal dimensions for values (paper: ~50)
    spectral_apply_qjl: bool = True  # apply QJL on signal dims only
    spectral_model_name: str = "model"  # identifier for rotation cache on disk
    # --- Metal kernel acceleration (Phase 1, 0.5.1+) -------------------
    # Three-state flag for VecInfer quantize/dequant Metal fast path:
    #   None  → auto-detect (use Metal if available, fall back silently)
    #   True  → require Metal; raise at cache-construction time if missing
    #   False → force pure-MLX path (debug / parity testing)
    use_metal_kernels: Optional[bool] = None
    # --- Fused dequant+SDPA Metal kernel (Phase 2, 0.6.0+) -------------
    # When True (or auto-detected at None), the cache stores K/V as
    # codebook indices only and exposes a fused_sdpa() method that
    # mlx_lm's dispatcher (after patch_mlx_lm_for_fused_sdpa()) routes
    # attention to.  Avoids materializing the fp16 K_hat tensor entirely.
    #   None  → False today (opt-in; will flip to auto-detect later)
    #   True  → require, raise if Metal/shape unsupported
    #   False → run the standard dequant→SDPA path (current 0.5.x default)
    fused_sdpa: Optional[bool] = False
    # Pre-allocated index ring-buffer capacity (in tokens) when
    # fused_sdpa=True.  At construction time we allocate
    # [B, H_kv, fused_sdpa_max_ctx, n_sub] uint32 once and slice-write
    # into it on each update_and_fetch — avoids O(S²) per-step concat.
    # If a generation exceeds this length, the cache raises RuntimeError.
    fused_sdpa_max_ctx: int = 8192

    def __repr__(self) -> str:
        return (
            f"KVCacheConfig(method={self.method!r}, d={self.head_dim}, "
            f"b={self.bit_width_inlier}, seed={self.seed})"
        )


class KVCacheFactory:
    """Factory for creating KVCache instances from a KVCacheConfig."""

    @staticmethod
    def create(config: KVCacheConfig) -> KVCache:
        """Instantiate a KVCache from the given configuration.

        Args:
            config: KVCacheConfig instance.

        Returns:
            Configured KVCache.
        """
        from veloxquant_mlx.cache.adakv_cache import AdaKVCache
        from veloxquant_mlx.cache.xquant_cache import XQuantKVCache
        from veloxquant_mlx.cache.kvquant_cache import KVQuantKVCache
        from veloxquant_mlx.cache.palu_cache import PALUKVCache
        from veloxquant_mlx.cache.cachegen_cache import CacheGenKVCache
        from veloxquant_mlx.cache.minicache_cache import MiniCacheKVCache
        from veloxquant_mlx.cache.gear_cache import GEARKVCache
        from veloxquant_mlx.cache.zipcache_cache import ZipCacheKVCache
        from veloxquant_mlx.cache.snapkv_cache import SnapKVKVCache
        from veloxquant_mlx.cache.streaming_llm_cache import StreamingLLMKVCache
        from veloxquant_mlx.cache.h2o_cache import H2OKVCache
        from veloxquant_mlx.cache.tova_cache import TOVAKVCache
        from veloxquant_mlx.cache.pyramidkv_cache import PyramidKVCache
        from veloxquant_mlx.cache.squeeze_cache import SqueezeAttentionCache
        from veloxquant_mlx.cache.chunkkv_cache import ChunkKVCache
        from veloxquant_mlx.cache.cam_cache import CaMKVCache
        from veloxquant_mlx.cache.xkv_cache import XKVCache
        from veloxquant_mlx.cache.nsnquant_cache import NSNQuantKVCache
        from veloxquant_mlx.cache.knorm_cache import L2NormKVCache
        from veloxquant_mlx.cache.skvq_cache import SKVQKVCache
        from veloxquant_mlx.cache.qfilters_cache import QFiltersKVCache
        from veloxquant_mlx.cache.keyformer_cache import KeyformerKVCache
        from veloxquant_mlx.cache.morphkv_cache import MorphKVKVCache
        from veloxquant_mlx.cache.kvzip_cache import KVzipKVCache
        from veloxquant_mlx.cache.kvtc_cache import KVTCKVCache
        from veloxquant_mlx.cache.curdkv_cache import CurDKVKVCache
        from veloxquant_mlx.cache.nestedkv_cache import NestedKVKVCache
        from veloxquant_mlx.cache.amc_cache import AMCKVCache
        from veloxquant_mlx.cache.a2ats_cache import A2ATSKVCache
        from veloxquant_mlx.cache.anchorkv_cache import AnchorKVKVCache
        from veloxquant_mlx.cache.rocketkv_cache import RocketKVKVCache
        from veloxquant_mlx.cache.age_tiered_cache import AgeTieredKVCache
        from veloxquant_mlx.cache.kitty_cache import KittyKVCache
        from veloxquant_mlx.cache.polar_cache import PolarQuantKVCache
        from veloxquant_mlx.cache.qjl_cache import QJLKVCache
        from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache
        from veloxquant_mlx.cache.kivi_cache import KIVIKVCache
        from veloxquant_mlx.cache.sink_cache import SinkProtectedKVCache
        from veloxquant_mlx.cache.spectral_cache import SpectralQuantKVCache
        from veloxquant_mlx.cache.svdq_cache import SVDqKVCache
        from veloxquant_mlx.cache.turboquant_cache import TurboQuantKVCache
        from veloxquant_mlx.cache.turboquant_rvq_cache import TurboQuantRVQKVCache
        from veloxquant_mlx.cache.vecinfer_cache import VecInferKVCache

        d = config.head_dim
        seed = config.seed
        b = config.bit_width_inlier
        if isinstance(b, list) and config.method != "vecinfer":
            raise QuantizerConfigError(
                "KVCacheFactory.create() requires bit_width_inlier to be a single int. "
                "List-form bit_width_inlier (per-layer allocation) is consumed by "
                "KVCacheBuilder.for_model(), which dispatches to create() once per layer."
            )
        m = config.jl_dim if config.jl_dim is not None else d
        store = config.store

        if config.method in ("turboquant_prod", "turboquant_mse"):
            cache: KVCache = TurboQuantKVCache(config)
        elif config.method == "turboquant_rvq":
            cache = TurboQuantRVQKVCache(config)
        elif config.method == "polar":
            cache = PolarQuantKVCache(config)
        elif config.method == "qjl":
            cache = QJLKVCache(config)
        elif config.method == "vecinfer":
            cache = VecInferKVCache(config)
        elif config.method == "spectral":
            cache = SpectralQuantKVCache(config)
        elif config.method == "kivi":
            cache = KIVIKVCache(config)
        elif config.method == "kivi_sink":
            cache = SinkProtectedKVCache(config)
        elif config.method == "svdq":
            cache = SVDqKVCache(config)
        elif config.method == "kitty":
            cache = KittyKVCache(config)
        elif config.method == "adakv":
            cache = AdaKVCache(config)
        elif config.method == "xquant":
            # Single-cache construction yields a degenerate (coordinator-less)
            # anchor. Cross-layer reuse requires KVCacheBuilder.for_model(), which
            # builds the shared XQuantCoordinator and assigns anchor/reuse roles.
            cache = XQuantKVCache(config)
        elif config.method == "kvquant":
            cache = KVQuantKVCache(config)
        elif config.method == "palu":
            cache = PALUKVCache(config)
        elif config.method == "cachegen":
            cache = CacheGenKVCache(config)
        elif config.method == "minicache":
            # Single-cache construction yields a degenerate (coordinator-less)
            # primary that behaves as lossless fp16 passthrough. Cross-layer
            # merging requires KVCacheBuilder.for_model(), which builds the
            # shared MiniCacheCoordinator and assigns primary/merge roles.
            cache = MiniCacheKVCache(config)
        elif config.method == "gear":
            cache = GEARKVCache(config)
        elif config.method == "zipcache":
            cache = ZipCacheKVCache(config)
        elif config.method == "snapkv":
            cache = SnapKVKVCache(config)
        elif config.method == "streaming_llm":
            cache = StreamingLLMKVCache(config)
        elif config.method == "h2o":
            cache = H2OKVCache(config)
        elif config.method == "tova":
            cache = TOVAKVCache(config)
        elif config.method == "pyramidkv":
            cache = PyramidKVCache(config)
        elif config.method == "squeeze":
            # Single-cache construction yields a coordinator-less layer that falls
            # back to squeeze_budget (uniform H2O). The 2D data-driven reallocation
            # requires KVCacheBuilder.for_model(), which builds the shared
            # SqueezeCoordinator and re-budgets after prefill.
            cache = SqueezeAttentionCache(config)
        elif config.method == "chunkkv":
            # No coordinator: each layer resolves its own chunks independently, so
            # the default for_model path (one ChunkKVCache per layer) is all it
            # needs. chunk_size=1 reduces bit-for-bit to H2O-adapted.
            cache = ChunkKVCache(config)
        elif config.method == "cam":
            # No coordinator: each layer merges independently, so the default
            # for_model path (one CaMKVCache per layer) is all it needs.
            # cam_merge="drop" reduces bit-for-bit to H2O-adapted.
            cache = CaMKVCache(config)
        elif config.method == "xkv":
            # Single-cache construction yields a degenerate (coordinator-less)
            # standalone member — behaves as per-layer SVD compression with no
            # basis sharing. Cross-layer subspace sharing requires
            # KVCacheBuilder.for_model(), which builds the shared
            # XKVCoordinator and assigns member/group roles.
            cache = XKVCache(config)
        elif config.method == "nsnquant":
            # No coordinator: single-layer wrapper with a chunk-flush residual
            # buffer; the universal codebook is model-independent (synthetic
            # Gaussian), so the default for_model path (one NSNQuantKVCache
            # per layer) is all it needs.
            cache = NSNQuantKVCache(config)
        elif config.method == "knorm":
            # No coordinator: intrinsic key-norm scores need no cross-layer or
            # cross-step state; the default for_model path (one L2NormKVCache
            # per layer) is all it needs.
            cache = L2NormKVCache(config)
        elif config.method == "skvq":
            # No coordinator: single-layer chunk-flush wrapper; the channel
            # permutations are per-layer state frozen from the first flushed
            # chunk, so the default for_model path (one SKVQKVCache per
            # layer) is all it needs.
            cache = SKVQKVCache(config)
        elif config.method == "qfilters":
            # No coordinator: the query-agnostic filter is per-layer state
            # frozen from the first observed chunk (projection scorer), so the
            # default for_model path (one QFiltersKVCache per layer) suffices.
            cache = QFiltersKVCache(config)
        elif config.method == "keyformer":
            # No coordinator: the Gumbel-regularized accumulating scorer is
            # per-layer per-head state; the default for_model path (one
            # KeyformerKVCache per layer) is all it needs.
            cache = KeyformerKVCache(config)
        elif config.method == "morphkv":
            # No coordinator: recent-window correlation retention is per-layer
            # per-head state recomputed each step; the default for_model path
            # (one MorphKVKVCache per layer) is all it needs.
            cache = MorphKVKVCache(config)
        elif config.method == "kvzip":
            # No coordinator: context-reconstruction reliance is per-layer
            # per-head state recomputed each step; the default for_model path
            # (one KVzipKVCache per layer) is all it needs.
            cache = KVzipKVCache(config)
        elif config.method == "kvtc":
            # No coordinator: local PCA basis + DP-optimal bit allocation are
            # fit once per (batch, head) at prefill and frozen thereafter; the
            # default for_model path (one KVTCKVCache per layer) is all it
            # needs.
            cache = KVTCKVCache(config)
        elif config.method == "curdkv":
            # No coordinator: value-aware leverage-score eviction is per-layer
            # per-head state recomputed each step; the default for_model path
            # (one CurDKVKVCache per layer) is all it needs.
            cache = CurDKVKVCache(config)
        elif config.method == "nestedkv":
            # No coordinator: multi-scale ensembled scoring, head-adaptive
            # blend, and cross-head budget competition all run once per
            # (batch, layer) at prefill end and freeze thereafter; the
            # default for_model path (one NestedKVKVCache per layer) is all
            # it needs.
            cache = NestedKVKVCache(config)
        elif config.method == "amc":
            # No coordinator: per-token saliency scoring, tier assignment, rank
            # masking, and quantization all run independently every step
            # (prefill and decode alike); the default for_model path (one
            # AMCKVCache per layer) is all it needs.
            cache = AMCKVCache(config)
        elif config.method == "a2ats":
            # No coordinator: windowed RoPE regime selection, retrieval-set
            # split, and query-aware codebook assignment all run
            # independently every step (prefill and decode alike, using
            # the incoming key as a proxy query — see a2ats_cache.py's
            # module docstring); the default for_model path (one
            # A2ATSKVCache per layer) is all it needs.
            cache = A2ATSKVCache(config)
        elif config.method == "anchorkv":
            # No coordinator: anchor selection, per-token assignment/
            # projection, utility scoring, and cross-head residual-budget
            # allocation all run once per (batch, layer) at prefill end and
            # freeze thereafter; the default for_model path (one
            # AnchorKVKVCache per layer) is all it needs.
            cache = AnchorKVKVCache(config)
        elif config.method == "rocketkv":
            # No coordinator: stage-1 SnapKV eviction and stage-2 HSA paged
            # summaries are both per-layer per-head state recomputed at
            # prefill/decode; the default for_model path (one RocketKVKVCache
            # per layer) is all it needs.
            cache = RocketKVKVCache(config)
        elif config.method == "age_tiered":
            # No coordinator: age-tier assignment and re-quantization are
            # per-layer per-head state recomputed independently every step
            # (prefill and decode alike, purely from cumulative position);
            # the default for_model path (one AgeTieredKVCache per layer) is
            # all it needs.
            cache = AgeTieredKVCache(config)
        else:
            raise QuantizerConfigError(
                f"KVCacheFactory: unknown method '{config.method}'. "
                f"Choices: turboquant_prod, turboquant_mse, turboquant_rvq, "
                f"polar, qjl, vecinfer, spectral, kivi, kivi_sink, svdq, kitty, "
                f"adakv, xquant, kvquant, palu, cachegen, minicache, gear, zipcache, snapkv, "
                f"streaming_llm, h2o, tova, pyramidkv, squeeze, chunkkv, cam, xkv, nsnquant, knorm, skvq, qfilters, keyformer, morphkv, kvzip, kvtc, curdkv, nestedkv, amc, a2ats, anchorkv, rocketkv, age_tiered."
            )

        if config.sliding_window is not None:
            if config.method not in STANDALONE_METHODS:
                raise QuantizerConfigError(
                    f"KVCacheFactory: sliding_window is only compatible with "
                    f"standalone methods (see STANDALONE_METHODS: "
                    f"{sorted(STANDALONE_METHODS)}) — method {config.method!r} "
                    f"produces a cache implementing mlx_lm.models.cache.KVCache's "
                    f"update_and_fetch protocol, not VeloxQuant's own "
                    f"append_key/append_value/attend interface that "
                    f"SlidingWindowKVCache wraps, so the result would have "
                    f"neither method working. Drop sliding_window for this "
                    f"method — several mlx_lm-protocol methods (h2o, tova, "
                    f"streaming_llm, snapkv, etc.) already implement their own "
                    f"budget/window-based eviction."
                )
            cache = SlidingWindowKVCache(cache, window_size=config.sliding_window)

        return cache


class KVCacheBuilder:
    """Fluent builder for KVCache construction with validation.

    Example::

        cache = (
            KVCacheBuilder()
            .with_method("turboquant_prod")
            .with_head_dim(128)
            .with_bit_width(inlier=2, outlier=3)
            .with_jl_dim(128)
            .with_seed(42)
            .build()
        )
    """

    def __init__(self) -> None:
        self._config = KVCacheConfig()

    def with_method(self, method: str) -> "KVCacheBuilder":
        """Set the quantisation method.

        Args:
            method: One of 'turboquant_prod', 'turboquant_mse', 'polar', 'qjl'.
        """
        self._config.method = method  # type: ignore[assignment]
        return self

    def with_head_dim(self, d: int) -> "KVCacheBuilder":
        """Set the attention head dimension.

        Args:
            d: Head dimension (must be a power of 2).
        """
        self._config.head_dim = d
        return self

    def with_bit_width(self, inlier, outlier: Optional[int] = None) -> "KVCacheBuilder":
        """Set bit-width(s).

        Args:
            inlier: Bit-width for inlier channels. Either an int (uniform across
                all layers) or a list[int] of length n_layers for RateQuant-style
                per-layer allocation. When a list is supplied, this builder
                must be consumed via ``KVCacheBuilder.for_model(model, config)``;
                direct ``.build()`` rejects the list.
            outlier: Bit-width for outlier channels (defaults to inlier if None).
        """
        self._config.bit_width_inlier = inlier
        self._config.bit_width_outlier = outlier
        return self

    def with_jl_dim(self, m: int) -> "KVCacheBuilder":
        """Set the JL projection dimension.

        Args:
            m: Must be <= head_dim.
        """
        self._config.jl_dim = m
        return self

    def with_n_outlier_channels(self, n: int) -> "KVCacheBuilder":
        """Set the number of outlier channels to detect.

        Args:
            n: Must be < head_dim.
        """
        self._config.n_outlier_channels = n
        return self

    def with_n_calib_tokens(self, n: int) -> "KVCacheBuilder":
        """Set calibration token count for outlier activation."""
        self._config.n_calib_tokens = n
        return self

    def with_vectorized_attend(self, enabled: bool = True) -> "KVCacheBuilder":
        """Enable vectorized packed-key unpack in attend()."""
        self._config.enable_vectorized_attend = enabled
        return self

    def with_outlier_two_stream(self, enabled: bool = True) -> "KVCacheBuilder":
        """Enable outlier/inlier split cache after calibration."""
        self._config.enable_outlier_two_stream = enabled
        return self

    def with_fused_query_dot(self, enabled: bool = True) -> "KVCacheBuilder":
        """Enable fused rotated-query + codebook-dot path."""
        self._config.enable_fused_query_dot = enabled
        return self

    def with_seed(self, seed: int) -> "KVCacheBuilder":
        """Set the random seed.

        Args:
            seed: Integer seed.
        """
        self._config.seed = seed
        return self

    def with_precision(self, dtype: Any) -> "KVCacheBuilder":
        """Set the compute dtype.

        Args:
            dtype: MLX dtype (e.g. mx.float16).
        """
        self._config.dtype = dtype
        return self

    def with_capacity(self, max_tokens: int) -> "KVCacheBuilder":
        """Set the maximum number of tokens to store.

        Args:
            max_tokens: Positive integer.
        """
        self._config.capacity = max_tokens
        return self

    def with_artifact_store(self, store: ArtifactStore) -> "KVCacheBuilder":
        """Provide an ArtifactStore for loading precomputed artifacts.

        Args:
            store: ArtifactStore instance.
        """
        self._config.store = store
        return self

    def with_observer(self, observer: QuantizationObserver) -> "KVCacheBuilder":
        """Attach a QuantizationObserver.

        Args:
            observer: Observer instance.
        """
        self._config.observers.append(observer)
        return self

    def with_sliding_window(self, window_size: int) -> "KVCacheBuilder":
        """Wrap the cache with a sliding-window eviction policy.

        Args:
            window_size: Number of tokens to keep.
        """
        self._config.sliding_window = window_size
        return self

    def build(self) -> KVCache:
        """Validate the configuration and construct the KVCache.

        Returns:
            Configured KVCache instance.

        Raises:
            QuantizerConfigError: If any validation check fails.
        """
        cfg = self._config
        d = cfg.head_dim

        if not (d >= 1 and (d & (d - 1)) == 0):
            raise QuantizerConfigError(f"KVCacheBuilder: head_dim={d} must be a power of 2.")
        if isinstance(cfg.bit_width_inlier, list):
            if not cfg.bit_width_inlier:
                raise QuantizerConfigError(
                    "KVCacheBuilder: bit_width_inlier list must not be empty."
                )
            if not all(isinstance(b, int) and b >= 1 for b in cfg.bit_width_inlier):
                raise QuantizerConfigError(
                    "KVCacheBuilder: every element of bit_width_inlier must be an int >= 1."
                )
        elif cfg.bit_width_inlier < 1:
            raise QuantizerConfigError(
                f"KVCacheBuilder: bit_width_inlier={cfg.bit_width_inlier} must be >= 1."
            )
        if cfg.jl_dim is not None and cfg.jl_dim > d:
            raise QuantizerConfigError(
                f"KVCacheBuilder: jl_dim={cfg.jl_dim} must be <= head_dim={d}."
            )
        if cfg.n_outlier_channels is not None and cfg.n_outlier_channels >= d:
            raise QuantizerConfigError(
                f"KVCacheBuilder: n_outlier_channels={cfg.n_outlier_channels} "
                f"must be < head_dim={d}."
            )
        if cfg.n_calib_tokens is not None and cfg.n_calib_tokens < 1:
            raise QuantizerConfigError(
                f"KVCacheBuilder: n_calib_tokens={cfg.n_calib_tokens} must be >= 1."
            )

        from veloxquant_mlx.metal._warmup import warmup_for_config

        cache = KVCacheFactory.create(cfg)
        warmup_for_config(cfg)
        return cache

    @staticmethod
    def for_model(model, config: "KVCacheConfig") -> list:
        """Build one KVCache per language-model layer, sized per-layer.

        Works for text-only and VLM models (Qwen2-VL, Qwen3-VL, Mistral, etc.).
        Layers without a self_attn attribute (MoE gates, etc.) fall back to a
        standard fp16 KVCache so the list length always matches model.layers.

        Per-layer bit-widths (RateQuant)
        --------------------------------
        If ``config.bit_width_inlier`` is a ``list[int]``, element ``i`` is
        used for layer ``i``. The list length must equal the number of
        attention-bearing layers (layers without self_attn are skipped from
        the count). This lets RateQuant-style mixed-precision allocations
        be passed through the standard API without manual cache wiring.

        Args:
            model: Loaded mlx_lm model instance.
            config: KVCacheConfig specifying method, bit_width_inlier, seed, etc.
                    head_dim is overridden per-layer.

        Returns:
            List of KVCache instances, one per language-model layer.

        Raises:
            QuantizerConfigError: If ``config.method`` is a standalone method
                (see :data:`STANDALONE_METHODS`) that does not implement the
                mlx_lm KVCache serving contract.
        """
        if config.method in STANDALONE_METHODS:
            raise QuantizerConfigError(
                f"KVCacheBuilder.for_model: method {config.method!r} is a standalone "
                f"method — its cache class implements VeloxQuant's own KVCache "
                f"interface, not mlx_lm.models.cache.KVCache, so it cannot serve "
                f"mlx_lm.generate() traffic and will fail deep inside generation "
                f"rather than here. Use it directly via KVCacheFactory.create() / "
                f"KVCacheBuilder.build() for standalone/research use instead, or "
                f"pick a serving-compatible method (see the method library's "
                f"'standalone' tag: https://github.com/rajveer43/VeloxQuant-MLX#method-library)."
            )

        from mlx_lm.models.cache import KVCache as _FallbackCache

        _PlainKVCache = _FallbackCache

        # Qwen2-VL exposes model.layers directly; text models expose model.model.layers
        layers = getattr(model, "layers", None) or model.model.layers

        # Hybrid-attention models (Qwen3.6 / qwen3_5, qwen3_next, Mamba hybrids)
        # interleave real attention layers with linear/recurrent ones, and the
        # non-attention slots need the model's OWN cache type -- qwen3_5's
        # GatedDeltaNet does `cache[0] = ...`, which a plain mlx_lm KVCache does
        # not support (no __setitem__), and its TextModel calls make_mask() on
        # the slot at ssm_idx, which a KVCache answers with the wrong signature.
        # Both raise TypeError deep inside generation rather than here. Ask the
        # model what it wants for each slot and reuse that for anything we do
        # not quantize; fall back to a plain KVCache only when the model offers
        # no usable make_cache.
        _native: list = []
        _native_fn = getattr(model, "make_cache", None)
        if callable(_native_fn):
            try:
                built = _native_fn()
                if isinstance(built, list) and len(built) == len(layers):
                    _native = built
            except Exception:  # pragma: no cover - model-specific constructors
                _native = []

        def _fallback_for(i: int):
            """Cache for a layer VeloxQuant does not quantize.

            Returns the model's native cache object for slot ``i`` when one is
            available (preserving recurrent/linear-attention state types), else
            a plain fp16 KVCache so the list length always matches the layers.
            """
            return _native[i] if i < len(_native) else _PlainKVCache()

        # VLM wrappers (Qwen2-VL) have model.args.text_config only;
        # real attention config lives in model.language_model.args
        args = getattr(model, "args", None)
        if args is not None and not hasattr(args, "hidden_size"):
            lm = getattr(model, "language_model", None)
            if lm is not None:
                args = getattr(lm, "args", args)

        # Resolve per-layer bit-width policy
        b_spec = config.bit_width_inlier
        is_per_layer = isinstance(b_spec, list)
        if is_per_layer:
            # Count attention-bearing layers up-front for validation
            n_attn = sum(
                1
                for L in layers
                if (getattr(L, "self_attn", None) or getattr(L, "attn", None)) is not None
            )
            if len(b_spec) != n_attn:
                raise QuantizerConfigError(
                    f"KVCacheBuilder.for_model: bit_width_inlier is a list of "
                    f"length {len(b_spec)}, but model has {n_attn} attention "
                    f"layers. The list must have one entry per attention layer."
                )

        # --- XQuant: cross-layer reuse needs a shared coordinator + roles ----
        if config.method == "xquant":
            return KVCacheBuilder._build_xquant(layers, args, config, _FallbackCache)

        # --- MiniCache: cross-layer merge needs a shared coordinator + roles --
        if config.method == "minicache":
            return KVCacheBuilder._build_minicache(layers, args, config, _FallbackCache)

        # --- PyramidKV: per-layer budget schedule injected at build time -------
        if config.method == "pyramidkv":
            return KVCacheBuilder._build_pyramidkv(layers, args, config, _FallbackCache)

        # --- CacheGen: per-layer bit-width schedule injected at build time -----
        if config.method == "cachegen":
            return KVCacheBuilder._build_cachegen(layers, args, config, _FallbackCache)

        # --- SqueezeAttention: shared coordinator re-budgets after prefill -----
        if config.method == "squeeze":
            return KVCacheBuilder._build_squeeze(layers, args, config, _FallbackCache)

        # --- xKV: cross-layer shared-subspace SVD needs a shared coordinator --
        if config.method == "xkv":
            return KVCacheBuilder._build_xkv(layers, args, config, _FallbackCache)

        # --- ChunkKV: layer-wise index reuse needs a shared coordinator -------
        if config.method == "chunkkv" and config.chunkkv_reuse_layers > 1:
            return KVCacheBuilder._build_chunkkv(layers, args, config, _FallbackCache)

        from veloxquant_mlx.metal._warmup import warmup_for_config

        caches = []
        attn_idx = 0  # index into b_spec, advances only for attention layers
        warmed_keys: set = set()
        for i, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is None:
                caches.append(_fallback_for(i))
                continue
            hd = getattr(attn, "head_dim", None)
            if hd is None:
                if args is not None:
                    hd = getattr(args, "head_dim", None) or (
                        args.hidden_size // args.num_attention_heads
                    )
            if hd is None:
                caches.append(_fallback_for(i))
                continue
            layer_b = b_spec[attn_idx] if is_per_layer else b_spec
            # Preserve every method-specific field (svdq_*, kitty_*, kvquant_*,
            # palu_*, …) from the user's config and override only the per-layer
            # head_dim / bit-width / seed.  Reconstructing the dataclass field by
            # field (as the old code did) silently dropped method hyperparameters.
            layer_cfg = dataclasses_replace(
                config,
                head_dim=hd,
                bit_width_inlier=layer_b,
                seed=config.seed + i,
                store=config.store,
            )
            # Warm each distinct (head_dim, bit_width) combo once rather than
            # per layer — most models use the same head_dim across layers, so
            # this usually compiles a single kernel variant for the whole model.
            warm_key = (hd, layer_b if not isinstance(layer_b, list) else tuple(layer_b))
            if warm_key not in warmed_keys:
                warmup_for_config(layer_cfg)
                warmed_keys.add(warm_key)
            caches.append(KVCacheFactory.create(layer_cfg))
            attn_idx += 1
        return caches

    @staticmethod
    def _build_xquant(layers, args, config: "KVCacheConfig", fallback_cls) -> list:
        """Build one shared XQuantCoordinator and role-assigned caches per layer.

        Anchor/reuse roles are assigned over *attention-bearing* layers only, so
        non-attention layers (MoE gates, etc.) get a plain fallback cache and do
        not consume a group slot.
        """
        from veloxquant_mlx.cache.xquant_cache import XQuantKVCache
        from veloxquant_mlx.cache.xquant_coordinator import XQuantCoordinator
        from veloxquant_mlx.quantizers.xquant import pair_layers

        def _head_dim(layer):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is None:
                return None
            hd = getattr(attn, "head_dim", None)
            if hd is None and args is not None:
                hd = getattr(args, "head_dim", None) or (
                    args.hidden_size // args.num_attention_heads
                )
            return hd

        attn_layer_idx = [i for i, L in enumerate(layers) if _head_dim(L) is not None]
        roles = pair_layers(len(attn_layer_idx), config.xquant_group_size)
        coordinator = XQuantCoordinator(max_ctx=config.xquant_max_ctx)

        role_by_layer: dict[int, tuple[str, int]] = {
            attn_layer_idx[k]: roles[k] for k in range(len(attn_layer_idx))
        }
        # Each anchor's published segment is consumed by every reuse layer in
        # its group before the coordinator can reclaim it (#80). Seed every
        # group at 0 first so a trailing degenerate group (anchor with no
        # reusers) doesn't fall through to some other group's reader count.
        n_readers_by_group: dict[int, int] = {group_id: 0 for _, group_id in roles}
        for role, group_id in roles:
            if role == "reuse":
                n_readers_by_group[group_id] += 1

        caches = []
        for i, layer in enumerate(layers):
            hd = _head_dim(layer)
            if hd is None:
                caches.append(fallback_cls())
                continue
            role, group_id = role_by_layer[i]
            layer_cfg = KVCacheConfig(
                method="xquant",
                head_dim=hd,
                seed=config.seed + i,
                xquant_group_size=config.xquant_group_size,
                xquant_base_bits=config.xquant_base_bits,
                xquant_residual_bits=config.xquant_residual_bits,
                xquant_group_quant_size=config.xquant_group_quant_size,
                xquant_max_ctx=config.xquant_max_ctx,
            )
            caches.append(
                XQuantKVCache(
                    layer_cfg,
                    role=role,
                    group_id=group_id,
                    coordinator=coordinator,
                    n_readers=n_readers_by_group[group_id],
                )
            )
        return caches

    @staticmethod
    def _build_xkv(layers, args, config: "KVCacheConfig", fallback_cls) -> list:
        """Build one shared XKVCoordinator and member/group-assigned caches
        per layer.

        Members are assigned over *attention-bearing* layers only, in fixed
        contiguous groups of ``xkv_group_size`` (a trailing partial group is
        still valid, with a smaller ``n_members``), so non-attention layers
        (MoE gates, etc.) get a plain fallback cache and do not consume a
        group slot.
        """
        from veloxquant_mlx.cache.xkv_cache import XKVCache
        from veloxquant_mlx.cache.xkv_coordinator import XKVCoordinator
        from veloxquant_mlx.quantizers.xkv import pair_layers_grouped

        def _head_dim(layer):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is None:
                return None
            hd = getattr(attn, "head_dim", None)
            if hd is None and args is not None:
                hd = getattr(args, "head_dim", None) or (
                    args.hidden_size // args.num_attention_heads
                )
            return hd

        attn_layer_idx = [i for i, L in enumerate(layers) if _head_dim(L) is not None]
        roles = pair_layers_grouped(len(attn_layer_idx), config.xkv_group_size)
        coordinator = XKVCoordinator(max_ctx=config.xkv_max_ctx)

        role_by_layer: dict[int, tuple[int, int, int]] = {
            attn_layer_idx[k]: roles[k] for k in range(len(attn_layer_idx))
        }

        caches = []
        for i, layer in enumerate(layers):
            hd = _head_dim(layer)
            if hd is None:
                caches.append(fallback_cls())
                continue
            member_idx, group_id, n_members = role_by_layer[i]
            layer_cfg = KVCacheConfig(
                method="xkv",
                head_dim=hd,
                seed=config.seed + i,
                xkv_group_size=config.xkv_group_size,
                xkv_rank=config.xkv_rank,
                xkv_energy_threshold=config.xkv_energy_threshold,
                xkv_latent_bits=config.xkv_latent_bits,
                xkv_group_quant_size=config.xkv_group_quant_size,
                xkv_max_ctx=config.xkv_max_ctx,
            )
            caches.append(
                XKVCache(
                    layer_cfg,
                    member_idx=member_idx,
                    group_id=group_id,
                    n_members=n_members,
                    coordinator=coordinator,
                )
            )
        return caches

    @staticmethod
    def _build_minicache(layers, args, config: "KVCacheConfig", fallback_cls) -> list:
        """Build one shared MiniCacheCoordinator and role-assigned caches per layer.

        Primary/merge roles are assigned over *attention-bearing* layers only,
        and only middle-to-deep layers (>= ``minicache_start_frac`` of depth) are
        eligible for merging — earlier layers are standalone primaries.
        """
        from veloxquant_mlx.cache.minicache_cache import MiniCacheKVCache
        from veloxquant_mlx.cache.minicache_coordinator import MiniCacheCoordinator
        from veloxquant_mlx.quantizers.minicache import pair_layers_depth

        def _head_dim(layer):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is None:
                return None
            hd = getattr(attn, "head_dim", None)
            if hd is None and args is not None:
                hd = getattr(args, "head_dim", None) or (
                    args.hidden_size // args.num_attention_heads
                )
            return hd

        attn_layer_idx = [i for i, L in enumerate(layers) if _head_dim(L) is not None]
        roles = pair_layers_depth(
            len(attn_layer_idx),
            start_frac=config.minicache_start_frac,
            group_size=config.minicache_group_size,
        )
        coordinator = MiniCacheCoordinator(max_ctx=config.minicache_max_ctx)

        role_by_layer: dict[int, tuple[str, int]] = {
            attn_layer_idx[k]: roles[k] for k in range(len(attn_layer_idx))
        }
        # Each primary's published entry is consumed by every merge layer in
        # its group before the coordinator can reclaim it (#79). Seed every
        # group at 0 first so a standalone primary (no merge partners, e.g.
        # an early layer below minicache_start_frac) doesn't fall through to
        # some other group's reader count.
        n_readers_by_group: dict[int, int] = {group_id: 0 for _, group_id in roles}
        for role, group_id in roles:
            if role == "merge":
                n_readers_by_group[group_id] += 1

        caches = []
        for i, layer in enumerate(layers):
            hd = _head_dim(layer)
            if hd is None:
                caches.append(fallback_cls())
                continue
            role, group_id = role_by_layer[i]
            layer_cfg = KVCacheConfig(
                method="minicache",
                head_dim=hd,
                seed=config.seed + i,
                minicache_start_frac=config.minicache_start_frac,
                minicache_group_size=config.minicache_group_size,
                minicache_retention_threshold=config.minicache_retention_threshold,
                minicache_slerp_t=config.minicache_slerp_t,
                minicache_max_ctx=config.minicache_max_ctx,
            )
            caches.append(
                MiniCacheKVCache(
                    layer_cfg,
                    role=role,
                    group_id=group_id,
                    coordinator=coordinator,
                    n_readers=n_readers_by_group[group_id],
                )
            )
        return caches

    @staticmethod
    def _build_pyramidkv(layers, args, config: "KVCacheConfig", fallback_cls) -> list:
        """Build per-layer PyramidKV caches with a pyramid budget schedule.

        The pyramid budget is allocated over *attention-bearing* layers only
        (large budget early, small budget deep, mean == ``pyramid_budget``).
        Each attention layer receives its own resolved budget via
        ``pyramid_resolved_budget``; no runtime coordinator is needed.
        """
        from veloxquant_mlx.cache.pyramidkv_cache import PyramidKVCache
        from veloxquant_mlx.quantizers.pyramidkv import pyramid_budgets

        def _head_dim(layer):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is None:
                return None
            hd = getattr(attn, "head_dim", None)
            if hd is None and args is not None:
                hd = getattr(args, "head_dim", None) or (
                    args.hidden_size // args.num_attention_heads
                )
            return hd

        attn_layer_idx = [i for i, L in enumerate(layers) if _head_dim(L) is not None]
        schedule = pyramid_budgets(
            n_layers=len(attn_layer_idx),
            avg_budget=config.pyramid_budget,
            n_sink=config.pyramid_n_sink,
            beta=config.pyramid_beta,
        )
        budget_by_layer: dict[int, int] = {
            attn_layer_idx[k]: schedule[k] for k in range(len(attn_layer_idx))
        }

        caches = []
        for i, layer in enumerate(layers):
            hd = _head_dim(layer)
            if hd is None:
                caches.append(fallback_cls())
                continue
            layer_cfg = dataclasses_replace(
                config,
                head_dim=hd,
                seed=config.seed + i,
                store=config.store,
                pyramid_resolved_budget=budget_by_layer[i],
            )
            caches.append(PyramidKVCache(layer_cfg))
        return caches

    @staticmethod
    def _build_cachegen(layers, args, config: "KVCacheConfig", fallback_cls) -> list:
        """Build per-layer CacheGen caches with a layer-wise bit-width schedule.

        Implements the paper's layer-sensitivity observation (§5.1.2/§5.2):
        output quality is more sensitive to precision loss in shallow layers
        than in deep ones, so shallow layers get ``cachegen_bits`` and deeper
        layer groups get progressively fewer bits (``layer_group_bits``). Each
        attention layer receives its own resolved bit-width via
        ``cachegen_resolved_bits``; no runtime coordinator is needed.
        """
        from veloxquant_mlx.cache.cachegen_cache import CacheGenKVCache
        from veloxquant_mlx.quantizers.cachegen import layer_group_bits

        def _head_dim(layer):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is None:
                return None
            hd = getattr(attn, "head_dim", None)
            if hd is None and args is not None:
                hd = getattr(args, "head_dim", None) or (
                    args.hidden_size // args.num_attention_heads
                )
            return hd

        attn_layer_idx = [i for i, L in enumerate(layers) if _head_dim(L) is not None]
        schedule = layer_group_bits(
            n_layers=len(attn_layer_idx),
            base_bits=config.cachegen_bits,
            n_groups=config.cachegen_layer_groups,
        )
        bits_by_layer: dict[int, int] = {
            attn_layer_idx[k]: schedule[k] for k in range(len(attn_layer_idx))
        }

        caches = []
        for i, layer in enumerate(layers):
            hd = _head_dim(layer)
            if hd is None:
                caches.append(fallback_cls())
                continue
            layer_cfg = dataclasses_replace(
                config,
                head_dim=hd,
                seed=config.seed + i,
                store=config.store,
                cachegen_resolved_bits=bits_by_layer[i],
            )
            caches.append(CacheGenKVCache(layer_cfg))
        return caches

    @staticmethod
    def _build_squeeze(layers, args, config: "KVCacheConfig", fallback_cls) -> list:
        """Build one shared SqueezeCoordinator and per-layer SqueezeAttention caches.

        Every attention-bearing layer shares one coordinator. During prefill each
        layer reports its measured concentration; once all have reported, the
        coordinator computes the data-driven budget schedule (``squeeze_budgets``)
        and each layer pulls its resolved budget. ``squeeze_strength=0.0`` yields a
        uniform schedule (reduces to H2O). Non-attention layers get a plain
        fallback cache and do not report.
        """
        from veloxquant_mlx.cache.squeeze_cache import SqueezeAttentionCache
        from veloxquant_mlx.cache.squeeze_coordinator import SqueezeCoordinator

        def _head_dim(layer):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is None:
                return None
            hd = getattr(attn, "head_dim", None)
            if hd is None and args is not None:
                hd = getattr(args, "head_dim", None) or (
                    args.hidden_size // args.num_attention_heads
                )
            return hd

        attn_layer_idx = [i for i, L in enumerate(layers) if _head_dim(L) is not None]
        coordinator = SqueezeCoordinator(
            n_layers=len(attn_layer_idx),
            avg_budget=config.squeeze_budget,
            n_sink=config.squeeze_n_sink,
            strength=config.squeeze_strength,
        )

        caches = []
        for i, layer in enumerate(layers):
            hd = _head_dim(layer)
            if hd is None:
                caches.append(fallback_cls())
                continue
            layer_cfg = dataclasses_replace(
                config,
                head_dim=hd,
                seed=config.seed + i,
                store=config.store,
            )
            caches.append(SqueezeAttentionCache(layer_cfg, layer_id=i, coordinator=coordinator))
        return caches

    @staticmethod
    def _build_chunkkv(layers, args, config: "KVCacheConfig", fallback_cls) -> list:
        """Build per-layer ChunkKV caches sharing a layer-wise index-reuse coordinator.

        Implements the paper's Algorithm 2: attention-bearing layers are grouped
        into consecutive blocks of ``chunkkv_reuse_layers``; the leader of each
        block runs its own eviction and publishes the kept-token positions every
        step, and the remaining layers in the block reuse those positions instead
        of scoring and evicting independently. Layer ids passed to the coordinator
        are indices into the attention-bearing layer sequence (0, 1, 2, ...), not
        raw model layer indices, so blocks are contiguous even when non-attention
        layers are interleaved.
        """
        from veloxquant_mlx.cache.chunkkv_cache import ChunkKVCache
        from veloxquant_mlx.cache.chunkkv_coordinator import ChunkKVIndexReuseCoordinator

        def _head_dim(layer):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is None:
                return None
            hd = getattr(attn, "head_dim", None)
            if hd is None and args is not None:
                hd = getattr(args, "head_dim", None) or (
                    args.hidden_size // args.num_attention_heads
                )
            return hd

        attn_layer_idx = [i for i, L in enumerate(layers) if _head_dim(L) is not None]
        coordinator = ChunkKVIndexReuseCoordinator(
            n_layers=len(attn_layer_idx),
            reuse_layers=config.chunkkv_reuse_layers,
        )

        caches = []
        attn_pos = 0  # 0-based index among attention-bearing layers only
        for i, layer in enumerate(layers):
            hd = _head_dim(layer)
            if hd is None:
                caches.append(fallback_cls())
                continue
            layer_cfg = dataclasses_replace(
                config,
                head_dim=hd,
                seed=config.seed + i,
                store=config.store,
            )
            caches.append(ChunkKVCache(layer_cfg, layer_id=attn_pos, coordinator=coordinator))
            attn_pos += 1
        return caches

    def __repr__(self) -> str:
        return f"KVCacheBuilder(config={self._config!r})"
