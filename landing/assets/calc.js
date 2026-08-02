/* ============================================================
   VeloxQuant-MLX — shared KV-cache sizing math.

   Single source of truth for the landing-page calculator and the
   playground. Both load this file; neither re-derives the numbers.

   Ported from veloxquant_mlx/tools/mac_recommender.py — keep parity.
   Zero-build: plain globals under window.VQCalc, no modules.
   ============================================================ */
(function (global) {
  'use strict';

  // RAM sizes Apple actually ships / the recommender accepts.
  var ALLOWED_RAM_GB = [8, 16, 24, 32, 36, 48, 64, 128];

  // 4-bit weight footprint per model class, in GB.
  var MODEL_WEIGHT_GB_4BIT = { "1B": 0.8, "3B": 2.0, "7B": 4.5, "14B": 8.0, "32B": 18.0 };

  // Reserve left for OS + apps, and the floor applied to a KV budget so a
  // negative result stays usable and the UI can explain why.
  var OS_RESERVE_GB = 4.0;
  var MIN_KV_BUDGET_GB = 0.5;

  /* ---------- core sizing ---------- */

  // Full K+V fp16 cache size in megabytes.
  // 2 tensors (K and V) x layers x kv_heads x head_dim x seq_len x 2 bytes.
  function estimateKvFp16Mb(nLayers, nKvHeads, headDim, seqLen) {
    var bytes = 2 * nLayers * nKvHeads * headDim * seqLen * 2;
    return bytes / Math.pow(1024, 2);
  }

  // RAM realistically available for the KV cache: unified RAM minus 4-bit
  // weights minus the same OS reserve the recommender uses.
  function deriveKvBudgetGb(ramGb, modelClass) {
    var weightGb = MODEL_WEIGHT_GB_4BIT[modelClass];
    if (weightGb === undefined) weightGb = 0;
    var free = ramGb - weightGb - OS_RESERVE_GB;
    return Math.max(MIN_KV_BUDGET_GB, Math.round(free * 10) / 10);
  }

  function round2(x) {
    return Math.round(x * 100) / 100;
  }

  /* ---------- model presets ----------
     Real architectures of the benchmarked checkpoints, so a visitor never
     has to know their model's internals. Matches the mlx-community 4-bit
     configs used across figures/. */
  var MODEL_PRESETS = [
    { id: "llama32-1b",  name: "Llama-3.2-1B",  model_class: "1B",  n_layers: 16, n_kv_heads: 8,  head_dim: 64 },
    { id: "llama32-3b",  name: "Llama-3.2-3B",  model_class: "3B",  n_layers: 28, n_kv_heads: 8,  head_dim: 128 },
    { id: "llama31-8b",  name: "Llama-3.1-8B",  model_class: "7B",  n_layers: 32, n_kv_heads: 8,  head_dim: 128 },
    { id: "mistral-7b",  name: "Mistral-7B",    model_class: "7B",  n_layers: 32, n_kv_heads: 8,  head_dim: 128 },
    { id: "qwen25-7b",   name: "Qwen2.5-7B",    model_class: "7B",  n_layers: 28, n_kv_heads: 4,  head_dim: 128 },
    { id: "qwen3-8b",    name: "Qwen3-8B",      model_class: "7B",  n_layers: 36, n_kv_heads: 8,  head_dim: 128 },
    { id: "phi4",        name: "Phi-4",         model_class: "14B", n_layers: 40, n_kv_heads: 10, head_dim: 128 },
    { id: "falcon3-7b",  name: "Falcon3-7B",    model_class: "7B",  n_layers: 28, n_kv_heads: 4,  head_dim: 256 },
    { id: "gemma3-4b",   name: "Gemma-3-4B",    model_class: "3B",  n_layers: 34, n_kv_heads: 4,  head_dim: 256 },
    { id: "qwen25-32b",  name: "Qwen2.5-32B",   model_class: "32B", n_layers: 64, n_kv_heads: 8,  head_dim: 128 }
  ];

  /* ---------- methods shown by the landing calculator ----------
     ratio    = compression factor for the "fits in RAM" estimate.
     resident = whether that ratio reflects real resident-RAM savings
                (full-KV path) or key-accounting only. Only the full-KV
                path frees resident RAM; the rest keep an fp16 parent
                cache by default, so their number is an accounting bound.
     Mirrors resident_savings_likely in mac_recommender.py. */
  var CALC_METHODS = [
    {
      id: "turboquant_rvq",
      name: "RVQ-1bit",
      ratio: 7.5,
      resident: false,
      note: "zero calibration",
      config: 'KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)'
    },
    {
      id: "vecinfer",
      name: "VecInfer-1bit",
      ratio: 16.0,
      resident: false,
      note: "needs calibration",
      config: 'KVCacheConfig(method="vecinfer", key_codebook_bits=8, key_sub_dim=8)'
    },
    {
      id: "rabitq",
      name: "RaBitQ",
      ratio: 6.0,
      resident: true,
      note: "full KV — real RAM savings",
      config: 'KVCacheConfig(method="rabitq")'
    }
  ];

  /* ---------- formatting ---------- */

  function formatMb(mb) {
    if (mb >= 1024) return (mb / 1024).toFixed(mb / 1024 >= 10 ? 0 : 2) + ' GB';
    if (mb >= 10) return Math.round(mb) + ' MB';
    return mb.toFixed(1) + ' MB';
  }

  // Context lengths are powers of two, so "k" here means 1024 — 32768 tokens
  // is "32k", not the "33k" a decimal divisor would print.
  function formatTokens(tokens) {
    if (tokens >= 1024) {
      var k = tokens / 1024;
      return (k >= 10 ? Math.round(k) : Math.round(k * 10) / 10) + 'k';
    }
    return String(Math.round(tokens));
  }

  /* ---------- the landing-page calculation ----------
     Given a preset, a context length and a machine's RAM, return one row
     per method plus the fp16 baseline, each marked fits / does not fit. */
  function computeRows(preset, seqLen, ramGb) {
    var budgetGb = deriveKvBudgetGb(ramGb, preset.model_class);
    var budgetMb = budgetGb * 1024;

    var fp16Mb = estimateKvFp16Mb(preset.n_layers, preset.n_kv_heads, preset.head_dim, seqLen);

    // Tokens that fit in the budget, at a given compression ratio. The cache
    // grows linearly in seq_len, so per-token cost is fp16Mb / seqLen.
    var mbPerToken = fp16Mb / seqLen;

    var rows = [{
      id: 'fp16',
      name: 'fp16 (no compression)',
      ratio: 1,
      resident: true,
      note: 'baseline',
      mb: round2(fp16Mb),
      fits: fp16Mb <= budgetMb,
      maxTokens: Math.floor(budgetMb / mbPerToken),
      config: null
    }];

    CALC_METHODS.forEach(function (m) {
      var mb = fp16Mb / m.ratio;
      rows.push({
        id: m.id,
        name: m.name,
        ratio: m.ratio,
        resident: m.resident,
        note: m.note,
        mb: round2(mb),
        fits: mb <= budgetMb,
        maxTokens: Math.floor(budgetMb / (mbPerToken / m.ratio)),
        config: m.config
      });
    });

    return {
      rows: rows,
      fp16Mb: round2(fp16Mb),
      budgetGb: budgetGb,
      budgetMb: round2(budgetMb),
      weightGb: MODEL_WEIGHT_GB_4BIT[preset.model_class] || 0
    };
  }

  // Preference order when several methods fit: the zero-calibration default
  // first, then full-KV (real resident savings), then the calibration-heavy
  // 16x path. Compression ratio is deliberately NOT the tiebreak — a bigger
  // ratio that costs a calibration pass is not the better default.
  var PREFERENCE = ['turboquant_rvq', 'rabitq', 'vecinfer'];

  // Recommend the most-preferred method that fits the budget. If none fit,
  // fall back to the highest ratio available and let the UI say it's tight.
  function pickRecommended(rows) {
    var byPref = function (list) {
      var ranked = list.slice().sort(function (a, b) {
        return PREFERENCE.indexOf(a.id) - PREFERENCE.indexOf(b.id);
      });
      return ranked[0];
    };
    var fitting = rows.filter(function (r) { return r.id !== 'fp16' && r.fits; });
    if (fitting.length) return byPref(fitting);

    var compressed = rows.filter(function (r) { return r.id !== 'fp16'; });
    return compressed.reduce(function (best, r) {
      return r.ratio > best.ratio ? r : best;
    });
  }

  global.VQCalc = {
    ALLOWED_RAM_GB: ALLOWED_RAM_GB,
    MODEL_WEIGHT_GB_4BIT: MODEL_WEIGHT_GB_4BIT,
    OS_RESERVE_GB: OS_RESERVE_GB,
    MIN_KV_BUDGET_GB: MIN_KV_BUDGET_GB,
    MODEL_PRESETS: MODEL_PRESETS,
    CALC_METHODS: CALC_METHODS,
    estimateKvFp16Mb: estimateKvFp16Mb,
    deriveKvBudgetGb: deriveKvBudgetGb,
    round2: round2,
    formatMb: formatMb,
    formatTokens: formatTokens,
    computeRows: computeRows,
    pickRecommended: pickRecommended
  };
})(window);
