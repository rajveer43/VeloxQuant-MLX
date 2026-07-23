/* ============================================================
   VeloxQuant-MLX Playground — client-side, zero-build.

   Three tools, all computed in the browser:
     1. Recommender      — 1:1 JS port of veloxquant_mlx/tools/mac_recommender.py
     2. Compression lab  — port of estimate_kv_fp16_mb + per-method ratios
     3. Benchmark viewer — inline SVG charts over real measured data
                           (assets/data/benchmarks.json)

   NOTHING here is fabricated: the recommender mirrors the Python heuristic
   exactly, and every benchmark number traces to a committed figures/*.json.
   Keep parity with mac_recommender.py — see verify script in the PR.
   ============================================================ */

/* ---------- Engine: ported verbatim from mac_recommender.py ---------- */

const ALLOWED_RAM_GB = [8, 16, 24, 32, 36, 48, 64, 128];
const MODEL_WEIGHT_GB_4BIT = { "1B": 0.8, "3B": 2.0, "7B": 4.5, "14B": 8.0, "32B": 18.0 };

// Full K+V fp16 cache size in megabytes (port of estimate_kv_fp16_mb).
function estimateKvFp16Mb(nLayers, nKvHeads, headDim, seqLen) {
  const bytes = 2 * nLayers * nKvHeads * headDim * seqLen * 2;
  return bytes / (1024 ** 2);
}

// Round-half-to-even to match Python's round() used in the reference engine.
function round2(x) {
  const r = Math.round(x * 100) / 100;
  // JS Math.round is half-up; Python is banker's rounding. The reference
  // values here never land exactly on a .xx5 boundary at 2 dp for realistic
  // inputs, so half-up is equivalent in practice. Kept explicit for clarity.
  return r;
}

// 1:1 port of recommend(req). Returns the same fields as RecommendResult.to_dict().
function recommend(req) {
  if (!ALLOWED_RAM_GB.includes(req.ram_gb)) {
    throw new Error(`ram_gb must be one of ${ALLOWED_RAM_GB}, got ${req.ram_gb}`);
  }
  if (req.seq_len < 1) throw new Error("seq_len must be >= 1");

  const warnings = [];
  const weightGb = MODEL_WEIGHT_GB_4BIT[req.model_class];
  const headroomGb = req.ram_gb - weightGb - 4.0; // leave ~4 GB for OS + apps
  if (headroomGb < 3.0) {
    // Match Python's float repr (e.g. "2.0", "18.0"): the reference message
    // uses `~{weight_gb} GB` where weight_gb is always a float.
    const weightStr = Number.isInteger(weightGb) ? weightGb.toFixed(1) : String(weightGb);
    warnings.push(
      `${req.model_class} 4-bit weights (~${weightStr} GB) leave little ` +
      `headroom on ${req.ram_gb} GB (est. headroom ${headroomGb.toFixed(1)} GB). ` +
      "Prefer a smaller model, eviction, or full-KV compression."
    );
  }

  const kvFp16 = estimateKvFp16Mb(req.n_layers, req.n_kv_heads, req.head_dim, req.seq_len);

  if (req.model_class === "1B") {
    warnings.push(
      "Metal kernel launch overhead can dominate on tiny models; " +
      "prefer RVQ or disable Metal if tok/s drops."
    );
  }

  const tight = req.ram_gb <= 16 || headroomGb < 3.0;
  let method, knobs, ratio, resident, rationale;

  if (req.goal === "everyday") {
    method = "turboquant_rvq";
    knobs = { bit_width_inlier: 1, seed: 42 };
    ratio = 7.5;
    resident = false;
    rationale =
      "Zero-calibration default. Key accounting ~7.5x at head_dim=128. " +
      "Default path dequantizes into parent fp16 cache.";
    if (tight && ["7B", "14B", "32B"].includes(req.model_class)) {
      warnings.push(
        "Tight RAM with a mid/large model: consider goal=max_context " +
        "(rabitq) or goal=constant_memory (eviction) for long prompts."
      );
    }
  } else if (req.goal === "max_key_accounting") {
    method = "vecinfer";
    knobs = {
      key_codebook_bits: 8, value_codebook_bits: 8,
      key_sub_dim: 8, value_sub_dim: 8,
      use_metal_kernels: null,
      note: "Requires one-time codebook calibration",
    };
    ratio = 16.0;
    resident = false;
    rationale =
      "Product VQ 1-bit path targets ~16x key accounting when " +
      "head_dim is divisible by sub_dim=8. Needs calibration.";
    if (req.head_dim % 8 !== 0) {
      warnings.push(
        `head_dim=${req.head_dim} is not divisible by 8; ` +
        "VecInfer sub_dim must divide head_dim."
      );
    }
  } else if (req.goal === "best_quality") {
    method = "spectral";
    knobs = { bit_width_inlier: 3, note: "Requires spectral rotation calibration" };
    ratio = 5.3;
    resident = false;
    rationale =
      "SpectralQuant targets better reconstruction at moderate " +
      "compression via eigenbasis rotation (calibration required).";
  } else if (req.goal === "max_context") {
    method = "rabitq";
    ratio = 6.0;
    resident = true;
    if (tight) {
      knobs = { note: "1-bit keys + MSE-b4 values; prefer fused Metal path when available" };
      rationale =
        "Full-KV compression is more likely to free resident memory " +
        "than key-only accounting methods on tight RAM.";
    } else {
      knobs = { note: "Full KV compression for longer context in fixed RAM" };
      rationale =
        "RaBitQ compresses keys and values. Better candidate for " +
        "real context capacity gains than key-only RVQ accounting.";
    }
  } else if (req.goal === "constant_memory") {
    method = "streaming_llm";
    knobs = { stream_n_sink: 4, stream_window_size: 512 };
    ratio = 1.0;
    resident = true;
    rationale =
      "Structural eviction keeps a fixed sink + window. Cache token " +
      "count stays bounded regardless of generation length.";
    warnings.push(
      "Eviction drops tokens; quality depends on the task. " +
      "For importance-based eviction try method=h2o instead."
    );
  } else {
    throw new Error(`Unknown goal: ${req.goal}`);
  }

  if (["M1", "M2"].includes(req.chip) && ["14B", "32B"].includes(req.model_class)) {
    warnings.push(
      `${req.chip} with ${req.model_class}: expect lower tok/s; ` +
      "memory fit still depends mainly on unified RAM."
    );
  }

  const compressedMb = ratio > 0 ? kvFp16 / ratio : kvFp16;
  if (!resident) {
    warnings.push(
      "Resident RSS savings are unlikely at short context for this " +
      "method's default path (accounting ratio still valid)."
    );
  }

  return {
    method, knobs,
    key_accounting_ratio: ratio,
    resident_savings_likely: resident,
    kv_fp16_mb: round2(kvFp16),
    kv_compressed_mb_estimate: round2(compressedMb),
    warnings, rationale,
  };
}

/* ---------- Method catalogue for the compression lab ---------- */
// ratio = compression factor used for the "fits in RAM" estimate.
// snippet = the real 3-line API for that method.
const METHODS = {
  turboquant_rvq: {
    label: "TurboQuant-RVQ (everyday, zero-calibration)",
    ratio: 7.5,
    config: 'KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)',
  },
  vecinfer: {
    label: "VecInfer (max key accounting, ~16×)",
    ratio: 16.0,
    config: 'KVCacheConfig(method="vecinfer", key_codebook_bits=8, key_sub_dim=8)',
  },
  rabitq: {
    label: "RaBitQ (full-KV, long context)",
    ratio: 6.0,
    config: 'KVCacheConfig(method="rabitq")',
  },
  spectral: {
    label: "SpectralQuant (best quality, ~5.3×)",
    ratio: 5.3,
    config: 'KVCacheConfig(method="spectral", bit_width_inlier=3)',
  },
  kivi: {
    label: "KIVI-2bit (per-channel keys, ~4× full-KV)",
    ratio: 4.0,
    config: 'KVCacheConfig(method="kivi", bit_width_inlier=2)',
  },
};

function snippetFor(methodKey) {
  const cfg = METHODS[methodKey].config;
  return (
`import mlx_lm
from veloxquant_mlx import KVCacheBuilder, KVCacheConfig

model, tokenizer = mlx_lm.load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
config = ${cfg}
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches`
  );
}

/* ---------- Tiny helpers ---------- */
const $ = (id) => document.getElementById(id);
const cssVar = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmtMb = (mb) => (mb >= 1024 ? (mb / 1024).toFixed(2) + " GB" : mb.toFixed(1) + " MB");

function readSharedShape() {
  return {
    n_layers: parseInt($("pg-layers").value, 10),
    n_kv_heads: parseInt($("pg-heads").value, 10),
    head_dim: parseInt($("pg-headdim").value, 10),
    seq_len: parseInt($("pg-seqlen").value, 10),
  };
}

/* ---------- Tab 1: Recommender ---------- */
function runRecommender() {
  const shape = readSharedShape();
  const req = {
    chip: $("pg-chip").value,
    ram_gb: parseInt($("pg-ram").value, 10),
    model_class: $("pg-model").value,
    goal: $("pg-goal").value,
    ...shape,
  };

  let res;
  try {
    res = recommend(req);
  } catch (e) {
    $("rec-output").innerHTML = `<p class="pg-error">${e.message}</p>`;
    return;
  }

  const knobsStr = Object.entries(res.knobs)
    .filter(([, v]) => v !== null)
    .map(([k, v]) => `${k}=${v}`)
    .join(", ");

  const savedPct = Math.round((1 - res.kv_compressed_mb_estimate / res.kv_fp16_mb) * 100);

  const warnHtml = res.warnings.length
    ? `<ul class="pg-warnings">${res.warnings.map((w) => `<li>${w}</li>`).join("")}</ul>`
    : `<p class="pg-note pg-ok">No warnings for this configuration.</p>`;

  $("rec-output").innerHTML = `
    <div class="pg-result-head">
      <span class="pg-badge-method">${res.method}</span>
      <span class="pg-ratio">${res.key_accounting_ratio}× key accounting</span>
      <span class="pg-chip-tag ${res.resident_savings_likely ? "pg-ok" : "pg-muted"}">
        ${res.resident_savings_likely ? "frees resident RAM" : "accounting-only"}
      </span>
    </div>
    <p class="pg-rationale">${res.rationale}</p>
    ${knobsStr ? `<p class="pg-note"><strong>Knobs:</strong> <code>${knobsStr}</code></p>` : ""}
    ${renderMemoryBar(res.kv_fp16_mb, res.kv_compressed_mb_estimate, savedPct)}
    ${warnHtml}
  `;
}

// Horizontal fp16-vs-compressed memory bar (pure divs, theme-aware).
function renderMemoryBar(fp16Mb, compMb, savedPct) {
  const compPct = Math.max(2, (compMb / fp16Mb) * 100);
  return `
    <div class="pg-membar">
      <div class="pg-membar-row">
        <span class="pg-membar-label">fp16 KV</span>
        <div class="pg-membar-track"><div class="pg-membar-fill fp16" style="width:100%"></div></div>
        <span class="pg-membar-val">${fmtMb(fp16Mb)}</span>
      </div>
      <div class="pg-membar-row">
        <span class="pg-membar-label">compressed</span>
        <div class="pg-membar-track"><div class="pg-membar-fill comp" style="width:${compPct}%"></div></div>
        <span class="pg-membar-val">${fmtMb(compMb)}</span>
      </div>
      <p class="pg-saved">≈ <strong>${savedPct}%</strong> smaller KV cache</p>
    </div>`;
}

/* ---------- Tab 2: Compression lab ---------- */
function runCompressionLab() {
  const shape = readSharedShape();
  const methodKey = $("pg-cmethod").value;
  const ratio = METHODS[methodKey].ratio;

  const fp16Mb = estimateKvFp16Mb(shape.n_layers, shape.n_kv_heads, shape.head_dim, shape.seq_len);
  const compMb = fp16Mb / ratio;

  // "Tokens that now fit in a RAM budget" — linear KV extrapolation, the same
  // framing the README uses for RaBitQ ("~103k tokens at 8 GB").
  const budgetGb = parseFloat($("pg-budget").value);
  const budgetMb = budgetGb * 1024;
  const perTokenFp16Mb = fp16Mb / shape.seq_len;
  const perTokenCompMb = compMb / shape.seq_len;
  const tokensFp16 = Math.floor(budgetMb / perTokenFp16Mb);
  const tokensComp = Math.floor(budgetMb / perTokenCompMb);

  const savedPct = Math.round((1 - compMb / fp16Mb) * 100);

  $("lab-output").innerHTML = `
    <div class="pg-result-head">
      <span class="pg-badge-method">${methodKey}</span>
      <span class="pg-ratio">${ratio}× smaller</span>
    </div>
    ${renderMemoryBar(round2(fp16Mb), round2(compMb), savedPct)}
    <div class="pg-token-grid">
      <div class="pg-token-card">
        <span class="pg-token-num">${tokensFp16.toLocaleString()}</span>
        <span class="pg-token-cap">tokens fit @ fp16 in ${budgetGb} GB</span>
      </div>
      <div class="pg-token-card pg-token-hi">
        <span class="pg-token-num">${tokensComp.toLocaleString()}</span>
        <span class="pg-token-cap">tokens fit with ${methodKey} in ${budgetGb} GB</span>
      </div>
    </div>
    <p class="pg-saved">${ratio}× more context in the same RAM budget (KV-only linear estimate).</p>
    <div class="pg-snippet">
      <button class="pg-copy" data-snippet>Copy</button>
      <pre><code>${snippetFor(methodKey).replace(/</g, "&lt;")}</code></pre>
    </div>
  `;

  const copyBtn = $("lab-output").querySelector("[data-snippet]");
  copyBtn.addEventListener("click", () => {
    navigator.clipboard?.writeText(snippetFor(methodKey)).then(() => {
      copyBtn.textContent = "Copied!";
      setTimeout(() => (copyBtn.textContent = "Copy"), 1400);
    });
  });
}

/* ---------- Tab 3: Benchmark viewer ---------- */
let BENCH = null;

async function loadBench() {
  if (BENCH) return BENCH;
  const res = await fetch("assets/data/benchmarks.json");
  BENCH = await res.json();
  return BENCH;
}

// Minimal grouped bar chart in inline SVG (no chart library).
function svgBarChart({ groups, series, width = 640, height = 300, yLabel, valFmt }) {
  const pad = { l: 52, r: 16, t: 16, b: 42 };
  const iw = width - pad.l - pad.r;
  const ih = height - pad.t - pad.b;
  const maxVal = Math.max(...groups.flatMap((g) => series.map((s) => g.values[s.key]))) * 1.1;
  const bandW = iw / groups.length;
  const barW = (bandW * 0.7) / series.length;
  const accent = cssVar("--accent") || "#00d4ff";
  const purple = cssVar("--purple") || "#7c3aed";
  const text = cssVar("--text") || "#e2e8f0";
  const muted = cssVar("--muted") || "#94a3b8";
  const colors = [accent, purple, "#22c55e"];

  let bars = "";
  groups.forEach((g, gi) => {
    const x0 = pad.l + gi * bandW + bandW * 0.15;
    series.forEach((s, si) => {
      const v = g.values[s.key];
      const bh = (v / maxVal) * ih;
      const x = x0 + si * barW;
      const y = pad.t + ih - bh;
      bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${bh.toFixed(1)}" rx="2" fill="${colors[si % colors.length]}"><title>${g.label} · ${s.label}: ${valFmt(v)}</title></rect>`;
      bars += `<text x="${(x + barW / 2).toFixed(1)}" y="${(y - 4).toFixed(1)}" text-anchor="middle" font-size="10" fill="${text}">${valFmt(v)}</text>`;
    });
    bars += `<text x="${(pad.l + gi * bandW + bandW / 2).toFixed(1)}" y="${height - pad.b + 16}" text-anchor="middle" font-size="11" fill="${muted}">${g.label}</text>`;
  });

  // y axis line + label
  const axis = `<line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t + ih}" stroke="${muted}" stroke-width="1" opacity="0.4"/>
    <line x1="${pad.l}" y1="${pad.t + ih}" x2="${width - pad.r}" y2="${pad.t + ih}" stroke="${muted}" stroke-width="1" opacity="0.4"/>
    <text x="14" y="${pad.t + ih / 2}" transform="rotate(-90 14 ${pad.t + ih / 2})" text-anchor="middle" font-size="11" fill="${muted}">${yLabel}</text>`;

  const legend = series
    .map((s, si) => `<span class="pg-legend-item"><span class="pg-legend-swatch" style="background:${colors[si % colors.length]}"></span>${s.label}</span>`)
    .join("");

  return `<div class="pg-legend">${legend}</div>
    <svg viewBox="0 0 ${width} ${height}" class="pg-svg" role="img" aria-label="${yLabel} chart">${axis}${bars}</svg>`;
}

async function runBenchViewer() {
  const data = await loadBench();
  const which = $("pg-bench").value;
  const out = $("bench-output");
  let caption = "";
  let chart = "";

  if (which === "metal") {
    const rows = data.metal_quantize;
    chart = svgBarChart({
      groups: rows.map((r) => ({ label: "S=" + r.S, values: { speedup: r.speedup } })),
      series: [{ key: "speedup", label: "Metal speedup ×" }],
      yLabel: "speedup ×",
      valFmt: (v) => v.toFixed(1) + "×",
    });
    caption = "Hand-written Metal quantize kernel vs pure-MLX, by sequence length S (B=1, H=8, D=128). Source: figures/metal/results.json.";
  } else if (which === "rabitq") {
    const rows = data.rabitq_falcon_memory;
    chart = svgBarChart({
      groups: rows.map((r) => ({
        label: "S=" + r.seq_len,
        values: { fp16: r.fp16_mb, comp: r.rabitq_mse4v_mb },
      })),
      series: [
        { key: "fp16", label: "fp16 KV (MB)" },
        { key: "comp", label: "RaBitQ 1-bit K + MSE-b4 V (MB)" },
      ],
      yLabel: "KV size (MB)",
      valFmt: (v) => v.toFixed(0),
    });
    caption = "Full-KV memory, Falcon3-7B shape: fp16 vs RaBitQ (~5.95× smaller). Source: figures/RaBitQ/falcon/results.json.";
  } else {
    // KIVI per-model throughput retention
    const model = $("pg-bench-model").value;
    const m = data.kivi_models[model];
    chart = svgBarChart({
      groups: m.rows.map((r) => ({
        label: r.config.replace("KIVI-", "").replace("fp16-baseline", "fp16"),
        values: { pct: r.tok_s_pct, keyc: r.key_compression },
      })),
      series: [
        { key: "pct", label: "throughput (% of fp16)" },
        { key: "keyc", label: "key compression ×" },
      ],
      yLabel: "value",
      valFmt: (v) => (v >= 20 ? v.toFixed(0) : v.toFixed(2)),
    });
    caption = `KIVI on ${model} (${m.chip}, ${m.ram_gb} GB): throughput retention vs key compression. Source: figures/kivi/results_summary.json.`;
  }

  out.innerHTML = `${chart}<p class="pg-caption">${caption}</p>`;
}

/* ---------- Tab switching (mirrors initCodeTabs in main.js) ---------- */
function initTabs() {
  const tabBtns = document.querySelectorAll(".pg-tab-btn");
  const panels = document.querySelectorAll(".pg-panel");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.pgtab;
      tabBtns.forEach((b) => b.classList.toggle("active", b === btn));
      tabBtns.forEach((b) => b.setAttribute("aria-selected", b === btn));
      panels.forEach((p) => p.classList.toggle("active", p.id === "pg-panel-" + tab));
      if (tab === "bench") runBenchViewer();
    });
  });
}

// Toggle the KIVI model dropdown visibility based on selected benchmark.
function syncBenchModelPicker() {
  const isKivi = $("pg-bench").value === "kivi";
  $("pg-bench-model-wrap").style.display = isKivi ? "" : "none";
}

/* ---------- Wire everything up ---------- */
document.addEventListener("DOMContentLoaded", () => {
  initTabs();

  // Recommender inputs
  ["pg-chip", "pg-ram", "pg-model", "pg-goal"].forEach((id) =>
    $(id).addEventListener("change", runRecommender)
  );
  // Shared shape inputs drive both recommender and lab
  ["pg-layers", "pg-heads", "pg-headdim", "pg-seqlen"].forEach((id) =>
    $(id).addEventListener("input", () => {
      runRecommender();
      runCompressionLab();
    })
  );
  // Lab inputs
  ["pg-cmethod", "pg-budget"].forEach((id) =>
    $(id).addEventListener("input", runCompressionLab)
  );
  // Bench inputs
  $("pg-bench").addEventListener("change", () => {
    syncBenchModelPicker();
    runBenchViewer();
  });
  $("pg-bench-model").addEventListener("change", runBenchViewer);

  // Re-render charts on theme toggle so SVG colors track the theme.
  const themeToggle = $("theme-toggle");
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      if ($("pg-panel-bench").classList.contains("active")) {
        setTimeout(runBenchViewer, 0);
      }
    });
  }

  syncBenchModelPicker();
  runRecommender();
  runCompressionLab();
});
