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

/* ---------- Model presets — real architectures of the benchmarked models ----
   One click fills the shared shape fields (class / layers / KV heads / head dim)
   so a newcomer never has to know their model's internals. Values match the
   configs of the mlx-community 4-bit checkpoints used across figures/.
   -------------------------------------------------------------------------- */
const MODEL_PRESETS = [
  { id: "llama32-3b",  name: "Llama-3.2-3B",  model_class: "3B",  n_layers: 28, n_kv_heads: 8,  head_dim: 128 },
  { id: "mistral-7b",  name: "Mistral-7B",    model_class: "7B",  n_layers: 32, n_kv_heads: 8,  head_dim: 128 },
  { id: "qwen25-7b",   name: "Qwen2.5-7B",    model_class: "7B",  n_layers: 28, n_kv_heads: 4,  head_dim: 128 },
  { id: "qwen3-8b",    name: "Qwen3-8B",      model_class: "7B",  n_layers: 36, n_kv_heads: 8,  head_dim: 128 },
  { id: "phi4",        name: "Phi-4",         model_class: "14B", n_layers: 40, n_kv_heads: 10, head_dim: 128 },
  { id: "falcon3-7b",  name: "Falcon3-7B",    model_class: "7B",  n_layers: 28, n_kv_heads: 4,  head_dim: 256 },
  { id: "gemma3-4b",   name: "Gemma-3-4B",    model_class: "3B",  n_layers: 34, n_kv_heads: 4,  head_dim: 256 },
  { id: "qwen25-32b",  name: "Qwen2.5-32B",   model_class: "32B", n_layers: 64, n_kv_heads: 8,  head_dim: 128 },
];

const PRESET_FIELDS = [
  ["pg-model", "model_class"],
  ["pg-layers", "n_layers"],
  ["pg-heads", "n_kv_heads"],
  ["pg-headdim", "head_dim"],
];

// Render the preset chip row (+ a Custom chip that reflects hand-edited shapes).
function renderPresets() {
  const row = $("pg-preset-row");
  if (!row) return;
  row.innerHTML =
    MODEL_PRESETS.map(
      (p) =>
        `<button type="button" class="pg-preset-chip" data-preset="${p.id}"
           title="${p.n_layers} layers · ${p.n_kv_heads} KV heads · head_dim ${p.head_dim}">
           ${p.name}
         </button>`
    ).join("") +
    `<button type="button" class="pg-preset-chip pg-preset-custom" data-preset="custom"
       title="Your own model shape">Custom</button>`;

  row.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.dataset.preset === "custom") return; // Custom is display-only
      applyPreset(btn.dataset.preset);
    });
  });
}

// Fill the shared fields from a preset, mark it active, and recompute.
function applyPreset(id) {
  const p = MODEL_PRESETS.find((x) => x.id === id);
  if (!p) return;
  PRESET_FIELDS.forEach(([elId, key]) => {
    $(elId).value = p[key];
  });
  markActivePreset();
  if (typeof updateAdvancedHint === "function") updateAdvancedHint();
  runRecommender();
  runCompressionLab();
}

// Highlight whichever preset matches the current fields, else the Custom chip.
function markActivePreset() {
  const cur = {
    model_class: $("pg-model").value,
    n_layers: parseInt($("pg-layers").value, 10),
    n_kv_heads: parseInt($("pg-heads").value, 10),
    head_dim: parseInt($("pg-headdim").value, 10),
  };
  const match = MODEL_PRESETS.find(
    (p) =>
      p.model_class === cur.model_class &&
      p.n_layers === cur.n_layers &&
      p.n_kv_heads === cur.n_kv_heads &&
      p.head_dim === cur.head_dim
  );
  document.querySelectorAll(".pg-preset-chip").forEach((c) => {
    const isActive = match ? c.dataset.preset === match.id : c.dataset.preset === "custom";
    c.classList.toggle("active", isActive);
    c.setAttribute("aria-pressed", isActive ? "true" : "false");
  });
}

/* ---------- Method catalogue for the compression lab ---------- */
// ratio = compression factor used for the "fits in RAM" estimate.
// snippet = the real 3-line API for that method.
const METHODS = {
  turboquant_rvq: {
    label: "TurboQuant-RVQ (everyday, zero-calibration)",
    short: "TurboQuant-RVQ",
    blurb: "Zero-calibration everyday default.",
    ratio: 7.5,
    config: 'KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)',
  },
  vecinfer: {
    label: "VecInfer (max key accounting, ~16×)",
    short: "VecInfer",
    blurb: "Max key accounting; needs calibration.",
    ratio: 16.0,
    config: 'KVCacheConfig(method="vecinfer", key_codebook_bits=8, key_sub_dim=8)',
  },
  rabitq: {
    label: "RaBitQ (full-KV, long context)",
    short: "RaBitQ",
    blurb: "Full-KV compression for long context.",
    ratio: 6.0,
    config: 'KVCacheConfig(method="rabitq")',
  },
  spectral: {
    label: "SpectralQuant (best quality, ~5.3×)",
    short: "SpectralQuant",
    blurb: "Best reconstruction quality.",
    ratio: 5.3,
    config: 'KVCacheConfig(method="spectral", bit_width_inlier=3)',
  },
  kivi: {
    label: "KIVI-2bit (per-channel keys, ~4× full-KV)",
    short: "KIVI-2bit",
    blurb: "Per-channel keys, throughput-neutral.",
    ratio: 4.0,
    config: 'KVCacheConfig(method="kivi", bit_width_inlier=2)',
  },
};

/* Order the method cards are shown in the lab (selection order). */
const METHOD_ORDER = ["turboquant_rvq", "vecinfer", "rabitq", "spectral", "kivi"];

/* A recommended method → which benchmark chart best backs it up.
   streaming_llm has no dedicated chart; fall back to the Metal speedup view. */
const METHOD_TO_BENCH = {
  turboquant_rvq: "metal",
  vecinfer: "metal",
  rabitq: "rabitq",
  spectral: "metal",
  kivi: "kivi",
  streaming_llm: "metal",
};

/* Nearest lab method for a recommender output (recommender emits methods
   the lab doesn't list one-for-one — map them to the closest lab entry). */
const REC_TO_LAB_METHOD = {
  turboquant_rvq: "turboquant_rvq",
  vecinfer: "vecinfer",
  rabitq: "rabitq",
  spectral: "spectral",
  kivi: "kivi",
  streaming_llm: "rabitq",
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

// Respect the user's motion preference for count-ups / bar animations.
const REDUCED_MOTION =
  window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function readSharedShape() {
  return {
    n_layers: parseInt($("pg-layers").value, 10),
    n_kv_heads: parseInt($("pg-heads").value, 10),
    head_dim: parseInt($("pg-headdim").value, 10),
    seq_len: parseInt($("pg-seqlen").value, 10),
  };
}

/* Animate a numeric element from its current value up to `to`.
   Purely presentational — the final rendered value is always the real one.
   `fmt` formats the running value; skipped entirely under reduced-motion. */
function countUp(el, to, fmt = (v) => String(Math.round(v))) {
  if (!el) return;
  if (REDUCED_MOTION) { el.textContent = fmt(to); return; }
  const from = 0;
  const dur = 480;
  const start = performance.now();
  function step(now) {
    const t = Math.min(1, (now - start) / dur);
    const eased = 1 - Math.pow(1 - t, 3); // easeOutCubic
    el.textContent = fmt(from + (to - from) * eased);
    if (t < 1) requestAnimationFrame(step);
    else el.textContent = fmt(to);
  }
  requestAnimationFrame(step);
}

/* Human-scale caption for a token count — a labelled ~450 tokens/page heuristic.
   Anchors are approximate published word/token counts, NOT measurements. */
const TOKENS_PER_PAGE = 450;
const BOOK_ANCHORS = [
  { tokens: 4_800_000, label: "the entire Wikipedia featured-article set" },
  { tokens: 780_000, label: "the full Lord of the Rings trilogy" },
  { tokens: 155_000, label: "a 340-page novel" },
  { tokens: 45_000, label: "a 100-page report" },
  { tokens: 9_000, label: "a 20-page paper" },
  { tokens: 1_800, label: "a 4-page memo" },
];
function bookScale(tokens) {
  const pages = Math.max(1, Math.round(tokens / TOKENS_PER_PAGE));
  const anchor = BOOK_ANCHORS.find((a) => tokens >= a.tokens);
  const tangible = anchor ? anchor.label : "a short note";
  return { pages, tangible };
}

/* ---------- Shareable URL state (optional, refresh-safe) ---------- */
const URL_KEYS = {
  chip: "pg-chip", ram: "pg-ram", goal: "pg-goal",
  model: "pg-model", seq: "pg-seqlen", layers: "pg-layers",
  heads: "pg-heads", headdim: "pg-headdim",
  method: "pg-cmethod", budget: "pg-budget",
};

// Restore setup from ?chip=…&ram=… before first compute. Silently ignores
// values the control doesn't offer, so a stale link can't break the page.
function restoreFromUrl() {
  const params = new URLSearchParams(window.location.search);
  if (![...params.keys()].length) return;
  Object.entries(URL_KEYS).forEach(([qkey, elId]) => {
    if (!params.has(qkey)) return;
    const el = $(elId);
    if (!el) return;
    const val = params.get(qkey);
    if (el.tagName === "SELECT") {
      if ([...el.options].some((o) => o.value === val)) el.value = val;
    } else {
      el.value = val;
    }
  });
}

// Build a shareable URL that restores the current setup on load.
function buildShareUrl() {
  const params = new URLSearchParams();
  Object.entries(URL_KEYS).forEach(([qkey, elId]) => {
    const el = $(elId);
    if (el && el.value !== "") params.set(qkey, el.value);
  });
  return `${window.location.origin}${window.location.pathname}?${params.toString()}`;
}

/* ---------- Tab 1: Recommender (hero card) ---------- */
// Escape user-derived strings before injecting into innerHTML.
const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ---------- Dashboard-pattern presentational helpers ----------
   Purely visual containers wrapped around the SAME honest engine
   data. No math lives here — every value passed in traces to the
   recommender / committed benchmark data. -------------------- */

// Tiny inline-SVG icon set (no icon font, no external requests).
// stroke=currentColor so each icon inherits its band/row color.
const PG_ICONS = {
  cpu: '<svg class="pg-ic" viewBox="0 0 24 24" aria-hidden="true"><rect x="7" y="7" width="10" height="10" rx="1.5"/><path d="M4 9h1M4 12h1M4 15h1M19 9h1M19 12h1M19 15h1M9 4v1M12 4v1M15 4v1M9 19v1M12 19v1M15 19v1"/></svg>',
  memory: '<svg class="pg-ic" viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="7" width="18" height="10" rx="1.5"/><path d="M7 17v2M11 17v2M15 17v2M9 10v4M13 10v4"/></svg>',
  chart: '<svg class="pg-ic" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 4v16h16"/><rect x="7" y="11" width="3" height="6"/><rect x="12" y="7" width="3" height="10"/><rect x="17" y="13" width="3" height="4"/></svg>',
  terminal: '<svg class="pg-ic" viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 3-3 3M13 15h4"/></svg>',
  warning: '<svg class="pg-ic" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17v.5"/></svg>',
  shape: '<svg class="pg-ic" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 3v18M4 7.5l8 4.5 8-4.5"/></svg>',
  code: '<svg class="pg-ic" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 8l-4 4 4 4M16 8l4 4-4 4M13 6l-2 12"/></svg>',
  copy: '<svg class="pg-ic pg-ic-copy" viewBox="0 0 24 24" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/></svg>',
  check: '<svg class="pg-ic pg-ic-check" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12l5 5L20 7"/></svg>',
};

// A labeled section band: subtle strip with an inline icon + uppercase
// title, then the block's content beneath. `iconKey` indexes PG_ICONS.
function renderBand(title, iconKey, bodyHtml) {
  const icon = PG_ICONS[iconKey] || "";
  return `<div class="pg-band">${icon}<span class="pg-band-title">${esc(title)}</span></div>
    <div class="pg-band-body">${bodyHtml}</div>`;
}

// One stat card. `opts.hi` accent-highlights it; `opts.id`/`opts.to`
// let the caller count the number up after paint (reusing countUp).
function renderStatCard({ label, value, unit = "", hi = false, id = "", to = "" }) {
  const numAttrs = (id ? ` id="${esc(id)}"` : "") + (to !== "" ? ` data-to="${esc(to)}"` : "");
  const unitHtml = unit ? ` <span class="pg-stat-unit">${esc(unit)}</span>` : "";
  return `<div class="pg-stat${hi ? " pg-stat-hi" : ""}">
    <span class="pg-stat-lbl">${esc(label)}</span>
    <span class="pg-stat-num"${numAttrs}>${esc(value)}${unitHtml}</span>
  </div>`;
}

// A status card (a state, not a number) — e.g. the RESIDENT RAM tile.
function renderStatStatus({ label, text, sub, ok }) {
  return `<div class="pg-stat">
    <span class="pg-stat-lbl">${esc(label)}</span>
    <span class="pg-stat-status ${ok ? "is-ok" : "is-muted"}">
      <span class="pg-stat-dot" aria-hidden="true"></span>${esc(text)}
    </span>
    <span class="pg-stat-unit">${esc(sub)}</span>
  </div>`;
}

function renderStatRow(cardsHtml) {
  return `<div class="pg-stat-row">${cardsHtml}</div>`;
}

// A copyable value row: label · mono value pill · copy icon-button.
// `copyValue` is what lands on the clipboard (defaults to the shown value).
// Copy buttons are wired after injection by wireCopyRows().
function renderCopyRow(label, value, ariaLabel, copyValue) {
  const toCopy = copyValue !== undefined ? copyValue : value;
  return `<div class="pg-row">
    <span class="pg-row-lbl">${esc(label)}</span>
    <span class="pg-row-right">
      <code class="pg-row-val">${esc(value)}</code>
      <button type="button" class="pg-row-copy" aria-label="${esc(ariaLabel)}"
        data-copy="${esc(toCopy)}">${PG_ICONS.copy}${PG_ICONS.check}</button>
    </span>
  </div>`;
}

// Wire every copy-row button inside `root` to the clipboard + the same
// "flip to check" success feedback used elsewhere. Success is announced
// through the surrounding aria-live="polite" region.
function wireCopyRows(root) {
  root.querySelectorAll(".pg-row-copy").forEach((btn) => {
    btn.addEventListener("click", () => {
      navigator.clipboard?.writeText(btn.dataset.copy).then(() => {
        btn.classList.add("copied");
        setTimeout(() => btn.classList.remove("copied"), 1400);
      });
    });
  });
}

// Build the real `veloxquant recommend` CLI command from the current setup.
// Flags verified against veloxquant_mlx/cli/recommend.py — do NOT invent flags:
//   --chip --ram-gb --model-class --goal --seq-len --n-layers --n-kv-heads --head-dim
function buildCliCommand(req) {
  return [
    "veloxquant recommend",
    `--chip ${req.chip}`,
    `--ram-gb ${req.ram_gb}`,
    `--model-class ${req.model_class}`,
    `--goal ${req.goal}`,
    `--seq-len ${req.seq_len}`,
    `--n-layers ${req.n_layers}`,
    `--n-kv-heads ${req.n_kv_heads}`,
    `--head-dim ${req.head_dim}`,
  ].join(" ");
}

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
    $("rec-output").innerHTML = `<p class="pg-error">${esc(e.message)}</p>`;
    updateHeadline(null);
    return;
  }

  const knobEntries = Object.entries(res.knobs).filter(([, v]) => v !== null);
  const knobsHtml = knobEntries.length
    ? `<div class="pg-knobs">${knobEntries
        .map(([k, v]) => `<code class="pg-knob-chip">${esc(k)}=${esc(v)}</code>`)
        .join("")}</div>`
    : "";
  // Copyable one-line summary of the same knobs (Pattern 4).
  const knobsInline = knobEntries.length
    ? knobEntries.map(([k, v]) => `${k}=${v}`).join(", ")
    : "";

  const savedPct = Math.round((1 - res.kv_compressed_mb_estimate / res.kv_fp16_mb) * 100);
  const resident = res.resident_savings_likely;
  const cli = buildCliCommand(req);

  const warnHtml = res.warnings.length
    ? `<div class="pg-callouts">${res.warnings
        .map(
          (w) => `<div class="pg-callout pg-callout-warn">
            <span class="pg-callout-ic" aria-hidden="true">▲</span>
            <span>${esc(w)}</span></div>`
        )
        .join("")}</div>`
    : `<div class="pg-callouts"><div class="pg-callout pg-callout-ok">
        <span class="pg-callout-ic" aria-hidden="true">✓</span>
        <span>No warnings for this configuration.</span></div></div>`;

  // Pattern 1 — headline stat-card row (ratio · saved · resident state).
  const statRow = renderStatRow(
    renderStatCard({
      label: "Key-accounting ratio",
      value: "0×",
      id: "pg-hero-ratio",
      to: res.key_accounting_ratio,
    }) +
    renderStatCard({
      label: "KV cache saved",
      value: "0%",
      id: "pg-hero-saved",
      to: savedPct,
    }) +
    renderStatStatus({
      label: "Resident RAM",
      text: resident ? "Frees RAM" : "Accounting-only",
      sub: resident ? "full-KV path" : "ratio still valid",
      ok: resident,
    })
  );

  // Pattern 3 — split-metric MODEL SHAPE readout above the memory bar.
  const splitCard = `<div class="pg-split">
    <div class="pg-split-cell">
      <span class="pg-stat-lbl">KV @ fp16</span>
      <span class="pg-stat-num">${esc(fmtMb(res.kv_fp16_mb))}</span>
    </div>
    <div class="pg-split-cell">
      <span class="pg-stat-lbl">KV compressed</span>
      <span class="pg-stat-num">${esc(fmtMb(res.kv_compressed_mb_estimate))}</span>
    </div>
  </div>`;

  // Pattern 4 — copyable rows: reproduce-in-terminal CLI + knobs.
  const reproRows =
    renderCopyRow("Reproduce in terminal", cli, "Copy CLI command") +
    (knobsInline ? renderCopyRow("Knobs", knobsInline, "Copy knobs") : "");

  $("rec-output").innerHTML = `
    <div class="pg-hero">
      <div class="pg-hero-head">
        <div class="pg-hero-badge">
          <span class="pg-hero-badge-k">Recommended</span>
          <span class="pg-badge-method">${esc(res.method)}</span>
        </div>
        <span class="pg-pill ${resident ? "pg-pill-ok" : "pg-pill-muted"}">
          ${resident ? "frees resident RAM" : "accounting-only"}
          <button type="button" class="pg-tip" tabindex="0"
            aria-label="What does this mean?"
            data-tip="${resident
              ? "This path actually shrinks the resident KV footprint in RAM, not just the accounting ratio."
              : "The compression ratio is real, but the default path may keep an fp16 parent cache — so resident RAM may not drop at short context."}">?</button>
        </span>
      </div>

      ${statRow}

      ${renderBand("Recommended method", "cpu",
        `<p class="pg-rationale">${esc(res.rationale)}</p>${knobsHtml}`)}

      ${renderBand("Memory vs context", "memory",
        splitCard + renderMemoryBar(res.kv_fp16_mb, res.kv_compressed_mb_estimate, savedPct, "rec"))}

      ${renderBand("Reproduce", "terminal", `<div class="pg-rows">${reproRows}</div>`)}

      ${renderBand("Notes & warnings", "warning", warnHtml)}

      <div class="pg-cta-row">
        <button type="button" class="pg-cta pg-cta-primary" id="pg-cta-lab"
          data-method="${esc(REC_TO_LAB_METHOD[res.method] || "turboquant_rvq")}">
          Open in Compression Lab →
        </button>
        <button type="button" class="pg-cta pg-cta-ghost" id="pg-cta-bench"
          data-bench="${esc(METHOD_TO_BENCH[res.method] || "metal")}">
          See the benchmark →
        </button>
      </div>
    </div>
  `;

  // Count the stat cards up and animate the bar after paint.
  countUp($("pg-hero-ratio"), res.key_accounting_ratio, (v) => {
    const r = Math.round(v * 10) / 10;
    return (Number.isInteger(r) ? r.toFixed(0) : r.toFixed(1)) + "×";
  });
  countUp($("pg-hero-saved"), savedPct, (v) => Math.round(v) + "%");
  animateMemoryBar("rec", savedPct);
  wireCopyRows($("rec-output"));

  // Wire the two hero CTAs — carry the method / benchmark across tabs.
  $("pg-cta-lab").addEventListener("click", (e) => {
    selectMethod(e.currentTarget.dataset.method);
    switchTab("lab");
  });
  $("pg-cta-bench").addEventListener("click", (e) => {
    // Route through selectBench so the segmented control re-marks itself,
    // then reveal the panel (switchTab re-renders the chart).
    selectBench(e.currentTarget.dataset.bench);
    switchTab("bench");
  });

  updateHeadline(res);
}

/* Horizontal fp16-vs-compressed memory bar (pure divs, theme-aware).
   The compressed fill starts at 100% and is animated down to compPct by
   animateMemoryBar(scope) after the node is in the DOM. */
function renderMemoryBar(fp16Mb, compMb, savedPct, scope) {
  const compPct = Math.max(2, (compMb / fp16Mb) * 100);
  const startPct = REDUCED_MOTION ? compPct : 100;
  return `
    <div class="pg-membar" data-scope="${scope}" data-comp-pct="${compPct.toFixed(2)}">
      <div class="pg-membar-row">
        <span class="pg-membar-label">fp16 KV</span>
        <div class="pg-membar-track"><div class="pg-membar-fill fp16" style="width:100%"></div></div>
        <span class="pg-membar-val">${fmtMb(fp16Mb)}</span>
      </div>
      <div class="pg-membar-row">
        <span class="pg-membar-label">compressed</span>
        <div class="pg-membar-track">
          <div class="pg-membar-fill comp" style="width:${startPct}%"></div>
        </div>
        <span class="pg-membar-val">${fmtMb(compMb)}</span>
      </div>
      <p class="pg-saved">≈ <strong class="pg-saved-num" id="pg-saved-${scope}">${savedPct}%</strong> smaller KV cache</p>
    </div>`;
}

// Animate the compressed bar full → compressed and count the "% smaller" up.
function animateMemoryBar(scope, savedPct) {
  const bar = document.querySelector(`.pg-membar[data-scope="${scope}"]`);
  if (!bar) return;
  const fill = bar.querySelector(".pg-membar-fill.comp");
  const target = bar.dataset.compPct + "%";
  const savedEl = $("pg-saved-" + scope);
  if (REDUCED_MOTION) {
    if (fill) fill.style.width = target;
    if (savedEl) savedEl.textContent = savedPct + "%";
    return;
  }
  // Force layout at 100%, then transition to the compressed width.
  if (fill) {
    fill.style.width = "100%";
    // eslint-disable-next-line no-unused-expressions
    fill.offsetWidth; // reflow so the width change animates
    requestAnimationFrame(() => (fill.style.width = target));
  }
  countUp(savedEl, savedPct, (v) => Math.round(v) + "%");
}

/* Animated headline metric — "Fit up to N× more context on your Mac".
   N is the recommender's own ratio; null clears it (on an input error). */
function updateHeadline(res) {
  const el = $("pg-headline");
  if (!el) return;
  if (!res) { el.textContent = ""; return; }
  const r = res.key_accounting_ratio;
  const rStr = Number.isInteger(r) ? r.toFixed(0) : r.toFixed(1);
  el.innerHTML = `Fit up to <strong>${rStr}×</strong> more context on your Mac.`;
}

/* ---------- Tab 2: Compression lab ---------- */

// Render the method selector as a row of radio cards (name · ratio · blurb).
function renderMethodCards() {
  const wrap = $("pg-method-cards");
  if (!wrap) return;
  const selected = $("pg-cmethod").value;
  wrap.innerHTML = METHOD_ORDER.map((key) => {
    const m = METHODS[key];
    const isSel = key === selected;
    return `<button type="button" class="pg-method-card ${isSel ? "active" : ""}"
      role="radio" aria-checked="${isSel}" data-method="${key}" tabindex="${isSel ? "0" : "-1"}">
      <span class="pg-method-name">${m.short}</span>
      <span class="pg-method-ratio">${m.ratio}×</span>
      <span class="pg-method-blurb">${m.blurb}</span>
    </button>`;
  }).join("");

  wrap.querySelectorAll(".pg-method-card").forEach((card) => {
    card.addEventListener("click", () => selectMethod(card.dataset.method));
    // Arrow-key navigation within the radiogroup.
    card.addEventListener("keydown", (e) => {
      if (!["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp"].includes(e.key)) return;
      e.preventDefault();
      const idx = METHOD_ORDER.indexOf($("pg-cmethod").value);
      const dir = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : -1;
      const next = (idx + dir + METHOD_ORDER.length) % METHOD_ORDER.length;
      selectMethod(METHOD_ORDER[next]);
      wrap.querySelector(".pg-method-card.active")?.focus();
    });
  });
}

// Single source of truth for the selected method: update hidden input,
// re-mark the cards, and recompute the lab.
function selectMethod(key) {
  if (!METHODS[key]) return;
  $("pg-cmethod").value = key;
  const wrap = $("pg-method-cards");
  if (wrap) {
    wrap.querySelectorAll(".pg-method-card").forEach((card) => {
      const on = card.dataset.method === key;
      card.classList.toggle("active", on);
      card.setAttribute("aria-checked", on ? "true" : "false");
      card.tabIndex = on ? 0 : -1;
    });
  }
  runCompressionLab();
}

function runCompressionLab() {
  const shape = readSharedShape();
  const methodKey = $("pg-cmethod").value;
  const m = METHODS[methodKey];
  const ratio = m.ratio;

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
  const book = bookScale(tokensComp);

  // Pattern 1 — headline 2-card stat row: tokens @ fp16 vs with <method>.
  const statRow = renderStatRow(
    renderStatCard({
      label: `Tokens @ fp16 · ${budgetGb} GB`,
      value: tokensFp16.toLocaleString(),
    }) +
    renderStatCard({
      label: `Tokens with ${m.short} · ${budgetGb} GB`,
      value: tokensComp.toLocaleString(),
      hi: true,
      id: "pg-tok-comp",
      to: tokensComp,
    })
  );

  $("lab-output").innerHTML = `
    <div class="pg-result-head">
      <span class="pg-badge-method">${esc(methodKey)}</span>
      <span class="pg-ratio">${ratio}× smaller</span>
    </div>

    ${statRow}

    ${renderBand("Context that fits", "chart",
      `<p class="pg-book">≈ <strong>${book.pages.toLocaleString()} pages</strong> — about ${book.tangible}
        <span class="pg-approx">approx · ~${TOKENS_PER_PAGE} tokens/page</span></p>
       <p class="pg-saved">${ratio}× more context in the same RAM budget (KV-only linear estimate).</p>`)}

    ${renderBand("Memory", "memory",
      renderMemoryBar(round2(fp16Mb), round2(compMb), savedPct, "lab"))}

    ${renderBand("Integration snippet", "code",
      `<div class="pg-snippet">
        <div class="pg-snippet-head">
          <span class="pg-snippet-note">Runs on any <code>mlx_lm</code> model</span>
          <button class="pg-copy" data-snippet>Copy</button>
        </div>
        <pre><code class="pg-code">${highlightSnippet(snippetFor(methodKey))}</code></pre>
      </div>`)}
  `;

  animateMemoryBar("lab", savedPct);
  countUp($("pg-tok-comp"), tokensComp, (v) => Math.round(v).toLocaleString());

  const copyBtn = $("lab-output").querySelector("[data-snippet]");
  copyBtn.addEventListener("click", () => {
    navigator.clipboard?.writeText(snippetFor(methodKey)).then(() => {
      copyBtn.textContent = "Copied!";
      copyBtn.classList.add("copied");
      setTimeout(() => {
        copyBtn.textContent = "Copy";
        copyBtn.classList.remove("copied");
      }, 1400);
    });
  });
}

// Light syntax tinting for the Python snippet (comments muted, strings accent).
// Escapes first, then wraps token spans — presentational only.
function highlightSnippet(code) {
  return esc(code)
    .split("\n")
    .map((line) => {
      if (line.trimStart().startsWith("#"))
        return `<span class="tok-comment">${line}</span>`;
      // Highlight double-quoted string literals.
      return line.replace(/(&quot;[^&]*?&quot;)/g, '<span class="tok-str">$1</span>');
    })
    .join("\n");
}

/* ---------- Tab 3: Benchmark viewer ---------- */
let BENCH = null;

async function loadBench() {
  if (BENCH) return BENCH;
  const res = await fetch("assets/data/benchmarks.json");
  BENCH = await res.json();
  return BENCH;
}

// "Nice" upper bound + step for the y axis, so gridlines land on round values.
function niceTicks(max, count = 4) {
  const raw = max / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const top = Math.ceil(max / step) * step;
  const ticks = [];
  for (let t = 0; t <= top + 1e-9; t += step) ticks.push(t);
  return { top, ticks };
}

// Minimal grouped bar chart in inline SVG (no chart library).
// Bars carry data-h and start at height 0; growBars() animates them in.
function svgBarChart({ groups, series, width = 640, height = 300, yLabel, valFmt }) {
  const pad = { l: 54, r: 16, t: 18, b: 44 };
  const iw = width - pad.l - pad.r;
  const ih = height - pad.t - pad.b;
  const dataMax = Math.max(...groups.flatMap((g) => series.map((s) => g.values[s.key])));
  const { top: maxVal, ticks } = niceTicks(dataMax * 1.05);
  const bandW = iw / groups.length;
  const barW = (bandW * 0.7) / series.length;
  const accent = cssVar("--accent") || "#00d4ff";
  const purple = cssVar("--purple") || "#7c3aed";
  const text = cssVar("--text") || "#e2e8f0";
  const muted = cssVar("--muted") || "#94a3b8";
  const colors = [accent, purple, "#22c55e"];

  // Horizontal gridlines + y tick labels.
  let grid = "";
  ticks.forEach((tv) => {
    const y = pad.t + ih - (tv / maxVal) * ih;
    grid += `<line x1="${pad.l}" y1="${y.toFixed(1)}" x2="${width - pad.r}" y2="${y.toFixed(1)}" stroke="${muted}" stroke-width="1" opacity="0.14"/>`;
    grid += `<text x="${pad.l - 8}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" font-size="10" fill="${muted}">${valFmt(tv)}</text>`;
  });

  let bars = "";
  groups.forEach((g, gi) => {
    const x0 = pad.l + gi * bandW + bandW * 0.15;
    series.forEach((s, si) => {
      const v = g.values[s.key];
      const bh = (v / maxVal) * ih;
      const x = x0 + si * barW;
      const y = pad.t + ih - bh;
      const baseY = pad.t + ih;
      bars += `<rect class="pg-bar" x="${x.toFixed(1)}" y="${baseY.toFixed(1)}" width="${barW.toFixed(1)}" height="0" data-y="${y.toFixed(1)}" data-h="${bh.toFixed(1)}" rx="2" fill="${colors[si % colors.length]}"><title>${g.label} · ${s.label}: ${valFmt(v)}</title></rect>`;
      bars += `<text class="pg-bar-val" x="${(x + barW / 2).toFixed(1)}" y="${(y - 4).toFixed(1)}" text-anchor="middle" font-size="10" fill="${text}" opacity="0">${valFmt(v)}</text>`;
    });
    bars += `<text x="${(pad.l + gi * bandW + bandW / 2).toFixed(1)}" y="${height - pad.b + 16}" text-anchor="middle" font-size="11" fill="${muted}">${g.label}</text>`;
  });

  // axes + rotated y label
  const axis = `<line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t + ih}" stroke="${muted}" stroke-width="1" opacity="0.4"/>
    <line x1="${pad.l}" y1="${pad.t + ih}" x2="${width - pad.r}" y2="${pad.t + ih}" stroke="${muted}" stroke-width="1" opacity="0.4"/>
    <text x="14" y="${pad.t + ih / 2}" transform="rotate(-90 14 ${pad.t + ih / 2})" text-anchor="middle" font-size="11" fill="${muted}">${yLabel}</text>`;

  const legend = series
    .map((s, si) => `<span class="pg-legend-item"><span class="pg-legend-swatch" style="background:${colors[si % colors.length]}"></span>${s.label}</span>`)
    .join("");

  return `<div class="pg-legend">${legend}</div>
    <div class="pg-svg-wrap"><svg viewBox="0 0 ${width} ${height}" class="pg-svg" role="img" aria-label="${yLabel} chart">${grid}${axis}${bars}</svg></div>`;
}

// Grow bars from the baseline into place, then fade in the value labels.
function growBars(container) {
  const rects = container.querySelectorAll(".pg-bar");
  const labels = container.querySelectorAll(".pg-bar-val");
  if (REDUCED_MOTION) {
    rects.forEach((r) => { r.setAttribute("y", r.dataset.y); r.setAttribute("height", r.dataset.h); });
    labels.forEach((l) => l.setAttribute("opacity", "1"));
    return;
  }
  requestAnimationFrame(() => {
    rects.forEach((r) => {
      r.style.transition = "y 0.5s cubic-bezier(.22,1,.36,1), height 0.5s cubic-bezier(.22,1,.36,1)";
      r.setAttribute("y", r.dataset.y);
      r.setAttribute("height", r.dataset.h);
    });
    setTimeout(() => labels.forEach((l) => {
      l.style.transition = "opacity 0.3s ease";
      l.setAttribute("opacity", "1");
    }), 260);
  });
}

async function runBenchViewer() {
  const out = $("bench-output");
  // Skeleton shimmer while the JSON loads (only visible on the first fetch).
  if (!BENCH) {
    out.innerHTML = `<div class="pg-skeleton" aria-hidden="true">
      <div class="pg-skel-legend"><span></span><span></span></div>
      <div class="pg-skel-chart"></div>
    </div><p class="pg-caption pg-muted">Loading measured benchmark data…</p>`;
  }

  let data;
  try {
    data = await loadBench();
  } catch (e) {
    out.innerHTML = `<p class="pg-error">Could not load benchmark data.</p>`;
    return;
  }

  const which = $("pg-bench").value;
  let caption = "";
  let source = "";
  let chart = "";

  if (which === "metal") {
    const rows = data.metal_quantize;
    chart = svgBarChart({
      groups: rows.map((r) => ({ label: "S=" + r.S, values: { speedup: r.speedup } })),
      series: [{ key: "speedup", label: "Metal speedup ×" }],
      yLabel: "speedup ×",
      valFmt: (v) => v.toFixed(1) + "×",
    });
    caption = "Hand-written Metal quantize kernel vs pure-MLX, by sequence length S (B=1, H=8, D=128).";
    source = "figures/metal/results.json";
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
    caption = "Full-KV memory, Falcon3-7B shape: fp16 vs RaBitQ (~5.95× smaller).";
    source = "figures/RaBitQ/falcon/results.json";
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
    caption = `KIVI on ${model} (${m.chip}, ${m.ram_gb} GB): throughput retention vs key compression.`;
    source = "figures/kivi/results_summary.json";
  }

  out.innerHTML = `${renderBand("Chart", "chart", chart)}
    <p class="pg-provenance"><span class="pg-prov-dot" aria-hidden="true"></span>
      Measured on real hardware · source <code>${source}</code></p>
    <p class="pg-caption">${caption}</p>`;

  growBars(out);
}

/* ---------- Stepper / tab switching ---------- */
const TAB_ORDER = ["rec", "lab", "bench"];

// Activate a tool tab by key. Shared by clicks, arrow keys, and hero CTAs.
function switchTab(tab) {
  const steps = document.querySelectorAll(".pg-step");
  const panels = document.querySelectorAll(".pg-panel");
  steps.forEach((b) => {
    const on = b.dataset.pgtab === tab;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
    b.tabIndex = on ? 0 : -1;
  });
  panels.forEach((p) => p.classList.toggle("active", p.id === "pg-panel-" + tab));
  if (tab === "bench") runBenchViewer();
}

function initTabs() {
  const steps = document.querySelectorAll(".pg-step");
  steps.forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.pgtab));
    // Arrow-key navigation across the tablist (WAI-ARIA tabs pattern).
    btn.addEventListener("keydown", (e) => {
      if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(e.key)) return;
      e.preventDefault();
      const idx = TAB_ORDER.indexOf(btn.dataset.pgtab);
      let next;
      if (e.key === "Home") next = 0;
      else if (e.key === "End") next = TAB_ORDER.length - 1;
      else next = (idx + (e.key === "ArrowRight" ? 1 : -1) + TAB_ORDER.length) % TAB_ORDER.length;
      switchTab(TAB_ORDER[next]);
      document.querySelector(`.pg-step[data-pgtab="${TAB_ORDER[next]}"]`).focus();
    });
  });
}

// Toggle the KIVI model dropdown visibility based on selected benchmark.
function syncBenchModelPicker() {
  const isKivi = $("pg-bench").value === "kivi";
  $("pg-bench-model-wrap").hidden = !isKivi;
}

/* Pattern 5 — segmented control over the benchmark choice.
   The hidden #pg-bench input stays the single source of truth; these
   connected radio-buttons are the accessible view over it (arrow-nav,
   roving tabindex — same shape as the method-card radiogroup). */
const BENCH_ORDER = ["metal", "rabitq", "kivi"];
const BENCH_LABELS = { metal: "Metal", rabitq: "RaBitQ", kivi: "KIVI" };

function renderSegmentedBench() {
  const seg = $("pg-bench-seg");
  if (!seg) return;
  // Guard against a stale/unknown restored ?bench= value (the input is now
  // hidden, so it can't self-reject like the old <select> did).
  if (!BENCH_ORDER.includes($("pg-bench").value)) $("pg-bench").value = "metal";
  const cur = $("pg-bench").value;
  seg.innerHTML = BENCH_ORDER.map((key) => {
    const on = key === cur;
    return `<button type="button" class="pg-seg-btn ${on ? "active" : ""}"
      role="radio" aria-checked="${on}" data-bench="${key}" tabindex="${on ? "0" : "-1"}">
      ${BENCH_LABELS[key]}</button>`;
  }).join("");

  seg.querySelectorAll(".pg-seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => selectBench(btn.dataset.bench));
    btn.addEventListener("keydown", (e) => {
      if (!["ArrowRight", "ArrowLeft", "ArrowDown", "ArrowUp"].includes(e.key)) return;
      e.preventDefault();
      const idx = BENCH_ORDER.indexOf($("pg-bench").value);
      const dir = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : -1;
      const next = (idx + dir + BENCH_ORDER.length) % BENCH_ORDER.length;
      selectBench(BENCH_ORDER[next]);
      $("pg-bench-seg").querySelector(".pg-seg-btn.active")?.focus();
    });
  });
}

// Single source of truth for the benchmark choice: update hidden input,
// re-mark the segmented buttons, sync the model picker, and re-render.
function selectBench(key) {
  if (!BENCH_ORDER.includes(key)) return;
  $("pg-bench").value = key;
  const seg = $("pg-bench-seg");
  if (seg) {
    seg.querySelectorAll(".pg-seg-btn").forEach((btn) => {
      const on = btn.dataset.bench === key;
      btn.classList.toggle("active", on);
      btn.setAttribute("aria-checked", on ? "true" : "false");
      btn.tabIndex = on ? 0 : -1;
    });
  }
  syncBenchModelPicker();
  runBenchViewer();
}

// Reflect the current architecture shape in the collapsed Advanced summary,
// so a newcomer sees what's inside without expanding it.
function updateAdvancedHint() {
  const hint = $("pg-adv-hint");
  if (!hint) return;
  const s = readSharedShape();
  hint.textContent = `${$("pg-model").value} · ${s.n_layers}L · ${s.n_kv_heads} KV heads · dim ${s.head_dim} · seq ${s.seq_len.toLocaleString()}`;
}

/* ---------- Wire everything up ---------- */
document.addEventListener("DOMContentLoaded", () => {
  initTabs();

  const recompute = () => {
    markActivePreset();
    updateAdvancedHint();
    runRecommender();
    runCompressionLab();
  };

  // Setup-bar inputs (chip / RAM / goal) drive the recommender + headline.
  ["pg-chip", "pg-ram", "pg-goal"].forEach((id) =>
    $(id).addEventListener("change", runRecommender)
  );
  // Shared shape inputs drive both recommender and lab; hand-editing a shape
  // field re-evaluates which preset (if any) still matches, else falls to Custom.
  ["pg-layers", "pg-heads", "pg-headdim", "pg-seqlen"].forEach((id) =>
    $(id).addEventListener("input", recompute)
  );
  // model_class is both a recommender input and part of a preset's identity.
  $("pg-model").addEventListener("change", recompute);
  // Lab budget (method is chosen via cards → selectMethod).
  $("pg-budget").addEventListener("input", runCompressionLab);
  // Bench inputs — the benchmark itself is a segmented control (wired in
  // renderSegmentedBench → selectBench); the KIVI model stays a <select>.
  $("pg-bench-model").addEventListener("change", runBenchViewer);

  // "Copy my setup" — a refresh-safe link that restores the current setup.
  const shareBtn = $("pg-share-btn");
  if (shareBtn) {
    shareBtn.addEventListener("click", () => {
      navigator.clipboard?.writeText(buildShareUrl()).then(() => {
        const orig = shareBtn.innerHTML;
        shareBtn.innerHTML = '<span aria-hidden="true">✓</span> Copied!';
        shareBtn.classList.add("copied");
        setTimeout(() => {
          shareBtn.innerHTML = orig;
          shareBtn.classList.remove("copied");
        }, 1400);
      });
    });
  }

  // Re-render charts on theme toggle so SVG colors track the theme.
  const themeToggle = $("theme-toggle");
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      if ($("pg-panel-bench").classList.contains("active")) {
        setTimeout(runBenchViewer, 0);
      }
    });
  }

  renderPresets();
  renderMethodCards();

  // Restore any shared setup, then seed a sensible default only if the URL
  // didn't already pin the shape (so a shared link is never overwritten).
  const hadUrlState = new URLSearchParams(window.location.search).toString() !== "";
  restoreFromUrl();
  if (!hadUrlState) {
    applyPreset("mistral-7b"); // sensible default that matches the code snippet
  } else {
    selectMethod($("pg-cmethod").value); // re-mark cards from restored method
  }

  markActivePreset();
  updateAdvancedHint();
  renderSegmentedBench(); // reflects any restored ?bench= value
  syncBenchModelPicker();
  runRecommender();
  runCompressionLab();
  runBenchViewer();
});
