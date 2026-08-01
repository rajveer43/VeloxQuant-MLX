// ============================================================
// VeloxQuant-MLX landing page behavior
// Zero-build static JS — served as-is by Netlify (cp -r landing/* dist/)
//
// Deliberately small. The decorative layer this file used to carry (matrix
// rain canvas, aurora parallax, floating particles, magnetic buttons, stat
// counters, code type-on) was removed in the consumer redesign: six
// simultaneous animation systems cost real battery on the Apple Silicon
// laptops this project targets, and read as marketing over substance. What
// remains is behaviour a visitor actually uses.
// ============================================================

// ── COPY-TO-CLIPBOARD ──
function copyText(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = orig; btn.classList.remove('copied'); }, 1800);
  });
}

function initCopyButtons() {
  const heroInstall = document.getElementById('hero-install');
  if (heroInstall) {
    heroInstall.addEventListener('click', () => {
      copyText('pip install VeloxQuant-MLX', document.getElementById('hero-copy-btn'));
    });
  }

  document.querySelectorAll('.code-copy').forEach(btn => {
    btn.addEventListener('click', () => {
      const pre = document.getElementById(btn.dataset.target);
      copyText(pre.innerText, btn);
    });
  });
}

// ── QUICKSTART CODE TABS ──
function initCodeTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.code-panel');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      tabBtns.forEach(b => {
        b.classList.remove('active', 'active-purple', 'active-green', 'active-teal');
        b.setAttribute('aria-selected', 'false');
      });
      tabPanels.forEach(p => p.classList.remove('active'));
      btn.setAttribute('aria-selected', 'true');
      btn.classList.add(
        tab === 'vecinfer' ? 'active-purple' :
        (tab === 'spectral' || tab === 'chunkkv') ? 'active-green' :
        tab === 'squeeze' ? 'active-teal' : 'active'
      );
      const panel = document.getElementById('tab-' + tab);
      if (panel) panel.classList.add('active');
      // Keep the active tab button in view on the horizontally-scrolling
      // mobile tab strip.
      btn.scrollIntoView({ inline: 'nearest', block: 'nearest' });
    });
  });
}

// ── HERO MEMORY BARS ──
// The hero's only visual. Numbers are computed by VQCalc from the same math
// the calculator and playground use — never hand-written into the markup, so
// they cannot drift from the rest of the page.
const HERO_SCENARIO = { presetId: 'llama31-8b', seqLen: 65536, ramGb: 16 };

function initHeroViz() {
  const host = document.getElementById('hero-viz-rows');
  if (!host || typeof VQCalc === 'undefined') return;

  const preset = VQCalc.MODEL_PRESETS.find(p => p.id === HERO_SCENARIO.presetId);
  if (!preset) return;

  const res = VQCalc.computeRows(preset, HERO_SCENARIO.seqLen, HERO_SCENARIO.ramGb);
  const fp16 = res.rows[0];

  // Only the headline three: fp16 baseline, the everyday default, and the
  // maximum-compression path. The rest live in the calculator.
  const show = ['fp16', 'turboquant_rvq', 'vecinfer'];
  const rows = show.map(id => res.rows.find(r => r.id === id)).filter(Boolean);

  const barClass = { fp16: 'b-fp16', turboquant_rvq: 'b-rvq', vecinfer: 'b-max' };

  host.innerHTML = rows.map(r => {
    const pct = Math.max(1.5, (r.mb / fp16.mb) * 100);
    const ratio = r.id === 'fp16' ? '' : `<span class="viz-x">${r.ratio}×</span>`;
    return `
      <div class="viz-row${r.id === 'fp16' ? ' is-fp16' : ''}">
        <span class="viz-label">${r.name}</span>
        <span class="viz-track">
          <span class="viz-bar ${barClass[r.id]}" data-pct="${pct}"></span>
        </span>
        <span class="viz-val">${VQCalc.formatMb(r.mb)}${ratio}</span>
      </div>`;
  }).join('');

  // Fill the bars. One transition on load, skipped under reduced motion.
  const bars = host.querySelectorAll('.viz-bar');
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduce) {
    bars.forEach(b => { b.style.transition = 'none'; b.style.width = b.dataset.pct + '%'; });
  } else {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        bars.forEach(b => { b.style.width = b.dataset.pct + '%'; });
      });
    });
  }

  const foot = document.getElementById('hero-viz-foot');
  if (foot) {
    const rvq = res.rows.find(r => r.id === 'turboquant_rvq');
    foot.innerHTML =
      `On a ${HERO_SCENARIO.ramGb} GB Mac about ${res.budgetGb} GB is left for the KV cache ` +
      `after 4-bit weights and the OS. fp16 needs ${VQCalc.formatMb(fp16.mb)} — ` +
      `<span class="verdict-bad">it doesn't fit</span>. ` +
      `RVQ-1bit needs ${VQCalc.formatMb(rvq.mb)} — ` +
      `<span class="verdict-good">it does</span>.`;
  }
}

// ── INLINE MEMORY CALCULATOR ──
// Three inputs only; the full parameter surface stays in the playground.
// State round-trips through the query string so a result can be shared.
function initCalculator() {
  const shell = document.getElementById('calc');
  if (!shell || typeof VQCalc === 'undefined') return;

  const modelSel = document.getElementById('calc-model');
  const ctxRange = document.getElementById('calc-ctx');
  const ctxOut   = document.getElementById('calc-ctx-val');
  const ramSel   = document.getElementById('calc-ram');
  const out      = document.getElementById('calc-out');
  const nojs     = document.getElementById('calc-nojs');
  if (!modelSel || !ctxRange || !ramSel || !out) return;

  // Context slider works in powers of two: 4k … 128k.
  const CTX_STEPS = [4096, 8192, 16384, 32768, 65536, 131072];

  modelSel.innerHTML = VQCalc.MODEL_PRESETS
    .map(p => `<option value="${p.id}">${p.name}</option>`).join('');
  ramSel.innerHTML = VQCalc.ALLOWED_RAM_GB
    .map(g => `<option value="${g}">${g} GB</option>`).join('');

  ctxRange.min = '0';
  ctxRange.max = String(CTX_STEPS.length - 1);
  ctxRange.step = '1';

  // ── query-string state ──
  function readState() {
    const q = new URLSearchParams(location.search);
    const model = q.get('model');
    const ctx = parseInt(q.get('ctx'), 10);
    const ram = parseInt(q.get('ram'), 10);

    if (model && VQCalc.MODEL_PRESETS.some(p => p.id === model)) modelSel.value = model;
    else modelSel.value = 'llama31-8b';

    const ctxIdx = CTX_STEPS.indexOf(ctx);
    ctxRange.value = String(ctxIdx >= 0 ? ctxIdx : CTX_STEPS.indexOf(32768));

    ramSel.value = String(VQCalc.ALLOWED_RAM_GB.includes(ram) ? ram : 16);
  }

  function writeState(seqLen) {
    const q = new URLSearchParams(location.search);
    q.set('model', modelSel.value);
    q.set('ctx', String(seqLen));
    q.set('ram', ramSel.value);
    history.replaceState(null, '', location.pathname + '?' + q.toString() + location.hash);
  }

  function render() {
    const preset = VQCalc.MODEL_PRESETS.find(p => p.id === modelSel.value);
    const seqLen = CTX_STEPS[parseInt(ctxRange.value, 10)];
    const ramGb = parseInt(ramSel.value, 10);
    if (!preset) return;

    ctxOut.textContent = VQCalc.formatTokens(seqLen) + ' tokens';

    const res = VQCalc.computeRows(preset, seqLen, ramGb);
    const rec = VQCalc.pickRecommended(res.rows);
    const fp16 = res.rows[0];

    const verdict = fp16.fits
      ? `<span class="t-strong">fp16 already fits.</span> ${preset.name} at ` +
        `${VQCalc.formatTokens(seqLen)} tokens needs ${VQCalc.formatMb(fp16.mb)} of KV cache, ` +
        `and you have about ${res.budgetGb} GB free. Compression still buys you ` +
        `headroom: <code class="inline plain t-accent">${rec.id}</code> takes you to ` +
        `~${VQCalc.formatTokens(rec.maxTokens)} tokens in the same RAM.`
      : `<span class="t-strong">fp16 does not fit.</span> ${preset.name} at ` +
        `${VQCalc.formatTokens(seqLen)} tokens needs ${VQCalc.formatMb(fp16.mb)}, but only ` +
        `about ${res.budgetGb} GB is free after weights and the OS. ` +
        `<code class="inline plain t-accent">${rec.id}</code> fits in ` +
        `${VQCalc.formatMb(rec.mb)}.`;

    const rowsHtml = res.rows.map(r => {
      const isRec = r.id === rec.id;
      const fit = r.fits
        ? '<span class="calc-row-fit y">✓ fits</span>'
        : '<span class="calc-row-fit n">✗ too big</span>';
      const label = r.id === 'fp16' ? r.name : `${r.name} · ${r.ratio}×`;
      return `
        <div class="calc-row${isRec ? ' rec' : ''}">
          <span class="calc-row-name">${label}</span>
          <span class="viz-track">
            <span class="viz-bar ${r.id === 'fp16' ? 'b-fp16' : (r.id === 'vecinfer' ? 'b-max' : 'b-rvq')}"
                  style="width:${Math.max(1.5, (r.mb / fp16.mb) * 100)}%"></span>
          </span>
          <span class="calc-row-size">${VQCalc.formatMb(r.mb)}</span>
          ${fit}
        </div>`;
    }).join('');

    // Key-accounting methods keep an fp16 parent cache on the default path,
    // so their ratio is an accounting bound rather than a resident-RAM
    // promise. Say so rather than letting the bar imply otherwise.
    const residentNote = rec.resident
      ? `<code class="inline plain">${rec.id}</code> compresses keys <em>and</em> values, so the saving shows up as real resident RAM.`
      : `<code class="inline plain">${rec.id}</code> is a key-accounting method: on the default path it dequantizes into an fp16 parent cache, so the ratio is an accounting bound rather than a guaranteed drop in resident RAM. For real resident savings use <code class="inline plain">rabitq</code>.`;

    out.innerHTML =
      `<div class="calc-verdict ${fp16.fits ? 'good' : 'bad'}">${verdict}</div>` +
      `<div class="calc-rows">${rowsHtml}</div>` +
      `<p class="calc-note">Estimated from your model's real shape ` +
      `(${preset.n_layers} layers · ${preset.n_kv_heads} KV heads · head_dim ${preset.head_dim}), ` +
      `assuming 4-bit weights (~${res.weightGb} GB) and a 4 GB OS reserve. ` +
      `${residentNote}</p>` +
      `<div class="calc-cta">` +
        `<a class="btn btn-filled" href="#quickstart">Use <code class="inline plain">${rec.id}</code> →</a>` +
        `<a class="btn btn-outline" href="playground.html">Full playground →</a>` +
      `</div>`;

    writeState(seqLen);
  }

  // The no-JS fallback is in the markup and removed only once we know the
  // calculator can actually take over.
  readState();
  if (nojs) nojs.remove();
  render();

  modelSel.addEventListener('change', render);
  ramSel.addEventListener('change', render);
  ctxRange.addEventListener('input', render);
}

// ── SCROLL FADE-IN ──
function initScrollFadeIn() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        setTimeout(() => entry.target.classList.add('visible'), i * 60);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));
}

// ── ACTIVE NAV LINK ON SCROLL ──
function initActiveNav() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-links a');

  const navObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(a => a.classList.remove('active'));
        const active = document.querySelector(`.nav-links a[href="#${entry.target.id}"]`);
        if (active) active.classList.add('active');
      }
    });
  }, { rootMargin: '-40% 0px -55% 0px' });

  sections.forEach(s => navObserver.observe(s));
}

// ── MOBILE HAMBURGER MENU ──
function initHamburgerMenu() {
  const toggle = document.getElementById('nav-toggle');
  const links = document.getElementById('nav-links');
  if (!toggle || !links) return;

  toggle.addEventListener('click', () => {
    const open = links.classList.toggle('open');
    toggle.classList.toggle('open', open);
    toggle.setAttribute('aria-expanded', open);
    document.body.style.overflow = open ? 'hidden' : '';
  });

  links.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', () => {
      links.classList.remove('open');
      toggle.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
      document.body.style.overflow = '';
    });
  });
}

// ── THEME TOGGLE (light / dark, persisted in localStorage) ──
function initThemeToggle() {
  const toggle = document.getElementById('theme-toggle');
  if (!toggle) return;

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme')
      || (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('vq-theme', theme); } catch (e) { /* storage unavailable */ }
    toggle.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  }

  toggle.addEventListener('click', () => {
    applyTheme(currentTheme() === 'dark' ? 'light' : 'dark');
  });
}

// ── INIT ──
document.addEventListener('DOMContentLoaded', () => {
  initCopyButtons();
  initCodeTabs();
  initHeroViz();
  initCalculator();
  initScrollFadeIn();
  initActiveNav();
  initHamburgerMenu();
  initThemeToggle();
});
