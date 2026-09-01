// ============================================================
// VeloxQuant-MLX landing page behavior
// Zero-build static JS — served as-is by Netlify (cp -r landing/* dist/)
//
// The hero's atmospheric layer (matrix rain canvas, aurora blobs, floating
// particles, scroll parallax) was removed in the calmer "oMLX-style" restyle
// — the hero background is flat/static now. Badge typing and magnetic
// buttons stay: they're subtle, one-time or hover-only effects, gated on
// prefers-reduced-motion like everything else here.
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
function activateTab(tab, { scrollIntoView = true } = {}) {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.code-panel');
  const btn = document.querySelector(`.tab-btn[data-tab="${tab}"]`);
  if (!btn) return;

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
  // The advanced examples (VecInfer, RateQuant, VLM) live inside a
  // collapsed <details> — open it so activating a tab actually shows it,
  // instead of leaving it collapsed.
  const details = btn.closest('details');
  if (details && !details.open) details.open = true;
  // Keep the active tab button in view on the horizontally-scrolling
  // mobile tab strip.
  if (scrollIntoView) btn.scrollIntoView({ inline: 'nearest', block: 'nearest' });
}

function initCodeTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => activateTab(btn.dataset.tab));
  });
}

// ── HERO BADGE TYPING ANIMATION ──
function initBadgeTyping() {
  const badge = document.getElementById('hero-badge');
  if (!badge) return;
  const text = badge.dataset.text || badge.textContent.trim();
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    badge.textContent = text;
    return;
  }
  // Lock the badge at its final rendered width before emptying it, so the
  // typing animation grows text inside a fixed box instead of reflowing the
  // hero on every keystroke (was the largest single contributor to CLS).
  badge.style.minWidth = badge.getBoundingClientRect().width + 'px';
  let i = 0;
  badge.textContent = '';
  const cursor = document.createElement('span');
  cursor.className = 'badge-cursor';
  badge.appendChild(cursor);

  const interval = setInterval(() => {
    badge.insertBefore(document.createTextNode(text[i]), cursor);
    i++;
    if (i >= text.length) {
      clearInterval(interval);
      setTimeout(() => cursor.remove(), 1500);
    }
  }, 35);
}

// ── MAGNETIC BUTTONS (desktop hover only — mousemove never fires on touch) ──
function initMagneticButtons() {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  document.querySelectorAll('.btn').forEach(btn => {
    btn.addEventListener('mousemove', e => {
      const rect = btn.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = (e.clientX - cx) / (rect.width / 2);
      const dy = (e.clientY - cy) / (rect.height / 2);
      btn.style.transform = `translate(${dx * 7}px, ${dy * 5}px)`;
    });
    btn.addEventListener('mouseleave', () => { btn.style.transform = ''; });
  });
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

  function render() {
    const preset = VQCalc.MODEL_PRESETS.find(p => p.id === modelSel.value);
    const seqLen = CTX_STEPS[parseInt(ctxRange.value, 10)];
    const ramGb = parseInt(ramSel.value, 10);
    if (!preset) return;

    ctxOut.textContent = VQCalc.formatTokens(seqLen) + ' tokens';

    const res = VQCalc.computeRows(preset, seqLen, ramGb);
    const rec = VQCalc.pickRecommended(res.rows);
    const fp16 = res.rows[0];

    // One-line headline result: does it fit, and with how much room. Detail
    // (all methods, the accounting caveat) moves behind "See all options" so
    // the default view is a single glanceable answer, not a wall of bars.
    const headline = fp16.fits
      ? `<span class="t-strong">${preset.name} already fits</span> in ${res.budgetGb} GB free ` +
        `— and <code class="inline plain t-accent">${rec.id}</code> stretches that to ` +
        `~${VQCalc.formatTokens(rec.maxTokens)} tokens.`
      : `<span class="t-strong">${preset.name} needs ${VQCalc.formatMb(fp16.mb)}</span> — more than ` +
        `the ${res.budgetGb} GB you have free. <code class="inline plain t-accent">${rec.id}</code> ` +
        `brings it down to <span class="t-strong">${VQCalc.formatMb(rec.mb)}</span>.`;

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
      `<div class="calc-verdict ${fp16.fits ? 'good' : 'bad'}">${headline}</div>` +
      `<div class="calc-cta">` +
        `<a class="btn btn-filled" href="#quickstart">Use <code class="inline plain">${rec.id}</code> →</a>` +
        `<a class="btn btn-outline" href="playground.html">Full playground →</a>` +
      `</div>` +
      `<details class="reveal calc-reveal">` +
        `<summary>See all options</summary>` +
        `<div class="reveal-body">` +
          `<div class="calc-rows">${rowsHtml}</div>` +
          `<p class="calc-note">Estimated from your model's real shape ` +
          `(${preset.n_layers} layers · ${preset.n_kv_heads} KV heads · head_dim ${preset.head_dim}), ` +
          `assuming 4-bit weights (~${res.weightGb} GB) and a 4 GB OS reserve. ` +
          `${residentNote} Heavier compression can affect answer quality — ` +
          `see the <a href="/docs/algorithms/overview" class="t-accent">algorithm reference</a>.</p>` +
        `</div>` +
      `</details>`;
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

// ── CODE BLOCK "TERMINAL BOOT" TYPE-ON EFFECT ──
// Only ever applied to blocks that opt in with data-boot: the quickstart diff
// is meant to be copied, and making someone wait for it to type is hostile.
function bootCode(preEl) {
  if (preEl.dataset.booted) return;
  preEl.dataset.booted = 'true';
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const original = preEl.innerHTML;
  preEl.innerHTML = '';

  const tmp = document.createElement('div');
  tmp.innerHTML = original;

  const cursor = document.createElement('span');
  cursor.className = 'code-cursor-blink';
  preEl.appendChild(cursor);

  const nodes = Array.from(tmp.childNodes);
  let nodeIdx = 0;
  let charIdx = 0;
  let delay = 0;

  function nextTick() {
    if (nodeIdx >= nodes.length) { cursor.remove(); return; }
    const node = nodes[nodeIdx];
    if (node.nodeType === Node.TEXT_NODE) {
      const text = node.textContent;
      if (charIdx < text.length) {
        preEl.insertBefore(document.createTextNode(text[charIdx]), cursor);
        charIdx++;
        delay = 12;
      } else {
        nodeIdx++; charIdx = 0; delay = 0;
      }
    } else {
      preEl.insertBefore(node.cloneNode(true), cursor);
      nodeIdx++; charIdx = 0; delay = 18;
    }
    setTimeout(nextTick, delay);
  }

  setTimeout(nextTick, 120);
}

function initCodeBootAnimation() {
  const targets = document.querySelectorAll('.code-wrap[data-boot]');
  if (!targets.length) return;
  const codeObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const pre = entry.target.querySelector('pre');
        if (pre) bootCode(pre);
        codeObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.25 });
  targets.forEach(el => codeObserver.observe(el));
}

// ── STAT NUMBER COUNTER ──
function animateCounter(element, target, suffix) {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    element.textContent = target + suffix;
    element.classList.add('counting');
    return;
  }
  const steps = 50;
  const duration = 900;
  let i = 0;
  const interval = setInterval(() => {
    i++;
    const progress = i / steps;
    const eased = 1 - Math.pow(1 - progress, 3);
    const val = target * eased;
    const disp = Number.isInteger(target) ? Math.round(val) : parseFloat(val.toFixed(1));
    element.textContent = disp + suffix;
    if (i >= steps) {
      element.textContent = target + suffix;
      element.classList.add('counting');
      clearInterval(interval);
    }
  }, duration / steps);
}

// ── SCROLL FADE-IN (+ triggers stat counters) ──
function initScrollFadeIn() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        setTimeout(() => {
          entry.target.classList.add('visible');
          if (entry.target.classList.contains('proof-item')) {
            const val = entry.target.querySelector('[data-count]');
            if (val && !val.dataset.animated) {
              val.dataset.animated = 'true';
              const raw = val.textContent.trim();
              const num = parseFloat(raw.replace(/[^\d.]/g, ''));
              const suffix = raw.replace(/[\d.,\s]/g, '');
              if (!isNaN(num)) animateCounter(val, num, suffix);
            }
          }
        }, i * 60);
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

// ── TOAST ──
function showToast({ title, desc, duration = 6000 }) {
  const stack = document.getElementById('toast-stack');
  if (!stack) return;

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.setAttribute('role', 'status');
  toast.innerHTML = `
    <span class="toast-icon" aria-hidden="true">✓</span>
    <span class="toast-body">
      <p class="toast-title"></p>
      <p class="toast-desc"></p>
    </span>
    <button class="toast-close" aria-label="Dismiss notification">×</button>
  `;
  toast.querySelector('.toast-title').textContent = title;
  toast.querySelector('.toast-desc').textContent = desc;

  function dismiss() {
    toast.classList.add('toast-leaving');
    toast.addEventListener('animationend', () => toast.remove(), { once: true });
  }

  toast.querySelector('.toast-close').addEventListener('click', dismiss);
  stack.appendChild(toast);

  setTimeout(dismiss, duration);
}

// ── WAITLIST FORM ──
// Netlify Forms: the static <form data-netlify="true"> is enough for Netlify's
// build-time HTML parser to register the "waitlist" form and start capturing
// submissions server-side, with zero backend of our own. This just upgrades
// the UX from "reload to a plain success page" to an inline swap plus a
// toast, and falls back to a normal (non-JS) POST if fetch is unavailable
// or fails.
function initWaitlistForm() {
  const form = document.getElementById('waitlist-form');
  if (!form) return;

  const success = document.getElementById('waitlist-success');
  const submitBtn = document.getElementById('waitlist-submit');

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    submitBtn.disabled = true;
    submitBtn.textContent = 'Joining…';

    const body = new URLSearchParams(new FormData(form)).toString();

    fetch('/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body,
    })
      .then((res) => {
        if (!res.ok) throw new Error('Network response was not ok');
        form.hidden = true;
        success.hidden = false;
        showToast({
          title: "You're on the waitlist",
          desc: "We'll announce VeloxQuant Studio here as soon as we start building — watch your inbox.",
        });
      })
      .catch(() => {
        // Fall back to a real form submission (full Netlify redirect flow)
        // rather than stranding the user on a form that silently did nothing.
        form.submit();
      });
  });
}

// ── INIT ──
document.addEventListener('DOMContentLoaded', () => {
  initCopyButtons();
  initCodeTabs();
  initBadgeTyping();
  initMagneticButtons();
  initCodeBootAnimation();
  initCalculator();
  initScrollFadeIn();
  initActiveNav();
  initHamburgerMenu();
  initThemeToggle();
  initWaitlistForm();
});
