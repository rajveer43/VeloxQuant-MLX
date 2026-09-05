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

  document.querySelectorAll('.code-copy, .code-copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const pre = document.getElementById(btn.dataset.target);
      if (pre) copyText(pre.innerText, btn);
    });
  });

  document.querySelectorAll('.eco-copy-trigger').forEach(btn => {
    btn.addEventListener('click', () => {
      const textToCopy = btn.getAttribute('data-copy') || 'npm i @veloxquant/sdk';
      const label = btn.querySelector('.eco-copy-label');
      navigator.clipboard.writeText(textToCopy).then(() => {
        if (label) {
          const orig = label.textContent;
          label.textContent = '✓ Copied to clipboard!';
          setTimeout(() => { label.textContent = orig; }, 2000);
        }
      });
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
  const navLinks = document.querySelectorAll('#nav-links a');

  const navObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(a => a.classList.remove('active'));
        const active = document.querySelector(`#nav-links a[href="#${entry.target.id}"]`);
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

// ── ANNOUNCEMENT BANNER (dismissible, persisted in localStorage) ──
// Visibility itself is decided pre-paint by the inline script in <head>
// (sets data-show-announcement="1" on <html>), so it never pops in after
// layout has settled. This just wires up the dismiss button.
function initAnnouncementBanner() {
  const banner = document.getElementById('announcement-banner');
  const closeBtn = document.getElementById('announcement-banner-close');
  if (!banner || !closeBtn) return;

  const dismissKey = 'vq-announcement-dismissed-studio-v0.1.0';
  closeBtn.addEventListener('click', () => {
    document.documentElement.removeAttribute('data-show-announcement');
    try { localStorage.setItem(dismissKey, '1'); } catch (e) { /* storage unavailable */ }
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

// ── BENCHMARK PAGE INTERACTIVITY ──
function initBenchmarkFilters() {
  const filterWrap = document.getElementById('model-filter-pills');
  const cards = document.querySelectorAll('#benchmark-model-grid .model-card');
  if (!filterWrap || !cards.length) return;

  const pills = filterWrap.querySelectorAll('.filter-pill');
  pills.forEach((pill) => {
    pill.addEventListener('click', () => {
      pills.forEach((p) => p.classList.remove('active'));
      pill.classList.add('active');

      const filter = pill.dataset.filter;
      cards.forEach((card) => {
        const size = card.dataset.size;
        const arch = card.dataset.arch;
        let match = false;

        if (filter === 'all') match = true;
        else if (filter === 'large') match = size === 'large';
        else if (filter === 'mid') match = size === 'mid';
        else if (filter === 'small') match = size === 'small';
        else if (filter === 'moe') match = size === 'moe' || arch === 'moe';

        if (match) {
          card.style.display = 'flex';
          setTimeout(() => {
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
          }, 10);
        } else {
          card.style.opacity = '0';
          card.style.transform = 'translateY(6px)';
          setTimeout(() => {
            if (card.style.opacity === '0') card.style.display = 'none';
          }, 200);
        }
      });
    });
  });

  // Highlight active benchmark category nav pill on scroll
  const navPills = document.querySelectorAll('.bench-nav-pill');
  if (navPills.length) {
    const sections = document.querySelectorAll('.bench-section');
    window.addEventListener('scroll', () => {
      let currentId = '';
      sections.forEach((sec) => {
        const top = sec.offsetTop - 140;
        if (window.scrollY >= top) {
          currentId = sec.getAttribute('id');
        }
      });
      if (currentId) {
        navPills.forEach((p) => {
          p.classList.toggle('active', p.getAttribute('href') === '#' + currentId);
        });
      }
    }, { passive: true });
  }
}

// ── DYNAMIC FLUID AURORA & MEMORY FLUX (Hero Background) ──
function initHeroAurora() {
  const canvas = document.getElementById('hero-aurora-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const hero = document.getElementById('hero');
  if (!hero) return;

  let width = 0;
  let height = 0;
  let animationFrameId = null;
  let isVisible = true;

  // Mouse parallax state
  let mouse = { x: 0.5, y: 0.5 };
  let targetMouse = { x: 0.5, y: 0.5 };

  function onMouseMove(e) {
    const rect = hero.getBoundingClientRect();
    if (e.clientY < rect.top || e.clientY > rect.bottom) return;
    targetMouse.x = (e.clientX - rect.left) / rect.width;
    targetMouse.y = (e.clientY - rect.top) / rect.height;
  }
  window.addEventListener('mousemove', onMouseMove, { passive: true });

  // Handle high-DPI crisp rendering across 100% full-bleed screen width
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
    width = hero.getBoundingClientRect().width || window.innerWidth;
    height = hero.getBoundingClientRect().height || window.innerHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener('resize', resize, { passive: true });
  resize();

  // 5 Organic Fluid Blobs spanning the entire edge-to-edge width (left, center, right)
  const blobs = [
    {
      // Left flank: Electric Cyan
      baseX: 0.20, baseY: 0.40, radius: 0.62,
      speedX: 0.00065, speedY: 0.00085,
      colorDark: [56, 189, 248, 0.45],
      colorLight: [56, 189, 248, 0.24]
    },
    {
      // Right flank: Deep Apple Silicon Violet
      baseX: 0.80, baseY: 0.42, radius: 0.65,
      speedX: 0.00055, speedY: 0.00072,
      colorDark: [124, 58, 237, 0.48],
      colorLight: [147, 51, 234, 0.22]
    },
    {
      // Center: Royal Indigo / Sapphire depth
      baseX: 0.50, baseY: 0.35, radius: 0.70,
      speedX: 0.00045, speedY: 0.00062,
      colorDark: [79, 70, 229, 0.40],
      colorLight: [99, 102, 241, 0.18]
    },
    {
      // Lower Left-Center: Ice Mint energy spark
      baseX: 0.34, baseY: 0.58, radius: 0.45,
      speedX: 0.00095, speedY: 0.00110,
      colorDark: [45, 212, 191, 0.30],
      colorLight: [20, 184, 166, 0.15]
    },
    {
      // Upper Right-Center: Luminous Lavender
      baseX: 0.66, baseY: 0.26, radius: 0.50,
      speedX: 0.00075, speedY: 0.00090,
      colorDark: [168, 85, 247, 0.38],
      colorLight: [192, 132, 252, 0.20]
    }
  ];

  // 65 Floating Quantized Memory Tokens (Product vision particle flux across full screen)
  const particleCount = 65;
  const particles = [];
  for (let i = 0; i < particleCount; i++) {
    particles.push({
      x: Math.random() * (width || window.innerWidth || 1400),
      y: Math.random() * (height || 800),
      radius: Math.random() * 1.5 + 0.8,
      vy: -(Math.random() * 0.35 + 0.15),
      vxFactor: Math.random() * 0.008 + 0.004,
      phase: Math.random() * Math.PI * 2,
      alpha: Math.random() * 0.45 + 0.15,
      hue: Math.random() > 0.5 ? 'cyan' : 'violet'
    });
  }

  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const startTime = performance.now();

  function render(now) {
    if (!isVisible) return;

    const t = prefersReduced ? 1000 : (now - startTime);

    // Smooth mouse parallax lerp
    mouse.x += (targetMouse.x - mouse.x) * 0.04;
    mouse.y += (targetMouse.y - mouse.y) * 0.04;

    ctx.clearRect(0, 0, width, height);

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';

    // 1. Draw Organic Fluid Blobs
    ctx.globalCompositeOperation = isLight ? 'multiply' : 'screen';

    blobs.forEach((b, idx) => {
      const offsetX = Math.sin(t * b.speedX + idx) * 0.16 + Math.cos(t * b.speedX * 0.7) * 0.08;
      const offsetY = Math.cos(t * b.speedY + idx * 1.5) * 0.14 + Math.sin(t * b.speedY * 0.6) * 0.06;

      // Parallax offset
      const px = (b.baseX + offsetX) * width + (mouse.x - 0.5) * 70;
      const py = (b.baseY + offsetY) * height + (mouse.y - 0.5) * 50;
      const r = Math.min(width, height) * b.radius * (1 + Math.sin(t * 0.0008 + idx) * 0.08);

      const color = isLight ? b.colorLight : b.colorDark;
      const grad = ctx.createRadialGradient(px, py, 0, px, py, Math.max(r, 10));
      grad.addColorStop(0, `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${color[3]})`);
      grad.addColorStop(0.45, `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${color[3] * 0.5})`);
      grad.addColorStop(1, `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0)`);

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(px, py, Math.max(r, 10), 0, Math.PI * 2);
      ctx.fill();
    });

    // 2. Draw Subtle Memory Token Flux Particles
    ctx.globalCompositeOperation = 'source-over';

    if (!prefersReduced) {
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.y += p.vy;
        p.x += Math.sin(p.y * p.vxFactor + p.phase) * 0.35;

        // Wrap around vertically
        if (p.y < -10) {
          p.y = height + 10;
          p.x = Math.random() * width;
        }

        // Draw particle
        const pColor = p.hue === 'cyan'
          ? (isLight ? 'rgba(2, 132, 199, ' : 'rgba(56, 189, 248, ')
          : (isLight ? 'rgba(126, 34, 206, ' : 'rgba(192, 132, 252, ');

        ctx.fillStyle = pColor + p.alpha + ')';
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fill();

        // Connect nearby tokens with faint memory tensor links
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const dx = p.x - p2.x;
          const dy = p.y - p2.y;
          const dist = Math.hypot(dx, dy);

          if (dist < 80) {
            const lineAlpha = (1 - dist / 80) * (isLight ? 0.08 : 0.16);
            ctx.strokeStyle = isLight
              ? `rgba(99, 102, 241, ${lineAlpha})`
              : `rgba(124, 58, 237, ${lineAlpha})`;
            ctx.lineWidth = 0.8;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }
        }
      }
    }

    if (!prefersReduced) {
      animationFrameId = requestAnimationFrame(render);
    }
  }

  // IntersectionObserver: Pause completely when scrolled past hero to save battery
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          isVisible = entry.isIntersecting;
          if (isVisible && !prefersReduced) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = requestAnimationFrame(render);
          } else {
            cancelAnimationFrame(animationFrameId);
          }
        });
      },
      { threshold: 0.05 }
    );
    observer.observe(hero);
  } else {
    animationFrameId = requestAnimationFrame(render);
  }
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
  initAnnouncementBanner();
  initBenchmarkFilters();
  initHeroAurora();
});


