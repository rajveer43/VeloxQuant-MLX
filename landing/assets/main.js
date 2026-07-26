// ============================================================
// VeloxQuant-MLX landing page behavior
// Zero-build static JS — served as-is by Netlify (cp -r landing/* dist/)
// ============================================================

function hexToRgba(hex, alpha) {
  const h = hex.replace('#', '');
  const r = parseInt(h.length === 3 ? h[0] + h[0] : h.slice(0, 2), 16);
  const g = parseInt(h.length === 3 ? h[1] + h[1] : h.slice(2, 4), 16);
  const b = parseInt(h.length === 3 ? h[2] + h[2] : h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

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

// ── HERO BADGE TYPING ANIMATION ──
function initBadgeTyping() {
  const badge = document.getElementById('hero-badge');
  if (!badge) return;
  const text = "v0.41.0 — mlx-vlm vision-language model support shipped";
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    badge.textContent = text;
    return;
  }
  let i = 0;
  badge.innerHTML = '';
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

// ── HERO MATRIX-RAIN CANVAS ──
// Decorative only; skipped for reduced-motion and on narrow/touch viewports
// where it costs battery for no visible benefit (canvas covers the hero,
// which is mostly obscured by content on small screens anyway).
function initMatrixRain() {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  if (window.matchMedia('(max-width: 640px)').matches) return;

  const canvas = document.getElementById('matrix-canvas');
  const hero = document.getElementById('hero');
  if (!canvas || !hero) return;
  const ctx = canvas.getContext('2d');
  const chars = '01ABCDEFx+=<>[]{}∑∫ΔΩπ';
  const fontSize = 13;
  let cols, drops;

  function resize() {
    canvas.width = hero.offsetWidth;
    canvas.height = hero.offsetHeight;
    cols = Math.floor(canvas.width / fontSize);
    drops = Array.from({ length: cols }, () => Math.random() * -50);
  }

  resize();
  new ResizeObserver(resize).observe(hero);

  function themeColors() {
    const styles = getComputedStyle(document.documentElement);
    return {
      bg: styles.getPropertyValue('--bg').trim() || '#08080f',
      accent: styles.getPropertyValue('--accent').trim() || '#00d4ff',
    };
  }

  function draw() {
    const { bg, accent } = themeColors();
    ctx.fillStyle = hexToRgba(bg, 0.05);
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = accent;
    ctx.font = fontSize + 'px JetBrains Mono, monospace';
    for (let i = 0; i < cols; i++) {
      const ch = chars[Math.floor(Math.random() * chars.length)];
      ctx.fillText(ch, i * fontSize, drops[i] * fontSize);
      if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0;
      drops[i] += 0.4;
    }
  }

  setInterval(draw, 60);
}

// ── STAT NUMBER COUNTER ──
function animateCounter(element, target, suffix) {
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
          if (entry.target.classList.contains('stat-card')) {
            const statNum = entry.target.querySelector('.stat-number');
            if (statNum && !statNum.dataset.animated) {
              statNum.dataset.animated = 'true';
              const raw = statNum.textContent.trim();
              const num = parseFloat(raw);
              const suffix = raw.replace(/[\d.]/g, '');
              if (!isNaN(num)) animateCounter(statNum, num, suffix);
            }
          }
        }, i * 80);
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

// ── HERO SCROLL PARALLAX (aurora blobs) ──
function initScrollParallax() {
  const heroEl = document.getElementById('hero');
  if (!heroEl) return;
  const auroraContainer = heroEl.querySelector('.aurora-container');
  if (!auroraContainer) return;

  window.addEventListener('scroll', () => {
    const scrollY = window.scrollY;
    const heroH = heroEl.offsetHeight;
    if (scrollY > heroH) return;
    const pct = scrollY / heroH;
    auroraContainer.style.transform = `translateY(${pct * 60}px)`;
  }, { passive: true });
}

// ── CODE BLOCK "TERMINAL BOOT" TYPE-ON EFFECT ──
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
    if (nodeIdx >= nodes.length) {
      cursor.remove();
      return;
    }
    const node = nodes[nodeIdx];
    if (node.nodeType === Node.TEXT_NODE) {
      const text = node.textContent;
      if (charIdx < text.length) {
        preEl.insertBefore(document.createTextNode(text[charIdx]), cursor);
        charIdx++;
        delay = 12;
      } else {
        nodeIdx++;
        charIdx = 0;
        delay = 0;
      }
    } else {
      const clone = node.cloneNode(true);
      preEl.insertBefore(clone, cursor);
      nodeIdx++;
      charIdx = 0;
      delay = 18;
    }
    setTimeout(nextTick, delay);
  }

  setTimeout(nextTick, 120);
}

function initCodeBootAnimation() {
  const codeObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const pre = entry.target.querySelector('pre');
        if (pre) bootCode(pre);
        codeObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.25 });

  document.querySelectorAll('.code-wrap').forEach(el => codeObserver.observe(el));
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

// ── MAGNETIC BUTTONS (desktop hover only — mousemove never fires on touch) ──
function initMagneticButtons() {
  document.querySelectorAll('.btn').forEach(btn => {
    btn.addEventListener('mousemove', e => {
      const rect = btn.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = (e.clientX - cx) / (rect.width / 2);
      const dy = (e.clientY - cy) / (rect.height / 2);
      btn.style.transform = `translate(${dx * 7}px, ${dy * 5}px)`;
    });

    btn.addEventListener('mouseleave', () => {
      btn.style.transform = '';
    });
  });
}

// ── INIT ──
document.addEventListener('DOMContentLoaded', () => {
  initCopyButtons();
  initCodeTabs();
  initBadgeTyping();
  initMatrixRain();
  initScrollFadeIn();
  initActiveNav();
  initScrollParallax();
  initCodeBootAnimation();
  initHamburgerMenu();
  initMagneticButtons();
  initThemeToggle();
});
