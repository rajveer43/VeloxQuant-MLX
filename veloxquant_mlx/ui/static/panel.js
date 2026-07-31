/* VeloxQuant-MLX control panel — talks to the local control API.
   No framework, no build step: this file is served as-is. */

const $ = (id) => document.getElementById(id);

const FIELDS = {
  model: 'f-model', method: 'f-method', bits: 'f-bits',
  host: 'f-host', port: 'f-port',
  max_tokens: 'f-max-tokens', temp: 'f-temp', top_p: 'f-top-p',
};

let METHODS = [];
let state = 'stopped';
let logCount = 0;
let poll = null;

/* ── Theme ─────────────────────────────────────────────── */
$('theme-toggle').addEventListener('click', () => {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('vq-theme', next); } catch (e) { /* ignore */ }
});

/* ── API ───────────────────────────────────────────────── */
async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `request failed (${res.status})`);
  return body;
}

function readForm() {
  const cfg = {};
  for (const [key, id] of Object.entries(FIELDS)) {
    const el = $(id);
    cfg[key] = el.type === 'number' ? Number(el.value) : el.value;
  }
  return cfg;
}

function applyConfig(cfg) {
  for (const [key, id] of Object.entries(FIELDS)) {
    if (cfg[key] !== undefined && cfg[key] !== null) $(id).value = cfg[key];
  }
  onHostChange();
}

/* ── Methods ───────────────────────────────────────────── */
async function loadMethods() {
  const data = await api('/api/methods');
  METHODS = data.methods;

  const select = $('f-method');
  select.innerHTML = '';

  const groups = { quantization: 'Quantization', eviction: 'Eviction', hybrid: 'Hybrid' };
  for (const [family, label] of Object.entries(groups)) {
    const inFamily = METHODS.filter((m) => m.family === family);
    if (!inFamily.length) continue;

    const group = document.createElement('optgroup');
    group.label = label;
    for (const m of inFamily) {
      const opt = document.createElement('option');
      opt.value = m.name;
      // Unsupported methods stay visible but unselectable, so the catalog
      // stays honest instead of quietly hiding five of the forty.
      opt.textContent = m.is_servable ? m.name : `${m.name} — unsupported`;
      opt.disabled = !m.is_servable;
      group.appendChild(opt);
    }
    select.appendChild(group);
  }
  select.value = data.default_serve_method;
  onMethodChange();
}

function onMethodChange() {
  const info = METHODS.find((m) => m.name === $('f-method').value);
  const box = $('method-info');
  if (!info) { box.innerHTML = ''; return; }

  const tierBadge = info.is_servable
    ? '<span class="badge badge-accounting">accounting-only</span>'
    : '<span class="badge badge-unsupported">unsupported</span>';

  let html = `<div class="blurb">`
    + `<span class="badge badge-family">${info.family}</span>${tierBadge}`
    + `</div><div class="blurb">${escapeHtml(info.blurb)}</div>`;

  if (info.unsupported_reason) {
    html += `<div class="deviation"><strong>Cannot serve.</strong> ${escapeHtml(info.unsupported_reason)}</div>`;
  }
  if (info.paper_deviation) {
    html += `<div class="deviation"><strong>Adapted.</strong> ${escapeHtml(info.paper_deviation)}</div>`;
  }
  if (info.config_fields && info.config_fields.length) {
    html += `<div class="hint">Knobs: <code>${info.config_fields.join('</code>, <code>')}</code></div>`;
  }
  box.innerHTML = html;
  refreshPrimary();
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

/* ── Status ────────────────────────────────────────────── */
function setState(next, detail) {
  state = next;
  const labels = { stopped: 'Stopped', starting: 'Starting', running: 'Running', error: 'Error' };
  $('status-pill').setAttribute('data-state', next);
  $('status-text').textContent = labels[next] || next;
  if (detail !== undefined) $('status-detail').textContent = detail;

  const btn = $('primary-btn');
  if (next === 'running' || next === 'starting') {
    btn.textContent = 'Stop Server';
    btn.className = 'btn btn-danger';
  } else {
    btn.textContent = 'Start Server';
    btn.className = 'btn btn-filled';
  }

  // Config is locked while a server owns it; changing it mid-flight would
  // make the panel describe a server that isn't what's running.
  const locked = next === 'running' || next === 'starting';
  for (const id of Object.values(FIELDS)) $(id).disabled = locked;

  refreshPrimary();
}

function refreshPrimary() {
  if (state === 'running' || state === 'starting') return;
  const info = METHODS.find((m) => m.name === $('f-method').value);
  const ok = $('f-model').value.trim() && info && info.is_servable;
  $('primary-btn').disabled = !ok;
}

function showError(message) {
  const banner = $('error-banner');
  if (!message) { banner.hidden = true; return; }
  banner.hidden = false;
  banner.innerHTML = `<strong>Error.</strong> ${escapeHtml(message)}`;
}

/* ── Endpoints ─────────────────────────────────────────── */
function renderEndpoints(ready) {
  const list = $('endpoint-list');
  const empty = $('endpoints-empty');
  list.innerHTML = '';

  if (!ready || !ready.endpoints) { empty.hidden = false; return; }
  empty.hidden = true;

  const labels = {
    openai_base_url: 'Base URL',
    chat_completions: 'Chat',
    completions: 'Completions',
    models: 'Models',
  };

  // Rendered from the handshake only — the panel never invents an endpoint.
  for (const [key, url] of Object.entries(ready.endpoints)) {
    const li = document.createElement('li');
    li.className = 'endpoint';
    li.innerHTML = `<span class="endpoint-label">${labels[key] || key}</span>`
      + `<span class="endpoint-url">${escapeHtml(url)}</span>`
      + `<button class="copy-btn" type="button">Copy</button>`;
    li.querySelector('.copy-btn').addEventListener('click', (e) => {
      navigator.clipboard.writeText(url).then(() => {
        const b = e.target;
        b.textContent = 'Copied';
        b.classList.add('copied');
        setTimeout(() => { b.textContent = 'Copy'; b.classList.remove('copied'); }, 1400);
      });
    });
    list.appendChild(li);
  }
}

/* ── Logs ──────────────────────────────────────────────── */
async function refreshLogs() {
  const data = await api(`/api/logs?since=${logCount}`);
  if (!data.lines.length) return;

  const box = $('log');
  const placeholder = box.querySelector('.log-empty');
  if (placeholder) placeholder.remove();

  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
  for (const line of data.lines) {
    const span = document.createElement('span');
    span.className = `log-line log-${line.stream}`;
    span.textContent = line.text;
    box.appendChild(span);
  }
  logCount = data.total;
  if (atBottom) box.scrollTop = box.scrollHeight;
}

$('clear-log').addEventListener('click', () => {
  $('log').innerHTML = '<span class="log-empty">No output yet.</span>';
});
$('copy-log').addEventListener('click', () => {
  navigator.clipboard.writeText($('log').innerText);
});

/* ── Polling ───────────────────────────────────────────── */
async function tick() {
  try {
    const status = await api('/api/status');
    const ready = status.ready;

    if (status.version && !$('nav-version').textContent) {
      $('nav-version').textContent = `v${status.version}`;
    }

    if (status.state !== state) {
      let detail = 'Pick a model and press Start Server.';
      if (status.state === 'starting') detail = 'Loading model and wiring caches …';
      else if (status.state === 'running' && ready) {
        detail = `${ready.model} · ${ready.method} · ${ready.bits}-bit · ${ready.layer_caches} layer caches`;
      } else if (status.state === 'error') detail = 'The server stopped unexpectedly.';
      setState(status.state, detail);
    }

    renderEndpoints(status.state === 'running' ? ready : null);
    showError(status.state === 'error' ? status.error : null);
    await refreshLogs();
  } catch (e) {
    // The control plane itself is unreachable — say so rather than freezing.
    setState('error', 'Lost contact with the panel backend.');
    showError(e.message);
  }
}

/* ── Actions ───────────────────────────────────────────── */
$('primary-btn').addEventListener('click', async () => {
  showError(null);
  const btn = $('primary-btn');
  btn.disabled = true;

  try {
    if (state === 'running' || state === 'starting') {
      await api('/api/stop', { method: 'POST' });
      logCount = 0;
      $('log').innerHTML = '<span class="log-empty">No output yet.</span>';
      setState('stopped', 'Pick a model and press Start Server.');
      renderEndpoints(null);
    } else {
      logCount = 0;
      $('log').innerHTML = '<span class="log-empty">No output yet.</span>';
      setState('starting', 'Loading model and wiring caches …');
      await api('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(readForm()),
      });
    }
  } catch (e) {
    setState('error', 'Could not start the server.');
    showError(e.message);
  }
  await tick();
});

$('f-method').addEventListener('change', onMethodChange);
$('f-model').addEventListener('input', refreshPrimary);
$('f-host').addEventListener('change', onHostChange);

function onHostChange() {
  $('host-warning').hidden = $('f-host').value !== '0.0.0.0';
}

/* ── Boot ──────────────────────────────────────────────── */
(async function init() {
  try {
    await loadMethods();
    applyConfig(await api('/api/config'));
    await tick();
    poll = setInterval(tick, 1000);
  } catch (e) {
    showError(`Could not reach the panel backend: ${e.message}`);
  }
})();
