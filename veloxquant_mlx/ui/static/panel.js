/* VeloxQuant-MLX control panel — talks to the local control API.
   No framework, no build step: this file is served as-is.

   Presentation rules this file must uphold (see docs/control-panel.md):
     - every number states its provenance: measured vs estimate
     - absent telemetry says "not reported", never 0 or a blank
     - key-only ratios are labelled key-only, never shown as whole-cache */

const $ = (id) => document.getElementById(id);

const FIELDS = {
  model: 'f-model', method: 'f-method', bits: 'f-bits',
  host: 'f-host', port: 'f-port',
  max_tokens: 'f-max-tokens', temp: 'f-temp', top_p: 'f-top-p',
};

let METHODS = [];
let state = 'stopped';
let logCount = 0;
let selectedMethod = null;
let filterFamily = '';
let filterServable = false;
let filterQuery = '';

/* ── Theme ─────────────────────────────────────────────── */
$('theme-toggle').addEventListener('click', () => {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('vq-theme', next); } catch (e) { /* ignore */ }
});

/* ── Views ─────────────────────────────────────────────── */
const VIEWS = ['server', 'methods', 'about'];

function showView(name, updateHash = true) {
  if (!VIEWS.includes(name)) name = 'server';
  document.querySelectorAll('.view').forEach((v) => v.classList.remove('is-active'));
  $(`view-${name}`).classList.add('is-active');
  document.querySelectorAll('.side-link').forEach((b) => {
    b.classList.toggle('is-active', b.dataset.view === name);
  });
  // Hash routing so a view can be linked and reload-stable. Kept out of
  // history so Back leaves the panel rather than cycling through tabs.
  if (updateHash) location.replace(`#${name}`);
  window.scrollTo(0, 0);
}

document.querySelectorAll('.side-link').forEach((b) => {
  b.addEventListener('click', () => showView(b.dataset.view));
});
window.addEventListener('hashchange', () => {
  showView(location.hash.replace('#', ''), false);
});
$('browse-methods').addEventListener('click', () => {
  selectMethodDetail($('f-method').value);
  showView('methods');
});

/* ── API ───────────────────────────────────────────────── */
async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `request failed (${res.status})`);
  return body;
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s == null ? '' : s;
  return d.innerHTML;
}

function humanBytes(n) {
  if (n == null) return null;
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = n, i = 0;
  while (Math.abs(v) >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${i === 0 ? v : v.toFixed(1)} ${units[i]}`;
}

/* ── Form ──────────────────────────────────────────────── */
function readForm() {
  const cfg = {};
  for (const [key, id] of Object.entries(FIELDS)) {
    const el = $(id);
    cfg[key] = el.type === 'number' ? Number(el.value) : el.value;
  }
  cfg.overrides = readKnobs();
  return cfg;
}

function applyConfig(cfg) {
  for (const [key, id] of Object.entries(FIELDS)) {
    if (cfg[key] !== undefined && cfg[key] !== null) $(id).value = cfg[key];
  }
  onHostChange();
  // Refresh the blurb/badges/telemetry line for the restored method before
  // painting knobs — otherwise the description belongs to the dropdown's
  // previous value while the selection says something else.
  onMethodChange(cfg.overrides || {});
}

/* ── Method knobs ──────────────────────────────────────── */
function renderKnobs(values) {
  const info = METHODS.find((m) => m.name === $('f-method').value);
  const box = $('knobs');
  const schema = (info && info.field_schema) || [];

  // bit_width_inlier already has a dedicated Bits input; showing it twice
  // would let the two disagree.
  const shown = schema.filter((f) => f.name !== 'bit_width_inlier');
  if (!shown.length) { box.innerHTML = ''; return; }

  const inputs = shown.map((f) => {
    const v = values && values[f.name] != null ? values[f.name] : f.default;
    const step = f.type === 'float' ? '0.01' : '1';
    const type = f.type === 'int' || f.type === 'float' ? 'number' : 'text';
    const placeholder = f.optional ? 'auto' : '';
    return `<div class="knob field">
      <label for="knob-${f.name}">${escapeHtml(f.name)}</label>
      <input id="knob-${f.name}" data-knob="${escapeHtml(f.name)}"
             data-type="${f.type}" type="${type}" step="${step}"
             placeholder="${placeholder}" value="${v == null ? '' : escapeHtml(v)}" />
      ${f.help ? `<p class="hint">${escapeHtml(f.help)}</p>` : ''}
    </div>`;
  }).join('');

  box.innerHTML = `<div class="knobs-head">${escapeHtml(info.name)} settings</div>
                   <div class="knob-grid">${inputs}</div>`;

  const locked = state === 'running' || state === 'starting';
  box.querySelectorAll('input').forEach((el) => { el.disabled = locked; });
}

function readKnobs() {
  const out = {};
  document.querySelectorAll('#knobs input[data-knob]').forEach((el) => {
    const raw = el.value.trim();
    if (raw === '') return;                       // blank = use the config default
    const t = el.dataset.type;
    out[el.dataset.knob] = t === 'int' ? parseInt(raw, 10)
                         : t === 'float' ? parseFloat(raw)
                         : raw;
  });
  return out;
}

/* ── Methods (server view dropdown) ────────────────────── */
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
      // Unsupported methods stay visible but unselectable — hiding five of
      // forty would misrepresent the catalog.
      opt.textContent = m.is_servable ? m.name : `${m.name} — unsupported`;
      opt.disabled = !m.is_servable;
      group.appendChild(opt);
    }
    select.appendChild(group);
  }
  select.value = data.default_serve_method;

  const servable = METHODS.filter((m) => m.is_servable).length;
  $('methods-summary').textContent =
    `${METHODS.length} methods · ${servable} servable · ${METHODS.length - servable} unsupported`;

  onMethodChange();
  renderMethodTable();
}

function onMethodChange(knobValues) {
  const info = METHODS.find((m) => m.name === $('f-method').value);
  const box = $('method-info');
  if (!info) { box.innerHTML = ''; return; }

  const tierBadge = info.is_servable
    ? '<span class="badge badge-accounting">accounting-only</span>'
    : '<span class="badge badge-unsupported">unsupported</span>';

  let html = `<div class="blurb"><span class="badge badge-family">${info.family}</span>${tierBadge}</div>`
    + `<div class="blurb">${escapeHtml(info.blurb)}</div>`;

  if (info.unsupported_reason) {
    html += `<div class="deviation"><strong>Cannot serve.</strong> ${escapeHtml(info.unsupported_reason)}</div>`;
  }
  if (info.paper_deviation) {
    html += `<div class="deviation"><strong>Adapted.</strong> ${escapeHtml(info.paper_deviation)}</div>`;
  }
  html += `<div class="hint">Telemetry: ${escapeHtml(info.coverage_label)}</div>`;

  box.innerHTML = html;
  renderKnobs(knobValues || {});
  refreshPrimary();
}

/* ── Methods view ──────────────────────────────────────── */
function visibleMethods() {
  return METHODS.filter((m) => {
    if (filterFamily && m.family !== filterFamily) return false;
    if (filterServable && !m.is_servable) return false;
    if (filterQuery && !m.name.includes(filterQuery) &&
        !m.blurb.toLowerCase().includes(filterQuery)) return false;
    return true;
  });
}

function renderMethodTable() {
  const rows = visibleMethods();
  const body = $('method-rows');

  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="4" class="method-empty">No methods match these filters.</td></tr>`;
    return;
  }

  body.innerHTML = rows.map((m) => `
    <tr data-method="${escapeHtml(m.name)}"
        class="${m.is_servable ? '' : 'is-unsupported'} ${m.name === selectedMethod ? 'is-selected' : ''}">
      <td><span class="method-name">${escapeHtml(m.name)}</span>${m.is_adapted ? ' <span class="badge badge-family">adapted</span>' : ''}</td>
      <td>${escapeHtml(m.family)}</td>
      <td class="${m.is_servable ? 'tier tier-ok' : 'tier tier-no'}">${escapeHtml(m.serve_tier_label)}</td>
      <td class="cov ${m.coverage === 'none' ? 'cov-none' : ''}">${escapeHtml(m.coverage_label)}</td>
    </tr>`).join('');

  body.querySelectorAll('tr[data-method]').forEach((tr) => {
    tr.addEventListener('click', () => selectMethodDetail(tr.dataset.method));
  });
}

function selectMethodDetail(name) {
  selectedMethod = name;
  const m = METHODS.find((x) => x.name === name);
  const box = $('method-detail');
  if (!m) { box.innerHTML = '<p class="hint">Select a method to see details.</p>'; return; }

  let html = `<h3>${escapeHtml(m.name)}</h3>
    <div class="detail-row">
      <span class="badge badge-family">${escapeHtml(m.family)}</span>
      <span class="badge ${m.is_servable ? 'badge-accounting' : 'badge-unsupported'}">${escapeHtml(m.serve_tier_label)}</span>
    </div>
    <p class="detail-blurb">${escapeHtml(m.blurb)}</p>`;

  if (m.unsupported_reason) {
    html += `<div class="deviation"><strong>Cannot serve.</strong> ${escapeHtml(m.unsupported_reason)}</div>`;
  }
  if (m.paper_deviation) {
    html += `<div class="deviation"><strong>Adapted from the paper.</strong> ${escapeHtml(m.paper_deviation)}</div>`;
  }

  // Coverage is spelled out because "not reported" and "zero compression"
  // look identical in a UI that only prints numbers.
  const covText = {
    keys_and_values: 'Reports compressed vs fp16 bytes for keys and values.',
    keys_only: 'Reports byte counters for <strong>keys only</strong> — not a whole-cache ratio.',
    none: 'Reports no byte counters. Eviction methods drop tokens rather than compress bytes, so a compression ratio would not be meaningful.',
  }[m.coverage] || '';
  html += `<div class="detail-row"><strong>Telemetry:</strong> ${covText}</div>`;

  if (m.config_fields && m.config_fields.length) {
    html += `<div class="detail-row"><strong>Settings:</strong> <code>${m.config_fields.join('</code>, <code>')}</code></div>`;
  }

  html += `<div class="detail-actions">`;
  if (m.is_servable) {
    html += `<button type="button" class="btn btn-filled" id="use-method">Use this method</button>`;
  }
  html += `<a class="detail-link" href="${escapeHtml(m.docs_url)}" target="_blank" rel="noopener">Docs ↗</a></div>`;

  box.innerHTML = html;

  const use = $('use-method');
  if (use) {
    use.addEventListener('click', () => {
      if (state === 'running' || state === 'starting') {
        alert('Stop the server before changing the method.');
        return;
      }
      $('f-method').value = m.name;
      onMethodChange();
      showView('server');
    });
  }
  renderMethodTable();
}

$('method-search').addEventListener('input', (e) => {
  filterQuery = e.target.value.trim().toLowerCase();
  renderMethodTable();
});
$('servable-only').addEventListener('change', (e) => {
  filterServable = e.target.checked;
  renderMethodTable();
});
document.querySelectorAll('#family-chips .chip').forEach((chip) => {
  chip.addEventListener('click', () => {
    filterFamily = chip.dataset.family;
    document.querySelectorAll('#family-chips .chip').forEach((c) => c.classList.remove('is-active'));
    chip.classList.add('is-active');
    renderMethodTable();
  });
});

/* ── Models ────────────────────────────────────────────── */
async function loadModels() {
  let data;
  try { data = await api('/api/models'); } catch (e) { return; }
  const list = $('model-suggestions');
  list.innerHTML = data.models.map(
    (m) => `<option value="${escapeHtml(m.repo_id)}">${escapeHtml(m.size_label)}</option>`
  ).join('');
  if (data.models.length) {
    $('model-hint').textContent =
      `Required. ${data.models.length} model(s) found in your local cache — start typing to pick one.`;
  }
}

/* ── Memory ────────────────────────────────────────────── */
function renderMemory(report) {
  const body = $('memory-body');
  const prov = $('memory-prov');

  if (!report) {
    prov.hidden = true;
    body.innerHTML = '<p class="hint">Available once the server is running.</p>';
    return;
  }
  prov.hidden = false;

  const row = (label, value, reason) => {
    // An absent value is stated as text. Rendering 0 here would assert
    // "no memory used", which is a claim we have not measured.
    const cell = value != null
      ? `<span class="mem-value">${escapeHtml(value)}</span>`
      : `<span class="mem-value is-absent">${escapeHtml(reason || 'not reported')}</span>`;
    return `<div class="mem-row"><span class="mem-label">${label}</span>${cell}</div>`;
  };

  body.innerHTML =
      row('Server process (RSS)', humanBytes(report.process.rss_bytes), report.process.unavailable_reason)
    + row('MLX active', humanBytes(report.mlx.active_bytes), report.mlx.unavailable_reason)
    + row('MLX peak', humanBytes(report.mlx.peak_bytes), report.mlx.unavailable_reason)
    + `<p class="mem-note">${escapeHtml(report.note)}</p>`;
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

  // Config locks while a server owns it: an editable form would describe a
  // server other than the one running.
  const locked = next === 'running' || next === 'starting';
  for (const id of Object.values(FIELDS)) $(id).disabled = locked;
  document.querySelectorAll('#knobs input').forEach((el) => { el.disabled = locked; });

  refreshPrimary();
}

function refreshPrimary() {
  if (state === 'running' || state === 'starting') return;
  const info = METHODS.find((m) => m.name === $('f-method').value);
  $('primary-btn').disabled = !($('f-model').value.trim() && info && info.is_servable);
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
    openai_base_url: 'Base URL', chat_completions: 'Chat',
    completions: 'Completions', models: 'Models',
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
        b.textContent = 'Copied'; b.classList.add('copied');
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

    if (status.state === 'running') {
      try { renderMemory(await api('/api/memory')); } catch (e) { renderMemory(null); }
    } else {
      renderMemory(null);
    }

    await refreshLogs();
  } catch (e) {
    setState('error', 'Lost contact with the panel backend.');
    showError(e.message);
  }
}

/* ── Actions ───────────────────────────────────────────── */
$('primary-btn').addEventListener('click', async () => {
  showError(null);
  $('primary-btn').disabled = true;

  try {
    if (state === 'running' || state === 'starting') {
      await api('/api/stop', { method: 'POST' });
      logCount = 0;
      $('log').innerHTML = '<span class="log-empty">No output yet.</span>';
      setState('stopped', 'Pick a model and press Start Server.');
      renderEndpoints(null);
      renderMemory(null);
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

// Wrapped, not passed directly: the listener receives an Event, which
// renderKnobs would treat as a values object.
$('f-method').addEventListener('change', () => onMethodChange());
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
    loadModels();
    if (location.hash) showView(location.hash.replace('#', ''), false);
    await tick();
    setInterval(tick, 1000);
  } catch (e) {
    showError(`Could not reach the panel backend: ${e.message}`);
  }
})();
