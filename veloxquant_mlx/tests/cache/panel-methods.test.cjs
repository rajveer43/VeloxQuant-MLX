const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(process.env.PANEL_JS || path.resolve(__dirname, '../../ui/static/panel.js'), 'utf8');

function harness(methods, defaultMethod = 'kivi') {
  const elements = new Map();
  const element = () => ({ value: '', innerHTML: '', children: [], appendChild(child) { this.children.push(child); } });
  const context = vm.createContext({
    METHODS: [], methodDiscoveryError: null,
    FIELDS: { model: 'f-model', method: 'f-method' },
    $: (id) => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
    document: { createElement: element },
    api: async () => ({ methods, default_serve_method: defaultMethod }),
    onMethodChange() {}, renderMethodTable() {}, onHostChange() {}, showError(message) { context.error = message; },
  });
  vm.runInContext(source.slice(source.indexOf('function availabilityLabel'), source.indexOf('function onMethodChange')), context);
  vm.runInContext(source.slice(source.indexOf('function applyConfig'), source.indexOf('/* ── Method knobs')), context);
  return { context, elements };
}

test('missing dependency keeps options disabled, clears selection, and exposes repair details', async () => {
  const { context, elements } = harness([{ name: 'kivi', family: 'quantization', is_servable: false, unsupported_reason: "cache construction failed: ModuleNotFoundError: No module named 'scipy'" }]);
  await context.loadMethods();
  context.applyConfig({ method: 'kivi', model: 'test-model' });
  assert.equal(elements.get('f-method').value, '');
  const option = elements.get('f-method').children[0].children[0];
  assert.equal(option.disabled, true);
  assert.match(option.textContent, /dependency error/);
  assert.match(context.error, /scipy/);
  assert.match(context.error, /restart/);
  assert.equal(elements.get('f-model').value, 'test-model');
});

test('unsupported saved/default method falls back to a supported option', async () => {
  const { context, elements } = harness([
    { name: 'polar', family: 'quantization', is_servable: false },
    { name: 'kivi', family: 'quantization', is_servable: true },
    { name: 'snapkv', family: 'eviction', is_servable: true },
  ], 'polar');
  await context.loadMethods();
  context.applyConfig({ method: 'polar' });
  assert.equal(elements.get('f-method').value, 'kivi');
  context.applyConfig({ method: 'snapkv' });
  assert.equal(elements.get('f-method').value, 'snapkv');
  assert.equal(context.error, null);
  assert.equal(elements.get('f-method').children[0].children[1].disabled, false);
  assert.match(elements.get('f-method').children[0].children[0].textContent, /unsupported for serving/);
});

test('malformed discovery data is rejected instead of interpreted as unsupported', async () => {
  const { context } = harness([{ name: 'kivi' }]);
  await assert.rejects(context.loadMethods(), /Invalid method-discovery response/);
});

test('empty catalog gives an actionable error and no selection', async () => {
  const { context, elements } = harness([]);
  await context.loadMethods();
  assert.equal(elements.get('f-method').value, '');
  assert.match(context.error, /No serving methods/);
});

test('status polling cannot clear a discovery error', () => {
  const { context, elements } = harness([]);
  context.escapeHtml = (value) => value;
  vm.runInContext(source.slice(source.indexOf('function showError'), source.indexOf('/* ── Endpoints')), context);
  context.methodDiscoveryError = 'Missing scipy';
  context.showError(null);
  assert.equal(elements.get('error-banner').hidden, false);
  assert.match(elements.get('error-banner').innerHTML, /Missing scipy/);
});
