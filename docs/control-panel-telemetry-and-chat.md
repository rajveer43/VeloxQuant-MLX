# Control Panel — Live Telemetry & Chat Tester

Status: **implemented** (see §10 for what actually shipped vs. this plan's
original design)
Implements: [`docs/control-panel-enhancements.md`](control-panel-enhancements.md) §4 + §5
Targets: [#36](https://github.com/rajveer43/VeloxQuant-MLX/issues/36), [#27](https://github.com/rajveer43/VeloxQuant-MLX/issues/27)
Depends on: §3 (measured memory panel) — already shipped in `veloxquant_mlx/ui/memory.py`

This is Slice 2 of the enhancement plan, written as one piece deliberately: a
stats endpoint without the presentation rules below is precisely the
"confident, wrong numbers" failure #27 exists to prevent, and the chat tester
without visible telemetry is a toy. Ship both together.

---

## 0. What exists today (verified against `master` @ `0762c9b`)

- **Two separate processes.** `veloxquant panel` (`veloxquant_mlx/ui/server.py`)
  is a stdlib `ThreadingHTTPServer` on `127.0.0.1:7860`. Pressing Start spawns
  `python -m veloxquant_mlx serve ...` as a **subprocess**
  (`veloxquant_mlx/ui/supervisor.py:ServerSupervisor.start`, line 99). The panel
  never imports the inference server — it only pipes its stdout/stderr and
  waits for a handshake line.
- **The handshake.** `veloxquant_mlx/cli/serve.py:emit_ready()` (line 264)
  prints `VELOXQUANT_READY {json}` once the model loads and the cache is
  wired, containing `host`, `port`, `endpoints.openai_base_url`, `method`,
  `bits`, `layer_caches`. `ServerSupervisor` parses this into `self._ready` and
  flips state to `"running"` only then (never on a timer) — see
  `supervisor.py:295`.
- **The actual server loop.** `run_server()` in `serve.py` (line 379) calls
  `mlx_lm.server.run(args.host, args.port, _Provider(server_args))` — a bare
  positional call, **no `handler_class` passed**, so it uses upstream's
  `APIHandler` untouched.
- **Upstream's hook point.** `mlx_lm.server.run()` (installed at
  `.venv/lib/python3.12/site-packages/mlx_lm/server.py:1735`) accepts
  `handler_class=APIHandler` as a keyword with a default — passing a subclass
  is a supported extension point, not a monkeypatch.
- **The live cache handle.** `mlx_lm.server.APIHandler.__init__` receives
  `response_generator` (a `ResponseGenerator`, `server.py:440`) as its first
  argument. `response_generator.prompt_cache` is an `LRUPromptCache` that
  already exposes `.nbytes` and `.stats_by_type()` (used internally at
  `server.py:462-465`). This is the live object to read from — not something
  we build.
- **Panel routing convention.** `PanelHandler.do_GET`/`do_POST`
  (`ui/server.py:82`, `147`) is a flat `if route == "/api/x": ...; return`
  chain. New routes follow this exact shape — no router abstraction exists or
  should be introduced.
- **Frontend polling convention.** `panel.js:tick()` (line 573) polls
  `/api/status`, conditionally `/api/memory`, and `/api/logs` every 1s via
  `setInterval(tick, 1000)` (line 664). New polling follows this pattern.
- **No WebSocket/SSE infrastructure exists anywhere in this codebase yet.**
  This plan is the first to introduce a streaming response.

**Consequence for this design:** since the panel and the inference server are
different processes bound to different ports, `/v1/kv/stats` must be served
*by the inference server* (it alone holds `response_generator`), and the panel
must *proxy* it, not read it directly. Same for chat completions: the panel
proxies `/v1/chat/completions` on the child server rather than the browser
talking to the child's port directly, because the child's port may be bound to
`0.0.0.0` with no auth (see `ui/server.py`'s module docstring on why the panel
itself stays loopback-only) — but more importantly because the panel is the
one place that knows whether a server is currently running at all.

---

## 1. Scope

**In scope:**
1. `/v1/kv/stats` endpoint on the `veloxquant serve` child process.
2. A telemetry proxy route + polling panel section that renders those stats
   under Finding A's coverage rules (§3 of the enhancement doc).
3. An SSE chat proxy + a minimal chat UI in the panel (message list, input,
   token count, tok/s).

**Out of scope (unchanged from the enhancement doc):**
- History/system-prompt editing, markdown rendering in the chat tester.
- Multi-server management, benchmark tab, native shell — all still deferred
  per §7 of the enhancement doc, untouched by this plan.
- Any change to what counters a cache method reports. This plan only
  *surfaces* existing `nbytes`/`stats_by_type()` data honestly; it does not
  add new instrumentation to the 17 eviction methods that report nothing.

---

## 2. `/v1/kv/stats` on the inference server

### 2.1 New handler class

Add `veloxquant_mlx/cli/telemetry.py`:

```python
"""GET /v1/kv/stats — read-only cache telemetry for the control panel.

Subclasses mlx_lm's APIHandler rather than patching it: `run()` already
accepts `handler_class` as a keyword (mlx_lm/server.py:1735), so this is
the extension point upstream provides, not a monkeypatch.
"""

from __future__ import annotations

import json
import time
from typing import Any

from mlx_lm.server import APIHandler


class TelemetryHandler(APIHandler):
    # Set by attach_cache() in serve.py once the cache list is known; a plain
    # module-level dict keyed by id(self.response_generator) would also work,
    # but a per-instance attribute set post-construction is simpler here
    # because ModelProvider._load runs once per process, not per request.
    kv_config: Any = None  # veloxquant_mlx.cache.KVCacheConfig, set externally
    kv_method: str | None = None
    kv_bits: int | None = None

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/kv/stats":
            self._handle_kv_stats()
            return
        super().do_GET()

    def _handle_kv_stats(self) -> None:
        payload = build_stats_payload(
            response_generator=self.response_generator,
            method=self.kv_method,
            bits=self.kv_bits,
        )
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def build_stats_payload(
    *, response_generator: Any, method: str | None, bits: int | None
) -> dict[str, Any]:
    """Pure function, unit-testable without an HTTP server or a real model.

    Encodes Finding A from control-panel-enhancements.md: never guesses a
    whole-cache ratio from partial data, and never renders a bare 0 for
    "not reported" — those two are indistinguishable to a user and #27 is
    explicit that indistinguishable-from-real is the failure mode to avoid.
    """
    from veloxquant_mlx.cache.registry import get_method_info

    info = get_method_info(method) if method else None
    coverage = (info.telemetry_coverage if info else None) or "unknown"
    # telemetry_coverage: "keys_and_values" | "keys_only" | "none" | "unknown"
    # (new registry field — see §2.3)

    prompt_cache = getattr(response_generator, "prompt_cache", None)
    n_sequences = len(prompt_cache) if prompt_cache is not None else 0

    result: dict[str, Any] = {
        "method": method,
        "bits": bits,
        "accounting_only": True,
        "coverage": coverage,
        "keys": None,
        "values": None,
        "tokens": None,
        "memory": _memory_block(),
    }

    if coverage == "none":
        # Eviction methods: report tokens kept vs seen, never a byte ratio.
        result["tokens"] = _token_counts(prompt_cache)
        return result

    if coverage == "unknown":
        # Explicitly distinct from "none": we don't know this method's shape.
        # Rendered by the client as "not reported", never as 0 or blank.
        return result

    keys, values = _byte_counts(prompt_cache, coverage)
    result["keys"] = keys
    result["values"] = values if coverage == "keys_and_values" else None
    return result


def _memory_block() -> dict[str, Any]:
    import psutil  # optional dep, already used by ui/memory.py

    try:
        import mlx.core as mx

        mlx_active = mx.get_active_memory()
        mlx_peak = mx.get_peak_memory()
    except Exception:
        mlx_active = mlx_peak = None

    try:
        rss = psutil.Process().memory_info().rss
    except Exception:
        rss = None

    return {
        "rss_bytes": rss,
        "mlx_active_bytes": mlx_active,
        "mlx_peak_bytes": mlx_peak,
        "source": "measured",
    }


def _byte_counts(prompt_cache: Any, coverage: str) -> tuple[dict, dict | None]:
    ...  # walk prompt_cache's per-layer caches, summing .nbytes-style
         # attributes that already exist per Finding B; see §2.4 for the
         # exact attribute path once confirmed against LRUPromptCache.


def _token_counts(prompt_cache: Any) -> dict[str, int]:
    ...  # seen vs retained, from cache.base's existing eviction bookkeeping
```

### 2.2 Wiring it into `serve.py`

`run_server()` (line 379) currently does:

```python
run(args.host, args.port, _Provider(server_args))
```

Change to:

```python
from veloxquant_mlx.cli.telemetry import TelemetryHandler

run(
    args.host,
    args.port,
    _Provider(server_args),
    handler_class=TelemetryHandler,
)
```

and inside `_Provider._load` (where `attach_cache()` already runs and
`n_layers`/`config` are in scope, line 411), set the class-level attributes
once:

```python
TelemetryHandler.kv_config = config
TelemetryHandler.kv_method = args.method
TelemetryHandler.kv_bits = args.bits
```

Class-level, not instance-level: `APIHandler` instances are constructed fresh
per-request by `_run_http_server`'s lambda factory
(`mlx_lm/server.py:1714-1721`), so there is no single long-lived handler
instance to attach state to. This mirrors how `response_generator` itself is
threaded through — a factory closure — but `response_generator` is a
constructor argument already, while `method`/`bits`/`config` are not, hence
the class attribute.

**Also add to `emit_ready()`'s payload** (`serve.py:264`): a
`"kv_stats_url": f"{base}/v1/kv/stats"` entry, so the panel does not have to
hardcode the path — same pattern already used for `endpoints.chat_completions`
etc. This keeps "endpoints advertised only if actually served" (enhancement
doc rule §4) true by construction, not by convention.

### 2.3 Registry: `telemetry_coverage` field

`veloxquant_mlx/cache/registry.py`'s method metadata (already carries `family`,
`tier`, `blurb`, `config_fields`, `docs_url` per the explored codebase) needs
one more field: `telemetry_coverage: Literal["keys_and_values", "keys_only",
"none", "unknown"]`.

This is exactly Finding A's table from the enhancement doc, made explicit and
machine-readable instead of living only in that markdown file:

| Coverage | Methods (from Finding A's probe) |
|---|---|
| `keys_and_values` | `kivi`, `vecinfer`, `xquant`, `palu`, … (13 total) |
| `keys_only` | `adakv`, `kitty`, `svdq`, `turboquant_rvq`, `xkv` (5 total) |
| `none` | all 17 eviction methods — `h2o`, `snapkv`, `tova`, `streaming_llm`, `pyramidkv`, … |

Populate this by re-running the same probe the enhancement doc's Finding A
used (it says "probing all 35 servable methods for byte counters" — reuse
that script/logic rather than hand-typing 35 entries, to keep it derived
from measurement, not asserted).

**Test:** one parametrized test asserting every servable method in the
registry has a non-`None` `telemetry_coverage`, so a newly added method can't
silently ship as `"unknown"` by omission.

### 2.4 Byte-counting implementation detail — needs one confirmation before coding

`_byte_counts()` above is stubbed because the exact attribute path from
`LRUPromptCache` down to per-layer byte counts wasn't fully traced in this
planning pass. Before implementing, spend ~30 minutes reading:
- `mlx_lm/server.py`'s `LRUPromptCache` class (constructor + `.nbytes`,
  `.stats_by_type()` — referenced at `server.py:462-465`).
- `veloxquant_mlx/cache/base.py`'s `KVCacheBuilder` and whatever per-cache
  attribute already backs the existing measured-memory panel's "compressed
  vs fp16 counters" row (`ui/memory.py` — §3, already shipped, so this
  attribute path already exists and should be reused, not reinvented).

The goal is one function that walks the same per-layer list `ui/memory.py`
already walks for the shipped memory panel, so `/v1/kv/stats` and the memory
card never disagree about what a "keys_only" cache reports.

---

## 3. Panel-side proxy + telemetry view

### 3.1 New route on the panel: `GET /api/telemetry`

In `veloxquant_mlx/ui/server.py`, following the exact convention of
`/api/memory` (line 138, which also reads `self.supervisor.status()` for the
child's PID first):

```python
if route == "/api/telemetry":
    status = self.supervisor.status()
    ready = status.get("ready")
    if status.get("state") != "running" or not ready:
        self._send_json({"available": False})
        return

    import urllib.request

    url = ready.get("kv_stats_url")
    if not url:
        # Older/mismatched child process without the new handshake field.
        self._send_json({"available": False, "reason": "server predates telemetry"})
        return

    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            stats = json.loads(resp.read())
    except Exception as exc:
        self._send_json({"available": False, "reason": str(exc)})
        return

    self._send_json({"available": True, **stats})
    return
```

`urllib.request` — stdlib, matching the module docstring's "stdlib only, the
panel must not add a runtime dependency" constraint. A 2s timeout keeps a
wedged child from freezing the panel's own request thread (the panel is
`ThreadingHTTPServer` per `ui/server.py`, so this blocks one worker thread,
not the whole panel — acceptable, matches how `/api/memory` already makes a
blocking `psutil` call inline).

### 3.2 Frontend: extend `tick()`, add a telemetry card

In `panel.js`'s `tick()` (line 573), alongside the existing
`if (status.state === 'running') { renderMemory(...) }` block (line 594):

```javascript
if (status.state === 'running') {
  try { renderMemory(await api('/api/memory')); } catch (e) { renderMemory(null); }
  try { renderTelemetry(await api('/api/telemetry')); } catch (e) { renderTelemetry(null); }
} else {
  renderMemory(null);
  renderTelemetry(null);
}
```

`renderTelemetry(data)` implements the enhancement doc's presentation rules
directly — this is the part that must not be shortcut:

```javascript
function renderTelemetry(data) {
  const el = $('telemetry-card');
  if (!data || !data.available) {
    el.innerHTML = '<span class="muted">No telemetry — start a server to see live cache stats.</span>';
    return;
  }

  const rows = [];
  if (data.coverage === 'keys_and_values') {
    rows.push(row('Keys', ratioText(data.keys), 'estimate'));
    rows.push(row('Values', ratioText(data.values), 'estimate'));
  } else if (data.coverage === 'keys_only') {
    rows.push(row('Keys (keys only — no value counters)', ratioText(data.keys), 'estimate'));
  } else if (data.coverage === 'none') {
    rows.push(row('Tokens retained', `${data.tokens.retained} / ${data.tokens.seen}`, 'measured'));
  } else {
    rows.push(row('Byte counters', 'not reported by this method', null));
  }

  rows.push(row('Process RSS', bytesText(data.memory.rss_bytes), 'measured'));
  rows.push(row('MLX active / peak', `${bytesText(data.memory.mlx_active_bytes)} / ${bytesText(data.memory.mlx_peak_bytes)}`, 'measured'));

  el.innerHTML = rows.join('');
}

function row(label, value, badge) {
  const badgeHtml = badge ? `<span class="badge badge-${badge}">${badge}</span>` : '';
  return `<div class="tele-row"><span>${label}</span><span>${value} ${badgeHtml}</span></div>`;
}
```

This is not decoration — `rows.push(row('Byte counters', 'not reported by
this method', null))` for the `unknown` branch is rule §8 from the
enhancement doc made literal: absent telemetry must render as a sentence, not
a blank, a dash, or a zero. Same for `'estimate'` vs `'measured'` badges on
every row — rule §7, "every number states its provenance in the UI, not just
in docs."

**Placement:** a `#telemetry-card` element in the Server view, directly under
the existing memory card (`ui/static/index.html`) — same section, so the
"estimate vs measured" comparison Finding B calls out ("the estimate says
3.8×, measured memory says otherwise, and the gap is visible") is literally
adjacent on screen, not on a separate tab a user has to remember to check.

### 3.3 Polling cost

Adds one more request per 1s tick while a server is `running` (matching the
existing `/api/memory` call already made on the same condition) — no new
interval, reuses `tick()`. No change to idle (`stopped`) behavior.

---

## 4. Chat tester

### 4.1 Panel-side SSE proxy: `POST /api/chat`

New route in `ui/server.py`'s `do_POST` (following the `/api/start` pattern at
line 150 for reading the body, but streaming the response instead of
buffering it — the first streaming response in this codebase):

```python
if route == "/api/chat":
    status = self.supervisor.status()
    ready = status.get("ready")
    if status.get("state") != "running" or not ready:
        self._send_json({"error": "no server is running"}, status=409)
        return

    body = self._read_json()
    base_url = ready["endpoints"]["chat_completions"]

    import urllib.request

    upstream_payload = json.dumps(
        {
            "model": ready["model"],
            "messages": body.get("messages", []),
            "stream": True,
            "max_tokens": body.get("max_tokens", 512),
            "temperature": body.get("temperature", 0.7),
        }
    ).encode()

    req = urllib.request.Request(
        base_url,
        data=upstream_payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Cache-Control", "no-store")
    self.end_headers()

    try:
        with urllib.request.urlopen(req, timeout=120) as upstream:
            for line in upstream:
                self.wfile.write(line)
                self.wfile.flush()
    except Exception as exc:
        # Best-effort: the SSE stream is already open, so report the error
        # as an SSE event rather than an HTTP error status (headers are sent).
        err = f"data: {json.dumps({'error': str(exc)})}\n\n".encode()
        self.wfile.write(err)
    return
```

This is a byte-for-byte relay of `mlx_lm.server`'s own SSE stream (the child
already speaks OpenAI-compatible SSE for `stream: true` — confirmed by
`endpoints.chat_completions` existing in the handshake and by `mlx_lm.server`
being an OpenAI-compatible server per `serve.py`'s module docstring). The
panel does not parse or reconstruct SSE framing — it forwards it, which is
the lowest-risk way to add the first streaming route to this codebase.

**Why proxy instead of the browser hitting the child directly:** the child
may be bound to `0.0.0.0` (user's choice, with the existing host-warning in
the UI) with no auth. Routing chat through the loopback-only panel means the
panel's own access pattern doesn't change based on how the user configured
the child's bind address — one code path regardless.

### 4.2 Frontend: chat view

New `ChatView` section (sibling to the existing `Server`/`Methods`/`About`
views, `#chat` hash), minimal per the enhancement doc's explicit non-goals
(no history persistence, no system prompt, no markdown):

```javascript
async function sendChat(userText) {
  appendMessage('user', userText);
  const messages = collectMessageHistory(); // in-memory array, cleared on reload — no persistence by design

  const el = appendMessage('assistant', '');
  const start = performance.now();
  let tokenCount = 0;

  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages }),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buf.indexOf('\n\n')) !== -1) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const line = chunk.replace(/^data: /, '').trim();
      if (line === '[DONE]' || !line) continue;

      const parsed = JSON.parse(line);
      if (parsed.error) { el.textContent += `\n[error: ${parsed.error}]`; continue; }
      const delta = parsed.choices?.[0]?.delta?.content || '';
      el.textContent += delta;
      if (delta) tokenCount++;
    }
  }

  const elapsed = (performance.now() - start) / 1000;
  appendStats(el, tokenCount, elapsed);
}
```

Message list: `<div>` elements appended to a scrolling container, `textContent`
only — explicitly no `innerHTML` on model output, since that's a stored-XSS
vector the moment a model is asked to echo HTML (deliberately not "no markdown
rendering" as a scope choice alone; it is also a security boundary, worth
stating as such even though the enhancement doc frames it as scope).

### 4.3 Chat view guard

The chat input is disabled whenever `status.state !== 'running'` (reuse the
existing `state` tracking `tick()` already maintains) — mirrors how the panel
already disables the Start button while `starting`, so there's no new pattern
to design, just one more place that reads the same state variable.

---

## 5. Testing

Following the existing test file structure (`tests/cache/test_panel.py`,
`test_registry.py`, `test_serve_cli.py`):

1. **`build_stats_payload()` unit tests** (new, e.g.
   `tests/cache/test_telemetry.py`) — pure function, no HTTP server or real
   model needed. Cover all four `coverage` branches with fake
   `response_generator`-like stand-ins. This is the highest-value test in
   this plan: it directly encodes rules §7–9 from the enhancement doc and
   should fail loudly if a future edit reintroduces a bare `0` or blank for
   the `unknown` case.
2. **Registry test**: every servable method has non-`None`
   `telemetry_coverage` (§2.3).
3. **Panel proxy tests** (`test_panel.py`): `/api/telemetry` returns
   `{"available": false}` when `state != "running"`, and forwards/wraps a
   fake upstream response when running (mock `urllib.request.urlopen`, don't
   spin up a real child process).
4. **`/api/chat` test**: returns 409 when no server running; for the
   streaming case, assert the proxy forwards bytes verbatim from a mocked
   upstream iterator without attempting to parse them (i.e., test that a
   malformed/partial SSE chunk from upstream doesn't crash the proxy — it
   should relay-and-flush regardless of content).
5. **No test asserts a specific compression ratio number** — per existing
   codebase convention (measured values live in `BENCHMARK_RESULTS.md`, not
   hardcoded in tests), only that the *labeling and branching logic* is
   correct.

---

## 6. Sequencing within this plan

```
1. Registry: add telemetry_coverage field + backing test         (no server changes, safe standalone)
2. telemetry.py: build_stats_payload() + unit tests               (pure function, no HTTP yet)
3. TelemetryHandler + serve.py wiring + emit_ready() kv_stats_url  (first real endpoint)
4. Panel: /api/telemetry proxy route                              (stdlib urllib, follows /api/memory)
5. Frontend: renderTelemetry() + tick() wiring + telemetry card    (Slice 2 "done" checkpoint)
6. Panel: /api/chat SSE proxy route                                (first streaming route — extra care)
7. Frontend: ChatView + sendChat()                                 (ships after telemetry is visibly working)
```

Steps 1–5 are independently shippable and should land as their own PR — they
are the `/v1/kv/stats` half of this plan and the enhancement doc's stated
dependency ("§4 depends on §3 landing first... §3 already shipped").
Steps 6–7 depend on 1–5 being live, per the enhancement doc's own framing
("combined with §3, lets telemetry move while you watch — which is the actual
demo").

---

## 7. Rules this plan must not break

Carried forward verbatim from `control-panel-enhancements.md` §9 — restated
here because this plan is the first to actually exercise most of them:

1. No silent fp16 fallback.
2. Byte figures labelled by mode; RSS shown only when measured.
3. Method lists from the registry probe, never a hardcoded count.
4. Endpoints advertised only if actually served — this is why
   `kv_stats_url` is added to `emit_ready()`'s payload rather than assumed
   client-side from a fixed path.
5. `running` requires the handshake, never a timer — the `/api/telemetry`
   and `/api/chat` proxy routes both check `status.get("state") ==
   "running"` before doing anything, not just "is there a PID."
6. Config locked while a server owns it — unaffected by this plan, no new
   config surface.
7. Every number states its provenance (*measured* / *estimate*) in the UI —
   §3.2's `renderTelemetry()`.
8. Absent telemetry says so, never `0`/`—`/blank — §3.2's `unknown` branch,
   §5's test #1.
9. Key-only ratios labelled key-only, never presented as whole-cache — §3.2's
   `keys_only` branch renders the label `"Keys (keys only — no value
   counters)"` inline, not just a tooltip that can be missed.

---

## 8. Open questions to resolve before writing code

1. **Exact attribute path for per-layer byte counts** (§2.4) — the plan
   identifies where to look (`ui/memory.py`'s existing walk +
   `LRUPromptCache`) but does not hand-derive it, since guessing here risks
   the exact "confident, wrong numbers" failure this whole plan exists to
   avoid. Trace it before implementing `_byte_counts()`.
2. **Token seen/retained bookkeeping for eviction methods** (`_token_counts()`
   stub) — confirm which existing attribute on eviction caches
   (`cache/base.py` or per-method files) already tracks this, since several
   eviction methods must already count evicted-vs-total for their own
   trimming logic; reuse rather than add new counters.
3. **`get_method_info()` naming** — the registry function name used in
   `build_stats_payload()` above is inferred from the sibling
   `get_method()`/`list_methods()` names seen in `serve.py`/`ui/server.py`;
   confirm the actual accessor for a single method's metadata before coding.

---

## 10. What actually shipped (post-implementation notes)

Both open questions in §8 resolved against the real code before
implementation began, and one design point in §2 turned out to be wrong once
traced against the installed `mlx_lm` package. Recorded here so this doc
reflects what's actually running, not the pre-implementation guess.

### §8.1 and §8.3 resolved

- Registry accessor is `get_method(name)` (not `get_method_info`), and
  `telemetry_coverage` / `TelemetryCoverage` already existed on `master`
  before this work started (`cache/registry.py`) — Slice 1 had shipped
  further than this plan's initial exploration pass caught. `MethodInfo`
  already carries a `coverage: TelemetryCoverage` field.
- Byte counters: `compressed_key_bytes`, `fp16_key_bytes`,
  `compressed_value_bytes`, `fp16_value_bytes` are real per-cache
  `@property` attributes already present on every method with byte
  telemetry (e.g. `cachegen_cache.py`, `gear_cache.py`). `ui/memory.py`
  does **not** walk these today — it only reports process RSS from
  outside the server process, and explicitly defers MLX active/peak
  memory as "coming soon" pending exactly this endpoint. `_byte_counts()`
  in `cli/telemetry.py` is the first place these per-layer counters are
  actually aggregated.
- Token counters: eviction caches share `tokens_seen` / `tokens_kept`
  properties (`h2o_cache.py`, `tova_cache.py`, `pyramidkv_cache.py`,
  `streaming_llm_cache.py`; `snapkv_cache.py` has `tokens_kept` only).
  `_token_counts()` uses `getattr(..., 0)` defensively per-attribute, but
  reports `None` (not a fabricated `{"seen": 0, "retained": 0}`) when a
  cache has neither attribute at all.

### §2's live-cache handle was wrong — corrected design

The plan's original §2.1 assumed `response_generator.prompt_cache`
(`mlx_lm.server.ResponseGenerator`, backed by `LRUPromptCache`) was a live
handle to "the cache currently being generated with." Tracing
`mlx_lm/models/cache.py` and `mlx_lm/server.py` directly showed this is
wrong: `LRUPromptCache` is a trie/LRU of *historical* prompt-prefix
snapshots, keyed by token prefix, used for cross-request cache reuse. The
cache actually mutating during a single request's generation is a **local
variable** inside `ResponseGenerator._generate` (either pulled from the
trie or built fresh via `model.make_cache()`), never stored as an attribute
anywhere reachable from outside that method.

**What shipped instead** (`veloxquant_mlx/cli/telemetry.py`): `attach_cache()`
in `cli/serve.py` is the one place a per-layer cache list is actually
constructed (`model.make_cache = <closure calling KVCacheBuilder.for_model>`,
per `attach_cache`'s existing docstring). The closure now also calls
`telemetry.record_live_caches(caches, method=..., bits=...)` on every
invocation, storing the list in a module-level `_LIVE_CACHES` (not an
instance attribute — `APIHandler` instances are constructed fresh per
request by `mlx_lm`'s own factory closure, so there is no single long-lived
handler instance to hold this on). `TelemetryHandler._handle_kv_stats` reads
`_LIVE_CACHES` directly. This is simpler than the original design and needs
no weak references: `mlx_lm` calls `make_cache()` fresh per request, so the
module-level slot always holds whichever list is currently in use.

### Bug found and fixed along the way: `attach_cache` recursion

While testing `attach_cache`, calling the newly-patched `model.make_cache()`
after `attach_cache` had run triggered a silent, pre-existing recursion bug
on `master` (present before this work, not introduced by it): `for_model()`
in `cache/base.py` unconditionally probes `getattr(model, "make_cache",
None)` as a hybrid-attention-layer heuristic and calls it. Once
`attach_cache` patches `model.make_cache` to a closure that itself calls
`for_model`, that probe finds the closure itself and recurses until
`RecursionError`, silently caught by `for_model`'s broad `except Exception`
and falling back correctly — but paying for a full recursion-limit's worth
of wasted stack frames and an exception catch on **every single
`make_cache()` call in production**, not just in tests.

Fixed in `attach_cache` (`cli/serve.py`): capture the model's true native
`make_cache` (or a stand-in returning `None` if it had none) once, and swap
it in for the duration of each `for_model()` call inside the closure, so
`for_model`'s probe sees the real native behavior instead of our own
wrapper. Regression-tested in `test_serve_cli.py`
(`test_attach_cache_does_not_recurse_into_its_own_wrapper`,
`test_attach_cache_preserves_a_real_native_make_cache`).

### Files touched

- `veloxquant_mlx/cli/telemetry.py` (new) — `TelemetryHandler`,
  `build_stats_payload`, `_byte_counts`, `_token_counts`, `_memory_block`,
  `record_live_caches`.
- `veloxquant_mlx/cli/serve.py` — `attach_cache` records live caches and
  fixes the recursion bug; `emit_ready` adds `endpoints.kv_stats` (schema
  v2); `run_server` passes `handler_class=TelemetryHandler`.
- `veloxquant_mlx/ui/server.py` — `/api/telemetry` (GET, proxies
  `kv_stats`) and `/api/chat` (POST, proxies `chat_completions` as SSE)
  routes, plus `_fetch_telemetry`/`_proxy_chat` helpers.
- `veloxquant_mlx/ui/static/{index.html,panel.css,panel.js}` — telemetry
  card (reuses the existing `.mem-row`/`.prov` vocabulary rather than
  inventing new classes) and a new Chat view/nav entry.
- Tests: `tests/cache/test_telemetry.py` (new),
  `tests/cache/test_panel_slice2.py` (new), additions to
  `tests/cache/test_serve_cli.py`.

All existing tests continued to pass throughout (1076 passed, 1 skipped,
one unrelated pre-existing warning, across the full `tests/cache/` suite).
