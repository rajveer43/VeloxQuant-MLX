# VeloxQuant Control Panel

Run a local, OpenAI-compatible server backed by VeloxQuant KV-cache compression
— without writing the `KVCacheConfig` → `KVCacheBuilder` → `model.make_cache`
recipe.

Implements [#34](https://github.com/rajveer43/VeloxQuant-MLX/issues/34) under
epic [#33](https://github.com/rajveer43/VeloxQuant-MLX/issues/33).

---

## Quick start

```bash
veloxquant panel
```

A browser opens at `http://127.0.0.1:7860`. Then:

1. **Model** — enter an MLX model id or local path, e.g.
   `mlx-community/Llama-3.2-1B-Instruct-4bit`
2. **KV compression** — pick a method and bit width (defaults to
   `turboquant_rvq`, 2-bit)
3. **Start Server** — the pill goes `Starting` → `Running`
4. Copy the **Base URL** and point any OpenAI-compatible client at it

From a source checkout, use `python3 -m veloxquant_mlx panel` instead.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--port` | `7860` | Port for the panel itself |
| `--no-browser` | off | Don't open a browser window |

The panel always binds `127.0.0.1`. This is **not** configurable: its API spawns
processes, and exposing that to a network would be a remote-execution hole. The
*inference* server's address is separate and can be set to `0.0.0.0` in the UI.

---

## Using the server

Once running, the panel shows copyable endpoints. With the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="default_model",
    messages=[{"role": "user", "content": "Say hi"}],
    max_tokens=50,
)
print(response.choices[0].message.content)
```

Or with curl:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default_model","messages":[{"role":"user","content":"Say hi"}],"max_tokens":50}'
```

`model` is `default_model` — the server is pinned to whatever you started it
with. Naming a different model returns an error rather than quietly loading it,
so the method and bits shown in the UI always describe the model that answered.

---

## CLI equivalents

The panel is a front-end for these; both work standalone.

```bash
# List methods and their serving support tiers
veloxquant methods
veloxquant methods --servable-only
veloxquant methods --json          # machine-readable

# Start a server directly
veloxquant serve \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --method turboquant_rvq --bits 2 \
  --host 127.0.0.1 --port 8000
```

`veloxquant serve` prints a `VELOXQUANT_READY {json}` line on stdout once the
model is loaded and caches are wired. The panel watches for it to distinguish
"still loading" from "actually ready" — a health poll alone cannot tell those
apart.

---

## What the numbers mean

> **Compression is accounting-only.** Caches store *dequantized* fp16 tensors.
> The byte counters measure compression fidelity, not memory saved at runtime.
> Do not read them as RSS reduction.

This is why the panel carries a permanent banner. The library's purpose is
measuring fidelity/compression trade-offs, which is entirely valid — but a
server is exactly where users assume a memory claim without reading caveats. See
[#27](https://github.com/rajveer43/VeloxQuant-MLX/issues/27) for the full
analysis and the plan to make storage genuinely compressed.

### Reading the numbers

Every figure in the panel is tagged with where it came from:

| Tag | Meaning |
|---|---|
| **measured** | Read from the running process. A fact. |
| **estimate** | Cache byte counters. Accounting-only — see above. |

They are **not comparable**, and the panel says so: measured RSS will not fall
as compression ratios rise, because no memory is actually saved yet.

MLX's own memory counters are deliberately *not* shown. They are process-local,
so querying them from the panel would describe the panel rather than the server
— true, but misleading beside the server's RSS. They become available with
`/v1/kv/stats` ([#27](https://github.com/rajveer43/VeloxQuant-MLX/issues/27)).

### Telemetry coverage

Byte counters are uneven across the catalog, so the Methods tab states which
case applies per method:

| Coverage | Count | Shown as |
|---|---|---|
| Keys and values | 13 | full ratio |
| Keys only | 5 | ratio labelled **keys only** — never as whole-cache |
| Not reported | 17 | *"not reported"*, never `0` |

The 17 are the eviction methods: they drop tokens rather than compress bytes, so
a byte ratio would not be meaningful. Note the serve default,
`turboquant_rvq`, is **keys only**.

### Method support tiers

Derived by probing each cache at runtime, never hand-maintained:

| Tier | Count | Meaning |
|---|---|---|
| Serves, honest bytes | 0 | Requires real compressed storage (#27) |
| Serves, accounting-only | 35 | Works today; byte counters over-report |
| Unsupported | 5 | Crashes under `mlx_lm.server` |

The five unsupported methods — `turboquant_prod`, `turboquant_mse`, `polar`,
`qjl`, `spectral` — do not subclass `mlx_lm`'s `KVCache`, so they inherit
neither `update_and_fetch` nor `is_trimmable`. They appear in the UI but are
disabled, with the reason shown; they are never silently substituted.

**Note:** `turboquant_prod` is the *library's* default and is in the crash tier.
`veloxquant serve` therefore defaults to `turboquant_rvq` instead.

---

## Architecture

```
Browser  ──HTTP──▶  Control plane        ──spawn──▶  veloxquant serve
(panel.js)          (ui/server.py)                   (cli/serve.py)
                    (ui/supervisor.py)                     │
                                                           ▼
                                                     mlx_lm.server
                                                    + VeloxQuant cache
```

The panel never runs inference in its own process. It spawns exactly the command
you could type yourself, and reports what that process announces.

| File | Role |
|---|---|
| [`ui/server.py`](../veloxquant_mlx/ui/server.py) | Static files + JSON control API |
| [`ui/supervisor.py`](../veloxquant_mlx/ui/supervisor.py) | Owns the child process, log buffer, state machine |
| [`ui/config.py`](../veloxquant_mlx/ui/config.py) | Persists settings to `~/.veloxquant/panel.json` |
| [`ui/static/`](../veloxquant_mlx/ui/static/) | Dependency-free HTML/CSS/JS |
| [`cache/registry.py`](../veloxquant_mlx/cache/registry.py) | Method catalog + serve-tier probe |
| [`cli/serve.py`](../veloxquant_mlx/cli/serve.py) | The launcher |

### Control API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/status` | State, ready payload, version, last error |
| `GET` | `/api/methods` | Catalog with tiers, coverage, field schema |
| `GET` | `/api/models` | Models already in the local HF cache |
| `GET` | `/api/memory` | Measured process memory |
| `GET` | `/api/logs?since=N` | Incremental log lines |
| `GET`/`POST` | `/api/config` | Persisted settings |
| `POST` | `/api/start` | Start a server |
| `POST` | `/api/stop` | Stop it, freeing the port |

### Views

`Server` · `Methods` · `About`, deep-linkable via `#server`, `#methods`,
`#about`.

The **Methods** tab lists every method with family filters, search, serve tier,
telemetry coverage, paper-deviation notes and a docs link. Unsupported methods
stay listed and disabled with their reason — never hidden.

### Method-specific settings

Knobs are derived from `KVCacheConfig` itself — names, types, defaults and
optionality — so the form cannot drift from what the config accepts. Selecting
KIVI shows `kivi_group_size`; SVDq shows `svdq_rank` and friends. Values reach
the server as repeated `--set FIELD=VALUE`, which works on the CLI too:

```bash
veloxquant serve --model <id> --method kivi --bits 2 --set kivi_group_size=64
```

Validation runs in the panel *and* the CLI, so an invalid knob is refused
immediately with the field named, rather than dying in the subprocess.

### Lifecycle

```
stopped ──start──▶ starting ──READY handshake──▶ running
   ▲                   │                            │
   └───────────────────┴──── stop ─────────────────┘
                       │
                       ▼ (child exits early)
                     error
```

`running` is entered **only** on the handshake — never on a timer — so the UI
cannot claim readiness while a model is still loading.

---

## Design rules

These exist because #27 identified the server as where users assume memory
claims. Each maps to a test.

1. **No silent fp16 fallback.** Unsupported methods fail at Start with a
   readable reason.
2. **Byte figures are labelled.** Accounting-only never renders as memory saved.
3. **Method lists come from the registry probe**, never a literal "41".
4. **Endpoints are advertised only if the backend reports them.** `/health` and
   `/metrics` do not appear, because `mlx_lm.server` does not serve them.
5. **The launcher default differs from the library default**, and says so.
6. **Config locks while running**, so the UI cannot describe a server that is
   not what is running.
7. **Every number states its provenance** — *measured* or *estimate* — in the
   UI, not only in docs.
8. **Absent telemetry says so.** A method without counters shows "not reported",
   never `0` or a blank that reads as zero.
9. **Key-only ratios are labelled key-only**, never presented as whole-cache.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `port N is already in use` | Another server holds it. Change the port or stop the other process. |
| `'<method>' cannot be served` | Crash-tier method. Pick a servable one. |
| `model not found on Hugging Face` | Check the id, or pass a local path. |
| `ran out of memory loading the model` | Model too large for available RAM. Try a smaller or more quantized one. |
| Pill flips to `Error` while running | The child died. The log pane keeps its output. |

Startup failures are translated into one actionable line rather than a raw
traceback; the full output stays in the log pane.

---

## Not yet built

- Live KV telemetry / compression charts — [#36](https://github.com/rajveer43/VeloxQuant-MLX/issues/36), needs a stats endpoint from #27
- Full method browser with filters and presets — [#35](https://github.com/rajveer43/VeloxQuant-MLX/issues/35)
- Model download manager, auth, themes beyond light/dark
- Native macOS (SwiftUI) shell — deferred; see
  [control-panel-plan.md](control-panel-plan.md)
