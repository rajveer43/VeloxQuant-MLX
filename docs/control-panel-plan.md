# Development Plan — VeloxQuant Control Panel (#33)

Status: **proposed** · Epic: [#33](https://github.com/rajveer43/VeloxQuant-MLX/issues/33) · Children: [#34](https://github.com/rajveer43/VeloxQuant-MLX/issues/34), [#35](https://github.com/rajveer43/VeloxQuant-MLX/issues/35), [#36](https://github.com/rajveer43/VeloxQuant-MLX/issues/36) · Hard dependency: [#27](https://github.com/rajveer43/VeloxQuant-MLX/issues/27)

This document answers the four open questions in #33 and lays out an implementation
sequence. It is a plan, not an implementation — no UI or server code lands with it.

---

## 1. Findings from the current tree

Verified against `master` at `0.42.1`:

| Claim | State on disk |
|---|---|
| `veloxquant serve` exists | **No.** [`veloxquant_mlx/__main__.py`](../veloxquant_mlx/__main__.py) dispatches only `precompute`, `benchmark`, `recommend`. |
| A patch hook the server can use exists | **Yes.** `patch_model_kv_cache(model, config)` in [`veloxquant_mlx/integration/mlx_lm_patch.py:24`](../veloxquant_mlx/integration/mlx_lm_patch.py#L24). |
| Cache methods are enumerable at runtime | **Partly.** 40+ cache modules under [`veloxquant_mlx/cache/`](../veloxquant_mlx/cache/), but there is no single exported registry mapping method-name → class → serve-tier. #35 needs one. |
| UI packaging surface exists | **No.** `pyproject.toml` has only a `dev` extra; `[project.scripts]` exposes `veloxquant` / `mlx-kv-quant`. |

Two consequences: **#34 cannot start** until a launcher exists, and **#35 needs a
registry built first** — it must not hardcode a marketing list of "41 methods".

---

## 2. Decisions on #33's open questions

### Q1 — Web UI vs native macOS first?

**Decision: local web UI (Python backend + single-page frontend), shipped as an
optional `[ui]` extra.**

Rationale:

- The panel's whole job is managing a Python subprocess and polling HTTP. A
  SwiftUI app would still shell out to the same Python process, so it buys
  fidelity of *feel* at the cost of a second toolchain, a second release
  pipeline, and code signing — none of which move the "under 5 minutes" success
  criterion.
- Everything ships through the existing PyPI release flow. No notarization.
- A Tauri/SwiftUI shell can wrap this later without rewriting the control plane,
  because the control plane is an HTTP API, not UI code.

Cost accepted: it will look like a local web app, not a Mac app. That is the
right trade for MVP; revisit after #34–#36 land and there is real usage.

### Q2 — Bundle a Hugging Face model list, or require a local path/id?

**Decision: accept a model id or local path; validate before Start; no download
manager.**

MVP resolves whatever the user types through the normal `mlx_lm.load` path and
remembers the last-used value. A curated shortlist of 3–5 known-good
`mlx-community` ids ships as *suggestions in a datalist*, not as a browsable hub
— #33 lists a download manager as an explicit non-goal.

### Q3 — Does MVP block on #27 option (d) (real compressed storage)?

**Decision: no. Ship on the (a) thin launcher, gated behind an honest banner.**

This is the load-bearing call in the plan, so the reasoning is explicit:

- #27's own recommendation is "(d), then (a)", and it names the counter-argument:
  (d) touches the core of a library whose credibility rests on its benchmark
  numbers, to chase a memory benefit no user has asked for yet.
- Blocking the panel on (d) means the adoption hook waits on the highest-risk
  change in the project. That inverts the risk ordering.
- The credibility risk #27 identifies is *a server that implies memory relief it
  cannot deliver*. That risk is addressable in the UI layer, and this plan
  addresses it in §4 as a hard requirement, not a nicety.

So: MVP serves via option (a), and every surface that shows bytes is labelled
**"Accounting-only — no runtime memory reduction"** until the backend proves
otherwise. #36's ratio display stays behind that same gate.

### Q4 — Which methods appear in the UI on day one?

Driven by #27's tier matrix, read from a runtime registry (§3, Phase 0):

| Tier | Count (per #27) | UI treatment |
|---|---|---|
| Serves, honest bytes | 0 | — (requires #27 (d)) |
| Serves, over-reports bytes | 35 | Selectable; badged `accounting-only` |
| Crashes | 5 — incl. the default `turboquant_prod`, plus `turboquant_mse`, `polar`, `qjl`, `spectral` | Listed, **disabled**, with the reason shown; never silently substituted |

Note the sharp edge: the library's *default* method is in the crash tier. The
launcher's default must be an explicitly chosen serve-tier method
(`turboquant_rvq`), not the library default — and the UI must say it changed it.

---

## 3. Implementation sequence

### Phase 0 — Method registry (new, unlisted prerequisite)

`veloxquant_mlx/cache/registry.py`: one exported mapping of method name →
`{class, family (quant/eviction/hybrid), serve_tier, blurb, docs_url,
relevant_config_fields, paper_deviation_note}`.

Serve-tier is *derived by probe*, not hand-maintained: a method is `serves` if it
subclasses `mlx_lm.models.cache.KVCache`, passes `can_trim_prompt_cache`, and
survives `deepcopy`. A test asserts the derived counts match #27's 35/5 split, so
drift fails CI instead of misleading the UI.

Unblocks: #35 entirely, #34's method dropdown, #27's docs matrix.

### Phase 1 — `veloxquant serve` (#27 option (a))

`veloxquant_mlx/cli/serve.py`, registered in `__main__.py`:

```
veloxquant serve --model <id|path> --method turboquant_rvq --bits 1 \
                 --host 127.0.0.1 --port 8000
```

Loads the model, applies `patch_model_kv_cache`, hands off to `mlx_lm.server`.
Roughly the ~10 lines #27 predicts, plus the honesty rules:

- Validate method against the registry **before** loading the model; exit
  non-zero with a readable message on a crash-tier method. No fp16 fallback,
  silent or otherwise.
- Print the accounting-only warning to stderr on every start.
- Emit a machine-readable ready line on stdout (`VELOXQUANT_READY {json}`) so the
  panel can distinguish "starting" from "running" without racing the health poll.

### Phase 2 — Control plane + panel MVP (#34)

`veloxquant_mlx/ui/` behind a `[ui]` extra, launched by `veloxquant panel`:

- Supervisor: spawn/terminate the Phase 1 subprocess, capture stdout/stderr into
  a ring buffer, expose lifecycle state (`stopped|starting|running|error`).
- Control API: `POST /api/server/start`, `POST /api/server/stop`,
  `GET /api/server/status`, `GET /api/config`, `PUT /api/config`.
- Frontend: status pill, Start/Stop, host/port, copyable `/v1` and `/health`
  endpoints, model bind, generation knobs. Endpoints are rendered from what the
  backend reports — `/metrics` and an Anthropic base URL appear only if actually
  served.
- Config persisted to `~/.veloxquant/panel.json`.

The control plane binds `127.0.0.1` only, always — separate from the *inference*
server's host setting, which may be `0.0.0.0`. Exposing the process-spawning API
on a LAN is not a user-configurable option.

### Phase 3 — Method panel (#35)

Consumes the Phase 0 registry: browser with family filter and tier badges,
method-specific knobs, and 2–3 presets. Preset names/numbers are drawn from
measured `BENCHMARK_RESULTS.md` values rather than invented.

### Phase 4 — Telemetry & logs (#36)

Log viewer works off the Phase 2 ring buffer and needs nothing new. The stats
half needs `/v1/kv/stats` on the server side (per #27) exposing
`compressed_*_bytes` / `fp16_*_bytes`; until then the panel shows the
accounting-only banner in place of a ratio, and shows nothing about RSS.

---

## 4. Honesty requirements (non-negotiable)

These exist because #27 identifies the server as exactly where users assume the
memory claim without reading caveats. Each maps to a test.

1. No silent fp16 fallback anywhere — unsupported method fails loudly at Start.
2. Byte figures are labelled with their mode; `accounting-only` never renders as
   a memory-saved claim, and RSS is never displayed unless measured.
3. Method counts in the UI come from the registry probe, never a literal `41`.
4. Endpoints are advertised only if the running backend serves them.
5. The launcher's default method differs from the library default, and the UI
   states this rather than papering over it.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| #27 chooses (d)-first, invalidating Phase 1 | Phase 1 is ~10 lines and Phase 2 talks to it over a process boundary; (d) changes what stats say, not the launcher's shape. |
| Registry probe drifts from #27's matrix | CI test asserts the 35/5 split; drift fails the build. |
| `[ui]` extra bloats the core install | Frontend is dependency-free static assets; the extra adds only the control-plane HTTP dep. Core `pip install VeloxQuant-MLX` is unchanged. |
| Panel is mistaken for a memory-savings demo | §4 banners; docs link it as "run VeloxQuant without Python", not "save memory". |

---

## 6. Definition of done for the epic

Install → `veloxquant panel` → pick model + method → Start → chat through the
OpenAI SDK at `http://127.0.0.1:8000/v1`, in under 5 minutes, with the active
method, its bits, and its accounting mode visible on screen the whole time.
