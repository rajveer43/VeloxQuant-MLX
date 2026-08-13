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

**Decision: native macOS app (SwiftUI), managing a Python subprocess.**

Phases 0 and 1 are identical under either choice — they are Python either way —
so this decision only governs Phase 2.

Going native also *removes* a component rather than adding one. A web panel needs
a localhost control-plane HTTP server whose only purpose is letting a browser
spawn processes; that is both extra code and the most uncomfortable security
surface in the design. SwiftUI calls `Process` directly, so there is no control
API, no browser-can-spawn-processes problem, and no `[ui]` extra on PyPI.

Costs accepted, in order of how much they matter:

1. **Python discovery.** The app must find an interpreter with `veloxquant_mlx`
   and `mlx_lm` installed. This is the main source of "it doesn't work on my
   machine" and does not exist in the web version, which already runs inside the
   right interpreter. Mitigation: first-run picker that validates by import.
2. **Distribution.** Unsigned `.app` bundles are quarantined by Gatekeeper.
   Proper signing/notarization needs a paid Apple Developer account; without one,
   distribution is "build from source in Xcode," which is a worse first-run story
   than `pip install`. **Open — see §5.**
3. Two release pipelines, on separate cadences.

Toolchain verified present: Xcode 26.3, Swift 6.2.4, macOS 26.5.

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

### Phase 0 — Method registry — **shipped**

[`veloxquant_mlx/cache/registry.py`](../veloxquant_mlx/cache/registry.py) maps
method name → `{family, serve_tier, blurb, docs_url, config_fields,
paper_deviation}`. Method names are read from `KVCacheConfig`'s own `Literal`,
so declaring a cache registers it.

Serve tier is *derived by probe*, never declared: a method serves only if it
subclasses `mlx_lm`'s `KVCache`, survives a live `update_and_fetch`, passes
`can_trim_prompt_cache`, and survives `deepcopy`.

Probing confirmed #27's matrix exactly — **35 servable, 5 crashing**
(`turboquant_prod`, `turboquant_mse`, `polar`, `qjl`, `spectral`), all five for
the same reason: they do not subclass `mlx_lm`'s `KVCache` and so inherit
neither `update_and_fetch` nor `is_trimmable`.

`veloxquant methods [--json]` exposes this. The `--json` form is the contract
the Swift app decodes, so tier logic lives in Python only.

### Phase 1 — `veloxquant serve` — **shipped**

[`veloxquant_mlx/cli/serve.py`](../veloxquant_mlx/cli/serve.py):

```
veloxquant serve --model <id|path> --method turboquant_rvq --bits 2 \
                 --host 127.0.0.1 --port 8000
```

Verified end-to-end against `mlx-community/Llama-3.2-1B-Instruct-4bit`:
`/v1/chat/completions` and `/v1/completions` both generate correctly through a
real `TurboQuantRVQKVCache`, on `turboquant_rvq` and `kivi`.

Honesty rules, each covered by a test:

- Crash-tier and unknown methods exit non-zero **before** the model loads. No
  fp16 fallback.
- Accounting-only warning on stderr at every start, and in the ready handshake.
- `VELOXQUANT_READY {json}` on stdout once the cache is wired, so the panel can
  distinguish "starting" from "running" without racing a health poll.
- Requests naming a different model are refused, not silently served by the
  pinned one — otherwise the panel would display a method the response did not use.

**#27's "~10 lines" estimate was optimistic**, for three reasons found only by
running it:

1. `ModelProvider` reads ~23 fields off `cli_args`. Hand-listing them breaks
   whenever upstream adds an option, so we harvest mlx_lm's own parser defaults
   and override only our flags.
2. **MLX arrays are bound to the thread that created them.** Loading the model on
   the main thread and generating on mlx_lm's worker fails with
   `There is no Stream(gpu, N) in current thread.` Upstream avoids this by
   calling `load_default()` inside `_generate`; we hook `_load` to land on that
   same thread.
3. `is_batchable` is computed at the end of `_load`. Overriding the load without
   recomputing it silently routes every request down the unbatched path, which
   generates on the HTTP thread with no stream scope — the same crash by a
   different route.

None of these are visible without a live model, which is why the launcher could
not have been signed off from reading the code.

### Phase 2 — SwiftUI control panel (#34) — next

`VeloxQuantPanel.app`, macOS 14+, no third-party dependencies:

- `ServerController` (`@Observable`) — owns `Process` + `Pipe`, lifecycle
  `stopped | starting | running | error`, flips to `running` on `VELOXQUANT_READY`.
- `PythonEnvironment` — locates and validates an interpreter (`import
  veloxquant_mlx`) before enabling Start.
- `MethodCatalog` — shells `veloxquant methods --json`; no method metadata in Swift.
- UI — sidebar (Server / Methods / Logs / About), status pill, Start/Stop,
  host/port, copy-buttons, generation knobs. `UserDefaults` + `NSPasteboard`.

Endpoints are rendered from the handshake, so the panel cannot advertise
`/health` or `/metrics` — `mlx_lm.server` does not serve them.

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
| **No Apple Developer account → Gatekeeper quarantine** | **Open question.** Without signing, MVP distribution is "build from source", which undercuts the 5-minute goal for non-developers. Decide before Phase 2 ships, not at release. |
| Panel can't find the user's Python | `PythonEnvironment` validates by import and remembers the choice; Start stays disabled until an interpreter resolves. |
| mlx_lm changes its server internals | We hook `_load` and harvest its parser rather than copying its arg list; `test_serve_cli.py` fails loudly if either stops working. |
| #27 chooses (d)-first, invalidating Phase 1 | Phase 1 talks to the app over a process boundary; (d) changes what the stats *say*, not the launcher's shape. |
| Registry probe drifts from #27's matrix | `test_registry.py` asserts the 35/5 split and that no method claims `HONEST_BYTES`; drift fails the build. |
| Panel is mistaken for a memory-savings demo | §4 banners; docs link it as "run VeloxQuant without Python", not "save memory". |

---

## 6. Definition of done for the epic

Install → `veloxquant panel` → pick model + method → Start → chat through the
OpenAI SDK at `http://127.0.0.1:8000/v1`, in under 5 minutes, with the active
method, its bits, and its accounting mode visible on screen the whole time.
