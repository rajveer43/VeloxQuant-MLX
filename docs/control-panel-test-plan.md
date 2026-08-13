# Control Panel — Test Plan and Results

What was tested for [#34](https://github.com/rajveer43/VeloxQuant-MLX/issues/34)
(and its prerequisites), how, and what it found. Written after execution, so
everything below was run rather than planned.

**Environment:** macOS 26.5.2, Apple Silicon (`applegpu_g16g`), Python 3.12.9,
`mlx_lm` 0.31.3, VeloxQuant-MLX 0.42.1.
**Model under test:** `mlx-community/Llama-3.2-1B-Instruct-4bit` — small enough
to reload repeatedly, real enough to exercise the full path.

---

## Strategy

The risk here is not "does the code run" but **"does it tell the truth."** The
panel makes claims about compression, endpoints, and readiness, and #27
established that confident-but-wrong numbers are the credibility risk for this
project. So the plan weights two things heavily:

1. **Live end-to-end runs over unit tests** for the server path. The three
   defects that mattered were all invisible to static reading and to isolated
   unit tests — they needed a real model on a real thread (see §4).
2. **Negative paths as first-class cases.** Refusals, crashes and lies are the
   failure modes worth catching, so unsupported methods, port conflicts and
   endpoint invention each get an explicit test.

Automated tests cover everything that does not need model weights. Anything
requiring a multi-GB download is manual and recorded in §3, because gating CI on
a model download would make the suite unrunnable for contributors.

---

## 1. Automated coverage

`pytest veloxquant_mlx/tests` — **1527 passed** (1506 pre-existing + 21 new).

### Registry — `tests/cache/test_registry.py` (14)

| Test | Guards against |
|---|---|
| `test_all_methods_discovered` | Registry silently losing a method |
| `test_crash_tier_matches_issue_27` | Drift from the published 35/5 split |
| `test_no_method_claims_honest_bytes_yet` | A false memory-savings claim reaching the UI |
| `test_every_servable_method_is_accounting_only` | Same, per method |
| `test_default_serve_method_is_servable` | Shipping a default that crashes |
| `test_unsupported_methods_explain_themselves` | Refusing without a reason |
| `test_adapted_methods_carry_deviation_notes` | "-adapted" without an explanation |
| `test_methods_cli_json_contract` | Breaking the schema a UI decodes |
| + 6 more | Memoization, filters, sorting, serialization, unknown-method errors |

### Serve CLI — `tests/cache/test_serve_cli.py` (8)

| Test | Guards against |
|---|---|
| `test_validate_method_rejects_crash_tier` | **Silent fp16 fallback** |
| `test_ready_handshake_shape` | Panel unable to detect readiness |
| `test_ready_handshake_advertises_only_real_endpoints` | Copy-buttons for endpoints that 404 |
| `test_mlx_server_defaults_are_readable` | Upstream changing its arg surface |
| + 4 more | Defaults, unknown methods, arg overrides |

### Panel — `tests/cache/test_panel.py` (21)

| Area | Tests | Notable |
|---|---|---|
| Refusal rules | 3 | Empty model, crash-tier, unknown method |
| Lifecycle | 4 | Idempotent stop, handshake promotion, malformed handshake ignored |
| Diagnostics | 4 | Port-in-use / OOM / model-not-found translated; stack frames skipped |
| Control API | 5 | Status, methods, 400s, static serving, **path traversal → 404** |
| Config | 3 | Round-trip, unknown-key allowlist, corrupt-file recovery |
| Log buffer | 2 | Bounded at capacity |

---

## 2. What is deliberately not automated

| Gap | Why | Mitigation |
|---|---|---|
| Real model load + inference | Multi-GB download would make CI unrunnable | Manual, §3 |
| Browser interaction (clicks) | No JS test infrastructure in this repo | Manual + screenshots, §3 |
| Concurrent requests | Single-user tool; batching is upstream's | Batchability asserted at load |
| `0.0.0.0` binding | Needs a second machine | Warning shown in UI and stderr |
| Dark theme rendering | Headless Chrome starts with empty `localStorage` | Tokens shared with landing page |

---

## 3. Manual end-to-end results

### 3.1 Phase 1 — `veloxquant serve`

| # | Check | Result |
|---|---|---|
| 1 | Server binds and `/v1/models` responds | ✅ up in ~1s |
| 2 | `/v1/chat/completions` generates | ✅ `"Yes"` (42 prompt tokens) |
| 3 | `/v1/completions` generates | ✅ `"Paris.\nThe capital of France"` |
| 4 | Handshake reports 16 layer caches | ✅ |
| 5 | Second method (`kivi`) works | ✅ `"Red."` |
| 6 | Crash-tier method refused pre-load | ✅ exit 1, no model load |
| 7 | Unknown method refused | ✅ lists valid methods |
| 8 | Request for a different model refused | ✅ readable error, not silently served |

**Compression actually in the path** — the important one, since a passing
request proves nothing if the cache was bypassed. Instrumenting `make_cache`
during generation:

```
cache class:         TurboQuantRVQKVCache
compressed_key_bytes: 3536
fp16_key_bytes:      13312     (≈3.8× accounting ratio)
```

### 3.2 Phase 2 — control panel

| # | Check | Result |
|---|---|---|
| 1 | Panel serves HTML/CSS/JS | ✅ all 200 |
| 2 | `/api/methods` returns 40, 5 unsupported | ✅ |
| 3 | Start via API → `starting` → `running` | ✅ ~3s |
| 4 | Handshake surfaces model/method/bits/layers | ✅ |
| 5 | Inference through panel-spawned server | ✅ `"Hello! How can I assist you today?"` |
| 6 | Endpoints rendered from handshake only | ✅ 4 shown; no invented `/health` |
| 7 | Log pane captures child stdout+stderr | ✅ 11 lines |
| 8 | **Stop frees the port** (#34 criterion 2) | ✅ 0 listeners after |
| 9 | Crash-tier method → 400 with reason | ✅ |
| 10 | Empty model → 400 | ✅ |
| 11 | Path traversal → 404 | ✅ |
| 12 | Port conflict → `error` state | ✅ (see §5) |
| 13 | Config locks while running | ✅ visible in screenshot |
| 14 | Version shown in nav | ✅ v0.42.1 |

Screenshots captured in both `running` and `error` states via headless Chrome.

---

## 4. Defects found

All three were in **my own** integration code, found only by running against a
live model. None were visible from reading the source.

### 4.1 `AttributeError: 'Namespace' object has no attribute 'draft_model'`

Pinning a pre-loaded model bypassed `ModelProvider._load`, but `mlx_lm.server`
reads ~23 fields off `cli_args`. **Fix:** harvest mlx_lm's own parser defaults
and override only our flags, so new upstream options cannot break us.

### 4.2 `RuntimeError: There is no Stream(gpu, 0) in current thread`

The one that cost the most time, and where I was **wrong twice**:

- First hypothesis — caches pinned across threads. Rebuilt them lazily. *Still failed.*
- Second hypothesis — `mlx_lm.generate`'s module-level thread-local stream.
  Tested directly. *Disproved.*
- What actually found it: running **stock `mlx_lm.server`** as a control. It
  worked, so the fault was ours, not MLX's.

Root cause: **MLX arrays are bound to the thread that created them.** We loaded
the model on the main thread; mlx_lm generates on a worker. Upstream avoids this
by calling `load_default()` inside `_generate`. **Fix:** hook `_load` so the
model is created on the generation thread.

*Lesson recorded because it will recur:* isolated unit tests passed in every
thread configuration. Only the full model path reproduced it, and only a control
run against upstream localised it.

### 4.3 Unbatched path taken silently

`is_batchable` is computed at the end of `_load`. Overriding the load left it
`False`, routing every request to `_serve_single` — which generates on the HTTP
thread with no stream scope, i.e. the same crash by another route. **Fix:**
recompute it using upstream's own criterion (`hasattr(c, "merge")`), so a cache
lacking `merge` still correctly opts out.

### 4.4 Unreadable startup errors

A port conflict surfaced as a `concurrent.futures` traceback tail. **Fix:**
`_diagnose()` matches known failure signatures and prefers exception lines over
stack frames.

Before → after:

```
server exited with code 1 before becoming ready.   File ".../thread.py", line 173, in submit |  raise R
server exited with code 1 before becoming ready: port 8203 is already in use
```

---

## 5. Verification of the honesty rules

Each design rule in [control-panel.md](control-panel.md#design-rules), and how
it is held:

| Rule | Verified by |
|---|---|
| No silent fp16 fallback | `test_validate_method_rejects_crash_tier` + manual #6; exits before model load |
| Byte figures labelled | Banner in UI, stderr, and handshake; `test_no_method_claims_honest_bytes_yet` |
| No hardcoded "41" | `test_all_methods_discovered`, `test_crash_tier_matches_issue_27` |
| Only real endpoints | `test_ready_handshake_advertises_only_real_endpoints` + manual #6 |
| Launcher default ≠ library default | `test_default_serve_method_is_servable` |
| Config locks while running | Manual #13 |

---

## 6. Known limitations

- **Accounting-only.** No runtime memory reduction; that needs #27 option (d).
- **One server at a time** per panel instance.
- **No auth** on the inference server — `0.0.0.0` warns but does not protect.
- **Not restart-safe**: quitting the panel stops the server it spawned (by
  design — it does not leave orphans holding ports).
- **35 of 40 methods** manually smoke-tested at 2 of 40; the remaining methods
  rely on the registry probe plus their own existing unit tests.

---

## 7. Reproducing

```bash
# Automated
.venv/bin/python3 -m pytest veloxquant_mlx/tests -q

# Panel only
.venv/bin/python3 -m pytest veloxquant_mlx/tests/cache/test_panel.py -q

# Manual end-to-end
.venv/bin/python3 -m veloxquant_mlx panel
# → Start with mlx-community/Llama-3.2-1B-Instruct-4bit, then:
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default_model","messages":[{"role":"user","content":"Say hi"}],"max_tokens":20}'
```
