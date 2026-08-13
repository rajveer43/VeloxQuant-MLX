# Control Panel — Enhancement Plan

Status: **proposed** · Builds on [#37](https://github.com/rajveer43/VeloxQuant-MLX/pull/37) · Targets [#35](https://github.com/rajveer43/VeloxQuant-MLX/issues/35), [#36](https://github.com/rajveer43/VeloxQuant-MLX/issues/36)

What to build next on the shipped panel, ordered by value per unit of risk. Every
item below was checked against the current tree; findings that change the plan
are called out rather than assumed.

---

## 0. Findings that shape this plan

Measured on `master` + #37, not inferred.

### Finding A — telemetry coverage is uneven, and this is the headline constraint

Probing all 35 servable methods for byte counters:

| Counters exposed | Count | Methods |
|---|---|---|
| Keys **and** values | 13 | `kivi`, `vecinfer`, `xquant`, `palu`, … |
| Keys only | 5 | `adakv`, `kitty`, `svdq`, `turboquant_rvq`, `xkv` |
| **None** | **17** | all eviction methods — `h2o`, `snapkv`, `tova`, `streaming_llm`, `pyramidkv`, … |

**This inverts the naive version of #36.** A "compression ratio" panel would be
blank or misleading for half the catalog, and *silently* so. Note the default
serve method, `turboquant_rvq`, is key-only — so even the happy path cannot show
a whole-cache ratio.

It also makes sense: eviction methods don't compress bytes, they *drop tokens*.
Their honest metric is tokens-kept-vs-seen, not bytes. One number cannot describe
both families, and pretending otherwise is exactly the failure mode #27 warns about.

### Finding B — real memory *is* measurable today

`mx.get_active_memory()` and `mx.get_peak_memory()` both exist in the installed
MLX, and `psutil` 7.2.2 is available for process RSS.

This matters more than it first appears. #27 says we must not *claim* memory
savings — it does not say we must stay silent about memory. We can show
**measured** process memory as fact, next to the accounting-only counters
labelled as estimate. That is strictly more honest than today's panel, which
shows neither, and it turns the accounting-only caveat from an apology into a
visible, verifiable comparison.

### Finding C — no stats endpoint exists

`veloxquant serve` exposes only what `mlx_lm.server` does. Live telemetry needs a
new endpoint, which is #27's `/v1/kv/stats` proposal — still unbuilt.

---

## 1. Priorities

Ordered by *user-visible value ÷ risk*, not by issue number.

| # | Enhancement | Issue | Value | Risk | Verdict |
|---|---|---|---|---|---|
| 1 | Method browser + presets | #35 | High | Low | **Do first** |
| 2 | Measured memory panel | #36 | High | Low | **Do first** |
| 3 | `/v1/kv/stats` + live telemetry | #36/#27 | High | Medium | Do second |
| 4 | Built-in chat tester | — | High | Low | Do second |
| 5 | Model picker from local cache | #34 | Medium | Low | Cheap win |
| 6 | Multi-server management | — | Medium | Medium | Defer |
| 7 | Native macOS shell | #33 | Medium | High | Defer |
| 8 | Benchmark tab | — | Medium | High | Defer |

**Recommended first slice: 1, 2, 4, 5.** All are additive, none touch the core
library, and together they change the panel from "a launcher" into "the thing
that shows what VeloxQuant does" — which is #33's actual differentiator.

---

## 2. Method browser and presets (#35)

The registry already carries family, tier, blurb, config fields, deviation notes
and docs URL. Today the panel shows a dropdown. This surfaces what's already there.

**Scope**

- Sidebar navigation: **Server · Methods · Logs · About** (the panel is currently
  one long column; it will not absorb a third section without this).
- Method table: filter by family, search by name, badges for tier and `-adapted`.
- Detail view: blurb, deviation note, relevant knobs, link to docs site.
- Method-specific knobs driven by `config_fields` — showing `kivi_group_size` for
  KIVI and `svdq_rank` for SVDq instead of one generic bits box.
- 3 presets drawn from **measured** `BENCHMARK_RESULTS.md` values, not invented:
  `Balanced`, `Max compression`, `Best quality`.

**Honesty rules**

- Unsupported methods stay listed and disabled with reasons — never hidden.
- Presets cite their source numbers; no preset claims a figure we haven't measured.

**Work:** extend `/api/methods` with a `presets` block; add `MethodsView` to the
frontend. No backend risk — the registry is already probed and tested.

---

## 3. Measured memory panel (#36, partial)

The highest-value honesty upgrade available, and it needs no new library work.

Show three things side by side, each labelled with **how it was obtained**:

| Metric | Source | Label |
|---|---|---|
| Process RSS | `psutil` | *measured* |
| MLX active / peak | `mx.get_active_memory()` / `get_peak_memory()` | *measured* |
| Compressed vs fp16 counters | cache attributes, when present | *accounting estimate* |

**Why this beats a bare ratio.** Today the panel asserts "accounting-only" and
asks the user to trust it. This *shows* it: the estimate says 3.8×, measured
memory says otherwise, and the gap is visible rather than argued. It converts
#27's caveat into evidence.

**Handling Finding A.** Per method, render whichever applies:

- Full counters → keys + values ratio, marked estimate
- Key-only → key ratio **explicitly labelled "keys only"**, never presented as whole-cache
- No counters (eviction) → tokens kept vs seen, **not** a byte ratio
- Unknown → "this method does not report byte counters", not a blank or a zero

That last row is the one that keeps the panel honest across all 35 methods.

---

## 4. `/v1/kv/stats` and live telemetry (#36)

Depends on #3 landing first, since the presentation rules are what make the
numbers safe to show.

**Endpoint** (matches #27's proposal, so it stays useful outside the panel):

```json
{
  "method": "turboquant_rvq", "bits": 2, "layers": 16,
  "accounting_only": true,
  "coverage": "keys_only",
  "keys":   {"compressed_bytes": 3536, "fp16_bytes": 13312, "ratio": 3.76},
  "values": null,
  "tokens": {"seen": 512, "retained": 512},
  "memory": {"rss_bytes": 0, "mlx_active_bytes": 0, "mlx_peak_bytes": 0,
             "source": "measured"}
}
```

The `coverage` field is load-bearing: it tells any consumer which of Finding A's
three cases applies, so a client cannot mistake a key-only ratio for a whole-cache one.

**Implementation:** subclass mlx_lm's handler to add the route, holding a weak
reference to the live cache list. Poll on an interval while `running`; stop
cleanly when the process dies.

**Risk:** this is the first place the panel reads *live* cache state. Aggregation
across 16 layers must not allocate or force evaluation — telemetry that perturbs
generation is worse than no telemetry.

---

## 5. Built-in chat tester

Currently a user must leave the panel and write `curl` to see anything work. A
small chat box closes the loop inside the product, and — combined with §3 — lets
telemetry move *while you watch*, which is the actual demo.

Scope: message list, input, streaming via SSE, token count and tok/s from the
response. Deliberately not a chat app: no history, no system prompt editor, no
markdown rendering.

---

## 6. Model picker from local cache (cheap win)

`huggingface_hub.scan_cache_dir()` enumerates already-downloaded models. Populate
the datalist from it, showing size on disk, and mark ids that are cached.

Stays inside #33's non-goal ("no download manager") — it only surfaces what's
already on disk. Roughly an afternoon.

---

## 7. Deferred, with reasons

**Multi-server management.** Running two methods side by side to compare is a
genuinely good demo, but it multiplies lifecycle states, port allocation, and
memory pressure on a machine already holding model weights. Revisit after
telemetry, when there's something worth comparing.

**Native macOS shell.** Still the right long-term answer, and the plan doc records
why. Blocked on the code-signing decision, not on engineering — an unsigned app is
a *worse* first-run story than `pip install`. The web UI's control API is the
boundary a SwiftUI shell would talk to, so nothing here is wasted.

**Benchmark tab.** Runs take minutes and would need a job queue, cancellation, and
progress reporting inside the panel. Large surface for a feature the CLI already
covers well.

---

## 8. Sequencing

```
Slice 1 (additive, no core changes)
  §6 model picker  →  §2 method browser  →  §3 measured memory
                                                  ↓
Slice 2 (needs a server-side endpoint)
  §4 /v1/kv/stats  →  live telemetry  →  §5 chat tester
                                                  ↓
Slice 3 (product decisions, not engineering)
  §7 native shell (needs signing decision) · multi-server · benchmarks
```

Slice 1 is safe to ship incrementally. Slice 2 should land as one piece — a stats
endpoint without the presentation rules from §3 is precisely the "confident,
wrong numbers" risk #27 exists to prevent.

---

## 9. Rules any of this must not break

Carried forward from #37; each already has a test:

1. No silent fp16 fallback.
2. Byte figures labelled by mode; RSS shown only when measured.
3. Method lists from the registry probe, never a hardcoded count.
4. Endpoints advertised only if actually served.
5. `running` requires the handshake, never a timer.
6. Config locked while a server owns it.

New rules this plan adds:

7. **Every number states its provenance** — *measured* or *estimate* — in the UI,
   not just in docs.
8. **Absent telemetry says so.** A method without counters shows "not reported",
   never `0`, `—`, or a blank cell that reads as zero.
9. **Key-only ratios are labelled key-only** and never presented as whole-cache.
