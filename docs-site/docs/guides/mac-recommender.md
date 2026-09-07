---
id: mac-recommender
title: Mac Method Recommender
sidebar_label: Mac Recommender
slug: /guides/mac-recommender
description: Explains the veloxquant recommend CLI, which picks a compression method based on your Mac's chip, unified RAM, model size class, and goal, and clarifies key accounting versus resident memory savings.
keywords: [mac recommender, veloxquant recommend, chip ram heuristics, model size class, key accounting]
---

# Mac chip + RAM method recommender

Pick a VeloxQuant-MLX method from your Apple Silicon chip, unified RAM, model
size class, and goal. The CLI is the source of truth; the table below mirrors
the same heuristics.

## Honesty first

- **Key accounting** ratios (7.5× RVQ, 16× VecInfer) are packed-byte counters.
- Default RVQ / VecInfer often **dequantize into fp16** parent storage.
- **Resident** savings are more likely for full-KV (`rabitq`) or eviction
  (`streaming_llm` / `h2o`), especially at long context.

## CLI

```bash
veloxquant recommend \
  --chip M4 \
  --ram-gb 48 \
  --model-class 7B \
  --goal everyday

# Machine-readable JSON
veloxquant recommend --chip M1 --ram-gb 16 --model-class 3B --goal max_context --json

# Static ruleset (for automation / docs sync)
veloxquant recommend --dump-ruleset > docs-site/static/mac-recommender-ruleset.json
```

## Goal → method map

| Goal | Method | Typical key accounting | Resident savings likely? |
| --- | --- | ---: | --- |
| `everyday` | `turboquant_rvq` b=1 | ~7.5× | No (default dequant path) |
| `max_key_accounting` | `vecinfer` 1-bit | ~16× | No (default dequant path) |
| `best_quality` | `spectral` | ~5.3× | No |
| `max_context` | `rabitq` | ~6× full KV | Yes (more likely) |
| `constant_memory` | `streaming_llm` | n/a (eviction) | Yes (bounded tokens) |

## RAM fit warnings (rule of thumb)

Approximate 4-bit weight footprints used by the recommender:

| Model class | ~Weights |
| --- | ---: |
| 1B | 0.8 GB |
| 3B | 2.0 GB |
| 7B | 4.5 GB |
| 14B | 8.0 GB |
| 32B | 18.0 GB |

The CLI also subtracts ~4 GB for OS / apps. If headroom is under ~1 GB it
warns. Example: 32B on 24 GB is expected to warn (see also
`docs/MEMORY_CONSTRAINT_FINDINGS.md`).

## Example (M4 Pro, 48 GB, 7B, everyday)

```text
method=turboquant_rvq
knobs={'bit_width_inlier': 1, 'seed': 42}
key_accounting_ratio≈7.5x
resident_savings_likely=False
```

## See also

- [Validation Report](./validation-report) for measuring accounting vs peak memory
- Python module: `veloxquant_mlx/tools/mac_recommender.py`
