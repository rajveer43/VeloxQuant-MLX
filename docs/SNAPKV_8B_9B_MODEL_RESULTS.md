# SnapKV 8–9B model validation

Each model: two warmups and five measured trials per backend/scenario, alternating backend order. Plain cache plus SnapKV reference, MLX and Metal. Fresh caches; 16 greedy decode steps. Manual loop includes scalar token reads. This is backend parity and timing validation on one repetitive prompt, not a quality evaluation.

| Model / budget / chunk | Backend | Prefill median [min–max] ms | Decode median ms/step |
|---|---|---:|---:|
| GLM-4-9B-0414-4bit / 512 / 192 | plain | 1330.0 [1283.8–1338.5] | 65.28 |
| GLM-4-9B-0414-4bit / 512 / 192 | reference | 1319.1 [1301.5–1342.5] | 65.52 |
| GLM-4-9B-0414-4bit / 512 / 192 | mlx | 1323.2 [1296.4–1347.7] | 65.77 |
| GLM-4-9B-0414-4bit / 512 / 192 | metal | 1316.2 [1313.2–1345.9] | 65.67 |
| GLM-4-9B-0414-4bit / 64 / 192 | plain | 1761.2 [1668.7–2036.0] | 65.92 |
| GLM-4-9B-0414-4bit / 64 / 192 | reference | 1763.2 [1550.6–2100.5] | 66.14 |
| GLM-4-9B-0414-4bit / 64 / 192 | mlx | 1812.8 [1549.9–2028.5] | 66.33 |
| GLM-4-9B-0414-4bit / 64 / 192 | metal | 1951.6 [1544.7–1980.7] | 65.98 |
| GLM-4-9B-0414-4bit / 64 / 48 | plain | 2599.4 [2553.3–2767.4] | 62.64 |
| GLM-4-9B-0414-4bit / 64 / 48 | reference | 2654.4 [2552.7–2719.4] | 62.55 |
| GLM-4-9B-0414-4bit / 64 / 48 | mlx | 2646.7 [2535.8–2915.8] | 62.35 |
| GLM-4-9B-0414-4bit / 64 / 48 | metal | 2586.2 [2497.5–2675.3] | 62.25 |

GLM-4-9B-0414-4bit: 84 successful runs; prompt 170 tokens. Revision `fecf35efe04b11a7e11ae434bf30693ac4c38e2c`. Raw data: [snapkv_glm9b_repeated.json](benchmarks/snapkv_glm9b_repeated.json).

| Model / budget / chunk | Backend | Prefill median [min–max] ms | Decode median ms/step |
|---|---|---:|---:|
| Qwen3-8B-4bit / 512 / 192 | plain | 1217.3 [1202.6–1312.6] | 51.26 |
| Qwen3-8B-4bit / 512 / 192 | reference | 1386.1 [1375.6–1570.1] | 78.26 |
| Qwen3-8B-4bit / 512 / 192 | mlx | 1386.5 [1374.5–1613.5] | 79.09 |
| Qwen3-8B-4bit / 512 / 192 | metal | 1385.3 [1377.3–1560.8] | 77.89 |
| Qwen3-8B-4bit / 64 / 192 | plain | 1750.2 [1500.1–1766.7] | 57.63 |
| Qwen3-8B-4bit / 64 / 192 | reference | 1984.8 [1736.5–2181.5] | 80.75 |
| Qwen3-8B-4bit / 64 / 192 | mlx | 2002.3 [1859.2–2365.5] | 80.35 |
| Qwen3-8B-4bit / 64 / 192 | metal | 1961.5 [1841.5–2263.8] | 80.24 |
| Qwen3-8B-4bit / 64 / 48 | plain | 2240.3 [2054.6–2489.0] | 54.42 |
| Qwen3-8B-4bit / 64 / 48 | reference | 2704.0 [2479.8–2992.8] | 76.09 |
| Qwen3-8B-4bit / 64 / 48 | mlx | 2785.3 [2564.3–2944.4] | 81.51 |
| Qwen3-8B-4bit / 64 / 48 | metal | 2780.5 [2575.4–3212.9] | 77.62 |

Qwen3-8B-4bit: 84 successful runs; prompt 170 tokens. Revision `545dc4251c05440727734bcd94334791f6ab0192`. Raw data: [snapkv_qwen8b_repeated.json](benchmarks/snapkv_qwen8b_repeated.json).

## Interpretation

Both models pass the tested backend parity checks. Metal timings are mixed and do not establish a consistent model-level speedup. Auto remains MLX. System load, thermal state and power mode were not controlled; the Qwen download overlapped early GLM inference. Five measured samples per group are insufficient for a useful tail-latency estimate. No peak-memory or broad task-quality claims follow from these tests.

Qwen shows a substantial whole-cache-path regression: plain-cache decode medians are 51–58 ms/step versus roughly 76–82 ms/step for SnapKV across backends, including the no-eviction case. This is not explained by the optional Metal selector, which does not run during singleton decode. Dtype conversions, attention dispatch, and cache bookkeeping require separate profiling; these measurements do not establish the cause.
