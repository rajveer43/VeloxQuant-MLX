---
id: architecture
title: System Architecture
sidebar_label: Architecture
slug: /getting-started/architecture
description: End-to-end architecture of VeloxQuant-MLX, its MLX and Metal kernel layers, Python worker protocol, and JavaScript SDK integration.
keywords: [architecture, MLX, Metal, Python worker, TypeScript SDK, KV cache]
---

# System Architecture

VeloxQuant-MLX is a Python/MLX runtime for compressing transformer KV caches on Apple Silicon. The JavaScript SDK is an orchestration layer around that runtime. This page explains how the pieces fit together, where tensors live, how Metal kernels are dispatched, and which boundaries are intentionally kept stable.

## System overview

```text
Application
  ├── Python API: KVCacheBuilder / mlx_lm integration
  ├── TypeScript API: @veloxquant/sdk
  │       │
  │       └── JSON-lines worker + .npy tensor transport
  │                    │
  └────────────────────┘
                       ▼
              veloxquant_mlx worker
                       │
             MLX tensors and cache objects
                       │
                mx.fast.metal_kernel
                       │
                    Apple GPU
```

The Python package remains the source of truth for tensor execution, cache semantics, model compatibility, and Metal dispatch. The npm package does not reimplement MLX or compile arbitrary Metal source.

## Runtime layers

### 1. User and framework layer

Users can call the Python API directly, use the command line, launch the control panel, or use the npm SDK from Node.js.

Relevant entry points:

- [Quickstart](./quickstart)
- [Python API reference](../api/core-api)
- [CLI and installation](./installation)
- [Control panel guide](../guides/control-panel)
- [JavaScript SDK](https://github.com/rajveer43/veloxquant-sdk)

### 2. Model integration layer

The model integration layer connects VeloxQuant caches to `mlx_lm` model generation. `KVCacheBuilder` creates per-layer cache instances from a `KVCacheConfig`. The cache contract is designed to match the `mlx_lm` cache interface so model generation can continue to own tokenization, sampling, and scheduling.

Read more:

- [MLX-LM integration](../guides/mlx-lm-integration)
- [Cache API](../api/cache)
- [Core API](../api/core-api)
- [Concepts](./concepts)

### 3. Cache and algorithm layer

Every method is selected through a common configuration and registry path:

```text
KVCacheConfig(method="...")
          ▼
       Registry
          ▼
   Cache / quantizer handler
          ▼
   Encode → store → decode/attend
```

The methods fall into three broad families:

- Quantization: represent every token with fewer bits.
- Eviction: remove tokens that contribute less to future attention.
- Hybrid and cross-layer methods: combine compression, eviction, or layer sharing.

See the [algorithm overview](../algorithms/overview) and [method API reference](../api/quantizers).

### 4. MLX tensor layer

MLX owns the device arrays, lazy evaluation graph, dtype conversion, shape propagation, and synchronization. Kernel wrappers pass MLX arrays into custom operations and call `mx.eval()` at explicit synchronization points when results must be materialized.

A `.metal` file by itself is not a complete VeloxQuant operation. The Python wrapper supplies input names, output names, templates, grids, threadgroups, output shapes, output dtypes, and MLX graph integration.

### 5. Metal kernel layer

Custom kernels are compiled lazily through `mx.fast.metal_kernel`. Sources live under:

```text
veloxquant_mlx/metal/src/*.metal
```

Python dispatch wrappers live under:

```text
veloxquant_mlx/metal/*.py
```

The [Metal API reference](../api/metal-api) lists the supported kernel families. The [Metal kernel guide](../guides/metal-kernels) explains dispatch, lazy compilation, synchronization, and performance measurement.

## Kernel families

| Family | Python wrapper | Main use |
|---|---|---|
| Bit packing | `metal/_bit_packing.py` | Store low-bit indices compactly |
| VecInfer | `metal/_vecinfer.py` | Product VQ encode/decode |
| Scalar quantization | `metal/_scalar_quant.py` | Affine and scalar quantization |
| RaBitQ | `metal/_rabitq*.py` | Binary encoding and Hamming operations |
| QJL | `metal/_qjl.py` | Quantized inner products and encoding |
| RVQ | `metal/_rvq_*.py` | Fused residual VQ operations |
| Attention | `metal/_scalar_attend.py`, `metal/fused_sdpa.py` | Decode and prefill attention |
| Eviction | `metal/_h2o_evict.py`, `_tova_evict.py`, `_keyformer_evict.py`, `_qfilters_evict.py` | Token selection and removal |
| Cross-model transfer | `metal/_crosskv_rope.py` | Fused RoPE re-encoding |

Not every kernel is exposed through the npm worker. The Python API remains broader than the Node.js transport surface.

## Python worker architecture

The worker is an optional long-lived process started with:

```bash
python -m veloxquant_mlx worker
```

It communicates over newline-delimited JSON:

```json
{"protocol_version":1,"id":"request-1","op":"capabilities","args":{}}
```

Responses preserve the request ID:

```json
{
  "protocol_version": 1,
  "id": "request-1",
  "ok": true,
  "result": {
    "metalAvailable": true,
    "device": "Device(gpu, 0)"
  }
}
```

The worker currently exposes a reviewed operation set:

- `ping`
- `capabilities`
- `metal_probe`
- `bit_pack`
- `bit_pack_file`
- `rope_recode_file`

Unknown operations and invalid arguments return structured errors. Arbitrary Metal source is never accepted from the client.

## Tensor transport

Small control values may be sent directly in JSON. Large tensors use `.npy` files:

```text
Node writes input.npy
        ▼
Python loads input.npy into MLX
        ▼
Metal kernel executes
        ▼
Python writes output.npy
        ▼
Node consumes output.npy
```

The file transport is deliberately explicit and debuggable. It avoids inventing a native tensor ABI before performance measurements justify zero-copy memory. Future transport options include memory-mapped files and shared memory.

See the [npm worker architecture decision](https://github.com/rajveer43/veloxquant-sdk/pull/33) and the [npm worker implementation](https://github.com/rajveer43/veloxquant-sdk/pull/33/files).

## JavaScript SDK boundary

The npm SDK exposes the worker through typed methods:

```ts
import { startWorker } from "@veloxquant/sdk";

const worker = startWorker();
const capabilities = await worker.capabilities();
const packed = await worker.bitPack([0, 1, 2, 3, 0, 1, 2, 3], 2);
await worker.close();
```

The SDK is responsible for:

- Python interpreter resolution
- Worker startup and shutdown
- Request IDs and timeouts
- Response validation
- TypeScript result types
- Node-side errors and lifecycle behavior

The Python package is responsible for:

- MLX imports and device selection
- Tensor validation
- Kernel compilation and dispatch
- Cache and model integration
- Metal synchronization
- Numerical correctness

## Capability and fallback model

Applications should call `capabilities()` before using a Metal operation. A valid result identifies the backend and device. The SDK must not label a result as Metal if it used a fallback.

Supported environments:

- Apple Silicon macOS with MLX and Metal: Metal backend.
- macOS without a usable GPU: structured capability error.
- Linux, Windows, Intel macOS, or unsupported Python environments: SDK can remain installable, but Metal operations are unavailable unless a documented fallback exists.

The fallback hierarchy is:

1. Metal-backed MLX implementation.
2. MLX non-Metal implementation, where available.
3. NumPy/reference implementation, where correctness and performance are acceptable.
4. Explicit unsupported-operation error.

## Testing architecture

Correctness is checked at multiple boundaries:

1. Metal kernel unit tests validate shapes, dtypes, grid sizes, and edge cases.
2. Reference parity tests compare Metal output with pure MLX or NumPy output.
3. Worker protocol tests validate envelopes, IDs, errors, timeouts, and shutdown.
4. End-to-end tests validate Node → worker → MLX → Metal → output transport.
5. Model tests validate generation and KV-cache behavior.
6. Benchmarks measure warm-up, kernel latency, IPC, file transport, memory, and end-to-end throughput.

Relevant resources:

- [Metal tests](https://github.com/rajveer43/VeloxQuant-MLX/tree/codex/metal-worker-protocol/veloxquant_mlx/tests/metal)
- [Worker parity tests](https://github.com/rajveer43/VeloxQuant-MLX/blob/codex/metal-worker-protocol/veloxquant_mlx/tests/metal/test_worker_kernel_parity.py)
- [Benchmarking guide](../guides/benchmarking)
- [Validation report](../guides/validation-report)
- [Metal kernel research notes](../blog/turboquant-metal-kernels)

## Packaging and release boundaries

The base npm package does not bundle MLX, Metal, or a native Node addon. Users install the Python package separately. This keeps the npm package portable and avoids coupling Node ABI compatibility to MLX and macOS GPU distribution.

The current integration is implemented across two companion pull requests:

- [npm SDK PR #33](https://github.com/rajveer43/veloxquant-sdk/pull/33)
- [Python worker PR #336](https://github.com/rajveer43/VeloxQuant-MLX/pull/336)

A native Node-API or Swift bridge is a future research option, not the current execution path. It would require a stable tensor ABI, numerical parity, zero-copy measurements, Apple Silicon packaging, code signing, notarization, and fallback behavior before adoption.

## Design principles

- Keep MLX and Metal execution in Python where the mature runtime already exists.
- Expose reviewed high-level operations, not arbitrary GPU code execution.
- Treat compression and memory numbers honestly: accounting estimates are not automatically resident RSS savings.
- Require parity tests before exposing a kernel through the worker.
- Measure IPC and file transport before introducing native bindings.
- Keep unsupported hardware explicit and diagnosable.

## Related documentation

- [Getting started](./intro)
- [Core concepts](./concepts)
- [Algorithm overview](../algorithms/overview)
- [Metal kernel guide](../guides/metal-kernels)
- [Metal API reference](../api/metal-api)
- [Cache API](../api/cache)
- [Cross-model transfer](../algorithms/cross-model-transfer)
- [Profiling](../guides/profiling)
- [Benchmarking](../guides/benchmarking)
- [Installation](./installation)

