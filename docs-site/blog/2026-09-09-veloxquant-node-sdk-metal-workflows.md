---
slug: veloxquant-node-sdk-metal-workflows
title: Build Local AI Apps with the VeloxQuant Node.js SDK
date: 2026-09-09
authors: rajveer
tags: [javascript, typescript, nodejs, apple-silicon, mlx, metal, llm]
description: The VeloxQuant Node.js SDK now makes hardware-aware local AI and accelerated tensor workflows easier to use from JavaScript and TypeScript.
---

Local AI development should feel familiar to web developers: install a package, choose a model, and start building. The VeloxQuant Node.js SDK makes it easier to create private, hardware-aware AI applications with JavaScript and TypeScript on Apple Silicon.

{/* truncate */}

## Install the SDK

Install the published package from npm:

```bash
npm install @veloxquant/sdk
```

You can find the package, release history, and usage information here:

[View `@veloxquant/sdk` on npm](https://www.npmjs.com/package/@veloxquant/sdk)

The SDK is designed for Apple Silicon Macs and works with local models supported by [VeloxQuant-MLX](https://github.com/rajveer43/VeloxQuant-MLX).

## What you can build today

The SDK brings hardware-aware local inference and KV-cache optimization into familiar JavaScript APIs.

### Hardware-aware recommendations

Get a recommendation based on the model size, context length, device, and available memory:

```ts
import { VeloxQuant } from "@veloxquant/sdk";

const vq = new VeloxQuant();

const recommendation = await vq.recommendModel({
  modelClass: "7B",
  goal: "max_context",
  seqLen: 32768,
});

console.log(recommendation.recommendation.method);
console.log(recommendation.recommendation.rationale);
```

### Memory estimation

Estimate KV-cache requirements before loading a model:

```ts
const estimate = await vq.memory.estimate({
  seqLen: 32768,
  headDim: 128,
  nLayers: 32,
});

console.log(estimate.recommendedMethod);
console.log(estimate.memorySavedBytes);
```

### Local model serving

Start a local model and use the OpenAI-compatible chat interface:

```ts
const model = await vq.load({
  model: "mlx-community/Qwen3-4B-4bit",
  optimize: true,
});

const response = await model.chat({
  prompt: "Explain KV-cache compression in simple terms.",
  maxTokens: 200,
});

console.log(response.message.content);
await model.stop();
```

Streaming is also supported:

```ts
for await (const chunk of model.stream({
  prompt: "Write a short welcome message.",
})) {
  process.stdout.write(chunk.text);
}
```

### Method discovery

Inspect the available compression methods and choose the right family for your application:

```ts
const methods = await vq.models.list({ servableOnly: true });

for (const method of methods.methods) {
  console.log(method.name, method.family, method.serveTierLabel);
}
```

### Benchmarks

Measure real performance on your machine:

```ts
const result = await vq.benchmark({
  model: "mlx-community/Qwen3-4B-4bit",
});

console.log(result.tokensPerSecond);
console.log(result.timeToFirstTokenMs);
console.log(result.toMarkdown());
```

## Framework integrations

The SDK includes integrations for popular JavaScript AI frameworks:

- [Vercel AI SDK integration](https://github.com/rajveer43/veloxquant-sdk#vercel-ai-sdk)
- [LangChain.js integration](https://github.com/rajveer43/veloxquant-sdk#langchainjs)
- [LlamaIndex.TS integration](https://github.com/rajveer43/veloxquant-sdk#llamaindexts)
- [MCP and tool-using agents](https://github.com/rajveer43/veloxquant-sdk#agents-and-mcp)

Example with the Vercel AI SDK:

```ts
import { generateText } from "ai";
import { veloxquant } from "@veloxquant/sdk/ai-sdk";

const result = await generateText({
  model: veloxquant(model),
  prompt: "Give me three ideas for a local AI app.",
});

console.log(result.text);
```

## Command-line tools

The package also includes a CLI:

```bash
npx veloxquant doctor
npx veloxquant recommend
npx veloxquant analyze --seq-len 32768 --head-dim 128 --n-layers 32
npx veloxquant serve --model mlx-community/Qwen3-4B-4bit
```

## Explore the documentation

- [Install VeloxQuant-MLX](https://veloxquant-mlx.netlify.app/docs/getting-started/installation)
- [VeloxQuant quickstart](https://veloxquant-mlx.netlify.app/docs/getting-started/quickstart)
- [Architecture guide](https://veloxquant-mlx.netlify.app/docs/getting-started/architecture)
- [Algorithm overview](https://veloxquant-mlx.netlify.app/docs/algorithms/overview)
- [VeloxQuant SDK on npm](https://www.npmjs.com/package/@veloxquant/sdk)
- [VeloxQuant SDK source code](https://github.com/rajveer43/veloxquant-sdk)

Whether you are building a desktop assistant, a private coding tool, a long-context research application, or an on-device agent, the goal is the same: make local AI development on Apple Silicon approachable from the tools you already use.
