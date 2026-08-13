---
id: control-panel
title: Control Panel
sidebar_label: Control Panel
slug: /guides/control-panel
---

# Control Panel

Run a local model on your Mac with a compressed memory footprint, using a
point-and-click web app — no Python required.

:::info Runs on your Mac, not in your browser alone
This app starts a real local server on your own machine, using your Mac's own
chip. There's no hosted version — install VeloxQuant-MLX first, then run the
command below.
:::

---

## Install

```bash
pip install veloxquant-mlx
```

## Start the panel

```bash
veloxquant panel
```

Your browser opens automatically at `http://127.0.0.1:7860`.

## Set it up

1. **Model** — paste a model name, e.g. `mlx-community/Llama-3.2-1B-Instruct-4bit`.
   (Don't have one yet? Any model from [mlx-community on Hugging Face](https://huggingface.co/mlx-community) works.)
2. **Compression method** — leave it on the default if you're not sure. It's a
   safe, broadly-compatible choice.
3. Click **Start Server**. The status pill goes `Starting` → `Running`.
4. Copy the **Base URL** shown on the page.

That's it — you now have a local, OpenAI-compatible endpoint running on your
Mac.

---

## Chat with it

Paste the Base URL into any tool that supports a custom OpenAI-compatible
endpoint — Cursor, a chat UI, or your own script.

With Python:

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

Or from the terminal:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default_model","messages":[{"role":"user","content":"Say hi"}],"max_tokens":50}'
```

When you're done, click **Stop Server** in the panel to shut it down and free
the port.

---

## What "compression" means here

VeloxQuant shrinks the memory a model's conversation history takes up (its
"KV cache"), so you can hold longer conversations or run bigger models on the
same Mac.

**One honest caveat:** the compression ratio shown in the panel today
describes how well the data compresses, not how much memory your Mac actually
saves while running — that part is still on our roadmap. So don't expect your
Mac's memory usage to drop by the ratio shown; think of it as a preview of
where we're headed, not a live memory saving yet.

---

## If something goes wrong

| What you see | What it means |
|---|---|
| "port already in use" | Something else is using that port — try a different one, or stop the other program. |
| "cannot be served" for a method | That method isn't supported by the server yet — pick a different one from the list. |
| "model not found" | Double check the model name, or make sure you're online the first time (it needs to download). |
| "ran out of memory loading the model" | The model is too big for your Mac's RAM — try a smaller one. |
| Status flips to `Error` while running | The server crashed. Scroll the log at the bottom of the page to see why. |

---

## Prefer the command line?

Everything the panel does, you can also run directly:

```bash
veloxquant serve \
  --model mlx-community/Llama-3.2-1B-Instruct-4bit \
  --host 127.0.0.1 --port 8000
```
