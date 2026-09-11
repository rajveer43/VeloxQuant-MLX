"""Short parity probe. Usage: python scripts/pyramidkv_model_probe.py MODEL OUTPUT.json."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import time

import mlx.core as mx
from mlx_lm import load

import veloxquant_mlx.metal._pyramidkv_evict as kernel
from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig

original = kernel.pyramidkv_fused_evict
calls = 0


def counted(*args, **kwargs):
    global calls
    calls += 1
    return original(*args, **kwargs)


kernel.pyramidkv_fused_evict = counted
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model")
parser.add_argument("output", type=Path)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--warmups", type=int, default=1)
parser.add_argument("--budget", type=int, default=64)
parser.add_argument("--prompt-tokens", type=int, default=128)
parser.add_argument("--decode-steps", type=int, default=16)
parser.add_argument("--chunk", type=int, default=128)
args = parser.parse_args()
if (
    min(args.repeats, args.budget, args.prompt_tokens, args.decode_steps, args.chunk) < 1
    or args.warmups < 0
):
    parser.error("positive counts required; warmups must be nonnegative")
model, tokenizer = load(args.model)
ids = tokenizer.encode(
    "Explain why the sky appears blue. Compare scattering at sunrise and noon. "
    * args.prompt_tokens
)[: args.prompt_tokens]
results = []
references = {}
for trial in range(args.warmups + args.repeats):
    for backend in (
        ("reference", "mlx", "metal") if trial % 2 == 0 else ("metal", "mlx", "reference")
    ):
        calls = 0
        caches = KVCacheBuilder.for_model(
            model,
            KVCacheConfig(method="pyramidkv", pyramid_budget=args.budget, pyramid_backend=backend),
        )
        start = time.perf_counter()
        for offset in range(0, len(ids), args.chunk):
            logits = model(mx.array(ids[offset : offset + args.chunk])[None], cache=caches)
            mx.eval(logits)
        prefill = (time.perf_counter() - start) * 1000
        first = logits[:, -1, :].astype(mx.float32)
        mx.eval(first)
        tokens = []
        start = time.perf_counter()
        for _ in range(args.decode_steps):
            token = mx.argmax(logits[:, -1, :], axis=-1)
            tokens.append(token.item())
            logits = model(token[:, None], cache=caches)
            mx.eval(logits)
        row = dict(
            backend=backend,
            trial=trial,
            warmup=trial < args.warmups,
            prefill_ms=prefill,
            decode_ms=(time.perf_counter() - start) * 1000 / args.decode_steps,
            kernel_calls=calls,
            tokens=tokens,
        )
        if backend == "reference":
            references["first"] = first
            references["tokens"] = tokens
        row["logit_error"] = mx.max(mx.abs(first - references["first"])).item()
        row["tokens_match"] = tokens == references["tokens"]
        results.append(row)
        print(json.dumps(row), flush=True)
args.output.write_text(
    json.dumps(
        dict(
            model=args.model,
            settings=vars(args) | {"output": str(args.output)},
            prompt_tokens=len(ids),
            results=results,
        ),
        indent=2,
    )
)
if any(row["logit_error"] != 0 or not row["tokens_match"] for row in results):
    raise SystemExit("Backend parity failed; see output JSON")
