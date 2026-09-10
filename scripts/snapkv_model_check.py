"""Offline real-model SnapKV backend check; reports failures without hiding them."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache

from veloxquant_mlx.cache.base import KVCacheBuilder, KVCacheConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmups", type=int, default=0)
    args = p.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        p.error("repeats must be positive and warmups nonnegative")
    model, tokenizer = load(args.model)
    ids = tokenizer.encode("Explain why the sky appears blue. " * 24)[:192]
    results = []
    saved = {}
    for budget, chunk in ((512, 192), (64, 192), (64, 48)):
        for trial in range(args.warmups + args.repeats):
            for backend in (
                ("plain", "reference", "mlx", "metal")
                if trial % 2 == 0
                else ("metal", "mlx", "reference", "plain")
            ):
                record = dict(
                    budget=budget,
                    chunk=chunk,
                    backend=backend,
                    trial=trial,
                    warmup=trial < args.warmups,
                )
                try:
                    caches = (
                        [KVCache() for _ in model.layers]
                        if backend == "plain"
                        else KVCacheBuilder.for_model(
                            model,
                            KVCacheConfig(
                                method="snapkv", snap_budget=budget, snap_backend=backend
                            ),
                        )
                    )
                    start = time.perf_counter()
                    for offset in range(0, len(ids), chunk):
                        logits = model(mx.array(ids[offset : offset + chunk])[None], cache=caches)
                        mx.eval(logits)
                    record["prefill_ms"] = (time.perf_counter() - start) * 1000
                    first = logits[:, -1, :].astype(mx.float32)
                    mx.eval(first)
                    key = (budget, chunk)
                    if backend == "reference":
                        saved[key] = first
                    if backend in ("mlx", "metal") and key in saved:
                        record["reference_max_logit_error"] = mx.max(
                            mx.abs(first - saved[key])
                        ).item()
                    tokens = []
                    start = time.perf_counter()
                    for _ in range(16):
                        token = mx.argmax(logits[:, -1, :], axis=-1)
                        mx.eval(token)
                        tokens.append(token.item())
                        logits = model(token[:, None], cache=caches)
                        mx.eval(logits)
                    record["decode_ms_per_step"] = (time.perf_counter() - start) * 1000 / 16
                    record["tokens"] = tokens
                    record["text"] = tokenizer.decode(tokens)
                    record["offset"] = caches[0].offset
                    record["status"] = "passed"
                except Exception as e:
                    record["status"] = "failed"
                    record["error"] = f"{type(e).__name__}: {e}"
                results.append(record)
                print(record, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                model=args.model,
                prompt_tokens=len(ids),
                note="Alternating backend order; fresh caches; manual greedy loop includes token scalar reads.",
                repeats=args.repeats,
                warmups=args.warmups,
                results=results,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
