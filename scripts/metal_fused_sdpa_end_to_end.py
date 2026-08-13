"""End-to-end model validation for the Phase 2 fused VecInfer SDPA kernel.

Runs ``mlx_lm.generate`` three times on the same prompt and compares:

  Path A: fp16-baseline (no compression)
  Path B: VecInfer-1bit pure-MLX (current 0.5.1 behavior)
  Path C: VecInfer-1bit fused (new Phase 2)

Reports throughput, peak memory, tokens generated, and a response preview
for each.  Verifies Path C beats Path B end-to-end.

Run from repo root:

    PYTHONPATH=. python scripts/metal_fused_sdpa_end_to_end.py [--model HF_ID]

Defaults to a small fast-iterating model.  Pass --model to test a larger
one (e.g. ``mlx-community/Llama-3.1-8B-Instruct-4bit``).
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import mlx.core as mx

from veloxquant_mlx.metal import metal_available

DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
# Default prompt: short, fast iteration.  Override with --long-prompt to use
# a multi-KB context that exercises the fused kernel's sweet spot (S_kv > 2k).
PROMPT = (
    "Explain the theory of relativity in simple terms, covering both "
    "special and general relativity with concrete examples."
)
# Long-context prompt that pushes S_kv into the regime where the fused
# kernel actually beats MLX SDPA (S_kv > 2048).  About 4-5k tokens.
LONG_PROMPT = (
    "You are a physics professor preparing extensive lecture notes on "
    "Albert Einstein's contributions to physics.  Below is your draft "
    "material; please continue it with a detailed exposition of general "
    "relativity, including the field equations, geodesic motion, the "
    "Schwarzschild solution, gravitational time dilation, frame dragging, "
    "the cosmological constant, and the relationship between matter, "
    "curvature, and the stress-energy tensor.  Be thorough.\n\n"
    + (
        "The story of relativity begins in the late 19th century when "
        "Maxwell's equations predicted a constant speed for light that "
        "contradicted Newtonian mechanics.  Michelson and Morley sought "
        "to detect the luminiferous ether through interferometry but "
        "found no fringe shift, an outcome that puzzled physicists for "
        "decades.  Lorentz proposed length contraction; FitzGerald did "
        "similarly.  Poincare formulated principles of relativity that "
        "anticipated Einstein's eventual formulation. "
    )
    * 16
)
MAX_TOKENS = 120


def _peak_mb() -> float:
    try:
        return float(mx.get_peak_memory()) / (1024**2)
    except Exception:
        try:
            return float(mx.metal.get_peak_memory()) / (1024**2)
        except Exception:
            return float("nan")


def _reset_peak() -> None:
    try:
        mx.reset_peak_memory()
    except Exception:
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass


def _build_caches(
    model,
    *,
    method: str,  # "fp16" | "vecinfer-pure" | "vecinfer-fused"
    key_sub_dim: int = 8,
):
    """Return a list of caches, one per attention-bearing layer."""
    from mlx_lm.models.cache import KVCache as _FB
    from veloxquant_mlx import KVCacheConfig
    from veloxquant_mlx.cache.vecinfer_cache import VecInferKVCache

    layers = getattr(model, "layers", None) or model.model.layers
    args = getattr(model, "args", None)
    if args is not None and not hasattr(args, "hidden_size"):
        lm = getattr(model, "language_model", None)
        if lm is not None:
            args = getattr(lm, "args", args)

    caches = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn is None:
            caches.append(_FB())
            continue
        hd = getattr(attn, "head_dim", None)
        if hd is None and args is not None:
            hd = getattr(args, "head_dim", None) or (args.hidden_size // args.num_attention_heads)
        if hd is None or method == "fp16":
            caches.append(_FB())
            continue

        sub_dim = key_sub_dim if hd % key_sub_dim == 0 else 4
        cfg = KVCacheConfig(
            method="vecinfer",
            head_dim=hd,
            key_sub_dim=sub_dim,
            value_sub_dim=sub_dim,
            key_codebook_bits=8,
            value_codebook_bits=8,
            seed=42 + i,
            fused_sdpa=(method == "vecinfer-fused"),
        )
        caches.append(VecInferKVCache(cfg))
    return caches


def _run_one(model, tokenizer, *, label: str, method: str) -> dict:
    import mlx_lm

    caches = _build_caches(model, method=method)
    messages = [{"role": "user", "content": PROMPT}]
    try:
        prompt_txt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_txt = PROMPT

    _reset_peak()
    mx.clear_cache()

    t0 = time.perf_counter()
    try:
        response = mlx_lm.generate(
            model,
            tokenizer,
            prompt=prompt_txt,
            max_tokens=MAX_TOKENS,
            verbose=False,
            prompt_cache=caches,
        )
    except Exception as e:
        return {
            "label": label,
            "method": method,
            "error": str(e),
            "tput": 0.0,
            "peak_mb": float("nan"),
            "n_tok": 0,
            "elapsed": 0.0,
            "preview": "",
        }
    elapsed = time.perf_counter() - t0
    n_tok = len(tokenizer.encode(response)) if response else 0
    peak = _peak_mb()

    return {
        "label": label,
        "method": method,
        "tput": n_tok / max(elapsed, 1e-6),
        "peak_mb": peak,
        "n_tok": n_tok,
        "elapsed": elapsed,
        "preview": response[:140].replace("\n", " "),
        "error": None,
    }


def main() -> int:
    global MAX_TOKENS, PROMPT
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument(
        "--long-prompt",
        action="store_true",
        help="Use a ~4-5k-token prompt to push S_kv into the fused-kernel sweet spot.",
    )
    args = parser.parse_args()
    MAX_TOKENS = args.max_tokens
    if args.long_prompt:
        PROMPT = LONG_PROMPT

    if not metal_available():
        print("Metal unavailable — aborting.")
        return 1

    from mlx_lm import load
    from veloxquant_mlx.metal.fused_sdpa import (
        patch_mlx_lm_for_fused_sdpa,
        unpatch_mlx_lm,
    )

    print(f"Model: {args.model}")
    print(f"Loading...")
    model, tokenizer = load(args.model)

    runs = []
    # Path A — fp16
    print("\n[A] fp16-baseline ...")
    runs.append(_run_one(model, tokenizer, label="fp16", method="fp16"))

    # Path B — VecInfer pure (no patch)
    print("[B] VecInfer pure-MLX (no fused_sdpa) ...")
    runs.append(_run_one(model, tokenizer, label="VecInfer-pure", method="vecinfer-pure"))

    # Path C — VecInfer fused (patch dispatcher first)
    print("[C] VecInfer fused (Metal fused SDPA) ...")
    patch_mlx_lm_for_fused_sdpa()
    try:
        runs.append(_run_one(model, tokenizer, label="VecInfer-fused", method="vecinfer-fused"))
    finally:
        unpatch_mlx_lm()

    print("\n" + "=" * 78)
    print(f"  {'label':<18s}  {'tput tok/s':>11s}  {'peak MB':>9s}  {'n_tok':>6s}  preview")
    print(f"  {'-' * 18}  {'-' * 11}  {'-' * 9}  {'-' * 6}  {'-' * 40}")
    for r in runs:
        if r.get("error"):
            print(f"  {r['label']:<18s}  ERROR: {r['error']}")
            continue
        print(
            f"  {r['label']:<18s}  {r['tput']:>11.1f}  {r['peak_mb']:>9.0f}  "
            f"{r['n_tok']:>6d}  {r['preview'][:60]!r}"
        )

    # Verdict
    fp16 = next((r for r in runs if r["method"] == "fp16" and not r.get("error")), None)
    pure = next((r for r in runs if r["method"] == "vecinfer-pure" and not r.get("error")), None)
    fused = next((r for r in runs if r["method"] == "vecinfer-fused" and not r.get("error")), None)

    if fused and pure and fused["tput"] > pure["tput"]:
        print(
            f"\nSUCCESS: fused {fused['tput']:.1f} tok/s beats pure {pure['tput']:.1f} tok/s "
            f"({fused['tput'] / pure['tput']:.2f}x)."
        )
    elif fused and pure:
        print(
            f"\nNOTE: fused {fused['tput']:.1f} tok/s did not beat pure "
            f"{pure['tput']:.1f} tok/s on this shape."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
