"""Real-model validation for A2ATS-adapted.

``benchmark_a2ats.py`` is synthetic: it never loads a language model. This
script runs the cache end-to-end through an actual mlx_lm model and measures,
token by token in generation mode (every next-token prediction is scored
against the compressed cache):

  1. Perplexity vs the fp16 cache, for several ``a2ats_window`` values.
  2. A diagnostic arm with a window larger than the context, which isolates the
     VQ error from the windowed-RoPE error.
  3. A random-codebook arm, the documented "no calibration" footgun.
  4. Bytes actually held by the cache vs the theoretical ``compression_ratio``.
  5. Decode throughput relative to the fp16 cache.
  6. A RoPE round-trip check: de-rotate with ``a2ats_rope_base`` and re-rotate
     with the model's own RoPE, to see whether the cache's RoPE assumption
     matches the model.

The codebook is calibrated per layer on real pre-RoPE keys and values from text
that is different from the evaluation text.

Usage:
    python benchmark_scripts/benchmark_a2ats_real_model.py
    python benchmark_scripts/benchmark_a2ats_real_model.py --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --tokens 1024

Prints tables; saves a JSON summary to figures/a2ats/real_model_results.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np
from mlx_lm.models.cache import KVCache

from veloxquant_mlx.allocators.vecinfer import train_codebook
from veloxquant_mlx.cache.a2ats_cache import A2ATSKVCache
from veloxquant_mlx.cache.base import KVCacheConfig
from veloxquant_mlx.quantizers.a2ats_rope import rope_freqs_from_scaling, rope_remap_positions

_repo_root = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def calib_texts(tok, n_tokens: int = 6144, seg: int = 512):
    """Calibration corpus: repo documentation prose, ~6k tokens in 512-token
    segments. Different content and style from EVAL (printing, longitude,
    birds, light, maps), so calibration is not fitted to the evaluation text.
    """
    import re

    parts = []
    for f in sorted((_repo_root / "docs-site" / "docs" / "algorithms").glob("*.md")):
        if f.name == "a2ats.md":
            continue
        t = re.sub(r"```.*?```", " ", f.read_text(), flags=re.S)
        t = re.sub(r"[#*`|>\[\]()_:-]+", " ", t)
        parts.append(re.sub(r"\s+", " ", t))
    ids = tok.encode(" ".join(parts))[:n_tokens]
    return [tok.decode(ids[i : i + seg]) for i in range(0, len(ids), seg)]


EVAL = (
    "The invention of movable type in Europe is generally credited to Johannes "
    "Gutenberg, a goldsmith working in Mainz during the middle of the fifteenth "
    "century. His press combined several existing technologies in a novel way: "
    "the screw mechanism borrowed from wine and olive presses, an oil-based ink "
    "that adhered to metal rather than running off it, and above all a hand mould "
    "that allowed individual letters to be cast quickly and to a consistent "
    "height. Earlier printing in East Asia had used carved blocks and, later, "
    "movable characters of clay and metal, but the enormous character sets of "
    "written Chinese limited the advantage such systems offered over careful "
    "copying. The alphabet changed that calculation entirely. A printer working "
    "in a European vernacular needed only a few hundred distinct sorts to "
    "compose any text whatsoever, and those sorts could be redistributed and "
    "reused indefinitely. Within fifty years presses had been established in "
    "more than two hundred cities, and the price of a book had fallen to a small "
    "fraction of what a manuscript copy had cost. Standardised texts made it "
    "possible for scholars in distant places to refer to precisely the same "
    "passage, which mattered a great deal for astronomical tables, anatomical "
    "drawings and legal codes where a copyist's slip could propagate silently for "
    "generations. Printers became publishers, choosing what to issue and in what "
    "form, and the resulting competition pushed them toward vernacular languages "
    "and shorter, cheaper formats aimed at readers who were not clergy or lawyers. "
    "Longitude at sea resisted solution for far longer than latitude, which any "
    "navigator could fix by measuring the sun at noon or the pole star at night. "
    "Longitude required knowing the time at a reference meridian at the same "
    "instant as local time, and no pendulum clock could keep that reference "
    "aboard a rolling ship through changes of temperature and humidity. The "
    "astronomical alternative, the method of lunar distances, demanded tables of "
    "the moon's position accurate to a degree no observatory had yet reached, and "
    "a set of calculations that took a trained officer four hours to complete. "
    "John Harrison spent decades building sea clocks against this problem, moving "
    "from large machines with interlocking grasshopper escapements to a watch "
    "barely five inches across, and the board charged with judging his work kept "
    "raising what would count as proof. "
    "Bird migration was for centuries explained by theories that now read as "
    "fantasy, including the belief that swallows spent the winter in the mud at "
    "the bottom of ponds, an idea repeated by serious naturalists into the "
    "eighteenth century. The evidence that settled the question arrived piecemeal: "
    "a stork shot in Germany carrying a central African spear through its neck, "
    "then systematic ringing programmes that recovered numbered bands from birds "
    "thousands of miles from where they were fitted. Individual birds return not "
    "just to the same country but often to the same hedgerow, navigating by a "
    "combination of the sun's arc, the pattern of polarised light at dusk, star "
    "rotation around the celestial pole, and a magnetic sense whose receptor is "
    "still argued over. "
    "The problem of measuring the speed of light was thought hopeless by "
    "observers who assumed propagation was instantaneous. Galileo proposed "
    "stationing two people on hilltops with shuttered lanterns and found, "
    "unsurprisingly, only the delay of human reaction. The first real value came "
    "not from a terrestrial experiment but from watching the moons of Jupiter, "
    "whose eclipses ran early when Earth approached and late when it receded, a "
    "discrepancy Ole Romer interpreted as the time light took to cross the "
    "diameter of Earth's orbit. Later terrestrial measurements using toothed "
    "wheels and rotating mirrors closed the gap, and the constant eventually "
    "became so well determined that the metre was redefined in terms of it. "
    "Cartographic projection forces an unavoidable compromise, since no flat "
    "sheet can represent a sphere without distorting area, angle, or distance "
    "somewhere. The Mercator projection preserves angles, which is exactly what a "
    "navigator wants because a straight line drawn on the chart is a course of "
    "constant compass bearing, but it inflates land near the poles without limit. "
    "Equal-area projections correct the sizes and pay for it by shearing shapes "
    "into forms that look wrong to eyes trained on the familiar arrangement. "
) * 2


class _Recorder(KVCache):
    """Plain cache that records pre-RoPE keys (de-rotated) and values."""

    def __init__(self, rope_base: float, freqs=None):
        super().__init__()
        self._base = rope_base
        self._freqs = freqs
        self.keys_pre, self.vals = [], []

    def update_and_fetch(self, keys, values):
        B, H, S, D = keys.shape
        pos = mx.arange(self.offset, self.offset + S)
        zero = mx.zeros((S,), dtype=mx.float32)
        for h in range(H):
            self.keys_pre.append(
                rope_remap_positions(keys[0, h], pos, zero, base=self._base, freqs=self._freqs)
            )
        self.vals.append(values[0].reshape(-1, D))
        return super().update_and_fetch(keys, values)


def calibrate(model, tok, rope_base: float, freqs, sub_dim: int, bits: int, seed: int = 42):
    calib = calib_texts(tok)
    """Per-layer codebook trained on pooled pre-RoPE keys + values."""
    inner = getattr(model, "model", model)
    recs = [_Recorder(rope_base, freqs) for _ in inner.layers]
    for text in calib:
        for r in recs:
            r.keys, r.values, r.offset = None, None, 0
        model(mx.array([tok.encode(text)]), cache=recs)
    books = []
    for r in recs:
        ks = mx.concatenate([k.astype(mx.float32) for k in r.keys_pre], axis=0)
        vs = mx.concatenate([v.astype(mx.float32) for v in r.vals], axis=0)
        x = mx.concatenate([ks, vs], axis=0).reshape(-1, sub_dim)
        books.append(train_codebook(x, n_centroids=2**bits, seed=seed))
        mx.eval(books[-1])
    return books


def rope_roundtrip_error(model, tok, rope_base: float, freqs) -> float:
    """Relative error between the model's own RoPE applied to the cache's
    de-rotated keys and the post-RoPE keys the model actually produced.

    ~0 means the cache's plain-RoPE assumption matches the model; a large value
    means de-rotation does not invert the model's RoPE (scaled or different
    convention), so even "exact" window tokens are not bit-identical.
    """
    inner = getattr(model, "model", model)
    rec = _Recorder(rope_base, freqs)
    caches = [KVCache() for _ in inner.layers]
    caches[0] = rec
    model(mx.array([tok.encode(calib_texts(tok, 512)[0])]), cache=caches)
    post = rec.keys[0, :, : rec.offset, :].astype(mx.float32)  # [H, S, D]
    pre = mx.stack(rec.keys_pre[: post.shape[0]], axis=0)  # [H, S, D]
    rope = inner.layers[0].self_attn.rope
    again = rope(pre.astype(rec.keys.dtype)[None], offset=0)[0].astype(mx.float32)
    return float(mx.max(mx.abs(again - post)) / mx.max(mx.abs(post)))


def perplexity(model, ids, caches, label):
    total_nll, n = 0.0, 0
    t0 = time.perf_counter()
    for t in range(len(ids) - 1):
        logits = model(mx.array([[ids[t]]]), cache=caches)
        logp = nn.log_softmax(logits[0, -1].astype(mx.float32))
        total_nll += -float(np.array(logp[ids[t + 1]]))
        n += 1
    dt = time.perf_counter() - t0
    ppl = float(np.exp(total_nll / n))
    print(f"  {label:44s} ppl {ppl:8.3f}   {n / dt:6.1f} tok/s")
    return ppl, n / dt


def held_bytes(caches):
    """Bytes the cache actually holds in memory (K + V buffers)."""
    total = 0
    for c in caches:
        if c.keys is not None:
            total += c.keys[..., : c.offset, :].nbytes + c.values[..., : c.offset, :].nbytes
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument(
        "--configs",
        nargs="*",
        default=["8x8", "4x8", "2x8"],
        help="SUB_DIMxBITS, e.g. 8x8 = 1 bit/number",
    )
    ap.add_argument(
        "--no-rope-scaling",
        dest="rope_scaling",
        action="store_false",
        help="ignore the model's rope_scaling (reproduces the pre-fix behaviour on Llama 3.x)",
    )
    ap.add_argument("--windows", type=int, nargs="*", default=[128, 16])
    args = ap.parse_args()

    model, tok = mlx_lm.load(args.model)
    inner = getattr(model, "model", model)
    n_layers = len(inner.layers)
    head_dim = (
        getattr(model.args, "head_dim", None)
        or model.args.hidden_size // model.args.num_attention_heads
    )
    rope_base = float(getattr(model.args, "rope_theta", 10000.0))
    rope_scaling = getattr(model.args, "rope_scaling", None)
    freqs = (
        rope_freqs_from_scaling(head_dim, rope_base, rope_scaling) if args.rope_scaling else None
    )
    print(
        f"\n=== {args.model} | layers={n_layers} head_dim={head_dim} "
        f"rope_theta={rope_base:g} rope_scaling={rope_scaling} ==="
    )

    ids = tok.encode(EVAL)[: args.tokens]
    print(f"eval tokens: {len(ids)}")

    rt = rope_roundtrip_error(model, tok, rope_base, freqs)
    print(f"RoPE round-trip relative error (model rope vs cache assumption): {rt:.2e}")

    def a2ats(window, books_, sub_dim, bits):
        out = []
        for i in range(n_layers):
            cfg = KVCacheConfig(
                method="a2ats",
                head_dim=head_dim,
                a2ats_window=window,
                a2ats_sub_dim=sub_dim,
                a2ats_codebook_bits=bits,
                a2ats_codebook=books_[i] if books_ is not None else None,
                a2ats_rope_base=rope_base,
                a2ats_rope_scaling=rope_scaling if args.rope_scaling else None,
            )
            out.append(A2ATSKVCache(cfg))
        return out

    results = {
        "model": args.model,
        "tokens": len(ids),
        "rope_theta": rope_base,
        "rope_roundtrip_rel_err": rt,
        "arms": {},
    }
    fp16 = [KVCache() for _ in range(n_layers)]
    base, base_tps = perplexity(model, ids, fp16, "fp16 cache (baseline)")
    base_bytes = held_bytes(fp16)
    results["arms"]["fp16"] = {"ppl": base, "tok_s": base_tps, "held_bytes": base_bytes}

    def run(label, window, books_, sub_dim, bits):
        caches = a2ats(window, books_, sub_dim, bits)
        ppl, tps = perplexity(model, ids, caches, label)
        hb = held_bytes(caches)
        ratio = float(np.mean([c.compression_ratio for c in caches]))
        results["arms"][label] = {
            "sub_dim": sub_dim,
            "bits": bits,
            "window": window,
            "bits_per_number": sub_dim and bits / sub_dim,
            "ppl": ppl,
            "ppl_delta_pct": 100 * (ppl - base) / base,
            "ppl_ratio": ppl / base,
            "tok_s": tps,
            "tok_s_vs_fp16": tps / base_tps,
            "held_bytes": hb,
            "held_vs_fp16": hb / base_bytes,
            "reported_compression_ratio": ratio,
            "codebook_kb": float(np.mean([c.codebook_bytes for c in caches])) / 1024,
        }
        print(
            f"      ppl x{ppl / base:6.2f} vs fp16 | held {hb / 1e6:7.2f} MB "
            f"(fp16 {base_bytes / 1e6:.2f} MB) | reported ratio {ratio:.1f}x | "
            f"speed {tps / base_tps:.2f}x"
        )

    for ci, cfg in enumerate(args.configs):
        sub_dim, bits = (int(v) for v in cfg.split("x"))
        t0 = time.time()
        books = calibrate(model, tok, rope_base, freqs, sub_dim, bits)
        print(
            f"\n-- sub_dim={sub_dim} bits={bits} ({bits / sub_dim:g} bit/number): calibrated in {time.time() - t0:.0f}s"
        )
        wins = args.windows if ci == 0 else args.windows[:1]
        for w in wins:
            run(f"A2ATS {cfg} window={w}", w, books, sub_dim, bits)
        run(f"A2ATS {cfg} VQ-only (window>=ctx)", len(ids) + 64, books, sub_dim, bits)
        if ci == 0:
            run(f"A2ATS {cfg} RANDOM codebook window=128", 128, None, sub_dim, bits)

    out = _repo_root / "figures" / "a2ats" / "real_model_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except json.JSONDecodeError:
            existing = []
    existing = [e for e in existing if e.get("model") != args.model] + [results]
    out.write_text(json.dumps(existing, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
