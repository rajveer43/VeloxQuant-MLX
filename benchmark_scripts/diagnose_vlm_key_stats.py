"""Diagnose per-layer key distribution on Qwen2-VL.

For each transformer layer we capture the keys produced by ``self_attn`` during
a single forward pass on a representative prompt and compute statistics
separately for the "image" portion (first N_IMG tokens of the prefill — the
patch tokens) and the "text" portion (the remaining tokens). We report:

  * mean / std of per-vector L2 norms
  * kurtosis of post-rotation coordinates (excess; Gaussian = 0, Laplacian = 3)
  * cosine similarity between original keys and RVQ-2bit reconstructions

Because Qwen2-VL in mlx_lm strips the visual encoder, we cannot run an actual
image through the model. Instead we synthesize a deterministic ``input_embeds``
tensor whose first N_IMG rows mimic the larger-norm patch-embedding statistics
(scale 5–20× the text token norm — matches what the ViT projection produces),
and whose remaining rows come from the language model's own embedding table at
some text token ids. That gives us a prefill key tensor that includes a real
"image-like" prefix flowing through the actual attention weights.

Outputs:
  * Per-layer table on stdout
  * figures/updated_tests/qwen2_vl/key_stats/layer_NN_norms.png
  * figures/updated_tests/qwen2_vl/key_stats/layer_NN_rot_hist.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ

DEFAULT_MODEL = "mlx-community/Qwen2-VL-7B-Instruct-bf16"
N_IMG = 256  # synthetic image patch count (typical ViT output for one image)
N_TEXT = 32  # text tokens following the image
IMG_NORM_SCALE = 12.0  # image-token norm multiplier vs text-token norm
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "figures",
    "updated_tests",
    "qwen2_vl",
    "key_stats",
)


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Gaussian = 0)."""
    x = x.astype(np.float64)
    mu = x.mean()
    var = x.var()
    if var < 1e-12:
        return 0.0
    return float(((x - mu) ** 4).mean() / (var**2) - 3.0)


def _to_numpy(arr: Any) -> np.ndarray:
    return np.array(arr.astype(mx.float32))


def _synth_inputs(model, n_img: int, n_text: int, seed: int = 0):
    """Build (input_embeddings, total_tokens) that drives a single forward pass.

    The first n_img rows are random vectors scaled to have norms ~IMG_NORM_SCALE×
    typical text-embedding norms. The remaining n_text rows are real embedding
    rows pulled from the model's embedding table.
    """
    layers = getattr(model, "layers", None) or model.model.layers
    # Find the embedding module — Qwen2-VL: model.language_model.model.embed_tokens
    inner = getattr(model, "language_model", model)
    inner_model = getattr(inner, "model", inner)
    embed = inner_model.embed_tokens
    hidden = embed.weight.shape[1]

    # Sample text-token embeddings from the table (deterministic ids).
    rng = np.random.default_rng(seed)
    vocab = embed.weight.shape[0]
    text_ids = rng.integers(0, vocab, size=(n_text,)).astype(np.int32)
    text_embeds = embed(mx.array(text_ids))  # (n_text, hidden)
    text_norms = _to_numpy(mx.linalg.norm(text_embeds.astype(mx.float32), axis=-1))
    text_norm_mean = float(text_norms.mean())

    # Synthesize image-like embeddings: random Gaussian, normalised, then scaled
    # to IMG_NORM_SCALE × text_norm_mean to mimic ViT output magnitudes.
    raw = rng.standard_normal((n_img, hidden)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
    img_target_norm = IMG_NORM_SCALE * text_norm_mean
    img_np = raw * img_target_norm

    embeds_np = np.concatenate([img_np, _to_numpy(text_embeds)], axis=0).astype(np.float32)
    embeds = mx.array(embeds_np).astype(embed.weight.dtype)
    embeds = embeds.reshape(1, n_img + n_text, hidden)

    print(f"  text_norm_mean = {text_norm_mean:.3f}")
    print(f"  img_target_norm = {img_target_norm:.3f}")
    return embeds, n_img + n_text


def _layer_stats(
    layer_idx: int,
    keys_img: mx.array,  # (H, n_img, D) for one layer
    keys_text: mx.array,  # (H, n_text, D)
    save_hists: bool,
    out_dir: str,
) -> dict:
    """Compute per-layer image vs text stats."""
    H, _, D = keys_img.shape

    def _norms(k: mx.array) -> np.ndarray:
        flat = k.reshape(-1, D).astype(mx.float32)
        return _to_numpy(mx.linalg.norm(flat, axis=-1))

    img_norms = _norms(keys_img)
    text_norms = _norms(keys_text)

    # Quantize with RVQ-2bit, measure cosine on the two segments separately.
    quant = TurboQuantRVQ(d=D, b=2, seed=42 + layer_idx, use_hadamard=True)

    def _q_cosine(k: mx.array) -> tuple[float, np.ndarray]:
        """Return (cosine, post-rotation coords as numpy array)."""
        flat = k.reshape(-1, D).astype(mx.float16)
        norms = mx.linalg.norm(flat.astype(mx.float32), axis=-1, keepdims=True)
        safe = mx.maximum(norms, mx.array(1e-4))
        unit = (flat.astype(mx.float32) / safe).astype(mx.float16)
        # Post-rotation coords (for kurtosis):
        rotated = quant._rotation.apply(unit)
        rot_np = _to_numpy(rotated).ravel()
        # RVQ encode/decode:
        ev = quant.encode(unit)
        unit_hat = quant.decode(ev)
        cos = float(
            mx.mean(
                mx.sum(unit * unit_hat, axis=-1)
                / (mx.linalg.norm(unit, axis=-1) * mx.linalg.norm(unit_hat, axis=-1) + 1e-8)
            )
        )
        return cos, rot_np

    cos_img, rot_img = _q_cosine(keys_img)
    cos_text, rot_text = _q_cosine(keys_text)
    kurt_img = _kurtosis(rot_img)
    kurt_text = _kurtosis(rot_text)

    if save_hists:
        os.makedirs(out_dir, exist_ok=True)
        # Norms histogram
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(img_norms, bins=50, alpha=0.6, label=f"image (n={len(img_norms)})", color="#C44E52")
        ax.hist(
            text_norms, bins=50, alpha=0.6, label=f"text (n={len(text_norms)})", color="#4C72B0"
        )
        ax.set_xlabel("L2 norm")
        ax.set_ylabel("count")
        ax.set_title(f"Layer {layer_idx:02d}  Key Norms")
        ax.legend()
        plt.tight_layout()
        fig.savefig(f"{out_dir}/layer_{layer_idx:02d}_norms.png", dpi=120)
        plt.close()

        # Post-rotation coord histogram
        fig, ax = plt.subplots(figsize=(7, 3.5))
        clip = np.percentile(np.concatenate([np.abs(rot_img), np.abs(rot_text)]), 99)
        ax.hist(
            np.clip(rot_img, -clip, clip),
            bins=80,
            alpha=0.55,
            label=f"image (kurt={kurt_img:+.2f})",
            color="#C44E52",
            density=True,
        )
        ax.hist(
            np.clip(rot_text, -clip, clip),
            bins=80,
            alpha=0.55,
            label=f"text (kurt={kurt_text:+.2f})",
            color="#4C72B0",
            density=True,
        )
        ax.set_xlabel("post-rotation coord")
        ax.set_ylabel("density")
        ax.set_title(f"Layer {layer_idx:02d}  Rotated-Key Distribution")
        ax.legend()
        plt.tight_layout()
        fig.savefig(f"{out_dir}/layer_{layer_idx:02d}_rot_hist.png", dpi=120)
        plt.close()

    return {
        "layer": layer_idx,
        "img_norm_mean": float(img_norms.mean()),
        "img_norm_std": float(img_norms.std()),
        "text_norm_mean": float(text_norms.mean()),
        "text_norm_std": float(text_norms.std()),
        "img_kurt": kurt_img,
        "text_kurt": kurt_text,
        "img_cos": cos_img,
        "text_cos": cos_text,
    }


def diagnose(model_id: str, n_layers_to_plot: int = 4) -> None:
    print(f"Loading {model_id} ...")
    model, tokenizer = mlx_lm.load(model_id)

    layers = getattr(model, "layers", None) or model.model.layers
    n_layers = len(layers)
    print(f"  layers={n_layers}")

    embeds, total = _synth_inputs(model, N_IMG, N_TEXT)
    print(f"  synth input: {N_IMG} image rows + {N_TEXT} text rows = {total} tokens")

    # Capture keys via a stub KVCache list whose ``update_and_fetch`` saves the
    # incoming key tensor. mlx_lm's attention module hands the post-rope keys
    # to cache.update_and_fetch(keys, values) on every forward — that's exactly
    # the tensor our quantizer sees in production.
    captured: dict[int, mx.array] = {}

    from mlx_lm.models.cache import KVCache

    class _TapCache(KVCache):
        def __init__(self, idx):
            super().__init__()
            self._idx = idx

        def update_and_fetch(self, keys, values):
            captured[self._idx] = keys
            return super().update_and_fetch(keys, values)

    cache_list = [_TapCache(i) for i in range(len(layers))]
    print("  running forward pass to capture keys ...")
    _ = model(
        mx.zeros((1, 1), dtype=mx.int32),  # dummy token ids; input_embeddings overrides
        cache=cache_list,
        input_embeddings=embeds,
    )
    mx.eval(list(captured.values()))
    print(f"  captured {len(captured)} layer key tensors")

    if not captured:
        print("ERROR: no keys captured — attention module shape unexpected.")
        return

    # Per-layer stats
    print(f"\n{'=' * 100}")
    print(
        f"{'lyr':>3}  {'img_norm':>16}  {'text_norm':>16}  "
        f"{'kurt(img)':>9}  {'kurt(txt)':>9}  {'cos(img)':>8}  {'cos(txt)':>8}"
    )
    print("-" * 100)
    rows = []
    layers_to_plot = set(np.linspace(0, n_layers - 1, n_layers_to_plot, dtype=int))
    for idx in sorted(captured.keys()):
        k = captured[idx][0]  # drop batch axis: (H, S, D)
        keys_img = k[:, :N_IMG, :]
        keys_text = k[:, N_IMG:, :]
        r = _layer_stats(
            idx,
            keys_img,
            keys_text,
            save_hists=(idx in layers_to_plot),
            out_dir=OUT_DIR,
        )
        rows.append(r)
        print(
            f"{r['layer']:>3}  "
            f"{r['img_norm_mean']:>8.3f}±{r['img_norm_std']:<6.3f}  "
            f"{r['text_norm_mean']:>8.3f}±{r['text_norm_std']:<6.3f}  "
            f"{r['img_kurt']:>+9.3f}  {r['text_kurt']:>+9.3f}  "
            f"{r['img_cos']:>8.4f}  {r['text_cos']:>8.4f}"
        )
    print("=" * 100)

    # Aggregate
    arr = np.array(
        [
            [
                r["img_norm_mean"],
                r["text_norm_mean"],
                r["img_kurt"],
                r["text_kurt"],
                r["img_cos"],
                r["text_cos"],
            ]
            for r in rows
        ]
    )
    print(f"\nMEANS across layers:")
    print(f"  img_norm   = {arr[:, 0].mean():.3f}   (max={arr[:, 0].max():.3f})")
    print(f"  text_norm  = {arr[:, 1].mean():.3f}   (max={arr[:, 1].max():.3f})")
    print(f"  kurt(img)  = {arr[:, 2].mean():+.3f}")
    print(f"  kurt(text) = {arr[:, 3].mean():+.3f}")
    print(f"  cos(img)   = {arr[:, 4].mean():.4f}   (min={arr[:, 4].min():.4f})")
    print(f"  cos(text)  = {arr[:, 5].mean():.4f}   (min={arr[:, 5].min():.4f})")

    print(f"\nHistogram plots saved to {OUT_DIR}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--plots", type=int, default=4, help="Number of evenly-spaced layers to plot"
    )
    args = parser.parse_args()
    diagnose(args.model, args.plots)


if __name__ == "__main__":
    main()
