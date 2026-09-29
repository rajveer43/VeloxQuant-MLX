# VeloxQuant-MLX

VeloxQuant-MLX shrinks the KV cache of any `mlx_lm` model on Apple Silicon, up to 16x smaller with near-lossless quality, in three lines of code. If you run models locally and keep hitting a context-length or memory wall, you swap in a compressed cache and change nothing else about the model.

Inside are 43 compression methods, each adapted from a published paper, spanning zero-calibration 1-bit quantizers, token-eviction caches, and cross-layer merging. All of them share the same 3-line API, so switching means changing `method="..."`. The hot path runs on hand-written Metal kernels, and it's validated on 12 production models (Llama, Mistral, Qwen, Phi, Gemma 3/4, Falcon) plus vision-language models via [mlx-vlm](https://github.com/Blaizzy/mlx-vlm).

This site is the **API reference**, generated from the package's docstrings. For guides, algorithm write-ups, and the playground, see [veloxquant.dev](https://veloxquant.dev/).

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart
api/index
changelog
```
