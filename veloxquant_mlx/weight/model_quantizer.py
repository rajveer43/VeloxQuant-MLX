from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from veloxquant_mlx.weight.quantized_linear import QuantizedLinear


def _dequantize_mlx_linear(child: nn.Module) -> tuple:
    """Extract fp16 weight (and optional bias) from an mlx-lm QuantizedLinear.

    mlx-lm's QuantizedLinear stores weights as (scales, biases, w_q) in affine
    format. mx.dequantize reconstructs the fp16 weight matrix.

    Returns:
        (weight, bias) where weight is fp32 mx.array and bias may be None.
    """
    # mx.dequantize signature: (w_q, scales, biases, group_size, bits)
    weight = mx.dequantize(
        child.weight,
        child.scales,
        child.biases,
        child.group_size,
        child.bits,
    ).astype(mx.float32)
    bias = getattr(child, "bias", None)
    return weight, bias


def _is_quantized_linear(child: nn.Module) -> bool:
    """True if child is mlx-lm's QuantizedLinear (already affine-quantized)."""
    return (
        type(child).__name__ == "QuantizedLinear"
        and hasattr(child, "scales")
        and hasattr(child, "biases")
        and hasattr(child, "group_size")
        and hasattr(child, "bits")
    )


def quantize_model(
    model: nn.Module,
    bits: int = 4,
    use_hadamard: bool = True,
    skip_embeddings: bool = True,
    seed: int = 42,
) -> nn.Module:
    """Replace all Linear layers in a model with TurboQuant QuantizedLinear.

    Handles both:
    - nn.Linear (fp16 weights) — compressed directly
    - mlx-lm's QuantizedLinear (affine-quantized) — dequantized first, then
      re-quantized with TurboQuant's rotation + Lloyd-Max scheme

    Args:
        model: Any mlx.nn.Module (e.g. a loaded mlx-lm model).
        bits: Bit-width for TurboQuant compression (2, 3, or 4).
        use_hadamard: Use Metal-accelerated Hadamard rotation (recommended).
        skip_embeddings: Skip embedding-like layers.
        seed: Base random seed. Each layer gets seed + layer_index.

    Returns:
        The same model with Linear layers replaced (in-place mutation + return).

    Example::

        import mlx_lm
        model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")
        model = quantize_model(model, bits=3, use_hadamard=True)
    """
    _replace_linears(model, bits=bits, use_hadamard=use_hadamard, seed=seed, counter=[0])
    mx.eval(model.parameters())
    return model


def _replace_linears(
    module: nn.Module,
    bits: int,
    use_hadamard: bool,
    seed: int,
    counter: list,
) -> None:
    """Recursively walk module tree and replace Linear layers in-place."""
    for name, child in module.named_modules():
        is_nn_linear = isinstance(child, nn.Linear) and not _is_quantized_linear(child)
        is_mlx_lm_quantized = _is_quantized_linear(child)

        if not (is_nn_linear or is_mlx_lm_quantized):
            continue

        if _is_embedding_like(name):
            continue

        # Extract weight + bias regardless of source type
        if is_mlx_lm_quantized:
            weight, bias = _dequantize_mlx_linear(child)
        else:
            weight = child.weight.astype(mx.float32)
            bias = getattr(child, "bias", None)

        layer_seed = seed + counter[0]
        counter[0] += 1

        q_layer = QuantizedLinear(
            in_features=weight.shape[1],
            out_features=weight.shape[0],
            bits=bits,
            use_hadamard=use_hadamard,
            bias=bias is not None,
            seed=layer_seed,
        )
        q_layer.quantize_weights(weight, bias=bias)

        # Replace the child in the parent module
        _set_nested_attr(module, name, q_layer)


def _is_embedding_like(name: str) -> bool:
    """Heuristic: skip layers whose name suggests they are embedding projections."""
    lower = name.lower()
    return any(kw in lower for kw in ("embed", "lm_head", "tok_emb"))


def _set_nested_attr(root: nn.Module, dotted_name: str, value: nn.Module) -> None:
    """Set a nested attribute given a dotted path like 'model.layers.27.mlp.up_proj'.

    Handles both attribute access (strings) and list indexing (integer strings).
    """
    parts = dotted_name.split(".")
    obj = root
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def compression_report(model: nn.Module) -> dict:
    """Summarise memory savings across all QuantizedLinear layers.

    Args:
        model: Model after quantize_model() has been called.

    Returns:
        Dict with total_compressed_bytes, total_fp16_bytes, ratio, n_layers.
    """
    total_compressed = 0
    total_fp16 = 0
    n_layers = 0

    for _, child in model.named_modules():
        if isinstance(child, QuantizedLinear):
            total_compressed += child.memory_bytes
            total_fp16 += child.fp16_bytes
            n_layers += 1

    ratio = total_fp16 / total_compressed if total_compressed > 0 else 0.0
    return {
        "n_layers": n_layers,
        "total_compressed_mb": total_compressed / 1024**2,
        "total_fp16_mb": total_fp16 / 1024**2,
        "compression_ratio": ratio,
    }
