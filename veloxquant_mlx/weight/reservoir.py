from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from veloxquant_mlx.codebooks.scalar_codebook import ScalarCodebook
from veloxquant_mlx.math.rotation import make_rotation_matrix
from veloxquant_mlx.preconditioners.rotation import HadamardPreconditioner, RotationPreconditioner
from veloxquant_mlx.weight.quantized_linear import QuantizedLinear

_MAGIC = b"VQRS"  # VeloxQuant Reservoir
_VERSION = 2
_PAGE_SIZE = 16384  # Apple Silicon page size; also a safe alignment for x86 mmap


def _pad_to_page(n: int) -> int:
    return (n + _PAGE_SIZE - 1) // _PAGE_SIZE * _PAGE_SIZE


def save_reservoir(model: nn.Module, path: Union[str, Path], persist_rotation: bool = False) -> None:
    """Serialize a quantize_model()-processed model's QuantizedLinear layers.

    Writes a single flat file: magic/version, a JSON header describing each
    layer's shape and quantization parameters, followed by page-aligned
    index/norms/rotation/centroids blobs. See
    docs/WEIGHT_RESERVOIR_IDEATION.md for format rationale and the measured
    findings on why this is a load-time / compressed-copy-size optimization
    rather than a cross-process RAM-sharing one.

    Args:
        model: A model whose Linear layers have already been replaced by
            quantize_model() (or manually constructed QuantizedLinear layers
            with quantize_weights() called).
        path: Output file path.
        persist_rotation: If True, persist each QR-fallback layer's full
            d x d rotation matrix so load_reservoir() never calls
            np.linalg.qr. This makes load ~9-230x faster (measured; see
            Finding 3 in docs/WEIGHT_RESERVOIR_IDEATION.md) but a dense
            random d x d orthogonal matrix is inherently O(d^2) information
            with no compact encoding, so this can make the file MUCH larger
            than the source model for architectures with non-Hadamard-
            compatible layer widths (measured: 9.5x larger for
            Qwen2.5-0.5B-Instruct-4bit, Finding 4). Default False: rotation
            matrices are re-derived from the layer's seed on load (same QR
            cost quantize_model() already pays), keeping the file close to
            the size of the compressed weights themselves. Hadamard-
            compatible layers persist their (d,) diagonal either way --
            that storage is cheap regardless of this flag.
    """
    layers: list[tuple[str, QuantizedLinear]] = [
        (name, child) for name, child in model.named_modules() if isinstance(child, QuantizedLinear)
    ]

    header_layers = []
    index_segments: list[np.ndarray] = []
    norms_segments: list[np.ndarray] = []
    rotation_segments: list[np.ndarray] = []
    centroids_segments: list[np.ndarray] = []
    index_offset = 0
    norms_offset = 0
    rotation_offset = 0
    centroids_offset = 0

    for name, layer in layers:
        idx_np = np.array(layer._w_indices, copy=False, dtype=np.uint8)
        norms_np = np.array(layer._w_norms, copy=False, dtype=np.float32)
        bias_np = None if layer._bias is None else np.array(layer._bias, copy=False, dtype=np.float16)

        is_hadamard = isinstance(layer._preconditioner, HadamardPreconditioner)
        # Hadamard diagonals are (d,) -- always cheap, always persisted.
        # QR rotation matrices are (d,d) -- only persisted when the caller
        # opts in via persist_rotation, since they can dominate file size
        # (Finding 4). When not persisting, store an empty array; the
        # loader re-derives the matrix from (seed, in_features) instead.
        if is_hadamard:
            rotation_np = np.array(layer._preconditioner._D, copy=False, dtype=np.float32)
        elif persist_rotation:
            rotation_np = np.array(layer._preconditioner._Pi, copy=False, dtype=np.float32)
        else:
            rotation_np = np.zeros(0, dtype=np.float32)
        centroids_np = np.array(layer._codebook.centroids_numpy(), copy=False, dtype=np.float32)

        idx_bytes = idx_np.nbytes
        norms_bytes = norms_np.nbytes
        rotation_bytes = rotation_np.nbytes
        centroids_bytes = centroids_np.nbytes
        idx_padded = _pad_to_page(idx_bytes)
        norms_padded_bytes = _pad_to_page(norms_bytes)
        norms_padded_elems = norms_padded_bytes // 4
        rotation_padded_bytes = _pad_to_page(rotation_bytes)
        rotation_padded_elems = rotation_padded_bytes // 4
        centroids_padded_bytes = _pad_to_page(centroids_bytes)
        centroids_padded_elems = centroids_padded_bytes // 4

        header_layers.append(
            {
                "name": name,
                "out_features": layer._out,
                "in_features": layer._in,
                "bits": layer._bits,
                "seed": layer._seed,
                "use_hadamard": is_hadamard,
                "has_bias": bias_np is not None,
                "bias": bias_np.tobytes().hex() if bias_np is not None else None,
                "index_offset": index_offset,
                "index_nbytes": idx_bytes,
                "norms_offset": norms_offset,
                "norms_nbytes": norms_bytes,
                "rotation_offset": rotation_offset,
                "rotation_nbytes": rotation_bytes,
                "centroids_offset": centroids_offset,
                "centroids_nbytes": centroids_bytes,
            }
        )

        norms_flat = norms_np.reshape(-1)
        rotation_flat = rotation_np.reshape(-1)
        centroids_flat = centroids_np.reshape(-1)
        index_segments.append(np.pad(idx_np.reshape(-1), (0, idx_padded - idx_bytes)))
        norms_segments.append(np.pad(norms_flat, (0, norms_padded_elems - norms_flat.size)))
        rotation_segments.append(np.pad(rotation_flat, (0, rotation_padded_elems - rotation_flat.size)))
        centroids_segments.append(np.pad(centroids_flat, (0, centroids_padded_elems - centroids_flat.size)))
        index_offset += idx_padded
        norms_offset += norms_padded_bytes
        rotation_offset += rotation_padded_bytes
        centroids_offset += centroids_padded_bytes

    header = {
        "version": _VERSION,
        "layers": header_layers,
        "index_blob_nbytes": index_offset,
        "norms_blob_nbytes": norms_offset,
        "rotation_blob_nbytes": rotation_offset,
        "centroids_blob_nbytes": centroids_offset,
    }
    header_bytes = json.dumps(header).encode("utf-8")
    header_padded = _pad_to_page(len(header_bytes) + 12)  # magic(4) + version(4) + header_len(4)

    index_blob = np.concatenate(index_segments) if index_segments else np.zeros(0, dtype=np.uint8)
    norms_blob = np.concatenate(norms_segments) if norms_segments else np.zeros(0, dtype=np.float32)
    rotation_blob = np.concatenate(rotation_segments) if rotation_segments else np.zeros(0, dtype=np.float32)
    centroids_blob = np.concatenate(centroids_segments) if centroids_segments else np.zeros(0, dtype=np.float32)

    path = Path(path)
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<I", _VERSION))
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(b"\x00" * (header_padded - 12 - len(header_bytes)))
        f.write(index_blob.tobytes())
        f.write(norms_blob.tobytes())
        f.write(rotation_blob.tobytes())
        f.write(centroids_blob.tobytes())


def _fast_quantized_linear(entry: dict, rotation_np: np.ndarray, centroids_np: np.ndarray) -> QuantizedLinear:
    """Construct a QuantizedLinear without paying QuantizedLinear.__init__'s
    rotation-matrix / Lloyd-Max setup cost -- both are regenerated instead
    from the persisted rotation_np / centroids_np arrays.

    QuantizedLinear.__init__ always derives the preconditioner and codebook
    from (seed, in_features), which for QR-fallback layers means a full
    d x d np.linalg.qr -- measured at 55s of a 66.6s single-process load for
    a 0.5B model's 24 non-Hadamard-compatible layers (see
    docs/WEIGHT_RESERVOIR_IDEATION.md, PoC scope note). Since the reservoir
    already has that exact rotation persisted, calling __init__ would
    silently redo (and discard) that work. This bypasses it via
    nn.Module.__init__ + direct field assignment, mirroring exactly what
    QuantizedLinear.__init__ sets up.
    """
    layer = QuantizedLinear.__new__(QuantizedLinear)
    nn.Module.__init__(layer)
    layer._in = entry["in_features"]
    layer._out = entry["out_features"]
    layer._bits = entry["bits"]
    layer._seed = entry["seed"]

    if entry["use_hadamard"]:
        layer._preconditioner = HadamardPreconditioner(mx.array(rotation_np))
    else:
        layer._preconditioner = RotationPreconditioner(mx.array(rotation_np))

    layer._codebook = ScalarCodebook(centroids_np)
    layer._centroids = layer._codebook.centroids_mx()

    layer._w_indices = mx.zeros((entry["out_features"], entry["in_features"]), dtype=mx.uint8)
    layer._w_norms = mx.ones((entry["out_features"], 1), dtype=mx.float32)
    layer._bias = None
    layer._has_bias = entry["has_bias"]
    return layer


def load_reservoir(path: Union[str, Path]) -> dict[str, Any]:
    """Deserialize a reservoir file into a flat {layer_name: QuantizedLinear} dict.

    Reconstructs each layer from its persisted rotation matrix/diagonal and
    codebook centroids (not by recomputing them from the seed -- see
    _fast_quantized_linear), then assigns the persisted _w_indices /
    _w_norms directly. This skips both the dequantize -> rotate ->
    Lloyd-Max-argmin quantization pass AND the rotation/codebook setup cost,
    which for QR-fallback layers otherwise dominates load time.

    The file is memory-mapped (np.memmap) for the read; per Finding 2 in
    docs/WEIGHT_RESERVOIR_IDEATION.md, mx.array() still performs one copy
    out of that mapping into MLX's unified-memory arena — there is no
    zero-copy constructor in the installed MLX version. The win is a
    smaller copy (2-4 bit indices instead of fp16 weights) and no
    re-quantization/re-setup compute, not zero RAM cost.

    Args:
        path: Reservoir file written by save_reservoir().

    Returns:
        Dict mapping each layer's dotted module name to its reconstructed
        QuantizedLinear instance. Callers that need a full model should
        graft these back onto a skeleton module tree by name.
    """
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != _MAGIC:
            raise ValueError(f"Not a VeloxQuant reservoir file: {path}")
        (version,) = struct.unpack("<I", f.read(4))
        if version != _VERSION:
            raise ValueError(f"Unsupported reservoir version {version}, expected {_VERSION}")
        (header_len,) = struct.unpack("<I", f.read(4))
        header = json.loads(f.read(header_len).decode("utf-8"))
        header_padded = _pad_to_page(header_len + 12)
        data_start = header_padded

    index_blob = np.memmap(
        path, dtype=np.uint8, mode="r", offset=data_start, shape=(header["index_blob_nbytes"],)
    )
    norms_start = data_start + header["index_blob_nbytes"]
    norms_blob = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=norms_start,
        shape=(header["norms_blob_nbytes"],),
    ).view(np.float32)
    rotation_start = norms_start + header["norms_blob_nbytes"]
    rotation_blob = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=rotation_start,
        shape=(header["rotation_blob_nbytes"],),
    ).view(np.float32)
    centroids_start = rotation_start + header["rotation_blob_nbytes"]
    centroids_blob = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=centroids_start,
        shape=(header["centroids_blob_nbytes"],),
    ).view(np.float32)

    layers: dict[str, QuantizedLinear] = {}
    for entry in header["layers"]:
        rot_elems = entry["rotation_nbytes"] // 4
        rot_off = entry["rotation_offset"] // 4
        rotation_flat = np.array(rotation_blob[rot_off : rot_off + rot_elems])
        if entry["use_hadamard"]:
            rotation_np = rotation_flat  # (d,)
        elif rot_elems > 0:
            d = entry["in_features"]
            rotation_np = rotation_flat.reshape(d, d)
        else:
            # persist_rotation=False at save time: not persisted, re-derive
            # deterministically from seed (same QR cost quantize_model()
            # already pays -- see Finding 4 in docs/WEIGHT_RESERVOIR_IDEATION.md).
            rotation_np = make_rotation_matrix(entry["in_features"], seed=entry["seed"]).astype(np.float32)

        cen_elems = entry["centroids_nbytes"] // 4
        cen_off = entry["centroids_offset"] // 4
        centroids_np = np.array(centroids_blob[cen_off : cen_off + cen_elems])

        layer = _fast_quantized_linear(entry, rotation_np, centroids_np)

        idx_flat = np.array(
            index_blob[entry["index_offset"] : entry["index_offset"] + entry["index_nbytes"]]
        )
        norms_flat = np.array(
            norms_blob[entry["norms_offset"] // 4 : entry["norms_offset"] // 4 + entry["norms_nbytes"] // 4]
        )

        layer._w_indices = mx.array(idx_flat.reshape(entry["out_features"], entry["in_features"]))
        layer._w_norms = mx.array(norms_flat.reshape(entry["out_features"], 1))
        if entry["has_bias"] and entry["bias"] is not None:
            bias_np = np.frombuffer(bytes.fromhex(entry["bias"]), dtype=np.float16)
            layer._bias = mx.array(bias_np)
        mx.eval(layer._w_indices, layer._w_norms)

        layers[entry["name"]] = layer

    return layers


def graft_reservoir(model: nn.Module, path: Union[str, Path]) -> nn.Module:
    """Load a reservoir and replace the matching named modules in `model` in place.

    `model` must have Linear/QuantizedLinear layers at the same dotted names
    the reservoir was saved with (i.e. it should be the same architecture
    quantize_model() would have walked). Layers not present in the reservoir
    are left untouched.

    Args:
        model: Model skeleton to graft reservoir layers onto.
        path: Reservoir file written by save_reservoir().

    Returns:
        The same model with matching layers replaced (in-place mutation + return).
    """
    from veloxquant_mlx.weight.model_quantizer import _set_nested_attr

    layers = load_reservoir(path)
    for name, layer in layers.items():
        _set_nested_attr(model, name, layer)
    mx.eval(model.parameters())
    return model
