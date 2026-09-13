"""JSON-lines worker used by the JavaScript SDK."""

from __future__ import annotations

import json
import sys
from typing import Any

PROTOCOL_VERSION = 1


def _reply(request_id: str, *, result: Any = None, error: Any = None) -> None:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "id": request_id,
        "ok": error is None,
    }
    payload["result" if error is None else "error"] = result if error is None else error
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def main() -> None:
    from veloxquant_mlx import __version__

    for line in sys.stdin:
        request: Any = None
        try:
            request = json.loads(line)
            request_id = str(request["id"])
            op = request.get("op")
            if op == "ping":
                _reply(request_id, result={"version": __version__})
            elif op == "capabilities":
                _reply(request_id, result=_capabilities(__version__))
            elif op == "metal_probe":
                _reply(request_id, result=_metal_probe())
            elif op == "bit_pack":
                _reply(request_id, result=_bit_pack(request.get("args", {})))
            elif op == "bit_pack_file":
                _reply(request_id, result=_bit_pack_file(request.get("args", {})))
            elif op == "rope_recode_file":
                _reply(request_id, result=_rope_recode_file(request.get("args", {})))
            elif op == "shutdown":
                _reply(request_id, result={"stopped": True})
                return
            else:
                _reply(request_id, error=f"unknown worker operation: {op!r}")
        except Exception as exc:
            _reply(
                str(request.get("id", "unknown")) if isinstance(request, dict) else "unknown",
                error={"code": "WORKER_ERROR", "message": str(exc)},
            )


def _capabilities(version: str) -> dict[str, Any]:
    import mlx.core as mx

    return {
        "veloxquantVersion": version,
        "mlxVersion": getattr(mx, "__version__", None),
        "device": str(mx.default_device()),
        "metalAvailable": True,
        "supportedOperations": [
            "ping",
            "capabilities",
            "metal_probe",
            "bit_pack",
            "bit_pack_file",
            "rope_recode_file",
        ],
    }


def _bit_pack(args: dict[str, Any]) -> dict[str, Any]:
    bits = int(args.get("bits", 0))
    values = args.get("values")
    if bits not in (1, 2, 4):
        raise ValueError("INVALID_BITS: bits must be one of 1, 2, or 4")
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise ValueError("INVALID_VALUES: values must be an integer array")
    import mlx.core as mx

    from veloxquant_mlx.metal._bit_packing import turboquant_bit_pack

    packed = turboquant_bit_pack(mx.array(values, dtype=mx.uint8), bits)
    mx.eval(packed)
    return {
        "bits": bits,
        "inputLength": len(values),
        "values": [int(value) for value in packed.tolist()],
        "backend": "metal",
        "device": str(mx.default_device()),
    }


def _bit_pack_file(args: dict[str, Any]) -> dict[str, Any]:
    from pathlib import Path

    import numpy as np

    input_path = Path(str(args.get("inputPath", ""))).resolve()
    output_path = Path(str(args.get("outputPath", ""))).resolve()
    if input_path.suffix != ".npy" or output_path.suffix != ".npy":
        raise ValueError("INVALID_PATH: inputPath and outputPath must be .npy files")
    if not input_path.is_file():
        raise FileNotFoundError(f"INPUT_NOT_FOUND: {input_path}")
    values = np.load(input_path, allow_pickle=False)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("INVALID_ARRAY: input must be a one-dimensional integer .npy array")
    result = _bit_pack(
        {"values": [int(value) for value in values.tolist()], "bits": args.get("bits")}
    )
    np.save(output_path, np.asarray(result["values"], dtype=np.uint8), allow_pickle=False)
    return {
        "outputPath": str(output_path),
        "shape": [len(result["values"])],
        "dtype": "uint8",
        "inputLength": result["inputLength"],
        "bits": result["bits"],
        "backend": result["backend"],
        "device": result["device"],
    }


def _rope_recode_file(args: dict[str, Any]) -> dict[str, Any]:
    from pathlib import Path

    import mlx.core as mx
    import numpy as np

    from veloxquant_mlx.metal._crosskv_rope import crosskv_rope_recode

    input_path = Path(str(args.get("inputPath", ""))).resolve()
    positions_path = Path(str(args.get("positionsPath", ""))).resolve()
    output_path = Path(str(args.get("outputPath", ""))).resolve()
    if any(path.suffix != ".npy" for path in (input_path, positions_path, output_path)):
        raise ValueError(
            "INVALID_PATH: inputPath, positionsPath, and outputPath must be .npy files"
        )
    if not input_path.is_file() or not positions_path.is_file():
        raise FileNotFoundError("INPUT_NOT_FOUND: input or positions file does not exist")
    keys = np.load(input_path, allow_pickle=False)
    positions = np.load(positions_path, allow_pickle=False)
    if keys.ndim != 3 or keys.shape[2] % 2 != 0:
        raise ValueError("INVALID_ARRAY: keys must have shape [BH, N, even_head_dim]")
    if positions.ndim != 1 or positions.shape[0] != keys.shape[1]:
        raise ValueError("INVALID_POSITIONS: positions must be [N] matching keys")
    source_base = float(args["sourceBase"])
    target_base = float(args["targetBase"])
    result = crosskv_rope_recode(mx.array(keys), mx.array(positions), source_base, target_base)
    mx.eval(result)
    np.save(output_path, np.asarray(result.tolist(), dtype=keys.dtype), allow_pickle=False)
    return {
        "outputPath": str(output_path),
        "shape": list(keys.shape),
        "dtype": str(keys.dtype),
        "backend": "metal",
        "device": str(mx.default_device()),
    }


def _metal_probe() -> dict[str, Any]:
    """Compile and execute one tiny MLX custom Metal kernel."""
    import mlx.core as mx

    source = """
        uint i = thread_position_in_grid.x;
        if (i < x_shape[0]) out[i] = x[i] + 1.0f;
    """
    kernel = mx.fast.metal_kernel(
        name="veloxquant_worker_probe",
        input_names=["x"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )
    values = mx.array([1.0, 2.0, 3.0], dtype=mx.float32)
    output = kernel(
        inputs=[values],
        grid=(3, 1, 1),
        threadgroup=(3, 1, 1),
        output_shapes=[(3,)],
        output_dtypes=[mx.float32],
    )[0]
    mx.eval(output)
    result = [float(value) for value in output.tolist()]
    return {
        "device": str(mx.default_device()),
        "output": result,
        "passed": result == [2.0, 3.0, 4.0],
    }
