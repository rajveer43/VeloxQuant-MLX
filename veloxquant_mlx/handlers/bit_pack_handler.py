from __future__ import annotations

import numpy as np

from veloxquant_mlx.core.abstractions import QuantizationHandler
from veloxquant_mlx.core.context import QuantizationContext
from veloxquant_mlx.dsa.bit_pack import BitPackBuffer


class BitPackingHandler(QuantizationHandler):
    """Pack codebook indices into b-bit compact storage.

    On encode:
        Reads ctx.indices (batch, d) uint8.
        Packs each row independently using BitPackBuffer.
        ctx.packed_bits = packed byte array.

    On decode:
        Reads ctx.packed_bits and the original n (d).
        Unpacks back to (batch, d) uint8 and stores in ctx.indices.

    Args:
        b: Bit-width per index.
        d: Vector dimension (needed for unpack).
    """

    def __init__(self, b: int, d: int) -> None:
        self._buf = BitPackBuffer(b=b)
        self._d = d
        self._b = b

    def handle(self, ctx: QuantizationContext) -> QuantizationContext:
        """Pack or unpack codebook indices.

        Args:
            ctx: Quantization context.

        Returns:
            Updated context.
        """
        if ctx.mode == "encode":
            if ctx.indices is not None:
                idx_np = np.array(ctx.indices, dtype=np.uint8)
                batch = idx_np.shape[0]
                packed_rows = [self._buf.pack(idx_np[i]) for i in range(batch)]
                ctx.packed_bits = np.stack(packed_rows, axis=0)  # (batch, n_bytes)
        else:
            if ctx.packed_bits is not None:
                batch = ctx.packed_bits.shape[0]
                unpacked = [self._buf.unpack(ctx.packed_bits[i], self._d) for i in range(batch)]
                idx_np = np.stack(unpacked, axis=0)  # (batch, d)
                import mlx.core as mx

                ctx.indices = mx.array(idx_np.astype(np.uint8))

        return self._pass_to_next(ctx)

    @property
    def handler_name(self) -> str:
        return "bit_packing"

    def __repr__(self) -> str:
        return f"BitPackingHandler(b={self._b}, d={self._d})"
