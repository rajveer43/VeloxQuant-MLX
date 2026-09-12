"""Hand-implemented data structures backing VeloxQuant-MLX's quantizer internals.

Groups the manually written (no ``heapq``/stdlib shortcuts) structures used by
the codebook, outlier-tracking, and pipeline-validation code: ``AVLTree`` /
``VoronoiTree`` for nearest-centroid search, ``BitPackBuffer`` for sub-byte
index packing, ``QuantizationGraph`` for DAG validation and topological
ordering of handler chains, ``MaxHeap`` / ``SortedChannelIndex`` for streaming
top-k channel tracking, and ``RingBuffer`` for fixed-capacity FIFO windows.
"""

from __future__ import annotations

from veloxquant_mlx.dsa.avl_tree import AVLNode, AVLTree, VoronoiTree
from veloxquant_mlx.dsa.bit_pack import BitPackBuffer
from veloxquant_mlx.dsa.dag import QuantizationGraph
from veloxquant_mlx.dsa.heap import MaxHeap, SortedChannelIndex
from veloxquant_mlx.dsa.ring_buffer import RingBuffer

__all__ = [
    "AVLNode",
    "AVLTree",
    "VoronoiTree",
    "BitPackBuffer",
    "QuantizationGraph",
    "MaxHeap",
    "SortedChannelIndex",
    "RingBuffer",
]
