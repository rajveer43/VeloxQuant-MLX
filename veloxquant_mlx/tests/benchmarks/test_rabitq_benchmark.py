"""The RaBitQ benchmark's memory claim must come from what encode() really stores."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from veloxquant_mlx.benchmarks import rabitq_benchmark as rb
from veloxquant_mlx.quantizers.rabitq import RaBitQQuantizer


def test_measured_bytes_per_vector_includes_correction_factors():
    """Regression: the memory figure claimed D // 8 bytes/vector (16x vs fp16), but
    encode() also stores per-vector scalars, so the real footprint is larger and
    the real compression ratio lower."""
    d = 128
    rng = np.random.default_rng(0)
    q = RaBitQQuantizer(d=d, nlist=4, nprobe=2, rerank=8, seed=0)
    q.fit(mx.array(rng.standard_normal((256, d)).astype(np.float16)), max_samples=256)
    ev = q.encode(mx.array(rng.standard_normal((64, d)).astype(np.float16)))
    mx.eval(ev.indices, ev.norm)

    bytes_per_vector = rb.measured_bytes_per_vector(ev)
    assert bytes_per_vector > d // 8  # codes alone understate the footprint
    assert bytes_per_vector == ev.memory_bytes() / 64
    assert (d * 2) / bytes_per_vector < 16  # not the advertised 16x
