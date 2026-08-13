"""Parity tests for the fused cross-model RoPE re-encode kernel.

``crosskv_rope_recode`` must reproduce the MLX reference path
(``apply_rope(strip_rope(k, p, θ_s), p, θ_t)`` in
``veloxquant_mlx/transfer/rope.py``) to floating-point tolerance. The kernel
folds both rotations into a single per-dimension angle difference, so an error
in the exponent, the base ordering, or the NeoX half-split pairing would show
up as a wrong-but-plausible key rather than a crash — hence parity testing
against the reference rather than spot-checking outputs.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from veloxquant_mlx.metal import metal_available
from veloxquant_mlx.metal.kernels import crosskv_rope_recode
from veloxquant_mlx.transfer.rope import apply_rope, recode_rope, strip_rope

pytestmark = pytest.mark.skipif(
    not metal_available(),
    reason="Metal compute kernels not available on this build of mlx.",
)

SRC_BASE = 10000.0
TGT_BASE = 1000000.0


def _reference(keys, positions, src=SRC_BASE, tgt=TGT_BASE):
    """Two-pass MLX path the kernel replaces."""
    return apply_rope(strip_rope(keys, positions, src), positions, tgt)


@pytest.mark.parametrize(
    "bh,n,d",
    [
        (1, 1, 8),  # single element — index arithmetic edge case
        (4, 17, 64),  # non-power-of-two token count
        (2, 128, 128),  # realistic head_dim
        (8, 512, 64),  # many groups
    ],
)
def test_matches_mlx_reference(bh, n, d):
    mx.random.seed(0)
    keys = mx.random.normal((bh, n, d)).astype(mx.float32)
    pos = mx.arange(n)
    err = mx.max(mx.abs(_reference(keys, pos) - crosskv_rope_recode(keys, pos, SRC_BASE, TGT_BASE)))
    assert float(err) < 1e-4


def test_matches_reference_in_fp16():
    """Cache keys are fp16 in practice; the kernel must handle that dtype."""
    mx.random.seed(1)
    keys = mx.random.normal((4, 64, 64)).astype(mx.float16)
    pos = mx.arange(64)
    got = crosskv_rope_recode(keys, pos, SRC_BASE, TGT_BASE)
    assert got.dtype == mx.float16
    err = mx.max(mx.abs(_reference(keys, pos).astype(mx.float32) - got.astype(mx.float32)))
    assert float(err) < 1e-2


def test_equal_bases_is_identity():
    """No net rotation when source and target share a base — pins the sign."""
    mx.random.seed(2)
    keys = mx.random.normal((2, 32, 64)).astype(mx.float32)
    pos = mx.arange(32)
    out = crosskv_rope_recode(keys, pos, SRC_BASE, SRC_BASE)
    assert float(mx.max(mx.abs(out - keys))) < 1e-5


def test_preserves_norm():
    """The kernel must implement a rotation, not an arbitrary linear map."""
    mx.random.seed(3)
    keys = mx.random.normal((3, 48, 64)).astype(mx.float32)
    pos = mx.arange(48)
    out = crosskv_rope_recode(keys, pos, SRC_BASE, TGT_BASE)
    delta = mx.linalg.norm(out, axis=-1) - mx.linalg.norm(keys, axis=-1)
    assert float(mx.max(mx.abs(delta))) < 1e-3


def test_long_context_parity():
    """Precision must hold at the context lengths this exists to serve."""
    mx.random.seed(4)
    n = 4096
    keys = mx.random.normal((2, n, 128)).astype(mx.float32)
    pos = mx.arange(n)
    err = mx.max(mx.abs(_reference(keys, pos) - crosskv_rope_recode(keys, pos, SRC_BASE, TGT_BASE)))
    assert float(err) < 1e-3


def test_noncontiguous_positions():
    """Positions come from the cache and need not be 0..N-1."""
    mx.random.seed(5)
    n = 32
    keys = mx.random.normal((2, n, 64)).astype(mx.float32)
    pos = mx.array([0, 3, 9] + list(range(500, 529)))
    err = mx.max(mx.abs(_reference(keys, pos) - crosskv_rope_recode(keys, pos, SRC_BASE, TGT_BASE)))
    assert float(err) < 1e-3


def test_recode_rope_dispatches_to_kernel_and_agrees_with_mlx():
    """The public wrapper's Metal and MLX paths must not diverge."""
    mx.random.seed(6)
    keys = mx.random.normal((4, 96, 64)).astype(mx.float32)
    pos = mx.arange(96)
    gpu = recode_rope(keys, pos, SRC_BASE, TGT_BASE, use_metal=True)
    cpu = recode_rope(keys, pos, SRC_BASE, TGT_BASE, use_metal=False)
    assert float(mx.max(mx.abs(gpu - cpu))) < 1e-4


def test_empty_sequence_passthrough():
    empty = mx.zeros((2, 0, 64))
    assert crosskv_rope_recode(empty, mx.zeros((0,)), SRC_BASE, TGT_BASE).shape == (2, 0, 64)


def test_rejects_bad_shapes():
    pos = mx.arange(8)
    with pytest.raises(ValueError, match=r"\[BH, N, D\]"):
        crosskv_rope_recode(mx.zeros((8, 64)), pos, SRC_BASE, TGT_BASE)
    with pytest.raises(ValueError, match="even"):
        crosskv_rope_recode(mx.zeros((1, 8, 7)), pos, SRC_BASE, TGT_BASE)
    with pytest.raises(ValueError, match="positions has"):
        crosskv_rope_recode(mx.zeros((1, 8, 64)), mx.arange(7), SRC_BASE, TGT_BASE)
