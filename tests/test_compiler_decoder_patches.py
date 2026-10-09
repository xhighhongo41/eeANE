"""Tests for the replacement bodies shared by the decoder-style backends.

The functions in ``eeane.compiler.backends.decoder_patches`` stand in for
upstream helpers of more than one architecture, so they are checked here
against every upstream implementation a backend rebinds them over: each
must compute the same values, bit for bit.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from transformers.models.gemma3 import modeling_gemma3
from transformers.models.qwen3 import modeling_qwen3

from eeane.compiler.backends import decoder_patches
from eeane.compiler.backends import qwen3 as q3

# The upstream implementations, captured at import time, before any test
# of this process can have rebound them.
_UPSTREAM_ROTATE_HALF: dict[str, Callable[..., torch.Tensor]] = {
    "qwen3": modeling_qwen3.rotate_half,
    "gemma3": modeling_gemma3.rotate_half,
}
_UPSTREAM_REPEAT_KV: dict[str, Callable[..., torch.Tensor]] = {
    "qwen3": modeling_qwen3.repeat_kv,
    "gemma3": modeling_gemma3.repeat_kv,
}


@pytest.mark.parametrize("family", sorted(_UPSTREAM_ROTATE_HALF))
@pytest.mark.parametrize("head_dim", [2, 8, 16])
def test_rotate_half_matches_the_upstream_formula(family: str, head_dim: int) -> None:
    """The chunk-based split must equal the slice-based upstream one, bit for bit."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 6, head_dim)

    assert torch.equal(decoder_patches.rotate_half(x), _UPSTREAM_ROTATE_HALF[family](x))


@pytest.mark.parametrize("family", sorted(_UPSTREAM_REPEAT_KV))
@pytest.mark.parametrize("n_rep", [2, 3, 4])
def test_repeat_kv_reproduces_the_upstream_head_order(family: str, n_rep: int) -> None:
    """The rank-4 expansion must place every repeated head where upstream puts it."""
    torch.manual_seed(1)
    hidden_states = torch.randn(2, 3, 5, 8)

    expanded = decoder_patches.repeat_kv(hidden_states, n_rep)

    assert expanded.dim() == 4
    assert torch.equal(expanded, _UPSTREAM_REPEAT_KV[family](hidden_states, n_rep))


def test_repeat_kv_returns_the_input_when_nothing_is_shared() -> None:
    """``n_rep == 1`` must short-circuit to the very input tensor, as upstream does."""
    hidden_states = torch.randn(2, 3, 4, 5)

    assert decoder_patches.repeat_kv(hidden_states, 1) is hidden_states


def test_the_mask_fill_value_survives_the_precision_the_graph_runs_in() -> None:
    """A fill value that becomes -inf in FP16 would turn a masked softmax row into NaN."""
    as_half = np.float16(decoder_patches.MASK_FILL_VALUE)

    assert np.isfinite(as_half)
    assert float(as_half) == decoder_patches.MASK_FILL_VALUE
    assert float(np.exp(as_half.astype(np.float32))) == 0.0


def test_the_backends_share_one_fill_value() -> None:
    """A backend re-exporting the constant must not carry a value of its own."""
    assert q3.MASK_FILL_VALUE == decoder_patches.MASK_FILL_VALUE
