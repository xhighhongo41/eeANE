"""Replacement bodies shared by the decoder-style compile backends.

Several decoder architectures carry the same two helper functions, copied
verbatim into each architecture's own module, and both stand in the way of
a conversion in the same way:

* ``rotate_half`` splits the last dimension with a Python-level division
  on a traced shape, which leaves a node in the traced graph that the
  Core ML converter cannot fold into a static slice;
* ``repeat_kv`` expands grouped key/value heads through a rank-5
  intermediate, which makes the Neural Engine compiler reject the
  attention subgraph -- and rejecting one subgraph makes the whole
  compiled model fall back to the CPU.

This module holds the replacements as plain functions, plus the finite
attention-mask fill value those backends build their in-graph masks with.
It deliberately rebinds nothing: the function to replace lives in a
different module for every architecture, so each backend decides which
module's symbol it points at these bodies. That also keeps this module
free of any architecture import.

Importing this module pulls in ``torch``; it therefore requires the
``[compile]`` extra and must never be imported from the ``eeane serve``
code path (see :mod:`eeane.compiler`).
"""

from __future__ import annotations

import torch

__all__ = ["MASK_FILL_VALUE", "repeat_kv", "rotate_half"]

# Additive value written into the masked-out positions of an attention
# mask. The value the framework itself uses is the float32 minimum, which
# is not representable in FP16: once the converted graph runs in FP16 it
# becomes -inf, and a row whose keys are all masked then computes
# -inf - (-inf) = NaN inside the softmax. -1e4 is exact in FP16, drives
# exp() to exactly 0.0 in both precisions, and keeps such a row finite.
MASK_FILL_VALUE = -1e4


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of ``x``, without consulting its shape.

    Upstream selects the two halves with ``x[..., : x.shape[-1] // 2]``.
    ``torch.chunk`` with a constant chunk count selects the same halves
    without any shape arithmetic at graph level; the results are identical
    whenever the last dimension is even, which the calling backend has to
    enforce before pointing a model at this function.

    Args:
        x: Tensor whose last dimension is the (even) rotary head width.

    Returns:
        ``cat((-second half, first half))`` along the last dimension.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped key/value heads to the query head count, staying rank-4.

    Grouped-query attention stores fewer key/value heads than query heads
    and expands them before the attention product. Upstream does that
    through ``x[:, :, None, :, :].expand(...).reshape(...)``;
    ``repeat_interleave`` repeats each key/value head ``n_rep`` times in
    place instead, which is exactly the upstream head order
    ``head = kv_index * n_rep + rep_index``. As upstream does, an
    expansion by one returns the input untouched.

    Args:
        hidden_states: Key or value states of shape
            ``(batch, num_key_value_heads, seq_len, head_dim)``.
        n_rep: Number of query heads sharing one key/value head.

    Returns:
        States of shape ``(batch, num_key_value_heads * n_rep, seq_len,
        head_dim)``.
    """
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(n_rep, dim=1)
