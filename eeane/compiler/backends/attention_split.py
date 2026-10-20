"""Eager attention computed over ranges of query rows, for any architecture.

On the Neural Engine, an attention whose score matrix for one head --
query rows x keys -- holds 2**20 elements or more returns wrong numbers,
while the very same program is correct on the CPU. Every row of an
attention depends on that row alone, since the softmax runs over the
keys, so the rows of a long sequence can be computed in a few ranges,
each with a score matrix below that size, and concatenated back: the
result is what the unsplit attention computes.

The eager attention function that does this differs from upstream in
nothing but the split, and upstream carries one copy of it per
architecture module, each calling that module's own key/value expansion.
This module therefore holds a factory, :func:`split_eager_attention`,
which builds the replacement for any such module, plus the pieces it is
built from. It deliberately rebinds nothing: each backend decides which
module's symbol it points at the function it builds. That also keeps
this module free of any architecture import.

Importing this module pulls in ``torch``; it therefore requires the
``[compile]`` extra and must never be imported from the ``eeane serve``
code path (see :mod:`eeane.compiler`).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch

# Public surface of this module.
__all__ = [
    "ATTENTION_SCORE_ELEMENT_LIMIT",
    "attention_query_rows",
    "attention_rows",
    "split_eager_attention",
    "static_size",
]

# Largest number of elements one head's attention score matrix (query rows
# x keys) may hold; :func:`attention_query_rows` keeps every score matrix
# strictly below it. On the Neural Engine an attention whose score matrix
# reaches 2**20 elements returns wrong numbers -- the result no longer
# resembles the model's -- while the very same program is correct on the
# CPU, so the conversion itself gives no sign of it. The boundary was
# observed exactly here: a sequence of 1023 tokens (1023 x 1023 scores) is
# computed correctly, one of 1024 (2**20 scores) is not, with or without
# an attention mask and however that mask is built. Computing the rows of
# a 1024-token sequence in two halves (512 x 1024 scores each) is correct
# again.
ATTENTION_SCORE_ELEMENT_LIMIT = 1 << 20


def static_size(tensor: torch.Tensor, dim: int) -> int:
    """Return one dimension of ``tensor`` as a plain Python integer.

    While a module is being traced, a tensor's size is handed out as a
    traced value, and arithmetic or slicing on it is recorded as shape
    operations in the graph. Every graph compiled here has fixed shapes,
    so the size is deliberately read as the constant it is; the tracer's
    warning that the value will be treated as a constant states the
    intention and is therefore silenced for this one conversion.

    Args:
        tensor: Tensor whose size is read.
        dim: Dimension to read.

    Returns:
        The size of ``tensor`` along ``dim``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        return int(tensor.shape[dim])


def attention_query_rows(num_queries: int, num_keys: int, limit: int | None = None) -> int:
    """Return how many query rows one attention score matrix may hold.

    The row count starts at the whole sequence and is halved (rounding
    up) until rows x keys falls strictly below the limit. A sequence
    whose full score matrix is already below the limit therefore keeps
    all of its rows, which means its attention is not split at all.

    Args:
        num_queries: Number of query positions.
        num_keys: Number of key positions.
        limit: Number of elements one score matrix must stay below.
            ``None`` reads :data:`ATTENTION_SCORE_ELEMENT_LIMIT` when the
            function is called, so a value set on this module afterwards
            takes effect.

    Returns:
        The largest row count reached by halving that keeps one score
        matrix below the limit; never less than 1, which is as far as rows
        can be split when the keys alone reach the limit.
    """
    if limit is None:
        limit = ATTENTION_SCORE_ELEMENT_LIMIT
    rows = num_queries
    while rows > 1 and rows * num_keys >= limit:
        rows = (rows + 1) // 2
    return rows


def attention_rows(
    module: torch.nn.Module,
    query: torch.Tensor,
    keys_transposed: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float,
    scaling: float,
    softcap: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend from a range of query rows to all keys, as upstream does.

    This is the upstream eager attention from the score product up to the
    product with the values, in the same order of operations. Each query
    row is computed from that row alone -- the softmax runs over the keys
    -- so the rows of a sequence may be computed in any number of ranges.

    Args:
        module: The attention module; only its training flag is read.
        query: Query states of the rows, shape ``(B, heads, rows, D)``.
        keys_transposed: Key states, shape ``(B, heads, D, keys)``.
        value_states: Value states, shape ``(B, heads, keys, D)``.
        attention_mask: Additive mask for these rows, or ``None``.
        dropout: Dropout probability of the attention weights.
        scaling: Factor applied to the scores.
        softcap: Soft cap applied to the scores, or ``None``.

    Returns:
        Tuple ``(output, weights)`` of shapes ``(B, heads, rows, D)`` and
        ``(B, heads, rows, keys)``.
    """
    attn_weights = torch.matmul(query, keys_transposed) * scaling
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    # The softmax is taken in float32 whatever the precision of the states.
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    return torch.matmul(attn_weights, value_states), attn_weights


def split_eager_attention(
    modeling_module: Any,
    query_rows: Callable[[int, int], int] | None = None,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Build an eager attention that splits the query rows of a long sequence.

    The function built takes the arguments of the upstream eager
    attention of a transformers architecture module and returns the same
    two tensors, so it can be bound in that module in place of the
    upstream one.

    Args:
        modeling_module: The architecture module whose ``repeat_kv`` the
            attention expands the key/value heads with. The function is
            looked up there at every call, so a replacement bound in that
            module -- before or after this factory runs -- takes effect.
        query_rows: Rule giving the number of query rows one range may
            hold, from the number of queries and the number of keys.
            ``None`` uses :func:`attention_query_rows` with the limit this
            module holds at call time.

    Returns:
        The eager attention function, called exactly as the upstream one
        is.
    """

    def eager_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        softcap: float | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute eager attention, splitting the query rows of a long sequence.

        Takes the arguments of the upstream function it replaces and
        returns the same two tensors. When the row rule grants every query
        row to one range, the operations are the upstream ones in the
        upstream order. Otherwise the query rows are cut into ranges of
        the size the rule allows, each range is attended with its own rows
        of the mask, and the results are concatenated back along the
        sequence axis. No query row depends on another, so the two ways
        produce the same numbers.

        The ranges are cut with constant indices rather than with a
        chunking operation: the sequence length is a constant of the
        traced graph, a chunk operation fails to convert, and constant
        slices leave no shape arithmetic behind. The key/value expansion
        is looked up in the architecture module at call time, so its own
        replacement stays in effect.

        Args:
            module: The attention module calling this function.
            query: Query states, shape ``(B, heads, S, D)``.
            key: Key states, shape ``(B, key/value heads, K, D)``.
            value: Value states, shape ``(B, key/value heads, K, D)``.
            attention_mask: Additive mask of shape ``(B, 1, S, >=K)`` (or
                with a single row broadcast over all queries), or ``None``.
            dropout: Dropout probability of the attention weights.
            scaling: Factor applied to the scores; the head dimension's
                inverse square root when ``None``.
            softcap: Soft cap applied to the scores, or ``None``.
            **kwargs: Accepted and ignored, as upstream does; this covers
                the keywords some architectures add, such as
                ``sliding_window``.

        Returns:
            Tuple ``(attn_output, attn_weights)`` of shapes ``(B, S,
            heads, D)`` and ``(B, heads, S, K)``.
        """
        if scaling is None:
            scaling = module.head_dim**-0.5

        key_states = modeling_module.repeat_kv(key, module.num_key_value_groups)
        value_states = modeling_module.repeat_kv(value, module.num_key_value_groups)
        keys_transposed = key_states.transpose(2, 3)

        num_queries = static_size(query, 2)
        num_keys = static_size(key_states, 2)
        if attention_mask is not None:
            # Whatever the length of the mask, only the columns of the keys
            # at hand are used.
            attention_mask = attention_mask[:, :, :, :num_keys]

        if query_rows is None:
            rows = attention_query_rows(num_queries, num_keys)
        else:
            rows = query_rows(num_queries, num_keys)
        if rows >= num_queries:
            # Nothing to split. This path must not go through a split and a
            # concatenation of one piece: besides being pointless, a
            # single-piece split is something the converter does not accept.
            attn_output, attn_weights = attention_rows(
                module,
                query,
                keys_transposed,
                value_states,
                attention_mask,
                dropout,
                scaling,
                softcap,
            )
            return attn_output.transpose(1, 2).contiguous(), attn_weights

        # A mask holding one row for all queries is broadcast, not cut.
        mask_has_query_rows = attention_mask is not None and static_size(attention_mask, 2) != 1
        outputs: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for start in range(0, num_queries, rows):
            # The last range is shorter when the rows do not divide evenly.
            stop = min(start + rows, num_queries)
            mask_rows = attention_mask
            if attention_mask is not None and mask_has_query_rows:
                mask_rows = attention_mask[:, :, start:stop, :]
            range_output, range_weights = attention_rows(
                module,
                query[:, :, start:stop, :],
                keys_transposed,
                value_states,
                mask_rows,
                dropout,
                scaling,
                softcap,
            )
            outputs.append(range_output)
            weights.append(range_weights)
        attn_output = torch.cat(outputs, dim=2)
        attn_weights = torch.cat(weights, dim=2)
        return attn_output.transpose(1, 2).contiguous(), attn_weights

    return eager_attention_forward
