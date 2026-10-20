"""Additive attention masks built in-graph, shared by the encoder compile backends.

Since transformers 5, several encoder architectures build their attention
masks through the framework's shared mask helpers
(``transformers.masking_utils``), and that path stands in the way of a
conversion in two ways:

* it materializes the mask with a ``new_ones`` op, which the Core ML
  converter has no translation for;
* it writes the float32 minimum into the masked-out positions, which is
  not representable in FP16: once the converted graph runs in FP16 it
  becomes -inf.

The same helpers return an attention mask unchanged when it is already
4-D: such a mask is taken as prepared, and nothing is built. The
remedy shared by every backend this affects is therefore to build the
additive 4-D mask in the traced wrapper itself, from the 2-D mask the
compiled graph takes, with a finite fill value and static-shape operations
only, and to hand the model that mask instead of the 2-D one.

This module holds the builders -- :func:`padding_mask` for a stack whose
every layer attends to every real token, :func:`bidirectional_masks` for
one that also has sliding-window layers -- and :class:`AdditiveMaskModel`,
the adapter that feeds the first of them to a model behind the shared
pooling wrappers.

Importing this module pulls in ``torch``; it therefore requires the
``[compile]`` extra and must never be imported from the ``eeane serve``
code path (see :mod:`eeane.compiler`).
"""

from __future__ import annotations

from typing import Any

import torch

from eeane.compiler.backends import decoder_patches

# Public surface of this module.
__all__ = ["MASK_FILL_VALUE", "AdditiveMaskModel", "bidirectional_masks", "padding_mask"]

# Additive value written into the masked-out positions of every mask built
# here; see :data:`eeane.compiler.backends.decoder_patches.MASK_FILL_VALUE`
# for why it is a finite number.
MASK_FILL_VALUE = decoder_patches.MASK_FILL_VALUE


def padding_mask(attention_mask: torch.Tensor, fill_value: float = MASK_FILL_VALUE) -> torch.Tensor:
    """Build the additive 4-D padding mask a BERT-style encoder adds to its scores.

    The mask is ``0.0`` where the key is a real token and ``fill_value``
    where it is padding. Only keys are hidden, so a row with at least one
    real token leaves every query row with at least one visible key.

    It is 4-D because the framework passes a 4-D mask through untouched:
    handed this tensor, the model builds no mask of its own and adds this
    one, broadcast over the heads and the query axis, to its attention
    scores. Only a cast, an indexing broadcast and ``torch.where`` are
    used, with no data-dependent control flow.

    Args:
        attention_mask: 2-D mask of shape ``(B, S)``, integer or float,
            where a non-zero value marks a real token.
        fill_value: Additive value written into masked positions; see
            :data:`MASK_FILL_VALUE` for why it is a finite number.

    Returns:
        Float32 tensor of shape ``(B, 1, 1, S)``.
    """
    device = attention_mask.device
    key_is_real = attention_mask.to(torch.bool)[:, None, None, :]  # (B, 1, 1, S)
    zero = torch.zeros((), dtype=torch.float32, device=device)
    fill = torch.full((), fill_value, dtype=torch.float32, device=device)
    return torch.where(key_is_real, zero, fill)


def bidirectional_masks(
    attention_mask: torch.Tensor, window: int, fill_value: float = MASK_FILL_VALUE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the two additive 4-D attention masks of a bidirectional stack.

    Both masks are ``0.0`` where a query position may attend to a key
    position and ``fill_value`` everywhere else:

    * the *full* mask hides a key only when it is padding;
    * the *sliding* mask additionally hides every key that is ``window``
      or more positions away from the query, in either direction -- the
      bound is exclusive, so a key is visible when
      ``abs(query - key) < window``.

    Handed to the backbone in place of its 2-D mask, one per layer type,
    these replace the framework's own mask construction with these few
    operations. The numbers are the ones the framework's path produces
    (with the finite fill value substituted), which is what keeps the FP32
    baseline -- which does go through the framework's path -- an
    independent check.

    Both conditions are combined as booleans before anything is filled, so
    a key that is hidden twice (padding *and* out of reach) is still
    filled exactly once. Only ``torch.arange`` comparisons, a broadcast
    and ``torch.where`` are used, with no data-dependent control flow: the
    distance half is a constant the tracer folds away, and the padding
    half stays a broadcast of the input mask. Both masks are always built,
    whichever layer types the model at hand happens to hold.

    Args:
        attention_mask: 2-D mask of shape ``(B, S)``, integer or float,
            where a non-zero value marks a real token.
        window: Exclusive half-width of the sliding layers' window. A
            model whose configuration states an inclusive radius ``r`` (a
            key visible when ``abs(query - key) <= r``) is masked with
            ``r + 1``. A window of ``S`` or more hides nothing, and the
            sliding mask then equals the full one.
        fill_value: Additive value written into masked positions; see
            :data:`MASK_FILL_VALUE` for why it is a finite number.

    Returns:
        Tuple ``(full, sliding)`` of float32 tensors, each of shape
        ``(B, 1, S, S)``.

    Raises:
        ValueError: If ``window`` is not a positive integer.
    """
    # bool is a subclass of int, but ``True`` is not a window.
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError(f"window must be a positive integer (got {window!r})")
    device = attention_mask.device
    seq_len = attention_mask.shape[-1]
    positions = torch.arange(seq_len, device=device)
    # distance[i, j] is how far key j is from query i, in either direction.
    distance = (positions.unsqueeze(0) - positions.unsqueeze(-1)).abs()  # (S, S)
    within_window = distance < window  # (S, S)
    # An all-true (S, S) grid, so the full mask gets its query axis by the
    # same broadcast as the sliding one instead of by an expand over a
    # traced size. It is written as a comparison, like the window above,
    # because that is what reaches the Core ML converter as a boolean: a
    # ``ones_like`` of a boolean tensor arrives there as a float, which
    # the logical "and" below does not accept.
    everywhere = distance >= 0
    key_is_real = attention_mask.to(torch.bool)[:, None, None, :]  # (B, 1, 1, S)
    zero = torch.zeros((), dtype=torch.float32, device=device)
    fill = torch.full((), fill_value, dtype=torch.float32, device=device)
    full = torch.where(everywhere[None, None, :, :] & key_is_real, zero, fill)
    sliding = torch.where(within_window[None, None, :, :] & key_is_real, zero, fill)
    return full, sliding


class AdditiveMaskModel(torch.nn.Module):
    """Adapter that hands an encoder its padding mask already built, in 4-D.

    The compiled graph takes a 2-D ``attention_mask``, and handed that,
    the encoder would build its 4-D mask through the framework's mask
    helpers. This adapter builds the mask with :func:`padding_mask`
    instead and hands the encoder the result:

    * the framework passes a 4-D mask through unchanged, so no mask
      construction -- nor the ``new_ones`` op it records -- reaches the
      traced graph;
    * the masked positions hold the finite :data:`MASK_FILL_VALUE` rather
      than the float32 minimum, which would become -inf in FP16.

    The positions it hides are the ones the framework hides, so the
    encoder computes the same function: only the fill value differs, and
    it drives the attention weight of a padded key to zero all the same.

    Placed between the backbone and the pooling wrappers of
    :mod:`eeane.compiler.backends.common`, so those stay
    architecture-independent and keep pooling with the 2-D mask.
    """

    def __init__(self, model: torch.nn.Module, fill_value: float = MASK_FILL_VALUE) -> None:
        """Store the encoder to call and the fill value of its mask.

        Args:
            model: Encoder loaded in eval/FP32 mode with
                ``config.return_dict = False``.
            fill_value: Additive value for masked positions; see
                :data:`MASK_FILL_VALUE`.
        """
        super().__init__()
        self.model = model
        self.fill_value = float(fill_value)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        """Run the encoder with the padding mask built by :func:`padding_mask`.

        Args:
            input_ids: Token ids, shape (B, S).
            attention_mask: Attention mask, shape (B, S); non-zero marks a
                real token.

        Returns:
            The model's own output tuple, unchanged.
        """
        return self.model(
            input_ids=input_ids, attention_mask=padding_mask(attention_mask, self.fill_value)
        )
