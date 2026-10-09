"""Gemma 3 compile backend, for bidirectional embedding models only.

Covers the text model of the Gemma 3 family when it is configured to
attend in both directions and published as a sentence-transformers
embedding model (e.g. ``google/embeddinggemma-300m``). The implementation
is a decoder stack -- grouped attention heads, rotary positions -- but
with bidirectional attention it behaves as an encoder: every position
sees the whole sequence, and the sentence vector is a masked mean over
all of them.

Four properties decide how the conversion is built here:

* **Two attention masks are built in-graph, not one.** The layers of this
  architecture come in two types. A *full* layer lets every position
  attend to every real token; a *sliding* layer additionally limits each
  position to the keys less than a window away from it, on either side.
  The wrapper therefore hands the backbone one additive mask per layer
  type, keyed by that type, and each layer picks its own. Handing over a
  mapping also bypasses the framework's own mask construction, while
  computing the same numbers. A single 4-D mask shared by every layer --
  what a purely causal decoder is masked with -- would silently drop the
  window and compute a different model for any sequence longer than it.
* **Three upstream constructs are replaced** before tracing:
  :func:`patch_rotate_half` removes shape arithmetic the Core ML
  converter cannot fold, :func:`patch_repeat_kv` removes the rank-5
  intermediate that makes the Neural Engine compiler reject the attention
  subgraph, and :func:`patch_attention_query_chunks` computes the
  attention of a long sequence over a few ranges of query rows, because
  the Neural Engine returns wrong numbers for an attention score matrix
  of 2**20 elements or more. All three replacements compute exactly what
  upstream computes; the bodies of the first two are shared with the
  other decoder-style backends through
  :mod:`eeane.compiler.backends.decoder_patches`.
* **The weights of the compiled copy are rescaled into the FP16 range.**
  The upstream model card states that the activations of this model do
  not fit float16, and the compiled program runs in float16: the residual
  stream grows past the largest float16 number in the later layers, and
  the output of the later MLPs is so small that its square falls below
  the smallest normal float16 number. :func:`condition_fp16_ranges`
  divides the residual stream by one power of two and multiplies each MLP
  output by a power of two of its own, by rewriting weights and
  normalization constants only; every normalization layer undoes the
  factor it is fed, so the function the model computes is unchanged. The
  factors are measured on the loaded model by
  :func:`measure_activation_ranges` and recorded in the metadata. The
  FP32 baseline loads a copy of its own and is never rescaled.
* **Only the bidirectional embedding model is compiled.** The same
  architecture configured causally is a language model, with another
  mask and another way of reading a sentence off it, so a directory that
  does not declare bidirectional attention is refused, as is every kind
  but ``embedding`` and every pooling but the mean.

The pooling and the Dense projections applied after it are not part of
the HF configuration; they are declared by the sentence-transformers
modules in the model directory, which this backend reads (and refuses to
guess) through the shared readers in
:mod:`eeane.compiler.backends.common`. A trailing normalization module is
read past: the compiled graph always returns the unnormalized vector and
normalizing is left to the server's own configuration.

Importing this module pulls in ``torch``/``transformers``; it therefore
requires the ``[compile]`` extra and must never be imported from the
``eeane serve`` code path (see :mod:`eeane.compiler`).
"""

from __future__ import annotations

import copy
import gc
import json
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.models.gemma3 import modeling_gemma3

from eeane.compiler.backends import decoder_patches
from eeane.compiler.backends.base import (
    SCORE_SPACE_PROBABILITY,
    LoadedModel,
    PairTemplate,
    SanitySpec,
)
from eeane.compiler.backends.common import (
    POOLING_MEAN,
    SANITY_TEXT_SETS,
    encode_pytorch,
    load_dense,
    mean_pool,
    read_pooling_mode,
    tokenize_batch,
)

# Public surface of this module.
__all__ = [
    "ATTENTION_SCORE_ELEMENT_LIMIT",
    "BIDIRECTIONAL_KEY",
    "CONFIG_FILENAME",
    "EMBEDDING_WRAPPERS",
    "FP16_RANGE_CONDITIONING_KEY",
    "FULL_ATTENTION",
    "MASK_FILL_VALUE",
    "MAX_GAIN_EXPONENT",
    "MAX_POSITION_KEY",
    "MAX_RESIDUAL_DIVISOR_EXPONENT",
    "MLP_OUTPUT_RMS_TARGET",
    "MODEL_TYPE",
    "OUTPUT_NAMES",
    "PATCHES",
    "RESIDUAL_PEAK_TARGET",
    "SANITY_SPECS",
    "SLIDING_ATTENTION",
    "SLIDING_WINDOW_KEY",
    "SUPPORTED_KINDS",
    "ActivationRanges",
    "BidirectionalMeanWrapper",
    "Gemma3Backend",
    "attention_query_rows",
    "build_bidirectional_masks",
    "condition_fp16_ranges",
    "measure_activation_ranges",
    "mlp_output_gain",
    "patch_attention_query_chunks",
    "patch_repeat_kv",
    "patch_rotate_half",
    "residual_divisor",
]

# The only model kind this backend compiles.
KIND_EMBEDDING = "embedding"
SUPPORTED_KINDS: tuple[str, ...] = (KIND_EMBEDDING,)

# Appended to every refusal of a kind, since the kind a user is most
# likely to ask for by mistake is one the architecture name suggests: the
# same stack is also published as a causal language model.
KIND_REFUSAL = (
    "This backend compiles the bidirectional embedding models of the Gemma 3 family only: "
    "a text model that attends in both directions and is pooled into a sentence vector"
)

# Core ML graph output name per kind.
OUTPUT_NAMES: dict[str, str] = {KIND_EMBEDDING: "embedding"}

# Additive value written into the masked-out positions of both attention
# masks; see :data:`eeane.compiler.backends.decoder_patches.MASK_FILL_VALUE`
# for why it is a finite number.
MASK_FILL_VALUE = decoder_patches.MASK_FILL_VALUE

# The two layer types of this architecture, spelled as the configuration
# spells them. They are also the keys of the mask mapping the backbone
# takes: each layer looks its own mask up by its type.
FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"

# Model directory file, and the keys read from it. ``model_type`` is what
# tells the text model apart from the multimodal model and from the other
# generations of the family; the bidirectional flag is what tells an
# embedding model apart from the causal language model of the very same
# architecture. Positions are rotary and index from 0, so the configured
# position budget is the usable length.
CONFIG_FILENAME = "config.json"
MODEL_TYPE_KEY = "model_type"
BIDIRECTIONAL_KEY = "use_bidirectional_attention"
MAX_POSITION_KEY = "max_position_embeddings"

# Attribute of the *loaded* configuration holding the window of the
# sliding layers. It is never read from the file: a bidirectional
# configuration stores one number and, when constructed, replaces it with
# the half-width the masks are actually built from, so the loaded value is
# the only one that is right -- and deriving it once more would halve it
# twice.
SLIDING_WINDOW_KEY = "sliding_window"

# The HF ``model_type`` this backend implements.
MODEL_TYPE = "gemma3_text"

# Short English sentence used as the example input for torch.jit.trace.
TRACE_EXAMPLE_TEXT = "This is a short sample sentence used for conversion."

# Filler row used to pad the last sanity batch when the number of sanity
# inputs is not a multiple of B. It is a real sentence rather than an
# empty string, so the row keeps attendable positions whatever special
# tokens the tokenizer at hand does or does not add.
BATCH_PADDING_TEXT = "This sentence only fills an unused row of the batch."

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

# Metadata key under which the FP16 range conditioning is recorded.
FP16_RANGE_CONDITIONING_KEY = "fp16_range_conditioning"

# Peak magnitude the residual stream is brought near by
# :func:`residual_divisor`. Float16 ends at 65504, so a peak around 2**10
# leaves a factor of about 64 for inputs that drive the stream harder than
# the calibration sentences did, while keeping the stream itself far above
# the float16 resolution.
RESIDUAL_PEAK_TARGET = 1024.0

# Root mean square each MLP output is brought near by
# :func:`mlp_output_gain`. The normalization that reads the MLP output
# squares it first; around 1 the squares sit in the middle of the float16
# range, as far from its underflow as from its overflow.
MLP_OUTPUT_RMS_TARGET = 1.0

# Bounds on the exponents of the two kinds of power-of-two factors. A
# residual divisor below 1 is never used: a stream that already fits
# float16 is left as it is. The upper bounds are far beyond anything a
# working model calls for (they allow factors of 65536); they only keep a
# degenerate measurement from producing a normalization constant that
# float32 itself can no longer hold.
MAX_RESIDUAL_DIVISOR_EXPONENT = 16
MAX_GAIN_EXPONENT = 16

# Attribute of the model object under which the applied conditioning is
# remembered. The conditioning rewrites weights, so applying it a second
# time to the same model would compound it; the record on the model is
# what makes a repeated request return the first result instead.
_CONDITIONING_ATTRIBUTE = "_eeane_fp16_range_conditioning"

# Sanity fixtures per kind, as handed to the pipeline and the self-check:
# the shared per-language sets, unchanged. Embeddings are compared row by
# row against their own baseline and carry no ordering expectation.
SANITY_SPECS: dict[str, SanitySpec] = {KIND_EMBEDDING: SanitySpec(input_sets=SANITY_TEXT_SETS)}


def patch_rotate_half() -> None:
    """Replace Gemma 3's ``rotate_half`` with a static-shape equivalent.

    The upstream implementation splits the last dimension with
    ``x[..., : x.shape[-1] // 2]``. The Python-level division on a traced
    shape records an ``aten::Int`` node in the TorchScript graph, which
    the Core ML converter cannot fold into a static slice. The replacement
    (:func:`eeane.compiler.backends.decoder_patches.rotate_half`) selects
    the same two halves without consulting the shape at graph level; the
    results are identical whenever the head dimension is even, which
    :meth:`Gemma3Backend.apply_patches` enforces.

    Rebinds the module-level function, which the rotary helper looks up at
    call time, so the replacement takes effect for every Gemma 3 model in
    the process -- whichever attention implementation it runs, since the
    rotary helper is shared by all of them. Re-applying it is harmless.
    """
    modeling_gemma3.rotate_half = decoder_patches.rotate_half


def patch_repeat_kv() -> None:
    """Replace the grouped-query key/value expansion with a rank-4 form.

    Upstream expands the stored key/value heads to the query head count
    through a rank-5 intermediate; the Neural Engine compiler rejects that
    subgraph, and rejecting one subgraph makes the whole compiled model
    fall back to the CPU. The replacement
    (:func:`eeane.compiler.backends.decoder_patches.repeat_kv`) stays
    rank-4 and produces the very same head order.

    Rebinds the module-level function, which only the eager attention
    implementation of this architecture calls, so the replacement takes
    effect for every eager Gemma 3 model in the process. Re-applying it is
    harmless.
    """
    modeling_gemma3.repeat_kv = decoder_patches.repeat_kv


def _static_size(tensor: torch.Tensor, dim: int) -> int:
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


def attention_query_rows(num_queries: int, num_keys: int) -> int:
    """Return how many query rows one attention score matrix may hold.

    The row count starts at the whole sequence and is halved (rounding
    up) until rows x keys falls strictly below
    :data:`ATTENTION_SCORE_ELEMENT_LIMIT`. A sequence whose full score
    matrix is already below the limit therefore keeps all of its rows,
    which means its attention is not split at all.

    Args:
        num_queries: Number of query positions.
        num_keys: Number of key positions.

    Returns:
        The largest row count reached by halving that keeps one score
        matrix below the limit; never less than 1, which is as far as rows
        can be split when the keys alone reach the limit.
    """
    rows = num_queries
    while rows > 1 and rows * num_keys >= ATTENTION_SCORE_ELEMENT_LIMIT:
        rows = (rows + 1) // 2
    return rows


def _attention_rows(
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


def _eager_attention_forward(
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

    Takes the arguments of the upstream function it replaces and returns
    the same two tensors. When the score matrix of one head stays below
    :data:`ATTENTION_SCORE_ELEMENT_LIMIT` the operations are the upstream
    ones in the upstream order. Otherwise the query rows are cut into the
    ranges :func:`attention_query_rows` allows, each range is attended
    with its own rows of the mask, and the results are concatenated back
    along the sequence axis. No query row depends on another, so the two
    ways produce the same numbers.

    The ranges are cut with constant indices rather than with a chunking
    operation: the sequence length is a constant of the traced graph, a
    chunk operation fails to convert, and constant slices leave no shape
    arithmetic behind. The key/value expansion is looked up in the
    upstream module at call time, so its own replacement stays in effect.

    Args:
        module: The attention module calling this function.
        query: Query states, shape ``(B, heads, S, D)``.
        key: Key states, shape ``(B, key/value heads, K, D)``.
        value: Value states, shape ``(B, key/value heads, K, D)``.
        attention_mask: Additive mask of shape ``(B, 1, S, >=K)`` (or with
            a single row broadcast over all queries), or ``None``.
        dropout: Dropout probability of the attention weights.
        scaling: Factor applied to the scores; the head dimension's
            inverse square root when ``None``.
        softcap: Soft cap applied to the scores, or ``None``.
        **kwargs: Accepted and ignored, as upstream does.

    Returns:
        Tuple ``(attn_output, attn_weights)`` of shapes ``(B, S, heads,
        D)`` and ``(B, heads, S, K)``.
    """
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = modeling_gemma3.repeat_kv(key, module.num_key_value_groups)
    value_states = modeling_gemma3.repeat_kv(value, module.num_key_value_groups)
    keys_transposed = key_states.transpose(2, 3)

    num_queries = _static_size(query, 2)
    num_keys = _static_size(key_states, 2)
    if attention_mask is not None:
        # Whatever the length of the mask, only the columns of the keys
        # at hand are used.
        attention_mask = attention_mask[:, :, :, :num_keys]

    rows = attention_query_rows(num_queries, num_keys)
    if rows >= num_queries:
        # Nothing to split. This path must not go through a split and a
        # concatenation of one piece: besides being pointless, a
        # single-piece split is something the converter does not accept.
        attn_output, attn_weights = _attention_rows(
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
    mask_has_query_rows = attention_mask is not None and _static_size(attention_mask, 2) != 1
    outputs: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for start in range(0, num_queries, rows):
        # The last range is shorter when the rows do not divide evenly.
        stop = min(start + rows, num_queries)
        mask_rows = attention_mask
        if attention_mask is not None and mask_has_query_rows:
            mask_rows = attention_mask[:, :, start:stop, :]
        range_output, range_weights = _attention_rows(
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


def patch_attention_query_chunks() -> None:
    """Replace the eager attention with one that splits long query axes.

    On the Neural Engine, an attention whose per-head score matrix holds
    :data:`ATTENTION_SCORE_ELEMENT_LIMIT` elements or more returns wrong
    numbers, although the same program is correct on the CPU. The
    replacement computes the query rows of such a sequence in a few
    ranges, each with a score matrix below the limit, and concatenates
    them; every row of an attention is independent of the others, so the
    result is what upstream computes. Below the limit nothing is split
    and the operations are the upstream ones.

    Rebinds the module-level function, which the attention module looks
    up at call time and only uses for the eager implementation, so the
    replacement takes effect for every eager Gemma 3 model in the process.
    Re-applying it is harmless.
    """
    modeling_gemma3.eager_attention_forward = _eager_attention_forward


# The graph rewrites of this architecture, in the order they are applied,
# each under the name it is recorded with in a compiled variant's
# metadata. A rewrite added later -- another function rebinding something
# upstream -- only has to be appended here to be applied and recorded.
PATCHES: tuple[tuple[str, Callable[[], None]], ...] = (
    ("rotate_half_static", patch_rotate_half),
    ("repeat_kv_rank4", patch_repeat_kv),
    ("attention_query_chunks", patch_attention_query_chunks),
)


@dataclass(frozen=True)
class ActivationRanges:
    """What :func:`measure_activation_ranges` observed on one model.

    Attributes:
        residual_peak: Largest absolute value the residual stream took,
            over every layer, input and position; ``nan`` when any of it
            was not finite, ``0.0`` when nothing was measured.
        mlp_output_rms: Root mean square of each layer's MLP output, in
            layer order, over every input, position and dimension; ``0.0``
            for a layer nothing was measured on.
        inputs: Number of inputs the measurement ran over.
    """

    residual_peak: float
    mlp_output_rms: tuple[float, ...]
    inputs: int


def _calibration_texts() -> tuple[str, ...]:
    """Return the fixed sentences the activation ranges are measured on.

    The tracing example plus every language set of the shared sanity
    fixtures: a small fixed collection that covers the scripts the
    compiled model is checked on, so the measurement is reproducible and
    costs a handful of forward passes.
    """
    return (TRACE_EXAMPLE_TEXT, *(text for _, texts in SANITY_TEXT_SETS for text in texts))


def measure_activation_ranges(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int | None = None,
) -> ActivationRanges:
    """Measure the residual-stream peak and the MLP output level of a model.

    Each text is encoded on its own, without padding, and run through the
    model in the precision it was loaded in, with the plain 2-D attention
    mask. Two things are read off the inputs of the normalization layers,
    and nothing about the model is changed:

    * the residual stream is what each layer's two pre-normalizations and
      the final normalization read, so the largest absolute value among
      those inputs is the peak of the stream;
    * each layer's MLP output is what its post-feedforward normalization
      reads, so the mean square of that input gives the level of the MLP
      output.

    Args:
        model: The loaded text model, in eval mode.
        tokenizer: Its tokenizer.
        texts: Sentences to measure on.
        max_length: Length the encodings are truncated to; ``None`` for
            no truncation.

    Returns:
        The measured ranges. A text that encodes to no token at all is
        skipped and not counted.
    """
    layers = list(model.layers)
    residual_peaks: list[float] = []
    square_sums = [0.0] * len(layers)
    element_counts = [0] * len(layers)

    def watch_residual(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        residual_peaks.append(float(args[0].detach().abs().max()))

    def watch_mlp_output(index: int) -> Callable[[torch.nn.Module, tuple[Any, ...]], None]:
        def hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            values = args[0].detach().to(torch.float64)
            square_sums[index] += float(values.pow(2).sum())
            element_counts[index] += values.numel()

        return hook

    handles = []
    inputs = 0
    try:
        for index, layer in enumerate(layers):
            handles.append(layer.input_layernorm.register_forward_pre_hook(watch_residual))
            handles.append(
                layer.pre_feedforward_layernorm.register_forward_pre_hook(watch_residual)
            )
            handles.append(
                layer.post_feedforward_layernorm.register_forward_pre_hook(watch_mlp_output(index))
            )
        handles.append(model.norm.register_forward_pre_hook(watch_residual))
        device = next(model.parameters()).device
        with torch.no_grad():
            for text in texts:
                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=max_length is not None,
                    max_length=max_length,
                )
                input_ids = encoded["input_ids"]
                if input_ids.shape[-1] == 0:
                    continue
                model(
                    input_ids=input_ids.to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                )
                inputs += 1
    finally:
        for handle in handles:
            handle.remove()

    # ``max`` would silently step over a NaN; a stream that is not finite
    # must be reported as such, so that no factor is derived from it.
    if all(math.isfinite(peak) for peak in residual_peaks):
        residual_peak = max(residual_peaks, default=0.0)
    else:
        residual_peak = math.nan
    mlp_output_rms = tuple(
        math.sqrt(square_sum / count) if count else 0.0
        for square_sum, count in zip(square_sums, element_counts, strict=True)
    )
    return ActivationRanges(
        residual_peak=residual_peak, mlp_output_rms=mlp_output_rms, inputs=inputs
    )


def _power_of_two_factor(ratio: float, min_exponent: int, max_exponent: int) -> float:
    """Round ``ratio`` to the nearest power of two within exponent bounds.

    Every factor of the range conditioning is a power of two because
    multiplying or dividing a binary floating-point number by one changes
    its exponent only: neither float32 nor float16 rounds the mantissa,
    so the rescaled weights carry exactly the precision of the original
    ones.

    Args:
        ratio: The factor that would be exact for the measured value.
        min_exponent: Smallest exponent allowed.
        max_exponent: Largest exponent allowed.

    Returns:
        ``2.0 ** e`` with ``e`` the rounded binary logarithm of ``ratio``
        clamped to the bounds; ``1.0`` -- no rescaling -- when ``ratio``
        is not a positive finite number, since nothing can be derived from
        a measurement that is zero, negative, infinite or NaN.
    """
    if not math.isfinite(ratio) or ratio <= 0.0:
        return 1.0
    exponent = round(math.log2(ratio))
    return 2.0 ** max(min_exponent, min(max_exponent, exponent))


def residual_divisor(residual_peak: float) -> float:
    """Return the power of two the residual stream is divided by.

    The divisor brings the measured peak of the stream to about
    :data:`RESIDUAL_PEAK_TARGET`. It is never below 1: a stream that is
    already at or under the target is left alone.

    Args:
        residual_peak: Measured peak magnitude of the residual stream.

    Returns:
        The divisor, a power of two of at least 1; ``1.0`` when the peak
        is not a positive finite number.
    """
    if not math.isfinite(residual_peak) or residual_peak <= 0.0:
        return 1.0
    return _power_of_two_factor(
        residual_peak / RESIDUAL_PEAK_TARGET, 0, MAX_RESIDUAL_DIVISOR_EXPONENT
    )


def mlp_output_gain(mlp_output_rms: float) -> float:
    """Return the power of two one layer's MLP output is multiplied by.

    The gain brings the measured root mean square of the MLP output to
    about :data:`MLP_OUTPUT_RMS_TARGET`.

    Args:
        mlp_output_rms: Measured root mean square of the layer's MLP
            output.

    Returns:
        The gain, a power of two; ``1.0`` when the measurement is not a
        positive finite number.
    """
    if not math.isfinite(mlp_output_rms) or mlp_output_rms <= 0.0:
        return 1.0
    return _power_of_two_factor(
        MLP_OUTPUT_RMS_TARGET / mlp_output_rms, -MAX_GAIN_EXPONENT, MAX_GAIN_EXPONENT
    )


def _check_power_of_two(name: str, value: float) -> None:
    """Refuse a conditioning factor that is not a positive power of two.

    Raises:
        ValueError: If ``value`` is not finite, not positive, or has a
            mantissa other than that of a power of two.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0.0
        or math.frexp(value)[0] != 0.5
    ):
        raise ValueError(f"{name} must be a positive power of two (got {value!r})")


def condition_fp16_ranges(model: torch.nn.Module, divisor: float, gains: Sequence[float]) -> None:
    """Rescale a model's activations into the float16 range, in place.

    Only weights, one buffer and the ``eps`` attributes of normalization
    layers are rewritten; no forward code changes, and in float32 the
    model computes the same function as before. Two rescalings are
    applied, each undone by the normalization layers that read what it
    scaled:

    * **The residual stream is divided by ``divisor``.** Everything
      written to the stream is divided: the embedding, through its scale
      buffer, and the two branch outputs of every layer, through the
      weights of the post-attention and post-feedforward normalizations
      (upstream multiplies their output by ``1 + weight``, so storing
      ``(1 + weight) / divisor - 1`` divides it). Everything read from
      the stream goes through a normalization computing
      ``x / sqrt(mean(x**2) + eps)``; dividing its ``eps`` by
      ``divisor**2`` makes that expression return for ``x / divisor``
      exactly what it returned for ``x``.
    * **Each MLP output is multiplied by that layer's gain.** The output
      projection of the MLP is multiplied by the gain, and the ``eps`` of
      the post-feedforward normalization reading it by the gain squared,
      which by the same identity leaves that normalization's output
      unchanged.

    The first keeps the stream below the float16 maximum; the second
    lifts the squares the post-feedforward normalization computes out of
    the float16 underflow range. Both factors must be powers of two, so
    that rescaling a weight changes its exponent and nothing else.

    The function is not idempotent: applying it twice rescales twice.
    :meth:`Gemma3Backend.apply_patches` is what applies it at most once
    per model.

    Args:
        model: The loaded text model to rewrite.
        divisor: Power of two the residual stream is divided by; ``1.0``
            leaves it alone.
        gains: Power of two each layer's MLP output is multiplied by, in
            layer order; ``1.0`` leaves a layer alone.

    Raises:
        ValueError: If a factor is not a positive power of two, or if
            ``gains`` does not hold one factor per layer. Nothing is
            rewritten in either case.
    """
    layers = list(model.layers)
    if len(gains) != len(layers):
        raise ValueError(
            f"expected one MLP output gain per layer ({len(layers)}), got {len(gains)}"
        )
    _check_power_of_two("the residual divisor", divisor)
    for gain in gains:
        _check_power_of_two("an MLP output gain", gain)

    with torch.no_grad():
        # A factor of exactly 1 is skipped rather than applied: rewriting
        # a weight as ``(1 + w) / 1 - 1`` would round it for nothing.
        if divisor != 1.0:
            model.embed_tokens.embed_scale.div_(divisor)
            model.norm.eps = model.norm.eps / divisor**2
            for layer in layers:
                for norm in (layer.post_attention_layernorm, layer.post_feedforward_layernorm):
                    norm.weight.copy_((1.0 + norm.weight) / divisor - 1.0)
                for norm in (layer.input_layernorm, layer.pre_feedforward_layernorm):
                    norm.eps = norm.eps / divisor**2
        for layer, gain in zip(layers, gains, strict=True):
            if gain == 1.0:
                continue
            down_proj = layer.mlp.down_proj
            down_proj.weight.mul_(gain)
            if down_proj.bias is not None:
                down_proj.bias.mul_(gain)
            norm = layer.post_feedforward_layernorm
            norm.eps = norm.eps * gain**2


def _condition_loaded_model(loaded: LoadedModel) -> dict[str, Any]:
    """Measure, derive and apply the range conditioning of a loaded model.

    The record of what was applied is kept on the model object itself.
    A model that already carries one is not measured or rewritten again;
    its record is returned instead, so asking any number of times -- once
    per compile run or once per variant -- rescales the weights once.

    Args:
        loaded: Handle whose model is rewritten in place.

    Returns:
        A JSON-serializable record: the residual divisor, the MLP output
        gain of every layer and the number of calibration inputs.
    """
    model = loaded.model
    record = getattr(model, _CONDITIONING_ATTRIBUTE, None)
    if record is None:
        max_length = getattr(loaded.config, MAX_POSITION_KEY, None)
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            max_length = None
        ranges = measure_activation_ranges(
            model, loaded.tokenizer, _calibration_texts(), max_length=max_length
        )
        divisor = residual_divisor(ranges.residual_peak)
        gains = [mlp_output_gain(rms) for rms in ranges.mlp_output_rms]
        condition_fp16_ranges(model, divisor, gains)
        record = {
            "residual_divisor": divisor,
            "mlp_output_gains": gains,
            "calibration_inputs": ranges.inputs,
        }
        setattr(model, _CONDITIONING_ATTRIBUTE, record)
    # A copy, so that a caller editing its metadata cannot alter what the
    # model remembers.
    return copy.deepcopy(record)


def build_bidirectional_masks(
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

    Handing these to the backbone as a mapping from layer type to mask
    replaces the framework's own mask construction with these few
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
        window: Exclusive half-width of the sliding layers' window, as
            held by the *loaded* configuration (which has already derived
            it from the stored value). A window of ``S`` or more hides
            nothing, and the sliding mask then equals the full one.
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


class BidirectionalMeanWrapper(torch.nn.Module):
    """Wraps a bidirectional backbone, masks it in-graph and mean-pools it.

    The output matches a sentence-transformers model whose modules are a
    Transformer, mean Pooling and the declared Dense projections, without
    normalization: the compiled graph always returns the raw vector, and
    normalizing is left to the server's own configuration.

    Both attention masks and the pooling are computed inside the wrapper,
    so the exported program takes exactly the two int32 tensors Core ML
    feeds it and leaves no pre- or post-processing to reimplement on the
    host.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        window: int,
        dense: torch.nn.Module | None = None,
        fill_value: float = MASK_FILL_VALUE,
    ) -> None:
        """Store the backbone, the window, the projection and the fill value.

        Args:
            model: Backbone returning the last hidden state first, loaded
                in eval/FP32 mode.
            window: Exclusive half-width of the sliding layers' window,
                as held by the loaded configuration; see
                :func:`build_bidirectional_masks`.
            dense: Projection applied to the pooled vector, as built by
                ``build_dense``; ``None`` for a model that declares none.
            fill_value: Additive value for masked positions; see
                :data:`MASK_FILL_VALUE`.
        """
        super().__init__()
        self.model = model
        self.window = window
        self.dense = dense
        self.fill_value = fill_value

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute mean-pooled (and, if declared, projected) embeddings.

        Args:
            input_ids: Token ids, shape (B, S).
            attention_mask: Attention mask, shape (B, S); non-zero marks a
                real token.

        Returns:
            Embeddings of shape (B, hidden_size), or (B, projected width)
            when a projection was given. They are **not** L2-normalized.
        """
        full, sliding = build_bidirectional_masks(attention_mask, self.window, self.fill_value)
        outputs = self.model(
            input_ids=input_ids,
            # A mapping, not a tensor: the backbone then builds no mask of
            # its own, and each layer takes the one matching its type.
            attention_mask={FULL_ATTENTION: full, SLIDING_ATTENTION: sliding},
            # Stated here rather than left to the configuration: a cache
            # would end up in the traced graph as state a single-pass
            # encoder has no use for, and tracing needs tuple outputs.
            use_cache=False,
            return_dict=False,
        )
        hidden = outputs[0]  # (B, S, H)
        pooled = mean_pool(hidden, attention_mask)  # (B, H)
        if self.dense is None:
            return pooled
        return self.dense(pooled)


# Traceable wrapper per detected pooling mode. The published models of
# this family are trained with a mean over all real tokens; reading one
# position instead would return a vector the model was never trained to
# put a sentence in, so every other mode is refused rather than served.
EMBEDDING_WRAPPERS: dict[str, type[torch.nn.Module]] = {POOLING_MEAN: BidirectionalMeanWrapper}


class Gemma3Backend:
    """Compile backend for the bidirectional embedding models of Gemma 3.

    Implements the backend interface declared in
    :mod:`eeane.compiler.backends.base`, which documents what each member
    is for, in which order the pipeline calls them, and the rules an
    implementation must follow. Every method is stateless: all per-model
    state travels in the :class:`~eeane.compiler.backends.base.LoadedModel`
    handle, so one instance can serve several compile runs.

    ``reranker`` is not among :attr:`supported_kinds`: every kind-taking
    member refuses it with the reason spelled out in :data:`KIND_REFUSAL`.
    """

    name = "Gemma3Text"
    supported_kinds: tuple[str, ...] = SUPPORTED_KINDS

    def load(self, model_dir: Path, kind: str, attn: str = "eager") -> LoadedModel:
        """Load the FP32 model and its tokenizer from a HF model directory.

        Args:
            model_dir: Local HuggingFace-format model directory. It is
                only ever read from.
            kind: Must be ``"embedding"``.
            attn: Attention implementation to request. ``"eager"`` is the
                path the conversion traces; ``"sdpa"`` reaches a separate
                implementation, which is what makes it the FP32 reference
                path.

        Returns:
            A handle holding the model in eval/FP32 mode with
            ``config.return_dict = False`` (``torch.jit.trace`` needs tuple
            outputs) and ``config.use_cache = False``, its tokenizer, its
            configuration, the declared pooling and the declared Dense
            projection.

        Raises:
            ValueError: If ``kind`` is not supported by this backend, if
                the directory declares another architecture or a model
                that does not attend in both directions, if its pooling
                cannot be determined, or if its declared module chain
                cannot be reproduced.
        """
        self._check_kind(kind)
        # Everything that can make a model uncompilable is decided from
        # the declarations first: loading the FP32 parameters of a model
        # that is then refused is pure waste. The configuration is checked
        # before the rest because it is the more fundamental refusal.
        self._check_config(model_dir)
        pooling = read_pooling_mode(model_dir)
        dense, dense_config = load_dense(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir, attn_implementation=attn, dtype=torch.float32)
        model.config.return_dict = False
        # Without this the backbone builds and updates a key/value cache,
        # which a fixed-shape single-pass encoder has no use for.
        model.config.use_cache = False
        return LoadedModel(
            model=model.eval(),
            tokenizer=tokenizer,
            config=model.config,
            model_dir=model_dir,
            kind=kind,
            attn=attn,
            pooling=pooling,
            dense=dense,
            dense_config=dense_config,
        )

    def apply_patches(
        self, loaded: LoadedModel, mask_fill_value: float | None = None
    ) -> dict[str, Any]:
        """Apply the mandatory Gemma 3 graph patches and range conditioning.

        Every rewrite in :data:`PATCHES` is a constituent of the
        conversion rather than an optional tweak -- without them the model
        fails to convert, converts into a program that runs on the CPU,
        or computes wrong numbers on the Neural Engine -- so all of them
        are always applied. They patch global ``transformers`` symbols and
        therefore affect every Gemma 3 instance in the process; all are
        semantically equivalent to upstream, and re-applying them is
        harmless.

        After them, the activations of ``loaded.model`` are rescaled into
        the float16 range (:func:`condition_fp16_ranges`), with factors
        measured on that model (:func:`measure_activation_ranges`). This
        rewrites the weights of that one model object, not a global
        symbol: a copy loaded separately, such as the FP32 baseline's, is
        unaffected. The rescaled model computes the same function, and a
        model is rescaled once however often this method is called on it.

        Args:
            loaded: Handle returned by :meth:`load`.
            mask_fill_value: Optional finite attention-mask fill value.
                This architecture needs no mask patch at all, because its
                wrapper builds the masks itself with
                :data:`MASK_FILL_VALUE` (already a finite value); the
                argument is therefore only accepted when it names that
                very value.

        Returns:
            The applied rewrites, the mask fill value the traced graph
            uses and, under :data:`FP16_RANGE_CONDITIONING_KEY`, the
            factors of the range conditioning; recorded verbatim in the
            compiled variant's metadata.

        Raises:
            ValueError: If the rotary head dimension is odd (which would
                make the ``chunk``-based ``rotate_half`` inexact), or if a
                mask fill value other than :data:`MASK_FILL_VALUE` is
                requested. Nothing is rebound or rescaled in either case.
        """
        head_dim = _rope_head_dim(loaded.config)
        if head_dim % 2 != 0:
            raise ValueError(
                f"odd RoPE head dim ({head_dim}) is incompatible with patch_rotate_half"
            )
        if mask_fill_value is not None and mask_fill_value != MASK_FILL_VALUE:
            raise ValueError(
                f"{self.name} builds its attention masks inside the traced graph with a "
                f"mask fill value of {MASK_FILL_VALUE}; {mask_fill_value} cannot be applied"
            )
        applied: dict[str, Any] = {}
        for patch_name, patch in PATCHES:
            patch()
            applied[patch_name] = True
        applied["mask_fill_value"] = MASK_FILL_VALUE
        applied[FP16_RANGE_CONDITIONING_KEY] = _condition_loaded_model(loaded)
        return applied

    def wrap(self, loaded: LoadedModel) -> torch.nn.Module:
        """Wrap the loaded model into the traceable module for its kind.

        Args:
            loaded: Handle returned by :meth:`load`; its ``pooling``
                selects the wrapper, its configuration provides the
                window of the sliding layers and its ``dense`` is applied
                after the pooling.

        Returns:
            The wrapper matching ``loaded.pooling``, in eval mode.

        Raises:
            ValueError: If ``loaded.kind`` is not supported by this
                backend, if the handle carries a pooling mode no wrapper
                implements, or if its configuration holds no usable
                window.
        """
        self._check_kind(loaded.kind)
        wrapper_class = self._wrapper_class(loaded.pooling)
        window = _sliding_window(loaded.config)
        return wrapper_class(loaded.model, window=window, dense=loaded.dense).eval()

    def output_name(self, kind: str) -> str:
        """Return the Core ML graph output name used for ``kind``.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return OUTPUT_NAMES[kind]

    def max_seq_len(self, model_dir: Path) -> int | None:
        """Return the effective maximum sequence length of ``model_dir``.

        Positions are rotary and indexed from 0 with no reserved offset,
        so the configured position budget is the usable sequence length.
        The window of the sliding layers is not a length limit: it bounds
        what one such layer sees, not how long a sequence may be. Only
        ``config.json`` is read; no weights are loaded.

        Args:
            model_dir: Local HuggingFace-format model directory.

        Returns:
            The configured positive position budget, or ``None`` when the
            file is absent/unreadable/unparsable or the value is missing
            or not a positive integer. ``None`` means "unknown", and the
            caller then imposes no limit.
        """
        config = _read_config(model_dir)
        if config is None:
            # A malformed config is reported by the dispatch step; an
            # optional bucket check must not turn it into a second error.
            return None
        value = config.get(MAX_POSITION_KEY)
        # bool is a subclass of int, but a JSON ``true`` is not a length.
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        return value

    def pair_template(self, model_dir: Path, kind: str) -> PairTemplate | None:
        """Return how this model wants a (query, document) pair spelled out.

        Args:
            model_dir: Local HuggingFace-format model directory.
            kind: Model kind the template is asked for.

        Returns:
            Always ``None``: this backend compiles embedding models only,
            and an embedding model encodes a single text and shapes
            nothing.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return None

    def reranker_score_space(self) -> str:
        """Return the space this backend's reranker scores are faithful in.

        Returns:
            :data:`~eeane.compiler.backends.base.SCORE_SPACE_PROBABILITY`.
            This backend compiles no reranker, so the self-check never
            asks; the answer is the one the other embedding-only backend
            gives, which keeps the interface total.
        """
        return SCORE_SPACE_PROBABILITY

    def trace_example(self, kind: str) -> Any:
        """Return the fixed raw example input used for ``torch.jit.trace``.

        Args:
            kind: Model kind.

        Returns:
            A sentence. The caller replicates it to B rows before tracing
            so the traced graph already carries the target batch size.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return TRACE_EXAMPLE_TEXT

    def sanity_spec(self, kind: str) -> SanitySpec:
        """Return the fixed sanity-check inputs and metadata for ``kind``.

        Returns:
            The immutable specification for ``kind``: sentences compared
            row by row against their own baseline, with no expected
            ordering.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return SANITY_SPECS[kind]

    def padding_input(self, kind: str) -> Any:
        """Return the filler input used to pad a partial batch.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return BATCH_PADDING_TEXT

    def tokenize(
        self, loaded: LoadedModel, inputs: list[Any], seq_len: int
    ) -> dict[str, np.ndarray]:
        """Tokenize raw inputs into fixed-shape int32 Core ML arrays.

        The begin- and end-of-sequence tokens this family expects around a
        text are added by the tokenizer's own template, so nothing is
        added here.

        Args:
            loaded: Handle returned by :meth:`load`; its tokenizer encodes
                the inputs.
            inputs: Sentences to encode.
            seq_len: Fixed sequence length S.

        Returns:
            Dict with ``input_ids`` and ``attention_mask`` of shape
            ``(len(inputs), seq_len)`` and dtype ``np.int32``.

        Raises:
            ValueError: If ``loaded.kind`` is unsupported, ``inputs`` is
                empty, or ``seq_len`` is not positive.
        """
        self._check_kind(loaded.kind)
        if not inputs:
            raise ValueError("no inputs to tokenize")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be a positive integer (got {seq_len})")
        return tokenize_batch(loaded.tokenizer, list(inputs), seq_len)

    def reference_outputs(
        self, model_dir: Path, kind: str, inputs: list[Any], seq_len: int
    ) -> np.ndarray:
        """Compute the FP32 (sdpa) reference outputs for ``inputs``.

        Loads a second copy of the model with the ``sdpa`` attention path
        -- a separate implementation from the eager one the conversion
        traces, with a key/value expansion of its own -- runs it row by
        row and releases it again.

        The baseline also hands the model the plain 2-D attention mask, so
        both attention masks are built by the framework rather than by the
        wrapper under test: the two sides of the self-check compute the
        same function through two independent mask implementations.

        Args:
            model_dir: Local HuggingFace-format model directory.
            kind: Model kind.
            inputs: Sentences to encode.
            seq_len: Fixed sequence length S.

        Returns:
            Pooled embeddings of shape (N, width), dtype float32.

        Raises:
            ValueError: If ``kind`` is unsupported, ``inputs`` is empty,
                or the directory is one :meth:`load` or :meth:`wrap`
                refuses.
        """
        self._check_kind(kind)
        if not inputs:
            raise ValueError("no inputs to encode")
        loaded = self.load(model_dir, kind, attn="sdpa")
        try:
            # Refused exactly where the traced side refuses it, so the
            # baseline can never pool a way the wrapper could not.
            self._wrapper_class(loaded.pooling)
            # The baseline must pool and project exactly like the traced
            # wrapper, so it follows both declarations of this directory.
            return encode_pytorch(
                loaded.model,
                loaded.tokenizer,
                list(inputs),
                seq_len,
                pooling=loaded.pooling,
                dense=loaded.dense,
            )
        finally:
            del loaded
            gc.collect()

    def _wrapper_class(self, pooling: str | None) -> type[torch.nn.Module]:
        """Return the wrapper implementing a declared pooling mode.

        Args:
            pooling: Pooling mode carried by a handle.

        Returns:
            The wrapper class registered for ``pooling``.

        Raises:
            ValueError: If no wrapper implements ``pooling``.
        """
        wrapper_class = EMBEDDING_WRAPPERS.get(pooling or "")
        if wrapper_class is None:
            supported = ", ".join(EMBEDDING_WRAPPERS)
            raise ValueError(
                f"unsupported pooling '{pooling}' for {self.name} (supported: {supported})"
            )
        return wrapper_class

    def _check_kind(self, kind: str) -> None:
        """Validate a model kind against :data:`SUPPORTED_KINDS`.

        The refusal carries :data:`KIND_REFUSAL`, so that a user who asks
        for anything else learns what this backend compiles rather than
        only that some kind was rejected.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        if kind in SUPPORTED_KINDS:
            return
        supported = ", ".join(SUPPORTED_KINDS)
        raise ValueError(
            f"unsupported kind '{kind}' for {self.name} (supported: {supported}). {KIND_REFUSAL}"
        )

    def _check_config(self, model_dir: Path) -> None:
        """Validate that ``model_dir`` declares the model this backend implements.

        Two declarations are checked, the more fundamental one first:

        * ``model_type`` -- the only field that names the implementation.
          The architecture-name prefix this backend is selected by cannot
          stand in for it, since a configuration may name any class.
        * the bidirectional flag -- the same architecture without it is a
          causal language model, which needs another mask and is not
          pooled the way an embedding model is. Only a literal ``true``
          counts: anything else is a declaration this backend cannot
          claim to understand.

        Only ``config.json`` is read; no weights are loaded.

        Args:
            model_dir: Local HuggingFace-format model directory.

        Raises:
            ValueError: If the configuration cannot be read, declares
                another ``model_type``, or does not declare bidirectional
                attention.
        """
        config = _read_config(model_dir)
        config_path = model_dir / CONFIG_FILENAME
        declared = config.get(MODEL_TYPE_KEY) if config is not None else None
        if declared != MODEL_TYPE:
            raise ValueError(
                f"'{config_path}' declares {MODEL_TYPE_KEY}={declared!r}, but the {self.name} "
                f"backend implements the '{MODEL_TYPE}' architecture only; a related "
                "architecture of the same family needs a backend of its own"
            )
        bidirectional = config.get(BIDIRECTIONAL_KEY) if config is not None else None
        if bidirectional is not True:
            raise ValueError(
                f"'{config_path}' declares {BIDIRECTIONAL_KEY}={bidirectional!r}, so this is "
                f"not a model that attends in both directions. {KIND_REFUSAL}; the same "
                "architecture with causal attention is a language model, which is masked and "
                "read differently"
            )


def _read_config(model_dir: Path) -> dict[str, Any] | None:
    """Read a model directory's ``config.json`` without loading any weights.

    Args:
        model_dir: Local HuggingFace-format model directory.

    Returns:
        The parsed configuration, or ``None`` when the file is absent,
        unreadable, unparsable or not a JSON object.
    """
    try:
        raw = (model_dir / CONFIG_FILENAME).read_text(encoding="utf-8")
        config = json.loads(raw)
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def _rope_head_dim(config: Any) -> int:
    """Return the width one attention head applies the rotary embedding to.

    This family declares the head width explicitly, and that declaration
    is not ``hidden_size / num_attention_heads`` in general. The declared
    value therefore wins, and the split is only the fallback for a
    configuration that leaves it out.

    Args:
        config: Model configuration of the loaded model.

    Returns:
        The rotary head dimension.
    """
    declared = getattr(config, "head_dim", None)
    if isinstance(declared, int) and not isinstance(declared, bool):
        return declared
    return config.hidden_size // config.num_attention_heads


def _sliding_window(config: Any) -> int:
    """Return the window of the sliding layers, as the loaded model sees it.

    The value is taken from the loaded configuration as it stands. That
    configuration has already turned the stored window into the exclusive
    half-width its own masks are built from, so this is the number the
    in-graph mask must use; nothing is derived from it here.

    Args:
        config: Model configuration of the loaded model.

    Returns:
        The exclusive half-width of the window.

    Raises:
        ValueError: If the configuration holds no positive integer window.
            Without it the sliding layers cannot be masked, and assuming
            one would compute a different model.
    """
    window = getattr(config, SLIDING_WINDOW_KEY, None)
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError(
            f"the loaded configuration holds {SLIDING_WINDOW_KEY}={window!r}, but the sliding "
            "attention layers of this architecture can only be masked with a positive integer "
            "window"
        )
    return window
