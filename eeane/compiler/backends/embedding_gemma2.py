"""EmbeddingGemma 2 compile backend, for text embeddings only.

Covers the text model of EmbeddingGemma 2 (e.g. ``google/embeddinggemma-2``),
a sentence-transformers embedding model published as a multimodal root
model -- a text model plus optional image and audio towers -- whose text
model is a bidirectional encoder stack: grouped attention heads, rotary
positions, and a per-layer input signal projected from the token
embeddings and mixed into every layer. Every position sees the whole
sequence (or a window around itself), each token is projected to the
embedding width, and the sentence vector is a masked mean over them.

Six properties decide how the conversion is built here:

* **Only the text model is read out of the published directory.** The
  directory declares the multimodal root model; it is loaded without its
  image and audio towers, whose weights are then skipped, and the text
  model it holds is what is compiled. A directory holding a text model
  saved on its own declares another model type and is refused, so that
  what is compiled is always the published model's own text path.
* **Two attention masks are built in-graph, not one.** The layers of this
  architecture come in two types: a *full* layer lets every position
  attend to every real token, a *sliding* layer additionally limits each
  position to the keys within a radius of it, on either side. The wrapper
  hands the backbone one additive mask per layer type, keyed by that
  type, which also bypasses the framework's own mask construction while
  computing the same numbers. The configuration states the radius
  inclusively, so the exclusive half-width the shared mask builder takes
  is the radius plus one.
* **Three upstream constructs are replaced** before tracing:
  :func:`patch_rotate_half` removes shape arithmetic the Core ML
  converter cannot fold, :func:`patch_repeat_kv` removes the rank-5
  intermediate that makes the Neural Engine compiler reject the attention
  subgraph, and :func:`patch_attention_query_chunks` computes the
  attention of a long sequence over a few ranges of query rows, because
  the Neural Engine returns wrong numbers for an attention score matrix
  of 2**20 elements or more. All three compute exactly what upstream
  computes; their bodies are shared with the other decoder-style backends
  through :mod:`eeane.compiler.backends.decoder_patches` and
  :mod:`eeane.compiler.backends.attention_split`. Each replacement is
  checked to be in place afterwards.
* **The weights of the compiled copy are rescaled so that every RMS
  normalization squares numbers float16 can hold.** Several activations
  this model normalizes -- the residual stream, the query, key and value
  projections, the output of the per-layer input block -- reach hundreds
  or thousands, where the squares a normalization takes no longer fit
  float16 (a single one passes its largest number beyond 256).
  :func:`condition_fp16_ranges` divides them, and lifts the attention and
  MLP outputs whose squares would underflow, by powers of two written
  into weights and normalization constants only; every normalization
  undoes the factor it is fed, so the function the model computes is
  unchanged. The factors are measured on the loaded model by
  :func:`measure_activation_ranges`, derived by
  :func:`derive_range_factors` and recorded in the metadata. The FP32
  baseline loads a copy of its own and is never rescaled.
* **The sequence lengths the Neural Engine gets wrong are computed at a
  longer length.** A graph of a length in :data:`NE_UNSTABLE_LENGTHS`
  appends masked padding up to :data:`NE_PADDED_LENGTH` inside the graph,
  runs the backbone there and cuts the result back, which computes the
  same function.
* **The mean divides before it sums, and only the mean is compiled.** The
  token vectors of this model are large enough that their plain sum over
  a long sequence approaches the float16 maximum, so
  :func:`scaled_mean_pool` weights every token by its share first. Every
  kind but ``embedding`` and every pooling but the mean is refused.

The pooling and any Dense projection applied after it are not part of
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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.models.embedding_gemma2 import modeling_embedding_gemma2

from eeane.compiler.backends import attention_split, decoder_patches
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
    read_pooling_mode,
    tokenize_batch,
)
from eeane.compiler.backends.fp16_ranges import (
    FP16_RANGE_CONDITIONING_KEY,
    NORM_INPUT_PEAK_TARGET,
    ActivationRanges,
    check_power_of_two,
    mlp_output_gain,
    norm_input_divisor,
)
from eeane.compiler.backends.masks import bidirectional_masks as build_bidirectional_masks

# Public surface of this module.
__all__ = [
    "ATTENTION_OUTPUT_RMS_FLOOR",
    "CONFIG_FILENAME",
    "EMBEDDING_WRAPPERS",
    "FP16_RANGE_CONDITIONING_KEY",
    "FULL_ATTENTION",
    "MASK_FILL_VALUE",
    "MAX_POSITION_KEY",
    "MODEL_TYPE",
    "NE_PADDED_LENGTH",
    "NE_UNSTABLE_LENGTHS",
    "NORM_INPUT_PEAK_TARGET",
    "OUTPUT_NAMES",
    "PATCHES",
    "SANITY_SPECS",
    "SLIDING_ATTENTION",
    "SLIDING_WINDOW_KEY",
    "SUPPORTED_KINDS",
    "TEXT_CONFIG_KEY",
    "TEXT_MODEL_TYPE",
    "EmbeddingGemma2Backend",
    "EmbeddingGemma2MeanWrapper",
    "EmbeddingGemma2Ranges",
    "RangeFactors",
    "build_bidirectional_masks",
    "condition_fp16_ranges",
    "derive_range_factors",
    "measure_activation_ranges",
    "patch_attention_query_chunks",
    "patch_repeat_kv",
    "patch_rotate_half",
    "scaled_mean_pool",
]

# The only model kind this backend compiles.
KIND_EMBEDDING = "embedding"
SUPPORTED_KINDS: tuple[str, ...] = (KIND_EMBEDDING,)

# Appended to every refusal of a kind. The published model also embeds
# images and audio into the same space, so the kind a user is most likely
# to ask for by mistake is one of those; this backend compiles the text
# path alone.
KIND_REFUSAL = (
    "This backend compiles the text embeddings of EmbeddingGemma 2 only: a text is encoded "
    "and pooled into a sentence vector, and image or audio inputs are not handled"
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
# tells the published multimodal root apart from its text model saved on
# its own and from the other generations of the family. The text model's
# own settings, its position budget among them, live under
# ``text_config``. Positions are rotary and index from 0, so the
# configured position budget is the usable length.
CONFIG_FILENAME = "config.json"
MODEL_TYPE_KEY = "model_type"
TEXT_CONFIG_KEY = "text_config"
MAX_POSITION_KEY = "max_position_embeddings"

# The HF ``model_type`` of the published root this backend reads, and the
# one of the text model it compiles out of it.
MODEL_TYPE = "embedding_gemma2"
TEXT_MODEL_TYPE = "embedding_gemma2_text"

# Attribute of the *loaded* text configuration holding the window of the
# sliding layers: an inclusive radius (a key is visible when
# ``abs(query - key) <= radius``), stored and loaded as it is.
SLIDING_WINDOW_KEY = "sliding_window"

# Attribute of the loaded text configuration holding the padding id, which
# the extension of a band length writes into the appended positions.
PAD_TOKEN_KEY = "pad_token_id"

# Short English sentence used as the example input for torch.jit.trace.
TRACE_EXAMPLE_TEXT = "This is a short sample sentence used for conversion."

# Filler row used to pad the last sanity batch when the number of sanity
# inputs is not a multiple of B. It is a real sentence rather than an
# empty string, so the row keeps attendable positions whatever special
# tokens the tokenizer at hand does or does not add.
BATCH_PADDING_TEXT = "This sentence only fills an unused row of the batch."

# Sequence lengths the wrapper computes at :data:`NE_PADDED_LENGTH`
# instead. Run on the Neural Engine at a length of 510 to 512 (batch size
# 1), this model returns wrong numbers -- a cosine of 0.93 to 0.96 to the
# correct result, and NaN with some orderings of its operations --
# whatever the value ranges, the order of operations or the deployment
# target the program was converted with, while the very same program is
# correct on the CPU. A length of 500, and every length from 513 to 768,
# is correct. The lengths from 501 to 509 were not measured, so the band
# starts right after the last length known to be correct.
NE_UNSTABLE_LENGTHS: range = range(501, 513)

# Length a band length is extended to: above the band, and the multiple of
# 8 closest to 512 beyond it, which was measured to be computed correctly.
# Extending computes the same function -- a masked position contributes
# nothing to any attention, and every other operation works on each
# token on its own -- at the price of a somewhat slower bucket of 512.
NE_PADDED_LENGTH = 520

# Root mean square below which an attention output is lifted towards 1, a
# rule of thumb for where the squares the normalization reading it takes
# approach the smallest normal float16 number (2**-14, about 6e-5): at a
# level of 0.1 the mean square is only two orders of magnitude above it,
# so the squares of the smaller elements already fall below it and lose
# precision. An attention output above it is left at its own level.
ATTENTION_OUTPUT_RMS_FLOOR = 0.1

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
    """Replace this architecture's ``rotate_half`` with a static-shape equivalent.

    The upstream implementation splits the last dimension with
    ``x[..., : x.shape[-1] // 2]``. The Python-level division on a traced
    shape records an ``aten::Int`` node in the TorchScript graph, which
    the Core ML converter cannot fold into a static slice. The replacement
    (:func:`eeane.compiler.backends.decoder_patches.rotate_half`) selects
    the same two halves without consulting the shape at graph level; the
    results are identical whenever the head dimension is even, which
    :meth:`EmbeddingGemma2Backend.apply_patches` enforces for every layer.

    Rebinds the module-level function, which the rotary helper looks up at
    call time, so the replacement takes effect for every EmbeddingGemma 2
    model in the process -- whichever attention implementation it runs,
    since the rotary helper is shared by all of them. Re-applying it is
    harmless.
    """
    modeling_embedding_gemma2.rotate_half = decoder_patches.rotate_half


def patch_repeat_kv() -> None:
    """Replace the grouped-query key/value expansion with a rank-4 form.

    Upstream expands the stored key/value heads to the query head count
    through a rank-5 intermediate; the Neural Engine compiler rejects that
    subgraph, and rejecting one subgraph makes the whole compiled model
    fall back to the CPU. The replacement
    (:func:`eeane.compiler.backends.decoder_patches.repeat_kv`) stays
    rank-4 and produces the very same head order.

    Rebinds the module-level function, which only the eager attention
    calls, so the replacement takes effect for every eager EmbeddingGemma
    2 model in the process. Re-applying it is harmless.
    """
    modeling_embedding_gemma2.repeat_kv = decoder_patches.repeat_kv


# The eager attention :func:`patch_attention_query_chunks` installs: the
# upstream one with the query rows of a long sequence split into ranges
# under the shared limit (read when it is called), expanding the
# key/value heads with whatever ``repeat_kv`` the upstream module holds
# when it is called.
_eager_attention_forward = attention_split.split_eager_attention(modeling_embedding_gemma2)


def patch_attention_query_chunks() -> None:
    """Replace the eager attention with one that splits long query axes.

    On the Neural Engine, an attention whose per-head score matrix holds
    :data:`~eeane.compiler.backends.attention_split.ATTENTION_SCORE_ELEMENT_LIMIT`
    elements or more returns wrong numbers, although the same program is
    correct on the CPU. The replacement computes the query rows of such a
    sequence in a few ranges, each with a score matrix below the limit,
    and concatenates them; every row of an attention is independent of
    the others, so the result is what upstream computes. Below the limit
    nothing is split and the operations are the upstream ones.

    Rebinds the module-level function, which the attention module looks
    up at call time and only uses for the eager implementation, so the
    replacement takes effect for every eager EmbeddingGemma 2 model in the
    process. Re-applying it is harmless.
    """
    modeling_embedding_gemma2.eager_attention_forward = _eager_attention_forward


# The graph rewrites of this architecture, in the order they are applied,
# each under the name it is recorded with in a compiled variant's
# metadata. A rewrite added later -- another function rebinding something
# upstream -- only has to be appended here (and to the check below) to be
# applied and recorded.
PATCHES: tuple[tuple[str, Callable[[], None]], ...] = (
    ("rotate_half_static", patch_rotate_half),
    ("repeat_kv_rank4", patch_repeat_kv),
    ("attention_query_chunks", patch_attention_query_chunks),
)


def _expected_replacements() -> tuple[tuple[str, Callable[..., Any]], ...]:
    """Return each rebound upstream symbol with the function it must now be.

    Looked up when called, so the check compares against the very
    functions the patches bind at that moment.
    """
    return (
        ("rotate_half", decoder_patches.rotate_half),
        ("repeat_kv", decoder_patches.repeat_kv),
        ("eager_attention_forward", _eager_attention_forward),
    )


@dataclass(frozen=True)
class EmbeddingGemma2Ranges(ActivationRanges):
    """What a measurement of activation ranges observed on this model.

    Extends the shared record (the residual-stream peak and the level of
    every MLP output) by the activations of this architecture that are
    normalized as well, one entry per layer in layer order. A peak is
    ``nan`` when any of what it covers was not finite; every entry is
    ``0.0`` for a layer nothing was measured on.

    Attributes:
        q_peaks: Largest absolute value of each layer's query projection.
        k_peaks: Largest absolute value of each layer's key projection.
        v_peaks: Largest absolute value of each layer's value projection.
        ple_output_peaks: Largest absolute value of the projection that
            closes each layer's per-layer input block.
        attention_output_rms: Root mean square of each layer's attention
            output, over every input, position and dimension.
    """

    q_peaks: tuple[float, ...]
    k_peaks: tuple[float, ...]
    v_peaks: tuple[float, ...]
    ple_output_peaks: tuple[float, ...]
    attention_output_rms: tuple[float, ...]


@dataclass(frozen=True)
class RangeFactors:
    """The power-of-two factors :func:`condition_fp16_ranges` rewrites a model with.

    A factor of exactly 1 leaves what it applies to untouched.

    Attributes:
        residual_divisor: Factor the residual stream is divided by.
        q_divisor: Factor every layer's query projection is divided by.
        k_divisor: Factor every layer's key projection is divided by.
        v_divisor: Factor every layer's value projection is divided by.
        ple_output_gains: Factor each layer's per-layer input block output
            is multiplied by, in layer order.
        mlp_output_gains: Factor each layer's MLP output is multiplied by.
        attention_output_gains: Factor each layer's attention output is
            multiplied by.
    """

    residual_divisor: float
    q_divisor: float
    k_divisor: float
    v_divisor: float
    ple_output_gains: tuple[float, ...]
    mlp_output_gains: tuple[float, ...]
    attention_output_gains: tuple[float, ...]


def _calibration_texts() -> tuple[str, ...]:
    """Return the fixed sentences the activation ranges are measured on.

    The tracing example plus every language set of the shared sanity
    fixtures: a small fixed collection that covers the scripts the
    compiled model is checked on, so the measurement is reproducible and
    costs a handful of forward passes.
    """
    return (TRACE_EXAMPLE_TEXT, *(text for _, texts in SANITY_TEXT_SETS for text in texts))


def _merge_peak(current: float, observed: float) -> float:
    """Combine a running peak with a newly observed one.

    ``max`` would silently step over a NaN; a peak over values that were
    not all finite must be reported as such, so that no factor is derived
    from it.

    Args:
        current: The peak so far (``0.0`` before anything was observed).
        observed: The peak of the newest tensor.

    Returns:
        The larger of the two, or ``nan`` when either is not finite.
    """
    if not (math.isfinite(current) and math.isfinite(observed)):
        return math.nan
    return max(current, observed)


def _overall_peak(peaks: Iterable[float]) -> float:
    """Return the peak over several peaks, ``nan`` if any is not finite."""
    overall = 0.0
    for peak in peaks:
        overall = _merge_peak(overall, peak)
    return overall


def measure_activation_ranges(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int | None = None,
) -> EmbeddingGemma2Ranges:
    """Measure the activations of a model that its normalizations square.

    Each text is encoded on its own, without padding, and run through the
    model in the precision it was loaded in, with the plain 2-D attention
    mask. Hooks read the activations and change nothing about the model:

    * the residual stream is what each layer's two pre-normalizations, its
      per-layer input gate and the final normalization read, so the
      largest absolute value among those inputs is the peak of the stream;
    * the query, key and value projections and the projection closing the
      per-layer input block are read at their outputs, for their peaks;
    * each layer's MLP and attention outputs are what its post-feedforward
      and post-attention normalizations read, so the mean square of those
      inputs gives their levels.

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
    count = len(layers)
    residual_peak = [0.0]
    peaks: dict[str, list[float]] = {name: [0.0] * count for name in ("q", "k", "v", "ple")}
    square_sums: dict[str, list[float]] = {name: [0.0] * count for name in ("mlp", "attention")}
    element_counts: dict[str, list[int]] = {name: [0] * count for name in ("mlp", "attention")}

    def peak_of(tensor: torch.Tensor) -> float:
        return float(tensor.detach().abs().max())

    def watch_residual(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        residual_peak[0] = _merge_peak(residual_peak[0], peak_of(args[0]))

    def watch_peak(name: str, index: int) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: torch.Tensor) -> None:
            peaks[name][index] = _merge_peak(peaks[name][index], peak_of(output))

        return hook

    def watch_level(name: str, index: int) -> Callable[..., None]:
        def hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
            values = args[0].detach().to(torch.float64)
            square_sums[name][index] += float(values.pow(2).sum())
            element_counts[name][index] += values.numel()

        return hook

    handles = []
    inputs = 0
    try:
        for index, layer in enumerate(layers):
            for module in (
                layer.input_layernorm,
                layer.pre_feedforward_layernorm,
                layer.ple_block.per_layer_input_gate,
            ):
                handles.append(module.register_forward_pre_hook(watch_residual))
            attention = layer.self_attn
            handles.append(attention.q_proj.register_forward_hook(watch_peak("q", index)))
            handles.append(attention.k_proj.register_forward_hook(watch_peak("k", index)))
            handles.append(attention.v_proj.register_forward_hook(watch_peak("v", index)))
            handles.append(
                layer.ple_block.per_layer_projection.register_forward_hook(watch_peak("ple", index))
            )
            handles.append(
                layer.post_feedforward_layernorm.register_forward_pre_hook(
                    watch_level("mlp", index)
                )
            )
            handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    watch_level("attention", index)
                )
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

    def levels(name: str) -> tuple[float, ...]:
        return tuple(
            math.sqrt(square_sum / elements) if elements else 0.0
            for square_sum, elements in zip(square_sums[name], element_counts[name], strict=True)
        )

    return EmbeddingGemma2Ranges(
        residual_peak=residual_peak[0],
        mlp_output_rms=levels("mlp"),
        inputs=inputs,
        q_peaks=tuple(peaks["q"]),
        k_peaks=tuple(peaks["k"]),
        v_peaks=tuple(peaks["v"]),
        ple_output_peaks=tuple(peaks["ple"]),
        attention_output_rms=levels("attention"),
    )


def _attention_output_gain(rms: float) -> float:
    """Return the gain of one attention output, lifting only a level below the floor."""
    if rms < ATTENTION_OUTPUT_RMS_FLOOR:
        return mlp_output_gain(rms)
    return 1.0


def derive_range_factors(ranges: EmbeddingGemma2Ranges) -> RangeFactors:
    """Derive the conditioning factors from measured activation ranges.

    Every factor is a power of two, derived by the shared rules of
    :mod:`eeane.compiler.backends.fp16_ranges`:

    * the residual stream, and each of the query, key and value
      projections (one factor per kind of projection, shared by all
      layers), is divided by the smallest power of two that brings its
      peak to or under
      :data:`~eeane.compiler.backends.fp16_ranges.NORM_INPUT_PEAK_TARGET`;
    * each layer's per-layer input block output is multiplied by the
      inverse of that divisor for its own peak;
    * each layer's MLP output is multiplied by the power of two that
      brings its level near 1;
    * each layer's attention output is lifted by the same rule, but only
      when its level is below :data:`ATTENTION_OUTPUT_RMS_FLOOR`.

    A measurement that is zero, negative or not finite yields a factor of
    1; for a factor shared by all layers, one such layer is enough.

    Args:
        ranges: What :func:`measure_activation_ranges` observed.

    Returns:
        The factors to rewrite the measured model with.
    """
    return RangeFactors(
        residual_divisor=norm_input_divisor(ranges.residual_peak),
        q_divisor=norm_input_divisor(_overall_peak(ranges.q_peaks)),
        k_divisor=norm_input_divisor(_overall_peak(ranges.k_peaks)),
        v_divisor=norm_input_divisor(_overall_peak(ranges.v_peaks)),
        ple_output_gains=tuple(1.0 / norm_input_divisor(peak) for peak in ranges.ple_output_peaks),
        mlp_output_gains=tuple(mlp_output_gain(rms) for rms in ranges.mlp_output_rms),
        attention_output_gains=tuple(
            _attention_output_gain(rms) for rms in ranges.attention_output_rms
        ),
    )


def _check_factors(factors: RangeFactors, num_layers: int) -> None:
    """Refuse factors that cannot be applied to a model of ``num_layers`` layers.

    Raises:
        ValueError: If a per-layer list does not hold one factor per
            layer, or a factor is not a positive power of two.
    """
    per_layer = (
        ("PLE output gain", factors.ple_output_gains),
        ("MLP output gain", factors.mlp_output_gains),
        ("attention output gain", factors.attention_output_gains),
    )
    for name, gains in per_layer:
        if len(gains) != num_layers:
            raise ValueError(f"expected one {name} per layer ({num_layers}), got {len(gains)}")
    check_power_of_two("the residual divisor", factors.residual_divisor)
    check_power_of_two("the query divisor", factors.q_divisor)
    check_power_of_two("the key divisor", factors.k_divisor)
    check_power_of_two("the value divisor", factors.v_divisor)
    for name, gains in per_layer:
        for gain in gains:
            check_power_of_two(f"a {name}", gain)


def _scale_output(linear: torch.nn.Linear, factor: float) -> None:
    """Multiply what a linear layer outputs by ``factor``, bias included."""
    linear.weight.mul_(factor)
    if linear.bias is not None:
        linear.bias.mul_(factor)


def condition_fp16_ranges(model: torch.nn.Module, factors: RangeFactors) -> None:
    """Rescale a model's normalized activations into the float16 range, in place.

    An RMS normalization computes ``x * (mean(x**2) + eps) ** -0.5`` (times
    its weight): fed a positive multiple ``c * x`` of its input, it returns
    almost what it returned for ``x``, and exactly that once its ``eps`` is
    multiplied by ``c**2``. In float16 the danger is the square it takes, which
    exceeds the largest float16 number (65,504) once ``abs(x)`` passes 256,
    or falls into the underflow range when ``x`` is small. On the published
    model the query, key and value projections reach peaks of about 167,
    153 and 712, the output of the per-layer input block about 1,979 and
    the residual stream about 469; in plain float16 its embeddings fall to
    a cosine of 0.93 on the CPU, and to 0.987 on the Neural Engine at a
    batch size of 2, and with this rescaling they are back above 0.9999.

    Every factor below is therefore written into the weights that produce
    what a normalization reads and into that normalization's ``eps``; no
    forward code changes, and in float32 the model computes the same
    function as before. The normalization weights of this model are used
    as they are stored (not as an offset from 1), so a branch output is
    divided by dividing the weight of the normalization producing it.
    Five rescalings are applied, all independent of one another:

    * **The residual stream is divided by ``residual_divisor``.**
      Everything written to the stream is divided: the embedding, through
      its scale buffer, and the three branch outputs of every layer,
      through the (plain) weights of the normalizations producing them.
      Everything reading the stream absorbs the factor: the input and
      pre-feedforward normalizations of every layer and the final one,
      through their ``eps``, and each per-layer input gate, which reads
      the stream without normalizing it, through a weight multiplied back
      by the divisor. The per-layer inputs are projected from the divided
      embedding, so their normalization's ``eps`` is divided as well. The
      layer scalar is linear and needs nothing.
    * **The query, key and value projections are divided** by their own
      divisors, undone by the ``eps`` of the query, key and value
      normalizations.
    * **Each per-layer input block output, MLP output and attention
      output is multiplied** by that layer's gain, undone by the ``eps``
      of the normalization reading it.

    Every factor must be a power of two, so that rescaling a weight
    changes its exponent and nothing else. A factor of exactly 1 is not
    applied at all.

    The function is not idempotent: applying it twice rescales twice.
    :meth:`EmbeddingGemma2Backend.apply_patches` is what applies it at most
    once per model.

    Args:
        model: The loaded text model to rewrite.
        factors: The factors to apply, as :func:`derive_range_factors`
            returns them.

    Raises:
        ValueError: If a factor is not a positive power of two, or a
            per-layer list does not hold one factor per layer. Nothing is
            rewritten in either case.
    """
    layers = list(model.layers)
    _check_factors(factors, len(layers))

    with torch.no_grad():
        divisor = factors.residual_divisor
        if divisor != 1.0:
            model.embed_tokens.embed_scale.div_(divisor)
            projection = model.ple.per_layer_model_projection
            if projection.bias is not None:
                # The per-layer inputs are projected from the divided
                # embedding; a bias would have to follow it.
                projection.bias.div_(divisor)
            for norm in (model.norm, model.ple.per_layer_projection_norm):
                norm.eps = norm.eps / divisor**2
            for layer in layers:
                for norm in (
                    layer.post_attention_layernorm,
                    layer.post_feedforward_layernorm,
                    layer.ple_block.post_per_layer_input_norm,
                ):
                    norm.weight.div_(divisor)
                for norm in (layer.input_layernorm, layer.pre_feedforward_layernorm):
                    norm.eps = norm.eps / divisor**2
                # The gate reads ``h / divisor``; a weight ``divisor`` times
                # larger gives back what it computed before, and its bias
                # (if any) is added afterwards, unchanged.
                layer.ple_block.per_layer_input_gate.weight.mul_(divisor)

        for name, projection_divisor in (
            ("q", factors.q_divisor),
            ("k", factors.k_divisor),
            ("v", factors.v_divisor),
        ):
            if projection_divisor == 1.0:
                continue
            for layer in layers:
                attention = layer.self_attn
                _scale_output(getattr(attention, f"{name}_proj"), 1.0 / projection_divisor)
                norm = getattr(attention, f"{name}_norm")
                norm.eps = norm.eps / projection_divisor**2

        for index, layer in enumerate(layers):
            for gain, linear, norm in (
                (
                    factors.ple_output_gains[index],
                    layer.ple_block.per_layer_projection,
                    layer.ple_block.post_per_layer_input_norm,
                ),
                (
                    factors.mlp_output_gains[index],
                    layer.mlp.down_proj,
                    layer.post_feedforward_layernorm,
                ),
                (
                    factors.attention_output_gains[index],
                    layer.self_attn.o_proj,
                    layer.post_attention_layernorm,
                ),
            ):
                if gain == 1.0:
                    continue
                _scale_output(linear, gain)
                norm.eps = norm.eps * gain**2


def _conditioning_record(factors: RangeFactors, inputs: int) -> dict[str, Any]:
    """Describe applied factors as the JSON-serializable metadata record.

    Args:
        factors: The factors that were applied.
        inputs: Number of calibration inputs they were measured on.

    Returns:
        The record, with every per-layer list as a JSON list.
    """
    return {
        "residual_divisor": factors.residual_divisor,
        "qkv_divisors": {"q": factors.q_divisor, "k": factors.k_divisor, "v": factors.v_divisor},
        "ple_output_gains": list(factors.ple_output_gains),
        "mlp_output_gains": list(factors.mlp_output_gains),
        "attention_output_gains": list(factors.attention_output_gains),
        "calibration_inputs": inputs,
    }


def _condition_loaded_model(loaded: LoadedModel) -> dict[str, Any]:
    """Measure, derive and apply the range conditioning of a loaded model.

    The record of what was applied is kept on the model object itself.
    A model that already carries one is not measured or rewritten again;
    its record is returned instead, so asking any number of times -- once
    per compile run or once per variant -- rescales the weights once.

    The model is measured once, before anything is rewritten. That is
    enough: the projection and branch outputs a gain or a divisor applies
    to are computed from inputs the residual divisor leaves unchanged --
    the outputs of normalizations, and of the per-layer input gate, whose
    rescaled weight undoes it.

    Args:
        loaded: Handle whose model is rewritten in place.

    Returns:
        A JSON-serializable record of the applied factors and the number
        of calibration inputs.
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
        factors = derive_range_factors(ranges)
        condition_fp16_ranges(model, factors)
        record = _conditioning_record(factors, ranges.inputs)
        setattr(model, _CONDITIONING_ATTRIBUTE, record)
    # A copy, so that a caller editing its metadata cannot alter what the
    # model remembers.
    return copy.deepcopy(record)


def scaled_mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over the sequence, each token weighted by its share first.

    The same mean as :func:`eeane.compiler.backends.common.mean_pool`, but
    divided before it is summed. The token vectors of this model are
    large (up to about 3,400), and their sum over 1,024 tokens comes close
    to the float16 maximum of 65,504 (about 63,000), so a float16 graph
    that sums first and divides afterwards may overflow; weighted first,
    no partial sum exceeds the largest token value. In float32 both orders
    agree to rounding.

    The count of real tokens is an integer, so bounding it below by 1 is
    exact for every row holding a real token, and turns a row without any
    into a zero vector, which is what the shared helper returns for it.
    (A bound as small as the shared helper's would itself be zero in
    float16.)

    Args:
        hidden: Token vectors, shape (B, S, H).
        attention_mask: Attention mask, shape (B, S); non-zero marks a
            real token.

    Returns:
        Pooled vectors, shape (B, H).
    """
    mask = attention_mask.to(hidden.dtype)  # (B, S)
    count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
    weights = mask / count  # (B, S)
    return (hidden * weights.unsqueeze(-1)).sum(dim=1)


def _check_band(unstable_lengths: Sequence[int], padded_length: int) -> None:
    """Refuse a band and padded length that would not avoid the band.

    Raises:
        ValueError: If the band holds anything but positive integers, if
            the padded length is not a positive integer above the band,
            or if it lies more positions beyond some band length than
            that length holds -- the appended ids are cut from the input
            itself, so a band length can be extended by at most its own
            length.
    """
    for length in unstable_lengths:
        if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            raise ValueError(f"the band may only hold positive integer lengths (got {length!r})")
    if isinstance(padded_length, bool) or not isinstance(padded_length, int) or padded_length <= 0:
        raise ValueError(f"the padded length must be a positive integer (got {padded_length!r})")
    if not unstable_lengths:
        return
    shortest, longest = min(unstable_lengths), max(unstable_lengths)
    if padded_length <= longest:
        raise ValueError(
            f"the padded length {padded_length} must lie above the band "
            f"({shortest}..{longest}), or a band length would stay in it"
        )
    if padded_length - shortest > shortest:
        raise ValueError(
            f"the padded length {padded_length} is more than {shortest} positions beyond the "
            f"band length {shortest}, which can be extended by at most its own length"
        )


class EmbeddingGemma2MeanWrapper(torch.nn.Module):
    """Wraps the text model, masks it in-graph and mean-pools its token vectors.

    The output matches a sentence-transformers model whose modules are a
    Transformer, mean Pooling and the declared Dense projections (if any),
    without normalization: the compiled graph always returns the raw
    vector, and normalizing is left to the server's own configuration.

    Both attention masks, the extension of a band length and the pooling
    are computed inside the wrapper, so the exported program takes exactly
    the two int32 tensors Core ML feeds it and leaves no pre- or
    post-processing to reimplement on the host.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        window: int,
        pad_token_id: int,
        dense: torch.nn.Module | None = None,
        fill_value: float = MASK_FILL_VALUE,
        unstable_lengths: Sequence[int] = NE_UNSTABLE_LENGTHS,
        padded_length: int = NE_PADDED_LENGTH,
    ) -> None:
        """Store the backbone and everything the graph is built from.

        Args:
            model: Text model returning the projected token vectors first,
                loaded in eval/FP32 mode.
            window: Exclusive half-width of the sliding layers' window;
                see :func:`build_bidirectional_masks`.
            pad_token_id: Id written into the positions appended to a band
                length. They are masked out, so it only has to be a valid
                id.
            dense: Projection applied to the pooled vector, as built by
                ``build_dense``; ``None`` for a model that declares none.
            fill_value: Additive value for masked positions; see
                :data:`MASK_FILL_VALUE`.
            unstable_lengths: Sequence lengths computed at
                ``padded_length`` instead; see :data:`NE_UNSTABLE_LENGTHS`.
                An empty band extends nothing.
            padded_length: Length a band length is extended to; see
                :data:`NE_PADDED_LENGTH`.

        Raises:
            ValueError: If the band and the padded length would not take
                every band length out of the band.
        """
        super().__init__()
        unstable = tuple(unstable_lengths)
        _check_band(unstable, padded_length)
        self.model = model
        self.window = window
        self.pad_token_id = pad_token_id
        self.dense = dense
        self.fill_value = fill_value
        self.unstable_lengths = unstable
        self.padded_length = padded_length

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute mean-pooled (and, if declared, projected) embeddings.

        Args:
            input_ids: Token ids, shape (B, S).
            attention_mask: Attention mask, shape (B, S); non-zero marks a
                real token.

        Returns:
            Embeddings of shape (B, embedding width), or (B, projected
            width) when a projection was given. They are **not**
            L2-normalized.
        """
        # A constant of the traced graph: every graph has one fixed length.
        seq_len = attention_split.static_size(input_ids, 1)
        ids, mask = input_ids, attention_mask
        if seq_len in self.unstable_lengths:
            # Appended as copies of constant-width slices of the inputs,
            # rather than by padding to a length computed from a traced
            # shape, so that the graph holds no shape arithmetic.
            extra = self.padded_length - seq_len
            pad_ids = torch.full_like(input_ids[:, :extra], self.pad_token_id)
            ids = torch.cat([input_ids, pad_ids], dim=1)
            mask = torch.cat([attention_mask, torch.zeros_like(attention_mask[:, :extra])], dim=1)
        full, sliding = build_bidirectional_masks(mask, self.window, self.fill_value)
        hidden = self.model(
            input_ids=ids,
            # A mapping, not a tensor: the backbone then builds no mask of
            # its own, and each layer takes the one matching its type.
            attention_mask={FULL_ATTENTION: full, SLIDING_ATTENTION: sliding},
            # Stated here rather than left to the configuration: tracing
            # needs tuple outputs.
            return_dict=False,
        )[0]  # (B, S', E)
        if ids is not input_ids:
            hidden = hidden[:, :seq_len, :]
        pooled = scaled_mean_pool(hidden, attention_mask)  # (B, E)
        if self.dense is None:
            return pooled
        return self.dense(pooled)


# Traceable wrapper per detected pooling mode. The published model is
# trained with a mean over all real tokens; reading one position instead
# would return a vector the model was never trained to put a sentence in,
# so every other mode is refused rather than served.
EMBEDDING_WRAPPERS: dict[str, type[torch.nn.Module]] = {POOLING_MEAN: EmbeddingGemma2MeanWrapper}


class EmbeddingGemma2Backend:
    """Compile backend for the text embeddings of EmbeddingGemma 2.

    Implements the backend interface declared in
    :mod:`eeane.compiler.backends.base`, which documents what each member
    is for, in which order the pipeline calls them, and the rules an
    implementation must follow. Every method is stateless: all per-model
    state travels in the :class:`~eeane.compiler.backends.base.LoadedModel`
    handle, so one instance can serve several compile runs.

    ``reranker`` is not among :attr:`supported_kinds`: every kind-taking
    member refuses it with the reason spelled out in :data:`KIND_REFUSAL`.
    """

    name = "EmbeddingGemma2"
    supported_kinds: tuple[str, ...] = SUPPORTED_KINDS

    def load(self, model_dir: Path, kind: str, attn: str = "eager") -> LoadedModel:
        """Load the FP32 text model and its tokenizer from a HF model directory.

        The directory holds the multimodal root model. It is loaded without
        its image and audio towers -- their weights are skipped -- and only
        the text model it holds is kept.

        Args:
            model_dir: Local HuggingFace-format model directory. It is
                only ever read from.
            kind: Must be ``"embedding"``.
            attn: Attention implementation to request. ``"eager"`` is the
                path the conversion traces; ``"sdpa"`` reaches a separate
                implementation, which is what makes it the FP32 reference
                path.

        Returns:
            A handle holding the text model in eval/FP32 mode with
            ``config.return_dict = False`` (``torch.jit.trace`` needs tuple
            outputs), its tokenizer, its configuration, the declared
            pooling and the declared Dense projection.

        Raises:
            ValueError: If ``kind`` is not supported by this backend, if
                the directory declares another architecture or no text
                configuration, if its pooling cannot be determined, or if
                its declared module chain cannot be reproduced.
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
        root = AutoModel.from_pretrained(
            model_dir,
            # No tower is built, so none is held in memory; the weights
            # stored for them are skipped by the root model's own rule.
            vision_config=None,
            audio_config=None,
            attn_implementation=attn,
            dtype=torch.float32,
        )
        model = root.language_model
        model.config.return_dict = False
        if model.config._attn_implementation != attn:
            # The request is made through the root, so the text model is
            # put on the requested path explicitly whenever it was not.
            model.set_attn_implementation(attn)
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
        """Apply the mandatory graph patches and the FP16 range conditioning.

        Every rewrite in :data:`PATCHES` is a constituent of the
        conversion rather than an optional tweak -- without them the model
        fails to convert, converts into a program that runs on the CPU,
        or computes wrong numbers on the Neural Engine -- so all of them
        are always applied, and each is checked to be in place afterwards.
        They patch global ``transformers`` symbols and therefore affect
        every EmbeddingGemma 2 instance in the process; all are
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
            ValueError: If the head dimension of any layer is odd (which
                would make the ``chunk``-based ``rotate_half`` inexact), or
                if a mask fill value other than :data:`MASK_FILL_VALUE` is
                requested. Nothing is rebound or rescaled in either case.
            RuntimeError: If a replacement is not in place after its patch
                ran. Nothing is rescaled then.
        """
        # The two layer types have head widths of their own, so every
        # layer is checked rather than the configured default width.
        for index, layer in enumerate(loaded.model.layers):
            head_dim = layer.self_attn.head_dim
            if head_dim % 2 != 0:
                raise ValueError(
                    f"odd RoPE head dim ({head_dim}) in layer {index} is incompatible with "
                    "patch_rotate_half"
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
        _check_replacements()
        applied["mask_fill_value"] = MASK_FILL_VALUE
        applied[FP16_RANGE_CONDITIONING_KEY] = _condition_loaded_model(loaded)
        return applied

    def wrap(self, loaded: LoadedModel) -> torch.nn.Module:
        """Wrap the loaded model into the traceable module for its kind.

        Args:
            loaded: Handle returned by :meth:`load`; its ``pooling``
                selects the wrapper, its configuration provides the
                window of the sliding layers and the padding id, and its
                ``dense`` is applied after the pooling.

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
        return wrapper_class(
            loaded.model,
            window=_sliding_window(loaded.config),
            pad_token_id=_pad_token_id(loaded.config),
            dense=loaded.dense,
        ).eval()

    def output_name(self, kind: str) -> str:
        """Return the Core ML graph output name used for ``kind``.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return OUTPUT_NAMES[kind]

    def max_seq_len(self, model_dir: Path) -> int | None:
        """Return the effective maximum sequence length of ``model_dir``.

        The budget is the text model's own, stated in the text
        configuration the root configuration holds. Positions are rotary
        and indexed from 0 with no reserved offset, so the configured
        position budget is the usable sequence length; the window of the
        sliding layers bounds what one such layer sees, not how long a
        sequence may be. Only ``config.json`` is read; no weights are
        loaded.

        Args:
            model_dir: Local HuggingFace-format model directory.

        Returns:
            The configured positive position budget, or ``None`` when the
            file is absent/unreadable/unparsable, holds no text
            configuration, or the value is missing or not a positive
            integer. ``None`` means "unknown", and the caller then imposes
            no limit.
        """
        config = _read_config(model_dir)
        if config is None:
            # A malformed config is reported by the dispatch step; an
            # optional bucket check must not turn it into a second error.
            return None
        text_config = config.get(TEXT_CONFIG_KEY)
        if not isinstance(text_config, dict):
            return None
        value = text_config.get(MAX_POSITION_KEY)
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
            asks; the answer is the one the other embedding-only backends
            give, which keeps the interface total.
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

        Loads a second copy of the text model with the ``sdpa`` attention
        path -- a separate implementation from the eager one the
        conversion traces, with a key/value expansion of its own and
        without the range conditioning -- runs it row by row and releases
        it again.

        The baseline also hands the model the plain 2-D attention mask, so
        both attention masks are built by the framework rather than by the
        wrapper under test, and it pools with the shared mean: the two
        sides of the self-check compute the same function through
        independent implementations.

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
          stand in for it, since a configuration may name any class. The
          text model saved on its own declares a model type of its own
          and is refused with a reason: this backend reads the published
          root and nothing else.
        * ``text_config`` -- the text model's configuration, which is what
          this backend compiles; a root without one has nothing to offer.

        Only ``config.json`` is read; no weights are loaded.

        Args:
            model_dir: Local HuggingFace-format model directory.

        Raises:
            ValueError: If the configuration cannot be read, declares
                another ``model_type``, or holds no text configuration.
        """
        config = _read_config(model_dir)
        config_path = model_dir / CONFIG_FILENAME
        declared = config.get(MODEL_TYPE_KEY) if config is not None else None
        if declared == TEXT_MODEL_TYPE:
            raise ValueError(
                f"'{config_path}' declares {MODEL_TYPE_KEY}={declared!r}: the text model saved "
                f"on its own. The {self.name} backend reads the directory of the published "
                f"multimodal configuration -- the root model, {MODEL_TYPE_KEY}={MODEL_TYPE!r} -- "
                "and compiles the text model it holds; point it at that directory instead"
            )
        if declared != MODEL_TYPE:
            raise ValueError(
                f"'{config_path}' declares {MODEL_TYPE_KEY}={declared!r}, but the {self.name} "
                f"backend implements the {MODEL_TYPE!r} architecture only; a related "
                "architecture of the same family needs a backend of its own"
            )
        if not isinstance(config.get(TEXT_CONFIG_KEY), dict):
            raise ValueError(
                f"'{config_path}' holds no {TEXT_CONFIG_KEY} object, so it declares no text "
                f"model to compile. {KIND_REFUSAL}"
            )


def _check_replacements() -> None:
    """Verify that every patched upstream symbol is the replacement now.

    Raises:
        RuntimeError: If a symbol still holds another function after its
            patch ran. Something else then binds that symbol -- the
            ``kernels`` package, for one, can put its own functions in
            place of upstream ones -- and the traced graph would not
            contain the rewrite the metadata records.
    """
    missing = [
        symbol
        for symbol, replacement in _expected_replacements()
        if getattr(modeling_embedding_gemma2, symbol, None) is not replacement
    ]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} in {modeling_embedding_gemma2.__name__} is not the "
            "replacement this backend installs, so the traced graph would not contain the "
            "rewrite the conversion relies on. Something else binds that symbol -- the "
            "'kernels' package, for one, can replace upstream functions -- and must be "
            "disabled for the conversion"
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


def _sliding_window(config: Any) -> int:
    """Return the exclusive half-width of the sliding layers' window.

    The text configuration states the window as an inclusive radius, and
    loading it keeps that number as it is; the shared mask builder takes
    an exclusive bound, which is the radius plus one.

    Args:
        config: Text configuration of the loaded model.

    Returns:
        The exclusive half-width of the window.

    Raises:
        ValueError: If the configuration holds no positive integer radius.
            Without it the sliding layers cannot be masked, and assuming
            one would compute a different model.
    """
    radius = getattr(config, SLIDING_WINDOW_KEY, None)
    if isinstance(radius, bool) or not isinstance(radius, int) or radius <= 0:
        raise ValueError(
            f"the loaded configuration holds {SLIDING_WINDOW_KEY}={radius!r}, but the sliding "
            "attention layers of this architecture can only be masked with a positive integer "
            "window"
        )
    return radius + 1


def _pad_token_id(config: Any) -> int:
    """Return the id the extension of a band length appends.

    The appended positions are masked out, so the id has no effect on the
    result; it only has to be one the embedding can look up. A missing or
    unusable one is therefore replaced by 0 rather than refused.

    Args:
        config: Text configuration of the loaded model.

    Returns:
        The configured padding id when it is a non-negative integer within
        the vocabulary (as far as the configuration states one), else 0.
    """
    value = getattr(config, PAD_TOKEN_KEY, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    vocab_size = getattr(config, "vocab_size", None)
    if isinstance(vocab_size, int) and not isinstance(vocab_size, bool) and value >= vocab_size:
        return 0
    return value
