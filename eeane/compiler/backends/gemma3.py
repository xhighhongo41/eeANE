"""Gemma 3 compile backend, for bidirectional embedding models only.

Covers the text model of the Gemma 3 family when it is configured to
attend in both directions and published as a sentence-transformers
embedding model (e.g. ``google/embeddinggemma-300m``). The implementation
is a decoder stack -- grouped attention heads, rotary positions -- but
with bidirectional attention it behaves as an encoder: every position
sees the whole sequence, and the sentence vector is a masked mean over
all of them.

Three properties decide how the conversion is built here:

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
* **Two upstream constructs are replaced** before tracing:
  :func:`patch_rotate_half` removes shape arithmetic the Core ML
  converter cannot fold, and :func:`patch_repeat_kv` removes the rank-5
  intermediate that makes the Neural Engine compiler reject the attention
  subgraph. Both replacements compute exactly what upstream computes;
  their bodies are shared with the other decoder-style backends through
  :mod:`eeane.compiler.backends.decoder_patches`.
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

import gc
import json
from collections.abc import Callable
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
    "BIDIRECTIONAL_KEY",
    "CONFIG_FILENAME",
    "EMBEDDING_WRAPPERS",
    "FULL_ATTENTION",
    "MASK_FILL_VALUE",
    "MAX_POSITION_KEY",
    "MODEL_TYPE",
    "OUTPUT_NAMES",
    "PATCHES",
    "SANITY_SPECS",
    "SLIDING_ATTENTION",
    "SLIDING_WINDOW_KEY",
    "SUPPORTED_KINDS",
    "BidirectionalMeanWrapper",
    "Gemma3Backend",
    "build_bidirectional_masks",
    "patch_repeat_kv",
    "patch_rotate_half",
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


# The graph rewrites of this architecture, in the order they are applied,
# each under the name it is recorded with in a compiled variant's
# metadata. A rewrite added later -- another function rebinding something
# upstream -- only has to be appended here to be applied and recorded.
PATCHES: tuple[tuple[str, Callable[[], None]], ...] = (
    ("rotate_half_static", patch_rotate_half),
    ("repeat_kv_rank4", patch_repeat_kv),
)


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
        """Apply the mandatory Gemma 3 graph patches.

        Every rewrite in :data:`PATCHES` is a constituent of the
        conversion rather than an optional tweak -- without them the model
        either fails to convert or converts into a program that runs on
        the CPU -- so all of them are always applied. They patch global
        ``transformers`` symbols and therefore affect every Gemma 3
        instance in the process; all are semantically equivalent to
        upstream, and re-applying them is harmless.

        Args:
            loaded: Handle returned by :meth:`load`.
            mask_fill_value: Optional finite attention-mask fill value.
                This architecture needs no mask patch at all, because its
                wrapper builds the masks itself with
                :data:`MASK_FILL_VALUE` (already a finite value); the
                argument is therefore only accepted when it names that
                very value.

        Returns:
            The applied rewrites plus the mask fill value the traced graph
            uses, recorded verbatim in the compiled variant's metadata.

        Raises:
            ValueError: If the rotary head dimension is odd (which would
                make the ``chunk``-based ``rotate_half`` inexact), or if a
                mask fill value other than :data:`MASK_FILL_VALUE` is
                requested. Nothing is rebound in either case.
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
