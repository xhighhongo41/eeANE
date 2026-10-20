"""BERT compile backend, for embedding models only.

Covers the original BERT encoder family whose ``config.json`` reports an
architecture starting with ``Bert``: embedding models (``BertModel`` plus
a sentence-transformers pooling module, e.g. ``BAAI/bge-large-en-v1.5``).
Cross-encoder rerankers of this family (``BertForSequenceClassification``)
are rejected with an explanation -- see the segment-id property below.

Three properties set this family apart from the other backends:

* Positions are learned absolute embeddings indexed from 0, so the whole
  configured position budget is usable and
  :meth:`BertBackend.max_seq_len` reports it unchanged. They are handed
  to the model explicitly while tracing, because deriving them inside the
  embedding layer records a graph node the Core ML converter rejects (see
  :class:`ZeroTokenTypeModel`).
* The embedding layer adds a segment (``token_type_ids``) embedding. The
  compiled graph takes ``input_ids`` and ``attention_mask`` only, so the
  segment ids are pinned to zeros inside the traced module by
  :class:`ZeroTokenTypeModel` -- the same value HuggingFace itself falls
  back to when the argument is omitted. A single sequence is segment 0
  throughout, so this is exact for an embedding model. A cross-encoder is
  not: it is trained on ``[CLS] query [SEP] document [SEP]`` with the
  document in segment 1, and pinning that to zero silently changes what
  the model is asked to score, which is why this backend compiles
  embedding models only.
* Attention is a plain rank-4 implementation with no rotary embeddings,
  so the conversion needs no rewrite of the model's code. What it does
  need is the attention mask built outside the framework's own mask
  helpers: those record an op the Core ML converter has no translation
  for and write a value FP16 cannot hold. :class:`ZeroTokenTypeModel`
  therefore hands the model its padding mask already built, in 4-D, with
  a finite fill value, and :meth:`BertBackend.apply_patches` only records
  that.

The pooling of an embedding model is not part of the HF configuration; it
is declared by the sentence-transformers pooling module in the model
directory, which this backend reads (and refuses to guess) through the
shared reader in :mod:`eeane.compiler.backends.common`. The same directory
may declare Dense projections applied after that pooling, which the shared
readers describe, build and hand to both the traced wrapper and the FP32
baseline; a declared module chain that cannot be reproduced is refused
before any weight is read.

The trace example below is English on purpose: checkpoints of this family
commonly ship an English-only WordPiece vocabulary, and a trace example
in another language would encode to little more than unknown-token rows.
The sanity fixtures are the shared per-language sets, of which the
self-check needs only one to clear its threshold, so an English-only
checkpoint is carried by the English set while a multilingual one of this
family is measured on whichever set it reads best.

Importing this module pulls in ``torch``/``transformers``; it therefore
requires the ``[compile]`` extra and must never be imported from the
``eeane serve`` code path (see :mod:`eeane.compiler`).
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from eeane.compiler.backends.base import (
    SCORE_SPACE_PROBABILITY,
    LoadedModel,
    PairTemplate,
    SanitySpec,
)
from eeane.compiler.backends.common import (
    POOLING_CLS,
    POOLING_MEAN,
    SANITY_TEXT_SETS,
    ClsEmbeddingWrapper,
    EmbeddingWrapper,
    encode_pytorch,
    load_dense,
    read_pooling_mode,
    tokenize_batch,
)
from eeane.compiler.backends.masks import MASK_FILL_VALUE, padding_mask

# The only model kind this backend compiles.
KIND_EMBEDDING = "embedding"
SUPPORTED_KINDS: tuple[str, ...] = (KIND_EMBEDDING,)

# Kind this backend knows by name in order to refuse it, and why. A BERT
# cross-encoder scores a pair whose two halves the model tells apart by
# their segment id, which the compiled graph cannot carry.
KIND_RERANKER = "reranker"
RERANKER_REFUSAL = (
    "BERT cross-encoder rerankers are not supported: the compiled graph takes "
    "input_ids and attention_mask only and fixes the segment ids to zero, which "
    "changes the meaning of a query/document pair for this architecture"
)

# Core ML graph output name per kind.
OUTPUT_NAMES: dict[str, str] = {KIND_EMBEDDING: "embedding"}

# Traceable wrapper per detected pooling mode.
EMBEDDING_WRAPPERS: dict[str, type[torch.nn.Module]] = {
    POOLING_MEAN: EmbeddingWrapper,
    POOLING_CLS: ClsEmbeddingWrapper,
}

# Model directory file and key the position budget is read from. This
# architecture indexes positions from 0 with no reserved offset, so the
# configured budget is the usable sequence length.
CONFIG_FILENAME = "config.json"
MAX_POSITION_KEY = "max_position_embeddings"

# Short English sentence used as the example input for torch.jit.trace.
TRACE_EXAMPLE_TEXT = "A short English sentence used as the conversion sample."

# Sanity fixtures per kind, as handed to the pipeline and the self-check:
# the shared per-language sets, unchanged. Embeddings are compared row by
# row against their own baseline and carry no ordering expectation.
SANITY_SPECS: dict[str, SanitySpec] = {KIND_EMBEDDING: SanitySpec(input_sets=SANITY_TEXT_SETS)}

# Filler row used to pad the last sanity batch when the number of sanity
# inputs is not a multiple of B. The empty string encodes to special tokens
# only, so the row still has a non-empty attention mask (a fully masked row
# would risk NaN).
BATCH_PADDING_TEXT = ""


class ZeroTokenTypeModel(torch.nn.Module):
    """Adapter that supplies the implicit inputs of the model explicitly.

    The compiled graph is built from ``input_ids`` and ``attention_mask``
    alone, so everything the model would otherwise derive by itself has to
    be decided while tracing:

    * ``token_type_ids`` are pinned to zeros, reproducing what HuggingFace
      does when the argument is omitted (its ``token_type_ids`` buffer is
      all zeros), but stating the assumption in the traced graph instead
      of relying on that fallback.
    * ``position_ids`` are built as ``arange(S)``, the same values the
      omitted argument would take. Left out, HuggingFace slices its
      position buffer with the traced sequence length, which records an
      ``aten::Int`` of a dynamic size that the Core ML converter cannot
      translate; passing the positions in removes that node. Each bucket
      is traced at its own fixed length, so the constant is exactly the
      one that bucket needs.
    * The attention mask is handed over already built, as the additive 4-D
      mask of :func:`~eeane.compiler.backends.masks.padding_mask`, with the
      finite fill value :data:`~eeane.compiler.backends.masks.MASK_FILL_VALUE`.
      The framework passes a 4-D mask through unchanged, so no mask is
      built in the graph: neither the ``new_ones`` op its own construction
      records nor the float32 minimum it fills with (which FP16 cannot
      hold) reaches the converter. The positions that are hidden are the
      framework's own -- the padded keys -- so the model computes the same
      function and only the fill value differs.

    Placed between the backbone and the pooling wrappers of
    :mod:`eeane.compiler.backends.common`, so those stay
    architecture-independent.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Store the BERT model to call.

        Args:
            model: ``BertModel`` loaded in eval/FP32 mode with
                ``config.return_dict = False``.
        """
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Run the model with segment ids, positions and the 4-D mask fixed.

        Args:
            input_ids: Token ids, shape (B, S).
            attention_mask: Attention mask, shape (B, S); non-zero marks a
                real token. The model itself receives it as the additive
                4-D mask built from it.

        Returns:
            The model's own output tuple, unchanged.
        """
        # zeros_like keeps the dtype and shape of the ids the graph takes,
        # so the traced constant matches the declared int input exactly.
        token_type_ids = torch.zeros_like(input_ids)
        # Shape (1, S): the embedding layer broadcasts one row of positions
        # over the batch, exactly as its own position buffer does.
        position_ids = torch.arange(
            input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
        ).unsqueeze(0)
        return self.model(
            input_ids=input_ids,
            attention_mask=padding_mask(attention_mask, MASK_FILL_VALUE),
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )


class BertBackend:
    """Compile backend for the embedding models of the BERT family.

    Implements the backend interface declared in
    :mod:`eeane.compiler.backends.base`, which documents what each member
    is for, in which order the pipeline calls them, and the rules an
    implementation must follow. Every method is stateless: all per-model
    state travels in the :class:`~eeane.compiler.backends.base.LoadedModel`
    handle, so one instance can serve several compile runs.

    ``reranker`` is not among :attr:`supported_kinds`: every kind-taking
    member refuses it with the reason spelled out in
    :data:`RERANKER_REFUSAL`.
    """

    name = "Bert"
    supported_kinds: tuple[str, ...] = SUPPORTED_KINDS

    def load(self, model_dir: Path, kind: str, attn: str = "eager") -> LoadedModel:
        """Load the FP32 model and its tokenizer from a HF model directory.

        Args:
            model_dir: Local HuggingFace-format model directory. It is
                only ever read from.
            kind: Must be ``"embedding"``.
            attn: Attention implementation to request.

        Returns:
            A handle holding the model in eval/FP32 mode with
            ``config.return_dict = False`` (``torch.jit.trace`` needs tuple
            outputs), its tokenizer, its configuration, the pooling
            declared by the model directory and the Dense projection it
            declares after that pooling, if any.

        Raises:
            ValueError: If ``kind`` is not supported by this backend, if
                the pooling of the model cannot be determined, or if the
                declared module chain cannot be reproduced.
        """
        self._check_kind(kind)
        # Both declarations are read before the weights: an undeclared
        # pooling or an unreproducible module chain makes the model
        # uncompilable, and finding that out first avoids loading
        # gigabytes of FP32 parameters for nothing.
        pooling = read_pooling_mode(model_dir)
        dense, dense_config = load_dense(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # Pooling happens in the wrapper; the BERT pooler head is unused
        # here, so skip it instead of tracing a dead subgraph.
        model = AutoModel.from_pretrained(
            model_dir,
            attn_implementation=attn,
            dtype=torch.float32,
            add_pooling_layer=False,
        )
        model.config.return_dict = False
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
        """Record how the traced graph builds its attention mask; rebind nothing.

        Its attention is a plain rank-4 implementation without rotary
        embeddings, so neither the shape rewrites of other families nor a
        patch of the framework's mask helpers apply here: nothing in
        transformers is rebound and the model is left as loaded. The mask
        remedy lives in the wrapper instead -- :class:`ZeroTokenTypeModel`
        builds the 4-D padding mask in the traced graph with
        :data:`~eeane.compiler.backends.masks.MASK_FILL_VALUE` -- and the
        record returned here states that, for the metadata of the compiled
        variant.

        Args:
            loaded: Handle returned by :meth:`load`.
            mask_fill_value: Optional finite attention-mask fill value.
                The wrapper fixes the fill value of its mask, so the
                argument is only accepted when it names that very value.

        Returns:
            A dict with ``mask_4d_static`` set to ``True`` and
            ``mask_fill_value`` set to the fill value of the traced
            graph's mask; recorded verbatim in the compiled variant's
            metadata.

        Raises:
            ValueError: If a mask fill value other than
                :data:`~eeane.compiler.backends.masks.MASK_FILL_VALUE` is
                requested.
        """
        if mask_fill_value is not None and mask_fill_value != MASK_FILL_VALUE:
            raise ValueError(
                f"{self.name} builds its attention masks inside the traced graph with a "
                f"mask fill value of {MASK_FILL_VALUE}; {mask_fill_value} cannot be applied"
            )
        return {"mask_4d_static": True, "mask_fill_value": MASK_FILL_VALUE}

    def wrap(self, loaded: LoadedModel) -> torch.nn.Module:
        """Wrap the loaded model into the traceable module for its kind.

        Args:
            loaded: Handle returned by :meth:`load`; its ``pooling``
                selects the wrapper and its ``dense`` is applied after
                that pooling.

        Returns:
            The pooling wrapper matching ``loaded.pooling``, in eval mode,
            over the adapter that fixes the segment ids, the positions and
            the 4-D mask. The wrapper pools with the 2-D mask it takes.

        Raises:
            ValueError: If ``loaded.kind`` is not supported by this
                backend, or if the handle carries a pooling mode no
                wrapper implements.
        """
        self._check_kind(loaded.kind)
        wrapper_class = EMBEDDING_WRAPPERS.get(loaded.pooling or "")
        if wrapper_class is None:
            supported = ", ".join(EMBEDDING_WRAPPERS)
            raise ValueError(
                f"unsupported pooling '{loaded.pooling}' for {self.name} (supported: {supported})"
            )
        return wrapper_class(ZeroTokenTypeModel(loaded.model), dense=loaded.dense).eval()

    def output_name(self, kind: str) -> str:
        """Return the Core ML graph output name used for ``kind``.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return OUTPUT_NAMES[kind]

    def max_seq_len(self, model_dir: Path) -> int | None:
        """Return the effective maximum sequence length of ``model_dir``.

        Positions are indexed from 0 with no reserved offset, so the
        configured position budget is the usable sequence length. Only
        ``config.json`` is read; no weights are loaded.

        Args:
            model_dir: Local HuggingFace-format model directory.

        Returns:
            The configured positive position budget, or ``None`` when the
            file is absent/unreadable/unparsable or the value is missing
            or not a positive integer. ``None`` means "unknown", and the
            caller then imposes no limit.
        """
        try:
            raw = (model_dir / CONFIG_FILENAME).read_text(encoding="utf-8")
            config = json.loads(raw)
        except (OSError, ValueError):
            # A malformed config is reported by the dispatch step; an
            # optional bucket check must not turn it into a second error.
            return None
        if not isinstance(config, dict):
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
            Always ``None``: this architecture encodes a pair through the
            tokenizer's own pair encoding, so there is nothing to shape.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        # This architecture hands both sequences to the tokenizer, whose
        # own post_processor produces the model's pair template; there is
        # nothing left for a caller to spell out.
        return None

    def reranker_score_space(self) -> str:
        """Return the space this backend's reranker scores are faithful in.

        Returns:
            :data:`~eeane.compiler.backends.base.SCORE_SPACE_PROBABILITY`:
            the classification head emits one calibrated relevance value
            per pair, and the probability its sigmoid produces is the
            number the server hands out.
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
            The immutable specification for ``kind``: sentences, which are
            compared row by row against their own baseline and therefore
            carry no ordering expectation.

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

        The ``token_type_ids`` a BERT tokenizer produces are dropped here:
        the compiled graph takes two inputs and pins the segment ids to
        zeros itself (see :class:`ZeroTokenTypeModel`).

        Args:
            loaded: Handle returned by :meth:`load`; its tokenizer encodes
                the inputs.
            inputs: Sentences.
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

        Loads its own copy of the model with the ``sdpa`` attention path,
        runs it row by row, and releases it again. The baseline calls the
        model without ``token_type_ids``, which HuggingFace fills with
        zeros -- exactly what the traced graph pins them to, so both sides
        of the self-check compute the same function.

        Args:
            model_dir: Local HuggingFace-format model directory.
            kind: Model kind.
            inputs: Sentences.
            seq_len: Fixed sequence length S.

        Returns:
            Pooled embeddings of shape (N, hidden_size), dtype float32.

        Raises:
            ValueError: If ``kind`` is unsupported or ``inputs`` is empty.
        """
        self._check_kind(kind)
        if not inputs:
            raise ValueError("no inputs to encode")
        loaded = self.load(model_dir, kind, attn="sdpa")
        try:
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

    def _check_kind(self, kind: str) -> None:
        """Validate a model kind against :data:`SUPPORTED_KINDS`.

        A ``reranker`` is refused with :data:`RERANKER_REFUSAL` appended,
        so that a user who points ``eeane compile`` at a BERT
        cross-encoder learns why it cannot be compiled rather than only
        that some kind was rejected.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        if kind in SUPPORTED_KINDS:
            return
        supported = ", ".join(SUPPORTED_KINDS)
        message = f"unsupported kind '{kind}' for {self.name} (supported: {supported})"
        if kind == KIND_RERANKER:
            message = f"{message}. {RERANKER_REFUSAL}"
        raise ValueError(message)
