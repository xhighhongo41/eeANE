"""XLM-RoBERTa compile backend.

Covers the encoder family whose ``config.json`` reports an architecture
starting with ``XLMRoberta`` -- multilingual embedding models
(``XLMRobertaModel`` plus a sentence-transformers pooling module, e.g.
``intfloat/multilingual-e5-base``) and cross-encoder rerankers
(``XLMRobertaForSequenceClassification`` with a single output label, e.g.
``BAAI/bge-reranker-v2-m3``).

Two properties set this family apart from the other backends:

* Positions are learned absolute embeddings indexed from ``padding_idx +
  1``, so the first two rows of the position table are never addressable
  and the usable sequence length is two tokens shorter than the
  configured position budget.
* Attention is a plain rank-4 implementation with no rotary embeddings,
  so the conversion needs no graph rewrites at all and
  :meth:`XlmRobertaBackend.apply_patches` is a no-op.

The pooling of an embedding model is not part of the HF configuration; it
is declared by the sentence-transformers pooling module in the model
directory, as are the Dense projections that may follow it. This backend
reads both (and refuses to guess either) through the shared readers in
:mod:`eeane.compiler.backends.common`, re-exported here so that the names
stay reachable under this module.

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
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from eeane.compiler.backends.base import LoadedModel, SanitySpec
from eeane.compiler.backends.common import (
    POOLING_CLS,
    POOLING_DIRNAME,
    POOLING_MEAN,
    POOLING_MODE_KEYS,
    POOLING_MODE_PREFIX,
    SANITY_IRRELEVANT_INDEX,
    SANITY_PAIR_SETS,
    SANITY_RELEVANT_INDEX,
    SANITY_TEXT_SETS,
    ClsEmbeddingWrapper,
    EmbeddingWrapper,
    RerankerWrapper,
    encode_pytorch,
    load_dense,
    read_pooling_mode,
    score_pytorch,
    tokenize_batch,
    tokenize_pairs,
)

# Public surface of this module, including the pooling-declaration reader
# and its constants, re-exported from
# :mod:`eeane.compiler.backends.common`.
__all__ = [
    "CONFIG_FILENAME",
    "EMBEDDING_WRAPPERS",
    "MAX_POSITION_KEY",
    "OUTPUT_NAMES",
    "POOLING_DIRNAME",
    "POOLING_MODE_KEYS",
    "POOLING_MODE_PREFIX",
    "POSITION_OFFSET",
    "SANITY_PAIR_SETS",
    "SANITY_SPECS",
    "SANITY_TEXT_SETS",
    "SUPPORTED_KINDS",
    "XlmRobertaBackend",
    "load_dense",
    "read_pooling_mode",
]

# Model kinds understood by this backend.
KIND_EMBEDDING = "embedding"
KIND_RERANKER = "reranker"
SUPPORTED_KINDS: tuple[str, ...] = (KIND_EMBEDDING, KIND_RERANKER)

# Core ML graph output name per kind (embeddings vs raw relevance logits).
OUTPUT_NAMES: dict[str, str] = {KIND_EMBEDDING: "embedding", KIND_RERANKER: "logits"}

# Traceable wrapper per detected pooling mode.
EMBEDDING_WRAPPERS: dict[str, type[torch.nn.Module]] = {
    POOLING_MEAN: EmbeddingWrapper,
    POOLING_CLS: ClsEmbeddingWrapper,
}

# Model directory file and key the position budget is read from.
CONFIG_FILENAME = "config.json"
MAX_POSITION_KEY = "max_position_embeddings"

# Positions are derived from the padding index (position_ids start at
# padding_idx + 1), so the first two rows of the position embedding table
# are unreachable and the usable sequence length is that much shorter.
POSITION_OFFSET = 2

# Short Japanese sentence used as the example input for torch.jit.trace.
TRACE_EXAMPLE_TEXT = "変換に使う短い日本語のサンプル文です。"

# Short Japanese (query, document) pair used as the example input for
# torch.jit.trace of a reranker.
TRACE_EXAMPLE_PAIR: tuple[str, str] = (
    "変換に使うサンプルの質問です。",
    "変換に使うサンプルの文書です。",
)

# Sanity fixtures per kind, as handed to the pipeline and the self-check:
# the shared per-language sets, unchanged -- this family is multilingual,
# so it has no reason to prefer fixtures of its own for any language. The
# reranker is expected to score pair 0 of a set above pair 1; embeddings
# are compared row by row against their own baseline and carry no
# ordering expectation.
SANITY_SPECS: dict[str, SanitySpec] = {
    KIND_EMBEDDING: SanitySpec(input_sets=SANITY_TEXT_SETS),
    KIND_RERANKER: SanitySpec(
        input_sets=SANITY_PAIR_SETS,
        relevant_index=SANITY_RELEVANT_INDEX,
        irrelevant_index=SANITY_IRRELEVANT_INDEX,
    ),
}

# Filler rows used to pad the last sanity batch when the number of sanity
# inputs is not a multiple of B. The empty strings encode to special tokens
# only, so the row still has a non-empty attention mask (a fully masked row
# would risk NaN).
BATCH_PADDING_TEXT = ""
BATCH_PADDING_PAIR: tuple[str, str] = ("", "")


class XlmRobertaBackend:
    """Compile backend for the XLM-RoBERTa architecture family.

    Implements the backend interface declared in
    :mod:`eeane.compiler.backends.base`, which documents what each member
    is for, in which order the pipeline calls them, and the rules an
    implementation must follow. Every method is stateless: all per-model
    state travels in the :class:`~eeane.compiler.backends.base.LoadedModel`
    handle, so one instance can serve several compile runs.
    """

    name = "XLMRoberta"
    supported_kinds: tuple[str, ...] = SUPPORTED_KINDS

    def load(self, model_dir: Path, kind: str, attn: str = "eager") -> LoadedModel:
        """Load the FP32 model and its tokenizer from a HF model directory.

        Args:
            model_dir: Local HuggingFace-format model directory. It is
                only ever read from.
            kind: ``"embedding"`` (``AutoModel``) or ``"reranker"``
                (``AutoModelForSequenceClassification``).
            attn: Attention implementation to request.

        Returns:
            A handle holding the model in eval/FP32 mode with
            ``config.return_dict = False`` (``torch.jit.trace`` needs tuple
            outputs), its tokenizer, its configuration and -- for an
            embedding model -- the pooling and the Dense projection
            declared by the model directory.

        Raises:
            ValueError: If ``kind`` is not supported by this backend, if
                the pooling of an embedding model cannot be determined, or
                if its declared module chain cannot be reproduced.
        """
        self._check_kind(kind)
        # Both declarations are read before the weights: an undeclared
        # pooling or an unreproducible module chain makes the model
        # uncompilable, and finding that out first avoids loading
        # gigabytes of FP32 parameters for nothing. A reranker has
        # neither: its score comes from its own classification head.
        pooling = read_pooling_mode(model_dir) if kind == KIND_EMBEDDING else None
        dense, dense_config = load_dense(model_dir) if kind == KIND_EMBEDDING else (None, None)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if kind == KIND_EMBEDDING:
            # Pooling happens in the wrapper; the HF pooler head is unused
            # (and absent from typical embedding checkpoints), so skip it
            # instead of tracing a dead subgraph over random weights.
            model = AutoModel.from_pretrained(
                model_dir,
                attn_implementation=attn,
                dtype=torch.float32,
                add_pooling_layer=False,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_dir, attn_implementation=attn, dtype=torch.float32
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
        """Do nothing: this architecture converts without any rewrite.

        Its attention is a plain rank-4 implementation without rotary
        embeddings, so neither the shape rewrites nor the mask remedy of
        other families apply here. A requested mask fill value is refused
        rather than ignored, so that a caller cannot believe a remedy was
        applied when it was not.

        Args:
            loaded: Handle returned by :meth:`load`.
            mask_fill_value: Not supported by this backend.

        Returns:
            Always an empty dict: no patch is ever applied.

        Raises:
            ValueError: If ``mask_fill_value`` is given.
        """
        if mask_fill_value is not None:
            raise ValueError(
                f"{self.name} does not implement a mask fill patch "
                f"(requested fill value: {mask_fill_value})"
            )
        return {}

    def wrap(self, loaded: LoadedModel) -> torch.nn.Module:
        """Wrap the loaded model into the traceable module for its kind.

        Args:
            loaded: Handle returned by :meth:`load`; for an embedding
                model its ``pooling`` selects the wrapper and its
                ``dense`` is applied after that pooling.

        Returns:
            The pooling wrapper matching ``loaded.pooling`` for an
            embedding model, or the raw-logits wrapper for a reranker, in
            eval mode.

        Raises:
            ValueError: If ``loaded.kind`` is not supported by this
                backend, or if an embedding handle carries a pooling mode
                no wrapper implements.
        """
        self._check_kind(loaded.kind)
        if loaded.kind == KIND_RERANKER:
            return RerankerWrapper(loaded.model).eval()
        wrapper_class = EMBEDDING_WRAPPERS.get(loaded.pooling or "")
        if wrapper_class is None:
            supported = ", ".join(EMBEDDING_WRAPPERS)
            raise ValueError(
                f"unsupported pooling '{loaded.pooling}' for {self.name} (supported: {supported})"
            )
        return wrapper_class(loaded.model, dense=loaded.dense).eval()

    def output_name(self, kind: str) -> str:
        """Return the Core ML graph output name used for ``kind``.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return OUTPUT_NAMES[kind]

    def max_seq_len(self, model_dir: Path) -> int | None:
        """Return the effective maximum sequence length of ``model_dir``.

        The configured position budget counts the reserved leading
        positions, which no input token can ever address, so
        :data:`POSITION_OFFSET` is subtracted. Only ``config.json`` is
        read; no weights are loaded.

        Args:
            model_dir: Local HuggingFace-format model directory.

        Returns:
            The usable sequence length, or ``None`` when the file is
            absent/unreadable/unparsable or the configured value is
            missing, not an integer, or too small to leave any usable
            position. ``None`` means "unknown", and the caller then
            imposes no limit.
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
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        usable = value - POSITION_OFFSET
        return usable if usable > 0 else None

    def trace_example(self, kind: str) -> Any:
        """Return the fixed raw example input used for ``torch.jit.trace``.

        Args:
            kind: Model kind.

        Returns:
            A sentence (embedding) or a (query, document) pair (reranker).
            The caller replicates it to B rows before tracing so the traced
            graph already carries the target batch size.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return TRACE_EXAMPLE_TEXT if kind == KIND_EMBEDDING else TRACE_EXAMPLE_PAIR

    def sanity_spec(self, kind: str) -> SanitySpec:
        """Return the fixed sanity-check inputs and metadata for ``kind``.

        Returns:
            The immutable specification for ``kind``: sentences for an
            embedding model, (query, document) pairs plus the expected
            relevant/irrelevant indices for a reranker.

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
        return BATCH_PADDING_TEXT if kind == KIND_EMBEDDING else BATCH_PADDING_PAIR

    def tokenize(
        self, loaded: LoadedModel, inputs: list[Any], seq_len: int
    ) -> dict[str, np.ndarray]:
        """Tokenize raw inputs into fixed-shape int32 Core ML arrays.

        Args:
            loaded: Handle returned by :meth:`load`; its tokenizer encodes
                the inputs and its kind selects single-sequence vs pair
                encoding.
            inputs: Sentences (embedding) or (query, document) pairs
                (reranker).
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
        if loaded.kind == KIND_EMBEDDING:
            return tokenize_batch(loaded.tokenizer, list(inputs), seq_len)
        return tokenize_pairs(loaded.tokenizer, [(query, doc) for query, doc in inputs], seq_len)

    def reference_outputs(
        self, model_dir: Path, kind: str, inputs: list[Any], seq_len: int
    ) -> np.ndarray:
        """Compute the FP32 (sdpa) reference outputs for ``inputs``.

        Loads its own copy of the model with the ``sdpa`` attention path,
        runs it row by row, and releases it again.

        Args:
            model_dir: Local HuggingFace-format model directory.
            kind: Model kind.
            inputs: Sentences or (query, document) pairs.
            seq_len: Fixed sequence length S.

        Returns:
            Pooled embeddings of shape (N, hidden_size) for ``embedding``,
            raw logits of shape (N,) for ``reranker``; dtype float32.

        Raises:
            ValueError: If ``kind`` is unsupported or ``inputs`` is empty.
        """
        self._check_kind(kind)
        if not inputs:
            raise ValueError("no inputs to score")
        loaded = self.load(model_dir, kind, attn="sdpa")
        try:
            if kind == KIND_EMBEDDING:
                # The baseline must pool and project exactly like the
                # traced wrapper, so it follows both declarations of this
                # directory.
                return encode_pytorch(
                    loaded.model,
                    loaded.tokenizer,
                    list(inputs),
                    seq_len,
                    pooling=loaded.pooling,
                    dense=loaded.dense,
                )
            return score_pytorch(
                loaded.model, loaded.tokenizer, [(q, d) for q, d in inputs], seq_len
            )
        finally:
            del loaded
            gc.collect()

    def _check_kind(self, kind: str) -> None:
        """Validate a model kind against :data:`SUPPORTED_KINDS`.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        if kind not in SUPPORTED_KINDS:
            supported = ", ".join(SUPPORTED_KINDS)
            raise ValueError(f"unsupported kind '{kind}' for {self.name} (supported: {supported})")
