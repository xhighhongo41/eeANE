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
directory, which this backend reads (and refuses to guess).

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
    POOLING_MEAN,
    ClsEmbeddingWrapper,
    EmbeddingWrapper,
    RerankerWrapper,
    encode_pytorch,
    score_pytorch,
    tokenize_batch,
    tokenize_pairs,
)

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

# sentence-transformers pooling module: directory holding the pooling
# declaration of an embedding model, and the flags it can set.
POOLING_DIRNAME = "1_Pooling"
POOLING_MODE_PREFIX = "pooling_mode_"
POOLING_MODE_KEYS: dict[str, str] = {
    "pooling_mode_mean_tokens": POOLING_MEAN,
    "pooling_mode_cls_token": POOLING_CLS,
}

# Appended to every pooling-detection error: an embedding model whose
# pooling cannot be read must fail loudly rather than default silently,
# because the wrong pooling produces a plausible but wrong embedding.
_POOLING_REQUIREMENT = (
    "An embedding model must declare its pooling in the sentence-transformers "
    f"'{POOLING_DIRNAME}/{CONFIG_FILENAME}' with exactly one of "
    f"{' / '.join(POOLING_MODE_KEYS)} set to true."
)

# Short Japanese sentence used as the example input for torch.jit.trace.
TRACE_EXAMPLE_TEXT = "変換に使う短い日本語のサンプル文です。"

# Fixed sanity-check sentences (short / medium / long) exercising different
# amounts of padding under the same fixed sequence length.
SANITY_TEXTS: list[str] = [
    "質問: 富士山の標高は何メートルですか。",
    "文書: 富士山は静岡県と山梨県にまたがる標高3776メートルの山であり、"
    "日本の最高峰として知られている。",
    "話題: 大量の文書をあらかじめベクトルに変換して保存しておくと、"
    "検索のたびに本文を読み直さずに近い意味の文書を取り出せる。",
]

# Short Japanese (query, document) pair used as the example input for
# torch.jit.trace of a reranker.
TRACE_EXAMPLE_PAIR: tuple[str, str] = (
    "変換に使うサンプルの質問です。",
    "変換に使うサンプルの文書です。",
)

# Fixed sanity-check pairs: relevant / irrelevant / partially related. The
# first two share their query, so only the document decides which of them
# must score higher.
SANITY_PAIRS: list[tuple[str, str]] = [
    # Relevant pair
    (
        "富士山の標高は何メートルですか。",
        "富士山は静岡県と山梨県にまたがる標高3776メートルの山であり、日本の最高峰として知られている。",
    ),
    # Irrelevant pair
    (
        "富士山の標高は何メートルですか。",
        "味噌汁の出汁は昆布と鰹節を組み合わせると香りが良くなると言われている。",
    ),
    # Partially related pair
    (
        "ベクトル検索の仕組みを知りたい。",
        "図書館では蔵書を著者名の五十音順に並べて管理している。",
    ),
]

# Sanity fixtures per kind, as handed to the pipeline and the self-check.
# The reranker is expected to score pair 0 above pair 1; embeddings are
# compared row by row against their own baseline and carry no ordering
# expectation.
SANITY_SPECS: dict[str, SanitySpec] = {
    KIND_EMBEDDING: SanitySpec(inputs=tuple(SANITY_TEXTS)),
    KIND_RERANKER: SanitySpec(inputs=tuple(SANITY_PAIRS), relevant_index=0, irrelevant_index=1),
}

# Filler rows used to pad the last sanity batch when the number of sanity
# inputs is not a multiple of B. The empty strings encode to special tokens
# only, so the row still has a non-empty attention mask (a fully masked row
# would risk NaN).
BATCH_PADDING_TEXT = ""
BATCH_PADDING_PAIR: tuple[str, str] = ("", "")


def read_pooling_mode(model_dir: Path) -> str:
    """Read the pooling mode an embedding model directory declares.

    Args:
        model_dir: Local HuggingFace-format model directory, expected to
            carry a sentence-transformers pooling module.

    Returns:
        ``"mean"`` or ``"cls"``.

    Raises:
        ValueError: If the pooling declaration is missing, unreadable,
            malformed, or does not select exactly one supported mode.
    """
    path = model_dir / POOLING_DIRNAME / CONFIG_FILENAME
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"cannot read the pooling module '{path}': {exc}. {_POOLING_REQUIREMENT}"
        ) from exc
    try:
        declaration = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"'{path}' is not valid JSON: {exc}. {_POOLING_REQUIREMENT}") from exc
    if not isinstance(declaration, dict):
        raise ValueError(f"'{path}' does not contain a JSON object. {_POOLING_REQUIREMENT}")
    # Only a literal ``true`` counts: anything else (a string, a number)
    # is a declaration this backend cannot claim to understand.
    enabled = [
        key
        for key, value in declaration.items()
        if key.startswith(POOLING_MODE_PREFIX) and value is True
    ]
    if len(enabled) != 1 or enabled[0] not in POOLING_MODE_KEYS:
        declared = ", ".join(sorted(enabled)) if enabled else "none"
        raise ValueError(
            f"'{path}' does not enable exactly one supported pooling mode "
            f"(enabled: {declared}). {_POOLING_REQUIREMENT}"
        )
    return POOLING_MODE_KEYS[enabled[0]]


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
            embedding model -- the pooling declared by the model directory.

        Raises:
            ValueError: If ``kind`` is not supported by this backend, or if
                the pooling of an embedding model cannot be determined.
        """
        self._check_kind(kind)
        # Detected before the weights are read: an undeclared pooling makes
        # the model uncompilable, and finding that out first avoids loading
        # gigabytes of FP32 parameters for nothing.
        pooling = read_pooling_mode(model_dir) if kind == KIND_EMBEDDING else None
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
                model its ``pooling`` selects the wrapper.

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
        return wrapper_class(loaded.model).eval()

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
                # The baseline must pool exactly like the traced wrapper,
                # so it follows the pooling detected for this directory.
                return encode_pytorch(
                    loaded.model,
                    loaded.tokenizer,
                    list(inputs),
                    seq_len,
                    pooling=loaded.pooling,
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
