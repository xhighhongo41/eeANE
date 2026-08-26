"""Architecture-independent helpers shared by the compile backends.

Everything in here is plumbing that does not depend on a particular model
architecture: masked pooling, the stable sigmoid used for reranker
post-processing, fixed-shape tokenization, the FP32 PyTorch baselines,
the traceable wrapper modules and the reader for the pooling mode a
sentence-transformers model directory declares. Architecture-specific
code (graph patches, fixtures, position-embedding offsets) stays in the
per-family backend modules that import from here.

The pooling helpers and the wrappers are the single source of truth for
both sides of the self-check: the module that is traced into the Core ML
graph and the PyTorch baseline it is compared against must compute the
same function, or the comparison is meaningless.

Importing this module pulls in ``torch``/``transformers``; it therefore
requires the ``[compile]`` extra and must never be imported from the
``eeane serve`` code path (see :mod:`eeane.compiler`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Pooling modes the shared embedding helpers implement. A backend records
# the mode it detected on its LoadedModel handle, and both the wrapper
# selection and the FP32 baseline are driven by that value.
POOLING_MEAN = "mean"
POOLING_CLS = "cls"
POOLING_MODES: tuple[str, ...] = (POOLING_MEAN, POOLING_CLS)

# sentence-transformers pooling module: directory holding the pooling
# declaration of an embedding model, the file inside it, and the flags it
# can set. The declaration is not part of the HF configuration and does
# not depend on the architecture, so every backend whose embedding models
# come from sentence-transformers reads the same file the same way.
POOLING_DIRNAME = "1_Pooling"
POOLING_CONFIG_FILENAME = "config.json"
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
    f"'{POOLING_DIRNAME}/{POOLING_CONFIG_FILENAME}' with exactly one of "
    f"{' / '.join(POOLING_MODE_KEYS)} set to true."
)


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
    path = model_dir / POOLING_DIRNAME / POOLING_CONFIG_FILENAME
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


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over the sequence dimension.

    Single source of truth shared by the in-graph wrapper
    (:class:`EmbeddingWrapper`) and the PyTorch baseline
    (:func:`encode_pytorch`); changing the formula for one consumer
    without the other would make the self-check compare two different
    functions.

    Args:
        hidden: Last hidden state, shape (B, S, H).
        attention_mask: Attention mask, shape (B, S).

    Returns:
        Pooled embeddings, shape (B, H).
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, S, 1)
    summed = (hidden * mask).sum(dim=1)  # (B, H)
    count = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
    return summed / count


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Compute a numerically stable sigmoid.

    Branches on the sign of ``x`` so ``np.exp`` is only ever evaluated on
    non-positive arguments, avoiding overflow for large-magnitude inputs.

    Args:
        x: Input array (raw logits).

    Returns:
        Array of the same shape as ``x``, with values in (0, 1).
    """
    is_positive = x >= 0
    exp_neg_abs = np.exp(-np.abs(x))
    return np.where(is_positive, 1.0 / (1.0 + exp_neg_abs), exp_neg_abs / (1.0 + exp_neg_abs))


def tokenize_batch(
    tokenizer: PreTrainedTokenizerBase, texts: list[str], seq_len: int
) -> dict[str, np.ndarray]:
    """Tokenize texts into fixed-shape int32 arrays for Core ML input.

    Args:
        tokenizer: Tokenizer of the model directory being compiled.
        texts: Input sentences (prefixes, if any, must already be applied
            by the caller).
        seq_len: Fixed sequence length used for padding/truncation.

    Returns:
        Dict with ``input_ids`` and ``attention_mask``, each of shape
        ``(len(texts), seq_len)`` and dtype ``np.int32``.
    """
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"].astype(np.int32),
        "attention_mask": encoded["attention_mask"].astype(np.int32),
    }


def tokenize_pairs(
    tokenizer: PreTrainedTokenizerBase, pairs: list[tuple[str, str]], seq_len: int
) -> dict[str, np.ndarray]:
    """Tokenize (query, document) pairs into fixed-shape int32 arrays.

    Delegates to the tokenizer's built-in pair encoding
    (``tokenizer(queries, documents, ...)``) so that the pair template of
    the model at hand is produced by the tokenizer's own post_processor
    rather than reimplemented here. ``truncation=True`` uses the
    tokenizer's default ``longest_first`` strategy across both sequences.
    Any key other than ``input_ids``/``attention_mask`` returned by the
    tokenizer (e.g. ``token_type_ids``) is discarded, since the compiled
    graph only accepts those two inputs.

    Args:
        tokenizer: Tokenizer of the model directory being compiled.
        pairs: List of (query, document) pairs.
        seq_len: Fixed sequence length used for padding/truncation.

    Returns:
        Dict with ``input_ids`` and ``attention_mask``, each of shape
        ``(len(pairs), seq_len)`` and dtype ``np.int32``.
    """
    queries = [query for query, _ in pairs]
    documents = [document for _, document in pairs]
    encoded = tokenizer(
        queries,
        documents,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"].astype(np.int32),
        "attention_mask": encoded["attention_mask"].astype(np.int32),
    }


def encode_pytorch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    seq_len: int,
    pooling: str = POOLING_MEAN,
) -> np.ndarray:
    """Compute FP32 baseline embeddings with a batch-size-1 loop.

    Args:
        model: Embedding model loaded by a backend's ``load``.
        tokenizer: Tokenizer for the same model directory.
        texts: Input sentences.
        seq_len: Fixed sequence length used for tokenization.
        pooling: Pooling mode to apply, one of :data:`POOLING_MODES`. It
            must match the pooling the traced wrapper performs.

    Returns:
        Embeddings array of shape (len(texts), hidden_size), dtype float32.

    Raises:
        ValueError: If ``pooling`` is not a supported pooling mode.
    """
    if pooling not in POOLING_MODES:
        supported = ", ".join(POOLING_MODES)
        raise ValueError(f"unsupported pooling '{pooling}' (supported: {supported})")
    batch = tokenize_batch(tokenizer, texts, seq_len)
    hidden_size = model.config.hidden_size
    embeddings = np.empty((len(texts), hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(len(texts)):
            # nn.Embedding lookup requires int64 indices; tokenize_batch
            # returns int32 for Core ML compatibility, so cast here.
            input_ids = torch.from_numpy(batch["input_ids"][i : i + 1]).long()
            attention_mask = torch.from_numpy(batch["attention_mask"][i : i + 1]).long()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs[0]  # (1, S, H)
            if pooling == POOLING_MEAN:
                pooled = mean_pool(hidden, attention_mask)  # (1, H)
            else:
                pooled = hidden[:, 0]  # CLS token, (1, H)
            embeddings[i] = pooled.numpy().astype(np.float32)
    return embeddings


def score_pytorch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: list[tuple[str, str]],
    seq_len: int,
) -> np.ndarray:
    """Compute FP32 baseline raw reranker logits with a batch-size-1 loop.

    Args:
        model: Reranker model loaded by a backend's ``load``.
        tokenizer: Tokenizer for the same model directory.
        pairs: List of (query, document) pairs.
        seq_len: Fixed sequence length used for tokenization.

    Returns:
        Raw logits array of shape (len(pairs),), dtype float32.
    """
    batch = tokenize_pairs(tokenizer, pairs, seq_len)
    scores = np.empty(len(pairs), dtype=np.float32)
    with torch.no_grad():
        for i in range(len(pairs)):
            # nn.Embedding lookup requires int64 indices; tokenize_pairs
            # returns int32 for Core ML compatibility, so cast here.
            input_ids = torch.from_numpy(batch["input_ids"][i : i + 1]).long()
            attention_mask = torch.from_numpy(batch["attention_mask"][i : i + 1]).long()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]  # (1, 1)
            scores[i] = logits.reshape(-1)[0].item()
    return scores


class EmbeddingWrapper(torch.nn.Module):
    """Wraps a backbone model and performs masked mean pooling in-graph.

    The output matches a sentence-transformers model whose modules are a
    Transformer followed by mean Pooling, without normalization.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Store the backbone model.

        Args:
            model: Backbone loaded in eval/FP32 mode with
                ``config.return_dict = False``.
        """
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute pooled sentence embeddings.

        Args:
            input_ids: Token ids, shape (B, S).
            attention_mask: Attention mask, shape (B, S).

        Returns:
            Pooled embeddings, shape (B, hidden_size).
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs[0]  # (B, S, H)
        return mean_pool(hidden, attention_mask)


class ClsEmbeddingWrapper(torch.nn.Module):
    """Wraps a backbone model and takes the first token's state in-graph.

    The output matches a sentence-transformers model whose modules are a
    Transformer followed by CLS Pooling, without normalization. The
    attention mask still reaches the backbone, but it does not take part
    in the pooling: the first position is never padding.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Store the backbone model.

        Args:
            model: Backbone loaded in eval/FP32 mode with
                ``config.return_dict = False``.
        """
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute CLS-pooled sentence embeddings.

        Args:
            input_ids: Token ids, shape (B, S).
            attention_mask: Attention mask, shape (B, S).

        Returns:
            Hidden state of the first token, shape (B, hidden_size).
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs[0]  # (B, S, H)
        return hidden[:, 0]


class RerankerWrapper(torch.nn.Module):
    """Wraps a sequence-classification model and exposes raw logits.

    The Core ML graph reproduces the HF forward as-is (the model's own
    pooling plus its classification head). Sigmoid is applied outside the
    graph, in Python post-processing.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Store the classification model.

        Args:
            model: Sequence-classification model loaded in eval/FP32 mode
                with ``config.return_dict = False``.
        """
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute raw relevance logits.

        Args:
            input_ids: Token ids, shape (B, S).
            attention_mask: Attention mask, shape (B, S).

        Returns:
            Raw logits, shape (B, 1).
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs[0]  # logits (B, 1)
