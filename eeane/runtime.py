"""Pure-function runtime helpers for the eeANE server (v0.4実装計画.md §4.3).

These functions have no side effects and depend only on numpy/transformers
(for type hints), so they can be unit-tested without loading a Core ML
model. ``eeane.engine``/``eeane.server`` compose them with the actual
model artifacts.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence

import numpy as np
from transformers import PreTrainedTokenizerBase


def select_bucket(n_tokens: int, buckets: Sequence[int]) -> tuple[int, bool]:
    """Select the smallest sequence-length bucket that fits ``n_tokens``.

    ``buckets`` must already be sorted in ascending order (not verified
    here). Returns the smallest bucket that is greater than or equal to
    ``n_tokens``. If ``n_tokens`` exceeds every bucket, the largest bucket
    is returned with ``truncated=True`` to signal that the input must be
    truncated to fit. Non-positive ``n_tokens`` (0 or negative) is treated
    like any small input and maps to the smallest bucket without
    truncation.

    Args:
        n_tokens: Number of tokens in the input (as counted without
            padding/truncation).
        buckets: Ascending sequence of fixed sequence lengths supported by
            the deployed Core ML artifacts (e.g. ``(128, 512, 1024)``).

    Returns:
        A ``(bucket, truncated)`` tuple: the selected bucket size, and
        whether ``n_tokens`` exceeds it (requiring truncation).
    """
    for bucket in buckets:
        if n_tokens <= bucket:
            return bucket, False
    return buckets[-1], True


def count_text_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Count the tokens a single text encodes to (special tokens included).

    Args:
        tokenizer: Tokenizer used to encode ``text``.
        text: Input text.

    Returns:
        Number of tokens produced by ``tokenizer(text)``, with the
        tokenizer's default ``add_special_tokens=True`` and no
        truncation/padding applied.
    """
    return len(tokenizer(text)["input_ids"])


def count_pair_tokens(tokenizer: PreTrainedTokenizerBase, query: str, document: str) -> int:
    """Count the tokens a (query, document) pair encodes to.

    Args:
        tokenizer: Tokenizer used to encode the pair.
        query: Query text (first sequence).
        document: Document text (second sequence).

    Returns:
        Number of tokens produced by ``tokenizer(query, document)`` (pair
        encoding via the tokenizer's post_processor), with no truncation
        applied.
    """
    return len(tokenizer(query, document)["input_ids"])


def tokenize_texts(
    tokenizer: PreTrainedTokenizerBase, texts: list[str], seq_len: int
) -> dict[str, np.ndarray]:
    """Tokenize texts into fixed-shape int32 arrays for Core ML input.

    Behaves identically to ``poc.common.tokenize_batch``: padding to
    ``seq_len``, truncation enabled, and only ``input_ids``/
    ``attention_mask`` are kept.

    Args:
        tokenizer: Tokenizer used to encode ``texts``.
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

    Behaves identically to ``poc.common.tokenize_pairs``: the tokenizer's
    built-in pair encoding produces the ``<s> query </s> <s> document
    </s>`` template, ``truncation=True`` uses the tokenizer's default
    ``longest_first`` strategy across both sequences, and any key other
    than ``input_ids``/``attention_mask`` (e.g. ``token_type_ids``) is
    discarded.

    Args:
        tokenizer: Tokenizer used to encode ``pairs``.
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute a numerically stable sigmoid.

    Branches on the sign of ``x`` so ``np.exp`` is only ever evaluated on
    non-positive arguments, avoiding overflow for large-magnitude inputs
    (same formula as ``poc.common.sigmoid_np``).

    Args:
        x: Input array (raw logits).

    Returns:
        Array of the same shape as ``x``, with values in (0, 1).
    """
    is_positive = x >= 0
    exp_neg_abs = np.exp(-np.abs(x))
    return np.where(is_positive, 1.0 / (1.0 + exp_neg_abs), exp_neg_abs / (1.0 + exp_neg_abs))


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row of a (N, D) matrix.

    Args:
        matrix: Array of shape (N, D).
        eps: Lower bound applied to each row's norm to avoid division by
            zero for zero rows.

    Returns:
        Row-normalized array of the same shape and dtype as ``matrix``.
    """
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return matrix / norm


def floats_to_base64(vector: np.ndarray) -> str:
    """Encode a 1-D float vector as an OpenAI-compatible base64 string.

    Matches the OpenAI SDK convention: bytes are float32, native (little)
    endian, decoded on the client side via
    ``np.frombuffer(base64.b64decode(data), dtype="float32")``.

    Args:
        vector: 1-D array of values to encode.

    Returns:
        ASCII base64 string encoding the vector as float32 bytes.
    """
    raw = np.asarray(vector, dtype="<f4").tobytes()
    return base64.b64encode(raw).decode("ascii")


def base64_to_floats(data: str) -> np.ndarray:
    """Decode an OpenAI-compatible base64 string back into a float vector.

    Inverse of :func:`floats_to_base64`.

    Args:
        data: ASCII base64 string encoding float32 little-endian bytes.

    Returns:
        1-D array of dtype ``float32``.
    """
    return np.frombuffer(base64.b64decode(data), dtype="<f4")
