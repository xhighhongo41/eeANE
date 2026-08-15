"""Runtime helpers for the eeANE server.

Everything here depends only on numpy and the ``tokenizers`` library, so
it can be unit-tested without loading a Core ML model -- and, since v0.6,
without ``transformers`` being installed at all: the server consumes a
``tokenizer.json`` frozen at compile time
(:mod:`eeane.compiler.tokenizer_freeze`) instead of building a tokenizer
from a HuggingFace model directory. ``eeane.engine``/``eeane.server``
compose these helpers with the actual model artifacts.

Apart from :func:`load_frozen_tokenizer` (which reads a file) the
functions are pure with one caveat: encoding mutates the Rust-side
padding/truncation state of the tokenizer it is given, so callers must
serialize them with the engine's tokenizer lock.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Encoding, Tokenizer


@dataclass(frozen=True)
class FrozenTokenizer:
    """A compile-time frozen tokenizer plus the pad settings baked into it.

    The pad settings are captured once, at load time, because the
    counting helpers below have to switch the backend to "no padding" and
    would otherwise lose them (``Tokenizer.padding`` then reads ``None``).

    Attributes:
        backend: The ``tokenizers`` tokenizer loaded from the frozen
            ``tokenizer.json``.
        pad_id: Token id used to fill the padded positions.
        pad_token: Token string matching :attr:`pad_id`.
        pad_type_id: ``token_type_ids`` value written at padded
            positions.
        pad_direction: ``"right"`` or ``"left"``; which end of the
            sequence padding is appended to.
    """

    backend: Tokenizer
    pad_id: int
    pad_token: str
    pad_type_id: int
    pad_direction: str


def load_frozen_tokenizer(path: Path) -> FrozenTokenizer:
    """Load a frozen ``tokenizer.json`` and read its pad settings.

    Args:
        path: Frozen tokenizer file produced by ``eeane compile``
            (:func:`eeane.compiler.tokenizer_freeze.freeze_tokenizer`).

    Returns:
        The loaded tokenizer together with its pad settings.

    Raises:
        ValueError: If the file carries no ``padding`` section (i.e. it is
            a plain HuggingFace-distributed ``tokenizer.json`` rather than
            an eeANE-frozen one), so the runtime cannot know how to pad.
    """
    backend = Tokenizer.from_file(str(path))
    padding = backend.padding
    if padding is None or padding.get("pad_id") is None or padding.get("pad_token") is None:
        raise ValueError(
            f"frozen tokenizer '{path}' carries no padding settings; it looks like a "
            "HuggingFace-distributed tokenizer.json rather than an eeANE-frozen one. "
            "Regenerate it with: eeane compile <model>"
        )
    pad_type_id = padding.get("pad_type_id")
    direction = padding.get("direction")
    return FrozenTokenizer(
        backend=backend,
        pad_id=int(padding["pad_id"]),
        pad_token=str(padding["pad_token"]),
        pad_type_id=0 if pad_type_id is None else int(pad_type_id),
        pad_direction="right" if direction is None else str(direction),
    )


def _apply_fixed_length_state(tokenizer: FrozenTokenizer, seq_len: int) -> Tokenizer:
    """Switch a tokenizer to "truncate and pad to ``seq_len``" and return its backend.

    Reproduces what ``PreTrainedTokenizerFast.set_truncation_and_padding``
    does for ``padding="max_length", truncation=True, max_length=seq_len``.
    The truncation strategy/direction are left at the ``tokenizers``
    defaults (``longest_first``, ``right``), which is what HuggingFace
    passes for these models; a model whose ``truncation_side`` differs
    would be caught by the compile-time freeze verification.

    Args:
        tokenizer: Frozen tokenizer to reconfigure.
        seq_len: Fixed sequence length.

    Returns:
        The reconfigured backend tokenizer.

    Raises:
        ValueError: If ``seq_len`` is not strictly positive.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be a positive bucket length, got {seq_len}")
    backend = tokenizer.backend
    backend.enable_truncation(max_length=seq_len)
    backend.enable_padding(
        length=seq_len,
        direction=tokenizer.pad_direction,
        pad_id=tokenizer.pad_id,
        pad_token=tokenizer.pad_token,
        pad_type_id=tokenizer.pad_type_id,
    )
    return backend


def _apply_counting_state(tokenizer: FrozenTokenizer) -> Tokenizer:
    """Switch a tokenizer to "no truncation, no padding" and return its backend.

    Args:
        tokenizer: Frozen tokenizer to reconfigure.

    Returns:
        The reconfigured backend tokenizer, so that an encoding's length
        is the real token count that drives bucket selection.
    """
    backend = tokenizer.backend
    backend.no_truncation()
    backend.no_padding()
    return backend


def _stack_encodings(encodings: list[Encoding], seq_len: int) -> dict[str, np.ndarray]:
    """Stack fixed-length encodings into the two Core ML input arrays.

    Args:
        encodings: Encodings produced with padding/truncation set to
            ``seq_len``.
        seq_len: Fixed sequence length, used to shape an empty batch (for
            which there is no row to infer the width from).

    Returns:
        Dict with ``input_ids`` and ``attention_mask`` of shape
        ``(len(encodings), seq_len)``, dtype ``np.int32``.
    """
    if not encodings:
        return {
            "input_ids": np.empty((0, seq_len), dtype=np.int32),
            "attention_mask": np.empty((0, seq_len), dtype=np.int32),
        }
    return {
        "input_ids": np.asarray([encoding.ids for encoding in encodings], dtype=np.int32),
        "attention_mask": np.asarray(
            [encoding.attention_mask for encoding in encodings], dtype=np.int32
        ),
    }


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


def count_text_tokens(tokenizer: FrozenTokenizer, text: str) -> int:
    """Count the tokens a single text encodes to (special tokens included).

    Args:
        tokenizer: Tokenizer used to encode ``text``.
        text: Input text.

    Returns:
        Number of tokens the text encodes to, with the tokenizer's
        default ``add_special_tokens=True`` and no truncation/padding
        applied.
    """
    backend = _apply_counting_state(tokenizer)
    return len(backend.encode(text).ids)


def count_pair_tokens(tokenizer: FrozenTokenizer, query: str, document: str) -> int:
    """Count the tokens a (query, document) pair encodes to.

    Args:
        tokenizer: Tokenizer used to encode the pair.
        query: Query text (first sequence).
        document: Document text (second sequence).

    Returns:
        Number of tokens the pair encodes to (pair encoding via the
        tokenizer's post_processor), with no truncation/padding applied.
    """
    backend = _apply_counting_state(tokenizer)
    return len(backend.encode(query, document).ids)


def tokenize_texts(
    tokenizer: FrozenTokenizer, texts: list[str], seq_len: int
) -> dict[str, np.ndarray]:
    """Tokenize texts into fixed-shape int32 arrays for Core ML input.

    Reproduces ``tokenizer(texts, padding="max_length", truncation=True,
    max_length=seq_len, return_tensors="np")`` (the v0.4/v0.5 behaviour,
    still the reference in ``poc.common.tokenize_batch``) on the frozen
    tokenizer, keeping only ``input_ids``/``attention_mask``.

    Args:
        tokenizer: Tokenizer used to encode ``texts``.
        texts: Input sentences (prefixes, if any, must already be applied
            by the caller).
        seq_len: Fixed sequence length used for padding/truncation.

    Returns:
        Dict with ``input_ids`` and ``attention_mask``, each of shape
        ``(len(texts), seq_len)`` and dtype ``np.int32``.

    Raises:
        ValueError: If ``seq_len`` is not strictly positive.
    """
    backend = _apply_fixed_length_state(tokenizer, seq_len)
    return _stack_encodings(backend.encode_batch(texts), seq_len)


def tokenize_pairs(
    tokenizer: FrozenTokenizer, pairs: list[tuple[str, str]], seq_len: int
) -> dict[str, np.ndarray]:
    """Tokenize (query, document) pairs into fixed-shape int32 arrays.

    Reproduces ``tokenizer(queries, documents, padding="max_length",
    truncation=True, max_length=seq_len, return_tensors="np")``: the
    tokenizer's built-in pair encoding produces the ``<s> query </s> <s>
    document </s>`` template, truncation uses the default
    ``longest_first`` strategy across both sequences, and any output other
    than ``input_ids``/``attention_mask`` (e.g. ``token_type_ids``) is
    discarded.

    Args:
        tokenizer: Tokenizer used to encode ``pairs``.
        pairs: List of (query, document) pairs.
        seq_len: Fixed sequence length used for padding/truncation.

    Returns:
        Dict with ``input_ids`` and ``attention_mask``, each of shape
        ``(len(pairs), seq_len)`` and dtype ``np.int32``.

    Raises:
        ValueError: If ``seq_len`` is not strictly positive.
    """
    backend = _apply_fixed_length_state(tokenizer, seq_len)
    # Rebuilt as tuples on purpose: `tokenizers` reads a *list* of two
    # strings as a pre-tokenized single sequence, not as a pair.
    encode_inputs = [(query, document) for query, document in pairs]
    return _stack_encodings(backend.encode_batch(encode_inputs), seq_len)


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
