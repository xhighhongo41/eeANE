"""Token-count-bounded text chunking for the eeANE PoC (v0.3実装計画.md §4.3).

Splits a document into substrings ("chunks") of the original text such
that each chunk re-tokenizes (with special tokens included) to at most
``chunk_tokens`` tokens. Uses the offset-mapping approach rather than
decode-round-tripping, because SentencePiece decoding does not always
reconstruct the exact original text, which would break the coverage
guarantee this module provides.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

# Reserved headroom (in tokens) absorbed by the shrink-and-retry validation
# pass below, to cover boundary-token drift that can occur when a chunk
# substring is re-tokenized in isolation (e.g. a token that spans the chunk
# boundary in the full-text tokenization splits differently once the text is
# cut, or a skipped boundary character such as a newline attaches an extra
# token to the next chunk).
SAFETY_MARGIN = 2

# Maximum number of shrink-and-retry attempts (beyond the initial check) when
# the re-tokenized chunk still exceeds chunk_tokens.
_MAX_RETRIES = 8


def chunk_by_tokens(
    tokenizer: PreTrainedTokenizerBase, text: str, chunk_tokens: int, stride: int = 0
) -> list[str]:
    """Split text into chunks bounded by a re-tokenized token count.

    Args:
        tokenizer: A fast tokenizer (``tokenizer.is_fast`` must be True)
            that supports ``return_offsets_mapping``, e.g. the tokenizer
            returned by :func:`poc.common.load_tokenizer`.
        text: Full input text to split.
        chunk_tokens: Maximum token count per chunk, including the special
            tokens (``<s>``/``</s>``) added when the chunk is re-tokenized
            on its own.
        stride: Number of overlapping tokens between consecutive chunks.
            Must satisfy ``0 <= stride < effective_window``, where
            ``effective_window = chunk_tokens - 2 - SAFETY_MARGIN``. The
            default of 0 produces non-overlapping, contiguous chunks whose
            concatenation exactly reproduces ``text``.

    Returns:
        List of chunk substrings of ``text``, in document order. Returns
        an empty list if ``text`` tokenizes to zero tokens (e.g. an empty
        string).

    Raises:
        ValueError: If ``tokenizer`` is not a fast tokenizer, if
            ``chunk_tokens`` is too small to leave a positive effective
            window, or if ``stride`` is out of range.
        RuntimeError: If the shrink-and-retry validation loop exhausts its
            retry budget (or empties its window) for a given chunk.
    """
    if not tokenizer.is_fast:
        raise ValueError(
            "chunk_by_tokens requires a fast tokenizer (backed by tokenizer.json) "
            "that supports return_offsets_mapping; got a slow tokenizer instead."
        )

    effective_window = chunk_tokens - 2 - SAFETY_MARGIN
    if effective_window < 1:
        raise ValueError(
            f"chunk_tokens={chunk_tokens} is too small: effective_window "
            f"({effective_window}) must be at least 1 (chunk_tokens - 2 for "
            f"<s>/</s> - SAFETY_MARGIN={SAFETY_MARGIN})."
        )
    if not (0 <= stride < effective_window):
        raise ValueError(
            f"stride={stride} must satisfy 0 <= stride < effective_window ({effective_window})."
        )

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets: list[tuple[int, int]] = encoded["offset_mapping"]
    if not offsets:
        return []

    if stride == 0:
        return _chunk_contiguous(tokenizer, text, offsets, effective_window, chunk_tokens)
    return _chunk_with_overlap(tokenizer, text, offsets, effective_window, stride, chunk_tokens)


def _chunk_contiguous(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    offsets: list[tuple[int, int]],
    effective_window: int,
    chunk_tokens: int,
) -> list[str]:
    """Build non-overlapping chunks that concatenate back to ``text``.

    Each chunk starts exactly where the previous one ended (the first
    starts at char 0, the last ends at ``len(text)``), so any characters
    skipped by the tokenizer at a chunk boundary (e.g. whitespace) are
    carried into the following chunk rather than dropped. This makes
    coverage a structural guarantee independent of tokenizer quirks.

    Args:
        tokenizer: Fast tokenizer used for the validation re-tokenize pass.
        text: Full input text.
        offsets: Offset mapping for ``text`` tokenized without special
            tokens.
        effective_window: Maximum raw (non-special) tokens per window.
        chunk_tokens: Maximum re-tokenized (special-included) token count.

    Returns:
        List of chunk substrings, in document order.
    """
    num_tokens = len(offsets)
    chunks: list[str] = []
    char_pos = 0
    tok_idx = 0
    while tok_idx < num_tokens:
        window_end = min(tok_idx + effective_window, num_tokens) - 1
        chunk, final_end, end_char = _shrink_and_validate(
            tokenizer,
            text,
            offsets,
            start_char=char_pos,
            window_start=tok_idx,
            window_end=window_end,
            chunk_tokens=chunk_tokens,
            extend_to_text_end=True,
        )
        chunks.append(chunk)
        char_pos = end_char
        tok_idx = final_end + 1
    return chunks


def _chunk_with_overlap(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    offsets: list[tuple[int, int]],
    effective_window: int,
    stride: int,
    chunk_tokens: int,
) -> list[str]:
    """Build overlapping chunks, each window taken as-is from its offsets.

    Args:
        tokenizer: Fast tokenizer used for the validation re-tokenize pass.
        text: Full input text.
        offsets: Offset mapping for ``text`` tokenized without special
            tokens.
        effective_window: Maximum raw (non-special) tokens per window.
        stride: Number of overlapping tokens requested between consecutive
            windows.
        chunk_tokens: Maximum re-tokenized (special-included) token count.

    Returns:
        List of chunk substrings, in document order, with overlapping
        text between consecutive chunks.
    """
    num_tokens = len(offsets)
    chunks: list[str] = []
    tok_idx = 0
    while tok_idx < num_tokens:
        window_end = min(tok_idx + effective_window, num_tokens) - 1
        start_char = offsets[tok_idx][0]
        chunk, final_end, _ = _shrink_and_validate(
            tokenizer,
            text,
            offsets,
            start_char=start_char,
            window_start=tok_idx,
            window_end=window_end,
            chunk_tokens=chunk_tokens,
            extend_to_text_end=False,
        )
        chunks.append(chunk)
        if final_end == num_tokens - 1:
            break
        # Advance the window start by (confirmed window size - stride),
        # guarded to always move forward by at least one token even if the
        # window shrank below the requested stride during validation.
        window_len = final_end - tok_idx + 1
        tok_idx += max(window_len - stride, 1)
    return chunks


def _shrink_and_validate(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    offsets: list[tuple[int, int]],
    start_char: int,
    window_start: int,
    window_end: int,
    chunk_tokens: int,
    extend_to_text_end: bool,
) -> tuple[str, int, int]:
    """Slice, re-tokenize, and shrink a window until it fits chunk_tokens.

    Args:
        tokenizer: Fast tokenizer used for the validation re-tokenize pass.
        text: Full input text.
        offsets: Offset mapping for ``text`` tokenized without special
            tokens.
        start_char: Character offset where the chunk begins.
        window_start: Token index (inclusive) where the window begins.
        window_end: Token index (inclusive) where the window initially
            ends; shrunk downward on validation failure.
        chunk_tokens: Maximum re-tokenized (special-included) token count.
        extend_to_text_end: If True, and the window's end token is the
            last token of the whole document, the chunk end is extended to
            ``len(text)`` instead of that token's offset end (used by the
            contiguous stride=0 mode to cover trailing untokenized
            characters).

    Returns:
        Tuple of (chunk text, final window-end token index, chunk end
        character offset).

    Raises:
        RuntimeError: If the window empties out or the retry budget is
            exhausted before the re-tokenized chunk fits chunk_tokens.
    """
    num_tokens = len(offsets)
    end = window_end
    attempts = 0
    while True:
        if end < window_start:
            raise RuntimeError(
                "chunk_by_tokens: shrinking the window emptied it before the "
                f"re-tokenized chunk fit within chunk_tokens={chunk_tokens}; the "
                "tokenizer may be adding an unexpectedly large number of tokens."
            )
        if extend_to_text_end and end == num_tokens - 1:
            end_char = len(text)
        else:
            end_char = offsets[end][1]
        chunk = text[start_char:end_char]
        retokenized_len = len(tokenizer(chunk)["input_ids"])
        if retokenized_len <= chunk_tokens:
            return chunk, end, end_char
        if attempts >= _MAX_RETRIES:
            raise RuntimeError(
                f"chunk_by_tokens: exceeded {_MAX_RETRIES} shrink-and-retry attempts "
                f"without fitting chunk_tokens={chunk_tokens} (last re-tokenized "
                f"length was {retokenized_len})."
            )
        end -= 1
        attempts += 1
