"""Tests for poc.chunking (T2 of v0.3実装計画.md §4.3)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from poc.chunking import SAFETY_MARGIN, chunk_by_tokens
from poc.common import CORPUS_DIR, DEFAULT_MODEL_DIR, load_tokenizer

_MODEL_AVAILABLE = (DEFAULT_MODEL_DIR / "config.json").exists()
_CORPUS_AVAILABLE = (CORPUS_DIR / "kokoro.txt").exists()

# Synthetic Japanese text (multiple short sentences) used for the short-text
# coverage check without depending on the corpus fixture.
_SHORT_TEXT = (
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"
    "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"
    "吾輩はここで始めて人間というものを見た。"
    "しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。"
    "この書生というのは時々我々を捕えて煮て食うという話である。"
)


@pytest.fixture(scope="module")
def tokenizer():
    """Load the ruri-v3-310m fast tokenizer once for all tests in this module."""
    if not _MODEL_AVAILABLE:
        pytest.skip("ruri-v3-310m model directory not found")
    return load_tokenizer(DEFAULT_MODEL_DIR)


@pytest.fixture(scope="module")
def kokoro_text() -> str:
    """Load the full kokoro.txt corpus text once for all tests in this module."""
    if not _CORPUS_AVAILABLE:
        pytest.skip("testdata/corpus/kokoro.txt not found (T3 pending)")
    return (CORPUS_DIR / "kokoro.txt").read_text(encoding="utf-8")


def test_chunk_by_tokens_coverage_short_synthetic_text(tokenizer) -> None:
    """stride=0 chunks of a short synthetic text must concatenate back to it."""
    chunks = chunk_by_tokens(tokenizer, _SHORT_TEXT, chunk_tokens=16)

    assert len(chunks) > 1
    assert "".join(chunks) == _SHORT_TEXT


@pytest.mark.parametrize("chunk_tokens", [128, 512])
def test_chunk_by_tokens_coverage_and_limit_kokoro(tokenizer, kokoro_text, chunk_tokens) -> None:
    """stride=0 chunks of kokoro.txt must cover the text and respect chunk_tokens."""
    chunks = chunk_by_tokens(tokenizer, kokoro_text, chunk_tokens=chunk_tokens)

    assert len(chunks) > 1
    assert "".join(chunks) == kokoro_text
    for chunk in chunks:
        retokenized_len = len(tokenizer(chunk)["input_ids"])
        assert retokenized_len <= chunk_tokens


def test_chunk_by_tokens_stride_overlap_increases_chunk_count(tokenizer, kokoro_text) -> None:
    """stride>0 must yield more, overlapping chunks than stride=0 for the same text."""
    excerpt = kokoro_text[:20000]

    no_stride_chunks = chunk_by_tokens(tokenizer, excerpt, chunk_tokens=128, stride=0)
    strided_chunks = chunk_by_tokens(tokenizer, excerpt, chunk_tokens=128, stride=32)

    assert len(strided_chunks) > len(no_stride_chunks)
    for prev_chunk, next_chunk in zip(strided_chunks, strided_chunks[1:], strict=False):
        # The overlapping token window guarantees shared characters between
        # consecutive windows, so the next chunk's leading text must appear
        # somewhere within the previous chunk.
        overlap_probe = next_chunk[:20]
        assert overlap_probe in prev_chunk


def test_chunk_by_tokens_short_input_returns_single_chunk(tokenizer) -> None:
    """Input that fits within chunk_tokens must be returned as [text] verbatim."""
    text = "これは短いテスト文です。"

    chunks = chunk_by_tokens(tokenizer, text, chunk_tokens=128)

    assert chunks == [text]


def test_chunk_by_tokens_empty_text_returns_empty_list(tokenizer) -> None:
    """Empty input must return an empty list rather than [""]."""
    assert chunk_by_tokens(tokenizer, "", chunk_tokens=128) == []


def test_chunk_by_tokens_rejects_slow_tokenizer() -> None:
    """A tokenizer with is_fast=False must raise ValueError instead of falling back."""
    fake_tokenizer = SimpleNamespace(is_fast=False)

    with pytest.raises(ValueError, match="fast tokenizer"):
        chunk_by_tokens(fake_tokenizer, "text", chunk_tokens=128)


def test_chunk_by_tokens_rejects_out_of_range_stride(tokenizer) -> None:
    """stride >= effective_window must raise ValueError."""
    chunk_tokens = 16
    effective_window = chunk_tokens - 2 - SAFETY_MARGIN

    with pytest.raises(ValueError, match="stride"):
        chunk_by_tokens(tokenizer, _SHORT_TEXT, chunk_tokens=chunk_tokens, stride=effective_window)
