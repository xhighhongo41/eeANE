"""Tests for eeane.runtime."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from eeane.config import default_config
from eeane.runtime import (
    base64_to_floats,
    count_pair_tokens,
    count_text_tokens,
    floats_to_base64,
    l2_normalize,
    load_frozen_tokenizer,
    select_bucket,
    sigmoid,
    tokenize_pairs,
    tokenize_texts,
)

_EMBEDDING_TOKENIZER_PATH = default_config().embedding_model.tokenizer
_MODEL_AVAILABLE = _EMBEDDING_TOKENIZER_PATH.is_file()

_BUCKETS = (128, 512, 1024)


@pytest.mark.parametrize(
    ("n_tokens", "expected"),
    [
        (1, (128, False)),
        (128, (128, False)),
        (129, (512, False)),
        (512, (512, False)),
        (513, (1024, False)),
        (1024, (1024, False)),
        (1025, (1024, True)),
        (0, (128, False)),
    ],
)
def test_select_bucket_boundaries(n_tokens: int, expected: tuple[int, bool]) -> None:
    """select_bucket must return the smallest bucket >= n_tokens, truncating above the max."""
    assert select_bucket(n_tokens, _BUCKETS) == expected


def test_sigmoid_zero_is_one_half() -> None:
    """sigmoid(0) must equal exactly 0.5."""
    result = sigmoid(np.array([0.0]))

    np.testing.assert_allclose(result, [0.5])


def test_sigmoid_large_inputs_saturate_without_overflow_warning() -> None:
    """sigmoid must not raise/warn on overflow and must saturate to 0/1 near +-1000."""
    x = np.array([1000.0, -1000.0])

    with warnings.catch_warnings():
        # Promote RuntimeWarning (e.g. numpy overflow in exp) to an error so
        # any overflow in a naive 1 / (1 + exp(-x)) implementation is caught.
        warnings.simplefilter("error")
        result = sigmoid(x)

    np.testing.assert_allclose(result, [1.0, 0.0], atol=1e-9)


def test_sigmoid_monotonic_and_shape_preserving() -> None:
    """sigmoid must be strictly increasing and preserve the input shape."""
    x = np.linspace(-10.0, 10.0, 21)

    result = sigmoid(x)

    assert result.shape == x.shape
    assert np.all(np.diff(result) > 0)


def test_l2_normalize_unit_norm_and_direction() -> None:
    """l2_normalize must produce unit-norm rows pointing in the original direction."""
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((5, 8)).astype(np.float32)

    normalized = l2_normalize(matrix)

    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, np.ones(5), atol=1e-6)

    dot = np.sum(matrix * normalized, axis=1)
    cosine = dot / (np.linalg.norm(matrix, axis=1) * norms)
    np.testing.assert_allclose(cosine, np.ones(5), atol=1e-6)

    assert normalized.dtype == np.float32


def test_l2_normalize_zero_row_has_no_nan_or_inf() -> None:
    """l2_normalize must not produce nan/inf for an all-zero row (eps floor)."""
    matrix = np.zeros((2, 4), dtype=np.float32)

    normalized = l2_normalize(matrix)

    assert np.all(np.isfinite(normalized))


def test_base64_roundtrip_is_bit_exact() -> None:
    """floats_to_base64 -> base64_to_floats must round-trip float32 bits exactly."""
    rng = np.random.default_rng(1)
    vector = rng.standard_normal(768).astype(np.float32)

    decoded = base64_to_floats(floats_to_base64(vector))

    assert np.array_equal(vector, decoded)


# --- frozen tokenizer loading (no model artifacts needed) ----------------


def _write_toy_tokenizer(path: Path, *, with_padding: bool) -> Path:
    """Write a minimal word-level tokenizer.json, with or without a padding section.

    Args:
        path: Destination file.
        with_padding: Whether to bake a padding section (as
            ``eeane compile`` does) before saving.

    Returns:
        ``path``, for chaining.
    """
    vocab = {"<pad>": 0, "a": 1, "b": 2, "c": 3}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    if with_padding:
        tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    tokenizer.save(str(path))
    return path


def test_load_frozen_tokenizer_reads_pad_settings(tmp_path: Path) -> None:
    """The pad id/token/direction must come from the frozen file's padding section."""
    path = _write_toy_tokenizer(tmp_path / "tokenizer.json", with_padding=True)

    frozen = load_frozen_tokenizer(path)

    assert frozen.pad_id == 0
    assert frozen.pad_token == "<pad>"
    assert frozen.pad_type_id == 0
    assert frozen.pad_direction == "right"


def test_load_frozen_tokenizer_without_padding_section_raises(tmp_path: Path) -> None:
    """A plain (non-frozen) tokenizer.json must be rejected with an `eeane compile` hint."""
    path = _write_toy_tokenizer(tmp_path / "tokenizer.json", with_padding=False)

    with pytest.raises(ValueError, match="eeane compile"):
        load_frozen_tokenizer(path)


def test_load_frozen_tokenizer_survives_a_counting_call(tmp_path: Path) -> None:
    """Counting clears the backend's padding state; the captured settings must survive it."""
    path = _write_toy_tokenizer(tmp_path / "tokenizer.json", with_padding=True)
    frozen = load_frozen_tokenizer(path)

    assert count_text_tokens(frozen, "a b c") == 3
    batch = tokenize_texts(frozen, ["a b c"], 5)

    assert batch["input_ids"].tolist() == [[1, 2, 3, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0]]


@pytest.mark.parametrize("seq_len", [0, -1])
def test_tokenize_rejects_non_positive_seq_len(tmp_path: Path, seq_len: int) -> None:
    """A zero/negative bucket length must raise instead of reaching the Rust tokenizer."""
    path = _write_toy_tokenizer(tmp_path / "tokenizer.json", with_padding=True)
    frozen = load_frozen_tokenizer(path)

    with pytest.raises(ValueError, match="seq_len"):
        tokenize_texts(frozen, ["a b"], seq_len)
    with pytest.raises(ValueError, match="seq_len"):
        tokenize_pairs(frozen, [("a", "b")], seq_len)


def test_tokenize_empty_batch_keeps_the_two_dimensional_shape(tmp_path: Path) -> None:
    """An empty input list must still yield (0, seq_len) int32 arrays."""
    path = _write_toy_tokenizer(tmp_path / "tokenizer.json", with_padding=True)
    frozen = load_frozen_tokenizer(path)

    texts = tokenize_texts(frozen, [], 4)
    pairs = tokenize_pairs(frozen, [], 4)

    for batch in (texts, pairs):
        assert batch["input_ids"].shape == (0, 4)
        assert batch["attention_mask"].shape == (0, 4)
        assert batch["input_ids"].dtype == np.int32
        assert batch["attention_mask"].dtype == np.int32


# --- real ruri-v3-310m tokenizer -----------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    """Load the frozen ruri-v3-310m tokenizer once for the tests below."""
    return load_frozen_tokenizer(_EMBEDDING_TOKENIZER_PATH)


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="frozen ruri-v3-310m tokenizer not found")
def test_tokenize_texts_shape_and_dtype(tokenizer) -> None:
    """tokenize_texts must produce fixed-shape int32 arrays with only the two expected keys."""
    batch = tokenize_texts(tokenizer, ["これはテストです。", "もう一つの短い日本語の文です。"], 128)

    assert set(batch.keys()) == {"input_ids", "attention_mask"}
    assert batch["input_ids"].shape == (2, 128)
    assert batch["attention_mask"].shape == (2, 128)
    assert batch["input_ids"].dtype == np.int32
    assert batch["attention_mask"].dtype == np.int32


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="frozen ruri-v3-310m tokenizer not found")
def test_count_text_tokens_short_text(tokenizer) -> None:
    """count_text_tokens must include the <s>/</s> special tokens around the content."""
    n = count_text_tokens(tokenizer, "テスト")

    assert n >= 3


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="frozen ruri-v3-310m tokenizer not found")
def test_count_pair_tokens_exceeds_either_side_alone(tokenizer) -> None:
    """count_pair_tokens must exceed the token count of either sequence alone."""
    query = "日本の首都はどこですか。"
    document = "東京は日本の首都であり、政治と経済の中心地として発展してきた都市である。"

    pair_count = count_pair_tokens(tokenizer, query, document)

    assert pair_count > count_text_tokens(tokenizer, query)
    assert pair_count > count_text_tokens(tokenizer, document)


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="frozen ruri-v3-310m tokenizer not found")
def test_long_text_exceeds_1024_tokens_and_fits_after_bucket_truncation(tokenizer) -> None:
    """A long text must exceed 1024 tokens, and tokenize_texts must truncate it to seq_len."""
    long_text = "今日は天気が良いです。" * 300

    n = count_text_tokens(tokenizer, long_text)
    assert n > 1024

    batch = tokenize_texts(tokenizer, [long_text], 128)
    assert batch["input_ids"].shape == (1, 128)
