"""Tests for eeane.runtime (T2 of v0.4実装計画.md §4.2, §4.3)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from transformers import AutoTokenizer

from eeane.config import default_config
from eeane.runtime import (
    base64_to_floats,
    count_pair_tokens,
    count_text_tokens,
    floats_to_base64,
    l2_normalize,
    select_bucket,
    sigmoid,
    tokenize_texts,
)

_EMBEDDING_MODEL_DIR = default_config().embedding_model.model_dir
_MODEL_AVAILABLE = _EMBEDDING_MODEL_DIR.exists()

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


@pytest.fixture(scope="module")
def tokenizer():
    """Load the ruri-v3-310m tokenizer once for the tokenizer-dependent tests below."""
    return AutoTokenizer.from_pretrained(_EMBEDDING_MODEL_DIR)


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="ruri-v3-310m model directory not found")
def test_tokenize_texts_shape_and_dtype(tokenizer) -> None:
    """tokenize_texts must produce fixed-shape int32 arrays with only the two expected keys."""
    batch = tokenize_texts(tokenizer, ["これはテストです。", "もう一つの短い日本語の文です。"], 128)

    assert set(batch.keys()) == {"input_ids", "attention_mask"}
    assert batch["input_ids"].shape == (2, 128)
    assert batch["attention_mask"].shape == (2, 128)
    assert batch["input_ids"].dtype == np.int32
    assert batch["attention_mask"].dtype == np.int32


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="ruri-v3-310m model directory not found")
def test_count_text_tokens_short_text(tokenizer) -> None:
    """count_text_tokens must include the <s>/</s> special tokens around the content."""
    n = count_text_tokens(tokenizer, "テスト")

    assert n >= 3


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="ruri-v3-310m model directory not found")
def test_count_pair_tokens_exceeds_either_side_alone(tokenizer) -> None:
    """count_pair_tokens must exceed the token count of either sequence alone."""
    query = "日本の首都はどこですか。"
    document = "東京は日本の首都であり、政治と経済の中心地として発展してきた都市である。"

    pair_count = count_pair_tokens(tokenizer, query, document)

    assert pair_count > count_text_tokens(tokenizer, query)
    assert pair_count > count_text_tokens(tokenizer, document)


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="ruri-v3-310m model directory not found")
def test_long_text_exceeds_1024_tokens_and_fits_after_bucket_truncation(tokenizer) -> None:
    """A long text must exceed 1024 tokens, and tokenize_texts must truncate it to seq_len."""
    long_text = "今日は天気が良いです。" * 300

    n = count_text_tokens(tokenizer, long_text)
    assert n > 1024

    batch = tokenize_texts(tokenizer, [long_text], 128)
    assert batch["input_ids"].shape == (1, 128)
