"""Tests for poc.common (T4: PoC common module)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from poc.common import (
    CORPUS_DIR,
    DEFAULT_MODEL_DIR,
    cosine_rowwise,
    load_corpus_paragraphs,
    load_tokenizer,
    mean_pool,
    tokenize_batch,
)

_MODEL_AVAILABLE = (DEFAULT_MODEL_DIR / "config.json").exists()
_CORPUS_AVAILABLE = (CORPUS_DIR / "kokoro.txt").exists()


def test_mean_pool_matches_manual_average() -> None:
    """mean_pool must equal the manual average of unmasked positions only."""
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 4)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ]
    )

    result = mean_pool(hidden, attention_mask)

    expected = torch.stack(
        [
            hidden[0].mean(dim=0),
            hidden[1, :3].mean(dim=0),
        ]
    )
    assert torch.allclose(result, expected, atol=1e-6)


def test_cosine_rowwise_known_vectors() -> None:
    """cosine_rowwise must return 1.0/0.0/-1.0 for identical/orthogonal/opposite pairs."""
    a = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    result = cosine_rowwise(a, b)

    np.testing.assert_allclose(result, [1.0, 0.0, -1.0], atol=1e-6)


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="ruri-v3-310m model directory not found")
def test_tokenize_batch_shape_and_special_tokens() -> None:
    """tokenize_batch must produce fixed-shape int32 arrays with <s>/</s> placed correctly."""
    tokenizer = load_tokenizer(DEFAULT_MODEL_DIR)
    seq_len = 32
    texts = ["これはテストです。", "もう一つの短い日本語の文です。"]

    batch = tokenize_batch(tokenizer, texts, seq_len)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    assert input_ids.shape == (len(texts), seq_len)
    assert attention_mask.shape == (len(texts), seq_len)
    assert input_ids.dtype == np.int32
    assert attention_mask.dtype == np.int32

    for row_ids, row_mask in zip(input_ids, attention_mask, strict=True):
        # Every sequence must start with the <s> token (id=1).
        assert row_ids[0] == 1
        valid_len = int(row_mask.sum())
        # The </s> token (id=2) must sit right before the padding region.
        assert row_ids[valid_len - 1] == 2
        if valid_len < seq_len:
            assert row_mask[valid_len] == 0


@pytest.mark.skipif(
    not _CORPUS_AVAILABLE, reason="testdata/corpus/kokoro.txt not found (T3 pending)"
)
def test_load_corpus_paragraphs_deterministic() -> None:
    """load_corpus_paragraphs must be deterministic and respect the min_chars filter."""
    first = load_corpus_paragraphs()
    second = load_corpus_paragraphs()

    assert len(first) > 0
    assert first == second
    assert all(len(paragraph) >= 40 for paragraph in first)
