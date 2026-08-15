"""Tests for poc.common."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from poc.common import (
    CORPUS_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_RERANKER_DIR,
    cosine_rowwise,
    load_corpus_paragraphs,
    load_rerank_queries,
    load_tokenizer,
    mean_pool,
    sigmoid_np,
    spearman,
    tokenize_batch,
    tokenize_pairs,
)

_MODEL_AVAILABLE = (DEFAULT_MODEL_DIR / "config.json").exists()
_CORPUS_AVAILABLE = (CORPUS_DIR / "kokoro.txt").exists()
_RERANKER_AVAILABLE = (DEFAULT_RERANKER_DIR / "config.json").exists()


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


def test_spearman_perfect_agreement() -> None:
    """spearman must return 1.0 for identically ordered score arrays."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    assert spearman(a, b) == pytest.approx(1.0)


def test_spearman_perfect_disagreement() -> None:
    """spearman must return -1.0 for exactly reverse-ordered score arrays."""
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])

    assert spearman(a, b) == pytest.approx(-1.0)


def test_spearman_partial_correlation() -> None:
    """spearman must match the classic no-tie rank-correlation formula.

    a and b are permutations of 1..5 (rank == value, no ties), so the
    textbook formula rho = 1 - 6*sum(d**2) / (n*(n**2-1)) applies directly:
    d = [0, -1, 1, -1, 1], sum(d**2) = 4, n = 5 -> rho = 1 - 24/120 = 0.8.
    """
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 3.0, 2.0, 5.0, 4.0])

    assert spearman(a, b) == pytest.approx(0.8)


def test_sigmoid_np_zero_is_one_half() -> None:
    """sigmoid_np(0) must equal exactly 0.5."""
    result = sigmoid_np(np.array([0.0]))

    np.testing.assert_allclose(result, [0.5])


def test_sigmoid_np_large_inputs_converge_without_overflow_warning() -> None:
    """sigmoid_np must not raise/warn on overflow and must saturate to 0/1."""
    x = np.array([1e4, -1e4, 700.0, -700.0])

    with warnings.catch_warnings():
        # Promote RuntimeWarning (e.g. numpy overflow in exp) to an error so
        # any overflow in a naive 1 / (1 + exp(-x)) implementation is caught.
        warnings.simplefilter("error")
        result = sigmoid_np(x)

    np.testing.assert_allclose(result, [1.0, 0.0, 1.0, 0.0], atol=1e-9)


def test_load_rerank_queries_contents() -> None:
    """load_rerank_queries must return the 9 hand-written queries."""
    queries = load_rerank_queries()

    assert len(queries) == 9
    expected_keys = {"id", "query", "source_work"}
    for entry in queries:
        assert set(entry.keys()) == expected_keys
    assert {entry["source_work"] for entry in queries} == {"kumonoito", "sangetsuki", "kokoro"}


@pytest.mark.skipif(
    not _RERANKER_AVAILABLE, reason="ruri-v3-reranker-310m model directory not found"
)
def test_tokenize_pairs_shape_and_pair_template() -> None:
    """tokenize_pairs must produce the <s> q </s> <s> d </s> template."""
    tokenizer = load_tokenizer(DEFAULT_RERANKER_DIR)
    seq_len = 64
    pairs = [
        (
            "日本の首都はどこですか。",
            "東京は日本の首都であり、政治と経済の中心地として発展してきた都市である。",
        ),
        (
            "機械学習の推論を高速化する方法",
            "毎朝のコーヒーにはカフェインが含まれており集中力を高めてくれる。",
        ),
    ]

    batch = tokenize_pairs(tokenizer, pairs, seq_len)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    assert input_ids.shape == (len(pairs), seq_len)
    assert attention_mask.shape == (len(pairs), seq_len)
    assert input_ids.dtype == np.int32
    assert attention_mask.dtype == np.int32

    for row_ids in input_ids:
        # Every sequence must start with the <s> token (id=1).
        assert row_ids[0] == 1
        eos_positions = np.flatnonzero(row_ids == 2)  # </s>
        # Exactly two </s>: one closing the query, one closing the document.
        assert len(eos_positions) == 2
        # The query's </s> must be immediately followed by the document's
        # <s>, confirming the <s> q </s> <s> d </s> template.
        first_eos = eos_positions[0]
        assert row_ids[first_eos + 1] == 1
