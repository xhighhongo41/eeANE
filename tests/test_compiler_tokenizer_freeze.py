"""Tests for eeane.compiler.tokenizer_freeze (v0.6 T6, 開発資料/v0.6実装計画.md §4.6).

The freeze/verify gate is what makes it safe for ``eeane serve`` to drop
``transformers``: these tests run the gate on the real ruri-v3 tokenizers
over the fixed Aozora corpus and every deployed bucket length. They are
skipped automatically when the HuggingFace model directories are absent
(CI-like environments), like the other real-artifact test modules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eeane.compiler.tokenizer_freeze import (
    TokenizerFreezeError,
    freeze_tokenizer,
    verify_frozen_tokenizer,
)
from eeane.runtime import load_frozen_tokenizer
from poc.common import (
    DEFAULT_MODEL_DIR,
    DEFAULT_RERANKER_DIR,
    load_corpus_paragraphs,
    load_rerank_queries,
)

_MODEL_DIRS = (DEFAULT_MODEL_DIR, DEFAULT_RERANKER_DIR)
if not all((path / "tokenizer.json").is_file() for path in _MODEL_DIRS):
    pytest.skip("HuggingFace model directories are missing", allow_module_level=True)

# Buckets the two models are actually deployed with (eeane.config.default_config).
_EMBEDDING_BUCKETS = [128, 512, 1024]
_RERANKER_BUCKETS = [512, 1024]

# Short inputs mirroring the conversion sanity texts, plus the degenerate
# cases (empty, whitespace-only, single character, far-past-the-longest-
# bucket) that must tokenize identically as well.
_SANITY_TEXTS = [
    "検索クエリ: 東京の天気",
    "検索文書: 今日の東京は晴れときどき曇りです。",
    "テスト",
    "",
    "   ",
    "a",
    "今日は天気が良いです。" * 300,
]


def _all_corpus_paragraphs() -> list[str]:
    """Return every paragraph of all three works (not just the benchmark subset)."""
    # max_kokoro is effectively unbounded here, and min_chars=1 keeps the
    # short paragraphs the benchmarks drop: the gate must cover them too.
    return load_corpus_paragraphs(max_kokoro=10**6, min_chars=1)


def _verification_texts() -> list[str]:
    """Build the single-sequence verification inputs (sanity texts + whole corpus)."""
    return _SANITY_TEXTS + _all_corpus_paragraphs()


def _verification_pairs() -> list[tuple[str, str]]:
    """Build the (query, document) verification inputs from the fixed corpus."""
    queries = [entry["query"] for entry in load_rerank_queries()]
    return [(query, document) for query in queries for document in _all_corpus_paragraphs()]


@pytest.fixture(scope="module")
def texts() -> list[str]:
    """Single-sequence verification inputs, built once for the module."""
    return _verification_texts()


@pytest.fixture(scope="module")
def pairs() -> list[tuple[str, str]]:
    """Pair verification inputs, built once for the module."""
    return _verification_pairs()


@pytest.fixture(scope="module")
def frozen_embedding(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Freeze the embedding model's tokenizer once for the module."""
    out_path = tmp_path_factory.mktemp("frozen-embedding") / "tokenizer.json"
    freeze_tokenizer(DEFAULT_MODEL_DIR, out_path)
    return out_path


@pytest.fixture(scope="module")
def frozen_reranker(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Freeze the reranker model's tokenizer once for the module."""
    out_path = tmp_path_factory.mktemp("frozen-reranker") / "tokenizer.json"
    freeze_tokenizer(DEFAULT_RERANKER_DIR, out_path)
    return out_path


# --- freeze ---------------------------------------------------------------


def test_freeze_tokenizer_bakes_padding_but_not_truncation(frozen_embedding: Path) -> None:
    """The frozen file must carry pad_id/pad_token and no truncation section."""
    raw = json.loads(frozen_embedding.read_text(encoding="utf-8"))

    assert raw["padding"]["pad_token"] == "<pad>"
    assert isinstance(raw["padding"]["pad_id"], int)
    # The bucket length is a per-request decision, so no fixed length or
    # truncation rule may be frozen into the file.
    assert raw["padding"]["strategy"] == "BatchLongest"
    assert raw["truncation"] is None


def test_freeze_tokenizer_reports_what_it_froze(tmp_path: Path) -> None:
    """freeze_tokenizer must report the tokenizer class and the baked pad settings."""
    out_path = tmp_path / "nested" / "tokenizer.json"

    info = freeze_tokenizer(DEFAULT_RERANKER_DIR, out_path)

    assert out_path.is_file()  # parent directories are created
    assert info["tokenizer_class"] == "LlamaTokenizerFast"
    assert info["pad_token"] == "<pad>"
    assert info["padding_direction"] == "right"
    assert Path(info["path"]) == out_path


def test_freeze_tokenizer_does_not_touch_the_model_directory(tmp_path: Path) -> None:
    """The input model directory is read-only (v0.6実装計画.md §2-11)."""
    before = {path.name: path.stat().st_mtime_ns for path in DEFAULT_MODEL_DIR.iterdir()}

    freeze_tokenizer(DEFAULT_MODEL_DIR, tmp_path / "tokenizer.json")

    after = {path.name: path.stat().st_mtime_ns for path in DEFAULT_MODEL_DIR.iterdir()}
    assert after == before


def test_frozen_tokenizer_is_loadable_by_the_runtime(frozen_embedding: Path) -> None:
    """The runtime loader must accept the frozen file and read its pad settings."""
    frozen = load_frozen_tokenizer(frozen_embedding)

    assert frozen.pad_token == "<pad>"
    assert frozen.pad_id >= 0
    assert frozen.pad_direction == "right"


# --- verify (the compile gate) --------------------------------------------


def test_verify_embedding_tokenizer_matches_every_bucket(
    frozen_embedding: Path, texts: list[str], pairs: list[tuple[str, str]]
) -> None:
    """The frozen embedding tokenizer must equal AutoTokenizer on the whole corpus."""
    report = verify_frozen_tokenizer(
        DEFAULT_MODEL_DIR, frozen_embedding, texts, pairs, _EMBEDDING_BUCKETS
    )

    assert report["passed"] is True
    assert report["buckets"] == _EMBEDDING_BUCKETS
    assert report["n_texts"] == len(texts)
    assert report["n_pairs"] == len(pairs)


def test_verify_reranker_tokenizer_matches_every_bucket(
    frozen_reranker: Path, texts: list[str], pairs: list[tuple[str, str]]
) -> None:
    """The frozen reranker tokenizer (dynamic Llama post_processor) must match too."""
    report = verify_frozen_tokenizer(
        DEFAULT_RERANKER_DIR, frozen_reranker, texts, pairs, _RERANKER_BUCKETS
    )

    assert report["passed"] is True
    assert report["buckets"] == _RERANKER_BUCKETS


def test_verify_detects_a_tampered_pad_id(tmp_path: Path, frozen_embedding: Path) -> None:
    """A frozen file whose pad_id was altered must fail the gate with details."""
    raw = json.loads(frozen_embedding.read_text(encoding="utf-8"))
    raw["padding"]["pad_id"] = raw["padding"]["pad_id"] + 1
    tampered = tmp_path / "tokenizer.json"
    tampered.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(TokenizerFreezeError) as excinfo:
        verify_frozen_tokenizer(DEFAULT_MODEL_DIR, tampered, ["短い文"], [], [128])

    message = str(excinfo.value)
    assert "input_ids" in message
    assert "bucket 128" in message
    assert excinfo.value.report["passed"] is False
    assert excinfo.value.report["mismatches"]


def test_verify_detects_a_missing_post_processor(tmp_path: Path, frozen_embedding: Path) -> None:
    """Dropping the post_processor (the <s>/</s> template) must fail the gate."""
    raw = json.loads(frozen_embedding.read_text(encoding="utf-8"))
    raw["post_processor"] = None
    tampered = tmp_path / "tokenizer.json"
    tampered.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(TokenizerFreezeError):
        verify_frozen_tokenizer(DEFAULT_MODEL_DIR, tampered, ["短い文"], [], [128])


@pytest.mark.parametrize("buckets", [[], [0], [-1], [128, 0]])
def test_verify_rejects_invalid_buckets(frozen_embedding: Path, buckets: list[int]) -> None:
    """An empty or non-positive bucket list must raise ValueError, not run a bogus check."""
    with pytest.raises(ValueError, match="bucket"):
        verify_frozen_tokenizer(DEFAULT_MODEL_DIR, frozen_embedding, ["文"], [], buckets)


def test_verify_accepts_an_empty_pair_list(frozen_embedding: Path) -> None:
    """Embedding-only models have no pairs to verify; that must not fail the gate."""
    report = verify_frozen_tokenizer(DEFAULT_MODEL_DIR, frozen_embedding, ["文"], [], [128])

    assert report["passed"] is True
    assert report["n_pairs"] == 0
