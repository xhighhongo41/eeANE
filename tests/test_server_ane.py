"""Integration tests against the real Core ML artifacts (T4 of v0.4実装計画.md §4.8).

These check the wiring (HTTP -> engine -> ANE -> HTTP) on the actual
compiled models; the heavy accuracy work belongs to tools/verify_server.py.
The whole module is skipped when the artifacts or the HuggingFace model
directories are absent, so CI-like environments stay green.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from eeane.config import default_config
from eeane.server import create_app

_CONFIG = default_config()
_EMBEDDING = _CONFIG.embedding_model
_RERANKER = _CONFIG.reranker_model
assert _RERANKER is not None, "the built-in default configuration always has a reranker"

_REQUIRED_PATHS = [
    _EMBEDDING.model_dir,
    _RERANKER.model_dir,
    *_EMBEDDING.artifacts.values(),
    *_RERANKER.artifacts.values(),
]
if not all(path.exists() for path in _REQUIRED_PATHS):
    pytest.skip(
        "Core ML artifacts or HuggingFace model directories are missing",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Client backed by the real engine, loaded once for the whole module."""
    with TestClient(create_app(_CONFIG)) as test_client:
        yield test_client


def test_health_lists_the_real_buckets(client: TestClient) -> None:
    """/health must expose the buckets of the loaded artifacts."""
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["models"] == {
        "embedding": list(_EMBEDDING.buckets),
        "reranker": list(_RERANKER.buckets),
    }


def test_embeddings_return_finite_unit_vectors(client: TestClient) -> None:
    """Short Japanese texts must yield finite, unit-norm 768-d embeddings."""
    texts = ["検索クエリ: 東京の天気", "検索文書: 今日の東京は晴れときどき曇りです。"]

    response = client.post("/v1/embeddings", json={"input": texts})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["data"]) == len(texts)
    for item in payload["data"]:
        vector = np.asarray(item["embedding"], dtype=np.float64)
        assert vector.shape == (768,)
        assert np.all(np.isfinite(vector))
        # The default config normalizes embeddings (Infinity parity).
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-3)
    # Both inputs are far below 128 tokens, so they route to the S128 bucket.
    assert payload["usage"]["prompt_tokens"] <= 2 * 128


def test_concurrent_requests_hitting_different_buckets_succeed(client: TestClient) -> None:
    """Parallel requests must not corrupt the shared tokenizer/model state.

    FastAPI runs the sync endpoints in a thread pool, so two requests can
    tokenize at the same time. Fast tokenizers keep mutable padding state,
    which used to fail with "RuntimeError: Already borrowed" when the two
    requests asked for different sequence lengths.
    """
    short_text = "東京の天気"
    long_text = "これは長めの日本語の文章です。" * 30  # routes to a larger bucket

    def post(text: str) -> tuple[int, int]:
        """POST one embedding request and return (status code, vector length)."""
        response = client.post("/v1/embeddings", json={"input": [text]})
        length = len(response.json()["data"][0]["embedding"]) if response.status_code == 200 else 0
        return response.status_code, length

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(post, [short_text, long_text] * 4))

    assert outcomes == [(200, 768)] * 8


def test_rerank_returns_sorted_probabilities(client: TestClient) -> None:
    """Rerank must score every document in (0, 1) and sort them descending."""
    documents = [
        "今日の東京は晴れときどき曇りで、最高気温は三十度の見込みです。",
        "パスタを茹でる時間は袋の表示より一分短くするとよい。",
        "この会社の四半期決算は増収増益となった。",
    ]

    response = client.post("/rerank", json={"query": "東京の天気を教えて", "documents": documents})

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == len(documents)
    assert sorted(result["index"] for result in results) == list(range(len(documents)))
    scores = [result["relevance_score"] for result in results]
    assert all(0.0 < score < 1.0 for score in scores)
    assert scores == sorted(scores, reverse=True)
