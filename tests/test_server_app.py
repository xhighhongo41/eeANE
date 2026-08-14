"""Tests for eeane.server with a stub engine (T4 of v0.4実装計画.md §4.8).

These tests never touch Core ML: a deterministic in-memory engine is
injected into :func:`eeane.server.create_app`, so the endpoints' shapes,
sorting, encodings and error handling can be checked in any environment.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from eeane import runtime
from eeane.engine import EmbeddingBatch, RerankBatch
from eeane.server import create_app

# Stub vectors are intentionally low-dimensional: the endpoints treat the
# width as opaque, so 4 keeps the expected values readable.
_STUB_DIM = 4


def stub_vector(text: str) -> np.ndarray:
    """Build the deterministic embedding the stub returns for ``text``.

    Args:
        text: Input text; only its length matters.

    Returns:
        ``[len(text), len(text) + 1, ...]`` as a float32 vector of length
        :data:`_STUB_DIM`.
    """
    return np.arange(_STUB_DIM, dtype=np.float32) + float(len(text))


def stub_logit(document: str) -> float:
    """Build the deterministic logit the stub returns for ``document``.

    Args:
        document: Candidate document; only its length matters.

    Returns:
        A value in ``[-2.0, 2.0]`` derived from ``len(document)``.
    """
    return float(len(document) % 5) - 2.0


class StubEngine:
    """Deterministic engine implementing the InferenceEngine protocol."""

    embedding_buckets: tuple[int, ...] = (128, 512, 1024)
    reranker_buckets: tuple[int, ...] = (512,)

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        """Return :func:`stub_vector` for each text, counting 1 token per character."""
        if not texts:
            return EmbeddingBatch(
                vectors=np.empty((0, _STUB_DIM), dtype=np.float32),
                used_tokens=[],
                orig_tokens=[],
                buckets=[],
                truncated_indices=[],
            )
        tokens = [len(text) for text in texts]
        return EmbeddingBatch(
            vectors=np.stack([stub_vector(text) for text in texts]),
            used_tokens=list(tokens),
            orig_tokens=list(tokens),
            buckets=[runtime.select_bucket(n, self.embedding_buckets)[0] for n in tokens],
            truncated_indices=[],
        )

    def rerank(self, query: str, documents: list[str]) -> RerankBatch:
        """Return :func:`stub_logit` for each document, ignoring the query's content."""
        tokens = [len(query) + len(document) for document in documents]
        return RerankBatch(
            logits=np.asarray([stub_logit(document) for document in documents], dtype=np.float32),
            used_tokens=list(tokens),
            orig_tokens=list(tokens),
            truncated_indices=[],
        )


class TruncatingStubEngine(StubEngine):
    """Stub reporting every input as truncated, to exercise the warning path."""

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        """Pretend each text exceeded the largest bucket and was cut down to it."""
        batch = super().embed(texts)
        batch.orig_tokens = [2000] * len(texts)
        batch.used_tokens = [1024] * len(texts)
        batch.buckets = [1024] * len(texts)
        batch.truncated_indices = list(range(len(texts)))
        return batch

    def rerank(self, query: str, documents: list[str]) -> RerankBatch:
        """Pretend each pair exceeded the reranker bucket and was cut down to it."""
        batch = super().rerank(query, documents)
        batch.orig_tokens = [2000] * len(documents)
        batch.used_tokens = [512] * len(documents)
        batch.truncated_indices = list(range(len(documents)))
        return batch


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Client for an app that returns raw (un-normalized) stub vectors."""
    # The with-block runs the lifespan handler, which populates app.state.
    with TestClient(create_app(engine=StubEngine(), normalize=False)) as test_client:
        yield test_client


@pytest.fixture
def truncating_client() -> Iterator[TestClient]:
    """Client for an app whose engine always reports truncated inputs."""
    with TestClient(create_app(engine=TruncatingStubEngine(), normalize=False)) as test_client:
        yield test_client


@pytest.fixture
def normalizing_client() -> Iterator[TestClient]:
    """Client for an app configured to L2-normalize embeddings."""
    with TestClient(create_app(engine=StubEngine(), normalize=True)) as test_client:
        yield test_client


def test_health_reports_status_version_and_buckets(client: TestClient) -> None:
    """/health must return ok plus the buckets of both models."""
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["version"], str) and payload["version"]
    assert payload["models"] == {"embedding": [128, 512, 1024], "reranker": [512]}


def test_embeddings_accepts_single_string(client: TestClient) -> None:
    """A bare string input must produce exactly one embedding at index 0."""
    response = client.post("/v1/embeddings", json={"input": "hello"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert len(payload["data"]) == 1
    assert payload["data"][0]["object"] == "embedding"
    assert payload["data"][0]["index"] == 0
    assert payload["model"] == "ruri-v3-310m"


def test_embeddings_root_alias_matches_v1_path(client: TestClient) -> None:
    """/embeddings (Infinity-style root path) must behave like /v1/embeddings."""
    body = {"input": ["hello", "world"]}

    root = client.post("/embeddings", json=body)
    v1 = client.post("/v1/embeddings", json=body)

    assert root.status_code == 200
    assert root.json() == v1.json()


def test_embeddings_preserve_input_order_and_values(client: TestClient) -> None:
    """List input must keep request order, index it, and echo the stub values."""
    texts = ["a", "bbb", "cc"]

    response = client.post("/v1/embeddings", json={"input": texts})

    assert response.status_code == 200
    data = response.json()["data"]
    assert [item["index"] for item in data] == [0, 1, 2]
    for item, text in zip(data, texts, strict=True):
        np.testing.assert_allclose(item["embedding"], stub_vector(text))


def test_embeddings_usage_counts_every_input(client: TestClient) -> None:
    """usage must sum the per-input token counts reported by the engine."""
    texts = ["a", "bbb", "cc"]
    expected = sum(len(text) for text in texts)

    payload = client.post("/v1/embeddings", json={"input": texts}).json()

    assert payload["usage"] == {"prompt_tokens": expected, "total_tokens": expected}


def test_embeddings_empty_list_returns_no_data(client: TestClient) -> None:
    """An empty input list is valid and yields empty data with zero usage."""
    response = client.post("/v1/embeddings", json={"input": []})

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"] == []
    assert payload["usage"] == {"prompt_tokens": 0, "total_tokens": 0}


def test_embeddings_empty_list_survives_normalization(normalizing_client: TestClient) -> None:
    """Normalizing an empty (0, D) result must not fail."""
    response = normalizing_client.post("/v1/embeddings", json={"input": []})

    assert response.status_code == 200
    assert response.json()["data"] == []


def test_embeddings_base64_round_trip(client: TestClient) -> None:
    """base64 mode must return a string decoding to the float mode values."""
    response = client.post("/v1/embeddings", json={"input": ["hello"], "encoding_format": "base64"})

    assert response.status_code == 200
    encoded = response.json()["data"][0]["embedding"]
    assert isinstance(encoded, str)
    np.testing.assert_allclose(runtime.base64_to_floats(encoded), stub_vector("hello"))


def test_embeddings_are_normalized_when_enabled(normalizing_client: TestClient) -> None:
    """With normalize=True every returned vector must have unit L2 norm."""
    response = normalizing_client.post("/v1/embeddings", json={"input": ["a", "bbb"]})

    assert response.status_code == 200
    for item in response.json()["data"]:
        assert np.linalg.norm(item["embedding"]) == pytest.approx(1.0, abs=1e-6)


def test_embeddings_ignore_unknown_fields(client: TestClient) -> None:
    """Extra fields sent by compatible clients must not cause a 422."""
    response = client.post(
        "/v1/embeddings",
        json={"input": "hello", "model": "whatever", "input_type": "query", "user": "u1"},
        headers={"Authorization": "Bearer "},
    )

    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


def test_embeddings_reject_token_arrays(client: TestClient) -> None:
    """Token id arrays are outside eeANE's scope and must be rejected."""
    response = client.post("/v1/embeddings", json={"input": [[1, 2]]})

    assert response.status_code == 422


def test_embeddings_require_input(client: TestClient) -> None:
    """A request without input must be a validation error."""
    response = client.post("/v1/embeddings", json={"model": "ruri-v3-310m"})

    assert response.status_code == 422


@pytest.mark.parametrize("path", ["/rerank", "/v1/rerank"])
def test_rerank_scores_are_sigmoid_and_sorted(client: TestClient, path: str) -> None:
    """Both paths must return sigmoid scores, sorted descending, with original indices."""
    documents = ["a", "bbbb", "cc"]

    response = client.post(path, json={"query": "q", "documents": documents})

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "rerank"
    assert payload["model"] == "ruri-v3-reranker-310m"
    results = payload["results"]
    expected = sorted(
        ((index, stub_logit(document)) for index, document in enumerate(documents)),
        key=lambda item: item[1],
        reverse=True,
    )
    assert [result["index"] for result in results] == [index for index, _ in expected]
    for result, (_, logit) in zip(results, expected, strict=True):
        assert result["relevance_score"] == pytest.approx(
            float(runtime.sigmoid(np.asarray(logit))), abs=1e-6
        )
    scores = [result["relevance_score"] for result in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("path", ["/rerank", "/v1/rerank"])
def test_rerank_usage_counts_every_pair(client: TestClient, path: str) -> None:
    """usage must cover all scored pairs, whatever the sort order is."""
    documents = ["a", "bbbb", "cc"]
    expected = sum(len("q") + len(document) for document in documents)

    payload = client.post(path, json={"query": "q", "documents": documents}).json()

    assert payload["usage"] == {"prompt_tokens": expected, "total_tokens": expected}


def test_rerank_raw_scores_return_logits(client: TestClient) -> None:
    """raw_scores=true must bypass the sigmoid mapping."""
    documents = ["a", "bbbb", "cc"]

    response = client.post(
        "/rerank", json={"query": "q", "documents": documents, "raw_scores": True}
    )

    assert response.status_code == 200
    for result in response.json()["results"]:
        assert result["relevance_score"] == pytest.approx(stub_logit(documents[result["index"]]))


def test_rerank_top_n_truncates_results(client: TestClient) -> None:
    """top_n must keep only the highest scoring results."""
    response = client.post(
        "/rerank", json={"query": "q", "documents": ["a", "bbbb", "cc", "ddddd"], "top_n": 2}
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


def test_rerank_rejects_zero_top_n(client: TestClient) -> None:
    """top_n=0 is out of range and must be a validation error."""
    response = client.post("/rerank", json={"query": "q", "documents": ["a"], "top_n": 0})

    assert response.status_code == 422


def test_rerank_returns_documents_on_demand(client: TestClient) -> None:
    """return_documents=true must echo the document text back."""
    documents = ["a", "bbbb"]

    response = client.post(
        "/rerank", json={"query": "q", "documents": documents, "return_documents": True}
    )

    assert response.status_code == 200
    for result in response.json()["results"]:
        assert result["document"] == documents[result["index"]]


def test_rerank_omits_document_key_by_default(client: TestClient) -> None:
    """Without return_documents the key must be absent, not null."""
    response = client.post("/rerank", json={"query": "q", "documents": ["a", "bbbb"]})

    assert response.status_code == 200
    for result in response.json()["results"]:
        assert "document" not in result


def test_rerank_empty_documents_returns_no_results(client: TestClient) -> None:
    """An empty documents list is valid and yields an empty result set."""
    response = client.post("/rerank", json={"query": "q", "documents": []})

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"] == []
    assert payload["usage"] == {"prompt_tokens": 0, "total_tokens": 0}


def test_rerank_requires_query(client: TestClient) -> None:
    """A request without a query must be a validation error."""
    response = client.post("/rerank", json={"documents": ["a"]})

    assert response.status_code == 422


def test_embeddings_log_truncation_warning(
    truncating_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Truncated embedding inputs must be served and reported one WARNING each."""
    with caplog.at_level(logging.WARNING, logger="eeane.server"):
        response = truncating_client.post("/v1/embeddings", json={"input": ["a", "bb"]})

    assert response.status_code == 200
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert warnings == [
        "/v1/embeddings: input 0 truncated from 2000 tokens to bucket 1024",
        "/v1/embeddings: input 1 truncated from 2000 tokens to bucket 1024",
    ]


def test_rerank_logs_truncation_warning(
    truncating_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Truncated rerank pairs must be reported against the reranker bucket."""
    with caplog.at_level(logging.WARNING, logger="eeane.server"):
        response = truncating_client.post("/rerank", json={"query": "q", "documents": ["a"]})

    assert response.status_code == 200
    warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
    assert warnings == ["/rerank: input 0 truncated from 2000 tokens to bucket 512"]
