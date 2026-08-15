"""Tests for eeane.server with a stub engine (v0.4実装計画.md §4.8, v0.5 §4.8).

These tests never touch Core ML: a deterministic in-memory engine is
injected into :func:`eeane.server.create_app`, so the endpoints' shapes,
sorting, encodings and error handling can be checked in any environment.
The v0.5 additions (API key auth, ``/models``, ``/health`` rate limiting,
embedding-only deployments) are exercised the same way, by feeding
``create_app`` a configuration built from the built-in default.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from eeane import runtime
from eeane.config import EeaneConfig, ModelEntry, ServerConfig, default_config
from eeane.engine import EmbeddingBatch, RerankBatch
from eeane.server import HealthRateLimiter, create_app

# Stub vectors are intentionally low-dimensional: the endpoints treat the
# width as opaque, so 4 keeps the expected values readable.
_STUB_DIM = 4

# API key used by the authenticated fixtures below (test-only value).
_API_KEY = "unit-test-api-key"


def make_config(
    *,
    normalize: bool = True,
    api_key: str | None = None,
    host: str = "127.0.0.1",
    health_rate_limit: int = 60,
    with_reranker: bool = True,
) -> EeaneConfig:
    """Build a test configuration by tweaking the built-in default.

    Args:
        normalize: Value of the embedding entry's ``normalize`` flag.
        api_key: Bearer key required by protected endpoints, or ``None``
            to keep the server unauthenticated.
        host: Bind address recorded in the config (only the start-up
            warning depends on it; nothing is actually bound in tests).
        health_rate_limit: ``/health`` requests allowed per minute per IP.
        with_reranker: When ``False``, drop the reranker entry to build an
            embedding-only deployment.

    Returns:
        A validated configuration serving the default model ids.
    """
    base = default_config()
    models: list[ModelEntry] = [base.embedding_model.model_copy(update={"normalize": normalize})]
    reranker = base.reranker_model
    if with_reranker and reranker is not None:
        models.append(reranker)
    return EeaneConfig(
        server=ServerConfig(host=host, api_key=api_key, health_rate_limit=health_rate_limit),
        models=models,
    )


class FakeClock:
    """Manually advanced monotonic clock for the rate limiter tests."""

    def __init__(self) -> None:
        """Start at zero seconds."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current fake time in seconds."""
        return self.now


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
    app = create_app(make_config(normalize=False), engine=StubEngine())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def truncating_client() -> Iterator[TestClient]:
    """Client for an app whose engine always reports truncated inputs."""
    app = create_app(make_config(normalize=False), engine=TruncatingStubEngine())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def normalizing_client() -> Iterator[TestClient]:
    """Client for an app configured to L2-normalize embeddings."""
    app = create_app(make_config(normalize=True), engine=StubEngine())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_client() -> Iterator[TestClient]:
    """Client for an app protected by :data:`_API_KEY`."""
    app = create_app(make_config(normalize=False, api_key=_API_KEY), engine=StubEngine())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def embedding_only_client() -> Iterator[TestClient]:
    """Client for an app configured without a reranker model."""
    app = create_app(make_config(normalize=False, with_reranker=False), engine=StubEngine())
    with TestClient(app) as test_client:
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


# --- API key authentication (v0.5実装計画.md §4.4) ------------------------

# Every endpoint that must be protected once an API key is configured,
# with a minimal valid body for the POST ones.
_PROTECTED_REQUESTS = [
    ("POST", "/embeddings", {"input": "hello"}),
    ("POST", "/v1/embeddings", {"input": "hello"}),
    ("POST", "/rerank", {"query": "q", "documents": ["a"]}),
    ("POST", "/v1/rerank", {"query": "q", "documents": ["a"]}),
    ("GET", "/models", None),
    ("GET", "/v1/models", None),
]


@pytest.mark.parametrize(("method", "path", "body"), _PROTECTED_REQUESTS)
def test_protected_endpoints_reject_missing_authorization(
    auth_client: TestClient, method: str, path: str, body: dict[str, object] | None
) -> None:
    """Without an Authorization header every protected endpoint must answer 401."""
    response = auth_client.request(method, path, json=body)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(("method", "path", "body"), _PROTECTED_REQUESTS)
def test_protected_endpoints_reject_wrong_key(
    auth_client: TestClient, method: str, path: str, body: dict[str, object] | None
) -> None:
    """A Bearer token that does not match the configured key must answer 401."""
    response = auth_client.request(
        method, path, json=body, headers={"Authorization": f"Bearer {_API_KEY}x"}
    )

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "header",
    [
        "Basic dXNlcjpwYXNz",
        "Bearer",
        f"Bearer {_API_KEY} extra",
        _API_KEY,
        "",
    ],
)
def test_protected_endpoints_reject_malformed_authorization(
    auth_client: TestClient, header: str
) -> None:
    """Anything that is not exactly 'Bearer <key>' must answer 401."""
    response = auth_client.post(
        "/v1/embeddings", json={"input": "hello"}, headers={"Authorization": header}
    )

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(("method", "path", "body"), _PROTECTED_REQUESTS)
def test_protected_endpoints_accept_the_configured_key(
    auth_client: TestClient, method: str, path: str, body: dict[str, object] | None
) -> None:
    """The configured key must unlock every protected endpoint."""
    response = auth_client.request(
        method, path, json=body, headers={"Authorization": f"Bearer {_API_KEY}"}
    )

    assert response.status_code == 200


def test_bearer_scheme_is_case_insensitive(auth_client: TestClient) -> None:
    """RFC 7235 makes the auth scheme case-insensitive, unlike the key itself."""
    response = auth_client.get("/models", headers={"Authorization": f"bearer {_API_KEY}"})

    assert response.status_code == 200


def test_health_stays_open_when_a_key_is_configured(auth_client: TestClient) -> None:
    """/health is the monitoring surface and must never require the key."""
    response = auth_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_key_startup_log_never_leaks_the_key(caplog: pytest.LogCaptureFixture) -> None:
    """Enabling auth must be reported at INFO without ever printing the key."""
    app = create_app(make_config(api_key=_API_KEY), engine=StubEngine())

    with caplog.at_level(logging.INFO, logger="eeane.server"), TestClient(app):
        pass

    messages = [record.getMessage() for record in caplog.records if record.name == "eeane.server"]
    assert "API key auth enabled" in messages
    assert all(_API_KEY not in message for message in messages)


# --- GET /models (v0.5実装計画.md §4.5) -----------------------------------


def test_models_lists_the_configured_models(client: TestClient) -> None:
    """/models must return an OpenAI-style list of every configured model."""
    response = client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert [card["id"] for card in payload["data"]] == [
        "ruri-v3-310m",
        "ruri-v3-reranker-310m",
    ]
    for card in payload["data"]:
        assert card["object"] == "model"
        assert card["owned_by"] == "eeane"
        assert isinstance(card["created"], int)
    # created is the server start-up time, shared by every card.
    created = {card["created"] for card in payload["data"]}
    assert len(created) == 1
    assert created.pop() == client.app.state.started_at


def test_models_root_alias_matches_v1_path(client: TestClient) -> None:
    """/models and /v1/models must return the very same payload."""
    root = client.get("/models")
    v1 = client.get("/v1/models")

    assert root.status_code == 200
    assert root.json() == v1.json()


def test_models_without_reranker_lists_only_the_embedding(
    embedding_only_client: TestClient,
) -> None:
    """An embedding-only deployment must advertise exactly one model."""
    response = embedding_only_client.get("/models")

    assert response.status_code == 200
    assert [card["id"] for card in response.json()["data"]] == ["ruri-v3-310m"]


# --- start-up security warnings (v0.5実装計画.md §4.4) --------------------


def test_non_loopback_host_without_api_key_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Binding to the LAN without a key must be reported once at WARNING."""
    app = create_app(make_config(host="192.168.10.20"), engine=StubEngine())

    with caplog.at_level(logging.WARNING, logger="eeane.server"), TestClient(app):
        pass

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "eeane.server" and record.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    assert "192.168.10.20" in warnings[0]


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_host_without_api_key_does_not_warn(
    host: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A loopback bind is the safe default and must stay silent."""
    app = create_app(make_config(host=host), engine=StubEngine())

    with caplog.at_level(logging.WARNING, logger="eeane.server"), TestClient(app):
        pass

    warnings = [
        record
        for record in caplog.records
        if record.name == "eeane.server" and record.levelname == "WARNING"
    ]
    assert warnings == []


def test_non_loopback_host_with_api_key_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """The warning is about the missing key, so a configured key silences it."""
    app = create_app(make_config(host="192.168.10.20", api_key=_API_KEY), engine=StubEngine())

    with caplog.at_level(logging.WARNING, logger="eeane.server"), TestClient(app):
        pass

    warnings = [
        record
        for record in caplog.records
        if record.name == "eeane.server" and record.levelname == "WARNING"
    ]
    assert warnings == []


# --- embedding-only deployment (v0.5実装計画.md §0-2, §4.6) ---------------


@pytest.mark.parametrize("path", ["/rerank", "/v1/rerank"])
def test_rerank_returns_503_without_a_configured_reranker(
    embedding_only_client: TestClient, path: str
) -> None:
    """Both rerank paths must report the missing reranker as 503."""
    response = embedding_only_client.post(path, json={"query": "q", "documents": ["a"]})

    assert response.status_code == 503
    assert response.json()["detail"] == "reranker is not configured"


def test_health_reports_no_reranker_buckets_without_a_reranker(
    embedding_only_client: TestClient,
) -> None:
    """The reranker bucket list must be empty whatever the engine exposes."""
    payload = embedding_only_client.get("/health").json()

    assert payload["models"] == {"embedding": [128, 512, 1024], "reranker": []}


def test_embeddings_still_work_without_a_reranker(embedding_only_client: TestClient) -> None:
    """Dropping the reranker must not affect the embedding endpoint."""
    response = embedding_only_client.post("/v1/embeddings", json={"input": "hello"})

    assert response.status_code == 200
    assert response.json()["model"] == "ruri-v3-310m"


# --- /health rate limiting (v0.5実装計画.md §4.4) -------------------------


def test_rate_limiter_rejects_after_the_limit_and_recovers_next_window() -> None:
    """The counter must cap a client within a window and reset in the next one."""
    clock = FakeClock()
    limiter = HealthRateLimiter(3, clock=clock)

    assert [limiter.allow("10.0.0.1") for _ in range(4)] == [True, True, True, False]

    # Still inside the same 60s window: still rejected.
    clock.now = 59.9
    assert limiter.allow("10.0.0.1") is False

    # Next window: the budget is restored.
    clock.now = 60.0
    assert limiter.allow("10.0.0.1") is True


@pytest.mark.parametrize("limit", [0, -1])
def test_rate_limiter_is_disabled_for_non_positive_limits(limit: int) -> None:
    """A limit of zero (or lower) must let every request through."""
    limiter = HealthRateLimiter(limit, clock=FakeClock())

    assert all(limiter.allow("10.0.0.1") for _ in range(1000))


def test_rate_limiter_counts_clients_independently() -> None:
    """One noisy client must not consume another client's budget."""
    clock = FakeClock()
    limiter = HealthRateLimiter(2, clock=clock)

    assert [limiter.allow("10.0.0.1") for _ in range(3)] == [True, True, False]
    assert [limiter.allow("10.0.0.2") for _ in range(3)] == [True, True, False]
    assert limiter.allow("unknown") is True


def test_rate_limiter_prunes_clients_from_closed_windows() -> None:
    """Tracked IPs must not accumulate forever across windows."""
    clock = FakeClock()
    limiter = HealthRateLimiter(10, clock=clock)
    for index in range(2000):
        limiter.allow(f"10.0.{index // 256}.{index % 256}")

    clock.now = 60.0
    assert limiter.allow("10.9.9.9") is True

    # Only the new window's client survives the pruning pass.
    assert list(limiter._counters) == ["10.9.9.9"]


def test_health_returns_429_after_the_configured_limit() -> None:
    """The /health endpoint must apply server.health_rate_limit."""
    app = create_app(make_config(normalize=False, health_rate_limit=2), engine=StubEngine())

    with TestClient(app) as test_client:
        statuses = [test_client.get("/health").status_code for _ in range(3)]

    assert statuses == [200, 200, 429]


def test_health_is_not_limited_when_the_limit_is_zero() -> None:
    """health_rate_limit=0 must disable the limiter entirely."""
    app = create_app(make_config(normalize=False, health_rate_limit=0), engine=StubEngine())

    with TestClient(app) as test_client:
        statuses = [test_client.get("/health").status_code for _ in range(10)]

    assert statuses == [200] * 10
