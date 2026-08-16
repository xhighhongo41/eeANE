"""Tests for eeane.server with a stub engine.

These tests never touch Core ML: a deterministic in-memory engine (see
``tests/conftest.py``) is injected into :func:`eeane.server.create_app`,
so the endpoints' shapes, sorting, encodings, model routing and error
handling can be checked in any environment. Deployments are described by
a configuration built from the built-in default, which is also what the
injected engine mirrors.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pytest
from conftest import (
    STUB_BUCKETS,
    StubEngine,
    make_config,
    make_model_entry,
    stub_logit,
    stub_vector,
)
from fastapi.testclient import TestClient

from eeane import runtime
from eeane.config import EeaneConfig
from eeane.engine import EmbeddingBatch, RerankBatch
from eeane.server import HealthRateLimiter, create_app

# API key used by the authenticated fixtures below (test-only value).
_API_KEY = "unit-test-api-key"

# Ids of the default deployment, as listed by the built-in configuration.
_EMBEDDING_ID = "ruri-v3-310m"
_RERANKER_ID = "ruri-v3-reranker-310m"

# Extra models appended to the default deployment by the routing tests.
_SECOND_EMBEDDING_ID = "second-embedding"
_SECOND_RERANKER_ID = "second-reranker"

# Buckets those extra models serve, chosen so /health tells them apart
# from the default ones.
_SECOND_BUCKETS = (256,)


class FakeClock:
    """Manually advanced monotonic clock for the rate limiter tests."""

    def __init__(self) -> None:
        """Start at zero seconds."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current fake time in seconds."""
        return self.now


class TruncatingStubEngine(StubEngine):
    """Stub reporting every input as truncated, to exercise the warning path."""

    def embed(self, texts: list[str], model_id: str | None = None) -> EmbeddingBatch:
        """Pretend each text exceeded the largest bucket and was cut down to it."""
        batch = super().embed(texts, model_id)
        batch.orig_tokens = [2000] * len(texts)
        batch.used_tokens = [1024] * len(texts)
        batch.buckets = [1024] * len(texts)
        batch.truncated_indices = list(range(len(texts)))
        return batch

    def rerank(self, query: str, documents: list[str], model_id: str | None = None) -> RerankBatch:
        """Pretend each pair exceeded the reranker bucket and was cut down to it."""
        batch = super().rerank(query, documents, model_id)
        batch.orig_tokens = [2000] * len(documents)
        batch.used_tokens = [512] * len(documents)
        batch.truncated_indices = list(range(len(documents)))
        return batch


class LoadedStateStubEngine(StubEngine):
    """Stub whose ``loaded()`` answer is set per model id by the test.

    Attributes:
        loaded_ids: Ids currently reported as in memory; every other
            configured id is reported as not loaded.
    """

    def __init__(self, config: EeaneConfig, loaded_ids: set[str]) -> None:
        """Register the configured models and remember which are loaded.

        Args:
            config: Deployment to mirror (forwarded to :class:`StubEngine`).
            loaded_ids: Ids :meth:`loaded` must report ``True`` for.
        """
        super().__init__(config)
        self.loaded_ids = loaded_ids

    def loaded(self, model_id: str) -> bool:
        """Report ``model_id in loaded_ids`` (``KeyError`` if unknown)."""
        if model_id not in self._kinds:
            raise KeyError(model_id)
        return model_id in self.loaded_ids


class CallCountingStubEngine(StubEngine):
    """Stub counting every :meth:`embed`/:meth:`rerank` call it serves.

    Used to prove an endpoint reads only state (``loaded()``, ``buckets()``)
    without triggering inference.
    """

    def __init__(self, config: EeaneConfig) -> None:
        """Register the configured models and start both counters at zero."""
        super().__init__(config)
        self.embed_calls = 0
        self.rerank_calls = 0

    def embed(self, texts: list[str], model_id: str | None = None) -> EmbeddingBatch:
        """Count the call, then serve it as :class:`StubEngine` would."""
        self.embed_calls += 1
        return super().embed(texts, model_id)

    def rerank(self, query: str, documents: list[str], model_id: str | None = None) -> RerankBatch:
        """Count the call, then serve it as :class:`StubEngine` would."""
        self.rerank_calls += 1
        return super().rerank(query, documents, model_id)


class CloseRecordingStubEngine(StubEngine):
    """Stub recording whether :meth:`close` was called (lifespan-teardown test)."""

    def __init__(self, config: EeaneConfig) -> None:
        """Register the configured models and start unclosed."""
        super().__init__(config)
        self.closed = False

    def close(self) -> None:
        """Record that shutdown reached this engine."""
        self.closed = True


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Client for an app that returns raw (un-normalized) stub vectors."""
    config = make_config(normalize=False)
    # The with-block runs the lifespan handler, which populates app.state.
    app = create_app(config, engine=StubEngine(config))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def truncating_client() -> Iterator[TestClient]:
    """Client for an app whose engine always reports truncated inputs."""
    config = make_config(normalize=False)
    app = create_app(config, engine=TruncatingStubEngine(config))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def normalizing_client() -> Iterator[TestClient]:
    """Client for an app configured to L2-normalize embeddings."""
    config = make_config(normalize=True)
    app = create_app(config, engine=StubEngine(config))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_client() -> Iterator[TestClient]:
    """Client for an app protected by :data:`_API_KEY`."""
    config = make_config(normalize=False, api_key=_API_KEY)
    app = create_app(config, engine=StubEngine(config))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def embedding_only_client() -> Iterator[TestClient]:
    """Client for an app configured without a reranker model."""
    config = make_config(normalize=False, with_reranker=False)
    app = create_app(config, engine=StubEngine(config))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def multi_model() -> Iterator[tuple[TestClient, StubEngine]]:
    """Client and engine of a deployment serving two models of each kind.

    The default embedding model returns raw vectors while the second one
    normalizes, so a response tells which model served it.
    """
    config = make_config(
        normalize=False,
        extra_models=[
            make_model_entry(_SECOND_EMBEDDING_ID, normalize=True, buckets=_SECOND_BUCKETS),
            make_model_entry(_SECOND_RERANKER_ID, kind="reranker", buckets=_SECOND_BUCKETS),
        ],
    )
    engine = StubEngine(
        config,
        buckets={_SECOND_EMBEDDING_ID: _SECOND_BUCKETS, _SECOND_RERANKER_ID: _SECOND_BUCKETS},
    )
    with TestClient(create_app(config, engine=engine)) as test_client:
        yield test_client, engine


def test_health_reports_status_version_and_buckets(client: TestClient) -> None:
    """/health must return ok plus one entry per served model."""
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["version"], str) and payload["version"]
    assert payload["models"] == [
        {
            "id": _EMBEDDING_ID,
            "kind": "embedding",
            "buckets": list(STUB_BUCKETS["embedding"]),
            "loaded": True,
        },
        {
            "id": _RERANKER_ID,
            "kind": "reranker",
            "buckets": list(STUB_BUCKETS["reranker"]),
            "loaded": True,
        },
    ]


def test_health_reports_the_loaded_state_of_each_model() -> None:
    """/health must reflect the engine's per-model ``loaded()`` answer."""
    config = make_config(normalize=False)
    engine = LoadedStateStubEngine(config, loaded_ids={_EMBEDDING_ID})

    with TestClient(create_app(config, engine=engine)) as test_client:
        payload = test_client.get("/health").json()

    loaded_by_id = {model["id"]: model["loaded"] for model in payload["models"]}
    assert loaded_by_id == {_EMBEDDING_ID: True, _RERANKER_ID: False}


def test_health_and_models_never_trigger_inference() -> None:
    """/health and /models must only read state, never call embed()/rerank()."""
    config = make_config(normalize=False)
    engine = CallCountingStubEngine(config)

    with TestClient(create_app(config, engine=engine)) as test_client:
        health_response = test_client.get("/health")
        models_response = test_client.get("/models")

    assert health_response.status_code == 200
    assert models_response.status_code == 200
    assert engine.embed_calls == 0
    assert engine.rerank_calls == 0


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
        json={"input": "hello", "model": _EMBEDDING_ID, "input_type": "query", "user": "u1"},
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


# --- API key authentication ----------------------------------------------

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
    config = make_config(api_key=_API_KEY)
    app = create_app(config, engine=StubEngine(config))

    with caplog.at_level(logging.INFO, logger="eeane.server"), TestClient(app):
        pass

    messages = [record.getMessage() for record in caplog.records if record.name == "eeane.server"]
    assert "API key auth enabled" in messages
    assert all(_API_KEY not in message for message in messages)


# --- lifespan teardown -----------------------------------------------------


def test_lifespan_shutdown_closes_the_engine() -> None:
    """Shutdown must call the injected engine's ``close()`` exactly once."""
    config = make_config(normalize=False)
    engine = CloseRecordingStubEngine(config)
    app = create_app(config, engine=engine)

    with TestClient(app):
        assert engine.closed is False

    assert engine.closed is True


def test_lifespan_shutdown_tolerates_an_engine_without_close() -> None:
    """An injected engine that defines no ``close`` must not break shutdown."""
    config = make_config(normalize=False)
    engine = StubEngine(config)
    app = create_app(config, engine=engine)

    assert not hasattr(engine, "close")
    with TestClient(app):
        pass


# --- GET /models ----------------------------------------------------------


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


# --- start-up security warnings -------------------------------------------


def test_non_loopback_host_without_api_key_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Binding to the LAN without a key must be reported once at WARNING."""
    config = make_config(host="192.168.10.20")
    app = create_app(config, engine=StubEngine(config))

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
    config = make_config(host=host)
    app = create_app(config, engine=StubEngine(config))

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
    config = make_config(host="192.168.10.20", api_key=_API_KEY)
    app = create_app(config, engine=StubEngine(config))

    with caplog.at_level(logging.WARNING, logger="eeane.server"), TestClient(app):
        pass

    warnings = [
        record
        for record in caplog.records
        if record.name == "eeane.server" and record.levelname == "WARNING"
    ]
    assert warnings == []


# --- embedding-only deployment --------------------------------------------


@pytest.mark.parametrize("path", ["/rerank", "/v1/rerank"])
def test_rerank_returns_503_without_a_configured_reranker(
    embedding_only_client: TestClient, path: str
) -> None:
    """Both rerank paths must report the missing reranker as 503."""
    response = embedding_only_client.post(path, json={"query": "q", "documents": ["a"]})

    assert response.status_code == 503
    assert response.json()["detail"] == "reranker is not configured"


def test_health_lists_no_reranker_without_a_reranker(
    embedding_only_client: TestClient,
) -> None:
    """Only configured models may be listed, whatever the engine exposes."""
    payload = embedding_only_client.get("/health").json()

    assert payload["models"] == [
        {
            "id": _EMBEDDING_ID,
            "kind": "embedding",
            "buckets": list(STUB_BUCKETS["embedding"]),
            "loaded": True,
        }
    ]


def test_embeddings_still_work_without_a_reranker(embedding_only_client: TestClient) -> None:
    """Dropping the reranker must not affect the embedding endpoint."""
    response = embedding_only_client.post("/v1/embeddings", json={"input": "hello"})

    assert response.status_code == 200
    assert response.json()["model"] == "ruri-v3-310m"


# --- /health rate limiting ------------------------------------------------


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
    config = make_config(normalize=False, health_rate_limit=2)
    app = create_app(config, engine=StubEngine(config))

    with TestClient(app) as test_client:
        statuses = [test_client.get("/health").status_code for _ in range(3)]

    assert statuses == [200, 200, 429]


def test_health_is_not_limited_when_the_limit_is_zero() -> None:
    """health_rate_limit=0 must disable the limiter entirely."""
    config = make_config(normalize=False, health_rate_limit=0)
    app = create_app(config, engine=StubEngine(config))

    with TestClient(app) as test_client:
        statuses = [test_client.get("/health").status_code for _ in range(10)]

    assert statuses == [200] * 10


# --- model routing --------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "body", "expected"),
    [
        ("/v1/embeddings", {"input": "hello"}, _EMBEDDING_ID),
        ("/rerank", {"query": "q", "documents": ["a"]}, _RERANKER_ID),
    ],
)
def test_request_without_a_model_uses_the_default_of_its_kind(
    multi_model: tuple[TestClient, StubEngine],
    path: str,
    body: dict[str, object],
    expected: str,
) -> None:
    """An omitted model id must resolve to the first-listed model of the kind."""
    client, engine = multi_model

    response = client.post(path, json=body)

    assert response.status_code == 200
    assert response.json()["model"] == expected
    assert (engine.embed_model_ids + engine.rerank_model_ids) == [expected]


@pytest.mark.parametrize(
    ("path", "body", "requested"),
    [
        ("/v1/embeddings", {"input": "hello"}, _SECOND_EMBEDDING_ID),
        ("/rerank", {"query": "q", "documents": ["a"]}, _SECOND_RERANKER_ID),
    ],
)
def test_request_naming_a_configured_model_is_routed_to_it(
    multi_model: tuple[TestClient, StubEngine],
    path: str,
    body: dict[str, object],
    requested: str,
) -> None:
    """A known model id must reach the engine and be echoed back."""
    client, engine = multi_model

    response = client.post(path, json={**body, "model": requested})

    assert response.status_code == 200
    assert response.json()["model"] == requested
    assert (engine.embed_model_ids + engine.rerank_model_ids) == [requested]


@pytest.mark.parametrize(
    ("path", "body", "kind", "known_id"),
    [
        ("/v1/embeddings", {"input": "hello"}, "embedding", _SECOND_EMBEDDING_ID),
        ("/rerank", {"query": "q", "documents": ["a"]}, "reranker", _SECOND_RERANKER_ID),
    ],
)
def test_request_naming_an_unknown_model_returns_404_with_the_available_ids(
    multi_model: tuple[TestClient, StubEngine],
    path: str,
    body: dict[str, object],
    kind: str,
    known_id: str,
) -> None:
    """An unknown id must be a 404 that tells the client what it can ask for."""
    client, engine = multi_model

    response = client.post(path, json={**body, "model": "no-such-model"})

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "no-such-model" in detail
    assert known_id in detail
    # Nothing may be inferred with a fallback model.
    assert engine.embed_model_ids == []
    assert engine.rerank_model_ids == []


@pytest.mark.parametrize(
    ("path", "body", "wrong_id"),
    [
        ("/v1/embeddings", {"input": "hello"}, _RERANKER_ID),
        ("/rerank", {"query": "q", "documents": ["a"]}, _EMBEDDING_ID),
    ],
)
def test_request_naming_a_model_of_the_other_kind_returns_400(
    multi_model: tuple[TestClient, StubEngine],
    path: str,
    body: dict[str, object],
    wrong_id: str,
) -> None:
    """A configured id of the wrong kind is a bad request, not a missing model."""
    client, engine = multi_model

    response = client.post(path, json={**body, "model": wrong_id})

    assert response.status_code == 400
    assert wrong_id in response.json()["detail"]
    assert engine.embed_model_ids == []
    assert engine.rerank_model_ids == []


def test_normalization_follows_the_resolved_model(
    multi_model: tuple[TestClient, StubEngine],
) -> None:
    """Each embedding model applies its own normalize flag, not a global one."""
    client, _ = multi_model

    raw = client.post("/v1/embeddings", json={"input": "hello"})
    normalized = client.post(
        "/v1/embeddings", json={"input": "hello", "model": _SECOND_EMBEDDING_ID}
    )

    assert raw.status_code == 200 and normalized.status_code == 200
    np.testing.assert_allclose(raw.json()["data"][0]["embedding"], stub_vector("hello"))
    assert np.linalg.norm(normalized.json()["data"][0]["embedding"]) == pytest.approx(1.0, abs=1e-6)


def test_models_lists_every_configured_model_in_order(
    multi_model: tuple[TestClient, StubEngine],
) -> None:
    """/models must advertise every entry, not just the default of each kind."""
    client, _ = multi_model

    response = client.get("/models")

    assert response.status_code == 200
    assert [card["id"] for card in response.json()["data"]] == [
        _EMBEDDING_ID,
        _RERANKER_ID,
        _SECOND_EMBEDDING_ID,
        _SECOND_RERANKER_ID,
    ]


def test_health_lists_every_model_with_its_kind_and_buckets(
    multi_model: tuple[TestClient, StubEngine],
) -> None:
    """/health must describe each served model on its own."""
    client, _ = multi_model

    payload = client.get("/health").json()

    assert payload["models"] == [
        {
            "id": _EMBEDDING_ID,
            "kind": "embedding",
            "buckets": list(STUB_BUCKETS["embedding"]),
            "loaded": True,
        },
        {
            "id": _RERANKER_ID,
            "kind": "reranker",
            "buckets": list(STUB_BUCKETS["reranker"]),
            "loaded": True,
        },
        {
            "id": _SECOND_EMBEDDING_ID,
            "kind": "embedding",
            "buckets": list(_SECOND_BUCKETS),
            "loaded": True,
        },
        {
            "id": _SECOND_RERANKER_ID,
            "kind": "reranker",
            "buckets": list(_SECOND_BUCKETS),
            "loaded": True,
        },
    ]


# --- start-up model reporting ---------------------------------------------


def test_startup_logs_one_line_per_served_model(caplog: pytest.LogCaptureFixture) -> None:
    """Every model must be reported at INFO with its kind and buckets."""
    config = make_config(
        normalize=False,
        extra_models=[make_model_entry(_SECOND_EMBEDDING_ID, buckets=_SECOND_BUCKETS)],
    )
    app = create_app(config, engine=StubEngine(config))

    with caplog.at_level(logging.INFO, logger="eeane.server"), TestClient(app):
        pass

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "eeane.server" and record.levelname == "INFO"
    ]
    for entry in config.models:
        assert any(
            entry.id in message and str(entry.kind) in message and str(entry.buckets[0]) in message
            for message in messages
        ), entry.id


def test_startup_warns_about_buckets_the_cache_recommends_against(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Excluded buckets must be reported once, naming the model and the buckets."""
    config = make_config(
        normalize=False,
        extra_models=[
            make_model_entry(
                _SECOND_EMBEDDING_ID, buckets=_SECOND_BUCKETS, excluded_buckets=(1024,)
            )
        ],
    )
    app = create_app(config, engine=StubEngine(config))

    with caplog.at_level(logging.WARNING, logger="eeane.server"), TestClient(app):
        pass

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "eeane.server" and record.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    assert _SECOND_EMBEDDING_ID in warnings[0]
    assert "1024" in warnings[0]


def test_startup_stays_quiet_without_excluded_buckets(caplog: pytest.LogCaptureFixture) -> None:
    """A deployment loading every compiled bucket must not warn."""
    config = make_config(normalize=False)
    app = create_app(config, engine=StubEngine(config))

    with caplog.at_level(logging.WARNING, logger="eeane.server"), TestClient(app):
        pass

    assert [record for record in caplog.records if record.levelname == "WARNING"] == []


def test_startup_logs_policy_and_keep_alive_for_on_demand_models(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An on-demand entry's start-up line must state its policy and idle-unload delay."""
    on_demand_entry = make_model_entry(_SECOND_EMBEDDING_ID, buckets=_SECOND_BUCKETS).model_copy(
        update={"load_policy": "on_demand", "keep_alive": 42}
    )
    config = make_config(normalize=False, extra_models=[on_demand_entry])
    app = create_app(config, engine=StubEngine(config))

    with caplog.at_level(logging.INFO, logger="eeane.server"), TestClient(app):
        pass

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "eeane.server" and record.levelname == "INFO"
    ]
    assert any(
        _SECOND_EMBEDDING_ID in message
        and "policy=on_demand" in message
        and "keep_alive=42s" in message
        for message in messages
    )


def test_startup_logs_the_configured_memory_cap(caplog: pytest.LogCaptureFixture) -> None:
    """A configured max_loaded_models cap must be reported once at start-up."""
    config = make_config(normalize=False)
    config.server.max_loaded_models = 2
    app = create_app(config, engine=StubEngine(config))

    with caplog.at_level(logging.INFO, logger="eeane.server"), TestClient(app):
        pass

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "eeane.server" and record.levelname == "INFO"
    ]
    assert any("2" in message and "kept in memory" in message for message in messages)


# --- disabled models --------------------------------------------------------


def test_disabled_model_is_hidden_and_returns_404(caplog: pytest.LogCaptureFixture) -> None:
    """A load_policy='disabled' entry must be unroutable and reported at start-up."""
    disabled_entry = make_model_entry(_SECOND_EMBEDDING_ID, buckets=_SECOND_BUCKETS).model_copy(
        update={"load_policy": "disabled"}
    )
    config = make_config(normalize=False, extra_models=[disabled_entry])
    assert _SECOND_EMBEDDING_ID in config.disabled_models
    app = create_app(config, engine=StubEngine(config))

    with caplog.at_level(logging.INFO, logger="eeane.server"), TestClient(app) as test_client:
        models_response = test_client.get("/models")
        health_response = test_client.get("/health")
        embeddings_response = test_client.post(
            "/v1/embeddings", json={"input": "hello", "model": _SECOND_EMBEDDING_ID}
        )

    assert _SECOND_EMBEDDING_ID not in [card["id"] for card in models_response.json()["data"]]
    assert _SECOND_EMBEDDING_ID not in [model["id"] for model in health_response.json()["models"]]
    assert embeddings_response.status_code == 404

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "eeane.server" and record.levelname == "INFO"
    ]
    assert any(_SECOND_EMBEDDING_ID in message and "disabled" in message for message in messages)
