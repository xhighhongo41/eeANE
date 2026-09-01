"""Tests for eeane.schemas: request coercion, validation and response shapes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eeane.schemas import (
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsResponse,
    HealthModel,
    HealthResponse,
    RerankRequest,
    RerankResult,
    Usage,
)

# --- EmbeddingsRequest -------------------------------------------------


def test_embeddings_request_string_input_is_wrapped_in_a_list() -> None:
    """A bare string input must be coerced into a one-element list."""
    request = EmbeddingsRequest(input="hello")

    assert request.input == ["hello"]


def test_embeddings_request_list_of_strings_passes_through() -> None:
    """A list[str] input must be accepted unchanged."""
    request = EmbeddingsRequest(input=["hello", "world"])

    assert request.input == ["hello", "world"]


def test_embeddings_request_empty_list_is_allowed() -> None:
    """An empty input list must be accepted as-is."""
    request = EmbeddingsRequest(input=[])

    assert request.input == []


def test_embeddings_request_rejects_list_of_ints() -> None:
    """Token id arrays (list[int]) must be rejected, not treated as text."""
    with pytest.raises(ValidationError):
        EmbeddingsRequest(input=[1, 2, 3])


def test_embeddings_request_rejects_list_of_list_of_ints() -> None:
    """Batched token id arrays (list[list[int]]) must be rejected."""
    with pytest.raises(ValidationError):
        EmbeddingsRequest(input=[[1, 2], [3, 4]])


def test_embeddings_request_rejects_bare_int() -> None:
    """A non-string, non-list input must be rejected."""
    with pytest.raises(ValidationError):
        EmbeddingsRequest(input=123)


def test_embeddings_request_requires_input_field() -> None:
    """Omitting the required ``input`` field must fail validation."""
    with pytest.raises(ValidationError):
        EmbeddingsRequest()


def test_embeddings_request_ignores_unknown_fields() -> None:
    """Unknown fields (e.g. Open WebUI's configurable prefix field) must be ignored."""
    request = EmbeddingsRequest(input="hello", input_type="query")

    assert not hasattr(request, "input_type")


def test_embeddings_request_accepts_base64_encoding_format() -> None:
    """encoding_format="base64" must be a valid, accepted value."""
    request = EmbeddingsRequest(input="hello", encoding_format="base64")

    assert request.encoding_format == "base64"


def test_embeddings_request_rejects_unknown_encoding_format() -> None:
    """An unsupported encoding_format value must fail validation."""
    with pytest.raises(ValidationError):
        EmbeddingsRequest(input="hello", encoding_format="float16")


def test_embeddings_request_defaults() -> None:
    """encoding_format defaults to "float" and model defaults to None."""
    request = EmbeddingsRequest(input="hello")

    assert request.encoding_format == "float"
    assert request.model is None


def test_embeddings_request_accepts_dimensions() -> None:
    """dimensions must be accepted as a positive int (OpenAI-compatible MRL truncation)."""
    request = EmbeddingsRequest(input="hello", dimensions=256)

    assert request.dimensions == 256


def test_embeddings_request_dimensions_defaults_to_none() -> None:
    """Omitting dimensions must leave it unset (full embedding width returned)."""
    request = EmbeddingsRequest(input="hello")

    assert request.dimensions is None


def test_embeddings_request_rejects_dimensions_zero() -> None:
    """dimensions must be >= 1; 0 must fail validation."""
    with pytest.raises(ValidationError):
        EmbeddingsRequest(input="hello", dimensions=0)


def test_embeddings_request_rejects_negative_dimensions() -> None:
    """A negative dimensions value must fail validation."""
    with pytest.raises(ValidationError):
        EmbeddingsRequest(input="hello", dimensions=-1)


# --- RerankRequest -------------------------------------------------


def test_rerank_request_minimal_form_and_defaults() -> None:
    """query + documents alone must be accepted, with the documented defaults."""
    request = RerankRequest(query="q", documents=["a", "b"])

    assert request.top_n is None
    assert request.return_documents is False
    assert request.raw_scores is False


def test_rerank_request_empty_documents_is_allowed() -> None:
    """An empty documents list must be accepted."""
    request = RerankRequest(query="q", documents=[])

    assert request.documents == []


def test_rerank_request_rejects_top_n_zero() -> None:
    """top_n must be >= 1; 0 must fail validation."""
    with pytest.raises(ValidationError):
        RerankRequest(query="q", documents=["a"], top_n=0)


def test_rerank_request_accepts_top_n_one() -> None:
    """top_n=1 is the minimum accepted value."""
    request = RerankRequest(query="q", documents=["a"], top_n=1)

    assert request.top_n == 1


def test_rerank_request_requires_query_field() -> None:
    """Omitting the required ``query`` field must fail validation."""
    with pytest.raises(ValidationError):
        RerankRequest(documents=["a"])


def test_rerank_request_ignores_unknown_fields() -> None:
    """Unknown fields must be ignored rather than rejected."""
    request = RerankRequest(query="q", documents=["a"], extra_field="ignored")

    assert not hasattr(request, "extra_field")


# --- Response models -------------------------------------------------


def test_embeddings_response_default_object_is_list() -> None:
    """EmbeddingsResponse.object must default to the OpenAI-compatible "list"."""
    response = EmbeddingsResponse(
        data=[],
        model="ruri-v3-310m",
        usage=Usage(prompt_tokens=0, total_tokens=0),
    )

    assert response.object == "list"


def test_embedding_object_accepts_float_list_or_base64_string() -> None:
    """EmbeddingObject.embedding must accept both list[float] and str (base64) forms."""
    float_form = EmbeddingObject(index=0, embedding=[0.1, 0.2, 0.3])
    base64_form = EmbeddingObject(index=1, embedding="AACAPwAAAEA=")

    assert float_form.embedding == [0.1, 0.2, 0.3]
    assert base64_form.embedding == "AACAPwAAAEA="


def test_rerank_result_document_none_is_excluded_from_dump() -> None:
    """RerankResult.document must be dropped by model_dump(exclude_none=True) when unset."""
    result = RerankResult(index=0, relevance_score=0.5)

    dumped = result.model_dump(exclude_none=True)

    assert "document" not in dumped


def test_health_response_construction() -> None:
    """HealthResponse must accept one entry per served model."""
    response = HealthResponse(
        status="ok",
        version="0.5.0.dev0",
        models=[
            HealthModel(id="emb", kind="embedding", buckets=[128, 512, 1024], loaded=True),
            HealthModel(id="rr", kind="reranker", buckets=[512], loaded=False),
        ],
    )

    assert response.status == "ok"
    assert [model.id for model in response.models] == ["emb", "rr"]
    assert response.models[0].buckets == [128, 512, 1024]
    assert response.models[1].kind == "reranker"


def test_health_response_rejects_the_legacy_kind_to_buckets_mapping() -> None:
    """The per-kind mapping is gone: a dict must not silently validate."""
    with pytest.raises(ValidationError):
        HealthResponse(
            status="ok",
            version="0.5.0.dev0",
            models={"embedding": [128], "reranker": [512]},
        )
