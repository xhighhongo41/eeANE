"""Request/response models for eeANE's HTTP API (v0.4 T3, see 開発資料/v0.4実装計画.md §4.4-§4.6).

eeANE exposes an OpenAI-compatible ``/v1/embeddings`` endpoint and an
Infinity-compatible ``/rerank`` / ``/v1/rerank`` endpoint pair. Compatible
clients (e.g. Open WebUI) may send additional fields that this server does
not use (for example a configurable prefix field); request models therefore
ignore unknown fields instead of rejecting them with a 422.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Usage(BaseModel):
    """Token accounting reported alongside embedding/rerank responses.

    Attributes:
        prompt_tokens: Number of tokens consumed by the request, after any
            truncation performed by the server.
        total_tokens: Total tokens processed. Currently equal to
            ``prompt_tokens`` (no completion tokens are produced by these
            endpoints).
    """

    prompt_tokens: int
    total_tokens: int


class EmbeddingsRequest(BaseModel):
    """Request body for ``POST /v1/embeddings`` (OpenAI-compatible).

    Attributes:
        input: Text(s) to embed. A single string is normalized to a
            one-element list. Token id arrays (as allowed by the OpenAI API)
            are not supported and are rejected with a validation error.
        model: Requested model id. Accepted but ignored (eeANE serves a
            single fixed embedding model per v0.4 scope).
        encoding_format: Either ``"float"`` (plain JSON floats) or
            ``"base64"`` (base64-encoded little-endian float32 bytes, as
            requested by the official OpenAI SDK by default).
        user: Opaque client identifier. Accepted but ignored.
    """

    model_config = ConfigDict(extra="ignore")

    input: list[str]
    model: str | None = None
    encoding_format: Literal["float", "base64"] = "float"
    user: str | None = None

    @field_validator("input", mode="before")
    @classmethod
    def _coerce_and_validate_input(cls, value: object) -> list[str]:
        """Normalize ``input`` into a list of strings.

        Args:
            value: Raw value supplied by the client: a single string, a
                list, or (invalid) anything else.

        Returns:
            The input coerced into a list of strings. An empty list is
            passed through unchanged.

        Raises:
            ValueError: If ``value`` is a list containing non-string
                elements (e.g. token id arrays), or if ``value`` is neither
                a string nor a list at all.
        """
        if isinstance(value, str):
            return [value]
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
        raise ValueError(
            "eeANE only supports text input for embeddings; token arrays are not accepted."
        )


class EmbeddingObject(BaseModel):
    """A single embedding entry within an embeddings response.

    Attributes:
        object: Discriminator, always ``"embedding"``.
        index: Position of this embedding within the request's ``input``
            list.
        embedding: The embedding vector as a list of floats, or (when
            ``encoding_format="base64"`` was requested) a base64-encoded
            string of little-endian float32 bytes.
    """

    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float] | str


class EmbeddingsResponse(BaseModel):
    """Response body for ``POST /v1/embeddings`` (OpenAI-compatible).

    Attributes:
        object: Discriminator, always ``"list"``.
        data: Embeddings, one per input, in request order.
        model: Model id that served the request.
        usage: Token accounting for the request.
    """

    object: Literal["list"] = "list"
    data: list[EmbeddingObject]
    model: str
    usage: Usage


class RerankRequest(BaseModel):
    """Request body for ``POST /rerank`` and ``POST /v1/rerank`` (Infinity-compatible).

    Attributes:
        query: The query text to score documents against.
        documents: Candidate documents to rerank. An empty list is allowed
            and yields an empty result set.
        top_n: If set, only the top-N results (by descending relevance) are
            returned. Must be at least 1 when provided.
        return_documents: If true, echo each document's text back in the
            corresponding result entry.
        raw_scores: If true, return raw logits instead of sigmoid-mapped
            relevance scores.
        model: Requested model id. Accepted but ignored (eeANE serves a
            single fixed reranker model per v0.4 scope).
    """

    model_config = ConfigDict(extra="ignore")

    query: str
    documents: list[str]
    top_n: int | None = Field(default=None, ge=1)
    return_documents: bool = False
    raw_scores: bool = False
    model: str | None = None


class RerankResult(BaseModel):
    """A single scored document within a rerank response.

    Attributes:
        index: Position of this document within the request's ``documents``
            list.
        relevance_score: Sigmoid-mapped relevance score (or a raw logit if
            ``raw_scores=True`` was requested).
        document: The document text, present only when
            ``return_documents=True`` was requested. Response builders
            should serialize with ``exclude_none=True`` so this field is
            omitted rather than emitted as ``null`` when unset.
    """

    index: int
    relevance_score: float
    document: str | None = None


class RerankResponse(BaseModel):
    """Response body for ``POST /rerank`` and ``POST /v1/rerank`` (Infinity-compatible).

    Attributes:
        object: Discriminator, always ``"rerank"``.
        results: Scored documents, sorted by descending relevance and
            truncated to ``top_n`` if requested.
        model: Model id that served the request.
        usage: Token accounting for the request.
    """

    object: Literal["rerank"] = "rerank"
    results: list[RerankResult]
    model: str
    usage: Usage


class HealthResponse(BaseModel):
    """Response body for ``GET /health``.

    Attributes:
        status: Server status string (e.g. ``"ok"``).
        version: eeANE package version (``eeane.__version__``).
        models: Mapping from model kind (``"embedding"``, ``"reranker"``)
            to the list of supported sequence-length buckets, e.g.
            ``{"embedding": [128, 512, 1024], "reranker": [512]}``.
    """

    status: str
    version: str
    models: dict[str, list[int]]
