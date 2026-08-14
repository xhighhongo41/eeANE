"""FastAPI application for the eeANE server (v0.4実装計画.md §4.1, §4.4-§4.6).

Exposes an OpenAI-compatible ``/v1/embeddings`` endpoint and an
Infinity-compatible ``/rerank`` + ``/v1/rerank`` pair on top of an
:class:`~eeane.engine.InferenceEngine`. The engine is built once during
startup and kept resident; every inference call is serialized inside the
engine, so the endpoints are plain ``def`` functions that FastAPI runs in
its thread pool (``/health`` stays ``async`` and answers immediately).

Run it with ``uv run python -m eeane.server`` (single process, single
worker: multiple workers would load the models several times).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request

from eeane import __version__, runtime, settings
from eeane.engine import CoreMLEngine, InferenceEngine
from eeane.schemas import (
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsResponse,
    HealthResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    Usage,
)

logger = logging.getLogger("eeane.server")


def _format_buckets(buckets: Sequence[int]) -> str:
    """Summarize per-input bucket usage for the request log.

    Args:
        buckets: Bucket used by each input of the request.

    Returns:
        A compact summary such as ``"128x3,512x1"``, or ``"-"`` when the
        request had no inputs.
    """
    if not buckets:
        return "-"
    counts = Counter(buckets)
    return ",".join(f"{bucket}x{counts[bucket]}" for bucket in sorted(counts))


def _warn_truncated(
    path: str,
    truncated_indices: Sequence[int],
    orig_tokens: Sequence[int],
    buckets: Sequence[int],
) -> None:
    """Log one WARNING per truncated input.

    Args:
        path: Request path, to tell embedding and rerank logs apart.
        truncated_indices: Indices reported by the engine as truncated.
        orig_tokens: Pre-truncation token count of every input.
        buckets: Bucket used by every input.
    """
    for index in truncated_indices:
        logger.warning(
            "%s: input %d truncated from %d tokens to bucket %d",
            path,
            index,
            orig_tokens[index],
            buckets[index],
        )


def create_app(engine: InferenceEngine | None = None, normalize: bool | None = None) -> FastAPI:
    """Build the eeANE FastAPI application.

    Args:
        engine: Engine used to serve requests. ``None`` (production
            default) builds a :class:`~eeane.engine.CoreMLEngine` from
            ``eeane.settings`` during startup; tests inject a stub so no
            Core ML artifact is needed.
        normalize: Whether embeddings are L2-normalized before they are
            returned. ``None`` falls back to
            ``settings.NORMALIZE_EMBEDDINGS``.

    Returns:
        The configured application. Nothing is loaded until the lifespan
        handler runs, so building the app is cheap and side-effect free.
    """
    normalize_embeddings = settings.NORMALIZE_EMBEDDINGS if normalize is None else normalize

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load the engine once at startup and keep it for the process' life."""
        if engine is None:
            started = time.perf_counter()
            active_engine: InferenceEngine = CoreMLEngine.from_settings()
            elapsed = time.perf_counter() - started
            logger.info(
                "loaded Core ML engine in %.2fs "
                "(embedding buckets=%s, reranker buckets=%s, normalize=%s)",
                elapsed,
                list(active_engine.embedding_buckets),
                list(active_engine.reranker_buckets),
                normalize_embeddings,
            )
        else:
            active_engine = engine
            logger.info(
                "using injected engine (embedding buckets=%s, reranker buckets=%s, normalize=%s)",
                list(active_engine.embedding_buckets),
                list(active_engine.reranker_buckets),
                normalize_embeddings,
            )
        app.state.engine = active_engine
        app.state.normalize_embeddings = normalize_embeddings
        yield

    app = FastAPI(title="eeANE", version=__version__, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Report server status and the buckets each model serves."""
        engine_impl: InferenceEngine = request.app.state.engine
        return HealthResponse(
            status="ok",
            version=__version__,
            models={
                "embedding": list(engine_impl.embedding_buckets),
                "reranker": list(engine_impl.reranker_buckets),
            },
        )

    # Registered under /v1 (OpenAI convention) and at the root (where
    # Infinity_emb serves it), so either base-URL style works unchanged.
    @app.post("/embeddings", response_model=EmbeddingsResponse)
    @app.post("/v1/embeddings", response_model=EmbeddingsResponse)
    def create_embeddings(request: Request, body: EmbeddingsRequest) -> EmbeddingsResponse:
        """Embed the request's texts (OpenAI-compatible, §4.4)."""
        started = time.perf_counter()
        engine_impl: InferenceEngine = request.app.state.engine
        batch = engine_impl.embed(body.input)

        vectors = batch.vectors
        # Skip normalization for an empty request so the (0, D) shape is
        # never divided by an empty norm array.
        if request.app.state.normalize_embeddings and vectors.shape[0] > 0:
            vectors = runtime.l2_normalize(vectors)
        # The official OpenAI SDK asks for base64 by default; both
        # encodings carry the very same float32 values.
        data = [
            EmbeddingObject(
                index=index,
                embedding=(
                    runtime.floats_to_base64(vector)
                    if body.encoding_format == "base64"
                    else vector.tolist()
                ),
            )
            for index, vector in enumerate(vectors)
        ]
        used_tokens = int(sum(batch.used_tokens))

        path = request.url.path
        _warn_truncated(path, batch.truncated_indices, batch.orig_tokens, batch.buckets)
        logger.info(
            "POST %s inputs=%d buckets=%s truncated=%d elapsed_ms=%.1f",
            path,
            len(body.input),
            _format_buckets(batch.buckets),
            len(batch.truncated_indices),
            (time.perf_counter() - started) * 1000.0,
        )
        return EmbeddingsResponse(
            data=data,
            model=settings.EMBEDDING_MODEL_ID,
            usage=Usage(prompt_tokens=used_tokens, total_tokens=used_tokens),
        )

    @app.post("/rerank", response_model=RerankResponse, response_model_exclude_none=True)
    @app.post("/v1/rerank", response_model=RerankResponse, response_model_exclude_none=True)
    def rerank(request: Request, body: RerankRequest) -> RerankResponse:
        """Score the request's documents against its query (Infinity-compatible, §4.5)."""
        started = time.perf_counter()
        engine_impl: InferenceEngine = request.app.state.engine
        batch = engine_impl.rerank(body.query, body.documents)

        scores = batch.logits if body.raw_scores else runtime.sigmoid(batch.logits)
        results = [
            RerankResult(
                index=index,
                relevance_score=float(score),
                document=body.documents[index] if body.return_documents else None,
            )
            for index, score in enumerate(scores)
        ]
        # Infinity returns the most relevant document first; ties keep the
        # request order because Python's sort is stable.
        results.sort(key=lambda result: result.relevance_score, reverse=True)
        if body.top_n is not None:
            results = results[: body.top_n]
        # usage counts every scored pair, including the ones top_n drops.
        used_tokens = int(sum(batch.used_tokens))

        # select_bucket is a pure function of the pre-truncation token
        # count, so recomputing it here reproduces the engine's routing
        # for logging without widening RerankBatch.
        buckets = [
            runtime.select_bucket(n_tokens, engine_impl.reranker_buckets)[0]
            for n_tokens in batch.orig_tokens
        ]
        path = request.url.path
        _warn_truncated(path, batch.truncated_indices, batch.orig_tokens, buckets)
        logger.info(
            "POST %s inputs=%d buckets=%s truncated=%d elapsed_ms=%.1f",
            path,
            len(body.documents),
            _format_buckets(buckets),
            len(batch.truncated_indices),
            (time.perf_counter() - started) * 1000.0,
        )
        return RerankResponse(
            results=results,
            model=settings.RERANKER_MODEL_ID,
            usage=Usage(prompt_tokens=used_tokens, total_tokens=used_tokens),
        )

    return app


def main() -> None:
    """Serve the app on ``settings.HOST``/``settings.PORT`` in one process."""
    # uvicorn only configures its own loggers, so enable INFO output for
    # "eeane.server" here (the app itself never touches global logging).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # An app instance (not an import string) is passed on purpose: it makes
    # the multi-worker mode unavailable, which would load the models once
    # per worker (§4.1).
    uvicorn.run(create_app(), host=settings.HOST, port=settings.PORT)


if __name__ == "__main__":
    main()
