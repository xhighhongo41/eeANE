"""FastAPI application for the eeANE server (v0.5実装計画.md §4.4-§4.6).

Exposes an OpenAI-compatible ``/v1/embeddings`` endpoint, an
Infinity-compatible ``/rerank`` + ``/v1/rerank`` pair and an
OpenAI-compatible ``/models`` listing on top of an
:class:`~eeane.engine.InferenceEngine`. Everything the server serves comes
from a validated :class:`~eeane.config.EeaneConfig`: model ids, bucket
artifacts, bind address, the optional Bearer API key and the ``/health``
rate limit.

The engine is built once during startup and kept resident; every inference
call is serialized inside the engine, so the endpoints are plain ``def``
functions that FastAPI runs in its thread pool (``/health`` and
``/models`` stay ``async`` and answer immediately).

Run it with ``uv run python -m eeane serve`` (single process, single
worker: multiple workers would load the models several times).
``uv run python -m eeane.server`` remains as a thin alias for the same
command until v0.10 (see :func:`main`).
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from eeane import __version__, runtime
from eeane.cli import main as cli_main
from eeane.config import EeaneConfig
from eeane.engine import CoreMLEngine, InferenceEngine
from eeane.schemas import (
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsResponse,
    HealthResponse,
    ModelCard,
    ModelListResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    Usage,
)

logger = logging.getLogger("eeane.server")

# Bind addresses that only accept connections from this machine. Anything
# else exposes the server to the network, which is worth a warning when no
# API key is configured (v0.5実装計画.md §4.4).
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Fixed window of the /health rate limiter, in seconds.
_RATE_LIMIT_WINDOW_SECONDS = 60

# Upper bound on the client IPs tracked by the /health rate limiter. Once
# exceeded, entries of past windows are dropped so a spoofed-IP flood
# cannot grow the counter dict without bound.
_MAX_TRACKED_CLIENTS = 1024


class HealthRateLimiter:
    """Fixed-window request limiter for the unauthenticated ``/health`` endpoint.

    ``/health`` is the only endpoint that stays open when API key auth is
    enabled, so it gets a dependency-free, in-memory limiter: each client
    IP may issue ``limit_per_minute`` requests per 60-second window. This
    only blunts trivial floods; connection-level protection belongs to a
    reverse proxy or firewall (v0.5実装計画.md §4.4).
    """

    def __init__(self, limit_per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        """Initialize an empty limiter.

        Args:
            limit_per_minute: Allowed requests per window per client IP.
                Zero (or any non-positive value) disables the limiter.
            clock: Monotonic seconds source; injectable so tests can move
                between windows without sleeping.
        """
        self._limit = limit_per_minute
        self._clock = clock
        # client IP -> (window index, requests served in that window)
        self._counters: dict[str, tuple[int, int]] = {}
        # /health is async (event loop only), but the limiter is cheap to
        # make thread-safe and may be shared with thread-pool callers.
        self._lock = threading.Lock()

    def allow(self, client_ip: str) -> bool:
        """Account for one request and report whether it may be served.

        Args:
            client_ip: Remote address of the caller (``"unknown"`` when the
                transport does not expose one, which buckets all such
                callers together).

        Returns:
            ``True`` if the request is within the client's budget for the
            current window, ``False`` if it must be rejected with 429.
        """
        if self._limit <= 0:
            return True

        window = int(self._clock() // _RATE_LIMIT_WINDOW_SECONDS)
        with self._lock:
            previous = self._counters.get(client_ip)
            # A stale entry from an earlier window restarts at zero.
            count = previous[1] if previous is not None and previous[0] == window else 0
            if count >= self._limit:
                return False
            self._counters[client_ip] = (window, count + 1)
            if len(self._counters) > _MAX_TRACKED_CLIENTS:
                self._prune(window)
        return True

    def _prune(self, window: int) -> None:
        """Drop counters that belong to an already-closed window.

        Args:
            window: Index of the window currently being served; entries of
                every other window are removed.
        """
        self._counters = {
            client_ip: entry for client_ip, entry in self._counters.items() if entry[0] == window
        }


def _unauthorized(detail: str) -> HTTPException:
    """Build the 401 raised by the API key dependency.

    Args:
        detail: Human-readable reason placed in the JSON body.

    Returns:
        A 401 ``HTTPException`` carrying the ``WWW-Authenticate`` header
        required by RFC 7235 for a Bearer-protected resource.
    """
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _log_startup_security(config: EeaneConfig) -> None:
    """Report the security posture of the resolved configuration at startup.

    Args:
        config: Resolved configuration. Only ``server.host`` and whether
            ``server.api_key`` is set are inspected; the key value itself
            is never logged.
    """
    if config.server.api_key is not None:
        logger.info("API key auth enabled")
        return
    if config.server.host not in _LOOPBACK_HOSTS:
        logger.warning(
            "listening on non-loopback address %s without an API key: "
            "every host that can reach this address can use the server; "
            "set server.api_key (or EEANE_API_KEY) to require authentication",
            config.server.host,
        )


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


def create_app(config: EeaneConfig, engine: InferenceEngine | None = None) -> FastAPI:
    """Build the eeANE FastAPI application for a resolved configuration.

    Args:
        config: Validated configuration (see :mod:`eeane.config`). It
            supplies the served model ids, the ``normalize`` flag, the
            optional API key and the ``/health`` rate limit. When its
            reranker entry is absent, the rerank endpoints answer 503.
        engine: Engine used to serve requests. ``None`` (production
            default) builds a :class:`~eeane.engine.CoreMLEngine` from
            ``config`` during startup; tests inject a stub so no Core ML
            artifact is needed.

    Returns:
        The configured application. Nothing is loaded until the lifespan
        handler runs, so building the app is cheap and side-effect free.
    """
    embedding_entry = config.embedding_model
    reranker_entry = config.reranker_model
    normalize_embeddings = embedding_entry.normalize
    api_key = config.server.api_key

    async def require_api_key(request: Request) -> None:
        """Enforce ``Authorization: Bearer <api key>`` on protected routes.

        Args:
            request: Incoming request; only its ``Authorization`` header is
                read.

        Raises:
            HTTPException: 401 when a key is configured and the header is
                missing, malformed or does not match. Without a configured
                key the dependency is a no-op, so behaviour is identical to
                v0.4.
        """
        if api_key is None:
            return
        header = request.headers.get("Authorization")
        if header is None:
            raise _unauthorized("Missing Authorization header; expected 'Bearer <api key>'")
        parts = header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise _unauthorized("Malformed Authorization header; expected 'Bearer <api key>'")
        # Constant-time comparison so a wrong key cannot be recovered by
        # timing the response (v0.5実装計画.md §4.4).
        if not secrets.compare_digest(parts[1].encode("utf-8"), api_key.encode("utf-8")):
            raise _unauthorized("Invalid API key")

    auth = [Depends(require_api_key)]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load the engine once at startup and keep it for the process' life."""
        _log_startup_security(config)
        if engine is None:
            started = time.perf_counter()
            active_engine: InferenceEngine = CoreMLEngine.from_config(config)
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
        # Reported as every model card's "created" timestamp.
        app.state.started_at = int(time.time())
        app.state.health_limiter = HealthRateLimiter(config.server.health_rate_limit)
        yield

    app = FastAPI(title="eeANE", version=__version__, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Report server status and the buckets each model serves.

        Raises:
            HTTPException: 429 when the caller exceeded
                ``server.health_rate_limit`` requests in the current
                minute. ``/health`` stays unauthenticated, so it is rate
                limited instead.
        """
        limiter: HealthRateLimiter = request.app.state.health_limiter
        # request.client is None for transports without a peer address.
        client_ip = request.client.host if request.client is not None else "unknown"
        if not limiter.allow(client_ip):
            raise HTTPException(
                status_code=429, detail="Too many /health requests; try again later"
            )

        engine_impl: InferenceEngine = request.app.state.engine
        return HealthResponse(
            status="ok",
            version=__version__,
            models={
                "embedding": list(engine_impl.embedding_buckets),
                # An embedding-only deployment serves no reranker bucket,
                # whatever the injected engine happens to expose.
                "reranker": [] if reranker_entry is None else list(engine_impl.reranker_buckets),
            },
        )

    @app.get("/models", response_model=ModelListResponse, dependencies=auth)
    @app.get("/v1/models", response_model=ModelListResponse, dependencies=auth)
    async def list_models(request: Request) -> ModelListResponse:
        """List the configured models (OpenAI-compatible, §4.5)."""
        created = request.app.state.started_at
        # Deterministic order (embedding first) regardless of how the
        # config file listed the entries.
        entries = [embedding_entry]
        if reranker_entry is not None:
            entries.append(reranker_entry)
        return ModelListResponse(
            data=[ModelCard(id=entry.id, created=created) for entry in entries]
        )

    # Registered under /v1 (OpenAI convention) and at the root (where
    # Infinity_emb serves it), so either base-URL style works unchanged.
    @app.post("/embeddings", response_model=EmbeddingsResponse, dependencies=auth)
    @app.post("/v1/embeddings", response_model=EmbeddingsResponse, dependencies=auth)
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
            model=embedding_entry.id,
            usage=Usage(prompt_tokens=used_tokens, total_tokens=used_tokens),
        )

    @app.post(
        "/rerank",
        response_model=RerankResponse,
        response_model_exclude_none=True,
        dependencies=auth,
    )
    @app.post(
        "/v1/rerank",
        response_model=RerankResponse,
        response_model_exclude_none=True,
        dependencies=auth,
    )
    def rerank(request: Request, body: RerankRequest) -> RerankResponse:
        """Score the request's documents against its query (Infinity-compatible, §4.5).

        Raises:
            HTTPException: 503 when the configuration has no reranker
                entry (embedding-only deployment).
        """
        if reranker_entry is None:
            raise HTTPException(status_code=503, detail="reranker is not configured")

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
            model=reranker_entry.id,
            usage=Usage(prompt_tokens=used_tokens, total_tokens=used_tokens),
        )

    return app


def main() -> None:
    """Interim alias for ``python -m eeane serve`` (kept until v0.10).

    ``python -m eeane.server`` predates the ``eeane`` CLI (v0.4); it is
    kept as a thin wrapper so existing invocations keep working, per
    開発資料/v0.5実装計画.md §0-6.
    """
    raise SystemExit(cli_main(["serve"]))


if __name__ == "__main__":
    main()
