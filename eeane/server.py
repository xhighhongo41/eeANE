"""FastAPI application for the eeANE server.

Exposes an OpenAI-compatible ``/v1/embeddings`` endpoint, an
Infinity-compatible ``/rerank`` + ``/v1/rerank`` pair and an
OpenAI-compatible ``/models`` listing on top of an
:class:`~eeane.engine.InferenceEngine`. Everything the server serves comes
from a validated :class:`~eeane.config.EeaneConfig`: model ids, bucket
artifacts, bind address, the optional Bearer API key and the ``/health``
rate limit.

Several embedding and reranker models can be served at once. A request's
``model`` field selects one of them by id; omitting it selects the
first-listed model of the kind the endpoint serves. The resolved entry
also decides per-request behaviour such as embedding normalization, and
its id is echoed in the response.

The engine is built once during startup and kept resident; every inference
call is serialized inside the engine, so the endpoints are plain ``def``
functions that FastAPI runs in its thread pool (``/health`` and
``/models`` stay ``async`` and answer immediately).

Run it with ``eeane serve`` (single process, single worker: multiple
workers would load the models several times). ``python -m eeane serve``
and ``python -m eeane.server`` remain as thin aliases for the same
command (see :func:`main`).
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
from eeane.config import EeaneConfig, ModelEntry
from eeane.engine import CoreMLEngine, InferenceEngine, NonFiniteOutputError, QueueTimeoutError
from eeane.schemas import (
    EmbeddingObject,
    EmbeddingsRequest,
    EmbeddingsResponse,
    HealthModel,
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
# API key is configured.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Fixed window of the /health rate limiter, in seconds.
_RATE_LIMIT_WINDOW_SECONDS = 60

# Upper bound on the client IPs tracked by the /health rate limiter. Once
# exceeded, entries of past windows are dropped so a spoofed-IP flood
# cannot grow the counter dict without bound.
_MAX_TRACKED_CLIENTS = 1024

# Seconds suggested to a rejected/timed-out client before it retries, sent
# as the Retry-After header on both the 429 (admission cap) and 503
# (queue timeout) responses.
_RETRY_AFTER_SECONDS = 5


class HealthRateLimiter:
    """Fixed-window request limiter for the unauthenticated ``/health`` endpoint.

    ``/health`` is the only endpoint that stays open when API key auth is
    enabled, so it gets a dependency-free, in-memory limiter: each client
    IP may issue ``limit_per_minute`` requests per 60-second window. This
    only blunts trivial floods; connection-level protection belongs to a
    reverse proxy or firewall.
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


class AdmissionController:
    """Caps the number of inference requests accepted at once.

    Every protected inference route enters the controller through the
    ``_admission`` dependency before it runs and leaves it once its
    response (or an error) is ready, so ``pending`` always reflects
    requests that are either waiting for the engine or already running.
    Follows the same in-memory, ``threading.Lock``-guarded style as
    :class:`HealthRateLimiter`.
    """

    def __init__(self, limit: int) -> None:
        """Initialize an empty controller.

        Args:
            limit: Maximum number of requests admitted at once. Zero (or
                any non-positive value) disables the cap.
        """
        self._limit = limit
        self._pending = 0
        self._lock = threading.Lock()

    @property
    def pending(self) -> int:
        """Number of requests currently admitted and not yet released."""
        with self._lock:
            return self._pending

    def try_enter(self) -> bool:
        """Admit one request if the configured cap has not been reached.

        Returns:
            ``True`` if the request is admitted (the caller must later
            call :meth:`leave` exactly once), ``False`` if the cap is
            already reached.
        """
        with self._lock:
            if self._limit > 0 and self._pending >= self._limit:
                return False
            # Counted even when the cap is disabled, so ``pending`` stays
            # an accurate gauge and every leave() has a matching entry.
            self._pending += 1
            return True

    def leave(self) -> None:
        """Release one request previously admitted by :meth:`try_enter`."""
        with self._lock:
            self._pending -= 1


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


def _log_served_models(config: EeaneConfig) -> None:
    """Report the served models at startup, one line each.

    Args:
        config: Resolved configuration. Every entry is reported at INFO
            with its kind, buckets and effective load policy: a resident
            entry's model is loaded right away, while an on-demand
            entry's line also states its idle-unload delay, since that
            model is only loaded once a request first needs it. An entry
            whose compiled-model cache recommends against some of its
            buckets also gets a WARNING, so an operator sees why a bucket
            is not in service. Ids configured with ``load_policy =
            "disabled"`` and a configured ``max_loaded_models`` cap are
            each reported in one extra INFO line, if set.
    """
    for entry in config.models:
        details = f"buckets={list(entry.buckets)}"
        if entry.kind == "embedding":
            details += f", normalize={entry.normalize}"
        if config.resolved_load_policy(entry) == "resident":
            details += ", policy=resident"
            logger.info("serving %s model '%s' (%s)", entry.kind, entry.id, details)
        else:
            keep_alive = config.resolved_keep_alive(entry)
            details += f", policy=on_demand, keep_alive={keep_alive}s"
            logger.info(
                "serving %s model '%s' on demand, loaded on first request (%s)",
                entry.kind,
                entry.id,
                details,
            )
        if entry.excluded_buckets:
            logger.warning(
                "model '%s': the compiled-model cache recommends against buckets %s, "
                "which are therefore excluded from service; see the model's compile "
                "self-check record for the measurements behind that recommendation",
                entry.id,
                list(entry.excluded_buckets),
            )

    if config.disabled_models:
        logger.info("configured but disabled: %s", ", ".join(config.disabled_models))

    if config.server.max_loaded_models is not None:
        logger.info(
            "at most %d model(s) kept in memory at once (max_loaded_models)",
            config.server.max_loaded_models,
        )


def _log_admission_control(config: EeaneConfig) -> None:
    """Report the request-admission settings at startup, in one INFO line.

    Args:
        config: Resolved configuration. ``server.max_pending_requests``
            of ``0`` is reported as "unlimited" and ``server.queue_timeout``
            of ``0`` as "disabled", matching what those values mean.
    """
    max_pending = config.server.max_pending_requests
    max_pending_desc = "unlimited" if max_pending <= 0 else str(max_pending)
    timeout = config.server.queue_timeout
    timeout_desc = "disabled" if timeout <= 0 else f"{timeout}s"
    logger.info(
        "admission control: max_pending_requests=%s, queue_timeout=%s, request coalescing=%s",
        max_pending_desc,
        timeout_desc,
        "on" if config.server.coalesce_requests else "off",
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


def _resolve_entry(config: EeaneConfig, kind: str, requested: str | None) -> ModelEntry:
    """Route one request to the model entry that must serve it.

    Args:
        config: Resolved configuration listing every served model.
        kind: Model kind the calling endpoint serves (``"embedding"`` or
            ``"reranker"``).
        requested: Model id sent by the client, or ``None`` when the
            request named none.

    Returns:
        The configured entry to serve the request with: the named one, or
        the first-listed entry of ``kind`` when the request named none.

    Raises:
        HTTPException: 404 when ``requested`` matches no configured model
            (the detail lists the ids this endpoint can serve), 400 when
            it matches a model of another kind, and 503 when the request
            named no model and none of ``kind`` is configured.
    """
    if requested is None:
        default_entry = config.default_model(kind)
        if default_entry is None:
            raise HTTPException(status_code=503, detail=f"{kind} is not configured")
        return default_entry

    entry = config.model_by_id(requested)
    if entry is None:
        # Listing the servable ids turns a typo into a self-service fix.
        available = ", ".join(f"'{candidate.id}'" for candidate in config.models_of_kind(kind))
        raise HTTPException(
            status_code=404,
            detail=(
                f"model '{requested}' not found; available {kind} models: {available or '(none)'}"
            ),
        )
    if entry.kind != kind:
        raise HTTPException(
            status_code=400,
            detail=(
                f"model '{requested}' is a {entry.kind} model; this endpoint serves {kind} models"
            ),
        )
    return entry


def _compute_deadline(request: Request, config: EeaneConfig) -> float | None:
    """Compute the absolute deadline this request's inference call may run until.

    Args:
        request: Current request; ``request.state.admitted_at`` must
            already hold the ``time.monotonic()`` reading of when the
            request was admitted (set by the ``_admission`` dependency).
        config: Resolved configuration; only ``server.queue_timeout`` is
            read.

    Returns:
        The ``time.monotonic()`` reading past which the request must
        give up, or ``None`` when ``server.queue_timeout`` is disabled
        (``0``), meaning the request may wait indefinitely.
    """
    timeout = config.server.queue_timeout
    if timeout <= 0:
        return None
    return request.state.admitted_at + timeout


def _queue_timeout_response(detail: str) -> HTTPException:
    """Build the 503 raised when a request's queue wait times out.

    Args:
        detail: Human-readable reason placed in the JSON body.

    Returns:
        A 503 ``HTTPException`` carrying a ``Retry-After`` header so a
        well-behaved client backs off before retrying.
    """
    return HTTPException(
        status_code=503,
        detail=detail,
        headers={"Retry-After": str(_RETRY_AFTER_SECONDS)},
    )


def create_app(config: EeaneConfig, engine: InferenceEngine | None = None) -> FastAPI:
    """Build the eeANE FastAPI application for a resolved configuration.

    Args:
        config: Validated configuration (see :mod:`eeane.config`). It
            supplies the served models (their ids, kinds and ``normalize``
            flags), the optional API key and the ``/health`` rate limit.
            Requests are routed to those models by id; when no reranker is
            configured at all, the rerank endpoints answer 503.
        engine: Engine used to serve requests. ``None`` (production
            default) builds a :class:`~eeane.engine.CoreMLEngine` from
            ``config`` during startup; tests inject a stub so no Core ML
            artifact is needed.

    Returns:
        The configured application. Nothing is loaded until the lifespan
        handler runs, so building the app is cheap and side-effect free.
    """
    api_key = config.server.api_key

    async def require_api_key(request: Request) -> None:
        """Enforce ``Authorization: Bearer <api key>`` on protected routes.

        Args:
            request: Incoming request; only its ``Authorization`` header is
                read.

        Raises:
            HTTPException: 401 when a key is configured and the header is
                missing, malformed or does not match. Without a configured
                key the dependency is a no-op.
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
        # timing the response.
        if not secrets.compare_digest(parts[1].encode("utf-8"), api_key.encode("utf-8")):
            raise _unauthorized("Invalid API key")

    auth = [Depends(require_api_key)]

    async def _admission(request: Request) -> AsyncIterator[None]:
        """Admit an inference request under ``server.max_pending_requests``.

        Also stamps ``request.state.admitted_at`` with the request's
        ``time.monotonic()`` queue time, which the endpoint functions use
        to compute their deadline. Runs after :data:`auth` in every
        protected inference route's dependency list, so a request that
        fails authentication never consumes a slot.

        Args:
            request: Incoming request; only ``request.app.state.admission``
                is read.

        Raises:
            HTTPException: 429 when the configured admission cap is
                already reached. Carries a ``Retry-After`` header so a
                well-behaved client backs off before retrying.
        """
        request.state.admitted_at = time.monotonic()
        controller: AdmissionController = request.app.state.admission
        if not controller.try_enter():
            raise HTTPException(
                status_code=429,
                detail=(
                    f"server is at capacity ({config.server.max_pending_requests} pending "
                    "requests); retry later"
                ),
                headers={"Retry-After": str(_RETRY_AFTER_SECONDS)},
            )
        try:
            yield
        finally:
            controller.leave()

    admission = [Depends(_admission)]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Build the engine at startup and stop its background work at shutdown.

        The engine itself decides what "stop" means: :class:`CoreMLEngine`
        stops its idle-unload sweeper thread but leaves whatever is loaded
        in memory, since the process is about to exit anyway. An injected
        engine that defines no ``close`` (as test stubs typically do) is
        left untouched.
        """
        _log_startup_security(config)
        if engine is None:
            started = time.perf_counter()
            active_engine: InferenceEngine = CoreMLEngine.from_config(config)
            logger.info("loaded Core ML engine in %.2fs", time.perf_counter() - started)
        else:
            active_engine = engine
            logger.info("using injected engine")
        _log_served_models(config)
        _log_admission_control(config)
        app.state.engine = active_engine
        # Reported as every model card's "created" timestamp.
        app.state.started_at = int(time.time())
        app.state.health_limiter = HealthRateLimiter(config.server.health_rate_limit)
        app.state.admission = AdmissionController(config.server.max_pending_requests)
        yield
        close = getattr(active_engine, "close", None)
        if callable(close):
            close()

    app = FastAPI(title="eeANE", version=__version__, lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        """Report server status and the buckets each served model covers.

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
        # Driven by the configuration, so only served models are listed
        # whatever else the engine happens to hold.
        return HealthResponse(
            status="ok",
            version=__version__,
            models=[
                HealthModel(
                    id=entry.id,
                    kind=str(entry.kind),
                    buckets=list(engine_impl.buckets(entry.id)),
                    loaded=engine_impl.loaded(entry.id),
                )
                for entry in config.models
            ],
        )

    @app.get("/models", response_model=ModelListResponse, dependencies=auth)
    @app.get("/v1/models", response_model=ModelListResponse, dependencies=auth)
    async def list_models(request: Request) -> ModelListResponse:
        """List every configured model, in configuration order (OpenAI-compatible)."""
        created = request.app.state.started_at
        return ModelListResponse(
            data=[ModelCard(id=entry.id, created=created) for entry in config.models]
        )

    # Registered under /v1 (OpenAI convention) and at the root (where
    # Infinity_emb serves it), so either base-URL style works unchanged.
    @app.post("/embeddings", response_model=EmbeddingsResponse, dependencies=auth + admission)
    @app.post("/v1/embeddings", response_model=EmbeddingsResponse, dependencies=auth + admission)
    def create_embeddings(request: Request, body: EmbeddingsRequest) -> EmbeddingsResponse:
        """Embed the request's texts with the requested model (OpenAI-compatible).

        Raises:
            HTTPException: 404 when ``body.model`` names no configured
                model, 400 when it names a model of another kind, 503
                when the request's wait exceeded ``server.queue_timeout``
                (either before or while it waited its turn at the
                engine), 500 when the model produced a non-finite output.
        """
        started = time.perf_counter()
        deadline = _compute_deadline(request, config)
        if deadline is not None and deadline - time.monotonic() <= 0:
            raise _queue_timeout_response("request timed out while waiting to be processed")
        entry = _resolve_entry(config, "embedding", body.model)
        engine_impl: InferenceEngine = request.app.state.engine
        try:
            batch = engine_impl.embed(body.input, model_id=entry.id, deadline=deadline)
        except QueueTimeoutError as exc:
            raise _queue_timeout_response(str(exc)) from exc
        except NonFiniteOutputError as exc:
            logger.error(
                "model '%s' produced a non-finite output on bucket %d", exc.model_id, exc.bucket
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        vectors = batch.vectors
        # Normalization is a per-model decision, and an empty request is
        # skipped so the (0, D) shape is never divided by an empty norm.
        if entry.normalize and vectors.shape[0] > 0:
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
            model=entry.id,
            usage=Usage(prompt_tokens=used_tokens, total_tokens=used_tokens),
        )

    @app.post(
        "/rerank",
        response_model=RerankResponse,
        response_model_exclude_none=True,
        dependencies=auth + admission,
    )
    @app.post(
        "/v1/rerank",
        response_model=RerankResponse,
        response_model_exclude_none=True,
        dependencies=auth + admission,
    )
    def rerank(request: Request, body: RerankRequest) -> RerankResponse:
        """Score the request's documents against its query (Infinity-compatible).

        Raises:
            HTTPException: 503 when the configuration has no reranker at
                all (embedding-only deployment) or when the request's
                wait exceeded ``server.queue_timeout`` (either before or
                while it waited its turn at the engine), 404 when
                ``body.model`` names no configured model, 400 when it
                names a model of another kind, 500 when the model
                produced a non-finite output.
        """
        started = time.perf_counter()
        deadline = _compute_deadline(request, config)
        if deadline is not None and deadline - time.monotonic() <= 0:
            raise _queue_timeout_response("request timed out while waiting to be processed")
        entry = _resolve_entry(config, "reranker", body.model)
        engine_impl: InferenceEngine = request.app.state.engine
        try:
            batch = engine_impl.rerank(
                body.query, body.documents, model_id=entry.id, deadline=deadline
            )
        except QueueTimeoutError as exc:
            raise _queue_timeout_response(str(exc)) from exc
        except NonFiniteOutputError as exc:
            logger.error(
                "model '%s' produced a non-finite output on bucket %d", exc.model_id, exc.bucket
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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
        reranker_buckets = engine_impl.buckets(entry.id)
        buckets = [
            runtime.select_bucket(n_tokens, reranker_buckets)[0] for n_tokens in batch.orig_tokens
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
            model=entry.id,
            usage=Usage(prompt_tokens=used_tokens, total_tokens=used_tokens),
        )

    return app


def main() -> None:
    """Interim alias for ``python -m eeane serve``.

    ``python -m eeane.server`` predates the ``eeane`` CLI; it is kept as a
    thin wrapper so existing invocations keep working.
    """
    raise SystemExit(cli_main(["serve"]))


if __name__ == "__main__":
    main()
