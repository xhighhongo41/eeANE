"""Load-test client for the eeANE server.

Standard library only (``urllib.request`` + ``concurrent.futures.
ThreadPoolExecutor``), reusing ``tools/corpus.py``'s deterministic Aozora
Bunko corpus loaders for request material. Like ``tools/verify_server.py``
and ``poc/benchmark_infinity_client.py``, every request is sent through an
opener that bypasses the ``HTTP_PROXY``/``HTTPS_PROXY`` environment
variables, since a proxy silently swallowing localhost traffic otherwise
turns into a confusing "server is broken" report. Requests are posted to
the server's non-versioned paths (``/embeddings``, ``/rerank``), matching
``poc/benchmark_infinity_client.py``.

Subcommands (scenarios):
    burst  Reproduces a bursty document-ingestion + query workload: ~30
           concurrent /embeddings requests (8-32 texts each) plus 3
           concurrent /rerank requests (24-36 documents each).
    short  Many concurrent /embeddings requests carrying short texts (a
           few dozen to 200 characters each), 64 texts per request, 20
           requests, to measure in-request batching/pipelining.
    limit  Sends the same mid-size /embeddings request N times
           concurrently and reports the HTTP status distribution. With
           --expect-429, also asserts that at least one HTTP 429 response
           was seen and that every 429 response carried a Retry-After
           header (non-zero exit otherwise).
    dup    Sends a byte-identical /embeddings request N times
           concurrently and asserts that every response was HTTP 200 with
           byte-identical embedding payloads (non-zero exit otherwise).

Every scenario supports --dry-run, which builds the request plan and
prints its size statistics without sending any HTTP requests. Given the
same --seed, the request plan a scenario builds is fully deterministic:
running the same scenario with --dry-run twice prints the same plan
statistics, including a SHA-256 fingerprint of the full plan.

Exit codes: 0 = the run completed (HTTP error statuses are reported, not
treated as failures, except by --expect-429/dup's own checks below), 1 =
a scenario's own consistency check failed (limit --expect-429, dup
byte-identity), 2 = the server could not be reached at all.

Usage:
    uv run python tools/load_test.py burst
    uv run python tools/load_test.py burst --dry-run
    uv run python tools/load_test.py short --concurrency 16
    uv run python tools/load_test.py limit --requests 20 --expect-429
    uv run python tools/load_test.py dup --requests 8
    uv run python tools/load_test.py burst --out results/burst.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python tools/load_test.py` to import tools.corpus regardless
    # of the current working directory.
    sys.path.insert(0, str(_REPO_ROOT))

from tools.corpus import load_corpus_paragraphs, load_rerank_queries  # noqa: E402

DEFAULT_BASE_URL = "http://127.0.0.1:7997"
DEFAULT_CONCURRENCY = 32
DEFAULT_TIMEOUT_SEC = 300.0
DEFAULT_SEED = 20260816
DEFAULT_LIMIT_REQUESTS = 12
DEFAULT_DUP_REQUESTS = 8

# burst scenario shape.
_BURST_EMBEDDING_REQUESTS = 30
_BURST_MIN_TEXTS = 8
_BURST_MAX_TEXTS = 32
_BURST_RERANK_REQUESTS = 3
_BURST_MIN_DOCS = 24
_BURST_MAX_DOCS = 36

# short scenario shape.
_SHORT_REQUESTS = 20
_SHORT_TEXTS_PER_REQUEST = 64
_SHORT_MIN_CHARS = 20
_SHORT_MAX_CHARS = 200

# limit/dup scenario shape.
_LIMIT_TEXTS_PER_REQUEST = 24
_DUP_TEXTS_PER_REQUEST = 16

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？])")


class ServerUnreachable(Exception):
    """Raised when the eeANE server did not answer with any HTTP response."""


@dataclass
class PlannedRequest:
    """One HTTP request this tool will send.

    Carries everything needed to send the request, print a per-request
    result row, and (with ``--dry-run``) summarize the plan without
    touching the network.

    Attributes:
        request_id: Stable, human-readable identifier used in per-request
            output rows and in the plan fingerprint.
        path: Request path relative to ``--base-url`` (``"/embeddings"``
            or ``"/rerank"``).
        payload: JSON-serializable request body.
        item_count: Number of texts (embedding) or documents (rerank)
            carried by ``payload``, used for ``--dry-run`` statistics.
        char_count: Total character count of every text/document/query
            carried by ``payload``, used for ``--dry-run`` statistics.
    """

    request_id: str
    path: str
    payload: dict[str, Any]
    item_count: int
    char_count: int


@dataclass
class RequestOutcome:
    """Result of sending one :class:`PlannedRequest`.

    Attributes:
        request_id: Matches the originating ``PlannedRequest.request_id``.
        path: Request path that was sent.
        status: HTTP status code.
        elapsed_sec: Wall time for this single request, in seconds.
        total_tokens: ``usage.total_tokens`` from a 200 response, or
            ``None`` when the response was not 200 or carried no usable
            usage field.
        retry_after: The response's ``Retry-After`` header value, or
            ``None`` when absent.
        embedding_fingerprint: SHA-256 hex digest of a 200
            ``/embeddings`` response's embedding vectors (ordered by
            index), or ``None`` for non-embedding or non-200 responses.
        error_detail: Short description of a non-200 response body, or
            ``""`` for 200 responses.
    """

    request_id: str
    path: str
    status: int
    elapsed_sec: float
    total_tokens: int | None
    retry_after: str | None
    embedding_fingerprint: str | None
    error_detail: str


@dataclass
class ScenarioSummary:
    """Aggregate statistics computed from a scenario run's outcomes.

    Attributes:
        status_counts: Mapping of HTTP status code to occurrence count.
        latency_p50: 50th percentile of per-request elapsed time, in
            seconds (nearest-rank).
        latency_p95: 95th percentile of per-request elapsed time, in
            seconds (nearest-rank).
        latency_max: Maximum per-request elapsed time, in seconds.
        wall_sec: Total wall-clock time for the whole concurrent send.
        total_tokens: Sum of ``total_tokens`` across requests that
            reported it.
        tokens_per_sec: ``total_tokens / wall_sec`` (``0.0`` when
            ``wall_sec`` is zero).
    """

    status_counts: dict[int, int]
    latency_p50: float
    latency_p95: float
    latency_max: float
    wall_sec: float
    total_tokens: int
    tokens_per_sec: float


# --------------------------------------------------------------------------
# Corpus loading
# --------------------------------------------------------------------------


def load_short_texts(
    paragraphs: list[str], min_chars: int = _SHORT_MIN_CHARS, max_chars: int = _SHORT_MAX_CHARS
) -> list[str]:
    """Split corpus paragraphs into short sentence-like texts.

    Splits each paragraph on Japanese sentence-ending punctuation
    (``。``, ``！``, ``？``) and keeps the pieces whose length falls
    within ``[min_chars, max_chars]``, used as the "short" scenario's
    sample pool.

    Args:
        paragraphs: Paragraphs to split, e.g. from
            :func:`tools.corpus.load_corpus_paragraphs`.
        min_chars: Minimum piece length (in characters) to keep.
        max_chars: Maximum piece length (in characters) to keep.

    Returns:
        Short texts in a fixed order (paragraph order, then sentence
        order within each paragraph).
    """
    texts: list[str] = []
    for paragraph in paragraphs:
        for piece in _SENTENCE_SPLIT_RE.split(paragraph):
            piece = piece.strip()
            if min_chars <= len(piece) <= max_chars:
                texts.append(piece)
    return texts


def _sample(rng: random.Random, pool: list[str], count: int) -> list[str]:
    """Deterministically draw ``count`` items from ``pool`` using ``rng``.

    Samples without replacement when the pool is large enough (the case
    for every scenario's pools below), falling back to sampling with
    replacement so a ``count`` larger than the pool never raises.

    Args:
        rng: Seeded random source.
        pool: Candidate items to sample from.
        count: Number of items to draw.

    Returns:
        A list of ``count`` items drawn from ``pool``.
    """
    if count <= len(pool):
        return rng.sample(pool, count)
    return rng.choices(pool, k=count)


# --------------------------------------------------------------------------
# Request plan builders
# --------------------------------------------------------------------------


def _embedding_payload(texts: list[str], model: str | None) -> dict[str, Any]:
    """Build an ``/embeddings`` request body, omitting ``model`` when unset."""
    payload: dict[str, Any] = {"input": texts}
    if model is not None:
        payload["model"] = model
    return payload


def _rerank_payload(query: str, documents: list[str], model: str | None) -> dict[str, Any]:
    """Build a ``/rerank`` request body, omitting ``model`` when unset."""
    payload: dict[str, Any] = {"query": query, "documents": documents}
    if model is not None:
        payload["model"] = model
    return payload


def build_burst_plan(
    rng: random.Random,
    paragraphs: list[str],
    queries: list[dict[str, Any]],
    embedding_model: str | None,
    reranker_model: str | None,
) -> list[PlannedRequest]:
    """Build the "burst" scenario plan.

    ``_BURST_EMBEDDING_REQUESTS`` ``/embeddings`` requests, each carrying
    ``_BURST_MIN_TEXTS``-``_BURST_MAX_TEXTS`` corpus paragraphs, followed
    by ``_BURST_RERANK_REQUESTS`` ``/rerank`` requests, each pairing one
    corpus query with ``_BURST_MIN_DOCS``-``_BURST_MAX_DOCS`` documents.

    Args:
        rng: Seeded random source, drawn from in a fixed order so the
            plan is fully determined by its seed.
        paragraphs: Corpus paragraph pool (e.g. from
            :func:`tools.corpus.load_corpus_paragraphs`).
        queries: Corpus rerank queries (e.g. from
            :func:`tools.corpus.load_rerank_queries`), each a dict with a
            ``"query"`` key.
        embedding_model: Value of the embedding requests' ``model``
            field, or ``None`` to omit it.
        reranker_model: Value of the rerank requests' ``model`` field, or
            ``None`` to omit it.

    Returns:
        Planned embedding requests followed by planned rerank requests.
    """
    planned: list[PlannedRequest] = []
    for index in range(_BURST_EMBEDDING_REQUESTS):
        size = rng.randint(_BURST_MIN_TEXTS, _BURST_MAX_TEXTS)
        texts = _sample(rng, paragraphs, size)
        payload = _embedding_payload(texts, embedding_model)
        planned.append(
            PlannedRequest(
                f"burst-embed-{index:02d}",
                "/embeddings",
                payload,
                len(texts),
                sum(len(text) for text in texts),
            )
        )

    selected_queries = rng.sample(queries, _BURST_RERANK_REQUESTS)
    for index, query in enumerate(selected_queries):
        size = rng.randint(_BURST_MIN_DOCS, _BURST_MAX_DOCS)
        documents = _sample(rng, paragraphs, size)
        payload = _rerank_payload(query["query"], documents, reranker_model)
        char_count = len(query["query"]) + sum(len(document) for document in documents)
        planned.append(
            PlannedRequest(
                f"burst-rerank-{index:02d}", "/rerank", payload, len(documents), char_count
            )
        )
    return planned


def build_short_plan(
    rng: random.Random, short_texts: list[str], embedding_model: str | None
) -> list[PlannedRequest]:
    """Build the "short" scenario plan.

    ``_SHORT_REQUESTS`` ``/embeddings`` requests, each carrying
    ``_SHORT_TEXTS_PER_REQUEST`` short texts drawn from ``short_texts``.

    Args:
        rng: Seeded random source.
        short_texts: Short-text pool, e.g. from :func:`load_short_texts`.
        embedding_model: Value of the ``model`` field, or ``None`` to
            omit it.

    Returns:
        Planned embedding requests.
    """
    planned: list[PlannedRequest] = []
    for index in range(_SHORT_REQUESTS):
        texts = _sample(rng, short_texts, _SHORT_TEXTS_PER_REQUEST)
        payload = _embedding_payload(texts, embedding_model)
        planned.append(
            PlannedRequest(
                f"short-{index:02d}",
                "/embeddings",
                payload,
                len(texts),
                sum(len(text) for text in texts),
            )
        )
    return planned


def _build_repeated_embedding_plan(
    rng: random.Random,
    paragraphs: list[str],
    embedding_model: str | None,
    texts_per_request: int,
    num_requests: int,
    id_prefix: str,
) -> list[PlannedRequest]:
    """Build ``num_requests`` planned requests sharing one identical payload.

    Shared by the "limit" and "dup" scenarios: both send the exact same
    ``/embeddings`` request body ``num_requests`` times, and only differ
    in how many texts that body carries and in how the responses are
    later validated.

    Args:
        rng: Seeded random source, used once to pick the shared text set.
        paragraphs: Corpus paragraph pool.
        embedding_model: Value of the ``model`` field, or ``None`` to
            omit it.
        texts_per_request: Number of paragraphs the shared request body
            carries.
        num_requests: Number of identical requests to plan.
        id_prefix: Prefix for each request's ``request_id``.

    Returns:
        ``num_requests`` planned requests, all carrying the same payload
        object (safe, since it is only ever read, never mutated).
    """
    texts = _sample(rng, paragraphs, texts_per_request)
    payload = _embedding_payload(texts, embedding_model)
    char_count = sum(len(text) for text in texts)
    return [
        PlannedRequest(f"{id_prefix}-{index:03d}", "/embeddings", payload, len(texts), char_count)
        for index in range(num_requests)
    ]


def build_limit_plan(
    rng: random.Random, paragraphs: list[str], embedding_model: str | None, num_requests: int
) -> list[PlannedRequest]:
    """Build the "limit" scenario plan.

    One mid-size ``/embeddings`` request sent ``num_requests`` times.
    """
    return _build_repeated_embedding_plan(
        rng, paragraphs, embedding_model, _LIMIT_TEXTS_PER_REQUEST, num_requests, "limit"
    )


def build_dup_plan(
    rng: random.Random, paragraphs: list[str], embedding_model: str | None, num_requests: int
) -> list[PlannedRequest]:
    """Build the "dup" scenario plan.

    One identical ``/embeddings`` request sent ``num_requests`` times.
    """
    return _build_repeated_embedding_plan(
        rng, paragraphs, embedding_model, _DUP_TEXTS_PER_REQUEST, num_requests, "dup"
    )


# --------------------------------------------------------------------------
# HTTP transport
# --------------------------------------------------------------------------


def _http_post(
    opener: urllib.request.OpenerDirector, url: str, payload: dict[str, Any], timeout: float
) -> tuple[int, Any, float, str | None]:
    """POST one JSON payload through the proxy-bypassing opener and time it.

    Args:
        opener: Opener built with ``ProxyHandler({})`` so localhost
            traffic is never routed through an unrelated proxy.
        url: Full request URL.
        payload: JSON-serializable request body.
        timeout: Per-request timeout, in seconds.

    Returns:
        Tuple of (HTTP status code, decoded JSON body -- or the raw text
        if it was not valid JSON -- elapsed wall time in seconds, and the
        ``Retry-After`` response header value or ``None``).

    Raises:
        ServerUnreachable: If no HTTP response was received at all
            (connection refused, DNS failure, timeout, ...).
    """
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    try:
        with opener.open(request, timeout=timeout) as response:
            status = response.status
            body_bytes = response.read()
            retry_after = response.headers.get("Retry-After")
    except urllib.error.HTTPError as exc:
        # A real HTTP response was received, just with an error status;
        # let the caller inspect and report it instead of aborting.
        status = exc.code
        body_bytes = exc.read()
        retry_after = exc.headers.get("Retry-After") if exc.headers is not None else None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ServerUnreachable(f"Cannot reach eeANE server at {url}: {exc}") from exc
    elapsed = time.perf_counter() - started

    try:
        body: Any = json.loads(body_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = body_bytes.decode("utf-8", errors="replace")
    return status, body, elapsed, retry_after


def _describe_error(status: int, body: Any) -> str:
    """Format a compact one-line description of a non-200 HTTP response."""
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
    return f"HTTP {status}: {text[:200]}"


def _extract_total_tokens(body: Any) -> int | None:
    """Extract ``usage.total_tokens`` from a decoded response body, if present and valid."""
    if isinstance(body, dict):
        usage = body.get("usage")
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            if isinstance(total, int) and not isinstance(total, bool):
                return total
    return None


def _embedding_fingerprint(body: Any) -> str | None:
    """SHA-256 hex digest of an ``/embeddings`` response's embedding vectors.

    Vectors are ordered by each item's ``index`` field before hashing, so
    the fingerprint is independent of response ordering. Used by the
    "dup" scenario to compare embedding payloads across many responses
    byte-for-byte (via a stable JSON serialization) without holding every
    response body in memory at once.

    Args:
        body: Decoded response body.

    Returns:
        A hex digest, or ``None`` if ``body`` does not carry a
        well-formed ``data`` list of ``{"index", "embedding"}`` items.
    """
    if not isinstance(body, dict):
        return None
    data = body.get("data")
    if not isinstance(data, list) or not data:
        return None
    indexed: list[tuple[int, Any]] = []
    for item in data:
        if not isinstance(item, dict) or "embedding" not in item:
            return None
        index = item.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            return None
        indexed.append((index, item["embedding"]))
    indexed.sort(key=lambda pair: pair[0])
    serialized = json.dumps([vector for _, vector in indexed], sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _send_one(
    opener: urllib.request.OpenerDirector, base_url: str, planned: PlannedRequest, timeout: float
) -> RequestOutcome:
    """Send one planned request and translate the HTTP outcome into a :class:`RequestOutcome`."""
    url = f"{base_url}{planned.path}"
    status, body, elapsed, retry_after = _http_post(opener, url, planned.payload, timeout)
    total_tokens = _extract_total_tokens(body) if status == 200 else None
    fingerprint = (
        _embedding_fingerprint(body) if status == 200 and planned.path == "/embeddings" else None
    )
    error_detail = "" if status == 200 else _describe_error(status, body)
    return RequestOutcome(
        planned.request_id,
        planned.path,
        status,
        elapsed,
        total_tokens,
        retry_after,
        fingerprint,
        error_detail,
    )


def send_all(
    opener: urllib.request.OpenerDirector,
    base_url: str,
    planned: list[PlannedRequest],
    timeout: float,
    concurrency: int,
) -> list[RequestOutcome]:
    """Submit every planned request to a shared thread pool all at once.

    Every request is submitted before any result is awaited, reproducing
    a burst of concurrent traffic rather than a steady trickle bounded by
    ``concurrency``; ``concurrency`` only bounds how many requests are
    in flight at the same time.

    Args:
        opener: Proxy-bypassing opener shared across every request.
        base_url: Server base URL (no trailing slash expected).
        planned: Requests to send.
        timeout: Per-request timeout, in seconds.
        concurrency: Maximum number of requests in flight at once.

    Returns:
        One outcome per planned request, in the same order as ``planned``.

    Raises:
        ServerUnreachable: If any request could not reach the server at
            all. Other already-submitted requests are still awaited
            (the underlying thread pool is shut down gracefully) before
            this propagates to the caller.
    """
    outcomes: list[RequestOutcome | None] = [None] * len(planned)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_index = {
            executor.submit(_send_one, opener, base_url, request, timeout): index
            for index, request in enumerate(planned)
        }
        for future in as_completed(future_to_index):
            outcomes[future_to_index[future]] = future.result()
    return cast(list[RequestOutcome], outcomes)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of a pre-sorted list (``fraction`` in ``[0, 1]``)."""
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(fraction * len(sorted_values)))
    return sorted_values[index]


def summarize(outcomes: list[RequestOutcome], wall_sec: float) -> ScenarioSummary:
    """Compute the aggregate statistics printed/reported for a completed scenario run."""
    status_counts = Counter(outcome.status for outcome in outcomes)
    latencies = sorted(outcome.elapsed_sec for outcome in outcomes)
    total_tokens = sum(
        outcome.total_tokens for outcome in outcomes if outcome.total_tokens is not None
    )
    return ScenarioSummary(
        status_counts=dict(sorted(status_counts.items())),
        latency_p50=_percentile(latencies, 0.50),
        latency_p95=_percentile(latencies, 0.95),
        latency_max=latencies[-1] if latencies else 0.0,
        wall_sec=wall_sec,
        total_tokens=total_tokens,
        tokens_per_sec=(total_tokens / wall_sec) if wall_sec > 0 else 0.0,
    )


def _plan_fingerprint(planned: list[PlannedRequest]) -> str:
    """SHA-256 hex digest of every planned request's (path, payload), in request_id order.

    Lets two runs built with the same ``--seed`` be confirmed identical
    (or a change in scenario parameters be confirmed to change the plan)
    by comparing one short string instead of the full request bodies.
    """
    ordered = sorted(planned, key=lambda request: request.request_id)
    serialized = json.dumps(
        [[request.request_id, request.path, request.payload] for request in ordered],
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _plan_stats(planned: list[PlannedRequest]) -> dict[str, Any]:
    """Compute count/size statistics of a request plan (used by ``--dry-run`` output)."""
    item_counts = [request.item_count for request in planned]
    char_counts = [request.char_count for request in planned]
    by_path = Counter(request.path for request in planned)
    return {
        "num_requests": len(planned),
        "by_path": dict(sorted(by_path.items())),
        "item_count_min": min(item_counts, default=0),
        "item_count_max": max(item_counts, default=0),
        "item_count_mean": (sum(item_counts) / len(item_counts)) if item_counts else 0.0,
        "char_count_min": min(char_counts, default=0),
        "char_count_max": max(char_counts, default=0),
        "char_count_total": sum(char_counts),
        "plan_fingerprint": _plan_fingerprint(planned),
    }


def print_dry_run_summary(scenario: str, planned: list[PlannedRequest]) -> None:
    """Print a human-readable summary of a request plan built with ``--dry-run``."""
    stats = _plan_stats(planned)
    print(f"=== {scenario} dry-run plan ===")
    print(f"requests        : {stats['num_requests']}  (by path: {stats['by_path']})")
    print(
        f"items/request   : min={stats['item_count_min']} max={stats['item_count_max']} "
        f"mean={stats['item_count_mean']:.1f}"
    )
    print(
        f"chars/request   : min={stats['char_count_min']} max={stats['char_count_max']} "
        f"total={stats['char_count_total']}"
    )
    print(f"plan_fingerprint: {stats['plan_fingerprint']}")


def print_run_summary(
    scenario: str, outcomes: list[RequestOutcome], summary: ScenarioSummary
) -> None:
    """Print per-request result rows followed by the aggregate summary."""
    print(f"=== {scenario}: per-request results ===")
    for outcome in outcomes:
        tokens = f" tokens={outcome.total_tokens}" if outcome.total_tokens is not None else ""
        retry = f" retry_after={outcome.retry_after}" if outcome.retry_after else ""
        detail = f" {outcome.error_detail}" if outcome.error_detail else ""
        print(
            f"[{outcome.status}] {outcome.request_id:<16} "
            f"elapsed={outcome.elapsed_sec:.3f}s{tokens}{retry}{detail}"
        )

    print(f"\n=== {scenario}: summary ===")
    print(f"status_counts   : {summary.status_counts}")
    print(
        f"latency p50/p95/max (s): {summary.latency_p50:.3f} / "
        f"{summary.latency_p95:.3f} / {summary.latency_max:.3f}"
    )
    print(f"wall_sec        : {summary.wall_sec:.3f}")
    print(f"total_tokens    : {summary.total_tokens}")
    print(f"tokens_per_sec  : {summary.tokens_per_sec:.1f}")


def _environment_info() -> dict[str, Any]:
    """Client-side platform/interpreter metadata recorded in ``--out`` JSON."""
    return {"python": platform.python_version(), "platform": platform.platform()}


def _scenario_parameters(args: argparse.Namespace) -> dict[str, Any]:
    """Serializable snapshot of the CLI arguments that define a run/plan."""
    params: dict[str, Any] = {
        "scenario": args.command,
        "base_url": args.base_url,
        "embedding_model": args.embedding_model,
        "reranker_model": args.reranker_model,
        "concurrency": args.concurrency,
        "timeout_sec": args.timeout_sec,
        "seed": args.seed,
        "dry_run": args.dry_run,
    }
    if hasattr(args, "requests"):
        params["requests"] = args.requests
    if hasattr(args, "expect_429"):
        params["expect_429"] = args.expect_429
    return params


def build_dry_run_report(planned: list[PlannedRequest], args: argparse.Namespace) -> dict[str, Any]:
    """Assemble the ``--out`` JSON document for a ``--dry-run`` invocation."""
    return {
        "executed_at": datetime.now(UTC).isoformat(),
        "parameters": _scenario_parameters(args),
        "environment": _environment_info(),
        "plan": _plan_stats(planned),
        "requests": [
            {
                "request_id": request.request_id,
                "path": request.path,
                "item_count": request.item_count,
                "char_count": request.char_count,
            }
            for request in planned
        ],
    }


def build_run_report(
    outcomes: list[RequestOutcome], summary: ScenarioSummary, args: argparse.Namespace
) -> dict[str, Any]:
    """Assemble the ``--out`` JSON document for a completed HTTP run."""
    return {
        "executed_at": datetime.now(UTC).isoformat(),
        "parameters": _scenario_parameters(args),
        "environment": _environment_info(),
        "summary": asdict(summary),
        "requests": [
            {
                "request_id": outcome.request_id,
                "path": outcome.path,
                "status": outcome.status,
                "elapsed_sec": outcome.elapsed_sec,
                "total_tokens": outcome.total_tokens,
                "retry_after": outcome.retry_after,
                "error_detail": outcome.error_detail,
            }
            for outcome in outcomes
        ],
    }


def _write_json(path: Path, report: dict[str, Any]) -> None:
    """Write ``report`` as indented JSON to ``path``, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# Scenario-specific validation
# --------------------------------------------------------------------------


def validate_limit_expect_429(outcomes: list[RequestOutcome]) -> str | None:
    """Check that at least one HTTP 429 response was seen and all carried Retry-After.

    Args:
        outcomes: Results from the "limit" scenario's concurrent send.

    Returns:
        ``None`` if the check passed, otherwise a short failure
        description.
    """
    rejected = [outcome for outcome in outcomes if outcome.status == 429]
    if not rejected:
        return "expected at least one HTTP 429 response, got none"
    missing = [outcome.request_id for outcome in rejected if not outcome.retry_after]
    if missing:
        return (
            f"{len(missing)}/{len(rejected)} HTTP 429 responses are missing a "
            f"Retry-After header: {', '.join(missing)}"
        )
    return None


def validate_dup_consistency(outcomes: list[RequestOutcome]) -> str | None:
    """Check that every "dup" response was HTTP 200 with byte-identical embedding payloads.

    Args:
        outcomes: Results from the "dup" scenario's concurrent send.

    Returns:
        ``None`` if the check passed, otherwise a short failure
        description.
    """
    non_200 = [outcome.request_id for outcome in outcomes if outcome.status != 200]
    if non_200:
        return (
            f"{len(non_200)}/{len(outcomes)} requests did not return HTTP 200: {', '.join(non_200)}"
        )
    unusable = [outcome.request_id for outcome in outcomes if outcome.embedding_fingerprint is None]
    if unusable:
        return f"{len(unusable)} responses had an unusable embedding payload: {', '.join(unusable)}"
    fingerprints = {outcome.embedding_fingerprint for outcome in outcomes}
    if len(fingerprints) != 1:
        return (
            f"embedding payloads differ across {len(fingerprints)} distinct fingerprints "
            f"among {len(outcomes)} responses"
        )
    return None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for every scenario subcommand."""
    parser = argparse.ArgumentParser(description="Load-test client for the eeANE server.")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"eeANE server base URL (default: {DEFAULT_BASE_URL}).",
    )
    common.add_argument(
        "--embedding-model",
        default=None,
        metavar="ID",
        help="Value of the 'model' field sent with embedding requests. "
        "Omit to send no 'model' field (server default embedding model).",
    )
    common.add_argument(
        "--reranker-model",
        default=None,
        metavar="ID",
        help="Value of the 'model' field sent with rerank requests (used by the 'burst' "
        "scenario). Omit to send no 'model' field (server default reranker).",
    )
    common.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Maximum number of requests in flight at once (default: {DEFAULT_CONCURRENCY}).",
    )
    common.add_argument(
        "--timeout-sec",
        type=float,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"Per-request HTTP timeout, in seconds (default: {DEFAULT_TIMEOUT_SEC}).",
    )
    common.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for deterministic request-plan construction (default: {DEFAULT_SEED}).",
    )
    common.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write results (or, with --dry-run, plan statistics) as JSON to this path.",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the request plan and print its statistics without sending any HTTP requests.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "burst",
        parents=[common],
        help="Burst of concurrent embedding + rerank requests (document-ingestion workload).",
    )
    subparsers.add_parser(
        "short",
        parents=[common],
        help="Many concurrent embedding requests carrying short texts.",
    )

    limit_parser = subparsers.add_parser(
        "limit",
        parents=[common],
        help="Send the same mid-size embedding request N times concurrently (queue-limit test).",
    )
    limit_parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_LIMIT_REQUESTS,
        help=(
            f"Number of concurrent identical requests to send (default: {DEFAULT_LIMIT_REQUESTS})."
        ),
    )
    limit_parser.add_argument(
        "--expect-429",
        action="store_true",
        help="Assert at least one HTTP 429 response with a Retry-After header (exit 1 otherwise).",
    )

    dup_parser = subparsers.add_parser(
        "dup",
        parents=[common],
        help="Send a byte-identical embedding request N times concurrently (dedup/merge probe).",
    )
    dup_parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_DUP_REQUESTS,
        help=f"Number of concurrent identical requests to send (default: {DEFAULT_DUP_REQUESTS}).",
    )

    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Validate numeric CLI arguments, raising ``SystemExit`` on violations.

    Args:
        args: Parsed arguments from :func:`build_parser`.

    Raises:
        SystemExit: If a strictly-positive argument is not positive.
    """
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be a positive integer")
    if args.timeout_sec <= 0:
        raise SystemExit("--timeout-sec must be a positive number")
    requests_count = getattr(args, "requests", None)
    if requests_count is not None and requests_count <= 0:
        raise SystemExit("--requests must be a positive integer")


def _build_plan(args: argparse.Namespace) -> list[PlannedRequest]:
    """Build the request plan for ``args.command``, deterministic given ``args.seed``.

    Args:
        args: Parsed and validated CLI arguments.

    Returns:
        The scenario's planned requests, in send order.
    """
    rng = random.Random(args.seed)
    paragraphs = load_corpus_paragraphs()
    if args.command == "burst":
        queries = load_rerank_queries()
        return build_burst_plan(rng, paragraphs, queries, args.embedding_model, args.reranker_model)
    if args.command == "short":
        short_texts = load_short_texts(paragraphs)
        return build_short_plan(rng, short_texts, args.embedding_model)
    if args.command == "limit":
        return build_limit_plan(rng, paragraphs, args.embedding_model, args.requests)
    return build_dup_plan(rng, paragraphs, args.embedding_model, args.requests)


def main(argv: list[str] | None = None) -> int:
    """Run the requested scenario and translate its outcome into an exit code.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        0 if the run completed (including one with only non-2xx HTTP
        responses), 1 if a scenario's own consistency check failed
        (``limit --expect-429``, ``dup``), 2 if the server could not be
        reached at all.
    """
    args = build_parser().parse_args(argv)
    _validate_args(args)

    planned = _build_plan(args)

    if args.dry_run:
        print_dry_run_summary(args.command, planned)
        if args.out is not None:
            _write_json(args.out, build_dry_run_report(planned, args))
        return 0

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        started = time.perf_counter()
        outcomes = send_all(opener, args.base_url, planned, args.timeout_sec, args.concurrency)
        wall_sec = time.perf_counter() - started
    except ServerUnreachable as exc:
        print(str(exc))
        print("Start the eeANE server first: uv run python -m eeane.server")
        return 2

    summary = summarize(outcomes, wall_sec)
    print_run_summary(args.command, outcomes, summary)
    if args.out is not None:
        _write_json(args.out, build_run_report(outcomes, summary, args))

    exit_code = 0
    if args.command == "limit" and args.expect_429:
        problem = validate_limit_expect_429(outcomes)
        if problem is None:
            print("[PASS] expect-429 check")
        else:
            print(f"[FAIL] expect-429 check: {problem}")
            exit_code = 1
    if args.command == "dup":
        problem = validate_dup_consistency(outcomes)
        if problem is None:
            print("[PASS] dedup consistency check")
        else:
            print(f"[FAIL] dedup consistency check: {problem}")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
