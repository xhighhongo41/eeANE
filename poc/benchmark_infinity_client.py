"""HTTP + server-overhead reference client for the Infinity_emb server (v0.3実装計画.md §4.8).

Measures the current production-shaped deployment end to end: a running
Infinity_emb server (started separately, serving ruri-v3-310m and
ruri-v3-reranker-310m with ``--batch-size 8 --device mps``) is hit over HTTP.
This is a *reference* number ("HTTP + server overhead included"); the
primary GPU comparison target for eeANE (Core ML/ANE) is the no-HTTP,
no-server PyTorch/MPS baseline in ``poc/benchmark_mps.py``.

Standard library only (``urllib.request`` + ``concurrent.futures.
ThreadPoolExecutor``); no external HTTP client dependency (e.g. requests/
httpx) is added, matching v0.3実装計画.md §2.1.

``--model embedding`` chunks kokoro.txt the same way as ``poc/benchmark_mps.py``
(via ``chunk_by_tokens``) and POSTs batches of ``--per-request`` chunks to the
server's OpenAI-compatible ``/embeddings`` endpoint, optionally with
``--concurrency`` concurrent request threads.

``--model reranker`` reranks the fixed 9-query x 36-paragraph pair set one
query per Cohere-compatible ``/rerank`` request, sequentially (no
concurrency), matching a single Open WebUI rerank request.

The server's model ids and payload schema were confirmed against a live
instance (``/models``, ``/openapi.json``) before writing this client; see the
default values of ``DEFAULT_EMBEDDING_MODEL_NAME``/``DEFAULT_RERANKER_MODEL_NAME``
below and the T8 implementation report for details.

Usage:
    uv run python poc/benchmark_infinity_client.py --model embedding \\
        --chunk-tokens 512 --per-request 32 --concurrency 1
    uv run python poc/benchmark_infinity_client.py --model reranker
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python poc/benchmark_infinity_client.py` to import the poc package.
    sys.path.insert(0, str(_REPO_ROOT))

from poc.chunking import chunk_by_tokens  # noqa: E402
from poc.common import (  # noqa: E402
    CORPUS_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_RERANKER_DIR,
    load_corpus_paragraphs,
    load_rerank_queries,
    load_tokenizer,
    tokenize_batch,
    tokenize_pairs,
)

# Fixed Aozora Bunko source used for the embedding chunk set, identical to
# poc/benchmark_mps.py and poc/benchmark_throughput.py (T6/T7).
KOKORO_PATH: Path = CORPUS_DIR / "kokoro.txt"

# Default Infinity_emb server URL (current operational deployment, §0-1).
DEFAULT_BASE_URL = "http://127.0.0.1:41889"

# Model ids as actually served by the running instance (confirmed via
# `curl http://127.0.0.1:41889/models`), NOT the full `--model-id` path
# passed at server startup: Infinity_emb truncates the local model
# directory path to its last two path components ("models/<dir-name>").
# Overridable via --model-name if a different deployment reports different ids.
DEFAULT_EMBEDDING_MODEL_NAME = "models/ruri-v3-310m"
DEFAULT_RERANKER_MODEL_NAME = "models/ruri-v3-reranker-310m"

# Fixed sequence length used for the reranker effective-token count,
# matching poc/benchmark_mps.py's --seq-len default.
RERANKER_SEQ_LEN = 512

# This is a local benchmarking client for a directly-reachable server, so
# always bypass any HTTP_PROXY/HTTPS_PROXY environment configuration
# (urllib.request otherwise honors it even for --base-url hosts such as
# 127.0.0.1, which would route measurement traffic through an unrelated
# proxy and corrupt the timing/results).
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description="Measure HTTP + server-overhead end-to-end performance against a "
        "running Infinity_emb server (reference value; primary comparison is "
        "poc/benchmark_mps.py)."
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="Infinity_emb server base URL."
    )
    parser.add_argument(
        "--model",
        choices=["embedding", "reranker"],
        required=True,
        help="Which model to measure.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override the 'model' field sent in request payloads (default: the id "
        "confirmed from the server's /models response for the requested --model).",
    )
    parser.add_argument(
        "--timeout-sec", type=int, default=120, help="Per-request HTTP timeout, in seconds."
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=512,
        help="[embedding] Chunk size C for chunk_by_tokens, in tokens.",
    )
    parser.add_argument(
        "--per-request",
        type=int,
        default=32,
        help="[embedding] Number of chunks K per /embeddings request.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="[embedding] Number of concurrent request threads N.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="[embedding] If > 0, keep only the first M chunks (smoke runs). 0 = all chunks.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="[reranker] If > 0, keep only the first Q queries (smoke runs). 0 = all 9.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate numeric CLI arguments, raising SystemExit on violations.

    Args:
        args: Parsed arguments from :func:`parse_args`.

    Raises:
        SystemExit: If a strictly-positive argument is <= 0, or if a
            zero-allowed argument (``--max-chunks``/``--max-queries``) is
            negative.
    """
    if args.timeout_sec <= 0:
        raise SystemExit("--timeout-sec must be a positive integer")
    if args.chunk_tokens <= 0:
        raise SystemExit("--chunk-tokens must be a positive integer")
    if args.per_request <= 0:
        raise SystemExit("--per-request must be a positive integer")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be a positive integer")
    if args.max_chunks < 0:
        raise SystemExit("--max-chunks must be >= 0")
    if args.max_queries < 0:
        raise SystemExit("--max-queries must be >= 0")


def build_environment_info(model: str) -> dict[str, Any]:
    """Assemble client-side environment metadata recorded alongside measurements.

    Args:
        model: One of the ``--model`` CLI choices (``"embedding"`` or
            ``"reranker"``), recorded so results JSONs are self-describing.

    Returns:
        Dict of client-side version/platform metadata. Server-side version
        info is intentionally out of scope (the server is a separately
        managed process; see v0.3実装計画.md §4.8).
    """
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": model,
    }


def embedding_result_path(chunk_tokens: int, per_request: int, concurrency: int) -> Path:
    """Build the embedding-mode results JSON output path.

    Args:
        chunk_tokens: Chunk size C.
        per_request: Chunks per request K.
        concurrency: Concurrent request threads N.

    Returns:
        Path to the output JSON file under ``poc/results/``.
    """
    results_dir = _REPO_ROOT / "poc" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / f"infinity_embedding_c{chunk_tokens}_k{per_request}_conc{concurrency}.json"


def reranker_result_path(num_queries: int) -> Path:
    """Build the reranker-mode results JSON output path.

    Args:
        num_queries: Number of queries actually measured.

    Returns:
        Path to the output JSON file under ``poc/results/``.
    """
    results_dir = _REPO_ROOT / "poc" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / f"infinity_reranker_q{num_queries}.json"


def _post_json(
    base_url: str, path: str, payload: dict[str, Any], timeout_sec: int
) -> tuple[int, dict[str, Any] | None, float]:
    """POST a JSON payload and time the round trip.

    Args:
        base_url: Server base URL (e.g. ``http://127.0.0.1:41889``).
        path: Endpoint path (e.g. ``/embeddings``).
        payload: JSON-serializable request body.
        timeout_sec: Per-request timeout, in seconds.

    Returns:
        Tuple of (HTTP status code, decoded JSON response body or None if
        the body could not be decoded as JSON, elapsed wall time in seconds
        for this single request).

    Raises:
        SystemExit: If the server cannot be reached at all (connection
            refused, DNS failure, timeout, etc. -- anything that does not
            produce an HTTP status code). Non-2xx HTTP responses do NOT
            raise; they are returned normally for the caller to tally.
    """
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    try:
        with _NO_PROXY_OPENER.open(request, timeout=timeout_sec) as response:
            status = response.status
            body_bytes = response.read()
    except urllib.error.HTTPError as exc:
        # Non-2xx: a real HTTP response was received, just with an error
        # status. Keep it in the status tally rather than aborting.
        status = exc.code
        body_bytes = exc.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise SystemExit(f"Cannot reach Infinity_emb server at {url}: {exc}") from exc
    elapsed = time.perf_counter() - start

    try:
        body: dict[str, Any] | None = json.loads(body_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = None
    return status, body, elapsed


def _send_embedding_request(
    base_url: str, model_name: str, texts: list[str], timeout_sec: int
) -> tuple[int, dict[str, Any] | None, float]:
    """Send one OpenAI-compatible ``/embeddings`` request.

    Args:
        base_url: Server base URL.
        model_name: Value of the request payload's ``model`` field.
        texts: Chunk texts for this request.
        timeout_sec: Per-request timeout, in seconds.

    Returns:
        Same tuple shape as :func:`_post_json`.
    """
    payload = {"model": model_name, "input": texts}
    return _post_json(base_url, "/embeddings", payload, timeout_sec)


def _send_rerank_request(
    base_url: str, model_name: str, query: str, documents: list[str], timeout_sec: int
) -> tuple[int, dict[str, Any] | None, float]:
    """Send one Cohere-compatible ``/rerank`` request.

    Args:
        base_url: Server base URL.
        model_name: Value of the request payload's ``model`` field.
        query: Query text.
        documents: Candidate documents to rerank.
        timeout_sec: Per-request timeout, in seconds.

    Returns:
        Same tuple shape as :func:`_post_json`.
    """
    payload = {"model": model_name, "query": query, "documents": documents}
    return _post_json(base_url, "/rerank", payload, timeout_sec)


def _validate_embedding_response(body: dict[str, Any] | None) -> None:
    """Log a minimal sanity check of the first /embeddings response.

    Args:
        body: Decoded JSON body of the first measured request, or None if
            it could not be decoded.
    """
    if not body or not body.get("data"):
        print("  [warn] first embedding response has no usable 'data' field")
        return
    dim = len(body["data"][0]["embedding"])
    print(f"  first response check: {len(body['data'])} embeddings, dim={dim}")


def _validate_rerank_response(body: dict[str, Any] | None, expected_num_results: int) -> None:
    """Log a minimal sanity check of the first /rerank response.

    Args:
        body: Decoded JSON body of the first measured request, or None if
            it could not be decoded.
        expected_num_results: Number of documents submitted in that request
            (the results list should have the same length).
    """
    if not body or "results" not in body:
        print("  [warn] first rerank response has no usable 'results' field")
        return
    num_results = len(body["results"])
    print(f"  first response check: {num_results} results (expected {expected_num_results})")


def run_embedding(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    """Run the embedding HTTP benchmark against the Infinity_emb server.

    Args:
        args: Parsed and validated arguments from :func:`parse_args`.

    Returns:
        Tuple of (result dict, output JSON path). ``result`` does not yet
        include ``peak_rss_bytes``; the caller adds it right before writing.
    """
    tokenizer = load_tokenizer(DEFAULT_MODEL_DIR)
    text = KOKORO_PATH.read_text(encoding="utf-8")
    chunks = chunk_by_tokens(tokenizer, text, args.chunk_tokens)
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
    if not chunks:
        raise SystemExit("chunk_by_tokens produced no chunks from the kokoro corpus")

    # Same effective-token definition as poc/benchmark_mps.py: attention_mask
    # sum over a fixed-length (max_length=C) tokenization.
    effective_tokens = int(
        tokenize_batch(tokenizer, chunks, args.chunk_tokens)["attention_mask"].sum()
    )

    request_batches = [
        chunks[i : i + args.per_request] for i in range(0, len(chunks), args.per_request)
    ]
    model_name = args.model_name or DEFAULT_EMBEDDING_MODEL_NAME

    print(f"Warming up against {args.base_url} (model={model_name})")
    warmup_status, _warmup_body, _warmup_elapsed = _send_embedding_request(
        args.base_url, model_name, request_batches[0], args.timeout_sec
    )
    print(f"  warmup status: {warmup_status}")

    print(
        f"Running {len(request_batches)} /embeddings requests over {len(chunks)} chunks "
        f"(per_request={args.per_request}, concurrency={args.concurrency})"
    )
    status_counts: dict[int, int] = {}
    per_request_sec: list[float] = []
    first_body: dict[str, Any] | None = None

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                _send_embedding_request, args.base_url, model_name, batch, args.timeout_sec
            )
            for batch in request_batches
        ]
        # .result() preserves request_batches order regardless of completion
        # order, so index 0 below is always the first-submitted request.
        results = [future.result() for future in futures]
    total_wall_sec = time.perf_counter() - start

    for index, (status, body, elapsed) in enumerate(results):
        status_counts[status] = status_counts.get(status, 0) + 1
        per_request_sec.append(elapsed)
        if index == 0:
            first_body = body

    _validate_embedding_response(first_body)
    had_errors = any(not (200 <= status < 300) for status in status_counts)

    result: dict[str, Any] = {
        "mode": "benchmark",
        "model": "embedding",
        "num_chunks": len(chunks),
        "num_requests": len(request_batches),
        "chunk_tokens": args.chunk_tokens,
        "per_request": args.per_request,
        "concurrency": args.concurrency,
        "per_request_sec": per_request_sec,
        "total_wall_sec": total_wall_sec,
        "effective_tokens": effective_tokens,
        "effective_tokens_per_sec": effective_tokens / total_wall_sec,
        "status_counts": status_counts,
        "had_errors": had_errors,
        "base_url": args.base_url,
        "model_name": model_name,
        "environment": build_environment_info("embedding"),
    }
    out_path = embedding_result_path(args.chunk_tokens, args.per_request, args.concurrency)
    return result, out_path


def run_reranker(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    """Run the reranker HTTP benchmark against the Infinity_emb server.

    Each query is reranked against all corpus paragraphs with an
    independent ``/rerank`` request, sent sequentially (no concurrency),
    matching a single Open WebUI rerank request.

    Args:
        args: Parsed and validated arguments from :func:`parse_args`.

    Returns:
        Tuple of (result dict, output JSON path). ``result`` does not yet
        include ``peak_rss_bytes``; the caller adds it right before writing.
    """
    tokenizer = load_tokenizer(DEFAULT_RERANKER_DIR)
    queries = load_rerank_queries()
    if args.max_queries > 0:
        queries = queries[: args.max_queries]
    if not queries:
        raise SystemExit("no reranker queries selected")
    paragraphs = load_corpus_paragraphs()
    if not paragraphs:
        raise SystemExit("no corpus paragraphs available")

    # Same effective-token definition as poc/benchmark_mps.py: attention_mask
    # sum over a fixed-length (max_length=S) pair tokenization across every
    # (query, paragraph) pair used in the run.
    all_pairs = [(query["query"], paragraph) for query in queries for paragraph in paragraphs]
    effective_tokens = int(
        tokenize_pairs(tokenizer, all_pairs, RERANKER_SEQ_LEN)["attention_mask"].sum()
    )

    model_name = args.model_name or DEFAULT_RERANKER_MODEL_NAME

    print(f"Warming up against {args.base_url} (model={model_name})")
    warmup_status, _warmup_body, _warmup_elapsed = _send_rerank_request(
        args.base_url, model_name, queries[0]["query"], paragraphs, args.timeout_sec
    )
    print(f"  warmup status: {warmup_status}")

    print(
        f"Running {len(queries)} /rerank requests ({len(paragraphs)} documents/query, sequential)"
    )
    status_counts: dict[int, int] = {}
    per_query_sec: list[float] = []
    first_body: dict[str, Any] | None = None

    start = time.perf_counter()
    for index, query in enumerate(queries):
        status, body, elapsed = _send_rerank_request(
            args.base_url, model_name, query["query"], paragraphs, args.timeout_sec
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        per_query_sec.append(elapsed)
        if index == 0:
            first_body = body
    total_wall_sec = time.perf_counter() - start

    _validate_rerank_response(first_body, len(paragraphs))
    had_errors = any(not (200 <= status < 300) for status in status_counts)

    result: dict[str, Any] = {
        "mode": "benchmark",
        "model": "reranker",
        "num_queries": len(queries),
        "num_documents_per_query": len(paragraphs),
        "per_query_sec": per_query_sec,
        "per_query_median_sec": statistics.median(per_query_sec),
        "total_wall_sec": total_wall_sec,
        "effective_tokens": effective_tokens,
        "effective_tokens_per_sec": effective_tokens / total_wall_sec,
        "status_counts": status_counts,
        "had_errors": had_errors,
        "base_url": args.base_url,
        "model_name": model_name,
        "environment": build_environment_info("reranker"),
    }
    out_path = reranker_result_path(len(queries))
    return result, out_path


def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary of the benchmark result to stdout."""
    print(f"model           : {result['model']}")
    print(f"model_name      : {result['model_name']}")
    print(f"base_url        : {result['base_url']}")
    if result["model"] == "embedding":
        print(f"num_chunks      : {result['num_chunks']}")
        print(f"num_requests    : {result['num_requests']}")
        print(f"per_request     : {result['per_request']}")
        print(f"concurrency     : {result['concurrency']}")
        print(f"total_wall_sec  : {result['total_wall_sec']:.4f}")
        print(f"effective_tok/s : {result['effective_tokens_per_sec']:.2f}")
    else:
        print(f"num_queries     : {result['num_queries']}")
        print(f"num_docs/query  : {result['num_documents_per_query']}")
        print(f"total_wall_sec  : {result['total_wall_sec']:.4f}")
        print(f"per_query_median: {result['per_query_median_sec']:.4f}")
        print(f"effective_tok/s : {result['effective_tokens_per_sec']:.2f}")
    print(f"status_counts   : {result['status_counts']}")
    print(f"had_errors      : {result['had_errors']}")
    print(f"peak_rss_bytes  : {result['peak_rss_bytes']}")


def main(argv: list[str] | None = None) -> int:
    """Run the Infinity_emb HTTP reference benchmark for the requested model.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (always 0; this script is not a pass/fail gate --
        HTTP errors are tallied in the result JSON via ``had_errors``
        rather than causing a non-zero exit).

    Raises:
        SystemExit: If a CLI argument fails validation (see
            :func:`_validate_args`), no chunks/queries are selected, or the
            server cannot be reached at all.
    """
    args = parse_args(argv)
    _validate_args(args)

    if args.model == "embedding":
        result, out_path = run_embedding(args)
    else:
        result, out_path = run_reranker(args)

    # Recorded right before the measurement ends (macOS reports ru_maxrss in
    # bytes, unlike Linux's KB).
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print_summary(result)
    print(f"results         : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
