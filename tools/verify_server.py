"""Verification client for the eeANE server (v0.4実装計画.md §4.7, T5).

Standard library only (``urllib.request``), with the eeANE package and a
handful of read-only test-data loaders from ``poc/common.py`` imported for
building inputs and computing the Core ML direct-predict baseline. Like
``poc/benchmark_infinity_client.py`` (v0.3 T8), every request is sent
through an opener that explicitly bypasses ``HTTP_PROXY``/``HTTPS_PROXY``
environment variables, since a proxy silently swallowing localhost traffic
was a real incident in v0.3.

Subcommands:
    health           GET /health shape check (R1).
    verify-embedding /v1/embeddings OpenAI compatibility + accuracy (R2).
    verify-rerank    /rerank, /v1/rerank Infinity compatibility + accuracy (R3).
    bench            HTTP round-trip latency measurement (part of R5).
    all              Runs the four subcommands above in order.

Exit codes: 0 = every check passed, 1 = at least one check failed, 2 = the
server could not be reached at all (start it first with
``uv run python -m eeane.server``).

Usage:
    uv run python tools/verify_server.py health
    uv run python tools/verify_server.py verify-embedding
    uv run python tools/verify_server.py verify-rerank
    uv run python tools/verify_server.py bench
    uv run python tools/verify_server.py all
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python tools/verify_server.py` to import the eeane/poc packages
    # regardless of the current working directory.
    sys.path.insert(0, str(_REPO_ROOT))

from eeane import runtime, settings  # noqa: E402
from eeane.engine import CoreMLEngine  # noqa: E402
from poc.common import (  # noqa: E402
    CORPUS_DIR,
    PREFIX_TEST_SENTENCES,
    PREFIXES,
    _split_paragraphs,
    load_corpus_paragraphs,
    load_rerank_queries,
)

DEFAULT_BASE_URL = "http://127.0.0.1:7997"
DEFAULT_TIMEOUT = 120.0

# (source work, corpus file, kokoro-only paragraph cap) in the exact file
# order load_corpus_paragraphs() uses, so per-paragraph work tags can be
# reconstructed alongside it (poc.common discards the work name).
_WORK_FILES: list[tuple[str, Path, int | None]] = [
    ("kumonoito", CORPUS_DIR / "kumonoito.txt", None),
    ("sangetsuki", CORPUS_DIR / "sangetsuki.txt", None),
    ("kokoro", CORPUS_DIR / "kokoro.txt", 30),
]


class ServerUnreachable(Exception):
    """Raised when the eeANE server did not answer with any HTTP response."""


@dataclass
class CheckResult:
    """One row of a PASS/FAIL summary table.

    Attributes:
        name: Short machine-readable identifier for the check.
        passed: Whether the check succeeded.
        detail: Human-readable context (measured values, mismatches, ...),
            shown next to the PASS/FAIL badge.
    """

    name: str
    passed: bool
    detail: str = ""


class BaselineEngine:
    """Lazily builds and caches the direct-predict Core ML baseline engine.

    ``verify-embedding`` and ``verify-rerank`` both need
    :class:`~eeane.engine.CoreMLEngine` as ground truth; when both run
    (the ``all`` subcommand), the artifacts must be loaded only once.
    """

    def __init__(self) -> None:
        """Initialize with no engine loaded yet."""
        self._engine: CoreMLEngine | None = None

    def get(self) -> CoreMLEngine:
        """Return the cached engine, building it on first use."""
        if self._engine is None:
            print("Loading baseline CoreMLEngine (direct Core ML predict, in-process)...")
            started = time.perf_counter()
            self._engine = CoreMLEngine.from_settings()
            print(f"  loaded in {time.perf_counter() - started:.2f}s")
        return self._engine


def _cosine_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two equally-shaped (N, D) arrays.

    Args:
        a: Array of shape (N, D).
        b: Array of shape (N, D).

    Returns:
        Cosine similarities, shape (N,). Zero-norm rows are protected with
        a small epsilon to avoid division by zero.
    """
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    denom = np.maximum(norm_a * norm_b, 1e-12)
    return dot / denom


def load_paragraph_works(min_chars: int = 40) -> list[str]:
    """Load the source-work tag of every paragraph in ``load_corpus_paragraphs()``.

    Rebuilds the paragraph list from the three corpus files using exactly
    the same filter as ``poc.common.load_corpus_paragraphs`` (file order,
    ``min_chars``, kokoro capped at its first 30 filtered paragraphs), but
    keeps the source work name that the shared loader discards. The
    reconstructed paragraph texts are compared against
    ``load_corpus_paragraphs()`` and must match exactly, so any future
    drift between the two implementations fails loudly instead of silently
    mislabeling paragraphs.

    Args:
        min_chars: Minimum paragraph length, matching
            ``load_corpus_paragraphs``'s default.

    Returns:
        Source work name (``"kumonoito"``, ``"sangetsuki"``, or
        ``"kokoro"``) for each paragraph, aligned index-for-index with
        ``load_corpus_paragraphs()``.

    Raises:
        RuntimeError: If the reconstructed paragraph list does not exactly
            match ``load_corpus_paragraphs()`` (count or content).
    """
    paragraphs: list[str] = []
    works: list[str] = []
    for work, path, limit in _WORK_FILES:
        text = path.read_text(encoding="utf-8")
        blocks = _split_paragraphs(text)
        blocks = [block for block in blocks if len(block) >= min_chars]
        if limit is not None:
            blocks = blocks[:limit]
        paragraphs.extend(blocks)
        works.extend([work] * len(blocks))

    expected = load_corpus_paragraphs()
    if paragraphs != expected:
        raise RuntimeError(
            "load_paragraph_works() diverged from poc.common.load_corpus_paragraphs(): "
            f"got {len(paragraphs)} paragraphs, expected {len(expected)}"
        )
    return works


def _request_json(
    opener: urllib.request.OpenerDirector,
    method: str,
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> tuple[int, Any, float]:
    """Send one HTTP request through the proxy-bypassing opener and time it.

    Args:
        opener: Opener built with ``ProxyHandler({})`` so localhost traffic
            is never routed through an unrelated proxy (§4.7).
        method: HTTP method (``"GET"`` or ``"POST"``).
        url: Full request URL.
        payload: JSON-serializable request body for POST; ``None`` for GET.
        timeout: Per-request timeout, in seconds.

    Returns:
        Tuple of (HTTP status code, decoded JSON body -- or the raw text if
        it was not valid JSON -- and elapsed wall time in seconds).

    Raises:
        ServerUnreachable: If no HTTP response was received at all
            (connection refused, DNS failure, timeout, ...).
    """
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with opener.open(request, timeout=timeout) as response:
            status = response.status
            body_bytes = response.read()
    except urllib.error.HTTPError as exc:
        # A real HTTP response was received, just with an error status;
        # let the caller inspect and report it instead of aborting.
        status = exc.code
        body_bytes = exc.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise ServerUnreachable(f"Cannot reach eeANE server at {url}: {exc}") from exc
    elapsed = time.perf_counter() - started

    try:
        body: Any = json.loads(body_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = body_bytes.decode("utf-8", errors="replace")
    return status, body, elapsed


def _describe_error(status: int, body: Any) -> str:
    """Format a compact one-line description of a non-200 HTTP response.

    Args:
        status: HTTP status code.
        body: Decoded response body (dict, list, or raw text).

    Returns:
        ``"HTTP <status>: <first 200 chars of the body>"``.
    """
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
    return f"HTTP {status}: {text[:200]}"


def _results_list(body: Any) -> list[Any]:
    """Extract a rerank response's ``results`` list, defaulting to empty.

    Args:
        body: Decoded response body.

    Returns:
        ``body["results"]`` if it is present and a list, otherwise ``[]``.
    """
    if isinstance(body, dict) and isinstance(body.get("results"), list):
        return body["results"]
    return []


def _print_summary(title: str, checks: list[CheckResult]) -> bool:
    """Print a PASS/FAIL summary table and return whether every check passed.

    Args:
        title: Subcommand name, used as the table header.
        checks: Checks accumulated while running the subcommand.

    Returns:
        ``True`` if every check in ``checks`` passed.
    """
    print(f"\n=== {title} summary ===")
    name_width = max((len(check.name) for check in checks), default=4)
    for check in checks:
        badge = "PASS" if check.passed else "FAIL"
        line = f"[{badge}] {check.name.ljust(name_width)}"
        if check.detail:
            line += f"  {check.detail}"
        print(line)
    passed_count = sum(check.passed for check in checks)
    overall = passed_count == len(checks)
    print(f"{title}: {'PASS' if overall else 'FAIL'} ({passed_count}/{len(checks)})")
    return overall


def cmd_health(args: argparse.Namespace, opener: urllib.request.OpenerDirector) -> bool:
    """Verify ``GET /health`` responds with the expected shape (R1).

    Args:
        args: Parsed CLI arguments (uses ``base_url``, ``timeout``).
        opener: Proxy-bypassing opener shared across subcommands.

    Returns:
        Whether every check passed.
    """
    status, body, elapsed = _request_json(
        opener, "GET", f"{args.base_url}/health", None, args.timeout
    )
    checks: list[CheckResult] = [CheckResult("http_200", status == 200, f"elapsed={elapsed:.3f}s")]
    if status != 200:
        checks[-1].detail = _describe_error(status, body)
        skipped_names = (
            "status_ok",
            "version_present",
            "models_embedding_list",
            "models_reranker_list",
        )
        for name in skipped_names:
            checks.append(CheckResult(name, False, "skipped: non-200 response"))
        return _print_summary("health", checks)

    payload = body if isinstance(body, dict) else {}
    checks.append(
        CheckResult("status_ok", payload.get("status") == "ok", f"status={payload.get('status')!r}")
    )
    version = payload.get("version")
    checks.append(
        CheckResult(
            "version_present", isinstance(version, str) and bool(version), f"version={version!r}"
        )
    )
    models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
    embedding_list = models.get("embedding")
    reranker_list = models.get("reranker")
    checks.append(
        CheckResult(
            "models_embedding_list",
            isinstance(embedding_list, list),
            f"embedding={embedding_list!r}",
        )
    )
    checks.append(
        CheckResult(
            "models_reranker_list", isinstance(reranker_list, list), f"reranker={reranker_list!r}"
        )
    )
    return _print_summary("health", checks)


def cmd_verify_embedding(
    args: argparse.Namespace, opener: urllib.request.OpenerDirector, baseline: BaselineEngine
) -> bool:
    """Verify ``/v1/embeddings`` OpenAI compatibility and accuracy (R2).

    Args:
        args: Parsed CLI arguments (uses ``base_url``, ``timeout``).
        opener: Proxy-bypassing opener shared across subcommands.
        baseline: Shared, lazily-built Core ML baseline engine.

    Returns:
        Whether every check passed.
    """
    checks: list[CheckResult] = []

    paragraphs = load_corpus_paragraphs()
    query_sentences = [PREFIXES["query"] + sentence for sentence in PREFIX_TEST_SENTENCES[:4]]
    document_sentences = [PREFIXES["document"] + sentence for sentence in PREFIX_TEST_SENTENCES[4:]]
    texts = paragraphs + query_sentences + document_sentences
    checks.append(CheckResult("input_count_44", len(texts) == 44, f"got {len(texts)}"))

    status, body, elapsed = _request_json(
        opener, "POST", f"{args.base_url}/v1/embeddings", {"input": texts}, args.timeout
    )
    if status != 200:
        checks.append(CheckResult("http_200", False, _describe_error(status, body)))
        for name in ("data_count", "index_order", "embedding_dim_768"):
            checks.append(CheckResult(name, False, "skipped: non-200 response"))
        no_response = "skipped: no successful embeddings response"
        for name in ("cosine_min_0.999999", "usage_prompt_tokens", "bucket_routing_128_and_512"):
            checks.append(CheckResult(name, False, no_response))
        checks.append(CheckResult("base64_roundtrip", False, no_response))
        return _print_summary("verify-embedding", checks)
    checks.append(CheckResult("http_200", True, f"elapsed={elapsed:.3f}s"))

    data = body.get("data") if isinstance(body, dict) else None
    shape_ok = isinstance(data, list) and len(data) == len(texts)
    checks.append(
        CheckResult("data_count", shape_ok, f"got {len(data) if isinstance(data, list) else 'n/a'}")
    )
    if not shape_ok:
        for name in ("index_order", "embedding_dim_768"):
            checks.append(CheckResult(name, False, "skipped: bad data shape"))
        for name in ("cosine_min_0.999999", "usage_prompt_tokens", "bucket_routing_128_and_512"):
            checks.append(CheckResult(name, False, "skipped: bad data shape"))
        checks.append(CheckResult("base64_roundtrip", False, "skipped: bad data shape"))
        return _print_summary("verify-embedding", checks)

    checks.append(
        CheckResult("index_order", [item["index"] for item in data] == list(range(len(texts))))
    )
    dims_ok = all(len(item["embedding"]) == 768 for item in data)
    checks.append(CheckResult("embedding_dim_768", dims_ok))

    http_vectors = np.asarray([item["embedding"] for item in data], dtype=np.float64)

    engine = baseline.get()
    baseline_batch = engine.embed(texts)
    baseline_vectors = baseline_batch.vectors.astype(np.float64)
    if settings.NORMALIZE_EMBEDDINGS:
        baseline_vectors = runtime.l2_normalize(baseline_vectors)

    cosine = _cosine_rowwise(http_vectors, baseline_vectors)
    cosine_min = float(cosine.min())
    cosine_mean = float(cosine.mean())
    worst_index = int(cosine.argmin())
    checks.append(
        CheckResult(
            "cosine_min_0.999999",
            cosine_min >= 0.999999,
            f"min={cosine_min:.8f} mean={cosine_mean:.8f} worst_index={worst_index}",
        )
    )

    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    http_prompt_tokens = usage.get("prompt_tokens")
    baseline_prompt_tokens = int(sum(baseline_batch.used_tokens))
    checks.append(
        CheckResult(
            "usage_prompt_tokens",
            http_prompt_tokens == baseline_prompt_tokens,
            f"http={http_prompt_tokens} baseline_sum={baseline_prompt_tokens}",
        )
    )

    bucket_counts = Counter(baseline_batch.buckets)
    checks.append(
        CheckResult(
            "bucket_routing_128_and_512",
            128 in bucket_counts and 512 in bucket_counts,
            f"distribution={dict(sorted(bucket_counts.items()))}",
        )
    )

    b64_status, b64_body, _elapsed = _request_json(
        opener,
        "POST",
        f"{args.base_url}/v1/embeddings",
        {"input": [texts[0]], "encoding_format": "base64"},
        args.timeout,
    )
    if b64_status == 200 and isinstance(b64_body, dict):
        b64_data = b64_body.get("data") or [{}]
        encoded = b64_data[0].get("embedding")
        if isinstance(encoded, str):
            decoded = runtime.base64_to_floats(encoded).astype(np.float64)
            b64_ok = decoded.shape == (768,) and np.allclose(decoded, http_vectors[0], atol=1e-6)
            detail = "" if b64_ok else f"max|delta|={np.max(np.abs(decoded - http_vectors[0])):.3e}"
        else:
            b64_ok = False
            detail = f"embedding field is not a string: {type(encoded)!r}"
    else:
        b64_ok = False
        detail = _describe_error(b64_status, b64_body)
    checks.append(CheckResult("base64_roundtrip", b64_ok, detail))

    return _print_summary("verify-embedding", checks)


def cmd_verify_rerank(
    args: argparse.Namespace, opener: urllib.request.OpenerDirector, baseline: BaselineEngine
) -> bool:
    """Verify ``/rerank``/``/v1/rerank`` Infinity compatibility and accuracy (R3).

    Args:
        args: Parsed CLI arguments (uses ``base_url``, ``timeout``).
        opener: Proxy-bypassing opener shared across subcommands.
        baseline: Shared, lazily-built Core ML baseline engine.

    Returns:
        Whether every check passed.
    """
    checks: list[CheckResult] = []

    queries = load_rerank_queries()
    paragraphs = load_corpus_paragraphs()
    works = load_paragraph_works()
    engine = baseline.get()

    request_issues: list[str] = []
    sort_issues: list[str] = []
    top1_mismatches: list[str] = []
    max_score_delta = 0.0
    worst_detail = "n/a"

    for query in queries:
        status, body, _elapsed = _request_json(
            opener,
            "POST",
            f"{args.base_url}/rerank",
            {"query": query["query"], "documents": paragraphs},
            args.timeout,
        )
        if status != 200:
            request_issues.append(f"{query['id']}: {_describe_error(status, body)}")
            continue
        results = _results_list(body)
        if len(results) != len(paragraphs):
            request_issues.append(f"{query['id']}: bad results shape (got {len(results)})")
            continue

        scores_seq = [result["relevance_score"] for result in results]
        if scores_seq != sorted(scores_seq, reverse=True):
            sort_issues.append(query["id"])

        top_index = results[0]["index"]
        actual_work = works[top_index]
        expected_work = query["source_work"]
        if actual_work != expected_work:
            top1_mismatches.append(
                f"{query['id']}: expected={expected_work} actual={actual_work} index={top_index}"
            )

        rerank_batch = engine.rerank(query["query"], paragraphs)
        baseline_scores = runtime.sigmoid(rerank_batch.logits)
        for result in results:
            index = result["index"]
            delta = abs(float(result["relevance_score"]) - float(baseline_scores[index]))
            if delta > max_score_delta:
                max_score_delta = delta
                worst_detail = f"query={query['id']} index={index}"

    requests_detail = "; ".join(request_issues) if request_issues else "9/9 ok"
    checks.append(CheckResult("requests_200", not request_issues, requests_detail))
    checks.append(
        CheckResult(
            "results_sorted_desc",
            not sort_issues,
            "; ".join(sort_issues) if sort_issues else "ok",
        )
    )
    checks.append(
        CheckResult(
            "top1_source_work_9_9",
            not request_issues and not top1_mismatches,
            "; ".join(top1_mismatches)
            if top1_mismatches
            else f"{len(queries) - len(request_issues)}/{len(queries)}",
        )
    )
    checks.append(
        CheckResult(
            "score_max_delta_1e-6",
            not request_issues and max_score_delta <= 1e-6,
            f"max|delta|={max_score_delta:.3e} ({worst_detail})",
        )
    )

    sample_query = queries[0]["query"]

    status, body, _elapsed = _request_json(
        opener,
        "POST",
        f"{args.base_url}/rerank",
        {"query": sample_query, "documents": paragraphs, "top_n": 5},
        args.timeout,
    )
    top_n_results = _results_list(body)
    checks.append(
        CheckResult(
            "top_n_5",
            status == 200 and len(top_n_results) == 5,
            f"status={status} results={len(top_n_results)}",
        )
    )

    status, body, _elapsed = _request_json(
        opener,
        "POST",
        f"{args.base_url}/rerank",
        {"query": sample_query, "documents": paragraphs, "return_documents": True},
        args.timeout,
    )
    return_documents_results = _results_list(body)
    return_documents_ok = len(return_documents_results) == len(paragraphs) and all(
        result.get("document") == paragraphs[result["index"]] for result in return_documents_results
    )
    checks.append(
        CheckResult("return_documents", status == 200 and return_documents_ok, f"status={status}")
    )

    status, body, _elapsed = _request_json(
        opener,
        "POST",
        f"{args.base_url}/rerank",
        {"query": sample_query, "documents": paragraphs, "raw_scores": True},
        args.timeout,
    )
    raw_results = _results_list(body)
    if status == 200 and len(raw_results) == len(paragraphs):
        raw_batch = engine.rerank(sample_query, paragraphs)
        raw_max_delta = max(
            abs(float(result["relevance_score"]) - float(raw_batch.logits[result["index"]]))
            for result in raw_results
        )
        raw_scores_ok = raw_max_delta <= 1e-5
        raw_detail = f"max|delta|={raw_max_delta:.3e}"
    else:
        raw_scores_ok = False
        raw_detail = _describe_error(status, body)
    checks.append(CheckResult("raw_scores_max_delta_1e-5", raw_scores_ok, raw_detail))

    status_v1, body_v1, _elapsed = _request_json(
        opener,
        "POST",
        f"{args.base_url}/v1/rerank",
        {"query": sample_query, "documents": paragraphs},
        args.timeout,
    )
    status_legacy, body_legacy, _elapsed = _request_json(
        opener,
        "POST",
        f"{args.base_url}/rerank",
        {"query": sample_query, "documents": paragraphs},
        args.timeout,
    )
    v1_parity_ok = (
        status_v1 == 200
        and status_legacy == 200
        and _results_list(body_v1) == _results_list(body_legacy)
        and len(_results_list(body_v1)) == len(paragraphs)
    )
    checks.append(
        CheckResult(
            "v1_rerank_matches_rerank",
            v1_parity_ok,
            f"status_v1={status_v1} status_legacy={status_legacy}",
        )
    )

    return _print_summary("verify-rerank", checks)


def cmd_bench(args: argparse.Namespace, opener: urllib.request.OpenerDirector) -> bool:
    """Measure HTTP round-trip latency for rerank and embedding (part of R5).

    No pass/fail judgment is made here (all reported measurements are
    informational); the return value is always ``True`` unless the server
    is unreachable, in which case :class:`ServerUnreachable` propagates to
    the caller.

    Args:
        args: Parsed CLI arguments (uses ``base_url``, ``timeout``).
        opener: Proxy-bypassing opener shared across subcommands.

    Returns:
        Always ``True``.
    """
    queries = load_rerank_queries()
    paragraphs = load_corpus_paragraphs()

    print(
        f"Running {len(queries)} sequential /rerank requests ({len(paragraphs)} documents each)..."
    )
    rerank_times: list[float] = []
    for query in queries:
        status, _body, elapsed = _request_json(
            opener,
            "POST",
            f"{args.base_url}/rerank",
            {"query": query["query"], "documents": paragraphs},
            args.timeout,
        )
        rerank_times.append(elapsed)
        print(f"  {query['id']}: status={status} elapsed={elapsed:.3f}s")

    print(f"Running 1 /v1/embeddings request with {len(paragraphs)} paragraphs...")
    status, _body, embedding_elapsed = _request_json(
        opener, "POST", f"{args.base_url}/v1/embeddings", {"input": paragraphs}, args.timeout
    )
    print(f"  status={status} elapsed={embedding_elapsed:.3f}s")

    result: dict[str, Any] = {
        "base_url": args.base_url,
        "executed_at": datetime.now(UTC).isoformat(),
        "rerank": {
            "num_queries": len(queries),
            "num_documents_per_query": len(paragraphs),
            "per_query_sec": rerank_times,
            "median_sec": statistics.median(rerank_times),
            "mean_sec": statistics.mean(rerank_times),
            "min_sec": min(rerank_times),
            "max_sec": max(rerank_times),
        },
        "embedding": {
            "num_paragraphs": len(paragraphs),
            "elapsed_sec": embedding_elapsed,
        },
    }

    results_dir = _REPO_ROOT / "poc" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"server_bench_{timestamp}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    rerank_stats = result["rerank"]
    rerank_detail = (
        f"median={rerank_stats['median_sec']:.3f}s mean={rerank_stats['mean_sec']:.3f}s "
        f"min={rerank_stats['min_sec']:.3f}s max={rerank_stats['max_sec']:.3f}s"
    )
    checks = [
        CheckResult("rerank_requests_completed", True, rerank_detail),
        CheckResult("embedding_request_completed", True, f"elapsed={embedding_elapsed:.3f}s"),
    ]
    _print_summary("bench", checks)
    print(f"results saved to {out_path}")
    return True


def cmd_all(args: argparse.Namespace, opener: urllib.request.OpenerDirector) -> bool:
    """Run health, verify-embedding, verify-rerank, then bench in order.

    Args:
        args: Parsed CLI arguments (uses ``base_url``, ``timeout``).
        opener: Proxy-bypassing opener shared across subcommands.

    Returns:
        Whether health, verify-embedding, and verify-rerank all passed
        (bench makes no pass/fail judgment, see :func:`cmd_bench`).
    """
    baseline = BaselineEngine()
    outcomes = {
        "health": cmd_health(args, opener),
        "verify-embedding": cmd_verify_embedding(args, opener, baseline),
        "verify-rerank": cmd_verify_rerank(args, opener, baseline),
    }
    cmd_bench(args, opener)

    overall = all(outcomes.values())
    print("\n=== all: overall summary ===")
    for name, passed in outcomes.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print("[ n/a] bench (no pass/fail judgment, see poc/results/ for the measurement)")
    print(f"all: {'PASS' if overall else 'FAIL'}")
    return overall


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for all subcommands."""
    parser = argparse.ArgumentParser(description="Verification client for the eeANE server.")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"eeANE server base URL (default: {DEFAULT_BASE_URL})",
    )
    common.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Per-request HTTP timeout, in seconds (default: {DEFAULT_TIMEOUT}).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health", parents=[common], help="Check GET /health (R1).")
    subparsers.add_parser(
        "verify-embedding",
        parents=[common],
        help="Verify /v1/embeddings compatibility and accuracy (R2).",
    )
    subparsers.add_parser(
        "verify-rerank",
        parents=[common],
        help="Verify /rerank and /v1/rerank compatibility and accuracy (R3).",
    )
    subparsers.add_parser("bench", parents=[common], help="Measure HTTP round-trip latency (R5).")
    subparsers.add_parser(
        "all", parents=[common], help="Run health, verify-embedding, verify-rerank, bench."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested subcommand and translate its outcome into an exit code.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        0 if every check passed, 1 if at least one check failed, 2 if the
        server could not be reached at all.
    """
    args = build_parser().parse_args(argv)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        if args.command == "health":
            passed = cmd_health(args, opener)
        elif args.command == "verify-embedding":
            passed = cmd_verify_embedding(args, opener, BaselineEngine())
        elif args.command == "verify-rerank":
            passed = cmd_verify_rerank(args, opener, BaselineEngine())
        elif args.command == "bench":
            passed = cmd_bench(args, opener)
        else:
            passed = cmd_all(args, opener)
    except ServerUnreachable as exc:
        print(str(exc))
        print("Start the eeANE server first: uv run python -m eeane.server")
        return 2

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
