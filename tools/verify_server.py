"""Verification client for the eeANE server.

Standard library only (``urllib.request``), with the eeANE package and a
handful of read-only test-data loaders from ``tools/corpus.py`` imported
for building inputs and computing the Core ML direct-predict baseline.
Like ``poc/benchmark_infinity_client.py``, every request is sent through
an opener that explicitly bypasses the ``HTTP_PROXY``/``HTTPS_PROXY``
environment variables: a proxy that silently swallows localhost traffic
otherwise turns into a confusing "server is broken" report.

Subcommands:
    health           GET /health shape check.
    verify-embedding /v1/embeddings OpenAI compatibility and accuracy.
    verify-rerank    /rerank, /v1/rerank Infinity compatibility and accuracy.
    bench            HTTP round-trip latency measurement.
    all              Runs the four subcommands above in order.

Both verify subcommands accept ``--model <id>``. Without it they run their
full default-model suite, whose expected values are tied to the corpus and
the model this repository ships. With it they run a model-neutral suite
against the named model: HTTP responses are compared against direct Core
ML predictions from that same model, and nothing is assumed about the
model's width, buckets or retrieval quality.

Exit codes: 0 = every check passed, 1 = at least one check failed, 2 = the
server could not be reached at all (start it first with
``uv run python -m eeane.server``).

Usage:
    uv run python tools/verify_server.py health
    uv run python tools/verify_server.py verify-embedding
    uv run python tools/verify_server.py verify-embedding --model MODEL_ID
    uv run python tools/verify_server.py verify-rerank
    uv run python tools/verify_server.py verify-rerank --model MODEL_ID
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
    # Allow `python tools/verify_server.py` to import the eeane package
    # regardless of the current working directory.
    sys.path.insert(0, str(_REPO_ROOT))

from eeane import runtime  # noqa: E402
from eeane.config import ModelEntry, load_config  # noqa: E402
from eeane.engine import CoreMLEngine  # noqa: E402
from tools.corpus import (  # noqa: E402
    CORPUS_DIR,
    PREFIX_TEST_SENTENCES,
    PREFIXES,
    _split_paragraphs,
    load_corpus_paragraphs,
    load_rerank_queries,
)

DEFAULT_BASE_URL = "http://127.0.0.1:7997"
DEFAULT_TIMEOUT = 120.0

# Resolved once, with the same search order the server uses, so the
# baseline engine and the normalization check match the running server's
# configuration.
CONFIG = load_config().config

# (source work, corpus file, kokoro-only paragraph cap) in the exact file
# order load_corpus_paragraphs() uses, so per-paragraph work tags can be
# reconstructed alongside it (tools.corpus discards the work name).
_WORK_FILES: list[tuple[str, Path, int | None]] = [
    ("kumonoito", CORPUS_DIR / "kumonoito.txt", None),
    ("sangetsuki", CORPUS_DIR / "sangetsuki.txt", None),
    ("kokoro", CORPUS_DIR / "kokoro.txt", 30),
]

# Every check the ``--model`` (model-neutral) suites report, in output
# order. Whatever is not reached is filled in as a failed "skipped" row, so
# a run always answers the same set of questions.
_EMBEDDING_MODEL_CHECKS: tuple[str, ...] = (
    "http_200",
    "data_count",
    "index_order",
    "response_model_id",
    "embedding_dim_consistent",
    "cosine_min_0.999999",
    "base64_roundtrip",
)
_RERANK_MODEL_CHECKS: tuple[str, ...] = (
    "requests_200",
    "response_model_id",
    "index_coverage",
    "results_sorted",
    "score_match",
    "v1_rerank_matches_rerank",
)


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
            self._engine = CoreMLEngine.from_config(CONFIG)
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
    the same filter as ``tools.corpus.load_corpus_paragraphs`` (file order,
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
            "load_paragraph_works() diverged from tools.corpus.load_corpus_paragraphs(): "
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
            is never routed through an unrelated proxy.
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


def _fill_skipped(checks: list[CheckResult], names: tuple[str, ...], reason: str) -> None:
    """Append a failed placeholder row for every check that could not run.

    Checks already present in ``checks`` are left alone, so a caller can
    hand over the whole suite's name list at any point of the run without
    tracking how far it got.

    Args:
        checks: Accumulated checks (appended to in place).
        names: Every check name the suite reports, in output order.
        reason: Short explanation shown as ``"skipped: <reason>"``.
    """
    recorded = {check.name for check in checks}
    checks.extend(
        CheckResult(name, False, f"skipped: {reason}") for name in names if name not in recorded
    )


def _resolve_model_check(model_id: str, kind: str) -> tuple[ModelEntry | None, CheckResult]:
    """Look a ``--model`` id up in the configuration this tool loaded.

    The baseline engine can only score models it serves, so an id that is
    absent from the configuration (or names the other kind of model) is a
    check failure here rather than an HTTP call that is doomed anyway.

    Args:
        model_id: Id requested on the command line.
        kind: Kind the subcommand verifies (``"embedding"`` or
            ``"reranker"``).

    Returns:
        A ``(entry, check)`` pair. ``entry`` is ``None`` exactly when the
        check failed; the check's detail then names the usable ids.
    """
    entry = CONFIG.model_by_id(model_id)
    if entry is None:
        available = ", ".join(f"'{candidate.id}'" for candidate in CONFIG.models_of_kind(kind))
        return None, CheckResult(
            "model_configured",
            False,
            f"'{model_id}' is not in the configuration this tool loaded; "
            f"available {kind} models: {available or '(none)'}",
        )
    if entry.kind != kind:
        return None, CheckResult(
            "model_configured",
            False,
            f"'{model_id}' is a {entry.kind} model, but this subcommand verifies {kind} models",
        )
    return entry, CheckResult("model_configured", True, f"id='{model_id}' kind={kind}")


def _embedding_vectors(data: list[Any]) -> list[list[float]] | None:
    """Extract the float vectors of an ``/v1/embeddings`` response.

    Args:
        data: The response's ``data`` list.

    Returns:
        One vector per entry, in response order, or ``None`` if any entry
        is not a dict carrying a non-empty numeric ``embedding`` list
        (booleans are numbers in Python, so they are rejected explicitly).
    """
    vectors: list[list[float]] = []
    for item in data:
        if not isinstance(item, dict):
            return None
        vector = item.get("embedding")
        if not isinstance(vector, list) or not vector:
            return None
        if any(not isinstance(value, int | float) or isinstance(value, bool) for value in vector):
            return None
        vectors.append([float(value) for value in vector])
    return vectors


def _parse_results(results: list[Any], count: int) -> tuple[list[int], list[float]] | None:
    """Extract the ``(index, relevance_score)`` pairs of a rerank response.

    Args:
        results: The response's ``results`` list.
        count: Number of documents the request sent.

    Returns:
        Indices and scores in response order, or ``None`` when an entry is
        malformed or the indices are not exactly ``0..count-1`` with no
        repeats -- in which case comparing scores by index would be
        meaningless (and could even read out of range).
    """
    indices: list[int] = []
    scores: list[float] = []
    for result in results:
        if not isinstance(result, dict):
            return None
        index = result.get("index")
        score = result.get("relevance_score")
        if not isinstance(index, int) or isinstance(index, bool):
            return None
        if not isinstance(score, int | float) or isinstance(score, bool):
            return None
        indices.append(index)
        scores.append(float(score))
    if sorted(indices) != list(range(count)):
        return None
    return indices, scores


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


def _health_entry_problem(entry: Any) -> str | None:
    """Describe what is wrong with one ``/health`` model entry.

    Args:
        entry: One element of the response's ``models`` list.

    Returns:
        A short description of the first problem found, or ``None`` if the
        entry carries a non-empty ``id``, a supported ``kind`` and a list
        of integer ``buckets``.
    """
    if not isinstance(entry, dict):
        return f"not an object: {entry!r}"
    model_id = entry.get("id")
    if not isinstance(model_id, str) or not model_id:
        return f"bad id: {model_id!r}"
    kind = entry.get("kind")
    if kind not in ("embedding", "reranker"):
        return f"{model_id}: bad kind {kind!r}"
    buckets = entry.get("buckets")
    if not isinstance(buckets, list) or any(
        not isinstance(bucket, int) or isinstance(bucket, bool) for bucket in buckets
    ):
        return f"{model_id}: bad buckets {buckets!r}"
    return None


def cmd_health(args: argparse.Namespace, opener: urllib.request.OpenerDirector) -> bool:
    """Verify ``GET /health`` responds with the expected shape.

    The response lists one entry per served model, so the checks are
    driven by that list rather than by any particular model being present.

    Args:
        args: Parsed CLI arguments (uses ``base_url``, ``timeout``).
        opener: Proxy-bypassing opener shared across subcommands.

    Returns:
        Whether every check passed.
    """
    status, body, elapsed = _request_json(
        opener, "GET", f"{args.base_url}/health", None, args.timeout
    )
    check_names = ("status_ok", "version_present", "models_list", "models_entries_wellformed")
    checks: list[CheckResult] = [CheckResult("http_200", status == 200, f"elapsed={elapsed:.3f}s")]
    if status != 200:
        checks[-1].detail = _describe_error(status, body)
        _fill_skipped(checks, check_names, "non-200 response")
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

    models = payload.get("models")
    checks.append(
        CheckResult(
            "models_list",
            isinstance(models, list),
            f"{len(models)} entries" if isinstance(models, list) else f"models={models!r}",
        )
    )
    if not isinstance(models, list):
        _fill_skipped(checks, check_names, "models is not a list")
        return _print_summary("health", checks)

    problems = [
        problem for entry in models if (problem := _health_entry_problem(entry)) is not None
    ]
    if problems:
        detail = "; ".join(problems)
    else:
        # Safe to index only once every entry is known to be well-formed.
        detail = ", ".join(f"{entry['id']}({entry['kind']})" for entry in models)
    checks.append(CheckResult("models_entries_wellformed", not problems, detail))
    return _print_summary("health", checks)


def cmd_verify_embedding(
    args: argparse.Namespace, opener: urllib.request.OpenerDirector, baseline: BaselineEngine
) -> bool:
    """Verify ``/v1/embeddings`` OpenAI compatibility and accuracy.

    Args:
        args: Parsed CLI arguments (uses ``base_url``, ``timeout``, and
            ``model`` when the subcommand defines it).
        opener: Proxy-bypassing opener shared across subcommands.
        baseline: Shared, lazily-built Core ML baseline engine.

    Returns:
        Whether every check passed.
    """
    model_id = getattr(args, "model", None)
    if model_id is not None:
        return _verify_embedding_model(args, opener, baseline, model_id)
    return _verify_embedding_default(args, opener, baseline)


def _verify_embedding_default(
    args: argparse.Namespace, opener: urllib.request.OpenerDirector, baseline: BaselineEngine
) -> bool:
    """Run the full ``/v1/embeddings`` suite against the server's default model.

    Sends no ``model`` field, so the server answers with the first-listed
    embedding model. The expected values (input count, embedding width,
    bucket routing) describe the model and corpus this repository ships;
    use ``--model`` for a model-neutral run.

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
    if CONFIG.embedding_model.normalize:
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


def _verify_embedding_model(
    args: argparse.Namespace,
    opener: urllib.request.OpenerDirector,
    baseline: BaselineEngine,
    model_id: str,
) -> bool:
    """Verify ``/v1/embeddings`` against one named model, model-neutrally.

    The corpus paragraphs are embedded twice -- over HTTP with ``model``
    set to ``model_id``, and by direct Core ML prediction with the same
    model -- and only the agreement between the two, plus the response's
    own self-consistency, is judged. No embedding width, bucket layout or
    retrieval quality is assumed, so the suite applies to any configured
    embedding model.

    Args:
        args: Parsed CLI arguments (uses ``base_url``, ``timeout``).
        opener: Proxy-bypassing opener shared across subcommands.
        baseline: Shared, lazily-built Core ML baseline engine.
        model_id: Id sent as the request's ``model`` field.

    Returns:
        Whether every check passed.
    """
    entry, model_check = _resolve_model_check(model_id, "embedding")
    checks: list[CheckResult] = [model_check]
    if entry is None:
        _fill_skipped(checks, _EMBEDDING_MODEL_CHECKS, "unusable --model")
        return _print_summary("verify-embedding", checks)

    texts = load_corpus_paragraphs()
    if not texts:
        _fill_skipped(checks, _EMBEDDING_MODEL_CHECKS, "the corpus loader returned no paragraph")
        return _print_summary("verify-embedding", checks)

    url = f"{args.base_url}/v1/embeddings"
    status, body, elapsed = _request_json(
        opener, "POST", url, {"input": texts, "model": model_id}, args.timeout
    )
    if status != 200:
        checks.append(CheckResult("http_200", False, _describe_error(status, body)))
        _fill_skipped(checks, _EMBEDDING_MODEL_CHECKS, "non-200 response")
        return _print_summary("verify-embedding", checks)
    checks.append(CheckResult("http_200", True, f"elapsed={elapsed:.3f}s"))

    payload = body if isinstance(body, dict) else {}
    data = payload.get("data")
    count_ok = isinstance(data, list) and len(data) == len(texts)
    checks.append(
        CheckResult(
            "data_count",
            count_ok,
            f"got {len(data) if isinstance(data, list) else 'n/a'} of {len(texts)}",
        )
    )
    if not count_ok:
        _fill_skipped(checks, _EMBEDDING_MODEL_CHECKS, "bad data shape")
        return _print_summary("verify-embedding", checks)

    order = [item.get("index") if isinstance(item, dict) else None for item in data]
    checks.append(CheckResult("index_order", order == list(range(len(texts)))))

    served_model = payload.get("model")
    checks.append(
        CheckResult(
            "response_model_id",
            served_model == model_id,
            f"response model={served_model!r} requested={model_id!r}",
        )
    )

    # A single width is what makes the vectors comparable at all; the
    # configured width, when the cache recorded one, must agree with it.
    vectors = _embedding_vectors(data)
    dims = sorted({len(vector) for vector in vectors}) if vectors is not None else []
    comparable = len(dims) == 1
    checks.append(
        CheckResult(
            "embedding_dim_consistent",
            comparable and (entry.embedding_dim is None or dims[0] == entry.embedding_dim),
            f"dims={dims or 'unusable embedding values'} configured={entry.embedding_dim}",
        )
    )
    if vectors is None or not comparable:
        _fill_skipped(checks, _EMBEDDING_MODEL_CHECKS, "embeddings are not a uniform float matrix")
        return _print_summary("verify-embedding", checks)

    http_vectors = np.asarray(vectors, dtype=np.float64)
    engine = baseline.get()
    baseline_vectors = engine.embed(texts, model_id=model_id).vectors.astype(np.float64)
    if entry.normalize:
        baseline_vectors = runtime.l2_normalize(baseline_vectors)
    if baseline_vectors.shape != http_vectors.shape:
        checks.append(
            CheckResult(
                "cosine_min_0.999999",
                False,
                f"shape mismatch: http={http_vectors.shape} baseline={baseline_vectors.shape}",
            )
        )
    else:
        cosine = _cosine_rowwise(http_vectors, baseline_vectors)
        cosine_min = float(cosine.min())
        checks.append(
            CheckResult(
                "cosine_min_0.999999",
                cosine_min >= 0.999999,
                f"min={cosine_min:.8f} mean={float(cosine.mean()):.8f} "
                f"worst_index={int(cosine.argmin())}",
            )
        )

    b64_status, b64_body, _elapsed = _request_json(
        opener,
        "POST",
        url,
        {"input": [texts[0]], "encoding_format": "base64", "model": model_id},
        args.timeout,
    )
    if b64_status == 200 and isinstance(b64_body, dict):
        b64_data = b64_body.get("data") or [{}]
        first = b64_data[0] if isinstance(b64_data[0], dict) else {}
        encoded = first.get("embedding")
        if isinstance(encoded, str):
            decoded = runtime.base64_to_floats(encoded).astype(np.float64)
            shape_ok = decoded.shape == (dims[0],)
            b64_ok = shape_ok and np.allclose(decoded, http_vectors[0], atol=1e-6)
            if b64_ok:
                detail = ""
            elif not shape_ok:
                detail = f"decoded shape={decoded.shape}, expected ({dims[0]},)"
            else:
                detail = f"max|delta|={np.max(np.abs(decoded - http_vectors[0])):.3e}"
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
    """Verify ``/rerank``/``/v1/rerank`` Infinity compatibility and accuracy.

    Args:
        args: Parsed CLI arguments (uses ``base_url``, ``timeout``, and
            ``model`` when the subcommand defines it).
        opener: Proxy-bypassing opener shared across subcommands.
        baseline: Shared, lazily-built Core ML baseline engine.

    Returns:
        Whether every check passed.
    """
    model_id = getattr(args, "model", None)
    if model_id is not None:
        return _verify_rerank_model(args, opener, baseline, model_id)
    return _verify_rerank_default(args, opener, baseline)


def _verify_rerank_default(
    args: argparse.Namespace, opener: urllib.request.OpenerDirector, baseline: BaselineEngine
) -> bool:
    """Run the full rerank suite against the server's default reranker.

    Sends no ``model`` field, so the server answers with the first-listed
    reranker. Besides the protocol and accuracy checks, this suite also
    asserts the retrieval quality expected of the model and corpus this
    repository ships (the top-1 hit must come from the query's source
    work); use ``--model`` for a model-neutral run.

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


def _verify_rerank_model(
    args: argparse.Namespace,
    opener: urllib.request.OpenerDirector,
    baseline: BaselineEngine,
    model_id: str,
) -> bool:
    """Verify ``/rerank``/``/v1/rerank`` against one named reranker, model-neutrally.

    Every stored query is scored against the corpus paragraphs twice --
    over HTTP with ``model`` set to ``model_id``, and by direct Core ML
    prediction with the same model -- and only the agreement between the
    two, plus the response's own consistency (sorting, index coverage,
    reported model id, both URL paths), is judged. Which document a model
    ranks first is its own business and is not checked, so the suite
    applies to any configured reranker.

    Args:
        args: Parsed CLI arguments (uses ``base_url``, ``timeout``).
        opener: Proxy-bypassing opener shared across subcommands.
        baseline: Shared, lazily-built Core ML baseline engine.
        model_id: Id sent as the request's ``model`` field.

    Returns:
        Whether every check passed.
    """
    entry, model_check = _resolve_model_check(model_id, "reranker")
    checks: list[CheckResult] = [model_check]
    if entry is None:
        _fill_skipped(checks, _RERANK_MODEL_CHECKS, "unusable --model")
        return _print_summary("verify-rerank", checks)

    queries = load_rerank_queries()
    paragraphs = load_corpus_paragraphs()
    if not queries or not paragraphs:
        _fill_skipped(checks, _RERANK_MODEL_CHECKS, "the test-data loaders returned nothing")
        return _print_summary("verify-rerank", checks)
    engine = baseline.get()

    request_issues: list[str] = []
    model_issues: list[str] = []
    index_issues: list[str] = []
    sort_issues: list[str] = []
    max_score_delta = 0.0
    worst_detail = "n/a"

    for query in queries:
        status, body, _elapsed = _request_json(
            opener,
            "POST",
            f"{args.base_url}/rerank",
            {"query": query["query"], "documents": paragraphs, "model": model_id},
            args.timeout,
        )
        if status != 200:
            request_issues.append(f"{query['id']}: {_describe_error(status, body)}")
            continue

        served_model = body.get("model") if isinstance(body, dict) else None
        if served_model != model_id:
            model_issues.append(f"{query['id']}: model={served_model!r}")

        results = _results_list(body)
        parsed = _parse_results(results, len(paragraphs))
        if parsed is None:
            # Without a clean 0..N-1 index set, scores cannot be matched
            # up with the baseline at all, so this query is not scored.
            index_issues.append(
                f"{query['id']}: results do not cover the {len(paragraphs)} sent documents "
                f"exactly once (got {len(results)} entries)"
            )
            continue
        indices, scores = parsed

        if scores != sorted(scores, reverse=True):
            sort_issues.append(query["id"])

        rerank_batch = engine.rerank(query["query"], paragraphs, model_id=model_id)
        baseline_scores = runtime.sigmoid(rerank_batch.logits)
        for index, score in zip(indices, scores, strict=True):
            delta = abs(score - float(baseline_scores[index]))
            if delta > max_score_delta:
                max_score_delta = delta
                worst_detail = f"query={query['id']} index={index}"

    scored = len(queries) - len(request_issues) - len(index_issues)
    blocked = bool(request_issues or index_issues)
    checks.append(
        CheckResult(
            "requests_200",
            not request_issues,
            "; ".join(request_issues) or f"{len(queries)}/{len(queries)} ok",
        )
    )
    checks.append(
        CheckResult(
            "response_model_id",
            not request_issues and not model_issues,
            "; ".join(model_issues) or f"all responses report '{model_id}'",
        )
    )
    checks.append(
        CheckResult("index_coverage", not blocked, "; ".join(index_issues) or f"{scored} queries")
    )
    checks.append(
        CheckResult("results_sorted", not blocked and not sort_issues, "; ".join(sort_issues))
    )
    checks.append(
        CheckResult(
            "score_match",
            not blocked and max_score_delta <= 1e-6,
            f"max|delta|={max_score_delta:.3e} ({worst_detail})",
        )
    )

    # Both URL paths must answer identically: /v1/rerank is the OpenAI-style
    # spelling of the same endpoint, not a separate implementation.
    sample_query = queries[0]["query"]
    payload = {"query": sample_query, "documents": paragraphs, "model": model_id}
    status_v1, body_v1, _elapsed = _request_json(
        opener, "POST", f"{args.base_url}/v1/rerank", payload, args.timeout
    )
    status_legacy, body_legacy, _elapsed = _request_json(
        opener, "POST", f"{args.base_url}/rerank", payload, args.timeout
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
    """Measure HTTP round-trip latency for rerank and embedding.

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

    # Only the two verify subcommands can target a specific model: bench
    # measures the default path, and `all` runs the default suites.
    model_option = argparse.ArgumentParser(add_help=False)
    model_option.add_argument(
        "--model",
        default=None,
        metavar="ID",
        help=(
            "Verify this configured model id instead of the server's default model. "
            "Runs a model-neutral suite: HTTP responses are compared against the same "
            "model's direct Core ML predictions, with no model-specific expectation."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("health", parents=[common], help="Check GET /health.")
    subparsers.add_parser(
        "verify-embedding",
        parents=[common, model_option],
        help="Verify /v1/embeddings compatibility and accuracy.",
    )
    subparsers.add_parser(
        "verify-rerank",
        parents=[common, model_option],
        help="Verify /rerank and /v1/rerank compatibility and accuracy.",
    )
    subparsers.add_parser("bench", parents=[common], help="Measure HTTP round-trip latency.")
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
