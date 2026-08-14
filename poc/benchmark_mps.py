"""Measure the PyTorch/MPS (GPU) baseline for ruri-v3-310m and its reranker.

This is the primary comparison target for eeANE (Core ML/ANE): it measures
"PyTorch batch inference on MPS" -- the actual engine underneath the current
Infinity_emb deployment -- with no HTTP or server overhead (see
開発資料/v0.3実装計画.md §4.7).

``--model embedding`` encodes the same kokoro.txt chunk set used by
``poc/benchmark_throughput.py`` (T6) via ``sentence_transformers.
SentenceTransformer.encode``. ``--model reranker`` reranks the fixed 9-query
x 36-paragraph pair set one query at a time via ``sentence_transformers.
CrossEncoder.predict``, matching a single Open WebUI rerank request.

``sentence_transformers``' ``encode``/``predict`` perform length-sorted
dynamic padding internally; this is left untouched (it mirrors real GPU
serving behavior, including Infinity_emb's own optimization) so the
comparison is "total wall time for the same chunk/pair set", not a
padding-matched microbenchmark.

Usage:
    uv run python poc/benchmark_mps.py --model embedding --chunk-tokens 512 --batch 16
    uv run python poc/benchmark_mps.py --model reranker --batch 16
    uv run python poc/benchmark_mps.py --model embedding --chunk-tokens 512 --batch 16 --sustain 60
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import sentence_transformers
import torch
import transformers
from sentence_transformers import CrossEncoder, SentenceTransformer

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python poc/benchmark_mps.py` to import the poc package.
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

# Fixed Aozora Bunko source used for the embedding chunk set (same input as
# poc/benchmark_throughput.py's T6 embedding mode).
KOKORO_PATH: Path = CORPUS_DIR / "kokoro.txt"

# Reporting interval (seconds) for the --sustain progress log.
SUSTAIN_REPORT_INTERVAL_SEC = 5.0

# Number of leading rows/pairs used for the pre-timing warmup call.
WARMUP_CAP = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description="Benchmark the PyTorch/MPS baseline for ruri-v3-310m (embedding) "
        "or ruri-v3-reranker-310m (reranker)."
    )
    parser.add_argument(
        "--model",
        choices=["embedding", "reranker"],
        required=True,
        help="Which model to benchmark.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=512,
        help="[embedding] Chunk size C for chunk_by_tokens, in tokens.",
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="Batch size B passed to encode()/predict()."
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Number of timed measurement repeats."
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="[embedding] If > 0, keep only the first M chunks (smoke runs). 0 = all chunks.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="[reranker] Max sequence length S for the CrossEncoder.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="[reranker] If > 0, keep only the first Q queries (smoke runs). 0 = all 9.",
    )
    parser.add_argument(
        "--sustain",
        type=int,
        default=0,
        help="If > 0, run a sustained encode()/predict() loop for this many seconds "
        "(for manual `powermetrics` GPU power observation) instead of the timed benchmark.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    """Validate numeric CLI arguments, raising SystemExit on violations.

    Args:
        args: Parsed arguments from :func:`parse_args`.

    Raises:
        SystemExit: If a strictly-positive argument is <= 0, or if a
            zero-allowed argument (``--max-chunks``/``--max-queries``/
            ``--sustain``) is negative.
    """
    if args.chunk_tokens <= 0:
        raise SystemExit("--chunk-tokens must be a positive integer")
    if args.batch <= 0:
        raise SystemExit("--batch must be a positive integer")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be a positive integer")
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be a positive integer")
    if args.max_chunks < 0:
        raise SystemExit("--max-chunks must be >= 0")
    if args.max_queries < 0:
        raise SystemExit("--max-queries must be >= 0")
    if args.sustain < 0:
        raise SystemExit("--sustain must be >= 0")


def build_environment_info(model: str) -> dict[str, Any]:
    """Assemble environment/version metadata recorded alongside measurements.

    Args:
        model: One of the ``--model`` CLI choices (``"embedding"`` or
            ``"reranker"``), recorded so results JSONs are self-describing.

    Returns:
        Dict of version/platform metadata.
    """
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "mps_available": torch.backends.mps.is_available(),
        "device": "mps",
        "model": model,
    }


def result_path(
    model: str,
    sustain: bool,
    batch: int,
    chunk_tokens: int | None = None,
    seq_len: int | None = None,
) -> Path:
    """Build the results JSON output path.

    Args:
        model: One of the ``--model`` CLI choices.
        sustain: Whether this run used ``--sustain`` mode (prefixes the
            filename with ``mps_sustain_`` instead of ``mps_``).
        batch: Batch size B.
        chunk_tokens: Chunk size C (embedding mode only).
        seq_len: Max sequence length S (reranker mode only).

    Returns:
        Path to the output JSON file under ``poc/results/``.
    """
    results_dir = _REPO_ROOT / "poc" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = "mps_sustain" if sustain else "mps"
    if model == "embedding":
        stem = f"embedding_c{chunk_tokens}_b{batch}"
    else:
        stem = f"reranker_s{seq_len}_b{batch}"
    return results_dir / f"{prefix}_{stem}.json"


def _run_sustain_loop(step_fn: Callable[[], None], sustain_seconds: int) -> dict[str, Any]:
    """Run ``step_fn`` repeatedly for ``sustain_seconds``, reporting progress.

    Args:
        step_fn: Zero-argument callable performing one loop iteration (an
            ``encode()`` or ``predict()`` call).
        sustain_seconds: Duration to keep looping, in seconds.

    Returns:
        Dict with the requested duration, actual elapsed time, and loop count.
    """
    print("Run 'sudo powermetrics --samplers gpu_power -i 1000' in another terminal now.")
    loops = 0
    next_report_at = SUSTAIN_REPORT_INTERVAL_SEC
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= sustain_seconds:
            break
        step_fn()
        loops += 1
        elapsed = time.perf_counter() - start
        if elapsed >= next_report_at:
            print(f"  elapsed={elapsed:.1f}s loops={loops}")
            next_report_at += SUSTAIN_REPORT_INTERVAL_SEC
    total_elapsed = time.perf_counter() - start
    print(f"Sustain loop finished: elapsed={total_elapsed:.1f}s loops={loops}")
    return {"sustain_seconds": sustain_seconds, "elapsed_sec": total_elapsed, "loops": loops}


def run_embedding(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    """Run the embedding MPS benchmark (timed repeats or --sustain loop).

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

    # Same effective-token definition as the eeANE-side throughput bench:
    # attention_mask sum over a fixed-length (max_length=C) tokenization.
    effective_tokens = int(
        tokenize_batch(tokenizer, chunks, args.chunk_tokens)["attention_mask"].sum()
    )

    print(f"Loading SentenceTransformer from {DEFAULT_MODEL_DIR} (device=mps)")
    step = time.perf_counter()
    model = SentenceTransformer(str(DEFAULT_MODEL_DIR), device="mps")
    load_sec = time.perf_counter() - step

    default_max_seq_length = model.max_seq_length
    model.max_seq_length = args.chunk_tokens
    max_seq_length_info = {"configured": args.chunk_tokens, "default": default_max_seq_length}

    warmup_n = min(args.batch, WARMUP_CAP)
    model.encode(
        chunks[:warmup_n], batch_size=args.batch, convert_to_numpy=True, show_progress_bar=False
    )

    environment = build_environment_info("embedding")

    if args.sustain > 0:
        sustain = _run_sustain_loop(
            lambda: model.encode(
                chunks, batch_size=args.batch, convert_to_numpy=True, show_progress_bar=False
            ),
            args.sustain,
        )
        result: dict[str, Any] = {
            "mode": "sustain",
            "model": "embedding",
            "num_chunks": len(chunks),
            "chunk_tokens": args.chunk_tokens,
            "batch": args.batch,
            "load_sec": load_sec,
            "max_seq_length": max_seq_length_info,
            "sustain": sustain,
            "environment": environment,
        }
        out_path = result_path(
            "embedding", sustain=True, batch=args.batch, chunk_tokens=args.chunk_tokens
        )
        return result, out_path

    print(
        f"Running {args.repeats} timed encode() repeats over {len(chunks)} chunks "
        f"(batch={args.batch})"
    )
    per_repeat_sec: list[float] = []
    for _ in range(args.repeats):
        step = time.perf_counter()
        model.encode(chunks, batch_size=args.batch, convert_to_numpy=True, show_progress_bar=False)
        per_repeat_sec.append(time.perf_counter() - step)
    median_sec = statistics.median(per_repeat_sec)

    result = {
        "mode": "benchmark",
        "model": "embedding",
        "num_chunks": len(chunks),
        "chunk_tokens": args.chunk_tokens,
        "batch": args.batch,
        "repeats": args.repeats,
        "per_repeat_sec": per_repeat_sec,
        "median_sec": median_sec,
        "effective_tokens": effective_tokens,
        "effective_tokens_per_sec": effective_tokens / median_sec,
        "load_sec": load_sec,
        "max_seq_length": max_seq_length_info,
        "environment": environment,
    }
    out_path = result_path(
        "embedding", sustain=False, batch=args.batch, chunk_tokens=args.chunk_tokens
    )
    return result, out_path


def run_reranker(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    """Run the reranker MPS benchmark (timed repeats or --sustain loop).

    Each query is reranked against all corpus paragraphs with an
    independent ``CrossEncoder.predict()`` call, matching a single Open
    WebUI rerank request rather than one large batched call across all
    queries.

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

    # One pair list per query, in the [query, document] list-of-lists form
    # sentence_transformers' CrossEncoder.predict expects.
    query_pairs: list[list[list[str]]] = [
        [[query["query"], paragraph] for paragraph in paragraphs] for query in queries
    ]

    # Same effective-token definition as the eeANE-side throughput bench:
    # attention_mask sum over a fixed-length (max_length=S) pair tokenization
    # across every (query, paragraph) pair used in one full pass.
    all_pairs = [(query["query"], paragraph) for query in queries for paragraph in paragraphs]
    effective_tokens = int(
        tokenize_pairs(tokenizer, all_pairs, args.seq_len)["attention_mask"].sum()
    )

    print(
        f"Loading CrossEncoder from {DEFAULT_RERANKER_DIR} (device=mps, max_length={args.seq_len})"
    )
    step = time.perf_counter()
    model = CrossEncoder(str(DEFAULT_RERANKER_DIR), device="mps", max_length=args.seq_len)
    load_sec = time.perf_counter() - step

    warmup_n = min(args.batch, WARMUP_CAP)
    model.predict(query_pairs[0][:warmup_n], batch_size=args.batch, show_progress_bar=False)

    environment = build_environment_info("reranker")

    if args.sustain > 0:
        sustain = _run_sustain_loop(
            lambda: model.predict(query_pairs[0], batch_size=args.batch, show_progress_bar=False),
            args.sustain,
        )
        result: dict[str, Any] = {
            "mode": "sustain",
            "model": "reranker",
            "num_queries": len(queries),
            "num_pairs_per_query": len(paragraphs),
            "batch": args.batch,
            "load_sec": load_sec,
            "max_length": args.seq_len,
            "sustain": sustain,
            "environment": environment,
        }
        out_path = result_path("reranker", sustain=True, batch=args.batch, seq_len=args.seq_len)
        return result, out_path

    print(
        f"Running {args.repeats} timed repeats over {len(queries)} queries "
        f"(batch={args.batch}, {len(paragraphs)} pairs/query)"
    )
    per_repeat_total_sec: list[float] = []
    per_query_sec: list[float] = []
    for _ in range(args.repeats):
        per_query_sec = []
        repeat_start = time.perf_counter()
        for pairs in query_pairs:
            step = time.perf_counter()
            model.predict(pairs, batch_size=args.batch, show_progress_bar=False)
            per_query_sec.append(time.perf_counter() - step)
        per_repeat_total_sec.append(time.perf_counter() - repeat_start)

    median_total_sec = statistics.median(per_repeat_total_sec)
    per_query_median_sec = statistics.median(per_query_sec)

    result = {
        "mode": "benchmark",
        "model": "reranker",
        "num_queries": len(queries),
        "num_pairs_per_query": len(paragraphs),
        "batch": args.batch,
        "repeats": args.repeats,
        "per_query_sec": per_query_sec,
        "per_query_median_sec": per_query_median_sec,
        "per_repeat_total_sec": per_repeat_total_sec,
        "median_total_sec": median_total_sec,
        "effective_tokens": effective_tokens,
        "effective_tokens_per_sec": effective_tokens / median_total_sec,
        "load_sec": load_sec,
        "max_length": args.seq_len,
        "environment": environment,
    }
    out_path = result_path("reranker", sustain=False, batch=args.batch, seq_len=args.seq_len)
    return result, out_path


def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary of the benchmark result to stdout."""
    print(f"mode            : {result['mode']}")
    print(f"model           : {result['model']}")
    print(f"load_sec        : {result['load_sec']:.4f}")
    if result["mode"] == "sustain":
        sustain = result["sustain"]
        print(f"sustain_seconds : {sustain['sustain_seconds']}")
        print(f"elapsed_sec     : {sustain['elapsed_sec']:.2f}")
        print(f"loops           : {sustain['loops']}")
        print(f"peak_rss_bytes  : {result['peak_rss_bytes']}")
        return
    if result["model"] == "embedding":
        print(f"num_chunks      : {result['num_chunks']}")
        print(f"chunk_tokens    : {result['chunk_tokens']}")
        print(f"batch           : {result['batch']}")
        print(f"median_sec      : {result['median_sec']:.4f}")
        print(f"effective_tok/s : {result['effective_tokens_per_sec']:.2f}")
    else:
        print(f"num_queries     : {result['num_queries']}")
        print(f"num_pairs/query : {result['num_pairs_per_query']}")
        print(f"median_total_sec: {result['median_total_sec']:.4f}")
        print(f"per_query_median: {result['per_query_median_sec']:.4f}")
        print(f"effective_tok/s : {result['effective_tokens_per_sec']:.2f}")
    print(f"peak_rss_bytes  : {result['peak_rss_bytes']}")


def main(argv: list[str] | None = None) -> int:
    """Run the PyTorch/MPS baseline benchmark for the requested model.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (always 0; this script is not a pass/fail gate).

    Raises:
        SystemExit: If MPS is unavailable, or a CLI argument fails
            validation (see :func:`_validate_args`).
    """
    if not torch.backends.mps.is_available():
        raise SystemExit(
            "torch.backends.mps.is_available() is False; this benchmark requires an "
            "MPS-capable machine."
        )

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
