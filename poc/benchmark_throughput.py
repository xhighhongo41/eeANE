"""Measure "one real job" throughput for the eeANE PoC (v0.3実装計画.md §4.6).

Complements ``poc/benchmark_latency.py``'s round-robin single-predict
latency benchmark with a document-scale measurement: embedding a whole
document (chunked with :func:`poc.chunking.chunk_by_tokens`) or reranking a
whole query against the full paragraph pool. Both report ``effective
tokens/s`` (real work, padding excluded) alongside ``padded tokens/s`` (raw
ANE-fed throughput), per the definitions in v0.3実装計画.md §4.1.

Usage:
    uv run python poc/benchmark_throughput.py --model embedding \\
        --chunk-tokens 128 --batch 1
    uv run python poc/benchmark_throughput.py --model reranker --batch 4
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
from transformers import PreTrainedTokenizerBase

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python poc/benchmark_throughput.py` to import the poc package.
    sys.path.insert(0, str(_REPO_ROOT))

from poc.benchmark_latency import build_environment_info, default_mlmodelc_path  # noqa: E402
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

# Compute units are fixed for this benchmark (§4.6): document-scale
# throughput is only meaningful for the deployment target, CPU_AND_NE.
_COMPUTE_UNITS_LABEL = "CPU_AND_NE"

# Predict calls run (and discarded) on the first batch before timing begins,
# fixed at 2 for this benchmark (§4.6, distinct from benchmark_latency.py's
# 5-call warmup for the round-robin benchmark).
NUM_WARMUP_PREDICTS = 2

# Filler text/pair for the rows that pad the last batch of a document when
# its row count is not a multiple of B. Mirrors convert_embedding.py's
# BATCH_PADDING_TEXT: the empty string still yields a non-empty attention
# mask (special tokens only), avoiding the all-zero-mask NaN risk
# (v0.3実装計画.md §4.2).
_DUMMY_TEXT = ""
_DUMMY_PAIR = ("", "")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description="Measure real-document throughput (embedding chunking / reranker "
        "batch scoring) for the eeANE PoC (v0.3実装計画.md §4.6)."
    )
    parser.add_argument(
        "--model",
        choices=["embedding", "reranker"],
        required=True,
        help="Which PoC model to benchmark.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=None,
        help="Chunk token budget C, passed to chunk_by_tokens(). Required for "
        "--model embedding; not accepted for --model reranker.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        required=True,
        help="Batch size B, matching a converted s{S}_b{B}_eager_macos13.mlmodelc.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Fixed sequence length S. Defaults to --chunk-tokens for --model "
        "embedding, or 512 for --model reranker.",
    )
    return parser.parse_args(argv)


def _ceil_div(a: int, b: int) -> int:
    """Return ``ceil(a / b)`` for positive integers."""
    return -(-a // b)


def _pad_to_batch_multiple(
    tokens: dict[str, np.ndarray], dummy_row: dict[str, np.ndarray], batch: int
) -> dict[str, np.ndarray]:
    """Pad a tokenized row block up to a multiple of ``batch`` rows.

    Appends copies of ``dummy_row`` so that fixed-batch predict() calls can
    consume the block in full B-row chunks; the caller must exclude the
    appended rows' outputs and token counts from any aggregate statistics.

    Args:
        tokens: Tokenized rows, each value of shape (N, S).
        dummy_row: Tokenized filler row, each value of shape (1, S)
            (e.g. from ``tokenize_batch(tokenizer, [""], S)``).
        batch: Fixed batch size B.

    Returns:
        Dict with the same keys, each of shape (ceil(N/B)*B, S).
    """
    n = tokens["input_ids"].shape[0]
    remainder = n % batch
    if remainder == 0:
        return tokens
    missing = batch - remainder
    return {
        key: np.concatenate([tokens[key], np.repeat(dummy_row[key], missing, axis=0)], axis=0)
        for key in ("input_ids", "attention_mask")
    }


def _slice_batch(padded: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    """Slice one B-row predict() input out of a padded row block."""
    return {key: padded[key][start:end] for key in ("input_ids", "attention_mask")}


def _run_timed_batches(
    model: Any, padded: dict[str, np.ndarray], batch: int, num_predicts: int
) -> tuple[float, list[float]]:
    """Warm up on the first batch, then time one full predict() pass.

    Args:
        model: Loaded ``CompiledMLModel``.
        padded: Row block already padded to a multiple of ``batch``
            (:func:`_pad_to_batch_multiple`).
        batch: Fixed batch size B.
        num_predicts: Number of B-row batches to run (``padded`` row count
            divided by ``batch``).

    Returns:
        Tuple of (total wall time across all timed predicts, per-batch
        predict() durations).
    """
    first_batch = _slice_batch(padded, 0, batch)
    for _ in range(NUM_WARMUP_PREDICTS):
        model.predict(first_batch)

    per_batch_sec: list[float] = []
    pass_start = time.perf_counter()
    for i in range(num_predicts):
        start = i * batch
        end = start + batch
        step = time.perf_counter()
        model.predict(_slice_batch(padded, start, end))
        per_batch_sec.append(time.perf_counter() - step)
    inference_sec = time.perf_counter() - pass_start
    return inference_sec, per_batch_sec


def prepare_embedding_document(
    tokenizer: PreTrainedTokenizerBase, chunk_tokens: int, seq_len: int
) -> tuple[list[str], dict[str, np.ndarray], float]:
    """Load kokoro.txt, chunk it, and tokenize the chunks at a fixed length.

    Args:
        tokenizer: Tokenizer returned by :func:`poc.common.load_tokenizer`.
        chunk_tokens: Chunk token budget C, passed to
            :func:`poc.chunking.chunk_by_tokens`.
        seq_len: Fixed sequence length S used for tokenization.

    Returns:
        Tuple of (chunks, tokenized rows, combined chunk+tokenize wall time
        in seconds).
    """
    step = time.perf_counter()
    text = (CORPUS_DIR / "kokoro.txt").read_text(encoding="utf-8")
    chunks = chunk_by_tokens(tokenizer, text, chunk_tokens, stride=0)
    tokens = tokenize_batch(tokenizer, chunks, seq_len)
    prep_sec = time.perf_counter() - step
    return chunks, tokens, prep_sec


def run_embedding_throughput(args: argparse.Namespace, seq_len: int) -> dict[str, Any]:
    """Run the embedding-mode document throughput benchmark.

    Args:
        args: Parsed CLI arguments (``--chunk-tokens``/``--batch`` used).
        seq_len: Resolved fixed sequence length S.

    Returns:
        Result dict, ready to be serialized to JSON (§4.6).
    """
    model_dir = DEFAULT_MODEL_DIR
    mlmodelc_path = default_mlmodelc_path(model_dir, seq_len, args.batch)
    if not mlmodelc_path.exists():
        raise SystemExit(f"compiled model not found: {mlmodelc_path}")

    tokenizer = load_tokenizer(model_dir)
    print(f"Chunking testdata/corpus/kokoro.txt (chunk_tokens={args.chunk_tokens}, S={seq_len})")
    chunks, tokens, tokenize_prep_sec = prepare_embedding_document(
        tokenizer, args.chunk_tokens, seq_len
    )
    num_chunks = len(chunks)
    effective_tokens = int(tokens["attention_mask"].sum())
    dummy_row = tokenize_batch(tokenizer, [_DUMMY_TEXT], seq_len)
    padded = _pad_to_batch_multiple(tokens, dummy_row, args.batch)
    num_predicts = padded["input_ids"].shape[0] // args.batch

    print(f"Cold-loading {mlmodelc_path} (compute_units={_COMPUTE_UNITS_LABEL})")
    step = time.perf_counter()
    model = ct.models.CompiledMLModel(str(mlmodelc_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    load_sec = time.perf_counter() - step

    print(f"Running {NUM_WARMUP_PREDICTS} warmup + {num_predicts} timed predicts")
    inference_sec, per_batch_sec = _run_timed_batches(model, padded, args.batch, num_predicts)

    padding_ratio = 1.0 - effective_tokens / (seq_len * num_chunks)
    result: dict[str, Any] = {
        "mode": "embedding",
        "chunk_tokens": args.chunk_tokens,
        "seq_len": seq_len,
        "batch": args.batch,
        "num_chunks": num_chunks,
        "num_predicts": num_predicts,
        "effective_tokens": effective_tokens,
        "padding_ratio": padding_ratio,
        "tokenize_prep_sec": tokenize_prep_sec,
        "load_sec": load_sec,
        "inference_sec": inference_sec,
        "total_wall_sec": tokenize_prep_sec + load_sec + inference_sec,
        "effective_tokens_per_sec": effective_tokens / inference_sec,
        "padded_tokens_per_sec": seq_len * args.batch * num_predicts / inference_sec,
        "per_batch_sec": per_batch_sec,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": build_environment_info(mlmodelc_path, _COMPUTE_UNITS_LABEL, "embedding"),
    }
    return result


def prepare_reranker_document(
    tokenizer: PreTrainedTokenizerBase, seq_len: int
) -> tuple[list[dict], list[str], list[dict[str, np.ndarray]], float]:
    """Tokenize every query's (query, paragraph) block at a fixed length.

    Args:
        tokenizer: Tokenizer returned by :func:`poc.common.load_tokenizer`.
        seq_len: Fixed sequence length S used for tokenization.

    Returns:
        Tuple of (queries, paragraphs, per-query tokenized row blocks (one
        per query, each shape (len(paragraphs), S)), combined tokenize wall
        time in seconds).
    """
    step = time.perf_counter()
    queries = load_rerank_queries()
    paragraphs = load_corpus_paragraphs()
    per_query_tokens = [
        tokenize_pairs(
            tokenizer, [(query["query"], paragraph) for paragraph in paragraphs], seq_len
        )
        for query in queries
    ]
    prep_sec = time.perf_counter() - step
    return queries, paragraphs, per_query_tokens, prep_sec


def run_reranker_throughput(args: argparse.Namespace, seq_len: int) -> dict[str, Any]:
    """Run the reranker-mode per-query throughput benchmark.

    Args:
        args: Parsed CLI arguments (``--batch`` used).
        seq_len: Resolved fixed sequence length S.

    Returns:
        Result dict, ready to be serialized to JSON (§4.6).
    """
    model_dir = DEFAULT_RERANKER_DIR
    mlmodelc_path = default_mlmodelc_path(model_dir, seq_len, args.batch)
    if not mlmodelc_path.exists():
        raise SystemExit(f"compiled model not found: {mlmodelc_path}")

    tokenizer = load_tokenizer(model_dir)
    print(f"Tokenizing rerank_queries x corpus_paragraphs (S={seq_len})")
    queries, paragraphs, per_query_tokens, tokenize_prep_sec = prepare_reranker_document(
        tokenizer, seq_len
    )
    num_queries = len(queries)
    num_pairs_per_query = len(paragraphs)
    dummy_row = tokenize_pairs(tokenizer, [_DUMMY_PAIR], seq_len)
    per_query_padded = [
        _pad_to_batch_multiple(tokens, dummy_row, args.batch) for tokens in per_query_tokens
    ]

    print(f"Cold-loading {mlmodelc_path} (compute_units={_COMPUTE_UNITS_LABEL})")
    step = time.perf_counter()
    model = ct.models.CompiledMLModel(str(mlmodelc_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    load_sec = time.perf_counter() - step

    print(f"Running reranking for {num_queries} queries x {num_pairs_per_query} paragraphs")
    per_query_sec: list[float] = []
    for qi, padded in enumerate(per_query_padded):
        num_predicts = padded["input_ids"].shape[0] // args.batch
        if qi == 0:
            # Warmup happens once, on the first query's first batch, before
            # any query's timing starts (§4.6).
            inference_sec, _ = _run_timed_batches(model, padded, args.batch, num_predicts)
        else:
            pass_start = time.perf_counter()
            for i in range(num_predicts):
                start = i * args.batch
                end = start + args.batch
                model.predict(_slice_batch(padded, start, end))
            inference_sec = time.perf_counter() - pass_start
        per_query_sec.append(inference_sec)

    effective_tokens = int(sum(tokens["attention_mask"].sum() for tokens in per_query_tokens))
    num_pairs_total = num_queries * num_pairs_per_query
    num_predicts_per_query = _ceil_div(num_pairs_per_query, args.batch)
    total_inference_sec = sum(per_query_sec)
    padding_ratio = 1.0 - effective_tokens / (seq_len * num_pairs_total)

    result: dict[str, Any] = {
        "mode": "reranker",
        "seq_len": seq_len,
        "batch": args.batch,
        "num_queries": num_queries,
        "num_pairs_per_query": num_pairs_per_query,
        "num_pairs_total": num_pairs_total,
        "num_predicts_per_query": num_predicts_per_query,
        "per_query_sec": per_query_sec,
        "per_query_median_sec": statistics.median(per_query_sec),
        "total_inference_sec": total_inference_sec,
        "effective_tokens": effective_tokens,
        "padding_ratio": padding_ratio,
        "effective_tokens_per_sec": effective_tokens / total_inference_sec,
        "padded_tokens_per_sec": (
            seq_len * args.batch * num_predicts_per_query * num_queries / total_inference_sec
        ),
        "load_sec": load_sec,
        "tokenize_prep_sec": tokenize_prep_sec,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": build_environment_info(mlmodelc_path, _COMPUTE_UNITS_LABEL, "reranker"),
    }
    return result


def result_path(result: dict[str, Any]) -> Path:
    """Build the results JSON output path per §4.6's naming convention."""
    results_dir = _REPO_ROOT / "poc" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if result["mode"] == "embedding":
        c, s, b = result["chunk_tokens"], result["seq_len"], result["batch"]
        name = f"throughput_embedding_c{c}_s{s}_b{b}.json"
    else:
        s, b = result["seq_len"], result["batch"]
        name = f"throughput_reranker_s{s}_b{b}.json"
    return results_dir / name


def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary of the benchmark result to stdout."""
    print(f"mode                  : {result['mode']}")
    print(f"seq_len               : {result['seq_len']}")
    print(f"batch                 : {result['batch']}")
    print(f"mlmodelc              : {result['environment']['mlmodelc_path']}")
    print(f"tokenize_prep_sec     : {result['tokenize_prep_sec']:.4f}")
    print(f"load_sec              : {result['load_sec']:.4f}")
    print(f"padding_ratio         : {result['padding_ratio']:.4f}")
    print(f"effective_tokens      : {result['effective_tokens']}")
    print(f"effective_tokens/s    : {result['effective_tokens_per_sec']:.2f}")
    print(f"padded_tokens/s       : {result['padded_tokens_per_sec']:.2f}")
    print(f"peak_rss_bytes        : {result['peak_rss_bytes']}")
    if result["mode"] == "embedding":
        print(f"chunk_tokens          : {result['chunk_tokens']}")
        print(f"num_chunks            : {result['num_chunks']}")
        print(f"num_predicts          : {result['num_predicts']}")
        print(f"inference_sec         : {result['inference_sec']:.4f}")
        print(f"total_wall_sec        : {result['total_wall_sec']:.4f}")
    else:
        print(f"num_queries           : {result['num_queries']}")
        print(f"num_pairs_per_query   : {result['num_pairs_per_query']}")
        print(f"total_inference_sec   : {result['total_inference_sec']:.4f}")
        print(f"per_query_median_sec  : {result['per_query_median_sec']:.4f}")
        print(f"per_query_sec         : {[round(s, 4) for s in result['per_query_sec']]}")


def main(argv: list[str] | None = None) -> int:
    """Run the real-document throughput benchmark for a single configuration.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (always 0; this script is not a pass/fail gate).
    """
    args = parse_args(argv)
    if args.batch <= 0:
        raise SystemExit("--batch must be a positive integer")

    if args.model == "embedding":
        if args.chunk_tokens is None:
            raise SystemExit("--chunk-tokens is required for --model embedding")
        if args.chunk_tokens <= 0:
            raise SystemExit("--chunk-tokens must be a positive integer")
        seq_len = args.seq_len if args.seq_len is not None else args.chunk_tokens
    else:
        if args.chunk_tokens is not None:
            raise SystemExit("--chunk-tokens is not supported for --model reranker")
        seq_len = args.seq_len if args.seq_len is not None else 512
    if seq_len <= 0:
        raise SystemExit("--seq-len must be a positive integer")

    if args.model == "embedding":
        result = run_embedding_throughput(args, seq_len)
    else:
        result = run_reranker_throughput(args, seq_len)

    out_path = result_path(result)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print_summary(result)
    print(f"results               : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
