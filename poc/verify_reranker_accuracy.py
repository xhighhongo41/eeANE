"""Verify Core ML reranker accuracy against the PyTorch FP32 baseline.

Implements 開発資料/v0.2実装計画.md §4.6 (accuracy verification design) and
the "T5: 精度検証スクリプト" task. Loads a pre-compiled ``.mlmodelc``
(produced by ``poc/convert_reranker.py``), scores a deterministic
(query, paragraph) grid through both the Core ML model and the PyTorch
FP32 baseline (:func:`poc.common.score_pytorch`), and reports per-query
Spearman rank correlation, sigmoid score error, top-1/top-5 agreement,
NaN/Inf detection, a source-work sanity cross-check, a padding-invariance
cross-check between the S=128 and S=512 artifacts, and an optional
sentence-transformers ``CrossEncoder`` score-semantics check.

Usage:
    uv run python poc/verify_reranker_accuracy.py --seq-len 512 \
        [--compute-units CPU_AND_NE] [--check-st] [--limit 0]
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
import transformers
from transformers import PreTrainedTokenizerBase

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python poc/verify_reranker_accuracy.py` to import the poc package.
    sys.path.insert(0, str(_REPO_ROOT))

from poc.common import (  # noqa: E402
    CORPUS_DIR,
    DEFAULT_RERANKER_DIR,
    load_corpus_paragraphs,
    load_rerank_queries,
    load_reranker_torch_model,
    load_tokenizer,
    score_pytorch,
    sigmoid_np,
    spearman,
    tokenize_pairs,
)
from poc.convert_common import resolve_output_key  # noqa: E402

# Standard artifact naming produced by convert_reranker.py (§4.1): eager
# attention, macOS13 target, FP16, default mask fill. This script only
# consumes pre-built artifacts.
_ARTIFACT_STEM_TEMPLATE = "s{seq_len}_b1_eager_macos13"

_COMPUTE_UNITS: dict[str, ct.ComputeUnit] = {
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
}

# The padding-invariance check always compares these two fixed sequence
# lengths, independent of --seq-len (§4.6 step 6), and always runs on
# CPU_AND_NE regardless of --compute-units.
_PADDING_INVARIANCE_SEQ_LENS = (128, 512)
_PADDING_INVARIANCE_COMPUTE_UNITS = ct.ComputeUnit.CPU_AND_NE

# Number of (query, paragraph) pairs used for the padding-invariance check.
_PADDING_INVARIANCE_N = 10

# poc.common.load_corpus_paragraphs' default min_chars, duplicated here so
# the per-work paragraph counts used by the source-work sanity check
# (§4.6 step 5) can be recomputed without changing common.py's public
# interface. Keep in sync with poc/common.py's load_corpus_paragraphs
# default.
_CORPUS_MIN_CHARS = 40

# sentence-transformers CrossEncoder score-semantics mismatch threshold
# (§4.8 C4): a max abs diff above this indicates the activation function
# assumption in v0.2実装計画.md §2.2 (raw logits + Python sigmoid) may not
# match CrossEncoder.predict()'s own post-processing.
CHECK_ST_MISMATCH_THRESHOLD = 0.01
# Expected max abs diff when the score-semantics assumption holds.
CHECK_ST_EXPECTED_THRESHOLD = 1e-3

RESULTS_DIR = _REPO_ROOT / "poc" / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description="Verify Core ML reranker accuracy against the PyTorch FP32 baseline."
    )
    parser.add_argument("--seq-len", type=int, required=True, help="Fixed sequence length S.")
    parser.add_argument(
        "--compute-units",
        choices=sorted(_COMPUTE_UNITS),
        default="CPU_AND_NE",
        help="Core ML compute unit restriction.",
    )
    parser.add_argument(
        "--check-st",
        action="store_true",
        help="Also compare against sentence_transformers' CrossEncoder.predict() output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Truncate the documents-per-query to the first N paragraphs (0 = unlimited).",
    )
    return parser.parse_args(argv)


def default_mlmodelc_path(seq_len: int, model_dir: Path = DEFAULT_RERANKER_DIR) -> Path:
    """Return the standard compiled reranker model path for a given sequence length.

    Args:
        seq_len: Fixed sequence length S.
        model_dir: Local HF model directory whose resolved directory name
            selects the artifact subdirectory (e.g. ``ruri-v3-reranker-310m``).

    Returns:
        Path such as ``models/compiled/ruri-v3-reranker-310m/s512_b1_eager_macos13.mlmodelc``.
    """
    stem = _ARTIFACT_STEM_TEMPLATE.format(seq_len=seq_len)
    return _REPO_ROOT / "models" / "compiled" / model_dir.resolve().name / f"{stem}.mlmodelc"


def build_test_set(limit: int) -> tuple[list[dict], list[str], list[tuple[str, str]]]:
    """Build the deterministic (query, paragraph) test grid (§4.6).

    Args:
        limit: If > 0, truncate the paragraph pool to its first ``limit``
            items before pairing with every query (0 = unlimited).

    Returns:
        Tuple of (queries, paragraphs, pairs) where ``pairs`` enumerates
        every (query, paragraph) combination in query-major order
        (query order x paragraph order, both fixed).
    """
    queries = load_rerank_queries()
    paragraphs = load_corpus_paragraphs()
    if limit > 0:
        paragraphs = paragraphs[:limit]
    pairs = [(query["query"], paragraph) for query in queries for paragraph in paragraphs]
    return queries, paragraphs, pairs


def _count_filtered_paragraphs(path: Path, min_chars: int = _CORPUS_MIN_CHARS) -> int:
    """Count paragraphs in ``path`` that survive the corpus min-length filter.

    Reimplements the blank-line paragraph split and min-length filter used
    internally by :func:`poc.common.load_corpus_paragraphs`, applied to a
    single corpus file. This lets the per-work paragraph counts needed by
    the source-work sanity check (§4.6 step 5) be (re)computed at runtime
    without exposing new internals from ``poc/common.py``.

    Args:
        path: Corpus text file (blank-line separated paragraphs).
        min_chars: Minimum paragraph length (in characters) to keep.

    Returns:
        Number of paragraphs at least ``min_chars`` characters long.
    """
    text = path.read_text(encoding="utf-8")
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                blocks.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return len([block for block in blocks if len(block) >= min_chars])


def paragraph_source_works(n_paragraphs: int) -> list[str]:
    """Label the first ``n_paragraphs`` corpus paragraphs by source work.

    Mirrors ``load_corpus_paragraphs``' fixed composition order (kumonoito
    in full, then sangetsuki in full, then kokoro's leading paragraphs).
    The kumonoito/sangetsuki paragraph counts are determined dynamically by
    :func:`_count_filtered_paragraphs`; any remaining index is labeled
    ``"kokoro"``.

    Args:
        n_paragraphs: Number of leading corpus paragraphs to label (i.e.
            the size of the paragraph pool actually used, after --limit).

    Returns:
        List of ``"kumonoito"``/``"sangetsuki"``/``"kokoro"`` labels, one
        per paragraph index in ``[0, n_paragraphs)``.
    """
    n_kumonoito = _count_filtered_paragraphs(CORPUS_DIR / "kumonoito.txt")
    n_sangetsuki = _count_filtered_paragraphs(CORPUS_DIR / "sangetsuki.txt")
    labels: list[str] = []
    for i in range(n_paragraphs):
        if i < n_kumonoito:
            labels.append("kumonoito")
        elif i < n_kumonoito + n_sangetsuki:
            labels.append("sangetsuki")
        else:
            labels.append("kokoro")
    return labels


def score_coreml(
    model: ct.models.CompiledMLModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: list[tuple[str, str]],
    seq_len: int,
) -> tuple[np.ndarray, str]:
    """Run a batch-size-1 Core ML inference loop over ``pairs``.

    Args:
        model: Loaded compiled reranker model.
        tokenizer: Tokenizer returned by :func:`poc.common.load_tokenizer`.
        pairs: List of (query, document) pairs.
        seq_len: Fixed sequence length used for tokenization.

    Returns:
        Tuple of (raw logits, shape (len(pairs),), and the resolved output
        key).
    """
    batch = tokenize_pairs(tokenizer, pairs, seq_len)
    output_key: str | None = None
    scores = np.empty(len(pairs), dtype=np.float32)
    for i in range(len(pairs)):
        prediction = model.predict(
            {
                "input_ids": batch["input_ids"][i : i + 1],
                "attention_mask": batch["attention_mask"][i : i + 1],
            }
        )
        output_key = output_key or resolve_output_key(prediction, "logits")
        # The graph output is (1, 1); flatten to a scalar logit per pair.
        scores[i] = float(np.asarray(prediction[output_key], dtype=np.float32).reshape(-1)[0])
    return scores, output_key or "logits"


def compute_metrics(
    coreml_logits: np.ndarray,
    fp32_logits: np.ndarray,
    queries: list[dict],
    n_docs: int,
) -> dict[str, Any]:
    """Compute the per-query and aggregate accuracy metrics (§4.6 steps 1-5).

    Args:
        coreml_logits: Core ML raw logits, shape (len(queries) * n_docs,),
            in query-major order.
        fp32_logits: PyTorch FP32 baseline raw logits, same shape/order.
        queries: Query dicts as returned by :func:`poc.common.load_rerank_queries`.
        n_docs: Number of paragraphs per query (after --limit).

    Returns:
        Dict with the ``per_query`` breakdown and aggregate metrics.
    """
    n_queries = len(queries)
    coreml_mat = coreml_logits.reshape(n_queries, n_docs)
    fp32_mat = fp32_logits.reshape(n_queries, n_docs)
    coreml_sigmoid = sigmoid_np(coreml_logits)
    fp32_sigmoid = sigmoid_np(fp32_logits)
    sigmoid_abs_diff = np.abs(coreml_sigmoid - fp32_sigmoid)

    source_labels = paragraph_source_works(n_docs)
    top_k = min(5, n_docs)
    per_query: list[dict[str, Any]] = []
    spearmans: list[float] = []
    top1_match_count = 0
    for qi, query in enumerate(queries):
        coreml_row = coreml_mat[qi]
        fp32_row = fp32_mat[qi]
        rho = spearman(coreml_row, fp32_row)
        spearmans.append(rho)
        top1_coreml = int(np.argmax(coreml_row))
        top1_fp32 = int(np.argmax(fp32_row))
        top1_match = top1_coreml == top1_fp32
        top1_match_count += int(top1_match)
        top5_coreml = set(np.argsort(coreml_row)[::-1][:top_k].tolist())
        top5_fp32 = set(np.argsort(fp32_row)[::-1][:top_k].tolist())
        per_query.append(
            {
                "query_id": query["id"],
                "spearman": rho,
                "top1_match": top1_match,
                "top1_doc_index": top1_fp32,
                "top5_overlap": len(top5_coreml & top5_fp32),
                "source_work_hit": source_labels[top1_fp32] == query["source_work"],
            }
        )

    nan_count = int(np.isnan(coreml_logits).sum())
    inf_count = int(np.isinf(coreml_logits).sum())
    return {
        "per_query": per_query,
        "spearman_min": float(min(spearmans)) if spearmans else None,
        "spearman_mean": float(sum(spearmans) / len(spearmans)) if spearmans else None,
        "sigmoid_mae": float(sigmoid_abs_diff.mean()),
        "sigmoid_max_abs_diff": float(sigmoid_abs_diff.max()),
        "top1_match_count": top1_match_count,
        "top1_match_rate": (top1_match_count / n_queries) if n_queries else None,
        "top_k": top_k,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nonfinite_count": nan_count + inf_count,
        "fp32_nonfinite_count": int((~np.isfinite(fp32_logits)).sum()),
    }


def select_padding_invariance_pairs(
    pairs: list[tuple[str, str]], tokenizer: PreTrainedTokenizerBase, max_tokens: int, n: int
) -> list[tuple[str, str]]:
    """Select the first ``n`` pairs whose untruncated pair token length fits ``max_tokens``.

    Args:
        pairs: Candidate (query, document) pairs, in query-major order.
        tokenizer: Tokenizer returned by :func:`poc.common.load_tokenizer`.
        max_tokens: Maximum untruncated pair token length (inclusive).
        n: Maximum number of pairs to select.

    Returns:
        Up to ``n`` pairs, in the same relative order as ``pairs``.
    """
    selected: list[tuple[str, str]] = []
    for pair in pairs:
        if len(selected) >= n:
            break
        length = len(tokenizer(pair[0], pair[1])["input_ids"])
        if length <= max_tokens:
            selected.append(pair)
    return selected


def run_padding_invariance_check(
    pairs: list[tuple[str, str]], tokenizer: PreTrainedTokenizerBase
) -> dict[str, Any]:
    """Cross-check logits from the S=128 and S=512 artifacts (§4.6 step 6).

    Both compiled models are loaded at their standard artifact paths
    (independent of the CLI ``--seq-len`` used for the main comparison),
    always on CPU_AND_NE, and scored on the same subset of pairs.

    Args:
        pairs: The main test grid's (query, document) pairs (query-major
            order, already reflecting --limit); candidates are drawn from
            this list.
        tokenizer: Tokenizer returned by :func:`poc.common.load_tokenizer`.

    Returns:
        Dict describing whether the check ran, and if so, the max absolute
        logit/sigmoid differences between the two shapes. If either
        artifact is missing, or no pair fits within the token budget,
        ``checked`` is False and the reason is recorded instead of raising.
    """
    max_tokens = min(_PADDING_INVARIANCE_SEQ_LENS)
    candidates = select_padding_invariance_pairs(
        pairs, tokenizer, max_tokens, _PADDING_INVARIANCE_N
    )
    if not candidates:
        print(
            "WARNING: padding-invariance check skipped: no pair fits within "
            f"{max_tokens} untruncated tokens",
            file=sys.stderr,
        )
        return {
            "checked": False,
            "skipped_reason": f"no pair fits within {max_tokens} untruncated tokens",
            "seq_lens": list(_PADDING_INVARIANCE_SEQ_LENS),
        }

    logits: dict[int, np.ndarray] = {}
    for seq_len in _PADDING_INVARIANCE_SEQ_LENS:
        path = default_mlmodelc_path(seq_len)
        if not path.exists():
            print(
                f"WARNING: padding-invariance check skipped: {path} not found",
                file=sys.stderr,
            )
            return {
                "checked": False,
                "skipped_reason": f"missing model: {path}",
                "seq_lens": list(_PADDING_INVARIANCE_SEQ_LENS),
            }
        model = ct.models.CompiledMLModel(
            str(path), compute_units=_PADDING_INVARIANCE_COMPUTE_UNITS
        )
        scores, _ = score_coreml(model, tokenizer, candidates, seq_len)
        logits[seq_len] = scores
        del model
        gc.collect()

    low, high = _PADDING_INVARIANCE_SEQ_LENS
    logit_diff = np.abs(logits[low] - logits[high])
    sigmoid_diff = np.abs(sigmoid_np(logits[low]) - sigmoid_np(logits[high]))
    return {
        "checked": True,
        "skipped_reason": None,
        "seq_lens": list(_PADDING_INVARIANCE_SEQ_LENS),
        "n_pairs": len(candidates),
        "n_requested": _PADDING_INVARIANCE_N,
        "logit_max_abs_diff": float(logit_diff.max()),
        "sigmoid_max_abs_diff": float(sigmoid_diff.max()),
        "sigmoid_mean_abs_diff": float(sigmoid_diff.mean()),
    }


def run_cross_encoder_check(
    model_dir: Path,
    queries: list[dict],
    paragraphs: list[str],
    fp32_logits: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
) -> dict[str, Any] | None:
    """Compare sentence_transformers' CrossEncoder.predict() with the manual baseline.

    Runs on the first query's block only (§4.6 step 7), restricted to the
    pairs whose untruncated pair token length fits within ``seq_len``:
    CrossEncoder tokenizes with its own (longer) dynamic max length, so
    pairs that our fixed-length tokenization truncates would diverge for
    reasons unrelated to score semantics (the activation-function
    assumption this check exists to verify).

    If ``sentence_transformers`` cannot be imported or fails to run, a
    warning is printed and ``None`` is returned so the caller can continue
    without affecting the process exit code. If the max absolute
    difference exceeds :data:`CHECK_ST_MISMATCH_THRESHOLD`, an explicit
    error is printed (§4.8 C4) but the run still completes and the result
    is still recorded in the returned dict.

    Args:
        model_dir: Local HF/sentence-transformers reranker model directory.
        queries: Query dicts as returned by :func:`poc.common.load_rerank_queries`.
        paragraphs: The (possibly --limit-truncated) paragraph pool.
        fp32_logits: PyTorch FP32 baseline raw logits for the full test
            grid, query-major order; the first ``len(paragraphs)`` entries
            correspond to the first query's block.
        tokenizer: Tokenizer returned by :func:`poc.common.load_tokenizer`.
        seq_len: Fixed sequence length S used by the manual baseline.

    Returns:
        Dict with the max/mean absolute difference and the mismatch flag,
        or None if the check was skipped.
    """
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:  # pragma: no cover - optional dependency
        print(
            f"WARNING: --check-st skipped, sentence-transformers unavailable: {exc}",
            file=sys.stderr,
        )
        return None
    try:
        first_query = queries[0]
        indices = [
            i
            for i, paragraph in enumerate(paragraphs)
            if len(tokenizer(first_query["query"], paragraph)["input_ids"]) <= seq_len
        ]
        if not indices:
            print(
                f"WARNING: --check-st skipped: no pair fits within {seq_len} untruncated tokens",
                file=sys.stderr,
            )
            return None
        block_pairs = [[first_query["query"], paragraphs[i]] for i in indices]
        baseline_scores = sigmoid_np(fp32_logits[np.asarray(indices)])

        ce_model = CrossEncoder(str(model_dir))
        ce_scores = np.asarray(ce_model.predict(block_pairs), dtype=np.float32).reshape(-1)
        del ce_model
        gc.collect()

        abs_diff = np.abs(ce_scores - baseline_scores)
        max_abs_diff = float(abs_diff.max())
        mismatch = max_abs_diff > CHECK_ST_MISMATCH_THRESHOLD
        if mismatch:
            print(
                "ERROR: --check-st score semantics mismatch: max|CrossEncoder - "
                f"sigmoid(fp32 logits)| = {max_abs_diff:.6f} exceeds "
                f"{CHECK_ST_MISMATCH_THRESHOLD} (§2.2/§4.8 C4). The sigmoid "
                "post-processing assumption may not match CrossEncoder's own "
                "activation; investigate before trusting sigmoid-space scores.",
                file=sys.stderr,
            )
        return {
            "query_id": first_query["id"],
            "n": len(indices),
            "n_pool": len(paragraphs),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": float(abs_diff.mean()),
            "expected_threshold": CHECK_ST_EXPECTED_THRESHOLD,
            "mismatch_threshold": CHECK_ST_MISMATCH_THRESHOLD,
            "semantics_mismatch": mismatch,
        }
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"WARNING: --check-st failed: {exc}", file=sys.stderr)
        return None


def print_summary(result: dict[str, Any], out_path: Path) -> None:
    """Print the human-readable summary table to stdout."""
    metrics = result["metrics"]
    padding = result["padding_invariance"]
    print("\n=== Reranker accuracy summary ===")
    n_queries = result["test_set"]["n_queries"]
    n_docs = result["test_set"]["n_docs_per_query"]
    top1_rate = metrics["top1_match_rate"]
    top1_count = metrics["top1_match_count"]
    print(f"seq_len              : {result['seq_len']}")
    print(f"compute_units        : {result['compute_units']}")
    print(f"n queries x docs     : {n_queries} x {n_docs}")
    print(f"spearman min / mean  : {metrics['spearman_min']:.6f} / {metrics['spearman_mean']:.6f}")
    print(f"sigmoid MAE          : {metrics['sigmoid_mae']:.6f}")
    print(f"sigmoid max abs diff : {metrics['sigmoid_max_abs_diff']:.6f}")
    print(f"top-1 match rate     : {top1_rate:.3f} ({top1_count}/{n_queries})")
    print(
        f"non-finite (coreml)  : {metrics['nonfinite_count']} "
        f"(nan={metrics['nan_count']}, inf={metrics['inf_count']})"
    )
    if padding["checked"]:
        print(
            f"padding invariance (S{padding['seq_lens'][0]} vs S{padding['seq_lens'][1]}, "
            f"n={padding['n_pairs']}/{padding['n_requested']}) "
            f"logit max|Δ|: {padding['logit_max_abs_diff']:.6f}, "
            f"sigmoid max|Δ|: {padding['sigmoid_max_abs_diff']:.6f}"
        )
    else:
        print(f"padding invariance: skipped ({padding['skipped_reason']})")
    if result["cross_encoder_check"] is not None:
        ce = result["cross_encoder_check"]
        print(
            f"--check-st max|Δ|    : {ce['max_abs_diff']:.6f} "
            f"(n={ce['n']}/{ce['n_pool']}, mismatch={ce['semantics_mismatch']})"
        )
    print(f"elapsed (total)      : {result['timings_sec']['total']:.2f}s")
    print(f"results written      : {out_path}")


def main(argv: list[str] | None = None) -> int:
    """Run the full reranker accuracy verification pipeline.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 1 if the Core ML output contains any NaN/Inf
        logit, 0 otherwise (Spearman/sigmoid thresholds never fail the
        exit code; see §4.6).
    """
    args = parse_args(argv)
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be a positive integer")
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    compute_units = _COMPUTE_UNITS[args.compute_units]
    mlmodelc_path = default_mlmodelc_path(args.seq_len)
    if not mlmodelc_path.exists():
        raise SystemExit(f"compiled model not found: {mlmodelc_path}")

    started = time.perf_counter()
    timings: dict[str, float] = {}

    print(f"[1/6] Building deterministic test grid (limit={args.limit or 'none'})")
    queries, paragraphs, pairs = build_test_set(args.limit)
    print(f"      {len(queries)} queries x {len(paragraphs)} paragraphs = {len(pairs)} pairs")

    print(f"[2/6] Computing PyTorch FP32 baseline (attn=sdpa, seq_len={args.seq_len})")
    step = time.perf_counter()
    tokenizer = load_tokenizer(DEFAULT_RERANKER_DIR)
    baseline_model = load_reranker_torch_model(DEFAULT_RERANKER_DIR, attn="sdpa")
    fp32_logits = score_pytorch(baseline_model, tokenizer, pairs, args.seq_len)
    timings["baseline"] = time.perf_counter() - step

    ce_result: dict[str, Any] | None = None
    if args.check_st:
        print("[2b/6] Comparing against sentence-transformers CrossEncoder.predict()")
        step = time.perf_counter()
        ce_result = run_cross_encoder_check(
            DEFAULT_RERANKER_DIR, queries, paragraphs, fp32_logits, tokenizer, args.seq_len
        )
        timings["check_st"] = time.perf_counter() - step

    del baseline_model
    gc.collect()

    print(f"[3/6] Running Core ML inference on {mlmodelc_path} ({args.compute_units})")
    step = time.perf_counter()
    coreml_model = ct.models.CompiledMLModel(str(mlmodelc_path), compute_units=compute_units)
    coreml_logits, output_key = score_coreml(coreml_model, tokenizer, pairs, args.seq_len)
    del coreml_model
    gc.collect()
    timings["coreml"] = time.perf_counter() - step

    print("[4/6] Computing accuracy metrics")
    metrics = compute_metrics(coreml_logits, fp32_logits, queries, len(paragraphs))

    print("[5/6] Checking padding invariance (S=128 vs S=512)")
    step = time.perf_counter()
    padding_result = run_padding_invariance_check(pairs, tokenizer)
    timings["padding_invariance"] = time.perf_counter() - step

    timings["total"] = time.perf_counter() - started

    print("[6/6] Writing results")
    result: dict[str, Any] = {
        "seq_len": args.seq_len,
        "compute_units": args.compute_units,
        "mlmodelc_path": str(mlmodelc_path),
        "reranker_model_dir": str(DEFAULT_RERANKER_DIR),
        "output_key": output_key,
        "limit": args.limit,
        "test_set": {
            "n_queries": len(queries),
            "n_docs_per_query": len(paragraphs),
            "n_pairs": len(pairs),
        },
        "metrics": {k: v for k, v in metrics.items() if k != "per_query"},
        "per_query": metrics["per_query"],
        "padding_invariance": padding_result,
        "cross_encoder_check": ce_result,
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "coremltools": ct.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "timings_sec": {k: round(v, 3) for k, v in timings.items()},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"rerank_accuracy_s{args.seq_len}_{args.compute_units}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(result, out_path)

    has_nonfinite = metrics["nonfinite_count"] > 0
    return 1 if has_nonfinite else 0


if __name__ == "__main__":
    raise SystemExit(main())
