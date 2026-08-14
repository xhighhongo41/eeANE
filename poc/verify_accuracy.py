"""Verify Core ML embedding accuracy against the PyTorch FP32 baseline.

Implements 開発資料/v0.1実装計画.md §4.6 (accuracy verification design) and
the "T6: 精度検証スクリプト" task. Loads a pre-compiled ``.mlmodelc``
(produced by ``poc/convert_embedding.py``), runs a deterministic test set
through both the Core ML model and the PyTorch FP32 baseline
(:func:`poc.common.encode_pytorch`), and reports row-wise cosine
similarity statistics plus a padding-invariance cross-check between the
S=128 and S=512 artifacts.

Usage:
    uv run python poc/verify_accuracy.py --seq-len 512 \
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
from transformers import PreTrainedModel, PreTrainedTokenizerBase

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python poc/verify_accuracy.py` to import the poc package.
    sys.path.insert(0, str(_REPO_ROOT))

from poc.common import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    PREFIX_TEST_SENTENCES,
    PREFIXES,
    cosine_rowwise,
    encode_pytorch,
    load_corpus_paragraphs,
    load_tokenizer,
    load_torch_model,
    tokenize_batch,
)

# Standard artifact naming produced by convert_embedding.py (§4.4): eager
# attention, macOS13 target. This script only consumes pre-built artifacts.
_ARTIFACT_STEM_TEMPLATE = "s{seq_len}_b1_eager_macos13"

_COMPUTE_UNITS: dict[str, ct.ComputeUnit] = {
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "ALL": ct.ComputeUnit.ALL,
}

# Minimum acceptable cosine similarity for the padding-invariance check
# (§4.6): verifies that the attention-mask-driven mean pooling produces
# (near) identical embeddings regardless of the padding amount.
PADDING_INVARIANCE_THRESHOLD = 0.999

# The padding-invariance check always compares these two fixed sequence
# lengths, independent of --seq-len (§4.6 step 5).
_PADDING_INVARIANCE_SEQ_LENS = (128, 512)

# Number of texts used for the padding-invariance check.
_PADDING_INVARIANCE_N = 10

RESULTS_DIR = _REPO_ROOT / "poc" / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description="Verify Core ML embedding accuracy against the PyTorch FP32 baseline."
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
        help="Also compare against sentence-transformers' own encode() output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Truncate the test set to the first N items (0 = unlimited; for debugging).",
    )
    parser.add_argument(
        "--mlmodelc",
        type=Path,
        default=None,
        help=(
            "Override the compiled model path used for --seq-len "
            "(defaults to the standard artifact path)."
        ),
    )
    return parser.parse_args(argv)


def default_mlmodelc_path(seq_len: int, model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """Return the standard compiled model path for a given sequence length.

    Args:
        seq_len: Fixed sequence length S.
        model_dir: Local HF model directory whose resolved directory name
            selects the artifact subdirectory (e.g. ``ruri-v3-310m``).

    Returns:
        Path such as ``models/compiled/ruri-v3-310m/s512_b1_eager_macos13.mlmodelc``.
    """
    stem = _ARTIFACT_STEM_TEMPLATE.format(seq_len=seq_len)
    return _REPO_ROOT / "models" / "compiled" / model_dir.resolve().name / f"{stem}.mlmodelc"


def build_test_set(limit: int) -> list[str]:
    """Build the deterministic accuracy test set (§4.6).

    Concatenates (a) all corpus paragraphs from
    :func:`poc.common.load_corpus_paragraphs` with (b) the prefix
    confirmation set: each of :data:`poc.common.PREFIXES` (in definition
    order) prepended to each of :data:`poc.common.PREFIX_TEST_SENTENCES`
    (in list order), giving 4 x 8 = 32 texts.

    Args:
        limit: If > 0, truncate the resulting list to its first ``limit``
            items (0 = unlimited).

    Returns:
        The ordered list of test texts.
    """
    corpus = load_corpus_paragraphs()
    prefix_set = [
        prefix + sentence for prefix in PREFIXES.values() for sentence in PREFIX_TEST_SENTENCES
    ]
    texts = corpus + prefix_set
    if limit > 0:
        texts = texts[:limit]
    return texts


def build_padding_invariance_texts(tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Build the fixed 10-text set used by the padding-invariance check.

    Takes the first 10 of :data:`poc.common.PREFIX_TEST_SENTENCES`; since
    there are only 8, pads with the first corpus paragraphs that fit
    within the smaller compared shape (§4.6 step 5). Longer paragraphs
    would be truncated at S=128 and the check would then measure
    truncation differences instead of padding invariance.

    Args:
        tokenizer: Tokenizer used to measure untruncated token lengths.

    Returns:
        A list of exactly 10 texts (fewer only if the corpus has too few
        short paragraphs, which does not happen for the fixed corpus).
    """
    texts = list(PREFIX_TEST_SENTENCES[:_PADDING_INVARIANCE_N])
    max_tokens = min(_PADDING_INVARIANCE_SEQ_LENS)
    for paragraph in load_corpus_paragraphs():
        if len(texts) >= _PADDING_INVARIANCE_N:
            break
        # Length without padding/truncation, special tokens included.
        n_tokens = len(tokenizer(paragraph)["input_ids"])
        if n_tokens <= max_tokens:
            texts.append(paragraph)
    return texts


def _resolve_output_key(prediction: dict[str, Any]) -> str:
    """Pick the embedding output key from a ``predict`` result dict.

    Raises:
        RuntimeError: If the model returned no outputs.
    """
    keys = list(prediction)
    if not keys:
        raise RuntimeError("Core ML model returned no outputs")
    # Prefer the requested name but tolerate a renamed output.
    return "embedding" if "embedding" in keys else keys[0]


def encode_coreml(
    model: ct.models.CompiledMLModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    seq_len: int,
) -> tuple[np.ndarray, str]:
    """Run a batch-size-1 Core ML inference loop over ``texts``.

    Args:
        model: Loaded compiled model (see ``ct.models.CompiledMLModel``).
        tokenizer: Tokenizer returned by :func:`poc.common.load_tokenizer`.
        texts: Input sentences.
        seq_len: Fixed sequence length used for tokenization.

    Returns:
        Tuple of (embeddings of shape (len(texts), H), resolved output key).
    """
    batch = tokenize_batch(tokenizer, texts, seq_len)
    output_key: str | None = None
    rows: list[np.ndarray] = []
    for i in range(len(texts)):
        prediction = model.predict(
            {
                "input_ids": batch["input_ids"][i : i + 1],
                "attention_mask": batch["attention_mask"][i : i + 1],
            }
        )
        output_key = output_key or _resolve_output_key(prediction)
        rows.append(np.asarray(prediction[output_key], dtype=np.float32).reshape(-1))
    return np.stack(rows), output_key or "embedding"


def compute_cosine_metrics(coreml_emb: np.ndarray, baseline_emb: np.ndarray) -> dict[str, Any]:
    """Compute row-wise cosine similarity statistics and NaN/Inf counts.

    Args:
        coreml_emb: Core ML (FP16) embeddings, shape (N, H).
        baseline_emb: PyTorch FP32 baseline embeddings, shape (N, H).

    Returns:
        Dict with cosine mean/min/p5, non-finite element/row counts, and
        the per-text cosine list.
    """
    cosines = cosine_rowwise(coreml_emb, baseline_emb)
    nonfinite_rows = ~np.isfinite(coreml_emb).all(axis=1)
    return {
        "n": int(len(cosines)),
        "cosine_mean": float(np.mean(cosines)),
        "cosine_min": float(np.min(cosines)),
        "cosine_p5": float(np.percentile(cosines, 5)),
        "nan_element_count": int(np.isnan(coreml_emb).sum()),
        "inf_element_count": int(np.isinf(coreml_emb).sum()),
        "nonfinite_row_count": int(nonfinite_rows.sum()),
        "cosine_per_text": [float(c) for c in cosines],
    }


def run_padding_invariance_check(
    tokenizer: PreTrainedTokenizerBase, compute_units: ct.ComputeUnit
) -> dict[str, Any]:
    """Cross-check embeddings from the S=128 and S=512 artifacts (§4.6 step 5).

    Both compiled models are loaded at their standard artifact paths
    (independent of the CLI ``--seq-len``/``--mlmodelc`` used for the main
    comparison) so the check always exercises the same two shapes.

    Args:
        tokenizer: Tokenizer returned by :func:`poc.common.load_tokenizer`.
        compute_units: Compute unit restriction to use for both models
            (same setting as the main run).

    Returns:
        Dict describing whether the check ran, and if so, the min/mean
        cross-shape cosine similarity and pass/fail against the threshold.
        If either artifact is missing, ``checked`` is False and the reason
        is recorded instead of raising.
    """
    texts = build_padding_invariance_texts(tokenizer)
    embeddings: dict[int, np.ndarray] = {}
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
        model = ct.models.CompiledMLModel(str(path), compute_units=compute_units)
        emb, _ = encode_coreml(model, tokenizer, texts, seq_len)
        embeddings[seq_len] = emb
        del model
        gc.collect()

    low, high = _PADDING_INVARIANCE_SEQ_LENS
    cosines = cosine_rowwise(embeddings[low], embeddings[high])
    return {
        "checked": True,
        "skipped_reason": None,
        "seq_lens": list(_PADDING_INVARIANCE_SEQ_LENS),
        "n_texts": len(texts),
        "cosine_min": float(cosines.min()),
        "cosine_mean": float(cosines.mean()),
        "cosine_per_text": [float(c) for c in cosines],
        "threshold": PADDING_INVARIANCE_THRESHOLD,
        "passed": bool(cosines.min() >= PADDING_INVARIANCE_THRESHOLD),
    }


def run_sentence_transformers_check(
    model_dir: Path,
    baseline_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
) -> dict[str, Any] | None:
    """Compare sentence-transformers' own encode() with the manual baseline.

    This is a sanity check on the manual baseline itself (§4.6, optional):
    if sentence-transformers cannot be imported or fails to run, a warning
    is printed and ``None`` is returned so the caller can continue without
    affecting the process exit code.

    Args:
        model_dir: Local HF/sentence-transformers model directory.
        baseline_model: Already-loaded PyTorch baseline model (reused to
            avoid a second load).
        tokenizer: Tokenizer returned by :func:`poc.common.load_tokenizer`.
        seq_len: Fixed sequence length used for the manual baseline side.

    Returns:
        Dict with per-sentence cosine similarities, or None if skipped.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - optional dependency
        print(
            f"WARNING: --check-st skipped, sentence-transformers unavailable: {exc}",
            file=sys.stderr,
        )
        return None
    try:
        st_model = SentenceTransformer(str(model_dir))
        st_emb = st_model.encode(
            PREFIX_TEST_SENTENCES, convert_to_numpy=True, normalize_embeddings=False
        )
        st_emb = np.asarray(st_emb, dtype=np.float32)
        # encode_pytorch uses fixed-shape padding to seq_len, while
        # SentenceTransformer.encode() tokenizes each batch to its own
        # (shorter) max length; any residual gap below 1.0 partly reflects
        # that difference rather than only FP32 rounding noise.
        baseline_emb = encode_pytorch(baseline_model, tokenizer, PREFIX_TEST_SENTENCES, seq_len)
        cosines = cosine_rowwise(st_emb, baseline_emb)
        del st_model
        gc.collect()
        return {
            "n": len(PREFIX_TEST_SENTENCES),
            "cosine_min": float(cosines.min()),
            "cosine_mean": float(cosines.mean()),
            "cosine_per_sentence": [float(c) for c in cosines],
            "note": (
                "sentence-transformers tokenizes to a dynamic max length "
                "while encode_pytorch pads/truncates to the fixed seq_len; "
                "this is recorded for reference and is not a pass/fail gate."
            ),
        }
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"WARNING: --check-st failed: {exc}", file=sys.stderr)
        return None


def print_summary(result: dict[str, Any], out_path: Path) -> None:
    """Print the human-readable summary table to stdout."""
    metrics = result["metrics"]
    padding = result["padding_invariance"]
    print("\n=== Accuracy summary ===")
    print(f"seq_len          : {result['seq_len']}")
    print(f"compute_units    : {result['compute_units']}")
    print(f"n texts          : {metrics['n']}")
    print(f"cosine mean      : {metrics['cosine_mean']:.6f}")
    print(f"cosine min       : {metrics['cosine_min']:.6f}")
    print(f"cosine p5        : {metrics['cosine_p5']:.6f}")
    print(f"non-finite rows  : {metrics['nonfinite_row_count']}")
    if padding["checked"]:
        print(
            f"padding invariance (S{padding['seq_lens'][0]} vs S{padding['seq_lens'][1]}) "
            f"min cosine: {padding['cosine_min']:.6f} "
            f"(threshold {padding['threshold']}, passed={padding['passed']})"
        )
    else:
        print(f"padding invariance: skipped ({padding['skipped_reason']})")
    if result["sentence_transformers_check"] is not None:
        st = result["sentence_transformers_check"]
        print(f"sentence-transformers check min cosine: {st['cosine_min']:.6f}")
    print(f"elapsed (total)  : {result['timings_sec']['total']:.2f}s")
    print(f"results written  : {out_path}")


def main(argv: list[str] | None = None) -> int:
    """Run the full accuracy verification pipeline.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 1 if the Core ML output contains any NaN/Inf
        row, 0 otherwise (cosine thresholds never fail the exit code; see
        §4.6).
    """
    args = parse_args(argv)
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be a positive integer")
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    compute_units = _COMPUTE_UNITS[args.compute_units]
    mlmodelc_path = (
        args.mlmodelc if args.mlmodelc is not None else default_mlmodelc_path(args.seq_len)
    )
    if not mlmodelc_path.exists():
        raise SystemExit(f"compiled model not found: {mlmodelc_path}")

    started = time.perf_counter()
    timings: dict[str, float] = {}

    print(f"[1/5] Building deterministic test set (limit={args.limit or 'none'})")
    texts = build_test_set(args.limit)
    print(f"      {len(texts)} texts")

    print(f"[2/5] Computing PyTorch FP32 baseline (attn=sdpa, seq_len={args.seq_len})")
    step = time.perf_counter()
    tokenizer = load_tokenizer(DEFAULT_MODEL_DIR)
    baseline_model = load_torch_model(DEFAULT_MODEL_DIR, attn="sdpa")
    baseline_emb = encode_pytorch(baseline_model, tokenizer, texts, args.seq_len)
    timings["baseline"] = time.perf_counter() - step

    st_result: dict[str, Any] | None = None
    if args.check_st:
        print("[2b/5] Comparing against sentence-transformers encode()")
        step = time.perf_counter()
        st_result = run_sentence_transformers_check(
            DEFAULT_MODEL_DIR, baseline_model, tokenizer, args.seq_len
        )
        timings["check_st"] = time.perf_counter() - step

    del baseline_model
    gc.collect()

    print(f"[3/5] Running Core ML inference on {mlmodelc_path} ({args.compute_units})")
    step = time.perf_counter()
    coreml_model = ct.models.CompiledMLModel(str(mlmodelc_path), compute_units=compute_units)
    coreml_emb, output_key = encode_coreml(coreml_model, tokenizer, texts, args.seq_len)
    del coreml_model
    gc.collect()
    timings["coreml"] = time.perf_counter() - step

    print("[4/5] Computing cosine metrics")
    metrics = compute_cosine_metrics(coreml_emb, baseline_emb)

    print("[5/5] Checking padding invariance (S=128 vs S=512)")
    step = time.perf_counter()
    padding_result = run_padding_invariance_check(tokenizer, compute_units)
    timings["padding_invariance"] = time.perf_counter() - step

    timings["total"] = time.perf_counter() - started

    result: dict[str, Any] = {
        "seq_len": args.seq_len,
        "compute_units": args.compute_units,
        "mlmodelc_path": str(mlmodelc_path),
        "output_key": output_key,
        "limit": args.limit,
        "test_set": {"n_total": len(texts)},
        "metrics": {k: v for k, v in metrics.items() if k != "cosine_per_text"},
        "cosine_per_text": metrics["cosine_per_text"],
        "texts": texts,
        "padding_invariance": padding_result,
        "sentence_transformers_check": st_result,
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
    out_path = RESULTS_DIR / f"accuracy_s{args.seq_len}_{args.compute_units}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(result, out_path)

    has_nonfinite = metrics["nonfinite_row_count"] > 0
    return 1 if has_nonfinite else 0


if __name__ == "__main__":
    raise SystemExit(main())
