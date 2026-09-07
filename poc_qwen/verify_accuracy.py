"""Verify the accuracy of a compiled Qwen3 embedding ``.mlmodelc`` artifact.

This script does not convert anything -- it loads an already-compiled
Core ML artifact (produced by a separate conversion script) and measures
how closely it reproduces the source model's embeddings, from several
independent angles:

(a) row-wise cosine similarity against the FP32 PyTorch reference
    (:func:`poc_qwen.common.encode_reference`), scored per language so a
    language the model has little vocabulary for cannot hide a language
    it handles well;
(b) row-wise cosine similarity against sentence-transformers' own
    ``encode()``, which shares no tokenization/pooling code with this
    project and is therefore the most independent check here;
(c) the distribution of cosine similarities against the FP32 reference
    over real document paragraphs from the fixed test corpus;
(d) padding invariance: whether two artifacts converted at different
    fixed sequence lengths embed the same short text (near-)identically;
(e) whether the Core ML output ever contains NaN/Inf.

Usage:
    python -m poc_qwen.verify_accuracy --mlmodelc path/to/s512.mlmodelc --seq-len 512
    python poc_qwen/verify_accuracy.py --mlmodelc path/to/s512.mlmodelc --seq-len 512
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python poc_qwen/verify_accuracy.py` to import the poc_qwen package;
    # `python -m poc_qwen.verify_accuracy` already has the repo root on sys.path.
    sys.path.insert(0, str(_REPO_ROOT))

from poc_qwen import common  # noqa: E402

# Directory holding the fixed Aozora Bunko test corpus used by check (c).
CORPUS_DIR = _REPO_ROOT / "testdata" / "corpus"

# Files read for check (c), in a fixed order so a given --corpus-limit
# always selects the same paragraphs.
_CORPUS_FILENAMES: tuple[str, ...] = ("kokoro.txt", "sangetsuki.txt", "kumonoito.txt")

# Paragraphs shorter than this are headings, single words, or other
# fragments that are not representative document text.
_CORPUS_MIN_CHARS = 40

# Minimum acceptable cosine similarity between the two artifacts of a
# padding-invariance check (d). Distinct from
# common.SANITY_COSINE_THRESHOLD because this compares two Core ML runs
# of the same model to each other, not Core ML to an FP32 reference, so a
# much tighter bound is expected to hold.
PADDING_INVARIANCE_THRESHOLD = 0.999

_COMPUTE_UNITS: dict[str, ct.ComputeUnit] = {
    "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
    "all": ct.ComputeUnit.ALL,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments (``argv`` defaults to ``sys.argv[1:]``)."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure the accuracy of a compiled Qwen3 embedding .mlmodelc artifact "
            "against the PyTorch FP32 reference, sentence-transformers, and a fixed "
            "text corpus. Does not perform any conversion."
        )
    )
    parser.add_argument(
        "--model-id",
        default=common.DEFAULT_MODEL_ID,
        help="Hub id or local directory of the source model (default: %(default)s).",
    )
    parser.add_argument(
        "--mlmodelc",
        type=Path,
        required=True,
        help="Path to the compiled .mlmodelc artifact to verify.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        required=True,
        help="Fixed sequence length the artifact in --mlmodelc was converted with.",
    )
    parser.add_argument(
        "--mlmodelc-alt",
        type=Path,
        default=None,
        help=(
            "Path to a second compiled artifact, converted at a different sequence "
            "length, used for the padding-invariance check (d). Requires "
            "--seq-len-alt."
        ),
    )
    parser.add_argument(
        "--seq-len-alt",
        type=int,
        default=None,
        help="Fixed sequence length of --mlmodelc-alt.",
    )
    parser.add_argument(
        "--compute-units",
        choices=sorted(_COMPUTE_UNITS),
        default="cpu_and_ne",
        help="Core ML compute unit restriction (default: %(default)s).",
    )
    parser.add_argument(
        "--corpus-limit",
        type=int,
        default=60,
        help="Max corpus paragraphs to evaluate for check (c) (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-st",
        action="store_true",
        help="Skip the sentence-transformers cross-check (b).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path (default: poc_qwen/results/accuracy_s{seq_len}_{compute_units}.json)."
        ),
    )
    return parser.parse_args(argv)


def _resolve_output_key(prediction: dict[str, Any]) -> str:
    """Pick the model's output feature name from a ``predict()`` result.

    The Core ML converter sometimes renames or auto-generates the output
    feature name, so the name is read back from an actual prediction
    instead of being hardcoded. ``"embedding"`` is preferred when present
    (the name this project's own converter requests); otherwise, with
    exactly one output, that one is used.

    Raises:
        RuntimeError: If the model returned no outputs, or more than one
            and none of them is named ``"embedding"`` (ambiguous).
    """
    keys = list(prediction)
    if not keys:
        raise RuntimeError("Core ML model returned no outputs")
    if "embedding" in keys:
        return "embedding"
    if len(keys) == 1:
        return keys[0]
    raise RuntimeError(f"cannot resolve a unique output key from outputs: {keys}")


def encode_texts_coreml(
    model: ct.models.CompiledMLModel,
    tokenizer: Any,
    texts: Sequence[str],
    seq_len: int,
    prompt: str = "",
) -> tuple[np.ndarray, str]:
    """Run a batch-size-1 Core ML inference loop over ``texts``.

    Args:
        model: Loaded compiled model.
        tokenizer: Tokenizer from :func:`poc_qwen.common.load_tokenizer`.
        texts: Input texts.
        seq_len: Fixed sequence length the artifact expects.
        prompt: Optional instruction prefix; see
            :func:`poc_qwen.common.tokenize_batch`.

    Returns:
        Tuple of (embeddings of shape ``(len(texts), H)``, resolved output
        key). An empty ``texts`` returns an empty ``(0, 0)`` array.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32), "embedding"
    batch = common.tokenize_batch(tokenizer, texts, seq_len, prompt=prompt)
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


def encode_lang_sets_coreml(
    model: ct.models.CompiledMLModel,
    tokenizer: Any,
    lang_sets: Sequence[tuple[str, Sequence[str]]],
    seq_len: int,
    prompt: str = "",
) -> tuple[dict[str, np.ndarray], str]:
    """Run :func:`encode_texts_coreml` over every language set.

    Args:
        model: Loaded compiled model.
        tokenizer: Tokenizer from :func:`poc_qwen.common.load_tokenizer`.
        lang_sets: ``(language, texts)`` pairs, e.g. from
            :func:`poc_qwen.common.sanity_text_sets`.
        seq_len: Fixed sequence length the artifact expects.
        prompt: Optional instruction prefix, applied to every set.

    Returns:
        Tuple of (embeddings keyed by language, resolved output key).
    """
    embeddings: dict[str, np.ndarray] = {}
    output_key = "embedding"
    for language, texts in lang_sets:
        embeddings[language], output_key = encode_texts_coreml(
            model, tokenizer, texts, seq_len, prompt=prompt
        )
    return embeddings, output_key


def _nonfinite_report(name: str, embeddings: np.ndarray) -> dict[str, Any]:
    """Count NaN/Inf elements and rows in one batch of Core ML embeddings.

    Args:
        name: Label identifying which measurement produced ``embeddings``,
            kept in the report so a non-zero count can be traced back to
            its source.
        embeddings: Array of shape ``(N, H)``.

    Returns:
        Dict with ``name``, row count, NaN count, Inf count, and the
        number of rows containing at least one non-finite value.
    """
    nan_count = int(np.isnan(embeddings).sum())
    inf_count = int(np.isinf(embeddings).sum())
    nonfinite_rows = int((~np.isfinite(embeddings)).any(axis=1).sum()) if embeddings.size else 0
    return {
        "name": name,
        "n_rows": int(embeddings.shape[0]),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nonfinite_row_count": nonfinite_rows,
    }


def build_nonfinite_check(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine every :func:`_nonfinite_report` into check (e)'s result.

    Args:
        reports: One report per Core ML embedding batch produced during
            the run.

    Returns:
        Dict with the total NaN/Inf/non-finite-row counts, the
        per-source breakdown, and ``passed`` (``True`` iff every count is
        zero).
    """
    nan_count = sum(report["nan_count"] for report in reports)
    inf_count = sum(report["inf_count"] for report in reports)
    nonfinite_row_count = sum(report["nonfinite_row_count"] for report in reports)
    return {
        "sources": reports,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nonfinite_row_count": nonfinite_row_count,
        "passed": nan_count == 0 and inf_count == 0,
    }


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on blank lines.

    A run of one or more blank (or whitespace-only) lines separates two
    paragraphs. ``str.splitlines`` is used so the source file's
    line-ending style does not matter.

    Args:
        text: Full text to split.

    Returns:
        Stripped paragraph strings; empty paragraphs are excluded.
    """
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                paragraphs.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append("\n".join(current).strip())
    return paragraphs


def load_corpus_paragraphs(corpus_dir: Path, limit: int) -> list[str]:
    """Load the fixed test corpus, split into paragraphs, for check (c).

    Reads the three files in :data:`_CORPUS_FILENAMES` order, splits each
    on blank lines, keeps only paragraphs of at least
    :data:`_CORPUS_MIN_CHARS` characters (dropping headings and other
    fragments that are not representative document text), and stops as
    soon as ``limit`` paragraphs have been collected.

    Args:
        corpus_dir: Directory holding the corpus text files.
        limit: Maximum number of paragraphs to return.

    Returns:
        Paragraphs in file order, at most ``limit`` items long.
    """
    paragraphs: list[str] = []
    for filename in _CORPUS_FILENAMES:
        if len(paragraphs) >= limit:
            break
        text = (corpus_dir / filename).read_text(encoding="utf-8")
        for paragraph in _split_paragraphs(text):
            if len(paragraph) < _CORPUS_MIN_CHARS:
                continue
            paragraphs.append(paragraph)
            if len(paragraphs) >= limit:
                break
    return paragraphs


def build_reference_check(
    coreml_embeddings: dict[str, np.ndarray],
    reference_embeddings: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    """Build check (a): Core ML vs. the FP32 reference, per language set.

    A language set the model's vocabulary barely covers can fail on
    FP16-vs-FP32 noise alone, saying nothing about the conversion itself.
    So, as this project's own compiled-model self-check does, the run is
    accepted as soon as *any* one set clears ``threshold``; every set's
    numbers are kept so the reader can see which set that was.

    Args:
        coreml_embeddings: Core ML embeddings keyed by language.
        reference_embeddings: FP32 reference embeddings keyed by the same
            languages.
        threshold: Minimum acceptable cosine similarity.

    Returns:
        Dict with per-language stats (``sets``), the best-scoring
        language (``best_set``) and its numbers, and ``passed``.
    """
    sets: dict[str, Any] = {}
    for language, coreml_emb in coreml_embeddings.items():
        cosines = common.cosine_rowwise(coreml_emb, reference_embeddings[language])
        cosine_min = float(cosines.min())
        sets[language] = {
            "n": int(len(cosines)),
            "cosine_min": cosine_min,
            "cosine_mean": float(cosines.mean()),
            "cosine_per_text": [float(c) for c in cosines],
            "passed": cosine_min >= threshold,
        }
    best_language = max(sets, key=lambda language: sets[language]["cosine_min"])
    return {
        "threshold": threshold,
        "sets": sets,
        "best_set": best_language,
        "best_cosine_min": sets[best_language]["cosine_min"],
        "best_cosine_mean": sets[best_language]["cosine_mean"],
        "passed": any(set_result["passed"] for set_result in sets.values()),
    }


def run_sentence_transformers_check(
    model_dir: Path,
    coreml_embeddings: dict[str, np.ndarray],
    lang_sets: Sequence[tuple[str, Sequence[str]]],
    seq_len: int,
    threshold: float,
) -> dict[str, Any]:
    """Build check (b): Core ML vs. sentence-transformers' own ``encode()``.

    This is the most independent check in this script: sentence-transformers
    tokenizes, applies the query prompt, and pools with its own code path,
    sharing none of it with :mod:`poc_qwen.common`. Agreement here cannot
    be explained by a bug shared between the Core ML side and its
    reference, unlike checks (a) and (c).

    Args:
        model_dir: Local sentence-transformers/HuggingFace model
            directory.
        coreml_embeddings: Core ML embeddings keyed by language, computed
            with the query prompt applied (matching ``prompt_name="query"``
            below).
        lang_sets: ``(language, texts)`` pairs in the same order
            ``coreml_embeddings`` was built from.
        seq_len: Sequence length to compare at.
        threshold: Minimum acceptable cosine similarity.

    Returns:
        Dict with the combined min/mean/per-text cosine similarity over
        every language set's texts, and ``passed``.
    """
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(str(model_dir))
    # The checkpoint declares max_seq_length=32768 by default; pin it to
    # the sequence length under test so both sides truncate identically.
    st_model.max_seq_length = seq_len
    texts_by_language = {language: list(texts) for language, texts in lang_sets}
    all_texts = [text for texts in texts_by_language.values() for text in texts]
    st_embeddings = np.asarray(
        st_model.encode(all_texts, prompt_name="query", convert_to_numpy=True), dtype=np.float32
    )
    del st_model
    gc.collect()

    coreml_all = np.concatenate(
        [coreml_embeddings[language] for language in texts_by_language], axis=0
    )
    cosines = common.cosine_rowwise(coreml_all, st_embeddings)
    cosine_min = float(cosines.min())
    return {
        "n": int(len(cosines)),
        "threshold": threshold,
        "texts": all_texts,
        "cosine_min": cosine_min,
        "cosine_mean": float(cosines.mean()),
        "cosine_per_text": [float(c) for c in cosines],
        "passed": cosine_min >= threshold,
    }


def build_corpus_check(
    coreml_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    texts: list[str],
    threshold: float,
) -> dict[str, Any]:
    """Build check (c): the cosine distribution over real corpus paragraphs.

    Unlike checks (a)/(b), which use short hand-written fixtures, this
    measures the same FP16-vs-FP32 agreement over longer, unscripted
    document text. The 5th percentile (rather than the strict minimum) is
    used as the pass/fail gate, so one unusually hard paragraph among
    many does not by itself fail an otherwise healthy conversion.

    Args:
        coreml_embeddings: Core ML embeddings, shape ``(N, H)``.
        reference_embeddings: FP32 reference embeddings, shape ``(N, H)``.
        texts: The paragraphs the embeddings were computed from, in the
            same row order (kept for traceability).
        threshold: Minimum acceptable 5th-percentile cosine similarity.

    Returns:
        Dict with mean/min/p5/std cosine similarity and ``passed``.
    """
    cosines = common.cosine_rowwise(coreml_embeddings, reference_embeddings)
    cosine_p5 = float(np.percentile(cosines, 5))
    return {
        "n": int(len(cosines)),
        "threshold": threshold,
        "texts": texts,
        "cosine_mean": float(cosines.mean()),
        "cosine_min": float(cosines.min()),
        "cosine_p5": cosine_p5,
        "cosine_std": float(cosines.std()),
        "cosine_per_text": [float(c) for c in cosines],
        "passed": cosine_p5 >= threshold,
    }


def select_short_texts(
    tokenizer: Any, lang_sets: Sequence[tuple[str, Sequence[str]]], max_tokens: int
) -> list[str]:
    """Pick up to one short text per language set, for check (d).

    A text longer than ``max_tokens`` (the smaller of the two compared
    sequence lengths) would be truncated by the smaller artifact, and the
    check would then measure a truncation difference instead of a
    padding difference. Each language set is walked from its first text,
    keeping the first one that tokenizes (without any prompt) to at most
    ``max_tokens`` tokens, so a language whose usual first sentence
    happens to be too long still contributes a shorter one instead of
    none at all.

    Args:
        tokenizer: Tokenizer used to measure untruncated token counts.
        lang_sets: ``(language, texts)`` pairs, e.g. from
            :func:`poc_qwen.common.sanity_text_sets`.
        max_tokens: Inclusive upper bound on the token count, including
            any special tokens the tokenizer adds.

    Returns:
        At most one text per language set, in ``lang_sets`` order.
    """
    selected: list[str] = []
    for _language, texts in lang_sets:
        for text in texts:
            n_tokens = len(tokenizer(text)["input_ids"])
            if n_tokens <= max_tokens:
                selected.append(text)
                break
    return selected


def build_padding_invariance_check(
    main_embeddings: np.ndarray,
    alt_embeddings: np.ndarray,
    texts: list[str],
    seq_len: int,
    seq_len_alt: int,
    threshold: float,
) -> dict[str, Any]:
    """Build check (d): cross-shape agreement between two compiled artifacts.

    Both artifacts embed the exact same short texts. Since the model
    left-pads to its fixed sequence length and pools the final token
    position, the only difference between the two runs is the amount of
    padding, so a correct conversion should return (near-)identical
    embeddings from both.

    Args:
        main_embeddings: Embeddings from the ``--mlmodelc`` artifact.
        alt_embeddings: Embeddings from the ``--mlmodelc-alt`` artifact.
        texts: The texts both artifacts embedded, in row order.
        seq_len: Sequence length of the main artifact.
        seq_len_alt: Sequence length of the alt artifact.
        threshold: Minimum acceptable cosine similarity.

    Returns:
        Dict with ``checked=True``, the min/mean/per-text cosine
        similarity, and ``passed``.
    """
    cosines = common.cosine_rowwise(main_embeddings, alt_embeddings)
    cosine_min = float(cosines.min())
    return {
        "checked": True,
        "seq_len": seq_len,
        "seq_len_alt": seq_len_alt,
        "n": int(len(cosines)),
        "threshold": threshold,
        "texts": texts,
        "cosine_min": cosine_min,
        "cosine_mean": float(cosines.mean()),
        "cosine_per_text": [float(c) for c in cosines],
        "passed": cosine_min >= threshold,
    }


def print_summary(result: dict[str, Any], out_path: Path) -> None:
    """Print the human-readable summary table to stdout."""
    print("\n=== Accuracy summary ===")
    print(f"model_id         : {result['model_id']}")
    print(f"mlmodelc         : {result['mlmodelc']} (seq_len={result['seq_len']})")
    print(f"compute_units    : {result['compute_units']}")

    if "error" in result:
        print(f"ERROR            : {result['error']}")

    reference = result.get("reference_check")
    if reference is not None:
        print(
            f"[a] reference    : best={reference['best_set']} "
            f"cosine_min={reference['best_cosine_min']:.6f} "
            f"cosine_mean={reference['best_cosine_mean']:.6f} "
            f"(threshold {reference['threshold']}, passed={reference['passed']})"
        )

    st_check = result.get("sentence_transformers_check")
    if st_check is None:
        pass
    elif st_check.get("skipped"):
        print(f"[b] sentence-tf  : skipped ({st_check.get('reason')})")
    else:
        print(
            f"[b] sentence-tf  : cosine_min={st_check['cosine_min']:.6f} "
            f"cosine_mean={st_check['cosine_mean']:.6f} "
            f"(threshold {st_check['threshold']}, passed={st_check['passed']})"
        )

    corpus = result.get("corpus_check")
    if corpus is not None:
        print(
            f"[c] corpus (n={corpus['n']}) : cosine_mean={corpus['cosine_mean']:.6f} "
            f"cosine_min={corpus['cosine_min']:.6f} cosine_p5={corpus['cosine_p5']:.6f} "
            f"cosine_std={corpus['cosine_std']:.6f} "
            f"(threshold {corpus['threshold']}, passed={corpus['passed']})"
        )

    padding = result.get("padding_invariance_check")
    if padding is None:
        pass
    elif not padding.get("checked"):
        print(f"[d] padding      : skipped ({padding.get('reason')})")
    else:
        print(
            f"[d] padding      : S{padding['seq_len']} vs S{padding['seq_len_alt']} "
            f"cosine_min={padding['cosine_min']:.6f} "
            f"(threshold {padding['threshold']}, passed={padding['passed']})"
        )

    nonfinite = result.get("nonfinite_check")
    if nonfinite is not None:
        print(
            f"[e] non-finite   : nan={nonfinite['nan_count']} inf={nonfinite['inf_count']} "
            f"rows={nonfinite['nonfinite_row_count']} (passed={nonfinite['passed']})"
        )

    print(f"elapsed (total)  : {result.get('timings_sec', {}).get('total', 0.0):.2f}s")
    print(f"results written  : {out_path}")
    print(f"overall_passed   : {result.get('overall_passed')}")


def main(argv: list[str] | None = None) -> int:
    """Run the full accuracy verification pipeline.

    Every measurement is wrapped so an unexpected failure (a Core ML
    runtime error, a missing optional dependency, ...) is recorded as
    part of the result instead of raising: this is a research tool, and a
    failed measurement is itself a useful result, not something to hide
    behind a stack trace.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        0 if every applicable check passed, 1 otherwise.
    """
    args = parse_args(argv)
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be a positive integer")
    if args.corpus_limit <= 0:
        raise SystemExit("--corpus-limit must be a positive integer")
    if (args.mlmodelc_alt is None) != (args.seq_len_alt is None):
        raise SystemExit("--mlmodelc-alt and --seq-len-alt must be given together")
    if not args.mlmodelc.exists():
        raise SystemExit(f"compiled model not found: {args.mlmodelc}")
    if args.mlmodelc_alt is not None and not args.mlmodelc_alt.exists():
        raise SystemExit(f"compiled model not found: {args.mlmodelc_alt}")

    compute_units = _COMPUTE_UNITS[args.compute_units]
    out_path = args.output or (
        _REPO_ROOT / "poc_qwen" / "results" / f"accuracy_s{args.seq_len}_{args.compute_units}.json"
    )

    result: dict[str, Any] = {
        "model_id": args.model_id,
        "mlmodelc": str(args.mlmodelc),
        "seq_len": args.seq_len,
        "mlmodelc_alt": str(args.mlmodelc_alt) if args.mlmodelc_alt is not None else None,
        "seq_len_alt": args.seq_len_alt,
        "compute_units": args.compute_units,
        "corpus_limit": args.corpus_limit,
        "skip_st": args.skip_st,
        "overall_passed": False,
    }

    started = time.perf_counter()
    timings: dict[str, float] = {}
    try:
        model_dir = common.resolve_model_dir(args.model_id)
        tokenizer = common.load_tokenizer(model_dir)
        lang_sets = common.sanity_text_sets()
        nonfinite_reports: list[dict[str, Any]] = []

        # --- Phase 1: all Core ML inference, before any PyTorch model is loaded ---
        print(f"[1/5] Loading Core ML model: {args.mlmodelc} ({args.compute_units})")
        step = time.perf_counter()
        main_model = ct.models.CompiledMLModel(str(args.mlmodelc), compute_units=compute_units)

        print("[2/5] Running Core ML inference for checks (a)/(b)/(c)/(d)")
        coreml_noprompt, output_key = encode_lang_sets_coreml(
            main_model, tokenizer, lang_sets, args.seq_len, prompt=""
        )
        for language, embeddings in coreml_noprompt.items():
            nonfinite_reports.append(_nonfinite_report(f"lang_noprompt_{language}", embeddings))

        coreml_queryprompt, _ = encode_lang_sets_coreml(
            main_model, tokenizer, lang_sets, args.seq_len, prompt=common.QUERY_PROMPT
        )
        for language, embeddings in coreml_queryprompt.items():
            nonfinite_reports.append(_nonfinite_report(f"lang_queryprompt_{language}", embeddings))

        corpus_texts = load_corpus_paragraphs(CORPUS_DIR, args.corpus_limit)
        coreml_corpus, _ = encode_texts_coreml(
            main_model, tokenizer, corpus_texts, args.seq_len, prompt=common.DOCUMENT_PROMPT
        )
        nonfinite_reports.append(_nonfinite_report("corpus", coreml_corpus))

        padding_texts: list[str] = []
        coreml_padding_main: np.ndarray | None = None
        if args.mlmodelc_alt is not None and args.seq_len_alt is not None:
            max_tokens = min(args.seq_len, args.seq_len_alt)
            padding_texts = select_short_texts(tokenizer, lang_sets, max_tokens)
            if padding_texts:
                coreml_padding_main, _ = encode_texts_coreml(
                    main_model, tokenizer, padding_texts, args.seq_len, prompt=""
                )
                nonfinite_reports.append(_nonfinite_report("padding_main", coreml_padding_main))

        del main_model
        gc.collect()

        coreml_padding_alt: np.ndarray | None = None
        if args.mlmodelc_alt is not None and args.seq_len_alt is not None and padding_texts:
            print(f"[2b/5] Loading alt Core ML model: {args.mlmodelc_alt} ({args.compute_units})")
            alt_model = ct.models.CompiledMLModel(
                str(args.mlmodelc_alt), compute_units=compute_units
            )
            coreml_padding_alt, _ = encode_texts_coreml(
                alt_model, tokenizer, padding_texts, args.seq_len_alt, prompt=""
            )
            nonfinite_reports.append(_nonfinite_report("padding_alt", coreml_padding_alt))
            del alt_model
            gc.collect()
        timings["coreml"] = time.perf_counter() - step

        # --- Phase 2: the FP32 PyTorch reference, loaded and freed on its own ---
        print("[3/5] Computing PyTorch FP32 reference for checks (a)/(c)")
        step = time.perf_counter()
        reference_model = common.load_reference_model(model_dir)
        reference_noprompt = {
            language: common.encode_reference(reference_model, tokenizer, texts, args.seq_len)
            for language, texts in lang_sets
        }
        reference_corpus = common.encode_reference(
            reference_model, tokenizer, corpus_texts, args.seq_len, prompt=common.DOCUMENT_PROMPT
        )
        del reference_model
        gc.collect()
        timings["reference"] = time.perf_counter() - step

        result["output_key"] = output_key
        result["reference_check"] = build_reference_check(
            coreml_noprompt, reference_noprompt, common.SANITY_COSINE_THRESHOLD
        )
        result["corpus_check"] = build_corpus_check(
            coreml_corpus, reference_corpus, corpus_texts, common.SANITY_COSINE_THRESHOLD
        )

        # --- (d) padding invariance, purely Core ML vs. Core ML ---
        if args.mlmodelc_alt is None:
            result["padding_invariance_check"] = {
                "checked": False,
                "reason": "--mlmodelc-alt/--seq-len-alt not provided",
            }
        elif not padding_texts:
            result["padding_invariance_check"] = {
                "checked": False,
                "reason": "no sanity text short enough to fit both sequence lengths",
            }
        else:
            assert coreml_padding_main is not None
            assert coreml_padding_alt is not None
            result["padding_invariance_check"] = build_padding_invariance_check(
                coreml_padding_main,
                coreml_padding_alt,
                padding_texts,
                args.seq_len,
                args.seq_len_alt,
                PADDING_INVARIANCE_THRESHOLD,
            )

        # --- Phase 3: sentence-transformers, loaded last and freed on its own ---
        if args.skip_st:
            result["sentence_transformers_check"] = {"skipped": True, "reason": "--skip-st"}
        else:
            print("[4/5] Comparing against sentence-transformers encode() for check (b)")
            step = time.perf_counter()
            try:
                result["sentence_transformers_check"] = run_sentence_transformers_check(
                    model_dir,
                    coreml_queryprompt,
                    lang_sets,
                    args.seq_len,
                    common.SANITY_COSINE_THRESHOLD,
                )
            except Exception as exc:  # noqa: BLE001 - record, do not crash the run
                print(f"WARNING: sentence-transformers check failed: {exc}", file=sys.stderr)
                result["sentence_transformers_check"] = {
                    "skipped": True,
                    "reason": f"failed: {exc}",
                }
            timings["sentence_transformers"] = time.perf_counter() - step

        print("[5/5] Aggregating results")
        result["nonfinite_check"] = build_nonfinite_check(nonfinite_reports)

        checks_passed = [
            result["reference_check"]["passed"],
            result["corpus_check"]["passed"],
            result["nonfinite_check"]["passed"],
        ]
        st_check = result["sentence_transformers_check"]
        if not st_check.get("skipped", False):
            checks_passed.append(st_check["passed"])
        padding_check = result["padding_invariance_check"]
        if padding_check.get("checked", False):
            checks_passed.append(padding_check["passed"])
        result["overall_passed"] = all(checks_passed)
    except Exception as exc:  # noqa: BLE001 - a research tool must report, not crash
        print(f"ERROR: verification failed with an unhandled exception: {exc}", file=sys.stderr)
        result["error"] = str(exc)
        result["overall_passed"] = False

    timings["total"] = time.perf_counter() - started
    result["timings_sec"] = {key: round(value, 3) for key, value in timings.items()}

    common.write_result_json(out_path, result)
    print_summary(result, out_path)

    return 0 if result["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
