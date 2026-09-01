"""Post-compile self-check: accuracy sanity, ANE placement, latency.

Plugs into :data:`eeane.compiler.pipeline.SelfcheckFn`: it is handed one
just-compiled ``.mlmodelc`` (as a :class:`~eeane.compiler.pipeline.
SelfcheckContext`) and returns a JSON-serializable report. It never raises
-- every failure mode, expected or not, is turned into the report's
``status``/``error`` keys instead, per the pipeline's hook contract.

Three checks run in order, in the same process that just compiled the
variant (the model stays warm on disk, no reload of anything but the
compiled artifact itself):

1. **Accuracy sanity** (:func:`_run_sanity`): the compiled model is loaded
   on ``CPU_AND_NE`` and compared against an FP32 (``sdpa``) reference,
   using the thresholds proven by ``poc/convert_embedding.py``/
   ``poc/convert_reranker.py``. The backend's fixtures come as one set
   per language and each set is scored on its own, because a set in a
   language the model has no vocabulary for encodes to little more than
   unknown-token rows -- whose FP16-vs-FP32 difference says nothing about
   the model, yet can still miss the threshold. The variant therefore
   passes as soon as *any* set does, and the report records every set's
   numbers plus the one it was accepted (or, having failed, best
   described) on. A failure here -- no set clearing the threshold, or a
   batch whose rows influence each other -- makes the whole report
   ``status="failed"``; this is the only check the pipeline reacts to.
2. **NE placement** (:func:`_compute_plan_report`): per-op device
   placement via ``MLComputePlan``, ported from
   ``poc/benchmark_latency.py``. Below
   :data:`NE_PLACEMENT_WARN_THRESHOLD` degrades the report to
   ``status="warned"`` but never fails the compile; when the API is
   unavailable the failure is recorded, not raised, and does not change
   the status either way.
3. **Warm latency** (:func:`_measure_latency`): a simplified, informative
   only measurement (poc/benchmark_latency.py remains the tool for actual
   benchmarking) reusing the sanity check's already-tokenized input.

The tokenization for (1) and (3) goes through the frozen tokenizer
(:mod:`eeane.runtime`) rather than ``AutoTokenizer``, unlike the poc
scripts: this is what proves the served request path (which only ever
sees the frozen tokenizer) produces the token sequence being measured.
"""

from __future__ import annotations

import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import coremltools as ct
import numpy as np

from eeane import runtime
from eeane.compiler import conversion

if TYPE_CHECKING:
    from eeane.compiler.pipeline import SelfcheckContext

# Statuses this module can produce. "skipped" is the pipeline's own status
# for when no self-check ran at all; it is never returned from here.
STATUS_PASSED = "passed"
STATUS_WARNED = "warned"
STATUS_FAILED = "failed"

# --- accuracy sanity thresholds, ported verbatim from
# poc/convert_embedding.py / poc/convert_reranker.py ---

# Minimum cosine similarity against the PyTorch FP32 baseline (embedding).
SANITY_COSINE_THRESHOLD = 0.99

# Maximum tolerated |sigmoid(coreml) - sigmoid(fp32)| (reranker).
SANITY_SIGMOID_TOLERANCE = 0.02

# Minimum cosine similarity between rows of a batch holding the same text
# (embedding batch consistency check).
BATCH_CONSISTENCY_COSINE_THRESHOLD = 0.99999

# Maximum tolerated |logit(row k) - logit(row 0)| between rows of a batch
# holding the same pair (reranker batch consistency check).
BATCH_CONSISTENCY_LOGIT_TOLERANCE = 0.01

# --- NE placement ---

# Below this NE-placement percentage the report is downgraded to "warned"
# (never "failed": the goal is data collection on unverified machines).
NE_PLACEMENT_WARN_THRESHOLD = 90.0

# --- warm latency (a simplified measurement) ---

LATENCY_WARMUP_PREDICTS = 3
LATENCY_TIMED_PREDICTS = 20


def run_selfcheck(context: SelfcheckContext) -> dict[str, Any]:
    """Run the accuracy/ANE-placement/latency self-check for one variant.

    Args:
        context: Everything needed to measure one compiled variant (see
            :class:`eeane.compiler.pipeline.SelfcheckContext`).

    Returns:
        A JSON-serializable report with ``status`` (``"passed"``,
        ``"warned"`` or ``"failed"``), ``sanity``, ``compute_plan``,
        ``latency`` and ``machine`` keys. On an unexpected internal
        failure, ``status`` is ``"failed"`` and an ``error`` key is added
        instead of the three measurement keys (only ``machine`` is still
        recorded).
    """
    machine = _machine_info()
    try:
        compiled = ct.models.CompiledMLModel(
            str(context.mlmodelc_path), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        frozen_tokenizer = runtime.load_frozen_tokenizer(context.tokenizer_path)
        sanity, latency_batch = _run_sanity(context, compiled, frozen_tokenizer)
        compute_plan = _compute_plan_report(context.mlmodelc_path)
        latency = _measure_latency(compiled, latency_batch)
    except Exception as exc:  # noqa: BLE001 - the hook contract forbids raising
        report = {"status": STATUS_FAILED, "error": str(exc), "machine": machine}
        _print_summary(context, report)
        return report

    report = {
        "status": _derive_status(sanity, compute_plan),
        "sanity": sanity,
        "compute_plan": compute_plan,
        "latency": latency,
        "machine": machine,
    }
    _print_summary(context, report)
    return report


def _derive_status(sanity: dict[str, Any], compute_plan: dict[str, Any]) -> str:
    """Combine the sanity and NE-placement results into the overall status.

    Args:
        sanity: Report from :func:`_run_sanity`.
        compute_plan: Report from :func:`_compute_plan_report`.

    Returns:
        ``"failed"`` if the sanity check did not pass; otherwise
        ``"warned"`` if NE placement was measured and is below
        :data:`NE_PLACEMENT_WARN_THRESHOLD`; otherwise ``"passed"``
        (including when NE placement could not be measured at all).
    """
    if not sanity.get("passed", False):
        return STATUS_FAILED
    ne_placement_pct = compute_plan.get("ne_placement_pct")
    if isinstance(ne_placement_pct, int | float) and ne_placement_pct < NE_PLACEMENT_WARN_THRESHOLD:
        return STATUS_WARNED
    return STATUS_PASSED


# --- accuracy sanity ----------------------------------------------------------


def _run_sanity(
    context: SelfcheckContext, compiled: ct.models.CompiledMLModel, frozen_tokenizer: Any
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Dispatch the accuracy sanity check to the embedding/reranker variant.

    Args:
        context: Self-check context.
        compiled: The just-loaded compiled model (``CPU_AND_NE``).
        frozen_tokenizer: Tokenizer loaded via
            :func:`eeane.runtime.load_frozen_tokenizer`.

    Returns:
        Tuple of the sanity report and one batch-size-sized, already
        tokenized predict() input -- reused by :func:`_measure_latency` so
        the warm-latency measurement exercises the same tokens the sanity
        check just verified.
    """
    if context.kind == "reranker":
        return _sanity_reranker(context, compiled, frozen_tokenizer)
    return _sanity_embedding(context, compiled, frozen_tokenizer)


def _sanity_embedding(
    context: SelfcheckContext, compiled: ct.models.CompiledMLModel, frozen_tokenizer: Any
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Accuracy sanity check for an embedding variant (ported from poc/convert_embedding.py).

    The fixtures arrive as one set per language and the variant passes as
    soon as any set clears the threshold (see :func:`_best_sanity_set`).
    Every set's texts are predicted and referenced in one pass -- the sets
    are concatenated, then split again by row -- so offering more of them
    costs one more row each rather than another FP32 model load.
    """
    spec = context.backend.sanity_spec(context.kind)
    raw_texts: list[str] = list(spec.all_inputs)
    tokens = runtime.tokenize_texts(frozen_tokenizer, raw_texts, context.seq_len)
    padding = runtime.tokenize_texts(
        frozen_tokenizer, [context.backend.padding_input(context.kind)], context.seq_len
    )
    output_key, coreml_emb = _predict_rows(compiled, tokens, padding, context.batch_size, context)
    baseline_emb = context.backend.reference_outputs(
        context.model_dir, context.kind, raw_texts, context.seq_len
    )
    cosines = _cosine_rowwise(coreml_emb, baseline_emb)

    rows_per_set = _set_row_slices(spec.input_sets)
    set_reports: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for language, rows in rows_per_set.items():
        set_cosines = cosines[rows]
        cosine_min = float(set_cosines.min())
        finite = bool(np.isfinite(coreml_emb[rows]).all())
        set_reports[language] = {
            "cosine_per_text": [float(value) for value in set_cosines],
            "cosine_min": cosine_min,
            "cosine_mean": float(set_cosines.mean()),
            "finite": finite,
            "passed": finite and bool(cosine_min >= SANITY_COSINE_THRESHOLD),
        }
        scores[language] = _set_score(cosine_min, finite)

    best = _best_sanity_set(set_reports, scores)
    best_rows = rows_per_set[best]

    consistency = None
    if context.batch_size > 1:
        consistency = _check_batch_consistency_embedding(
            compiled,
            frozen_tokenizer,
            raw_texts[best_rows.start],
            context.seq_len,
            context.batch_size,
            output_key,
        )

    best_report = set_reports[best]
    sanity = {
        "output_key": output_key,
        "batch_size": context.batch_size,
        "embedding_dim": int(coreml_emb.shape[1]),
        # The established keys carry the best set's numbers, so a reader
        # that predates the per-set sets sees the measurement the variant
        # was actually accepted (or rejected) on.
        "cosine_per_text": best_report["cosine_per_text"],
        "cosine_min": best_report["cosine_min"],
        "cosine_mean": best_report["cosine_mean"],
        "cosine_threshold": SANITY_COSINE_THRESHOLD,
        "batch_consistency": consistency,
        "finite": best_report["finite"],
        "sets": set_reports,
        "best_set": best,
        "passed": (
            any(report["passed"] for report in set_reports.values())
            and (consistency is None or bool(consistency["passed"]))
        ),
    }
    best_tokens = _slice_tokens(tokens, best_rows)
    return sanity, _first_predict_batch(best_tokens, padding, context.batch_size)


def _sanity_reranker(
    context: SelfcheckContext, compiled: ct.models.CompiledMLModel, frozen_tokenizer: Any
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Accuracy sanity check for a reranker variant (ported from poc/convert_reranker.py).

    Like the embedding check, the fixtures arrive as one set per language,
    every set is predicted and referenced in one pass, and the variant
    passes as soon as any set does.

    The expected ordering comes from the backend's sanity specification
    and is checked *within* each set, whose pairs are built in the same
    roles; when the spec declares no relevant/irrelevant pair, the
    ordering check is skipped and the report says so (``ordering_checked``)
    instead of silently passing an unperformed check.
    """
    spec = context.backend.sanity_spec(context.kind)
    raw_pairs: list[tuple[str, str]] = [(query, document) for query, document in spec.all_inputs]
    tokens = runtime.tokenize_pairs(frozen_tokenizer, raw_pairs, context.seq_len)
    padding = runtime.tokenize_pairs(
        frozen_tokenizer, [context.backend.padding_input(context.kind)], context.seq_len
    )
    output_key, coreml_out = _predict_rows(compiled, tokens, padding, context.batch_size, context)
    coreml_logits = coreml_out.reshape(-1)
    fp32_logits = context.backend.reference_outputs(
        context.model_dir, context.kind, raw_pairs, context.seq_len
    )
    coreml_scores = runtime.sigmoid(coreml_logits)
    fp32_scores = runtime.sigmoid(fp32_logits)
    abs_diff = np.abs(coreml_scores - fp32_scores)
    ordering_checked = spec.relevant_index is not None and spec.irrelevant_index is not None

    rows_per_set = _set_row_slices(spec.input_sets)
    set_reports: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    for language, rows in rows_per_set.items():
        max_abs_diff = float(abs_diff[rows].max())
        finite = bool(
            np.isfinite(coreml_logits[rows]).all() and np.isfinite(fp32_logits[rows]).all()
        )
        ordering_coreml = _check_ordering(coreml_scores[rows], spec)
        ordering_fp32 = _check_ordering(fp32_scores[rows], spec)
        set_reports[language] = {
            "coreml_logits": [float(value) for value in coreml_logits[rows]],
            "fp32_logits": [float(value) for value in fp32_logits[rows]],
            "coreml_scores": [float(value) for value in coreml_scores[rows]],
            "fp32_scores": [float(value) for value in fp32_scores[rows]],
            "sigmoid_abs_diff": [float(value) for value in abs_diff[rows]],
            "sigmoid_max_abs_diff": max_abs_diff,
            "ordering_checked": ordering_checked,
            "ordering_ok_coreml": ordering_coreml,
            "ordering_ok_fp32": ordering_fp32,
            "finite": finite,
            "passed": (
                finite
                and max_abs_diff <= SANITY_SIGMOID_TOLERANCE
                and (not ordering_checked or (bool(ordering_coreml) and bool(ordering_fp32)))
            ),
        }
        # Negated: the closer to the baseline, the better the set, whereas
        # _best_sanity_set always keeps the highest score.
        scores[language] = _set_score(-max_abs_diff, finite)

    best = _best_sanity_set(set_reports, scores)
    best_rows = rows_per_set[best]

    consistency = None
    if context.batch_size > 1:
        consistency = _check_batch_consistency_reranker(
            compiled,
            frozen_tokenizer,
            raw_pairs[best_rows.start],
            context.seq_len,
            context.batch_size,
            output_key,
        )

    best_report = set_reports[best]
    sanity = {
        "output_key": output_key,
        "batch_size": context.batch_size,
        # As for an embedding variant: the established keys describe the
        # best set, the per-set table below describes all of them.
        "coreml_logits": best_report["coreml_logits"],
        "fp32_logits": best_report["fp32_logits"],
        "coreml_scores": best_report["coreml_scores"],
        "fp32_scores": best_report["fp32_scores"],
        "sigmoid_abs_diff": best_report["sigmoid_abs_diff"],
        "sigmoid_max_abs_diff": best_report["sigmoid_max_abs_diff"],
        "ordering_checked": best_report["ordering_checked"],
        "ordering_ok_coreml": best_report["ordering_ok_coreml"],
        "ordering_ok_fp32": best_report["ordering_ok_fp32"],
        "finite": best_report["finite"],
        "sigmoid_tolerance": SANITY_SIGMOID_TOLERANCE,
        "batch_consistency": consistency,
        "sets": set_reports,
        "best_set": best,
        "passed": (
            any(report["passed"] for report in set_reports.values())
            and (consistency is None or bool(consistency["passed"]))
        ),
    }
    best_tokens = _slice_tokens(tokens, best_rows)
    return sanity, _first_predict_batch(best_tokens, padding, context.batch_size)


def _set_row_slices(input_sets: Sequence[tuple[str, Sequence[Any]]]) -> dict[str, slice]:
    """Map every language set to the rows it occupies in the concatenated inputs.

    The sets are predicted (and referenced) as one flat sequence of rows,
    so each of them has to be found again by position afterwards.

    Args:
        input_sets: A :class:`eeane.compiler.backends.base.SanitySpec`'s
            ordered ``(language, inputs)`` pairs.

    Returns:
        Language -> row slice, in the order the sets were declared in.
    """
    slices: dict[str, slice] = {}
    start = 0
    for language, inputs in input_sets:
        end = start + len(inputs)
        slices[language] = slice(start, end)
        start = end
    return slices


def _set_score(metric: float, finite: bool) -> float:
    """Rank one language set by its headline metric, higher being better.

    Args:
        metric: The set's metric, already oriented so that a larger value
            is a better result.
        finite: Whether the set's raw outputs were finite.

    Returns:
        ``metric``, or negative infinity when the set produced a
        non-finite output or a metric that is not a number: such a set
        must never be reported as the best one, whatever the others did.
    """
    return metric if finite and np.isfinite(metric) else float("-inf")


def _best_sanity_set(set_reports: dict[str, dict[str, Any]], scores: dict[str, float]) -> str:
    """Pick the language set a sanity report is summarized by.

    A model is only measurable on fixtures its vocabulary covers, so the
    self-check evaluates one set per language and accepts the variant as
    soon as one of them passes. This picks which of them the report's
    established keys describe: a passing set whenever there is one (the
    reason the variant was accepted), else the closest set (the most
    informative account of why it was not).

    Args:
        set_reports: Per-language report, in declaration order.
        scores: Per-language rank from :func:`_set_score`.

    Returns:
        The chosen language. Two equally good sets resolve to the one
        declared first, so a rerun of the same variant reports the same
        set.
    """
    passing = [language for language, report in set_reports.items() if report["passed"]]
    candidates = passing or list(set_reports)
    return max(candidates, key=lambda language: scores[language])


def _slice_tokens(tokens: dict[str, np.ndarray], rows: slice) -> dict[str, np.ndarray]:
    """Keep only ``rows`` of a tokenized batch.

    Args:
        tokens: Tokenized rows, each value of shape (N, S).
        rows: Rows to keep.

    Returns:
        Dict with the same keys, each value of shape (len(rows), S).
    """
    return {key: value[rows] for key, value in tokens.items()}


def _check_ordering(scores: np.ndarray, spec: Any) -> bool | None:
    """Report whether the relevant input scored above the irrelevant one.

    Args:
        scores: One score per input of *one* language set, in that set's
            own order. The spec's indices address every set alike, since
            each of them puts its pairs in the same roles.
        spec: The backend's sanity specification
            (:class:`eeane.compiler.backends.base.SanitySpec`).

    Returns:
        ``True``/``False`` when the spec declares both the relevant and
        the irrelevant index, ``None`` when it declares no expected
        ordering (nothing to check).
    """
    relevant, irrelevant = spec.relevant_index, spec.irrelevant_index
    if relevant is None or irrelevant is None:
        return None
    return bool(scores[relevant] > scores[irrelevant])


def _fill_batch(rows: np.ndarray, padding_row: np.ndarray, batch_size: int) -> np.ndarray:
    """Pad a partial input chunk up to ``batch_size`` rows.

    Ported from ``poc/convert_embedding.py``/``poc/convert_reranker.py``
    (identical in both).

    Args:
        rows: Chunk of tokenized rows, shape (n, S) with ``n <= batch_size``.
        padding_row: Filler row, shape (1, S).
        batch_size: Fixed batch size B expected by the model.

    Returns:
        Array of shape (batch_size, S); ``rows`` itself when already full.
    """
    missing = batch_size - rows.shape[0]
    if missing <= 0:
        return rows
    return np.concatenate([rows, np.repeat(padding_row, missing, axis=0)], axis=0)


def _first_predict_batch(
    tokens: dict[str, np.ndarray], padding: dict[str, np.ndarray], batch_size: int
) -> dict[str, np.ndarray]:
    """Build the first batch-size-sized predict() input (the first chunk of :func:`_predict_rows`).

    Returned so :func:`_measure_latency` can reuse it instead of
    re-tokenizing an input.

    Args:
        tokens: Tokenized sanity rows.
        padding: Tokenized filler row, shape (1, S).
        batch_size: Fixed batch size B of the model.

    Returns:
        Dict with ``input_ids``/``attention_mask`` of shape (batch_size, S).
    """
    end = min(batch_size, int(tokens["input_ids"].shape[0]))
    return {
        key: _fill_batch(tokens[key][:end], padding[key], batch_size)
        for key in ("input_ids", "attention_mask")
    }


def _predict_rows(
    compiled: ct.models.CompiledMLModel,
    tokens: dict[str, np.ndarray],
    padding: dict[str, np.ndarray],
    batch_size: int,
    context: SelfcheckContext,
) -> tuple[str, np.ndarray]:
    """Run every tokenized row through a model whose batch size is fixed to B.

    Ported from ``poc/convert_embedding.py``/``poc/convert_reranker.py``'s
    identical ``_predict_rows`` helper: rows are grouped into chunks of
    ``batch_size``; a trailing partial chunk is padded with ``padding``
    rows whose outputs are dropped. Unlike the poc originals, the per-row
    output is always reshaped to ``(batch_size, -1)`` regardless of kind
    (a reranker's ``(B, 1)`` logits included), so the two kinds share one
    implementation; callers needing a flat logits vector reshape the
    stacked result themselves (see :func:`_sanity_reranker`).

    Args:
        compiled: Loaded compiled model.
        tokens: Tokenized rows, each value of shape (N, S).
        padding: Tokenized filler row, each value of shape (1, S).
        batch_size: Fixed batch size B of the model.
        context: Self-check context (only ``output_name`` is used).

    Returns:
        Tuple of the resolved output key and the stacked per-row outputs,
        shape (N, output_dim).

    Raises:
        ValueError: If ``tokens`` holds no rows.
    """
    n_rows = int(tokens["input_ids"].shape[0])
    if n_rows == 0:
        raise ValueError("no rows to predict")
    output_key = ""
    rows: list[np.ndarray] = []
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        prediction = compiled.predict(
            {
                key: _fill_batch(tokens[key][start:end], padding[key], batch_size)
                for key in ("input_ids", "attention_mask")
            }
        )
        output_key = output_key or conversion.resolve_output_key(prediction, context.output_name)
        values = np.asarray(prediction[output_key], dtype=np.float32).reshape(batch_size, -1)
        rows.extend(values[: end - start])
    return output_key, np.stack(rows)


def _check_batch_consistency_embedding(
    compiled: ct.models.CompiledMLModel,
    frozen_tokenizer: Any,
    text: str,
    seq_len: int,
    batch_size: int,
    output_key: str,
) -> dict[str, Any]:
    """Verify that rows of one embedding batch do not influence each other.

    Ported from ``poc/convert_embedding.py``'s ``check_batch_consistency``.

    Args:
        compiled: Loaded compiled model.
        frozen_tokenizer: Tokenizer used to encode ``text``.
        text: Sanity text replicated to every row of the batch.
        seq_len: Fixed sequence length S.
        batch_size: Fixed batch size B of the model (must be > 1).
        output_key: Output key resolved by :func:`_predict_rows`.

    Returns:
        Dict with the per-row cosine similarities against row 0, their
        min/max, the threshold, and the pass/fail flag.
    """
    tokens = runtime.tokenize_texts(frozen_tokenizer, [text] * batch_size, seq_len)
    prediction = compiled.predict(
        {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}
    )
    embeddings = np.asarray(prediction[output_key], dtype=np.float32).reshape(batch_size, -1)
    reference = np.repeat(embeddings[:1], batch_size - 1, axis=0)
    cosines = _cosine_rowwise(reference, embeddings[1:])
    return {
        "text_index": 0,
        "cosine_per_row": [float(c) for c in cosines],
        "cosine_min": float(cosines.min()),
        "cosine_max": float(cosines.max()),
        "cosine_threshold": BATCH_CONSISTENCY_COSINE_THRESHOLD,
        "passed": bool(np.isfinite(cosines).all())
        and bool(cosines.min() >= BATCH_CONSISTENCY_COSINE_THRESHOLD),
    }


def _check_batch_consistency_reranker(
    compiled: ct.models.CompiledMLModel,
    frozen_tokenizer: Any,
    pair: tuple[str, str],
    seq_len: int,
    batch_size: int,
    output_key: str,
) -> dict[str, Any]:
    """Verify that rows of one reranker batch do not influence each other.

    Ported from ``poc/convert_reranker.py``'s ``check_batch_consistency``.

    Args:
        compiled: Loaded compiled model.
        frozen_tokenizer: Tokenizer used to encode ``pair``.
        pair: Sanity (query, document) pair replicated to every row.
        seq_len: Fixed sequence length S.
        batch_size: Fixed batch size B of the model (must be > 1).
        output_key: Output key resolved by :func:`_predict_rows`.

    Returns:
        Dict with the per-row absolute logit differences against row 0,
        their min/max, the tolerance, and the pass/fail flag.
    """
    tokens = runtime.tokenize_pairs(frozen_tokenizer, [pair] * batch_size, seq_len)
    prediction = compiled.predict(
        {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}
    )
    logits = np.asarray(prediction[output_key], dtype=np.float32).reshape(-1)
    abs_diff = np.abs(logits[1:] - logits[0])
    return {
        "pair_index": 0,
        "logit_per_row": [float(v) for v in logits],
        "logit_abs_diff_per_row": [float(v) for v in abs_diff],
        "logit_abs_diff_min": float(abs_diff.min()),
        "logit_abs_diff_max": float(abs_diff.max()),
        "logit_tolerance": BATCH_CONSISTENCY_LOGIT_TOLERANCE,
        "passed": bool(np.isfinite(logits).all())
        and bool(abs_diff.max() <= BATCH_CONSISTENCY_LOGIT_TOLERANCE),
    }


def _cosine_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two equally-shaped arrays.

    Ported from ``poc/common.py``'s ``cosine_rowwise``.

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


# --- NE placement ---------------------------------------------------------------


def _compute_plan_report(mlmodelc_path: Any) -> dict[str, Any]:
    """Summarize per-op compute device placement via MLComputePlan.

    Ported from ``poc/benchmark_latency.py``'s ``compute_plan_report``:
    always measured against ``CPU_AND_NE``, matching the sanity
    check's own compute unit selection. If the API is unavailable or
    raises at runtime, the failure is recorded rather than propagated so
    that a self-check never fails because of this best-effort report.

    Args:
        mlmodelc_path: Compiled model directory.

    Returns:
        Dict describing op counts per device and the NE placement
        percentage, or ``{"status": "unavailable", "reason": ...}``.
    """
    try:
        from coremltools.models.compute_device import (
            MLCPUComputeDevice,
            MLGPUComputeDevice,
            MLNeuralEngineComputeDevice,
        )
        from coremltools.models.compute_plan import MLComputePlan

        plan = MLComputePlan.load_from_path(
            str(mlmodelc_path), compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        program = plan.model_structure.program
        if program is None:
            raise RuntimeError("compiled model has no MLProgram structure")

        operations = _collect_program_operations(program.functions["main"].block)
        device_counts = {"NE": 0, "GPU": 0, "CPU": 0, "unspecified": 0}
        for op in operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                # Typically `const` ops: no dispatch decision is made for them.
                device_counts["unspecified"] += 1
                continue
            device = usage.preferred_compute_device
            if isinstance(device, MLNeuralEngineComputeDevice):
                device_counts["NE"] += 1
            elif isinstance(device, MLGPUComputeDevice):
                device_counts["GPU"] += 1
            elif isinstance(device, MLCPUComputeDevice):
                device_counts["CPU"] += 1
            else:
                device_counts["unspecified"] += 1

        dispatched = device_counts["NE"] + device_counts["GPU"] + device_counts["CPU"]
        ne_placement_pct = 100.0 * device_counts["NE"] / dispatched if dispatched > 0 else 0.0
        return {
            "status": "measured",
            "total_ops": len(operations),
            "device_counts": device_counts,
            "ne_placement_pct": ne_placement_pct,
        }
    except Exception as exc:  # noqa: BLE001 - degrade to a recorded skip, never raise
        return {"status": "unavailable", "reason": str(exc)}


def _collect_program_operations(block: Any) -> list[Any]:
    """Recursively collect all operations from an MLProgram block.

    Ported from ``poc/benchmark_latency.py``. Descends into nested blocks
    (e.g. control-flow ops) so no op is missed.

    Args:
        block: An ``MLModelStructureProgramBlock``.

    Returns:
        Flat list of ``MLModelStructureProgramOperation`` instances.
    """
    operations: list[Any] = []
    for op in block.operations:
        operations.append(op)
        for nested_block in op.blocks:
            operations.extend(_collect_program_operations(nested_block))
    return operations


# --- warm latency -----------------------------------------------------------------


def _measure_latency(
    compiled: ct.models.CompiledMLModel, batch: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Measure warm predict() latency on one fixed batch (informational only).

    A simplified measurement compared to ``poc/benchmark_latency.py``
    (which remains the tool for real benchmarking): one batch, no cold
    timing, no round-robin input pool.

    Args:
        compiled: Loaded compiled model.
        batch: Predict() input, already sized to the model's fixed batch B
            (see :func:`_first_predict_batch`).

    Returns:
        Dict with the call count and median/p95/min latency in
        milliseconds.
    """
    for _ in range(LATENCY_WARMUP_PREDICTS):
        compiled.predict(batch)

    timings_ms: list[float] = []
    for _ in range(LATENCY_TIMED_PREDICTS):
        started = time.perf_counter()
        compiled.predict(batch)
        timings_ms.append((time.perf_counter() - started) * 1000.0)

    return {
        "n": len(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "p95_ms": float(np.percentile(timings_ms, 95)),
        "min_ms": min(timings_ms),
    }


# --- machine info & summary -------------------------------------------------------


def _machine_info() -> dict[str, str]:
    """Collect the machine info recorded in every report.

    Returns:
        Dict with the OS ``platform.platform()`` string and the CPU brand
        (``sysctl -n machdep.cpu.brand_string``, ``"unknown"`` if it
        cannot be determined -- e.g. a non-Apple-Silicon Mac reports a
        different sysctl key, or the binary is unavailable).
    """
    return {"platform": platform.platform(), "cpu": _cpu_brand()}


def _cpu_brand() -> str:
    """Read the CPU brand string via ``sysctl``, or ``"unknown"`` on any failure."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    brand = result.stdout.strip()
    return brand if result.returncode == 0 and brand else "unknown"


def _print_summary(context: SelfcheckContext, report: dict[str, Any]) -> None:
    """Print the self-check compatibility summary to stderr.

    One block per compiled bucket, in the pipeline's own progress style,
    meant to be pasted verbatim into a "works on my machine" report (the
    motivation for recording :func:`_machine_info` in the report too).

    Args:
        context: Self-check context (for the bucket/kind header).
        report: Report as returned by :func:`run_selfcheck`.
    """
    machine = report.get("machine", {})
    lines = [
        "      self-check summary:",
        f"        machine       : {machine.get('platform', 'unknown')} / "
        f"{machine.get('cpu', 'unknown')}",
        f"        bucket        : S={context.seq_len} B={context.batch_size} kind={context.kind}",
        f"        status        : {report.get('status', 'unknown')}",
    ]
    sanity = report.get("sanity")
    if isinstance(sanity, dict):
        lines.append(
            f"        sanity        : passed={sanity.get('passed')} "
            f"finite={sanity.get('finite')} {_sanity_summary_metric(sanity)}"
        )
    compute_plan = report.get("compute_plan")
    if isinstance(compute_plan, dict):
        if compute_plan.get("status") == "unavailable":
            lines.append(f"        NE placement  : unavailable ({compute_plan.get('reason')})")
        else:
            lines.append(
                f"        NE placement  : {compute_plan.get('ne_placement_pct', 0.0):.1f}% "
                f"({compute_plan.get('total_ops', 0)} ops)"
            )
    latency = report.get("latency")
    if isinstance(latency, dict):
        lines.append(
            f"        latency (ms)  : median={latency.get('median_ms', 0.0):.2f} "
            f"p95={latency.get('p95_ms', 0.0):.2f} min={latency.get('min_ms', 0.0):.2f}"
        )
    error = report.get("error")
    if error is not None:
        lines.append(f"        error         : {error}")
    for line in lines:
        print(line, file=sys.stderr, flush=True)


def _sanity_summary_metric(sanity: dict[str, Any]) -> str:
    """Pick the headline accuracy number for the summary line.

    Args:
        sanity: Report from :func:`_sanity_embedding` or
            :func:`_sanity_reranker`.

    Returns:
        ``"cosine_min=..."`` for an embedding sanity report,
        ``"sigmoid_max_abs_diff=..."`` for a reranker one, or an empty
        string if neither key is present.
    """
    if "cosine_min" in sanity:
        return f"cosine_min={sanity['cosine_min']:.5f}"
    if "sigmoid_max_abs_diff" in sanity:
        return f"sigmoid_max_abs_diff={sanity['sigmoid_max_abs_diff']:.5f}"
    return ""
