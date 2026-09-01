"""Tests for eeane.compiler.selfcheck.

Two layers, like the other compile-pipeline test modules:

* Unit tests (CI-safe) that mock the three external dependencies the
  self-check talks to -- ``CompiledMLModel.predict``, ``MLComputePlan``
  and a backend's ``reference_outputs`` -- while exercising the real
  threshold constants and status-derivation logic. A tiny byte-level
  tokenizer stands in for a frozen model tokenizer so the tokenization
  path (:mod:`eeane.runtime`) is genuinely exercised, not mocked.
* One end-to-end run of the real ``run_selfcheck`` over the same
  synthetic ModernBERT that :mod:`tests.test_compiler_pipeline` uses,
  skipped unless that module's local-machine marker is present.
"""

from __future__ import annotations

import contextlib
import io
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import coremltools.models.compute_device as compute_device_module
import coremltools.models.compute_plan as compute_plan_module
import numpy as np
import pytest
from test_compiler_pipeline import _E2E_AVAILABLE, E2E_SEQ_LEN, _build_synthetic_model
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers.models.modernbert import modeling_modernbert

from eeane import cli, runtime
from eeane.compiler import pipeline, selfcheck
from eeane.compiler.backends import base

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True, scope="module")
def _restore_transformers_patches() -> Iterator[None]:
    """Undo the global ModernBert monkeypatches after this module's tests."""
    original_rotate_half = modeling_modernbert.rotate_half
    original_forward = modeling_modernbert.ModernBertAttention.forward
    yield
    modeling_modernbert.rotate_half = original_rotate_half
    modeling_modernbert.ModernBertAttention.forward = original_forward


# --- test doubles: frozen tokenizer, backend, CompiledMLModel, MLComputePlan --


def _build_toy_tokenizer(path: Path) -> None:
    """Write a tiny byte-level frozen ``tokenizer.json`` usable for any UTF-8 text.

    Mirrors what ``eeane.compiler.tokenizer_freeze.freeze_tokenizer`` bakes
    (a ``tokenizers`` backend with a padding section already enabled), so
    :func:`eeane.runtime.load_frozen_tokenizer` accepts it without needing
    a HuggingFace model directory.

    Args:
        path: Destination ``tokenizer.json`` path.
    """
    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
    for index, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet())):
        vocab[character] = index + 4
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[("<s>", 2), ("</s>", 3)],
    )
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    tokenizer.save(str(path))


class _FakeBackend:
    """Minimal, fully controllable stand-in for a compile backend."""

    def __init__(
        self,
        kind: str,
        sanity_sets: dict[str, list[Any]],
        padding: Any,
        reference: np.ndarray,
        *,
        ordering: bool = True,
    ) -> None:
        """Store the fixed fixtures this fake backend serves.

        Args:
            kind: Model kind the fake backend answers for.
            sanity_sets: Inputs per language of the
                :class:`~eeane.compiler.backends.base.SanitySpec` returned
                by :meth:`sanity_spec`, in evaluation order.
            padding: Value returned by :meth:`padding_input`.
            reference: Value returned by :meth:`reference_outputs`: one row
                per input of every set, concatenated in set order.
            ordering: Whether a reranker's spec declares the
                relevant/irrelevant indices (index 0 / index 1) its
                ordering check needs. An embedding spec never declares
                them, as in a real backend.
        """
        self._kind = kind
        self._sanity_sets = sanity_sets
        self._padding = padding
        self._reference = reference
        self._ordering = ordering
        # Recorded so a test can prove the FP32 baseline is computed once
        # for every set together, not once per set (each call reloads the
        # model in a real backend).
        self.reference_calls: list[list[Any]] = []

    def sanity_spec(self, kind: str) -> base.SanitySpec:
        """Return the fixed sanity specification, asserting the expected kind."""
        assert kind == self._kind
        input_sets = tuple(
            (language, tuple(inputs)) for language, inputs in self._sanity_sets.items()
        )
        if kind != "reranker" or not self._ordering:
            return base.SanitySpec(input_sets=input_sets)
        return base.SanitySpec(input_sets=input_sets, relevant_index=0, irrelevant_index=1)

    def padding_input(self, kind: str) -> Any:
        """Return the fixed padding fixture, asserting the expected kind."""
        assert kind == self._kind
        return self._padding

    def reference_outputs(
        self, model_dir: Path, kind: str, inputs: list[Any], seq_len: int
    ) -> np.ndarray:
        """Return the fixed FP32 baseline, recording which inputs were asked for."""
        assert kind == self._kind
        self.reference_calls.append(list(inputs))
        return self._reference


def _make_context(
    tmp_path: Path,
    kind: str,
    sanity_sets: dict[str, list[Any]],
    padding_input: Any,
    reference: np.ndarray,
    seq_len: int = 16,
    batch_size: int = 1,
    ordering: bool = True,
) -> tuple[pipeline.SelfcheckContext, runtime.FrozenTokenizer]:
    """Build a :class:`~eeane.compiler.pipeline.SelfcheckContext` for a unit test.

    Args:
        tmp_path: Test-scoped scratch directory.
        kind: ``"embedding"`` or ``"reranker"``.
        sanity_sets: Inputs per language of the spec served by the fake
            backend's ``sanity_spec``, in evaluation order.
        padding_input: Fixture served by the fake backend's ``padding_input``.
        reference: Fixture served by the fake backend's ``reference_outputs``.
        seq_len: Fixed sequence length S.
        batch_size: Fixed batch size B.
        ordering: Whether the fake reranker spec declares the expected
            relevant/irrelevant pair indices.

    Returns:
        Tuple of the context and the frozen tokenizer it points to (handed
        back so a test can tokenize its own expected rows).
    """
    tokenizer_path = tmp_path / "tokenizer.json"
    _build_toy_tokenizer(tokenizer_path)
    frozen = runtime.load_frozen_tokenizer(tokenizer_path)
    mlmodelc_path = tmp_path / "variant.mlmodelc"
    mlmodelc_path.mkdir()
    backend = _FakeBackend(kind, sanity_sets, padding_input, reference, ordering=ordering)
    context = pipeline.SelfcheckContext(
        backend=backend,
        model_dir=tmp_path,
        kind=kind,
        seq_len=seq_len,
        batch_size=batch_size,
        output_name="logits" if kind == "reranker" else "embedding",
        mlmodelc_path=mlmodelc_path,
        tokenizer_path=tokenizer_path,
    )
    return context, frozen


def _row_key(row: np.ndarray) -> tuple[int, ...]:
    """Turn a tokenized row into a hashable key for the predict-output lookup."""
    return tuple(int(value) for value in row)


def _row_outputs(
    frozen: runtime.FrozenTokenizer,
    kind: str,
    inputs: list[Any],
    seq_len: int,
    outputs: np.ndarray,
) -> dict[tuple[int, ...], np.ndarray]:
    """Map every fixture's tokenized row to the raw output predict() must return.

    Args:
        frozen: Tokenizer the self-check will encode the fixtures with.
        kind: ``"embedding"`` or ``"reranker"``; decides the encoding.
        inputs: Every set's fixtures, concatenated in set order.
        seq_len: Fixed sequence length S.
        outputs: One raw output row per input, in the same order.

    Returns:
        The lookup :func:`_install_row_predict` takes.
    """
    tokenize = runtime.tokenize_pairs if kind == "reranker" else runtime.tokenize_texts
    rows = tokenize(frozen, inputs, seq_len)["input_ids"]
    return {_row_key(rows[index]): outputs[index] for index in range(len(inputs))}


def _install_row_predict(
    monkeypatch: pytest.MonkeyPatch, output_key: str, row_outputs: dict[tuple[int, ...], np.ndarray]
) -> list[dict[str, np.ndarray]]:
    """Patch ``ct.models.CompiledMLModel`` so predict() looks up a fixed row per input.

    Every row of a predict() call's ``input_ids`` is looked up in
    ``row_outputs``; this keeps the fake in step with real tokenization
    (the tests tokenize the same fixtures with the same toy tokenizer to
    build the lookup) while never touching an actual Core ML model.

    Args:
        monkeypatch: Test's monkeypatch fixture.
        output_key: Output name the fake prediction dict is keyed by.
        row_outputs: Map from a tokenized row (see :func:`_row_key`) to its
            desired raw output row.

    Returns:
        The list every predict() input is appended to, so a test can assert
        which rows were predicted, and in how many calls.
    """
    calls: list[dict[str, np.ndarray]] = []

    def _predict(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        calls.append(inputs)
        rows = inputs["input_ids"]
        stacked = np.stack([row_outputs[_row_key(row)] for row in rows])
        return {output_key: stacked}

    fake_compiled = SimpleNamespace(predict=_predict)
    monkeypatch.setattr(
        selfcheck.ct.models, "CompiledMLModel", lambda *args, **kwargs: fake_compiled
    )
    return calls


class _FakeOp:
    """Stand-in for an ``MLModelStructureProgramOperation`` with no nested blocks."""

    def __init__(self) -> None:
        """Initialize an operation with an empty nested-block list."""
        self.blocks: list[Any] = []


class _FakePlan:
    """Stand-in for an ``MLComputePlan`` reporting a fixed op/device mapping."""

    def __init__(self, device_by_op: dict[_FakeOp, Any]) -> None:
        """Build the fake plan's structure from a fixed op -> device mapping.

        Args:
            device_by_op: Maps each fake op to its device (an instance of
                one of the ``MLxComputeDevice`` classes, or ``None`` for an
                op with no dispatch decision, e.g. a ``const``).
        """
        self._device_by_op = device_by_op
        block = SimpleNamespace(operations=list(device_by_op))
        program = SimpleNamespace(functions={"main": SimpleNamespace(block=block)})
        self.model_structure = SimpleNamespace(program=program)

    def get_compute_device_usage_for_mlprogram_operation(self, op: _FakeOp) -> Any:
        """Return a fake usage object carrying ``op``'s preferred device, or ``None``."""
        device = self._device_by_op[op]
        return None if device is None else SimpleNamespace(preferred_compute_device=device)


def _fake_device(cls: type) -> Any:
    """Build a bare instance of an ``MLxComputeDevice`` class for ``isinstance`` checks.

    The real classes require a native proxy object to construct; since
    :func:`eeane.compiler.selfcheck._compute_plan_report` only ever calls
    ``isinstance`` on a device, bypassing ``__init__`` is sufficient and
    keeps the fake device real for the type hierarchy involved.
    """
    return object.__new__(cls)


def _build_fake_plan(ne_count: int, cpu_count: int) -> _FakePlan:
    """Build a fake compute plan with ``ne_count`` NE ops and ``cpu_count`` CPU ops."""
    ops = [_FakeOp() for _ in range(ne_count + cpu_count)]
    devices: list[Any] = [
        _fake_device(compute_device_module.MLNeuralEngineComputeDevice) for _ in range(ne_count)
    ] + [_fake_device(compute_device_module.MLCPUComputeDevice) for _ in range(cpu_count)]
    return _FakePlan(dict(zip(ops, devices, strict=True)))


def _install_compute_plan(monkeypatch: pytest.MonkeyPatch, plan: _FakePlan) -> None:
    """Patch ``MLComputePlan.load_from_path`` to return a fixed fake plan."""
    monkeypatch.setattr(
        compute_plan_module.MLComputePlan,
        "load_from_path",
        staticmethod(lambda path, compute_units=None: plan),
    )


def _install_compute_plan_failure(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    """Patch ``MLComputePlan.load_from_path`` to raise, simulating an unavailable API."""

    def _raise(path: Any, compute_units: Any = None) -> Any:
        raise RuntimeError(message)

    monkeypatch.setattr(compute_plan_module.MLComputePlan, "load_from_path", staticmethod(_raise))


# --- _derive_status: pure status-combination logic ----------------------------


@pytest.mark.parametrize(
    ("sanity_passed", "compute_plan", "expected"),
    [
        (True, {"status": "measured", "ne_placement_pct": 98.0}, selfcheck.STATUS_PASSED),
        (True, {"status": "measured", "ne_placement_pct": 89.0}, selfcheck.STATUS_WARNED),
        (True, {"status": "measured", "ne_placement_pct": 90.0}, selfcheck.STATUS_PASSED),
        (True, {"status": "unavailable", "reason": "boom"}, selfcheck.STATUS_PASSED),
        (False, {"status": "measured", "ne_placement_pct": 99.0}, selfcheck.STATUS_FAILED),
        (False, {"status": "unavailable", "reason": "boom"}, selfcheck.STATUS_FAILED),
    ],
)
def test_derive_status_branches(
    sanity_passed: bool, compute_plan: dict[str, Any], expected: str
) -> None:
    """Every combination of sanity pass/fail and NE placement must map correctly."""
    sanity = {"passed": sanity_passed}

    assert selfcheck._derive_status(sanity, compute_plan) == expected


# --- _compute_plan_report -----------------------------------------------------


def test_compute_plan_report_measures_ne_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful MLComputePlan load must report per-device op counts and NE percentage."""
    _install_compute_plan(monkeypatch, _build_fake_plan(ne_count=9, cpu_count=1))

    report = selfcheck._compute_plan_report(Path("/fake/model.mlmodelc"))

    assert report["status"] == "measured"
    assert report["total_ops"] == 10
    assert report["device_counts"] == {"NE": 9, "GPU": 0, "CPU": 1, "unspecified": 0}
    assert report["ne_placement_pct"] == pytest.approx(90.0)


def test_compute_plan_report_degrades_to_unavailable_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An MLComputePlan failure must be recorded, not raised."""
    _install_compute_plan_failure(monkeypatch, "MLComputePlan is not supported here")

    report = selfcheck._compute_plan_report(Path("/fake/model.mlmodelc"))

    assert report == {
        "status": "unavailable",
        "reason": "MLComputePlan is not supported here",
    }


# --- _measure_latency ----------------------------------------------------------


def test_measure_latency_computes_median_p95_min(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warm latency statistics must be derived correctly from the measured deltas."""
    # 19 fast calls (1ms) plus one slow outlier (101ms); perf_counter is
    # consulted twice (start/end) per timed call, none during warmup.
    elapsed_sec = [0.001] * 19 + [0.101]
    timestamps: list[float] = []
    clock = 0.0
    for delta in elapsed_sec:
        timestamps.append(clock)
        clock += delta
        timestamps.append(clock)
    timestamps_iter = iter(timestamps)
    monkeypatch.setattr(selfcheck.time, "perf_counter", lambda: next(timestamps_iter))
    fake_compiled = SimpleNamespace(predict=lambda batch: None)
    batch = {
        "input_ids": np.zeros((1, 1), dtype=np.int32),
        "attention_mask": np.ones((1, 1), dtype=np.int32),
    }

    result = selfcheck._measure_latency(fake_compiled, batch)

    assert result["n"] == selfcheck.LATENCY_TIMED_PREDICTS == len(elapsed_sec)
    assert result["min_ms"] == pytest.approx(1.0)
    assert result["median_ms"] == pytest.approx(1.0)
    assert result["p95_ms"] == pytest.approx(6.0)


# --- run_selfcheck: embedding ---------------------------------------------------


def test_run_selfcheck_embedding_passed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A high-NE, on-threshold-accuracy embedding variant must report status='passed'."""
    sanity_texts = ["alpha", "beta", "gamma"]
    reference = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    context, frozen = _make_context(tmp_path, "embedding", {"en": sanity_texts}, "", reference)
    rows = runtime.tokenize_texts(frozen, sanity_texts, context.seq_len)["input_ids"]
    row_outputs = {_row_key(rows[i]): reference[i] for i in range(len(sanity_texts))}
    _install_row_predict(monkeypatch, "embedding", row_outputs)
    _install_compute_plan(monkeypatch, _build_fake_plan(ne_count=95, cpu_count=5))

    report = selfcheck.run_selfcheck(context)

    assert report["status"] == selfcheck.STATUS_PASSED
    assert report["sanity"]["passed"] is True
    assert report["sanity"]["cosine_min"] == pytest.approx(1.0)
    assert report["sanity"]["embedding_dim"] == reference.shape[1]
    assert report["compute_plan"]["ne_placement_pct"] == pytest.approx(95.0)
    assert report["latency"]["n"] == selfcheck.LATENCY_TIMED_PREDICTS
    assert report["machine"]["platform"]
    json.dumps(report)  # must be JSON-serializable


def test_run_selfcheck_embedding_warned_on_low_ne_placement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Passing accuracy with NE placement below the threshold must warn, not fail."""
    sanity_texts = ["alpha", "beta"]
    reference = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    context, frozen = _make_context(tmp_path, "embedding", {"en": sanity_texts}, "", reference)
    rows = runtime.tokenize_texts(frozen, sanity_texts, context.seq_len)["input_ids"]
    row_outputs = {_row_key(rows[i]): reference[i] for i in range(len(sanity_texts))}
    _install_row_predict(monkeypatch, "embedding", row_outputs)
    _install_compute_plan(monkeypatch, _build_fake_plan(ne_count=60, cpu_count=40))

    report = selfcheck.run_selfcheck(context)

    assert report["status"] == selfcheck.STATUS_WARNED
    assert report["sanity"]["passed"] is True
    assert report["compute_plan"]["ne_placement_pct"] == pytest.approx(60.0)
    json.dumps(report)


def test_run_selfcheck_embedding_failed_on_low_cosine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A coreml output far from the FP32 baseline must fail the sanity check."""
    sanity_texts = ["alpha", "beta"]
    reference = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    context, frozen = _make_context(tmp_path, "embedding", {"en": sanity_texts}, "", reference)
    rows = runtime.tokenize_texts(frozen, sanity_texts, context.seq_len)["input_ids"]
    # Orthogonal to the reference row: cosine == 0.0, far below the threshold.
    orthogonal = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    row_outputs = {_row_key(rows[i]): orthogonal[i] for i in range(len(sanity_texts))}
    _install_row_predict(monkeypatch, "embedding", row_outputs)
    _install_compute_plan(monkeypatch, _build_fake_plan(ne_count=99, cpu_count=1))

    report = selfcheck.run_selfcheck(context)

    assert report["status"] == selfcheck.STATUS_FAILED
    assert report["sanity"]["passed"] is False
    assert report["sanity"]["cosine_min"] < selfcheck.SANITY_COSINE_THRESHOLD
    json.dumps(report)


def test_run_selfcheck_embedding_failed_on_nan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-finite coreml output must fail the sanity check regardless of cosine."""
    sanity_texts = ["alpha", "beta"]
    reference = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    context, frozen = _make_context(tmp_path, "embedding", {"en": sanity_texts}, "", reference)
    rows = runtime.tokenize_texts(frozen, sanity_texts, context.seq_len)["input_ids"]
    outputs = {
        _row_key(rows[0]): np.array([np.nan, 0.0], dtype=np.float32),
        _row_key(rows[1]): reference[1],
    }
    _install_row_predict(monkeypatch, "embedding", outputs)
    _install_compute_plan(monkeypatch, _build_fake_plan(ne_count=99, cpu_count=1))

    report = selfcheck.run_selfcheck(context)

    assert report["status"] == selfcheck.STATUS_FAILED
    assert report["sanity"]["finite"] is False
    assert report["sanity"]["passed"] is False
    json.dumps(report)


def test_run_selfcheck_compute_plan_unavailable_does_not_change_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A passing sanity check plus an unavailable compute plan must still report 'passed'."""
    sanity_texts = ["alpha"]
    reference = np.array([[1.0, 0.0]], dtype=np.float32)
    context, frozen = _make_context(tmp_path, "embedding", {"en": sanity_texts}, "", reference)
    rows = runtime.tokenize_texts(frozen, sanity_texts, context.seq_len)["input_ids"]
    _install_row_predict(monkeypatch, "embedding", {_row_key(rows[0]): reference[0]})
    _install_compute_plan_failure(monkeypatch, "MLComputePlan API missing")

    report = selfcheck.run_selfcheck(context)

    expected_compute_plan = {"status": "unavailable", "reason": "MLComputePlan API missing"}
    assert report["compute_plan"] == expected_compute_plan
    assert report["status"] == selfcheck.STATUS_PASSED
    json.dumps(report)


def test_run_selfcheck_internal_exception_returns_failed_with_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unexpected internal failure must be reported, never raised."""

    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("cannot load the compiled model")

    monkeypatch.setattr(selfcheck.ct.models, "CompiledMLModel", _raise)
    tokenizer_path = tmp_path / "tokenizer.json"
    _build_toy_tokenizer(tokenizer_path)
    backend = _FakeBackend("embedding", {"en": ["alpha"]}, "", np.zeros((1, 2), dtype=np.float32))
    context = pipeline.SelfcheckContext(
        backend=backend,
        model_dir=tmp_path,
        kind="embedding",
        seq_len=8,
        batch_size=1,
        output_name="embedding",
        mlmodelc_path=tmp_path / "broken.mlmodelc",
        tokenizer_path=tokenizer_path,
    )

    report = selfcheck.run_selfcheck(context)

    assert report["status"] == selfcheck.STATUS_FAILED
    assert report["error"] == "cannot load the compiled model"
    assert "machine" in report and report["machine"]["platform"]
    assert "sanity" not in report
    assert "compute_plan" not in report
    assert "latency" not in report
    json.dumps(report)


# --- run_selfcheck: one embedding set per language ------------------------------

# Two sanity sets whose four texts are all different, so a tokenized row
# identifies both the set it belongs to and the text it came from.
_TWO_SET_TEXTS: dict[str, list[Any]] = {"en": ["alpha", "beta"], "ja": ["gamma", "delta"]}
_TWO_SET_ALL_TEXTS: list[Any] = ["alpha", "beta", "gamma", "delta"]

# One FP32 baseline embedding per row of _TWO_SET_ALL_TEXTS.
_TWO_SET_REFERENCE = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

# Compiled outputs orthogonal to the baseline on the English rows and
# identical to it on the Japanese ones: only the second set can pass.
_JAPANESE_ONLY = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def _run_two_set_embedding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    coreml_outputs: np.ndarray,
    batch_size: int = 1,
) -> tuple[dict[str, Any], pipeline.SelfcheckContext, list[dict[str, np.ndarray]]]:
    """Run the self-check over one English and one Japanese embedding set.

    Args:
        monkeypatch: Test's monkeypatch fixture.
        tmp_path: Test-scoped scratch directory.
        coreml_outputs: One compiled-model output row per text of
            :data:`_TWO_SET_ALL_TEXTS`.
        batch_size: Fixed batch size B of the variant.

    Returns:
        Tuple of the report, the context (whose fake backend records the
        ``reference_outputs`` calls) and every predict() input.
    """
    context, frozen = _make_context(
        tmp_path, "embedding", _TWO_SET_TEXTS, "", _TWO_SET_REFERENCE, batch_size=batch_size
    )
    calls = _install_row_predict(
        monkeypatch,
        "embedding",
        _row_outputs(frozen, "embedding", _TWO_SET_ALL_TEXTS, context.seq_len, coreml_outputs),
    )
    _install_compute_plan(monkeypatch, _build_fake_plan(ne_count=99, cpu_count=1))
    return selfcheck.run_selfcheck(context), context, calls


def test_run_selfcheck_passes_when_a_single_language_set_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One set clearing the threshold carries the variant, however the others scored."""
    report, _, _ = _run_two_set_embedding(monkeypatch, tmp_path, _JAPANESE_ONLY)

    assert report["status"] == selfcheck.STATUS_PASSED
    assert report["sanity"]["passed"] is True
    assert report["sanity"]["best_set"] == "ja"
    json.dumps(report)


def test_run_selfcheck_records_the_result_of_every_language_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A set that missed the threshold must still be reported, with its own numbers."""
    report, _, _ = _run_two_set_embedding(monkeypatch, tmp_path, _JAPANESE_ONLY)
    sets = report["sanity"]["sets"]

    assert list(sets) == ["en", "ja"]
    assert sets["en"]["passed"] is False
    assert sets["ja"]["passed"] is True
    assert sets["en"]["cosine_min"] == pytest.approx(0.0)
    assert sets["ja"]["cosine_min"] == pytest.approx(1.0)
    assert sets["en"]["cosine_per_text"] == [pytest.approx(0.0), pytest.approx(0.0)]


def test_run_selfcheck_top_level_values_describe_the_best_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A reader of the established keys must see the set the variant was accepted on."""
    report, _, _ = _run_two_set_embedding(monkeypatch, tmp_path, _JAPANESE_ONLY)
    sanity = report["sanity"]
    best = sanity["sets"][sanity["best_set"]]

    assert sanity["cosine_min"] == best["cosine_min"]
    assert sanity["cosine_mean"] == best["cosine_mean"]
    assert sanity["cosine_per_text"] == best["cosine_per_text"]
    assert sanity["finite"] == best["finite"]
    assert sanity["cosine_threshold"] == selfcheck.SANITY_COSINE_THRESHOLD
    assert sanity["embedding_dim"] == _TWO_SET_REFERENCE.shape[1]


def test_run_selfcheck_fails_when_no_language_set_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every set missing the threshold is a failure, with the closest one reported."""
    # English rows orthogonal to their baseline (cosine 0.0), Japanese rows
    # at cosine 0.6 -- closer, but still far below the threshold.
    outputs = np.array(
        [[0.0, 1.0], [1.0, 0.0], [0.6, 0.8], [0.8, 0.6]],
        dtype=np.float32,
    )

    report, _, _ = _run_two_set_embedding(monkeypatch, tmp_path, outputs)

    assert report["status"] == selfcheck.STATUS_FAILED
    assert report["sanity"]["passed"] is False
    assert report["sanity"]["best_set"] == "ja"
    assert report["sanity"]["cosine_min"] == pytest.approx(0.6)
    assert not any(set_report["passed"] for set_report in report["sanity"]["sets"].values())
    json.dumps(report)


@pytest.mark.parametrize(
    ("outputs", "expected_best"),
    [
        ([[1.0, 0.1], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], "ja"),
        ([[1.0, 0.0], [0.0, 1.0], [1.0, 0.1], [0.0, 1.0]], "en"),
    ],
    ids=["japanese-closer", "english-closer"],
)
def test_run_selfcheck_reports_the_closest_of_the_passing_sets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outputs: list[list[float]],
    expected_best: str,
) -> None:
    """With more than one set passing, the one nearest its baseline is the reported one."""
    report, _, _ = _run_two_set_embedding(
        monkeypatch, tmp_path, np.array(outputs, dtype=np.float32)
    )

    assert report["sanity"]["passed"] is True
    assert all(set_report["passed"] for set_report in report["sanity"]["sets"].values())
    assert report["sanity"]["best_set"] == expected_best
    assert report["sanity"]["cosine_min"] == pytest.approx(1.0)


def test_run_selfcheck_breaks_a_tie_between_sets_by_declaration_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Two equally good sets must always resolve to the same one, run after run."""
    report, _, _ = _run_two_set_embedding(monkeypatch, tmp_path, _TWO_SET_REFERENCE.copy())

    assert report["sanity"]["best_set"] == "en"


def test_run_selfcheck_never_reports_a_non_finite_set_as_the_best_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A NaN row makes a cosine meaningless, so that set must not represent the variant."""
    outputs = np.array(
        [[np.nan, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    report, _, _ = _run_two_set_embedding(monkeypatch, tmp_path, outputs)

    assert report["sanity"]["sets"]["en"]["finite"] is False
    assert report["sanity"]["sets"]["en"]["passed"] is False
    assert report["sanity"]["best_set"] == "ja"
    assert report["sanity"]["passed"] is True
    assert report["status"] == selfcheck.STATUS_PASSED
    json.dumps(report)


def test_run_selfcheck_computes_the_fp32_baseline_of_every_set_in_one_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One call per set would reload the FP32 model once per set; the sets share one."""
    _, context, _ = _run_two_set_embedding(monkeypatch, tmp_path, _JAPANESE_ONLY)

    assert context.backend.reference_calls == [_TWO_SET_ALL_TEXTS]


def test_run_selfcheck_predicts_every_row_of_every_set_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The sets are evaluated from one pass over the compiled model, not one pass each."""

    def _no_latency(compiled: Any, batch: dict[str, np.ndarray]) -> dict[str, Any]:
        return {"n": 0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0}

    # The warm-latency measurement predicts too; leaving it out keeps the
    # count to the rows the sanity check itself runs.
    monkeypatch.setattr(selfcheck, "_measure_latency", _no_latency)

    _, _, calls = _run_two_set_embedding(monkeypatch, tmp_path, _JAPANESE_ONLY)

    predicted = [row for call in calls for row in call["input_ids"]]
    assert len(predicted) == len(_TWO_SET_ALL_TEXTS)


def test_run_selfcheck_measures_latency_on_the_best_sets_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The warm-latency batch must hold tokens the reported set was actually verified on."""
    batches: list[dict[str, np.ndarray]] = []

    def _fake_latency(compiled: Any, batch: dict[str, np.ndarray]) -> dict[str, Any]:
        batches.append(batch)
        return {"n": 0, "median_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0}

    monkeypatch.setattr(selfcheck, "_measure_latency", _fake_latency)
    context, frozen = _make_context(tmp_path, "embedding", _TWO_SET_TEXTS, "", _TWO_SET_REFERENCE)
    _install_row_predict(
        monkeypatch,
        "embedding",
        _row_outputs(frozen, "embedding", _TWO_SET_ALL_TEXTS, context.seq_len, _JAPANESE_ONLY),
    )
    _install_compute_plan(monkeypatch, _build_fake_plan(ne_count=99, cpu_count=1))

    report = selfcheck.run_selfcheck(context)

    assert report["sanity"]["best_set"] == "ja"
    expected = runtime.tokenize_texts(frozen, ["gamma"], context.seq_len)
    assert np.array_equal(batches[0]["input_ids"], expected["input_ids"])


def test_run_selfcheck_checks_batch_consistency_on_the_best_sets_first_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A batch is only worth checking on an input the model demonstrably handles."""
    texts: list[str] = []

    def _fake_consistency(
        compiled: Any,
        frozen_tokenizer: Any,
        text: str,
        seq_len: int,
        batch_size: int,
        output_key: str,
    ) -> dict[str, Any]:
        texts.append(text)
        return {"passed": True}

    monkeypatch.setattr(selfcheck, "_check_batch_consistency_embedding", _fake_consistency)

    report, _, _ = _run_two_set_embedding(monkeypatch, tmp_path, _JAPANESE_ONLY, batch_size=2)

    assert report["sanity"]["best_set"] == "ja"
    assert texts == [_TWO_SET_TEXTS["ja"][0]]


def test_run_selfcheck_fails_an_inconsistent_batch_even_when_a_set_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Rows influencing each other invalidates the accuracy the sets just proved."""
    monkeypatch.setattr(
        selfcheck,
        "_check_batch_consistency_embedding",
        lambda *args, **kwargs: {"passed": False, "cosine_min": 0.5},
    )

    report, _, _ = _run_two_set_embedding(monkeypatch, tmp_path, _JAPANESE_ONLY, batch_size=2)

    assert report["sanity"]["sets"]["ja"]["passed"] is True
    assert report["sanity"]["passed"] is False
    assert report["status"] == selfcheck.STATUS_FAILED
    assert report["sanity"]["batch_consistency"] == {"passed": False, "cosine_min": 0.5}


# --- run_selfcheck: reranker ----------------------------------------------------


def test_run_selfcheck_reranker_passed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A reranker variant must be scored via sigmoid difference and pair ordering."""
    sanity_pairs = [("q-relevant", "d-relevant"), ("q-irrelevant", "d-irrelevant")]
    reference = np.array([2.0, -2.0], dtype=np.float32)
    context, frozen = _make_context(tmp_path, "reranker", {"en": sanity_pairs}, ("", ""), reference)
    rows = runtime.tokenize_pairs(frozen, sanity_pairs, context.seq_len)["input_ids"]
    row_outputs = {
        _row_key(rows[i]): np.array([reference[i]], dtype=np.float32)
        for i in range(len(sanity_pairs))
    }
    _install_row_predict(monkeypatch, "logits", row_outputs)
    _install_compute_plan(monkeypatch, _build_fake_plan(ne_count=99, cpu_count=1))

    report = selfcheck.run_selfcheck(context)

    assert report["status"] == selfcheck.STATUS_PASSED
    assert report["sanity"]["passed"] is True
    assert report["sanity"]["ordering_checked"] is True
    assert report["sanity"]["ordering_ok_coreml"] is True
    assert report["sanity"]["ordering_ok_fp32"] is True
    assert report["sanity"]["sigmoid_max_abs_diff"] == pytest.approx(0.0)
    assert "embedding_dim" not in report["sanity"]  # embedding-only field
    json.dumps(report)


def test_run_selfcheck_reranker_failed_on_wrong_ordering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A compiled model ranking the irrelevant pair higher must fail the sanity check."""
    sanity_pairs = [("q-relevant", "d-relevant"), ("q-irrelevant", "d-irrelevant")]
    reference = np.array([2.0, -2.0], dtype=np.float32)
    context, frozen = _make_context(tmp_path, "reranker", {"en": sanity_pairs}, ("", ""), reference)
    rows = runtime.tokenize_pairs(frozen, sanity_pairs, context.seq_len)["input_ids"]
    # The compiled model returns the two logits swapped, so the relevant
    # pair scores lower than the irrelevant one.
    row_outputs = {
        _row_key(rows[0]): np.array([reference[1]], dtype=np.float32),
        _row_key(rows[1]): np.array([reference[0]], dtype=np.float32),
    }
    _install_row_predict(monkeypatch, "logits", row_outputs)
    _install_compute_plan(monkeypatch, _build_fake_plan(ne_count=99, cpu_count=1))

    report = selfcheck.run_selfcheck(context)

    assert report["status"] == selfcheck.STATUS_FAILED
    assert report["sanity"]["ordering_checked"] is True
    assert report["sanity"]["ordering_ok_coreml"] is False
    assert report["sanity"]["ordering_ok_fp32"] is True
    assert report["sanity"]["passed"] is False
    json.dumps(report)


def test_run_selfcheck_reranker_without_ordering_metadata_skips_that_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A spec without relevant/irrelevant indices must record the skip, not crash or fail."""
    sanity_pairs = [("q-a", "d-a"), ("q-b", "d-b")]
    reference = np.array([2.0, -2.0], dtype=np.float32)
    context, frozen = _make_context(
        tmp_path, "reranker", {"en": sanity_pairs}, ("", ""), reference, ordering=False
    )
    rows = runtime.tokenize_pairs(frozen, sanity_pairs, context.seq_len)["input_ids"]
    row_outputs = {
        _row_key(rows[i]): np.array([reference[i]], dtype=np.float32)
        for i in range(len(sanity_pairs))
    }
    _install_row_predict(monkeypatch, "logits", row_outputs)
    _install_compute_plan(monkeypatch, _build_fake_plan(ne_count=99, cpu_count=1))

    report = selfcheck.run_selfcheck(context)

    assert report["status"] == selfcheck.STATUS_PASSED
    assert report["sanity"]["ordering_checked"] is False
    assert report["sanity"]["ordering_ok_coreml"] is None
    assert report["sanity"]["ordering_ok_fp32"] is None
    assert report["sanity"]["passed"] is True
    json.dumps(report)


# --- run_selfcheck: one reranker set per language --------------------------------

# Two pair sets whose four pairs are all different, so a tokenized row
# identifies the set it belongs to. Within a set the two pairs share their
# query, exactly as a backend's fixtures do, so only the document decides
# which of them must score higher.
_TWO_SET_PAIRS: dict[str, list[Any]] = {
    "en": [("q-en", "d-en-relevant"), ("q-en", "d-en-irrelevant")],
    "ja": [("q-ja", "d-ja-relevant"), ("q-ja", "d-ja-irrelevant")],
}
_TWO_SET_ALL_PAIRS: list[Any] = [*_TWO_SET_PAIRS["en"], *_TWO_SET_PAIRS["ja"]]


def _run_two_set_reranker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fp32_logits: list[float],
    coreml_logits: list[float],
    batch_size: int = 1,
) -> tuple[dict[str, Any], pipeline.SelfcheckContext]:
    """Run the self-check over one English and one Japanese pair set.

    Args:
        monkeypatch: Test's monkeypatch fixture.
        tmp_path: Test-scoped scratch directory.
        fp32_logits: FP32 baseline logit per pair of
            :data:`_TWO_SET_ALL_PAIRS`.
        coreml_logits: Compiled-model logit per pair, in the same order.
        batch_size: Fixed batch size B of the variant.

    Returns:
        Tuple of the report and the context it was produced from.
    """
    reference = np.array(fp32_logits, dtype=np.float32)
    context, frozen = _make_context(
        tmp_path, "reranker", _TWO_SET_PAIRS, ("", ""), reference, batch_size=batch_size
    )
    _install_row_predict(
        monkeypatch,
        "logits",
        _row_outputs(
            frozen,
            "reranker",
            _TWO_SET_ALL_PAIRS,
            context.seq_len,
            np.array(coreml_logits, dtype=np.float32).reshape(-1, 1),
        ),
    )
    _install_compute_plan(monkeypatch, _build_fake_plan(ne_count=99, cpu_count=1))
    return selfcheck.run_selfcheck(context), context


def test_run_selfcheck_reranker_passes_when_one_language_set_orders_correctly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ordering expectation holds within a set, so one set may order wrongly."""
    # English: the relevant pair scores below the irrelevant one; Japanese:
    # the expected way round. The compiled logits match the baseline in both.
    report, _ = _run_two_set_reranker(
        monkeypatch, tmp_path, [-2.0, 2.0, 2.0, -2.0], [-2.0, 2.0, 2.0, -2.0]
    )
    sets = report["sanity"]["sets"]

    assert report["status"] == selfcheck.STATUS_PASSED
    assert report["sanity"]["passed"] is True
    assert report["sanity"]["best_set"] == "ja"
    assert sets["en"]["ordering_ok_coreml"] is False
    assert sets["en"]["ordering_ok_fp32"] is False
    assert sets["en"]["passed"] is False
    assert sets["ja"]["ordering_ok_coreml"] is True
    assert sets["ja"]["passed"] is True
    json.dumps(report)


def test_run_selfcheck_reranker_fails_when_no_language_set_orders_correctly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both sets ordering the pairs wrongly is a failure, whatever the score difference."""
    # Both sets order the relevant pair below the irrelevant one; the
    # English compiled logits match their baseline exactly, so English is
    # the closest set and the one the report is summarized by.
    report, _ = _run_two_set_reranker(
        monkeypatch, tmp_path, [-2.0, 2.0, -1.0, 1.0], [-2.0, 2.0, -1.5, 1.0]
    )

    assert report["status"] == selfcheck.STATUS_FAILED
    assert report["sanity"]["passed"] is False
    assert report["sanity"]["best_set"] == "en"
    assert report["sanity"]["sigmoid_max_abs_diff"] == pytest.approx(0.0)
    json.dumps(report)


def test_run_selfcheck_reranker_set_can_miss_the_tolerance_while_ordering_correctly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A set is only passed when it clears the score tolerance too, not the ordering alone."""
    # English: ordering preserved, but the compiled score is far from the
    # baseline; Japanese: an exact match.
    report, _ = _run_two_set_reranker(
        monkeypatch, tmp_path, [2.0, -2.0, 2.0, -2.0], [0.5, -3.0, 2.0, -2.0]
    )
    sets = report["sanity"]["sets"]

    assert sets["en"]["ordering_ok_coreml"] is True
    assert sets["en"]["sigmoid_max_abs_diff"] > selfcheck.SANITY_SIGMOID_TOLERANCE
    assert sets["en"]["passed"] is False
    assert report["sanity"]["best_set"] == "ja"
    assert report["sanity"]["passed"] is True


def test_run_selfcheck_reranker_top_level_values_describe_the_best_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The established reranker keys must carry the numbers of the reported set alone."""
    report, _ = _run_two_set_reranker(
        monkeypatch, tmp_path, [-2.0, 2.0, 2.0, -2.0], [-2.0, 2.0, 2.0, -2.0]
    )
    sanity = report["sanity"]
    best = sanity["sets"][sanity["best_set"]]

    for key in (
        "coreml_logits",
        "fp32_logits",
        "coreml_scores",
        "fp32_scores",
        "sigmoid_abs_diff",
        "sigmoid_max_abs_diff",
        "ordering_checked",
        "ordering_ok_coreml",
        "ordering_ok_fp32",
        "finite",
    ):
        assert sanity[key] == best[key], key
    assert sanity["coreml_logits"] == [pytest.approx(2.0), pytest.approx(-2.0)]
    assert sanity["sigmoid_tolerance"] == selfcheck.SANITY_SIGMOID_TOLERANCE


def test_run_selfcheck_reranker_checks_batch_consistency_on_the_best_sets_first_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A batch is only worth checking on a pair the model demonstrably handles."""
    pairs: list[tuple[str, str]] = []

    def _fake_consistency(
        compiled: Any,
        frozen_tokenizer: Any,
        pair: tuple[str, str],
        seq_len: int,
        batch_size: int,
        output_key: str,
    ) -> dict[str, Any]:
        pairs.append(pair)
        return {"passed": True}

    monkeypatch.setattr(selfcheck, "_check_batch_consistency_reranker", _fake_consistency)

    report, _ = _run_two_set_reranker(
        monkeypatch, tmp_path, [-2.0, 2.0, 2.0, -2.0], [-2.0, 2.0, 2.0, -2.0], batch_size=2
    )

    assert report["sanity"]["best_set"] == "ja"
    assert pairs == [_TWO_SET_PAIRS["ja"][0]]


# --- end-to-end on a synthetic ModernBERT (local only) --------------------------


@pytest.fixture(scope="module")
def synthetic_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the synthetic model directory once for the end-to-end test."""
    if not _E2E_AVAILABLE:
        pytest.skip("end-to-end self-check needs a local machine with xcrun")
    workspace = tmp_path_factory.mktemp("synthetic-selfcheck")
    return _build_synthetic_model(workspace / "tiny-modernbert")


def test_e2e_real_selfcheck_records_every_section(
    synthetic_model_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """The real self-check must run end to end and populate every metadata section."""
    workspace = tmp_path_factory.mktemp("compile-selfcheck")
    out_dir = workspace / "cache"
    arguments = [str(synthetic_model_dir), "--buckets", str(E2E_SEQ_LEN), "--out-dir", str(out_dir)]

    args = cli.build_parser().parse_args(["compile", *arguments])
    with contextlib.redirect_stdout(io.StringIO()):
        exit_code = pipeline.run(args, selfcheck_fn=selfcheck.run_selfcheck)

    assert exit_code == 0
    model_root = out_dir / "compiled" / synthetic_model_dir.name
    metadata_path = model_root / f"s{E2E_SEQ_LEN}_b1_eager_macos13.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    report = metadata["selfcheck"]

    assert report["status"] in {selfcheck.STATUS_PASSED, selfcheck.STATUS_WARNED}
    assert report["sanity"]["passed"] is True
    assert report["sanity"]["embedding_dim"] > 0
    assert set(report) >= {"status", "sanity", "compute_plan", "latency", "machine"}
    assert report["latency"]["n"] == selfcheck.LATENCY_TIMED_PREDICTS
    assert report["machine"]["platform"]
    json.dumps(report)  # the metadata file itself already proves this, but be explicit
