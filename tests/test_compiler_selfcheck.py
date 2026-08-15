"""Tests for eeane.compiler.selfcheck (v0.6 T5, 開発資料/v0.6実装計画.md §4.5).

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

    def __init__(self, kind: str, sanity: list[Any], padding: Any, reference: np.ndarray) -> None:
        """Store the fixed fixtures this fake backend serves.

        Args:
            kind: Model kind the fake backend answers for.
            sanity: Value returned by :meth:`sanity_inputs`.
            padding: Value returned by :meth:`padding_input`.
            reference: Value returned by :meth:`reference_outputs`.
        """
        self._kind = kind
        self._sanity = sanity
        self._padding = padding
        self._reference = reference

    def sanity_inputs(self, kind: str) -> list[Any]:
        """Return the fixed sanity fixtures, asserting the expected kind."""
        assert kind == self._kind
        return list(self._sanity)

    def padding_input(self, kind: str) -> Any:
        """Return the fixed padding fixture, asserting the expected kind."""
        assert kind == self._kind
        return self._padding

    def reference_outputs(
        self, model_dir: Path, kind: str, inputs: list[Any], seq_len: int
    ) -> np.ndarray:
        """Return the fixed FP32 baseline, ignoring everything but ``kind``."""
        assert kind == self._kind
        return self._reference


def _make_context(
    tmp_path: Path,
    kind: str,
    sanity_inputs: list[Any],
    padding_input: Any,
    reference: np.ndarray,
    seq_len: int = 16,
    batch_size: int = 1,
) -> tuple[pipeline.SelfcheckContext, runtime.FrozenTokenizer]:
    """Build a :class:`~eeane.compiler.pipeline.SelfcheckContext` for a unit test.

    Args:
        tmp_path: Test-scoped scratch directory.
        kind: ``"embedding"`` or ``"reranker"``.
        sanity_inputs: Fixture served by the fake backend's ``sanity_inputs``.
        padding_input: Fixture served by the fake backend's ``padding_input``.
        reference: Fixture served by the fake backend's ``reference_outputs``.
        seq_len: Fixed sequence length S.
        batch_size: Fixed batch size B.

    Returns:
        Tuple of the context and the frozen tokenizer it points to (handed
        back so a test can tokenize its own expected rows).
    """
    tokenizer_path = tmp_path / "tokenizer.json"
    _build_toy_tokenizer(tokenizer_path)
    frozen = runtime.load_frozen_tokenizer(tokenizer_path)
    mlmodelc_path = tmp_path / "variant.mlmodelc"
    mlmodelc_path.mkdir()
    backend = _FakeBackend(kind, sanity_inputs, padding_input, reference)
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


def _install_row_predict(
    monkeypatch: pytest.MonkeyPatch, output_key: str, row_outputs: dict[tuple[int, ...], np.ndarray]
) -> None:
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
    """

    def _predict(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        rows = inputs["input_ids"]
        stacked = np.stack([row_outputs[_row_key(row)] for row in rows])
        return {output_key: stacked}

    fake_compiled = SimpleNamespace(predict=_predict)
    monkeypatch.setattr(
        selfcheck.ct.models, "CompiledMLModel", lambda *args, **kwargs: fake_compiled
    )


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
    context, frozen = _make_context(tmp_path, "embedding", sanity_texts, "", reference)
    rows = runtime.tokenize_texts(frozen, sanity_texts, context.seq_len)["input_ids"]
    row_outputs = {_row_key(rows[i]): reference[i] for i in range(len(sanity_texts))}
    _install_row_predict(monkeypatch, "embedding", row_outputs)
    _install_compute_plan(monkeypatch, _build_fake_plan(ne_count=95, cpu_count=5))

    report = selfcheck.run_selfcheck(context)

    assert report["status"] == selfcheck.STATUS_PASSED
    assert report["sanity"]["passed"] is True
    assert report["sanity"]["cosine_min"] == pytest.approx(1.0)
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
    context, frozen = _make_context(tmp_path, "embedding", sanity_texts, "", reference)
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
    context, frozen = _make_context(tmp_path, "embedding", sanity_texts, "", reference)
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
    context, frozen = _make_context(tmp_path, "embedding", sanity_texts, "", reference)
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
    context, frozen = _make_context(tmp_path, "embedding", sanity_texts, "", reference)
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
    backend = _FakeBackend("embedding", ["alpha"], "", np.zeros((1, 2), dtype=np.float32))
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


# --- run_selfcheck: reranker ----------------------------------------------------


def test_run_selfcheck_reranker_passed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A reranker variant must be scored via sigmoid difference and pair ordering."""
    sanity_pairs = [("q-relevant", "d-relevant"), ("q-irrelevant", "d-irrelevant")]
    reference = np.array([2.0, -2.0], dtype=np.float32)
    context, frozen = _make_context(tmp_path, "reranker", sanity_pairs, ("", ""), reference)
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
    assert report["sanity"]["ordering_ok_coreml"] is True
    assert report["sanity"]["ordering_ok_fp32"] is True
    assert report["sanity"]["sigmoid_max_abs_diff"] == pytest.approx(0.0)
    json.dumps(report)


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
    assert set(report) >= {"status", "sanity", "compute_plan", "latency", "machine"}
    assert report["latency"]["n"] == selfcheck.LATENCY_TIMED_PREDICTS
    assert report["machine"]["platform"]
    json.dumps(report)  # the metadata file itself already proves this, but be explicit
