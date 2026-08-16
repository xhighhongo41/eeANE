"""Tests for predicting several inputs of one request together.

An embedding model may be served with batched artifacts for some of its
buckets. The inputs of one request that share such a bucket are then
predicted together instead of one at a time, while everything else --
other buckets, leftover inputs, rerank requests -- keeps being predicted
one input per call. A request never waits for another one to fill a
group.

Nothing here needs a Neural Engine: the artifact loader is injected, so
the stand-in artifacts answer ``predict`` from memory, record the shape
they were fed and can be told to park a prediction or to answer with
values no caller could use. Tokenizers are real (tiny word-level ones
written on the fly), so bucket selection, truncation and the token
accounting run exactly as in production.

The concurrent tests are written to be deterministic rather than fast: a
prediction is parked inside the engine's prediction lock, and the test
waits for an observable state change instead of sleeping and hoping.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from eeane import runtime
from eeane.config import ModelEntry
from eeane.engine import CoreMLEngine, NonFiniteOutputError, QueueTimeoutError, _ServedModel
from eeane.engine_pipeline import _InputRoute, _plan_groups
from eeane.engine_types import _as_rows, _collect_missing

# Width of the rows the stand-in artifacts answer with.
_WIDTH = 8

# Sequence-length buckets every stand-in model is compiled for, and the
# one a batched artifact is compiled for unless a test says otherwise.
_BUCKETS = (8, 16)
_BATCHED_BUCKETS = (8,)

# Seconds any wait in this module may take before the test gives up. Long
# enough never to be reached on a loaded machine, since it is only ever
# waited out when the code under test is broken.
_TIMEOUT = 10.0

# Seconds a request whose deadline is meant to pass is given. It is spent
# waiting for a lock a parked thread is known to hold, so the wait always
# ends in the timeout being reported, never in a race.
_SHORT_WAIT = 0.05

# How often the polling helper re-checks the engine's state.
_POLL_INTERVAL = 0.005

# Five texts of one, two, three, four and five tokens: they all fit the
# smallest bucket, and no two of them produce the same row, so an input
# placed at the wrong position is visible in the answer.
_SAME_BUCKET_TEXTS = ["a", "a b", "a b c", "a b a b", "a b c a b"]


class PredictGate:
    """Hook that parks a prediction inside the engine's prediction lock.

    The stand-in artifacts call it at the start of every prediction. The
    first caller reports that it has arrived and then waits until the
    test lets it go, which gives the test a window in which the engine is
    known to be busy predicting. Once released, later predictions run
    straight through.
    """

    def __init__(self) -> None:
        """Register a gate no prediction has reached yet."""
        self.entered = threading.Event()
        self.released = threading.Event()

    def pass_through(self) -> None:
        """Report that a prediction started and wait for the test.

        Raises:
            AssertionError: If the test never releases the gate, so a
                mistake shows up as a failure instead of a hung suite.
        """
        self.entered.set()
        if not self.released.wait(timeout=_TIMEOUT):
            raise AssertionError("the parked prediction was never released")

    def wait_for_entry(self) -> None:
        """Block until a prediction has reached the gate."""
        assert self.entered.wait(timeout=_TIMEOUT), "no prediction reached the artifacts"

    def release(self) -> None:
        """Let the parked prediction, and every later one, run."""
        self.released.set()


class StubArtifacts:
    """Stand-in for every compiled artifact of every model.

    One instance is shared by every bucket and every batch size the
    loader builds, so a test can watch a whole engine's predictions in
    one place. Answers are derived from each row's attention mask, so
    every input comes back with a value of its own and a row placed at
    the wrong position is visible.

    Attributes:
        shapes: ``(rows, S)`` shape of every prediction, in call order.
        gate: Hook run at the start of each prediction, or ``None``.
        fill: Value every output element carries, or ``None`` to derive
            it from each row's attention mask.
    """

    def __init__(self, clock: FakeClock | None = None, advance: float = 0.0) -> None:
        """Register artifacts that have served no prediction yet.

        Args:
            clock: Fake clock the predictions move forward.
            advance: Seconds one prediction takes on that clock.
        """
        self.shapes: list[tuple[int, int]] = []
        self.gate: PredictGate | None = None
        self.fill: float | None = None
        self._clock = clock
        self._advance = advance

    @property
    def count(self) -> int:
        """Return how many predictions have been served."""
        return len(self.shapes)

    @property
    def rows(self) -> list[int]:
        """Return the number of inputs of every prediction, in call order."""
        return [shape[0] for shape in self.shapes]

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Answer one prediction, obeying whatever the test asked for.

        Args:
            inputs: ``input_ids``/``attention_mask`` of shape
                ``(rows, S)``.

        Returns:
            Both output names eeANE knows about, with one row per input.
        """
        mask = inputs["attention_mask"]
        used = [int(row.sum()) for row in mask]
        self.shapes.append((int(inputs["input_ids"].shape[0]), int(inputs["input_ids"].shape[1])))
        if self.gate is not None:
            self.gate.pass_through()
        if self._clock is not None and self._advance:
            self._clock.advance(self._advance)
        if self.fill is not None:
            return {
                "embedding": np.full((len(used), _WIDTH), self.fill, dtype=np.float32),
                "logits": np.full((len(used), 1), self.fill, dtype=np.float32),
            }
        return {
            "embedding": np.stack([_expected_row(count) for count in used]),
            "logits": np.asarray([[float(count)] for count in used], dtype=np.float32),
        }


class FakeClock:
    """Monotonic clock a test moves forward by hand.

    Attributes:
        now: Current reading, in seconds.
    """

    def __init__(self, start: float = 0.0) -> None:
        """Start the clock at ``start`` seconds."""
        self.now = start

    def __call__(self) -> float:
        """Return the current reading, as the engine's clock does."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds``."""
        self.now += seconds


class RecordingLoader:
    """Replacement for the engine's artifact loader that never touches Core ML.

    Mirrors the real loader's decisions: the entry's own tokenizer, one
    stand-in per bucket, and one per batched bucket for an embedding
    entry that configures any.

    Attributes:
        calls: Model id of every load, in call order.
    """

    def __init__(self, artifacts: StubArtifacts) -> None:
        """Register a loader handing out ``artifacts`` for every bucket.

        Args:
            artifacts: Stand-in every loaded bucket is served by.
        """
        self.calls: list[str] = []
        self._artifacts = artifacts

    def __call__(self, entry: ModelEntry) -> _ServedModel:
        """Build the stand-in artifacts for one entry, recording the call.

        Args:
            entry: Entry the engine asked for.

        Returns:
            A served model backed by the entry's real frozen tokenizer.
        """
        self.calls.append(entry.id)
        assert entry.tokenizer is not None
        kind = str(entry.kind)
        buckets = tuple(sorted(entry.artifacts or ()))
        batched = tuple(sorted(entry.batch_artifacts or ())) if kind == "embedding" else ()
        return _ServedModel(
            id=entry.id,
            kind=kind,
            tokenizer=runtime.load_frozen_tokenizer(entry.tokenizer),
            tokenizer_lock=threading.Lock(),
            compiled=dict.fromkeys(buckets, self._artifacts),
            compiled_batch=dict.fromkeys(batched, self._artifacts),
            buckets=buckets,
            output_name=str(entry.output_name),
            normalize=entry.normalize,
            embedding_dim=entry.embedding_dim if kind == "embedding" else None,
        )


class Worker(threading.Thread):
    """Thread running one engine call and keeping whatever came out of it.

    Attributes:
        result: What the call returned, or ``None`` if it raised.
        error: What the call raised, or ``None`` if it returned.
    """

    def __init__(self, call: Callable[[], Any]) -> None:
        """Register a thread that has not run ``call`` yet.

        Args:
            call: Engine call to run in this thread.
        """
        super().__init__(daemon=True)
        self._call = call
        self.result: Any = None
        self.error: Exception | None = None

    def run(self) -> None:
        """Run the call, keeping its result or its error for the test."""
        try:
            self.result = self._call()
        except Exception as error:
            self.error = error

    def finish(self) -> None:
        """Join the thread, failing the test if it never finishes."""
        self.join(timeout=_TIMEOUT)
        assert not self.is_alive(), "the worker thread did not finish"


def _expected_row(used_tokens: int) -> np.ndarray:
    """Build the row the stand-in artifacts answer one input with.

    Args:
        used_tokens: Tokens the model was fed for that input.

    Returns:
        A 1-D float32 row of :data:`_WIDTH` values that no other token
        count produces.
    """
    return np.arange(_WIDTH, dtype=np.float32) + float(used_tokens)


def _write_toy_tokenizer(path: Path) -> Path:
    """Write a minimal word-level tokenizer.json carrying pad settings.

    Args:
        path: Destination file.

    Returns:
        ``path``, for chaining.
    """
    vocab = {"<pad>": 0, "a": 1, "b": 2, "c": 3}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    tokenizer.save(str(path))
    return path


def _make_entry(
    root: Path,
    model_id: str,
    kind: str = "embedding",
    *,
    buckets: Sequence[int] = _BUCKETS,
    batched_buckets: Sequence[int] = (),
    create_batched: bool = True,
) -> ModelEntry:
    """Build a model entry together with the files it points at.

    Args:
        root: Directory holding one sub-directory per model.
        model_id: Id the entry is routed by.
        kind: ``"embedding"`` or ``"reranker"``.
        buckets: Sequence lengths to create artifacts for.
        batched_buckets: Sequence lengths to create batched artifacts
            for; empty for a model served one input at a time.
        create_batched: Whether the batched artifacts are created on
            disk, so a test can build an entry that points at one that
            was never compiled.

    Returns:
        A validated entry whose artifacts exist (unless a test asked for
        a batched one that does not).
    """
    model_dir = root / model_id
    tokenizer_path = model_dir / "tokenizer.json"
    artifacts = {bucket: model_dir / f"s{bucket}.mlmodelc" for bucket in buckets}
    batched = {bucket: model_dir / f"s{bucket}_b2.mlmodelc" for bucket in batched_buckets}
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_toy_tokenizer(tokenizer_path)
    for path in artifacts.values():
        # A compiled Core ML artifact is a directory, not a file.
        path.mkdir(exist_ok=True)
    if create_batched:
        for path in batched.values():
            path.mkdir(exist_ok=True)
    fields: dict[str, object] = {
        "id": model_id,
        "kind": kind,
        "tokenizer": tokenizer_path,
        "artifacts": artifacts,
    }
    if batched:
        fields["batch_artifacts"] = batched
    return ModelEntry(**fields)


def _reference_embedding(entry: ModelEntry, texts: Sequence[str]) -> dict[str, Any]:
    """Work out what an inline, unpipelined loop would answer ``texts`` with.

    Written against the runtime helpers and its own copy of the
    tokenizer, so it is an independent reference rather than a replay of
    whatever the engine did -- and grouping must not change any of it.

    Args:
        entry: Entry the request is served by.
        texts: Input texts in request order.

    Returns:
        The expected value of every field of the embedding batch.
    """
    assert entry.tokenizer is not None
    tokenizer = runtime.load_frozen_tokenizer(entry.tokenizer)
    buckets = tuple(sorted(entry.artifacts or ()))
    expected: dict[str, Any] = {
        "vectors": [],
        "used_tokens": [],
        "orig_tokens": [],
        "buckets": [],
        "truncated_indices": [],
    }
    for index, text in enumerate(texts):
        n_tokens = runtime.count_text_tokens(tokenizer, text)
        bucket, truncated = runtime.select_bucket(n_tokens, buckets)
        inputs = runtime.tokenize_texts(tokenizer, [text], bucket)
        used = int(inputs["attention_mask"].sum())
        expected["vectors"].append(_expected_row(used))
        expected["used_tokens"].append(used)
        expected["orig_tokens"].append(n_tokens)
        expected["buckets"].append(bucket)
        if truncated:
            expected["truncated_indices"].append(index)
    expected["vectors"] = np.stack(expected["vectors"])
    return expected


def _assert_embedding_matches(batch: Any, expected: dict[str, Any]) -> None:
    """Fail unless every field of ``batch`` is the expected one.

    Args:
        batch: Batch the engine answered with.
        expected: Reference fields (see :func:`_reference_embedding`).
    """
    assert np.array_equal(batch.vectors, expected["vectors"])
    assert batch.used_tokens == expected["used_tokens"]
    assert batch.orig_tokens == expected["orig_tokens"]
    assert batch.buckets == expected["buckets"]
    assert batch.truncated_indices == expected["truncated_indices"]


def _wait_for_in_flight(engine: CoreMLEngine, model_id: str, count: int) -> None:
    """Block until ``count`` requests are counted against ``model_id``.

    A request is counted in before it starts waiting, so this is how a
    test knows that a second request has reached the engine -- without
    guessing how long that takes.

    Args:
        engine: Engine under test.
        model_id: Model the requests routed to.
        count: Number of requests to wait for.
    """
    deadline = time.monotonic() + _TIMEOUT
    while engine._models[model_id].in_flight < count:
        assert time.monotonic() < deadline, f"only {engine._models[model_id].in_flight} arrived"
        time.sleep(_POLL_INTERVAL)


@pytest.fixture
def clock() -> FakeClock:
    """Clock the test moves forward instead of waiting."""
    return FakeClock()


@pytest.fixture
def artifacts() -> StubArtifacts:
    """Stand-in artifacts shared by every bucket of every model."""
    return StubArtifacts()


@pytest.fixture
def loader(artifacts: StubArtifacts) -> RecordingLoader:
    """Artifact loader that records its calls and loads nothing real."""
    return RecordingLoader(artifacts)


@pytest.fixture
def batched_entries(tmp_path: Path) -> list[ModelEntry]:
    """An embedding model with a batched artifact for its smallest bucket."""
    return [
        _make_entry(tmp_path, "emb-a", batched_buckets=_BATCHED_BUCKETS),
        _make_entry(tmp_path, "rr-a", kind="reranker"),
    ]


@pytest.fixture
def plain_entries(tmp_path: Path) -> list[ModelEntry]:
    """The same deployment without any batched artifact."""
    return [
        _make_entry(tmp_path, "emb-a"),
        _make_entry(tmp_path, "rr-a", kind="reranker"),
    ]


@pytest.fixture
def build_engine() -> Iterator[Callable[..., CoreMLEngine]]:
    """Build engines that are closed again when the test ends.

    Background sweeping is off, so no thread interferes with the timing a
    test sets up.
    """
    built: list[CoreMLEngine] = []

    def factory(model_entries: Sequence[ModelEntry], **kwargs: object) -> CoreMLEngine:
        kwargs.setdefault("sweep_interval", 0.0)
        engine = CoreMLEngine(model_entries, **kwargs)  # type: ignore[arg-type]
        built.append(engine)
        return engine

    yield factory
    for engine in built:
        engine.close()


# --- shaping one prediction's output --------------------------------------


def test_as_rows_splits_a_batched_output_into_one_row_per_input() -> None:
    """A prediction over several inputs must come back as one row each."""
    output = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    rows = _as_rows(output, "embedding", 2)

    assert rows.shape == (2, 3)
    assert np.array_equal(rows[0], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.array_equal(rows[1], np.asarray([4.0, 5.0, 6.0], dtype=np.float32))


def test_as_rows_accepts_a_single_input_and_a_flat_output() -> None:
    """A group of one, and an output of any shape, must still split cleanly."""
    rows = _as_rows([1.0, 2.0, 3.0, 4.0], "embedding", 1)
    pairs = _as_rows([1.0, 2.0, 3.0, 4.0], "embedding", 2)

    assert rows.shape == (1, 4)
    assert pairs.shape == (2, 2)
    assert rows.dtype == np.float32 and pairs.dtype == np.float32


def test_as_rows_rejects_an_empty_output() -> None:
    """An output holding nothing is not one row per input, whatever the count."""
    with pytest.raises(RuntimeError, match="empty"):
        _as_rows(np.empty((0, 4), dtype=np.float32), "embedding", 2)


def test_as_rows_rejects_an_output_that_is_not_whole_rows() -> None:
    """A model answering with fewer values than inputs must be reported, not reshaped."""
    with pytest.raises(RuntimeError, match="rows"):
        _as_rows([1.0, 2.0, 3.0], "embedding", 2)


def test_as_rows_rejects_a_prediction_of_no_inputs() -> None:
    """A prediction always carries at least one input."""
    with pytest.raises(ValueError, match="at least one"):
        _as_rows([1.0, 2.0], "embedding", 0)


# --- planning the groups --------------------------------------------------


def _routes(*buckets: int) -> list[_InputRoute]:
    """Build one routing decision per bucket, in request order."""
    return [_InputRoute(bucket=bucket, n_tokens=1, truncated=False) for bucket in buckets]


def test_plan_groups_pairs_inputs_of_a_batched_bucket() -> None:
    """Inputs of one bucket must be paired up in request order, the odd one alone."""
    groups = _plan_groups(_routes(8, 8, 8, 8, 8), {8})

    assert [group.indices for group in groups] == [(0, 1), (2, 3), (4,)]
    assert {group.bucket for group in groups} == {8}


def test_plan_groups_leaves_a_bucket_without_a_batched_artifact_alone() -> None:
    """Only the buckets a batched artifact is loaded for are ever grouped."""
    groups = _plan_groups(_routes(8, 16, 8, 16, 8), {8})

    assert [group.indices for group in groups] == [(0, 2), (1,), (3,), (4,)]
    assert [group.bucket for group in groups] == [8, 16, 16, 8]


def test_plan_groups_covers_every_input_exactly_once() -> None:
    """Grouping must never drop, duplicate or reorder a request's inputs."""
    groups = _plan_groups(_routes(16, 8, 16, 8, 16, 8, 8), {8, 16})

    covered = [index for group in groups for index in group.indices]
    assert sorted(covered) == list(range(7))
    # Groups are handed out in the order of their first input.
    assert [group.indices[0] for group in groups] == sorted(group.indices[0] for group in groups)


def test_plan_groups_without_any_batched_bucket_predicts_one_by_one() -> None:
    """A model served without batched artifacts must plan single-input groups."""
    groups = _plan_groups(_routes(8, 8, 16), set())

    assert [group.indices for group in groups] == [(0,), (1,), (2,)]


# --- grouping a request ---------------------------------------------------


def test_inputs_of_a_batched_bucket_are_predicted_two_at_a_time(
    batched_entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Five inputs of one batched bucket must be predicted (2, 2, 1)."""
    engine = build_engine(batched_entries, loader=loader)

    batch = engine.embed(list(_SAME_BUCKET_TEXTS), "emb-a")

    assert artifacts.shapes == [(2, 8), (2, 8), (1, 8)]
    assert batch.vectors.shape == (5, _WIDTH)


def test_a_grouped_request_answers_exactly_what_an_inline_loop_would(
    batched_entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Grouping must change how the answer is computed, not what it is."""
    engine = build_engine(batched_entries, loader=loader)
    expected = _reference_embedding(batched_entries[0], _SAME_BUCKET_TEXTS)

    batch = engine.embed(list(_SAME_BUCKET_TEXTS), "emb-a")

    _assert_embedding_matches(batch, expected)


def test_only_the_buckets_with_a_batched_artifact_are_grouped(
    batched_entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Inputs of a bucket without a batched artifact stay one per prediction."""
    engine = build_engine(batched_entries, loader=loader)
    # Two short inputs (bucket 8, batched) around two long ones (bucket
    # 16, not batched), plus a fifth short one with no partner left.
    texts = ["a", " ".join(["a"] * 10), "a b", " ".join(["b"] * 12), "a b c"]
    expected = _reference_embedding(batched_entries[0], texts)

    batch = engine.embed(list(texts), "emb-a")

    assert artifacts.shapes == [(2, 8), (1, 16), (1, 16), (1, 8)]
    _assert_embedding_matches(batch, expected)


def test_a_model_without_batched_artifacts_predicts_one_input_at_a_time(
    plain_entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """The path a model without batched artifacts takes must be unchanged."""
    engine = build_engine(plain_entries, loader=loader)
    expected = _reference_embedding(plain_entries[0], _SAME_BUCKET_TEXTS)

    batch = engine.embed(list(_SAME_BUCKET_TEXTS), "emb-a")

    assert artifacts.shapes == [(1, 8)] * 5
    _assert_embedding_matches(batch, expected)


def test_a_single_input_request_is_predicted_on_its_own_artifact(
    batched_entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """One input has nobody to pair with, so it runs on the ordinary artifact."""
    engine = build_engine(batched_entries, loader=loader)

    batch = engine.embed(["a b"], "emb-a")

    assert artifacts.shapes == [(1, 8)]
    assert batch.vectors.shape == (1, _WIDTH)


def test_used_tokens_are_counted_per_row_of_a_group(
    batched_entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Inputs padded together must still report their own token counts."""
    engine = build_engine(batched_entries, loader=loader)

    batch = engine.embed(["a", "a b c a b"], "emb-a")

    assert batch.used_tokens == [1, 5]
    assert batch.orig_tokens == [1, 5]
    assert batch.buckets == [8, 8]
    # Every row is the one its own input produced, not the group's first.
    assert np.array_equal(batch.vectors[0], _expected_row(1))
    assert np.array_equal(batch.vectors[1], _expected_row(5))


def test_truncation_is_reported_for_the_input_that_did_not_fit(
    batched_entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """An input past the largest bucket is truncated and reported, grouped or not."""
    engine = build_engine(batched_entries, loader=loader)
    # Both run on the largest bucket, which has no batched artifact; only
    # the second one outgrows it.
    texts = [" ".join(["a"] * 12), " ".join(["b"] * 40)]
    expected = _reference_embedding(batched_entries[0], texts)

    batch = engine.embed(list(texts), "emb-a")

    assert batch.truncated_indices == [1]
    assert batch.used_tokens == [12, 16]
    assert artifacts.shapes == [(1, 16), (1, 16)]
    _assert_embedding_matches(batch, expected)


def test_truncation_inside_a_group_is_reported_for_the_input_that_did_not_fit(
    tmp_path: Path,
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Grouped inputs are truncated and accounted for one by one, as single ones are."""
    entry = _make_entry(tmp_path, "emb-a", batched_buckets=_BUCKETS)
    engine = build_engine([entry], loader=loader)
    # Both run on the largest bucket -- which is batched here -- and only
    # the first one outgrows it.
    texts = [" ".join(["a"] * 40), " ".join(["b"] * 12)]
    expected = _reference_embedding(entry, texts)

    batch = engine.embed(list(texts), "emb-a")

    assert artifacts.shapes == [(2, 16)]
    assert batch.truncated_indices == [0]
    assert batch.used_tokens == [16, 12]
    assert batch.orig_tokens == [40, 12]
    _assert_embedding_matches(batch, expected)


def test_an_empty_request_is_answered_without_predicting_anything(
    batched_entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """A request with no texts must keep its (0, D) contract and load nothing."""
    engine = build_engine(batched_entries, loader=loader)

    batch = engine.embed([], "emb-a")

    assert batch.vectors.shape == (0, 0)
    assert artifacts.count == 0


def test_reranking_is_never_grouped(
    batched_entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """A reranker is served one pair per prediction, whatever the embedding model does."""
    engine = build_engine(batched_entries, loader=loader)

    scores = engine.rerank("a", ["b", "c", "a b"], "rr-a")

    assert artifacts.rows == [1, 1, 1]
    assert scores.logits.shape == (3,)


# --- the output guard -----------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_row_of_a_group_fails_the_request(
    batched_entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
    value: float,
) -> None:
    """An unusable answer must be reported for a grouped prediction too."""
    engine = build_engine(batched_entries, loader=loader)
    artifacts.fill = value

    with pytest.raises(NonFiniteOutputError) as raised:
        engine.embed(["a", "a b"], "emb-a")

    assert raised.value.model_id == "emb-a"
    # Both inputs are short, so the group ran on the smallest bucket.
    assert raised.value.bucket == 8
    assert artifacts.shapes == [(2, 8)]
    assert engine._models["emb-a"].in_flight == 0
    assert engine._inflight == {}


# --- deadlines ------------------------------------------------------------


def test_a_grouped_request_gives_up_while_it_waits_for_the_engine(
    batched_entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """The first prediction of a grouped request is bounded by the deadline."""
    engine = build_engine(batched_entries, loader=loader, clock=clock)
    gate = PredictGate()
    artifacts.gate = gate

    blocker = Worker(lambda: engine.embed(["a b"], "emb-a"))
    blocker.start()
    gate.wait_for_entry()
    # Other texts, so this request waits for the prediction lock instead
    # of attaching to the parked request.
    with pytest.raises(QueueTimeoutError):
        engine.embed(["a b c", "a b c a"], "emb-a", deadline=clock.now + _SHORT_WAIT)
    gate.release()
    blocker.finish()

    assert blocker.error is None
    # The request that gave up must have predicted nothing at all.
    assert artifacts.count == 1
    assert engine._models["emb-a"].in_flight == 0
    assert engine._inflight == {}


def test_a_grouped_request_that_has_started_predicting_ignores_its_deadline(
    batched_entries: list[ModelEntry],
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Only the wait before the first group is bounded; the rest always runs."""
    artifacts = StubArtifacts(clock=clock, advance=100.0)
    engine = build_engine(batched_entries, loader=RecordingLoader(artifacts), clock=clock)

    batch = engine.embed(list(_SAME_BUCKET_TEXTS), "emb-a", deadline=clock.now + 10.0)

    # The first prediction alone took the request past its deadline, and
    # the two groups that follow it still ran.
    assert clock.now == pytest.approx(300.0)
    assert artifacts.shapes == [(2, 8), (2, 8), (1, 8)]
    assert batch.vectors.shape == (5, _WIDTH)


# --- coalescing -----------------------------------------------------------


def test_two_identical_grouped_requests_run_one_computation(
    batched_entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Coalescing still serves both requests from one request's predictions."""
    engine = build_engine(batched_entries, loader=loader)
    gate = PredictGate()
    artifacts.gate = gate

    leader = Worker(lambda: engine.embed(list(_SAME_BUCKET_TEXTS), "emb-a"))
    leader.start()
    gate.wait_for_entry()
    waiter = Worker(lambda: engine.embed(list(_SAME_BUCKET_TEXTS), "emb-a"))
    waiter.start()
    # The second request is counted against the model as soon as it has
    # attached, so this is the point where it is known to be waiting.
    _wait_for_in_flight(engine, "emb-a", 2)
    gate.release()
    leader.finish()
    waiter.finish()

    assert (leader.error, waiter.error) == (None, None)
    assert waiter.result is leader.result
    # Three predictions for five inputs: one request's worth, not two.
    assert artifacts.shapes == [(2, 8), (2, 8), (1, 8)]
    assert engine._inflight == {}


# --- start-up validation --------------------------------------------------


def test_a_missing_batched_artifact_is_reported_at_start_up(tmp_path: Path) -> None:
    """A configured batched artifact that was never compiled must fail the start-up."""
    entry = _make_entry(tmp_path, "emb-a", batched_buckets=_BATCHED_BUCKETS, create_batched=False)

    problems = _collect_missing(entry)

    assert len(problems) == 1
    assert "s8_b2.mlmodelc" in problems[0]
    assert "emb-a" in problems[0]
    # The hint names the run that produces a batched artifact.
    assert "--batch 2" in problems[0]


def test_an_engine_refuses_to_start_without_the_batched_artifact(
    tmp_path: Path, loader: RecordingLoader, build_engine: Callable[..., CoreMLEngine]
) -> None:
    """The missing artifact is reported before anything is loaded."""
    entry = _make_entry(tmp_path, "emb-a", batched_buckets=_BATCHED_BUCKETS, create_batched=False)

    with pytest.raises(RuntimeError, match="s8_b2.mlmodelc"):
        build_engine([entry], loader=loader)

    assert loader.calls == []
