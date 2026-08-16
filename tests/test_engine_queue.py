"""Tests for the engine's request coalescing, deadlines and output guard.

Nothing here needs a Neural Engine: the artifact loader is injected, so
the stand-in artifacts answer ``predict`` from memory and can be told to
hold a prediction, to fail, or to answer with values no caller could use.
Tokenizers are real (tiny word-level ones written on the fly), so bucket
selection and the token accounting run exactly as in production.

The concurrent tests are written to be deterministic rather than fast:
one thread is parked inside a prediction, which keeps the engine's
prediction lock held for as long as the test needs, and the test waits
for an observable state change (a second request counted against the
model) instead of sleeping and hoping.
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
from eeane.engine import (
    CoreMLEngine,
    ModelPolicy,
    NonFiniteOutputError,
    QueueTimeoutError,
    _ServedModel,
)

# Width of the row the stand-in artifacts return under the "embedding" key.
_WIDTH = 8

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
    """Stand-in for the compiled artifacts of every bucket of every model.

    One instance is shared by every bucket the loader builds, so a test
    can count the predictions of a whole engine in one place and change
    what they answer at any point.

    Attributes:
        predictions: Sequence length of every prediction, in call order.
        gate: Hook run at the start of each prediction, or ``None``.
        fill: Value every output element carries, or ``None`` to derive
            it from the attention mask (which makes two different inputs
            answer differently).
        error: Exception raised instead of answering, or ``None``.
    """

    def __init__(self, clock: FakeClock | None = None, advance: float = 0.0) -> None:
        """Register artifacts that have served no prediction yet.

        Args:
            clock: Fake clock the predictions move forward.
            advance: Seconds one prediction takes on that clock.
        """
        self.predictions: list[int] = []
        self.gate: PredictGate | None = None
        self.fill: float | None = None
        self.error: BaseException | None = None
        self._clock = clock
        self._advance = advance

    @property
    def count(self) -> int:
        """Return how many predictions have been served."""
        return len(self.predictions)

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Answer one prediction, obeying whatever the test asked for.

        Args:
            inputs: ``input_ids``/``attention_mask`` of shape ``(1, S)``.

        Returns:
            Both output names eeANE knows about, filled with one value.

        Raises:
            BaseException: Whatever :attr:`error` holds, standing in for
                a model that cannot answer.
        """
        used = int(inputs["attention_mask"].sum())
        self.predictions.append(int(inputs["input_ids"].shape[1]))
        if self.gate is not None:
            self.gate.pass_through()
        if self._clock is not None and self._advance:
            self._clock.advance(self._advance)
        if self.error is not None:
            raise self.error
        value = float(used) if self.fill is None else self.fill
        return {
            "embedding": np.full((1, _WIDTH), value, dtype=np.float32),
            "logits": np.asarray([[value]], dtype=np.float32),
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
        return _ServedModel(
            id=entry.id,
            kind=kind,
            tokenizer=runtime.load_frozen_tokenizer(entry.tokenizer),
            tokenizer_lock=threading.Lock(),
            compiled=dict.fromkeys(buckets, self._artifacts),
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
    buckets: Sequence[int] = (8, 16),
) -> ModelEntry:
    """Build a model entry together with the files it points at.

    Args:
        root: Directory holding one sub-directory per model.
        model_id: Id the entry is routed by.
        kind: ``"embedding"`` or ``"reranker"``.
        buckets: Sequence lengths to create artifacts for.

    Returns:
        A validated entry whose artifacts all exist.
    """
    model_dir = root / model_id
    tokenizer_path = model_dir / "tokenizer.json"
    artifacts = {bucket: model_dir / f"s{bucket}.mlmodelc" for bucket in buckets}
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_toy_tokenizer(tokenizer_path)
    for path in artifacts.values():
        # A compiled Core ML artifact is a directory, not a file.
        path.mkdir(exist_ok=True)
    fields: dict[str, object] = {
        "id": model_id,
        "kind": kind,
        "tokenizer": tokenizer_path,
        "artifacts": artifacts,
    }
    return ModelEntry(**fields)


def _on_demand(*model_ids: str, keep_alive: int = 300) -> dict[str, ModelPolicy]:
    """Build an on-demand policy for each of ``model_ids``.

    Args:
        model_ids: Ids to serve on demand; any id left out stays
            resident, which is the engine's default.
        keep_alive: Idle delay applied to all of them.

    Returns:
        The policy mapping to hand to the engine.
    """
    return {model_id: ModelPolicy("on_demand", keep_alive) for model_id in model_ids}


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
def entries(tmp_path: Path) -> list[ModelEntry]:
    """One embedding model and one reranker, both with existing artifacts."""
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


# --- coalescing ----------------------------------------------------------


def test_two_identical_requests_run_one_prediction(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """A request arriving while an identical one runs must be served from it."""
    engine = build_engine(entries, loader=loader)
    gate = PredictGate()
    artifacts.gate = gate

    leader = Worker(lambda: engine.embed(["a b"], "emb-a"))
    leader.start()
    gate.wait_for_entry()
    waiter = Worker(lambda: engine.embed(["a b"], "emb-a"))
    waiter.start()
    # The second request is counted against the model as soon as it has
    # attached, so this is the point where it is known to be waiting.
    _wait_for_in_flight(engine, "emb-a", 2)
    gate.release()
    leader.finish()
    waiter.finish()

    assert (leader.error, waiter.error) == (None, None)
    assert artifacts.count == 1
    # Both requests are served the very same batch, which is exactly why
    # callers must treat it as read-only.
    assert waiter.result is leader.result
    assert leader.result.vectors.shape == (1, _WIDTH)
    assert engine._inflight == {}
    assert engine._models["emb-a"].in_flight == 0


def test_requests_with_different_texts_each_run_their_own_prediction(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Coalescing must never answer a request with another request's texts."""
    engine = build_engine(entries, loader=loader)
    gate = PredictGate()
    artifacts.gate = gate

    first = Worker(lambda: engine.embed(["a b"], "emb-a"))
    first.start()
    gate.wait_for_entry()
    second = Worker(lambda: engine.embed(["a b c"], "emb-a"))
    second.start()
    _wait_for_in_flight(engine, "emb-a", 2)
    gate.release()
    first.finish()
    second.finish()

    assert (first.error, second.error) == (None, None)
    assert artifacts.count == 2
    # The stand-in answers with the token count, so the two requests must
    # not have been given the same row.
    assert not np.array_equal(first.result.vectors, second.result.vectors)
    assert engine._inflight == {}


def test_coalescing_can_be_switched_off(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """With coalescing off, identical requests each compute their own answer."""
    engine = build_engine(entries, loader=loader, coalesce=False)
    gate = PredictGate()
    artifacts.gate = gate

    first = Worker(lambda: engine.embed(["a b"], "emb-a"))
    first.start()
    gate.wait_for_entry()
    second = Worker(lambda: engine.embed(["a b"], "emb-a"))
    second.start()
    _wait_for_in_flight(engine, "emb-a", 2)
    gate.release()
    first.finish()
    second.finish()

    assert (first.error, second.error) == (None, None)
    assert artifacts.count == 2
    assert first.result is not second.result
    assert np.array_equal(first.result.vectors, second.result.vectors)
    # Nothing is registered at all when coalescing is off.
    assert engine._inflight == {}


def test_the_leaders_failure_is_raised_by_the_waiting_request(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Identical requests fail identically, so the waiter inherits the error."""
    engine = build_engine(entries, loader=loader)
    gate = PredictGate()
    artifacts.gate = gate
    artifacts.error = RuntimeError("the model gave up")

    leader = Worker(lambda: engine.embed(["a b"], "emb-a"))
    leader.start()
    gate.wait_for_entry()
    waiter = Worker(lambda: engine.embed(["a b"], "emb-a"))
    waiter.start()
    _wait_for_in_flight(engine, "emb-a", 2)
    gate.release()
    leader.finish()
    waiter.finish()

    assert isinstance(leader.error, RuntimeError)
    assert waiter.error is leader.error
    assert artifacts.count == 1
    # A failed computation must leave nothing behind for later requests.
    assert engine._inflight == {}
    assert engine._models["emb-a"].in_flight == 0


def test_two_identical_rerank_requests_run_one_prediction(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """The reranker path coalesces on the query and the documents together."""
    engine = build_engine(entries, loader=loader)
    gate = PredictGate()
    artifacts.gate = gate

    leader = Worker(lambda: engine.rerank("a", ["b c"], "rr-a"))
    leader.start()
    gate.wait_for_entry()
    waiter = Worker(lambda: engine.rerank("a", ["b c"], "rr-a"))
    waiter.start()
    _wait_for_in_flight(engine, "rr-a", 2)
    gate.release()
    leader.finish()
    waiter.finish()

    assert (leader.error, waiter.error) == (None, None)
    assert artifacts.count == 1
    assert waiter.result is leader.result
    assert leader.result.logits.shape == (1,)
    assert engine._inflight == {}


def test_a_rerank_with_another_query_is_not_coalesced(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """The same documents scored against another query are another request."""
    engine = build_engine(entries, loader=loader)
    gate = PredictGate()
    artifacts.gate = gate

    first = Worker(lambda: engine.rerank("a", ["b c"], "rr-a"))
    first.start()
    gate.wait_for_entry()
    second = Worker(lambda: engine.rerank("c", ["b c"], "rr-a"))
    second.start()
    _wait_for_in_flight(engine, "rr-a", 2)
    gate.release()
    first.finish()
    second.finish()

    assert (first.error, second.error) == (None, None)
    assert artifacts.count == 2
    assert first.result is not second.result
    assert engine._inflight == {}


def test_an_empty_request_is_never_coalesced(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """An empty request computes nothing, so it registers nothing either."""
    engine = build_engine(entries, policies=_on_demand("emb-a", "rr-a"), loader=loader)

    assert engine.embed([], "emb-a").vectors.shape == (0, 0)
    assert engine.rerank("a", [], "rr-a").logits.shape == (0,)

    assert engine._inflight == {}
    assert artifacts.count == 0
    assert loader.calls == []


def test_the_request_key_tells_apart_what_must_not_be_shared(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Two requests share a key only if kind, model and texts all match."""
    engine = build_engine(entries, loader=loader)
    key = engine._request_key

    assert key("embedding", "emb-a", ["a", "b"]) == key("embedding", "emb-a", ["a", "b"])
    # Framing every text with its length keeps a split from being forged.
    assert key("embedding", "emb-a", ["ab", "c"]) != key("embedding", "emb-a", ["a", "bc"])
    assert key("embedding", "emb-a", ["a", "b"]) != key("embedding", "emb-a", ["b", "a"])
    assert key("embedding", "emb-a", ["a"]) != key("embedding", "emb-b", ["a"])
    assert key("embedding", "emb-a", ["a"]) != key("reranker", "emb-a", ["a"])


def test_no_key_is_built_when_coalescing_is_off(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Switching coalescing off must keep requests out of the table entirely."""
    engine = build_engine(entries, loader=loader, coalesce=False)

    assert engine._request_key("embedding", "emb-a", ["a"]) is None


# --- deadlines -----------------------------------------------------------


@pytest.mark.parametrize("elapsed", [0.0, 1.0, 3600.0])
def test_an_expired_deadline_is_rejected_before_the_model_is_loaded(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
    elapsed: float,
) -> None:
    """A request that has already given up must not trigger an on-demand load."""
    engine = build_engine(entries, policies=_on_demand("emb-a", "rr-a"), loader=loader, clock=clock)

    with pytest.raises(QueueTimeoutError):
        engine.embed(["a b"], "emb-a", deadline=clock.now - elapsed)
    with pytest.raises(QueueTimeoutError):
        engine.rerank("a", ["b"], "rr-a", deadline=clock.now - elapsed)

    assert loader.calls == []
    assert engine.loaded("emb-a") is False
    assert engine.loaded("rr-a") is False
    assert artifacts.count == 0
    assert engine._inflight == {}


def test_a_deadline_in_the_future_serves_the_request_normally(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """A deadline nobody is late for must change nothing about the answer."""
    engine = build_engine(entries, loader=loader, clock=clock)

    batch = engine.embed(["a b"], "emb-a", deadline=clock.now + 30.0)
    scores = engine.rerank("a", ["b"], "rr-a", deadline=clock.now + 30.0)

    assert batch.vectors.shape == (1, _WIDTH)
    assert scores.logits.shape == (1,)
    assert artifacts.count == 2


def test_the_first_prediction_gives_up_when_the_engine_stays_busy(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """A request waiting for the engine must give up once its deadline passes."""
    engine = build_engine(entries, loader=loader, clock=clock)
    gate = PredictGate()
    artifacts.gate = gate

    blocker = Worker(lambda: engine.embed(["a b"], "emb-a"))
    blocker.start()
    gate.wait_for_entry()
    # Another text, so this request waits for the prediction lock instead
    # of attaching to the parked request.
    with pytest.raises(QueueTimeoutError):
        engine.embed(["a b c"], "emb-a", deadline=clock.now + _SHORT_WAIT)
    gate.release()
    blocker.finish()

    assert blocker.error is None
    # The request that gave up must have predicted nothing at all.
    assert artifacts.count == 1
    assert engine._models["emb-a"].in_flight == 0
    assert engine._inflight == {}


def test_a_waiting_request_gives_up_when_the_one_it_attached_to_is_slow(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """A deadline is honoured while attached to another request, too."""
    engine = build_engine(entries, loader=loader, clock=clock)
    gate = PredictGate()
    artifacts.gate = gate

    leader = Worker(lambda: engine.embed(["a b"], "emb-a"))
    leader.start()
    gate.wait_for_entry()
    with pytest.raises(QueueTimeoutError):
        engine.embed(["a b"], "emb-a", deadline=clock.now + _SHORT_WAIT)
    gate.release()
    leader.finish()

    assert leader.error is None
    assert artifacts.count == 1
    assert engine._models["emb-a"].in_flight == 0
    assert engine._inflight == {}


def test_a_request_that_has_started_predicting_ignores_its_deadline(
    entries: list[ModelEntry],
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Inference is never interrupted: only the wait before it is bounded."""
    artifacts = StubArtifacts(clock=clock, advance=100.0)
    engine = build_engine(entries, loader=RecordingLoader(artifacts), clock=clock)

    batch = engine.embed(["a b", "a b c", "a"], "emb-a", deadline=clock.now + 10.0)

    # The first prediction alone took the request past its deadline, and
    # the two that follow it still ran.
    assert clock.now == pytest.approx(300.0)
    assert artifacts.count == 3
    assert batch.vectors.shape == (3, _WIDTH)


# --- the output guard ----------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_embedding_fails_the_request(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
    value: float,
) -> None:
    """An unusable vector must be reported, not handed to the caller."""
    engine = build_engine(entries, loader=loader)
    artifacts.fill = value

    with pytest.raises(NonFiniteOutputError) as raised:
        engine.embed(["a b"], "emb-a")

    assert raised.value.model_id == "emb-a"
    # "a b" is two tokens, so it ran on the smallest bucket.
    assert raised.value.bucket == 8
    message = str(raised.value)
    assert "emb-a" in message
    assert "8" in message
    assert engine._models["emb-a"].in_flight == 0
    assert engine._inflight == {}


def test_a_non_finite_logit_fails_the_request(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """The reranker path guards its output the same way the embedding one does."""
    engine = build_engine(entries, loader=loader)
    artifacts.fill = float("nan")

    with pytest.raises(NonFiniteOutputError) as raised:
        engine.rerank("a", ["b"], "rr-a")

    assert raised.value.model_id == "rr-a"
    assert raised.value.bucket == 8
    assert "rr-a" in str(raised.value)
    assert engine._models["rr-a"].in_flight == 0


def test_a_finite_output_is_served_unchanged(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """The guard must leave a healthy answer exactly as the model produced it."""
    engine = build_engine(entries, loader=loader)

    batch = engine.embed(["a b"], "emb-a")
    scores = engine.rerank("a", ["b"], "rr-a")

    # The stand-in answers with the number of tokens the model consumed.
    assert np.array_equal(batch.vectors, np.full((1, _WIDTH), 2.0, dtype=np.float32))
    assert batch.used_tokens == [2]
    assert scores.logits.shape == (1,)
    assert artifacts.count == 2
