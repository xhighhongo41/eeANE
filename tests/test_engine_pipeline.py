"""Tests for the engine's one-input-ahead tokenization pipeline.

Nothing here needs a Neural Engine: the artifact loader is injected, so
the stand-in artifacts answer ``predict`` from memory and can be told to
park a prediction or to fail. Tokenizers are real (tiny word-level ones
written on the fly), so bucket selection, truncation and the token
accounting run exactly as in production, and every model is served with a
recording stand-in for its tokenizer lock: the engine takes that lock
around the whole of one input's tokenization, so one acquisition is one
tokenization and a test can see which inputs were tokenized, when, and on
which thread.

The concurrent tests are written to be deterministic rather than fast.
Nothing waits out a sleep and hopes: a prediction is parked inside the
prediction lock, and the test waits for an observable event (a
tokenization that has started, a prediction that has arrived) with a
generous timeout that is only ever reached when the code under test is
broken.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from eeane import runtime
from eeane.config import ModelEntry
from eeane.engine import _TOKENIZE_THREAD_PREFIX, CoreMLEngine, _ServedModel

# Width of the row the stand-in artifacts return under the "embedding" key.
_WIDTH = 8

# Sequence-length buckets every stand-in model is compiled for.
_BUCKETS = (8, 16)

# Seconds any wait in this module may take before the test gives up. Long
# enough never to be reached on a loaded machine, since it is only ever
# waited out when the code under test is broken.
_TIMEOUT = 10.0

# Seconds given to something that must *not* happen, i.e. how long a
# request that has to wait for a running tokenization is watched before
# it is accepted as still waiting.
_SETTLE = 0.1


class RecordingLock:
    """Stand-in for a model's tokenizer lock that records every tokenization.

    The engine holds a model's tokenizer lock for the whole of one
    input's counting, routing and tokenization, so one acquisition is one
    tokenization. A real lock does the excluding, so the tokenizer is
    protected exactly as in production; the recording only observes.

    A test can also park the n-th tokenization inside the lock, which
    gives it a window in which one tokenization is known to be running.

    Attributes:
        threads: Thread of every tokenization that started, in the order
            they started. They are the threads of this model's engine
            only, so a test can tell them from another engine's.
    """

    def __init__(self) -> None:
        """Register a lock no tokenization has taken yet."""
        self.threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._observed = threading.Condition()
        self._park_at: int | None = None
        self._released = threading.Event()

    def park_at(self, position: int) -> None:
        """Hold the ``position``-th tokenization inside the lock.

        Args:
            position: One-based position of the tokenization to park.
        """
        self._park_at = position

    def release(self) -> None:
        """Let the parked tokenization, and every later one, run."""
        self._released.set()

    def wait_for(self, count: int) -> None:
        """Block until ``count`` tokenizations have started.

        Args:
            count: Number of tokenizations to wait for.
        """
        with self._observed:
            started = self._observed.wait_for(lambda: len(self.threads) >= count, timeout=_TIMEOUT)
        assert started, f"only {self.started} tokenization(s) started, expected {count}"

    @property
    def started(self) -> int:
        """Return how many tokenizations have started so far."""
        with self._observed:
            return len(self.threads)

    @property
    def names(self) -> list[str]:
        """Return the name of the thread of every tokenization, in order."""
        with self._observed:
            return [thread.name for thread in self.threads]

    def __enter__(self) -> RecordingLock:
        """Take the real lock, record the tokenization and park it if asked."""
        self._lock.acquire()
        with self._observed:
            self.threads.append(threading.current_thread())
            position = len(self.threads)
            self._observed.notify_all()
        if self._park_at is not None and position == self._park_at:
            assert self._released.wait(timeout=_TIMEOUT), "the parked tokenization was never let go"
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release the real lock, whatever the tokenization did."""
        self._lock.release()


class PredictGate:
    """Hook that parks a prediction inside the engine's prediction lock.

    The stand-in artifacts call it at the start of every prediction. The
    first caller reports that it has arrived and then waits until the
    test lets it go, which gives the test a window in which the engine is
    known to be predicting. Once released, later predictions run straight
    through.
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
        """Block until a prediction has reached the artifacts."""
        assert self.entered.wait(timeout=_TIMEOUT), "no prediction reached the artifacts"

    def release(self) -> None:
        """Let the parked prediction, and every later one, run."""
        self.released.set()


class StubArtifacts:
    """Stand-in for the compiled artifacts of every bucket of every model.

    One instance is shared by every bucket the loader builds, so a test
    can watch the predictions of a whole engine in one place. Answers are
    derived from the attention mask, so the values a request comes back
    with depend on the input that produced them -- which is what makes an
    out-of-order pipeline visible.

    Attributes:
        seq_lens: Sequence length of every prediction, in call order.
        gate: Hook run at the start of each prediction, or ``None``.
        error: Exception raised instead of answering, or ``None``.
        error_at: Zero-based prediction the error is raised at.
    """

    def __init__(self) -> None:
        """Register artifacts that have served no prediction yet."""
        self.seq_lens: list[int] = []
        self.gate: PredictGate | None = None
        self.error: BaseException | None = None
        self.error_at = 0

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Answer one prediction, obeying whatever the test asked for.

        Args:
            inputs: ``input_ids``/``attention_mask`` of shape ``(1, S)``.

        Returns:
            Both output names eeANE knows about, derived from the number
            of tokens the model was fed.

        Raises:
            BaseException: Whatever :attr:`error` holds, standing in for
                a model that cannot answer.
        """
        used = int(inputs["attention_mask"].sum())
        self.seq_lens.append(int(inputs["input_ids"].shape[1]))
        if self.gate is not None:
            self.gate.pass_through()
        if self.error is not None and len(self.seq_lens) == self.error_at + 1:
            raise self.error
        return {
            "embedding": _expected_row(used),
            "logits": np.asarray([[float(used)]], dtype=np.float32),
        }


class RecordingLoader:
    """Replacement for the engine's artifact loader that never touches Core ML.

    Every model is served with the entry's real frozen tokenizer and with
    a recording tokenizer lock, kept per model id so a test can inspect
    the tokenizations of the model it cares about.

    Attributes:
        calls: Model id of every load, in call order.
        locks: Recording tokenizer lock of every loaded model, by id.
    """

    def __init__(self, artifacts: StubArtifacts) -> None:
        """Register a loader handing out ``artifacts`` for every bucket.

        Args:
            artifacts: Stand-in every loaded bucket is served by.
        """
        self.calls: list[str] = []
        self.locks: dict[str, RecordingLock] = {}
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
        lock = RecordingLock()
        self.locks[entry.id] = lock
        return _ServedModel(
            id=entry.id,
            kind=kind,
            tokenizer=runtime.load_frozen_tokenizer(entry.tokenizer),
            tokenizer_lock=lock,  # type: ignore[arg-type]
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


def _expected_row(used_tokens: int) -> np.ndarray:
    """Build the row the stand-in artifacts answer one input with.

    Args:
        used_tokens: Tokens the model was fed for that input.

    Returns:
        A ``(1, _WIDTH)`` float32 row that no other token count produces.
    """
    return (np.arange(_WIDTH, dtype=np.float32) + float(used_tokens)).reshape(1, -1)


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


def _reference_embedding(entry: ModelEntry, texts: Sequence[str]) -> dict[str, Any]:
    """Work out what an inline, unpipelined loop would answer ``texts`` with.

    Written against the runtime helpers and its own copy of the
    tokenizer, so it is an independent reference rather than a replay of
    whatever the engine did.

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
        expected["vectors"].append(_expected_row(used).reshape(-1))
        expected["used_tokens"].append(used)
        expected["orig_tokens"].append(n_tokens)
        expected["buckets"].append(bucket)
        if truncated:
            expected["truncated_indices"].append(index)
    expected["vectors"] = np.stack(expected["vectors"])
    return expected


def _reference_rerank(entry: ModelEntry, query: str, documents: Sequence[str]) -> dict[str, Any]:
    """Work out what an inline loop would score ``documents`` with.

    Args:
        entry: Reranker entry the request is served by.
        query: Query text of every pair.
        documents: Candidate documents in request order.

    Returns:
        The expected value of every field of the rerank batch.
    """
    assert entry.tokenizer is not None
    tokenizer = runtime.load_frozen_tokenizer(entry.tokenizer)
    buckets = tuple(sorted(entry.artifacts or ()))
    expected: dict[str, Any] = {
        "logits": [],
        "used_tokens": [],
        "orig_tokens": [],
        "truncated_indices": [],
    }
    for index, document in enumerate(documents):
        n_tokens = runtime.count_pair_tokens(tokenizer, query, document)
        bucket, truncated = runtime.select_bucket(n_tokens, buckets)
        inputs = runtime.tokenize_pairs(tokenizer, [(query, document)], bucket)
        used = int(inputs["attention_mask"].sum())
        expected["logits"].append(float(used))
        expected["used_tokens"].append(used)
        expected["orig_tokens"].append(n_tokens)
        if truncated:
            expected["truncated_indices"].append(index)
    expected["logits"] = np.asarray(expected["logits"], dtype=np.float32)
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


# Texts of the multi-input requests: five different token counts, one of
# them past the largest bucket, so an input handed out in the wrong order
# would change every field of the answer.
_TEXTS = ["a", "a b c", "a b", " ".join(["a", "b"] * 5), " ".join(["c"] * 20)]

# Documents of the multi-input rerank requests, chosen the same way.
_DOCUMENTS = ["b", "a b c a", " ".join(["a"] * 12), "c", " ".join(["b"] * 25)]


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


# --- order and values ----------------------------------------------------


def test_a_multi_input_request_answers_in_request_order(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Reading ahead must answer exactly what an inline loop would answer."""
    engine = build_engine(entries, loader=loader)

    batch = engine.embed(list(_TEXTS), "emb-a")

    _assert_embedding_matches(batch, _reference_embedding(entries[0], _TEXTS))
    # One prediction per input, in request order and on the input's own
    # bucket: a pipeline that reordered inputs would show up here too.
    assert artifacts.seq_lens == _reference_embedding(entries[0], _TEXTS)["buckets"]
    assert engine._models["emb-a"].in_flight == 0
    assert engine._inflight == {}


def test_a_multi_input_rerank_answers_in_request_order(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """The reranker path reads ahead without reordering its pairs either."""
    engine = build_engine(entries, loader=loader)

    batch = engine.rerank("a b", list(_DOCUMENTS), "rr-a")

    expected = _reference_rerank(entries[1], "a b", _DOCUMENTS)
    assert np.array_equal(batch.logits, expected["logits"])
    assert batch.used_tokens == expected["used_tokens"]
    assert batch.orig_tokens == expected["orig_tokens"]
    assert batch.truncated_indices == expected["truncated_indices"]
    assert engine._models["rr-a"].in_flight == 0


def test_every_input_of_a_multi_input_request_is_tokenized_once(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Reading ahead must not tokenize an input twice, nor skip one."""
    engine = build_engine(entries, loader=loader)

    engine.embed(list(_TEXTS), "emb-a")

    lock = loader.locks["emb-a"]
    assert lock.started == len(_TEXTS)
    # Every tokenization of a multi-input request runs on the engine's
    # shared workers, not on the thread waiting for the predictions.
    assert all(name.startswith(_TOKENIZE_THREAD_PREFIX) for name in lock.names)


def test_a_single_input_request_is_tokenized_inline(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """A request with nothing to read ahead of must not hand off to a worker."""
    engine = build_engine(entries, loader=loader)

    batch = engine.embed(["a b"], "emb-a")
    scores = engine.rerank("a", ["b c"], "rr-a")

    _assert_embedding_matches(batch, _reference_embedding(entries[0], ["a b"]))
    assert np.array_equal(scores.logits, _reference_rerank(entries[1], "a", ["b c"])["logits"])
    assert loader.locks["emb-a"].threads == [threading.current_thread()]
    assert loader.locks["rr-a"].threads == [threading.current_thread()]


# --- overlapping tokenization with prediction ----------------------------


def test_the_next_input_is_tokenized_before_the_current_prediction_ends(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """The point of the pipeline: tokenizing must overlap with predicting."""
    engine = build_engine(entries, loader=loader)
    gate = PredictGate()
    artifacts.gate = gate

    request = Worker(lambda: engine.embed(list(_TEXTS), "emb-a"))
    request.start()
    # The first prediction is parked inside the prediction lock, so it
    # cannot have finished before the gate is released below.
    gate.wait_for_entry()
    lock = loader.locks["emb-a"]
    lock.wait_for(2)
    # Read-ahead depth is one: the second input is being tokenized while
    # the first is predicted, and the third is not started before the
    # second prediction needs it.
    assert lock.started == 2
    gate.release()
    request.finish()

    assert request.error is None
    _assert_embedding_matches(request.result, _reference_embedding(entries[0], _TEXTS))


def test_concurrent_multi_input_requests_keep_their_own_answers(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Requests sharing the engine's workers must not share their inputs."""
    engine = build_engine(entries, loader=loader)
    requests = [list(_TEXTS[index:]) + list(_TEXTS[:index]) for index in range(3)]

    workers = [Worker(lambda texts=texts: engine.embed(texts, "emb-a")) for texts in requests]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.finish()

    for worker, texts in zip(workers, requests, strict=True):
        assert worker.error is None
        _assert_embedding_matches(worker.result, _reference_embedding(entries[0], texts))
    assert engine._models["emb-a"].in_flight == 0
    assert engine._inflight == {}


# --- a request that fails half-way ---------------------------------------


def test_a_failed_request_waits_for_the_tokenization_it_read_ahead(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """No worker may touch the tokenizer of a model the request has released."""
    engine = build_engine(entries, loader=loader)
    artifacts.error = RuntimeError("the model gave up")
    gate = PredictGate()
    # The first prediction is parked, then fails: parking it is what lets
    # the read-ahead tokenization of the second input really start before
    # the request gives up, instead of being cancelled unstarted.
    artifacts.gate = gate
    lock = loader.locks["emb-a"]
    lock.park_at(2)

    request = Worker(lambda: engine.embed(list(_TEXTS), "emb-a"))
    request.start()
    gate.wait_for_entry()
    lock.wait_for(2)
    # From here the second input is known to be tokenizing, and the first
    # prediction is known to be about to fail.
    gate.release()
    request.join(timeout=_SETTLE)

    # The failing request must not let go of the model while a worker is
    # still using its tokenizer.
    assert request.is_alive(), "the request released its model with a tokenization running"
    assert engine._models["emb-a"].in_flight == 1
    lock.release()
    request.finish()

    assert isinstance(request.error, RuntimeError)
    assert engine._models["emb-a"].in_flight == 0
    assert engine._inflight == {}


def test_a_failed_request_leaves_no_tokenization_behind(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """A request that gives up must start no tokenization it will never use."""
    engine = build_engine(entries, loader=loader)
    artifacts.error = RuntimeError("the model gave up")

    with pytest.raises(RuntimeError):
        engine.embed(list(_TEXTS), "emb-a")
    lock = loader.locks["emb-a"]
    given_up_at = lock.started

    # Closing waits for the workers, so anything the failed request had
    # left queued would have run by the time close() returns.
    engine.close()

    # The first input, plus at most the one read ahead of it (which is
    # cancelled when the failure beats a worker to it): the inputs behind
    # them are never tokenized, before or after the request gave up.
    assert given_up_at <= 2
    assert lock.started == given_up_at
    assert engine._models["emb-a"].in_flight == 0
    assert engine._inflight == {}


def test_a_request_after_a_failure_is_served_normally(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    artifacts: StubArtifacts,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """A failure must leave the engine's workers usable for the next request."""
    engine = build_engine(entries, loader=loader)
    artifacts.error = RuntimeError("the model gave up")

    with pytest.raises(RuntimeError):
        engine.embed(list(_TEXTS), "emb-a")
    lock = loader.locks["emb-a"]
    given_up_at = lock.started
    artifacts.error = None
    batch = engine.embed(list(_TEXTS), "emb-a")

    _assert_embedding_matches(batch, _reference_embedding(entries[0], _TEXTS))
    # The retry tokenized every one of its own inputs, and none of the
    # failed request's leftovers were served to it.
    assert lock.started == given_up_at + len(_TEXTS)


# --- shutdown ------------------------------------------------------------


def test_close_stops_the_tokenize_workers(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Closing the engine must leave no tokenize thread running behind it."""
    engine = build_engine(entries, loader=loader)
    engine.embed(list(_TEXTS), "emb-a")
    workers = loader.locks["emb-a"].threads
    assert workers, "the request tokenized nothing"

    engine.close()

    # The threads this engine's request really ran on, so another
    # engine's workers cannot make this pass or fail.
    assert not [thread for thread in workers if thread.is_alive()]
    with pytest.raises(RuntimeError):
        engine._tokenize_pool.submit(lambda: None)


def test_close_is_idempotent(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Closing twice must be as harmless as closing once."""
    engine = build_engine(entries, loader=loader)
    engine.embed(list(_TEXTS), "emb-a")

    engine.close()
    engine.close()

    assert engine._models["emb-a"].in_flight == 0


def test_a_request_after_close_is_tokenized_inline(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """A closed engine still answers, one input at a time, on the caller's thread."""
    engine = build_engine(entries, loader=loader)
    engine.close()

    batch = engine.embed(list(_TEXTS), "emb-a")

    _assert_embedding_matches(batch, _reference_embedding(entries[0], _TEXTS))
    lock = loader.locks["emb-a"]
    assert lock.threads == [threading.current_thread()] * len(_TEXTS)
