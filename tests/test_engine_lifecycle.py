"""Tests for the engine's load/unload lifecycle, without a Neural Engine.

The artifact loader is injected, so a model can be "loaded" and unloaded
as often as a test likes on any machine: the stand-in pairs the entry's
real frozen tokenizer (so bucket selection and the token accounting run
exactly as in production) with in-memory stand-ins for the compiled
artifacts, and records every call. The engine's clock is injected too,
which makes the idle accounting deterministic and lets the unit tests run
without a background thread; the one test that does exercise the
background sweeper polls with a deadline instead.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from eeane import engine as engine_module
from eeane import runtime
from eeane.config import EeaneConfig, ModelEntry, ServerConfig
from eeane.engine import CoreMLEngine, ModelPolicy, _ServedModel

# Width of the row the stand-in artifacts return under the "embedding" key.
_WIDTH = 8

# Embedding width some entries state up front, standing in for the value
# the compiled-model cache records.
_CONFIGURED_WIDTH = 768


class StubCompiledModel:
    """Stand-in for a loaded ``CompiledMLModel``.

    Answers with both output names eeANE knows about, derived from the
    attention mask, so a test can tell that a request really reached the
    artifacts.

    Attributes:
        clock: Fake clock moved forward on every prediction, or ``None``
            to leave the clock alone.
        advance: Seconds one prediction takes on that clock, standing in
            for a long-running request.
    """

    def __init__(self, clock: FakeClock | None = None, advance: float = 0.0) -> None:
        """Register an artifact that has served no prediction yet."""
        self.clock = clock
        self.advance = advance

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return deterministic outputs derived from the attention mask."""
        used = int(inputs["attention_mask"].sum())
        if self.clock is not None and self.advance:
            self.clock.advance(self.advance)
        return {
            "embedding": (np.arange(_WIDTH, dtype=np.float32) + float(used)).reshape(1, -1),
            "logits": np.asarray([[float(used)]], dtype=np.float32),
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
        failing_ids: Ids whose load raises instead of returning; a test
            empties the set to let a retry succeed.
    """

    def __init__(
        self,
        *,
        clock: FakeClock | None = None,
        delay: float = 0.0,
        advance_on_predict: float = 0.0,
    ) -> None:
        """Register a loader that has loaded nothing yet.

        Args:
            clock: Fake clock the stand-in artifacts move forward.
            delay: Seconds each load blocks for, widening the window two
                concurrent first requests race in.
            advance_on_predict: Seconds one prediction takes on ``clock``.
        """
        self.calls: list[str] = []
        self.failing_ids: set[str] = set()
        self._clock = clock
        self._delay = delay
        self._advance = advance_on_predict

    def __call__(self, entry: ModelEntry) -> _ServedModel:
        """Build the stand-in artifacts for one entry, recording the call.

        Args:
            entry: Entry the engine asked for.

        Returns:
            A served model backed by the entry's real frozen tokenizer.

        Raises:
            RuntimeError: If the entry's id is in :attr:`failing_ids`,
                standing in for an artifact that cannot be loaded.
        """
        self.calls.append(entry.id)
        if entry.id in self.failing_ids:
            raise RuntimeError(f"cannot load model '{entry.id}'")
        if self._delay:
            time.sleep(self._delay)
        assert entry.tokenizer is not None
        kind = str(entry.kind)
        buckets = tuple(sorted(entry.artifacts or ()))
        return _ServedModel(
            id=entry.id,
            kind=kind,
            tokenizer=runtime.load_frozen_tokenizer(entry.tokenizer),
            tokenizer_lock=threading.Lock(),
            compiled={bucket: StubCompiledModel(self._clock, self._advance) for bucket in buckets},
            buckets=buckets,
            output_name=str(entry.output_name),
            normalize=entry.normalize,
            embedding_dim=entry.embedding_dim if kind == "embedding" else None,
        )

    def count(self, model_id: str) -> int:
        """Return how many times ``model_id`` was loaded."""
        return self.calls.count(model_id)


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
    embedding_dim: int | None = None,
    load_policy: str | None = None,
    keep_alive: int | None = None,
) -> ModelEntry:
    """Build a model entry together with the files it points at.

    Every entry's artifacts exist, since the engine checks them at
    start-up whatever the model's load policy.

    Args:
        root: Directory holding one sub-directory per model.
        model_id: Id the entry is routed by.
        kind: ``"embedding"`` or ``"reranker"``.
        buckets: Sequence lengths to create artifacts for.
        embedding_dim: Embedding width stated by the entry, when known.
        load_policy: Per-entry load policy, or ``None`` to inherit the
            server default.
        keep_alive: Per-entry idle delay, or ``None`` to inherit the
            server default.

    Returns:
        A validated entry.
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
        "embedding_dim": embedding_dim,
        "load_policy": load_policy,
        "keep_alive": keep_alive,
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


@pytest.fixture
def clock() -> FakeClock:
    """Clock the test moves forward instead of waiting."""
    return FakeClock()


@pytest.fixture
def loader() -> RecordingLoader:
    """Artifact loader that records its calls and loads nothing real."""
    return RecordingLoader()


@pytest.fixture
def entries(tmp_path: Path) -> list[ModelEntry]:
    """One embedding model with a stated width, and one reranker."""
    return [
        _make_entry(tmp_path, "emb-a", embedding_dim=_CONFIGURED_WIDTH),
        _make_entry(tmp_path, "rr-a", kind="reranker"),
    ]


@pytest.fixture
def build_engine() -> Iterator[Callable[..., CoreMLEngine]]:
    """Build engines that are closed again when the test ends.

    Background sweeping is off by default, so a test drives
    ``_sweep_idle`` itself and no thread interferes with its assertions.
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


# --- policies ------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"load_policy": "sometimes"}, "unsupported load policy"),
        ({"load_policy": "disabled"}, "unsupported load policy"),
        ({"keep_alive": -1}, "keep_alive"),
    ],
)
def test_an_unusable_policy_is_rejected(overrides: dict[str, object], expected: str) -> None:
    """A policy the engine could not act on must be refused where it is built."""
    with pytest.raises(ValueError, match=expected):
        ModelPolicy(**overrides)  # type: ignore[arg-type]


def test_a_non_positive_model_limit_is_rejected(
    entries: list[ModelEntry], loader: RecordingLoader
) -> None:
    """A limit of zero could never hold a model, so it is a start-up error."""
    with pytest.raises(ValueError, match="max_loaded_models"):
        CoreMLEngine(entries, loader=loader, max_loaded_models=0, sweep_interval=0.0)


def test_models_are_resident_when_no_policies_are_given(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Without policies every model is resident, as the pre-lifecycle engine was."""
    engine = build_engine(entries, loader=loader)

    assert loader.calls == ["emb-a", "rr-a"]
    assert engine.loaded("emb-a") is True
    assert engine.loaded("rr-a") is True
    # However long they sit idle, resident models stay in memory.
    assert engine._sweep_idle(1e6) == 0
    assert engine.loaded("emb-a") is True


def test_resident_models_are_loaded_during_construction(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """An explicit resident policy loads the model before any request arrives."""
    policies = {"emb-a": ModelPolicy("resident"), "rr-a": ModelPolicy("resident")}

    engine = build_engine(entries, policies=policies, loader=loader)

    assert loader.calls == ["emb-a", "rr-a"]
    assert engine.loaded("emb-a") is True
    assert engine.loaded("rr-a") is True


# --- on-demand loading ---------------------------------------------------


def test_an_on_demand_model_loads_on_the_first_embed(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Nothing is loaded until a request needs it, and then the answer is normal."""
    engine = build_engine(entries, policies=_on_demand("emb-a", "rr-a"), loader=loader)
    assert loader.calls == []
    assert engine.loaded("emb-a") is False

    batch = engine.embed(["a b"], "emb-a")

    assert loader.calls == ["emb-a"]
    assert engine.loaded("emb-a") is True
    assert batch.vectors.shape == (1, _WIDTH)
    assert batch.buckets == [8]
    assert batch.orig_tokens == [2]
    assert batch.used_tokens == [2]
    # A request for one model must not drag the others in.
    assert engine.loaded("rr-a") is False


def test_an_on_demand_reranker_loads_on_the_first_rerank(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """The reranker path acquires its model the same way the embedding one does."""
    engine = build_engine(entries, policies=_on_demand("emb-a", "rr-a"), loader=loader)

    batch = engine.rerank("a", ["b", "c c"])

    assert loader.calls == ["rr-a"]
    assert engine.loaded("rr-a") is True
    assert batch.logits.shape == (2,)
    assert batch.orig_tokens == [2, 3]


def test_concurrent_first_requests_load_the_model_once(
    entries: list[ModelEntry], build_engine: Callable[..., CoreMLEngine]
) -> None:
    """Two threads racing on an unloaded model must load it once and both be served."""
    slow_loader = RecordingLoader(delay=0.05)
    engine = build_engine(entries, policies=_on_demand("emb-a", "rr-a"), loader=slow_loader)
    barrier = threading.Barrier(2)
    results: list[np.ndarray] = []
    failures: list[BaseException] = []

    def worker() -> None:
        """Embed the same text as the other thread, as simultaneously as possible."""
        barrier.wait(timeout=10)
        try:
            results.append(engine.embed(["a b"], "emb-a").vectors)
        except BaseException as exc:  # noqa: BLE001 - reported by the assertions below
            failures.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not [str(failure) for failure in failures]
    assert not [thread.name for thread in threads if thread.is_alive()]
    assert slow_loader.calls == ["emb-a"]
    assert len(results) == 2
    assert np.array_equal(results[0], results[1])
    assert engine._models["emb-a"].in_flight == 0


def test_a_failed_load_leaves_the_model_unloaded_and_is_retried(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """A loader error must propagate without leaving a half-built state behind."""
    engine = build_engine(entries, policies=_on_demand("emb-a"), loader=loader)
    loader.failing_ids.add("emb-a")

    with pytest.raises(RuntimeError, match="cannot load model 'emb-a'"):
        engine.embed(["a b"], "emb-a")

    managed = engine._models["emb-a"]
    assert engine.loaded("emb-a") is False
    assert managed.in_flight == 0
    assert not managed.load_lock.locked()

    loader.failing_ids.clear()
    batch = engine.embed(["a b"], "emb-a")

    assert loader.count("emb-a") == 2
    assert engine.loaded("emb-a") is True
    assert batch.vectors.shape == (1, _WIDTH)


# --- idle unloading ------------------------------------------------------


@pytest.mark.parametrize(("elapsed", "still_loaded"), [(9.0, True), (10.0, False), (11.0, False)])
def test_a_sweep_unloads_a_model_only_once_keep_alive_elapsed(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
    elapsed: float,
    still_loaded: bool,
) -> None:
    """The idle delay is inclusive: exactly ``keep_alive`` seconds is long enough."""
    engine = build_engine(
        entries, policies=_on_demand("emb-a", keep_alive=10), loader=loader, clock=clock
    )
    engine.embed(["a b"], "emb-a")
    finished = clock.now

    unloaded = engine._sweep_idle(finished + elapsed)

    assert engine.loaded("emb-a") is still_loaded
    assert unloaded == (0 if still_loaded else 1)


def test_a_zero_keep_alive_unloads_at_the_next_sweep(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """``keep_alive=0`` means "unload as soon as the model is found idle"."""
    engine = build_engine(
        entries, policies=_on_demand("emb-a", keep_alive=0), loader=loader, clock=clock
    )
    engine.embed(["a b"], "emb-a")

    assert engine._sweep_idle(clock.now) == 1
    assert engine.loaded("emb-a") is False


def test_the_idle_delay_is_measured_from_the_request_completion(
    entries: list[ModelEntry], clock: FakeClock, build_engine: Callable[..., CoreMLEngine]
) -> None:
    """A long request must not be followed by an immediate unload."""
    slow_loader = RecordingLoader(clock=clock, advance_on_predict=5.0)
    engine = build_engine(
        entries, policies=_on_demand("emb-a", keep_alive=10), loader=slow_loader, clock=clock
    )

    engine.embed(["a b"], "emb-a")

    assert clock.now == pytest.approx(5.0)
    # Measured from when the request started, the model would already have
    # been idle for 12s here; measured from when it finished, only for 7s.
    assert engine._sweep_idle(12.0) == 0
    assert engine.loaded("emb-a") is True
    assert engine._sweep_idle(15.0) == 1
    assert engine.loaded("emb-a") is False


def test_a_model_with_a_request_in_flight_is_never_swept(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """However long a request runs, the model it is using stays in memory."""
    engine = build_engine(
        entries, policies=_on_demand("emb-a", keep_alive=0), loader=loader, clock=clock
    )
    managed = engine._models["emb-a"]

    engine._acquire(managed)
    try:
        assert engine._sweep_idle(1e6) == 0
        assert engine.loaded("emb-a") is True
    finally:
        engine._release(managed)

    assert engine._sweep_idle(1e6) == 1
    assert engine.loaded("emb-a") is False


def test_a_request_after_an_idle_unload_reloads_the_model(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """An unloaded model must serve the next request as if nothing had happened."""
    engine = build_engine(
        entries, policies=_on_demand("emb-a", "rr-a", keep_alive=10), loader=loader, clock=clock
    )
    engine.embed(["a b"], "emb-a")
    engine._sweep_idle(clock.now + 10)
    assert engine.loaded("emb-a") is False

    batch = engine.embed(["a b"], "emb-a")

    assert loader.calls == ["emb-a", "emb-a"]
    assert engine.loaded("emb-a") is True
    assert batch.vectors.shape == (1, _WIDTH)
    assert batch.buckets == [8]


def test_the_measured_embedding_width_survives_an_unload(
    tmp_path: Path,
    loader: RecordingLoader,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """The (0, D) contract must hold for a model that is no longer in memory."""
    entries = [_make_entry(tmp_path, "emb-b")]
    engine = build_engine(
        entries, policies=_on_demand("emb-b", keep_alive=0), loader=loader, clock=clock
    )
    # Nothing measured and nothing configured yet: no width is invented.
    assert engine.embed([], "emb-b").vectors.shape == (0, 0)

    engine.embed(["a b"], "emb-b")
    engine._sweep_idle(clock.now)
    assert engine.loaded("emb-b") is False
    batch = engine.embed([], "emb-b")

    assert batch.vectors.shape == (0, _WIDTH)
    # The empty requests must not have loaded the model back in.
    assert loader.count("emb-b") == 1


# --- eviction ------------------------------------------------------------


def test_eviction_unloads_the_least_recently_used_on_demand_model(
    tmp_path: Path,
    loader: RecordingLoader,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Above the limit, the model idle for the longest makes room for the new one."""
    entries = [_make_entry(tmp_path, f"emb-{name}") for name in ("a", "b", "c")]
    engine = build_engine(
        entries,
        policies=_on_demand("emb-a", "emb-b", "emb-c"),
        loader=loader,
        max_loaded_models=2,
        clock=clock,
    )

    engine.embed(["a"], "emb-a")
    clock.advance(1.0)
    engine.embed(["a"], "emb-b")
    clock.advance(1.0)
    with caplog.at_level(logging.INFO, logger="eeane.engine"):
        engine.embed(["a"], "emb-c")

    assert engine.loaded("emb-a") is False
    assert engine.loaded("emb-b") is True
    assert engine.loaded("emb-c") is True
    assert loader.calls == ["emb-a", "emb-b", "emb-c"]
    assert "evicted to make room for model 'emb-c'" in caplog.text


def test_eviction_leaves_resident_models_alone(
    tmp_path: Path,
    loader: RecordingLoader,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """A resident model is never a victim, however long it has been idle."""
    entries = [_make_entry(tmp_path, name) for name in ("emb-r", "emb-a", "emb-b")]
    policies = {"emb-r": ModelPolicy("resident"), **_on_demand("emb-a", "emb-b")}
    engine = build_engine(
        entries, policies=policies, loader=loader, max_loaded_models=2, clock=clock
    )

    # The resident model was loaded first, so it is the least recently
    # used of all three by the time the limit is reached.
    clock.advance(1.0)
    engine.embed(["a"], "emb-a")
    clock.advance(1.0)
    engine.embed(["a"], "emb-b")

    assert engine.loaded("emb-r") is True
    assert engine.loaded("emb-a") is False
    assert engine.loaded("emb-b") is True


def test_a_model_in_flight_is_not_evicted_and_the_limit_is_only_warned_about(
    tmp_path: Path,
    loader: RecordingLoader,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Serving the request wins over the limit when nothing can be unloaded."""
    entries = [_make_entry(tmp_path, "emb-a"), _make_entry(tmp_path, "emb-b")]
    engine = build_engine(
        entries,
        policies=_on_demand("emb-a", "emb-b"),
        loader=loader,
        max_loaded_models=1,
        clock=clock,
    )
    managed = engine._models["emb-a"]

    engine._acquire(managed)
    try:
        with caplog.at_level(logging.WARNING, logger="eeane.engine"):
            batch = engine.embed(["a"], "emb-b")
    finally:
        engine._release(managed)

    assert batch.vectors.shape == (1, _WIDTH)
    assert engine.loaded("emb-a") is True
    assert engine.loaded("emb-b") is True
    assert "max_loaded_models=1" in caplog.text


# --- state reads ---------------------------------------------------------


def test_reading_the_engines_state_never_loads_a_model(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Routing, bucket and width lookups must all work while nothing is loaded."""
    engine = build_engine(entries, policies=_on_demand("emb-a", "rr-a"), loader=loader)

    assert engine.loaded("emb-a") is False
    assert engine.buckets("emb-a") == (8, 16)
    assert engine.default_model_id("embedding") == "emb-a"
    assert engine.default_model_id("reranker") == "rr-a"
    assert engine.embed([], "emb-a").vectors.shape == (0, _CONFIGURED_WIDTH)
    assert engine.rerank("a", [], "rr-a").logits.shape == (0,)

    assert loader.calls == []
    assert engine.loaded("emb-a") is False
    assert engine.loaded("rr-a") is False


def test_loaded_rejects_an_unknown_model_id(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """State lookups are keyed by model id and must not invent an answer."""
    engine = build_engine(entries, loader=loader)

    with pytest.raises(KeyError):
        engine.loaded("nope")


# --- the background sweeper ----------------------------------------------


def test_no_sweeper_runs_when_every_model_is_resident(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """A fully resident engine has nothing to sweep, so it stays single-threaded."""
    engine = build_engine(entries, loader=loader, sweep_interval=0.05)

    assert engine._sweeper is None
    engine.close()
    engine.close()


def test_close_stops_the_sweeper_and_is_idempotent(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """An engine with on-demand models sweeps in the background until it is closed."""
    engine = build_engine(entries, policies=_on_demand("emb-a"), loader=loader, sweep_interval=0.05)
    sweeper = engine._sweeper
    assert sweeper is not None
    assert sweeper.is_alive()

    engine.close()

    assert not sweeper.is_alive()
    engine.close()
    assert engine._sweeper is None


def test_the_sweeper_unloads_an_idle_model_in_the_background(
    entries: list[ModelEntry],
    loader: RecordingLoader,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """The real thread, on the real clock, must unload a model nobody uses."""
    engine = build_engine(
        entries,
        policies=_on_demand("emb-a", keep_alive=0),
        loader=loader,
        sweep_interval=0.01,
    )
    engine.embed(["a b"], "emb-a")

    deadline = time.monotonic() + 5.0
    while engine.loaded("emb-a") and time.monotonic() < deadline:
        time.sleep(0.01)

    assert engine.loaded("emb-a") is False


# --- configuration -------------------------------------------------------


def test_policies_are_read_from_the_resolved_configuration(tmp_path: Path) -> None:
    """Server defaults apply to the entries that state neither value."""
    config = EeaneConfig(
        server=ServerConfig(default_load_policy="on_demand", keep_alive=120, max_loaded_models=2),
        models=[
            _make_entry(tmp_path, "emb-a"),
            _make_entry(tmp_path, "emb-b", load_policy="resident", keep_alive=30),
        ],
    )

    assert engine_module._policies_from_config(config) == {
        "emb-a": ModelPolicy("on_demand", 120),
        "emb-b": ModelPolicy("resident", 30),
    }


def test_from_config_applies_the_policies_and_the_model_limit(tmp_path: Path) -> None:
    """The configured lifecycle settings must reach the engine unchanged."""
    config = EeaneConfig(
        server=ServerConfig(default_load_policy="on_demand", keep_alive=120, max_loaded_models=2),
        models=[
            _make_entry(tmp_path, "emb-a"),
            _make_entry(tmp_path, "emb-b", keep_alive=45),
        ],
    )

    engine = CoreMLEngine.from_config(config)
    try:
        assert engine._models["emb-a"].policy == ModelPolicy("on_demand", 120)
        assert engine._models["emb-b"].policy == ModelPolicy("on_demand", 45)
        assert engine._max_loaded_models == 2
        # Every entry is on-demand here, so building the engine loaded no
        # artifact at all and needed no Neural Engine.
        assert engine.loaded("emb-a") is False
        assert engine.loaded("emb-b") is False
    finally:
        engine.close()
