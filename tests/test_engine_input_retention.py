"""Tests that the engine keeps the arrays it hands to Core ML referenced.

Core ML releases the feature values wrapping a prediction's inputs from an
internal dispatch queue, seconds after the prediction and from a thread
that carries no Python state, so the engine -- not Core ML -- must be the
one dropping the last reference to those arrays. These tests watch the
arrays through weak references: nothing here keeps one alive by itself, so
an array that is still reachable is one the engine is holding on to, and a
collected one is an array the engine has let go of.

The compiled artifacts are stand-ins, as in the other engine tests, so the
whole module runs on any machine, without a Neural Engine.
"""

from __future__ import annotations

import gc
import threading
import time
import weakref
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from eeane import engine as engine_module
from eeane import runtime
from eeane.config import ModelEntry
from eeane.engine import CoreMLEngine, ModelPolicy, _ServedModel

# Width of the row the stand-in artifacts answer with.
_WIDTH = 8

# Sequence-length buckets every stand-in model is compiled for.
_BUCKETS = (8, 16)

# Texts that all fit the smallest bucket, and one that only fits the
# largest, so a test can reach two different compiled artifacts.
_SHORT_TEXTS = ["a b", "a c", "b c"]
_LONG_TEXT = "a b c a b c a b c"

# Seconds any polling wait in this module may take before the test gives
# up. Only ever waited out when the code under test is broken.
_TIMEOUT = 5.0

# How often a polling wait re-checks what it is waiting for.
_POLL_INTERVAL = 0.005


class WatchingModel:
    """Stand-in for one bucket's compiled model that watches its inputs.

    Answers like a real artifact, derived from the attention mask, and
    keeps a weak reference to the ``input_ids`` array of every prediction
    it was given. Weak references keep nothing alive, so what a test reads
    back through them is exactly what the engine is still holding.

    Attributes:
        seen: Weak reference to the ``input_ids`` array of every
            prediction, in call order.
        raises: Whether ``predict`` fails instead of answering, standing
            in for an artifact that rejects its inputs.
    """

    def __init__(self) -> None:
        """Register an artifact that has served no prediction yet."""
        self.seen: list[weakref.ref[np.ndarray]] = []
        self.raises = False

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Record the inputs weakly, then answer one prediction.

        Args:
            inputs: ``input_ids``/``attention_mask`` of shape
                ``(rows, S)``.

        Returns:
            Both output names eeANE knows about, with one row per input.

        Raises:
            RuntimeError: If :attr:`raises` is set.
        """
        self.seen.append(weakref.ref(inputs["input_ids"]))
        if self.raises:
            raise RuntimeError("the stand-in artifact refused the inputs")
        used = [int(row.sum()) for row in inputs["attention_mask"]]
        return {
            "embedding": np.stack(
                [np.arange(_WIDTH, dtype=np.float32) + float(count) for count in used]
            ),
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


class WatchingLoader:
    """Artifact loader building one watching stand-in per bucket.

    Every load builds fresh artifacts, exactly as the real loader does, so
    a reload really does hand the engine new model objects.

    Attributes:
        calls: Model id of every load, in call order.
        models: The stand-in built for each ``(model id, bucket)`` by the
            most recent load of that model.
    """

    def __init__(self) -> None:
        """Register a loader that has loaded nothing yet."""
        self.calls: list[str] = []
        self.models: dict[tuple[str, int], WatchingModel] = {}

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
        compiled: dict[int, WatchingModel] = {}
        for bucket in buckets:
            compiled[bucket] = WatchingModel()
            self.models[entry.id, bucket] = compiled[bucket]
        return _ServedModel(
            id=entry.id,
            kind=kind,
            tokenizer=runtime.load_frozen_tokenizer(entry.tokenizer),
            tokenizer_lock=threading.Lock(),
            compiled=dict(compiled),
            buckets=buckets,
            output_name=str(entry.output_name),
            normalize=entry.normalize,
            embedding_dim=entry.embedding_dim if kind == "embedding" else None,
        )


class FakeMonotonic:
    """Stand-in for ``time.monotonic`` a test moves forward by hand.

    Attributes:
        now: Current reading, in seconds. Started well above zero so a
            reading of this clock is never mistaken for a real one.
    """

    def __init__(self, start: float = 1_000.0) -> None:
        """Start the clock at ``start`` seconds."""
        self.now = start

    def __call__(self) -> float:
        """Return the current reading."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds``."""
        self.now += seconds


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


def _make_entry(root: Path, model_id: str, kind: str = "embedding") -> ModelEntry:
    """Build a model entry together with the files it points at.

    Args:
        root: Directory holding one sub-directory per model.
        model_id: Id the entry is routed by.
        kind: ``"embedding"`` or ``"reranker"``.

    Returns:
        A validated entry whose artifacts all exist.
    """
    model_dir = root / model_id
    tokenizer_path = model_dir / "tokenizer.json"
    artifacts = {bucket: model_dir / f"s{bucket}.mlmodelc" for bucket in _BUCKETS}
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
        model_ids: Ids to serve on demand; any id left out stays resident.
        keep_alive: Idle delay applied to all of them.

    Returns:
        The policy mapping to hand to the engine.
    """
    return {model_id: ModelPolicy("on_demand", keep_alive) for model_id in model_ids}


def _alive(ref: weakref.ref[np.ndarray]) -> bool:
    """Report whether the array behind ``ref`` is still referenced.

    Args:
        ref: Weak reference taken by the stand-in artifacts.

    Returns:
        ``True`` while something still holds the array. A collection is
        forced first, so an array that is only kept alive by a reference
        cycle counts as released.
    """
    gc.collect()
    return ref() is not None


def _purge_after_grace(engine: CoreMLEngine) -> None:
    """Let go of everything quarantined more than the grace period ago.

    Args:
        engine: Engine whose quarantine is swept. Nothing happens when it
            holds nothing, so a test may call this unconditionally.
    """
    with engine._retention_lock:
        if not engine._retired_inputs:
            return
        oldest = engine._retired_inputs[0][0]
        engine._purge_retired_locked(oldest + engine_module._INPUT_RETENTION_SECONDS + 1.0)


def _wait_until(condition: Callable[[], bool]) -> bool:
    """Poll ``condition`` until it holds or the test's patience runs out.

    Args:
        condition: Predicate re-evaluated until it is true.

    Returns:
        Whether the condition became true within :data:`_TIMEOUT`.
    """
    deadline = time.monotonic() + _TIMEOUT
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(_POLL_INTERVAL)
    return condition()


@pytest.fixture
def clock() -> FakeClock:
    """Clock the test moves forward instead of waiting."""
    return FakeClock()


@pytest.fixture
def loader() -> WatchingLoader:
    """Artifact loader handing out stand-ins that watch their inputs."""
    return WatchingLoader()


@pytest.fixture
def entries(tmp_path: Path) -> list[ModelEntry]:
    """Two embedding models, so eviction has something to choose from."""
    return [_make_entry(tmp_path, "emb-a"), _make_entry(tmp_path, "emb-b")]


@pytest.fixture
def build_engine() -> Iterator[Callable[..., CoreMLEngine]]:
    """Build engines that are closed again when the test ends.

    Background sweeping is off by default, so a test drives the engine's
    housekeeping itself and no thread interferes with its assertions.
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


# --- retention across predictions ----------------------------------------


def test_the_inputs_of_a_prediction_outlive_the_request(
    entries: list[ModelEntry], loader: WatchingLoader, build_engine: Callable[..., CoreMLEngine]
) -> None:
    """Core ML may still hold the last inputs, so the engine must too."""
    engine = build_engine(entries, loader=loader)

    engine.embed([_SHORT_TEXTS[0]], "emb-a")

    model = loader.models["emb-a", 8]
    assert len(model.seen) == 1
    assert _alive(model.seen[0]), "the engine dropped the inputs of its last prediction"


def test_replaced_inputs_are_quarantined_rather_than_released(
    entries: list[ModelEntry], loader: WatchingLoader, build_engine: Callable[..., CoreMLEngine]
) -> None:
    """The inputs of the previous prediction may still be wrapped by Core ML."""
    engine = build_engine(entries, loader=loader)

    engine.embed([_SHORT_TEXTS[0]], "emb-a")
    engine.embed([_SHORT_TEXTS[1]], "emb-a")

    model = loader.models["emb-a", 8]
    assert len(model.seen) == 2
    assert _alive(model.seen[0]), "the replaced inputs were released straight away"
    assert _alive(model.seen[1])
    assert len(engine._retired_inputs) == 1


def test_each_compiled_artifact_keeps_its_own_last_inputs(
    entries: list[ModelEntry], loader: WatchingLoader, build_engine: Callable[..., CoreMLEngine]
) -> None:
    """Two buckets are two models, so neither prediction replaces the other."""
    engine = build_engine(entries, loader=loader)

    engine.embed([_SHORT_TEXTS[0]], "emb-a")
    engine.embed([_LONG_TEXT], "emb-a")

    short_model = loader.models["emb-a", 8]
    long_model = loader.models["emb-a", 16]
    assert _alive(short_model.seen[0])
    assert _alive(long_model.seen[0])
    # Nothing was replaced, so nothing had to be quarantined at all.
    assert not engine._retired_inputs


def test_inputs_are_retained_even_when_the_prediction_fails(
    entries: list[ModelEntry], loader: WatchingLoader, build_engine: Callable[..., CoreMLEngine]
) -> None:
    """A failed prediction may have bound its inputs before it failed."""
    engine = build_engine(entries, loader=loader)
    engine.embed([_SHORT_TEXTS[0]], "emb-a")
    model = loader.models["emb-a", 8]
    model.raises = True

    with pytest.raises(RuntimeError, match="refused the inputs"):
        engine.embed([_SHORT_TEXTS[1]], "emb-a")

    # Compared by identity through the recorded ids: the retained array is
    # alive, so no other object can have been given its id meanwhile.
    retained = engine._retained_inputs[id(model)]
    failed = model.seen[1]()
    assert failed is not None
    assert retained["input_ids"] is failed


# --- the grace period ----------------------------------------------------


def test_a_quarantined_input_is_released_once_the_grace_period_passes(
    entries: list[ModelEntry],
    loader: WatchingLoader,
    build_engine: Callable[..., CoreMLEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retention is a grace period, not a leak: the queue drains as it is used."""
    fake_monotonic = FakeMonotonic()
    monkeypatch.setattr(engine_module.time, "monotonic", fake_monotonic)
    engine = build_engine(entries, loader=loader)
    engine.embed([_SHORT_TEXTS[0]], "emb-a")
    engine.embed([_SHORT_TEXTS[1]], "emb-a")

    fake_monotonic.advance(engine_module._INPUT_RETENTION_SECONDS + 1.0)
    engine.embed([_SHORT_TEXTS[2]], "emb-a")

    model = loader.models["emb-a", 8]
    assert not _alive(model.seen[0]), "the quarantine never released the oldest inputs"
    # The inputs quarantined by the third prediction have just arrived, and
    # its own inputs are the ones Core ML may still be holding.
    assert _alive(model.seen[1])
    assert _alive(model.seen[2])


def test_a_purge_leaves_inputs_that_are_still_within_the_grace_period(
    entries: list[ModelEntry],
    loader: WatchingLoader,
    build_engine: Callable[..., CoreMLEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An input quarantined a moment ago must survive a sweep that follows it."""
    fake_monotonic = FakeMonotonic()
    monkeypatch.setattr(engine_module.time, "monotonic", fake_monotonic)
    engine = build_engine(entries, loader=loader)
    engine.embed([_SHORT_TEXTS[0]], "emb-a")
    engine.embed([_SHORT_TEXTS[1]], "emb-a")

    fake_monotonic.advance(engine_module._INPUT_RETENTION_SECONDS - 1.0)
    with engine._retention_lock:
        engine._purge_retired_locked(fake_monotonic.now)

    model = loader.models["emb-a", 8]
    assert _alive(model.seen[0])
    assert len(engine._retired_inputs) == 1


def test_the_background_sweeper_drains_the_quarantine(
    entries: list[ModelEntry],
    loader: WatchingLoader,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An engine that stops being used must still let go of what it quarantined."""
    fake_monotonic = FakeMonotonic()
    monkeypatch.setattr(engine_module.time, "monotonic", fake_monotonic)
    engine = build_engine(
        entries,
        policies=_on_demand("emb-a", "emb-b"),
        loader=loader,
        clock=clock,
        sweep_interval=0.01,
    )
    engine.embed([_SHORT_TEXTS[0]], "emb-a")
    engine.embed([_SHORT_TEXTS[1]], "emb-a")
    model = loader.models["emb-a", 8]
    assert _alive(model.seen[0])

    fake_monotonic.advance(engine_module._INPUT_RETENTION_SECONDS + 1.0)

    assert _wait_until(lambda: not _alive(model.seen[0])), "the sweeper never purged the quarantine"
    # The model is still loaded (its keep_alive has not passed on the
    # engine's own clock), so its last inputs are still retained.
    assert engine.loaded("emb-a") is True
    assert _alive(model.seen[1])


# --- retention across unloads --------------------------------------------


def test_an_idle_unload_quarantines_the_models_last_inputs(
    entries: list[ModelEntry],
    loader: WatchingLoader,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """Core ML defers its release past the unload, so the unload must not free it."""
    engine = build_engine(
        entries, policies=_on_demand("emb-a", keep_alive=0), loader=loader, clock=clock
    )
    engine.embed([_SHORT_TEXTS[0]], "emb-a")
    model = loader.models["emb-a", 8]

    assert engine._sweep_idle(clock.now) == 1

    assert engine.loaded("emb-a") is False
    assert _alive(model.seen[0]), "the unload released inputs Core ML may still hold"
    assert len(engine._retired_inputs) == 1
    # Once quarantined, they are no longer tied to the unloaded artifact.
    assert id(model) not in engine._retained_inputs

    _purge_after_grace(engine)

    assert not _alive(model.seen[0])


def test_an_eviction_quarantines_the_evicted_models_last_inputs(
    entries: list[ModelEntry],
    loader: WatchingLoader,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """A model unloaded to make room for another one is unloaded just as carefully."""
    engine = build_engine(
        entries,
        policies=_on_demand("emb-a", "emb-b"),
        loader=loader,
        clock=clock,
        max_loaded_models=1,
    )
    engine.embed([_SHORT_TEXTS[0]], "emb-a")
    model = loader.models["emb-a", 8]

    engine.embed([_SHORT_TEXTS[1]], "emb-b")

    assert engine.loaded("emb-a") is False
    assert _alive(model.seen[0]), "the eviction released inputs Core ML may still hold"
    assert len(engine._retired_inputs) == 1

    _purge_after_grace(engine)

    assert not _alive(model.seen[0])
    # The model that took its place keeps its own inputs, as always.
    assert _alive(loader.models["emb-b", 8].seen[0])


def test_a_reload_after_an_unload_starts_from_a_clean_retention(
    entries: list[ModelEntry],
    loader: WatchingLoader,
    clock: FakeClock,
    build_engine: Callable[..., CoreMLEngine],
) -> None:
    """The retention must follow the artifacts, not the model id."""
    engine = build_engine(
        entries, policies=_on_demand("emb-a", keep_alive=0), loader=loader, clock=clock
    )
    engine.embed([_SHORT_TEXTS[0]], "emb-a")
    first_model = loader.models["emb-a", 8]
    engine._sweep_idle(clock.now)

    engine.embed([_SHORT_TEXTS[1]], "emb-a")

    second_model = loader.models["emb-a", 8]
    assert second_model is not first_model
    assert _alive(second_model.seen[0])
    # The reloaded artifacts start with no inputs of their own to replace,
    # so the quarantine still holds only what the unload put there.
    assert len(engine._retired_inputs) == 1
