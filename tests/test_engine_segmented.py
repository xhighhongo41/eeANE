"""Tests for serving a model stored as several compiled parts.

A model too large for the Neural Engine is stored as a directory of
compiled parts next to a manifest, and the engine runs the parts one
after another. The parts directories here are real -- a manifest written
by :func:`eeane.segmented.write_segments` next to empty part directories
-- so the engine's own loader reads them, while loading one compiled part
is replaced by stand-ins that answer from memory: nothing here needs a
Neural Engine.

The stand-in parts compute something only a correct chaining produces:
the first part turns the token ids into an intermediate array, and the
last one reads that array together with the model's own attention mask,
each matched by name.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from eeane import engine as engine_module
from eeane.config import ModelEntry
from eeane.engine import CoreMLEngine
from eeane.segmented import (
    SEGMENTS_FILENAME,
    Segment,
    SegmentedModel,
    SegmentsError,
    SegmentsManifest,
    write_segments,
)

# Width of the rows the stand-in artifacts answer with.
_WIDTH = 8

# Directory names of the two parts every parts directory here holds.
_FIRST_PART = "part01.mlmodelc"
_LAST_PART = "part02.mlmodelc"

# Factor the first part scales the token ids by, and weight the last part
# gives each real token of the attention mask. Distinct enough that an
# answer which skipped either part, or read the wrong tensor, is visible.
_HIDDEN_SCALE = 10.0
_MASK_WEIGHT = 0.5

# Token ids of the toy vocabulary the tokenizers below are built with.
_VOCAB = {"<pad>": 0, "a": 1, "b": 2, "c": 3}

# A text of nine tokens: too long for bucket 8, so it lands in bucket 16.
_LONG_TEXT = "a b c a b c a b c"


class FirstPart:
    """Stand-in for the first compiled part: turns token ids into an intermediate.

    Attributes:
        path: Part directory it was "loaded" from.
        feeds: Every feed it was given, in call order.
        produced: The intermediate array of every prediction.
    """

    def __init__(self, path: Path) -> None:
        """Register a part that has served no prediction yet."""
        self.path = path
        self.feeds: list[dict[str, np.ndarray]] = []
        self.produced: list[np.ndarray] = []

    def predict(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Answer with the token ids scaled into a float intermediate."""
        self.feeds.append(dict(feed))
        hidden = feed["input_ids"].astype(np.float32) * _HIDDEN_SCALE
        self.produced.append(hidden)
        return {"hidden": hidden}


class LastPart:
    """Stand-in for the last compiled part: reads the intermediate and the mask.

    Answers with both output names eeANE knows about, like a real
    artifact's stand-in in the other engine tests, so the engine's own
    choice of output decides what a request gets.

    Attributes:
        path: Part directory it was "loaded" from.
        feeds: Every feed it was given, in call order.
    """

    def __init__(self, path: Path) -> None:
        """Register a part that has served no prediction yet."""
        self.path = path
        self.feeds: list[dict[str, np.ndarray]] = []

    def predict(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Answer one score per row, from the intermediate and the mask."""
        self.feeds.append(dict(feed))
        mask = feed["attention_mask"].astype(np.float32)
        score = (feed["hidden"] * mask).sum(axis=1) + _MASK_WEIGHT * mask.sum(axis=1)
        return {
            "embedding": np.arange(_WIDTH, dtype=np.float32)[None, :] + score[:, None],
            "logits": score[:, None].astype(np.float32),
        }


class PlainModel:
    """Stand-in for a model stored as one compiled model.

    Its answer differs from any split model's, so a test can tell which
    kind of artifact served an input.

    Attributes:
        path: Artifact directory it was "loaded" from.
        feeds: Every feed it was given, in call order.
    """

    def __init__(self, path: Path) -> None:
        """Register an artifact that has served no prediction yet."""
        self.path = path
        self.feeds: list[dict[str, np.ndarray]] = []

    def predict(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Answer with rows counting the real tokens down instead of up."""
        self.feeds.append(dict(feed))
        used = feed["attention_mask"].sum(axis=1).astype(np.float32)
        return {
            "embedding": np.arange(_WIDTH, dtype=np.float32)[None, :] - used[:, None],
            "logits": -used[:, None],
        }


class PartLoader:
    """Replacement for ``eeane.engine._load_compiled_part`` that opens no file.

    Attributes:
        loaded: The stand-in built for every loaded path, keyed by path.
    """

    def __init__(self) -> None:
        """Start with nothing loaded."""
        self.loaded: dict[Path, FirstPart | LastPart | PlainModel] = {}

    def __call__(self, path: Path) -> FirstPart | LastPart | PlainModel:
        """Build (and remember) the stand-in for one compiled directory."""
        path = Path(path)
        model: FirstPart | LastPart | PlainModel
        if path.name == _FIRST_PART:
            model = FirstPart(path)
        elif path.name == _LAST_PART:
            model = LastPart(path)
        else:
            model = PlainModel(path)
        self.loaded[path] = model
        return model

    def first(self, parts_dir: Path) -> FirstPart:
        """Return the stand-in loaded for the first part of ``parts_dir``."""
        part = self.loaded[parts_dir / _FIRST_PART]
        assert isinstance(part, FirstPart)
        return part

    def last(self, parts_dir: Path) -> LastPart:
        """Return the stand-in loaded for the last part of ``parts_dir``."""
        part = self.loaded[parts_dir / _LAST_PART]
        assert isinstance(part, LastPart)
        return part


@pytest.fixture
def part_loader(monkeypatch: pytest.MonkeyPatch) -> PartLoader:
    """Replace the loading of one compiled part for the duration of a test."""
    loader = PartLoader()
    monkeypatch.setattr(engine_module, "_load_compiled_part", loader)
    return loader


@pytest.fixture
def build_engine() -> Iterator[Callable[..., CoreMLEngine]]:
    """Build engines that are closed again when the test ends."""
    built: list[CoreMLEngine] = []

    def factory(model_entries: Sequence[ModelEntry], **kwargs: object) -> CoreMLEngine:
        engine = CoreMLEngine(model_entries, **kwargs)  # type: ignore[arg-type]
        built.append(engine)
        return engine

    yield factory
    for engine in built:
        engine.close()


def _write_toy_tokenizer(path: Path) -> Path:
    """Write a minimal word-level tokenizer.json carrying pad settings."""
    tokenizer = Tokenizer(models.WordLevel(vocab=_VOCAB, unk_token="<pad>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    tokenizer.save(str(path))
    return path


def _write_parts(parts_dir: Path, output_name: str) -> Path:
    """Write a parts directory of two chained parts.

    The first part reads only the token ids; the last one reads the first
    part's output and the model's own attention mask.

    Args:
        parts_dir: Directory to create.
        output_name: Name the manifest records as the model's output.

    Returns:
        ``parts_dir``, for chaining.
    """
    parts_dir.mkdir(parents=True)
    for name in (_FIRST_PART, _LAST_PART):
        (parts_dir / name).mkdir()
    manifest = SegmentsManifest(
        output_name=output_name,
        inputs=("input_ids", "attention_mask"),
        segments=(
            Segment(mlmodelc=_FIRST_PART, inputs=("input_ids",), outputs=("hidden",)),
            Segment(
                mlmodelc=_LAST_PART,
                inputs=("hidden", "attention_mask"),
                outputs=("embedding", "logits"),
            ),
        ),
    )
    write_segments(parts_dir, manifest)
    return parts_dir


def _make_entry(
    root: Path,
    model_id: str,
    kind: str = "embedding",
    *,
    split: Sequence[int] = (8, 16),
    plain: Sequence[int] = (),
    split_batch: Sequence[int] = (),
) -> ModelEntry:
    """Build a model entry together with the files it points at.

    Args:
        root: Directory holding one sub-directory per model.
        model_id: Id the entry is routed by.
        kind: ``"embedding"`` or ``"reranker"``.
        split: Buckets whose artifact is a parts directory.
        plain: Buckets whose artifact is one compiled model.
        split_batch: Buckets given a batched artifact that is a parts
            directory.

    Returns:
        A validated entry whose artifacts all exist.
    """
    model_dir = root / model_id
    model_dir.mkdir(parents=True)
    tokenizer_path = _write_toy_tokenizer(model_dir / "tokenizer.json")
    output_name = "embedding" if kind == "embedding" else "logits"
    artifacts: dict[int, Path] = {}
    for bucket in split:
        artifacts[bucket] = _write_parts(model_dir / f"s{bucket}.parts", output_name)
    for bucket in plain:
        artifacts[bucket] = model_dir / f"s{bucket}.mlmodelc"
        artifacts[bucket].mkdir()
    fields: dict[str, object] = {
        "id": model_id,
        "kind": kind,
        "tokenizer": tokenizer_path,
        "artifacts": artifacts,
    }
    if split_batch:
        fields["batch_artifacts"] = {
            bucket: _write_parts(model_dir / f"s{bucket}_b2.parts", output_name)
            for bucket in split_batch
        }
    return ModelEntry(**fields)


def _ids(text: str) -> list[int]:
    """Return the toy vocabulary's token ids of a whitespace-separated text."""
    return [_VOCAB[word] for word in text.split()]


def _split_score(ids: Sequence[int]) -> float:
    """Return what the split stand-ins score one input of these token ids at."""
    return _HIDDEN_SCALE * sum(ids) + _MASK_WEIGHT * len(ids)


def _split_row(text: str) -> np.ndarray:
    """Return the embedding row the split stand-ins answer for ``text``."""
    return np.arange(_WIDTH, dtype=np.float32) + _split_score(_ids(text))


def _plain_row(text: str) -> np.ndarray:
    """Return the embedding row the plain stand-in answers for ``text``."""
    return np.arange(_WIDTH, dtype=np.float32) - len(_ids(text))


# --- loading ---------------------------------------------------------------


def test_a_parts_directory_is_loaded_as_a_segmented_model(
    tmp_path: Path, part_loader: PartLoader
) -> None:
    parts_dir = _write_parts(tmp_path / "s8.parts", "embedding")

    model = engine_module._load_compiled(parts_dir)

    assert isinstance(model, SegmentedModel)
    assert model.parts == (part_loader.first(parts_dir), part_loader.last(parts_dir))
    assert [segment.mlmodelc for segment in model.manifest.segments] == [_FIRST_PART, _LAST_PART]


def test_a_single_compiled_model_is_loaded_as_it_always_was(
    tmp_path: Path, part_loader: PartLoader
) -> None:
    artifact = tmp_path / "s8.mlmodelc"
    artifact.mkdir()

    model = engine_module._load_compiled(artifact)

    assert isinstance(model, PlainModel)
    assert part_loader.loaded == {artifact: model}


def test_a_broken_manifest_fails_the_load_before_any_part_is_loaded(
    tmp_path: Path, part_loader: PartLoader, build_engine: Callable[..., CoreMLEngine]
) -> None:
    entry = _make_entry(tmp_path, "emb", split=(8,))
    assert entry.artifacts is not None
    (entry.artifacts[8] / SEGMENTS_FILENAME).write_text("{not json", encoding="utf-8")

    with pytest.raises(SegmentsError, match="not valid JSON"):
        build_engine([entry])

    assert part_loader.loaded == {}


def test_every_bucket_gets_its_own_parts(
    tmp_path: Path, part_loader: PartLoader, build_engine: Callable[..., CoreMLEngine]
) -> None:
    entry = _make_entry(tmp_path, "emb", split=(8, 16))
    assert entry.artifacts is not None

    build_engine([entry])

    assert set(part_loader.loaded) == {
        entry.artifacts[bucket] / name for bucket in (8, 16) for name in (_FIRST_PART, _LAST_PART)
    }


# --- requests ----------------------------------------------------------------


def test_an_embedding_request_returns_the_last_parts_output(
    tmp_path: Path, part_loader: PartLoader, build_engine: Callable[..., CoreMLEngine]
) -> None:
    entry = _make_entry(tmp_path, "emb", split=(8, 16))
    assert entry.artifacts is not None
    engine = build_engine([entry])

    batch = engine.embed(["a b", _LONG_TEXT])

    np.testing.assert_array_equal(
        batch.vectors, np.stack([_split_row("a b"), _split_row(_LONG_TEXT)])
    )
    assert batch.buckets == [8, 16]
    assert batch.used_tokens == [2, 9]


def test_each_part_gets_exactly_the_tensors_its_manifest_names(
    tmp_path: Path, part_loader: PartLoader, build_engine: Callable[..., CoreMLEngine]
) -> None:
    entry = _make_entry(tmp_path, "emb", split=(8,))
    assert entry.artifacts is not None
    engine = build_engine([entry])

    engine.embed(["a b c"])

    first = part_loader.first(entry.artifacts[8])
    last = part_loader.last(entry.artifacts[8])
    assert [sorted(feed) for feed in first.feeds] == [["input_ids"]]
    assert [sorted(feed) for feed in last.feeds] == [["attention_mask", "hidden"]]
    # The last part reads the very array the first one produced, and the
    # model's own attention mask, which no part produced.
    assert last.feeds[0]["hidden"] is first.produced[0]
    np.testing.assert_array_equal(
        last.feeds[0]["attention_mask"], np.asarray([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int32)
    )


def test_a_rerank_request_returns_the_last_parts_logits(
    tmp_path: Path, part_loader: PartLoader, build_engine: Callable[..., CoreMLEngine]
) -> None:
    embedder = _make_entry(tmp_path, "emb", plain=(8,), split=())
    reranker = _make_entry(tmp_path, "rr", kind="reranker", split=(8, 16))
    engine = build_engine([embedder, reranker])

    batch = engine.rerank("a", ["b", "c c"])

    # The toy tokenizer encodes a pair as the two texts back to back.
    np.testing.assert_array_equal(
        batch.logits,
        np.asarray([_split_score(_ids("a b")), _split_score(_ids("a c c"))], dtype=np.float32),
    )


def test_split_and_plain_buckets_of_one_entry_are_each_served_their_way(
    tmp_path: Path, part_loader: PartLoader, build_engine: Callable[..., CoreMLEngine]
) -> None:
    entry = _make_entry(tmp_path, "emb", plain=(8,), split=(16,))
    assert entry.artifacts is not None
    engine = build_engine([entry])

    batch = engine.embed(["a b", _LONG_TEXT])

    np.testing.assert_array_equal(
        batch.vectors, np.stack([_plain_row("a b"), _split_row(_LONG_TEXT)])
    )
    plain = part_loader.loaded[entry.artifacts[8]]
    assert isinstance(plain, PlainModel)
    assert len(plain.feeds) == 1
    assert len(part_loader.first(entry.artifacts[16]).feeds) == 1


def test_split_batched_artifacts_predict_a_group_of_inputs_together(
    tmp_path: Path, part_loader: PartLoader, build_engine: Callable[..., CoreMLEngine]
) -> None:
    entry = _make_entry(tmp_path, "emb", split=(8, 16), split_batch=(8,))
    assert entry.artifacts is not None
    assert entry.batch_artifacts is not None
    engine = build_engine([entry])

    batch = engine.embed(["a", "a b"])

    np.testing.assert_array_equal(batch.vectors, np.stack([_split_row("a"), _split_row("a b")]))
    batched_first = part_loader.first(entry.batch_artifacts[8])
    batched_last = part_loader.last(entry.batch_artifacts[8])
    assert [feed["input_ids"].shape for feed in batched_first.feeds] == [(2, 8)]
    assert [feed["hidden"].shape for feed in batched_last.feeds] == [(2, 8)]
    # The group went to the batched parts only.
    assert part_loader.first(entry.artifacts[8]).feeds == []


def test_every_part_runs_inside_the_prediction_lock(
    tmp_path: Path, part_loader: PartLoader, build_engine: Callable[..., CoreMLEngine]
) -> None:
    """The Neural Engine runs one prediction at a time, whichever part it is."""
    entry = _make_entry(tmp_path, "emb", split=(8,))
    assert entry.artifacts is not None
    engine = build_engine([entry])
    held: list[bool] = []
    for part in (part_loader.first(entry.artifacts[8]), part_loader.last(entry.artifacts[8])):
        answer = part.predict

        def probing_predict(
            feed: dict[str, np.ndarray],
            answer: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] = answer,
        ) -> dict[str, np.ndarray]:
            # Nothing else predicts in this test, so a held lock is held
            # by this very request.
            held.append(engine._lock.locked())
            return answer(feed)

        part.predict = probing_predict  # type: ignore[method-assign]

    engine.embed(["a b"])

    assert held == [True, True]
    assert not engine._lock.locked()
