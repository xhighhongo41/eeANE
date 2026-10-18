"""Tests for the manifest and runner of a model stored as several compiled parts."""

from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from eeane import segmented
from eeane.segmented import (
    SEGMENTS_FILENAME,
    SEGMENTS_FORMAT_VERSION,
    Segment,
    SegmentedModel,
    SegmentsError,
    SegmentsManifest,
    is_segmented,
    load_segments,
    part_paths,
    write_segments,
)

# --- helpers -------------------------------------------------------------------


def _part_name(index: int) -> str:
    """Directory name of the ``index``-th (0-based) part."""
    return f"part{index + 1:02d}.mlmodelc"


def _chain_manifest(
    count: int, *, mask_in: Callable[[int], bool] = lambda _i: True
) -> SegmentsManifest:
    """Describe ``count`` parts chained one after another.

    The first part reads ``input_ids``, every later part the previous
    part's hidden state; ``mask_in(i)`` tells whether part ``i`` also
    reads the model's own ``attention_mask``. The last part's output is
    named ``embedding``.
    """
    segments = []
    for i in range(count):
        source = "input_ids" if i == 0 else f"hidden{i - 1}"
        inputs = (source, "attention_mask") if mask_in(i) else (source,)
        output = "embedding" if i == count - 1 else f"hidden{i}"
        segments.append(Segment(mlmodelc=_part_name(i), inputs=inputs, outputs=(output,)))
    return SegmentsManifest(
        output_name="embedding",
        inputs=("input_ids", "attention_mask"),
        segments=tuple(segments),
    )


class _FakePart:
    """Stand-in for a loaded compiled part: adds its inputs up, then adds ``bias``."""

    def __init__(self, outputs: tuple[str, ...], bias: float = 0.0) -> None:
        self.outputs = outputs
        self.bias = bias
        self.feeds: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []

    def predict(self, feed: Mapping[str, Any]) -> dict[str, Any]:
        self.feeds.append(dict(feed))
        total = sum(feed.values())
        result = {name: total + self.bias for name in self.outputs}
        self.results.append(result)
        return result


class _FailingPart(_FakePart):
    """A part whose prediction raises."""

    def predict(self, feed: Mapping[str, Any]) -> dict[str, Any]:
        self.feeds.append(dict(feed))
        raise RuntimeError("part failed")


def _fake_parts(manifest: SegmentsManifest) -> list[_FakePart]:
    """One fake part per segment, adding 1 on top of the sum of its inputs."""
    return [_FakePart(segment.outputs, bias=1.0) for segment in manifest.segments]


def _model_inputs() -> dict[str, Any]:
    return {
        "input_ids": np.array([[1.0, 2.0, 3.0]]),
        "attention_mask": np.array([[1.0, 1.0, 0.0]]),
    }


def _make_parts_dir(root: Path, manifest: SegmentsManifest) -> Path:
    """Create a parts directory holding the manifest and an empty directory per part."""
    parts_dir = root / "model.parts"
    parts_dir.mkdir()
    for segment in manifest.segments:
        (parts_dir / segment.mlmodelc).mkdir()
    write_segments(parts_dir, manifest)
    return parts_dir


def _valid_payload() -> dict[str, Any]:
    """The JSON payload of a valid two-part manifest."""
    return {
        "format_version": SEGMENTS_FORMAT_VERSION,
        "output_name": "embedding",
        "inputs": ["input_ids", "attention_mask"],
        "segments": [
            {
                "mlmodelc": "part01.mlmodelc",
                "inputs": ["input_ids", "attention_mask"],
                "outputs": ["hidden"],
            },
            {
                "mlmodelc": "part02.mlmodelc",
                "inputs": ["hidden", "attention_mask"],
                "outputs": ["embedding", "extra"],
            },
        ],
    }


def _manifest_from_payload(payload: Mapping[str, Any]) -> SegmentsManifest:
    """Build the dataclasses straight from a payload, bypassing every check."""
    return SegmentsManifest(
        output_name=payload["output_name"],
        inputs=tuple(payload["inputs"]),
        segments=tuple(
            Segment(
                mlmodelc=item["mlmodelc"],
                inputs=tuple(item["inputs"]),
                outputs=tuple(item["outputs"]),
            )
            for item in payload["segments"]
        ),
    )


# --- manifest: JSON shape ----------------------------------------------------------


def test_constants() -> None:
    assert SEGMENTS_FILENAME == "segments.json"
    assert SEGMENTS_FORMAT_VERSION == 1


def test_to_json_has_the_documented_shape() -> None:
    manifest = _manifest_from_payload(_valid_payload())

    assert manifest.to_json() == _valid_payload()
    json.dumps(manifest.to_json())  # plain JSON types only


def test_from_json_turns_lists_into_tuples() -> None:
    manifest = SegmentsManifest.from_json(_valid_payload(), source="x")

    assert manifest.inputs == ("input_ids", "attention_mask")
    assert isinstance(manifest.segments, tuple)
    assert manifest.segments[1] == Segment(
        mlmodelc="part02.mlmodelc",
        inputs=("hidden", "attention_mask"),
        outputs=("embedding", "extra"),
    )
    assert manifest.output_name == "embedding"


def test_from_json_inverts_to_json() -> None:
    manifest = _chain_manifest(5)

    assert SegmentsManifest.from_json(manifest.to_json(), source="x") == manifest


def test_the_dataclasses_are_frozen() -> None:
    manifest = _chain_manifest(2)

    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.output_name = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.segments[0].mlmodelc = "other"  # type: ignore[misc]


def test_segments_error_is_a_value_error() -> None:
    assert issubclass(SegmentsError, ValueError)


# --- manifest: validation ----------------------------------------------------------


def _set(path: tuple[Any, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    """Mutation that sets ``payload[path[0]][path[1]]... = value``."""

    def mutate(payload: dict[str, Any]) -> None:
        target: Any = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _delete(key: str) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        del payload[key]

    return mutate


# Mutations that leave a payload a dataclass can still express, so
# ``write_segments`` must refuse the same cases ``from_json`` does.
_STRUCTURAL_CASES: dict[str, Callable[[dict[str, Any]], None]] = {
    "empty-output-name": _set(("output_name",), ""),
    "empty-model-inputs": _set(("inputs",), []),
    "duplicate-model-inputs": _set(("inputs",), ["input_ids", "input_ids"]),
    "empty-model-input-name": _set(("inputs",), ["input_ids", ""]),
    "no-segments": _set(("segments",), []),
    "empty-mlmodelc": _set(("segments", 0, "mlmodelc"), ""),
    "mlmodelc-with-slash": _set(("segments", 0, "mlmodelc"), "sub/part01.mlmodelc"),
    "mlmodelc-with-backslash": _set(("segments", 0, "mlmodelc"), "sub\\part01.mlmodelc"),
    "mlmodelc-absolute": _set(("segments", 0, "mlmodelc"), "/part01.mlmodelc"),
    "mlmodelc-dot": _set(("segments", 0, "mlmodelc"), "."),
    "mlmodelc-dot-dot": _set(("segments", 0, "mlmodelc"), ".."),
    "duplicate-mlmodelc": _set(("segments", 1, "mlmodelc"), "part01.mlmodelc"),
    "empty-segment-inputs": _set(("segments", 0, "inputs"), []),
    "duplicate-segment-inputs": _set(("segments", 0, "inputs"), ["input_ids", "input_ids"]),
    "empty-segment-input-name": _set(("segments", 0, "inputs"), [""]),
    "empty-segment-outputs": _set(("segments", 0, "outputs"), []),
    "duplicate-segment-outputs": _set(("segments", 1, "outputs"), ["embedding", "embedding"]),
    "empty-segment-output-name": _set(("segments", 0, "outputs"), [""]),
    "input-nobody-produces": _set(("segments", 0, "inputs"), ["input_ids", "nowhere"]),
    "input-produced-only-later": _set(("segments", 0, "inputs"), ["input_ids", "embedding"]),
    "input-of-its-own-output": _set(("segments", 0, "inputs"), ["hidden"]),
    "output-name-not-produced": _set(("output_name",), "missing"),
    "output-name-only-in-an-earlier-part": _set(("output_name",), "hidden"),
}

# Mutations only a JSON payload can carry.
_PAYLOAD_ONLY_CASES: dict[str, Callable[[dict[str, Any]], None]] = {
    "no-format-version": _delete("format_version"),
    "format-version-string": _set(("format_version",), "1"),
    "format-version-bool": _set(("format_version",), True),
    "format-version-float": _set(("format_version",), 1.0),
    "format-version-zero": _set(("format_version",), 0),
    "format-version-negative": _set(("format_version",), -1),
    "no-output-name": _delete("output_name"),
    "output-name-number": _set(("output_name",), 3),
    "no-model-inputs": _delete("inputs"),
    "model-inputs-string": _set(("inputs",), "input_ids"),
    "model-inputs-non-string": _set(("inputs",), ["input_ids", 7]),
    "no-segments-key": _delete("segments"),
    "segments-not-a-list": _set(("segments",), {"mlmodelc": "part01.mlmodelc"}),
    "segment-not-an-object": _set(("segments", 0), "part01.mlmodelc"),
    "mlmodelc-number": _set(("segments", 0, "mlmodelc"), 1),
    "mlmodelc-missing": lambda payload: payload["segments"][0].pop("mlmodelc"),
    "segment-inputs-missing": lambda payload: payload["segments"][0].pop("inputs"),
    "segment-outputs-missing": lambda payload: payload["segments"][0].pop("outputs"),
    "segment-outputs-not-a-list": _set(("segments", 0, "outputs"), "hidden"),
    "segment-inputs-non-string": _set(("segments", 0, "inputs"), [None]),
}


def _mutated(mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    payload = copy.deepcopy(_valid_payload())
    mutate(payload)
    return payload


def test_the_baseline_payload_is_valid() -> None:
    SegmentsManifest.from_json(_valid_payload(), source="x")


@pytest.mark.parametrize("name", [*_STRUCTURAL_CASES, *_PAYLOAD_ONLY_CASES])
def test_from_json_rejects_an_invalid_payload(name: str) -> None:
    mutate = {**_STRUCTURAL_CASES, **_PAYLOAD_ONLY_CASES}[name]

    with pytest.raises(SegmentsError) as excinfo:
        SegmentsManifest.from_json(_mutated(mutate), source="the-source")

    assert "the-source" in str(excinfo.value)


@pytest.mark.parametrize("payload", [[], "text", 3, None, [_valid_payload()]])
def test_from_json_rejects_a_payload_that_is_not_an_object(payload: Any) -> None:
    with pytest.raises(SegmentsError, match="the-source"):
        SegmentsManifest.from_json(payload, source="the-source")


@pytest.mark.parametrize("name", list(_STRUCTURAL_CASES))
def test_write_segments_rejects_an_invalid_manifest_and_writes_nothing(
    tmp_path: Path, name: str
) -> None:
    manifest = _manifest_from_payload(_mutated(_STRUCTURAL_CASES[name]))
    parts_dir = tmp_path / "model.parts"
    parts_dir.mkdir()

    with pytest.raises(SegmentsError):
        write_segments(parts_dir, manifest)

    assert not (parts_dir / SEGMENTS_FILENAME).exists()


def test_a_later_part_may_read_a_model_input_and_an_earlier_output_together() -> None:
    payload = _valid_payload()
    payload["segments"].append(
        {
            "mlmodelc": "part03.mlmodelc",
            "inputs": ["attention_mask", "hidden", "extra"],
            "outputs": ["embedding"],
        }
    )

    manifest = SegmentsManifest.from_json(payload, source="x")

    assert len(manifest.segments) == 3


def test_a_newer_format_version_asks_for_an_upgrade_or_a_recompile() -> None:
    payload = _valid_payload()
    payload["format_version"] = SEGMENTS_FORMAT_VERSION + 1

    with pytest.raises(SegmentsError) as excinfo:
        SegmentsManifest.from_json(payload, source="the-source")

    message = str(excinfo.value)
    assert "the-source" in message
    assert "newer eeANE" in message
    assert "upgrade eeANE" in message
    assert "recompile" in message


# --- reading and writing the manifest ----------------------------------------------


def test_write_then_load_round_trips(tmp_path: Path) -> None:
    manifest = _chain_manifest(3)
    parts_dir = tmp_path / "model.parts"
    parts_dir.mkdir()
    for segment in manifest.segments:
        (parts_dir / segment.mlmodelc).mkdir()

    written = write_segments(parts_dir, manifest)

    assert written == parts_dir / SEGMENTS_FILENAME
    assert load_segments(parts_dir) == manifest


def test_write_segments_formats_the_file(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["segments"][1]["outputs"] = ["embedding", "出力"]
    manifest = _manifest_from_payload(payload)
    parts_dir = tmp_path / "model.parts"
    parts_dir.mkdir()

    written = write_segments(parts_dir, manifest)

    text = written.read_text(encoding="utf-8")
    assert text == json.dumps(manifest.to_json(), indent=2, ensure_ascii=False) + "\n"
    assert "出力" in text


def test_write_segments_propagates_os_errors(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        write_segments(tmp_path / "absent.parts", _chain_manifest(2))


def test_part_paths_follow_the_manifest_order(tmp_path: Path) -> None:
    manifest = _chain_manifest(4)

    assert part_paths(tmp_path, manifest) == [tmp_path / _part_name(i) for i in range(4)]


def test_load_segments_without_a_manifest_file(tmp_path: Path) -> None:
    with pytest.raises(SegmentsError, match=SEGMENTS_FILENAME):
        load_segments(tmp_path)


def test_load_segments_of_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(SegmentsError):
        load_segments(tmp_path / "absent.parts")


@pytest.mark.parametrize(
    "content",
    [b"{not json", b"", b"[1, 2]", b'"text"', b"\xff\xfe\x00bad", b'{"format_version": 1}'],
    ids=["broken-json", "empty", "list", "string", "not-utf8", "keys-missing"],
)
def test_load_segments_of_an_unusable_file(tmp_path: Path, content: bytes) -> None:
    (tmp_path / SEGMENTS_FILENAME).write_bytes(content)

    with pytest.raises(SegmentsError, match=SEGMENTS_FILENAME):
        load_segments(tmp_path)


def test_load_segments_when_the_manifest_is_a_directory(tmp_path: Path) -> None:
    (tmp_path / SEGMENTS_FILENAME).mkdir()

    with pytest.raises(SegmentsError):
        load_segments(tmp_path)


def test_load_segments_names_the_missing_part(tmp_path: Path) -> None:
    manifest = _chain_manifest(3)
    parts_dir = _make_parts_dir(tmp_path, manifest)
    (parts_dir / _part_name(1)).rmdir()

    with pytest.raises(SegmentsError) as excinfo:
        load_segments(parts_dir)

    assert _part_name(1) in str(excinfo.value)


def test_load_segments_refuses_a_part_that_is_a_file(tmp_path: Path) -> None:
    manifest = _chain_manifest(2)
    parts_dir = _make_parts_dir(tmp_path, manifest)
    (parts_dir / _part_name(0)).rmdir()
    (parts_dir / _part_name(0)).write_bytes(b"not a directory")

    with pytest.raises(SegmentsError, match=_part_name(0)):
        load_segments(parts_dir)


def test_load_segments_reports_a_newer_format(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["format_version"] = SEGMENTS_FORMAT_VERSION + 1
    (tmp_path / SEGMENTS_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SegmentsError, match="newer eeANE"):
        load_segments(tmp_path)


# --- is_segmented ------------------------------------------------------------------


def test_is_segmented_is_true_for_a_directory_holding_the_manifest(tmp_path: Path) -> None:
    parts_dir = _make_parts_dir(tmp_path, _chain_manifest(2))

    assert is_segmented(parts_dir) is True


def test_is_segmented_does_not_check_the_parts(tmp_path: Path) -> None:
    (tmp_path / SEGMENTS_FILENAME).write_text("{}", encoding="utf-8")

    assert is_segmented(tmp_path) is True


def test_is_segmented_is_false_for_a_directory_without_the_manifest(tmp_path: Path) -> None:
    (tmp_path / "model.mlmodelc").mkdir()

    assert is_segmented(tmp_path) is False
    assert is_segmented(tmp_path / "model.mlmodelc") is False


def test_is_segmented_is_false_for_a_missing_path(tmp_path: Path) -> None:
    assert is_segmented(tmp_path / "absent.parts") is False


def test_is_segmented_is_false_for_a_file(tmp_path: Path) -> None:
    file = tmp_path / "model.parts"
    file.write_text("x", encoding="utf-8")

    assert is_segmented(file) is False


def test_is_segmented_is_false_when_the_manifest_is_a_directory(tmp_path: Path) -> None:
    (tmp_path / SEGMENTS_FILENAME).mkdir()

    assert is_segmented(tmp_path) is False


# --- SegmentedModel: construction --------------------------------------------------


def test_the_part_count_must_match_the_manifest() -> None:
    manifest = _chain_manifest(3)

    with pytest.raises(ValueError, match="3"):
        SegmentedModel(_fake_parts(manifest)[:2], manifest)
    with pytest.raises(ValueError):
        SegmentedModel([*_fake_parts(manifest), _FakePart(("x",))], manifest)


def test_parts_and_manifest_are_exposed_read_only() -> None:
    manifest = _chain_manifest(2)
    parts = _fake_parts(manifest)

    model = SegmentedModel(parts, manifest)

    assert model.parts == tuple(parts)
    assert isinstance(model.parts, tuple)
    assert model.manifest is manifest
    with pytest.raises(AttributeError):
        model.parts = ()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        model.manifest = manifest  # type: ignore[misc]


def test_load_hands_the_part_paths_to_the_injected_loader_in_order(tmp_path: Path) -> None:
    manifest = _chain_manifest(4)
    parts_dir = _make_parts_dir(tmp_path, manifest)
    loaded: list[Path] = []

    def load_part(path: Path) -> str:
        loaded.append(path)
        return f"model-of-{path.name}"

    model = SegmentedModel.load(parts_dir, load_part)

    assert loaded == [parts_dir / _part_name(i) for i in range(4)]
    assert model.parts == tuple(f"model-of-{_part_name(i)}" for i in range(4))
    assert model.manifest == manifest


def test_load_loads_nothing_when_the_manifest_is_unusable(tmp_path: Path) -> None:
    manifest = _chain_manifest(2)
    parts_dir = _make_parts_dir(tmp_path, manifest)
    (parts_dir / _part_name(1)).rmdir()
    loaded: list[Path] = []

    with pytest.raises(SegmentsError):
        SegmentedModel.load(parts_dir, lambda path: loaded.append(path))

    assert loaded == []


# --- SegmentedModel.predict --------------------------------------------------------


@pytest.mark.parametrize("count", [2, 4, 16])
def test_parts_run_in_order_each_feeding_the_next(count: int) -> None:
    manifest = _chain_manifest(count)
    parts = _fake_parts(manifest)
    inputs = _model_inputs()

    result = SegmentedModel(parts, manifest).predict(inputs)

    # Every part adds the model's mask and 1 on top of the previous result.
    expected = inputs["input_ids"] + count * (inputs["attention_mask"] + 1.0)
    assert list(result) == ["embedding"]
    np.testing.assert_allclose(result["embedding"], expected)
    assert all(len(part.feeds) == 1 for part in parts)


@pytest.mark.parametrize("count", [2, 4, 16])
def test_each_part_receives_the_output_of_the_one_before(count: int) -> None:
    manifest = _chain_manifest(count)
    parts = _fake_parts(manifest)

    SegmentedModel(parts, manifest).predict(_model_inputs())

    for i in range(1, count):
        assert parts[i].feeds[0][f"hidden{i - 1}"] is parts[i - 1].results[0][f"hidden{i - 1}"]


def test_a_model_input_is_available_to_every_later_part() -> None:
    manifest = _chain_manifest(3, mask_in=lambda i: i in (0, 2))
    parts = _fake_parts(manifest)
    inputs = _model_inputs()

    SegmentedModel(parts, manifest).predict(inputs)

    assert set(parts[0].feeds[0]) == {"input_ids", "attention_mask"}
    assert set(parts[1].feeds[0]) == {"hidden0"}
    assert set(parts[2].feeds[0]) == {"hidden1", "attention_mask"}
    assert parts[2].feeds[0]["attention_mask"] is inputs["attention_mask"]


def test_a_part_gets_exactly_the_inputs_it_declares() -> None:
    manifest = _chain_manifest(3, mask_in=lambda i: i == 1)
    parts = _fake_parts(manifest)
    inputs = {**_model_inputs(), "not_declared_anywhere": np.array([9.0])}

    SegmentedModel(parts, manifest).predict(inputs)

    for part, segment in zip(parts, manifest.segments, strict=True):
        assert list(part.feeds[0]) == list(segment.inputs)


def test_the_callers_mapping_is_left_alone() -> None:
    manifest = _chain_manifest(3)
    inputs = _model_inputs()
    before = dict(inputs)

    SegmentedModel(_fake_parts(manifest), manifest).predict(inputs)

    assert inputs == before


def test_the_result_is_the_last_parts_whole_prediction_as_a_dict() -> None:
    manifest = SegmentsManifest(
        output_name="embedding",
        inputs=("input_ids",),
        segments=(
            Segment("part01.mlmodelc", ("input_ids",), ("hidden",)),
            Segment("part02.mlmodelc", ("hidden",), ("embedding", "extra")),
        ),
    )

    class _ReadOnlyPart(_FakePart):
        def predict(self, feed: Mapping[str, Any]) -> Any:  # type: ignore[override]
            return MappingProxyType(super().predict(feed))

    parts = [_FakePart(("hidden",)), _ReadOnlyPart(("embedding", "extra"))]

    result = SegmentedModel(parts, manifest).predict({"input_ids": np.array([1.0])})

    assert type(result) is dict
    assert set(result) == {"embedding", "extra"}


def test_a_part_with_nothing_to_hand_on_is_a_missing_input_for_the_next() -> None:
    manifest = _chain_manifest(2)
    parts = [_FakePart(("somewhere_else",)), _FakePart(("embedding",))]

    with pytest.raises(ValueError, match="hidden0"):
        SegmentedModel(parts, manifest).predict(_model_inputs())

    assert len(parts[0].feeds) == 1
    assert parts[1].feeds == []


def test_a_missing_model_input_is_named_before_any_part_runs() -> None:
    manifest = _chain_manifest(3)
    parts = _fake_parts(manifest)
    retained: list[Any] = []

    with pytest.raises(ValueError) as excinfo:
        SegmentedModel(parts, manifest).predict(
            {"input_ids": np.array([1.0])}, retain=lambda part, feed: retained.append(part)
        )

    assert "attention_mask" in str(excinfo.value)
    assert all(part.feeds == [] for part in parts)
    assert retained == []


def test_every_missing_input_is_named() -> None:
    manifest = _chain_manifest(2)

    with pytest.raises(ValueError) as excinfo:
        SegmentedModel(_fake_parts(manifest), manifest).predict({})

    assert "input_ids" in str(excinfo.value)
    assert "attention_mask" in str(excinfo.value)


# --- SegmentedModel.predict: retain ------------------------------------------------


@pytest.mark.parametrize("count", [2, 4, 16])
def test_retain_sees_the_feed_of_every_part_in_order(count: int) -> None:
    manifest = _chain_manifest(count)
    parts = _fake_parts(manifest)
    seen: list[tuple[Any, dict[str, Any]]] = []

    SegmentedModel(parts, manifest).predict(
        _model_inputs(), retain=lambda part, feed: seen.append((part, feed))
    )

    assert [part for part, _feed in seen] == parts
    for (part, feed), segment in zip(seen, manifest.segments, strict=True):
        assert list(feed) == list(segment.inputs)
        assert all(feed[name] is part.feeds[0][name] for name in feed)


def test_retain_keeps_the_intermediate_outputs_reachable() -> None:
    manifest = _chain_manifest(3)
    parts = _fake_parts(manifest)
    seen: list[dict[str, Any]] = []

    SegmentedModel(parts, manifest).predict(
        _model_inputs(), retain=lambda _part, feed: seen.append(feed)
    )

    assert seen[1]["hidden0"] is parts[0].results[0]["hidden0"]
    assert seen[2]["hidden1"] is parts[1].results[0]["hidden1"]


def test_retain_runs_after_the_part_call() -> None:
    manifest = _chain_manifest(2)
    parts = _fake_parts(manifest)
    seen_calls_at_retain: list[int] = []

    def retain(part: _FakePart, _feed: dict[str, Any]) -> None:
        seen_calls_at_retain.append(len(part.feeds))

    SegmentedModel(parts, manifest).predict(_model_inputs(), retain=retain)

    assert seen_calls_at_retain == [1, 1]


def test_retain_also_sees_the_feed_of_the_part_that_raises() -> None:
    manifest = _chain_manifest(4)
    parts: list[_FakePart] = [
        _FakePart(("hidden0",), bias=1.0),
        _FakePart(("hidden1",), bias=1.0),
        _FailingPart(("hidden2",)),
        _FakePart(("embedding",), bias=1.0),
    ]
    seen: list[tuple[Any, dict[str, Any]]] = []

    with pytest.raises(RuntimeError, match="part failed"):
        SegmentedModel(parts, manifest).predict(
            _model_inputs(), retain=lambda part, feed: seen.append((part, feed))
        )

    assert [part for part, _feed in seen] == parts[:3]
    assert seen[2][1]["hidden1"] is parts[1].results[0]["hidden1"]
    assert parts[3].feeds == []


def test_retain_is_called_for_a_failing_first_part() -> None:
    manifest = _chain_manifest(2)
    parts: list[_FakePart] = [_FailingPart(("hidden0",)), _FakePart(("embedding",))]
    seen: list[Any] = []

    with pytest.raises(RuntimeError):
        SegmentedModel(parts, manifest).predict(
            _model_inputs(), retain=lambda part, feed: seen.append(part)
        )

    assert seen == [parts[0]]


def test_predict_works_without_retain() -> None:
    manifest = _chain_manifest(2)

    result = SegmentedModel(_fake_parts(manifest), manifest).predict(_model_inputs())

    assert "embedding" in result


def test_the_same_model_can_predict_again() -> None:
    manifest = _chain_manifest(3)
    parts = _fake_parts(manifest)
    model = SegmentedModel(parts, manifest)

    first = model.predict(_model_inputs())
    second = model.predict(_model_inputs())

    np.testing.assert_allclose(first["embedding"], second["embedding"])
    assert all(len(part.feeds) == 2 for part in parts)


# --- module boundaries -------------------------------------------------------------


def test_module_does_not_import_the_conversion_stack() -> None:
    script = (
        "import eeane.segmented\n"
        "import sys\n"
        "for name in ('torch', 'transformers', 'coremltools', 'eeane.compiler'):\n"
        "    assert name not in sys.modules, name\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr


def test_public_api_is_exported() -> None:
    for name in (
        "SEGMENTS_FILENAME",
        "SEGMENTS_FORMAT_VERSION",
        "SegmentsError",
        "Segment",
        "SegmentsManifest",
        "is_segmented",
        "load_segments",
        "write_segments",
        "part_paths",
        "SegmentedModel",
    ):
        assert hasattr(segmented, name), name
