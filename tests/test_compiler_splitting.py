"""Tests for splitting a compiled model the Neural Engine would refuse.

``eeane.compiler.splitting`` cuts a converted model into compiled parts
that each fit the Neural Engine size limit. Two layers:

* Stub-driven tests of the splitting logic itself. Bisecting, compiling
  and reading a package's tensor names are replaced by
  :class:`_SplitStubs`, which works on plain directories and decides the
  compiled size of every chunk by where it sits in the split tree, so the
  recursion, the assembly of the parts and the clean-up after a failure
  are exercised without coremltools or the Core ML compiler.
* One test of the real coremltools bisection over a tiny converted
  model. It reads and writes packages only; nothing is compiled, so the
  Core ML compiler is not needed.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import coremltools as ct
import numpy as np
import pytest
import torch

from eeane import segmented
from eeane.compiler import artifacts, conversion, splitting
from eeane.compiler.artifacts import CompileError

# Base name of the variant the stub-driven tests split.
_STEM = "s16_b1_eager_macos13"

# Tensor names of the model as converted.
_MODEL_INPUTS = ("input_ids", "attention_mask")
_OUTPUT_NAME = "embedding"

# Stand-in for the Neural Engine size limit, and compiled sizes on either
# side of it, all small enough for a stub artifact of a few bytes.
_LIMIT = 100
_FITS = 60
_TOO_BIG = 150

# Marker file a fake chunk package carries: where it sits in the split tree
# and which tensors it reads and writes.
_FAKE_PACKAGE_FILE = "fake_package.json"


def _sizes(table: dict[str, int], default: int = _FITS) -> Callable[[str], int]:
    """Look a chunk's compiled size up by its label, with a default for the rest."""
    return lambda label: table.get(label, default)


class _SplitStubs:
    """Stand-ins for bisecting, compiling and reading a package.

    A package is a plain directory. The one the conversion produced
    carries no marker and is the root of the split tree (label ``""``);
    every chunk the fake bisection writes carries a marker naming its
    label -- the chunk indices from the root, so ``"21"`` is the first
    half of the second half -- and the tensors it reads and writes. The
    compiled size of every package is looked up by its label, which is
    how a test decides which chunks still exceed the limit.

    The fake bisection wires the chunks the way coremltools does: the
    first half reads every input of what it was cut from and hands one
    boundary tensor on; the second half reads that tensor plus the
    attention mask, which the whole model needs again further down.
    """

    def __init__(self, size_of: Callable[[str], int], output_name: str = _OUTPUT_NAME) -> None:
        """Record how large every chunk compiles to.

        Args:
            size_of: Compiled size in bytes of the package with a label.
            output_name: Output tensor of the converted model.
        """
        self.size_of = size_of
        self.output_name = output_name
        # Labels of the packages bisected and compiled, in call order.
        self.bisected: list[str] = []
        self.compiled: list[str] = []

    def describe(self, package: Path) -> dict[str, Any]:
        """Return the label and tensor names of a fake package."""
        marker = package / _FAKE_PACKAGE_FILE
        if marker.is_file():
            return json.loads(marker.read_text(encoding="utf-8"))
        return {"label": "", "inputs": list(_MODEL_INPUTS), "outputs": [self.output_name]}

    def bisect(self, package: Path, output_dir: Path) -> tuple[Path, Path]:
        """Write the two halves of ``package`` the way the real bisection names them."""
        assert package.is_dir(), "a package must still exist when it is bisected"
        parent = self.describe(package)
        label = parent["label"]
        self.bisected.append(label)
        boundary = f"hidden_{label or 'root'}"
        halves = (
            {"label": label + "1", "inputs": parent["inputs"], "outputs": [boundary]},
            {
                "label": label + "2",
                "inputs": [boundary, *[n for n in parent["inputs"] if n == "attention_mask"]],
                "outputs": parent["outputs"],
            },
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for index, description in enumerate(halves, start=1):
            path = output_dir / f"{package.stem}_chunk{index}.mlpackage"
            path.mkdir()
            (path / _FAKE_PACKAGE_FILE).write_text(json.dumps(description), encoding="utf-8")
            paths.append(path)
        return paths[0], paths[1]

    def io_names(self, package: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the inputs and outputs a fake package declares."""
        description = self.describe(package)
        return tuple(description["inputs"]), tuple(description["outputs"])

    def compile(self, package: Path, mlmodelc: Path) -> None:
        """Write a compiled model of the size the package's label is given."""
        label = self.describe(package)["label"]
        self.compiled.append(label)
        if mlmodelc.exists():
            shutil.rmtree(mlmodelc)
        mlmodelc.mkdir(parents=True)
        (mlmodelc / "weight.bin").write_bytes(b"\0" * self.size_of(label))

    def install(self, monkeypatch: pytest.MonkeyPatch) -> _SplitStubs:
        """Put the stand-ins in place of the real steps; return ``self``."""
        monkeypatch.setattr(splitting, "bisect_package", self.bisect)
        monkeypatch.setattr(splitting, "package_io_names", self.io_names)
        monkeypatch.setattr(conversion, "compile_model", self.compile)
        return self


def _prepare_variant(directory: Path, stubs: _SplitStubs) -> tuple[Path, Path]:
    """Create a converted package and its freshly compiled artifact.

    Returns:
        The ``.mlpackage`` and ``.mlmodelc`` paths.
    """
    package = directory / f"{_STEM}.mlpackage"
    package.mkdir(parents=True)
    mlmodelc = directory / f"{_STEM}.mlmodelc"
    stubs.compile(package, mlmodelc)
    stubs.compiled.clear()
    return package, mlmodelc


def _split(
    package: Path,
    mlmodelc: Path,
    *,
    limit_bytes: int | None = _LIMIT,
    keep_part_packages: bool = False,
    messages: list[str] | None = None,
) -> dict[str, Any] | None:
    """Call the function under test with this module's defaults."""
    return splitting.split_oversized_variant(
        package,
        mlmodelc,
        output_name=_OUTPUT_NAME,
        limit_bytes=limit_bytes,
        limit_source="os",
        keep_part_packages=keep_part_packages,
        progress=(messages if messages is not None else []).append,
    )


def _snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    """Every path below ``directory`` with its size and modification time."""
    return {
        str(path.relative_to(directory)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(directory.rglob("*"))
    }


def _assert_nothing_served(directory: Path) -> None:
    """No artifact, half-made or oversized, may be left to serve or reuse."""
    assert not (directory / f"{_STEM}.mlmodelc").exists()
    assert not (directory / f"{_STEM}.parts").exists()
    assert not (directory / f"{_STEM}.split_tmp").exists()


# --- a model that fits ------------------------------------------------------------


@pytest.mark.parametrize("size", [_LIMIT, _LIMIT - 1])
def test_a_model_within_the_limit_is_left_exactly_as_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    """A model of exactly the limit is still accepted, so nothing is bisected or touched."""
    stubs = _SplitStubs(_sizes({"": size})).install(monkeypatch)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)
    before = _snapshot(tmp_path)
    messages: list[str] = []

    assert _split(package, mlmodelc, messages=messages) is None

    assert _snapshot(tmp_path) == before
    assert stubs.bisected == [] and stubs.compiled == []
    assert messages == []


def test_without_a_limit_even_an_oversized_model_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``limit_bytes=None`` means no limit applies: nothing is split."""
    stubs = _SplitStubs(_sizes({"": _TOO_BIG * 100})).install(monkeypatch)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)
    before = _snapshot(tmp_path)

    assert _split(package, mlmodelc, limit_bytes=None) is None

    assert _snapshot(tmp_path) == before
    assert stubs.bisected == []


@pytest.mark.parametrize("limit_bytes", [_LIMIT, None])
def test_a_fitting_model_removes_the_parts_of_an_earlier_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit_bytes: int | None
) -> None:
    """Parts left by an earlier run would shadow the fresh single artifact."""
    stubs = _SplitStubs(_sizes({"": _FITS})).install(monkeypatch)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)
    stale = tmp_path / f"{_STEM}.parts"
    (stale / "part01.mlmodelc").mkdir(parents=True)
    (stale / segmented.SEGMENTS_FILENAME).write_text("{}", encoding="utf-8")

    assert _split(package, mlmodelc, limit_bytes=limit_bytes) is None

    assert not stale.exists()
    assert mlmodelc.is_dir()
    assert package.is_dir()
    assert artifacts.served_artifact_path(tmp_path, _STEM) == mlmodelc


# --- splitting ----------------------------------------------------------------------


def test_an_oversized_model_is_split_into_two_parts_that_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both halves fit: two compiled parts, a manifest closing the data flow, nothing else."""
    stubs = _SplitStubs(_sizes({"": _TOO_BIG})).install(monkeypatch)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)

    record = _split(package, mlmodelc)

    parts_dir = tmp_path / f"{_STEM}.parts"
    manifest = segmented.load_segments(parts_dir)
    assert manifest.output_name == _OUTPUT_NAME
    assert manifest.inputs == _MODEL_INPUTS
    assert manifest.segments == (
        segmented.Segment("part01.mlmodelc", _MODEL_INPUTS, ("hidden_root",)),
        segmented.Segment("part02.mlmodelc", ("hidden_root", "attention_mask"), (_OUTPUT_NAME,)),
    )
    assert sorted(path.name for path in parts_dir.iterdir()) == [
        "part01.mlmodelc",
        "part02.mlmodelc",
        segmented.SEGMENTS_FILENAME,
    ]
    # The oversized artifact and the work area are gone; the converted
    # package is the caller's to keep or drop.
    assert not mlmodelc.exists()
    assert not (tmp_path / f"{_STEM}.split_tmp").exists()
    assert package.is_dir()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        f"{_STEM}.mlpackage",
        f"{_STEM}.parts",
    ]
    assert stubs.bisected == [""]
    assert stubs.compiled == ["1", "2"]
    assert artifacts.served_artifact_path(tmp_path, _STEM) == parts_dir
    assert record == {
        "method": splitting.SPLIT_METHOD,
        "limit_bytes": _LIMIT,
        "limit_source": "os",
        "size_before_bytes": _TOO_BIG,
        "parts": [
            {
                "name": "part01.mlmodelc",
                "size_bytes": _FITS,
                "inputs": list(_MODEL_INPUTS),
                "outputs": ["hidden_root"],
            },
            {
                "name": "part02.mlmodelc",
                "size_bytes": _FITS,
                "inputs": ["hidden_root", "attention_mask"],
                "outputs": [_OUTPUT_NAME],
            },
        ],
    }
    json.dumps(record)


def test_the_split_is_announced_and_its_result_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One line says why the model is split, one says what came of it."""
    stubs = _SplitStubs(_sizes({"": _TOO_BIG})).install(monkeypatch)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)
    messages: list[str] = []

    _split(package, mlmodelc, messages=messages)

    assert len(messages) == 2
    assert "above the Neural Engine size limit (100 bytes)" in messages[0]
    assert "splitting it into parts that each fit" in messages[0]
    assert messages[1].startswith("split into 2 parts (")
    assert messages[1].count("GiB") == 2
    assert messages[1].endswith(" s")


def test_only_a_half_still_too_large_is_split_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first half fits, the second does not: three parts, depth first."""
    stubs = _SplitStubs(_sizes({"": _TOO_BIG, "2": _TOO_BIG})).install(monkeypatch)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)

    record = _split(package, mlmodelc)

    assert stubs.bisected == ["", "2"]
    manifest = segmented.load_segments(tmp_path / f"{_STEM}.parts")
    assert [segment.mlmodelc for segment in manifest.segments] == [
        "part01.mlmodelc",
        "part02.mlmodelc",
        "part03.mlmodelc",
    ]
    # part01 is the first half, part02/part03 the two quarters of the second.
    assert [segment.outputs for segment in manifest.segments] == [
        ("hidden_root",),
        ("hidden_2",),
        (_OUTPUT_NAME,),
    ]
    assert manifest.segments[1].inputs == ("hidden_root", "attention_mask")
    assert manifest.segments[2].inputs == ("hidden_2", "attention_mask")
    assert record is not None
    assert [part["size_bytes"] for part in record["parts"]] == [_FITS] * 3
    assert not (tmp_path / f"{_STEM}.split_tmp").exists()


def test_the_order_is_depth_first_when_the_first_half_is_split_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every part of the first half runs before the second half."""
    stubs = _SplitStubs(_sizes({"": _TOO_BIG, "1": _TOO_BIG})).install(monkeypatch)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)

    _split(package, mlmodelc)

    manifest = segmented.load_segments(tmp_path / f"{_STEM}.parts")
    assert [segment.outputs for segment in manifest.segments] == [
        ("hidden_1",),
        ("hidden_root",),
        (_OUTPUT_NAME,),
    ]


def test_the_depth_cap_still_allows_sixteen_parts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model needing every one of the 2**MAX_SPLIT_DEPTH parts is still split."""
    depth = splitting.MAX_SPLIT_DEPTH
    stubs = _SplitStubs(lambda label: _FITS if len(label) == depth else _TOO_BIG)
    stubs.install(monkeypatch)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)

    record = _split(package, mlmodelc)

    assert record is not None
    assert len(record["parts"]) == 2**depth == 16
    assert record["parts"][-1]["name"] == "part16.mlmodelc"
    segmented.load_segments(tmp_path / f"{_STEM}.parts")


def test_a_part_still_too_large_at_the_depth_cap_fails_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sixteen parts that do not fit are an error, and nothing is left to serve."""
    stubs = _SplitStubs(_sizes({}, default=_TOO_BIG)).install(monkeypatch)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)

    with pytest.raises(CompileError) as caught:
        _split(package, mlmodelc)

    message = str(caught.value)
    assert "16 parts" in message
    assert "100 bytes" in message
    assert "GiB" in message
    assert stubs.bisected == ["", "1", "11", "111"]
    _assert_nothing_served(tmp_path)
    assert package.is_dir()


def test_a_model_without_a_bisect_point_fails_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """coremltools' "no bisect point" reaches the user as a split failure."""
    # The real bisect_package, over a coremltools that finds no cut point.
    stubs = _SplitStubs(_sizes({"": _TOO_BIG}))
    monkeypatch.setattr(splitting, "package_io_names", stubs.io_names)
    monkeypatch.setattr(conversion, "compile_model", stubs.compile)

    def _no_bisect_point(*args: Any, **kwargs: Any) -> None:
        raise ValueError("Not able to find the bisect point in the model.")

    monkeypatch.setattr(splitting.ct.models.utils, "bisect_model", _no_bisect_point)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)

    with pytest.raises(CompileError, match="could not be split") as caught:
        _split(package, mlmodelc)

    assert "bisect point" in str(caught.value)
    _assert_nothing_served(tmp_path)


def test_a_chunk_that_fails_to_compile_fails_the_split_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A compiler error on any chunk leaves neither parts nor the oversized artifact."""
    stubs = _SplitStubs(_sizes({"": _TOO_BIG})).install(monkeypatch)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)

    def _failing_compile(chunk: Path, destination: Path) -> None:
        if stubs.describe(chunk)["label"] == "2":
            raise RuntimeError("coremlcompiler failed (1): boom")
        stubs.compile(chunk, destination)

    monkeypatch.setattr(conversion, "compile_model", _failing_compile)

    with pytest.raises(CompileError, match="boom") as caught:
        _split(package, mlmodelc)

    assert isinstance(caught.value.__cause__, RuntimeError)
    _assert_nothing_served(tmp_path)


def test_a_failed_split_also_removes_the_parts_of_an_earlier_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parts built from an older conversion must not survive a failed rebuild."""
    stubs = _SplitStubs(_sizes({}, default=_TOO_BIG)).install(monkeypatch)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)
    earlier = tmp_path / f"{_STEM}.parts"
    (earlier / "part01.mlmodelc").mkdir(parents=True)

    with pytest.raises(CompileError):
        _split(package, mlmodelc)

    _assert_nothing_served(tmp_path)


@pytest.mark.parametrize(
    ("label", "names"),
    [
        # The second half reads a tensor nobody produces.
        ("2", (("never_produced",), (_OUTPUT_NAME,))),
        # The last part does not produce the model's output.
        ("2", (("hidden_root",), ("something_else",))),
    ],
)
def test_parts_whose_data_flow_does_not_close_fail_the_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    names: tuple[tuple[str, ...], tuple[str, ...]],
) -> None:
    """A manifest that would not run is never written; the split fails instead."""
    stubs = _SplitStubs(_sizes({"": _TOO_BIG})).install(monkeypatch)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)

    def _io_names(chunk: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return names if stubs.describe(chunk)["label"] == label else stubs.io_names(chunk)

    monkeypatch.setattr(splitting, "package_io_names", _io_names)

    with pytest.raises(CompileError) as caught:
        _split(package, mlmodelc)

    assert isinstance(caught.value.__cause__, segmented.SegmentsError)
    _assert_nothing_served(tmp_path)


def test_the_part_packages_are_kept_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``keep_part_packages`` every part's package sits next to its compiled model."""
    stubs = _SplitStubs(_sizes({"": _TOO_BIG, "2": _TOO_BIG})).install(monkeypatch)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)

    _split(package, mlmodelc, keep_part_packages=True)

    parts_dir = tmp_path / f"{_STEM}.parts"
    assert sorted(path.name for path in parts_dir.iterdir()) == [
        "part01.mlmodelc",
        "part01.mlpackage",
        "part02.mlmodelc",
        "part02.mlpackage",
        "part03.mlmodelc",
        "part03.mlpackage",
        segmented.SEGMENTS_FILENAME,
    ]
    labels = [stubs.describe(parts_dir / f"part{i:02d}.mlpackage")["label"] for i in (1, 2, 3)]
    assert labels == ["1", "21", "22"]
    assert not (tmp_path / f"{_STEM}.split_tmp").exists()


def test_a_split_replaces_the_parts_of_an_earlier_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only this run's parts remain, whatever an earlier split left behind."""
    stubs = _SplitStubs(_sizes({"": _TOO_BIG})).install(monkeypatch)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)
    earlier = tmp_path / f"{_STEM}.parts"
    (earlier / "part09.mlmodelc").mkdir(parents=True)

    _split(package, mlmodelc)

    assert sorted(path.name for path in earlier.iterdir()) == [
        "part01.mlmodelc",
        "part02.mlmodelc",
        segmented.SEGMENTS_FILENAME,
    ]


def test_a_work_area_left_by_an_interrupted_run_is_cleared_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Debris of an earlier, interrupted split must not end up among the parts."""
    stubs = _SplitStubs(_sizes({"": _TOO_BIG})).install(monkeypatch)
    package, mlmodelc = _prepare_variant(tmp_path, stubs)
    leftover = tmp_path / f"{_STEM}.split_tmp"
    (leftover / "parts" / "part07.mlmodelc").mkdir(parents=True)

    record = _split(package, mlmodelc)

    assert record is not None and len(record["parts"]) == 2
    assert not leftover.exists()
    assert not (tmp_path / f"{_STEM}.parts" / "part07.mlmodelc").exists()


# --- bisect_package / package_io_names ---------------------------------------------


def test_bisect_package_asks_coremltools_for_two_unverified_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model is handed over by path, unmerged, with the correctness check off."""
    calls: list[tuple[Any, ...]] = []

    def _fake_bisect_model(model: Any, output_dir: str, **kwargs: Any) -> None:
        calls.append((model, output_dir, kwargs))
        for index in (1, 2):
            (Path(output_dir) / f"model_chunk{index}.mlpackage").mkdir(parents=True)

    monkeypatch.setattr(splitting.ct.models.utils, "bisect_model", _fake_bisect_model)
    package = tmp_path / "model.mlpackage"
    package.mkdir()

    chunks = splitting.bisect_package(package, tmp_path / "out")

    assert chunks == (
        tmp_path / "out" / "model_chunk1.mlpackage",
        tmp_path / "out" / "model_chunk2.mlpackage",
    )
    assert calls == [
        (
            str(package),
            str(tmp_path / "out"),
            {"merge_chunks_to_pipeline": False, "check_output_correctness": False},
        )
    ]


def test_bisect_package_reports_a_missing_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bisection that wrote fewer than two chunks is an error, not a one-part model."""

    def _one_chunk(model: Any, output_dir: str, **kwargs: Any) -> None:
        (Path(output_dir) / "model_chunk1.mlpackage").mkdir(parents=True)

    monkeypatch.setattr(splitting.ct.models.utils, "bisect_model", _one_chunk)
    package = tmp_path / "model.mlpackage"
    package.mkdir()

    with pytest.raises(CompileError, match="model_chunk2"):
        splitting.bisect_package(package, tmp_path / "out")


def test_bisect_package_turns_no_bisect_point_into_a_compile_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """coremltools signals a model it cannot cut with a ValueError."""

    def _no_bisect_point(*args: Any, **kwargs: Any) -> None:
        raise ValueError("Not able to find the bisect point in the model.")

    monkeypatch.setattr(splitting.ct.models.utils, "bisect_model", _no_bisect_point)

    with pytest.raises(CompileError, match="could not be split") as caught:
        splitting.bisect_package(tmp_path / "model.mlpackage", tmp_path / "out")

    assert isinstance(caught.value.__cause__, ValueError)


def _fake_spec(inputs: list[str], outputs: list[str]) -> Any:
    """Build an object shaped like a model spec's ``description``."""
    return SimpleNamespace(
        description=SimpleNamespace(
            input=[SimpleNamespace(name=name) for name in inputs],
            output=[SimpleNamespace(name=name) for name in outputs],
        )
    )


def test_package_io_names_reads_the_declared_tensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inputs and outputs come back in declaration order."""
    monkeypatch.setattr(
        splitting.ct.utils,
        "load_spec",
        lambda path: _fake_spec(["hidden_cast_fp16", "attention_mask"], ["embedding"]),
    )

    names = splitting.package_io_names(tmp_path / "chunk.mlpackage")

    assert names == (("hidden_cast_fp16", "attention_mask"), ("embedding",))


@pytest.mark.parametrize(("inputs", "outputs"), [([], ["embedding"]), (["input_ids"], [])])
def test_package_io_names_refuses_a_package_without_inputs_or_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inputs: list[str], outputs: list[str]
) -> None:
    """A part that reads or writes nothing cannot be chained."""
    monkeypatch.setattr(splitting.ct.utils, "load_spec", lambda path: _fake_spec(inputs, outputs))

    with pytest.raises(CompileError, match="chunk.mlpackage"):
        splitting.package_io_names(tmp_path / "chunk.mlpackage")


def test_package_io_names_reports_an_unreadable_package(tmp_path: Path) -> None:
    """A package that is not there is reported by name."""
    with pytest.raises(CompileError, match="absent.mlpackage"):
        splitting.package_io_names(tmp_path / "absent.mlpackage")


# --- the real coremltools bisection ------------------------------------------------

# Geometry of the tiny model: four equally sized projections with a
# nonlinearity between them, so the graph has single-child cut points
# around the middle of its weights.
_HIDDEN = 64
_TOKENS = 4
_REAL_OUTPUT = "out"


class _FourLayerModel(torch.nn.Module):
    """Four projections; the mask is applied late, so both halves need it."""

    def __init__(self) -> None:
        """Create the projections from a fixed seed."""
        super().__init__()
        torch.manual_seed(0)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(_HIDDEN, _HIDDEN) for _ in range(4))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the projections, masking the hidden state before the last one."""
        hidden = x
        for index, layer in enumerate(self.layers):
            if index == len(self.layers) - 1:
                hidden = hidden * mask
            hidden = torch.relu(layer(hidden))
        return hidden


def _save_four_layer_package(directory: Path, target: str) -> Path:
    """Convert the tiny model as the pipeline would (fp16) and save the package."""
    example = (torch.zeros((1, _TOKENS, _HIDDEN)), torch.ones((1, _TOKENS, 1)))
    with torch.no_grad():
        traced = torch.jit.trace(_FourLayerModel().eval(), example)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x", shape=(1, _TOKENS, _HIDDEN), dtype=np.float32),
            ct.TensorType(name="mask", shape=(1, _TOKENS, 1), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name=_REAL_OUTPUT)],
        convert_to="mlprogram",
        compute_precision=conversion.PRECISIONS["fp16"],
        minimum_deployment_target=conversion.TARGETS[target],
    )
    path = directory / "model.mlpackage"
    mlmodel.save(str(path))
    return path


@pytest.mark.parametrize("target", sorted(conversion.TARGETS))
def test_the_real_bisection_yields_two_packages_that_chain(tmp_path: Path, target: str) -> None:
    """Two packages whose declared tensors form a manifest that validates."""
    package = _save_four_layer_package(tmp_path, target)
    model_inputs, model_outputs = splitting.package_io_names(package)

    first, second = splitting.bisect_package(package, tmp_path / "chunks")

    assert first.is_dir() and second.is_dir()
    first_inputs, first_outputs = splitting.package_io_names(first)
    second_inputs, second_outputs = splitting.package_io_names(second)
    assert model_inputs == ("x", "mask")
    assert model_outputs == (_REAL_OUTPUT,)
    assert first_inputs == model_inputs
    assert set(first_outputs) & set(second_inputs)
    assert set(second_inputs) <= set(model_inputs) | set(first_outputs)
    assert second_outputs == model_outputs
    manifest = segmented.SegmentsManifest(
        output_name=_REAL_OUTPUT,
        inputs=model_inputs,
        segments=(
            segmented.Segment("part01.mlmodelc", first_inputs, first_outputs),
            segmented.Segment("part02.mlmodelc", second_inputs, second_outputs),
        ),
    )
    segmented.SegmentsManifest.from_json(manifest.to_json(), source="bisected chunks")
