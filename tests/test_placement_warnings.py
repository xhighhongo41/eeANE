"""Tests for the start-up Neural Engine placement checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from eeane import ne_limits, placement_warnings
from eeane.placement_warnings import artifact_placement_reasons
from eeane.segmented import SEGMENTS_FILENAME, Segment, SegmentsManifest, write_segments

_PLACED_LOW = "only {pct}% of its operations were placed on the Neural Engine when it was compiled"


def _make_artifact(directory: Path, name: str = "s128_b1_eager_macos13", size: int = 16) -> Path:
    """Create a fake ``.mlmodelc`` directory holding ``size`` bytes."""
    artifact = directory / f"{name}.mlmodelc"
    artifact.mkdir()
    (artifact / "weight.bin").write_bytes(b"\0" * size)
    return artifact


def _write_metadata(artifact: Path, selfcheck: object) -> None:
    """Write the metadata JSON next to ``artifact``."""
    artifact.with_suffix(".json").write_text(json.dumps({"selfcheck": selfcheck}))


def _measured(pct: object) -> dict[str, object]:
    return {"compute_plan": {"status": "measured", "ne_placement_pct": pct}}


@pytest.mark.parametrize("pct", [0.0, 89.9])
def test_low_recorded_placement_is_reported(tmp_path: Path, pct: float) -> None:
    artifact = _make_artifact(tmp_path)
    _write_metadata(artifact, _measured(pct))

    assert artifact_placement_reasons(artifact, mac_version="26.0") == [
        _PLACED_LOW.format(pct=f"{pct:.1f}")
    ]


@pytest.mark.parametrize("pct", [90.0, 99.3])
def test_sufficient_recorded_placement_is_not_reported(tmp_path: Path, pct: float) -> None:
    artifact = _make_artifact(tmp_path)
    _write_metadata(artifact, _measured(pct))

    assert artifact_placement_reasons(artifact, mac_version="26.0") == []


def test_missing_metadata_is_not_reported(tmp_path: Path) -> None:
    assert artifact_placement_reasons(_make_artifact(tmp_path), mac_version="26.0") == []


def test_broken_metadata_json_is_not_reported(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    artifact.with_suffix(".json").write_text("{not json")

    assert artifact_placement_reasons(artifact, mac_version="26.0") == []


@pytest.mark.parametrize(
    "selfcheck",
    [
        {"compute_plan": {"status": "unavailable", "reason": "x"}},
        {"compute_plan": {"status": "unavailable", "ne_placement_pct": 0.0}},
        {"compute_plan": {"status": "measured"}},
        {"compute_plan": _measured("12.0")["compute_plan"]},
        {"compute_plan": _measured(True)["compute_plan"]},
        {"compute_plan": "measured"},
        {},
        [],
        None,
    ],
)
def test_unusable_selfcheck_records_are_not_reported(tmp_path: Path, selfcheck: object) -> None:
    artifact = _make_artifact(tmp_path)
    _write_metadata(artifact, selfcheck)

    assert artifact_placement_reasons(artifact, mac_version="26.0") == []


def test_metadata_without_selfcheck_key_is_not_reported(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    artifact.with_suffix(".json").write_text(json.dumps({"other": 1}))

    assert artifact_placement_reasons(artifact, mac_version="26.0") == []


def test_non_object_metadata_is_not_reported(tmp_path: Path) -> None:
    artifact = _make_artifact(tmp_path)
    artifact.with_suffix(".json").write_text("[1, 2]")

    assert artifact_placement_reasons(artifact, mac_version="26.0") == []


_GIB = 1 << 30


def _fake_size(monkeypatch: pytest.MonkeyPatch, size: int) -> None:
    """Make every compiled model look ``size`` bytes large without writing them."""
    monkeypatch.setattr(ne_limits, "compiled_model_size_bytes", lambda _path: size)


def test_oversized_model_is_reported_with_the_limit_of_macos_27(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_size(monkeypatch, int(1.5 * _GIB))

    reasons = artifact_placement_reasons(_make_artifact(tmp_path), mac_version="27.0.1")

    assert reasons == [
        "it is 1.50 GiB, above the 1 GiB the Neural Engine accepts on macOS 27 and later"
    ]


def test_model_between_the_two_limits_is_reported_only_from_macos_27(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_size(monkeypatch, int(1.5 * _GIB))
    artifact = _make_artifact(tmp_path)

    assert len(artifact_placement_reasons(artifact, mac_version="27.0")) == 1
    assert artifact_placement_reasons(artifact, mac_version="26.6.2") == []


def test_model_above_the_legacy_limit_is_reported_up_to_macos_26(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_size(monkeypatch, int(2.5 * _GIB))

    reasons = artifact_placement_reasons(_make_artifact(tmp_path), mac_version="26.6.2")

    assert reasons == ["it is 2.50 GiB, above the 2 GiB the Neural Engine accepts up to macOS 26"]


def test_the_os_phrase_follows_the_first_limited_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ne_limits, "NE_SIZE_LIMIT_FIRST_MACOS_MAJOR", 30)
    _fake_size(monkeypatch, 3 * _GIB)
    artifact = _make_artifact(tmp_path)

    assert artifact_placement_reasons(artifact, mac_version="30.1")[0].endswith(
        "the Neural Engine accepts on macOS 30 and later"
    )
    assert artifact_placement_reasons(artifact, mac_version="29.9")[0].endswith(
        "the Neural Engine accepts up to macOS 29"
    )


@pytest.mark.parametrize("mac_version", ["", "unknown"])
def test_size_is_never_reported_when_the_os_version_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mac_version: str
) -> None:
    _fake_size(monkeypatch, 100 * _GIB)

    assert artifact_placement_reasons(_make_artifact(tmp_path), mac_version=mac_version) == []


@pytest.mark.parametrize(
    ("mac_version", "size"), [("27.0.1", _GIB), ("26.6.2", 2 * _GIB), ("27.0.1", 0)]
)
def test_model_up_to_the_limit_is_not_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mac_version: str, size: int
) -> None:
    _fake_size(monkeypatch, size)

    assert artifact_placement_reasons(_make_artifact(tmp_path), mac_version=mac_version) == []


def test_model_one_byte_over_the_limit_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_size(monkeypatch, _GIB + 1)

    reasons = artifact_placement_reasons(_make_artifact(tmp_path), mac_version="27.0.1")

    assert len(reasons) == 1
    assert reasons[0].startswith("it is 1.00 GiB,")


def test_size_is_reported_in_gibibytes_with_two_decimals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_size(monkeypatch, int(1.11 * _GIB))

    reasons = artifact_placement_reasons(_make_artifact(tmp_path), mac_version="27.0")

    assert reasons[0].startswith("it is 1.11 GiB,")


def test_unreadable_artifact_size_is_not_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail(_path: Path) -> int:
        raise OSError("unreadable")

    monkeypatch.setattr(ne_limits, "compiled_model_size_bytes", _fail)

    assert artifact_placement_reasons(_make_artifact(tmp_path), mac_version="27.0") == []


def test_both_reasons_are_returned_with_placement_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_size(monkeypatch, 2 * _GIB)
    artifact = _make_artifact(tmp_path)
    _write_metadata(artifact, _measured(0.0))

    reasons = artifact_placement_reasons(artifact, mac_version="27.0")

    assert len(reasons) == 2
    assert reasons[0] == _PLACED_LOW.format(pct="0.0")
    assert reasons[1].startswith("it is ")


def test_missing_artifact_gives_no_reasons(tmp_path: Path) -> None:
    missing = tmp_path / "absent.mlmodelc"

    assert artifact_placement_reasons(missing, mac_version="27.0") == []
    assert artifact_placement_reasons(missing) == []


# --- a model stored as parts ------------------------------------------------


def _part_name(index: int) -> str:
    """Directory name of the ``index``-th (0-based) part."""
    return f"part{index + 1:02d}.mlmodelc"


def _make_parts(
    directory: Path, name: str = "s128_b1_eager_macos13", sizes: tuple[int, ...] = (16, 16)
) -> Path:
    """Create a parts directory: chained parts holding ``sizes`` bytes each.

    Args:
        directory: Where the parts directory is created.
        name: Stem of the parts directory (``<name>.parts``).
        sizes: Bytes written into each part, one part per entry.

    Returns:
        The parts directory.
    """
    parts_dir = directory / f"{name}.parts"
    parts_dir.mkdir()
    segments = []
    for index, size in enumerate(sizes):
        part = parts_dir / _part_name(index)
        part.mkdir()
        (part / "weight.bin").write_bytes(b"\0" * size)
        source = "input_ids" if index == 0 else f"hidden{index - 1}"
        output = "embedding" if index == len(sizes) - 1 else f"hidden{index}"
        segments.append(Segment(mlmodelc=part.name, inputs=(source,), outputs=(output,)))
    write_segments(
        parts_dir,
        SegmentsManifest(output_name="embedding", inputs=("input_ids",), segments=tuple(segments)),
    )
    return parts_dir


def _fake_sizes(monkeypatch: pytest.MonkeyPatch, sizes: dict[str, int]) -> None:
    """Make each compiled directory look as large as ``sizes`` says, by name.

    A directory ``sizes`` does not name looks empty.
    """
    monkeypatch.setattr(
        ne_limits, "compiled_model_size_bytes", lambda path: sizes.get(Path(path).name, 0)
    )


def test_parts_that_all_fit_are_not_reported_however_large_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts_dir = _make_parts(tmp_path, sizes=(16, 16, 16))
    # The directory as a whole is far above the limit; each part is not.
    _fake_sizes(
        monkeypatch,
        {
            parts_dir.name: 3 * _GIB,
            _part_name(0): _GIB,
            _part_name(1): int(0.9 * _GIB),
            _part_name(2): int(0.9 * _GIB),
        },
    )

    assert artifact_placement_reasons(parts_dir, mac_version="27.0.1") == []


def test_parts_are_measured_on_disk_one_by_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A small limit, so real files can exceed it: 60 + 60 bytes is above
    # 100 in total, yet neither part is.
    monkeypatch.setattr(ne_limits, "ne_size_limit_bytes", lambda _mac_version=None: 100)

    fitting = _make_parts(tmp_path, name="fits", sizes=(60, 60))
    oversized = _make_parts(tmp_path, name="too_big", sizes=(60, 150))

    assert artifact_placement_reasons(fitting, mac_version="27.0") == []
    assert artifact_placement_reasons(oversized, mac_version="27.0") == [
        "1 of its 2 compiled parts are above the 100 bytes the Neural Engine accepts "
        "on macOS 27 and later (the largest is 0.00 GiB)"
    ]


def test_a_part_above_the_limit_is_reported_with_the_counts_and_the_largest_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts_dir = _make_parts(tmp_path, sizes=(16, 16, 16))
    _fake_sizes(
        monkeypatch,
        {_part_name(0): int(0.5 * _GIB), _part_name(1): int(1.2 * _GIB), _part_name(2): _GIB},
    )

    assert artifact_placement_reasons(parts_dir, mac_version="27.0.1") == [
        "1 of its 3 compiled parts are above the 1 GiB the Neural Engine accepts "
        "on macOS 27 and later (the largest is 1.20 GiB)"
    ]


def test_several_parts_above_the_limit_are_reported_as_one_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts_dir = _make_parts(tmp_path, sizes=(16, 16, 16))
    _fake_sizes(
        monkeypatch,
        {
            _part_name(0): int(1.1 * _GIB),
            _part_name(1): int(0.5 * _GIB),
            _part_name(2): int(1.75 * _GIB),
        },
    )

    assert artifact_placement_reasons(parts_dir, mac_version="27.0") == [
        "2 of its 3 compiled parts are above the 1 GiB the Neural Engine accepts "
        "on macOS 27 and later (the largest is 1.75 GiB)"
    ]


def test_parts_between_the_two_limits_are_reported_only_from_macos_27(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts_dir = _make_parts(tmp_path)
    _fake_sizes(monkeypatch, {_part_name(0): int(1.5 * _GIB), _part_name(1): int(1.5 * _GIB)})

    assert artifact_placement_reasons(parts_dir, mac_version="27.0") == [
        "2 of its 2 compiled parts are above the 1 GiB the Neural Engine accepts "
        "on macOS 27 and later (the largest is 1.50 GiB)"
    ]
    assert artifact_placement_reasons(parts_dir, mac_version="26.6.2") == []


def test_a_part_above_the_legacy_limit_is_reported_up_to_macos_26(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts_dir = _make_parts(tmp_path)
    _fake_sizes(monkeypatch, {_part_name(0): int(2.5 * _GIB), _part_name(1): _GIB})

    assert artifact_placement_reasons(parts_dir, mac_version="26.6.2") == [
        "1 of its 2 compiled parts are above the 2 GiB the Neural Engine accepts "
        "up to macOS 26 (the largest is 2.50 GiB)"
    ]


def test_parts_size_is_never_reported_when_the_os_version_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts_dir = _make_parts(tmp_path)
    _fake_sizes(monkeypatch, {_part_name(0): 100 * _GIB, _part_name(1): 100 * _GIB})

    assert artifact_placement_reasons(parts_dir, mac_version="") == []


def test_an_unreadable_part_counts_as_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts_dir = _make_parts(tmp_path)

    def _size(path: Path) -> int:
        if path.name == _part_name(0):
            raise OSError("unreadable")
        return 2 * _GIB

    monkeypatch.setattr(ne_limits, "compiled_model_size_bytes", _size)

    assert artifact_placement_reasons(parts_dir, mac_version="27.0") == [
        "1 of its 2 compiled parts are above the 1 GiB the Neural Engine accepts "
        "on macOS 27 and later (the largest is 2.00 GiB)"
    ]


@pytest.mark.parametrize(
    "manifest_text",
    [
        "{not json",
        "[]",
        json.dumps({"format_version": 99, "output_name": "x", "inputs": ["x"], "segments": []}),
        # Valid in shape, but lists a part that is not in the directory.
        json.dumps(
            {
                "format_version": 1,
                "output_name": "embedding",
                "inputs": ["input_ids"],
                "segments": [
                    {
                        "mlmodelc": "absent.mlmodelc",
                        "inputs": ["input_ids"],
                        "outputs": ["embedding"],
                    }
                ],
            }
        ),
    ],
)
def test_a_broken_manifest_gives_no_size_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_text: str
) -> None:
    """Loading the model reports a broken manifest; the start-up check stays quiet."""
    parts_dir = _make_parts(tmp_path)
    (parts_dir / SEGMENTS_FILENAME).write_text(manifest_text, encoding="utf-8")
    _fake_sizes(monkeypatch, {parts_dir.name: 5 * _GIB, _part_name(0): 5 * _GIB})

    assert artifact_placement_reasons(parts_dir, mac_version="27.0") == []


def test_a_broken_manifest_still_lets_the_recorded_placement_through(
    tmp_path: Path,
) -> None:
    parts_dir = _make_parts(tmp_path)
    (parts_dir / SEGMENTS_FILENAME).write_text("{not json", encoding="utf-8")
    _write_metadata(parts_dir, _measured(12.5))

    assert artifact_placement_reasons(parts_dir, mac_version="27.0") == [
        _PLACED_LOW.format(pct="12.5")
    ]


def test_the_placement_of_a_parts_directory_is_read_from_the_stems_metadata(
    tmp_path: Path,
) -> None:
    parts_dir = _make_parts(tmp_path, name="s512_b1_eager_macos13")
    (tmp_path / "s512_b1_eager_macos13.json").write_text(
        json.dumps({"selfcheck": _measured(42.0)}), encoding="utf-8"
    )

    assert artifact_placement_reasons(parts_dir, mac_version="27.0") == [
        _PLACED_LOW.format(pct="42.0")
    ]


def test_a_parts_directory_with_a_sufficient_placement_is_not_reported(
    tmp_path: Path,
) -> None:
    parts_dir = _make_parts(tmp_path)
    _write_metadata(parts_dir, _measured(97.0))

    assert artifact_placement_reasons(parts_dir, mac_version="27.0") == []


def test_both_reasons_are_returned_for_parts_with_placement_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts_dir = _make_parts(tmp_path)
    _write_metadata(parts_dir, _measured(0.0))
    _fake_sizes(monkeypatch, {_part_name(1): 2 * _GIB})

    reasons = artifact_placement_reasons(parts_dir, mac_version="27.0")

    assert reasons == [
        _PLACED_LOW.format(pct="0.0"),
        "1 of its 2 compiled parts are above the 1 GiB the Neural Engine accepts "
        "on macOS 27 and later (the largest is 2.00 GiB)",
    ]


def test_a_directory_named_like_parts_but_without_a_manifest_is_judged_as_one_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a manifest makes a parts directory; anything else is one compiled model."""
    directory = tmp_path / "s128.parts"
    directory.mkdir()
    _fake_sizes(monkeypatch, {directory.name: int(1.5 * _GIB)})

    assert artifact_placement_reasons(directory, mac_version="27.0") == [
        "it is 1.50 GiB, above the 1 GiB the Neural Engine accepts on macOS 27 and later"
    ]


def test_module_does_not_import_the_conversion_stack() -> None:
    script = (
        "import eeane.placement_warnings\n"
        "import sys\n"
        "for name in ('torch', 'transformers', 'coremltools', 'eeane.compiler'):\n"
        "    assert name not in sys.modules, name\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr


def test_public_api_is_exported() -> None:
    assert placement_warnings.artifact_placement_reasons is artifact_placement_reasons
