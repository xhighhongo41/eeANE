"""Tests for the start-up Neural Engine placement checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from eeane import ne_limits, placement_warnings
from eeane.placement_warnings import artifact_placement_reasons

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


def test_oversized_model_is_reported_only_where_the_limit_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ne_limits, "NE_MODEL_SIZE_LIMIT_BYTES", 10)
    artifact = _make_artifact(tmp_path, size=11)

    reasons = artifact_placement_reasons(artifact, mac_version="27.0.1")
    assert len(reasons) == 1
    assert "above the 1 GiB the Neural Engine accepts on macOS 27 and later" in reasons[0]
    assert reasons[0].startswith("it is ")
    assert artifact_placement_reasons(artifact, mac_version="26.6.2") == []


def test_model_exactly_at_the_limit_is_not_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ne_limits, "NE_MODEL_SIZE_LIMIT_BYTES", 10)
    artifact = _make_artifact(tmp_path, size=10)

    assert artifact_placement_reasons(artifact, mac_version="27.0.1") == []


def test_size_is_reported_in_gibibytes_with_two_decimals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ne_limits, "NE_MODEL_SIZE_LIMIT_BYTES", 10)
    monkeypatch.setattr(ne_limits, "compiled_model_size_bytes", lambda _path: int(1.11 * 2**30))
    artifact = _make_artifact(tmp_path)

    assert artifact_placement_reasons(artifact, mac_version="27.0")[0].startswith("it is 1.11 GiB,")


def test_both_reasons_are_returned_with_placement_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ne_limits, "NE_MODEL_SIZE_LIMIT_BYTES", 10)
    artifact = _make_artifact(tmp_path, size=11)
    _write_metadata(artifact, _measured(0.0))

    reasons = artifact_placement_reasons(artifact, mac_version="27.0")

    assert len(reasons) == 2
    assert reasons[0] == _PLACED_LOW.format(pct="0.0")
    assert reasons[1].startswith("it is ")


def test_missing_artifact_gives_no_reasons(tmp_path: Path) -> None:
    missing = tmp_path / "absent.mlmodelc"

    assert artifact_placement_reasons(missing, mac_version="27.0") == []
    assert artifact_placement_reasons(missing) == []


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
