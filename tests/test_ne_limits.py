"""Tests for the Neural Engine placement limits shared module."""

from __future__ import annotations

from pathlib import Path

import pytest

from eeane import ne_limits


def test_size_limit_is_one_gibibyte() -> None:
    assert ne_limits.NE_MODEL_SIZE_LIMIT_BYTES == 1_073_741_824


@pytest.mark.parametrize(
    ("version", "expected"),
    [("27.0.1", 27), ("26.6.2", 26), ("13", 13), ("", None), ("unknown", None), (" 28.1 ", 28)],
)
def test_macos_major_version_parses_the_leading_number(version: str, expected: int | None) -> None:
    assert ne_limits.macos_major_version(version) == expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [("27.0", True), ("27.0.1", True), ("28.3", True), ("26.6.2", False), ("13.0", False)],
)
def test_size_limit_applies_from_macos_27(version: str, expected: bool) -> None:
    assert ne_limits.ne_size_limit_applies(version) is expected


def test_size_limit_does_not_apply_when_the_version_is_unknown() -> None:
    assert ne_limits.ne_size_limit_applies("") is False


def test_size_limit_reads_the_running_system_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ne_limits.platform, "mac_ver", lambda: ("27.1", ("", "", ""), "arm64"))
    assert ne_limits.ne_size_limit_applies() is True
    monkeypatch.setattr(ne_limits.platform, "mac_ver", lambda: ("26.4", ("", "", ""), "arm64"))
    assert ne_limits.ne_size_limit_applies() is False


def test_compiled_model_size_sums_nested_files(tmp_path: Path) -> None:
    model = tmp_path / "m.mlmodelc"
    (model / "weights").mkdir(parents=True)
    (model / "weights" / "weight.bin").write_bytes(b"x" * 1000)
    (model / "model.mil").write_bytes(b"y" * 24)
    assert ne_limits.compiled_model_size_bytes(model) == 1024


def test_compiled_model_size_of_a_missing_directory_is_zero(tmp_path: Path) -> None:
    assert ne_limits.compiled_model_size_bytes(tmp_path / "absent.mlmodelc") == 0
