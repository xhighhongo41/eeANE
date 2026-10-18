"""Tests for the Neural Engine placement limits shared module."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from eeane import ne_limits


def test_size_limit_constants() -> None:
    assert ne_limits.NE_MODEL_SIZE_LIMIT_BYTES_MACOS27 == 1_073_741_824
    assert ne_limits.NE_MODEL_SIZE_LIMIT_BYTES_LEGACY == 2_147_483_648
    assert ne_limits.NE_SIZE_LIMIT_FIRST_MACOS_MAJOR == 27
    assert ne_limits.SIZE_LIMIT_DISABLED == "none"


@pytest.mark.parametrize(
    ("version", "expected"),
    [("27.0.1", 27), ("26.6.2", 26), ("13", 13), ("", None), ("unknown", None), (" 28.1 ", 28)],
)
def test_macos_major_version_parses_the_leading_number(version: str, expected: int | None) -> None:
    assert ne_limits.macos_major_version(version) == expected


# --- parse_size_limit ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1073741824", 1 << 30),
        ("1", 1),
        ("1GiB", 1 << 30),
        ("2 GiB", 2 << 30),
        ("800MiB", 800 << 20),
        ("800 MiB", 800 << 20),
        ("64KiB", 64 << 10),
        ("1.5GiB", 3 << 29),
        ("0.5 GiB", 1 << 29),
        ("1.5KiB", 1536),
        # Truncated toward zero, not rounded.
        ("1.9999KiB", 2047),
        ("  1gib  ", 1 << 30),
        ("1GIB", 1 << 30),
        ("800mib", 800 << 20),
        ("2\tGiB", 2 << 30),
    ],
)
def test_parse_size_limit_accepts_bytes_and_binary_units(text: str, expected: int) -> None:
    assert ne_limits.parse_size_limit(text) == expected


@pytest.mark.parametrize("text", ["none", "None", "NONE", "  none  "])
def test_parse_size_limit_none_disables_splitting(text: str) -> None:
    assert ne_limits.parse_size_limit(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "0",
        "0GiB",
        "0.0 MiB",
        # Truncates to zero bytes.
        "0.0001KiB",
        "-1",
        "-1GiB",
        "+1GiB",
        "1GB",
        "1G",
        "1gb",
        "500MB",
        "1MiBytes",
        "1.5",
        "1.0",
        "1.GiB",
        ".5GiB",
        "GiB",
        "1 2 GiB",
        "1e3",
        "1,5GiB",
        "abc",
        "nonee",
        "no ne",
        "1GiB2",
        "٣GiB",
    ],
)
def test_parse_size_limit_rejects_anything_else(text: str) -> None:
    with pytest.raises(ValueError, match="invalid Neural Engine size limit"):
        ne_limits.parse_size_limit(text)


def test_parse_size_limit_error_names_the_input_and_the_accepted_forms() -> None:
    with pytest.raises(ValueError) as excinfo:
        ne_limits.parse_size_limit("1GB")

    assert str(excinfo.value) == (
        "invalid Neural Engine size limit '1GB': give a size such as 1GiB, 800MiB or "
        "1073741824 (bytes), or 'none'"
    )


# --- ne_size_limit_bytes -------------------------------------------------------


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("27.0", 1 << 30),
        ("27.0.1", 1 << 30),
        ("28.3", 1 << 30),
        ("26.6.2", 2 << 30),
        ("13.0", 2 << 30),
    ],
)
def test_size_limit_follows_the_macos_version(version: str, expected: int) -> None:
    assert ne_limits.ne_size_limit_bytes(version) == expected


@pytest.mark.parametrize("version", ["", "unknown"])
def test_size_limit_is_absent_when_the_version_is_unknown(version: str) -> None:
    assert ne_limits.ne_size_limit_bytes(version) is None


def test_size_limit_reads_the_running_system_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ne_limits.platform, "mac_ver", lambda: ("27.1", ("", "", ""), "arm64"))
    assert ne_limits.ne_size_limit_bytes() == 1 << 30
    monkeypatch.setattr(ne_limits.platform, "mac_ver", lambda: ("26.4", ("", "", ""), "arm64"))
    assert ne_limits.ne_size_limit_bytes() == 2 << 30
    monkeypatch.setattr(ne_limits.platform, "mac_ver", lambda: ("", ("", "", ""), ""))
    assert ne_limits.ne_size_limit_bytes() is None


def test_an_override_replaces_the_limit_of_the_os() -> None:
    assert ne_limits.ne_size_limit_bytes("27.0", "2GiB") == 2 << 30
    assert ne_limits.ne_size_limit_bytes("26.0", "800MiB") == 800 << 20
    assert ne_limits.ne_size_limit_bytes("", "1073741824") == 1 << 30


def test_an_override_of_none_disables_the_limit() -> None:
    assert ne_limits.ne_size_limit_bytes("27.0", "none") is None


def test_an_invalid_override_is_refused() -> None:
    with pytest.raises(ValueError, match="invalid Neural Engine size limit"):
        ne_limits.ne_size_limit_bytes("27.0", "1GB")


def test_an_override_does_not_read_the_running_system(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> tuple[str, tuple[str, str, str], str]:
        raise AssertionError("the OS version must not be read")

    monkeypatch.setattr(ne_limits.platform, "mac_ver", _boom)

    assert ne_limits.ne_size_limit_bytes(override="1GiB") == 1 << 30


# --- describe_size_limit -------------------------------------------------------


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        (1 << 30, "1 GiB"),
        (2 << 30, "2 GiB"),
        (800 << 20, "800 MiB"),
        (1536 << 20, "1536 MiB"),
        ((1 << 30) + 1, f"{(1 << 30) + 1} bytes"),
        (10, "10 bytes"),
    ],
)
def test_describe_size_limit(limit: int, expected: str) -> None:
    assert ne_limits.describe_size_limit(limit) == expected


# --- compiled_model_size_bytes -------------------------------------------------


def test_compiled_model_size_sums_nested_files(tmp_path: Path) -> None:
    model = tmp_path / "m.mlmodelc"
    (model / "weights").mkdir(parents=True)
    (model / "weights" / "weight.bin").write_bytes(b"x" * 1000)
    (model / "model.mil").write_bytes(b"y" * 24)
    assert ne_limits.compiled_model_size_bytes(model) == 1024


def test_compiled_model_size_of_a_missing_directory_is_zero(tmp_path: Path) -> None:
    assert ne_limits.compiled_model_size_bytes(tmp_path / "absent.mlmodelc") == 0


def test_module_imports_nothing_beyond_the_standard_library() -> None:
    script = (
        "import eeane.ne_limits\n"
        "import sys\n"
        "for name in ('numpy', 'torch', 'transformers', 'coremltools', 'eeane.compiler'):\n"
        "    assert name not in sys.modules, name\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr
