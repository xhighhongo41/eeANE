"""Tests for eeane.cache: cache root resolution and model_info.json reading."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import eeane.cache as cache
from eeane.cache import (
    CACHE_SUBDIR,
    MODEL_INFO_FILENAME,
    CacheError,
    load_model_info,
    model_cache_dir,
    model_info_path,
    resolve_cache_root,
)

# --- resolve_cache_root ---------------------------------------------------


def test_override_wins_over_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit cache root must be used verbatim, ignoring XDG_CACHE_HOME."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert resolve_cache_root(tmp_path / "explicit") == tmp_path / "explicit"


def test_override_expands_a_leading_tilde(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A '~/...' override must be expanded to the user's home directory."""
    monkeypatch.setenv("HOME", str(tmp_path))

    assert resolve_cache_root(Path("~/somewhere")) == tmp_path / "somewhere"


def test_absolute_xdg_cache_home_is_used(tmp_path: Path) -> None:
    """With an absolute XDG_CACHE_HOME, the root is <XDG_CACHE_HOME>/eeane."""
    root = resolve_cache_root(env={"XDG_CACHE_HOME": str(tmp_path / "xdg")})

    assert root == tmp_path / "xdg" / "eeane"


def test_relative_xdg_cache_home_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative XDG_CACHE_HOME is invalid per the XDG spec and must fall back to ~/.cache."""
    monkeypatch.setenv("HOME", str(tmp_path))

    root = resolve_cache_root(env={"XDG_CACHE_HOME": "relative/cache"})

    assert root == tmp_path / ".cache" / "eeane"


def test_unset_xdg_cache_home_falls_back_to_home_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without XDG_CACHE_HOME, the root is ~/.cache/eeane."""
    monkeypatch.setenv("HOME", str(tmp_path))

    assert resolve_cache_root(env={}) == tmp_path / ".cache" / "eeane"


def test_blank_xdg_cache_home_falls_back_to_home_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A whitespace-only XDG_CACHE_HOME must be treated as unset, not as './eeane'."""
    monkeypatch.setenv("HOME", str(tmp_path))

    assert resolve_cache_root(env={"XDG_CACHE_HOME": "   "}) == tmp_path / ".cache" / "eeane"


def test_environment_is_read_from_os_environ_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an explicit env mapping, the process environment must be consulted."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert resolve_cache_root() == tmp_path / "xdg" / "eeane"


# --- model_cache_dir ------------------------------------------------------


def test_hub_id_is_normalised_to_a_single_directory_component(tmp_path: Path) -> None:
    """'org/name' must map to '<root>/compiled/org--name', not a nested directory."""
    assert model_cache_dir(tmp_path, "org/name") == tmp_path / CACHE_SUBDIR / "org--name"


def test_plain_model_name_is_used_as_is(tmp_path: Path) -> None:
    """A slash-free id must be used as the directory name unchanged."""
    assert model_cache_dir(tmp_path, "local-model") == tmp_path / CACHE_SUBDIR / "local-model"


def test_nested_hub_id_normalises_every_separator(tmp_path: Path) -> None:
    """Every '/' must be replaced, so the result stays one path component."""
    resolved = model_cache_dir(tmp_path, "org/team/name")

    assert resolved.parent == tmp_path / CACHE_SUBDIR
    assert resolved.name == "org--team--name"


@pytest.mark.parametrize("model_id", ["", "   ", ".", ".."])
def test_ids_that_do_not_name_a_directory_are_rejected(tmp_path: Path, model_id: str) -> None:
    """Empty and dot ids must raise instead of silently escaping the cache root."""
    with pytest.raises(CacheError, match="cache directory"):
        model_cache_dir(tmp_path, model_id)


def test_surrounding_whitespace_is_trimmed(tmp_path: Path) -> None:
    """A padded id must resolve to the same directory as the trimmed one."""
    assert model_cache_dir(tmp_path, "  org/name  ") == model_cache_dir(tmp_path, "org/name")


# --- load_model_info ------------------------------------------------------


def _write_model_info(model_dir: Path, payload: object) -> Path:
    """Write ``payload`` as the model's record and return its path."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / MODEL_INFO_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_model_info_path_points_into_the_model_directory(tmp_path: Path) -> None:
    """model_info_path must name the record inside the model's own directory."""
    assert model_info_path(tmp_path) == tmp_path / MODEL_INFO_FILENAME


def test_valid_record_is_returned_verbatim(tmp_path: Path) -> None:
    """A well-formed record must be returned as a plain dict, unmodified."""
    payload = {"format_version": 1, "id": "org/name", "buckets": [128, 512]}
    _write_model_info(tmp_path / "model", payload)

    assert load_model_info(tmp_path / "model") == payload


def test_missing_record_raises_cache_error_naming_the_path(tmp_path: Path) -> None:
    """A model directory without a record must raise, naming the file it looked for."""
    with pytest.raises(CacheError, match=MODEL_INFO_FILENAME):
        load_model_info(tmp_path / "absent")


def test_malformed_json_raises_cache_error(tmp_path: Path) -> None:
    """A truncated/invalid JSON record must raise rather than return a partial dict."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / MODEL_INFO_FILENAME).write_text('{"format_version": 1,', encoding="utf-8")

    with pytest.raises(CacheError, match="JSON"):
        load_model_info(model_dir)


def test_non_object_json_raises_cache_error(tmp_path: Path) -> None:
    """A record holding a JSON array (not an object) must be rejected."""
    _write_model_info(tmp_path / "model", [1, 2, 3])

    with pytest.raises(CacheError, match="object"):
        load_model_info(tmp_path / "model")


def test_unreadable_record_raises_cache_error(tmp_path: Path) -> None:
    """A record that cannot be read as a file (here: a directory) must be reported."""
    model_dir = tmp_path / "model"
    (model_dir / MODEL_INFO_FILENAME).mkdir(parents=True)

    with pytest.raises(CacheError, match=MODEL_INFO_FILENAME):
        load_model_info(model_dir)


def test_non_utf8_record_raises_cache_error(tmp_path: Path) -> None:
    """A record written in a non-UTF-8 encoding must be reported, not crash the loader."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / MODEL_INFO_FILENAME).write_bytes(b'{"id": "\xff\xfe"}')

    with pytest.raises(CacheError, match=MODEL_INFO_FILENAME):
        load_model_info(model_dir)


# --- dependency boundary --------------------------------------------------

# Packages that only the [compile] extra installs, plus the compiler
# package itself: the serving process is installed without them, so the
# cache reader and the config loader must not reach for either.
_FORBIDDEN_IMPORTS = ("torch", "transformers", "eeane.compiler")


def _forbidden_imports(source: str, filename: str) -> list[str]:
    """List [compile]-only imports found anywhere in a module's source.

    ``ast.walk`` is used rather than an import-time check so that imports
    hidden inside functions are caught too.

    Args:
        source: Python source code.
        filename: Name used in the returned messages.

    Returns:
        One ``"<filename>:<line>: <module>"`` string per offending import.
    """
    offenders: list[str] = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module] if node.level == 0 and node.module else []
        else:
            continue
        offenders += [
            f"{filename}:{node.lineno}: {name}"
            for name in names
            if any(
                name == package or name.startswith(f"{package}.") for package in _FORBIDDEN_IMPORTS
            )
        ]
    return offenders


def test_cache_and_config_modules_stay_free_of_compile_only_imports() -> None:
    """eeane.cache/eeane.config must run in an install without the [compile] extra."""
    package_dir = Path(cache.__file__).resolve().parent

    offenders: list[str] = []
    for filename in ("cache.py", "config.py"):
        path = package_dir / filename
        assert path.is_file(), f"module {path} is missing (renamed?)"
        offenders += _forbidden_imports(path.read_text(encoding="utf-8"), filename)

    assert offenders == []


def test_forbidden_import_detector_flags_module_and_function_level_imports() -> None:
    """The detector itself must catch both import forms, including inside a function."""
    source = (
        "import numpy\n"
        "from eeane.compiler import pipeline\n"
        "def load():\n"
        "    import torch.nn\n"
        "    return torch.nn, pipeline, numpy\n"
    )

    offenders = _forbidden_imports(source, "fake.py")

    assert offenders == ["fake.py:2: eeane.compiler", "fake.py:4: torch.nn"]
