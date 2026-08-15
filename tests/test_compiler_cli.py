"""Tests for `eeane compile` (v0.6 T2, see 開発資料/v0.6実装計画.md §4.1, §4.7, §4.9)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from eeane import cli, compiler

# --- --buckets / argument parsing -------------------------------------------


def test_build_parser_compile_defaults() -> None:
    """With no options, compile's optional fields must all be at their documented defaults."""
    parser = cli.build_parser()

    args = parser.parse_args(["compile", "some/path"])

    assert args.source == "some/path"
    assert args.kind == "auto"
    assert args.buckets is None
    assert args.out_dir is None
    assert args.emit_config is None
    assert args.force is False
    assert args.batch == 1
    assert args.precision == "fp16"
    assert args.target == "macos13"
    assert args.attn == "eager"
    assert args.keep_mlpackage is False
    assert args.skip_selfcheck is False


def test_build_parser_compile_parses_buckets_option() -> None:
    """--buckets 128,512 must parse to [128, 512], in order."""
    parser = cli.build_parser()

    args = parser.parse_args(["compile", "some/path", "--buckets", "128,512"])

    assert args.buckets == [128, 512]


@pytest.mark.parametrize("bad_value", ["abc", "0", "-1", "", "128,", "128,,512"])
def test_build_parser_compile_rejects_invalid_buckets(bad_value: str) -> None:
    """Non-integer, zero, negative, or empty --buckets elements must exit 2."""
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["compile", "some/path", "--buckets", bad_value])

    assert excinfo.value.code == 2


def test_build_parser_compile_rejects_invalid_kind() -> None:
    """An unknown --kind choice must exit 2."""
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["compile", "some/path", "--kind", "invalid"])

    assert excinfo.value.code == 2


def test_build_parser_compile_parses_all_other_options(tmp_path: Path) -> None:
    """Every remaining compile option must parse to the given value with the right type."""
    parser = cli.build_parser()
    out_dir = tmp_path / "cache"
    emit_config = tmp_path / "snippet.toml"

    args = parser.parse_args(
        [
            "compile",
            "cl-nagoya/ruri-v3-310m",
            "--kind",
            "embedding",
            "--out-dir",
            str(out_dir),
            "--emit-config",
            str(emit_config),
            "--force",
            "--batch",
            "4",
            "--precision",
            "fp32",
            "--target",
            "macos15",
            "--attn",
            "sdpa",
            "--keep-mlpackage",
            "--skip-selfcheck",
        ]
    )

    assert args.source == "cl-nagoya/ruri-v3-310m"
    assert args.kind == "embedding"
    assert args.out_dir == out_dir
    assert args.emit_config == emit_config
    assert args.force is True
    assert args.batch == 4
    assert isinstance(args.batch, int)
    assert args.precision == "fp32"
    assert args.target == "macos15"
    assert args.attn == "sdpa"
    assert args.keep_mlpackage is True
    assert args.skip_selfcheck is True


# --- dependency guard --------------------------------------------------------


def test_require_compile_dependencies_raises_with_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must name every missing package and how to install them."""
    monkeypatch.setattr(compiler.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(compiler.MissingCompileDependencyError) as excinfo:
        compiler.require_compile_dependencies()

    message = str(excinfo.value)
    assert "torch" in message
    assert "transformers" in message
    assert "uv sync --extra compile" in message


def test_require_compile_dependencies_passes_when_available() -> None:
    """When torch/transformers are actually importable, the guard must be a no-op."""
    compiler.require_compile_dependencies()  # must not raise


def test_compile_cli_missing_dependency_exits_1_with_install_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """With torch/transformers unavailable, `eeane compile` must exit 1 with an install hint."""
    monkeypatch.setattr(compiler.importlib.util, "find_spec", lambda name: None)

    exit_code = cli.main(["compile", "some/path"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "torch" in captured.err
    assert "transformers" in captured.err
    assert "uv sync --extra compile" in captured.err


# --- stub behaviour -----------------------------------------------------


def test_compile_cli_stub_exits_2_with_not_implemented_message(
    capsys: pytest.CaptureFixture,
) -> None:
    """With [compile] dependencies installed, `eeane compile` must exit 2 (unimplemented stub)."""
    exit_code = cli.main(["compile", "some/path"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "not implemented yet" in captured.err


# --- runtime/torch import isolation ------------------------------------

# Modules that must stay importable with the runtime dependency set only
# (no [compile] extra): everything `eeane serve` / `eeane check-config`
# loads. eeane.compiler is deliberately absent -- it is only imported
# from inside eeane.cli._cmd_compile.
_RUNTIME_MODULE_FILES = (
    "__init__.py",
    "__main__.py",
    "cli.py",
    "config.py",
    "engine.py",
    "runtime.py",
    "schemas.py",
    "server.py",
)

# Top-level packages that only the [compile] extra provides.
_COMPILE_ONLY_PACKAGES = ("torch", "transformers")


def _is_compile_only(module_name: str) -> bool:
    """Tell whether an imported module name comes from the [compile] extra.

    Args:
        module_name: Dotted module name as written in the import
            statement.

    Returns:
        ``True`` for a [compile]-only package or any of its submodules.
    """
    return any(
        module_name == package or module_name.startswith(f"{package}.")
        for package in _COMPILE_ONLY_PACKAGES
    )


def _compile_only_imports(source: str, filename: str) -> list[str]:
    """Find [compile]-only imports anywhere in a module's source.

    Uses ``ast.walk``, so imports nested inside functions/methods (the
    usual way an unwanted dependency sneaks back in) are reported too.
    Relative imports (``level > 0``) are skipped: they can only refer to
    ``eeane`` itself.

    Args:
        source: Python source code.
        filename: Name used in the returned messages.

    Returns:
        One ``"<filename>:<line>: <module>"`` string per offending
        import.
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
            f"{filename}:{node.lineno}: {name}" for name in names if _is_compile_only(name)
        ]
    return offenders


def test_runtime_modules_do_not_import_torch_or_transformers() -> None:
    """No runtime-path module may import torch/transformers, at module or function level.

    Checked with ``ast`` rather than by inspecting ``sys.modules`` after
    an import: ``coremltools`` (a genuine runtime dependency) try-imports
    torch and transformers itself, in ``coremltools/_deps/__init__.py``,
    to probe its optional converter frontends. A ``sys.modules`` check
    therefore can never pass, no matter what eeANE does, while what R5
    actually requires is that *eeANE's own* runtime code never reaches
    for the [compile] extra (開発資料/v0.6実装計画.md §4.7 担保テスト).
    """
    package_dir = Path(cli.__file__).resolve().parent

    offenders: list[str] = []
    for filename in _RUNTIME_MODULE_FILES:
        path = package_dir / filename
        assert path.is_file(), f"runtime module {path} is missing (renamed?)"
        offenders += _compile_only_imports(path.read_text(encoding="utf-8"), filename)

    assert offenders == []


def test_compile_only_import_detector_flags_module_and_function_level_imports() -> None:
    """The detector itself must catch both import forms, including inside a function."""
    source = (
        "import numpy\n"
        "from transformers import AutoTokenizer\n"
        "from . import runtime\n"
        "def load():\n"
        "    import torch.nn\n"
        "    return torch.nn, AutoTokenizer, numpy, runtime\n"
    )

    offenders = _compile_only_imports(source, "fake.py")

    assert offenders == ["fake.py:2: transformers", "fake.py:5: torch.nn"]
