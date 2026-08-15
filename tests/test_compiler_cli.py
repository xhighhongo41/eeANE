"""Tests for `eeane compile` (v0.6 T2, see 開発資料/v0.6実装計画.md §4.1, §4.7, §4.9)."""

from __future__ import annotations

import subprocess
import sys
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


@pytest.mark.xfail(
    reason=(
        "Not yet satisfiable within T2's scope: eeane/runtime.py:15 and "
        "eeane/engine.py:23 still do `from transformers import ...` "
        "(pre-existing v0.5 code; T2 must not change runtime/engine "
        "behaviour per its brief), and plain `import transformers` alone "
        "already imports torch as a side effect (verified: `python -c "
        "'import transformers; import sys; print(\"torch\" in sys.modules)'` "
        "-> True). Separately, `import coremltools` (a genuine runtime "
        "dependency) *also* try-imports torch and transformers itself, at "
        "coremltools/_deps/__init__.py -- to probe optional converter "
        "frontends -- independent of anything eeANE does. So this "
        "assertion cannot pass until T6 removes the direct transformers "
        "imports from runtime.py/engine.py (開発資料/v0.6実装計画.md §4.6, "
        "§5 T6), and even then coremltools's own probing means a strict "
        "'torch not in sys.modules' check is not a meaningful CI-safe "
        "signal for R5; see this task's final report for the recommended "
        "follow-up (an AST-based direct-import check, plus T7's real "
        "clean-venv `eeane serve` smoke test)."
    ),
    strict=True,
)
def test_runtime_modules_do_not_import_torch() -> None:
    """Importing every runtime-path module must never pull torch into sys.modules.

    Run in a subprocess: within the pytest process itself other tests
    (or ``eeane.compiler``'s own dependencies) may already have imported
    torch, which would make an in-process ``sys.modules`` check
    meaningless (開発資料/v0.6実装計画.md §4.7 担保テスト).
    """
    script = (
        "import eeane.config, eeane.schemas, eeane.runtime, eeane.engine, "
        "eeane.server, eeane.cli\n"
        "import sys\n"
        "assert 'torch' not in sys.modules, sorted(sys.modules)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
