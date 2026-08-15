"""Tests for eeane.cli (v0.5 T3, see 開発資料/v0.5実装計画.md §4.3-§4.4)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from eeane import cli

# --- fixtures / helpers --------------------------------------------------

_KEYLESS_TOML = """
[[models]]
id = "emb-only"
kind = "embedding"
tokenizer = "models/emb-only/tokenizer.json"

[models.artifacts]
256 = "compiled/emb-only/s256.mlmodelc"
"""

_KEYED_TOML = """
[server]
api_key = "top-secret-key"

[[models]]
id = "emb-only"
kind = "embedding"
tokenizer = "models/emb-only/tokenizer.json"

[models.artifacts]
256 = "compiled/emb-only/s256.mlmodelc"
"""


_DETAILED_TOML = """
[server]
cache_root = "cache"

[[models]]
id = "emb-only"
kind = "embedding"
tokenizer = "models/emb-only/tokenizer.json"
embedding_dim = 768
excluded_buckets = [1024]

[models.artifacts]
256 = "compiled/emb-only/s256.mlmodelc"
"""


def _write(path: Path, content: str) -> Path:
    """Write ``content`` to ``path`` and return ``path`` for chaining."""
    path.write_text(content, encoding="utf-8")
    return path


def _isolate_config_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the config search order (cwd, home) at empty tmp_path dirs.

    Ensures tests don't pick up an unrelated ``./eeane.toml`` or
    ``~/.config/eeane/eeane.toml`` from the machine running the tests.
    """
    monkeypatch.chdir(tmp_path)
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.delenv("EEANE_API_KEY", raising=False)


class _RecordingUvicornRun:
    """Stub replacing ``uvicorn.run``: records calls instead of serving."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        self.calls: list[dict] = []

    def __call__(self, app: object, *, host: str, port: int, **kwargs: object) -> None:
        """Record the call instead of starting a real server."""
        self.calls.append({"app": app, "host": host, "port": port})


@pytest.fixture
def stub_uvicorn_run(monkeypatch: pytest.MonkeyPatch) -> _RecordingUvicornRun:
    """Replace ``eeane.cli.uvicorn.run`` with a call recorder for the test."""
    recorder = _RecordingUvicornRun()
    monkeypatch.setattr(cli.uvicorn, "run", recorder)
    return recorder


# --- build_parser ----------------------------------------------------------


def test_build_parser_serve_defaults_are_none() -> None:
    """With no options, serve's override fields must all be None (not provided)."""
    parser = cli.build_parser()

    args = parser.parse_args(["serve"])

    assert args.config is None
    assert args.host is None
    assert args.port is None
    assert args.log_level is None


def test_build_parser_serve_parses_all_options(tmp_path: Path) -> None:
    """serve must parse --config/--host/--port/--log-level, with port as int."""
    parser = cli.build_parser()
    config_path = tmp_path / "eeane.toml"

    args = parser.parse_args(
        [
            "serve",
            "--config",
            str(config_path),
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--log-level",
            "debug",
        ]
    )

    assert args.config == config_path
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert isinstance(args.port, int)
    assert args.log_level == "debug"


def test_build_parser_serve_rejects_invalid_log_level() -> None:
    """An unknown --log-level choice must make parse_args exit with code 2."""
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["serve", "--log-level", "verbose"])

    assert excinfo.value.code == 2


def test_build_parser_check_config_parses_config_option(tmp_path: Path) -> None:
    """check-config must parse --config."""
    parser = cli.build_parser()
    config_path = tmp_path / "eeane.toml"

    args = parser.parse_args(["check-config", "--config", str(config_path)])

    assert args.config == config_path


# --- main(): no / unknown subcommand ----------------------------------------


def test_main_with_no_arguments_returns_usage_exit_code(capsys: pytest.CaptureFixture) -> None:
    """No subcommand must return exit code 2 and print usage to stderr."""
    exit_code = cli.main([])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "usage" in captured.err.lower()


def test_main_with_unknown_subcommand_raises_system_exit() -> None:
    """An unknown subcommand must raise SystemExit(2) (argparse's own error handling)."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["unknown"])

    assert excinfo.value.code == 2


# --- serve -------------------------------------------------------------


def test_serve_uses_builtin_defaults_when_no_config_or_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_uvicorn_run: _RecordingUvicornRun,
) -> None:
    """With no config file anywhere and no overrides, the built-in host/port are used."""
    _isolate_config_search(tmp_path, monkeypatch)

    exit_code = cli.main(["serve"])

    assert exit_code == 0
    assert len(stub_uvicorn_run.calls) == 1
    call = stub_uvicorn_run.calls[0]
    assert call["host"] == "127.0.0.1"
    assert call["port"] == 7997


def test_serve_host_and_port_overrides_take_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_uvicorn_run: _RecordingUvicornRun,
) -> None:
    """--host/--port must override the built-in defaults."""
    _isolate_config_search(tmp_path, monkeypatch)

    exit_code = cli.main(["serve", "--host", "0.0.0.0", "--port", "9000"])

    assert exit_code == 0
    call = stub_uvicorn_run.calls[0]
    assert call["host"] == "0.0.0.0"
    assert call["port"] == 9000


def test_serve_with_missing_config_file_exits_1_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    stub_uvicorn_run: _RecordingUvicornRun,
) -> None:
    """A --config path that does not exist must exit 1 with a clean stderr message."""
    missing = tmp_path / "does-not-exist.toml"

    exit_code = cli.main(["serve", "--config", str(missing)])

    assert exit_code == 1
    assert len(stub_uvicorn_run.calls) == 0
    captured = capsys.readouterr()
    assert "does-not-exist" in captured.err
    assert "Traceback" not in captured.err


def test_serve_log_level_override_is_passed_to_basic_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_uvicorn_run: _RecordingUvicornRun,
) -> None:
    """--log-level must control the level logging.basicConfig is initialized with."""
    _isolate_config_search(tmp_path, monkeypatch)
    basic_config_calls: list[dict] = []
    monkeypatch.setattr(
        cli.logging, "basicConfig", lambda **kwargs: basic_config_calls.append(kwargs)
    )

    exit_code = cli.main(["serve", "--log-level", "warning"])

    assert exit_code == 0
    assert len(basic_config_calls) == 1
    assert basic_config_calls[0]["level"] == logging.WARNING
    assert "%(asctime)s" in basic_config_calls[0]["format"]


@pytest.mark.parametrize("command", ["serve", "check-config"])
def test_group_other_readable_key_file_warns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    stub_uvicorn_run: _RecordingUvicornRun,
    command: str,
) -> None:
    """A key-holding config file readable by group/others must trigger a WARNING."""
    config_path = _write(tmp_path / "eeane.toml", _KEYED_TOML)
    config_path.chmod(0o644)
    caplog.set_level(logging.WARNING, logger="eeane.cli")

    exit_code = cli.main([command, "--config", str(config_path)])

    assert exit_code == 0
    assert any("readable by group/others" in record.message for record in caplog.records)


@pytest.mark.parametrize("command", ["serve", "check-config"])
def test_private_key_file_does_not_warn(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    stub_uvicorn_run: _RecordingUvicornRun,
    command: str,
) -> None:
    """A key-holding config file restricted to chmod 600 must not warn."""
    config_path = _write(tmp_path / "eeane.toml", _KEYED_TOML)
    config_path.chmod(0o600)
    caplog.set_level(logging.WARNING, logger="eeane.cli")

    exit_code = cli.main([command, "--config", str(config_path)])

    assert exit_code == 0
    assert not any("readable by group/others" in record.message for record in caplog.records)


def test_env_sourced_api_key_does_not_warn_about_file_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When EEANE_API_KEY wins over the file's key, file permissions must not be checked."""
    config_path = _write(tmp_path / "eeane.toml", _KEYED_TOML)
    config_path.chmod(0o644)
    monkeypatch.setenv("EEANE_API_KEY", "env-key")
    caplog.set_level(logging.WARNING, logger="eeane.cli")

    exit_code = cli.main(["check-config", "--config", str(config_path)])

    assert exit_code == 0
    assert not any("readable by group/others" in record.message for record in caplog.records)


# --- check-config --------------------------------------------------------


def test_check_config_success_prints_effective_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A valid config must exit 0 and print host/port/model id/buckets to stdout."""
    config_path = _write(tmp_path / "eeane.toml", _KEYLESS_TOML)

    exit_code = cli.main(["check-config", "--config", str(config_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "127.0.0.1" in captured.out
    assert "7997" in captured.out
    assert "emb-only" in captured.out
    assert "256" in captured.out


def test_check_config_never_prints_api_key_value(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The api_key value must never appear in stdout/stderr, only its "(set" status."""
    config_path = _write(tmp_path / "eeane.toml", _KEYED_TOML)
    config_path.chmod(0o600)  # keep the permission warning out of this assertion's way

    exit_code = cli.main(["check-config", "--config", str(config_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "top-secret-key" not in captured.out
    assert "top-secret-key" not in captured.err
    assert "(set" in captured.out


def test_check_config_reports_missing_artifact_path(
    tmp_path: Path, capsys: pytest.CaptureFixture, caplog: pytest.LogCaptureFixture
) -> None:
    """A configured artifact path that does not exist must be marked [MISSING] and warned."""
    config_path = _write(tmp_path / "eeane.toml", _KEYLESS_TOML)
    caplog.set_level(logging.WARNING, logger="eeane.cli")

    exit_code = cli.main(["check-config", "--config", str(config_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "[MISSING]" in captured.out
    assert any("does not exist" in record.message for record in caplog.records)


def test_check_config_with_malformed_toml_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A TOML syntax error must exit 1 with a clean stderr message (no traceback)."""
    config_path = _write(tmp_path / "eeane.toml", "this is not valid toml [[[")

    exit_code = cli.main(["check-config", "--config", str(config_path)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "eeane.toml" in captured.err
    assert "Traceback" not in captured.err


def test_check_config_prints_the_optional_cache_and_model_details(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A cache root, an embedding width and excluded buckets must be reported."""
    config_path = _write(tmp_path / "eeane.toml", _DETAILED_TOML)

    exit_code = cli.main(["check-config", "--config", str(config_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "cache_root:" in captured.out
    assert "embedding_dim: 768" in captured.out
    assert "excluded_buckets: 1024" in captured.out


def test_check_config_omits_the_optional_details_when_unset(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """What the configuration does not set must not be printed at all."""
    config_path = _write(tmp_path / "eeane.toml", _KEYLESS_TOML)

    exit_code = cli.main(["check-config", "--config", str(config_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "cache_root:" not in captured.out
    assert "embedding_dim:" not in captured.out
    assert "excluded_buckets:" not in captured.out


def test_check_config_without_explicit_path_uses_cwd_eeane_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """With no --config, ./eeane.toml (if present) must be resolved and reported."""
    _isolate_config_search(tmp_path, monkeypatch)
    _write(tmp_path / "eeane.toml", _KEYLESS_TOML)

    exit_code = cli.main(["check-config"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert str(tmp_path / "eeane.toml") in captured.out


# --- python -m eeane.server backward compatibility --------------------------


def test_server_main_delegates_to_cli_serve(monkeypatch: pytest.MonkeyPatch) -> None:
    """eeane.server.main() must call eeane.cli.main(["serve"]) and propagate its exit code."""
    from eeane import server as server_module

    calls: list[list[str]] = []

    def fake_cli_main(argv: list[str]) -> int:
        calls.append(argv)
        return 0

    monkeypatch.setattr(server_module, "cli_main", fake_cli_main)

    with pytest.raises(SystemExit) as excinfo:
        server_module.main()

    assert calls == [["serve"]]
    assert excinfo.value.code == 0
