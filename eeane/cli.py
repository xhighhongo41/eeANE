"""Command-line interface for eeANE (v0.5実装計画.md §4.3).

Provides two subcommands:

* ``eeane serve`` -- resolve the configuration, initialize logging, and
  run the FastAPI application with uvicorn (single process, single
  worker; see :func:`eeane.server.create_app`).
* ``eeane check-config`` -- resolve the configuration and print it in a
  human-readable form (with the API key value always masked), without
  starting the server. Useful to validate a config file before deploying
  it.

Both subcommands share the same configuration resolution
(:func:`eeane.config.load_config`) and precedence rules (CLI overrides >
``EEANE_API_KEY`` > config file > built-in defaults).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from eeane.config import CliOverrides, ConfigError, LoadedConfig, load_config

logger = logging.getLogger("eeane.cli")

# Config file permission bits that indicate group/other read access. Used
# to warn when a config file holding an api_key is not private (chmod 600
# recommended), per 開発資料/v0.5実装計画.md §4.4.
_GROUP_OTHER_READABLE = 0o044

_LOG_LEVEL_CHOICES = ("debug", "info", "warning", "error")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``eeane`` argument parser (``serve`` / ``check-config``).

    Returns:
        A configured :class:`argparse.ArgumentParser`. Invalid arguments
        (unknown subcommand, bad ``--log-level`` choice, non-int
        ``--port``, ...) make ``parse_args`` exit with code 2, which is
        argparse's standard behaviour.
    """
    parser = argparse.ArgumentParser(prog="eeane", description="eeANE embedding/reranker server.")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser(
        "serve", help="Resolve the configuration and run the server."
    )
    _add_config_option(serve_parser)
    serve_parser.add_argument("--host", default=None, help="Override server.host (bind address).")
    serve_parser.add_argument("--port", type=int, default=None, help="Override server.port.")
    serve_parser.add_argument(
        "--log-level",
        choices=_LOG_LEVEL_CHOICES,
        default=None,
        help="Override server.log_level.",
    )
    serve_parser.set_defaults(func=_cmd_serve)

    check_config_parser = subparsers.add_parser(
        "check-config",
        help="Resolve the configuration and print it without starting the server.",
    )
    _add_config_option(check_config_parser)
    check_config_parser.set_defaults(func=_cmd_check_config)

    return parser


def _add_config_option(subparser: argparse.ArgumentParser) -> None:
    """Add the ``--config PATH`` option shared by both subcommands.

    Args:
        subparser: Subcommand parser to add the option to.
    """
    subparser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a TOML config file (default: search ./eeane.toml, "
        "~/.config/eeane/eeane.toml, then built-in defaults).",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the requested subcommand.

    Args:
        argv: Command-line arguments excluding the program name. ``None``
            (the default) reads from ``sys.argv``.

    Returns:
        Process exit code: ``0`` on success, ``1`` on a resolved
        :class:`~eeane.config.ConfigError`, ``2`` when no subcommand was
        given (usage is printed to stderr).

    Raises:
        SystemExit: Raised by argparse itself for malformed arguments or
            an unknown subcommand (exit code 2).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_usage(sys.stderr)
        return 2

    return args.func(args)


def _cmd_serve(args: argparse.Namespace) -> int:
    """Resolve the configuration and run the server (the ``serve`` subcommand).

    Args:
        args: Parsed ``serve`` arguments (``config``/``host``/``port``/
            ``log_level``).

    Returns:
        ``0`` if ``uvicorn.run`` returns (it normally does not: it blocks
        until the process is asked to stop). ``1`` if the configuration
        could not be resolved.
    """
    overrides = CliOverrides(host=args.host, port=args.port, log_level=args.log_level)
    try:
        loaded = load_config(explicit_path=args.config, overrides=overrides)
    except ConfigError as exc:
        print(f"eeane: {exc}", file=sys.stderr)
        return 1

    config = loaded.config
    # uvicorn only configures its own loggers, so enable output for
    # "eeane.*" loggers here (mirrors the format eeane.server.main used
    # in v0.4/early v0.5).
    logging.basicConfig(
        level=logging.getLevelNamesMapping()[config.server.log_level.upper()],
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info(
        "configuration source: %s",
        loaded.source if loaded.source is not None else "built-in defaults",
    )
    _warn_if_key_file_readable(loaded)

    # Imported here, not at module load time: eeane.server imports
    # eeane.cli.main for its python -m eeane.server backward-compat alias,
    # so a top-level import here would create an import cycle.
    from eeane.server import create_app

    uvicorn.run(create_app(config), host=config.server.host, port=config.server.port)
    return 0


def _cmd_check_config(args: argparse.Namespace) -> int:
    """Resolve the configuration and print it (the ``check-config`` subcommand).

    Args:
        args: Parsed ``check-config`` arguments (``config``).

    Returns:
        ``0`` if the configuration resolved successfully (even if some
        artifact paths are missing; existence checks are diagnostic
        only). ``1`` if the configuration could not be resolved.
    """
    try:
        loaded = load_config(explicit_path=args.config)
    except ConfigError as exc:
        print(f"eeane: {exc}", file=sys.stderr)
        return 1

    _print_effective_config(loaded)
    _warn_if_key_file_readable(loaded)
    return 0


def _warn_if_key_file_readable(loaded: LoadedConfig) -> None:
    """Warn when a config file holding ``api_key`` is group/other readable.

    Args:
        loaded: Result of :func:`eeane.config.load_config`. A no-op
            unless the effective ``api_key`` came from a file
            (``api_key_source == "file"``): an environment-sourced key
            has no file whose permissions matter here.
    """
    if loaded.api_key_source != "file" or loaded.source is None:
        return
    mode = loaded.source.stat().st_mode
    if mode & _GROUP_OTHER_READABLE:
        logger.warning(
            "config file %s containing api_key is readable by group/others; chmod 600 recommended",
            loaded.source,
        )


def _describe_api_key(api_key_source: str | None) -> str:
    """Describe the effective ``api_key`` for display, without its value.

    Args:
        api_key_source: :attr:`eeane.config.LoadedConfig.api_key_source`.

    Returns:
        ``"(set, from file)"``, ``"(set, from environment)"`` or
        ``"(not set)"``.
    """
    if api_key_source == "file":
        return "(set, from file)"
    if api_key_source == "env":
        return "(set, from environment)"
    return "(not set)"


def _print_effective_config(loaded: LoadedConfig) -> None:
    """Print the resolved configuration to stdout in human-readable form.

    Also checks whether every configured artifact path exists; missing
    ones are marked ``[MISSING]`` inline and logged as one WARNING each
    (the configuration remains valid: artifact existence is checked at
    engine startup, not here).

    Args:
        loaded: Result of :func:`eeane.config.load_config`.
    """
    config = loaded.config
    source = loaded.source if loaded.source is not None else "built-in defaults"

    print(f"configuration source: {source}")
    print("server:")
    print(f"  host: {config.server.host}")
    print(f"  port: {config.server.port}")
    print(f"  log_level: {config.server.log_level}")
    print(f"  api_key: {_describe_api_key(loaded.api_key_source)}")
    print(f"  health_rate_limit: {config.server.health_rate_limit}")

    print("models:")
    for entry in config.models:
        print(f"  - id: {entry.id}")
        print(f"    kind: {entry.kind}")
        print(f"    model_dir: {entry.model_dir}")
        print(f"    buckets: {', '.join(str(bucket) for bucket in entry.buckets)}")
        if entry.kind == "embedding":
            print(f"    normalize: {entry.normalize}")
        print("    artifacts:")
        for bucket in entry.buckets:
            path = entry.artifacts[bucket]
            if path.exists():
                print(f"      {bucket}: {path}")
            else:
                print(f"      {bucket}: {path} [MISSING]")
                logger.warning(
                    "artifact for model '%s' bucket %d does not exist: %s",
                    entry.id,
                    bucket,
                    path,
                )
