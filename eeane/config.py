"""Configuration schema, TOML loader, and built-in defaults for eeANE.

A TOML config file plus CLI/environment overrides, validated via pydantic
v2. See 開発資料/v0.5実装計画.md §4.1-§4.2 for the authoritative design,
and 開発資料/v0.6実装計画.md §4.6 for the v0.6 change from a
HuggingFace model directory to a frozen ``tokenizer.json`` per model.

Precedence (lowest to highest): built-in default < config file <
``EEANE_API_KEY`` environment variable (``api_key`` only) < CLI overrides.
No deep merge is implemented: if a config file is used, its ``models`` list
fully replaces the built-in default model list.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """Raised when a config file or the resolved configuration is invalid.

    Wraps both TOML syntax errors (``tomllib.TOMLDecodeError``) and
    pydantic ``ValidationError`` instances with a human-readable message
    that includes the offending file name and/or field name.
    """


class ServerConfig(BaseModel):
    """Validated ``[server]`` section of an eeANE TOML config file.

    Attributes:
        host: Bind address. Must be non-empty.
        port: Bind port, 1-65535.
        log_level: Python logging level name applied at CLI startup.
        api_key: Bearer token required for protected endpoints, or
            ``None`` to disable authentication. If provided, must be
            non-empty.
        health_rate_limit: Maximum ``/health`` requests per minute per
            client IP. ``0`` disables the limit.
    """

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=7997, ge=1, le=65535)
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    api_key: str | None = Field(default=None, min_length=1)
    health_rate_limit: int = Field(default=60, ge=0)


class ModelEntry(BaseModel):
    """A single ``[[models]]`` entry: one served embedding or reranker model.

    Attributes:
        id: Model identifier reported in API responses. Must be non-empty.
        kind: Either ``"embedding"`` or ``"reranker"``.
        tokenizer: Frozen ``tokenizer.json`` file written by ``eeane
            compile`` (v0.6実装計画.md §4.6). Pointing this at a
            HuggingFace-distributed ``tokenizer.json`` is rejected at
            engine startup: it carries no padding section.
        artifacts: Map of fixed sequence-length bucket to compiled Core ML
            artifact path (``.mlmodelc``). TOML tables always have string
            keys, so this is coerced to ``int`` explicitly rather than
            relying on pydantic's implicit coercion.
        normalize: Whether to L2-normalize embedding output. Only valid
            for ``kind="embedding"``; explicitly setting this on a
            ``kind="reranker"`` entry is a configuration error.
        output_name: Name of the Core ML output tensor to read. If
            omitted, derived from ``kind`` (``"embedding"`` ->
            ``"embedding"``, ``"reranker"`` -> ``"logits"``).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    kind: Literal["embedding", "reranker"]
    tokenizer: Path
    artifacts: dict[int, Path]
    normalize: bool = True
    output_name: str | None = None

    @field_validator("artifacts", mode="before")
    @classmethod
    def _coerce_artifact_keys(cls, value: Any, info: ValidationInfo) -> dict[int, Path]:
        """Coerce TOML string bucket-length keys to positive ints.

        Args:
            value: Raw ``artifacts`` value as parsed from TOML (or as
                re-supplied by :func:`load_config` during re-validation).
            info: Validation context; used to look up the entry's ``id``
                (already validated, since it is declared earlier in the
                model) for error messages.

        Returns:
            A dict mapping positive bucket-length ints to ``Path``.

        Raises:
            ValueError: If ``value`` is not a non-empty mapping, or any
                key is not a positive integer.
        """
        entry_id = info.data.get("id", "<unknown>")

        if not isinstance(value, dict):
            raise ValueError(f"model '{entry_id}': 'artifacts' must be a table of bucket lengths")
        if not value:
            raise ValueError(f"model '{entry_id}': 'artifacts' must not be empty")

        coerced: dict[int, Path] = {}
        for raw_key, raw_path in value.items():
            try:
                bucket = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"model '{entry_id}': artifacts key '{raw_key}' is not a valid integer "
                    "bucket length"
                ) from exc
            if bucket <= 0:
                raise ValueError(
                    f"model '{entry_id}': artifacts key '{raw_key}' must be a positive "
                    "bucket length"
                )
            coerced[bucket] = Path(raw_path)
        return coerced

    @model_validator(mode="after")
    def _finalize(self) -> ModelEntry:
        """Reject reranker-with-explicit-``normalize``; derive ``output_name``.

        Raises:
            ValueError: If ``normalize`` was explicitly set on a
                ``kind="reranker"`` entry.

        Returns:
            ``self``, with ``output_name`` filled in from ``kind`` when it
            was not explicitly provided.
        """
        if self.kind == "reranker" and "normalize" in self.model_fields_set:
            raise ValueError(
                f"model '{self.id}': 'normalize' may only be set on kind='embedding' entries"
            )
        if self.output_name is None:
            self.output_name = "embedding" if self.kind == "embedding" else "logits"
        return self

    @property
    def buckets(self) -> tuple[int, ...]:
        """Ascending tuple of sequence-length buckets covered by ``artifacts``."""
        return tuple(sorted(self.artifacts))


class EeaneConfig(BaseModel):
    """Top-level validated eeANE configuration.

    Attributes:
        server: Server/network/auth settings.
        models: Served model entries. Exactly one ``kind="embedding"``
            entry is required; at most one ``kind="reranker"`` entry is
            allowed. Serving multiple models of the same kind
            simultaneously is out of scope until v0.7.
    """

    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    models: list[ModelEntry]

    @model_validator(mode="after")
    def _validate_model_composition(self) -> EeaneConfig:
        """Enforce embedding=1, reranker<=1, and unique ``id`` across ``models``.

        Raises:
            ValueError: If the composition constraints are violated.

        Returns:
            ``self``.
        """
        embeddings = [entry for entry in self.models if entry.kind == "embedding"]
        rerankers = [entry for entry in self.models if entry.kind == "reranker"]

        if not embeddings:
            raise ValueError("exactly one 'embedding' model entry is required, found 0")
        if len(embeddings) > 1:
            raise ValueError(
                f"exactly one 'embedding' model entry is required, found {len(embeddings)}; "
                "serving multiple models simultaneously is planned for v0.7"
            )
        if len(rerankers) > 1:
            raise ValueError(
                f"at most one 'reranker' model entry is allowed, found {len(rerankers)}; "
                "serving multiple models simultaneously is planned for v0.7"
            )

        seen_ids: set[str] = set()
        for entry in self.models:
            if entry.id in seen_ids:
                raise ValueError(f"duplicate model id '{entry.id}' in 'models'")
            seen_ids.add(entry.id)

        return self

    @property
    def embedding_model(self) -> ModelEntry:
        """Return the (always present) configured embedding model entry."""
        for entry in self.models:
            if entry.kind == "embedding":
                return entry
        # Unreachable: _validate_model_composition guarantees exactly one.
        raise RuntimeError("no embedding model configured")  # pragma: no cover

    @property
    def reranker_model(self) -> ModelEntry | None:
        """Return the configured reranker model entry, or ``None`` if absent."""
        for entry in self.models:
            if entry.kind == "reranker":
                return entry
        return None


@dataclass
class CliOverrides:
    """CLI-supplied overrides for ``[server]`` scalar settings.

    ``None`` means "not provided on the command line"; only non-``None``
    fields are applied on top of the file/environment-resolved config.

    Attributes:
        host: Overrides ``server.host``.
        port: Overrides ``server.port``.
        log_level: Overrides ``server.log_level``.
    """

    host: str | None = None
    port: int | None = None
    log_level: str | None = None


@dataclass
class LoadedConfig:
    """Result of :func:`load_config`, including provenance for logging.

    Attributes:
        config: The fully resolved and validated configuration.
        source: Path to the TOML file that was used, or ``None`` if no
            config file was found and the built-in default was used.
        api_key_source: Where the effective ``api_key`` came from:
            ``"file"``, ``"env"``, or ``None`` if no key is configured.
    """

    config: EeaneConfig
    source: Path | None
    api_key_source: str | None


def default_config() -> EeaneConfig:
    """Build eeANE's built-in default configuration.

    Describes the development repository layout (paths resolved relative
    to the repository root): one embedding model (``ruri-v3-310m``,
    buckets 128/512/1024) and one reranker model
    (``ruri-v3-reranker-310m``, buckets 512/1024), each served from the
    artifacts and the frozen tokenizer under ``models/compiled/``.

    Returns:
        The built-in default configuration.
    """
    compiled_root = REPO_ROOT / "models" / "compiled"

    embedding = ModelEntry(
        id="ruri-v3-310m",
        kind="embedding",
        tokenizer=compiled_root / "ruri-v3-310m" / "tokenizer.json",
        artifacts={
            128: compiled_root / "ruri-v3-310m" / "s128_b1_eager_macos13.mlmodelc",
            512: compiled_root / "ruri-v3-310m" / "s512_b1_eager_macos13.mlmodelc",
            1024: compiled_root / "ruri-v3-310m" / "s1024_b1_eager_macos13.mlmodelc",
        },
        normalize=True,
    )
    reranker = ModelEntry(
        id="ruri-v3-reranker-310m",
        kind="reranker",
        tokenizer=compiled_root / "ruri-v3-reranker-310m" / "tokenizer.json",
        artifacts={
            512: compiled_root / "ruri-v3-reranker-310m" / "s512_b1_eager_macos13.mlmodelc",
            1024: compiled_root / "ruri-v3-reranker-310m" / "s1024_b1_eager_macos13.mlmodelc",
        },
    )
    return EeaneConfig(server=ServerConfig(), models=[embedding, reranker])


def load_config(
    explicit_path: Path | None = None,
    overrides: CliOverrides | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> LoadedConfig:
    """Resolve the effective eeANE configuration.

    Search order for the config file (first match wins):

    1. ``explicit_path``, if given (``ConfigError`` if it does not exist).
    2. ``./eeane.toml`` (current working directory).
    3. ``~/.config/eeane/eeane.toml``.
    4. None of the above: use :func:`default_config`.

    When a config file is used, relative ``tokenizer``/``artifacts``
    paths are resolved against the config file's parent directory before
    pydantic validation runs.

    Overrides are applied in ascending precedence: built-in default <
    config file < ``EEANE_API_KEY`` (``api_key`` only, empty string is
    treated as unset) < ``overrides`` (``host``/``port``/``log_level``,
    simple assignment, no deep merge). The result is re-validated after
    overrides are applied so that an invalid override (e.g. a bad
    ``log_level``) still surfaces as a ``ConfigError``.

    Args:
        explicit_path: Config file path explicitly requested by the
            caller (e.g. via ``--config``), taking precedence over the
            search order.
        overrides: CLI-supplied scalar overrides for ``server``.
        env: Environment mapping to read ``EEANE_API_KEY`` from. Defaults
            to ``os.environ`` when ``None`` (kept as a parameter for
            testability).

    Returns:
        The resolved configuration plus provenance information.

    Raises:
        ConfigError: If ``explicit_path`` does not exist, the config file
            has a TOML syntax error, or the resolved configuration fails
            pydantic validation.
    """
    env = env if env is not None else os.environ

    source = _resolve_source_path(explicit_path)
    config = default_config() if source is None else _load_from_file(source)

    api_key_source: str | None = "file" if config.server.api_key else None

    env_api_key = env.get("EEANE_API_KEY", "")
    if env_api_key:
        config.server.api_key = env_api_key
        api_key_source = "env"

    if overrides is not None:
        if overrides.host is not None:
            config.server.host = overrides.host
        if overrides.port is not None:
            config.server.port = overrides.port
        if overrides.log_level is not None:
            config.server.log_level = overrides.log_level  # type: ignore[assignment]

    # Plain attribute assignment above does not re-run pydantic validation
    # (validate_assignment is not enabled), so re-validate the final result
    # to catch invalid overrides (e.g. a bad --log-level) before returning.
    config = _revalidate(config)

    return LoadedConfig(config=config, source=source, api_key_source=api_key_source)


def _resolve_source_path(explicit_path: Path | None) -> Path | None:
    """Determine which config file (if any) to load, per the search order.

    Args:
        explicit_path: Caller-requested config path, taking precedence.

    Returns:
        The resolved config file path, or ``None`` if none was found and
        the built-in default should be used.

    Raises:
        ConfigError: If ``explicit_path`` is given but does not exist.
    """
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise ConfigError(f"Config file not found: '{explicit_path}'")
        return explicit_path

    cwd_candidate = Path.cwd() / "eeane.toml"
    if cwd_candidate.is_file():
        return cwd_candidate

    home_candidate = Path.home() / ".config" / "eeane" / "eeane.toml"
    if home_candidate.is_file():
        return home_candidate

    return None


def _load_from_file(path: Path) -> EeaneConfig:
    """Parse, resolve relative paths in, and validate a config file.

    Args:
        path: Config file to load.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file has a TOML syntax error, cannot be read,
            or fails pydantic validation.
    """
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Failed to parse TOML config file '{path}': {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Failed to read config file '{path}': {exc}") from exc

    # path may itself be relative (e.g. --config eeane.example.toml), so
    # resolve it first: model paths must come out absolute either way.
    _resolve_relative_paths(raw, base_dir=path.resolve().parent)

    try:
        return EeaneConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config file '{path}': {exc}") from exc
    except TypeError as exc:
        # raw's top-level shape did not match EeaneConfig's constructor
        # (e.g. "models" given as a scalar instead of a list of tables).
        raise ConfigError(f"Invalid config file '{path}': {exc}") from exc


def _resolve_relative_paths(raw: dict[str, Any], *, base_dir: Path) -> None:
    """Absolutize relative tokenizer/artifacts paths in-place, before validation.

    Args:
        raw: Dict as parsed by ``tomllib.load`` (mutated in place).
        base_dir: Directory the config file lives in; relative paths are
            resolved against this directory.
    """
    models = raw.get("models")
    if not isinstance(models, list):
        return

    for entry in models:
        if not isinstance(entry, dict):
            continue

        tokenizer = entry.get("tokenizer")
        if isinstance(tokenizer, str):
            entry["tokenizer"] = _resolve_one_path(base_dir, tokenizer)

        artifacts = entry.get("artifacts")
        if isinstance(artifacts, dict):
            for bucket_key, artifact_path in list(artifacts.items()):
                if isinstance(artifact_path, str):
                    artifacts[bucket_key] = _resolve_one_path(base_dir, artifact_path)


def _resolve_one_path(base_dir: Path, value: str) -> Path:
    """Resolve a single possibly-relative path string against ``base_dir``.

    Args:
        base_dir: Directory to resolve relative paths against.
        value: Raw path string from the config file.

    Returns:
        ``value`` unchanged (as a ``Path``) if already absolute, otherwise
        ``base_dir / value``.
    """
    candidate = Path(value)
    return candidate if candidate.is_absolute() else base_dir / candidate


def _revalidate(config: EeaneConfig) -> EeaneConfig:
    """Rebuild ``config`` through pydantic validation after attribute-assignment overrides.

    Args:
        config: Config whose ``server`` fields may have been mutated by
            plain attribute assignment (which bypasses validation).

    Returns:
        A freshly validated ``EeaneConfig`` with the same values.

    Raises:
        ConfigError: If the mutated values no longer pass validation.
    """
    try:
        server = ServerConfig(**config.server.model_dump())
        return EeaneConfig(server=server, models=config.models)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration after applying overrides: {exc}") from exc
