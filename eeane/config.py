"""Configuration schema, TOML loader, and built-in defaults for eeANE.

A TOML config file plus CLI/environment overrides, validated via pydantic
v2.

Precedence (lowest to highest): built-in default < config file <
``EEANE_API_KEY`` environment variable (``api_key`` only) < CLI overrides.
No deep merge is implemented: if a config file is used, its ``models`` list
fully replaces the built-in default model list.

Any number of embedding and reranker models may be served at once; within
each kind, the first entry listed is the default one. A ``[[models]]``
entry either spells out everything it needs (``kind``, ``tokenizer``,
``artifacts``) or gives only an ``id``, in which case the omitted fields
are filled in from the compiled-model cache written by ``eeane compile``
(see :mod:`eeane.cache`). Both forms end up equally complete: after
loading, every entry has a kind, a frozen tokenizer, and at least one
compiled artifact.
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

from eeane.cache import (
    MODEL_INFO_FILENAME,
    CacheError,
    load_model_info,
    model_cache_dir,
    resolve_cache_root,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Highest ``model_info.json`` schema version this release knows how to
# read. A newer record may assign different meanings to the keys below,
# so it is rejected instead of guessed at.
MAX_MODEL_INFO_FORMAT_VERSION = 2


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
        cache_root: Root of the compiled-model cache used to auto-resolve
            model entries, or ``None`` for the default location (see
            :func:`eeane.cache.resolve_cache_root`). A relative path is
            resolved against the config file's directory.
        default_load_policy: Default load policy for a ``[[models]]``
            entry that omits ``load_policy``. Restricted to
            ``"resident"``/``"on_demand"``: ``"disabled"`` is only
            available as an explicit per-model choice, never as a
            server-wide default.
        keep_alive: Default number of seconds a loaded model may sit
            idle before it becomes eligible for unloading, for entries
            that omit ``keep_alive``. ``0`` means "eligible as soon as
            it is idle". Idle-unload behavior itself is the serving
            engine's responsibility; this field only holds and
            validates the configured value.
        max_loaded_models: Maximum number of models the serving engine
            may hold in memory at once, or ``None`` for no limit.
            Config-time validation only checks this against
            ``resident``-policy entries (see
            :meth:`EeaneConfig.resolved_load_policy`), since those are
            guaranteed to be loaded at all times; keeping on-demand
            entries within the limit at runtime is the serving engine's
            responsibility.
        max_pending_requests: Maximum number of inference requests the
            serving engine admits at once, counting requests currently
            being processed and requests still waiting for their turn.
            ``0`` means "unlimited". Enforcing this cap is the serving
            engine's responsibility; this field only holds and validates
            the configured value.
        queue_timeout: Maximum number of seconds a request may wait
            between being admitted and starting inference before it is
            abandoned. ``0`` means "no timeout". Enforcing this is the
            serving engine's responsibility; this field only holds and
            validates the configured value.
        coalesce_requests: Whether an incoming request with the same
            content as one already being processed is served by
            attaching to that in-flight request instead of running
            inference again. Implementing the coalescing itself is the
            serving engine's responsibility; this field only holds and
            validates the configured value.
        graceful_shutdown_timeout: Maximum number of seconds the server
            waits for in-flight requests to finish while shutting down,
            or ``None`` to wait for them indefinitely (the same default
            behavior as the underlying ASGI server). Implementing the
            wait itself is the serving engine's responsibility; this
            field only holds and validates the configured value.
    """

    model_config = ConfigDict(extra="forbid")

    host: str = Field(default="127.0.0.1", min_length=1)
    port: int = Field(default=7997, ge=1, le=65535)
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    api_key: str | None = Field(default=None, min_length=1)
    health_rate_limit: int = Field(default=60, ge=0)
    cache_root: Path | None = None
    default_load_policy: Literal["resident", "on_demand"] = "on_demand"
    keep_alive: int = Field(default=300, ge=0)
    max_loaded_models: int | None = Field(default=None, ge=1)
    max_pending_requests: int = Field(default=500, ge=0)
    queue_timeout: int = Field(default=600, ge=0)
    coalesce_requests: bool = True
    graceful_shutdown_timeout: int | None = Field(default=None, ge=1)


def _coerce_bucket_path_map(value: Any, *, entry_id: str, field: str) -> dict[int, Path] | None:
    """Coerce a TOML bucket-length-to-path table's string keys to positive ints.

    Shared by :class:`ModelEntry`'s ``artifacts`` and ``batch_artifacts``
    before-validators: TOML tables always have string keys, so this is
    coerced explicitly rather than relying on pydantic's implicit
    coercion.

    Args:
        value: Raw field value as parsed from TOML (or as re-supplied by
            :func:`load_config` during re-validation). ``None`` means
            "not configured" and is passed through unchanged.
        entry_id: The owning entry's ``id``, for error messages.
        field: Name of the field being coerced (``"artifacts"`` or
            ``"batch_artifacts"``), for error messages.

    Returns:
        A dict mapping positive bucket-length ints to ``Path``, or
        ``None``.

    Raises:
        ValueError: If ``value`` is not a non-empty mapping, or any key
            is not a positive integer.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"model '{entry_id}': '{field}' must be a table of bucket lengths")
    if not value:
        raise ValueError(f"model '{entry_id}': '{field}' must not be empty")

    coerced: dict[int, Path] = {}
    for raw_key, raw_path in value.items():
        try:
            bucket = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"model '{entry_id}': {field} key '{raw_key}' is not a valid integer bucket length"
            ) from exc
        if bucket <= 0:
            raise ValueError(
                f"model '{entry_id}': {field} key '{raw_key}' must be a positive bucket length"
            )
        coerced[bucket] = Path(raw_path)
    return coerced


class ModelEntry(BaseModel):
    """A single ``[[models]]`` entry: one served embedding or reranker model.

    Every field except ``id`` may be omitted in the config file and filled
    in from the compiled-model cache instead (see :func:`load_config`).
    Validation therefore runs *after* that resolution: an entry that still
    lacks a kind, a tokenizer, or artifacts by then is rejected.

    Attributes:
        id: Model identifier reported in API responses, and the key used
            to look the model up in the compiled-model cache. Must be
            non-empty.
        kind: Either ``"embedding"`` or ``"reranker"``.
        tokenizer: Frozen ``tokenizer.json`` file written by ``eeane
            compile``. Pointing this at a HuggingFace-distributed
            ``tokenizer.json`` is rejected at engine startup: it carries
            no padding section.
        artifacts: Map of fixed sequence-length bucket to compiled Core ML
            artifact path (``.mlmodelc``). TOML tables always have string
            keys, so this is coerced to ``int`` explicitly rather than
            relying on pydantic's implicit coercion.
        batch_artifacts: Map of fixed sequence-length bucket to a Core ML
            artifact compiled for a batch size of two, for the buckets
            where one is available; ``None`` when none is configured.
            Coerced from string TOML keys the same way as ``artifacts``.
            Only valid for ``kind="embedding"`` entries; every key must
            already be one of ``artifacts``' buckets; and setting this
            field requires ``artifacts`` itself to be stated explicitly
            (it cannot be combined with an id-only entry resolved from
            the compiled-model cache).
        normalize: Whether to L2-normalize embedding output. Only valid
            for ``kind="embedding"``; explicitly setting this on a
            ``kind="reranker"`` entry is a configuration error.
        output_name: Name of the Core ML output tensor to read. If
            omitted (and not recorded in the cache), derived from ``kind``
            (``"embedding"`` -> ``"embedding"``, ``"reranker"`` ->
            ``"logits"``).
        embedding_dim: Width of the embedding vectors, when known. Filled
            in from the cache for embedding models compiled by a release
            that records it; ``None`` otherwise, and always ``None`` for a
            reranker.
        excluded_buckets: Buckets present in the cache but left out of
            ``artifacts`` because the cache recommends against loading
            them. Informational only (reported at server startup).
        load_policy: How this model is loaded: ``"resident"`` (loaded at
            startup and kept loaded), ``"on_demand"`` (loaded on first
            use and unloaded once idle), or ``"disabled"`` (never
            served; the entry is removed from
            :attr:`EeaneConfig.models` during validation and its ``id``
            is recorded in :attr:`EeaneConfig.disabled_models` instead).
            ``None`` means "use ``server.default_load_policy``" (see
            :meth:`EeaneConfig.resolved_load_policy`); ``"disabled"`` is
            only available as an explicit per-model choice, since it
            cannot be the server-wide default.
        keep_alive: Number of seconds this model may sit idle before it
            becomes eligible for unloading, or ``None`` to use
            ``server.keep_alive`` (see
            :meth:`EeaneConfig.resolved_keep_alive`).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    kind: Literal["embedding", "reranker"] | None = None
    tokenizer: Path | None = None
    artifacts: dict[int, Path] | None = None
    batch_artifacts: dict[int, Path] | None = None
    normalize: bool = True
    output_name: str | None = None
    embedding_dim: int | None = Field(default=None, gt=0)
    excluded_buckets: tuple[int, ...] = ()
    load_policy: Literal["resident", "on_demand", "disabled"] | None = None
    keep_alive: int | None = Field(default=None, ge=0)

    @field_validator("artifacts", mode="before")
    @classmethod
    def _coerce_artifact_keys(cls, value: Any, info: ValidationInfo) -> dict[int, Path] | None:
        """Coerce TOML string bucket-length keys to positive ints.

        Args:
            value: Raw ``artifacts`` value as parsed from TOML (or as
                re-supplied by :func:`load_config` during re-validation).
                ``None`` means "not configured yet" and is passed through
                for the model-level validator to report.
            info: Validation context; used to look up the entry's ``id``
                (already validated, since it is declared earlier in the
                model) for error messages.

        Returns:
            A dict mapping positive bucket-length ints to ``Path``, or
            ``None``.

        Raises:
            ValueError: If ``value`` is not a non-empty mapping, or any
                key is not a positive integer.
        """
        entry_id = info.data.get("id", "<unknown>")
        return _coerce_bucket_path_map(value, entry_id=entry_id, field="artifacts")

    @field_validator("batch_artifacts", mode="before")
    @classmethod
    def _coerce_batch_artifact_keys(
        cls, value: Any, info: ValidationInfo
    ) -> dict[int, Path] | None:
        """Coerce TOML string bucket-length keys to positive ints, like ``artifacts``.

        Args:
            value: Raw ``batch_artifacts`` value as parsed from TOML (or
                as re-supplied by :func:`load_config` during
                re-validation). ``None`` means "not configured" and is
                passed through unchanged.
            info: Validation context; used to look up the entry's ``id``
                (already validated, since it is declared earlier in the
                model) for error messages.

        Returns:
            A dict mapping positive bucket-length ints to ``Path``, or
            ``None``.

        Raises:
            ValueError: If ``value`` is not a non-empty mapping, or any
                key is not a positive integer.
        """
        entry_id = info.data.get("id", "<unknown>")
        return _coerce_bucket_path_map(value, entry_id=entry_id, field="batch_artifacts")

    @model_validator(mode="after")
    def _finalize(self) -> ModelEntry:
        """Check completeness, cross-field constraints, and derive ``output_name``.

        Raises:
            ValueError: If ``kind``/``tokenizer``/``artifacts`` are still
                unset (neither configured nor resolved from the cache); if
                ``normalize`` was explicitly set on a ``kind="reranker"``
                entry; if ``batch_artifacts`` was set on a
                ``kind="reranker"`` entry; or if ``batch_artifacts``
                contains a bucket that is not one of ``artifacts``'
                buckets.

        Returns:
            ``self``, with ``output_name`` filled in from ``kind`` when it
            was not explicitly provided.
        """
        missing = [
            name for name in ("kind", "tokenizer", "artifacts") if getattr(self, name) is None
        ]
        if missing:
            listed = ", ".join(f"'{name}'" for name in missing)
            raise ValueError(
                f"model '{self.id}': {listed} must be set explicitly unless the entry is "
                "resolved from the compiled-model cache (omit 'tokenizer' and 'artifacts' "
                "to resolve it by id)"
            )
        if self.kind == "reranker" and "normalize" in self.model_fields_set:
            raise ValueError(
                f"model '{self.id}': 'normalize' may only be set on kind='embedding' entries"
            )
        if self.batch_artifacts is not None:
            if self.kind == "reranker":
                raise ValueError(
                    f"model '{self.id}': 'batch_artifacts' may only be set on "
                    "kind='embedding' entries"
                )
            unknown_buckets = sorted(set(self.batch_artifacts) - set(self.buckets))
            if unknown_buckets:
                raise ValueError(
                    f"model '{self.id}': batch_artifacts bucket(s) {unknown_buckets} are not "
                    f"among the configured artifacts buckets {list(self.buckets)}"
                )
        if self.output_name is None:
            self.output_name = "embedding" if self.kind == "embedding" else "logits"
        return self

    @property
    def buckets(self) -> tuple[int, ...]:
        """Ascending tuple of sequence-length buckets covered by ``artifacts``."""
        return tuple(sorted(self.artifacts or ()))


class EeaneConfig(BaseModel):
    """Top-level validated eeANE configuration.

    Attributes:
        server: Server/network/auth settings.
        models: Served model entries, in config-file order, after
            ``load_policy="disabled"`` entries have been removed (see
            ``disabled_models``). At least one ``kind="embedding"`` entry
            is required (the engine's embedding endpoints have no
            meaning without one); rerankers are optional. Any number of
            either kind may be listed, as long as every ``id`` is unique
            across kinds and across disabled entries too.
        disabled_models: Ids of ``[[models]]`` entries whose
            ``load_policy`` resolved to ``"disabled"``, in their
            original config-file order. These entries never appear in
            ``models``; this field exists only so callers can report
            which ids were configured but left unserved.
    """

    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    models: list[ModelEntry]
    disabled_models: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _split_disabled_models(self) -> EeaneConfig:
        """Move ``load_policy="disabled"`` entries out of ``models`` into ``disabled_models``.

        Runs before ``_validate_model_composition`` so the id-uniqueness
        and embedding-count checks there see the already-filtered
        ``models`` list. For a config file loaded via :func:`load_config`,
        disabled entries are already stripped out of the raw model list
        before this validator ever sees them (so that an id-only
        disabled entry does not need a compiled-model cache to exist);
        this validator's own filtering only matters for ``ModelEntry``
        objects constructed directly with ``load_policy="disabled"``.

        Returns:
            ``self``, with ``models``/``disabled_models`` updated to
            reflect the split.
        """
        active: list[ModelEntry] = []
        newly_disabled: list[str] = []
        for entry in self.models:
            if entry.load_policy == "disabled":
                newly_disabled.append(entry.id)
            else:
                active.append(entry)

        self.models = active
        self.disabled_models = (*self.disabled_models, *newly_disabled)
        return self

    @model_validator(mode="after")
    def _validate_model_composition(self) -> EeaneConfig:
        """Enforce unique ids, embedding>=1, and the resident-count cap.

        Raises:
            ValueError: If the composition constraints are violated.

        Returns:
            ``self``.
        """
        seen_ids: set[str] = set()
        for model_id in (*self.disabled_models, *(entry.id for entry in self.models)):
            if model_id in seen_ids:
                raise ValueError(f"duplicate model id '{model_id}' in 'models'")
            seen_ids.add(model_id)

        if not self.models_of_kind("embedding"):
            raise ValueError("at least one 'embedding' model entry is required, found 0")

        if self.server.max_loaded_models is not None:
            resident_count = sum(
                1 for entry in self.models if self.resolved_load_policy(entry) == "resident"
            )
            if resident_count > self.server.max_loaded_models:
                raise ValueError(
                    f"{resident_count} model(s) resolve to load_policy='resident', which "
                    f"exceeds server.max_loaded_models={self.server.max_loaded_models}"
                )

        return self

    def resolved_load_policy(self, entry: ModelEntry) -> Literal["resident", "on_demand"]:
        """Return ``entry``'s effective load policy, applying the server default.

        Args:
            entry: A model entry from ``models`` (i.e. not disabled).

        Returns:
            ``entry.load_policy`` if set, otherwise
            ``server.default_load_policy``.

        Raises:
            ValueError: If ``entry.load_policy`` is ``"disabled"``. This
                cannot happen for an entry still listed in ``models``,
                since disabled entries are filtered out during
                validation.
        """
        if entry.load_policy is None:
            return self.server.default_load_policy
        if entry.load_policy == "disabled":
            # Unreachable for an entry obtained via `models`/`model_by_id`:
            # disabled entries never survive validation into `models`.
            raise ValueError(
                f"model '{entry.id}' has load_policy='disabled' and is not servable"
            )  # pragma: no cover
        return entry.load_policy

    def resolved_keep_alive(self, entry: ModelEntry) -> int:
        """Return ``entry``'s effective idle-unload delay in seconds, applying the server default.

        Args:
            entry: A model entry from ``models``.

        Returns:
            ``entry.keep_alive`` if set, otherwise ``server.keep_alive``.
        """
        return entry.keep_alive if entry.keep_alive is not None else self.server.keep_alive

    def models_of_kind(self, kind: str) -> list[ModelEntry]:
        """Return every configured entry of ``kind``, in config-file order.

        Args:
            kind: ``"embedding"`` or ``"reranker"``.

        Returns:
            The matching entries; empty if none are configured.
        """
        return [entry for entry in self.models if entry.kind == kind]

    def default_model(self, kind: str) -> ModelEntry | None:
        """Return the default entry of ``kind``: the first one listed.

        Args:
            kind: ``"embedding"`` or ``"reranker"``.

        Returns:
            The first configured entry of that kind, or ``None`` if none
            is configured. An ``"embedding"`` lookup always succeeds: the
            composition rules require at least one.
        """
        for entry in self.models:
            if entry.kind == kind:
                return entry
        return None

    def model_by_id(self, model_id: str) -> ModelEntry | None:
        """Return the entry with the given ``id``.

        Args:
            model_id: Exact model id as configured.

        Returns:
            The matching entry (ids are unique), or ``None`` if no
            configured model has that id.
        """
        for entry in self.models:
            if entry.id == model_id:
                return entry
        return None

    @property
    def embedding_model(self) -> ModelEntry:
        """Return the default (first-listed) embedding model entry."""
        entry = self.default_model("embedding")
        if entry is None:
            # Unreachable: _validate_model_composition requires one.
            raise RuntimeError("no embedding model configured")  # pragma: no cover
        return entry

    @property
    def reranker_model(self) -> ModelEntry | None:
        """Return the default (first-listed) reranker entry, or ``None`` if absent."""
        return self.default_model("reranker")


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

    When a config file is used, relative ``tokenizer``/``artifacts``/
    ``cache_root`` paths are resolved against the config file's parent
    directory, and model entries that omit ``tokenizer`` or ``artifacts``
    are completed from the compiled-model cache, both before pydantic
    validation runs.

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
        env: Environment mapping to read ``EEANE_API_KEY`` and
            ``XDG_CACHE_HOME`` from. Defaults to ``os.environ`` when
            ``None`` (kept as a parameter for testability).

    Returns:
        The resolved configuration plus provenance information.

    Raises:
        ConfigError: If ``explicit_path`` does not exist, the config file
            has a TOML syntax error, a model entry cannot be resolved
            from the compiled-model cache, or the resolved configuration
            fails pydantic validation.
    """
    env = env if env is not None else os.environ

    source = _resolve_source_path(explicit_path)
    config = default_config() if source is None else _load_from_file(source, env=env)

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


def _load_from_file(path: Path, *, env: Mapping[str, str]) -> EeaneConfig:
    """Parse, resolve paths and cache references in, and validate a config file.

    Args:
        path: Config file to load.
        env: Environment mapping used to locate the compiled-model cache.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If the file has a TOML syntax error, cannot be read,
            references a model that is not in the compiled-model cache, or
            fails pydantic validation.
    """
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Failed to parse TOML config file '{path}': {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Failed to read config file '{path}': {exc}") from exc

    # Disabled entries are pulled out first, before cache resolution: an
    # id-only entry that is disabled must not need a compiled-model cache
    # to exist just to be skipped.
    _split_disabled_entries(raw)

    # path may itself be relative (e.g. --config eeane.example.toml), so
    # resolve it first: model paths must come out absolute either way.
    _resolve_relative_paths(raw, base_dir=path.resolve().parent)
    _resolve_from_cache(raw, source=path, env=env)

    try:
        return EeaneConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config file '{path}': {exc}") from exc
    except TypeError as exc:
        # raw's top-level shape did not match EeaneConfig's constructor
        # (e.g. "models" given as a scalar instead of a list of tables).
        raise ConfigError(f"Invalid config file '{path}': {exc}") from exc


def _split_disabled_entries(raw: dict[str, Any]) -> None:
    """Move ``load_policy = "disabled"`` model entries out of ``raw["models"]``, in place.

    Runs before ``_resolve_from_cache`` so a disabled entry that only
    states an ``id`` never triggers a compiled-model cache lookup: it is
    simply never served, whether or not that cache entry exists.

    Args:
        raw: Dict as parsed by ``tomllib.load`` (mutated in place).
    """
    models = raw.get("models")
    if not isinstance(models, list):
        return

    active: list[Any] = []
    disabled_ids: list[Any] = []
    for entry in models:
        if isinstance(entry, dict) and entry.get("load_policy") == "disabled":
            disabled_ids.append(entry.get("id"))
        else:
            active.append(entry)

    if not disabled_ids:
        return
    raw["models"] = active
    raw["disabled_models"] = [*raw.get("disabled_models", []), *disabled_ids]


def _resolve_relative_paths(raw: dict[str, Any], *, base_dir: Path) -> None:
    """Absolutize relative tokenizer/artifacts/cache_root paths in-place, before validation.

    Args:
        raw: Dict as parsed by ``tomllib.load`` (mutated in place).
        base_dir: Directory the config file lives in; relative paths are
            resolved against this directory.
    """
    server = raw.get("server")
    if isinstance(server, dict):
        cache_root = server.get("cache_root")
        if isinstance(cache_root, str):
            # ``~`` is expanded here rather than joined onto base_dir: a
            # cache root is a machine-level location, so a home-relative
            # spelling is the natural one to write.
            server["cache_root"] = _resolve_one_path(base_dir, cache_root, expand_user=True)

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

        batch_artifacts = entry.get("batch_artifacts")
        if isinstance(batch_artifacts, dict):
            for bucket_key, artifact_path in list(batch_artifacts.items()):
                if isinstance(artifact_path, str):
                    batch_artifacts[bucket_key] = _resolve_one_path(base_dir, artifact_path)


def _resolve_one_path(base_dir: Path, value: str, *, expand_user: bool = False) -> Path:
    """Resolve a single possibly-relative path string against ``base_dir``.

    Args:
        base_dir: Directory to resolve relative paths against.
        value: Raw path string from the config file.
        expand_user: Whether a leading ``~`` is expanded to the user's
            home directory before the relative/absolute decision.

    Returns:
        ``value`` unchanged (as a ``Path``) if already absolute, otherwise
        ``base_dir / value``.
    """
    candidate = Path(value)
    if expand_user:
        candidate = candidate.expanduser()
    return candidate if candidate.is_absolute() else base_dir / candidate


def _resolve_from_cache(raw: dict[str, Any], *, source: Path, env: Mapping[str, str]) -> None:
    """Complete cache-resolvable model entries in-place, before validation.

    An entry is left untouched when it spells out both ``tokenizer`` and
    ``artifacts``: that is the fully explicit form, which must keep
    working without any cache on disk. Otherwise the entry's ``id`` is
    looked up in the compiled-model cache and every omitted field is
    filled in from the recorded ``model_info.json``.

    Args:
        raw: Dict as parsed by ``tomllib.load`` (mutated in place).
        source: Config file the entries came from, for error messages.
        env: Environment mapping used to locate the cache root.

    Raises:
        ConfigError: If an entry needs the cache and the cache cannot be
            read or contradicts the entry.
    """
    models = raw.get("models")
    if not isinstance(models, list):
        return

    cache_root_value: Any = None
    server = raw.get("server")
    if isinstance(server, dict):
        cache_root_value = server.get("cache_root")
        if cache_root_value is not None and not isinstance(cache_root_value, str | Path):
            # Guessing a cache root here would hide the real problem;
            # leave the entries alone so pydantic reports the bad type.
            return

    cache_root: Path | None = None
    for entry in models:
        if not isinstance(entry, dict):
            continue
        if entry.get("tokenizer") is not None and entry.get("artifacts") is not None:
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            # An unusable id is a schema error; let pydantic name it.
            continue
        if cache_root is None:
            override = None if cache_root_value is None else Path(cache_root_value)
            cache_root = resolve_cache_root(override, env=env)
        try:
            _fill_entry_from_cache(entry, model_id, cache_root)
        except (CacheError, ConfigError) as exc:
            raise ConfigError(f"Invalid config file '{source}': model '{model_id}': {exc}") from exc


def _fill_entry_from_cache(entry: dict[str, Any], model_id: str, cache_root: Path) -> None:
    """Fill one raw model entry's omitted fields from the compiled-model cache.

    Args:
        entry: Raw ``[[models]]`` table (mutated in place).
        model_id: The entry's ``id``, used as the cache lookup key.
        cache_root: Root of the compiled-model cache.

    Raises:
        ConfigError: If the model is not in the cache, its record is
            unreadable/malformed, the record contradicts a field the
            entry states explicitly, or the entry sets ``batch_artifacts``
            without also stating ``artifacts`` explicitly. Messages are
            written without the config-file context, which the caller
            adds.
    """
    model_dir = model_cache_dir(cache_root, model_id)
    try:
        info = load_model_info(model_dir)
    except CacheError as exc:
        missing = ", ".join(
            f"'{name}'" for name in ("tokenizer", "artifacts") if entry.get(name) is None
        )
        raise ConfigError(
            f"{missing} not set and the compiled-model cache in '{model_dir}' cannot be "
            f"used ({exc}); run 'eeane compile {model_id}' first, or set 'kind', "
            "'tokenizer' and 'artifacts' explicitly"
        ) from exc

    record = f"'{model_dir / MODEL_INFO_FILENAME}'"
    version = info.get("format_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ConfigError(
            f"{record} has no usable 'format_version'; re-run 'eeane compile {model_id}'"
        )
    if version > MAX_MODEL_INFO_FORMAT_VERSION:
        raise ConfigError(
            f"{record} has format_version {version}, which this eeANE release cannot read "
            f"(it understands up to {MAX_MODEL_INFO_FORMAT_VERSION}); upgrade eeANE or "
            f"re-run 'eeane compile {model_id}' with this release"
        )

    kind = info.get("kind")
    if kind not in ("embedding", "reranker"):
        raise ConfigError(f"{record} records an unsupported model kind {kind!r}")
    declared_kind = entry.get("kind")
    if declared_kind is not None and declared_kind != kind:
        raise ConfigError(
            f"the config declares kind '{declared_kind}', but the compiled model in "
            f"'{model_dir}' is a '{kind}' model"
        )
    entry["kind"] = kind

    if entry.get("tokenizer") is None:
        entry["tokenizer"] = _cache_relative_path(
            model_dir, info.get("tokenizer"), field="tokenizer"
        )

    if entry.get("artifacts") is None:
        if entry.get("batch_artifacts") is not None:
            # batch_artifacts pairs each bucket with a specific compiled
            # artifact, so it only makes sense alongside an explicit
            # artifacts table; an id-only entry cannot supply one.
            raise ConfigError(
                "'batch_artifacts' requires explicit 'artifacts' (an id-only entry resolved "
                "from the compiled-model cache cannot use 'batch_artifacts')"
            )
        artifacts, excluded = _cached_artifacts(info, model_dir)
        entry["artifacts"] = artifacts
        entry["excluded_buckets"] = excluded

    if entry.get("output_name") is None:
        output_name = info.get("output_name")
        if output_name is not None:
            if not isinstance(output_name, str) or not output_name:
                raise ConfigError(f"{record} records an unusable 'output_name'")
            entry["output_name"] = output_name

    # Recorded only from format_version 2 on, and never for a reranker:
    # an absent or null value simply leaves the width unknown.
    if entry.get("embedding_dim") is None:
        embedding_dim = info.get("embedding_dim")
        if embedding_dim is not None:
            if (
                not isinstance(embedding_dim, int)
                or isinstance(embedding_dim, bool)
                or embedding_dim <= 0
            ):
                raise ConfigError(f"{record} records an unusable 'embedding_dim'")
            entry["embedding_dim"] = embedding_dim


def _cached_artifacts(
    info: Mapping[str, Any], model_dir: Path
) -> tuple[dict[int, Path], list[int]]:
    """Pick the artifacts to load from a cache record.

    Args:
        info: Parsed ``model_info.json`` contents.
        model_dir: Directory the record lives in; artifact file names are
            recorded relative to it.

    Returns:
        A ``(artifacts, excluded_buckets)`` pair. When the record carries
        a ``recommended_buckets`` list, only those buckets are loaded and
        the remaining compiled ones are reported as excluded.

    Raises:
        ConfigError: If the record lists no usable artifact, or its
            recommendation leaves nothing to load.
    """
    record = f"'{model_dir / MODEL_INFO_FILENAME}'"
    recorded = info.get("artifacts")
    if not isinstance(recorded, dict) or not recorded:
        raise ConfigError(f"{record} lists no compiled artifacts")

    artifacts: dict[int, Path] = {}
    for raw_key, raw_name in recorded.items():
        try:
            bucket = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{record} has artifacts key '{raw_key}', which is not a bucket length"
            ) from exc
        if bucket <= 0:
            raise ConfigError(
                f"{record} has artifacts key '{raw_key}', which is not a positive bucket length"
            )
        artifacts[bucket] = _cache_relative_path(model_dir, raw_name, field="artifacts")

    recommended = info.get("recommended_buckets")
    if recommended is None:
        # Recorded only from format_version 2 on: without it, every
        # compiled bucket is loaded, as earlier releases did.
        return artifacts, []
    if not isinstance(recommended, list) or any(
        not isinstance(bucket, int) or isinstance(bucket, bool) for bucket in recommended
    ):
        raise ConfigError(f"{record} has an unusable 'recommended_buckets' list")

    wanted = set(recommended)
    selected = {bucket: path for bucket, path in artifacts.items() if bucket in wanted}
    if not selected:
        raise ConfigError(
            f"{record} has 'recommended_buckets' {sorted(wanted)}, none of which are "
            f"compiled (available: {sorted(artifacts)})"
        )
    return selected, sorted(set(artifacts) - set(selected))


def _cache_relative_path(model_dir: Path, value: Any, *, field: str) -> Path:
    """Turn a file name recorded in the cache into an absolute path.

    Args:
        model_dir: Directory the record lives in.
        value: Recorded file name, expected to be a relative path that
            stays inside ``model_dir``.
        field: Field name, for error messages.

    Returns:
        ``model_dir / value``.

    Raises:
        ConfigError: If ``value`` is not a relative, non-escaping file
            name (a record pointing outside its own directory is treated
            as corrupt rather than followed).
    """
    record = f"'{model_dir / MODEL_INFO_FILENAME}'"
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{record} records no usable '{field}' file name")

    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigError(
            f"{record} points '{field}' at '{value}', outside its own cache directory"
        )
    return model_dir / relative


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
        return EeaneConfig(
            server=server, models=config.models, disabled_models=config.disabled_models
        )
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration after applying overrides: {exc}") from exc
