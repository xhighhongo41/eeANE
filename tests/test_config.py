"""Tests for eeane.config: TOML schema, overrides, and cache auto-resolution."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from eeane.config import (
    CliOverrides,
    ConfigError,
    EeaneConfig,
    ModelEntry,
    ServerConfig,
    default_config,
    load_config,
)

# --- fixtures / helpers --------------------------------------------------


def _write_toml(path: Path, content: str) -> Path:
    """Write ``content`` to ``path`` and return ``path`` for chaining."""
    path.write_text(content, encoding="utf-8")
    return path


def _write_cached_model(
    cache_root: Path,
    model_id: str,
    *,
    format_version: int = 2,
    kind: str = "embedding",
    buckets: Sequence[int] = (128, 512),
    recommended_buckets: Sequence[int] | None = None,
    embedding_dim: int | None = 768,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Create a compiled-model cache entry the way ``eeane compile`` does.

    Args:
        cache_root: Cache root to create the entry under.
        model_id: Model id; a Hub id is normalised for the directory name.
        format_version: ``model_info.json`` schema version to emit. Version
            1 omits the fields introduced later, so it exercises the
            degraded path.
        kind: ``"embedding"`` or ``"reranker"``.
        buckets: Compiled sequence-length buckets.
        recommended_buckets: ``recommended_buckets`` value (v2+); defaults
            to every compiled bucket.
        embedding_dim: ``embedding_dim`` value (v2+, embeddings only).
        overrides: Keys merged into the record last, to inject malformed
            or unknown values.

    Returns:
        The created model directory.
    """
    model_dir = cache_root / "compiled" / model_id.replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {str(bucket): f"s{bucket}_b1_eager_macos13.mlmodelc" for bucket in buckets}
    info: dict[str, Any] = {
        "format_version": format_version,
        "id": model_id,
        "kind": kind,
        "output_name": "embedding" if kind == "embedding" else "logits",
        "buckets": sorted(buckets),
        "tokenizer": "tokenizer.json",
        "artifacts": artifacts,
        "eeane_version": "test",
    }
    if format_version >= 2:
        info["embedding_dim"] = embedding_dim if kind == "embedding" else None
        info["recommended_buckets"] = list(
            recommended_buckets if recommended_buckets is not None else sorted(buckets)
        )
        # Recorded by the compiler for provenance; the config layer must
        # ignore its contents entirely.
        info["calibration"] = {"measured_at": "2026-01-01T00:00:00Z", "samples": 32}
    if overrides:
        info.update(overrides)

    (model_dir / "model_info.json").write_text(json.dumps(info), encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    for name in artifacts.values():
        (model_dir / name).mkdir(exist_ok=True)
    return model_dir


_FULL_TOML = """
[server]
host = "0.0.0.0"
port = 8000
log_level = "debug"
api_key = "secret-key"
health_rate_limit = 30

[[models]]
id = "emb-1"
kind = "embedding"
tokenizer = "models/emb-1/tokenizer.json"
normalize = false
output_name = "custom_embedding"

[models.artifacts]
128 = "compiled/emb-1/s128.mlmodelc"
512 = "compiled/emb-1/s512.mlmodelc"

[[models]]
id = "rr-1"
kind = "reranker"
tokenizer = "models/rr-1/tokenizer.json"

[models.artifacts]
512 = "compiled/rr-1/s512.mlmodelc"
"""

_MINIMAL_TOML = """
[[models]]
id = "emb-only"
kind = "embedding"
tokenizer = "models/emb-only/tokenizer.json"

[models.artifacts]
256 = "compiled/emb-only/s256.mlmodelc"
"""

_API_KEY_TOML = """
[server]
api_key = "file-key"

[[models]]
id = "emb"
kind = "embedding"
tokenizer = "models/emb/tokenizer.json"

[models.artifacts]
128 = "compiled/emb/s128.mlmodelc"
"""


# --- positive: full/minimal TOML, defaults, buckets, accessors -----------


def test_full_toml_loads_all_fields(tmp_path: Path) -> None:
    """A fully specified TOML file must round-trip every scalar/field value."""
    config_path = _write_toml(tmp_path / "eeane.toml", _FULL_TOML)

    loaded = load_config(explicit_path=config_path, env={})

    assert loaded.source == config_path
    assert loaded.config.server.host == "0.0.0.0"
    assert loaded.config.server.port == 8000
    assert loaded.config.server.log_level == "debug"
    assert loaded.config.server.api_key == "secret-key"
    assert loaded.config.server.health_rate_limit == 30
    assert loaded.api_key_source == "file"

    embedding = loaded.config.embedding_model
    assert embedding.id == "emb-1"
    assert embedding.normalize is False
    assert embedding.output_name == "custom_embedding"
    assert embedding.buckets == (128, 512)
    assert embedding.tokenizer == tmp_path / "models" / "emb-1" / "tokenizer.json"

    reranker = loaded.config.reranker_model
    assert reranker is not None
    assert reranker.id == "rr-1"
    assert reranker.normalize is True
    assert reranker.output_name == "logits"
    assert reranker.buckets == (512,)


def test_minimal_toml_fills_in_defaults(tmp_path: Path) -> None:
    """A minimal TOML file (models-only) must fall back to server/model defaults."""
    config_path = _write_toml(tmp_path / "eeane.toml", _MINIMAL_TOML)

    loaded = load_config(explicit_path=config_path, env={})

    assert loaded.config.server == ServerConfig()
    embedding = loaded.config.embedding_model
    assert embedding.normalize is True
    assert embedding.output_name == "embedding"
    assert embedding.buckets == (256,)
    assert loaded.config.reranker_model is None
    assert loaded.api_key_source is None


def test_buckets_are_sorted_ascending_regardless_of_toml_order() -> None:
    """The ``buckets`` property must return an ascending tuple even if artifacts is unordered."""
    entry = ModelEntry(
        id="m",
        kind="embedding",
        tokenizer=Path("models/m/tokenizer.json"),
        artifacts={1024: Path("a"), 128: Path("b"), 512: Path("c")},
    )

    assert entry.buckets == (128, 512, 1024)


@pytest.mark.parametrize(
    ("kind", "expected_output_name"),
    [("embedding", "embedding"), ("reranker", "logits")],
)
def test_output_name_is_derived_from_kind_when_omitted(
    kind: str, expected_output_name: str
) -> None:
    """output_name must be derived from kind whenever it is not explicitly provided."""
    entry = ModelEntry(
        id="m", kind=kind, tokenizer=Path("m/tokenizer.json"), artifacts={128: Path("a")}
    )

    assert entry.output_name == expected_output_name


def test_embedding_model_accessor_returns_the_single_embedding_entry() -> None:
    """embedding_model must return the configured embedding entry."""
    config = default_config()

    assert config.embedding_model.id == "ruri-v3-310m"


def test_reranker_model_accessor_returns_none_when_absent() -> None:
    """reranker_model must return None for an embedding-only configuration."""
    embedding = ModelEntry(
        id="emb",
        kind="embedding",
        tokenizer=Path("models/emb/tokenizer.json"),
        artifacts={128: Path("a")},
    )
    config = EeaneConfig(models=[embedding])

    assert config.reranker_model is None


def test_relative_paths_resolve_against_config_file_directory(tmp_path: Path) -> None:
    """tokenizer/artifacts relative paths must resolve against the config file's directory."""
    config_path = _write_toml(tmp_path / "eeane.toml", _MINIMAL_TOML)

    loaded = load_config(explicit_path=config_path, env={})

    embedding = loaded.config.embedding_model
    assert embedding.tokenizer == tmp_path / "models" / "emb-only" / "tokenizer.json"
    assert embedding.artifacts[256] == tmp_path / "compiled" / "emb-only" / "s256.mlmodelc"


def test_relative_config_path_still_yields_absolute_model_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CWD-relative --config path must still absolutize the model paths."""
    _write_toml(tmp_path / "eeane.toml", _MINIMAL_TOML)
    monkeypatch.chdir(tmp_path)

    loaded = load_config(explicit_path=Path("eeane.toml"), env={})

    embedding = loaded.config.embedding_model
    assert embedding.tokenizer.is_absolute()
    assert embedding.tokenizer == tmp_path / "models" / "emb-only" / "tokenizer.json"
    assert embedding.artifacts[256] == tmp_path / "compiled" / "emb-only" / "s256.mlmodelc"


def test_absolute_artifact_paths_pass_through_unresolved(tmp_path: Path) -> None:
    """An already-absolute artifact path must not be re-based on the config directory."""
    absolute_artifact = tmp_path / "elsewhere" / "s256.mlmodelc"
    toml_content = f"""
[[models]]
id = "emb-only"
kind = "embedding"
tokenizer = "models/emb-only/tokenizer.json"

[models.artifacts]
256 = "{absolute_artifact.as_posix()}"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    loaded = load_config(explicit_path=config_path, env={})

    assert loaded.config.embedding_model.artifacts[256] == absolute_artifact


# --- negative: unknown keys ------------------------------------------------


def test_unknown_top_level_key_raises_config_error(tmp_path: Path) -> None:
    """An unrecognized top-level key must be rejected with the key name in the message."""
    toml_content = """
bogus_top_level = true

[[models]]
id = "emb"
kind = "embedding"
tokenizer = "models/emb/tokenizer.json"

[models.artifacts]
128 = "compiled/emb/s128.mlmodelc"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    with pytest.raises(ConfigError, match="bogus_top_level"):
        load_config(explicit_path=config_path, env={})


def test_unknown_server_key_raises_config_error(tmp_path: Path) -> None:
    """An unrecognized key in [server] must be rejected with the key name in the message."""
    toml_content = """
[server]
bogus_server_key = true

[[models]]
id = "emb"
kind = "embedding"
tokenizer = "models/emb/tokenizer.json"

[models.artifacts]
128 = "compiled/emb/s128.mlmodelc"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    with pytest.raises(ConfigError, match="bogus_server_key"):
        load_config(explicit_path=config_path, env={})


def test_legacy_model_dir_key_raises_config_error(tmp_path: Path) -> None:
    """The pre-v0.6 'model_dir' key must be rejected (replaced by 'tokenizer')."""
    toml_content = """
[[models]]
id = "emb"
kind = "embedding"
model_dir = "models/emb"

[models.artifacts]
128 = "compiled/emb/s128.mlmodelc"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    with pytest.raises(ConfigError, match="model_dir"):
        load_config(explicit_path=config_path, env={})


def test_missing_tokenizer_key_raises_config_error(tmp_path: Path) -> None:
    """A [[models]] entry without 'tokenizer' must be rejected (it is required)."""
    toml_content = """
[[models]]
id = "emb"
kind = "embedding"

[models.artifacts]
128 = "compiled/emb/s128.mlmodelc"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    with pytest.raises(ConfigError, match="tokenizer"):
        load_config(explicit_path=config_path, env={})


def test_unknown_model_key_raises_config_error(tmp_path: Path) -> None:
    """An unrecognized key in a [[models]] entry must be rejected with the key name."""
    toml_content = """
[[models]]
id = "emb"
kind = "embedding"
tokenizer = "models/emb/tokenizer.json"
bogus_model_key = true

[models.artifacts]
128 = "compiled/emb/s128.mlmodelc"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    with pytest.raises(ConfigError, match="bogus_model_key"):
        load_config(explicit_path=config_path, env={})


# --- negative: [server] field constraints -----------------------------


def test_port_zero_raises_config_error(tmp_path: Path) -> None:
    """port=0 is below the valid 1-65535 range and must be rejected."""
    content = _MINIMAL_TOML.replace("[[models]]", "[server]\nport = 0\n\n[[models]]", 1)
    config_path = _write_toml(tmp_path / "eeane.toml", content)

    with pytest.raises(ConfigError, match="port"):
        load_config(explicit_path=config_path, env={})


def test_port_out_of_range_high_raises_config_error(tmp_path: Path) -> None:
    """port=65536 is above the valid 1-65535 range and must be rejected."""
    content = _MINIMAL_TOML.replace("[[models]]", "[server]\nport = 65536\n\n[[models]]", 1)
    config_path = _write_toml(tmp_path / "eeane.toml", content)

    with pytest.raises(ConfigError, match="port"):
        load_config(explicit_path=config_path, env={})


def test_invalid_log_level_raises_config_error(tmp_path: Path) -> None:
    """A log_level outside the allowed Literal set must be rejected."""
    content = _MINIMAL_TOML.replace(
        "[[models]]", '[server]\nlog_level = "verbose"\n\n[[models]]', 1
    )
    config_path = _write_toml(tmp_path / "eeane.toml", content)

    with pytest.raises(ConfigError, match="log_level"):
        load_config(explicit_path=config_path, env={})


def test_empty_api_key_raises_config_error(tmp_path: Path) -> None:
    """An empty-string api_key must be rejected (omit the key instead)."""
    content = _MINIMAL_TOML.replace("[[models]]", '[server]\napi_key = ""\n\n[[models]]', 1)
    config_path = _write_toml(tmp_path / "eeane.toml", content)

    with pytest.raises(ConfigError, match="api_key"):
        load_config(explicit_path=config_path, env={})


def test_negative_health_rate_limit_raises_config_error(tmp_path: Path) -> None:
    """A negative health_rate_limit must be rejected (0 disables, not negative)."""
    content = _MINIMAL_TOML.replace(
        "[[models]]", "[server]\nhealth_rate_limit = -1\n\n[[models]]", 1
    )
    config_path = _write_toml(tmp_path / "eeane.toml", content)

    with pytest.raises(ConfigError, match="health_rate_limit"):
        load_config(explicit_path=config_path, env={})


# --- negative: model composition (embedding/reranker counts, dup ids) --


def test_zero_embedding_entries_raises_config_error(tmp_path: Path) -> None:
    """A config with no embedding entry must be rejected (exactly one is required)."""
    toml_content = """
[[models]]
id = "rr"
kind = "reranker"
tokenizer = "models/rr/tokenizer.json"

[models.artifacts]
512 = "compiled/rr/s512.mlmodelc"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    with pytest.raises(ConfigError, match="embedding"):
        load_config(explicit_path=config_path, env={})


def test_two_embedding_entries_are_accepted_in_order(tmp_path: Path) -> None:
    """Several embedding models may be served at once, keeping the config-file order."""
    toml_content = """
[[models]]
id = "emb1"
kind = "embedding"
tokenizer = "models/emb1/tokenizer.json"

[models.artifacts]
128 = "compiled/emb1/s128.mlmodelc"

[[models]]
id = "emb2"
kind = "embedding"
tokenizer = "models/emb2/tokenizer.json"

[models.artifacts]
128 = "compiled/emb2/s128.mlmodelc"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    loaded = load_config(explicit_path=config_path, env={})

    assert [entry.id for entry in loaded.config.models_of_kind("embedding")] == ["emb1", "emb2"]
    assert loaded.config.embedding_model.id == "emb1"


def test_two_reranker_entries_are_accepted_in_order(tmp_path: Path) -> None:
    """Several reranker models may be served at once, keeping the config-file order."""
    toml_content = """
[[models]]
id = "emb"
kind = "embedding"
tokenizer = "models/emb/tokenizer.json"

[models.artifacts]
128 = "compiled/emb/s128.mlmodelc"

[[models]]
id = "rr1"
kind = "reranker"
tokenizer = "models/rr1/tokenizer.json"

[models.artifacts]
512 = "compiled/rr1/s512.mlmodelc"

[[models]]
id = "rr2"
kind = "reranker"
tokenizer = "models/rr2/tokenizer.json"

[models.artifacts]
512 = "compiled/rr2/s512.mlmodelc"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    loaded = load_config(explicit_path=config_path, env={})

    assert [entry.id for entry in loaded.config.models_of_kind("reranker")] == ["rr1", "rr2"]
    assert loaded.config.reranker_model is not None
    assert loaded.config.reranker_model.id == "rr1"


def test_duplicate_model_id_raises_config_error(tmp_path: Path) -> None:
    """Two model entries sharing the same id must be rejected."""
    toml_content = """
[[models]]
id = "same-id"
kind = "embedding"
tokenizer = "models/emb/tokenizer.json"

[models.artifacts]
128 = "compiled/emb/s128.mlmodelc"

[[models]]
id = "same-id"
kind = "reranker"
tokenizer = "models/rr/tokenizer.json"

[models.artifacts]
512 = "compiled/rr/s512.mlmodelc"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    with pytest.raises(ConfigError, match="duplicate"):
        load_config(explicit_path=config_path, env={})


# --- negative: artifacts field -----------------------------------------


def test_empty_artifacts_raises_config_error(tmp_path: Path) -> None:
    """An empty artifacts table must be rejected."""
    toml_content = """
[[models]]
id = "emb"
kind = "embedding"
tokenizer = "models/emb/tokenizer.json"
artifacts = {}
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    with pytest.raises(ConfigError, match="empty"):
        load_config(explicit_path=config_path, env={})


def test_non_numeric_artifact_key_raises_config_error(tmp_path: Path) -> None:
    """A non-numeric artifacts key ("abc") must be rejected with the key in the message."""
    toml_content = """
[[models]]
id = "emb"
kind = "embedding"
tokenizer = "models/emb/tokenizer.json"

[models.artifacts]
abc = "compiled/emb/sabc.mlmodelc"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    with pytest.raises(ConfigError, match="abc"):
        load_config(explicit_path=config_path, env={})


def test_non_positive_artifact_key_raises_config_error(tmp_path: Path) -> None:
    """A zero/negative artifacts key must be rejected as a non-positive bucket length."""
    toml_content = """
[[models]]
id = "emb"
kind = "embedding"
tokenizer = "models/emb/tokenizer.json"

[models.artifacts]
"0" = "compiled/emb/s0.mlmodelc"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    with pytest.raises(ConfigError, match="positive"):
        load_config(explicit_path=config_path, env={})


# --- negative: reranker normalize ---------------------------------------


def test_reranker_with_explicit_normalize_raises_config_error(tmp_path: Path) -> None:
    """Explicitly setting normalize on a reranker entry must be rejected."""
    toml_content = """
[[models]]
id = "emb"
kind = "embedding"
tokenizer = "models/emb/tokenizer.json"

[models.artifacts]
128 = "compiled/emb/s128.mlmodelc"

[[models]]
id = "rr"
kind = "reranker"
tokenizer = "models/rr/tokenizer.json"
normalize = true

[models.artifacts]
512 = "compiled/rr/s512.mlmodelc"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    with pytest.raises(ConfigError, match="normalize"):
        load_config(explicit_path=config_path, env={})


# --- negative: TOML syntax / missing explicit path ----------------------


def test_malformed_toml_raises_config_error(tmp_path: Path) -> None:
    """A TOML syntax error must be reported as a ConfigError mentioning the file."""
    config_path = _write_toml(tmp_path / "eeane.toml", "this is not valid toml [[[")

    with pytest.raises(ConfigError, match="eeane.toml"):
        load_config(explicit_path=config_path, env={})


def test_explicit_path_that_does_not_exist_raises_config_error(tmp_path: Path) -> None:
    """An explicit --config path that does not exist must be rejected."""
    missing = tmp_path / "does-not-exist.toml"

    with pytest.raises(ConfigError, match="does-not-exist"):
        load_config(explicit_path=missing, env={})


# --- priority: file < EEANE_API_KEY < CLI overrides ----------------------


def test_api_key_precedence_file_only(tmp_path: Path) -> None:
    """With no env override, the file's api_key is used and reported as its source."""
    config_path = _write_toml(tmp_path / "eeane.toml", _API_KEY_TOML)

    loaded = load_config(explicit_path=config_path, env={})

    assert loaded.config.server.api_key == "file-key"
    assert loaded.api_key_source == "file"


def test_api_key_precedence_env_overrides_file(tmp_path: Path) -> None:
    """EEANE_API_KEY must take precedence over a file-configured api_key."""
    config_path = _write_toml(tmp_path / "eeane.toml", _API_KEY_TOML)

    loaded = load_config(explicit_path=config_path, env={"EEANE_API_KEY": "env-key"})

    assert loaded.config.server.api_key == "env-key"
    assert loaded.api_key_source == "env"


def test_empty_env_api_key_is_ignored(tmp_path: Path) -> None:
    """An empty-string EEANE_API_KEY must be treated as unset, not applied."""
    config_path = _write_toml(tmp_path / "eeane.toml", _API_KEY_TOML)

    loaded = load_config(explicit_path=config_path, env={"EEANE_API_KEY": ""})

    assert loaded.config.server.api_key == "file-key"
    assert loaded.api_key_source == "file"


def test_no_api_key_anywhere_reports_none_source(tmp_path: Path) -> None:
    """With no file/env api_key, api_key_source must be None."""
    config_path = _write_toml(tmp_path / "eeane.toml", _MINIMAL_TOML)

    loaded = load_config(explicit_path=config_path, env={})

    assert loaded.config.server.api_key is None
    assert loaded.api_key_source is None


def test_cli_overrides_take_precedence_over_file_and_env(tmp_path: Path) -> None:
    """host/port/log_level CLI overrides must win over file/env, leaving api_key untouched."""
    config_path = _write_toml(tmp_path / "eeane.toml", _API_KEY_TOML)
    overrides = CliOverrides(host="0.0.0.0", port=9000, log_level="warning")

    loaded = load_config(
        explicit_path=config_path, overrides=overrides, env={"EEANE_API_KEY": "env-key"}
    )

    assert loaded.config.server.host == "0.0.0.0"
    assert loaded.config.server.port == 9000
    assert loaded.config.server.log_level == "warning"
    assert loaded.config.server.api_key == "env-key"
    assert loaded.api_key_source == "env"


def test_cli_override_with_invalid_log_level_raises_config_error(tmp_path: Path) -> None:
    """An invalid CLI-overridden log_level must still surface as a ConfigError."""
    config_path = _write_toml(tmp_path / "eeane.toml", _MINIMAL_TOML)
    overrides = CliOverrides(log_level="verbose")

    with pytest.raises(ConfigError, match="log_level"):
        load_config(explicit_path=config_path, overrides=overrides, env={})


# --- search order: explicit > cwd > home > built-in default --------------


def test_explicit_path_takes_priority_over_cwd_and_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit --config path must win even if ./eeane.toml also exists."""
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    _write_toml(cwd_dir / "eeane.toml", _MINIMAL_TOML)
    monkeypatch.chdir(cwd_dir)

    explicit_dir = tmp_path / "explicit"
    explicit_dir.mkdir()
    explicit_content = _MINIMAL_TOML.replace("emb-only", "emb-explicit")
    explicit_path = _write_toml(explicit_dir / "explicit.toml", explicit_content)

    loaded = load_config(explicit_path=explicit_path, env={})

    assert loaded.source == explicit_path
    assert loaded.config.embedding_model.id == "emb-explicit"


def test_cwd_eeane_toml_is_used_when_no_explicit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """./eeane.toml must be used when no explicit path is given."""
    monkeypatch.chdir(tmp_path)
    _write_toml(tmp_path / "eeane.toml", _MINIMAL_TOML)
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    loaded = load_config(env={})

    assert loaded.source == tmp_path / "eeane.toml"


def test_home_config_is_used_when_no_explicit_path_or_cwd_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """~/.config/eeane/eeane.toml must be used as a fallback when cwd has no config."""
    empty_cwd = tmp_path / "empty-cwd"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)

    fake_home = tmp_path / "fake-home"
    xdg_dir = fake_home / ".config" / "eeane"
    xdg_dir.mkdir(parents=True)
    _write_toml(xdg_dir / "eeane.toml", _MINIMAL_TOML)
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    loaded = load_config(env={})

    assert loaded.source == xdg_dir / "eeane.toml"


def test_no_config_file_anywhere_uses_built_in_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no config file anywhere in the search path, the built-in default is used."""
    empty_cwd = tmp_path / "empty-cwd"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)

    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    loaded = load_config(env={})

    assert loaded.source is None
    assert loaded.config == default_config()


# --- default_config() vs the v0.4 hard-coded values ----------------------

# Literal copies of the constants the v0.4 hard-coded settings module
# held (it is deleted in v0.5), except for the tokenizer paths, which
# v0.6 moved from the HuggingFace model directory to the frozen
# tokenizer.json under models/compiled/. The
# repository root is derived from this test file, independently of
# eeane.config, so the comparison below still checks the values instead
# of restating them.
_V04_REPO_ROOT = Path(__file__).resolve().parent.parent
_V04_COMPILED_ROOT = _V04_REPO_ROOT / "models" / "compiled"
_V04_EMBEDDING_COMPILED = {
    128: _V04_COMPILED_ROOT / "ruri-v3-310m" / "s128_b1_eager_macos13.mlmodelc",
    512: _V04_COMPILED_ROOT / "ruri-v3-310m" / "s512_b1_eager_macos13.mlmodelc",
    1024: _V04_COMPILED_ROOT / "ruri-v3-310m" / "s1024_b1_eager_macos13.mlmodelc",
}
_V04_RERANKER_COMPILED = {
    512: _V04_COMPILED_ROOT / "ruri-v3-reranker-310m" / "s512_b1_eager_macos13.mlmodelc",
    1024: _V04_COMPILED_ROOT / "ruri-v3-reranker-310m" / "s1024_b1_eager_macos13.mlmodelc",
}


def test_default_config_matches_v04_settings_values() -> None:
    """default_config() must reproduce every value of the v0.4 settings module."""
    config = default_config()

    assert config.server.host == "127.0.0.1"
    assert config.server.port == 7997

    embedding = config.embedding_model
    assert embedding.id == "ruri-v3-310m"
    assert embedding.tokenizer == _V04_COMPILED_ROOT / "ruri-v3-310m" / "tokenizer.json"
    assert embedding.artifacts == _V04_EMBEDDING_COMPILED
    assert embedding.buckets == tuple(sorted(_V04_EMBEDDING_COMPILED))
    assert embedding.normalize is True
    assert embedding.output_name == "embedding"

    reranker = config.reranker_model
    assert reranker is not None
    assert reranker.id == "ruri-v3-reranker-310m"
    assert reranker.tokenizer == _V04_COMPILED_ROOT / "ruri-v3-reranker-310m" / "tokenizer.json"
    assert reranker.artifacts == _V04_RERANKER_COMPILED
    assert reranker.buckets == tuple(sorted(_V04_RERANKER_COMPILED))
    assert reranker.output_name == "logits"


# --- kind-aware accessors -------------------------------------------------


def _explicit_entry(model_id: str, kind: str) -> ModelEntry:
    """Build a minimal, fully explicit entry for accessor tests."""
    return ModelEntry(
        id=model_id,
        kind=kind,
        tokenizer=Path(f"models/{model_id}/tokenizer.json"),
        artifacts={128: Path(f"compiled/{model_id}/s128.mlmodelc")},
    )


def test_models_of_kind_returns_every_entry_in_config_order() -> None:
    """models_of_kind must list all entries of one kind, keeping their declared order."""
    config = EeaneConfig(
        models=[
            _explicit_entry("emb1", "embedding"),
            _explicit_entry("rr1", "reranker"),
            _explicit_entry("emb2", "embedding"),
        ]
    )

    assert [entry.id for entry in config.models_of_kind("embedding")] == ["emb1", "emb2"]
    assert [entry.id for entry in config.models_of_kind("reranker")] == ["rr1"]


def test_default_model_is_the_first_listed_entry_of_its_kind() -> None:
    """The default model of a kind must be the first one listed, not e.g. the last."""
    config = EeaneConfig(
        models=[
            _explicit_entry("rr1", "reranker"),
            _explicit_entry("emb1", "embedding"),
            _explicit_entry("emb2", "embedding"),
            _explicit_entry("rr2", "reranker"),
        ]
    )

    default_embedding = config.default_model("embedding")
    default_reranker = config.default_model("reranker")

    assert default_embedding is not None and default_embedding.id == "emb1"
    assert default_reranker is not None and default_reranker.id == "rr1"
    # The compatibility properties must agree with the new accessors.
    assert config.embedding_model is default_embedding
    assert config.reranker_model is default_reranker


def test_default_model_returns_none_for_an_unconfigured_kind() -> None:
    """A kind with no entries must yield None and an empty list, not an error."""
    config = EeaneConfig(models=[_explicit_entry("emb", "embedding")])

    assert config.default_model("reranker") is None
    assert config.models_of_kind("reranker") == []
    assert config.reranker_model is None


def test_model_by_id_finds_entries_of_either_kind() -> None:
    """model_by_id must look across kinds and return None for an unknown id."""
    config = EeaneConfig(
        models=[_explicit_entry("emb", "embedding"), _explicit_entry("rr", "reranker")]
    )

    found_embedding = config.model_by_id("emb")
    found_reranker = config.model_by_id("rr")

    assert found_embedding is not None and found_embedding.kind == "embedding"
    assert found_reranker is not None and found_reranker.kind == "reranker"
    assert config.model_by_id("absent") is None
    assert config.model_by_id("") is None


def test_zero_embedding_entries_still_rejected_for_a_programmatic_config() -> None:
    """The composition rule (at least one embedding) must hold for direct construction too."""
    with pytest.raises(ValidationError, match="embedding"):
        EeaneConfig(models=[_explicit_entry("rr", "reranker")])


# --- cache auto-resolution ------------------------------------------------


_CACHE_ROOT_TOML = """
[server]
cache_root = "cache"

[[models]]
id = "org/emb"
"""


def test_id_only_entry_is_completed_from_the_cache(tmp_path: Path) -> None:
    """An entry holding just an id must gain kind/tokenizer/artifacts/output_name."""
    model_dir = _write_cached_model(tmp_path / "cache", "org/emb", buckets=(128, 512))
    config_path = _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML)

    loaded = load_config(explicit_path=config_path, env={})

    entry = loaded.config.embedding_model
    assert entry.id == "org/emb"
    assert entry.kind == "embedding"
    assert entry.tokenizer == model_dir / "tokenizer.json"
    assert entry.tokenizer.is_absolute()
    assert entry.artifacts == {
        128: model_dir / "s128_b1_eager_macos13.mlmodelc",
        512: model_dir / "s512_b1_eager_macos13.mlmodelc",
    }
    assert entry.output_name == "embedding"
    assert entry.embedding_dim == 768
    assert entry.excluded_buckets == ()
    assert entry.normalize is True


def test_id_only_reranker_entry_is_completed_from_the_cache(tmp_path: Path) -> None:
    """A cached reranker must resolve to kind='reranker', its logits output, and no dim."""
    _write_cached_model(tmp_path / "cache", "org/emb")
    _write_cached_model(tmp_path / "cache", "rr", kind="reranker", buckets=(512, 1024))
    config_path = _write_toml(
        tmp_path / "eeane.toml", _CACHE_ROOT_TOML + '\n[[models]]\nid = "rr"\n'
    )

    loaded = load_config(explicit_path=config_path, env={})

    reranker = loaded.config.reranker_model
    assert reranker is not None
    assert reranker.kind == "reranker"
    assert reranker.output_name == "logits"
    assert reranker.buckets == (512, 1024)
    assert reranker.embedding_dim is None


def test_recommended_buckets_limit_the_loaded_artifacts(tmp_path: Path) -> None:
    """Only the recommended buckets are loaded; the rest are recorded as excluded."""
    _write_cached_model(
        tmp_path / "cache",
        "org/emb",
        buckets=(128, 512, 1024),
        recommended_buckets=(512,),
    )
    config_path = _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML)

    loaded = load_config(explicit_path=config_path, env={})

    entry = loaded.config.embedding_model
    assert entry.buckets == (512,)
    assert entry.excluded_buckets == (128, 1024)


def test_format_version_1_record_degrades_gracefully(tmp_path: Path) -> None:
    """A v1 record has no embedding_dim/recommended_buckets: load every bucket, dim unknown."""
    _write_cached_model(tmp_path / "cache", "org/emb", format_version=1, buckets=(128, 512, 1024))
    config_path = _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML)

    loaded = load_config(explicit_path=config_path, env={})

    entry = loaded.config.embedding_model
    assert entry.buckets == (128, 512, 1024)
    assert entry.excluded_buckets == ()
    assert entry.embedding_dim is None
    assert entry.kind == "embedding"


def test_unknown_record_keys_are_ignored(tmp_path: Path) -> None:
    """Provenance blocks and future keys in the record must not leak into the entry."""
    _write_cached_model(
        tmp_path / "cache",
        "org/emb",
        overrides={"something_new": {"nested": True}},
    )
    config_path = _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML)

    loaded = load_config(explicit_path=config_path, env={})

    entry = loaded.config.embedding_model
    assert not hasattr(entry, "calibration")
    assert not hasattr(entry, "something_new")


def test_declared_kind_conflicting_with_the_cache_raises_config_error(tmp_path: Path) -> None:
    """A kind stated in the config that the cache contradicts must be a hard error."""
    _write_cached_model(tmp_path / "cache", "org/emb", kind="embedding")
    config_path = _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML + 'kind = "reranker"\n')

    with pytest.raises(ConfigError, match="kind"):
        load_config(explicit_path=config_path, env={})


def test_missing_cache_entry_names_the_directory_and_the_compile_command(
    tmp_path: Path,
) -> None:
    """A model that was never compiled must fail with the searched path and the fix."""
    config_path = _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML)
    expected_dir = tmp_path / "cache" / "compiled" / "org--emb"

    with pytest.raises(ConfigError) as excinfo:
        load_config(explicit_path=config_path, env={})

    message = str(excinfo.value)
    assert "eeane compile org/emb" in message
    assert str(expected_dir) in message
    assert "tokenizer" in message


def test_corrupt_model_info_reports_the_compile_command(tmp_path: Path) -> None:
    """A truncated record must be reported like a missing one, not silently ignored."""
    model_dir = _write_cached_model(tmp_path / "cache", "org/emb")
    (model_dir / "model_info.json").write_text('{"format_version": 2,', encoding="utf-8")
    config_path = _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML)

    with pytest.raises(ConfigError, match=re.escape("eeane compile org/emb")):
        load_config(explicit_path=config_path, env={})


def test_newer_record_format_version_is_rejected(tmp_path: Path) -> None:
    """A record from a newer eeANE must be refused rather than half-understood."""
    _write_cached_model(tmp_path / "cache", "org/emb", overrides={"format_version": 99})
    config_path = _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML)

    with pytest.raises(ConfigError, match="format_version"):
        load_config(explicit_path=config_path, env={})


def test_record_without_a_usable_format_version_is_rejected(tmp_path: Path) -> None:
    """A record whose format_version is not a positive integer must be refused."""
    _write_cached_model(tmp_path / "cache", "org/emb", overrides={"format_version": "one"})
    config_path = _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML)

    with pytest.raises(ConfigError, match="format_version"):
        load_config(explicit_path=config_path, env={})


def test_record_with_an_unknown_kind_is_rejected(tmp_path: Path) -> None:
    """A record naming a kind this release cannot serve must be refused."""
    _write_cached_model(tmp_path / "cache", "org/emb", overrides={"kind": "classifier"})
    config_path = _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML)

    with pytest.raises(ConfigError, match="kind"):
        load_config(explicit_path=config_path, env={})


def test_recommendation_matching_no_compiled_bucket_is_rejected(tmp_path: Path) -> None:
    """A recommendation that leaves nothing to load must not yield an empty artifact set."""
    _write_cached_model(tmp_path / "cache", "org/emb", buckets=(128,), recommended_buckets=(2048,))
    config_path = _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML)

    with pytest.raises(ConfigError, match="recommended"):
        load_config(explicit_path=config_path, env={})


def test_record_without_artifacts_is_rejected(tmp_path: Path) -> None:
    """A record listing no compiled artifact must be refused with the compile hint."""
    _write_cached_model(tmp_path / "cache", "org/emb", overrides={"artifacts": {}})
    config_path = _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML)

    with pytest.raises(ConfigError, match="artifact"):
        load_config(explicit_path=config_path, env={})


def test_record_with_a_non_positive_bucket_is_rejected(tmp_path: Path) -> None:
    """A bucket length of zero in the record must be refused like one in the config file."""
    _write_cached_model(
        tmp_path / "cache", "org/emb", overrides={"artifacts": {"0": "s0.mlmodelc"}}
    )
    config_path = _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML)

    with pytest.raises(ConfigError, match="positive"):
        load_config(explicit_path=config_path, env={})


@pytest.mark.parametrize(
    ("field", "value"),
    [("tokenizer", "/etc/passwd"), ("artifacts", {"128": "../../elsewhere.mlmodelc"})],
)
def test_record_pointing_outside_its_directory_is_rejected(
    tmp_path: Path, field: str, value: object
) -> None:
    """A record must only name files inside its own cache directory."""
    _write_cached_model(tmp_path / "cache", "org/emb", overrides={field: value})
    config_path = _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML)

    with pytest.raises(ConfigError, match=field):
        load_config(explicit_path=config_path, env={})


def test_resolved_reranker_still_rejects_an_explicit_normalize(tmp_path: Path) -> None:
    """Field validation must apply to cache-resolved entries just as to explicit ones."""
    _write_cached_model(tmp_path / "cache", "org/emb")
    _write_cached_model(tmp_path / "cache", "rr", kind="reranker", buckets=(512,))
    config_path = _write_toml(
        tmp_path / "eeane.toml",
        _CACHE_ROOT_TOML + '\n[[models]]\nid = "rr"\nnormalize = true\n',
    )

    with pytest.raises(ConfigError, match="normalize"):
        load_config(explicit_path=config_path, env={})


def test_explicit_tokenizer_and_artifacts_skip_cache_resolution(tmp_path: Path) -> None:
    """A v0.6-style entry must load unchanged even when no cache exists at all."""
    toml_content = """
[server]
cache_root = "no-such-cache"

[[models]]
id = "emb"
kind = "embedding"
tokenizer = "models/emb/tokenizer.json"

[models.artifacts]
128 = "compiled/emb/s128.mlmodelc"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    loaded = load_config(explicit_path=config_path, env={})

    entry = loaded.config.embedding_model
    assert entry.tokenizer == tmp_path / "models" / "emb" / "tokenizer.json"
    assert entry.artifacts == {128: tmp_path / "compiled" / "emb" / "s128.mlmodelc"}
    assert entry.embedding_dim is None
    assert entry.excluded_buckets == ()


def test_entry_with_only_a_tokenizer_gains_its_artifacts_from_the_cache(tmp_path: Path) -> None:
    """A half-specified entry must keep what it states and fill in only what it omits."""
    model_dir = _write_cached_model(tmp_path / "cache", "emb", buckets=(128, 512))
    toml_content = """
[server]
cache_root = "cache"

[[models]]
id = "emb"
tokenizer = "models/emb/tokenizer.json"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    loaded = load_config(explicit_path=config_path, env={})

    entry = loaded.config.embedding_model
    assert entry.tokenizer == tmp_path / "models" / "emb" / "tokenizer.json"
    assert entry.kind == "embedding"
    assert entry.artifacts == {
        128: model_dir / "s128_b1_eager_macos13.mlmodelc",
        512: model_dir / "s512_b1_eager_macos13.mlmodelc",
    }


def test_explicitly_configured_fields_win_over_the_cache(tmp_path: Path) -> None:
    """Values stated in the config file must survive resolution untouched."""
    _write_cached_model(tmp_path / "cache", "org/emb", embedding_dim=768)
    config_path = _write_toml(
        tmp_path / "eeane.toml",
        _CACHE_ROOT_TOML + 'output_name = "custom"\nembedding_dim = 64\nnormalize = false\n',
    )

    loaded = load_config(explicit_path=config_path, env={})

    entry = loaded.config.embedding_model
    assert entry.output_name == "custom"
    assert entry.embedding_dim == 64
    assert entry.normalize is False


def test_cache_root_is_resolved_against_the_config_file_directory(tmp_path: Path) -> None:
    """A relative cache_root must be anchored at the config file, like the other paths."""
    _write_cached_model(tmp_path / "cache", "org/emb")
    config_path = _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML)

    loaded = load_config(explicit_path=config_path, env={})

    assert loaded.config.server.cache_root == tmp_path / "cache"
    assert loaded.config.server.cache_root.is_absolute()


def test_relative_config_path_still_yields_an_absolute_cache_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CWD-relative --config path must still absolutize cache_root and the resolved paths."""
    _write_cached_model(tmp_path / "cache", "org/emb")
    _write_toml(tmp_path / "eeane.toml", _CACHE_ROOT_TOML)
    monkeypatch.chdir(tmp_path)

    loaded = load_config(explicit_path=Path("eeane.toml"), env={})

    assert loaded.config.server.cache_root == tmp_path / "cache"
    assert loaded.config.embedding_model.tokenizer.is_absolute()


def test_absolute_cache_root_passes_through(tmp_path: Path) -> None:
    """An absolute cache_root must not be re-based on the config directory."""
    cache_root = tmp_path / "elsewhere" / "cache"
    _write_cached_model(cache_root, "org/emb")
    toml_content = f"""
[server]
cache_root = "{cache_root.as_posix()}"

[[models]]
id = "org/emb"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    loaded = load_config(explicit_path=config_path, env={})

    assert loaded.config.server.cache_root == cache_root
    assert loaded.config.embedding_model.tokenizer.parent.parent == cache_root / "compiled"


def test_xdg_cache_home_locates_the_cache_when_cache_root_is_unset(tmp_path: Path) -> None:
    """Without cache_root, the cache is looked up below XDG_CACHE_HOME."""
    model_dir = _write_cached_model(tmp_path / "xdg" / "eeane", "emb")
    config_path = _write_toml(tmp_path / "eeane.toml", '[[models]]\nid = "emb"\n')

    loaded = load_config(explicit_path=config_path, env={"XDG_CACHE_HOME": str(tmp_path / "xdg")})

    assert loaded.config.server.cache_root is None
    assert loaded.config.embedding_model.tokenizer == model_dir / "tokenizer.json"


def test_explicit_entry_without_kind_raises_config_error(tmp_path: Path) -> None:
    """Stating tokenizer and artifacts disables resolution, so kind becomes mandatory."""
    toml_content = """
[[models]]
id = "emb"
tokenizer = "models/emb/tokenizer.json"

[models.artifacts]
128 = "compiled/emb/s128.mlmodelc"
"""
    config_path = _write_toml(tmp_path / "eeane.toml", toml_content)

    with pytest.raises(ConfigError, match="kind"):
        load_config(explicit_path=config_path, env={})
