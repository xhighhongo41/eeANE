"""Tests for eeane.config (v0.5 T2, see 開発資料/v0.5実装計画.md §4.1-§4.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_two_embedding_entries_raises_config_error_mentioning_v07(tmp_path: Path) -> None:
    """Two embedding entries must be rejected, with the message pointing to v0.7."""
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

    with pytest.raises(ConfigError, match="v0.7"):
        load_config(explicit_path=config_path, env={})


def test_two_reranker_entries_raises_config_error_mentioning_v07(tmp_path: Path) -> None:
    """Two reranker entries must be rejected, with the message pointing to v0.7."""
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

    with pytest.raises(ConfigError, match="v0.7"):
        load_config(explicit_path=config_path, env={})


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
# tokenizer.json under models/compiled/ (v0.6実装計画.md §4.6). The
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
