"""Tests for the compile pipeline.

Covers ``eeane.compiler.pipeline`` (the driver) together with
``eeane.compiler.artifacts`` (the layout/naming/record decisions it
drives), in two layers:

* Unit tests for the pure decisions (cache naming, output-root
  resolution, bucket defaults, variant naming, config snippet,
  calibration aggregation, idempotent-skip) -- these run anywhere,
  including CI.
* One end-to-end run over a *synthetic* randomly initialised ModernBERT
  (trace -> convert -> ``xcrun coremlcompiler`` -> metadata -> snippet).
  It is skipped unless the local development model directories are present
  (the established local-only marker) and ``xcrun`` is available, so CI
  stays green and fast. Converting the real, deployed-sized models happens
  outside this test suite.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import time
import tomllib
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
from transformers.models.modernbert import modeling_modernbert

from eeane import __version__, cli
from eeane.compiler import artifacts, pipeline
from eeane.compiler.backends import modernbert as mb
from eeane.config import ModelEntry, load_config

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The end-to-end test needs the Core ML compiler and is a local-only test
# by policy; the presence of the development model directory is the marker
# the other real-artifact modules use to detect a local machine.
_LOCAL_MACHINE = (_REPO_ROOT / "models" / "ruri-v3-310m" / "config.json").exists()
_E2E_AVAILABLE = _LOCAL_MACHINE and shutil.which("xcrun") is not None

# Sequence length of the end-to-end run: small enough to keep the whole
# conversion within a few seconds.
E2E_SEQ_LEN = 32
E2E_STEM = f"s{E2E_SEQ_LEN}_b1_eager_macos13"


@pytest.fixture(autouse=True, scope="module")
def _restore_transformers_patches() -> Iterator[None]:
    """Undo the global ModernBert monkeypatches after this module's tests."""
    original_rotate_half = modeling_modernbert.rotate_half
    original_forward = modeling_modernbert.ModernBertAttention.forward
    yield
    modeling_modernbert.rotate_half = original_rotate_half
    modeling_modernbert.ModernBertAttention.forward = original_forward


def _versions() -> dict[str, str]:
    """Build a version block shaped like the real one, without importing torch info."""
    return {key: f"{key}-1.0" for key in artifacts.SKIP_VERSION_KEYS}


def _write_variant(
    directory: Path,
    versions: dict[str, str],
    *,
    selfcheck_status: str = "skipped",
    pooling: str | None = None,
) -> tuple[Path, Path]:
    """Create a fake compiled variant (``.mlmodelc`` + metadata) on disk.

    Args:
        directory: Directory to create the variant in.
        versions: Version block to record in the metadata.
        selfcheck_status: Value recorded under ``selfcheck.status``.
        pooling: Value recorded under ``variant.pooling``, or ``None`` to
            omit the ``variant`` block entirely -- an old-format record
            that never recorded one.

    Returns:
        Tuple of the ``.mlmodelc`` and metadata paths.
    """
    mlmodelc_path = directory / f"{E2E_STEM}.mlmodelc"
    mlmodelc_path.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / f"{E2E_STEM}.json"
    payload: dict[str, Any] = {"versions": versions, "selfcheck": {"status": selfcheck_status}}
    if pooling is not None:
        payload["variant"] = {"pooling": pooling}
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    return mlmodelc_path, metadata_path


# --- cache naming ------------------------------------------------------------


def test_model_cache_name_normalizes_a_hub_id() -> None:
    """A Hub id must become the HF-style ``org--name`` directory name."""
    name = artifacts.model_cache_name("cl-nagoya/ruri-v3-310m", Path("/hf/cache/snapshots/abc"))

    assert name == "cl-nagoya--ruri-v3-310m"


def test_model_cache_name_uses_the_directory_name_for_a_local_source(tmp_path: Path) -> None:
    """A local source must be named after its resolved directory name."""
    model_dir = tmp_path / "ruri-v3-310m"
    model_dir.mkdir()

    assert artifacts.model_cache_name(f"{model_dir}/", model_dir) == "ruri-v3-310m"


def test_model_identifier_keeps_the_hub_id_but_uses_the_name_locally(tmp_path: Path) -> None:
    """The snippet id is the Hub id for a download and the directory name locally."""
    model_dir = tmp_path / "ruri-v3-310m"
    model_dir.mkdir()

    assert artifacts.model_identifier("cl-nagoya/ruri-v3-310m", Path("/hf/x")) == (
        "cl-nagoya/ruri-v3-310m"
    )
    assert artifacts.model_identifier(str(model_dir), model_dir) == "ruri-v3-310m"


def test_model_cache_name_rejects_a_nameless_directory() -> None:
    """A filesystem root has no usable cache name and must be rejected."""
    with pytest.raises(artifacts.CompileError, match="name"):
        artifacts.model_cache_name("/", Path("/"))


# --- output root resolution --------------------------------------------------


def test_resolve_out_root_defaults_to_home_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without XDG_CACHE_HOME the default must be ``~/.cache/eeane``."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert artifacts.resolve_out_root(None, env={}) == (tmp_path / ".cache" / "eeane").resolve()


def test_resolve_out_root_respects_xdg_cache_home(tmp_path: Path) -> None:
    """XDG_CACHE_HOME must move the cache root."""
    root = artifacts.resolve_out_root(None, env={"XDG_CACHE_HOME": str(tmp_path)})

    assert root == (tmp_path / "eeane").resolve()


def test_resolve_out_root_ignores_a_relative_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A relative XDG_CACHE_HOME is invalid per spec and must fall back to the home cache."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    root = artifacts.resolve_out_root(None, env={"XDG_CACHE_HOME": "relative/cache"})

    assert root == (tmp_path / ".cache" / "eeane").resolve()


def test_resolve_out_root_expands_an_explicit_out_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit --out-dir must be expanded and absolutized."""
    monkeypatch.setenv("HOME", str(tmp_path))

    assert artifacts.resolve_out_root(Path("~/artifacts")) == (tmp_path / "artifacts").resolve()


# --- bucket defaults ---------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("embedding", [128, 512, 1024]), ("reranker", [512, 1024])],
)
def test_resolve_buckets_defaults_per_kind(kind: str, expected: list[int]) -> None:
    """Omitting --buckets must reproduce the deployed v0.5 bucket configuration."""
    assert artifacts.resolve_buckets(None, kind) == expected


def test_resolve_buckets_sorts_and_deduplicates() -> None:
    """Explicit buckets must be compiled once each, in ascending order."""
    assert artifacts.resolve_buckets([512, 128, 512], "embedding") == [128, 512]


@pytest.mark.parametrize("buckets", [[], [0], [-1], [128, 0]])
def test_resolve_buckets_rejects_empty_or_non_positive(buckets: list[int]) -> None:
    """An empty or non-positive bucket list must be a clean compile error."""
    with pytest.raises(artifacts.CompileError, match="bucket"):
        artifacts.resolve_buckets(buckets, "embedding")


def test_resolve_buckets_rejects_an_unknown_kind() -> None:
    """A kind with no default bucket set must be reported, not silently emptied."""
    with pytest.raises(artifacts.CompileError, match="kind"):
        artifacts.resolve_buckets(None, "classifier")


# --- variant naming ----------------------------------------------------------


def test_variant_stem_matches_the_poc_naming() -> None:
    """The default variant name must stay byte-identical to the PoC artifacts."""
    assert artifacts.variant_stem(512, 1, "eager", "macos13", "fp16") == "s512_b1_eager_macos13"


def test_variant_stem_appends_fp32_only_for_fp32() -> None:
    """fp32 must get its own suffix so it cannot overwrite the fp16 baseline."""
    assert artifacts.variant_stem(128, 4, "sdpa", "macos15", "fp32") == "s128_b4_sdpa_macos15_fp32"


# --- config snippet ----------------------------------------------------------


def _parse_snippet(snippet: str) -> dict[str, Any]:
    """Parse a generated snippet's active (non-comment) ``[[models]]`` entry."""
    parsed = tomllib.loads(snippet)
    assert len(parsed["models"]) == 1
    return parsed["models"][0]


def _commented_value(snippet: str, key: str) -> Any:
    """TOML-parse the value of a commented-out ``# key = ...`` line.

    The minimal snippet activates only ``id``/``normalize``; every other
    field is written as a ready-to-uncomment comment (a TOML parser skips
    it). This is how a test checks what that comment *would* set without
    actually enabling it.
    """
    prefix = f"# {key} = "
    for line in snippet.splitlines():
        if line.startswith(prefix):
            return tomllib.loads(f"{key} = {line[len(prefix) :]}")[key]
    raise AssertionError(f"no commented '{key} = ...' line in snippet:\n{snippet}")


def _commented_buckets(snippet: str) -> list[str]:
    """Return the bucket keys named in a snippet's commented artifacts block."""
    return [
        line[2:].split(" = ", 1)[0]
        for line in snippet.splitlines()
        if line.startswith("# ") and line[2:].split(" = ", 1)[0].isdigit()
    ]


def _commented_table(snippet: str, header: str) -> dict[str, Any]:
    """TOML-parse the entries of one commented-out table of a snippet.

    The explicit form is written as comments, so a table is a
    ``# [<header>]`` line followed by its own ``# <key> = <value>`` lines,
    up to the next table header. This is how a test tells the entries of
    one table from another's when both are keyed by bucket.
    """
    entries: dict[str, Any] = {}
    collecting = False
    for line in snippet.splitlines():
        if line.startswith("# ["):
            collecting = line == f"# [{header}]"
            continue
        if not collecting or not line.startswith("# ") or " = " not in line:
            continue
        key, _, value = line[2:].partition(" = ")
        entries[key] = tomllib.loads(f"value = {value}")["value"]
    return entries


def test_config_snippet_minimal_form_only_sets_id_and_normalize(tmp_path: Path) -> None:
    """The active (uncommented) part of an embedding snippet must be id + normalize."""
    snippet = artifacts.build_config_snippet(
        model_id="cl-nagoya/ruri-v3-310m",
        kind="embedding",
        tokenizer_path=tmp_path / "tokenizer.json",
        artifacts={512: tmp_path / "s512.mlmodelc", 128: tmp_path / "s128.mlmodelc"},
    )

    assert _parse_snippet(snippet) == {"id": "cl-nagoya/ruri-v3-310m", "normalize": True}


def test_config_snippet_comments_name_every_bucket_with_absolute_paths(tmp_path: Path) -> None:
    """The commented-out explicit form must still name every bucket, absolutely."""
    compiled_artifacts = {512: tmp_path / "s512.mlmodelc", 128: tmp_path / "s128.mlmodelc"}

    snippet = artifacts.build_config_snippet(
        model_id="cl-nagoya/ruri-v3-310m",
        kind="embedding",
        tokenizer_path=tmp_path / "tokenizer.json",
        artifacts=compiled_artifacts,
    )

    assert _commented_value(snippet, "kind") == "embedding"
    assert Path(_commented_value(snippet, "tokenizer")).is_absolute()
    for bucket, path in compiled_artifacts.items():
        commented_path = Path(_commented_value(snippet, str(bucket)))
        assert commented_path.is_absolute()
        assert commented_path == path.resolve()


def test_config_snippet_is_accepted_by_the_config_schema_once_pinned(tmp_path: Path) -> None:
    """Uncommenting the minimal snippet's hints must build a valid ModelEntry."""
    snippet = artifacts.build_config_snippet(
        model_id="ruri-v3-310m",
        kind="embedding",
        tokenizer_path=tmp_path / "tokenizer.json",
        artifacts={128: tmp_path / "s128.mlmodelc"},
    )

    entry = ModelEntry(
        **_parse_snippet(snippet),
        kind=_commented_value(snippet, "kind"),
        tokenizer=_commented_value(snippet, "tokenizer"),
        artifacts={128: _commented_value(snippet, "128")},
    )

    assert entry.buckets == (128,)
    assert entry.output_name == "embedding"


def test_config_snippet_omits_normalize_for_a_reranker(tmp_path: Path) -> None:
    """`normalize` is embedding-only; a reranker's active entry must not carry it."""
    snippet = artifacts.build_config_snippet(
        model_id="ruri-v3-reranker-310m",
        kind="reranker",
        tokenizer_path=tmp_path / "tokenizer.json",
        artifacts={512: tmp_path / "s512.mlmodelc"},
    )

    entry = _parse_snippet(snippet)
    assert entry == {"id": "ruri-v3-reranker-310m"}
    assert _commented_value(snippet, "kind") == "reranker"


def test_config_snippet_comments_the_batched_artifacts_of_an_embedding_model(
    tmp_path: Path,
) -> None:
    """Batched artifacts must be offered as their own commented table, absolutely."""
    snippet = artifacts.build_config_snippet(
        model_id="emb",
        kind="embedding",
        tokenizer_path=tmp_path / "tokenizer.json",
        artifacts={128: tmp_path / "s128.mlmodelc", 512: tmp_path / "s512.mlmodelc"},
        batch_artifacts={128: tmp_path / "s128_b2.mlmodelc"},
    )

    # The active entry stays minimal: the cache resolves the rest.
    assert _parse_snippet(snippet) == {"id": "emb", "normalize": True}
    assert sorted(_commented_table(snippet, "models.artifacts")) == ["128", "512"]
    assert _commented_table(snippet, "models.batch_artifacts") == {
        "128": str((tmp_path / "s128_b2.mlmodelc").resolve())
    }


def test_config_snippet_batched_artifacts_are_accepted_by_the_config_schema(
    tmp_path: Path,
) -> None:
    """Uncommenting both tables must build a valid ModelEntry, not a rejected one."""
    snippet = artifacts.build_config_snippet(
        model_id="emb",
        kind="embedding",
        tokenizer_path=tmp_path / "tokenizer.json",
        artifacts={128: tmp_path / "s128.mlmodelc", 512: tmp_path / "s512.mlmodelc"},
        batch_artifacts={128: tmp_path / "s128_b2.mlmodelc"},
    )

    entry = ModelEntry(
        **_parse_snippet(snippet),
        kind=_commented_value(snippet, "kind"),
        tokenizer=_commented_value(snippet, "tokenizer"),
        artifacts=_commented_table(snippet, "models.artifacts"),
        batch_artifacts=_commented_table(snippet, "models.batch_artifacts"),
    )

    assert entry.buckets == (128, 512)
    assert entry.batch_artifacts == {128: (tmp_path / "s128_b2.mlmodelc").resolve()}


def test_config_snippet_omits_the_batched_artifacts_for_a_reranker(tmp_path: Path) -> None:
    """A reranker is served one input at a time, so its snippet must offer no such table."""
    snippet = artifacts.build_config_snippet(
        model_id="rr",
        kind="reranker",
        tokenizer_path=tmp_path / "tokenizer.json",
        artifacts={512: tmp_path / "s512.mlmodelc"},
        batch_artifacts={512: tmp_path / "s512_b2.mlmodelc"},
    )

    assert "batch_artifacts" not in snippet


def test_config_snippet_without_batched_artifacts_offers_no_such_table(tmp_path: Path) -> None:
    """A model compiled for one batch size only must produce the snippet it always did."""
    snippet = artifacts.build_config_snippet(
        model_id="emb",
        kind="embedding",
        tokenizer_path=tmp_path / "tokenizer.json",
        artifacts={128: tmp_path / "s128.mlmodelc"},
    )

    assert "batch_artifacts" not in snippet


def test_config_snippet_absolutizes_relative_paths() -> None:
    """Relative paths must never reach the snippet: configs are read from elsewhere."""
    snippet = artifacts.build_config_snippet(
        model_id="local",
        kind="embedding",
        tokenizer_path=Path("cache/tokenizer.json"),
        artifacts={128: Path("cache/s128.mlmodelc")},
    )

    assert Path(_commented_value(snippet, "tokenizer")).is_absolute()
    assert Path(_commented_value(snippet, "128")).is_absolute()


def test_config_snippet_escapes_special_characters_in_paths(tmp_path: Path) -> None:
    """A quote or backslash in a path must survive the TOML round trip."""
    weird = tmp_path / "we'i\"rd\\dir" / "tokenizer.json"

    snippet = artifacts.build_config_snippet(
        model_id='odd "id"',
        kind="embedding",
        tokenizer_path=weird,
        artifacts={128: tmp_path / "a b" / "s128.mlmodelc"},
    )

    assert _parse_snippet(snippet)["id"] == 'odd "id"'
    assert _commented_value(snippet, "tokenizer") == str(weird.resolve())
    assert _commented_value(snippet, "128") == str((tmp_path / "a b" / "s128.mlmodelc").resolve())


def test_config_snippet_adds_a_cache_root_hint_for_a_non_default_out_dir(tmp_path: Path) -> None:
    """A non-default cache root must be echoed as a commented [server] reminder."""
    non_default_root = tmp_path / "elsewhere"

    snippet = artifacts.build_config_snippet(
        model_id="local",
        kind="embedding",
        tokenizer_path=tmp_path / "tokenizer.json",
        artifacts={128: tmp_path / "s128.mlmodelc"},
        cache_root_hint=non_default_root,
    )

    lines = snippet.splitlines()
    assert "# [server]" in lines
    assert f"# cache_root = {artifacts._toml_string(str(non_default_root.resolve()))}" in lines
    # The reminder must stay a comment: the server config is the user's own.
    assert "server" not in tomllib.loads(snippet)


def test_config_snippet_omits_the_cache_root_hint_by_default(tmp_path: Path) -> None:
    """The default cache root needs no reminder: eeANE resolves it on its own."""
    snippet = artifacts.build_config_snippet(
        model_id="local",
        kind="embedding",
        tokenizer_path=tmp_path / "tokenizer.json",
        artifacts={128: tmp_path / "s128.mlmodelc"},
    )

    assert "cache_root" not in snippet
    assert "[server]" not in snippet


def test_minimal_snippet_resolves_through_the_real_cache_and_config_loader(
    tmp_path: Path,
) -> None:
    """The minimal snippet plus a real model_info.json must resolve via eeane.config.

    ``eeane.config``'s cache-based resolution is a prerequisite of this
    task; a real round trip through it is the strongest proof that the
    minimal snippet and the model_info.json schema (format_version 2)
    actually agree with each other.
    """
    cache_root = tmp_path / "eeane"
    model_root = cache_root / artifacts.CACHE_SUBDIR / "ruri-v3-310m"
    model_root.mkdir(parents=True)
    tokenizer_path = model_root / artifacts.TOKENIZER_FILENAME
    tokenizer_path.write_text("{}", encoding="utf-8")
    mlmodelc_path = model_root / "s128_b1_eager_macos13.mlmodelc"
    mlmodelc_path.mkdir()
    (model_root / artifacts.MODEL_INFO_FILENAME).write_text(
        json.dumps(
            {
                "format_version": 2,
                "id": "ruri-v3-310m",
                "kind": "embedding",
                "output_name": "embedding",
                "buckets": [128],
                "tokenizer": artifacts.TOKENIZER_FILENAME,
                "artifacts": {"128": mlmodelc_path.name},
                "embedding_dim": 768,
                "recommended_buckets": [128],
                "calibration": {"machine": None, "buckets": {}},
                "eeane_version": __version__,
            }
        ),
        encoding="utf-8",
    )

    snippet = artifacts.build_config_snippet(
        model_id="ruri-v3-310m",
        kind="embedding",
        tokenizer_path=tokenizer_path,
        artifacts={128: mlmodelc_path},
    )
    config_path = tmp_path / "eeane.toml"
    config_path.write_text(snippet, encoding="utf-8")

    loaded = load_config(explicit_path=config_path, env={"XDG_CACHE_HOME": str(tmp_path)})

    entry = loaded.config.models[0]
    assert entry.id == "ruri-v3-310m"
    assert entry.kind == "embedding"
    assert entry.output_name == "embedding"
    assert entry.tokenizer == tokenizer_path
    assert entry.artifacts == {128: mlmodelc_path}
    assert entry.embedding_dim == 768
    assert entry.excluded_buckets == ()


def test_write_config_snippet_creates_parent_directories(tmp_path: Path) -> None:
    """--emit-config must work even when the destination directory does not exist."""
    destination = tmp_path / "nested" / "eeane.toml"

    artifacts.write_config_snippet(destination, "[[models]]\n")

    assert destination.read_text(encoding="utf-8") == "[[models]]\n"


def test_write_config_snippet_reports_an_unwritable_destination(tmp_path: Path) -> None:
    """A destination that cannot be written must fail with a clean compile error."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    with pytest.raises(artifacts.CompileError, match="eeane.toml"):
        artifacts.write_config_snippet(blocker / "eeane.toml", "[[models]]\n")


# --- idempotent skip ---------------------------------------------------------


def test_needs_conversion_skips_an_up_to_date_variant(tmp_path: Path) -> None:
    """Matching versions plus both artifacts present means nothing to do."""
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions)

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, versions) is False


def test_needs_conversion_when_a_recorded_version_differs(tmp_path: Path) -> None:
    """A different torch/transformers/... version must trigger a reconversion."""
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions)

    current = {**versions, "coremltools": "9.1"}

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, current) is True


def test_needs_conversion_when_forced(tmp_path: Path) -> None:
    """--force must reconvert even a perfectly up-to-date variant."""
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions)

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, versions, force=True) is True


def test_needs_conversion_when_the_artifact_is_missing(tmp_path: Path) -> None:
    """Metadata without its .mlmodelc is not a reusable variant."""
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions)
    shutil.rmtree(mlmodelc_path)

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, versions) is True


def test_needs_conversion_when_the_metadata_is_missing_or_corrupt(tmp_path: Path) -> None:
    """Missing or unparsable metadata must never be treated as up to date."""
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions)
    metadata_path.unlink()

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, versions) is True

    metadata_path.write_text("{not json", encoding="utf-8")

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, versions) is True


def test_needs_conversion_when_the_recorded_selfcheck_failed(tmp_path: Path) -> None:
    """A variant whose self-check failed must be retried, not skipped."""
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions, selfcheck_status="failed")

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, versions) is True


def test_needs_conversion_when_the_recorded_pooling_matches(tmp_path: Path) -> None:
    """A recorded pooling equal to the one this run resolved is not a reason to reconvert."""
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions, pooling="mean")

    assert (
        artifacts.needs_conversion(mlmodelc_path, metadata_path, versions, pooling="mean") is False
    )


def test_needs_conversion_when_the_recorded_pooling_differs(tmp_path: Path) -> None:
    """A model whose declared pooling changed must be reconverted, not silently reused.

    A Hub repository can update its sentence-transformers declaration
    between two compiles of the exact same eeane version; nothing in
    SKIP_VERSION_KEYS would ever notice that on its own.
    """
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions, pooling="mean")

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, versions, pooling="cls") is True


def test_needs_conversion_when_no_pooling_was_ever_recorded(tmp_path: Path) -> None:
    """An old-format record without a recorded pooling counts as a mismatch, not a pass."""
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions)

    assert (
        artifacts.needs_conversion(mlmodelc_path, metadata_path, versions, pooling="mean") is True
    )


def test_needs_conversion_skips_the_pooling_check_when_none_is_given(tmp_path: Path) -> None:
    """pooling=None (a reranker, or an unresolved declaration) must leave the outcome alone."""
    versions = _versions()
    mlmodelc_path, metadata_path = _write_variant(tmp_path, versions)

    assert artifacts.needs_conversion(mlmodelc_path, metadata_path, versions, pooling=None) is False


# --- declared pooling ---------------------------------------------------------


def _write_pooling_declaration(model_dir: Path, pooling_flag: str) -> None:
    """Write a minimal sentence-transformers pooling declaration.

    Args:
        model_dir: Directory the ``1_Pooling/config.json`` is created under.
        pooling_flag: Flag to set to ``True`` (e.g.
            ``"pooling_mode_mean_tokens"``).
    """
    (model_dir / mb.POOLING_DIRNAME).mkdir(parents=True, exist_ok=True)
    (model_dir / mb.POOLING_DIRNAME / "config.json").write_text(
        json.dumps({pooling_flag: True}), encoding="utf-8"
    )


def test_declared_pooling_reads_an_embedding_models_declaration(tmp_path: Path) -> None:
    """A readable sentence-transformers declaration must be reported back verbatim."""
    _write_pooling_declaration(tmp_path, "pooling_mode_cls_token")

    assert pipeline._declared_pooling("embedding", tmp_path) == "cls"


def test_declared_pooling_is_none_when_the_declaration_cannot_be_read(tmp_path: Path) -> None:
    """A missing declaration must resolve to None here, not raise.

    The authoritative failure for a missing or malformed declaration is
    the backend's own load(); this helper only records what it can for
    later comparison and must never fail the run on its own.
    """
    assert pipeline._declared_pooling("embedding", tmp_path) is None


def test_declared_pooling_is_none_for_a_reranker(tmp_path: Path) -> None:
    """A reranker's pooling belongs to its own classification head, not a declaration."""
    _write_pooling_declaration(tmp_path, "pooling_mode_mean_tokens")

    assert pipeline._declared_pooling("reranker", tmp_path) is None


# --- calibration aggregation --------------------------------------------------


def _write_selfcheck(mlmodelc_path: Path, selfcheck: dict[str, Any] | None) -> None:
    """Write a variant metadata file exposing only its ``selfcheck`` block.

    Args:
        mlmodelc_path: The (not necessarily existing) artifact path whose
            sibling ``.json`` metadata file is written.
        selfcheck: Value to store under the ``selfcheck`` key, or ``None``
            to omit the key entirely (an old or foreign record).
    """
    metadata_path = mlmodelc_path.with_suffix(".json")
    payload: dict[str, Any] = {"variant": {"seq_len": 0}}
    if selfcheck is not None:
        payload["selfcheck"] = selfcheck
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_calibration_recommends_every_status_but_failed(tmp_path: Path) -> None:
    """Only a "failed" bucket must be excluded from recommended_buckets."""
    cache_artifacts = {
        128: tmp_path / "s128.mlmodelc",
        512: tmp_path / "s512.mlmodelc",
        1024: tmp_path / "s1024.mlmodelc",
        2048: tmp_path / "s2048.mlmodelc",
    }
    _write_selfcheck(cache_artifacts[128], {"status": "passed", "machine": {"cpu": "m1"}})
    _write_selfcheck(cache_artifacts[512], {"status": "warned"})
    _write_selfcheck(cache_artifacts[1024], {"status": "skipped", "reason": "no hook"})
    _write_selfcheck(cache_artifacts[2048], {"status": "failed", "error": "boom"})

    calibration, recommended, embedding_dim = artifacts.aggregate_calibration(
        "reranker", cache_artifacts, {}
    )

    assert recommended == [128, 512, 1024]
    assert embedding_dim is None
    assert calibration["buckets"]["128"]["status"] == "passed"
    assert calibration["buckets"]["128"]["measured"] is True
    assert calibration["buckets"]["512"]["status"] == "warned"
    assert calibration["buckets"]["1024"]["measured"] is False
    assert calibration["buckets"]["1024"]["status"] is None
    assert calibration["buckets"]["2048"]["status"] == "failed"
    assert calibration["buckets"]["2048"]["measured"] is True


def test_aggregate_calibration_fills_measured_fields(tmp_path: Path) -> None:
    """A measured bucket's sanity/compute_plan/latency values must be copied through."""
    cache_artifacts = {128: tmp_path / "s128.mlmodelc"}
    _write_selfcheck(
        cache_artifacts[128],
        {
            "status": "passed",
            "sanity": {"passed": True},
            "compute_plan": {"ne_placement_pct": 97.5},
            "latency": {"median_ms": 3.1, "p95_ms": 4.2},
        },
    )

    calibration, recommended, _ = artifacts.aggregate_calibration("embedding", cache_artifacts, {})

    assert recommended == [128]
    assert calibration["buckets"]["128"] == {
        "status": "passed",
        "sanity_passed": True,
        "ne_placement_pct": 97.5,
        "latency_median_ms": 3.1,
        "latency_p95_ms": 4.2,
        "measured": True,
    }


def test_aggregate_calibration_treats_unreadable_metadata_as_unmeasured(tmp_path: Path) -> None:
    """Missing or corrupt metadata must degrade to measured=False, not raise."""
    cache_artifacts = {128: tmp_path / "s128.mlmodelc", 512: tmp_path / "s512.mlmodelc"}
    cache_artifacts[512].with_suffix(".json").write_text("{not json", encoding="utf-8")
    # 128's metadata is never written at all.

    calibration, recommended, embedding_dim = artifacts.aggregate_calibration(
        "embedding", cache_artifacts, {}
    )

    assert recommended == [128, 512]
    assert embedding_dim is None
    for bucket in ("128", "512"):
        assert calibration["buckets"][bucket] == {
            "status": None,
            "sanity_passed": None,
            "ne_placement_pct": None,
            "latency_median_ms": None,
            "latency_p95_ms": None,
            "measured": False,
        }


def test_aggregate_calibration_prefers_this_runs_own_machine(tmp_path: Path) -> None:
    """A run's own machine block must win over one read back from an earlier run."""
    cache_artifacts = {128: tmp_path / "s128.mlmodelc", 512: tmp_path / "s512.mlmodelc"}
    _write_selfcheck(cache_artifacts[128], {"status": "passed", "machine": {"cpu": "earlier-run"}})
    run_reports = {512: {"status": "passed", "machine": {"cpu": "this-run"}}}

    calibration, _, _ = artifacts.aggregate_calibration("reranker", cache_artifacts, run_reports)

    assert calibration["machine"] == {"cpu": "this-run"}


def test_aggregate_calibration_falls_back_to_an_existing_machine(tmp_path: Path) -> None:
    """Without a machine block of its own, the run must fall back to a recorded one."""
    cache_artifacts = {128: tmp_path / "s128.mlmodelc", 512: tmp_path / "s512.mlmodelc"}
    _write_selfcheck(cache_artifacts[128], {"status": "skipped"})
    _write_selfcheck(cache_artifacts[512], {"status": "passed", "machine": {"cpu": "earlier-run"}})
    run_reports = {128: {"status": "skipped", "reason": "no hook"}}

    calibration, _, _ = artifacts.aggregate_calibration("reranker", cache_artifacts, run_reports)

    assert calibration["machine"] == {"cpu": "earlier-run"}


def test_aggregate_calibration_machine_is_none_when_nothing_was_ever_measured(
    tmp_path: Path,
) -> None:
    """A cache where every self-check was always skipped must report machine=None."""
    cache_artifacts = {128: tmp_path / "s128.mlmodelc"}
    _write_selfcheck(cache_artifacts[128], {"status": "skipped"})

    calibration, _, _ = artifacts.aggregate_calibration("reranker", cache_artifacts, {})

    assert calibration["machine"] is None


def test_aggregate_calibration_adopts_the_consistent_embedding_dim(tmp_path: Path) -> None:
    """The shared embedding_dim across every measured bucket must be adopted."""
    cache_artifacts = {128: tmp_path / "s128.mlmodelc", 512: tmp_path / "s512.mlmodelc"}
    _write_selfcheck(cache_artifacts[128], {"status": "passed", "sanity": {"embedding_dim": 768}})
    _write_selfcheck(cache_artifacts[512], {"status": "warned", "sanity": {"embedding_dim": 768}})

    _, _, embedding_dim = artifacts.aggregate_calibration("embedding", cache_artifacts, {})

    assert embedding_dim == 768


def test_aggregate_calibration_rejects_an_inconsistent_embedding_dim(tmp_path: Path) -> None:
    """Disagreeing embedding_dim values across buckets must be reported as a corrupt cache."""
    cache_artifacts = {128: tmp_path / "s128.mlmodelc", 512: tmp_path / "s512.mlmodelc"}
    _write_selfcheck(cache_artifacts[128], {"status": "passed", "sanity": {"embedding_dim": 768}})
    _write_selfcheck(cache_artifacts[512], {"status": "passed", "sanity": {"embedding_dim": 1024}})

    with pytest.raises(artifacts.CompileError, match="embedding_dim"):
        artifacts.aggregate_calibration("embedding", cache_artifacts, {})


def test_aggregate_calibration_embedding_dim_is_none_without_any_measurement(
    tmp_path: Path,
) -> None:
    """No cached bucket recording embedding_dim must leave it None, not raise."""
    cache_artifacts = {128: tmp_path / "s128.mlmodelc"}
    _write_selfcheck(cache_artifacts[128], {"status": "skipped"})

    _, _, embedding_dim = artifacts.aggregate_calibration("embedding", cache_artifacts, {})

    assert embedding_dim is None


def test_aggregate_calibration_embedding_dim_is_always_none_for_a_reranker(
    tmp_path: Path,
) -> None:
    """A reranker never records embedding_dim, even if a sanity block carries one."""
    cache_artifacts = {512: tmp_path / "s512.mlmodelc"}
    _write_selfcheck(cache_artifacts[512], {"status": "passed", "sanity": {"embedding_dim": 768}})

    _, _, embedding_dim = artifacts.aggregate_calibration("reranker", cache_artifacts, {})

    assert embedding_dim is None


# --- patches recording --------------------------------------------------------


def _stub_compile_context(
    tmp_path: Path, *, kind: str = "embedding", pooling: str | None = "mean"
) -> pipeline._CompileContext:
    """Build a minimal ``_CompileContext`` for a ``_build_metadata`` unit test."""
    return pipeline._CompileContext(
        args=argparse.Namespace(source="stub-source"),
        model_dir=tmp_path,
        model_id="stub-model",
        kind=kind,
        pooling=pooling,
        output_name="embedding",
        batch_size=1,
        model_root=tmp_path,
        tokenizer_path=tmp_path / "tokenizer.json",
        versions={"eeane": "0.0"},
        recorded_args={},
        backend=None,
        selfcheck_fn=None,
    )


def test_build_metadata_records_the_actual_apply_patches_return_value(tmp_path: Path) -> None:
    """The metadata's ``patches`` key must be the backend's own return value, verbatim.

    Regression test: the metadata used to hardcode ModernBERT-specific
    patch names for every backend; a backend that applies something else
    entirely (or nothing) must be described accurately, not assumed.
    """
    context = _stub_compile_context(tmp_path)
    plan = artifacts.VariantPlan(
        seq_len=128,
        stem="s128_b1_eager_macos13",
        mlpackage_path=tmp_path / "s128.mlpackage",
        mlmodelc_path=tmp_path / "s128.mlmodelc",
        metadata_path=tmp_path / "s128.json",
        convert=True,
    )
    patches = {"no_op": True, "note": "a backend that applies nothing still reports honestly"}

    metadata = pipeline._build_metadata(
        context, plan, {"load": 0.0}, {"mlmodelc": "x"}, patches, {"status": "skipped"}
    )

    assert metadata["patches"] == patches
    assert metadata["patches"] is not patches  # a defensive copy, not the same object


class _StubBackend:
    """Minimal compile backend stub driving ``_convert_variants`` without torch."""

    name = "Stub"

    def __init__(self, patches: dict[str, Any]) -> None:
        """Store the fixed record :meth:`apply_patches` must hand back."""
        self._patches = patches

    def load(self, model_dir: Path, kind: str, attn: str = "eager") -> Any:
        """Return an opaque handle; nothing downstream inspects it here."""
        return SimpleNamespace(kind=kind)

    def apply_patches(self, loaded: Any, mask_fill_value: float | None = None) -> dict[str, Any]:
        """Return the fixed patch record this stub was built with."""
        return dict(self._patches)

    def wrap(self, loaded: Any) -> str:
        """Return an opaque wrapper; ``conversion.trace_model`` is stubbed too."""
        return "wrapped"

    def trace_example(self, kind: str) -> str:
        """Return a fixed placeholder trace example."""
        return "example"

    def tokenize(self, loaded: Any, inputs: list[Any], seq_len: int) -> dict[str, np.ndarray]:
        """Return fixed-shape zero/one arrays; no real tokenizer is involved."""
        n = len(inputs)
        return {
            "input_ids": np.zeros((n, seq_len), dtype=np.int32),
            "attention_mask": np.ones((n, seq_len), dtype=np.int32),
        }


def _install_stub_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub trace/convert/compile so a variant needs neither torch nor xcrun."""

    def _fake_convert_model(
        traced: Any, seq_len: int, precision: str, target: str, output_name: str, *, batch_size: int
    ) -> Any:
        class _FakeMLModel:
            def save(self, path: str) -> None:
                Path(path).mkdir(parents=True, exist_ok=True)

        return _FakeMLModel()

    def _fake_compile_model(mlpackage_path: Path, mlmodelc_path: Path) -> None:
        mlmodelc_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pipeline.conversion, "trace_model", lambda wrapper, example: "traced")
    monkeypatch.setattr(pipeline.conversion, "convert_model", _fake_convert_model)
    monkeypatch.setattr(pipeline.conversion, "compile_model", _fake_compile_model)


def test_convert_variants_records_the_backends_own_patches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stub backend's apply_patches() return value must reach the variant's metadata.

    Drives ``_convert_variants`` (not the whole pipeline) with a stub
    backend and stubbed conversion steps, so this needs neither torch
    weights nor ``xcrun``: the point is that the pipeline records exactly
    what the backend reports, not an assumption tied to one architecture.
    """
    _install_stub_conversion(monkeypatch)
    patches = {"some_rewrite": True, "detail": 42}
    context = pipeline._CompileContext(
        args=argparse.Namespace(
            source="stub-source",
            attn="eager",
            precision="fp16",
            target="macos13",
            keep_mlpackage=False,
            skip_selfcheck=False,
        ),
        model_dir=tmp_path,
        model_id="stub-model",
        kind="embedding",
        pooling="mean",
        output_name="embedding",
        batch_size=1,
        model_root=tmp_path,
        tokenizer_path=tmp_path / "tokenizer.json",
        versions={"eeane": "0.0"},
        recorded_args={},
        backend=_StubBackend(patches),
        selfcheck_fn=None,
    )
    plan = artifacts.VariantPlan(
        seq_len=8,
        stem="s8_b1_eager_macos13",
        mlpackage_path=tmp_path / "s8.mlpackage",
        mlmodelc_path=tmp_path / "s8.mlmodelc",
        metadata_path=tmp_path / "s8.json",
        convert=True,
    )

    run_reports = pipeline._convert_variants(context, [plan])

    assert plan.mlmodelc_path.is_dir()
    assert run_reports == {
        8: {
            "status": artifacts.SELFCHECK_STATUS_SKIPPED,
            "reason": pipeline.SELFCHECK_REASON_UNAVAILABLE,
        }
    }
    metadata = json.loads(plan.metadata_path.read_text(encoding="utf-8"))
    assert metadata["patches"] == patches


# --- tokenizer verification inputs -------------------------------------------


def test_verification_inputs_are_self_contained_for_an_embedding_model() -> None:
    """The gate inputs must include the boundary cases and need no repository data."""
    backend = mb.ModernBertBackend()

    texts, pairs = pipeline.verification_inputs(backend, "embedding", [128])

    assert "" in texts  # empty input
    assert any(len(text) == 1 for text in texts)  # single character
    assert max(len(text) for text in texts) > 4 * 128  # far longer than the bucket
    assert all(isinstance(text, str) for text in texts)
    assert set(backend.sanity_spec("embedding").inputs).issubset(texts)
    assert pairs == []  # an embedding model never encodes pairs


def test_verification_inputs_include_pairs_for_a_reranker() -> None:
    """A reranker must be verified on pair encodings (the dynamic post_processor)."""
    backend = mb.ModernBertBackend()

    texts, pairs = pipeline.verification_inputs(backend, "reranker", [512])

    assert texts and "" in texts
    assert ("", "") in pairs
    assert all(len(pair) == 2 for pair in pairs)
    assert set(backend.sanity_spec("reranker").inputs).issubset(pairs)


def test_verification_long_input_scales_with_the_largest_bucket() -> None:
    """The long input must outgrow whatever the largest requested bucket is."""
    backend = mb.ModernBertBackend()

    short_texts, _ = pipeline.verification_inputs(backend, "embedding", [128])
    long_texts, _ = pipeline.verification_inputs(backend, "embedding", [128, 1024])

    assert max(len(text) for text in long_texts) > max(len(text) for text in short_texts)


# --- run(): argument and resolution failures ---------------------------------


def _compile_args(*arguments: str) -> Any:
    """Parse a ``compile`` command line into the namespace ``run`` expects."""
    return cli.build_parser().parse_args(["compile", *arguments])


def test_run_rejects_a_non_positive_batch(capsys: pytest.CaptureFixture) -> None:
    """--batch 0 must fail before anything is loaded or written."""
    exit_code = pipeline.run(_compile_args("some/path", "--batch", "0"))

    assert exit_code == 1
    assert "batch" in capsys.readouterr().err


def test_run_reports_an_unresolvable_source(capsys: pytest.CaptureFixture) -> None:
    """A source that is neither a directory nor a Hub id must exit non-zero, cleanly."""
    exit_code = pipeline.run(_compile_args("definitely/not/a/model/dir"))

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "definitely/not/a/model/dir" in captured.err
    assert "Traceback" not in captured.err


def test_run_reports_a_bert_reranker_as_unsupported(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A BERT cross-encoder must be refused with a reason, not a traceback.

    Its architecture is dispatched to the BERT backend, which compiles
    embedding models only, so the refusal has to travel from the backend
    to the single ``eeane compile: ...`` line the user sees.
    """
    source = tmp_path / "bert-reranker"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"architectures": ["BertForSequenceClassification"]}), encoding="utf-8"
    )

    exit_code = pipeline.run(_compile_args(str(source)))

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "reranker" in captured.err
    assert "segment ids" in captured.err


def test_run_reports_a_directory_without_config_json(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A directory that is not a HuggingFace model must be reported by name."""
    source = tmp_path / "not-a-model"
    source.mkdir()

    exit_code = pipeline.run(_compile_args(str(source)))

    assert exit_code == 1
    assert "config.json" in capsys.readouterr().err


def test_run_reports_an_unusable_out_dir(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A cache root that cannot be created must fail before the model is loaded."""
    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"architectures": ["ModernBertModel"]}), encoding="utf-8"
    )
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    exit_code = pipeline.run(_compile_args(str(source), "--out-dir", str(blocker)))

    assert exit_code == 1
    assert "director" in capsys.readouterr().err


# --- buckets vs the model's maximum sequence length --------------------------


class _LimitBackend:
    """Backend stub answering only the bucket-validation questions."""

    name = "Stub"
    supported_kinds: tuple[str, ...] = ("embedding", "reranker")

    def __init__(self, limit: int | None) -> None:
        """Store the effective maximum sequence length to report.

        Args:
            limit: Value returned by :meth:`max_seq_len` (``None`` = no
                known limit).
        """
        self._limit = limit
        self.seen: list[Path] = []

    def max_seq_len(self, model_dir: Path) -> int | None:
        """Record the queried directory and return the fixed limit."""
        self.seen.append(model_dir)
        return self._limit

    def output_name(self, kind: str) -> str:
        """Return the graph output name of ``kind`` (embedding-style stub)."""
        return "logits" if kind == "reranker" else "embedding"


@pytest.mark.parametrize("explicit", [True, False])
def test_buckets_within_the_model_limit_are_kept(
    explicit: bool, capsys: pytest.CaptureFixture
) -> None:
    """Buckets the model can actually process must pass through untouched and silently."""
    backend = _LimitBackend(1024)

    resolved = pipeline._apply_max_seq_len(
        backend, Path("/models/example"), [128, 512, 1024], explicit=explicit
    )

    assert resolved == [128, 512, 1024]
    assert backend.seen == [Path("/models/example")]
    assert "dropped" not in capsys.readouterr().err


def test_explicit_buckets_beyond_the_model_limit_are_an_error() -> None:
    """A user-requested bucket the model cannot process must be reported, not silently changed."""
    backend = _LimitBackend(512)

    with pytest.raises(artifacts.CompileError) as excinfo:
        pipeline._apply_max_seq_len(
            backend, Path("/models/example"), [128, 512, 1024, 2048], explicit=True
        )

    message = str(excinfo.value)
    assert "512" in message  # the model's effective maximum
    assert "1024" in message and "2048" in message  # every offending bucket


def test_default_buckets_beyond_the_model_limit_are_dropped_with_a_message(
    capsys: pytest.CaptureFixture,
) -> None:
    """Default buckets must be clipped to what the model supports, visibly."""
    backend = _LimitBackend(512)

    resolved = pipeline._apply_max_seq_len(
        backend, Path("/models/example"), [128, 512, 1024], explicit=False
    )

    assert resolved == [128, 512]
    stderr = capsys.readouterr().err
    assert "bucket 1024 dropped" in stderr
    assert "512" in stderr


def test_an_unknown_maximum_sequence_length_skips_the_validation() -> None:
    """A backend that cannot determine the limit must not block any bucket."""
    backend = _LimitBackend(None)

    resolved = pipeline._apply_max_seq_len(
        backend, Path("/models/example"), [128, 99999], explicit=True
    )

    assert resolved == [128, 99999]


def test_dropping_every_default_bucket_is_an_error() -> None:
    """Clipping must never leave the pipeline with nothing to compile."""
    backend = _LimitBackend(64)

    with pytest.raises(artifacts.CompileError, match="64"):
        pipeline._apply_max_seq_len(backend, Path("/models/example"), [128, 512], explicit=False)


def test_run_reports_explicit_buckets_beyond_the_model_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The bucket check must fail the run cleanly, before anything is loaded or written."""
    source = tmp_path / "model"
    source.mkdir()
    out_dir = tmp_path / "cache"

    class _StubDispatch:
        """Dispatch result pointing at the limit-reporting stub backend."""

        architecture = "StubModel"
        backend_name = "Stub"
        kind = "embedding"

        def load_backend(self) -> Any:
            """Return the stub backend instance."""
            return _LimitBackend(512)

    monkeypatch.setattr(pipeline, "resolve_dispatch", lambda model_dir, kind: _StubDispatch())

    exit_code = pipeline.run(
        _compile_args(str(source), "--buckets", "1024", "--out-dir", str(out_dir))
    )

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "1024" in stderr and "512" in stderr
    assert "Traceback" not in stderr
    assert not out_dir.exists()  # nothing was written


# --- end-to-end on a synthetic ModernBERT (local only) -----------------------


def _build_synthetic_model(path: Path, pooling_flag: str = "pooling_mode_mean_tokens") -> Path:
    """Create a tiny randomly initialised ModernBERT model directory.

    The directory is a complete HuggingFace distribution-format model
    (config.json + safetensors weights + a byte-level fast tokenizer + a
    sentence-transformers pooling declaration), so the pipeline can be
    driven exactly as it would be for a real model.

    Args:
        path: Directory to create (parents are created as needed).
        pooling_flag: sentence-transformers pooling flag to declare
            (``"pooling_mode_mean_tokens"`` or ``"pooling_mode_cls_token"``).

    Returns:
        ``path``.
    """
    config = modeling_modernbert.ModernBertConfig(
        vocab_size=300,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        max_position_embeddings=64,
        local_attention=8,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    modeling_modernbert.ModernBertModel(config).save_pretrained(path)

    # Byte-level vocabulary with no merges: every byte is its own token, so
    # Japanese text produces plenty of tokens without shipping a vocab file.
    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
    for index, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet())):
        vocab[character] = index + 4
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[("<s>", 2), ("</s>", 3)],
    )
    tokenizer.decoder = decoders.ByteLevel()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        model_max_length=64,
    ).save_pretrained(path)

    # An embedding model must declare its pooling now that the modernbert
    # backend detects it instead of hard-coding mean, so this is not optional.
    (path / mb.POOLING_DIRNAME).mkdir(parents=True, exist_ok=True)
    (path / mb.POOLING_DIRNAME / "config.json").write_text(
        json.dumps({"word_embedding_dimension": 32, pooling_flag: True}), encoding="utf-8"
    )
    return path


def _mtimes(directory: Path) -> dict[str, int]:
    """Snapshot every file's modification time under ``directory``."""
    return {
        str(path.relative_to(directory)): path.stat().st_mtime_ns
        for path in sorted(directory.rglob("*"))
    }


@pytest.fixture(scope="module")
def synthetic_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the synthetic model directory once for the end-to-end tests."""
    if not _E2E_AVAILABLE:
        pytest.skip("end-to-end conversion needs a local machine with xcrun")
    return _build_synthetic_model(tmp_path_factory.mktemp("synthetic") / "tiny-modernbert")


@pytest.fixture(scope="module")
def compiled(synthetic_model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the full pipeline once and return its inputs, outputs and stdout."""
    workspace = tmp_path_factory.mktemp("compile")
    out_dir = workspace / "cache"
    emit_config = workspace / "emitted.toml"
    arguments = [
        str(synthetic_model_dir),
        "--buckets",
        str(E2E_SEQ_LEN),
        "--out-dir",
        str(out_dir),
        "--emit-config",
        str(emit_config),
    ]

    before = _mtimes(synthetic_model_dir)
    stdout = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(stdout):
        exit_code = pipeline.run(_compile_args(*arguments))
    elapsed = time.perf_counter() - started

    return {
        "arguments": arguments,
        "exit_code": exit_code,
        "elapsed": elapsed,
        "stdout": stdout.getvalue(),
        "model_dir": synthetic_model_dir,
        "out_dir": out_dir,
        "model_root": out_dir / "compiled" / synthetic_model_dir.name,
        "emit_config": emit_config,
        "mtimes_before": before,
    }


def test_e2e_produces_the_compiled_artifact_and_drops_the_mlpackage(
    compiled: dict[str, Any],
) -> None:
    """A successful run must leave a .mlmodelc, a frozen tokenizer and no .mlpackage."""
    model_root = compiled["model_root"]

    assert compiled["exit_code"] == 0
    assert (model_root / f"{E2E_STEM}.mlmodelc").is_dir()
    assert not (model_root / f"{E2E_STEM}.mlpackage").exists()
    assert (model_root / artifacts.TOKENIZER_FILENAME).is_file()
    assert not list(model_root.glob("*.compile_tmp"))


def test_e2e_writes_variant_metadata(compiled: dict[str, Any]) -> None:
    """The variant metadata must describe the source, versions, patches and timings."""
    metadata = json.loads((compiled["model_root"] / f"{E2E_STEM}.json").read_text(encoding="utf-8"))

    assert metadata["format_version"] == artifacts.METADATA_FORMAT_VERSION
    assert metadata["source"]["requested"] == str(compiled["model_dir"])
    assert Path(metadata["source"]["resolved"]) == compiled["model_dir"].resolve()
    assert metadata["args"]["buckets"] == [E2E_SEQ_LEN]
    assert metadata["versions"]["eeane"] == __version__
    # Recorded verbatim from ModernBertBackend.apply_patches()'s return
    # value: the two mandatory rewrites, no mask fill (never requested).
    assert metadata["patches"] == {"rotate_half_static": True, "eager_attention_rank4": True}
    assert {"load", "trace", "convert", "compile", "total"} <= set(metadata["timings_sec"])
    assert Path(metadata["artifacts"]["mlmodelc"]).is_dir()
    assert "mlpackage" not in metadata["artifacts"]
    assert metadata["selfcheck"]["status"] == artifacts.SELFCHECK_STATUS_SKIPPED


def test_e2e_writes_model_info(compiled: dict[str, Any]) -> None:
    """model_info.json must summarise the model for later cache resolution."""
    info = json.loads(
        (compiled["model_root"] / artifacts.MODEL_INFO_FILENAME).read_text(encoding="utf-8")
    )

    assert info["format_version"] == artifacts.MODEL_INFO_FORMAT_VERSION
    assert info["id"] == compiled["model_dir"].name
    assert info["kind"] == "embedding"
    assert info["output_name"] == "embedding"
    assert info["buckets"] == [E2E_SEQ_LEN]
    assert info["tokenizer"] == artifacts.TOKENIZER_FILENAME
    assert info["eeane_version"] == __version__
    assert info["tokenizer_freeze"]["verified"] is True
    assert info["tokenizer_freeze"]["buckets"] == [E2E_SEQ_LEN]
    # No self-check hook was given: the one bucket is unmeasured, but still
    # recommended (only a "failed" status excludes a bucket).
    assert info["recommended_buckets"] == [E2E_SEQ_LEN]
    assert info["embedding_dim"] is None
    assert info["calibration"]["machine"] is None
    assert info["calibration"]["buckets"] == {
        str(E2E_SEQ_LEN): {
            "status": None,
            "sanity_passed": None,
            "ne_placement_pct": None,
            "latency_median_ms": None,
            "latency_p95_ms": None,
            "measured": False,
        }
    }


def test_e2e_prints_and_emits_a_usable_config_snippet(compiled: dict[str, Any]) -> None:
    """stdout must carry the snippet, and --emit-config must write the same text."""
    stdout = compiled["stdout"]
    emitted = compiled["emit_config"].read_text(encoding="utf-8")

    assert "[[models]]" in stdout
    assert emitted in stdout

    assert _parse_snippet(emitted) == {"id": compiled["model_dir"].name, "normalize": True}
    tokenizer_path = Path(_commented_value(emitted, "tokenizer"))
    artifact_path = Path(_commented_value(emitted, str(E2E_SEQ_LEN)))
    assert tokenizer_path == compiled["model_root"] / artifacts.TOKENIZER_FILENAME
    assert artifact_path == compiled["model_root"] / f"{E2E_STEM}.mlmodelc"
    assert tokenizer_path.is_file()
    assert artifact_path.is_dir()

    # --out-dir here is a pytest tmp directory, never the default cache
    # root, so the reminder to point the server at it must be present.
    expected_root = artifacts.resolve_out_root(compiled["out_dir"])
    assert f"# cache_root = {artifacts._toml_string(str(expected_root))}" in emitted


def test_e2e_leaves_the_input_model_directory_untouched(compiled: dict[str, Any]) -> None:
    """The input model directory is read-only."""
    assert _mtimes(compiled["model_dir"]) == compiled["mtimes_before"]


def test_e2e_second_run_skips_the_up_to_date_variant(
    compiled: dict[str, Any], capsys: pytest.CaptureFixture
) -> None:
    """Re-running without --force must reuse the artifact and say so."""
    mlmodelc_path = compiled["model_root"] / f"{E2E_STEM}.mlmodelc"
    before = mlmodelc_path.stat().st_mtime_ns

    exit_code = pipeline.run(_compile_args(*compiled["arguments"]))

    assert exit_code == 0
    assert mlmodelc_path.stat().st_mtime_ns == before
    assert "skip" in capsys.readouterr().err.lower()


def test_e2e_force_reconverts_and_keeps_the_mlpackage(compiled: dict[str, Any]) -> None:
    """--force must rebuild the artifact; --keep-mlpackage must retain the intermediate."""
    model_root = compiled["model_root"]
    mlmodelc_path = model_root / f"{E2E_STEM}.mlmodelc"
    before = mlmodelc_path.stat().st_mtime_ns

    with contextlib.redirect_stdout(io.StringIO()):
        exit_code = pipeline.run(
            _compile_args(*compiled["arguments"], "--force", "--keep-mlpackage")
        )

    assert exit_code == 0
    assert mlmodelc_path.stat().st_mtime_ns != before
    assert (model_root / f"{E2E_STEM}.mlpackage").is_dir()
    metadata = json.loads((model_root / f"{E2E_STEM}.json").read_text(encoding="utf-8"))
    assert Path(metadata["artifacts"]["mlpackage"]).is_dir()


def test_e2e_selfcheck_hook_result_is_recorded(compiled: dict[str, Any]) -> None:
    """A self-check implementation's report must land in the variant metadata."""
    contexts: list[pipeline.SelfcheckContext] = []

    def fake_selfcheck(context: pipeline.SelfcheckContext) -> dict[str, Any]:
        contexts.append(context)
        return {"status": "passed", "note": "fake"}

    with contextlib.redirect_stdout(io.StringIO()):
        exit_code = pipeline.run(
            _compile_args(*compiled["arguments"], "--force"), selfcheck_fn=fake_selfcheck
        )

    assert exit_code == 0
    assert len(contexts) == 1
    assert contexts[0].seq_len == E2E_SEQ_LEN
    assert contexts[0].kind == "embedding"
    assert contexts[0].output_name == "embedding"
    assert contexts[0].mlmodelc_path.is_dir()
    assert contexts[0].tokenizer_path.is_file()
    metadata = json.loads((compiled["model_root"] / f"{E2E_STEM}.json").read_text(encoding="utf-8"))
    assert metadata["selfcheck"] == {"status": "passed", "note": "fake"}

    info = json.loads(
        (compiled["model_root"] / artifacts.MODEL_INFO_FILENAME).read_text(encoding="utf-8")
    )
    assert info["recommended_buckets"] == [E2E_SEQ_LEN]
    assert info["calibration"]["buckets"][str(E2E_SEQ_LEN)]["status"] == "passed"
    assert info["calibration"]["buckets"][str(E2E_SEQ_LEN)]["measured"] is True


def test_e2e_failing_selfcheck_fails_the_compile(compiled: dict[str, Any]) -> None:
    """A failed self-check must exit non-zero and leave the variant non-skippable."""

    def failing_selfcheck(context: pipeline.SelfcheckContext) -> dict[str, Any]:
        return {"status": artifacts.SELFCHECK_STATUS_FAILED, "reason": "fake failure"}

    with contextlib.redirect_stdout(io.StringIO()):
        exit_code = pipeline.run(
            _compile_args(*compiled["arguments"], "--force"), selfcheck_fn=failing_selfcheck
        )

    assert exit_code == 1
    model_root = compiled["model_root"]
    metadata_path = model_root / f"{E2E_STEM}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["selfcheck"]["status"] == artifacts.SELFCHECK_STATUS_FAILED
    # The recorded failure must not be reusable on the next run.
    assert artifacts.needs_conversion(
        model_root / f"{E2E_STEM}.mlmodelc", metadata_path, metadata["versions"]
    )


# --- incremental bucket discovery (cache-wide snippet/model_info) ------------


def _write_family_metadata(
    directory: Path,
    seq_len: int,
    *,
    batch: int = 1,
    attn: str = "eager",
    target: str = "macos13",
    precision: str = "fp16",
    with_mlmodelc: bool = True,
    corrupt: bool = False,
) -> Path:
    """Write a fake variant metadata file (and optionally its .mlmodelc)."""
    stem = artifacts.variant_stem(seq_len, batch, attn, target, precision)
    metadata_path = directory / f"{stem}.json"
    if corrupt:
        metadata_path.write_text("{not json", encoding="utf-8")
    else:
        metadata_path.write_text(
            json.dumps(
                {
                    "variant": {"stem": stem, "seq_len": seq_len, "batch_size": batch},
                    "args": {"attn": attn, "target": target, "precision": precision},
                }
            ),
            encoding="utf-8",
        )
    mlmodelc_path = directory / f"{stem}.mlmodelc"
    if with_mlmodelc:
        mlmodelc_path.mkdir(parents=True, exist_ok=True)
    return mlmodelc_path


def test_discover_variants_lists_only_the_matching_family(tmp_path: Path) -> None:
    """Same-family variants with an existing .mlmodelc are found; everything else is not."""
    kept_128 = _write_family_metadata(tmp_path, 128)
    kept_512 = _write_family_metadata(tmp_path, 512)
    _write_family_metadata(tmp_path, 256, precision="fp32")  # different family
    _write_family_metadata(tmp_path, 96, batch=2)  # different family
    _write_family_metadata(tmp_path, 1024, with_mlmodelc=False)  # artifact gone
    _write_family_metadata(tmp_path, 64, corrupt=True)  # unreadable metadata

    found = artifacts.discover_variants(
        tmp_path, batch_size=1, attn="eager", target="macos13", precision="fp16"
    )

    assert found == {128: kept_128, 512: kept_512}


def test_discover_variants_of_a_missing_directory_is_empty(tmp_path: Path) -> None:
    """A model that was never compiled has no discoverable variants."""
    found = artifacts.discover_variants(
        tmp_path / "absent", batch_size=1, attn="eager", target="macos13", precision="fp16"
    )

    assert found == {}


def test_e2e_incremental_bucket_keeps_earlier_buckets(
    synthetic_model_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
    capfd: pytest.CaptureFixture,
) -> None:
    """Adding one bucket to an existing cache must keep listing the earlier ones.

    Regression test: a `--buckets 2048` run on a cache holding S512/S1024
    must not emit a snippet and model_info.json that list only 2048.
    """
    workspace = tmp_path_factory.mktemp("incremental")
    out_dir = workspace / "cache"
    second_bucket = E2E_SEQ_LEN + 16

    def run_bucket(bucket: int) -> str:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = pipeline.run(
                _compile_args(
                    str(synthetic_model_dir),
                    "--buckets",
                    str(bucket),
                    "--out-dir",
                    str(out_dir),
                )
            )
        assert exit_code == 0
        return stdout.getvalue()

    first_snippet = run_bucket(E2E_SEQ_LEN)
    assert sorted(_commented_buckets(first_snippet)) == [str(E2E_SEQ_LEN)]
    capfd.readouterr()

    second_snippet = run_bucket(second_bucket)

    stderr = capfd.readouterr().err
    assert f"previously compiled bucket(s) kept: {E2E_SEQ_LEN}" in stderr
    assert sorted(_commented_buckets(second_snippet), key=int) == [
        str(E2E_SEQ_LEN),
        str(second_bucket),
    ]
    model_info = json.loads(
        (out_dir / "compiled" / synthetic_model_dir.name / "model_info.json").read_text(
            encoding="utf-8"
        )
    )
    assert model_info["buckets"] == [E2E_SEQ_LEN, second_bucket]
    assert sorted(model_info["artifacts"], key=int) == [str(E2E_SEQ_LEN), str(second_bucket)]
    # Neither self-check ran (no --skip-selfcheck override here, but no
    # selfcheck_fn either): both buckets stay recommended (unmeasured).
    assert model_info["recommended_buckets"] == [E2E_SEQ_LEN, second_bucket]


def test_e2e_cls_pooling_embedding_model_converts(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A directory declaring CLS pooling must also convert through the full pipeline.

    Every other end-to-end test in this module shares the module-scoped
    ``synthetic_model_dir`` fixture, which declares mean pooling, so this
    builds its own directory to exercise the CLS branch as well.
    """
    if not _E2E_AVAILABLE:
        pytest.skip("end-to-end conversion needs a local machine with xcrun")
    model_dir = _build_synthetic_model(
        tmp_path_factory.mktemp("synthetic-cls") / "tiny-modernbert-cls",
        pooling_flag="pooling_mode_cls_token",
    )
    out_dir = tmp_path_factory.mktemp("compile-cls") / "cache"

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = pipeline.run(
            _compile_args(str(model_dir), "--buckets", str(E2E_SEQ_LEN), "--out-dir", str(out_dir))
        )

    model_root = out_dir / "compiled" / model_dir.name
    assert exit_code == 0
    assert (model_root / f"{E2E_STEM}.mlmodelc").is_dir()


# --- batch families in the record and the snippet ----------------------------

# Bucket the stub-driven runs of this section compile, small enough to
# keep the (stubbed) artifacts trivial.
_FAMILY_SEQ_LEN = 16


class _FamilyBackend(_StubBackend):
    """Stub backend answering every question a whole ``run()`` asks.

    Extends the conversion-level stub with the resolution-level answers
    (output name, sequence-length limit, sanity fixtures), so a complete
    pipeline run needs neither torch weights nor ``xcrun``.
    """

    supported_kinds: tuple[str, ...] = ("embedding", "reranker")

    def __init__(self) -> None:
        """Register a backend that applies no patch at all."""
        super().__init__({})

    def output_name(self, kind: str) -> str:
        """Return the graph output name of ``kind``."""
        return "logits" if kind == "reranker" else "embedding"

    def max_seq_len(self, model_dir: Path) -> int | None:
        """Report no known limit, so every requested bucket is compiled."""
        return None

    def sanity_spec(self, kind: str) -> Any:
        """Return the fixtures the tokenizer verification is built from."""
        inputs: list[Any] = [("q", "d")] if kind == "reranker" else ["hello"]
        return SimpleNamespace(inputs=inputs)

    def trace_example(self, kind: str) -> Any:
        """Return the fixed trace example of ``kind``."""
        return ("q", "d") if kind == "reranker" else "example"


def _install_family_stubs(monkeypatch: pytest.MonkeyPatch, kind: str = "embedding") -> None:
    """Stub dispatch, tokenizer freezing and conversion for a whole run.

    Args:
        monkeypatch: Fixture the stubs are installed with.
        kind: Model kind the stub dispatch reports.
    """

    class _StubDispatch:
        """Dispatch result pointing at the family backend stub."""

        architecture = "StubModel"
        backend_name = "Stub"

        def __init__(self, dispatched_kind: str) -> None:
            self.kind = dispatched_kind

        def load_backend(self) -> Any:
            """Return the stub backend instance."""
            return _FamilyBackend()

    def _fake_freeze(model_dir: Path, tokenizer_path: Path) -> dict[str, Any]:
        tokenizer_path.write_text("{}", encoding="utf-8")
        return {
            "tokenizer_class": "StubTokenizer",
            "pad_id": 0,
            "pad_token": "<pad>",
            "padding_direction": "right",
        }

    def _fake_verify(
        model_dir: Path,
        tokenizer_path: Path,
        texts: list[str],
        pairs: list[tuple[str, str]],
        buckets: list[int],
    ) -> dict[str, Any]:
        return {
            "passed": True,
            "buckets": list(buckets),
            "n_texts": len(texts),
            "n_pairs": len(pairs),
            "n_comparisons": len(texts) * len(buckets),
        }

    monkeypatch.setattr(pipeline, "resolve_dispatch", lambda model_dir, asked: _StubDispatch(kind))
    monkeypatch.setattr(pipeline, "freeze_tokenizer", _fake_freeze)
    monkeypatch.setattr(pipeline, "verify_frozen_tokenizer", _fake_verify)
    _install_stub_conversion(monkeypatch)


def _stub_source(tmp_path: Path, *, pooling_flag: str | None = None) -> Path:
    """Create the minimal model directory a run resolves its source to.

    Args:
        tmp_path: Directory the source is created under.
        pooling_flag: sentence-transformers pooling flag to declare under
            ``1_Pooling/config.json`` (e.g. ``"pooling_mode_mean_tokens"``).
            ``None`` (the default) leaves the model without one, matching
            every stub-driven test that never reads it.

    Returns:
        The created model directory.
    """
    source = tmp_path / "stub-model"
    source.mkdir(parents=True, exist_ok=True)
    (source / "config.json").write_text(
        json.dumps({"architectures": ["StubModel"]}), encoding="utf-8"
    )
    if pooling_flag is not None:
        _write_pooling_declaration(source, pooling_flag)
    return source


def _run_stub_compile(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    out_dir: Path,
    *,
    batch: int = 1,
    buckets: int = _FAMILY_SEQ_LEN,
    kind: str = "embedding",
) -> int:
    """Run the whole pipeline against the stubs and return its exit code."""
    _install_family_stubs(monkeypatch, kind=kind)
    return pipeline.run(
        _compile_args(
            str(source),
            "--buckets",
            str(buckets),
            "--batch",
            str(batch),
            "--out-dir",
            str(out_dir),
        )
    )


def _stub_model_info(out_dir: Path, source: Path) -> dict[str, Any]:
    """Read back the record a stub-driven run wrote."""
    path = out_dir / "compiled" / source.name / artifacts.MODEL_INFO_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_batch_1_run_records_no_batched_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache holding one batch size only must read exactly as it always did."""
    source = _stub_source(tmp_path)
    out_dir = tmp_path / "cache"

    assert _run_stub_compile(monkeypatch, source, out_dir) == 0

    info = _stub_model_info(out_dir, source)
    assert info["artifacts"] == {
        str(_FAMILY_SEQ_LEN): f"s{_FAMILY_SEQ_LEN}_b1_eager_macos13.mlmodelc"
    }
    assert "batch_artifacts" not in info
    assert info["format_version"] == artifacts.MODEL_INFO_FORMAT_VERSION


def test_a_batched_run_keeps_the_served_artifacts_and_records_the_batched_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batched run must add to the record, never overwrite the served table.

    Regression test: the record used to describe whichever family the run
    compiled, so a batched run replaced the artifacts an id-only entry is
    served with -- which would feed single inputs to a batched artifact.
    """
    source = _stub_source(tmp_path)
    out_dir = tmp_path / "cache"

    assert _run_stub_compile(monkeypatch, source, out_dir, batch=1) == 0
    assert _run_stub_compile(monkeypatch, source, out_dir, batch=2) == 0

    info = _stub_model_info(out_dir, source)
    bucket = str(_FAMILY_SEQ_LEN)
    assert info["artifacts"] == {bucket: f"s{_FAMILY_SEQ_LEN}_b1_eager_macos13.mlmodelc"}
    assert info["batch_artifacts"] == {
        "2": {bucket: f"s{_FAMILY_SEQ_LEN}_b2_eager_macos13.mlmodelc"}
    }
    assert info["buckets"] == [_FAMILY_SEQ_LEN]
    # The calibration still describes the served family only.
    assert info["recommended_buckets"] == [_FAMILY_SEQ_LEN]
    assert sorted(info["calibration"]["buckets"]) == [bucket]


def test_a_serving_run_after_a_batched_one_records_both_families(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whichever order the two runs happen in, the record must describe both."""
    source = _stub_source(tmp_path)
    out_dir = tmp_path / "cache"

    assert _run_stub_compile(monkeypatch, source, out_dir, batch=2) == 0
    assert _run_stub_compile(monkeypatch, source, out_dir, batch=1) == 0

    info = _stub_model_info(out_dir, source)
    bucket = str(_FAMILY_SEQ_LEN)
    assert info["artifacts"] == {bucket: f"s{_FAMILY_SEQ_LEN}_b1_eager_macos13.mlmodelc"}
    assert info["batch_artifacts"] == {
        "2": {bucket: f"s{_FAMILY_SEQ_LEN}_b2_eager_macos13.mlmodelc"}
    }


def test_a_batched_run_without_served_artifacts_warns_about_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Compiling only the batched family leaves nothing to serve, and must say so."""
    source = _stub_source(tmp_path)
    out_dir = tmp_path / "cache"

    assert _run_stub_compile(monkeypatch, source, out_dir, batch=2) == 0

    stderr = capsys.readouterr().err
    assert "WARNING" in stderr
    assert "batch-1" in stderr
    info = _stub_model_info(out_dir, source)
    # The record is still written, so the batched artifacts are not lost.
    assert info["artifacts"] == {}
    assert info["recommended_buckets"] == []
    assert info["batch_artifacts"] == {
        "2": {str(_FAMILY_SEQ_LEN): f"s{_FAMILY_SEQ_LEN}_b2_eager_macos13.mlmodelc"}
    }


def test_the_snippet_of_a_model_with_batched_artifacts_names_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """An embedding snippet must offer the batched table once the cache holds one."""
    source = _stub_source(tmp_path)
    out_dir = tmp_path / "cache"

    assert _run_stub_compile(monkeypatch, source, out_dir, batch=1) == 0
    capsys.readouterr()
    assert _run_stub_compile(monkeypatch, source, out_dir, batch=2) == 0

    snippet = capsys.readouterr().out
    model_root = out_dir / "compiled" / source.name
    assert _commented_table(snippet, "models.artifacts") == {
        str(_FAMILY_SEQ_LEN): str(model_root / f"s{_FAMILY_SEQ_LEN}_b1_eager_macos13.mlmodelc")
    }
    assert _commented_table(snippet, "models.batch_artifacts") == {
        str(_FAMILY_SEQ_LEN): str(model_root / f"s{_FAMILY_SEQ_LEN}_b2_eager_macos13.mlmodelc")
    }


def test_a_reranker_snippet_never_names_batched_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A reranker is served one pair at a time, whatever its cache happens to hold."""
    source = _stub_source(tmp_path)
    out_dir = tmp_path / "cache"

    assert _run_stub_compile(monkeypatch, source, out_dir, batch=1, kind="reranker") == 0
    capsys.readouterr()
    assert _run_stub_compile(monkeypatch, source, out_dir, batch=2, kind="reranker") == 0

    captured = capsys.readouterr()
    assert "batch_artifacts" not in captured.out
    # The record still describes the cache truthfully; only the served
    # configuration leaves the batched family out.
    assert "batch_artifacts" in _stub_model_info(out_dir, source)


# --- declared pooling recorded by a whole run ---------------------------------


def test_a_stub_run_records_the_declared_pooling_in_variant_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An embedding model's declared pooling must reach the variant metadata."""
    source = _stub_source(tmp_path, pooling_flag="pooling_mode_cls_token")
    out_dir = tmp_path / "cache"

    assert _run_stub_compile(monkeypatch, source, out_dir) == 0

    model_root = out_dir / "compiled" / source.name
    metadata_path = model_root / f"s{_FAMILY_SEQ_LEN}_b1_eager_macos13.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["variant"]["pooling"] == "cls"


def test_a_stub_run_records_the_declared_pooling_in_model_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An embedding model's declared pooling must reach model_info.json too."""
    source = _stub_source(tmp_path, pooling_flag="pooling_mode_mean_tokens")
    out_dir = tmp_path / "cache"

    assert _run_stub_compile(monkeypatch, source, out_dir) == 0

    assert _stub_model_info(out_dir, source)["pooling"] == "mean"


def test_a_reranker_stub_run_records_no_pooling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reranker's model_info.json must record pooling as null.

    Its pooling belongs to the model's own classification head, not a
    sentence-transformers declaration.
    """
    source = _stub_source(tmp_path)
    out_dir = tmp_path / "cache"

    assert _run_stub_compile(monkeypatch, source, out_dir, kind="reranker") == 0

    assert _stub_model_info(out_dir, source)["pooling"] is None


def test_a_stub_run_logs_the_declared_pooling_for_an_embedding_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The progress log must report the pooling mode an embedding model declares."""
    source = _stub_source(tmp_path, pooling_flag="pooling_mode_cls_token")
    out_dir = tmp_path / "cache"

    assert _run_stub_compile(monkeypatch, source, out_dir) == 0

    assert "pooling         : cls" in capsys.readouterr().err


def test_a_reranker_stub_run_logs_no_pooling_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A reranker never has a declared pooling to log, so no such line must appear."""
    source = _stub_source(tmp_path)
    out_dir = tmp_path / "cache"

    assert _run_stub_compile(monkeypatch, source, out_dir, kind="reranker") == 0

    assert "pooling" not in capsys.readouterr().err
