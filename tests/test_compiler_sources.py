"""Tests for eeane.compiler.sources."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import huggingface_hub
import pytest

from eeane.compiler import sources


def _make_snapshot(root: Path, filenames: list[str]) -> Path:
    """Create a fake snapshot directory holding ``filenames``."""
    root.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (root / name).write_text("{}", encoding="utf-8")
    return root


@pytest.fixture
def download_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Patch huggingface_hub.snapshot_download and record how it is called.

    ``kwargs`` holds the most recent call (for tests that only make one);
    ``calls`` holds every call in order (for tests asserting on a second,
    ``--allow-pickle``-triggered download).
    """
    calls: dict[str, Any] = {"kwargs": None, "calls": [], "snapshot": tmp_path / "snapshot"}

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls["kwargs"] = kwargs
        calls["calls"].append(kwargs)
        return str(calls["snapshot"])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    return calls


# --- local directories -------------------------------------------------------


def test_resolve_source_returns_existing_directory(tmp_path: Path) -> None:
    """An existing directory must be returned as an absolute path, unchanged."""
    model_dir = tmp_path / "ruri-v3-310m"
    _make_snapshot(model_dir, ["config.json"])

    resolved = sources.resolve_source(str(model_dir))

    assert resolved == model_dir.resolve()


def test_resolve_source_does_not_write_into_the_model_directory(tmp_path: Path) -> None:
    """Resolution must leave a local model directory byte-for-byte untouched."""
    model_dir = tmp_path / "ruri-v3-310m"
    _make_snapshot(model_dir, ["config.json"])
    before = {p.name: p.stat().st_mtime_ns for p in model_dir.iterdir()}

    sources.resolve_source(str(model_dir))

    assert {p.name: p.stat().st_mtime_ns for p in model_dir.iterdir()} == before


def test_resolve_source_local_directory_bin_only_without_allow_pickle_is_rejected(
    tmp_path: Path,
) -> None:
    """A local bin-only directory must be rejected unless --allow-pickle was passed."""
    model_dir = tmp_path / "local-model"
    _make_snapshot(model_dir, ["config.json", "pytorch_model.bin"])

    with pytest.raises(sources.MissingSafetensorsError) as excinfo:
        sources.resolve_source(str(model_dir))

    assert "--allow-pickle" in str(excinfo.value)


def test_resolve_source_local_directory_bin_only_with_allow_pickle_warns_and_succeeds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """--allow-pickle must accept a local bin-only directory, after logging a WARNING."""
    model_dir = tmp_path / "local-model"
    _make_snapshot(model_dir, ["config.json", "pytorch_model.bin"])

    with caplog.at_level(logging.WARNING):
        resolved = sources.resolve_source(str(model_dir), allow_pickle=True)

    assert resolved == model_dir.resolve()
    assert any(
        record.levelno == logging.WARNING and "pickle" in record.message.lower()
        for record in caplog.records
    )


@pytest.mark.parametrize("allow_pickle", [False, True])
def test_resolve_source_local_directory_with_safetensors_ignores_allow_pickle(
    tmp_path: Path, allow_pickle: bool
) -> None:
    """A local directory with safetensors must be accepted regardless of --allow-pickle."""
    model_dir = tmp_path / "local-model"
    _make_snapshot(model_dir, ["config.json", "model.safetensors"])

    resolved = sources.resolve_source(str(model_dir), allow_pickle=allow_pickle)

    assert resolved == model_dir.resolve()


def test_resolve_source_local_directory_without_any_weights_is_passed_through(
    tmp_path: Path,
) -> None:
    """A directory with neither safetensors nor .bin weights is left to transformers."""
    model_dir = tmp_path / "local-model"
    _make_snapshot(model_dir, ["config.json"])

    assert sources.resolve_source(str(model_dir)) == model_dir.resolve()


def test_resolve_source_rejects_a_file(tmp_path: Path) -> None:
    """An existing path that is not a directory must raise."""
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(sources.SourceError) as excinfo:
        sources.resolve_source(str(path))

    assert "directory" in str(excinfo.value)


@pytest.mark.parametrize(
    "source",
    ["./missing/model", "/nonexistent/eeane/model", "", "   ", "not/a/repo/id", "-bad/name"],
)
def test_resolve_source_rejects_unusable_sources(source: str) -> None:
    """Anything that is neither an existing directory nor a repo id must raise."""
    with pytest.raises(sources.SourceError):
        sources.resolve_source(source)


# --- repo id detection -------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    ["cl-nagoya/ruri-v3-310m", "org/name", "Org_1/model.v2", "a/b"],
)
def test_looks_like_hf_repo_id_accepts_repo_ids(source: str) -> None:
    """Well-formed ``org/name`` strings must be recognised as repo ids."""
    assert sources.looks_like_hf_repo_id(source) is True


@pytest.mark.parametrize(
    "source",
    ["./org/name", "/org/name", "org/name/extra", "orgname", "org/", "/name", "", "  /  "],
)
def test_looks_like_hf_repo_id_rejects_paths(source: str) -> None:
    """Path-like strings and malformed ids must not be treated as repo ids."""
    assert sources.looks_like_hf_repo_id(source) is False


def test_looks_like_hf_repo_id_is_shape_only(tmp_path: Path) -> None:
    """A relative path can be repo-id shaped; resolve_source checks the disk first."""
    model_dir = tmp_path / "models" / "ruri"
    _make_snapshot(model_dir, ["config.json"])

    assert sources.looks_like_hf_repo_id("models/ruri") is True
    # The existing directory wins over the repo-id interpretation.
    assert sources.resolve_source(str(model_dir)) == model_dir.resolve()


# --- Hub downloads (mocked) --------------------------------------------------


def test_resolve_source_downloads_repo_id_with_allow_patterns(
    download_calls: dict[str, Any],
) -> None:
    """A repo id must be fetched via snapshot_download with the restricted patterns."""
    _make_snapshot(download_calls["snapshot"], ["config.json", "model.safetensors"])

    resolved = sources.resolve_source("cl-nagoya/ruri-v3-310m")

    assert resolved == Path(download_calls["snapshot"]).resolve()
    kwargs = download_calls["kwargs"]
    assert kwargs["repo_id"] == "cl-nagoya/ruri-v3-310m"
    assert list(kwargs["allow_patterns"]) == list(sources.HF_ALLOW_PATTERNS)
    assert "config.json" in kwargs["allow_patterns"]
    assert "*.safetensors" in kwargs["allow_patterns"]


def test_resolve_source_passes_revision_through(download_calls: dict[str, Any]) -> None:
    """An explicit revision must reach snapshot_download unmodified."""
    _make_snapshot(download_calls["snapshot"], ["config.json", "model.safetensors"])

    sources.resolve_source("org/name", revision="refs/pr/1")

    assert download_calls["kwargs"]["revision"] == "refs/pr/1"


def test_resolve_source_accepts_sharded_safetensors(download_calls: dict[str, Any]) -> None:
    """A sharded checkpoint (index + shards) must be accepted."""
    _make_snapshot(
        download_calls["snapshot"],
        [
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
    )

    assert sources.resolve_source("org/name") == Path(download_calls["snapshot"]).resolve()


def test_resolve_source_rejects_snapshot_without_safetensors(
    download_calls: dict[str, Any],
) -> None:
    """A repo whose safetensors-only download carries no weights must raise a clear error."""
    _make_snapshot(download_calls["snapshot"], ["config.json"])

    with pytest.raises(sources.MissingSafetensorsError) as excinfo:
        sources.resolve_source("org/name")

    message = str(excinfo.value)
    assert "safetensors" in message
    assert "org/name" in message
    assert "--allow-pickle" in message


def test_resolve_source_hf_bin_only_without_allow_pickle_error_mentions_allow_pickle(
    download_calls: dict[str, Any],
) -> None:
    """A bin-only repo's error message must point at --allow-pickle as the opt-in."""
    _make_snapshot(download_calls["snapshot"], ["config.json", "pytorch_model.bin"])

    with pytest.raises(sources.MissingSafetensorsError) as excinfo:
        sources.resolve_source("org/name")

    assert "--allow-pickle" in str(excinfo.value)
    # Only the safetensors-only patterns must have been requested; the fallback
    # download only happens with --allow-pickle.
    assert len(download_calls["calls"]) == 1


def test_resolve_source_hf_bin_only_with_allow_pickle_redownloads_with_bin_patterns(
    download_calls: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """--allow-pickle must trigger a second download that also requests .bin weights."""
    _make_snapshot(download_calls["snapshot"], ["config.json", "pytorch_model.bin"])

    with caplog.at_level(logging.WARNING):
        resolved = sources.resolve_source("org/name", allow_pickle=True)

    assert resolved == Path(download_calls["snapshot"]).resolve()
    calls = download_calls["calls"]
    assert len(calls) == 2
    assert list(calls[0]["allow_patterns"]) == list(sources.HF_ALLOW_PATTERNS)
    assert "*.bin" not in calls[0]["allow_patterns"]
    assert "*.bin" in calls[1]["allow_patterns"]
    assert "*.bin.index.json" in calls[1]["allow_patterns"]
    assert calls[1]["repo_id"] == "org/name"
    assert any(
        record.levelno == logging.WARNING and "pickle" in record.message.lower()
        for record in caplog.records
    )


def test_resolve_source_hf_with_safetensors_does_not_redownload_even_with_allow_pickle(
    download_calls: dict[str, Any],
) -> None:
    """A repo that already has safetensors must not trigger a second (.bin) download."""
    _make_snapshot(download_calls["snapshot"], ["config.json", "model.safetensors"])

    resolved = sources.resolve_source("org/name", allow_pickle=True)

    assert resolved == Path(download_calls["snapshot"]).resolve()
    assert len(download_calls["calls"]) == 1


def test_resolve_source_wraps_download_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Hub failure must surface as a SourceError naming the repo id."""

    def boom(**_kwargs: Any) -> str:
        raise OSError("network unreachable")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)

    with pytest.raises(sources.SourceError) as excinfo:
        sources.resolve_source("org/name")

    assert "org/name" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)


def test_resolve_source_errors_when_snapshot_path_is_missing(
    download_calls: dict[str, Any],
) -> None:
    """A snapshot path that does not exist must raise instead of being returned."""
    with pytest.raises(sources.SourceError):
        sources.resolve_source("org/name")
