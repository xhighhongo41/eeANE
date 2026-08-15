"""Tests for eeane.compiler.dispatch (v0.6 T3, see 開発資料/v0.6実装計画.md §4.2, §4.9)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from eeane.compiler import dispatch


def _write_config(model_dir: Path, config: object) -> Path:
    """Write a synthetic config.json into ``model_dir`` and return the directory."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return model_dir


# --- architecture -> kind detection -----------------------------------------


def test_resolve_dispatch_detects_embedding_kind(tmp_path: Path) -> None:
    """A bare ModernBertModel must be dispatched as an embedding model."""
    model_dir = _write_config(tmp_path / "m", {"architectures": ["ModernBertModel"]})

    result = dispatch.resolve_dispatch(model_dir)

    assert result.kind == dispatch.KIND_EMBEDDING
    assert result.architecture == "ModernBertModel"
    assert result.backend_name == "ModernBert"


def test_resolve_dispatch_detects_reranker_kind(tmp_path: Path) -> None:
    """A ...ForSequenceClassification architecture must be dispatched as a reranker."""
    model_dir = _write_config(
        tmp_path / "m", {"architectures": ["ModernBertForSequenceClassification"]}
    )

    result = dispatch.resolve_dispatch(model_dir)

    assert result.kind == dispatch.KIND_RERANKER
    assert result.architecture == "ModernBertForSequenceClassification"
    assert result.backend_name == "ModernBert"


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("ModernBertModel", dispatch.KIND_EMBEDDING),
        ("BertModel", dispatch.KIND_EMBEDDING),
        ("ModernBertForSequenceClassification", dispatch.KIND_RERANKER),
        ("ModernBertForMaskedLM", None),
        ("", None),
    ],
)
def test_detect_kind_rules(architecture: str, expected: str | None) -> None:
    """detect_kind must follow the suffix rules and return None when undecidable."""
    assert dispatch.detect_kind(architecture) == expected


def test_resolve_dispatch_undetectable_kind_asks_for_kind_option(tmp_path: Path) -> None:
    """An architecture matching no kind rule must raise and point at --kind."""
    model_dir = _write_config(tmp_path / "m", {"architectures": ["ModernBertForMaskedLM"]})

    with pytest.raises(dispatch.KindDetectionError) as excinfo:
        dispatch.resolve_dispatch(model_dir)

    message = str(excinfo.value)
    assert "--kind" in message
    assert "ModernBertForMaskedLM" in message


def test_resolve_dispatch_ambiguous_architectures_ask_for_kind_option(tmp_path: Path) -> None:
    """Architectures implying different kinds must raise instead of guessing."""
    model_dir = _write_config(
        tmp_path / "m",
        {"architectures": ["ModernBertModel", "ModernBertForSequenceClassification"]},
    )

    with pytest.raises(dispatch.KindDetectionError) as excinfo:
        dispatch.resolve_dispatch(model_dir)

    assert "--kind" in str(excinfo.value)


# --- explicit --kind ---------------------------------------------------------


def test_resolve_dispatch_explicit_kind_wins_and_warns_on_conflict(tmp_path: Path) -> None:
    """An explicit --kind must override detection but warn about the contradiction."""
    model_dir = _write_config(
        tmp_path / "m", {"architectures": ["ModernBertForSequenceClassification"]}
    )

    with pytest.warns(UserWarning) as record:
        result = dispatch.resolve_dispatch(model_dir, kind=dispatch.KIND_EMBEDDING)

    assert result.kind == dispatch.KIND_EMBEDDING
    message = str(record[0].message)
    assert "embedding" in message
    assert "reranker" in message
    assert "ModernBertForSequenceClassification" in message


def test_resolve_dispatch_explicit_kind_matching_detection_does_not_warn(tmp_path: Path) -> None:
    """A matching explicit --kind must not produce a warning."""
    model_dir = _write_config(tmp_path / "m", {"architectures": ["ModernBertModel"]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = dispatch.resolve_dispatch(model_dir, kind=dispatch.KIND_EMBEDDING)

    assert result.kind == dispatch.KIND_EMBEDDING
    assert caught == []


def test_resolve_dispatch_explicit_kind_rescues_undetectable_architecture(tmp_path: Path) -> None:
    """An explicit --kind must make an otherwise undecidable model compilable."""
    model_dir = _write_config(tmp_path / "m", {"architectures": ["ModernBertForMaskedLM"]})

    result = dispatch.resolve_dispatch(model_dir, kind=dispatch.KIND_RERANKER)

    assert result.kind == dispatch.KIND_RERANKER


def test_resolve_dispatch_rejects_unknown_kind_value(tmp_path: Path) -> None:
    """A kind outside {auto, embedding, reranker} must raise ValueError."""
    model_dir = _write_config(tmp_path / "m", {"architectures": ["ModernBertModel"]})

    with pytest.raises(ValueError, match="kind"):
        dispatch.resolve_dispatch(model_dir, kind="classifier")


# --- unsupported / malformed configs ----------------------------------------


def test_resolve_dispatch_unsupported_architecture_lists_supported_models(tmp_path: Path) -> None:
    """An unregistered architecture must raise and name what v0.6 supports."""
    model_dir = _write_config(tmp_path / "m", {"architectures": ["LlamaForCausalLM"]})

    with pytest.raises(dispatch.UnsupportedArchitectureError) as excinfo:
        dispatch.resolve_dispatch(model_dir)

    message = str(excinfo.value)
    assert "Unsupported architecture 'LlamaForCausalLM'" in message
    assert "ModernBERT" in message
    assert "cl-nagoya/ruri-v3-310m" in message


def test_resolve_dispatch_missing_config_json(tmp_path: Path) -> None:
    """A directory without config.json must raise a config error naming the file."""
    model_dir = tmp_path / "empty"
    model_dir.mkdir()

    with pytest.raises(dispatch.ModelConfigError) as excinfo:
        dispatch.resolve_dispatch(model_dir)

    assert "config.json" in str(excinfo.value)


def test_resolve_dispatch_invalid_json(tmp_path: Path) -> None:
    """A malformed config.json must raise a config error, not a JSONDecodeError."""
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(dispatch.ModelConfigError) as excinfo:
        dispatch.resolve_dispatch(model_dir)

    assert "config.json" in str(excinfo.value)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"architectures": []},
        {"architectures": "ModernBertModel"},
        {"architectures": [123]},
        ["ModernBertModel"],
    ],
    ids=["absent", "empty", "not-a-list", "non-string-entry", "not-a-mapping"],
)
def test_resolve_dispatch_bad_architectures_field(tmp_path: Path, config: object) -> None:
    """Any config without a usable ``architectures`` list must raise a config error."""
    model_dir = _write_config(tmp_path / "m", config)

    with pytest.raises(dispatch.ModelConfigError) as excinfo:
        dispatch.resolve_dispatch(model_dir)

    assert "architectures" in str(excinfo.value)


# --- backend loading ---------------------------------------------------------


def test_dispatch_load_backend_returns_modernbert_backend(tmp_path: Path) -> None:
    """load_backend must import the registered class lazily and instantiate it."""
    model_dir = _write_config(tmp_path / "m", {"architectures": ["ModernBertModel"]})

    backend = dispatch.resolve_dispatch(model_dir).load_backend()

    assert type(backend).__name__ == "ModernBertBackend"
    assert backend.name == "ModernBert"
    for method in ("load", "apply_patches", "wrap", "trace_example", "reference_outputs"):
        assert callable(getattr(backend, method))


def test_dispatch_module_does_not_import_torch() -> None:
    """Importing eeane.compiler.dispatch alone must not pull in torch/transformers."""
    import subprocess
    import sys

    script = (
        "import eeane.compiler.dispatch\n"
        "import sys\n"
        "assert 'torch' not in sys.modules, sorted(sys.modules)\n"
        "assert 'transformers' not in sys.modules, sorted(sys.modules)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr
