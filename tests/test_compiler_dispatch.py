"""Tests for eeane.compiler.dispatch."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from eeane.compiler import dispatch

# Module types of the sentence-transformers declaration a decoder-style
# model states its kind through. The same role is published under more
# than one namespace, which is why the declaration is matched by suffix.
_POOLING_MODULE = "sentence_transformers.models.Pooling"
_NESTED_POOLING_MODULE = "sentence_transformers.base.modules.pooling.Pooling"
_LOGIT_SCORE_MODULE = "sentence_transformers.cross_encoder.modules.logit_score.LogitScore"
_TRANSFORMER_MODULE = "sentence_transformers.models.Transformer"


def _write_config(model_dir: Path, config: object) -> Path:
    """Write a synthetic config.json into ``model_dir`` and return the directory."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return model_dir


def _module_entries(module_types: list[str]) -> list[dict[str, object]]:
    """Build the modules.json entries a sentence-transformers model ships."""
    return [
        {"idx": index, "name": str(index), "path": "", "type": module_type}
        for index, module_type in enumerate(module_types)
    ]


def _write_modules(model_dir: Path, declaration: object) -> Path:
    """Write ``declaration`` verbatim as the modules.json of ``model_dir``."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "modules.json").write_text(
        json.dumps(declaration, ensure_ascii=False), encoding="utf-8"
    )
    return model_dir


def _causal_model_dir(tmp_path: Path, module_types: list[str] | None = None) -> Path:
    """Build a decoder-style model directory, optionally with a module declaration."""
    model_dir = _write_config(tmp_path / "m", {"architectures": ["Qwen3ForCausalLM"]})
    if module_types is not None:
        _write_modules(model_dir, _module_entries(module_types))
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


def test_resolve_dispatch_detects_a_bert_embedding_model(tmp_path: Path) -> None:
    """A bare BertModel must be dispatched to the BERT backend as an embedding model."""
    model_dir = _write_config(tmp_path / "m", {"architectures": ["BertModel"]})

    result = dispatch.resolve_dispatch(model_dir)

    assert result.kind == dispatch.KIND_EMBEDDING
    assert result.architecture == "BertModel"
    assert result.backend_name == "Bert"


def test_resolve_dispatch_detects_a_bert_reranker(tmp_path: Path) -> None:
    """BertForSequenceClassification must be dispatched as a reranker.

    Dispatch resolves architecture and kind only; the BERT backend then
    refuses that kind (it compiles embedding models only), which the
    compile pipeline reports as a plain error message.
    """
    model_dir = _write_config(tmp_path / "m", {"architectures": ["BertForSequenceClassification"]})

    result = dispatch.resolve_dispatch(model_dir)

    assert result.kind == dispatch.KIND_RERANKER
    assert result.architecture == "BertForSequenceClassification"
    assert result.backend_name == "Bert"


@pytest.mark.parametrize(
    "architecture",
    ["ModernBertModel", "ModernBertForSequenceClassification"],
)
def test_resolve_dispatch_prefers_the_modernbert_backend_over_bert(
    tmp_path: Path, architecture: str
) -> None:
    """A ModernBERT name must never be captured by the shorter ``Bert`` registry key."""
    model_dir = _write_config(tmp_path / "m", {"architectures": [architecture]})

    assert dispatch.resolve_dispatch(model_dir).backend_name == "ModernBert"


def test_resolve_dispatch_detects_an_xlm_roberta_embedding_model(tmp_path: Path) -> None:
    """A bare XLMRobertaModel must be dispatched to the XLM-RoBERTa backend."""
    model_dir = _write_config(tmp_path / "m", {"architectures": ["XLMRobertaModel"]})

    result = dispatch.resolve_dispatch(model_dir)

    assert result.kind == dispatch.KIND_EMBEDDING
    assert result.architecture == "XLMRobertaModel"
    assert result.backend_name == "XLMRoberta"


def test_resolve_dispatch_detects_an_xlm_roberta_reranker(tmp_path: Path) -> None:
    """XLMRobertaForSequenceClassification must be dispatched as a reranker."""
    model_dir = _write_config(
        tmp_path / "m", {"architectures": ["XLMRobertaForSequenceClassification"]}
    )

    result = dispatch.resolve_dispatch(model_dir)

    assert result.kind == dispatch.KIND_RERANKER
    assert result.architecture == "XLMRobertaForSequenceClassification"
    assert result.backend_name == "XLMRoberta"


def test_resolve_dispatch_detects_a_roberta_embedding_model(tmp_path: Path) -> None:
    """A bare RobertaModel must be dispatched to the XLM-RoBERTa backend.

    RoBERTa and XLM-RoBERTa are structurally identical in the HF
    implementation (they differ only in vocabulary), so the XLM-RoBERTa
    backend covers RoBERTa checkpoints too. It is matched through its own
    ``Roberta`` registry key, but that key resolves to the very same
    backend class as ``XLMRoberta``.
    """
    model_dir = _write_config(tmp_path / "m", {"architectures": ["RobertaModel"]})

    result = dispatch.resolve_dispatch(model_dir)

    assert result.kind == dispatch.KIND_EMBEDDING
    assert result.architecture == "RobertaModel"
    assert result.backend_name == "Roberta"
    assert result.load_backend().name == "XLMRoberta"


def test_resolve_dispatch_detects_a_roberta_reranker(tmp_path: Path) -> None:
    """RobertaForSequenceClassification must be dispatched as a reranker on XLM-RoBERTa."""
    model_dir = _write_config(
        tmp_path / "m", {"architectures": ["RobertaForSequenceClassification"]}
    )

    result = dispatch.resolve_dispatch(model_dir)

    assert result.kind == dispatch.KIND_RERANKER
    assert result.architecture == "RobertaForSequenceClassification"
    assert result.backend_name == "Roberta"
    assert result.load_backend().name == "XLMRoberta"


# --- decoder-style models: the kind comes from the module declaration --------


def test_resolve_dispatch_detects_a_declared_embedding_decoder(tmp_path: Path) -> None:
    """A causal-LM directory declaring a pooling module is an embedding model."""
    model_dir = _causal_model_dir(tmp_path, [_TRANSFORMER_MODULE, _POOLING_MODULE])

    result = dispatch.resolve_dispatch(model_dir)

    assert result.kind == dispatch.KIND_EMBEDDING
    assert result.architecture == "Qwen3ForCausalLM"
    assert result.backend_name == "Qwen3"


def test_resolve_dispatch_detects_a_declared_reranker_decoder(tmp_path: Path) -> None:
    """A causal-LM directory declaring a logit-score module is a reranker."""
    model_dir = _causal_model_dir(tmp_path, [_TRANSFORMER_MODULE, _LOGIT_SCORE_MODULE])

    result = dispatch.resolve_dispatch(model_dir)

    assert result.kind == dispatch.KIND_RERANKER
    assert result.architecture == "Qwen3ForCausalLM"
    assert result.backend_name == "Qwen3"


def test_resolve_dispatch_matches_a_module_type_of_any_namespace(tmp_path: Path) -> None:
    """The same role is published under several namespaces, so only the suffix decides."""
    model_dir = _causal_model_dir(tmp_path, [_TRANSFORMER_MODULE, _NESTED_POOLING_MODULE])

    assert dispatch.resolve_dispatch(model_dir).kind == dispatch.KIND_EMBEDDING


def test_resolve_dispatch_refuses_a_declaration_naming_both_roles(tmp_path: Path) -> None:
    """A directory claiming both roles decides nothing and must ask for --kind."""
    model_dir = _causal_model_dir(
        tmp_path, [_TRANSFORMER_MODULE, _POOLING_MODULE, _LOGIT_SCORE_MODULE]
    )

    with pytest.raises(dispatch.KindDetectionError):
        dispatch.resolve_dispatch(model_dir)


def test_resolve_dispatch_undeclared_decoder_explains_the_declaration_and_kind(
    tmp_path: Path,
) -> None:
    """Without a declaration the kind is unknowable; the error must say so and how to fix it."""
    model_dir = _causal_model_dir(tmp_path, [_TRANSFORMER_MODULE])

    with pytest.raises(dispatch.KindDetectionError) as excinfo:
        dispatch.resolve_dispatch(model_dir)

    message = str(excinfo.value)
    assert "Qwen3ForCausalLM" in message
    assert "modules.json" in message
    assert "--kind" in message


def test_resolve_dispatch_explicit_kind_rescues_an_undeclared_decoder(tmp_path: Path) -> None:
    """An explicit --kind must make a directory without a declaration compilable."""
    model_dir = _causal_model_dir(tmp_path)

    result = dispatch.resolve_dispatch(model_dir, kind=dispatch.KIND_EMBEDDING)

    assert result.kind == dispatch.KIND_EMBEDDING
    assert result.backend_name == "Qwen3"


@pytest.mark.parametrize(
    "declaration",
    [
        [],
        {"type": _POOLING_MODULE},
        ["sentence_transformers.models.Pooling"],
        [{"idx": 0, "path": ""}],
        "not a list at all",
    ],
    ids=["empty", "object", "bare-strings", "typeless-entry", "not-a-list"],
)
def test_detect_kind_from_modules_ignores_an_unusable_declaration(
    tmp_path: Path, declaration: object
) -> None:
    """A declaration this reader cannot parse means 'undecided', never an exception."""
    model_dir = _write_modules(tmp_path / "m", declaration)

    assert dispatch.detect_kind_from_modules(model_dir) is None


def test_detect_kind_from_modules_ignores_a_corrupt_declaration(tmp_path: Path) -> None:
    """Unparsable JSON must degrade to 'undecided', not raise out of dispatch."""
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "modules.json").write_text("{not json", encoding="utf-8")

    assert dispatch.detect_kind_from_modules(model_dir) is None


def test_detect_kind_from_modules_ignores_an_undecodable_declaration(tmp_path: Path) -> None:
    """A file that is not even UTF-8 text must degrade to 'undecided', not raise."""
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "modules.json").write_bytes(b"\xff\xfe not utf-8")

    assert dispatch.detect_kind_from_modules(model_dir) is None


def test_detect_kind_from_modules_of_a_directory_without_a_declaration(tmp_path: Path) -> None:
    """Most published models declare no module chain at all."""
    assert dispatch.detect_kind_from_modules(tmp_path / "absent") is None


def test_a_module_declaration_does_not_override_an_encoder_architecture(tmp_path: Path) -> None:
    """Only a name that decides nothing may fall through to the declaration."""
    model_dir = _write_config(tmp_path / "m", {"architectures": ["ModernBertModel"]})
    _write_modules(model_dir, _module_entries([_TRANSFORMER_MODULE, _LOGIT_SCORE_MODULE]))

    result = dispatch.resolve_dispatch(model_dir)

    assert result.kind == dispatch.KIND_EMBEDDING


def test_a_module_declaration_does_not_rescue_an_encoder_of_an_undecidable_name(
    tmp_path: Path,
) -> None:
    """The declaration is read for decoder-style names only; nothing else changes."""
    model_dir = _write_config(tmp_path / "m", {"architectures": ["ModernBertForMaskedLM"]})
    _write_modules(model_dir, _module_entries([_TRANSFORMER_MODULE, _POOLING_MODULE]))

    with pytest.raises(dispatch.KindDetectionError):
        dispatch.resolve_dispatch(model_dir)


def test_no_registry_key_is_a_prefix_of_another_one() -> None:
    """Prefix matching stays unambiguous only while no key starts with another key."""
    keys = list(dispatch.BACKEND_REGISTRY)

    for key in keys:
        for other in keys:
            assert key == other or not other.startswith(key), (
                f"'{other}' starts with '{key}': prefix matching would depend on dict order"
            )


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        ("ModernBertModel", dispatch.KIND_EMBEDDING),
        ("BertModel", dispatch.KIND_EMBEDDING),
        ("XLMRobertaModel", dispatch.KIND_EMBEDDING),
        ("RobertaModel", dispatch.KIND_EMBEDDING),
        ("ModernBertForSequenceClassification", dispatch.KIND_RERANKER),
        ("BertForSequenceClassification", dispatch.KIND_RERANKER),
        ("XLMRobertaForSequenceClassification", dispatch.KIND_RERANKER),
        ("RobertaForSequenceClassification", dispatch.KIND_RERANKER),
        ("ModernBertForMaskedLM", None),
        ("BertForMaskedLM", None),
        ("XLMRobertaForMaskedLM", None),
        # A decoder-style embedding model and a decoder-style reranker
        # carry the very same architecture name, so it decides nothing.
        ("Qwen3ForCausalLM", None),
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
    """An unregistered architecture must raise and name every supported family."""
    model_dir = _write_config(tmp_path / "m", {"architectures": ["LlamaForCausalLM"]})

    with pytest.raises(dispatch.UnsupportedArchitectureError) as excinfo:
        dispatch.resolve_dispatch(model_dir)

    message = str(excinfo.value)
    assert "Unsupported architecture 'LlamaForCausalLM'" in message
    assert "BAAI/bge-large-en-v1.5" in message
    assert "ModernBERT" in message
    assert "cl-nagoya/ruri-v3-310m" in message
    assert "XLM-RoBERTa" in message
    assert "intfloat/multilingual-e5-base" in message
    assert "BAAI/bge-reranker-v2-m3" in message
    assert "Qwen3" in message
    assert "Qwen/Qwen3-Embedding-0.6B" in message
    assert "Qwen/Qwen3-Reranker-0.6B" in message


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


@pytest.mark.parametrize(
    ("architecture", "class_name", "backend_name"),
    [
        ("BertModel", "BertBackend", "Bert"),
        ("ModernBertModel", "ModernBertBackend", "ModernBert"),
        ("XLMRobertaModel", "XlmRobertaBackend", "XLMRoberta"),
        ("RobertaModel", "XlmRobertaBackend", "XLMRoberta"),
    ],
)
def test_dispatch_load_backend_returns_the_registered_backend(
    tmp_path: Path, architecture: str, class_name: str, backend_name: str
) -> None:
    """load_backend must import the registered class lazily and instantiate it."""
    model_dir = _write_config(tmp_path / "m", {"architectures": [architecture]})

    backend = dispatch.resolve_dispatch(model_dir).load_backend()

    assert type(backend).__name__ == class_name
    assert backend.name == backend_name
    for method in ("load", "apply_patches", "wrap", "trace_example", "reference_outputs"):
        assert callable(getattr(backend, method))


def test_dispatch_load_backend_returns_the_decoder_backend(tmp_path: Path) -> None:
    """A declared decoder-style embedding model must resolve to the Qwen3 backend."""
    model_dir = _causal_model_dir(tmp_path, [_TRANSFORMER_MODULE, _POOLING_MODULE])

    backend = dispatch.resolve_dispatch(model_dir).load_backend()

    assert type(backend).__name__ == "Qwen3Backend"
    assert backend.name == "Qwen3"
    assert backend.supported_kinds == (dispatch.KIND_EMBEDDING,)


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
