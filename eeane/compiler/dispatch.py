"""Compile backend dispatch.

Reads the ``config.json`` of a HuggingFace-format model directory and
decides (a) which compile backend implements that architecture and (b)
whether the model is an embedding model or a reranker.

This module deliberately stays free of ``torch``/``transformers`` imports:
it only parses JSON, and the backend class itself is imported lazily by
:meth:`Dispatch.load_backend`. That keeps ``eeane compile``'s failure
modes for unsupported models fast and dependency-free.
"""

from __future__ import annotations

import importlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Model kinds (``auto`` is only accepted as the *requested* kind).
KIND_AUTO = "auto"
KIND_EMBEDDING = "embedding"
KIND_RERANKER = "reranker"
KINDS: tuple[str, ...] = (KIND_EMBEDDING, KIND_RERANKER)

# Human-readable list of the architectures a backend is registered for,
# used in the "unsupported architecture" error message.
SUPPORTED_ARCHITECTURES = (
    "BERT embedding models (e.g. BAAI/bge-large-en-v1.5), "
    "ModernBERT (e.g. cl-nagoya/ruri-v3-310m), "
    "XLM-RoBERTa / RoBERTa (e.g. intfloat/multilingual-e5-base, BAAI/bge-reranker-v2-m3) and "
    "Qwen3 decoder-style models (e.g. Qwen/Qwen3-Embedding-0.6B, Qwen/Qwen3-Reranker-0.6B)"
)

# Architecture-name prefix -> "module:attribute" of the backend class. The
# value is a string so that selecting a backend never imports torch. No key
# may start with another key, or prefix matching would depend on the order
# of this mapping.
BACKEND_REGISTRY: dict[str, str] = {
    "Bert": "eeane.compiler.backends.bert:BertBackend",
    "ModernBert": "eeane.compiler.backends.modernbert:ModernBertBackend",
    "XLMRoberta": "eeane.compiler.backends.xlm_roberta:XlmRobertaBackend",
    "Roberta": "eeane.compiler.backends.xlm_roberta:XlmRobertaBackend",
    "Qwen3": "eeane.compiler.backends.qwen3:Qwen3Backend",
}

# Architecture-name suffix that identifies a cross-encoder reranker.
_RERANKER_SUFFIX = "ForSequenceClassification"

# Architecture-name suffix of a bare backbone model (embedding kind).
_EMBEDDING_SUFFIX = "Model"

# Architecture-name suffix of a decoder-style (causal language model)
# checkpoint. Such a name says nothing about the kind: a decoder-style
# embedding model and a decoder-style reranker are published under the
# very same name, and only the model directory itself tells them apart.
_CAUSAL_LM_SUFFIX = "ForCausalLM"

# sentence-transformers module declaration, and the module types that
# state which of the two roles a model plays. The declaration lives next
# to config.json and lists the modules the model applies, in order.
#
# The types are matched by suffix because the same role is published under
# more than one namespace: a pooling module is spelled
# ``sentence_transformers.models.Pooling`` by one publisher and
# ``sentence_transformers.base.modules.pooling.Pooling`` by another, and
# the scoring module of a cross-encoder appears as
# ``sentence_transformers.cross_encoder.modules.logit_score.LogitScore``.
# Matching the trailing class name keeps every spelling recognizable.
_MODULES_FILENAME = "modules.json"
_MODULE_TYPE_KEY = "type"
_POOLING_MODULE_SUFFIX = ".Pooling"
_LOGIT_SCORE_MODULE_SUFFIX = ".LogitScore"


class DispatchError(RuntimeError):
    """Base class for every backend/kind resolution failure."""


class ModelConfigError(DispatchError):
    """Raised when ``config.json`` is missing, unreadable, or incomplete."""


class UnsupportedArchitectureError(DispatchError):
    """Raised when no registered backend implements the architecture."""


class KindDetectionError(DispatchError):
    """Raised when the model kind cannot be inferred from the architecture."""


@dataclass(frozen=True)
class Dispatch:
    """Resolved backend and model kind for one model directory.

    Attributes:
        architecture: The ``config.json`` architecture the backend was
            selected for (e.g. ``ModernBertForSequenceClassification``).
        kind: ``"embedding"`` or ``"reranker"``.
        backend_name: Registry key of the backend (e.g. ``ModernBert``).
        backend_target: ``"module:attribute"`` of the backend class,
            imported on demand by :meth:`load_backend`.
    """

    architecture: str
    kind: str
    backend_name: str
    backend_target: str

    def load_backend(self) -> Any:
        """Import and instantiate the selected backend class.

        The import happens here (and not at module import time) because
        backends pull in ``torch``/``transformers``, which only the
        ``[compile]`` extra provides. The return type stays untyped on
        purpose: annotating it would require importing the backend module
        (and thus torch) into this one.

        Returns:
            A new backend instance (e.g. ``ModernBertBackend()``).
        """
        module_name, _, attribute = self.backend_target.partition(":")
        module = importlib.import_module(module_name)
        return getattr(module, attribute)()


def read_architectures(model_dir: Path) -> list[str]:
    """Read the ``architectures`` list from a model directory's config.json.

    Args:
        model_dir: HuggingFace-format model directory.

    Returns:
        The non-empty list of architecture names, in config.json order.

    Raises:
        ModelConfigError: If config.json is missing, unreadable, not a JSON
            object, or has no usable ``architectures`` list.
    """
    config_path = model_dir / "config.json"
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelConfigError(
            f"cannot read {config_path}: {exc}. "
            "eeane compile expects a HuggingFace-format model directory."
        ) from exc
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelConfigError(f"{config_path} is not valid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise ModelConfigError(
            f"{config_path} must contain a JSON object with an 'architectures' list"
        )
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ModelConfigError(
            f"{config_path} has no non-empty 'architectures' list; "
            "eeane compile cannot tell which model this is"
        )
    if not all(isinstance(name, str) and name for name in architectures):
        raise ModelConfigError(f"{config_path} has a non-string entry in 'architectures'")
    return list(architectures)


def select_backend(architectures: list[str]) -> tuple[str, str, str]:
    """Pick the backend implementing one of ``architectures``.

    Matching is by name prefix (``ModernBertModel``,
    ``ModernBertForSequenceClassification``, ... all map to the
    ``ModernBert`` backend); the first matching entry wins, which is
    unambiguous as long as no registry key starts with another one.

    Args:
        architectures: Architecture names from config.json.

    Returns:
        Tuple of (architecture, backend name, backend target).

    Raises:
        UnsupportedArchitectureError: If no registered backend matches.
    """
    for architecture in architectures:
        for prefix, target in BACKEND_REGISTRY.items():
            if architecture.startswith(prefix):
                return architecture, prefix, target
    raise UnsupportedArchitectureError(
        f"Unsupported architecture '{architectures[0]}'. "
        f"eeane compile supports: {SUPPORTED_ARCHITECTURES}."
    )


def detect_kind(architecture: str) -> str | None:
    """Infer the model kind from a single architecture name.

    Args:
        architecture: An architecture name from config.json.

    Returns:
        ``"reranker"`` for ``...ForSequenceClassification`` (cross-encoder
        head), ``"embedding"`` for a bare backbone (``...Model``), or
        ``None`` when the name says neither (e.g. ``...ForMaskedLM``).
    """
    if architecture.endswith(_RERANKER_SUFFIX):
        return KIND_RERANKER
    if architecture.endswith(_EMBEDDING_SUFFIX):
        return KIND_EMBEDDING
    return None


def detect_kind_from_modules(model_dir: Path) -> str | None:
    """Infer the model kind from the sentence-transformers module declaration.

    Some checkpoints carry the same architecture name whatever they are
    for, so their kind has to come from what the directory says it does: a
    module chain ending in a pooling module produces a sentence vector, a
    chain ending in a logit-score module produces a relevance score.

    Nothing here raises. A directory that declares no module chain (the
    common case) or one this reader cannot parse simply does not answer
    the question, and the caller falls back to asking for an explicit
    kind.

    Args:
        model_dir: HuggingFace-format model directory.

    Returns:
        ``"embedding"`` if a pooling module is declared, ``"reranker"`` if
        a logit-score module is, and ``None`` when neither or both are.
    """
    module_types = _read_module_types(model_dir / _MODULES_FILENAME)
    pools = any(name.endswith(_POOLING_MODULE_SUFFIX) for name in module_types)
    scores = any(name.endswith(_LOGIT_SCORE_MODULE_SUFFIX) for name in module_types)
    if pools == scores:
        # Neither role is declared, or both are: either way the
        # declaration does not decide anything.
        return None
    return KIND_EMBEDDING if pools else KIND_RERANKER


def _read_module_types(modules_path: Path) -> list[str]:
    """Read the module types a ``modules.json`` declares, in declared order.

    Args:
        modules_path: The declaration file; it need not exist.

    Returns:
        One type name per declared module. Entries that name no type are
        left out, and the list is empty when the file is absent,
        unreadable, undecodable, not valid JSON, or not a list at all.
    """
    try:
        raw = modules_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # Anything the file itself makes impossible -- absent, a
        # directory, not text -- is simply an unanswered question here.
        return []
    try:
        declaration = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(declaration, list):
        return []
    types: list[str] = []
    for entry in declaration:
        module_type = entry.get(_MODULE_TYPE_KEY) if isinstance(entry, dict) else None
        if isinstance(module_type, str):
            types.append(module_type)
    return types


def resolve_dispatch(model_dir: Path, kind: str = KIND_AUTO) -> Dispatch:
    """Resolve the compile backend and model kind for ``model_dir``.

    An explicit ``kind`` always wins over detection, but a contradiction
    (e.g. ``--kind embedding`` for a ``...ForSequenceClassification``
    model) is reported with a :class:`UserWarning` because it is far more
    often a mistake than a deliberate choice.

    Args:
        model_dir: HuggingFace-format model directory.
        kind: ``"auto"`` (detect), ``"embedding"``, or ``"reranker"``.

    Returns:
        The resolved :class:`Dispatch`.

    Raises:
        ValueError: If ``kind`` is not one of auto/embedding/reranker.
        ModelConfigError: If config.json is missing or unusable.
        UnsupportedArchitectureError: If no backend implements the model.
        KindDetectionError: If ``kind`` is ``auto`` and the architecture(s)
            do not determine the kind unambiguously.
    """
    if kind != KIND_AUTO and kind not in KINDS:
        supported = ", ".join((KIND_AUTO, *KINDS))
        raise ValueError(f"unknown kind '{kind}' (expected one of: {supported})")

    architectures = read_architectures(model_dir)
    architecture, backend_name, backend_target = select_backend(architectures)
    detected = _detect_kind_from_all(architectures)
    if detected is None and any(name.endswith(_CAUSAL_LM_SUFFIX) for name in architectures):
        # A decoder-style name never implies a kind, so this is the one
        # case where the model directory's own declaration is consulted.
        # Every other architecture keeps deciding by name alone.
        detected = detect_kind_from_modules(model_dir)

    if kind == KIND_AUTO:
        if detected is None:
            raise KindDetectionError(
                f"cannot tell whether '{architecture}' is an embedding model or a reranker "
                f"(architectures: {', '.join(architectures)}). A decoder-style checkpoint "
                "carries the same architecture name whichever it is, and states its role in "
                f"the sentence-transformers module declaration '{_MODULES_FILENAME}' of the "
                "model directory: a pooling module for an embedding model, a logit-score "
                "module for a reranker. For a directory that declares neither, rerun with "
                "--kind embedding or --kind reranker"
            )
        resolved_kind = detected
    else:
        if detected is not None and detected != kind:
            # Not fatal: the user may know better than the config, but a
            # silent mismatch would produce a subtly wrong graph.
            warnings.warn(
                f"--kind {kind} contradicts architecture '{architecture}', "
                f"which looks like a {detected} model; continuing with {kind}",
                UserWarning,
                stacklevel=2,
            )
        resolved_kind = kind

    return Dispatch(
        architecture=architecture,
        kind=resolved_kind,
        backend_name=backend_name,
        backend_target=backend_target,
    )


def _detect_kind_from_all(architectures: list[str]) -> str | None:
    """Detect one kind from every architecture entry, or None if ambiguous.

    Args:
        architectures: Architecture names from config.json.

    Returns:
        The single kind implied by the entries, or ``None`` when no entry
        implies one or when the entries disagree.
    """
    kinds = {detected for detected in map(detect_kind, architectures) if detected is not None}
    if len(kinds) == 1:
        return kinds.pop()
    return None
