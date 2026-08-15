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
    "ModernBERT (e.g. cl-nagoya/ruri-v3-310m) and "
    "XLM-RoBERTa (e.g. intfloat/multilingual-e5-base, BAAI/bge-reranker-v2-m3)"
)

# Architecture-name prefix -> "module:attribute" of the backend class. The
# value is a string so that selecting a backend never imports torch. No key
# may start with another key, or prefix matching would depend on the order
# of this mapping.
BACKEND_REGISTRY: dict[str, str] = {
    "ModernBert": "eeane.compiler.backends.modernbert:ModernBertBackend",
    "XLMRoberta": "eeane.compiler.backends.xlm_roberta:XlmRobertaBackend",
}

# Architecture-name suffix that identifies a cross-encoder reranker.
_RERANKER_SUFFIX = "ForSequenceClassification"

# Architecture-name suffix of a bare backbone model (embedding kind).
_EMBEDDING_SUFFIX = "Model"


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

    if kind == KIND_AUTO:
        if detected is None:
            raise KindDetectionError(
                f"cannot tell whether '{architecture}' is an embedding model or a reranker "
                f"(architectures: {', '.join(architectures)}); rerun with "
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
