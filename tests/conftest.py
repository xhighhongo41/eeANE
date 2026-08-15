"""Shared helpers for the eeANE test suite: multi-model configs and a stub engine.

The server tests and the engine tests both need deployments with several
models per kind. The configuration builders and the deterministic stub
engine live here so both modules describe a deployment the same way, and
so the stub keeps implementing the very same
:class:`eeane.engine.InferenceEngine` protocol the real engine does.

Nothing here touches Core ML or reads a model file: the stub serves models
by id, and the entries built here point at paths that are never opened.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from eeane import runtime
from eeane.config import EeaneConfig, ModelEntry, ServerConfig, default_config
from eeane.engine import EmbeddingBatch, RerankBatch

# Stub vectors are intentionally low-dimensional: the endpoints treat the
# width as opaque, so 4 keeps the expected values readable.
STUB_DIM = 4

# Buckets the stub serves for a model of each kind, unless a test asks for
# something else. They mirror a realistic deployment: several buckets for
# the embedding models, one for the rerankers.
STUB_BUCKETS: dict[str, tuple[int, ...]] = {
    "embedding": (128, 512, 1024),
    "reranker": (512,),
}

# Root of the tokenizer/artifact paths handed to :func:`make_model_entry`.
# A configuration must carry paths, but no test ever opens them.
_FAKE_MODEL_ROOT = Path("/nonexistent/eeane-test-models")


def make_model_entry(
    model_id: str,
    kind: str = "embedding",
    *,
    normalize: bool = True,
    buckets: Sequence[int] = (128,),
    embedding_dim: int | None = None,
    excluded_buckets: Sequence[int] = (),
) -> ModelEntry:
    """Build one model entry pointing at paths that are never opened.

    Args:
        model_id: Id requests route by; also decides the (fake) paths.
        kind: ``"embedding"`` or ``"reranker"``.
        normalize: Embedding normalization flag. Only applied to an
            embedding entry: a reranker entry may not carry the flag.
        buckets: Sequence lengths the entry claims artifacts for.
        embedding_dim: Embedding width recorded for the entry, when known.
        excluded_buckets: Buckets the compiled-model cache recommends
            against loading (reported at start-up).

    Returns:
        A validated entry, ready to be put into a configuration.
    """
    model_dir = _FAKE_MODEL_ROOT / model_id
    fields: dict[str, object] = {
        "id": model_id,
        "kind": kind,
        "tokenizer": model_dir / "tokenizer.json",
        "artifacts": {bucket: model_dir / f"s{bucket}.mlmodelc" for bucket in buckets},
        "embedding_dim": embedding_dim,
        "excluded_buckets": tuple(excluded_buckets),
    }
    if kind == "embedding":
        fields["normalize"] = normalize
    return ModelEntry(**fields)


def make_config(
    *,
    normalize: bool = True,
    api_key: str | None = None,
    host: str = "127.0.0.1",
    health_rate_limit: int = 60,
    with_reranker: bool = True,
    extra_models: Sequence[ModelEntry] = (),
) -> EeaneConfig:
    """Build a test configuration by tweaking the built-in default.

    Args:
        normalize: Value of the default embedding entry's ``normalize``
            flag.
        api_key: Bearer key required by protected endpoints, or ``None``
            to keep the server unauthenticated.
        host: Bind address recorded in the config (only the start-up
            warning depends on it; nothing is actually bound in tests).
        health_rate_limit: ``/health`` requests allowed per minute per IP.
        with_reranker: When ``False``, drop the default reranker entry to
            build an embedding-only deployment.
        extra_models: Entries appended after the default ones, so the
            defaults stay first-listed (i.e. stay the default models).

    Returns:
        A validated configuration serving the default model ids plus
        ``extra_models``.
    """
    base = default_config()
    models: list[ModelEntry] = [base.embedding_model.model_copy(update={"normalize": normalize})]
    reranker = base.reranker_model
    if with_reranker and reranker is not None:
        models.append(reranker)
    models.extend(extra_models)
    return EeaneConfig(
        server=ServerConfig(host=host, api_key=api_key, health_rate_limit=health_rate_limit),
        models=models,
    )


def stub_vector(text: str) -> np.ndarray:
    """Build the deterministic embedding the stub returns for ``text``.

    Args:
        text: Input text; only its length matters.

    Returns:
        ``[len(text), len(text) + 1, ...]`` as a float32 vector of length
        :data:`STUB_DIM`.
    """
    return np.arange(STUB_DIM, dtype=np.float32) + float(len(text))


def stub_logit(document: str) -> float:
    """Build the deterministic logit the stub returns for ``document``.

    Args:
        document: Candidate document; only its length matters.

    Returns:
        A value in ``[-2.0, 2.0]`` derived from ``len(document)``.
    """
    return float(len(document) % 5) - 2.0


class StubEngine:
    """Deterministic multi-model engine implementing the InferenceEngine protocol.

    Serves exactly the models of the configuration it is built from, which
    keeps a test's deployment described in one place. Every call records
    the model id it was routed to, so a test can assert the routing
    without inspecting the returned values.

    Attributes:
        embed_model_ids: Resolved model id of every :meth:`embed` call, in
            call order.
        rerank_model_ids: Resolved model id of every :meth:`rerank` call,
            in call order.
    """

    def __init__(
        self, config: EeaneConfig, buckets: Mapping[str, Sequence[int]] | None = None
    ) -> None:
        """Register one stub model per configured entry.

        Args:
            config: Deployment to mirror. The first entry of a kind
                becomes that kind's default model, as in the real engine.
            buckets: Per-model-id bucket override; models left out serve
                the default buckets of their kind (:data:`STUB_BUCKETS`).
        """
        overrides = dict(buckets or {})
        self._kinds = {entry.id: str(entry.kind) for entry in config.models}
        self._buckets = {
            entry.id: tuple(overrides.get(entry.id, STUB_BUCKETS[str(entry.kind)]))
            for entry in config.models
        }
        self._defaults: dict[str, str] = {}
        for entry in config.models:
            self._defaults.setdefault(str(entry.kind), entry.id)
        self.embed_model_ids: list[str] = []
        self.rerank_model_ids: list[str] = []

    def buckets(self, model_id: str) -> tuple[int, ...]:
        """Return the buckets served by ``model_id`` (KeyError if unknown)."""
        return self._buckets[model_id]

    def default_model_id(self, kind: str) -> str | None:
        """Return the first-listed model id of ``kind``, or ``None``."""
        return self._defaults.get(kind)

    def embed(self, texts: list[str], model_id: str | None = None) -> EmbeddingBatch:
        """Return :func:`stub_vector` for each text, counting 1 token per character."""
        resolved = self._resolve("embedding", model_id)
        self.embed_model_ids.append(resolved)
        if not texts:
            return EmbeddingBatch(
                vectors=np.empty((0, STUB_DIM), dtype=np.float32),
                used_tokens=[],
                orig_tokens=[],
                buckets=[],
                truncated_indices=[],
            )
        tokens = [len(text) for text in texts]
        return EmbeddingBatch(
            vectors=np.stack([stub_vector(text) for text in texts]),
            used_tokens=list(tokens),
            orig_tokens=list(tokens),
            buckets=[runtime.select_bucket(n, self.buckets(resolved))[0] for n in tokens],
            truncated_indices=[],
        )

    def rerank(self, query: str, documents: list[str], model_id: str | None = None) -> RerankBatch:
        """Return :func:`stub_logit` for each document, ignoring the query's content."""
        resolved = self._resolve("reranker", model_id)
        self.rerank_model_ids.append(resolved)
        tokens = [len(query) + len(document) for document in documents]
        return RerankBatch(
            logits=np.asarray([stub_logit(document) for document in documents], dtype=np.float32),
            used_tokens=list(tokens),
            orig_tokens=list(tokens),
            truncated_indices=[],
        )

    def _resolve(self, kind: str, model_id: str | None) -> str:
        """Resolve a requested model id the way the real engine does.

        Args:
            kind: Kind the calling endpoint serves.
            model_id: Requested id, or ``None`` for the default model.

        Returns:
            The id of the model that serves the call.

        Raises:
            ValueError: If ``model_id`` is unknown or of another kind.
            RuntimeError: If no model of ``kind`` is served at all.
        """
        if model_id is None:
            default_id = self.default_model_id(kind)
            if default_id is None:
                raise RuntimeError(f"{kind} is not configured")
            return default_id
        if model_id not in self._kinds:
            raise ValueError(f"unknown model id '{model_id}'")
        if self._kinds[model_id] != kind:
            raise ValueError(f"model '{model_id}' is not a {kind} model")
        return model_id
