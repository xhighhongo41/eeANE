"""Core ML inference engine for the eeANE server.

The engine owns everything that touches Core ML: artifact validation,
tokenizer/model loading, sequence-length bucket routing and the
process-wide lock that serializes every ``predict`` call (the Neural
Engine runs one prediction at a time anyway, so concurrent calls buy
nothing).

Any number of embedding and reranker models can be served at once: every
configured entry becomes one resident model, and a request is routed to
it by model id, ``None`` meaning "the first-listed model of the kind the
endpoint serves".

The HTTP layer only sees :class:`InferenceEngine`, so tests can inject a
deterministic stub and a future on-demand-loading engine can replace
:class:`CoreMLEngine` without touching ``eeane.server``.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import coremltools as ct
import numpy as np

from eeane import runtime
from eeane.config import EeaneConfig, ModelEntry

# Conversion command quoted in the "missing artifact" errors, so the
# operator can regenerate what is missing without reading the docs.
_COMPILE_COMMAND = "eeane compile <model> --buckets {buckets}"

# Output tensor name assumed for a model whose entry does not state one.
_DEFAULT_OUTPUT_NAMES = {"embedding": "embedding", "reranker": "logits"}


@dataclass
class EmbeddingBatch:
    """Result of embedding one request's worth of texts.

    Attributes:
        vectors: Raw (un-normalized) embeddings of shape ``(N, D)``,
            dtype float32, in request order.
        used_tokens: Per-input token count actually fed to the model
            (sum of ``attention_mask``, i.e. after truncation).
        orig_tokens: Per-input token count before truncation.
        buckets: Per-input sequence-length bucket used for inference.
        truncated_indices: Indices of the inputs that did not fit into the
            largest bucket and were truncated.
    """

    vectors: np.ndarray
    used_tokens: list[int]
    orig_tokens: list[int]
    buckets: list[int]
    truncated_indices: list[int]


@dataclass
class RerankBatch:
    """Result of scoring one request's worth of (query, document) pairs.

    Attributes:
        logits: Raw cross-encoder logits of shape ``(N,)``, dtype float32,
            in request order (sigmoid mapping is the caller's choice).
        used_tokens: Per-pair token count actually fed to the model.
        orig_tokens: Per-pair token count before truncation.
        truncated_indices: Indices of the pairs that were truncated.
    """

    logits: np.ndarray
    used_tokens: list[int]
    orig_tokens: list[int]
    truncated_indices: list[int]


class InferenceEngine(Protocol):
    """Interface the HTTP layer depends on.

    Implementations serve zero or more models per kind and route by model
    id. The HTTP layer resolves the client-supplied id against the
    configuration before calling in, so these methods only have to defend
    themselves against an id that does not exist.
    """

    def embed(self, texts: list[str], model_id: str | None = None) -> EmbeddingBatch:
        """Embed ``texts`` in request order with the given embedding model."""
        ...

    def rerank(self, query: str, documents: list[str], model_id: str | None = None) -> RerankBatch:
        """Score every ``(query, document)`` pair with the given reranker."""
        ...

    def buckets(self, model_id: str) -> tuple[int, ...]:
        """Return the ascending sequence-length buckets served by ``model_id``."""
        ...

    def default_model_id(self, kind: str) -> str | None:
        """Return the id used when a request names no model of ``kind``."""
        ...


@dataclass
class _ServedModel:
    """One resident model: its tokenizer, its compiled artifacts and its metadata.

    Attributes:
        id: Model id requests route by.
        kind: Either ``"embedding"`` or ``"reranker"``.
        tokenizer: Frozen tokenizer the model is served with.
        tokenizer_lock: Serializes encodes on :attr:`tokenizer`. Fast
            tokenizers keep mutable padding/truncation state, so every
            model needs its own lock; sharing one across models would
            serialize unrelated requests for nothing.
        compiled: Loaded ``CompiledMLModel`` per sequence-length bucket.
        buckets: Ascending sequence-length buckets, i.e. the keys of
            :attr:`compiled`.
        output_name: Output tensor name requested at conversion time.
        normalize: The entry's ``normalize`` flag, recorded so callers can
            inspect the served model. The engine always returns raw
            vectors; normalization is the HTTP layer's decision.
        embedding_dim: Width of the embedding vectors when known: taken
            from the configuration, then filled in from the first real
            prediction. Always ``None`` for a reranker.
    """

    id: str
    kind: str
    tokenizer: runtime.FrozenTokenizer
    tokenizer_lock: threading.Lock
    compiled: dict[int, Any]
    buckets: tuple[int, ...]
    output_name: str
    normalize: bool
    embedding_dim: int | None


def _resolve_output_key(prediction: dict[str, Any], preferred: str) -> str:
    """Pick the output key of a ``predict`` result dict.

    Prefers the name chosen at conversion time but tolerates a renamed
    single output.

    Args:
        prediction: Dict returned by ``CompiledMLModel.predict``.
        preferred: Output name requested at conversion time.

    Returns:
        Key to read the output tensor from.

    Raises:
        RuntimeError: If the model returned no outputs at all.
    """
    keys = list(prediction)
    if not keys:
        raise RuntimeError("Core ML model returned no outputs")
    return preferred if preferred in keys else keys[0]


def _as_row(output: Any, name: str) -> np.ndarray:
    """Flatten a batch-of-one Core ML output into a 1-D float32 row.

    Args:
        output: Tensor returned by ``predict`` (shape ``(1, D)`` or
            ``(1, 1)`` for a reranker).
        name: Output key, used in the error message only.

    Returns:
        1-D float32 view of ``output``.

    Raises:
        RuntimeError: If the output holds no values.
    """
    row = np.asarray(output, dtype=np.float32).reshape(-1)
    if row.size == 0:
        raise RuntimeError(f"Core ML output {name!r} is empty")
    return row


def _require_complete(entry: ModelEntry) -> None:
    """Reject a model entry the engine cannot serve.

    Configuration validation already guarantees complete entries, so this
    only guards against a hand-built :class:`~eeane.config.ModelEntry`.

    Args:
        entry: Model entry about to be loaded.

    Raises:
        ValueError: If the entry has no supported kind, no tokenizer, no
            compiled artifact, or a non-positive embedding width (which
            would shape an empty response as ``(0, -n)``).
    """
    if entry.kind not in _DEFAULT_OUTPUT_NAMES:
        raise ValueError(f"model '{entry.id}': unsupported model kind {entry.kind!r}")
    if entry.tokenizer is None:
        raise ValueError(f"model '{entry.id}': no tokenizer file configured")
    if not entry.artifacts:
        raise ValueError(f"model '{entry.id}': no compiled artifact configured")
    if entry.embedding_dim is not None and entry.embedding_dim <= 0:
        raise ValueError(
            f"model '{entry.id}': embedding_dim must be positive, got {entry.embedding_dim}"
        )


def _collect_missing(entry: ModelEntry) -> list[str]:
    """Describe the artifacts of one entry that are not on disk.

    Args:
        entry: Complete model entry (see :func:`_require_complete`).

    Returns:
        One human-readable line per missing path, each naming the model it
        belongs to so a multi-model report stays readable. Empty when
        every path exists.
    """
    tokenizer_path = entry.tokenizer
    compiled = entry.artifacts or {}
    problems: list[str] = []
    if tokenizer_path is not None and not tokenizer_path.is_file():
        # Quote every bucket: the tokenizer is written by the same
        # `eeane compile` run that produces the artifacts.
        all_buckets = ",".join(str(bucket) for bucket in sorted(compiled))
        command = _COMPILE_COMMAND.format(buckets=all_buckets)
        problems.append(
            f"model '{entry.id}': missing tokenizer file {tokenizer_path}; "
            f"generate it with: {command}"
        )
    # Sorted so the reported order is deterministic across runs.
    for seq_len, path in sorted(compiled.items()):
        if not path.exists():
            command = _COMPILE_COMMAND.format(buckets=seq_len)
            problems.append(
                f"model '{entry.id}': missing Core ML artifact {path}; generate it with: {command}"
            )
    return problems


def _load_compiled(path: Path) -> Any:
    """Load a compiled Core ML model on the CPU+ANE compute units.

    Args:
        path: Path to a ``.mlmodelc`` directory.

    Returns:
        The loaded ``ct.models.CompiledMLModel``.
    """
    return ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)


def _load_entry(entry: ModelEntry) -> _ServedModel:
    """Load one configured entry's tokenizer and compiled artifacts.

    Args:
        entry: Complete model entry whose artifacts are known to exist.

    Returns:
        The resident model serving that entry.

    Raises:
        ValueError: If the frozen tokenizer file carries no padding
            section (i.e. it is not an ``eeane compile`` output).
    """
    kind = str(entry.kind)
    compiled = dict(entry.artifacts or {})
    tokenizer_path = entry.tokenizer
    if tokenizer_path is None:
        # Unreachable: every entry is checked before any of them is loaded.
        raise ValueError(f"model '{entry.id}': no tokenizer file configured")
    return _ServedModel(
        id=entry.id,
        kind=kind,
        tokenizer=runtime.load_frozen_tokenizer(tokenizer_path),
        tokenizer_lock=threading.Lock(),
        compiled={seq_len: _load_compiled(path) for seq_len, path in sorted(compiled.items())},
        buckets=tuple(sorted(compiled)),
        # output_name is derived during config validation; the fallback
        # only guards against a hand-built entry.
        output_name=entry.output_name or _DEFAULT_OUTPUT_NAMES[kind],
        normalize=entry.normalize,
        # A reranker has no embedding width, whatever the entry states.
        embedding_dim=entry.embedding_dim if kind == "embedding" else None,
    )


class CoreMLEngine:
    """Resident Core ML engine holding every configured model.

    All configured models and their tokenizers are loaded in ``__init__``
    and kept until the process exits (no on-demand loading yet). Every
    ``predict`` call is serialized by a single process-wide lock, whatever
    the model: the Neural Engine runs one prediction at a time, so a lock
    per model would only add contention without adding throughput.
    Tokenizer calls take one lock per model instead, since the mutable
    state they protect belongs to a single tokenizer.
    """

    def __init__(self, entries: Sequence[ModelEntry]) -> None:
        """Validate every entry's artifacts, then load tokenizers and models.

        Args:
            entries: Model entries to serve, in configuration order: the
                first entry of a kind is that kind's default model.
                Rerankers are optional, and a list without one builds an
                embedding-only engine whose :meth:`rerank` always raises
                (the HTTP layer answers 503 before reaching it).

        Raises:
            ValueError: If ``entries`` is empty, two entries share an id,
                an entry is incomplete, or a frozen tokenizer file carries
                no padding section.
            RuntimeError: If any tokenizer file or compiled artifact is
                missing. The message lists every problem of every model
                and the command that produces each missing artifact.
        """
        if not entries:
            raise ValueError("at least one model entry is required to build an engine")

        # Everything is checked before anything is loaded, so a broken
        # deployment is reported in one go instead of one model at a time.
        problems: list[str] = []
        seen_ids: set[str] = set()
        for entry in entries:
            if entry.id in seen_ids:
                raise ValueError(f"duplicate model id '{entry.id}' in the engine's model list")
            seen_ids.add(entry.id)
            _require_complete(entry)
            problems += _collect_missing(entry)
        if problems:
            raise RuntimeError(
                "eeANE cannot start, the following model artifacts are missing:\n  - "
                + "\n  - ".join(problems)
            )

        # Insertion order mirrors the configuration order, which is what
        # decides the default model of each kind.
        self._models: dict[str, _ServedModel] = {entry.id: _load_entry(entry) for entry in entries}
        # One predict lock for every model: the Neural Engine serializes
        # predictions anyway, and switching between models costs nothing.
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: EeaneConfig) -> CoreMLEngine:
        """Build the engine from a resolved eeANE configuration.

        Args:
            config: Validated configuration (see :mod:`eeane.config`).
                Every configured entry is served, and the first entry of
                each kind becomes that kind's default model.

        Returns:
            A loaded engine serving the configured artifacts.
        """
        return cls(config.models)

    def default_model_id(self, kind: str) -> str | None:
        """Return the id used when a request names no model of ``kind``.

        Args:
            kind: ``"embedding"`` or ``"reranker"``.

        Returns:
            The id of the first served model of that kind, or ``None``
            when the engine serves none.
        """
        for served in self._models.values():
            if served.kind == kind:
                return served.id
        return None

    def buckets(self, model_id: str) -> tuple[int, ...]:
        """Return the ascending sequence-length buckets served by ``model_id``.

        Args:
            model_id: Id of a served model.

        Returns:
            The sequence lengths the model was compiled for.

        Raises:
            KeyError: If no model with that id is served.
        """
        return self._models[model_id].buckets

    def embed(self, texts: list[str], model_id: str | None = None) -> EmbeddingBatch:
        """Embed ``texts`` one by one, routing each to its smallest bucket.

        Args:
            texts: Input texts in request order (prefixes, if any, are the
                client's responsibility).
            model_id: Embedding model to serve the request with; ``None``
                selects the default embedding model.

        Returns:
            Raw embeddings plus the token accounting needed for the
            response's ``usage`` field and the truncation warnings. An
            empty request keeps the ``(N, D)`` contract with ``N = 0`` and
            ``D`` the model's width: the configured ``embedding_dim``, or
            the width measured during an earlier prediction, or ``0`` as
            long as neither is available.

        Raises:
            ValueError: If ``model_id`` names no served model, or names a
                model of another kind.
            RuntimeError: If the engine serves no embedding model at all.
        """
        served = self._select("embedding", model_id)
        if not texts:
            # Keep the (N, D) contract for the empty request as well, with
            # the widest width known for this model.
            return EmbeddingBatch(
                vectors=np.empty((0, served.embedding_dim or 0), dtype=np.float32),
                used_tokens=[],
                orig_tokens=[],
                buckets=[],
                truncated_indices=[],
            )

        rows: list[np.ndarray] = []
        used_tokens: list[int] = []
        orig_tokens: list[int] = []
        buckets: list[int] = []
        truncated_indices: list[int] = []
        for index, text in enumerate(texts):
            with served.tokenizer_lock:
                n_tokens = runtime.count_text_tokens(served.tokenizer, text)
                bucket, truncated = runtime.select_bucket(n_tokens, served.buckets)
                inputs = runtime.tokenize_texts(served.tokenizer, [text], bucket)
            output = self._predict(served.compiled[bucket], inputs, served.output_name)
            rows.append(output)
            # attention_mask counts the tokens the model really consumed,
            # i.e. n_tokens capped at the bucket size.
            used_tokens.append(int(inputs["attention_mask"].sum()))
            orig_tokens.append(n_tokens)
            buckets.append(bucket)
            if truncated:
                truncated_indices.append(index)

        vectors = np.stack(rows)
        if served.embedding_dim is None:
            # Remember the width the model really produces, so a later
            # empty request can keep the (0, D) shape.
            served.embedding_dim = int(vectors.shape[1])
        return EmbeddingBatch(
            vectors=vectors,
            used_tokens=used_tokens,
            orig_tokens=orig_tokens,
            buckets=buckets,
            truncated_indices=truncated_indices,
        )

    def rerank(self, query: str, documents: list[str], model_id: str | None = None) -> RerankBatch:
        """Score every ``(query, document)`` pair with a cross-encoder.

        Args:
            query: Query text (first sequence of every pair).
            documents: Candidate documents in request order.
            model_id: Reranker to serve the request with; ``None`` selects
                the default reranker.

        Returns:
            Raw logits plus the token accounting; the sigmoid mapping is
            applied by the HTTP layer (``raw_scores`` decides).

        Raises:
            ValueError: If ``model_id`` names no served model, or names a
                model of another kind.
            RuntimeError: If no reranker is configured. The HTTP layer
                answers 503 before calling this, so this is a defensive
                guard for direct engine users.
        """
        served = self._select("reranker", model_id)
        if not documents:
            return RerankBatch(
                logits=np.empty((0,), dtype=np.float32),
                used_tokens=[],
                orig_tokens=[],
                truncated_indices=[],
            )

        logits: list[float] = []
        used_tokens: list[int] = []
        orig_tokens: list[int] = []
        truncated_indices: list[int] = []
        for index, document in enumerate(documents):
            with served.tokenizer_lock:
                n_tokens = runtime.count_pair_tokens(served.tokenizer, query, document)
                bucket, truncated = runtime.select_bucket(n_tokens, served.buckets)
                inputs = runtime.tokenize_pairs(served.tokenizer, [(query, document)], bucket)
            output = self._predict(served.compiled[bucket], inputs, served.output_name)
            # The reranker head emits a single logit per pair.
            logits.append(float(output[0]))
            used_tokens.append(int(inputs["attention_mask"].sum()))
            orig_tokens.append(n_tokens)
            if truncated:
                truncated_indices.append(index)

        return RerankBatch(
            logits=np.asarray(logits, dtype=np.float32),
            used_tokens=used_tokens,
            orig_tokens=orig_tokens,
            truncated_indices=truncated_indices,
        )

    def _select(self, kind: str, model_id: str | None) -> _ServedModel:
        """Resolve one request's model id to a served model.

        Args:
            kind: Model kind the calling endpoint serves.
            model_id: Requested model id, or ``None`` for the default
                model of ``kind``.

        Returns:
            The model that must serve the request.

        Raises:
            ValueError: If ``model_id`` is unknown or names a model of
                another kind. Turning a routing mistake into 404/400 is
                the HTTP layer's job; this is the runtime guard behind it.
            RuntimeError: If ``model_id`` is ``None`` and no model of
                ``kind`` is served.
        """
        if model_id is None:
            default_id = self.default_model_id(kind)
            if default_id is None:
                raise RuntimeError(f"{kind} is not configured")
            return self._models[default_id]

        served = self._models.get(model_id)
        if served is None:
            raise ValueError(f"unknown model id '{model_id}'")
        if served.kind != kind:
            raise ValueError(f"model '{model_id}' is a {served.kind} model, not a {kind} model")
        return served

    def _predict(self, model: Any, inputs: dict[str, np.ndarray], output_name: str) -> np.ndarray:
        """Run one batch-of-one prediction under the process-wide lock.

        Args:
            model: Loaded ``CompiledMLModel`` for the selected bucket.
            inputs: ``input_ids``/``attention_mask`` arrays of shape
                ``(1, S)``, dtype int32.
            output_name: Output name requested at conversion time.

        Returns:
            The prediction flattened to a 1-D float32 row.
        """
        with self._lock:
            prediction = model.predict(inputs)
        key = _resolve_output_key(prediction, output_name)
        return _as_row(prediction[key], key)
