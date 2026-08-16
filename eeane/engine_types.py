"""Data contracts and pure helpers for the eeANE Core ML engine.

This module holds the dataclasses, the errors and the
:class:`InferenceEngine` protocol the HTTP layer and the engine
implementation share, plus the small pure functions that validate a model
entry and shape a raw Core ML prediction into the engine's result types.
Nothing here touches Core ML itself or any mutable engine state;
:mod:`eeane.engine` composes these pieces with the actual model loading
and lifecycle management.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from eeane import runtime
from eeane.config import BATCH_ARTIFACT_BATCH_SIZE, ModelEntry

# Conversion command quoted in the "missing artifact" errors, so the
# operator can regenerate what is missing without reading the docs.
_COMPILE_COMMAND = "eeane compile <model> --buckets {buckets}"

# Same command for a batched artifact, which is produced by a run of its
# own rather than by the batch-1 one. The batch size it is quoted with is
# the one a served entry's batched artifacts are compiled for, i.e. the
# number of inputs of one request they predict at a time.
_COMPILE_BATCH_COMMAND = _COMPILE_COMMAND + " --batch {batch}"

# Output tensor name assumed for a model whose entry does not state one.
_DEFAULT_OUTPUT_NAMES = {"embedding": "embedding", "reranker": "logits"}

# Load policies the engine can act on. ``"disabled"`` is a configuration
# level decision: such an entry never reaches the engine.
_LOAD_POLICIES = ("resident", "on_demand")

# States one served model moves through. Only a "loaded" model can answer
# a request, and only a "loaded" one can be unloaded.
_UNLOADED = "unloaded"
_LOADING = "loading"
_LOADED = "loaded"

# Bytes the request digest frames every text's length with. Eight bytes
# hold any text a request could carry, so the framing never overflows.
_LENGTH_PREFIX_BYTES = 8


class QueueTimeoutError(RuntimeError):
    """Raised when a request gives up before its inference starts.

    Every wait a request goes through before its first prediction ends
    this way once its deadline has passed: the wait for the engine to
    become free, and the wait for an identical request it was attached
    to. Nothing has been computed at that point, so the caller is free to
    retry or to report the timeout as it sees fit. A request whose first
    prediction has started is never interrupted, so this is never raised
    once inference is under way.
    """


class NonFiniteOutputError(RuntimeError):
    """Raised when a model answers with NaN or infinite values.

    Such an output is not a usable result -- it would silently poison a
    similarity search or a ranking -- and it points at the model itself
    rather than at the request, so it fails loudly instead of being
    passed on.

    Attributes:
        model_id: Id of the model that produced the output.
        bucket: Sequence-length bucket the prediction ran on, i.e. the
            compiled artifact the output came from.
    """

    def __init__(self, model_id: str, bucket: int) -> None:
        """Describe which model and which artifact produced the output.

        Args:
            model_id: Id of the model that produced the output.
            bucket: Sequence length the prediction ran on.
        """
        super().__init__(
            f"model '{model_id}' produced a non-finite output for bucket {bucket}; "
            "the compiled model may have run on an unsupported compute path"
        )
        self.model_id = model_id
        self.bucket = bucket


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


@dataclass(frozen=True)
class ModelPolicy:
    """How one served model is loaded and how long it stays in memory.

    Attributes:
        load_policy: ``"resident"`` (loaded at start-up, never unloaded)
            or ``"on_demand"`` (loaded on first use, unloaded once idle).
        keep_alive: Seconds an idle ``"on_demand"`` model stays in memory
            before it is unloaded. ``0`` unloads it as soon as it is
            found idle. Ignored for a resident model.
    """

    load_policy: str = "resident"
    keep_alive: int = 300

    def __post_init__(self) -> None:
        """Reject a policy the engine could not act on.

        Raises:
            ValueError: If ``load_policy`` is not one of the supported
                values, or ``keep_alive`` is negative.
        """
        if self.load_policy not in _LOAD_POLICIES:
            raise ValueError(
                f"unsupported load policy {self.load_policy!r}, expected one of "
                + ", ".join(repr(name) for name in _LOAD_POLICIES)
            )
        if self.keep_alive < 0:
            raise ValueError(f"keep_alive must not be negative, got {self.keep_alive}")


class InferenceEngine(Protocol):
    """Interface the HTTP layer depends on.

    Implementations serve zero or more models per kind and route by model
    id. The HTTP layer resolves the client-supplied id against the
    configuration before calling in, so these methods only have to defend
    themselves against an id that does not exist.
    """

    def embed(
        self, texts: list[str], model_id: str | None = None, *, deadline: float | None = None
    ) -> EmbeddingBatch:
        """Embed ``texts`` in request order with the given embedding model.

        ``deadline`` is an absolute reading of the implementation's clock
        past which a request that is still waiting gives up, or ``None``
        to wait as long as it takes.
        """
        ...

    def rerank(
        self,
        query: str,
        documents: list[str],
        model_id: str | None = None,
        *,
        deadline: float | None = None,
    ) -> RerankBatch:
        """Score every ``(query, document)`` pair with the given reranker.

        ``deadline`` means what it does in :meth:`embed`.
        """
        ...

    def buckets(self, model_id: str) -> tuple[int, ...]:
        """Return the ascending sequence-length buckets served by ``model_id``."""
        ...

    def default_model_id(self, kind: str) -> str | None:
        """Return the id used when a request names no model of ``kind``."""
        ...

    def loaded(self, model_id: str) -> bool:
        """Report whether ``model_id``'s artifacts are in memory, loading nothing."""
        ...


@dataclass
class _ServedModel:
    """One loaded model: its tokenizer, its compiled artifacts and its metadata.

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
        compiled_batch: Loaded batched ``CompiledMLModel`` per bucket, for
            the buckets one is configured for; empty when none is, which
            is what an entry without batched artifacts is served with.
            Inputs of one request that share such a bucket are predicted
            together instead of one at a time.
        output_name: Output tensor name requested at conversion time.
        normalize: The entry's ``normalize`` flag, recorded so callers can
            inspect the served model. The engine always returns raw
            vectors; normalization is the HTTP layer's decision.
        embedding_dim: Width of the embedding vectors as stated by the
            configuration, or ``None`` when it states none (and always
            ``None`` for a reranker). The width measured at run time is
            tracked outside this record, which is discarded on every
            unload.
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
    compiled_batch: dict[int, Any] = field(default_factory=dict)


@dataclass
class _ManagedModel:
    """One served model's lifecycle state, whether or not it is in memory.

    Exists for the engine's whole life, so everything a request needs
    before any artifact is touched (routing, buckets, the empty-request
    width) survives an unload.

    Every mutable attribute (:attr:`state`, :attr:`served`,
    :attr:`last_used`, :attr:`in_flight`, :attr:`embedding_dim`) is read
    and written under the engine's state lock only. The immutable ones
    need no lock at all.

    Attributes:
        entry: Configuration entry this model serves, the source of its
            id, kind and artifact paths.
        policy: Load policy and idle delay applied to this model.
        buckets: Ascending sequence-length buckets, derived from the
            entry, so they can be reported while the model is unloaded.
        load_lock: Serializes this model's loads. Concurrent first
            requests therefore load it once, and a load never blocks
            another model's requests.
        state: ``"unloaded"``, ``"loading"`` or ``"loaded"``.
        served: The loaded artifacts while :attr:`state` is ``"loaded"``,
            ``None`` otherwise.
        last_used: Engine clock reading of the last request completion,
            i.e. what the idle delay is measured from.
        in_flight: Number of requests currently being served by this
            model. A model with requests in flight is never unloaded.
        embedding_dim: Width of the embedding vectors when known: taken
            from the configuration, then filled in from the first real
            prediction. Kept here rather than on :attr:`served` so an
            unload cannot lose it, since an empty request must keep
            answering with the ``(0, D)`` shape. Always ``None`` for a
            reranker.
    """

    entry: ModelEntry
    policy: ModelPolicy
    buckets: tuple[int, ...]
    embedding_dim: int | None
    load_lock: threading.Lock = field(default_factory=threading.Lock)
    state: str = _UNLOADED
    served: _ServedModel | None = None
    last_used: float = 0.0
    in_flight: int = 0


@dataclass
class _InflightRequest:
    """One running computation identical requests can be served from.

    The request that created the record computes the answer and publishes
    it here; every request that arrives with the same content while it
    runs waits on :attr:`done` instead of computing the same thing again.
    Exactly one of :attr:`result` and :attr:`error` is set by the time
    :attr:`done` is set.

    Attributes:
        done: Set once the outcome has been published, whatever it is.
        result: Batch the computation produced, or ``None`` while it is
            still running or has failed. It is shared by every request
            served from this record, so all of them must treat it as
            read-only.
        error: Exception the computation raised, or ``None``. Identical
            requests fail identically, so it is re-raised as is by every
            request waiting on the record.
    """

    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


def _text_digest(texts: Sequence[str]) -> str:
    """Hash a request's texts into the identity it is coalesced under.

    Every text is framed with its own UTF-8 byte length, so no separator
    can be forged by the texts themselves: ``["ab", "c"]`` and
    ``["a", "bc"]`` hash differently, and two requests therefore share a
    digest only if they carry exactly the same texts in the same order.

    Args:
        texts: Texts the request's model inputs are built from, in
            request order.

    Returns:
        The hex-encoded SHA-256 digest of the framed texts.
    """
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(_LENGTH_PREFIX_BYTES, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _require_finite(row: np.ndarray, model_id: str, bucket: int) -> None:
    """Reject a prediction that carries NaN or infinite values.

    Args:
        row: One prediction, flattened to a 1-D row.
        model_id: Id of the model that produced it.
        bucket: Sequence length the prediction ran on.

    Raises:
        NonFiniteOutputError: If any value of ``row`` is not finite.
    """
    if not np.isfinite(row).all():
        raise NonFiniteOutputError(model_id, bucket)


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


def _as_rows(output: Any, name: str, rows: int) -> np.ndarray:
    """Shape a Core ML output into one float32 row per predicted input.

    Args:
        output: Tensor returned by ``predict`` (shape ``(rows, D)``, or
            ``(rows, 1)`` for a single-value head).
        name: Output key, used in the error messages only.
        rows: Number of inputs the prediction carried, at least one.

    Returns:
        A ``(rows, D)`` float32 array, one row per input, in the order the
        inputs were fed to the model.

    Raises:
        ValueError: If ``rows`` is not at least one.
        RuntimeError: If the output holds no values, or holds a count of
            values that is not a whole number of equally wide rows -- a
            model whose output does not match the inputs it was given.
    """
    if rows < 1:
        raise ValueError(f"a prediction carries at least one input, got {rows}")
    values = np.asarray(output, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise RuntimeError(f"Core ML output {name!r} is empty")
    if values.size % rows:
        raise RuntimeError(
            f"Core ML output {name!r} holds {values.size} values, which is not "
            f"{rows} rows of equal width"
        )
    return values.reshape(rows, -1)


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
        every path exists. The batched artifacts of an entry that
        configures any are checked too, since a deployment that asks for
        them must fail at start-up rather than on a first request.
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
    # Batched artifacts are only ever served for an embedding model, so an
    # entry of another kind is not held to them whatever it states.
    batched = (entry.batch_artifacts or {}) if entry.kind == "embedding" else {}
    for seq_len, path in sorted(batched.items()):
        if not path.exists():
            command = _COMPILE_BATCH_COMMAND.format(
                buckets=seq_len, batch=BATCH_ARTIFACT_BATCH_SIZE
            )
            problems.append(
                f"model '{entry.id}': missing Core ML artifact {path}; generate it with: {command}"
            )
    return problems
