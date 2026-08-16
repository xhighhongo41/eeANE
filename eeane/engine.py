"""Core ML inference engine for the eeANE server.

The engine owns everything that touches Core ML: artifact validation,
tokenizer/model loading and unloading, sequence-length bucket routing and
the process-wide lock that serializes every ``predict`` call (the Neural
Engine runs one prediction at a time anyway, so concurrent calls buy
nothing).

Any number of embedding and reranker models can be served at once, and a
request is routed to one by model id, ``None`` meaning "the first-listed
model of the kind the endpoint serves". Each model is served under one of
two load policies: a ``"resident"`` model is loaded at start-up and kept
in memory for the process' life, while an ``"on_demand"`` model is loaded
when a request first needs it and unloaded again once it has been idle
for its ``keep_alive`` delay -- or earlier, when loading another model
would exceed ``max_loaded_models``.

The HTTP layer only sees :class:`InferenceEngine`, so tests can inject a
deterministic stub without touching ``eeane.server``.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import coremltools as ct
import numpy as np

from eeane import runtime
from eeane.config import EeaneConfig, ModelEntry

logger = logging.getLogger("eeane.engine")

# Conversion command quoted in the "missing artifact" errors, so the
# operator can regenerate what is missing without reading the docs.
_COMPILE_COMMAND = "eeane compile <model> --buckets {buckets}"

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


def _reclaim_memory(dropped: list[_ServedModel]) -> None:
    """Let go of unloaded artifacts and hand their memory back.

    Args:
        dropped: Models the caller has just unloaded, holding the last
            references to their tokenizers and compiled artifacts. The
            list is emptied, so the caller must hold no other reference
            to them.
    """
    if not dropped:
        return
    dropped.clear()
    # Compiled Core ML models and fast tokenizers own sizeable native
    # buffers that are only freed once the last Python reference goes
    # away. One collection per unload batch keeps the resident size
    # predictable without paying for one collection per model.
    gc.collect()


def _policies_from_config(config: EeaneConfig) -> dict[str, ModelPolicy]:
    """Read every served entry's effective load policy and idle delay.

    Args:
        config: Validated configuration (see :mod:`eeane.config`), whose
            accessors apply the ``[server]`` defaults to the entries that
            state neither value. Disabled entries are already gone from
            ``config.models``, so none can appear here.

    Returns:
        One policy per served entry, keyed by model id.
    """
    return {
        entry.id: ModelPolicy(
            load_policy=config.resolved_load_policy(entry),
            keep_alive=config.resolved_keep_alive(entry),
        )
        for entry in config.models
    }


class CoreMLEngine:
    """Core ML engine serving resident and on-demand models side by side.

    A resident model is loaded in ``__init__`` and kept until the process
    exits; an on-demand model is loaded when a request first needs it,
    then unloaded once it has been idle for its ``keep_alive`` delay, or
    when loading another model would exceed ``max_loaded_models``. Every
    entry's artifacts are checked in ``__init__`` whatever its policy, so
    a broken deployment fails at start-up rather than on a first request.

    Three locks are held, and the order they may be taken in is fixed to
    keep the engine deadlock-free:

    * ``load_lock`` (one per model) guards that model's loads and is the
      *outermost* lock. The loader itself runs holding this lock only, so
      a load that takes seconds blocks neither other models' requests nor
      anyone reading the engine's state.
    * ``_state_lock`` (one per engine) guards every model's lifecycle
      state. It may be taken while holding a ``load_lock``, but never the
      other way round, and no other lock may be taken while holding it.
    * ``_lock`` (one per engine) serializes ``predict`` calls, whatever
      the model: the Neural Engine runs one prediction at a time, so a
      lock per model would only add contention without adding throughput.
      It is taken on its own, never nested inside the other two.

    Tokenizer calls take one lock per model instead, since the mutable
    state they protect belongs to a single tokenizer.
    """

    def __init__(
        self,
        entries: Sequence[ModelEntry],
        *,
        policies: Mapping[str, ModelPolicy] | None = None,
        max_loaded_models: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        loader: Callable[[ModelEntry], _ServedModel] | None = None,
        sweep_interval: float = 5.0,
    ) -> None:
        """Validate every entry's artifacts, then load the resident models.

        Args:
            entries: Model entries to serve, in configuration order: the
                first entry of a kind is that kind's default model.
                Rerankers are optional, and a list without one builds an
                embedding-only engine whose :meth:`rerank` always raises
                (the HTTP layer answers 503 before reaching it).
            policies: Load policy per model id. Ids left out -- and
                ``None``, i.e. no policies at all -- are served as
                resident models, so a caller that does not care about
                loading behaviour gets a fully resident engine.
            max_loaded_models: Most models the engine may hold in memory
                at once, or ``None`` for no limit. Only idle on-demand
                models are ever unloaded to respect it, so a deployment
                whose resident models alone exceed the limit simply goes
                over it (with a warning) rather than failing requests.
            clock: Monotonic seconds source the idle accounting is
                measured with. Injectable so idle behaviour can be tested
                without waiting.
            loader: Loads one entry's tokenizer and compiled artifacts;
                defaults to the real Core ML loader. Injectable so the
                lifecycle can be tested without a Neural Engine.
            sweep_interval: Seconds between two idle-unload sweeps. A
                non-positive value runs no background sweeper at all,
                leaving idle unloading to whoever drives the engine.

        Raises:
            ValueError: If ``entries`` is empty, two entries share an id,
                an entry is incomplete, ``max_loaded_models`` is not
                strictly positive, or a frozen tokenizer file carries no
                padding section.
            RuntimeError: If any tokenizer file or compiled artifact is
                missing. The message lists every problem of every model
                and the command that produces each missing artifact.
        """
        if not entries:
            raise ValueError("at least one model entry is required to build an engine")
        if max_loaded_models is not None and max_loaded_models < 1:
            raise ValueError(f"max_loaded_models must be at least 1, got {max_loaded_models}")

        # Everything is checked before anything is loaded, so a broken
        # deployment is reported in one go instead of one model at a time
        # -- and an on-demand model's missing artifact is reported at
        # start-up too, not on the first request that needs it.
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

        given_policies = dict(policies or {})
        self._clock = clock
        self._loader = loader if loader is not None else _load_entry
        self._max_loaded_models = max_loaded_models
        self._sweep_interval = sweep_interval
        # Insertion order mirrors the configuration order, which is what
        # decides the default model of each kind.
        self._models: dict[str, _ManagedModel] = {
            entry.id: _ManagedModel(
                entry=entry,
                policy=given_policies.get(entry.id, ModelPolicy()),
                buckets=tuple(sorted(entry.artifacts or ())),
                # A reranker has no embedding width, whatever it states.
                embedding_dim=entry.embedding_dim if entry.kind == "embedding" else None,
            )
            for entry in entries
        }
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stopping = threading.Event()
        self._sweeper: threading.Thread | None = None

        for managed in self._models.values():
            if managed.policy.load_policy == "resident":
                # Loaded through the request path so a resident model goes
                # through exactly the same accounting as any other one.
                self._acquire(managed)
                self._release(managed)
        self._start_sweeper()

    @classmethod
    def from_config(cls, config: EeaneConfig) -> CoreMLEngine:
        """Build the engine from a resolved eeANE configuration.

        Args:
            config: Validated configuration (see :mod:`eeane.config`).
                Every configured entry is served, and the first entry of
                each kind becomes that kind's default model.

        Returns:
            An engine serving the configured artifacts, with the resident
            models already loaded.
        """
        return cls(
            config.models,
            policies=_policies_from_config(config),
            max_loaded_models=config.server.max_loaded_models,
        )

    def close(self) -> None:
        """Stop the background idle sweeper, if one is running.

        Idempotent, and safe to call on an engine that never started a
        sweeper. Loaded models are left in memory: the engine can still
        answer requests afterwards, it just stops unloading idle models
        on its own.
        """
        self._stopping.set()
        sweeper = self._sweeper
        self._sweeper = None
        if sweeper is not None:
            # The sweeper waits on the event, so it wakes immediately;
            # joining keeps a caller from racing a sweep it has stopped.
            sweeper.join()

    def default_model_id(self, kind: str) -> str | None:
        """Return the id used when a request names no model of ``kind``.

        Args:
            kind: ``"embedding"`` or ``"reranker"``.

        Returns:
            The id of the first served model of that kind, or ``None``
            when the engine serves none.
        """
        for managed in self._models.values():
            if managed.entry.kind == kind:
                return managed.entry.id
        return None

    def buckets(self, model_id: str) -> tuple[int, ...]:
        """Return the ascending sequence-length buckets served by ``model_id``.

        The buckets come from the model's configured artifacts, so they
        are reported whether or not the model is currently in memory, and
        asking for them never loads anything.

        Args:
            model_id: Id of a served model.

        Returns:
            The sequence lengths the model was compiled for.

        Raises:
            KeyError: If no model with that id is served.
        """
        return self._models[model_id].buckets

    def loaded(self, model_id: str) -> bool:
        """Report whether ``model_id``'s artifacts are in memory right now.

        Reads the model's state without touching it: a model that is
        still loading counts as not loaded, and asking never triggers a
        load.

        Args:
            model_id: Id of a served model.

        Returns:
            Whether the model's tokenizer and compiled artifacts are held
            in memory.

        Raises:
            KeyError: If no model with that id is served.
        """
        managed = self._models[model_id]
        with self._state_lock:
            return managed.state == _LOADED

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
                Loading the model can raise whatever the loader raises,
                leaving the model unloaded so the next request retries.
        """
        managed = self._select("embedding", model_id)
        if not texts:
            # Answered before the model is acquired: an empty request
            # needs no artifacts, so it must never load an unloaded
            # model. The (N, D) contract still holds, with the widest
            # width known for this model.
            return EmbeddingBatch(
                vectors=np.empty((0, self._known_width(managed)), dtype=np.float32),
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
        served = self._acquire(managed)
        try:
            for index, text in enumerate(texts):
                with served.tokenizer_lock:
                    n_tokens = runtime.count_text_tokens(served.tokenizer, text)
                    bucket, truncated = runtime.select_bucket(n_tokens, served.buckets)
                    inputs = runtime.tokenize_texts(served.tokenizer, [text], bucket)
                output = self._predict(served.compiled[bucket], inputs, served.output_name)
                rows.append(output)
                # attention_mask counts the tokens the model really
                # consumed, i.e. n_tokens capped at the bucket size.
                used_tokens.append(int(inputs["attention_mask"].sum()))
                orig_tokens.append(n_tokens)
                buckets.append(bucket)
                if truncated:
                    truncated_indices.append(index)
        finally:
            self._release(managed)

        vectors = np.stack(rows)
        # Remember the width the model really produces, so a later empty
        # request can keep the (0, D) shape.
        self._record_width(managed, int(vectors.shape[1]))
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
                guard for direct engine users. Loading the model can
                raise whatever the loader raises, leaving the model
                unloaded so the next request retries.
        """
        managed = self._select("reranker", model_id)
        if not documents:
            # As in embed(): an empty request must not load anything.
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
        served = self._acquire(managed)
        try:
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
        finally:
            self._release(managed)

        return RerankBatch(
            logits=np.asarray(logits, dtype=np.float32),
            used_tokens=used_tokens,
            orig_tokens=orig_tokens,
            truncated_indices=truncated_indices,
        )

    def _select(self, kind: str, model_id: str | None) -> _ManagedModel:
        """Resolve one request's model id to a served model.

        Routing only reads configuration-derived state, so it works the
        same whether or not the model is currently in memory.

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

        managed = self._models.get(model_id)
        if managed is None:
            raise ValueError(f"unknown model id '{model_id}'")
        if managed.entry.kind != kind:
            raise ValueError(
                f"model '{model_id}' is a {managed.entry.kind} model, not a {kind} model"
            )
        return managed

    def _acquire(self, managed: _ManagedModel) -> _ServedModel:
        """Make sure ``managed`` is in memory and count one request in.

        The caller must pair every call with a :meth:`_release`, and owns
        the returned reference for the whole request: even if the model
        is unloaded meanwhile, the artifacts it is already using stay
        alive.

        Concurrent first requests all wait on the model's load lock, so
        the loader runs once and every waiter is served by that one
        result.

        Args:
            managed: Model the request routed to.

        Returns:
            The loaded artifacts to serve the request with.

        Raises:
            Exception: Whatever the loader raises. The model is left
                unloaded, so a later request loads it again rather than
                inheriting a half-built state.
        """
        served = self._enter_if_loaded(managed)
        if served is not None:
            return served

        # Lock order: the load lock is taken first and the state lock
        # only inside it, never the other way round.
        with managed.load_lock:
            served = self._enter_if_loaded(managed)
            if served is not None:
                return served

            _reclaim_memory(self._begin_load(managed))
            started = self._clock()
            try:
                # Run outside the state lock: loading takes seconds, and
                # must block neither other models nor state readers.
                served = self._loader(managed.entry)
            except BaseException:
                self._abort_load(managed)
                raise
            self._finish_load(managed, served, started)
        return served

    def _release(self, managed: _ManagedModel) -> None:
        """Count one finished request out and restart its idle delay.

        The idle delay is measured from when a request *finished*, so a
        long request cannot be followed by an immediate unload.

        Args:
            managed: Model the finished request was served by.
        """
        with self._state_lock:
            if managed.in_flight > 0:
                managed.in_flight -= 1
            managed.last_used = self._clock()

    def _enter_if_loaded(self, managed: _ManagedModel) -> _ServedModel | None:
        """Count one request in if ``managed`` is already in memory.

        Args:
            managed: Model the request routed to.

        Returns:
            The loaded artifacts, or ``None`` if the model has to be
            loaded first. Taking the reference and counting the request
            in happen together under the state lock, so a model cannot be
            unloaded between the two.
        """
        with self._state_lock:
            served = managed.served
            if managed.state == _LOADED and served is not None:
                managed.in_flight += 1
                return served
        return None

    def _begin_load(self, managed: _ManagedModel) -> list[_ServedModel]:
        """Mark ``managed`` as loading and make room for it.

        Args:
            managed: Model whose load is about to start.

        Returns:
            The artifacts of every model unloaded to stay within
            ``max_loaded_models``, for the caller to release outside the
            state lock.
        """
        with self._state_lock:
            managed.state = _LOADING
            return self._evict_locked(managed)

    def _finish_load(self, managed: _ManagedModel, served: _ServedModel, started: float) -> None:
        """Publish a finished load and count the waiting request in.

        Args:
            managed: Model that has just been loaded.
            served: Artifacts the loader produced.
            started: Clock reading taken before the loader ran.
        """
        with self._state_lock:
            managed.served = served
            managed.state = _LOADED
            managed.in_flight += 1
            managed.last_used = self._clock()
            elapsed = managed.last_used - started
        logger.info("loaded model '%s' in %.2fs", managed.entry.id, elapsed)

    def _abort_load(self, managed: _ManagedModel) -> None:
        """Return a failed load to the unloaded state so it can be retried.

        Args:
            managed: Model whose loader raised.
        """
        with self._state_lock:
            managed.state = _UNLOADED
            managed.served = None

    def _evict_locked(self, incoming: _ManagedModel) -> list[_ServedModel]:
        """Unload idle on-demand models until ``incoming`` fits in the limit.

        Must be called with the state lock held and with ``incoming``
        already marked as loading, so it counts against the limit like
        any model already in memory. Only a model the engine is allowed
        to drop is a candidate: on-demand, loaded, and serving no
        request. A victim's own load lock is deliberately not taken --
        the lock order forbids it here -- which is safe because the state
        lock alone decides whether a model may be dropped.

        Args:
            incoming: Model that is about to be loaded.

        Returns:
            The artifacts of every unloaded model, for the caller to
            release outside the state lock. Empty when nothing had to be
            unloaded, including when nothing could be.
        """
        dropped: list[_ServedModel] = []
        limit = self._max_loaded_models
        if limit is None:
            return dropped

        while True:
            held = [
                managed for managed in self._models.values() if managed.state in (_LOADED, _LOADING)
            ]
            if len(held) <= limit:
                return dropped
            candidates = [
                managed
                for managed in held
                if managed is not incoming
                and managed.state == _LOADED
                and managed.in_flight == 0
                and managed.policy.load_policy == "on_demand"
            ]
            if not candidates:
                # Serving the request matters more than the limit: going
                # over it is reported, never turned into a failure.
                logger.warning(
                    "loading model '%s' leaves %d model(s) in memory, above "
                    "max_loaded_models=%d: no idle on-demand model can be unloaded",
                    incoming.entry.id,
                    len(held),
                    limit,
                )
                return dropped
            victim = min(candidates, key=lambda managed: managed.last_used)
            served = self._unload_locked(
                victim, f"evicted to make room for model '{incoming.entry.id}'"
            )
            if served is not None:
                dropped.append(served)

    def _unload_locked(self, managed: _ManagedModel, reason: str) -> _ServedModel | None:
        """Drop one loaded model's artifacts, keeping everything else about it.

        Must be called with the state lock held, on a model that is
        loaded and has no request in flight.

        Args:
            managed: Model to unload.
            reason: Why it is being unloaded, quoted in the log line.

        Returns:
            The artifacts that were held, so the caller can drop the last
            reference to them outside the state lock, or ``None`` if the
            model held none.
        """
        served = managed.served
        managed.served = None
        managed.state = _UNLOADED
        logger.info("unloaded model '%s' (%s)", managed.entry.id, reason)
        return served

    def _sweep_idle(self, now: float) -> int:
        """Unload every on-demand model that has been idle long enough.

        A model is unloaded once it has served no request for its
        ``keep_alive`` delay; ``keep_alive=0`` therefore unloads it at the
        first sweep that finds it idle. Resident models, models still
        loading and models with a request in flight are all left alone.

        Args:
            now: Current reading of the engine's clock.

        Returns:
            How many models were unloaded.
        """
        dropped: list[_ServedModel] = []
        with self._state_lock:
            for managed in self._models.values():
                if managed.policy.load_policy != "on_demand":
                    continue
                if managed.state != _LOADED or managed.in_flight > 0:
                    continue
                idle = now - managed.last_used
                if idle < managed.policy.keep_alive:
                    continue
                served = self._unload_locked(managed, f"idle for {idle:.1f}s")
                if served is not None:
                    dropped.append(served)

        unloaded = len(dropped)
        _reclaim_memory(dropped)
        return unloaded

    def _start_sweeper(self) -> None:
        """Start the background idle-unload sweeper if this engine needs one.

        Nothing is started when no model can ever be unloaded (every
        model is resident) or when sweeping is switched off by a
        non-positive interval, so a fully resident engine stays a
        single-threaded one.
        """
        if self._sweep_interval <= 0:
            return
        if not any(managed.policy.load_policy == "on_demand" for managed in self._models.values()):
            return
        self._sweeper = threading.Thread(
            target=self._sweep_loop, name="eeane-idle-unloader", daemon=True
        )
        self._sweeper.start()

    def _sweep_loop(self) -> None:
        """Sweep for idle models until :meth:`close` stops the engine."""
        # wait() returns True as soon as close() sets the event, which
        # both ends the loop and keeps the shutdown from waiting out a
        # whole interval.
        while not self._stopping.wait(self._sweep_interval):
            self._sweep_idle(self._clock())

    def _known_width(self, managed: _ManagedModel) -> int:
        """Return the embedding width known for ``managed``, or ``0``.

        Args:
            managed: Model an empty request routed to.

        Returns:
            The configured or previously measured width, or ``0`` while
            neither is available.
        """
        with self._state_lock:
            return managed.embedding_dim or 0

    def _record_width(self, managed: _ManagedModel, width: int) -> None:
        """Remember the width the model's first real prediction produced.

        Args:
            managed: Model that produced the vectors.
            width: Width measured on those vectors.
        """
        with self._state_lock:
            if managed.embedding_dim is None:
                managed.embedding_dim = width

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
