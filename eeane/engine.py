"""Core ML inference engine for the eeANE server (v0.4実装計画.md §4.1, §4.3).

The engine owns everything that touches Core ML: artifact validation,
tokenizer/model loading, sequence-length bucket routing and the
process-wide lock that serializes every ``predict`` call (v0.3 measured
that the ANE serializes predictions anyway, so concurrent calls buy
nothing).

The HTTP layer only sees :class:`InferenceEngine`, so tests can inject a
deterministic stub and a future on-demand-loading engine can replace
:class:`CoreMLEngine` without touching ``eeane.server``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer

from eeane import runtime, settings

# Hidden size of ruri-v3-310m. Only used to shape the empty result of an
# empty request; non-empty results take their width from the model output.
EMBEDDING_DIM = 768

# Conversion commands quoted in the "missing artifact" error, so the
# operator can regenerate a missing .mlmodelc without reading the docs.
_CONVERT_COMMAND = {
    "embedding": "uv run python poc/convert_embedding.py --seq-len {seq_len} --batch 1",
    "reranker": "uv run python poc/convert_reranker.py --seq-len {seq_len} --batch 1",
}


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

    Attributes:
        embedding_buckets: Ascending sequence lengths served by the
            embedding model.
        reranker_buckets: Ascending sequence lengths served by the
            reranker model.
    """

    embedding_buckets: tuple[int, ...]
    reranker_buckets: tuple[int, ...]

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        """Embed ``texts`` in request order."""
        ...

    def rerank(self, query: str, documents: list[str]) -> RerankBatch:
        """Score every ``(query, document)`` pair in request order."""
        ...


def _resolve_output_key(prediction: dict[str, Any], preferred: str) -> str:
    """Pick the output key of a ``predict`` result dict.

    Mirrors ``poc.convert_common.resolve_output_key``: prefer the name
    chosen at conversion time but tolerate a renamed single output.

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
            ``(1, 1)`` for the reranker).
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


def _collect_missing(kind: str, model_dir: Path, compiled: dict[int, Path]) -> list[str]:
    """Describe the missing artifacts of one model kind.

    Args:
        kind: Either ``"embedding"`` or ``"reranker"`` (selects the
            conversion command quoted in the message).
        model_dir: HuggingFace-format directory the tokenizer loads from.
        compiled: Mapping from sequence-length bucket to ``.mlmodelc``
            path.

    Returns:
        One human-readable line per missing path (empty if all exist).
    """
    problems: list[str] = []
    if not model_dir.is_dir():
        problems.append(
            f"missing {kind} model directory {model_dir}; place the HuggingFace-format "
            f"model there (see README, 'Place the models in HF distribution form')"
        )
    # Sorted so the reported order is deterministic across runs.
    for seq_len, path in sorted(compiled.items()):
        if not path.exists():
            command = _CONVERT_COMMAND[kind].format(seq_len=seq_len)
            problems.append(f"missing Core ML artifact {path}; generate it with: {command}")
    return problems


def _load_compiled(path: Path) -> Any:
    """Load a compiled Core ML model on the CPU+ANE compute units.

    Args:
        path: Path to a ``.mlmodelc`` directory.

    Returns:
        The loaded ``ct.models.CompiledMLModel``.
    """
    return ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)


class CoreMLEngine:
    """Resident Core ML engine holding one compiled model per bucket.

    All models and both tokenizers are loaded in ``__init__`` and kept
    until the process exits (v0.4 has no on-demand loading). Every
    ``predict`` call is serialized by a single process-wide lock, and
    tokenizer calls are serialized by a second one (fast tokenizers keep
    mutable padding/truncation state, see ``__init__``).
    """

    def __init__(
        self,
        *,
        embedding_model_dir: Path,
        reranker_model_dir: Path,
        embedding_compiled: dict[int, Path],
        reranker_compiled: dict[int, Path],
        embedding_output_name: str,
        reranker_output_name: str,
    ) -> None:
        """Validate the artifacts, then load tokenizers and compiled models.

        Args:
            embedding_model_dir: HuggingFace-format embedding model
                directory (tokenizer source).
            reranker_model_dir: HuggingFace-format reranker model
                directory (tokenizer source).
            embedding_compiled: Bucket -> ``.mlmodelc`` path for the
                embedding model.
            reranker_compiled: Bucket -> ``.mlmodelc`` path for the
                reranker model.
            embedding_output_name: Output tensor name chosen when the
                embedding model was converted.
            reranker_output_name: Output tensor name chosen when the
                reranker model was converted.

        Raises:
            RuntimeError: If any model directory or compiled artifact is
                missing. The message lists every problem and the command
                that produces each missing artifact.
        """
        problems = _collect_missing("embedding", embedding_model_dir, embedding_compiled)
        problems += _collect_missing("reranker", reranker_model_dir, reranker_compiled)
        if problems:
            raise RuntimeError(
                "eeANE cannot start, the following model artifacts are missing:\n  - "
                + "\n  - ".join(problems)
            )

        self.embedding_buckets: tuple[int, ...] = tuple(sorted(embedding_compiled))
        self.reranker_buckets: tuple[int, ...] = tuple(sorted(reranker_compiled))
        self._embedding_output_name = embedding_output_name
        self._reranker_output_name = reranker_output_name

        self._embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_dir)
        self._reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_dir)
        self._embedding_models = {
            seq_len: _load_compiled(path) for seq_len, path in sorted(embedding_compiled.items())
        }
        self._reranker_models = {
            seq_len: _load_compiled(path) for seq_len, path in sorted(reranker_compiled.items())
        }
        # One lock for both models: the ANE serializes predictions anyway
        # and switching between the two models costs nothing (v0.3 TB).
        self._lock = threading.Lock()
        # HuggingFace fast tokenizers are NOT thread-safe: every call
        # mutates the Rust tokenizer's padding/truncation state, so two
        # endpoint threads encoding at once raise "RuntimeError: Already
        # borrowed". Encoding is sub-millisecond, so guarding it with its
        # own lock costs nothing and keeps it off the predict lock.
        self._tokenizer_lock = threading.Lock()

    @classmethod
    def from_settings(cls) -> CoreMLEngine:
        """Build the engine from the hard-coded ``eeane.settings`` constants."""
        return cls(
            embedding_model_dir=settings.EMBEDDING_MODEL_DIR,
            reranker_model_dir=settings.RERANKER_MODEL_DIR,
            embedding_compiled=settings.EMBEDDING_COMPILED,
            reranker_compiled=settings.RERANKER_COMPILED,
            embedding_output_name=settings.EMBEDDING_OUTPUT_NAME,
            reranker_output_name=settings.RERANKER_OUTPUT_NAME,
        )

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        """Embed ``texts`` one by one, routing each to its smallest bucket.

        Args:
            texts: Input texts in request order (prefixes, if any, are the
                client's responsibility).

        Returns:
            Raw embeddings plus the token accounting needed for the
            response's ``usage`` field and the truncation warnings.
        """
        if not texts:
            # Keep the (N, D) contract for the empty request as well.
            return EmbeddingBatch(
                vectors=np.empty((0, EMBEDDING_DIM), dtype=np.float32),
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
            with self._tokenizer_lock:
                n_tokens = runtime.count_text_tokens(self._embedding_tokenizer, text)
                bucket, truncated = runtime.select_bucket(n_tokens, self.embedding_buckets)
                inputs = runtime.tokenize_texts(self._embedding_tokenizer, [text], bucket)
            output = self._predict(
                self._embedding_models[bucket], inputs, self._embedding_output_name
            )
            rows.append(output)
            # attention_mask counts the tokens the model really consumed,
            # i.e. n_tokens capped at the bucket size.
            used_tokens.append(int(inputs["attention_mask"].sum()))
            orig_tokens.append(n_tokens)
            buckets.append(bucket)
            if truncated:
                truncated_indices.append(index)

        return EmbeddingBatch(
            vectors=np.stack(rows),
            used_tokens=used_tokens,
            orig_tokens=orig_tokens,
            buckets=buckets,
            truncated_indices=truncated_indices,
        )

    def rerank(self, query: str, documents: list[str]) -> RerankBatch:
        """Score every ``(query, document)`` pair with the cross-encoder.

        Args:
            query: Query text (first sequence of every pair).
            documents: Candidate documents in request order.

        Returns:
            Raw logits plus the token accounting; the sigmoid mapping is
            applied by the HTTP layer (``raw_scores`` decides).
        """
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
            with self._tokenizer_lock:
                n_tokens = runtime.count_pair_tokens(self._reranker_tokenizer, query, document)
                bucket, truncated = runtime.select_bucket(n_tokens, self.reranker_buckets)
                inputs = runtime.tokenize_pairs(
                    self._reranker_tokenizer, [(query, document)], bucket
                )
            output = self._predict(
                self._reranker_models[bucket], inputs, self._reranker_output_name
            )
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
