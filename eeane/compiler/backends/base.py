"""Compile-backend interface shared by every supported model architecture.

``eeane compile`` itself is architecture-agnostic: it resolves a source
directory, freezes the tokenizer, traces, converts, compiles and
self-checks. Everything that depends on the model architecture -- how the
model is loaded, which graph rewrites it needs, how it is wrapped into a
traceable module, how raw inputs become fixed-shape tensors, and what the
FP32 reference of an output is -- lives behind the :class:`CompileBackend`
protocol and is implemented once per architecture family in this package.

This module deliberately stays free of ``torch``/``transformers``: it only
declares types, so it can be imported from code paths that must not pull
in the heavy compile-time dependencies.

Adding a backend
----------------

1. Create ``eeane/compiler/backends/<family>.py`` with a class implementing
   every member listed below (structural typing: no base class to inherit).
2. Register it in :data:`eeane.compiler.dispatch.BACKEND_REGISTRY`, keyed by
   the architecture-name prefix that ``config.json`` reports. An existing
   backend may be registered under more than one prefix when another
   architecture is, for compilation purposes, the same encoder (e.g.
   ``Roberta`` alongside ``XLMRoberta``): add the extra key pointing at
   the same backend target instead of duplicating the implementation.
3. Add unit tests covering the fixtures, the kind validation and the
   effective maximum sequence length; add local-only tests for anything
   that needs real weights.

Members to implement:

``name``
    Human-readable backend name, used in error messages and progress
    output. Matches the registry key.
``supported_kinds``
    Kinds this backend can compile, e.g. ``("embedding", "reranker")``.
    Every kind-taking member must reject anything else with a
    ``ValueError`` naming the supported kinds.
``load(model_dir, kind, attn="eager")``
    Load the model and its tokenizer and return them as a
    :class:`LoadedModel`; see the rules below.
``apply_patches(loaded, mask_fill_value=None)``
    Apply whatever graph rewrites the conversion of this architecture
    requires, in place, and return a JSON-serializable record of what was
    actually applied (an empty dict for a backend that needs none). The
    optional finite attention-mask fill value is a remedy for masks that
    become ``-inf`` (and thus NaN-prone) once the graph is cast to FP16;
    backends that cannot apply it may ignore it or raise.
``wrap(loaded)``
    Return the traceable ``torch.nn.Module`` for ``loaded.kind``: the
    module whose forward is exactly what the compiled graph must compute
    (pooling included for an embedding model, raw logits for a reranker).
``output_name(kind)``
    Name of the single Core ML graph output for ``kind``.
``max_seq_len(model_dir)``
    Effective maximum sequence length the model can process, read from the
    model directory without loading any weights, or ``None`` when the
    architecture imposes no limit. Position-embedding offsets (if the
    architecture reserves leading positions) must already be subtracted:
    the pipeline compares bucket lengths against this number directly.
``trace_example(kind)``
    One fixed raw input used to build the tracing example.
``sanity_spec(kind)``
    Fixed self-check inputs -- one set per language -- plus their
    expected-ordering metadata.
``padding_input(kind)``
    Filler raw input used to pad a partial batch. It must encode to a
    non-empty attention mask, since a fully masked row can produce NaN.
``tokenize(loaded, inputs, seq_len)``
    Encode raw inputs into the fixed-shape arrays the compiled graph takes.
``reference_outputs(model_dir, kind, inputs, seq_len)``
    Compute the FP32 reference the self-check compares the compiled model
    against.

The pipeline drives them in this order::

    loaded = backend.load(model_dir, kind, attn=attn)   # once per run
    patches = backend.apply_patches(loaded)
    for seq_len in buckets:                             # once per bucket
        wrapper = backend.wrap(loaded)
        example = backend.tokenize(
            loaded, [backend.trace_example(kind)] * batch_size, seq_len
        )
        ...                                             # trace/convert/compile
    spec = backend.sanity_spec(kind)                    # once per variant
    reference = backend.reference_outputs(
        model_dir, kind, list(spec.all_inputs), seq_len
    )

Rules every backend must follow:

* ``load`` returns FP32 weights in ``eval()`` mode with
  ``config.return_dict = False``: tracing needs tuple outputs, and the
  conversion to a lower precision happens in the Core ML conversion step,
  never in PyTorch.
* The model directory is strictly read-only. A backend reads
  ``config.json`` and its siblings; it never writes into, renames or
  deletes anything below ``model_dir``.
* ``reference_outputs`` loads its own copy of the model through a
  reference path that :meth:`CompileBackend.apply_patches` does not
  rewrite (e.g. a different attention implementation), so the baseline
  stays independent of the conversion patches, and releases it before
  returning.
* Every member is stateless with respect to the backend instance: all
  per-model state travels in the :class:`LoadedModel` handle, so one
  backend instance can serve several compiles.
* The fixtures are fixed and deterministic. Comparing a compiled model
  against its FP32 baseline is only meaningful when both see byte-identical
  inputs.
* ``tokenize`` returns exactly ``input_ids`` and ``attention_mask``, both
  of shape ``(len(inputs), seq_len)`` and dtype ``int32``; any extra key
  the tokenizer produces is dropped, since the compiled graph takes those
  two inputs only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class LoadedModel:
    """One loaded HF model plus everything later stages need.

    The handle is what :meth:`CompileBackend.load` hands to every later
    stage, so that the model is loaded (and its weights held) exactly once
    per compile run.

    Attributes:
        model: ``torch.nn.Module`` in eval mode, FP32, with
            ``config.return_dict = False``.
        tokenizer: Tokenizer of the same model directory.
        config: ``transformers`` configuration object; the very object
            reachable as ``model.config``.
        model_dir: Read-only HuggingFace-format directory the model came
            from.
        kind: ``"embedding"`` or ``"reranker"``.
        attn: Attention implementation the model was loaded with.
        pooling: Pooling mode of an embedding model (e.g. ``"mean"`` or
            ``"cls"``); ``None`` for a reranker, whose head is part of the
            model itself.
        dense: ``torch.nn.Module`` projecting the pooled vector of an
            embedding model whose sentence-transformers module chain
            declares one; ``None`` when nothing is declared (the common
            case) and always for a reranker. Both the traced wrapper and
            the FP32 baseline apply this very module, so that the two
            sides of the self-check keep computing the same function.
        dense_config: JSON-serializable description of ``dense`` -- one
            entry per projection stage -- recorded in the compiled
            variant's metadata so a later run can tell whether the model's
            declaration still matches the artifact. ``None`` whenever
            ``dense`` is.
    """

    model: Any
    tokenizer: Any
    config: Any
    model_dir: Path
    kind: str
    attn: str
    pooling: str | None = None
    dense: Any = None
    dense_config: tuple[dict[str, Any], ...] | None = None


@dataclass(frozen=True)
class SanitySpec:
    """Fixed sanity-check inputs, one set per language, plus their metadata.

    Everything the self-check knows about a backend's sanity fixtures is
    in here: it never reads a backend module's constants directly.

    The fixtures are grouped by language because a checkpoint's vocabulary
    decides how much a set can say about it: fixtures in a language the
    model has no vocabulary for encode to little more than unknown-token
    rows, whose FP16-vs-FP32 difference says nothing about the model but
    can still miss the accuracy threshold. The self-check therefore
    evaluates every set and accepts the variant when *any* of them passes,
    which is why the sets have to reach it separately rather than merged
    into one list.

    Attributes:
        input_sets: Ordered ``(language, inputs)`` pairs, one per language
            the backend offers fixtures in. ``inputs`` are raw inputs --
            texts for an embedding model, ``(query, document)`` pairs for
            a reranker. Both levels are tuples on purpose: the same
            fixtures are fed to the compiled model and to the FP32
            reference, so a caller must not be able to change them. The
            order is part of the contract -- it decides the order rows are
            predicted in and the set a tie between two equally good sets
            resolves to.
        relevant_index: Index *within every set* of the pair expected to
            score *higher*, or ``None`` when the fixtures carry no
            ordering expectation (always the case for embeddings). Every
            set is built with its pairs in the same roles, so one index
            applies to all of them.
        irrelevant_index: Index *within every set* of the pair expected
            to score *lower*, or ``None``. The ordering check runs only
            when both indices are given.
    """

    input_sets: tuple[tuple[str, tuple[Any, ...]], ...]
    relevant_index: int | None = None
    irrelevant_index: int | None = None

    def __post_init__(self) -> None:
        """Validate the declared sets and the indices addressing them.

        Raises:
            TypeError: If a set's inputs are not held in a tuple, which
                would leave the fixtures mutable by their consumers.
            ValueError: If no set is declared, a set is empty, two sets
                share a language (one would silently overwrite the other
                in the self-check's per-language report), or an index is
                negative or beyond the last input of a set; any of those
                would only surface deep inside the self-check.
        """
        if not self.input_sets:
            raise ValueError("a sanity spec declares no sanity input set")
        seen: set[str] = set()
        for language, inputs in self.input_sets:
            if language in seen:
                raise ValueError(f"the sanity language '{language}' is declared more than once")
            seen.add(language)
            if not isinstance(inputs, tuple):
                raise TypeError(
                    f"the '{language}' sanity inputs are a {type(inputs).__name__}, "
                    "expected a tuple"
                )
            if not inputs:
                raise ValueError(f"the '{language}' sanity input set is empty")
            self._check_indices(language, inputs)

    def _check_indices(self, language: str, inputs: tuple[Any, ...]) -> None:
        """Validate both ordering indices against one set's inputs.

        Args:
            language: Language of the set being validated, named in the
                error so the offending set is identifiable.
            inputs: That set's inputs.

        Raises:
            ValueError: If an index is negative or beyond the last input.
        """
        for field_name in ("relevant_index", "irrelevant_index"):
            index = getattr(self, field_name)
            if index is None:
                continue
            if not 0 <= index < len(inputs):
                raise ValueError(
                    f"{field_name}={index} is out of range for the {len(inputs)} "
                    f"'{language}' sanity input(s)"
                )

    @property
    def languages(self) -> tuple[str, ...]:
        """Return the declared languages, in :attr:`input_sets` order."""
        return tuple(language for language, _ in self.input_sets)

    @property
    def all_inputs(self) -> tuple[Any, ...]:
        """Return every set's inputs concatenated in :attr:`input_sets` order.

        For consumers that treat the fixtures as one flat collection (the
        tokenizer-verification gate, which compares token sequences and
        is therefore language-agnostic) rather than evaluating them set by
        set.
        """
        return tuple(item for _, inputs in self.input_sets for item in inputs)


class CompileBackend(Protocol):
    """Structural interface of an architecture-specific compile backend.

    See the module docstring for the responsibilities of each member, the
    order the pipeline calls them in, and the rules an implementation must
    follow.
    """

    name: str
    supported_kinds: tuple[str, ...]

    def load(self, model_dir: Path, kind: str, attn: str = "eager") -> LoadedModel:
        """Load the FP32 model and tokenizer of ``model_dir`` for ``kind``.

        Args:
            model_dir: Read-only HuggingFace-format model directory.
            kind: Model kind to load the model as.
            attn: Attention implementation to request.

        Returns:
            The :class:`LoadedModel` handle every later stage takes.

        Raises:
            ValueError: If ``kind`` is not supported by the backend.
        """
        ...

    def apply_patches(
        self, loaded: LoadedModel, mask_fill_value: float | None = None
    ) -> dict[str, Any]:
        """Apply the graph rewrites this architecture needs, in place.

        Args:
            loaded: Handle returned by :meth:`load`.
            mask_fill_value: Optional finite attention-mask fill value.

        Returns:
            A JSON-serializable record of the patches actually applied
            (an empty dict when the backend needs none), recorded verbatim
            in the compiled variant's metadata.

        Raises:
            ValueError: If the loaded model contradicts an assumption a
                rewrite relies on.
        """
        ...

    def wrap(self, loaded: LoadedModel) -> Any:
        """Wrap the loaded model into the traceable module for its kind.

        Args:
            loaded: Handle returned by :meth:`load`.

        Returns:
            A ``torch.nn.Module`` in eval mode whose forward takes
            ``(input_ids, attention_mask)`` and returns the tensor the
            compiled graph must produce.

        Raises:
            ValueError: If ``loaded.kind`` is not supported by the backend.
        """
        ...

    def output_name(self, kind: str) -> str:
        """Return the Core ML graph output name used for ``kind``.

        Raises:
            ValueError: If ``kind`` is not supported by the backend.
        """
        ...

    def max_seq_len(self, model_dir: Path) -> int | None:
        """Return the effective maximum sequence length of ``model_dir``.

        Read from the model directory's configuration files only; no
        weights are loaded.

        Args:
            model_dir: Read-only HuggingFace-format model directory.

        Returns:
            The largest sequence length the model can process, or ``None``
            when it cannot be determined or the architecture has no limit.
        """
        ...

    def trace_example(self, kind: str) -> Any:
        """Return the fixed raw example input used for tracing.

        Raises:
            ValueError: If ``kind`` is not supported by the backend.
        """
        ...

    def sanity_spec(self, kind: str) -> SanitySpec:
        """Return the fixed sanity-check inputs and their metadata.

        Raises:
            ValueError: If ``kind`` is not supported by the backend.
        """
        ...

    def padding_input(self, kind: str) -> Any:
        """Return the filler input used to pad a partial batch.

        Raises:
            ValueError: If ``kind`` is not supported by the backend.
        """
        ...

    def tokenize(
        self, loaded: LoadedModel, inputs: list[Any], seq_len: int
    ) -> dict[str, np.ndarray]:
        """Encode raw inputs into fixed-shape int32 arrays.

        Args:
            loaded: Handle returned by :meth:`load`; its tokenizer and
                kind decide how the inputs are encoded.
            inputs: Raw inputs of the shape ``loaded.kind`` implies.
            seq_len: Fixed sequence length S.

        Returns:
            Dict with ``input_ids`` and ``attention_mask``, each of shape
            ``(len(inputs), seq_len)`` and dtype ``int32``.

        Raises:
            ValueError: If ``loaded.kind`` is unsupported, ``inputs`` is
                empty, or ``seq_len`` is not positive.
        """
        ...

    def reference_outputs(
        self, model_dir: Path, kind: str, inputs: list[Any], seq_len: int
    ) -> np.ndarray:
        """Compute the FP32 reference outputs for ``inputs``.

        Args:
            model_dir: Read-only HuggingFace-format model directory.
            kind: Model kind.
            inputs: Raw inputs of the shape ``kind`` implies.
            seq_len: Fixed sequence length S.

        Returns:
            One row per input: pooled embeddings of shape ``(N, hidden)``
            for an embedding model, raw scores of shape ``(N,)`` for a
            reranker; dtype float32.

        Raises:
            ValueError: If ``kind`` is unsupported or ``inputs`` is empty.
        """
        ...
