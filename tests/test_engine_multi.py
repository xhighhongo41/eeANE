"""Tests for the multi-model engine, without loading a single Core ML model.

Artifact loading is replaced by a stub that answers ``predict`` from
memory, so the engine's loading, routing and shape contracts can be
exercised on any machine. Tokenizers are real (tiny word-level ones
written on the fly), so bucket selection, truncation and the token
accounting run exactly as they do in production.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from conftest import StubEngine
from tokenizers import Tokenizer, models, pre_tokenizers

from eeane import engine as engine_module
from eeane.config import EeaneConfig, ModelEntry, ServerConfig
from eeane.engine import CoreMLEngine, InferenceEngine

# Width of the row the stub artifacts return under the "embedding" key.
_EMBEDDING_WIDTH = 8

# Embedding width some entries state up front, standing in for the value
# the compiled-model cache records.
_CONFIGURED_WIDTH = 768


class StubCompiledModel:
    """Stand-in for a loaded ``CompiledMLModel``.

    Answers with both output names eeANE knows about, so a test can tell
    which one the engine asked for: ``"embedding"`` is a row of
    :data:`_EMBEDDING_WIDTH` values, ``"logits"`` a single one.

    Attributes:
        path: Artifact path this model was "loaded" from.
        seq_lens: Sequence length of every prediction, in call order.
    """

    def __init__(self, path: Path) -> None:
        """Register an artifact that has served no prediction yet."""
        self.path = path
        self.seq_lens: list[int] = []

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return deterministic outputs derived from the attention mask."""
        used = int(inputs["attention_mask"].sum())
        self.seq_lens.append(int(inputs["input_ids"].shape[1]))
        embedding = (np.arange(_EMBEDDING_WIDTH, dtype=np.float32) + float(used)).reshape(1, -1)
        return {
            "embedding": embedding,
            "logits": np.asarray([[float(used)]], dtype=np.float32),
        }


class StubLoader:
    """Replacement for ``eeane.engine._load_compiled`` that never opens a file.

    Attributes:
        loaded: Stub model created for every artifact path, keyed by path.
    """

    def __init__(self) -> None:
        """Start with nothing loaded."""
        self.loaded: dict[Path, StubCompiledModel] = {}

    def __call__(self, path: Path) -> StubCompiledModel:
        """Create (and remember) the stub model for ``path``."""
        model = StubCompiledModel(Path(path))
        self.loaded[Path(path)] = model
        return model

    def model_for(self, entry: ModelEntry, bucket: int) -> StubCompiledModel:
        """Return the stub model standing in for one entry's bucket.

        Args:
            entry: Entry whose artifact is wanted.
            bucket: Sequence-length bucket of that artifact.

        Returns:
            The stub model the engine loaded for that bucket.
        """
        assert entry.artifacts is not None
        return self.loaded[entry.artifacts[bucket]]


@pytest.fixture
def stub_loader(monkeypatch: pytest.MonkeyPatch) -> StubLoader:
    """Replace the Core ML loader for the duration of one test."""
    loader = StubLoader()
    monkeypatch.setattr(engine_module, "_load_compiled", loader)
    return loader


def _write_toy_tokenizer(path: Path) -> Path:
    """Write a minimal word-level tokenizer.json carrying pad settings.

    Args:
        path: Destination file.

    Returns:
        ``path``, for chaining.
    """
    vocab = {"<pad>": 0, "a": 1, "b": 2, "c": 3}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    tokenizer.save(str(path))
    return path


def _make_entry(
    root: Path,
    model_id: str,
    kind: str = "embedding",
    *,
    buckets: Sequence[int] = (8, 16),
    embedding_dim: int | None = None,
    create: bool = True,
) -> ModelEntry:
    """Build a model entry and, by default, the files it points at.

    Args:
        root: Directory holding one sub-directory per model.
        model_id: Id the entry is routed by.
        kind: ``"embedding"`` or ``"reranker"``.
        buckets: Sequence lengths to create artifacts for. They are kept
            small so a short text can exceed the largest one.
        embedding_dim: Embedding width stated by the entry, when known.
        create: When ``False``, no file is written, so the engine's
            pre-flight check reports the entry as missing.

    Returns:
        A validated entry.
    """
    model_dir = root / model_id
    tokenizer_path = model_dir / "tokenizer.json"
    artifacts = {bucket: model_dir / f"s{bucket}.mlmodelc" for bucket in buckets}
    if create:
        model_dir.mkdir(parents=True, exist_ok=True)
        _write_toy_tokenizer(tokenizer_path)
        for path in artifacts.values():
            # A compiled Core ML artifact is a directory, not a file.
            path.mkdir(exist_ok=True)
    fields: dict[str, object] = {
        "id": model_id,
        "kind": kind,
        "tokenizer": tokenizer_path,
        "artifacts": artifacts,
        "embedding_dim": embedding_dim,
    }
    return ModelEntry(**fields)


@pytest.fixture
def deployment(tmp_path: Path) -> list[ModelEntry]:
    """Two embedding models and two rerankers, in configuration order."""
    return [
        _make_entry(tmp_path, "emb-a", embedding_dim=_CONFIGURED_WIDTH),
        _make_entry(tmp_path, "emb-b", buckets=(16,)),
        _make_entry(tmp_path, "rr-a", kind="reranker"),
        _make_entry(tmp_path, "rr-b", kind="reranker", buckets=(16,)),
    ]


# --- interface -----------------------------------------------------------


@pytest.mark.parametrize("implementation", [CoreMLEngine, StubEngine])
@pytest.mark.parametrize("name", ["embed", "rerank", "buckets", "default_model_id", "loaded"])
def test_implementations_keep_the_inference_engine_signatures(
    implementation: type, name: str
) -> None:
    """The real engine and the test stub must stay interchangeable for the HTTP layer."""
    assert inspect.signature(getattr(implementation, name)) == inspect.signature(
        getattr(InferenceEngine, name)
    )


# --- loading -------------------------------------------------------------


def test_every_configured_model_is_loaded(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """Each entry's buckets must be loaded and reported under its own id."""
    engine = CoreMLEngine(deployment)

    assert engine.buckets("emb-a") == (8, 16)
    assert engine.buckets("emb-b") == (16,)
    assert engine.buckets("rr-a") == (8, 16)
    assert engine.buckets("rr-b") == (16,)
    assert len(stub_loader.loaded) == 6


def test_default_model_is_the_first_listed_of_its_kind(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """Configuration order decides the default model of each kind."""
    engine = CoreMLEngine(list(reversed(deployment)))

    assert engine.default_model_id("embedding") == "emb-b"
    assert engine.default_model_id("reranker") == "rr-b"
    assert engine.default_model_id("nonsense") is None


def test_from_config_serves_every_configured_entry(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """from_config must serve the whole models list, not just the defaults."""
    config = EeaneConfig(server=ServerConfig(), models=deployment)

    engine = CoreMLEngine.from_config(config)
    try:
        assert engine.default_model_id("embedding") == "emb-a"
        assert engine.buckets("emb-b") == (16,)
        assert engine.buckets("rr-b") == (16,)
    finally:
        # This configuration serves its models on demand, so the engine
        # runs an idle sweeper this test has to stop again.
        engine.close()


def test_embedding_only_deployment_reports_no_reranker(
    tmp_path: Path, stub_loader: StubLoader
) -> None:
    """Without a reranker entry the reranker kind must have no default."""
    engine = CoreMLEngine([_make_entry(tmp_path, "emb-a")])

    assert engine.default_model_id("reranker") is None


def test_tokenizer_locks_are_per_model_while_predict_stays_serialized(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """One predict lock for the whole engine, one tokenizer lock per model."""
    engine = CoreMLEngine(deployment)

    served = [managed.served for managed in engine._models.values()]
    assert all(model is not None for model in served)
    assert len({id(model.tokenizer_lock) for model in served}) == len(deployment)
    assert all(model.tokenizer_lock is not engine._lock for model in served)


# --- artifact validation -------------------------------------------------


def test_missing_artifacts_of_every_model_are_reported_together(
    tmp_path: Path, stub_loader: StubLoader
) -> None:
    """One start-up error must name every model whose files are absent."""
    entries = [
        _make_entry(tmp_path, "emb-a", create=False),
        _make_entry(tmp_path, "rr-a", kind="reranker", create=False),
    ]

    with pytest.raises(RuntimeError) as excinfo:
        CoreMLEngine(entries)

    message = str(excinfo.value)
    assert "emb-a" in message
    assert "rr-a" in message
    assert "tokenizer.json" in message
    assert "s8.mlmodelc" in message
    assert "eeane compile" in message
    # Nothing may be loaded once a path is known to be missing.
    assert stub_loader.loaded == {}


def test_a_model_with_complete_artifacts_is_still_reported_by_id(
    tmp_path: Path, stub_loader: StubLoader
) -> None:
    """A partially broken deployment must point at the offending model only."""
    entries = [
        _make_entry(tmp_path, "emb-a"),
        _make_entry(tmp_path, "emb-b", create=False),
    ]

    with pytest.raises(RuntimeError, match="emb-b"):
        CoreMLEngine(entries)


def test_empty_entry_list_is_rejected(stub_loader: StubLoader) -> None:
    """An engine without a single model could serve nothing."""
    with pytest.raises(ValueError, match="at least one"):
        CoreMLEngine([])


def test_duplicate_model_ids_are_rejected(tmp_path: Path, stub_loader: StubLoader) -> None:
    """Two entries sharing an id would make routing ambiguous."""
    entries = [_make_entry(tmp_path, "emb-a"), _make_entry(tmp_path, "emb-a", buckets=(16,))]

    with pytest.raises(ValueError, match="duplicate model id"):
        CoreMLEngine(entries)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"kind": None}, "kind"),
        ({"tokenizer": None}, "tokenizer"),
        ({"artifacts": {}}, "artifact"),
        ({"embedding_dim": 0}, "embedding_dim"),
        ({"embedding_dim": -8}, "embedding_dim"),
    ],
)
def test_incomplete_entries_are_rejected(
    tmp_path: Path,
    stub_loader: StubLoader,
    overrides: dict[str, object],
    expected: str,
) -> None:
    """A hand-built entry that config validation would reject must not load."""
    entry = _make_entry(tmp_path, "emb-a")
    # model_construct skips validation, which is exactly how such an entry
    # could reach the engine without going through load_config.
    broken = ModelEntry.model_construct(**{**entry.model_dump(), **overrides})

    with pytest.raises(ValueError, match=expected):
        CoreMLEngine([broken])


# --- routing -------------------------------------------------------------


def test_embed_uses_the_default_model_when_none_is_named(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """A request without a model id must reach the first-listed embedding model."""
    engine = CoreMLEngine(deployment)

    engine.embed(["a b"])

    assert stub_loader.model_for(deployment[0], 8).seq_lens == [8]
    assert stub_loader.model_for(deployment[1], 16).seq_lens == []


def test_embed_routes_to_the_named_model(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """An explicit id must select that model's artifacts and buckets."""
    engine = CoreMLEngine(deployment)

    batch = engine.embed(["a b"], "emb-b")

    assert batch.buckets == [16]
    assert stub_loader.model_for(deployment[1], 16).seq_lens == [16]
    assert stub_loader.model_for(deployment[0], 8).seq_lens == []


def test_rerank_routes_to_the_named_reranker(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """The reranker endpoints must route by id the same way."""
    engine = CoreMLEngine(deployment)

    engine.rerank("a", ["b"], "rr-b")

    assert stub_loader.model_for(deployment[3], 16).seq_lens == [16]
    assert stub_loader.model_for(deployment[2], 8).seq_lens == []


def test_rerank_uses_the_default_reranker_when_none_is_named(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """A request without a model id must reach the first-listed reranker."""
    engine = CoreMLEngine(deployment)

    engine.rerank("a", ["b"])

    assert stub_loader.model_for(deployment[2], 8).seq_lens == [8]


@pytest.mark.parametrize("model_id", ["nope", "", "EMB-A"])
def test_embed_rejects_an_unknown_model_id(
    deployment: list[ModelEntry], stub_loader: StubLoader, model_id: str
) -> None:
    """An id no model answers to must raise instead of falling back."""
    engine = CoreMLEngine(deployment)

    with pytest.raises(ValueError, match="unknown model id"):
        engine.embed(["a"], model_id)


def test_rerank_rejects_an_unknown_model_id(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """The reranker path must guard its ids as strictly as the embedding one."""
    engine = CoreMLEngine(deployment)

    with pytest.raises(ValueError, match="unknown model id"):
        engine.rerank("a", ["b"], "nope")


def test_embed_rejects_a_reranker_id(deployment: list[ModelEntry], stub_loader: StubLoader) -> None:
    """Embedding a text with a cross-encoder is a routing mistake, not a fallback."""
    engine = CoreMLEngine(deployment)

    with pytest.raises(ValueError, match="rr-a"):
        engine.embed(["a"], "rr-a")


def test_rerank_rejects_an_embedding_id(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """Scoring pairs with an embedding model must be rejected as well."""
    engine = CoreMLEngine(deployment)

    with pytest.raises(ValueError, match="emb-a"):
        engine.rerank("a", ["b"], "emb-a")


def test_rerank_without_a_configured_reranker_raises(
    tmp_path: Path, stub_loader: StubLoader
) -> None:
    """An embedding-only engine must report the missing reranker."""
    engine = CoreMLEngine([_make_entry(tmp_path, "emb-a")])

    with pytest.raises(RuntimeError, match="reranker is not configured"):
        engine.rerank("a", ["b"])


def test_buckets_of_an_unknown_model_raise_key_error(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """Bucket lookups are keyed by model id and must not invent an answer."""
    engine = CoreMLEngine(deployment)

    with pytest.raises(KeyError):
        engine.buckets("nope")


# --- inference results ---------------------------------------------------


def test_embed_reports_buckets_tokens_and_truncation(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """Every input must be routed to its smallest bucket and accounted for."""
    engine = CoreMLEngine(deployment)
    long_text = " ".join(["a"] * 20)  # 20 tokens, above the largest bucket (16)

    batch = engine.embed(["a b", long_text])

    assert batch.vectors.shape == (2, _EMBEDDING_WIDTH)
    assert batch.buckets == [8, 16]
    assert batch.orig_tokens == [2, 20]
    assert batch.used_tokens == [2, 16]
    assert batch.truncated_indices == [1]


def test_rerank_returns_one_logit_per_document(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """The reranker output must be read from the "logits" key, one value per pair."""
    engine = CoreMLEngine(deployment)

    batch = engine.rerank("a", ["b", "c c"])

    assert batch.logits.shape == (2,)
    assert batch.logits.dtype == np.float32
    assert batch.orig_tokens == [2, 3]
    assert batch.truncated_indices == []


def test_rerank_with_no_documents_returns_an_empty_batch(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """An empty documents list must not reach the model at all."""
    engine = CoreMLEngine(deployment)

    batch = engine.rerank("a", [])

    assert batch.logits.shape == (0,)
    assert stub_loader.model_for(deployment[2], 8).seq_lens == []


# --- empty-request width -------------------------------------------------


def test_empty_request_uses_the_configured_embedding_dim(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """A width known from the configuration shapes the empty result."""
    engine = CoreMLEngine(deployment)

    batch = engine.embed([], "emb-a")

    assert batch.vectors.shape == (0, _CONFIGURED_WIDTH)
    assert batch.vectors.dtype == np.float32


def test_empty_request_has_zero_width_while_the_dim_is_unknown(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """Without a configured width, nothing is invented before the first prediction."""
    engine = CoreMLEngine(deployment)

    batch = engine.embed([], "emb-b")

    assert batch.vectors.shape == (0, 0)


def test_empty_request_uses_the_width_measured_by_an_earlier_prediction(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """The first real prediction teaches the engine the model's width."""
    engine = CoreMLEngine(deployment)

    engine.embed(["a b"], "emb-b")
    batch = engine.embed([], "emb-b")

    assert batch.vectors.shape == (0, _EMBEDDING_WIDTH)


def test_measured_width_stays_model_local(
    deployment: list[ModelEntry], stub_loader: StubLoader
) -> None:
    """Measuring one model's width must not shape another model's empty result."""
    engine = CoreMLEngine(deployment)

    engine.embed(["a b"], "emb-b")
    batch = engine.embed([], "emb-a")

    assert batch.vectors.shape == (0, _CONFIGURED_WIDTH)
