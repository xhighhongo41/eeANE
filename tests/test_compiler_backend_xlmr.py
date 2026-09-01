"""Tests for the XLM-RoBERTa compile backend and the shared backend helpers.

Three layers:

* ``eeane.compiler.backends.xlm_roberta``: pooling detection from the
  sentence-transformers pooling module, the position-offset rule behind
  the effective sequence length, the fixtures, and the kind/pooling
  validation the pipeline relies on.
* ``eeane.compiler.backends.common``: the CLS pooling added for this
  family (wrapper and FP32 baseline), checked against hand-computed
  values.
* Conformance to the backend interface, driven by the protocol in
  ``eeane.compiler.backends.base``.

Everything here runs on synthetic directories and tiny stand-in modules;
nothing needs real model weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from eeane.compiler.backends import common
from eeane.compiler.backends import xlm_roberta as xlmr

# Ids the stand-in tokenizer below uses for the special/padding tokens.
_BOS_ID = 1
_EOS_ID = 2
_PAD_ID = 0


class _FakeTokenizer:
    """Deterministic stand-in for a HuggingFace tokenizer.

    Encodes each text as ``<bos> one id per character <eos>``, appends the
    same encoding of the second sequence when one is given, then truncates
    and pads to ``max_length``. It also returns a ``token_type_ids`` key,
    so tests can verify that the tokenize helpers drop everything the
    compiled graph does not take.
    """

    def __call__(
        self,
        texts: list[str],
        text_pairs: list[str] | None = None,
        padding: str | None = None,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Encode ``texts`` (optionally paired) into fixed-shape int64 arrays."""
        assert max_length is not None, "the helpers must always pass a fixed length"
        pairs = text_pairs if text_pairs is not None else [None] * len(texts)
        rows = [
            self._encode(text, pair, max_length) for text, pair in zip(texts, pairs, strict=True)
        ]
        input_ids = np.array(rows, dtype=np.int64)
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != _PAD_ID).astype(np.int64),
            "token_type_ids": np.zeros_like(input_ids),
        }

    @staticmethod
    def _encode(text: str, pair: str | None, max_length: int) -> list[int]:
        """Build one padded/truncated id row for a text and its optional pair."""
        ids = [_BOS_ID, *(ord(char) % 90 + 5 for char in text), _EOS_ID]
        if pair is not None:
            ids += [_BOS_ID, *(ord(char) % 90 + 5 for char in pair), _EOS_ID]
        ids = ids[:max_length]
        return ids + [_PAD_ID] * (max_length - len(ids))


class _FakeEncoder(torch.nn.Module):
    """Backbone stand-in whose hidden state is a known function of the ids.

    ``hidden[b, s, h] == input_ids[b, s] + h``, so pooled results can be
    computed by hand from the token ids alone.
    """

    def __init__(self, hidden_size: int = 3) -> None:
        """Store the hidden size the pooling helpers read off the config."""
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Return the derived hidden state as a tuple, like return_dict=False."""
        offsets = torch.arange(self.config.hidden_size, dtype=torch.float32)
        return (input_ids.to(torch.float32).unsqueeze(-1) + offsets,)


class _FixedHiddenModel(torch.nn.Module):
    """Backbone stand-in returning one fixed hidden state, ignoring its inputs."""

    def __init__(self, hidden: torch.Tensor) -> None:
        """Store the hidden state every forward call returns."""
        super().__init__()
        self.hidden = hidden

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Return the stored hidden state as a tuple, like return_dict=False."""
        return (self.hidden,)


def _loaded(
    model: Any,
    kind: str = "embedding",
    pooling: str | None = "mean",
    tokenizer: Any = None,
    model_dir: Path = Path("/nonexistent-model-dir"),
) -> Any:
    """Build the handle the backend interface passes between its stages."""
    from eeane.compiler.backends import base

    return base.LoadedModel(
        model=model,
        tokenizer=tokenizer,
        config=getattr(model, "config", None),
        model_dir=model_dir,
        kind=kind,
        attn="eager",
        pooling=pooling if kind == "embedding" else None,
    )


def _write_json(path: Path, payload: object) -> None:
    """Write ``payload`` as JSON, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _model_dir(tmp_path: Path, config: dict[str, Any] | None = None, **pooling: Any) -> Path:
    """Build a synthetic model directory with a config.json and pooling module."""
    model_dir = tmp_path / "model"
    _write_json(model_dir / xlmr.CONFIG_FILENAME, config or {"architectures": ["XLMRobertaModel"]})
    if pooling:
        _write_json(model_dir / xlmr.POOLING_DIRNAME / xlmr.CONFIG_FILENAME, pooling)
    return model_dir


# --- pooling detection -------------------------------------------------------


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ({"pooling_mode_mean_tokens": True, "pooling_mode_cls_token": False}, "mean"),
        ({"pooling_mode_mean_tokens": False, "pooling_mode_cls_token": True}, "cls"),
        # Only the flag that is set has to be present.
        ({"pooling_mode_mean_tokens": True}, "mean"),
        ({"pooling_mode_cls_token": True, "word_embedding_dimension": 768}, "cls"),
    ],
)
def test_read_pooling_mode_detects_the_declared_mode(
    tmp_path: Path, declaration: dict[str, Any], expected: str
) -> None:
    """Exactly one enabled supported flag must select that pooling mode."""
    model_dir = _model_dir(tmp_path, **declaration)

    assert xlmr.read_pooling_mode(model_dir) == expected


def test_read_pooling_mode_without_a_pooling_module_explains_what_is_missing(
    tmp_path: Path,
) -> None:
    """A missing declaration must name the file and the accepted flags, not default."""
    model_dir = _model_dir(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        xlmr.read_pooling_mode(model_dir)

    message = str(excinfo.value)
    assert xlmr.POOLING_DIRNAME in message
    assert "pooling_mode_mean_tokens" in message
    assert "pooling_mode_cls_token" in message


@pytest.mark.parametrize(
    "declaration",
    [
        {"pooling_mode_mean_tokens": True, "pooling_mode_cls_token": True},
        {"pooling_mode_mean_tokens": False, "pooling_mode_cls_token": False},
        {},
        {"pooling_mode_max_tokens": True},
        {"pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": True},
        # A non-boolean flag is not a declaration this backend can trust.
        {"pooling_mode_mean_tokens": "true"},
        {"pooling_mode_mean_tokens": 1},
    ],
    ids=["both", "neither", "empty", "unsupported", "mixed", "string-flag", "int-flag"],
)
def test_read_pooling_mode_rejects_an_ambiguous_declaration(
    tmp_path: Path, declaration: dict[str, Any]
) -> None:
    """Anything but exactly one supported flag must raise instead of guessing."""
    model_dir = _model_dir(tmp_path, **{"word_embedding_dimension": 768, **declaration})

    with pytest.raises(ValueError, match="pooling"):
        xlmr.read_pooling_mode(model_dir)


def test_read_pooling_mode_rejects_a_corrupt_declaration(tmp_path: Path) -> None:
    """Unparsable JSON must surface as a ValueError naming the file."""
    model_dir = _model_dir(tmp_path)
    pooling_config = model_dir / xlmr.POOLING_DIRNAME / xlmr.CONFIG_FILENAME
    pooling_config.parent.mkdir(parents=True, exist_ok=True)
    pooling_config.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match=xlmr.POOLING_DIRNAME):
        xlmr.read_pooling_mode(model_dir)


def test_read_pooling_mode_rejects_a_non_object_declaration(tmp_path: Path) -> None:
    """A JSON document that is not an object cannot declare a pooling mode."""
    model_dir = _model_dir(tmp_path)
    _write_json(model_dir / xlmr.POOLING_DIRNAME / xlmr.CONFIG_FILENAME, ["mean"])

    with pytest.raises(ValueError, match="pooling"):
        xlmr.read_pooling_mode(model_dir)


def test_load_reports_an_undeclared_pooling_before_loading_weights(tmp_path: Path) -> None:
    """An embedding model without a pooling declaration must fail fast and clearly."""
    model_dir = _model_dir(tmp_path)

    with pytest.raises(ValueError, match="pooling"):
        xlmr.XlmRobertaBackend().load(model_dir, "embedding")


# --- effective maximum sequence length ---------------------------------------


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(514, 512), (8194, 8192), (3, 1), (4, 2)],
)
def test_max_seq_len_subtracts_the_reserved_positions(
    tmp_path: Path, configured: int, expected: int
) -> None:
    """The reserved leading positions are unusable and must not be offered."""
    model_dir = _model_dir(
        tmp_path,
        config={"architectures": ["XLMRobertaModel"], "max_position_embeddings": configured},
    )

    assert xlmr.XlmRobertaBackend().max_seq_len(model_dir) == expected


def test_max_seq_len_of_a_missing_directory_is_none(tmp_path: Path) -> None:
    """No config.json means no known limit, not a crash."""
    assert xlmr.XlmRobertaBackend().max_seq_len(tmp_path / "absent") is None


@pytest.mark.parametrize(
    "config",
    [
        {"architectures": ["XLMRobertaModel"]},
        {"max_position_embeddings": None},
        {"max_position_embeddings": "514"},
        {"max_position_embeddings": 514.0},
        {"max_position_embeddings": True},
        {"max_position_embeddings": 0},
        {"max_position_embeddings": -1},
        # Nothing is left once the reserved positions are subtracted.
        {"max_position_embeddings": 1},
        {"max_position_embeddings": 2},
    ],
    ids=["absent", "null", "string", "float", "bool", "zero", "negative", "one", "offset-only"],
)
def test_max_seq_len_ignores_a_missing_or_unusable_value(
    tmp_path: Path, config: dict[str, Any]
) -> None:
    """A missing or nonsensical value must degrade to 'unknown', never to a bogus limit."""
    model_dir = _model_dir(tmp_path, config=config)

    assert xlmr.XlmRobertaBackend().max_seq_len(model_dir) is None


def test_max_seq_len_of_a_corrupt_config_is_none(tmp_path: Path) -> None:
    """Unparsable JSON must not turn an optional check into a compile failure."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / xlmr.CONFIG_FILENAME).write_text("{not json", encoding="utf-8")

    assert xlmr.XlmRobertaBackend().max_seq_len(model_dir) is None


# --- interface attributes and fixtures ---------------------------------------


def test_backend_declares_the_interface_attributes() -> None:
    """The backend must name itself (matching its registry key) and its kinds."""
    backend = xlmr.XlmRobertaBackend()

    assert backend.name == "XLMRoberta"
    assert backend.supported_kinds == ("embedding", "reranker")


def test_output_name_per_kind() -> None:
    """Embeddings and rerankers must keep the graph output names of the engine."""
    backend = xlmr.XlmRobertaBackend()

    assert backend.output_name("embedding") == "embedding"
    assert backend.output_name("reranker") == "logits"


def test_fixtures_have_the_expected_shape_per_kind() -> None:
    """trace/sanity/padding fixtures must be texts for embeddings, pairs for rerankers."""
    backend = xlmr.XlmRobertaBackend()

    assert isinstance(backend.trace_example("embedding"), str)
    assert all(
        isinstance(text, str) and text for text in backend.sanity_spec("embedding").all_inputs
    )
    assert backend.padding_input("embedding") == ""

    trace_pair = backend.trace_example("reranker")
    assert isinstance(trace_pair, tuple) and len(trace_pair) == 2
    assert all(len(pair) == 2 for pair in backend.sanity_spec("reranker").all_inputs)
    assert backend.padding_input("reranker") == ("", "")


def test_embedding_sanity_spec_declares_no_ordering() -> None:
    """Embedding fixtures are compared row-wise; there is no expected ordering."""
    spec = xlmr.XlmRobertaBackend().sanity_spec("embedding")

    assert spec.input_sets
    assert spec.relevant_index is None
    assert spec.irrelevant_index is None


def test_reranker_sanity_spec_points_at_the_relevant_and_irrelevant_pairs() -> None:
    """The reranker ordering check must be driven by the spec, not by module constants."""
    spec = xlmr.XlmRobertaBackend().sanity_spec("reranker")
    japanese = dict(spec.input_sets)["ja"]

    assert spec.relevant_index == 0
    assert spec.irrelevant_index == 1
    # The relevant pair shares its query with the irrelevant one, so only the
    # document decides the expected ordering.
    assert japanese[0][0] == japanese[1][0]
    assert japanese[0][1] != japanese[1][1]


def test_sanity_spec_inputs_are_immutable() -> None:
    """The fixtures a caller receives must not be corruptible module state."""
    input_sets = xlmr.XlmRobertaBackend().sanity_spec("embedding").input_sets

    assert isinstance(input_sets, tuple)
    with pytest.raises(TypeError):
        input_sets[0][1][0] = "mutated"  # type: ignore[index]


@pytest.mark.parametrize(
    "method",
    ["trace_example", "sanity_spec", "padding_input", "output_name"],
)
def test_unknown_kind_is_rejected(method: str) -> None:
    """Every kind-dispatching method must reject an unsupported kind."""
    backend = xlmr.XlmRobertaBackend()

    with pytest.raises(ValueError, match="kind"):
        getattr(backend, method)("classifier")


def test_load_rejects_unknown_kind(tmp_path: Path) -> None:
    """load must validate the kind before touching the filesystem."""
    with pytest.raises(ValueError, match="kind"):
        xlmr.XlmRobertaBackend().load(tmp_path, "classifier")


@pytest.mark.parametrize("method", ["wrap", "tokenize"])
def test_handle_taking_methods_reject_an_unsupported_kind(method: str) -> None:
    """A handle carrying an unsupported kind must be rejected, not silently wrapped."""
    backend = xlmr.XlmRobertaBackend()
    loaded = _loaded(torch.nn.Identity(), kind="classifier")
    arguments = {"tokenize": (["text"], 8)}.get(method, ())

    with pytest.raises(ValueError, match="kind"):
        getattr(backend, method)(loaded, *arguments)


def test_reference_outputs_rejects_an_unsupported_kind(tmp_path: Path) -> None:
    """The reference path must validate the kind before loading any weights."""
    with pytest.raises(ValueError, match="kind"):
        xlmr.XlmRobertaBackend().reference_outputs(tmp_path, "classifier", ["text"], 8)


def test_reference_outputs_rejects_empty_inputs(tmp_path: Path) -> None:
    """No inputs means nothing to compare; that must raise before loading weights."""
    with pytest.raises(ValueError, match="inputs"):
        xlmr.XlmRobertaBackend().reference_outputs(tmp_path, "embedding", [], 8)


# --- patches (this architecture needs none) ----------------------------------


def test_apply_patches_is_a_no_op() -> None:
    """A model that needs no rewrite must come back untouched and unrewritten."""
    backend = xlmr.XlmRobertaBackend()
    model = _FakeEncoder()
    input_ids = torch.tensor([[1, 7, 9, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    with torch.no_grad():
        before = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    assert backend.apply_patches(_loaded(model)) == {}

    with torch.no_grad():
        after = model(input_ids=input_ids, attention_mask=attention_mask)[0]
    assert torch.equal(before, after)


def test_apply_patches_rejects_a_mask_fill_value() -> None:
    """An unimplemented remedy must be refused loudly, not accepted and ignored."""
    backend = xlmr.XlmRobertaBackend()

    with pytest.raises(ValueError, match="mask fill"):
        backend.apply_patches(_loaded(_FakeEncoder()), mask_fill_value=-30000.0)


# --- wrappers ----------------------------------------------------------------


def test_wrap_selects_the_wrapper_of_the_detected_pooling() -> None:
    """The pooling recorded on the handle must decide which wrapper is traced."""
    backend = xlmr.XlmRobertaBackend()
    model = torch.nn.Identity()

    mean_wrapper = backend.wrap(_loaded(model, "embedding", pooling="mean"))
    cls_wrapper = backend.wrap(_loaded(model, "embedding", pooling="cls"))
    reranker_wrapper = backend.wrap(_loaded(model, "reranker"))

    assert isinstance(mean_wrapper, common.EmbeddingWrapper)
    assert isinstance(cls_wrapper, common.ClsEmbeddingWrapper)
    assert isinstance(reranker_wrapper, common.RerankerWrapper)
    assert not any(wrapper.training for wrapper in (mean_wrapper, cls_wrapper, reranker_wrapper))


@pytest.mark.parametrize("pooling", [None, "max", "lasttoken", ""])
def test_wrap_rejects_a_pooling_no_wrapper_implements(pooling: str | None) -> None:
    """An embedding handle with an unknown pooling must raise, not fall back to mean."""
    backend = xlmr.XlmRobertaBackend()

    with pytest.raises(ValueError, match="pooling"):
        backend.wrap(_loaded(torch.nn.Identity(), "embedding", pooling=pooling))


def test_mean_and_cls_wrappers_return_the_hand_computed_vectors() -> None:
    """Mean pooling must ignore padding; CLS pooling must return the first row."""
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # (1, 3, 2)
    attention_mask = torch.tensor([[1, 1, 0]])
    input_ids = torch.tensor([[7, 8, 0]])
    model = _FixedHiddenModel(hidden)

    with torch.no_grad():
        pooled_mean = common.EmbeddingWrapper(model)(input_ids, attention_mask)
        pooled_cls = common.ClsEmbeddingWrapper(model)(input_ids, attention_mask)

    assert torch.allclose(pooled_mean, torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(pooled_cls, torch.tensor([[1.0, 2.0]]))


def test_cls_wrapper_ignores_the_attention_mask() -> None:
    """The first position is never padding, so the mask must not change CLS pooling."""
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    model = _FixedHiddenModel(hidden)
    wrapper = common.ClsEmbeddingWrapper(model)
    input_ids = torch.tensor([[7, 8]])

    with torch.no_grad():
        full = wrapper(input_ids, torch.tensor([[1, 1]]))
        padded = wrapper(input_ids, torch.tensor([[1, 0]]))

    assert torch.equal(full, padded)


def test_reranker_wrapper_returns_the_raw_logits() -> None:
    """The graph must expose the raw logits; sigmoid happens outside it."""
    logits = torch.tensor([[0.25], [-1.5]])
    wrapper = common.RerankerWrapper(_FixedHiddenModel(logits))

    with torch.no_grad():
        result = wrapper(torch.tensor([[1, 2], [3, 4]]), torch.ones(2, 2, dtype=torch.long))

    assert torch.equal(result, logits)


# --- FP32 baselines ----------------------------------------------------------


def _expected_pooled(ids: np.ndarray, hidden_size: int, pooling: str) -> np.ndarray:
    """Compute the baseline the fake encoder implies, straight from the ids."""
    offsets = np.arange(hidden_size, dtype=np.float32)
    if pooling == "cls":
        return ids[:, 0].astype(np.float32)[:, None] + offsets
    mask = ids != _PAD_ID
    means = np.array(
        [row[row_mask].mean() for row, row_mask in zip(ids, mask, strict=True)], dtype=np.float32
    )
    return means[:, None] + offsets


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_encode_pytorch_matches_the_hand_computed_pooling(pooling: str) -> None:
    """Both baseline pooling paths must reproduce the pooled values exactly."""
    tokenizer = _FakeTokenizer()
    model = _FakeEncoder(hidden_size=3)
    texts = ["abc", "de"]
    seq_len = 8

    result = common.encode_pytorch(model, tokenizer, texts, seq_len, pooling=pooling)

    ids = tokenizer(texts, padding="max_length", truncation=True, max_length=seq_len)["input_ids"]
    expected = _expected_pooled(ids, model.config.hidden_size, pooling)
    assert result.dtype == np.float32
    assert result.shape == (len(texts), model.config.hidden_size)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-5)


def test_encode_pytorch_cls_differs_from_mean() -> None:
    """The two pooling modes must not silently collapse into one another."""
    tokenizer = _FakeTokenizer()
    model = _FakeEncoder()

    pooled_mean = common.encode_pytorch(model, tokenizer, ["abc"], 8, pooling="mean")
    pooled_cls = common.encode_pytorch(model, tokenizer, ["abc"], 8, pooling="cls")

    assert not np.allclose(pooled_mean, pooled_cls)


def test_encode_pytorch_defaults_to_mean_pooling() -> None:
    """The default must stay mean pooling, so existing callers keep their behaviour."""
    tokenizer = _FakeTokenizer()
    model = _FakeEncoder()

    default = common.encode_pytorch(model, tokenizer, ["abc"], 8)

    np.testing.assert_array_equal(
        default, common.encode_pytorch(model, tokenizer, ["abc"], 8, pooling="mean")
    )


@pytest.mark.parametrize("pooling", [None, "max", "MEAN", ""])
def test_encode_pytorch_rejects_an_unsupported_pooling(pooling: str | None) -> None:
    """An unknown pooling must raise before any forward pass, never default silently."""
    with pytest.raises(ValueError, match="pooling"):
        common.encode_pytorch(_FakeEncoder(), _FakeTokenizer(), ["abc"], 8, pooling=pooling)


# --- tokenization ------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "inputs"),
    [("embedding", ["abc", "de"]), ("reranker", [("ab", "cd"), ("ef", "gh")])],
)
def test_tokenize_returns_fixed_shape_int32_arrays(kind: str, inputs: list[Any]) -> None:
    """Tokenized Core ML inputs must be (N, S) int32 with only the two graph keys."""
    backend = xlmr.XlmRobertaBackend()
    loaded = _loaded(_FakeEncoder(), kind=kind, tokenizer=_FakeTokenizer())
    seq_len = 12

    tokens = backend.tokenize(loaded, inputs, seq_len)

    assert set(tokens) == {"input_ids", "attention_mask"}
    for key in ("input_ids", "attention_mask"):
        assert tokens[key].dtype == np.int32
        assert tokens[key].shape == (len(inputs), seq_len)
    assert tokens["attention_mask"].sum() > 0
    assert tokens["attention_mask"].sum() < len(inputs) * seq_len  # the rows really are padded


def test_tokenize_of_a_pair_encodes_both_sequences() -> None:
    """Pair encoding must feed the document to the tokenizer, not only the query."""
    backend = xlmr.XlmRobertaBackend()
    loaded = _loaded(_FakeEncoder(), kind="reranker", tokenizer=_FakeTokenizer())

    query_only = backend.tokenize(loaded, [("abc", "")], 16)["attention_mask"].sum()
    with_document = backend.tokenize(loaded, [("abc", "defg")], 16)["attention_mask"].sum()

    assert with_document > query_only


def test_tokenize_rejects_empty_inputs() -> None:
    """An empty batch would produce a zero-row graph input and must raise."""
    backend = xlmr.XlmRobertaBackend()
    loaded = _loaded(_FakeEncoder(), tokenizer=_FakeTokenizer())

    with pytest.raises(ValueError, match="inputs"):
        backend.tokenize(loaded, [], 8)


@pytest.mark.parametrize("seq_len", [0, -1])
def test_tokenize_rejects_a_non_positive_sequence_length(seq_len: int) -> None:
    """A non-positive fixed length is never a valid graph shape."""
    backend = xlmr.XlmRobertaBackend()
    loaded = _loaded(_FakeEncoder(), tokenizer=_FakeTokenizer())

    with pytest.raises(ValueError, match="seq_len"):
        backend.tokenize(loaded, ["abc"], seq_len)


# --- architecture assumptions, on a tiny randomly initialised model ----------

# Position budget of the tiny models below: small enough to run its own
# limit into the ground within a unit test.
_TINY_POSITIONS = 10


def _tiny_config(**overrides: Any) -> Any:
    """Build a minimal XLM-RoBERTa configuration for a unit-test-sized model."""
    from transformers.models.xlm_roberta import modeling_xlm_roberta

    return modeling_xlm_roberta.XLMRobertaConfig(
        vocab_size=64,
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=32,
        max_position_embeddings=_TINY_POSITIONS,
        type_vocab_size=1,
        pad_token_id=1,
        **overrides,
    )


def _tiny_model(for_classification: bool = False, seed: int = 0) -> torch.nn.Module:
    """Build a randomly initialised tiny model with tuple outputs, in eval mode."""
    from transformers.models.xlm_roberta import modeling_xlm_roberta

    config = _tiny_config(num_labels=1) if for_classification else _tiny_config()
    torch.manual_seed(seed)
    model_class = (
        modeling_xlm_roberta.XLMRobertaForSequenceClassification
        if for_classification
        else modeling_xlm_roberta.XLMRobertaModel
    )
    model = model_class(config)
    model.config.return_dict = False
    return model.eval()


def _tiny_inputs(seq_len: int, batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic (input_ids, attention_mask) tensors with padding."""
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(3, 64, (batch_size, seq_len), generator=generator)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[-1, seq_len // 2 :] = 0  # last row is half padding
    return input_ids, attention_mask


def test_max_seq_len_matches_the_length_the_model_actually_accepts(tmp_path: Path) -> None:
    """The reported limit must be the longest sequence the position table can index."""
    model = _tiny_model()
    model_dir = _model_dir(
        tmp_path,
        config={
            "architectures": ["XLMRobertaModel"],
            "max_position_embeddings": _TINY_POSITIONS,
        },
    )

    reported = xlmr.XlmRobertaBackend().max_seq_len(model_dir)

    assert reported == _TINY_POSITIONS - xlmr.POSITION_OFFSET
    usable_ids, usable_mask = _tiny_inputs(reported)
    too_long_ids, too_long_mask = _tiny_inputs(reported + 1)
    with torch.no_grad():
        model(input_ids=usable_ids, attention_mask=usable_mask)  # must be accepted
    with pytest.raises(IndexError):
        # One token more addresses a position the table does not have.
        with torch.no_grad():
            model(input_ids=too_long_ids, attention_mask=too_long_mask)


def test_backbone_returns_the_last_hidden_state_first() -> None:
    """The wrappers pool ``outputs[0]``; for this family that is the hidden state."""
    model = _tiny_model()
    input_ids, attention_mask = _tiny_inputs(6)

    with torch.no_grad():
        hidden = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooled = common.EmbeddingWrapper(model)(input_ids, attention_mask)
        cls_pooled = common.ClsEmbeddingWrapper(model)(input_ids, attention_mask)

    assert hidden.shape == (2, 6, model.config.hidden_size)
    assert torch.allclose(pooled, common.mean_pool(hidden, attention_mask))
    assert torch.allclose(cls_pooled, hidden[:, 0])


def test_sequence_classification_returns_the_logits_first() -> None:
    """The reranker wrapper exposes ``outputs[0]``; that must be the raw logits."""
    model = _tiny_model(for_classification=True)
    input_ids, attention_mask = _tiny_inputs(6)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        wrapped = common.RerankerWrapper(model)(input_ids, attention_mask)

    assert logits.shape == (2, 1)
    assert torch.equal(wrapped, logits)


def test_eager_and_sdpa_agree_without_any_patch() -> None:
    """No-op patching is only safe while the traced path equals the reference path."""
    model = _tiny_model()
    input_ids, attention_mask = _tiny_inputs(6)

    model.config._attn_implementation = "sdpa"
    with torch.no_grad():
        sdpa = model(input_ids=input_ids, attention_mask=attention_mask)[0]
    xlmr.XlmRobertaBackend().apply_patches(_loaded(model))
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        eager = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    assert torch.isfinite(eager).all()
    assert (eager - sdpa).abs().max().item() < 1e-6
