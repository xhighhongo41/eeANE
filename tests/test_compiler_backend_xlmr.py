"""Tests for the XLM-RoBERTa compile backend and the shared backend helpers.

Three layers:

* ``eeane.compiler.backends.xlm_roberta``: pooling detection from the
  sentence-transformers pooling module, the position-offset rule behind
  the effective sequence length, the fixtures, and the kind/pooling
  validation the pipeline relies on.
* ``eeane.compiler.backends.common``: the CLS pooling added for this
  family (wrapper and FP32 baseline), checked against hand-computed
  values.
* The architecture assumptions this backend encodes, checked against a
  tiny randomly initialised model: that handing the model its padding mask
  already built in 4-D computes what the framework's own mask path
  computes, and that the traced graphs of an embedding model and of a
  reranker really convert to Core ML and agree with the FP32 wrapper.
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

import coremltools as ct
import numpy as np
import pytest
import torch

from eeane.compiler import conversion
from eeane.compiler.backends import common, masks
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
    """Backbone stand-in returning one fixed hidden state and recording its mask."""

    def __init__(self, hidden: torch.Tensor) -> None:
        """Store the hidden state every forward call returns."""
        super().__init__()
        self.hidden = hidden
        self.seen_attention_mask: torch.Tensor | None = None

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Record the mask and return the stored hidden state as a tuple."""
        self.seen_attention_mask = attention_mask
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


# --- patches (nothing is rebound; the record states how the mask is built) --


def test_apply_patches_records_how_the_mask_is_built() -> None:
    """The record names the mask the graph is traced with, as a JSON-serializable dict."""
    backend = xlmr.XlmRobertaBackend()

    applied = backend.apply_patches(_loaded(_FakeEncoder()))

    assert applied == {"mask_4d_static": True, "mask_fill_value": masks.MASK_FILL_VALUE}
    assert json.loads(json.dumps(applied)) == applied


def test_apply_patches_leaves_the_model_untouched() -> None:
    """Nothing is rebound or rewritten: the model computes the same before and after."""
    backend = xlmr.XlmRobertaBackend()
    model = _FakeEncoder()
    input_ids = torch.tensor([[1, 7, 9, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    with torch.no_grad():
        before = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    backend.apply_patches(_loaded(model))

    with torch.no_grad():
        after = model(input_ids=input_ids, attention_mask=attention_mask)[0]
    assert torch.equal(before, after)


def test_apply_patches_rejects_a_mask_fill_value() -> None:
    """A fill value other than the one the graph uses must be refused, not claimed."""
    backend = xlmr.XlmRobertaBackend()

    with pytest.raises(ValueError, match="mask fill"):
        backend.apply_patches(_loaded(_FakeEncoder()), mask_fill_value=-30000.0)


def test_apply_patches_accepts_the_fill_value_the_graph_uses() -> None:
    """Asking for exactly the value the graph already uses is a no-op, not an error."""
    backend = xlmr.XlmRobertaBackend()

    applied = backend.apply_patches(_loaded(_FakeEncoder()), mask_fill_value=masks.MASK_FILL_VALUE)

    assert applied == backend.apply_patches(_loaded(_FakeEncoder()))


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


@pytest.mark.parametrize(
    ("kind", "pooling"), [("embedding", "mean"), ("embedding", "cls"), ("reranker", None)]
)
def test_wrap_puts_the_mask_adapter_around_the_model(kind: str, pooling: str | None) -> None:
    """Every wrapper must reach the model through the adapter that builds the 4-D mask."""
    backend = xlmr.XlmRobertaBackend()
    model = torch.nn.Identity()

    wrapper = backend.wrap(_loaded(model, kind, pooling=pooling))

    assert isinstance(wrapper.model, masks.AdditiveMaskModel)
    assert wrapper.model.model is model
    assert wrapper.model.fill_value == masks.MASK_FILL_VALUE
    assert wrapper.model.training is False


@pytest.mark.parametrize(
    ("kind", "pooling"), [("embedding", "mean"), ("embedding", "cls"), ("reranker", None)]
)
def test_wrapped_model_receives_the_4d_padding_mask(kind: str, pooling: str | None) -> None:
    """The model must get the built additive mask in place of the 2-D one, and nothing else."""
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    model = _FixedHiddenModel(hidden)
    wrapper = xlmr.XlmRobertaBackend().wrap(_loaded(model, kind, pooling=pooling))
    attention_mask = torch.tensor([[1, 1, 0]])

    with torch.no_grad():
        # _FixedHiddenModel takes the two keyword inputs only, so a position
        # or segment argument passed on top would raise here.
        wrapper(torch.tensor([[7, 8, 1]]), attention_mask)

    seen = model.seen_attention_mask
    assert seen is not None
    assert seen.shape == (1, 1, 1, 3)
    assert torch.equal(seen, masks.padding_mask(attention_mask))


def test_wrapped_embedding_still_pools_with_the_2d_mask() -> None:
    """The adapter must not change pooling: padding is still ignored by the mean."""
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    wrapper = xlmr.XlmRobertaBackend().wrap(
        _loaded(_FixedHiddenModel(hidden), "embedding", pooling="mean")
    )

    with torch.no_grad():
        pooled = wrapper(torch.tensor([[7, 8, 1]]), torch.tensor([[1, 1, 0]]))

    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0]]))


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

# Padding index of the tiny models below. This family derives its positions
# from the ids, so a padded position has to carry this id to be one.
_TINY_PAD_ID = 1


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
        pad_token_id=_TINY_PAD_ID,
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
    with pytest.raises((IndexError, RuntimeError)):
        # One token more addresses a position the tables do not have: the
        # position embedding (IndexError) or, first in transformers 5, the
        # default segment ids gathered by position (RuntimeError).
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


# --- the 4-D mask path against the framework's own ---------------------------

# Tolerance between the wrapped model (finite-fill 4-D mask, eager attention)
# and the framework's own path (2-D mask, sdpa attention); what is left is
# the difference between the finite fill and the float32 minimum, plus the
# kernel difference between the two attention implementations.
_FRAMEWORK_PATH_TOLERANCE = 1e-5


def _padded_inputs(seq_len: int, batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (input_ids, attention_mask) whose masked positions hold the padding id."""
    input_ids, attention_mask = _tiny_inputs(seq_len, batch_size)
    input_ids[attention_mask == 0] = _TINY_PAD_ID
    return input_ids, attention_mask


def _redraw(layer: torch.nn.Linear, generator: torch.Generator, scale: float) -> None:
    """Draw the weights of ``layer`` again from a normal distribution of the given scale."""
    with torch.no_grad():
        layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * scale)


def _sensitive_model(for_classification: bool = False) -> torch.nn.Module:
    """Build a tiny model that reacts visibly to a key it is wrongly allowed to attend to.

    The default initialisation leaves the attention output negligible next
    to the residual stream and, for a classification model, the logits near
    1e-3. There, neither a padded key left visible nor an FP16 rounding
    error would show, so the attention projections (and the classification
    head) are drawn again at a larger scale.
    """
    model = _tiny_model(for_classification=for_classification)
    generator = torch.Generator().manual_seed(2)
    backbone = model.roberta if for_classification else model
    for layer in backbone.encoder.layer:
        attention = layer.attention
        for projection in (
            attention.self.query,
            attention.self.key,
            attention.self.value,
            attention.output.dense,
        ):
            _redraw(projection, generator, scale=0.5)
    if for_classification:
        _redraw(model.classifier.dense, generator, scale=0.5)
        _redraw(model.classifier.out_proj, generator, scale=0.5)
    return model


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_the_wrapped_embedding_equals_the_framework_path(pooling: str) -> None:
    """The pooled output of the wrapper equals the pooled framework output for a padded batch."""
    model = _sensitive_model()
    input_ids, attention_mask = _padded_inputs(6, batch_size=3)
    attention_mask[0, 5:] = 0  # a second, differently padded row
    input_ids[0, 5:] = _TINY_PAD_ID

    model.config._attn_implementation = "sdpa"
    with torch.no_grad():
        hidden = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        unmasked = model(input_ids=input_ids, attention_mask=torch.ones_like(attention_mask))[0]
    expected = common.mean_pool(hidden, attention_mask) if pooling == "mean" else hidden[:, 0]
    model.config._attn_implementation = "eager"
    wrapper = xlmr.XlmRobertaBackend().wrap(_loaded(model, "embedding", pooling=pooling))
    with torch.no_grad():
        wrapped = wrapper(input_ids, attention_mask)

    # Guard: the padding must matter here, or the comparison proves nothing.
    assert (hidden[attention_mask.bool()] - unmasked[attention_mask.bool()]).abs().max() > 1e-3
    assert wrapped.shape == expected.shape
    assert torch.isfinite(wrapped).all()
    assert (wrapped - expected).abs().max().item() <= _FRAMEWORK_PATH_TOLERANCE


def test_the_wrapped_reranker_equals_the_framework_path() -> None:
    """The logits of the wrapper equal the framework's logits for a padded batch."""
    model = _sensitive_model(for_classification=True)
    input_ids, attention_mask = _padded_inputs(6, batch_size=3)
    attention_mask[0, 5:] = 0
    input_ids[0, 5:] = _TINY_PAD_ID

    model.config._attn_implementation = "sdpa"
    with torch.no_grad():
        expected = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        unmasked = model(input_ids=input_ids, attention_mask=torch.ones_like(attention_mask))[0]
    model.config._attn_implementation = "eager"
    wrapper = xlmr.XlmRobertaBackend().wrap(_loaded(model, "reranker"))
    with torch.no_grad():
        wrapped = wrapper(input_ids, attention_mask)

    # Guard: the padding must matter here, or the comparison proves nothing.
    assert (expected - unmasked).abs().max() > 1e-3
    assert wrapped.shape == expected.shape == (3, 1)
    assert torch.isfinite(wrapped).all()
    assert (wrapped - expected).abs().max().item() <= _FRAMEWORK_PATH_TOLERANCE


# --- Core ML conversion of the traced wrapper --------------------------------

# Graph nodes the converter cannot translate: the op the framework
# materializes its own attention mask with, which the 4-D mask the wrapper
# hands over avoids, and the cast of a traced size to an integer.
_NEW_ONES_NODE = "aten::new_ones"
_DYNAMIC_SIZE_NODE = "aten::Int"

# Fixed shape of the conversion tests: the longest sequence the tiny position
# budget leaves room for, so a wrong position row would be a real difference.
_CONVERSION_SEQ_LEN = _TINY_POSITIONS - xlmr.POSITION_OFFSET
_CONVERSION_BATCH_SIZE = 2

# Minimum cosine between the converted FP16 embedding and the FP32 wrapper.
_CONVERSION_COSINE_THRESHOLD = 0.99

# Largest absolute difference between the converted FP16 logits and the FP32
# wrapper's; far below what ignoring the padding would change them by.
_CONVERSION_LOGIT_TOLERANCE = 1e-2


def _traced_wrapper(kind: str) -> tuple[torch.nn.Module, Any, dict[str, np.ndarray]]:
    """Build, patch, wrap and trace a tiny model the way the pipeline does.

    Args:
        kind: ``"embedding"`` (mean pooling) or ``"reranker"``.

    Returns:
        The FP32 wrapper, its traced module and the int32 tokens it was
        traced on.
    """
    backend = xlmr.XlmRobertaBackend()
    model = _sensitive_model(for_classification=kind == "reranker")
    model.config._attn_implementation = "eager"  # what ``load`` requests by default
    loaded = _loaded(model, kind, pooling="mean")
    backend.apply_patches(loaded)
    wrapper = backend.wrap(loaded)
    input_ids, attention_mask = _padded_inputs(_CONVERSION_SEQ_LEN, _CONVERSION_BATCH_SIZE)
    tokens = {
        "input_ids": input_ids.numpy().astype(np.int32),
        "attention_mask": attention_mask.numpy().astype(np.int32),
    }
    return wrapper, conversion.trace_model(wrapper, tokens), tokens


def _fp32_output(wrapper: torch.nn.Module, tokens: dict[str, np.ndarray]) -> np.ndarray:
    """Run the FP32 wrapper on ``tokens``."""
    with torch.no_grad():
        output = wrapper(
            torch.from_numpy(tokens["input_ids"]).long(),
            torch.from_numpy(tokens["attention_mask"]).long(),
        )
    return output.numpy().astype(np.float32)


def _converted_output(
    traced: Any, tokens: dict[str, np.ndarray], kind: str
) -> tuple[Any, np.ndarray]:
    """Convert ``traced`` to FP16 Core ML and predict ``tokens`` on the CPU.

    Returns:
        The converted model and its output as float32 of shape ``(B, -1)``.
    """
    backend = xlmr.XlmRobertaBackend()
    name = backend.output_name(kind)
    mlmodel = conversion.convert_model(
        traced, _CONVERSION_SEQ_LEN, "fp16", "macos13", name, batch_size=_CONVERSION_BATCH_SIZE
    )
    # The default compute units may put the program on the GPU or the Neural
    # Engine; the CPU is the one place the comparison is reproducible.
    cpu_model = ct.models.MLModel(mlmodel.package_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    prediction = cpu_model.predict(dict(tokens))
    output = np.asarray(prediction[name], dtype=np.float32).reshape(_CONVERSION_BATCH_SIZE, -1)
    return mlmodel, output


@pytest.mark.parametrize("kind", ["embedding", "reranker"])
def test_the_traced_graph_records_no_node_the_converter_rejects(kind: str) -> None:
    """Neither a mask built in the graph nor a position cast to a size may be recorded."""
    _, traced, _ = _traced_wrapper(kind)

    graph = str(traced.inlined_graph)

    assert _NEW_ONES_NODE not in graph
    assert _DYNAMIC_SIZE_NODE not in graph


def test_the_traced_embedding_converts_and_agrees_with_the_fp32_wrapper() -> None:
    """The converted embedding wrapper must compute the same embedding as the FP32 one."""
    wrapper, traced, tokens = _traced_wrapper("embedding")
    hidden_size = _tiny_config().hidden_size
    expected = _fp32_output(wrapper, tokens)

    mlmodel, embeddings = _converted_output(traced, tokens, "embedding")

    description = mlmodel.get_spec().description
    assert [tensor.name for tensor in description.input] == ["input_ids", "attention_mask"]
    assert [tensor.name for tensor in description.output] == ["embedding"]
    assert embeddings.shape == expected.shape == (_CONVERSION_BATCH_SIZE, hidden_size)
    assert bool(np.isfinite(embeddings).all())
    cosines = np.sum(embeddings * expected, axis=1) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(expected, axis=1)
    )
    assert float(cosines.min()) >= _CONVERSION_COSINE_THRESHOLD


def test_the_traced_reranker_converts_and_agrees_with_the_fp32_wrapper() -> None:
    """The converted reranker wrapper must compute the same logits as the FP32 one."""
    wrapper, traced, tokens = _traced_wrapper("reranker")
    expected = _fp32_output(wrapper, tokens)

    mlmodel, logits = _converted_output(traced, tokens, "reranker")

    description = mlmodel.get_spec().description
    assert [tensor.name for tensor in description.input] == ["input_ids", "attention_mask"]
    assert [tensor.name for tensor in description.output] == ["logits"]
    assert logits.shape == expected.shape == (_CONVERSION_BATCH_SIZE, 1)
    assert bool(np.isfinite(logits).all())
    assert float(np.abs(logits - expected).max()) <= _CONVERSION_LOGIT_TOLERANCE
