"""Tests for the BERT compile backend.

Three layers:

* ``eeane.compiler.backends.bert``: pooling detection from the
  sentence-transformers pooling module, the offset-free rule behind the
  effective sequence length, the fixtures, the segment-id adapter and the
  kind/pooling validation the pipeline relies on.
* The architecture assumptions this backend encodes, checked against a
  tiny randomly initialised model: that pinning ``token_type_ids`` to
  zeros equals omitting them, in eager mode and through
  ``torch.jit.trace``, and that the full configured position budget is
  addressable.
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

from eeane.compiler.backends import bert, common

# Ids the stand-in tokenizer below uses for the special/padding tokens.
_CLS_ID = 101
_SEP_ID = 102
_PAD_ID = 0


class _FakeTokenizer:
    """Deterministic stand-in for a HuggingFace tokenizer.

    Encodes each text as ``[CLS] one id per character [SEP]``, then
    truncates and pads to ``max_length``. It also returns a
    ``token_type_ids`` key -- as a real BERT tokenizer does -- so tests can
    verify that the tokenize helpers drop everything the compiled graph
    does not take.
    """

    def __call__(
        self,
        texts: list[str],
        padding: str | None = None,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
    ) -> dict[str, np.ndarray]:
        """Encode ``texts`` into fixed-shape int64 arrays."""
        assert max_length is not None, "the helpers must always pass a fixed length"
        input_ids = np.array([self._encode(text, max_length) for text in texts], dtype=np.int64)
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != _PAD_ID).astype(np.int64),
            "token_type_ids": np.zeros_like(input_ids),
        }

    @staticmethod
    def _encode(text: str, max_length: int) -> list[int]:
        """Build one padded/truncated id row."""
        ids = [_CLS_ID, *(ord(char) % 90 + 5 for char in text), _SEP_ID][:max_length]
        return ids + [_PAD_ID] * (max_length - len(ids))


class _RecordingModel(torch.nn.Module):
    """Backbone stand-in returning a fixed output and recording its segment ids."""

    def __init__(self, output: torch.Tensor, hidden_size: int = 2) -> None:
        """Store the output every forward call returns."""
        super().__init__()
        self.output = output
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.seen_token_type_ids: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Record the segment ids and return the stored output as a tuple."""
        self.seen_token_type_ids = token_type_ids
        return (self.output,)


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
    _write_json(model_dir / bert.CONFIG_FILENAME, config or {"architectures": ["BertModel"]})
    if pooling:
        _write_json(model_dir / common.POOLING_DIRNAME / common.POOLING_CONFIG_FILENAME, pooling)
    return model_dir


# --- pooling detection -------------------------------------------------------


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ({"pooling_mode_mean_tokens": True, "pooling_mode_cls_token": False}, "mean"),
        ({"pooling_mode_mean_tokens": False, "pooling_mode_cls_token": True}, "cls"),
        ({"pooling_mode_cls_token": True, "word_embedding_dimension": 768}, "cls"),
    ],
)
def test_read_pooling_mode_detects_the_declared_mode(
    tmp_path: Path, declaration: dict[str, Any], expected: str
) -> None:
    """Exactly one enabled supported flag must select that pooling mode."""
    model_dir = _model_dir(tmp_path, **declaration)

    assert bert.read_pooling_mode(model_dir) == expected


@pytest.mark.parametrize(
    "declaration",
    [
        {"pooling_mode_mean_tokens": True, "pooling_mode_cls_token": True},
        {"pooling_mode_mean_tokens": False, "pooling_mode_cls_token": False},
        {"pooling_mode_max_tokens": True},
        {"pooling_mode_mean_tokens": "true"},
    ],
    ids=["both", "neither", "unsupported", "string-flag"],
)
def test_read_pooling_mode_rejects_an_ambiguous_declaration(
    tmp_path: Path, declaration: dict[str, Any]
) -> None:
    """Anything but exactly one supported flag must raise instead of guessing."""
    model_dir = _model_dir(tmp_path, **{"word_embedding_dimension": 768, **declaration})

    with pytest.raises(ValueError, match="pooling"):
        bert.read_pooling_mode(model_dir)


def test_load_reports_an_undeclared_pooling_before_loading_weights(tmp_path: Path) -> None:
    """An embedding model without a pooling declaration must fail fast and clearly."""
    model_dir = _model_dir(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        bert.BertBackend().load(model_dir, "embedding")

    message = str(excinfo.value)
    assert common.POOLING_DIRNAME in message
    assert "pooling_mode_mean_tokens" in message
    assert "pooling_mode_cls_token" in message


# --- effective maximum sequence length ---------------------------------------


@pytest.mark.parametrize("configured", [512, 8192, 1])
def test_max_seq_len_reports_the_whole_position_budget(tmp_path: Path, configured: int) -> None:
    """Positions start at 0 here, so nothing may be subtracted from the budget."""
    model_dir = _model_dir(
        tmp_path,
        config={"architectures": ["BertModel"], "max_position_embeddings": configured},
    )

    assert bert.BertBackend().max_seq_len(model_dir) == configured


def test_max_seq_len_of_a_missing_directory_is_none(tmp_path: Path) -> None:
    """No config.json means no known limit, not a crash."""
    assert bert.BertBackend().max_seq_len(tmp_path / "absent") is None


@pytest.mark.parametrize(
    "config",
    [
        {"architectures": ["BertModel"]},
        {"max_position_embeddings": None},
        {"max_position_embeddings": "512"},
        {"max_position_embeddings": 512.0},
        {"max_position_embeddings": True},
        {"max_position_embeddings": 0},
        {"max_position_embeddings": -1},
    ],
    ids=["absent", "null", "string", "float", "bool", "zero", "negative"],
)
def test_max_seq_len_ignores_a_missing_or_unusable_value(
    tmp_path: Path, config: dict[str, Any]
) -> None:
    """A missing or nonsensical value must degrade to 'unknown', never to a bogus limit."""
    model_dir = _model_dir(tmp_path, config=config)

    assert bert.BertBackend().max_seq_len(model_dir) is None


def test_max_seq_len_of_a_corrupt_config_is_none(tmp_path: Path) -> None:
    """Unparsable JSON must not turn an optional check into a compile failure."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / bert.CONFIG_FILENAME).write_text("{not json", encoding="utf-8")

    assert bert.BertBackend().max_seq_len(model_dir) is None


# --- interface attributes and fixtures ---------------------------------------


def test_backend_declares_the_interface_attributes() -> None:
    """The backend must name itself (matching its registry key) and its only kind."""
    backend = bert.BertBackend()

    assert backend.name == "Bert"
    assert backend.supported_kinds == ("embedding",)


def test_output_name_of_the_supported_kind() -> None:
    """Embeddings must keep the graph output name of the engine."""
    assert bert.BertBackend().output_name("embedding") == "embedding"


def test_fixtures_are_single_texts() -> None:
    """trace/sanity/padding fixtures are single sequences: this backend encodes no pairs."""
    backend = bert.BertBackend()

    assert isinstance(backend.trace_example("embedding"), str)
    assert all(isinstance(text, str) and text for text in backend.sanity_spec("embedding").inputs)
    assert backend.padding_input("embedding") == ""


def test_fixtures_are_english() -> None:
    """Checkpoints of this family commonly ship an English-only WordPiece vocabulary.

    Fixtures in another language would encode to little more than [UNK]
    rows, which still compare cleanly against their own FP32 baseline but
    tell the self-check nothing about the model.
    """
    backend = bert.BertBackend()
    texts = [backend.trace_example("embedding"), *backend.sanity_spec("embedding").inputs]

    assert all(text.isascii() for text in texts), texts


def test_embedding_sanity_spec_declares_no_ordering() -> None:
    """Embedding fixtures are compared row-wise; there is no expected ordering."""
    spec = bert.BertBackend().sanity_spec("embedding")

    assert spec.inputs
    assert spec.relevant_index is None
    assert spec.irrelevant_index is None


def test_sanity_spec_inputs_are_immutable() -> None:
    """The fixtures a caller receives must not be corruptible module state."""
    inputs = bert.BertBackend().sanity_spec("embedding").inputs

    assert isinstance(inputs, tuple)
    with pytest.raises(TypeError):
        inputs[0] = "mutated"  # type: ignore[index]


@pytest.mark.parametrize(
    "method",
    ["trace_example", "sanity_spec", "padding_input", "output_name"],
)
def test_unknown_kind_is_rejected(method: str) -> None:
    """Every kind-dispatching method must reject an unsupported kind."""
    backend = bert.BertBackend()

    with pytest.raises(ValueError, match="kind"):
        getattr(backend, method)("classifier")


def test_load_rejects_unknown_kind(tmp_path: Path) -> None:
    """load must validate the kind before touching the filesystem."""
    with pytest.raises(ValueError, match="kind"):
        bert.BertBackend().load(tmp_path, "classifier")


@pytest.mark.parametrize("method", ["wrap", "tokenize"])
def test_handle_taking_methods_reject_an_unsupported_kind(method: str) -> None:
    """A handle carrying an unsupported kind must be rejected, not silently wrapped."""
    backend = bert.BertBackend()
    loaded = _loaded(torch.nn.Identity(), kind="classifier")
    arguments = {"tokenize": (["text"], 8)}.get(method, ())

    with pytest.raises(ValueError, match="kind"):
        getattr(backend, method)(loaded, *arguments)


def test_reference_outputs_rejects_an_unsupported_kind(tmp_path: Path) -> None:
    """The reference path must validate the kind before loading any weights."""
    with pytest.raises(ValueError, match="kind"):
        bert.BertBackend().reference_outputs(tmp_path, "classifier", ["text"], 8)


# --- rerankers of this family are refused, with the reason -------------------


def _assert_reranker_refusal(excinfo: pytest.ExceptionInfo[ValueError]) -> None:
    """Assert that a refusal names the kind and explains the segment-id reason."""
    message = str(excinfo.value)
    assert "reranker" in message
    assert "segment ids" in message
    assert "query/document" in message


@pytest.mark.parametrize(
    "method",
    ["trace_example", "sanity_spec", "padding_input", "output_name"],
)
def test_kind_taking_methods_refuse_a_reranker_with_the_reason(method: str) -> None:
    """A refused reranker must say why, not just that some kind was rejected."""
    backend = bert.BertBackend()

    with pytest.raises(ValueError) as excinfo:
        getattr(backend, method)("reranker")

    _assert_reranker_refusal(excinfo)


def test_load_refuses_a_reranker_before_touching_the_filesystem(tmp_path: Path) -> None:
    """A BERT cross-encoder must be refused before any weights are read."""
    with pytest.raises(ValueError) as excinfo:
        bert.BertBackend().load(tmp_path, "reranker")

    _assert_reranker_refusal(excinfo)


@pytest.mark.parametrize("method", ["wrap", "tokenize"])
def test_handle_taking_methods_refuse_a_reranker(method: str) -> None:
    """A handle carrying the reranker kind must be refused, not wrapped or encoded."""
    backend = bert.BertBackend()
    loaded = _loaded(torch.nn.Identity(), kind="reranker")
    arguments = {"tokenize": ([("query", "document")], 8)}.get(method, ())

    with pytest.raises(ValueError) as excinfo:
        getattr(backend, method)(loaded, *arguments)

    _assert_reranker_refusal(excinfo)


def test_reference_outputs_refuses_a_reranker(tmp_path: Path) -> None:
    """The reference path must refuse a reranker before loading any weights."""
    with pytest.raises(ValueError) as excinfo:
        bert.BertBackend().reference_outputs(tmp_path, "reranker", [("q", "d")], 8)

    _assert_reranker_refusal(excinfo)


def test_the_module_offers_no_reranker_fixtures() -> None:
    """Dead pair fixtures would suggest a reranker path that does not exist."""
    assert "reranker" not in bert.SANITY_SPECS
    assert "reranker" not in bert.OUTPUT_NAMES
    assert not hasattr(bert, "SANITY_PAIRS")
    assert not hasattr(bert, "TRACE_EXAMPLE_PAIR")
    assert not hasattr(bert, "BATCH_PADDING_PAIR")


def test_reference_outputs_rejects_empty_inputs(tmp_path: Path) -> None:
    """No inputs means nothing to compare; that must raise before loading weights."""
    with pytest.raises(ValueError, match="inputs"):
        bert.BertBackend().reference_outputs(tmp_path, "embedding", [], 8)


# --- patches (this architecture needs none) ----------------------------------


def test_apply_patches_is_a_no_op() -> None:
    """A model that needs no rewrite must come back untouched and unrewritten."""
    backend = bert.BertBackend()
    model = _RecordingModel(torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]))
    input_ids = torch.tensor([[7, 9]])
    attention_mask = torch.tensor([[1, 1]])
    with torch.no_grad():
        before = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    assert backend.apply_patches(_loaded(model)) == {}

    with torch.no_grad():
        after = model(input_ids=input_ids, attention_mask=attention_mask)[0]
    assert torch.equal(before, after)


def test_apply_patches_rejects_a_mask_fill_value() -> None:
    """An unimplemented remedy must be refused loudly, not accepted and ignored."""
    backend = bert.BertBackend()

    with pytest.raises(ValueError, match="mask fill"):
        backend.apply_patches(_loaded(torch.nn.Identity()), mask_fill_value=-30000.0)


# --- wrappers and the zero-segment adapter -----------------------------------


def test_wrap_selects_the_wrapper_of_the_detected_pooling() -> None:
    """The pooling recorded on the handle must decide which wrapper is traced."""
    backend = bert.BertBackend()
    model = torch.nn.Identity()

    mean_wrapper = backend.wrap(_loaded(model, "embedding", pooling="mean"))
    cls_wrapper = backend.wrap(_loaded(model, "embedding", pooling="cls"))

    assert isinstance(mean_wrapper, common.EmbeddingWrapper)
    assert isinstance(cls_wrapper, common.ClsEmbeddingWrapper)
    assert not any(wrapper.training for wrapper in (mean_wrapper, cls_wrapper))


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_wrap_puts_the_zero_segment_adapter_around_the_model(pooling: str) -> None:
    """Every wrapper must reach the model through the segment-id adapter."""
    backend = bert.BertBackend()
    model = torch.nn.Identity()

    wrapper = backend.wrap(_loaded(model, "embedding", pooling=pooling))

    assert isinstance(wrapper.model, bert.ZeroTokenTypeModel)
    assert wrapper.model.model is model
    assert wrapper.model.training is False


@pytest.mark.parametrize("pooling", [None, "max", "lasttoken", ""])
def test_wrap_rejects_a_pooling_no_wrapper_implements(pooling: str | None) -> None:
    """An embedding handle with an unknown pooling must raise, not fall back to mean."""
    backend = bert.BertBackend()

    with pytest.raises(ValueError, match="pooling"):
        backend.wrap(_loaded(torch.nn.Identity(), "embedding", pooling=pooling))


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [("mean", [[2.0, 3.0]]), ("cls", [[1.0, 2.0]])],
)
def test_wrapped_pooling_returns_the_hand_computed_vectors(
    pooling: str, expected: list[list[float]]
) -> None:
    """Mean pooling must ignore padding; CLS pooling must return the first row."""
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # (1, 3, 2)
    model = _RecordingModel(hidden)
    wrapper = bert.BertBackend().wrap(_loaded(model, "embedding", pooling=pooling))

    with torch.no_grad():
        pooled = wrapper(torch.tensor([[7, 8, 0]]), torch.tensor([[1, 1, 0]]))

    assert torch.allclose(pooled, torch.tensor(expected))


@pytest.mark.parametrize("input_ids", [torch.tensor([[5, 6, 0]]), torch.tensor([[1, 2], [3, 4]])])
def test_adapter_passes_all_zero_segment_ids(input_ids: torch.Tensor) -> None:
    """The traced graph must pin the segment ids, not leave them to the caller."""
    model = _RecordingModel(torch.zeros(input_ids.shape[0], input_ids.shape[1], 2))
    adapter = bert.ZeroTokenTypeModel(model)

    with torch.no_grad():
        adapter(input_ids, torch.ones_like(input_ids))

    seen = model.seen_token_type_ids
    assert seen is not None
    assert seen.shape == input_ids.shape
    assert seen.dtype == input_ids.dtype
    assert torch.equal(seen, torch.zeros_like(input_ids))


def test_adapter_returns_the_model_output_tuple_unchanged() -> None:
    """The adapter must stay transparent: the wrappers read ``outputs[0]``."""
    output = torch.tensor([[[1.0, 2.0]]])
    adapter = bert.ZeroTokenTypeModel(_RecordingModel(output))

    with torch.no_grad():
        result = adapter(torch.tensor([[5]]), torch.tensor([[1]]))

    assert isinstance(result, tuple)
    assert torch.equal(result[0], output)


# --- tokenization ------------------------------------------------------------


def test_tokenize_returns_fixed_shape_int32_arrays() -> None:
    """Tokenized Core ML inputs must be (N, S) int32 with only the two graph keys."""
    backend = bert.BertBackend()
    loaded = _loaded(torch.nn.Identity(), tokenizer=_FakeTokenizer())
    inputs = ["abc", "de"]
    seq_len = 12

    tokens = backend.tokenize(loaded, inputs, seq_len)

    assert set(tokens) == {"input_ids", "attention_mask"}
    for key in ("input_ids", "attention_mask"):
        assert tokens[key].dtype == np.int32
        assert tokens[key].shape == (len(inputs), seq_len)
    assert tokens["attention_mask"].sum() > 0
    assert tokens["attention_mask"].sum() < len(inputs) * seq_len  # the rows really are padded


def test_tokenize_drops_the_segment_ids_a_bert_tokenizer_produces() -> None:
    """A BERT tokenizer returns token_type_ids, which the compiled graph does not take."""
    backend = bert.BertBackend()
    tokenizer = _FakeTokenizer()
    loaded = _loaded(torch.nn.Identity(), tokenizer=tokenizer)

    tokens = backend.tokenize(loaded, ["abc"], 12)

    assert "token_type_ids" in tokenizer(["abc"], max_length=12)
    assert "token_type_ids" not in tokens


def test_tokenize_rejects_empty_inputs() -> None:
    """An empty batch would produce a zero-row graph input and must raise."""
    backend = bert.BertBackend()
    loaded = _loaded(torch.nn.Identity(), tokenizer=_FakeTokenizer())

    with pytest.raises(ValueError, match="inputs"):
        backend.tokenize(loaded, [], 8)


@pytest.mark.parametrize("seq_len", [0, -1])
def test_tokenize_rejects_a_non_positive_sequence_length(seq_len: int) -> None:
    """A non-positive fixed length is never a valid graph shape."""
    backend = bert.BertBackend()
    loaded = _loaded(torch.nn.Identity(), tokenizer=_FakeTokenizer())

    with pytest.raises(ValueError, match="seq_len"):
        backend.tokenize(loaded, ["abc"], seq_len)


# --- architecture assumptions, on a tiny randomly initialised model ----------

# Position budget of the tiny models below: small enough to run its own
# limit into the ground within a unit test.
_TINY_POSITIONS = 10


def _tiny_model(seed: int = 0) -> torch.nn.Module:
    """Build a randomly initialised tiny backbone with tuple outputs, in eval mode."""
    from transformers.models.bert import modeling_bert

    config = modeling_bert.BertConfig(
        vocab_size=64,
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=32,
        max_position_embeddings=_TINY_POSITIONS,
        # Two segments, so a wrong token_type_ids value changes the output.
        type_vocab_size=2,
        pad_token_id=0,
    )
    torch.manual_seed(seed)
    model = modeling_bert.BertModel(config, add_pooling_layer=False)
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
        config={"architectures": ["BertModel"], "max_position_embeddings": _TINY_POSITIONS},
    )

    reported = bert.BertBackend().max_seq_len(model_dir)

    # No leading position is reserved here, unlike the offset families.
    assert reported == _TINY_POSITIONS
    usable_ids, usable_mask = _tiny_inputs(reported)
    too_long_ids, too_long_mask = _tiny_inputs(reported + 1)
    with torch.no_grad():
        model(input_ids=usable_ids, attention_mask=usable_mask)  # must be accepted
    with pytest.raises(RuntimeError):
        # One token more addresses a position the table does not have.
        with torch.no_grad():
            model(input_ids=too_long_ids, attention_mask=too_long_mask)


def test_zero_segment_ids_equal_the_omitted_argument() -> None:
    """Pinning the segment ids to zeros must reproduce HuggingFace's own fallback.

    This is what lets the compiled graph take ``input_ids`` and
    ``attention_mask`` only while the FP32 baseline calls the model
    without any segment ids at all.
    """
    model = _tiny_model()
    input_ids, attention_mask = _tiny_inputs(6)
    adapter = bert.ZeroTokenTypeModel(model)

    with torch.no_grad():
        omitted = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        pinned = adapter(input_ids, attention_mask)[0]

    assert torch.equal(omitted, pinned)


def test_the_segment_ids_really_change_the_output() -> None:
    """Guard for the test above: a model insensitive to segments would prove nothing."""
    model = _tiny_model()
    input_ids, attention_mask = _tiny_inputs(6)

    with torch.no_grad():
        zeros = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        ones = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.ones_like(input_ids),
        )[0]

    assert (zeros - ones).abs().max().item() > 1e-3


def test_tracing_keeps_the_zero_segment_ids() -> None:
    """The traced graph must keep the pinned zeros for inputs it was not traced on.

    Traces exactly what the pipeline traces -- the wrapper returned by
    ``wrap`` -- and replays it on a different batch, so a segment tensor
    accidentally captured from the tracing example would show up.
    """
    model = _tiny_model()
    wrapper = bert.BertBackend().wrap(_loaded(model, "embedding", pooling="cls"))

    traced = torch.jit.trace(wrapper, _tiny_inputs(6), strict=False)
    other_ids, other_mask = _tiny_inputs(6, batch_size=3)
    with torch.no_grad():
        expected = model(input_ids=other_ids, attention_mask=other_mask)[0][:, 0]
        replayed = traced(other_ids, other_mask)

    assert replayed.shape == expected.shape
    assert torch.equal(replayed, expected)


def test_backbone_returns_the_last_hidden_state_first() -> None:
    """The wrappers pool ``outputs[0]``; for this family that is the hidden state."""
    model = _tiny_model()
    input_ids, attention_mask = _tiny_inputs(6)
    backend = bert.BertBackend()

    mean_wrapper = backend.wrap(_loaded(model, "embedding", pooling="mean"))
    cls_wrapper = backend.wrap(_loaded(model, "embedding", pooling="cls"))
    with torch.no_grad():
        hidden = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooled = mean_wrapper(input_ids, attention_mask)
        cls_pooled = cls_wrapper(input_ids, attention_mask)

    assert hidden.shape == (2, 6, model.config.hidden_size)
    assert torch.allclose(pooled, common.mean_pool(hidden, attention_mask))
    assert torch.allclose(cls_pooled, hidden[:, 0])


def test_eager_and_sdpa_agree_without_any_patch() -> None:
    """No-op patching is only safe while the traced path equals the reference path."""
    model = _tiny_model()
    input_ids, attention_mask = _tiny_inputs(6)

    model.config._attn_implementation = "sdpa"
    with torch.no_grad():
        sdpa = model(input_ids=input_ids, attention_mask=attention_mask)[0]
    bert.BertBackend().apply_patches(_loaded(model))
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        eager = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    assert torch.isfinite(eager).all()
    assert (eager - sdpa).abs().max().item() < 1e-6


# --- round trip through a synthetic saved model directory --------------------

# Sequence length of the round-trip tests: within the tiny position budget
# and short enough to leave the fixtures visibly truncated and padded.
_ROUND_TRIP_SEQ_LEN = 8

# Tolerance between the wrapper (eager) and the FP32 baseline (sdpa); what
# is left is the kernel difference between the two attention paths.
_ROUND_TRIP_TOLERANCE = 1e-5


def _write_model_directory(directory: Path, pooling_flag: str) -> Path:
    """Save a tiny randomly initialised embedding model as a HuggingFace directory.

    Args:
        directory: Destination directory; created if needed.
        pooling_flag: sentence-transformers pooling flag to enable.

    Returns:
        ``directory``, holding weights, config, tokenizer files and the
        pooling declaration.
    """
    from transformers import BertTokenizerFast

    directory.mkdir(parents=True, exist_ok=True)
    # A hand-written vocabulary keeps the tokenizer offline and well below
    # the tiny model's vocab_size, so every id it emits is addressable.
    vocab_path = directory / "vocab.txt"
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    tokens += [chr(code) for code in range(ord("a"), ord("z") + 1)]
    tokens += ["mount", "fuji", "how", "tall", "is", "the", "question"]
    vocab_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    BertTokenizerFast(vocab_file=str(vocab_path)).save_pretrained(directory)
    _tiny_model().save_pretrained(directory)
    _write_json(
        directory / common.POOLING_DIRNAME / common.POOLING_CONFIG_FILENAME,
        {"word_embedding_dimension": 16, pooling_flag: True},
    )
    return directory


@pytest.fixture(scope="module")
def embedding_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic embedding model directory declaring CLS pooling."""
    return _write_model_directory(
        tmp_path_factory.mktemp("bert-embedding"), pooling_flag="pooling_mode_cls_token"
    )


def test_load_returns_a_conforming_handle(embedding_dir: Path) -> None:
    """load must hand back an eval/FP32/tuple-output model plus its tokenizer and config."""
    loaded = bert.BertBackend().load(embedding_dir, "embedding")

    assert loaded.kind == "embedding"
    assert loaded.attn == "eager"
    assert loaded.pooling == "cls"  # the pooling module of the directory
    assert loaded.model_dir == embedding_dir
    assert loaded.model.training is False
    assert next(loaded.model.parameters()).dtype == torch.float32
    assert loaded.config.return_dict is False
    assert loaded.config is loaded.model.config
    assert loaded.tokenizer("x")["input_ids"]


def test_load_skips_the_unused_pooler_head_of_an_embedding_model(embedding_dir: Path) -> None:
    """The pooler is dead weight for an embedding model and must not be built."""
    loaded = bert.BertBackend().load(embedding_dir, "embedding")

    assert loaded.model.pooler is None


def test_wrapper_matches_the_fp32_reference(embedding_dir: Path) -> None:
    """The traced module and the baseline must compute the same function.

    Both sides go through this backend's own stages, so a disagreement
    about the pinned segment ids, the pooling or the tokenization would
    show up here rather than only against real weights.
    """
    backend = bert.BertBackend()
    inputs = list(backend.sanity_spec("embedding").inputs)
    loaded = backend.load(embedding_dir, "embedding")
    wrapper = backend.wrap(loaded)
    tokens = backend.tokenize(loaded, inputs, _ROUND_TRIP_SEQ_LEN)

    with torch.no_grad():
        wrapped = wrapper(
            torch.from_numpy(tokens["input_ids"]).long(),
            torch.from_numpy(tokens["attention_mask"]).long(),
        )
    reference = backend.reference_outputs(embedding_dir, "embedding", inputs, _ROUND_TRIP_SEQ_LEN)

    assert tokens["input_ids"].shape == (len(inputs), _ROUND_TRIP_SEQ_LEN)
    assert np.isfinite(reference).all()
    np.testing.assert_allclose(
        wrapped.numpy().reshape(len(inputs), -1),
        np.asarray(reference, dtype=np.float32).reshape(len(inputs), -1),
        rtol=0,
        atol=_ROUND_TRIP_TOLERANCE,
    )
