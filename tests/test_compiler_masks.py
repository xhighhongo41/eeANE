"""Tests for the in-graph additive attention masks shared by the encoder backends.

Three layers:

* ``eeane.compiler.backends.masks`` on its own: the shape, dtype and exact
  values of the padding mask, and the single fill value it shares with
  the decoder-style backends.
* Equivalence with the framework's own mask construction, compared as
  sets of hidden positions on tiny configurations: the padding mask
  against the bidirectional mask of a BERT-style stack, and both masks of
  the sliding-window builder against those of a ModernBERT stack.
* The adapter around tiny randomly initialised models: that handing the
  model the built 4-D mask computes what the framework's own path
  computes (eager and sdpa), and that tracing it records none of the
  nodes the Core ML converter rejects.

Everything here runs on tiny stand-in modules and randomly initialised
models; nothing needs real model weights.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch
from transformers import masking_utils

from eeane.compiler.backends import decoder_patches, gemma3, masks

# Graph nodes the Core ML converter has no translation for: the op the
# framework's mask construction materializes its mask with, and the cast
# of a traced size to a Python integer.
_NEW_ONES_NODE = "aten::new_ones"
_DYNAMIC_SIZE_NODE = "aten::Int"

# Tolerance between the adapter and the framework's own path; what is left
# is the difference between the finite fill and the float32 minimum, plus
# the kernel difference between the attention implementations.
_OUTPUT_TOLERANCE = 1e-5

# A right-padded batch with one full row, one partly padded row and one
# row of a single real token.
_PADDED_ROWS = [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0],
]

# Dtypes a 2-D attention mask arrives in: the tokenizer's int64, the int32
# the compiled graph takes, and a float mask.
_MASK_DTYPES = [torch.long, torch.int32, torch.float32]


def _padded_mask(dtype: torch.dtype = torch.long) -> torch.Tensor:
    """Return :data:`_PADDED_ROWS` as a 2-D mask of the given dtype."""
    return torch.tensor(_PADDED_ROWS, dtype=dtype)


def _hidden(mask: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    """Return the positions an additive mask hides, broadcast to ``(B, 1, S, S)``."""
    return (mask != 0).expand(batch_size, 1, seq_len, seq_len)


def _node_kinds(module: torch.nn.Module, example: tuple[torch.Tensor, ...]) -> set[str]:
    """Trace ``module`` on ``example`` and return the kinds of its inlined graph nodes."""
    with warnings.catch_warnings():
        # Tracer warnings about Python values are expected from the
        # framework's modules and say nothing about the masks.
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(module, example, strict=False)
    return {node.kind() for node in traced.inlined_graph.nodes()}


# --- padding_mask ----------------------------------------------------------


@pytest.mark.parametrize("dtype", _MASK_DTYPES, ids=str)
def test_padding_mask_is_a_float32_mask_over_the_keys(dtype: torch.dtype) -> None:
    """The mask has one row per batch entry, broadcastable over heads and queries."""
    mask = masks.padding_mask(_padded_mask(dtype))

    assert mask.shape == (3, 1, 1, 6)
    assert mask.dtype == torch.float32


@pytest.mark.parametrize("dtype", _MASK_DTYPES, ids=str)
def test_padding_mask_is_zero_at_real_tokens_and_fill_at_padding(dtype: torch.dtype) -> None:
    """Real keys get exactly 0.0, padded keys exactly the fill value."""
    fill = -1e4
    expected = torch.tensor(
        [
            [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0, 0.0, fill, fill]]],
            [[[0.0, fill, fill, fill, fill, fill]]],
        ],
        dtype=torch.float32,
    )

    mask = masks.padding_mask(_padded_mask(dtype))

    assert torch.equal(mask, expected)


def test_padding_mask_honours_a_custom_fill_value() -> None:
    """A fill value given by the caller replaces the default one."""
    expected = torch.tensor([[[[0.0, 0.0, -7.5]]]], dtype=torch.float32)

    mask = masks.padding_mask(torch.tensor([[1, 1, 0]]), fill_value=-7.5)

    assert torch.equal(mask, expected)


def test_padding_mask_treats_any_non_zero_value_as_a_real_token() -> None:
    """A real token is marked by a non-zero value, not by the value 1 alone."""
    expected = torch.tensor([[[[0.0, 0.0, -1e4, 0.0]]]], dtype=torch.float32)

    from_int = masks.padding_mask(torch.tensor([[1, 3, 0, 2]]))
    from_float = masks.padding_mask(torch.tensor([[0.5, 1.0, 0.0, 2.0]]))

    assert torch.equal(from_int, expected)
    assert torch.equal(from_float, expected)


def test_padding_mask_of_a_row_without_any_real_token_stays_finite() -> None:
    """A fully padded row is filled everywhere, with a finite number."""
    mask = masks.padding_mask(torch.zeros(1, 4, dtype=torch.long))

    assert torch.isfinite(mask).all()
    assert torch.equal(mask, torch.full((1, 1, 1, 4), -1e4, dtype=torch.float32))


def test_the_fill_value_is_the_one_the_decoder_backends_use() -> None:
    """There is one finite fill value, shared with the decoder-style backends."""
    assert masks.MASK_FILL_VALUE == decoder_patches.MASK_FILL_VALUE
    assert masks.MASK_FILL_VALUE == -1e4


# --- equivalence with the framework's own masks ------------------------------


def _tiny_bert_config() -> Any:
    """Build a minimal BERT configuration, set to eager attention."""
    from transformers.models.bert import modeling_bert

    config = modeling_bert.BertConfig(
        vocab_size=64,
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=32,
        max_position_embeddings=10,
        type_vocab_size=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return config


def _tiny_xlmr_config() -> Any:
    """Build a minimal XLM-RoBERTa configuration, set to eager attention."""
    from transformers.models.xlm_roberta import modeling_xlm_roberta

    config = modeling_xlm_roberta.XLMRobertaConfig(
        vocab_size=64,
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=32,
        max_position_embeddings=10,
        type_vocab_size=1,
        pad_token_id=1,
    )
    config._attn_implementation = "eager"
    return config


def _tiny_modernbert_config() -> Any:
    """Build a minimal ModernBERT configuration with an inclusive window radius of 4."""
    from transformers.models.modernbert import modeling_modernbert

    config = modeling_modernbert.ModernBertConfig(
        vocab_size=128,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        max_position_embeddings=64,
        local_attention=8,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return config


@pytest.mark.parametrize("build_config", [_tiny_bert_config, _tiny_xlmr_config])
def test_padding_mask_hides_what_the_framework_hides(build_config: Callable[[], Any]) -> None:
    """The padding mask hides exactly the positions of the framework's bidirectional mask."""
    config = build_config()
    # One more row than _PADDED_ROWS: a row without any real token.
    mask_2d = torch.tensor([*_PADDED_ROWS, [0, 0, 0, 0, 0, 0]])
    batch_size, seq_len = mask_2d.shape
    embeds = torch.zeros(batch_size, seq_len, config.hidden_size)

    framework = masking_utils.create_bidirectional_mask(
        config=config, inputs_embeds=embeds, attention_mask=mask_2d
    )
    ours = masks.padding_mask(mask_2d)

    assert framework is not None
    assert framework.shape == (batch_size, 1, seq_len, seq_len)
    assert torch.equal(_hidden(ours, batch_size, seq_len), _hidden(framework, batch_size, seq_len))
    broadcast = ours.expand(batch_size, 1, seq_len, seq_len)
    assert (broadcast[framework == 0] == 0).all()
    assert (broadcast[framework != 0] == masks.MASK_FILL_VALUE).all()


def _sliding_example() -> tuple[Any, torch.Tensor, torch.Tensor]:
    """Return a tiny ModernBERT config, a padded (B, S) mask and matching embeddings.

    The sequence is four times the inclusive radius, so the window cuts
    keys off on both sides of the middle queries.
    """
    config = _tiny_modernbert_config()
    mask_2d = torch.ones(3, 16, dtype=torch.long)
    mask_2d[1, 8:] = 0
    mask_2d[2, 13:] = 0
    embeds = torch.zeros(3, 16, config.hidden_size)
    return config, mask_2d, embeds


def test_the_tiny_sliding_config_has_an_inclusive_radius_of_four() -> None:
    """Guard for the tests below: the window they pass is derived from this radius."""
    config, _, _ = _sliding_example()

    assert config.sliding_window == 4


def test_the_sliding_mask_hides_what_the_framework_hides() -> None:
    """The sliding mask, built with the radius plus one, matches the framework's."""
    config, mask_2d, embeds = _sliding_example()
    batch_size, seq_len = mask_2d.shape

    framework = masking_utils.create_bidirectional_sliding_window_mask(
        config=config, inputs_embeds=embeds, attention_mask=mask_2d
    )
    _, sliding = masks.bidirectional_masks(mask_2d, window=config.sliding_window + 1)

    assert framework is not None
    assert framework.shape == (batch_size, 1, seq_len, seq_len)
    assert torch.equal(
        _hidden(sliding, batch_size, seq_len), _hidden(framework, batch_size, seq_len)
    )
    assert (sliding[framework == 0] == 0).all()
    assert (sliding[framework != 0] == masks.MASK_FILL_VALUE).all()


def test_the_full_mask_hides_what_the_framework_hides() -> None:
    """The full mask of the sliding builder matches the framework's bidirectional mask."""
    config, mask_2d, embeds = _sliding_example()
    batch_size, seq_len = mask_2d.shape

    framework = masking_utils.create_bidirectional_mask(
        config=config, inputs_embeds=embeds, attention_mask=mask_2d
    )
    full, _ = masks.bidirectional_masks(mask_2d, window=config.sliding_window + 1)

    assert framework is not None
    assert framework.shape == (batch_size, 1, seq_len, seq_len)
    assert torch.equal(_hidden(full, batch_size, seq_len), _hidden(framework, batch_size, seq_len))
    assert (full[framework == 0] == 0).all()
    assert (full[framework != 0] == masks.MASK_FILL_VALUE).all()


def test_the_sliding_window_cuts_off_keys_on_both_sides() -> None:
    """A middle query sees the keys at most four positions away, and no further."""
    config, mask_2d, _ = _sliding_example()

    _, sliding = masks.bidirectional_masks(mask_2d, window=config.sliding_window + 1)

    visible = (sliding[0, 0, 8] == 0).nonzero().flatten().tolist()
    assert visible == [4, 5, 6, 7, 8, 9, 10, 11, 12]


def test_gemma3_builds_its_masks_with_the_shared_builder() -> None:
    """The Gemma 3 backend exposes the shared builder itself, not a copy of it."""
    assert gemma3.build_bidirectional_masks is masks.bidirectional_masks
    assert "build_bidirectional_masks" in gemma3.__all__


# --- AdditiveMaskModel -------------------------------------------------------


class _RecordingModel(torch.nn.Module):
    """Model stand-in that records its keyword inputs and returns a fixed tuple."""

    def __init__(self) -> None:
        """Prepare the fixed output and the record."""
        super().__init__()
        self.output = (torch.zeros(1),)
        self.seen: dict[str, torch.Tensor] = {}

    def forward(self, **kwargs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Record the inputs and return the stored output."""
        self.seen = kwargs
        return self.output


def test_the_adapter_hands_the_model_the_built_mask() -> None:
    """The model receives the ids unchanged and the 4-D mask in place of the 2-D one."""
    model = _RecordingModel()
    input_ids = torch.tensor([[5, 6, 7], [8, 9, 0]])
    mask_2d = torch.tensor([[1, 1, 1], [1, 1, 0]])
    expected_mask = torch.tensor(
        [[[[0.0, 0.0, 0.0]]], [[[0.0, 0.0, -1e4]]]],
        dtype=torch.float32,
    )

    output = masks.AdditiveMaskModel(model)(input_ids, mask_2d)

    assert output is model.output
    assert set(model.seen) == {"input_ids", "attention_mask"}
    assert torch.equal(model.seen["input_ids"], input_ids)
    assert torch.equal(model.seen["attention_mask"], expected_mask)


def test_the_adapter_masks_with_the_fill_value_it_was_given() -> None:
    """A fill value given to the adapter reaches the mask the model receives."""
    model = _RecordingModel()
    adapter = masks.AdditiveMaskModel(model, fill_value=-3)

    adapter(torch.tensor([[5, 0]]), torch.tensor([[1, 0]]))

    assert adapter.fill_value == -3.0
    assert isinstance(adapter.fill_value, float)
    assert torch.equal(
        model.seen["attention_mask"], torch.tensor([[[[0.0, -3.0]]]], dtype=torch.float32)
    )


def _tiny_bert() -> torch.nn.Module:
    """Build a randomly initialised tiny eager BertModel with tuple outputs."""
    from transformers.models.bert import modeling_bert

    torch.manual_seed(0)
    model = modeling_bert.BertModel(_tiny_bert_config(), add_pooling_layer=False)
    model.config._attn_implementation = "eager"
    model.config.return_dict = False
    return model.eval()


def _tiny_xlmr() -> torch.nn.Module:
    """Build a randomly initialised tiny eager XLMRobertaModel with tuple outputs."""
    from transformers.models.xlm_roberta import modeling_xlm_roberta

    torch.manual_seed(0)
    model = modeling_xlm_roberta.XLMRobertaModel(_tiny_xlmr_config())
    model.config._attn_implementation = "eager"
    model.config.return_dict = False
    return model.eval()


def _padded_inputs(model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic ids for :data:`_PADDED_ROWS`, padded with the model's pad id."""
    generator = torch.Generator().manual_seed(1)
    mask_2d = _padded_mask()
    input_ids = torch.randint(3, 64, mask_2d.shape, generator=generator)
    input_ids[mask_2d == 0] = model.config.pad_token_id
    return input_ids, mask_2d


_TINY_ENCODERS = pytest.mark.parametrize("build_model", [_tiny_bert, _tiny_xlmr])


@_TINY_ENCODERS
def test_the_padding_really_changes_the_output(build_model: Callable[[], torch.nn.Module]) -> None:
    """Guard for the tests below: a model ignoring its mask would prove nothing."""
    model = build_model()
    input_ids, mask_2d = _padded_inputs(model)
    real = mask_2d.to(torch.bool)

    with torch.no_grad():
        masked = model(input_ids=input_ids, attention_mask=mask_2d)[0]
        unmasked = model(input_ids=input_ids, attention_mask=torch.ones_like(mask_2d))[0]

    assert (masked[real] - unmasked[real]).abs().max().item() > 1e-3


@_TINY_ENCODERS
def test_the_adapter_computes_what_the_framework_path_computes(
    build_model: Callable[[], torch.nn.Module],
) -> None:
    """At every real token, the adapter's output equals the 2-D mask's through the framework."""
    model = build_model()
    input_ids, mask_2d = _padded_inputs(model)
    real = mask_2d.to(torch.bool)

    with torch.no_grad():
        reference = model(input_ids=input_ids, attention_mask=mask_2d)[0]
        adapted = masks.AdditiveMaskModel(model)(input_ids, mask_2d)[0]

    assert torch.isfinite(adapted).all()
    assert (adapted[real] - reference[real]).abs().max().item() <= _OUTPUT_TOLERANCE


@_TINY_ENCODERS
def test_the_adapter_computes_what_an_sdpa_copy_computes(
    build_model: Callable[[], torch.nn.Module],
) -> None:
    """At every real token, the eager adapter equals an sdpa copy given the 2-D mask."""
    model = build_model()
    sdpa_copy = copy.deepcopy(model)
    sdpa_copy.config._attn_implementation = "sdpa"
    input_ids, mask_2d = _padded_inputs(model)
    real = mask_2d.to(torch.bool)

    with torch.no_grad():
        reference = sdpa_copy(input_ids=input_ids, attention_mask=mask_2d)[0]
        adapted = masks.AdditiveMaskModel(model)(input_ids, mask_2d)[0]

    assert model.config._attn_implementation == "eager"
    assert (adapted[real] - reference[real]).abs().max().item() <= _OUTPUT_TOLERANCE


# --- tracing -----------------------------------------------------------------


class _TwoDimensionalMask(torch.nn.Module):
    """Hands a model the 2-D mask unchanged, as the shared pooling wrappers do."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Store the model to call."""
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        """Run the model on the 2-D mask."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


class _PinnedPositions(torch.nn.Module):
    """Hands a BERT model fixed segment ids and positions, and the mask it is given.

    The same inputs the BERT backend's adapter pins, without the mask that
    adapter builds as well: the tests below decide which mask reaches the
    model.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Store the model to call."""
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        """Run the model with zero segment ids and ``arange`` positions."""
        position_ids = torch.arange(
            input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
        ).unsqueeze(0)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(input_ids),
            position_ids=position_ids,
        )


def _unchanged(model: torch.nn.Module) -> torch.nn.Module:
    """Return ``model`` itself: a backbone that derives no dynamic size of its own."""
    return model


# Tiny backbones to trace, each with what removes a dynamic-size node of
# its own. Left to itself, BERT slices its position buffer with the traced
# sequence length and records one; pinning the positions, as its backend
# does, removes it, so what is left to look at is the mask. XLM-RoBERTa
# derives its positions from the ids and needs nothing.
_TRACED_ENCODERS = pytest.mark.parametrize(
    ("build_model", "pin_positions"),
    [(_tiny_bert, _PinnedPositions), (_tiny_xlmr, _unchanged)],
    ids=["bert", "xlmr"],
)


@_TRACED_ENCODERS
def test_the_frameworks_own_mask_path_records_new_ones(
    build_model: Callable[[], torch.nn.Module],
    pin_positions: Callable[[torch.nn.Module], torch.nn.Module],
) -> None:
    """Guard for the test below: handed the 2-D mask, the model records the rejected op."""
    model = build_model()
    example = _padded_inputs(model)

    kinds = _node_kinds(_TwoDimensionalMask(pin_positions(model)), example)

    assert _NEW_ONES_NODE in kinds


@_TRACED_ENCODERS
def test_the_traced_adapter_records_no_node_the_converter_rejects(
    build_model: Callable[[], torch.nn.Module],
    pin_positions: Callable[[torch.nn.Module], torch.nn.Module],
) -> None:
    """Handed the built mask, the model builds none and no dynamic size is recorded."""
    model = build_model()
    example = _padded_inputs(model)

    kinds = _node_kinds(masks.AdditiveMaskModel(pin_positions(model)), example)

    assert _NEW_ONES_NODE not in kinds
    assert _DYNAMIC_SIZE_NODE not in kinds


def test_the_traced_adapter_replays_another_padding_pattern() -> None:
    """The traced mask follows the input it is given, not the example it was traced on."""
    model = _tiny_bert()
    adapter = masks.AdditiveMaskModel(model)
    input_ids, mask_2d = _padded_inputs(model)
    other_mask = mask_2d.flip(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(adapter, (input_ids, mask_2d), strict=False)

    with torch.no_grad():
        expected = adapter(input_ids, other_mask)[0]
        replayed = traced(input_ids, other_mask)[0]

    assert torch.equal(replayed, expected)


class _SlidingMasks(torch.nn.Module):
    """Module returning the two masks of the sliding builder, for tracing."""

    def forward(self, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build both masks with a window of 3."""
        return masks.bidirectional_masks(attention_mask, window=3)


class _PaddingMask(torch.nn.Module):
    """Module returning the padding mask, for tracing."""

    def forward(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Build the padding mask."""
        return masks.padding_mask(attention_mask)


@pytest.mark.parametrize("module", [_SlidingMasks(), _PaddingMask()], ids=["sliding", "padding"])
def test_the_traced_builders_record_no_node_the_converter_rejects(module: torch.nn.Module) -> None:
    """Neither builder records the rejected op or a dynamic size."""
    kinds = _node_kinds(module, (_padded_mask(),))

    assert _NEW_ONES_NODE not in kinds
    assert _DYNAMIC_SIZE_NODE not in kinds
