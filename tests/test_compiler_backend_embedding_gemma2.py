"""Tests for the EmbeddingGemma 2 compile backend (text embeddings only).

Six layers:

* The three conversion patches, checked against the upstream functions
  they replace, together with the check that each replacement really took
  effect.
* The FP16 range conditioning: what is measured, how the factors are
  derived from it, and that a model whose weights were rescaled -- by
  each factor alone and by all of them at once -- still computes the same
  function.
* The in-graph masks and the traceable wrapper, checked against the
  upstream model on its own attention path and mask construction, the
  extension of the sequence lengths the Neural Engine gets wrong, and the
  pooling that divides before it sums.
* The traced graph: no shape arithmetic, no tensor above rank four, with
  and without the extension and with and without split query rows.
* Conformance to the backend interface declared in
  ``eeane.compiler.backends.base``, the refusal of every directory this
  backend does not implement, and the dispatch to it.
* The round trip through a synthetic saved model directory, laid out as
  sentence-transformers 6 writes one, up to the Core ML conversion of the
  traced wrapper.

Nothing here downloads weights: every model is a small randomly
initialised EmbeddingGemma 2 model built from an in-test configuration.
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import inspect
import json
import math
import string
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
from transformers.models.embedding_gemma2 import modeling_embedding_gemma2
from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import (
    EmbeddingGemma2Config,
    EmbeddingGemma2TextConfig,
)
from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
    EmbeddingGemma2Model,
    EmbeddingGemma2TextModel,
)
from transformers.models.gemma3 import modeling_gemma3

from eeane.compiler import conversion, dispatch
from eeane.compiler.backends import attention_split, base, common, decoder_patches, fp16_ranges
from eeane.compiler.backends import embedding_gemma2 as eg2

# The upstream implementations, captured at import time (that is, before
# any test can have replaced them). They are both the comparison baseline
# for the patch tests and the value the autouse fixture restores.
_UPSTREAM_ROTATE_HALF = modeling_embedding_gemma2.rotate_half
_UPSTREAM_REPEAT_KV = modeling_embedding_gemma2.repeat_kv
_UPSTREAM_EAGER_ATTENTION = modeling_embedding_gemma2.eager_attention_forward

# The same three symbols of another family, which the patches of this
# backend must leave alone.
_OTHER_FAMILY_SYMBOLS = (
    modeling_gemma3.rotate_half,
    modeling_gemma3.repeat_kv,
    modeling_gemma3.eager_attention_forward,
)

# Geometry of the tiny model used throughout: two query heads per
# key/value head, so ``repeat_kv`` really has to expand something; two
# sliding layers followed by a full one, whose heads are twice as wide,
# so both layer geometries and both masks are in play.
_TINY_VOCAB = 64
_TINY_HIDDEN = 16
_TINY_INTERMEDIATE = 32
_TINY_LAYER_TYPES = ("sliding_attention", "sliding_attention", "full_attention")
_TINY_HEADS = 2
_TINY_KV_HEADS = 1
_TINY_HEAD_DIM = 8
_TINY_FULL_HEAD_DIM = 16
_TINY_PER_LAYER_INPUT = 8
_TINY_EMBEDDING_DIM = 12
_TINY_POSITIONS = 64
_TINY_LAYERS = len(_TINY_LAYER_TYPES)

# The window as the configuration states it -- an inclusive radius -- and
# the exclusive half-width the in-graph mask is built from.
_TINY_RADIUS = 2
_TINY_WINDOW = _TINY_RADIUS + 1

_PAD_ID = 0
_EOS_ID = 1
_BOS_ID = 2
_UNK_ID = 3

# Sequence lengths on either side of the window. At the short one no two
# positions are farther apart than the radius, so the sliding layers see
# everything; at the long one most pairs are out of reach of each other.
_SEQ_LEN_INSIDE_WINDOW = 3
_SEQ_LEN_OUTSIDE_WINDOW = 10

# A small stand-in for the band of sequence lengths the wrapper extends,
# and the length it extends them to: above the band, and no more than the
# shortest length of the band away from it.
_TEST_BAND = range(9, 13)
_TEST_PADDED_LENGTH = 16

# Sequence length of the round-trip and conversion tests: longer than the
# window, and short enough to leave the sanity fixtures visibly truncated.
_ROUND_TRIP_SEQ_LEN = 16

# A band holding the round-trip length, for the conversion of a graph that
# extends its sequence.
_ROUND_TRIP_BAND = range(13, 17)
_ROUND_TRIP_PADDED_LENGTH = 20

# Minimum cosine between two computations that claim to be the same
# function in FP32: the in-graph masks against the framework's own, the
# patched against the upstream functions, the rescaled against the
# original weights, the extended against the plain sequence.
_EQUIVALENCE_COSINE_THRESHOLD = 0.999999

# Largest absolute difference, relative to the largest magnitude of the
# compared output, between a model before and after its weights were
# rescaled by powers of two.
_CONDITIONED_RELATIVE_TOLERANCE = 1e-5

# Tolerance between the traced wrapper (patched eager, in-graph masks) and
# the FP32 baseline (sdpa, framework masks). What is left is the kernel
# difference between the two attention implementations.
_ROUND_TRIP_TOLERANCE = 1e-5

# Minimum cosine between the converted FP16 program and the FP32 wrapper.
_CONVERSION_COSINE_THRESHOLD = 0.99

# Graph node the Core ML converter cannot fold into a static slice; it is
# what a Python-level division or index on a traced shape lowers to.
_DYNAMIC_SIZE_NODE = "aten::Int"

# Highest tensor rank the traced graph may hold: the Neural Engine
# compiler rejects a subgraph working on anything wider.
_MAX_GRAPH_RANK = 4

# Graph node of the attention softmax: one per layer when the query rows
# are attended in one piece, one per layer and range when they are split.
_SOFTMAX_NODE = "aten::softmax"

# sentence-transformers 6 module types, spelled out as that series writes
# them into modules.json.
_ST6_TRANSFORMER = "sentence_transformers.base.modules.transformer.Transformer"
_ST6_POOLING = "sentence_transformers.sentence_transformer.modules.pooling.Pooling"
_ST6_NORMALIZE = "sentence_transformers.base.modules.normalize.Normalize"

# Characters the toy tokenizer knows besides its four special tokens:
# letters, the byte-level spelling of a space and a little punctuation,
# exactly filling the tiny vocabulary. Every other byte is unknown.
_TOY_ALPHABET = string.ascii_lowercase + string.ascii_uppercase + "Ġ.,:?'-!"


def _restore_upstream_symbols() -> None:
    """Point every symbol the backend rebinds back at its upstream function."""
    modeling_embedding_gemma2.rotate_half = _UPSTREAM_ROTATE_HALF
    modeling_embedding_gemma2.repeat_kv = _UPSTREAM_REPEAT_KV
    modeling_embedding_gemma2.eager_attention_forward = _UPSTREAM_EAGER_ATTENTION


@pytest.fixture(autouse=True)
def _restore_transformers_patches() -> Iterator[None]:
    """Undo the process-wide EmbeddingGemma 2 monkeypatches after every test.

    The patch functions rebind names in the transformers module, which
    affects every model of this architecture in the process, so each test
    starts and ends on the upstream implementations.
    """
    _restore_upstream_symbols()
    try:
        yield
    finally:
        _restore_upstream_symbols()


def _tiny_root_config(attention_bias: bool = False) -> EmbeddingGemma2Config:
    """Build the configuration of the tiny model, without image or audio towers."""
    text_config = EmbeddingGemma2TextConfig(
        vocab_size=_TINY_VOCAB,
        hidden_size=_TINY_HIDDEN,
        intermediate_size=_TINY_INTERMEDIATE,
        num_hidden_layers=_TINY_LAYERS,
        layer_types=list(_TINY_LAYER_TYPES),
        num_attention_heads=_TINY_HEADS,
        num_key_value_heads=_TINY_KV_HEADS,
        head_dim=_TINY_HEAD_DIM,
        per_layer_config={
            index: {"head_dim": _TINY_FULL_HEAD_DIM, "num_key_value_heads": _TINY_KV_HEADS}
            for index, layer_type in enumerate(_TINY_LAYER_TYPES)
            if layer_type == "full_attention"
        },
        hidden_size_per_layer_input=_TINY_PER_LAYER_INPUT,
        embedding_dim=_TINY_EMBEDDING_DIM,
        sliding_window=_TINY_RADIUS,
        max_position_embeddings=_TINY_POSITIONS,
        attention_bias=attention_bias,
        pad_token_id=_PAD_ID,
        eos_token_id=_EOS_ID,
        bos_token_id=_BOS_ID,
    )
    return EmbeddingGemma2Config(text_config=text_config, vision_config=None, audio_config=None)


def _tiny_root(seed: int = 0, attention_bias: bool = False) -> EmbeddingGemma2Model:
    """Build the tiny multimodal root model, holding the text model only."""
    torch.manual_seed(seed)
    return EmbeddingGemma2Model(_tiny_root_config(attention_bias)).eval()


def _tiny_model(
    seed: int = 0, attn: str = "eager", attention_bias: bool = False
) -> EmbeddingGemma2TextModel:
    """Build the tiny text model in eval mode, on the given attention path.

    Two calls with the same seed build the same weights, so a second copy
    on another attention path is the same model computed another way.
    """
    model = _tiny_root(seed, attention_bias).language_model
    model.set_attn_implementation(attn)
    model.config.return_dict = False
    return model.eval()


def _scaled_norms(model: EmbeddingGemma2TextModel) -> list[torch.nn.Module]:
    """Return every normalization layer of ``model`` that carries a weight."""
    return [
        module
        for module in model.modules()
        if type(module).__name__ == "EmbeddingGemma2RMSNorm" and module.with_scale
    ]


def _shaped_model(
    seed: int = 0, attn: str = "eager", attention_bias: bool = False
) -> EmbeddingGemma2TextModel:
    """Build the tiny model with the activation ranges the conditioning is for.

    A fresh model holds unit normalization weights, unit layer scalars and
    projections so small that every activation fits float16 by a wide
    margin. Here the normalization weights and layer scalars differ from
    one, and the projections whose outputs the conditioning rescales are
    enlarged until their outputs cross the bounds it derives its factors
    from -- in some layers and not in others -- so every factor has
    something to do and every rewrite is exercised on values that are
    not the trivial ones.
    """
    model = _tiny_model(seed, attn, attention_bias)
    generator = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        for norm in _scaled_norms(model):
            norm.weight.copy_(torch.rand(norm.weight.shape, generator=generator) + 0.5)
        for index, layer in enumerate(model.layers):
            layer.layer_scalar.fill_(0.5 + 0.5 * index)
        model.embed_tokens.weight.mul_(3000.0)
        for layer in model.layers:
            layer.self_attn.q_proj.weight.mul_(4000.0)
            layer.self_attn.k_proj.weight.mul_(8000.0)
            layer.self_attn.v_proj.weight.mul_(16000.0)
        model.layers[0].ple_block.per_layer_projection.weight.mul_(1e5)
        model.layers[1].mlp.down_proj.weight.mul_(1e5)
        model.layers[1].self_attn.o_proj.weight.mul_(100.0)
    return model


def _right_padded_batch(
    seq_len: int = 12, lengths: tuple[int, ...] = (7, 10), seed: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a right-padded batch whose rows have the given real lengths."""
    torch.manual_seed(seed)
    input_ids = torch.randint(_UNK_ID + 1, _TINY_VOCAB, (len(lengths), seq_len))
    attention_mask = torch.ones(len(lengths), seq_len, dtype=torch.long)
    for row, length in enumerate(lengths):
        attention_mask[row, length:] = 0
        input_ids[row, length:] = _PAD_ID
    return input_ids, attention_mask


def _mask_mapping(attention_mask: torch.Tensor, window: int = _TINY_WINDOW) -> dict[str, Any]:
    """Build the per-layer-type mask mapping the wrapper hands the backbone."""
    full, sliding = eg2.build_bidirectional_masks(attention_mask, window)
    return {eg2.FULL_ATTENTION: full, eg2.SLIDING_ATTENTION: sliding}


def _row_cosines(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return the cosine similarity of every pair of rows along the last axis."""
    return torch.nn.functional.cosine_similarity(left, right, dim=-1)


def _upstream_embedding(
    model: EmbeddingGemma2TextModel, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Mean-pool what the model computes with the framework's own 2-D mask path."""
    with torch.no_grad():
        hidden = model(input_ids=input_ids, attention_mask=attention_mask)[0]
    return common.mean_pool(hidden, attention_mask)


def _toy_tokenizer(model_max_length: int = _TINY_POSITIONS) -> PreTrainedTokenizerFast:
    """Build a byte-level toy tokenizer whose every id fits the tiny vocabulary.

    Byte-level vocabulary with no merges: every known byte is its own
    token, and every byte outside :data:`_TOY_ALPHABET` -- every byte of a
    Japanese or Chinese text among them -- is the unknown token. The
    template wraps the text in the begin- and end-of-sequence tokens, as
    this family's own tokenizer does, and the padding goes on the right.

    Args:
        model_max_length: Length the tokenizer declares; only used to keep
            its own warnings quiet, since every encoding here is truncated
            by the caller.
    """
    vocab = {"<pad>": _PAD_ID, "<eos>": _EOS_ID, "<bos>": _BOS_ID, "<unk>": _UNK_ID}
    first_character_id = len(vocab)
    for index, character in enumerate(_TOY_ALPHABET):
        vocab[character] = index + first_character_id
    assert len(vocab) == _TINY_VOCAB
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B <eos>",
        special_tokens=[("<bos>", _BOS_ID), ("<eos>", _EOS_ID)],
    )
    tokenizer.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        padding_side="right",
        model_max_length=model_max_length,
    )


@cache
def _shared_tokenizer() -> PreTrainedTokenizerFast:
    """Return one toy tokenizer shared by every handle built in memory."""
    return _toy_tokenizer()


def _loaded(
    model: Any,
    kind: str = "embedding",
    pooling: str | None = common.POOLING_MEAN,
    tokenizer: Any = None,
    model_dir: Path = Path("/nonexistent-model-dir"),
    dense: Any = None,
) -> base.LoadedModel:
    """Build the handle the backend interface passes between its stages.

    A handle always carries a tokenizer, as one returned by ``load``
    does: applying the patches measures the model on encoded sentences.
    """
    return base.LoadedModel(
        model=model,
        tokenizer=tokenizer if tokenizer is not None else _shared_tokenizer(),
        config=getattr(model, "config", None),
        model_dir=model_dir,
        kind=kind,
        attn="eager",
        pooling=pooling,
        dense=dense,
    )


def _example(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, np.ndarray]:
    """Turn a batch into the int32 arrays the tracing helper takes."""
    return {
        "input_ids": input_ids.numpy().astype(np.int32),
        "attention_mask": attention_mask.numpy().astype(np.int32),
    }


def _traced_ranks(traced: torch.jit.ScriptModule) -> list[int]:
    """Return the rank of every tensor a traced graph produces."""
    ranks: list[int] = []
    for node in traced.inlined_graph.nodes():
        for output in node.outputs():
            value_type = output.type()
            if isinstance(value_type, torch.TensorType) and value_type.dim() is not None:
                ranks.append(int(value_type.dim()))
    return ranks


def _softmax_nodes(traced: torch.jit.ScriptModule) -> list[Any]:
    """Return the softmax nodes of a traced graph."""
    return [node for node in traced.inlined_graph.nodes() if node.kind() == _SOFTMAX_NODE]


def _softmax_key_lengths(traced: torch.jit.ScriptModule) -> set[int]:
    """Return the number of keys every softmax of a traced graph runs over."""
    lengths: set[int] = set()
    for node in _softmax_nodes(traced):
        sizes = next(iter(node.inputs())).type().sizes()
        lengths.add(int(sizes[-1]))
    return lengths


def _cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the cosine similarity of every pair of rows of two arrays."""
    return np.sum(left * right, axis=1) / (
        np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    )


def _write_json(path: Path, payload: object) -> None:
    """Write ``payload`` as JSON, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _declared_config(**overrides: Any) -> dict[str, Any]:
    """Build the config.json fields this backend decides a directory on."""
    config: dict[str, Any] = {
        "architectures": ["EmbeddingGemma2Model"],
        "model_type": eg2.MODEL_TYPE,
        "text_config": {
            "model_type": eg2.TEXT_MODEL_TYPE,
            "max_position_embeddings": _TINY_POSITIONS,
        },
    }
    config.update(overrides)
    return {key: value for key, value in config.items() if value is not None}


def _model_dir(tmp_path: Path, config: dict[str, Any] | None = None, **pooling: Any) -> Path:
    """Build a weightless model directory with a config.json and pooling module."""
    model_dir = tmp_path / "model"
    _write_json(
        model_dir / eg2.CONFIG_FILENAME, config if config is not None else _declared_config()
    )
    if pooling:
        _write_json(model_dir / common.POOLING_DIRNAME / common.POOLING_CONFIG_FILENAME, pooling)
    return model_dir


# --- the conversion patches --------------------------------------------------


@pytest.mark.parametrize("head_dim", [_TINY_HEAD_DIM, _TINY_FULL_HEAD_DIM])
def test_patched_rotate_half_matches_the_upstream_formula(head_dim: int) -> None:
    """The chunk-based split must equal the slice-based upstream one, for both head widths."""
    torch.manual_seed(0)
    x = torch.randn(2, _TINY_HEADS, 6, head_dim)
    expected = _UPSTREAM_ROTATE_HALF(x)

    eg2.patch_rotate_half()

    assert modeling_embedding_gemma2.rotate_half is decoder_patches.rotate_half
    assert torch.equal(modeling_embedding_gemma2.rotate_half(x), expected)


def test_patched_repeat_kv_reproduces_the_upstream_head_order() -> None:
    """The rank-4 expansion must place every repeated head where upstream puts it."""
    torch.manual_seed(1)
    hidden_states = torch.randn(2, _TINY_KV_HEADS, 3, _TINY_HEAD_DIM)
    n_rep = _TINY_HEADS // _TINY_KV_HEADS
    expected = _UPSTREAM_REPEAT_KV(hidden_states, n_rep)

    eg2.patch_repeat_kv()
    patched = modeling_embedding_gemma2.repeat_kv(hidden_states, n_rep)

    assert modeling_embedding_gemma2.repeat_kv is decoder_patches.repeat_kv
    assert patched.dim() == 4
    assert torch.equal(patched, expected)


def test_the_installed_attention_is_the_upstream_attention_below_the_limit() -> None:
    """Unsplit, the replacement computes upstream's numbers, keywords of this family included."""
    torch.manual_seed(2)
    module = SimpleNamespace(
        head_dim=_TINY_HEAD_DIM, num_key_value_groups=_TINY_HEADS // _TINY_KV_HEADS, training=False
    )
    query = torch.randn(2, _TINY_HEADS, 6, _TINY_HEAD_DIM)
    key = torch.randn(2, _TINY_KV_HEADS, 6, _TINY_HEAD_DIM)
    value = torch.randn(2, _TINY_KV_HEADS, 6, _TINY_HEAD_DIM)
    _, mask = eg2.build_bidirectional_masks(torch.ones(2, 6, dtype=torch.long), _TINY_WINDOW)
    arguments = {"scaling": 1.0, "sliding_window": _TINY_WINDOW}
    expected = _UPSTREAM_EAGER_ATTENTION(module, query, key, value, mask, **arguments)

    eg2.patch_attention_query_chunks()
    actual = modeling_embedding_gemma2.eager_attention_forward(
        module, query, key, value, mask, **arguments
    )

    assert modeling_embedding_gemma2.eager_attention_forward is eg2._eager_attention_forward
    assert all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True))


def test_apply_patches_rebinds_the_three_module_functions() -> None:
    """The record must describe rewrites that really happened."""
    eg2.EmbeddingGemma2Backend().apply_patches(_loaded(_tiny_model()))

    assert modeling_embedding_gemma2.rotate_half is decoder_patches.rotate_half
    assert modeling_embedding_gemma2.repeat_kv is decoder_patches.repeat_kv
    assert modeling_embedding_gemma2.eager_attention_forward is eg2._eager_attention_forward


def test_apply_patches_leaves_another_family_alone() -> None:
    """Each backend rebinds the symbols of its own architecture module only."""
    eg2.EmbeddingGemma2Backend().apply_patches(_loaded(_tiny_model()))

    assert (
        modeling_gemma3.rotate_half,
        modeling_gemma3.repeat_kv,
        modeling_gemma3.eager_attention_forward,
    ) == _OTHER_FAMILY_SYMBOLS


def test_apply_patches_returns_the_documented_record() -> None:
    """The record is stored verbatim in the artifact metadata, so it must be JSON."""
    applied = eg2.EmbeddingGemma2Backend().apply_patches(_loaded(_tiny_model()))

    conditioning = applied.pop(fp16_ranges.FP16_RANGE_CONDITIONING_KEY)
    assert applied == {
        "rotate_half_static": True,
        "repeat_kv_rank4": True,
        "attention_query_chunks": True,
        "mask_fill_value": decoder_patches.MASK_FILL_VALUE,
    }
    assert applied["mask_fill_value"] == -1e4
    assert set(conditioning) == {
        "residual_divisor",
        "qkv_divisors",
        "ple_output_gains",
        "mlp_output_gains",
        "attention_output_gains",
        "calibration_inputs",
    }
    assert set(conditioning["qkv_divisors"]) == {"q", "k", "v"}
    for name in ("ple_output_gains", "mlp_output_gains", "attention_output_gains"):
        assert len(conditioning[name]) == _TINY_LAYERS
    assert json.loads(json.dumps(applied)) == applied
    assert json.loads(json.dumps(conditioning)) == conditioning


def test_the_record_names_every_patch_of_the_patch_list() -> None:
    """A rewrite added to the list must show up in the metadata without further wiring."""
    applied = eg2.EmbeddingGemma2Backend().apply_patches(_loaded(_tiny_model()))

    assert [name for name, _ in eg2.PATCHES] == [
        "rotate_half_static",
        "repeat_kv_rank4",
        "attention_query_chunks",
    ]
    assert all(applied[name] is True for name, _ in eg2.PATCHES)


def test_applying_the_patches_twice_changes_nothing_more() -> None:
    """A second request rebinds the same functions and returns the same record."""
    backend = eg2.EmbeddingGemma2Backend()
    model = _tiny_model()
    first = backend.apply_patches(_loaded(model))
    bound = (
        modeling_embedding_gemma2.rotate_half,
        modeling_embedding_gemma2.repeat_kv,
        modeling_embedding_gemma2.eager_attention_forward,
    )

    second = backend.apply_patches(_loaded(model))

    assert second == first
    assert (
        modeling_embedding_gemma2.rotate_half,
        modeling_embedding_gemma2.repeat_kv,
        modeling_embedding_gemma2.eager_attention_forward,
    ) == bound


@pytest.mark.parametrize(
    ("patch_name", "symbol"),
    [
        ("rotate_half_static", "rotate_half"),
        ("repeat_kv_rank4", "repeat_kv"),
        ("attention_query_chunks", "eager_attention_forward"),
    ],
)
def test_apply_patches_refuses_a_replacement_that_did_not_take_effect(
    monkeypatch: pytest.MonkeyPatch, patch_name: str, symbol: str
) -> None:
    """A rewrite that leaves the upstream function in place must stop the conversion.

    The patch is made to do nothing, as if something else had bound the
    upstream function again; the record would otherwise claim a rewrite
    the traced graph does not contain.
    """
    patches = tuple(
        (name, (lambda: None) if name == patch_name else patch) for name, patch in eg2.PATCHES
    )
    monkeypatch.setattr(eg2, "PATCHES", patches)
    model = _tiny_model()
    before = copy.deepcopy(model.state_dict())

    with pytest.raises(RuntimeError) as excinfo:
        eg2.EmbeddingGemma2Backend().apply_patches(_loaded(model))

    message = str(excinfo.value)
    assert symbol in message
    assert "kernels" in message
    # Refused before the weights were rescaled.
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def _attention_stub(head_dims: tuple[int, ...]) -> SimpleNamespace:
    """Build a stand-in model whose layers have the given attention head widths."""
    layers = [SimpleNamespace(self_attn=SimpleNamespace(head_dim=dim)) for dim in head_dims]
    return SimpleNamespace(layers=layers, config=SimpleNamespace())


@pytest.mark.parametrize("head_dims", [(8, 8, 15), (7, 8, 16)], ids=["full-layer", "sliding-layer"])
def test_apply_patches_rejects_an_odd_head_dim_in_any_layer(head_dims: tuple[int, ...]) -> None:
    """Each layer type has a head width of its own, and every one must be even."""
    with pytest.raises(ValueError, match="head dim"):
        eg2.EmbeddingGemma2Backend().apply_patches(_loaded(_attention_stub(head_dims)))

    assert modeling_embedding_gemma2.rotate_half is _UPSTREAM_ROTATE_HALF
    assert modeling_embedding_gemma2.repeat_kv is _UPSTREAM_REPEAT_KV
    assert modeling_embedding_gemma2.eager_attention_forward is _UPSTREAM_EAGER_ATTENTION


def test_apply_patches_refuses_a_fill_value_the_wrapper_cannot_apply() -> None:
    """The graph's mask fill is fixed by the wrapper; a different one must not be claimed."""
    model = _tiny_model()
    before = copy.deepcopy(model.state_dict())

    with pytest.raises(ValueError, match="mask fill"):
        eg2.EmbeddingGemma2Backend().apply_patches(_loaded(model), mask_fill_value=-30000.0)

    assert modeling_embedding_gemma2.rotate_half is _UPSTREAM_ROTATE_HALF
    assert modeling_embedding_gemma2.eager_attention_forward is _UPSTREAM_EAGER_ATTENTION
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_apply_patches_accepts_the_fill_value_the_wrapper_uses() -> None:
    """Asking for exactly the value the graph already uses is a no-op, not an error."""
    applied = eg2.EmbeddingGemma2Backend().apply_patches(
        _loaded(_tiny_model()), mask_fill_value=decoder_patches.MASK_FILL_VALUE
    )

    assert applied["mask_fill_value"] == decoder_patches.MASK_FILL_VALUE


def test_the_patches_do_not_change_what_the_model_computes() -> None:
    """The three rewrites claim to be equivalent to upstream; the hidden states must show it."""
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch(
        seq_len=_SEQ_LEN_OUTSIDE_WINDOW, lengths=(_SEQ_LEN_OUTSIDE_WINDOW, 6, 2)
    )
    masks = _mask_mapping(attention_mask)

    with torch.no_grad():
        upstream = model(input_ids=input_ids, attention_mask=masks)[0]
        for _, patch in eg2.PATCHES:
            patch()
        patched = model(input_ids=input_ids, attention_mask=masks)[0]

    assert torch.allclose(patched, upstream, rtol=0, atol=1e-6)


# --- the FP16 range conditioning: measuring ----------------------------------


def _capture(
    model: EmbeddingGemma2TextModel, input_ids: torch.Tensor, attention_mask: Any
) -> dict[str, Any]:
    """Run ``model`` and capture the activations the conditioning is about.

    Returns:
        The model output under ``"output"``; the input of the final
        normalization under ``"final"``; every tensor read from the
        residual stream under ``"residual"``; and, one tensor per layer,
        the outputs of the query, key and value projections, of the
        projection closing the per-layer-input block, of the MLP and of
        the attention.
    """
    captured: dict[str, Any] = {
        name: [] for name in ("residual", "q", "k", "v", "ple", "mlp", "attention")
    }

    def keep_input(name: str) -> Any:
        return lambda _module, args: captured[name].append(args[0].detach().clone())

    def keep_output(name: str) -> Any:
        return lambda _module, _args, output: captured[name].append(output.detach().clone())

    handles = []
    for layer in model.layers:
        for module in (
            layer.input_layernorm,
            layer.pre_feedforward_layernorm,
            layer.ple_block.per_layer_input_gate,
        ):
            handles.append(module.register_forward_pre_hook(keep_input("residual")))
        handles.append(layer.self_attn.q_proj.register_forward_hook(keep_output("q")))
        handles.append(layer.self_attn.k_proj.register_forward_hook(keep_output("k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(keep_output("v")))
        handles.append(
            layer.ple_block.per_layer_projection.register_forward_hook(keep_output("ple"))
        )
        handles.append(
            layer.post_feedforward_layernorm.register_forward_pre_hook(keep_input("mlp"))
        )
        handles.append(
            layer.post_attention_layernorm.register_forward_pre_hook(keep_input("attention"))
        )
    final: list[torch.Tensor] = []
    handles.append(
        model.norm.register_forward_pre_hook(lambda _m, args: final.append(args[0].detach()))
    )
    try:
        with torch.no_grad():
            captured["output"] = model(input_ids=input_ids, attention_mask=attention_mask)[0]
    finally:
        for handle in handles:
            handle.remove()
    captured["final"] = final[0]
    captured["residual"].append(final[0])
    return captured


def _rms(tensors: list[torch.Tensor]) -> float:
    """Return the root mean square over every element of ``tensors``."""
    total = sum(float(tensor.double().pow(2).sum()) for tensor in tensors)
    count = sum(tensor.numel() for tensor in tensors)
    return math.sqrt(total / count)


def _peak(tensors: list[torch.Tensor]) -> float:
    """Return the largest absolute value over every element of ``tensors``."""
    return max(float(tensor.abs().max()) for tensor in tensors)


def test_measuring_reports_one_value_per_layer_for_every_field() -> None:
    """Every per-layer field holds one entry per layer, in layer order."""
    model = _shaped_model()

    ranges = eg2.measure_activation_ranges(model, _shared_tokenizer(), ["a sentence", "another"])

    assert isinstance(ranges, eg2.EmbeddingGemma2Ranges)
    assert isinstance(ranges, fp16_ranges.ActivationRanges)
    assert ranges.inputs == 2
    for name in (
        "mlp_output_rms",
        "q_peaks",
        "k_peaks",
        "v_peaks",
        "ple_output_peaks",
        "attention_output_rms",
    ):
        values = getattr(ranges, name)
        assert isinstance(values, tuple), name
        assert len(values) == _TINY_LAYERS, name
        assert all(math.isfinite(value) and value > 0.0 for value in values), name


def test_measuring_reads_the_tensors_the_model_computes() -> None:
    """Each field is the peak or the level of the tensor it names, over every input."""
    model = _shaped_model()
    tokenizer = _shared_tokenizer()
    texts = ["a first sentence", "another, somewhat longer sentence"]
    before = copy.deepcopy(model.state_dict())

    ranges = eg2.measure_activation_ranges(model, tokenizer, texts)

    runs = []
    for text in texts:
        encoded = tokenizer(text, return_tensors="pt")
        runs.append(_capture(model, encoded["input_ids"], encoded["attention_mask"]))

    def per_layer(name: str, index: int) -> list[torch.Tensor]:
        return [run[name][index] for run in runs]

    assert ranges.inputs == len(texts)
    assert ranges.residual_peak == pytest.approx(
        max(_peak(run["residual"]) for run in runs), rel=1e-6
    )
    for index in range(_TINY_LAYERS):
        assert ranges.q_peaks[index] == pytest.approx(_peak(per_layer("q", index)), rel=1e-6)
        assert ranges.k_peaks[index] == pytest.approx(_peak(per_layer("k", index)), rel=1e-6)
        assert ranges.v_peaks[index] == pytest.approx(_peak(per_layer("v", index)), rel=1e-6)
        assert ranges.ple_output_peaks[index] == pytest.approx(
            _peak(per_layer("ple", index)), rel=1e-6
        )
        assert ranges.mlp_output_rms[index] == pytest.approx(
            _rms(per_layer("mlp", index)), rel=1e-9
        )
        assert ranges.attention_output_rms[index] == pytest.approx(
            _rms(per_layer("attention", index)), rel=1e-9
        )
    # Measuring leaves no hook and no changed weight behind.
    assert all(
        not module._forward_pre_hooks and not module._forward_hooks for module in model.modules()
    )
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_measuring_sees_the_stream_the_per_layer_block_reads() -> None:
    """The stream between the MLP and the per-layer-input block counts as well."""
    model = _tiny_model()
    tokenizer = _shared_tokenizer()
    baseline = eg2.measure_activation_ranges(model, tokenizer, ["a sentence"])
    # Raising what the last layer's MLP adds raises only the stream its
    # per-layer-input block reads: the final normalization reads the
    # stream after the layer scalar, which is set to zero here.
    with torch.no_grad():
        model.layers[-1].post_feedforward_layernorm.weight.fill_(1e4)
        model.layers[-1].layer_scalar.fill_(0.0)

    raised = eg2.measure_activation_ranges(model, tokenizer, ["a sentence"])

    assert raised.residual_peak > 100 * baseline.residual_peak


def test_measuring_truncates_to_the_given_length() -> None:
    """A length limit bounds what is run; without it the whole text is."""
    model = _tiny_model()
    seen: list[int] = []
    handle = model.norm.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].shape[1])
    )
    try:
        eg2.measure_activation_ranges(model, _shared_tokenizer(), ["x" * 30], max_length=8)
        eg2.measure_activation_ranges(model, _shared_tokenizer(), ["x" * 30])
    finally:
        handle.remove()

    assert seen == [8, 32]


def test_measuring_reports_a_non_finite_activation_as_such() -> None:
    """A NaN must not be stepped over by a maximum; it has to reach the factors as one."""
    model = _tiny_model()
    with torch.no_grad():
        model.layers[1].self_attn.q_proj.weight.fill_(math.nan)

    ranges = eg2.measure_activation_ranges(model, _shared_tokenizer(), ["a sentence"])
    factors = eg2.derive_range_factors(ranges)

    assert math.isnan(ranges.q_peaks[1])
    assert math.isfinite(ranges.q_peaks[0])
    assert math.isnan(ranges.residual_peak)
    assert factors.q_divisor == 1.0
    assert factors.residual_divisor == 1.0


def test_measuring_nothing_reports_nothing() -> None:
    """Without any input there is no range, and so no rescaling to derive."""
    model = _tiny_model()

    ranges = eg2.measure_activation_ranges(model, _shared_tokenizer(), [])

    zeros = (0.0,) * _TINY_LAYERS
    assert ranges == eg2.EmbeddingGemma2Ranges(
        residual_peak=0.0,
        mlp_output_rms=zeros,
        inputs=0,
        q_peaks=zeros,
        k_peaks=zeros,
        v_peaks=zeros,
        ple_output_peaks=zeros,
        attention_output_rms=zeros,
    )
    assert eg2.derive_range_factors(ranges) == _unit_factors()


# --- the FP16 range conditioning: deriving the factors -----------------------


def _unit_factors() -> eg2.RangeFactors:
    """Return factors that rescale nothing, for the tiny model."""
    return eg2.RangeFactors(
        residual_divisor=1.0,
        q_divisor=1.0,
        k_divisor=1.0,
        v_divisor=1.0,
        ple_output_gains=(1.0,) * _TINY_LAYERS,
        mlp_output_gains=(1.0,) * _TINY_LAYERS,
        attention_output_gains=(1.0,) * _TINY_LAYERS,
    )


# Ranges of the size a real model of this family reaches (the residual,
# query, key, value and per-layer-input peaks of its first layers), and
# the factors they must lead to.
_REALISTIC_RANGES = eg2.EmbeddingGemma2Ranges(
    residual_peak=469.0,
    mlp_output_rms=(0.12, 4.0, 1.0),
    inputs=10,
    q_peaks=(167.0, 12.0, 3.0),
    k_peaks=(20.0, 153.0, 1.0),
    v_peaks=(712.0, 5.0, 2.0),
    ple_output_peaks=(1979.0, 10.0, 70.0),
    attention_output_rms=(0.05, 0.5, 0.0),
)
_REALISTIC_FACTORS = eg2.RangeFactors(
    residual_divisor=8.0,
    q_divisor=4.0,
    k_divisor=4.0,
    v_divisor=16.0,
    ple_output_gains=(1.0 / 32.0, 1.0, 0.5),
    mlp_output_gains=(8.0, 0.25, 1.0),
    attention_output_gains=(16.0, 1.0, 1.0),
)


def test_the_factors_follow_the_documented_rules() -> None:
    """Divisors bound the peaks under the target, gains bring the levels near one."""
    assert eg2.derive_range_factors(_REALISTIC_RANGES) == _REALISTIC_FACTORS


def test_the_factors_bring_every_measured_peak_under_the_target() -> None:
    """Guard for the rule above: what the divisors leave fits the squares into float16."""
    factors = eg2.derive_range_factors(_REALISTIC_RANGES)
    target = fp16_ranges.NORM_INPUT_PEAK_TARGET

    assert _REALISTIC_RANGES.residual_peak / factors.residual_divisor <= target
    assert max(_REALISTIC_RANGES.q_peaks) / factors.q_divisor <= target
    assert max(_REALISTIC_RANGES.k_peaks) / factors.k_divisor <= target
    assert max(_REALISTIC_RANGES.v_peaks) / factors.v_divisor <= target
    for peak, gain in zip(
        _REALISTIC_RANGES.ple_output_peaks, factors.ple_output_gains, strict=True
    ):
        assert peak * gain <= target


@pytest.mark.parametrize(
    ("rms", "expected"),
    [
        # Only a level below the floor is lifted.
        (0.1, 1.0),
        (0.5, 1.0),
        (3.0, 1.0),
        (0.099, 8.0),
        (0.002, 512.0),
        # Nothing measured, or nothing usable: left alone.
        (0.0, 1.0),
        (math.nan, 1.0),
    ],
)
def test_an_attention_output_is_lifted_only_below_the_floor(rms: float, expected: float) -> None:
    """An attention output that is not near the float16 underflow keeps its level."""
    assert eg2.ATTENTION_OUTPUT_RMS_FLOOR == 0.1
    ranges = dataclasses.replace(_REALISTIC_RANGES, attention_output_rms=(rms, 1.0, 1.0))

    assert eg2.derive_range_factors(ranges).attention_output_gains == (expected, 1.0, 1.0)


@pytest.mark.parametrize("measured", [0.0, -3.0, math.nan, math.inf, -math.inf])
def test_an_unusable_measurement_yields_no_rescaling(measured: float) -> None:
    """Nothing can be derived from a zero, negative or non-finite measurement."""
    values = (measured,) * _TINY_LAYERS
    ranges = eg2.EmbeddingGemma2Ranges(
        residual_peak=measured,
        mlp_output_rms=values,
        inputs=3,
        q_peaks=values,
        k_peaks=values,
        v_peaks=values,
        ple_output_peaks=values,
        attention_output_rms=values,
    )

    assert eg2.derive_range_factors(ranges) == _unit_factors()


def test_one_unusable_layer_does_not_hide_behind_a_usable_one() -> None:
    """A NaN in any layer leaves the shared divisor of that projection at 1."""
    ranges = dataclasses.replace(_REALISTIC_RANGES, q_peaks=(167.0, math.nan, 3.0))

    assert eg2.derive_range_factors(ranges).q_divisor == 1.0


def _is_power_of_two(value: float) -> bool:
    """Tell whether ``value`` is a positive power of two."""
    return value > 0 and math.isfinite(value) and math.frexp(value)[0] == 0.5


@pytest.mark.parametrize("exponent", range(-30, 31, 5))
def test_every_factor_is_a_power_of_two(exponent: int) -> None:
    """A power of two changes a float's exponent only, whatever was measured."""
    measured = 1.7 * 10.0**exponent
    values = (measured, measured / 3.0, measured * 7.0)
    ranges = eg2.EmbeddingGemma2Ranges(
        residual_peak=measured,
        mlp_output_rms=values,
        inputs=1,
        q_peaks=values,
        k_peaks=values,
        v_peaks=values,
        ple_output_peaks=values,
        attention_output_rms=values,
    )

    factors = eg2.derive_range_factors(ranges)

    assert all(
        _is_power_of_two(value)
        for value in (
            factors.residual_divisor,
            factors.q_divisor,
            factors.k_divisor,
            factors.v_divisor,
            *factors.ple_output_gains,
            *factors.mlp_output_gains,
            *factors.attention_output_gains,
        )
    )


# --- the FP16 range conditioning: rewriting the weights ----------------------

# Factors applied by hand: each kind of factor alone, and all of them at
# once. Every per-layer list holds a factor above 1, one below and one
# that leaves its layer alone.
_FACTOR_CASES: dict[str, eg2.RangeFactors] = {
    "residual": dataclasses.replace(_unit_factors(), residual_divisor=8.0),
    "query": dataclasses.replace(_unit_factors(), q_divisor=4.0),
    "key": dataclasses.replace(_unit_factors(), k_divisor=8.0),
    "value": dataclasses.replace(_unit_factors(), v_divisor=16.0),
    "per-layer-input": dataclasses.replace(
        _unit_factors(), ple_output_gains=(1.0 / 32.0, 1.0, 4.0)
    ),
    "mlp": dataclasses.replace(_unit_factors(), mlp_output_gains=(8.0, 0.5, 1.0)),
    "attention": dataclasses.replace(_unit_factors(), attention_output_gains=(16.0, 1.0, 0.25)),
    "all": eg2.RangeFactors(
        residual_divisor=8.0,
        q_divisor=4.0,
        k_divisor=8.0,
        v_divisor=16.0,
        ple_output_gains=(1.0 / 32.0, 1.0, 4.0),
        mlp_output_gains=(8.0, 0.5, 1.0),
        attention_output_gains=(16.0, 1.0, 0.25),
    ),
}

# Real lengths per row, for a batch of each sequence length with and
# without padding.
_BATCH_LENGTHS: dict[tuple[int, bool], tuple[int, ...]] = {
    (_SEQ_LEN_INSIDE_WINDOW, False): (3, 3),
    (_SEQ_LEN_INSIDE_WINDOW, True): (3, 2, 1),
    (_SEQ_LEN_OUTSIDE_WINDOW, False): (10, 10),
    (_SEQ_LEN_OUTSIDE_WINDOW, True): (10, 6, 2),
}


def _assert_same_function(actual: torch.Tensor, expected: torch.Tensor, real: torch.Tensor) -> None:
    """Assert two outputs agree on every real position, to rounding."""
    scale = float(expected[real].abs().max())
    assert float(_row_cosines(actual[real], expected[real]).min()) >= (
        _EQUIVALENCE_COSINE_THRESHOLD
    )
    assert float((actual[real] - expected[real]).abs().max()) <= (
        _CONDITIONED_RELATIVE_TOLERANCE * scale
    )


@pytest.mark.parametrize("case", list(_FACTOR_CASES))
@pytest.mark.parametrize(
    ("seq_len", "padded"),
    [(_SEQ_LEN_INSIDE_WINDOW, False), (_SEQ_LEN_OUTSIDE_WINDOW, True)],
    ids=["inside-window", "outside-window-padded"],
)
@pytest.mark.parametrize("mask_path", ["mapping", "framework"])
def test_a_conditioned_model_computes_the_same_function(
    case: str, seq_len: int, padded: bool, mask_path: str
) -> None:
    """Rescaled weights must leave the output and the pooled vector where they were."""
    original = _shaped_model()
    conditioned = copy.deepcopy(original)
    input_ids, attention_mask = _right_padded_batch(
        seq_len=seq_len, lengths=_BATCH_LENGTHS[(seq_len, padded)]
    )
    masks = _mask_mapping(attention_mask) if mask_path == "mapping" else attention_mask

    eg2.condition_fp16_ranges(conditioned, _FACTOR_CASES[case])
    with torch.no_grad():
        expected = original(input_ids=input_ids, attention_mask=masks)[0]
        actual = conditioned(input_ids=input_ids, attention_mask=masks)[0]

    real = attention_mask.to(torch.bool)
    _assert_same_function(actual, expected, real)
    pooled_actual = common.mean_pool(actual, attention_mask)
    pooled_expected = common.mean_pool(expected, attention_mask)
    assert float(_row_cosines(pooled_actual, pooled_expected).min()) >= (
        _EQUIVALENCE_COSINE_THRESHOLD
    )


def test_a_conditioned_model_with_attention_biases_computes_the_same_function() -> None:
    """A projection bias is rescaled with its weight, so the identity holds with biases too."""
    original = _shaped_model(attention_bias=True)
    with torch.no_grad():
        for layer in original.layers:
            for projection in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
            ):
                projection.bias.normal_(0.0, float(projection.weight.abs().max()))
    conditioned = copy.deepcopy(original)
    input_ids, attention_mask = _right_padded_batch(seq_len=_SEQ_LEN_OUTSIDE_WINDOW)

    eg2.condition_fp16_ranges(conditioned, _FACTOR_CASES["all"])
    with torch.no_grad():
        expected = original(input_ids=input_ids, attention_mask=attention_mask)[0]
        actual = conditioned(input_ids=input_ids, attention_mask=attention_mask)[0]

    _assert_same_function(actual, expected, attention_mask.to(torch.bool))


def test_conditioning_moves_the_ranges_it_is_meant_to_move() -> None:
    """Guard for the comparisons above: every rescaled activation really moved by its factor."""
    original = _shaped_model()
    conditioned = copy.deepcopy(original)
    factors = _FACTOR_CASES["all"]
    input_ids, attention_mask = _right_padded_batch(seq_len=_SEQ_LEN_OUTSIDE_WINDOW)

    eg2.condition_fp16_ranges(conditioned, factors)
    before = _capture(original, input_ids, attention_mask)
    after = _capture(conditioned, input_ids, attention_mask)

    def moved(name: str, index: int, factor: float) -> bool:
        return torch.allclose(after[name][index], before[name][index] * factor, rtol=1e-4)

    assert torch.allclose(
        after["final"] * factors.residual_divisor, before["final"], rtol=1e-4, atol=1e-3
    )
    for index in range(_TINY_LAYERS):
        assert moved("q", index, 1.0 / factors.q_divisor)
        assert moved("k", index, 1.0 / factors.k_divisor)
        assert moved("v", index, 1.0 / factors.v_divisor)
        assert moved("ple", index, factors.ple_output_gains[index])
        assert moved("mlp", index, factors.mlp_output_gains[index])
        assert moved("attention", index, factors.attention_output_gains[index])


def test_conditioning_rewrites_exactly_the_documented_constants() -> None:
    """Weights are scaled by exact powers of two, normalization constants follow, nothing else."""
    original = _shaped_model()
    conditioned = copy.deepcopy(original)
    factors = _FACTOR_CASES["all"]
    divisor = factors.residual_divisor
    eps = original.config.rms_norm_eps

    eg2.condition_fp16_ranges(conditioned, factors)

    assert torch.equal(
        conditioned.embed_tokens.embed_scale, original.embed_tokens.embed_scale / divisor
    )
    assert conditioned.norm.eps == eps / divisor**2
    assert conditioned.ple.per_layer_projection_norm.eps == eps / divisor**2
    changed: set[str] = set()
    for index, (before, after) in enumerate(zip(original.layers, conditioned.layers, strict=True)):
        ple_gain = factors.ple_output_gains[index]
        mlp_gain = factors.mlp_output_gains[index]
        attention_gain = factors.attention_output_gains[index]
        prefix = f"layers.{index}."
        # Everything written to the stream is divided by the divisor.
        for name in (
            "post_attention_layernorm",
            "post_feedforward_layernorm",
            "ple_block.post_per_layer_input_norm",
        ):
            old = before.get_submodule(name).weight
            assert torch.equal(after.get_submodule(name).weight, old / divisor), name
            changed.add(f"{prefix}{name}.weight")
        # The gate reads the stream without normalizing it.
        assert torch.equal(
            after.ple_block.per_layer_input_gate.weight,
            before.ple_block.per_layer_input_gate.weight * divisor,
        )
        changed.add(f"{prefix}ple_block.per_layer_input_gate.weight")
        # Everything normalizing the stream absorbs the divisor in its eps.
        assert after.input_layernorm.eps == eps / divisor**2
        assert after.pre_feedforward_layernorm.eps == eps / divisor**2
        # The three attention projections and their normalizations.
        for projection, norm, factor in (
            ("q_proj", "q_norm", factors.q_divisor),
            ("k_proj", "k_norm", factors.k_divisor),
            ("v_proj", "v_norm", factors.v_divisor),
        ):
            old = getattr(before.self_attn, projection).weight
            assert torch.equal(getattr(after.self_attn, projection).weight, old / factor)
            assert getattr(after.self_attn, norm).eps == eps / factor**2
            changed.add(f"{prefix}self_attn.{projection}.weight")
        # The three branch outputs and the normalizations reading them.
        for projection, norm, gain in (
            ("ple_block.per_layer_projection", "ple_block.post_per_layer_input_norm", ple_gain),
            ("mlp.down_proj", "post_feedforward_layernorm", mlp_gain),
            ("self_attn.o_proj", "post_attention_layernorm", attention_gain),
        ):
            old = before.get_submodule(projection).weight
            assert torch.equal(after.get_submodule(projection).weight, old * gain), projection
            assert after.get_submodule(norm).eps == eps * gain**2, norm
            if gain != 1.0:
                changed.add(f"{prefix}{projection}.weight")
        # A linear factor needs no rescaling.
        assert torch.equal(after.layer_scalar, before.layer_scalar)
    before_state = original.state_dict()
    after_state = conditioned.state_dict()
    # The embedding scale is a non-persistent buffer, compared above.
    assert {
        name for name, value in after_state.items() if not torch.equal(value, before_state[name])
    } == changed


def test_conditioning_with_unit_factors_changes_nothing() -> None:
    """Factors of 1 must leave every weight and constant bit-identical, not merely close."""
    original = _shaped_model()
    conditioned = copy.deepcopy(original)

    eg2.condition_fp16_ranges(conditioned, _unit_factors())

    before = original.state_dict()
    assert all(torch.equal(value, before[name]) for name, value in conditioned.state_dict().items())
    assert torch.equal(conditioned.embed_tokens.embed_scale, original.embed_tokens.embed_scale)
    eps_before = [m.eps for m in original.modules() if hasattr(m, "eps")]
    eps_after = [m.eps for m in conditioned.modules() if hasattr(m, "eps")]
    assert eps_after == eps_before


_ALL = _FACTOR_CASES["all"]


@pytest.mark.parametrize(
    "factors",
    [
        dataclasses.replace(_ALL, ple_output_gains=(1.0, 1.0)),
        dataclasses.replace(_ALL, mlp_output_gains=(1.0,) * (_TINY_LAYERS + 1)),
        dataclasses.replace(_ALL, attention_output_gains=()),
        dataclasses.replace(_ALL, residual_divisor=3.0),
        dataclasses.replace(_ALL, residual_divisor=0.0),
        dataclasses.replace(_ALL, residual_divisor=math.nan),
        dataclasses.replace(_ALL, residual_divisor=math.inf),
        dataclasses.replace(_ALL, q_divisor=-2.0),
        dataclasses.replace(_ALL, k_divisor=6.0),
        dataclasses.replace(_ALL, v_divisor=math.nan),
        dataclasses.replace(_ALL, ple_output_gains=(1.0, 0.0, 1.0)),
        dataclasses.replace(_ALL, mlp_output_gains=(1.0, 6.0, 1.0)),
        dataclasses.replace(_ALL, attention_output_gains=(math.nan, 1.0, 1.0)),
    ],
    ids=[
        "too-few-ple-gains",
        "too-many-mlp-gains",
        "no-attention-gains",
        "divisor-not-a-power",
        "zero-divisor",
        "nan-divisor",
        "infinite-divisor",
        "negative-query-divisor",
        "key-divisor-not-a-power",
        "nan-value-divisor",
        "zero-ple-gain",
        "mlp-gain-not-a-power",
        "nan-attention-gain",
    ],
)
def test_conditioning_refuses_unusable_factors_without_touching_the_model(
    factors: eg2.RangeFactors,
) -> None:
    """A factor that is not a power of two, or a missing one, must rewrite nothing at all.

    Every other factor of each case is a usable one that would change the
    model, so a rewrite started before the refusal would show up.
    """
    model = _shaped_model()
    before = copy.deepcopy(model.state_dict())
    scale = model.embed_tokens.embed_scale.clone()
    eps = [module.eps for module in model.modules() if hasattr(module, "eps")]

    with pytest.raises(ValueError, match="power of two|per layer"):
        eg2.condition_fp16_ranges(model, factors)

    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert torch.equal(model.embed_tokens.embed_scale, scale)
    assert [module.eps for module in model.modules() if hasattr(module, "eps")] == eps


# --- the FP16 range conditioning: applied through the backend ----------------


def test_apply_patches_derives_rescaling_factors_from_a_measured_model() -> None:
    """A model with real-sized activations is rescaled by every kind of factor, and unchanged."""
    original = _shaped_model()
    model = _shaped_model()
    input_ids, attention_mask = _right_padded_batch(
        seq_len=_SEQ_LEN_OUTSIDE_WINDOW, lengths=(_SEQ_LEN_OUTSIDE_WINDOW, 6)
    )
    expected = _upstream_embedding(original, input_ids, attention_mask)

    backend = eg2.EmbeddingGemma2Backend()
    record = backend.apply_patches(_loaded(model))[fp16_ranges.FP16_RANGE_CONDITIONING_KEY]
    with torch.no_grad():
        actual = backend.wrap(_loaded(model))(input_ids, attention_mask)

    assert record["residual_divisor"] > 1.0
    assert all(value > 1.0 for value in record["qkv_divisors"].values())
    assert any(gain < 1.0 for gain in record["ple_output_gains"])
    assert any(gain == 1.0 for gain in record["ple_output_gains"])
    assert any(gain < 1.0 for gain in record["mlp_output_gains"])
    assert any(gain > 1.0 for gain in record["mlp_output_gains"])
    assert any(gain > 1.0 for gain in record["attention_output_gains"])
    assert any(gain == 1.0 for gain in record["attention_output_gains"])
    assert torch.equal(
        model.embed_tokens.embed_scale,
        original.embed_tokens.embed_scale / record["residual_divisor"],
    )
    assert float(_row_cosines(actual, expected).min()) >= _EQUIVALENCE_COSINE_THRESHOLD


def test_apply_patches_records_the_factors_it_applied() -> None:
    """The record must name the factors the weights were really rescaled with."""
    original = _shaped_model()
    model = copy.deepcopy(original)
    expected_inputs = 1 + len(eg2.EmbeddingGemma2Backend().sanity_spec("embedding").all_inputs)

    applied = eg2.EmbeddingGemma2Backend().apply_patches(_loaded(model))

    record = applied[fp16_ranges.FP16_RANGE_CONDITIONING_KEY]
    assert record["calibration_inputs"] == expected_inputs
    divisors = record["qkv_divisors"]
    for index, (before, after) in enumerate(zip(original.layers, model.layers, strict=True)):
        for projection, key in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")):
            assert torch.equal(
                getattr(after.self_attn, projection).weight,
                getattr(before.self_attn, projection).weight / divisors[key],
            )
        assert torch.equal(
            after.ple_block.per_layer_projection.weight,
            before.ple_block.per_layer_projection.weight * record["ple_output_gains"][index],
        )
        assert torch.equal(
            after.mlp.down_proj.weight,
            before.mlp.down_proj.weight * record["mlp_output_gains"][index],
        )
        assert torch.equal(
            after.self_attn.o_proj.weight,
            before.self_attn.o_proj.weight * record["attention_output_gains"][index],
        )


def test_apply_patches_derives_the_factors_from_the_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What is measured decides the factors: a real model's ranges give a real model's factors."""
    monkeypatch.setattr(eg2, "measure_activation_ranges", lambda *args, **kwargs: _REALISTIC_RANGES)
    original = _tiny_model()
    model = copy.deepcopy(original)

    applied = eg2.EmbeddingGemma2Backend().apply_patches(_loaded(model))

    assert applied[fp16_ranges.FP16_RANGE_CONDITIONING_KEY] == {
        "residual_divisor": 8.0,
        "qkv_divisors": {"q": 4.0, "k": 4.0, "v": 16.0},
        "ple_output_gains": [1.0 / 32.0, 1.0, 0.5],
        "mlp_output_gains": [8.0, 0.25, 1.0],
        "attention_output_gains": [16.0, 1.0, 1.0],
        "calibration_inputs": 10,
    }
    assert torch.equal(model.embed_tokens.embed_scale, original.embed_tokens.embed_scale / 8.0)
    assert torch.equal(
        model.layers[0].self_attn.v_proj.weight, original.layers[0].self_attn.v_proj.weight / 16.0
    )


@pytest.mark.parametrize("unusable", [0.0, math.nan, math.inf, -1.0])
def test_apply_patches_leaves_the_weights_alone_on_an_unusable_measurement(
    monkeypatch: pytest.MonkeyPatch, unusable: float
) -> None:
    """A measurement nothing can be derived from must fall back to no rescaling at all."""
    model = _tiny_model()
    before = copy.deepcopy(model.state_dict())
    scale = model.embed_tokens.embed_scale.clone()
    values = (unusable,) * _TINY_LAYERS
    measured = eg2.EmbeddingGemma2Ranges(
        residual_peak=unusable,
        mlp_output_rms=values,
        inputs=10,
        q_peaks=values,
        k_peaks=values,
        v_peaks=values,
        ple_output_peaks=values,
        attention_output_rms=values,
    )
    monkeypatch.setattr(eg2, "measure_activation_ranges", lambda *args, **kwargs: measured)

    applied = eg2.EmbeddingGemma2Backend().apply_patches(_loaded(model))

    ones = [1.0] * _TINY_LAYERS
    assert applied[fp16_ranges.FP16_RANGE_CONDITIONING_KEY] == {
        "residual_divisor": 1.0,
        "qkv_divisors": {"q": 1.0, "k": 1.0, "v": 1.0},
        "ple_output_gains": ones,
        "mlp_output_gains": ones,
        "attention_output_gains": ones,
        "calibration_inputs": 10,
    }
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert torch.equal(model.embed_tokens.embed_scale, scale)


def test_apply_patches_measures_within_the_position_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The calibration sentences are cut to the length the model is configured for."""
    requested: list[int | None] = []
    measure = eg2.measure_activation_ranges

    def spy(model: Any, tokenizer: Any, texts: Any, max_length: int | None = None) -> Any:
        requested.append(max_length)
        return measure(model, tokenizer, texts, max_length=max_length)

    monkeypatch.setattr(eg2, "measure_activation_ranges", spy)

    eg2.EmbeddingGemma2Backend().apply_patches(_loaded(_tiny_model()))

    assert requested == [_TINY_POSITIONS]


def test_applying_the_patches_twice_rescales_once() -> None:
    """A second request on the same model must return the first record and change nothing."""
    backend = eg2.EmbeddingGemma2Backend()
    model = _shaped_model()
    first = backend.apply_patches(_loaded(model))
    after_first = copy.deepcopy(model.state_dict())
    scale = model.embed_tokens.embed_scale.clone()
    eps = [module.eps for module in model.modules() if hasattr(module, "eps")]

    # Through a second handle on the same model, as another caller would.
    second = backend.apply_patches(_loaded(model))

    assert second == first
    assert first[fp16_ranges.FP16_RANGE_CONDITIONING_KEY]["residual_divisor"] > 1.0
    assert all(torch.equal(value, after_first[name]) for name, value in model.state_dict().items())
    assert torch.equal(model.embed_tokens.embed_scale, scale)
    assert [module.eps for module in model.modules() if hasattr(module, "eps")] == eps


def test_a_returned_record_cannot_alter_the_remembered_one() -> None:
    """The record goes into metadata the caller owns; editing it must not leak back."""
    backend = eg2.EmbeddingGemma2Backend()
    model = _tiny_model()
    first = backend.apply_patches(_loaded(model))
    expected = copy.deepcopy(first)
    record = first[fp16_ranges.FP16_RANGE_CONDITIONING_KEY]
    record["mlp_output_gains"].append(99.0)
    record["qkv_divisors"]["q"] = 7.0
    record["residual_divisor"] = 7.0

    assert backend.apply_patches(_loaded(model)) == expected


def test_conditioning_one_model_leaves_a_separate_copy_alone() -> None:
    """The rescaling belongs to one model object; the baseline's own copy must not see it."""
    compiled = _shaped_model()
    baseline = _shaped_model()
    untouched = copy.deepcopy(baseline.state_dict())
    eps = [module.eps for module in baseline.modules() if hasattr(module, "eps")]

    applied = eg2.EmbeddingGemma2Backend().apply_patches(_loaded(compiled))

    assert applied[fp16_ranges.FP16_RANGE_CONDITIONING_KEY]["residual_divisor"] > 1.0
    assert all(torch.equal(value, untouched[name]) for name, value in baseline.state_dict().items())
    assert torch.equal(baseline.embed_tokens.embed_scale, torch.tensor(float(_TINY_HIDDEN) ** 0.5))
    assert [module.eps for module in baseline.modules() if hasattr(module, "eps")] == eps
    # A module-level default shared between instances would show up here.
    assert not hasattr(baseline, "_eeane_fp16_range_conditioning")


# --- the in-graph masks and the wrapper --------------------------------------


def test_the_tiny_model_exercises_both_layer_geometries() -> None:
    """Guard for the comparisons below: both layer types, both widths and the window in play."""
    model = _tiny_model()

    assert tuple(model.config.layer_types) == _TINY_LAYER_TYPES
    assert [layer.self_attn.head_dim for layer in model.layers] == [8, 8, 16]
    assert all(layer.self_attn.num_key_value_groups == 2 for layer in model.layers)
    assert model.config.sliding_window == _TINY_RADIUS
    assert _SEQ_LEN_INSIDE_WINDOW <= _TINY_RADIUS + 1 < _SEQ_LEN_OUTSIDE_WINDOW


def test_wrap_masks_with_the_radius_plus_one() -> None:
    """The configuration states an inclusive radius; the mask builder takes an exclusive bound."""
    wrapper = eg2.EmbeddingGemma2Backend().wrap(_loaded(_tiny_model()))

    assert isinstance(wrapper, eg2.EmbeddingGemma2MeanWrapper)
    assert wrapper.training is False
    assert wrapper.window == _TINY_RADIUS + 1
    assert wrapper.pad_token_id == _PAD_ID
    assert wrapper.dense is None
    assert wrapper.fill_value == decoder_patches.MASK_FILL_VALUE
    assert tuple(wrapper.unstable_lengths) == tuple(eg2.NE_UNSTABLE_LENGTHS)
    assert wrapper.padded_length == eg2.NE_PADDED_LENGTH


# Real lengths per row for the wrapper comparisons: a batch of one and of
# two, each with and without padding.
_WRAPPER_LENGTHS: dict[tuple[int, bool, int], tuple[int, ...]] = {
    (_SEQ_LEN_INSIDE_WINDOW, False, 1): (3,),
    (_SEQ_LEN_INSIDE_WINDOW, False, 2): (3, 3),
    (_SEQ_LEN_INSIDE_WINDOW, True, 1): (2,),
    (_SEQ_LEN_INSIDE_WINDOW, True, 2): (3, 1),
    (_SEQ_LEN_OUTSIDE_WINDOW, False, 1): (10,),
    (_SEQ_LEN_OUTSIDE_WINDOW, False, 2): (10, 10),
    (_SEQ_LEN_OUTSIDE_WINDOW, True, 1): (6,),
    (_SEQ_LEN_OUTSIDE_WINDOW, True, 2): (10, 4),
}


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
@pytest.mark.parametrize(
    "seq_len",
    [_SEQ_LEN_INSIDE_WINDOW, _SEQ_LEN_OUTSIDE_WINDOW],
    ids=["inside-window", "outside-window"],
)
def test_the_wrapper_matches_the_upstream_model(
    seq_len: int, padded: bool, batch_size: int
) -> None:
    """Patched eager attention over the in-graph masks must compute what upstream computes.

    Upstream is a separate copy on the sdpa path that builds its own masks
    from the plain 2-D mask, pooled by the shared helper.
    """
    reference = _tiny_model(attn="sdpa")
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch(
        seq_len=seq_len, lengths=_WRAPPER_LENGTHS[(seq_len, padded, batch_size)]
    )
    expected = _upstream_embedding(reference, input_ids, attention_mask)

    backend = eg2.EmbeddingGemma2Backend()
    backend.apply_patches(_loaded(model))
    with torch.no_grad():
        actual = backend.wrap(_loaded(model))(input_ids, attention_mask)

    assert bool((attention_mask == 0).any()) is padded  # the batch is what its name says
    assert actual.shape == (batch_size, _TINY_EMBEDDING_DIM)
    assert float(_row_cosines(actual, expected).min()) >= _EQUIVALENCE_COSINE_THRESHOLD


@pytest.mark.parametrize("window", [_TINY_RADIUS, _TINY_RADIUS + 2])
def test_a_window_off_by_one_is_not_this_model(window: int) -> None:
    """Guard for the comparison above: only the radius plus one reproduces the model."""
    reference = _tiny_model(attn="sdpa")
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch(
        seq_len=_SEQ_LEN_OUTSIDE_WINDOW, lengths=(_SEQ_LEN_OUTSIDE_WINDOW,)
    )
    expected = _upstream_embedding(reference, input_ids, attention_mask)

    with torch.no_grad():
        actual = eg2.EmbeddingGemma2MeanWrapper(model, window=window, pad_token_id=_PAD_ID)(
            input_ids, attention_mask
        )

    assert float(_row_cosines(actual, expected).min()) < _EQUIVALENCE_COSINE_THRESHOLD


def _band_wrapper(model: torch.nn.Module, **overrides: Any) -> eg2.EmbeddingGemma2MeanWrapper:
    """Build a wrapper that extends the lengths of the small test band."""
    arguments: dict[str, Any] = {
        "window": _TINY_WINDOW,
        "pad_token_id": _PAD_ID,
        "unstable_lengths": _TEST_BAND,
        "padded_length": _TEST_PADDED_LENGTH,
    }
    arguments.update(overrides)
    return eg2.EmbeddingGemma2MeanWrapper(model, **arguments).eval()


def _lengths_the_backbone_sees(wrapper: torch.nn.Module, seq_len: int) -> list[int]:
    """Run ``wrapper`` on a batch of ``seq_len`` and report the length its backbone ran at."""
    seen: list[int] = []
    handle = wrapper.model.norm.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].shape[1])
    )
    input_ids, attention_mask = _right_padded_batch(seq_len=seq_len, lengths=(seq_len,))
    try:
        with torch.no_grad():
            pooled = wrapper(input_ids, attention_mask)
    finally:
        handle.remove()
    assert pooled.shape == (1, _TINY_EMBEDDING_DIM)
    return seen


@pytest.mark.parametrize(
    ("seq_len", "expected"),
    [(8, 8), (9, _TEST_PADDED_LENGTH), (12, _TEST_PADDED_LENGTH), (13, 13), (16, 16)],
)
def test_a_length_inside_the_band_is_computed_at_the_padded_length(
    seq_len: int, expected: int
) -> None:
    """Only the lengths of the band are extended, and all of them to the one padded length."""
    wrapper = _band_wrapper(_tiny_model())

    assert _lengths_the_backbone_sees(wrapper, seq_len) == [expected]


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [8, 9, 12, 13], ids=["below", "lowest", "highest", "above"])
def test_the_extended_wrapper_matches_the_upstream_model(seq_len: int, batch_size: int) -> None:
    """Computing a band length at the padded length and cutting it back changes nothing."""
    reference = _tiny_model(attn="sdpa")
    model = _tiny_model()
    lengths = (seq_len, seq_len // 2)[:batch_size]
    input_ids, attention_mask = _right_padded_batch(seq_len=seq_len, lengths=lengths)
    expected = _upstream_embedding(reference, input_ids, attention_mask)

    eg2.EmbeddingGemma2Backend().apply_patches(_loaded(model))
    with torch.no_grad():
        actual = _band_wrapper(model)(input_ids, attention_mask)

    assert actual.shape == (batch_size, _TINY_EMBEDDING_DIM)
    assert float(_row_cosines(actual, expected).min()) >= _EQUIVALENCE_COSINE_THRESHOLD


def test_the_extension_does_not_change_the_wrapper_output() -> None:
    """The same wrapper with and without a band must agree on a length inside it."""
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch(seq_len=12, lengths=(12, 5))

    with torch.no_grad():
        extended = _band_wrapper(model)(input_ids, attention_mask)
        plain = _band_wrapper(model, unstable_lengths=range(0))(input_ids, attention_mask)

    assert torch.allclose(extended, plain, rtol=1e-5, atol=1e-6)


def test_the_default_band_covers_the_lengths_the_neural_engine_gets_wrong() -> None:
    """The observed lengths are inside the band; the observed correct ones are outside."""
    band = eg2.NE_UNSTABLE_LENGTHS

    assert all(length in band for length in (501, 509, 510, 511, 512))
    assert all(length not in band for length in (500, 513, 520, 768))
    assert eg2.NE_PADDED_LENGTH == 520
    assert eg2.NE_PADDED_LENGTH % 8 == 0
    assert eg2.NE_PADDED_LENGTH > max(band)
    assert eg2.NE_PADDED_LENGTH - min(band) <= min(band)


@pytest.mark.parametrize(
    ("band", "padded_length"),
    [
        (range(9, 13), 12),  # the padded length is the band's own top
        (range(9, 13), 10),  # inside the band
        (range(9, 13), 8),  # below the band
        (range(9, 13), 19),  # more padding than the shortest band length holds
        (range(9, 13), True),
        (range(9, 13), 16.0),
        ((9, 0, 12), 16),
        ((9, "10"), 16),
    ],
    ids=[
        "top-of-band",
        "inside-band",
        "below-band",
        "too-far",
        "bool-length",
        "float-length",
        "zero-in-band",
        "string-in-band",
    ],
)
def test_the_wrapper_refuses_an_unusable_band(band: Any, padded_length: Any) -> None:
    """A padded length that does not take every band length out of the band is refused."""
    with pytest.raises(ValueError, match="band|padded"):
        _band_wrapper(_tiny_model(), unstable_lengths=band, padded_length=padded_length)


def test_an_empty_band_extends_nothing() -> None:
    """Without a band no length is extended, whatever the padded length says."""
    wrapper = _band_wrapper(_tiny_model(), unstable_lengths=(), padded_length=_TEST_PADDED_LENGTH)

    assert _lengths_the_backbone_sees(wrapper, 12) == [12]


def test_the_extension_pads_with_the_pad_token() -> None:
    """The ids appended to a band length are the model's pad id, and they are masked."""
    model = _tiny_model()
    seen: dict[str, torch.Tensor] = {}
    original_forward = model.forward

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen["input_ids"] = kwargs["input_ids"]
        seen["sliding"] = kwargs["attention_mask"][eg2.SLIDING_ATTENTION]
        return original_forward(*args, **kwargs)

    model.forward = spy
    input_ids, attention_mask = _right_padded_batch(seq_len=12, lengths=(12,))
    with torch.no_grad():
        _band_wrapper(model, pad_token_id=5)(input_ids, attention_mask)

    assert seen["input_ids"].shape == (1, _TEST_PADDED_LENGTH)
    assert torch.equal(seen["input_ids"][:, :12], input_ids)
    assert bool((seen["input_ids"][:, 12:] == 5).all())
    assert bool((seen["sliding"][..., 12:] == decoder_patches.MASK_FILL_VALUE).all())


@pytest.mark.parametrize(
    ("pad_token_id", "expected"),
    [(0, 0), (5, 5), (None, 0), (-1, 0), (True, 0), ("0", 0), (_TINY_VOCAB, 0)],
)
def test_wrap_pads_a_band_length_with_a_usable_pad_id(pad_token_id: Any, expected: int) -> None:
    """A missing or unusable pad id falls back to 0; the positions are masked either way."""
    model = SimpleNamespace(
        config=SimpleNamespace(
            sliding_window=_TINY_RADIUS, pad_token_id=pad_token_id, vocab_size=_TINY_VOCAB
        )
    )

    wrapper = eg2.EmbeddingGemma2Backend().wrap(_loaded(model))

    assert wrapper.pad_token_id == expected


def test_scaled_mean_pool_matches_the_shared_mean_pool() -> None:
    """Dividing before summing is the same mean, padded and fully masked rows included."""
    torch.manual_seed(5)
    hidden = torch.randn(4, 7, 5) * 100.0
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )

    pooled = eg2.scaled_mean_pool(hidden, attention_mask)

    expected = common.mean_pool(hidden, attention_mask)
    assert pooled.shape == (4, 5)
    assert torch.allclose(pooled, expected, rtol=1e-5, atol=1e-5)
    assert torch.equal(pooled[3], torch.zeros(5))
    assert torch.allclose(
        eg2.scaled_mean_pool(hidden, attention_mask.to(torch.float32)), pooled, rtol=0, atol=0
    )


def test_scaled_mean_pool_stays_within_float16_where_summing_first_overflows() -> None:
    """At the size of this model's outputs, a float16 sum of 1,024 tokens overflows."""
    hidden = torch.full((1, 1024, 4), 3400.0, dtype=torch.float16)
    attention_mask = torch.ones(1, 1024, dtype=torch.long)

    pooled = eg2.scaled_mean_pool(hidden, attention_mask)

    assert not bool(torch.isfinite(common.mean_pool(hidden, attention_mask)).all())
    assert bool(torch.isfinite(pooled).all())
    assert torch.allclose(pooled.float(), torch.full((1, 4), 3400.0), rtol=1e-2)


@pytest.mark.parametrize("pooling", [None, "cls", "lasttoken", "max", ""])
def test_wrap_rejects_a_pooling_no_wrapper_implements(pooling: str | None) -> None:
    """Only the declared mean pooling is this model; anything else must raise."""
    with pytest.raises(ValueError, match="pooling"):
        eg2.EmbeddingGemma2Backend().wrap(_loaded(_tiny_model(), pooling=pooling))


@pytest.mark.parametrize("radius", [None, 0, -3, True, "5", 2.0])
def test_wrap_rejects_a_configuration_without_a_usable_window(radius: Any) -> None:
    """The sliding layers cannot be masked without the window the model declares."""
    model = SimpleNamespace(config=SimpleNamespace(sliding_window=radius, pad_token_id=0))

    with pytest.raises(ValueError, match="sliding_window"):
        eg2.EmbeddingGemma2Backend().wrap(_loaded(model))


def test_wrap_hands_the_dense_to_the_wrapper() -> None:
    """Whatever ``load`` resolved must be what the traced module applies, after the pooling."""
    model = _tiny_model()
    torch.manual_seed(7)
    dense = torch.nn.Sequential(torch.nn.Linear(_TINY_EMBEDDING_DIM, 5, bias=False)).eval()
    input_ids, attention_mask = _right_padded_batch()

    wrapper = eg2.EmbeddingGemma2Backend().wrap(_loaded(model, dense=dense))
    with torch.no_grad():
        projected = wrapper(input_ids, attention_mask)
        pooled = eg2.EmbeddingGemma2MeanWrapper(model, window=_TINY_WINDOW, pad_token_id=0)(
            input_ids, attention_mask
        )

    assert wrapper.dense is dense
    assert projected.shape == (input_ids.shape[0], 5)
    assert torch.allclose(projected, dense(pooled))


def test_the_wrapper_pools_with_the_divide_first_mean() -> None:
    """The traced module pools exactly as :func:`scaled_mean_pool` does."""
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch()

    with torch.no_grad():
        hidden = model(input_ids=input_ids, attention_mask=_mask_mapping(attention_mask))[0]
        pooled = eg2.EmbeddingGemma2MeanWrapper(model, window=_TINY_WINDOW, pad_token_id=0)(
            input_ids, attention_mask
        )

    assert torch.equal(pooled, eg2.scaled_mean_pool(hidden, attention_mask))


def test_the_wrapper_asks_for_tuple_outputs_itself() -> None:
    """The wrapper must not depend on the configuration having been switched to tuples."""
    model = _tiny_model()
    model.config.return_dict = True
    input_ids, attention_mask = _right_padded_batch()

    with torch.no_grad():
        pooled = eg2.EmbeddingGemma2MeanWrapper(model, window=_TINY_WINDOW, pad_token_id=0)(
            input_ids, attention_mask
        )

    assert pooled.shape == (input_ids.shape[0], _TINY_EMBEDDING_DIM)


def test_the_wrapper_does_not_normalize_its_output() -> None:
    """The graph always returns the raw vector; normalization is the server's choice."""
    model = _shaped_model()
    input_ids, attention_mask = _right_padded_batch()

    with torch.no_grad():
        pooled = eg2.EmbeddingGemma2MeanWrapper(model, window=_TINY_WINDOW, pad_token_id=0)(
            input_ids, attention_mask
        )

    norms = torch.linalg.norm(pooled, dim=1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_the_wrapper_keeps_a_row_without_any_real_token_finite() -> None:
    """A fully masked row is never asked for, but it must not poison the batch with NaN."""
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch(seq_len=8, lengths=(8, 0))

    with torch.no_grad():
        pooled = eg2.EmbeddingGemma2MeanWrapper(model, window=_TINY_WINDOW, pad_token_id=0)(
            input_ids, attention_mask
        )

    assert bool(torch.isfinite(pooled).all())
    assert torch.equal(pooled[1], torch.zeros(_TINY_EMBEDDING_DIM))


# --- the traced graph ----------------------------------------------------------

# Traced cases: the sequence length of the example, whether the wrapper
# extends the small band, and the score limit the query rows are split at
# (``None`` for the shared default, under which nothing here is split)
# together with the number of ranges it leads to at the length the
# backbone runs at.
_TRACE_CASES: dict[str, tuple[int, bool, int | None, int]] = {
    "plain": (_SEQ_LEN_OUTSIDE_WINDOW, False, None, 1),
    "extended": (12, True, None, 1),
    "split": (_SEQ_LEN_OUTSIDE_WINDOW, False, 51, 2),
    "extended-split": (12, True, 65, 4),
}


def _traced_case(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> tuple[torch.nn.Module, torch.jit.ScriptModule, int]:
    """Patch a tiny model, wrap it as the case says and trace it.

    Returns:
        The wrapper, its trace and the sequence length it was traced at.
    """
    seq_len, extended, limit, _ = _TRACE_CASES[case]
    if limit is not None:
        monkeypatch.setattr(attention_split, "ATTENTION_SCORE_ELEMENT_LIMIT", limit)
    backend = eg2.EmbeddingGemma2Backend()
    model = _shaped_model()
    backend.apply_patches(_loaded(model))
    wrapper = _band_wrapper(model) if extended else backend.wrap(_loaded(model))
    input_ids, attention_mask = _right_padded_batch(seq_len=seq_len, lengths=(seq_len - 3, seq_len))
    traced = conversion.trace_model(wrapper, _example(input_ids, attention_mask))
    return wrapper, traced, seq_len


@pytest.mark.parametrize("case", list(_TRACE_CASES))
def test_the_traced_graph_is_static_and_stays_within_rank_four(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Neither the extension nor the split may add shape arithmetic or a wider tensor."""
    _, traced, _ = _traced_case(monkeypatch, case)

    ranks = _traced_ranks(traced)
    assert ranks  # the graph really was inspected
    assert max(ranks) <= _MAX_GRAPH_RANK
    assert _DYNAMIC_SIZE_NODE not in str(traced.inlined_graph)
    # The case is what its name says.
    ranges = _TRACE_CASES[case][3]
    assert len(_softmax_nodes(traced)) == _TINY_LAYERS * ranges


@pytest.mark.parametrize(
    ("case", "keys"),
    [("plain", _SEQ_LEN_OUTSIDE_WINDOW), ("extended", _TEST_PADDED_LENGTH)],
)
def test_the_traced_graph_attends_at_the_length_the_backbone_runs_at(
    monkeypatch: pytest.MonkeyPatch, case: str, keys: int
) -> None:
    """An extended graph attends over the padded length; a plain one over its own."""
    _, traced, _ = _traced_case(monkeypatch, case)

    assert _softmax_key_lengths(traced) == {keys}


@pytest.mark.parametrize("case", list(_TRACE_CASES))
def test_the_traced_graph_replays_another_batch(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    """A graph that baked in the example's padding would still pass its own trace."""
    wrapper, traced, seq_len = _traced_case(monkeypatch, case)
    other_ids, other_mask = _right_padded_batch(seq_len=seq_len, lengths=(seq_len, 4), seed=8)

    with torch.no_grad():
        expected = wrapper(other_ids, other_mask)
        replayed = traced(other_ids, other_mask)

    assert torch.allclose(replayed, expected, rtol=1e-5, atol=1e-6)


def test_the_unpatched_graph_holds_what_the_patches_remove() -> None:
    """Guard for the graph tests above: without the patches both constructs are there."""
    wrapper = eg2.EmbeddingGemma2Backend().wrap(_loaded(_tiny_model()))
    input_ids, attention_mask = _right_padded_batch(seq_len=_SEQ_LEN_OUTSIDE_WINDOW)

    traced = conversion.trace_model(wrapper, _example(input_ids, attention_mask))

    assert _DYNAMIC_SIZE_NODE in str(traced.inlined_graph)
    assert max(_traced_ranks(traced)) > _MAX_GRAPH_RANK


# --- interface conformance, refusals and dispatch -----------------------------


def test_backend_declares_the_interface_attributes() -> None:
    """The backend must name itself (matching its registry key) and its only kind."""
    backend = eg2.EmbeddingGemma2Backend()

    assert backend.name == "EmbeddingGemma2"
    assert backend.supported_kinds == ("embedding",)


def test_the_backend_matches_the_declared_interface_signature() -> None:
    """Every protocol member must exist here with the declared parameters."""
    members = sorted(
        name
        for name, member in vars(base.CompileBackend).items()
        if not name.startswith("_") and callable(member)
    )
    assert members  # the protocol really was inspected

    for name in members:
        implemented = getattr(eg2.EmbeddingGemma2Backend, name, None)
        assert implemented is not None, f"EmbeddingGemma2Backend does not implement {name}()"
        assert [
            (parameter, value.default)
            for parameter, value in inspect.signature(implemented).parameters.items()
        ] == [
            (parameter, value.default)
            for parameter, value in inspect.signature(
                getattr(base.CompileBackend, name)
            ).parameters.items()
        ]


def test_output_name_of_the_supported_kind() -> None:
    """Embeddings must keep the graph output name of the engine."""
    assert eg2.EmbeddingGemma2Backend().output_name("embedding") == "embedding"


def test_the_backend_serves_the_shared_sanity_sets() -> None:
    """Nothing about this family calls for fixtures of its own."""
    spec = eg2.EmbeddingGemma2Backend().sanity_spec("embedding")

    assert spec.input_sets == common.SANITY_TEXT_SETS
    assert spec.languages == ("en", "ja", "zh")
    assert spec.relevant_index is None
    assert spec.irrelevant_index is None


def test_the_fixtures_are_non_empty_single_texts() -> None:
    """A fully masked row says nothing about the model, so no fixture may be empty."""
    backend = eg2.EmbeddingGemma2Backend()

    assert isinstance(backend.trace_example("embedding"), str)
    assert backend.trace_example("embedding")
    assert isinstance(backend.padding_input("embedding"), str)
    assert backend.padding_input("embedding")


def test_pair_template_of_an_embedding_model_is_none(tmp_path: Path) -> None:
    """An embedding model encodes one text and has no pair to shape."""
    assert eg2.EmbeddingGemma2Backend().pair_template(tmp_path, "embedding") is None


def test_the_reranker_score_space_follows_the_embedding_only_precedent() -> None:
    """A backend without a reranker answers as the other embedding-only backends do."""
    assert eg2.EmbeddingGemma2Backend().reranker_score_space() == base.SCORE_SPACE_PROBABILITY


@pytest.mark.parametrize("method", ["trace_example", "sanity_spec", "padding_input", "output_name"])
@pytest.mark.parametrize("kind", ["reranker", "classifier", "image"])
def test_unknown_kind_is_rejected(method: str, kind: str) -> None:
    """Every kind-dispatching method must reject a kind this backend cannot compile."""
    with pytest.raises(ValueError, match="kind"):
        getattr(eg2.EmbeddingGemma2Backend(), method)(kind)


@pytest.mark.parametrize("kind", ["reranker", "classifier"])
def test_pair_template_rejects_an_unsupported_kind(tmp_path: Path, kind: str) -> None:
    """A kind this backend cannot compile must be refused, not answered with None."""
    with pytest.raises(ValueError, match="kind"):
        eg2.EmbeddingGemma2Backend().pair_template(tmp_path, kind)


@pytest.mark.parametrize("kind", ["reranker", "classifier"])
def test_load_rejects_an_unsupported_kind(tmp_path: Path, kind: str) -> None:
    """load must validate the kind before touching the filesystem."""
    with pytest.raises(ValueError, match="kind"):
        eg2.EmbeddingGemma2Backend().load(tmp_path, kind)


def test_a_reranker_is_refused_with_the_reason(tmp_path: Path) -> None:
    """Asking for another kind must say what this backend compiles instead."""
    model_dir = _model_dir(tmp_path, pooling_mode="mean")

    with pytest.raises(ValueError) as excinfo:
        eg2.EmbeddingGemma2Backend().load(model_dir, "reranker")

    message = str(excinfo.value)
    assert "reranker" in message
    assert "EmbeddingGemma 2" in message
    assert "text" in message
    assert "image" in message
    assert "audio" in message


@pytest.mark.parametrize("method", ["wrap", "tokenize"])
@pytest.mark.parametrize("kind", ["reranker", "classifier"])
def test_handle_taking_methods_reject_an_unsupported_kind(method: str, kind: str) -> None:
    """A handle carrying an unsupported kind must be rejected, not silently wrapped."""
    loaded = _loaded(_tiny_model(), kind=kind)
    arguments = {"tokenize": (["text"], 8)}.get(method, ())

    with pytest.raises(ValueError, match="kind"):
        getattr(eg2.EmbeddingGemma2Backend(), method)(loaded, *arguments)


def test_reference_outputs_rejects_an_unsupported_kind(tmp_path: Path) -> None:
    """The reference path must validate the kind before loading any weights."""
    with pytest.raises(ValueError, match="kind"):
        eg2.EmbeddingGemma2Backend().reference_outputs(tmp_path, "reranker", [("q", "d")], 8)


def test_reference_outputs_rejects_empty_inputs(tmp_path: Path) -> None:
    """No inputs means nothing to compare; that must raise before loading weights."""
    with pytest.raises(ValueError, match="inputs"):
        eg2.EmbeddingGemma2Backend().reference_outputs(tmp_path, "embedding", [], 8)


# None of the directories below holds any weight, so an error that names
# the declaration proves it was decided before a checkpoint was looked for.


@pytest.mark.parametrize("declared", ["gemma3_text", "gemma3", "gemma4", "qwen3", None])
def test_load_rejects_a_directory_of_another_model_type(
    tmp_path: Path, declared: str | None
) -> None:
    """A related architecture must be refused before any weight is read."""
    model_dir = _model_dir(
        tmp_path, config=_declared_config(model_type=declared), pooling_mode="mean"
    )

    with pytest.raises(ValueError) as excinfo:
        eg2.EmbeddingGemma2Backend().load(model_dir, "embedding")

    message = str(excinfo.value)
    assert "model_type" in message
    assert repr(eg2.MODEL_TYPE) in message
    assert repr(declared) in message


def test_load_rejects_the_text_model_saved_on_its_own(tmp_path: Path) -> None:
    """A directory holding only the text model is refused with what to point at instead."""
    model_dir = _model_dir(
        tmp_path,
        config={
            "architectures": ["EmbeddingGemma2TextModel"],
            "model_type": eg2.TEXT_MODEL_TYPE,
            "max_position_embeddings": _TINY_POSITIONS,
        },
        pooling_mode="mean",
    )

    with pytest.raises(ValueError) as excinfo:
        eg2.EmbeddingGemma2Backend().load(model_dir, "embedding")

    message = str(excinfo.value)
    assert repr(eg2.TEXT_MODEL_TYPE) in message
    assert "multimodal" in message
    assert "root" in message


@pytest.mark.parametrize("text_config", [None, "embedding_gemma2_text", ["model_type"], 3], ids=str)
def test_load_rejects_a_directory_without_a_text_configuration(
    tmp_path: Path, text_config: Any
) -> None:
    """Without the text model's configuration there is nothing this backend can compile."""
    config = _declared_config()
    if text_config is None:
        del config["text_config"]
    else:
        config["text_config"] = text_config
    model_dir = _model_dir(tmp_path, config=config, pooling_mode="mean")

    with pytest.raises(ValueError, match="text_config"):
        eg2.EmbeddingGemma2Backend().load(model_dir, "embedding")


def test_load_rejects_a_directory_without_a_readable_config(tmp_path: Path) -> None:
    """No configuration means no declared architecture, which is a refusal of its own."""
    with pytest.raises(ValueError, match="model_type"):
        eg2.EmbeddingGemma2Backend().load(tmp_path / "absent", "embedding")


def test_the_architecture_is_refused_before_the_pooling(tmp_path: Path) -> None:
    """The more fundamental refusal comes first, so the message names the real problem."""
    model_dir = _model_dir(tmp_path, config=_declared_config(model_type="gemma3_text"))

    with pytest.raises(ValueError, match="model_type"):
        eg2.EmbeddingGemma2Backend().load(model_dir, "embedding")


def test_load_reports_an_undeclared_pooling_before_loading_weights(tmp_path: Path) -> None:
    """An embedding model without a pooling declaration must fail fast and clearly."""
    with pytest.raises(ValueError) as excinfo:
        eg2.EmbeddingGemma2Backend().load(_model_dir(tmp_path), "embedding")

    assert common.POOLING_DIRNAME in str(excinfo.value)


def test_reference_outputs_refuses_what_load_refuses(tmp_path: Path) -> None:
    """Whichever side of the self-check reaches a directory first, it fails the same way."""
    model_dir = _model_dir(tmp_path, config=_declared_config(model_type=eg2.TEXT_MODEL_TYPE))

    with pytest.raises(ValueError, match="model_type"):
        eg2.EmbeddingGemma2Backend().reference_outputs(model_dir, "embedding", ["text"], 8)


@pytest.mark.parametrize("configured", [2048, 512, 1])
def test_max_seq_len_reports_the_text_models_position_budget(
    tmp_path: Path, configured: int
) -> None:
    """The budget lives in the text configuration; rotary positions reserve no slot."""
    config = _declared_config(
        text_config={"max_position_embeddings": configured, "sliding_window": 512}
    )

    assert eg2.EmbeddingGemma2Backend().max_seq_len(_model_dir(tmp_path, config=config)) == (
        configured
    )


def test_max_seq_len_does_not_read_the_root_level(tmp_path: Path) -> None:
    """A value beside the text configuration is not the text model's budget."""
    config = _declared_config(max_position_embeddings=4096, text_config={})

    assert eg2.EmbeddingGemma2Backend().max_seq_len(_model_dir(tmp_path, config=config)) is None


@pytest.mark.parametrize(
    "text_config",
    [
        {},
        {"max_position_embeddings": None},
        {"max_position_embeddings": "512"},
        {"max_position_embeddings": 512.0},
        {"max_position_embeddings": True},
        {"max_position_embeddings": 0},
        {"max_position_embeddings": -1},
        None,
        "512",
    ],
    ids=["absent", "null", "string", "float", "bool", "zero", "negative", "no-text", "not-a-dict"],
)
def test_max_seq_len_ignores_a_missing_or_unusable_value(tmp_path: Path, text_config: Any) -> None:
    """A missing or nonsensical value must degrade to 'unknown', never to a bogus limit."""
    config = _declared_config()
    if text_config is None:
        del config["text_config"]
    else:
        config["text_config"] = text_config

    assert eg2.EmbeddingGemma2Backend().max_seq_len(_model_dir(tmp_path, config=config)) is None


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"], ids=["corrupt", "not-an-object"])
def test_max_seq_len_of_an_unusable_config_is_none(tmp_path: Path, content: str) -> None:
    """An unparsable config must not turn an optional check into a compile failure."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / eg2.CONFIG_FILENAME).write_text(content, encoding="utf-8")

    assert eg2.EmbeddingGemma2Backend().max_seq_len(model_dir) is None


def test_max_seq_len_of_a_missing_directory_is_none(tmp_path: Path) -> None:
    """No config.json means no known limit, not a crash."""
    assert eg2.EmbeddingGemma2Backend().max_seq_len(tmp_path / "absent") is None


def test_the_published_architecture_dispatches_to_this_backend(tmp_path: Path) -> None:
    """The root class the model is published under selects this backend, as an embedding model."""
    model_dir = _model_dir(tmp_path)

    result = dispatch.resolve_dispatch(model_dir)

    assert result.architecture == "EmbeddingGemma2Model"
    assert result.backend_name == "EmbeddingGemma2"
    assert result.kind == dispatch.KIND_EMBEDDING
    backend = result.load_backend()
    assert type(backend).__name__ == "EmbeddingGemma2Backend"
    assert backend.name == result.backend_name


def test_the_registered_prefix_collides_with_no_other_backend() -> None:
    """No key may start with another one, and the neighbouring families keep their backends."""
    keys = list(dispatch.BACKEND_REGISTRY)

    assert "EmbeddingGemma2" in keys
    for key in keys:
        for other in keys:
            assert key == other or not other.startswith(key)
    assert dispatch.select_backend(["Gemma3TextModel"])[1] == "Gemma3Text"
    assert dispatch.select_backend(["EmbeddingGemma2Model"])[1] == "EmbeddingGemma2"


@pytest.mark.parametrize("architecture", ["EmbeddingGemmaModel", "Gemma2Model", "Gemma4Model"])
def test_a_relative_whose_name_differs_is_not_dispatched_here(
    tmp_path: Path, architecture: str
) -> None:
    """Only names beginning with the registered prefix reach this backend."""
    model_dir = _model_dir(tmp_path, config=_declared_config(architectures=[architecture]))

    with pytest.raises(dispatch.UnsupportedArchitectureError):
        dispatch.resolve_dispatch(model_dir)


def test_the_unsupported_architecture_message_names_this_family() -> None:
    """The list of supported families names this one, and still names every earlier one."""
    message = dispatch.SUPPORTED_ARCHITECTURES

    assert "EmbeddingGemma 2" in message
    assert "google/embeddinggemma-2" in message
    assert "Gemma 3 bidirectional embedding models (e.g. google/embeddinggemma-300m)" in message


# --- round trip through a synthetic saved model directory --------------------


def _write_model_directory(directory: Path) -> Path:
    """Save the tiny root model as a sentence-transformers 6 embedding directory.

    The directory declares what the published model does: a Transformer,
    a mean pooling over the projected token vectors and a trailing
    normalization, in the module types and the pooling dialect of the 6
    series.

    Args:
        directory: Destination directory; created if needed.

    Returns:
        ``directory``, holding weights, config, tokenizer files and the
        sentence-transformers declarations.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _tiny_root().save_pretrained(directory)
    _toy_tokenizer().save_pretrained(directory)
    _write_json(
        directory / common.POOLING_DIRNAME / common.POOLING_CONFIG_FILENAME,
        {
            "embedding_dimension": _TINY_EMBEDDING_DIM,
            "pooling_mode": "mean",
            "include_prompt": True,
        },
    )
    _write_json(
        directory / common.ST_MODULES_FILENAME,
        [
            {"idx": 0, "name": "0", "path": "", "type": _ST6_TRANSFORMER},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": _ST6_POOLING},
            {"idx": 2, "name": "2", "path": "2_Normalize", "type": _ST6_NORMALIZE},
        ],
    )
    return directory


@pytest.fixture(scope="module")
def embedding_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic EmbeddingGemma 2 embedding model directory."""
    return _write_model_directory(tmp_path_factory.mktemp("embedding-gemma2"))


def test_the_saved_directory_is_the_multimodal_root(embedding_dir: Path) -> None:
    """Guard for the tests below: the directory is laid out as the published one is."""
    config = json.loads((embedding_dir / eg2.CONFIG_FILENAME).read_text(encoding="utf-8"))

    assert config["architectures"] == ["EmbeddingGemma2Model"]
    assert config["model_type"] == eg2.MODEL_TYPE
    assert config["text_config"]["model_type"] == eg2.TEXT_MODEL_TYPE
    assert config["text_config"]["sliding_window"] == _TINY_RADIUS
    assert config["vision_config"] is None
    assert config["audio_config"] is None


def test_load_returns_a_conforming_handle(embedding_dir: Path) -> None:
    """load must hand back the eval/FP32/tuple-output text model plus its tokenizer and config."""
    loaded = eg2.EmbeddingGemma2Backend().load(embedding_dir, "embedding")

    assert loaded.kind == "embedding"
    assert loaded.attn == "eager"
    assert loaded.pooling == common.POOLING_MEAN
    assert loaded.dense is None
    assert loaded.dense_config is None
    assert loaded.model_dir == embedding_dir
    assert isinstance(loaded.model, EmbeddingGemma2TextModel)
    assert loaded.model.training is False
    assert next(loaded.model.parameters()).dtype == torch.float32
    assert loaded.config.return_dict is False
    assert loaded.config is loaded.model.config
    assert loaded.config._attn_implementation == "eager"
    assert loaded.config.sliding_window == _TINY_RADIUS
    assert loaded.tokenizer("x")["input_ids"]


def test_load_reads_the_saved_text_weights(embedding_dir: Path) -> None:
    """The text model loaded out of the root holds the weights the root was saved with."""
    loaded = eg2.EmbeddingGemma2Backend().load(embedding_dir, "embedding")
    saved = _tiny_root().language_model.state_dict()

    state = loaded.model.state_dict()
    assert set(state) == set(saved)
    assert all(torch.equal(state[name], saved[name]) for name in saved)


def test_load_honours_the_requested_attention_path(embedding_dir: Path) -> None:
    """The reference path loads on sdpa, which the patched eager functions never reach."""
    loaded = eg2.EmbeddingGemma2Backend().load(embedding_dir, "embedding", attn="sdpa")

    assert loaded.attn == "sdpa"
    assert loaded.config._attn_implementation == "sdpa"


def test_wrap_takes_the_window_from_the_loaded_radius(embedding_dir: Path) -> None:
    """The loaded configuration keeps the stored radius, and the mask adds one to it."""
    backend = eg2.EmbeddingGemma2Backend()

    wrapper = backend.wrap(backend.load(embedding_dir, "embedding"))

    assert wrapper.window == _TINY_RADIUS + 1


def test_tokenize_returns_fixed_shape_int32_arrays(embedding_dir: Path) -> None:
    """Tokenized Core ML inputs must be (N, S) int32 with only the two graph keys."""
    backend = eg2.EmbeddingGemma2Backend()
    loaded = backend.load(embedding_dir, "embedding")
    inputs = ["a short text", "a considerably longer text than the first one"]

    tokens = backend.tokenize(loaded, inputs, _ROUND_TRIP_SEQ_LEN)

    assert set(tokens) == {"input_ids", "attention_mask"}
    for key in ("input_ids", "attention_mask"):
        assert tokens[key].dtype == np.int32
        assert tokens[key].shape == (len(inputs), _ROUND_TRIP_SEQ_LEN)
    assert tokens["attention_mask"][0, -1] == 0
    assert tokens["attention_mask"][0, 0] == 1


def test_tokenize_leaves_the_special_tokens_to_the_tokenizer(embedding_dir: Path) -> None:
    """The begin and end markers come from the tokenizer's own template, exactly once."""
    backend = eg2.EmbeddingGemma2Backend()
    loaded = backend.load(embedding_dir, "embedding")

    tokens = backend.tokenize(loaded, ["abc"], _ROUND_TRIP_SEQ_LEN)

    row = tokens["input_ids"][0]
    real = int(tokens["attention_mask"][0].sum())
    assert real == len("abc") + 2
    assert row[0] == _BOS_ID
    assert row[real - 1] == _EOS_ID
    assert int((row == _BOS_ID).sum()) == 1
    assert int((row == _EOS_ID).sum()) == 1


def test_the_padding_input_encodes_to_a_non_empty_mask(embedding_dir: Path) -> None:
    """A filler row that masked out everything would say nothing about the model."""
    backend = eg2.EmbeddingGemma2Backend()
    loaded = backend.load(embedding_dir, "embedding")

    tokens = backend.tokenize(loaded, [backend.padding_input("embedding")], _ROUND_TRIP_SEQ_LEN)

    assert tokens["attention_mask"].sum() > 0


def test_tokenize_rejects_empty_inputs(embedding_dir: Path) -> None:
    """An empty batch would produce a zero-row graph input and must raise."""
    backend = eg2.EmbeddingGemma2Backend()
    loaded = backend.load(embedding_dir, "embedding")

    with pytest.raises(ValueError, match="inputs"):
        backend.tokenize(loaded, [], _ROUND_TRIP_SEQ_LEN)


@pytest.mark.parametrize("seq_len", [0, -1])
def test_tokenize_rejects_a_non_positive_sequence_length(embedding_dir: Path, seq_len: int) -> None:
    """A non-positive fixed length is never a valid graph shape."""
    backend = eg2.EmbeddingGemma2Backend()
    loaded = backend.load(embedding_dir, "embedding")

    with pytest.raises(ValueError, match="seq_len"):
        backend.tokenize(loaded, ["abc"], seq_len)


def test_reference_outputs_loads_a_copy_of_its_own_on_the_other_attention_path(
    embedding_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The baseline must not reuse the implementation the conversion rewrites."""
    requested: list[str] = []
    original = eg2.EmbeddingGemma2Backend.load

    def spy(self: Any, model_dir: Path, kind: str, attn: str = "eager") -> base.LoadedModel:
        requested.append(attn)
        return original(self, model_dir, kind, attn=attn)

    monkeypatch.setattr(eg2.EmbeddingGemma2Backend, "load", spy)

    eg2.EmbeddingGemma2Backend().reference_outputs(embedding_dir, "embedding", ["text"], 8)

    assert requested == ["sdpa"]


def test_reference_outputs_refuses_a_pooling_the_wrapper_would_refuse(tmp_path: Path) -> None:
    """The baseline must not quietly pool another way than the traced module can."""
    model_dir = _write_model_directory(tmp_path / "cls")
    _write_json(
        model_dir / common.POOLING_DIRNAME / common.POOLING_CONFIG_FILENAME,
        {"embedding_dimension": _TINY_EMBEDDING_DIM, "pooling_mode": "cls"},
    )

    with pytest.raises(ValueError, match="pooling"):
        eg2.EmbeddingGemma2Backend().reference_outputs(model_dir, "embedding", ["text"], 8)


@pytest.fixture(scope="module")
def round_trip(embedding_dir: Path) -> Iterator[dict[str, Any]]:
    """Run the patched wrapper and the FP32 sdpa reference over the sanity fixtures."""
    backend = eg2.EmbeddingGemma2Backend()
    inputs = list(backend.sanity_spec("embedding").all_inputs)
    loaded = backend.load(embedding_dir, "embedding")
    try:
        applied = backend.apply_patches(loaded)
        wrapper = backend.wrap(loaded)
        tokens = backend.tokenize(loaded, inputs, _ROUND_TRIP_SEQ_LEN)
        with torch.no_grad():
            wrapped = wrapper(
                torch.from_numpy(tokens["input_ids"]).long(),
                torch.from_numpy(tokens["attention_mask"]).long(),
            )
        reference = backend.reference_outputs(
            embedding_dir, "embedding", inputs, _ROUND_TRIP_SEQ_LEN
        )
    finally:
        # A module-scoped fixture is set up before the function-scoped
        # restore fixture runs, so it puts the upstream symbols back itself.
        _restore_upstream_symbols()
    del loaded, wrapper
    gc.collect()
    yield {
        "inputs": inputs,
        "applied": applied,
        "tokens": tokens,
        "wrapped": wrapped.numpy().reshape(len(inputs), -1),
        "reference": np.asarray(reference, dtype=np.float32).reshape(len(inputs), -1),
    }


def test_the_round_trip_runs_beyond_the_window(round_trip: dict[str, Any]) -> None:
    """Guard for the comparison below: its rows must be longer than the window."""
    real_lengths = round_trip["tokens"]["attention_mask"].sum(axis=1)

    assert int(real_lengths.max()) > _TINY_WINDOW
    assert round_trip["applied"][fp16_ranges.FP16_RANGE_CONDITIONING_KEY][
        "calibration_inputs"
    ] == 1 + len(round_trip["inputs"])


def test_the_wrapper_matches_the_fp32_reference(round_trip: dict[str, Any]) -> None:
    """The traced module and the baseline must compute the same function.

    The two sides reach it by different routes -- the wrapper builds its
    own two masks over the patched eager attention of a rescaled model and
    pools by dividing first; the baseline lets the framework build the
    masks for the sdpa attention of an untouched copy and pools by summing
    first -- so a disagreement about the masks, the window, the rescaling,
    the pooling or the tokenization shows up here.
    """
    assert np.isfinite(round_trip["wrapped"]).all()
    assert np.isfinite(round_trip["reference"]).all()
    assert round_trip["reference"].shape == (len(round_trip["inputs"]), _TINY_EMBEDDING_DIM)
    assert round_trip["reference"].dtype == np.float32
    np.testing.assert_allclose(
        round_trip["wrapped"], round_trip["reference"], rtol=0, atol=_ROUND_TRIP_TOLERANCE
    )
    assert float(_cosines(round_trip["wrapped"], round_trip["reference"]).min()) >= (
        _EQUIVALENCE_COSINE_THRESHOLD
    )


def test_the_reference_distinguishes_the_english_fixtures(round_trip: dict[str, Any]) -> None:
    """Guard for the comparison above: identical rows would make it prove nothing.

    Only the English set is compared: the toy tokenizer knows no byte of
    the other two, which therefore encode to unknown tokens alone.
    """
    reference = round_trip["reference"]
    english = len(common.SANITY_TEXT_SETS[0][1])

    for row in range(1, english):
        assert not np.allclose(reference[0], reference[row], atol=_ROUND_TRIP_TOLERANCE)


def test_the_wrapper_matches_the_fp32_reference_on_padded_rows(embedding_dir: Path) -> None:
    """A batch mixing a short and a long row must agree with the baseline row by row.

    The byte-level tokenizer fills every English sanity fixture up to the
    fixed length, so the comparison over those never pads; this one does,
    with one row inside the window and one beyond it.
    """
    backend = eg2.EmbeddingGemma2Backend()
    inputs = ["a", "a text long enough to fill the whole row"]
    loaded = backend.load(embedding_dir, "embedding")
    backend.apply_patches(loaded)
    wrapper = backend.wrap(loaded)
    tokens = backend.tokenize(loaded, inputs, _ROUND_TRIP_SEQ_LEN)
    with torch.no_grad():
        wrapped = wrapper(
            torch.from_numpy(tokens["input_ids"]).long(),
            torch.from_numpy(tokens["attention_mask"]).long(),
        ).numpy()

    reference = backend.reference_outputs(embedding_dir, "embedding", inputs, _ROUND_TRIP_SEQ_LEN)

    real_lengths = tokens["attention_mask"].sum(axis=1).tolist()
    # The short row lies within the radius of every one of its positions.
    assert real_lengths[0] - 1 <= _TINY_RADIUS < real_lengths[1] == _ROUND_TRIP_SEQ_LEN
    np.testing.assert_allclose(wrapped, reference, rtol=0, atol=_ROUND_TRIP_TOLERANCE)


# --- Core ML conversion of the traced wrapper --------------------------------


def _convert_and_predict(
    backend: eg2.EmbeddingGemma2Backend,
    wrapper: torch.nn.Module,
    tokens: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, Any, torch.jit.ScriptModule]:
    """Trace, convert to an FP16 ML program and predict ``tokens``.

    Returns:
        The FP32 wrapper's output, the converted program's output, the
        program itself and the trace it was converted from.
    """
    batch_size = tokens["input_ids"].shape[0]
    with torch.no_grad():
        expected = (
            wrapper(
                torch.from_numpy(tokens["input_ids"]).long(),
                torch.from_numpy(tokens["attention_mask"]).long(),
            )
            .numpy()
            .astype(np.float32)
        )
    traced = conversion.trace_model(wrapper, tokens)
    mlmodel = conversion.convert_model(
        traced,
        _ROUND_TRIP_SEQ_LEN,
        "fp16",
        "macos13",
        backend.output_name("embedding"),
        batch_size=batch_size,
    )
    prediction = mlmodel.predict(dict(tokens))
    embeddings = np.asarray(prediction["embedding"], dtype=np.float32).reshape(batch_size, -1)
    return expected, embeddings, mlmodel, traced


def test_the_traced_wrapper_converts_and_agrees_with_the_fp32_wrapper(
    embedding_dir: Path,
) -> None:
    """The conversion must produce a program that still computes the same embedding."""
    backend = eg2.EmbeddingGemma2Backend()
    loaded = backend.load(embedding_dir, "embedding")
    backend.apply_patches(loaded)
    wrapper = backend.wrap(loaded)
    inputs = list(backend.sanity_spec("embedding").input_sets[0][1])
    tokens = backend.tokenize(loaded, inputs, _ROUND_TRIP_SEQ_LEN)

    expected, embeddings, mlmodel, _ = _convert_and_predict(backend, wrapper, tokens)

    description = mlmodel.get_spec().description
    assert [tensor.name for tensor in description.input] == ["input_ids", "attention_mask"]
    assert [tensor.name for tensor in description.output] == ["embedding"]
    assert embeddings.shape == expected.shape == (len(inputs), _TINY_EMBEDDING_DIM)
    assert bool(np.isfinite(embeddings).all())
    assert float(_cosines(embeddings, expected).min()) >= _CONVERSION_COSINE_THRESHOLD


def test_the_extended_wrapper_converts_and_agrees_with_the_fp32_wrapper(
    embedding_dir: Path,
) -> None:
    """A graph that extends its sequence before attending must convert as well."""
    backend = eg2.EmbeddingGemma2Backend()
    loaded = backend.load(embedding_dir, "embedding")
    backend.apply_patches(loaded)
    wrapper = eg2.EmbeddingGemma2MeanWrapper(
        loaded.model,
        window=_TINY_WINDOW,
        pad_token_id=_PAD_ID,
        unstable_lengths=_ROUND_TRIP_BAND,
        padded_length=_ROUND_TRIP_PADDED_LENGTH,
    ).eval()
    inputs = ["ab", *backend.sanity_spec("embedding").input_sets[0][1]]
    tokens = backend.tokenize(loaded, inputs, _ROUND_TRIP_SEQ_LEN)

    expected, embeddings, _, traced = _convert_and_predict(backend, wrapper, tokens)

    # The graph that was converted really is extended.
    assert _softmax_key_lengths(traced) == {_ROUND_TRIP_PADDED_LENGTH}
    assert embeddings.shape == expected.shape == (len(inputs), _TINY_EMBEDDING_DIM)
    assert bool(np.isfinite(embeddings).all())
    assert float(_cosines(embeddings, expected).min()) >= _CONVERSION_COSINE_THRESHOLD
