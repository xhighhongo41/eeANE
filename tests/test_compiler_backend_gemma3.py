"""Tests for the Gemma 3 compile backend (bidirectional embedding models).

Four layers:

* The two conversion patches, checked against the upstream implementations
  they replace, and the in-graph attention masks, checked against
  hand-written expectations and against the framework's own mask path.
* The traceable wrapper: what it pools, what it projects, and what its
  traced graph must no longer contain.
* Conformance to the backend interface declared in
  ``eeane.compiler.backends.base``: the kind validation, the fixtures and
  the refusal of every directory this backend does not implement.
* The round trip through a synthetic saved model directory, up to the
  Core ML conversion of the traced wrapper.

Nothing here downloads weights: every model is a small randomly
initialised Gemma 3 text model built from an in-test configuration.
"""

from __future__ import annotations

import gc
import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
from transformers.models.gemma3 import modeling_gemma3
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel
from transformers.models.qwen3 import modeling_qwen3

from eeane.compiler import conversion
from eeane.compiler.backends import base, common
from eeane.compiler.backends import gemma3 as g3

# The upstream implementations, captured at import time (that is, before
# any test can have replaced them). They are both the comparison baseline
# for the patch tests and the value the autouse fixture restores.
_UPSTREAM_ROTATE_HALF = modeling_gemma3.rotate_half
_UPSTREAM_REPEAT_KV = modeling_gemma3.repeat_kv

# The same two symbols of another decoder family, which the patches of
# this backend must leave alone.
_OTHER_FAMILY_ROTATE_HALF = modeling_qwen3.rotate_half
_OTHER_FAMILY_REPEAT_KV = modeling_qwen3.repeat_kv

# Geometry of the tiny model used throughout: two grouped query heads per
# key/value head, so ``repeat_kv`` really has to expand something, an even
# head dimension, which ``rotate_half`` requires, and both layer types, so
# that both of the two masks are consumed.
_TINY_HIDDEN = 32
_TINY_HEADS = 4
_TINY_KV_HEADS = 2
_TINY_HEAD_DIM = 8
_TINY_INTERMEDIATE = 48
_TINY_POSITIONS = 64
_TINY_LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]

# The window as a published configuration stores it, and as the model sees
# it: a bidirectional configuration rewrites the stored value to
# ``stored // 2 + 1`` when it is constructed.
_TINY_STORED_WINDOW = 8
_TINY_WINDOW = 5

# Sequence lengths on either side of the window. At the short one no two
# positions are a window apart, so the sliding layers see everything; at
# the long one most pairs are out of reach of each other.
_SEQ_LEN_INSIDE_WINDOW = 4
_SEQ_LEN_OUTSIDE_WINDOW = 16

# Vocabulary of the synthetic model directory: the byte-level tokenizer
# below emits up to ~260 distinct ids (256 bytes plus its special tokens),
# so every id it can produce stays addressable.
_TINY_VOCAB = 300
_PAD_ID = 0
_EOS_ID = 1
_BOS_ID = 2

# Widths of the two Dense projections the synthetic directory declares
# after its pooling: out to a wider space and back, both without a bias.
_DENSE_WIDE = 48
_DENSE_OUT = 32

# Sequence length of the round-trip and conversion tests: longer than the
# window, and short enough to leave the sanity fixtures visibly truncated.
_ROUND_TRIP_SEQ_LEN = 16

# Minimum cosine between the in-graph masks and the framework's own mask
# path, both in FP32 over the same weights.
_MASK_PATH_COSINE_THRESHOLD = 0.999999

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


@pytest.fixture(autouse=True)
def _restore_transformers_patches() -> Iterator[None]:
    """Undo the process-wide Gemma 3 monkeypatches after every test.

    ``patch_rotate_half`` and ``patch_repeat_kv`` rebind names in the
    transformers module, which affects every Gemma 3 model in the process,
    so each test starts and ends on the upstream implementations.
    """
    modeling_gemma3.rotate_half = _UPSTREAM_ROTATE_HALF
    modeling_gemma3.repeat_kv = _UPSTREAM_REPEAT_KV
    try:
        yield
    finally:
        modeling_gemma3.rotate_half = _UPSTREAM_ROTATE_HALF
        modeling_gemma3.repeat_kv = _UPSTREAM_REPEAT_KV


def _tiny_config(vocab_size: int = _TINY_VOCAB) -> Gemma3TextConfig:
    """Build the configuration of the tiny model used by these tests."""
    config = Gemma3TextConfig(
        vocab_size=vocab_size,
        hidden_size=_TINY_HIDDEN,
        intermediate_size=_TINY_INTERMEDIATE,
        num_hidden_layers=len(_TINY_LAYER_TYPES),
        num_attention_heads=_TINY_HEADS,
        num_key_value_heads=_TINY_KV_HEADS,
        head_dim=_TINY_HEAD_DIM,
        max_position_embeddings=_TINY_POSITIONS,
        sliding_window=_TINY_STORED_WINDOW,
        layer_types=list(_TINY_LAYER_TYPES),
        use_bidirectional_attention=True,
        query_pre_attn_scalar=_TINY_HEAD_DIM,
        pad_token_id=_PAD_ID,
        bos_token_id=_BOS_ID,
        eos_token_id=_EOS_ID,
        attn_implementation="eager",
    )
    config.use_cache = False
    config.return_dict = False
    return config


def _tiny_model(seed: int = 0, vocab_size: int = _TINY_VOCAB) -> Gemma3TextModel:
    """Build a small randomly initialised Gemma 3 text model in eval mode."""
    torch.manual_seed(seed)
    return Gemma3TextModel(_tiny_config(vocab_size)).eval()


def _right_padded_batch(
    seq_len: int = 12, lengths: tuple[int, ...] = (7, 10), seed: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a right-padded batch whose rows have the given real lengths."""
    torch.manual_seed(seed)
    input_ids = torch.randint(3, _TINY_VOCAB, (len(lengths), seq_len))
    attention_mask = torch.ones(len(lengths), seq_len, dtype=torch.long)
    for row, length in enumerate(lengths):
        attention_mask[row, length:] = 0
        input_ids[row, length:] = _PAD_ID
    return input_ids, attention_mask


def _mask_mapping(attention_mask: torch.Tensor, window: int = _TINY_WINDOW) -> dict[str, Any]:
    """Build the per-layer-type mask mapping the wrapper hands the backbone."""
    full, sliding = g3.build_bidirectional_masks(attention_mask, window)
    return {g3.FULL_ATTENTION: full, g3.SLIDING_ATTENTION: sliding}


def _row_cosines(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return the cosine similarity of every pair of rows along the last axis."""
    return torch.nn.functional.cosine_similarity(left, right, dim=-1)


def _loaded(
    model: Any,
    kind: str = "embedding",
    pooling: str | None = common.POOLING_MEAN,
    tokenizer: Any = None,
    model_dir: Path = Path("/nonexistent-model-dir"),
    dense: Any = None,
) -> base.LoadedModel:
    """Build the handle the backend interface passes between its stages."""
    return base.LoadedModel(
        model=model,
        tokenizer=tokenizer,
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


def _write_json(path: Path, payload: object) -> None:
    """Write ``payload`` as JSON, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _declared_config(**overrides: Any) -> dict[str, Any]:
    """Build the config.json fields this backend decides a directory on."""
    config: dict[str, Any] = {
        "architectures": ["Gemma3TextModel"],
        "model_type": g3.MODEL_TYPE,
        "use_bidirectional_attention": True,
    }
    config.update(overrides)
    return {key: value for key, value in config.items() if value is not None}


def _model_dir(tmp_path: Path, config: dict[str, Any] | None = None, **pooling: Any) -> Path:
    """Build a weightless model directory with a config.json and pooling module."""
    model_dir = tmp_path / "model"
    _write_json(
        model_dir / g3.CONFIG_FILENAME, config if config is not None else _declared_config()
    )
    if pooling:
        _write_json(model_dir / common.POOLING_DIRNAME / common.POOLING_CONFIG_FILENAME, pooling)
    return model_dir


# --- the two conversion patches ----------------------------------------------


def test_patched_rotate_half_matches_the_upstream_formula() -> None:
    """The chunk-based split must equal the slice-based upstream one, bit for bit."""
    torch.manual_seed(0)
    x = torch.randn(2, _TINY_HEADS, 6, _TINY_HEAD_DIM)
    expected = _UPSTREAM_ROTATE_HALF(x)

    g3.patch_rotate_half()

    assert modeling_gemma3.rotate_half is not _UPSTREAM_ROTATE_HALF
    assert torch.equal(modeling_gemma3.rotate_half(x), expected)


def test_patched_repeat_kv_reproduces_the_upstream_head_order() -> None:
    """The rank-4 expansion must place every repeated head where upstream puts it."""
    torch.manual_seed(1)
    hidden_states = torch.randn(2, _TINY_KV_HEADS, 3, _TINY_HEAD_DIM)
    n_rep = _TINY_HEADS // _TINY_KV_HEADS
    expected = _UPSTREAM_REPEAT_KV(hidden_states, n_rep)

    g3.patch_repeat_kv()
    patched = modeling_gemma3.repeat_kv(hidden_states, n_rep)

    assert patched.shape == expected.shape
    assert patched.dim() == 4
    assert torch.equal(patched, expected)


def test_patched_repeat_kv_returns_the_input_when_nothing_is_shared() -> None:
    """``n_rep == 1`` must short-circuit to the very input tensor, as upstream does."""
    torch.manual_seed(2)
    hidden_states = torch.randn(2, 3, 4, 5)

    g3.patch_repeat_kv()

    assert modeling_gemma3.repeat_kv(hidden_states, 1) is hidden_states


def test_apply_patches_rebinds_both_module_functions() -> None:
    """The record must describe rewrites that really happened."""
    backend = g3.Gemma3Backend()

    backend.apply_patches(_loaded(_tiny_model()))

    assert modeling_gemma3.rotate_half is not _UPSTREAM_ROTATE_HALF
    assert modeling_gemma3.repeat_kv is not _UPSTREAM_REPEAT_KV


def test_apply_patches_leaves_another_decoder_family_alone() -> None:
    """Each backend rebinds the symbols of its own architecture module only."""
    g3.Gemma3Backend().apply_patches(_loaded(_tiny_model()))

    assert modeling_qwen3.rotate_half is _OTHER_FAMILY_ROTATE_HALF
    assert modeling_qwen3.repeat_kv is _OTHER_FAMILY_REPEAT_KV


def test_apply_patches_returns_a_json_serializable_record() -> None:
    """The record is stored verbatim in the artifact metadata, so it must be JSON."""
    backend = g3.Gemma3Backend()

    applied = backend.apply_patches(_loaded(_tiny_model()))

    assert applied == {
        "rotate_half_static": True,
        "repeat_kv_rank4": True,
        "mask_fill_value": g3.MASK_FILL_VALUE,
    }
    assert json.loads(json.dumps(applied)) == applied


def test_the_record_names_every_patch_of_the_patch_list() -> None:
    """A rewrite added to the list must show up in the metadata without further wiring."""
    applied = g3.Gemma3Backend().apply_patches(_loaded(_tiny_model()))

    assert [name for name, _ in g3.PATCHES] == ["rotate_half_static", "repeat_kv_rank4"]
    assert all(applied[name] is True for name, _ in g3.PATCHES)


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(hidden_size=30, num_attention_heads=2, head_dim=15),
        # No declared head dimension: it is then the width one head gets.
        SimpleNamespace(hidden_size=30, num_attention_heads=2, head_dim=None),
    ],
    ids=["declared", "derived"],
)
def test_apply_patches_rejects_an_odd_rope_head_dim(config: SimpleNamespace) -> None:
    """An odd head dimension breaks the chunk-based rotate_half and must raise."""
    backend = g3.Gemma3Backend()

    with pytest.raises(ValueError, match="head dim"):
        backend.apply_patches(_loaded(SimpleNamespace(config=config)))

    assert modeling_gemma3.rotate_half is _UPSTREAM_ROTATE_HALF
    assert modeling_gemma3.repeat_kv is _UPSTREAM_REPEAT_KV


def test_apply_patches_refuses_a_fill_value_the_wrapper_cannot_apply() -> None:
    """The graph's mask fill is fixed by the wrapper; a different one must not be claimed."""
    backend = g3.Gemma3Backend()

    with pytest.raises(ValueError, match="mask fill"):
        backend.apply_patches(_loaded(_tiny_model()), mask_fill_value=-30000.0)

    assert modeling_gemma3.rotate_half is _UPSTREAM_ROTATE_HALF


def test_apply_patches_accepts_the_fill_value_the_wrapper_uses() -> None:
    """Asking for exactly the value the graph already uses is a no-op, not an error."""
    backend = g3.Gemma3Backend()

    applied = backend.apply_patches(_loaded(_tiny_model()), mask_fill_value=g3.MASK_FILL_VALUE)

    assert applied["mask_fill_value"] == g3.MASK_FILL_VALUE


def test_the_patches_do_not_change_what_the_model_computes() -> None:
    """Both rewrites claim to be equivalent to upstream; the hidden states must show it."""
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch(
        seq_len=_SEQ_LEN_OUTSIDE_WINDOW, lengths=(_SEQ_LEN_OUTSIDE_WINDOW, 9, 3)
    )
    masks = _mask_mapping(attention_mask)

    with torch.no_grad():
        upstream = model(input_ids=input_ids, attention_mask=masks)[0]
        g3.Gemma3Backend().apply_patches(_loaded(model))
        patched = model(input_ids=input_ids, attention_mask=masks)[0]

    assert torch.allclose(patched, upstream, rtol=0, atol=1e-6)


# --- the in-graph attention masks --------------------------------------------


def test_build_bidirectional_masks_matches_the_hand_computed_layout() -> None:
    """A padded key is hidden everywhere; the window additionally hides distant keys."""
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)
    fill = -5.0

    full, sliding = g3.build_bidirectional_masks(attention_mask, window=2, fill_value=fill)

    expected_full = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.0, fill]] * 5]],
        dtype=torch.float32,
    )
    expected_sliding = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, fill, fill, fill],
                    [0.0, 0.0, 0.0, fill, fill],
                    [fill, 0.0, 0.0, 0.0, fill],
                    [fill, fill, 0.0, 0.0, fill],
                    [fill, fill, fill, 0.0, fill],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.equal(full, expected_full)
    assert torch.equal(sliding, expected_sliding)


@pytest.mark.parametrize("window", [1, 2, 3, 5])
def test_the_window_bound_is_exclusive(window: int) -> None:
    """A key exactly ``window`` positions away is out of reach; one closer is not."""
    seq_len = 8
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)

    _, sliding = g3.build_bidirectional_masks(attention_mask, window)

    for query in range(seq_len):
        for key in range(seq_len):
            visible = abs(query - key) < window
            expected = 0.0 if visible else g3.MASK_FILL_VALUE
            assert float(sliding[0, 0, query, key]) == expected, (query, key)


def test_the_window_reaches_both_ways() -> None:
    """A bidirectional window is symmetric: no key is hidden for coming later."""
    attention_mask = torch.ones(1, 9, dtype=torch.long)

    full, sliding = g3.build_bidirectional_masks(attention_mask, window=3)

    assert torch.equal(sliding, sliding.transpose(-1, -2))
    assert torch.equal(full, torch.zeros_like(full))


def test_build_bidirectional_masks_returns_two_float32_batch_masks() -> None:
    """Both masks are (B, 1, S, S) float32 and hold nothing but zero and the fill value."""
    _, attention_mask = _right_padded_batch(seq_len=12, lengths=(7, 10))

    masks = g3.build_bidirectional_masks(attention_mask, window=4)

    assert len(masks) == 2
    for mask in masks:
        assert mask.shape == (2, 1, 12, 12)
        assert mask.dtype == torch.float32
        assert set(mask.unique().tolist()) == {0.0, g3.MASK_FILL_VALUE}


def test_a_key_hidden_twice_is_filled_once() -> None:
    """Padding and distance together must not stack into twice the fill value.

    Twice the fill value would still mask the position, but it is no
    longer the number the graph was checked to stay finite with.
    """
    # The last key is padding and, for the first query, also out of reach.
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.long)

    _, sliding = g3.build_bidirectional_masks(attention_mask, window=2)

    assert float(sliding[0, 0, 0, 5]) == g3.MASK_FILL_VALUE
    assert float(sliding.min()) == g3.MASK_FILL_VALUE


def test_a_window_wider_than_the_sequence_hides_nothing_more() -> None:
    """Inside the window the sliding mask is the full mask."""
    _, attention_mask = _right_padded_batch(seq_len=6, lengths=(6, 4))

    full, sliding = g3.build_bidirectional_masks(attention_mask, window=6)

    assert torch.equal(sliding, full)


def test_build_bidirectional_masks_hides_a_padded_key_from_every_query() -> None:
    """A padded column is filled on every row, the padded rows included."""
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.long)

    full, _ = g3.build_bidirectional_masks(attention_mask, window=8)

    assert torch.equal(full[0, 0, :, 2:], torch.full((4, 2), g3.MASK_FILL_VALUE))
    assert torch.equal(full[0, 0, :, :2], torch.zeros(4, 2))
    assert torch.equal(full[1], torch.zeros(1, 4, 4))


def test_build_bidirectional_masks_accepts_a_float_mask() -> None:
    """A float mask must give the same result as the integer one."""
    int_mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.long)

    from_int = g3.build_bidirectional_masks(int_mask, window=2)
    from_float = g3.build_bidirectional_masks(int_mask.to(torch.float32), window=2)

    assert all(torch.equal(a, b) for a, b in zip(from_int, from_float, strict=True))


def test_a_row_without_any_real_token_stays_finite() -> None:
    """A fully masked row must come out as a uniform softmax, not as NaN."""
    attention_mask = torch.zeros(1, 4, dtype=torch.long)

    for mask in g3.build_bidirectional_masks(attention_mask, window=2):
        assert torch.equal(mask, torch.full((1, 1, 4, 4), g3.MASK_FILL_VALUE))
        assert bool(torch.isfinite(torch.softmax(mask.to(torch.float16), dim=-1)).all())


@pytest.mark.parametrize("window", [0, -1, True, 2.0, None])
def test_build_bidirectional_masks_rejects_an_unusable_window(window: Any) -> None:
    """A window that is not a positive integer describes no mask at all."""
    with pytest.raises(ValueError, match="window"):
        g3.build_bidirectional_masks(torch.ones(1, 4, dtype=torch.long), window)


def test_the_mask_fill_value_survives_the_precision_the_graph_runs_in() -> None:
    """A fill value that becomes -inf in FP16 would turn a masked softmax row into NaN."""
    as_half = np.float16(g3.MASK_FILL_VALUE)

    assert np.isfinite(as_half)
    assert float(as_half) == g3.MASK_FILL_VALUE
    assert float(np.exp(as_half.astype(np.float32))) == 0.0


# --- the in-graph masks against the framework's own mask path ----------------

# Real lengths per row, for a batch of each sequence length with and
# without padding.
_BATCH_LENGTHS: dict[tuple[int, bool], tuple[int, ...]] = {
    (_SEQ_LEN_INSIDE_WINDOW, False): (4, 4),
    (_SEQ_LEN_INSIDE_WINDOW, True): (4, 2, 1),
    (_SEQ_LEN_OUTSIDE_WINDOW, False): (16, 16),
    (_SEQ_LEN_OUTSIDE_WINDOW, True): (16, 9, 3),
}


def test_the_tiny_model_exercises_both_masks() -> None:
    """Guard for the comparisons below: the window and both layer types must be in play."""
    config = _tiny_model().config

    assert config.sliding_window == _TINY_WINDOW
    assert set(config.layer_types) == {g3.FULL_ATTENTION, g3.SLIDING_ATTENTION}
    assert _SEQ_LEN_INSIDE_WINDOW <= _TINY_WINDOW < _SEQ_LEN_OUTSIDE_WINDOW


@pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
@pytest.mark.parametrize(
    "seq_len",
    [_SEQ_LEN_INSIDE_WINDOW, _SEQ_LEN_OUTSIDE_WINDOW],
    ids=["inside-window", "outside-window"],
)
def test_the_mask_mapping_path_matches_the_frameworks_own_mask_path(
    seq_len: int, padded: bool
) -> None:
    """Feeding the two masks must reproduce what the 2-D mask path computes.

    Only real (unpadded) positions are compared: a padded query position
    is one nothing downstream reads, so the two paths are free to differ
    there.
    """
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch(
        seq_len=seq_len, lengths=_BATCH_LENGTHS[(seq_len, padded)]
    )

    with torch.no_grad():
        from_2d = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        from_mapping = model(input_ids=input_ids, attention_mask=_mask_mapping(attention_mask))[0]

    real = attention_mask.to(torch.bool)
    assert bool((~real).any()) is padded  # the batch is what its name says
    assert from_mapping.shape == from_2d.shape
    assert float(_row_cosines(from_mapping[real], from_2d[real]).min()) >= (
        _MASK_PATH_COSINE_THRESHOLD
    )
    pooled_mapping = common.mean_pool(from_mapping, attention_mask)
    pooled_2d = common.mean_pool(from_2d, attention_mask)
    assert float(_row_cosines(pooled_mapping, pooled_2d).min()) >= _MASK_PATH_COSINE_THRESHOLD


@pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
def test_a_single_mask_for_every_layer_is_not_this_model(padded: bool) -> None:
    """One 4-D mask shared by all layers drops the window of the sliding layers.

    That is the layout a purely causal decoder is masked with, and it is
    wrong here: beyond the window it computes a different model. The
    comparison at the short length shows that the window is the reason,
    since there the very same call agrees.
    """
    model = _tiny_model()

    def paths(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask = _right_padded_batch(
            seq_len=seq_len, lengths=_BATCH_LENGTHS[(seq_len, padded)]
        )
        full, _ = g3.build_bidirectional_masks(attention_mask, _TINY_WINDOW)
        with torch.no_grad():
            from_2d = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            from_single = model(input_ids=input_ids, attention_mask=full)[0]
        return (
            common.mean_pool(from_single, attention_mask),
            common.mean_pool(from_2d, attention_mask),
        )

    inside = _row_cosines(*paths(_SEQ_LEN_INSIDE_WINDOW))
    outside = _row_cosines(*paths(_SEQ_LEN_OUTSIDE_WINDOW))

    assert float(inside.min()) >= _MASK_PATH_COSINE_THRESHOLD
    # The first row is the full-length one, where the window matters most.
    assert float(outside[0]) < _MASK_PATH_COSINE_THRESHOLD


# --- the traceable wrapper ---------------------------------------------------


def test_wrap_selects_the_mean_pooling_wrapper() -> None:
    """A handle declaring mean pooling must be wrapped with the model's own window."""
    backend = g3.Gemma3Backend()

    wrapper = backend.wrap(_loaded(_tiny_model()))

    assert isinstance(wrapper, g3.BidirectionalMeanWrapper)
    assert wrapper.training is False
    assert wrapper.window == _TINY_WINDOW
    assert wrapper.dense is None


def test_wrap_hands_the_dense_to_the_wrapper() -> None:
    """Whatever ``load`` resolved must be what the traced module applies."""
    dense = torch.nn.Sequential(torch.nn.Linear(_TINY_HIDDEN, 8)).eval()

    wrapper = g3.Gemma3Backend().wrap(_loaded(_tiny_model(), dense=dense))

    assert wrapper.dense is dense


@pytest.mark.parametrize("pooling", [None, "cls", "lasttoken", "max", ""])
def test_wrap_rejects_a_pooling_no_wrapper_implements(pooling: str | None) -> None:
    """Only the declared mean pooling is this model; anything else must raise."""
    backend = g3.Gemma3Backend()

    with pytest.raises(ValueError, match="pooling"):
        backend.wrap(_loaded(_tiny_model(), "embedding", pooling=pooling))


@pytest.mark.parametrize("window", [None, 0, -3, True, "5"])
def test_wrap_rejects_a_configuration_without_a_usable_window(window: Any) -> None:
    """The sliding layers cannot be masked without the window the model declares."""
    model = SimpleNamespace(config=SimpleNamespace(sliding_window=window))

    with pytest.raises(ValueError, match="sliding_window"):
        g3.Gemma3Backend().wrap(_loaded(model))


def test_the_wrapper_pools_what_the_shared_helper_pools() -> None:
    """The wrapper and the FP32 baseline must average the same states the same way."""
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch()
    wrapper = g3.BidirectionalMeanWrapper(model, window=_TINY_WINDOW)

    with torch.no_grad():
        hidden = model(input_ids=input_ids, attention_mask=_mask_mapping(attention_mask))[0]
        expected = common.mean_pool(hidden, attention_mask)
        pooled = wrapper(input_ids, attention_mask)

    assert pooled.shape == (input_ids.shape[0], _TINY_HIDDEN)
    assert torch.equal(pooled, expected)


def test_the_wrapper_asks_for_tuple_outputs_itself() -> None:
    """The wrapper must not depend on the configuration having been switched to tuples."""
    model = _tiny_model()
    model.config.return_dict = True
    model.config.use_cache = True
    input_ids, attention_mask = _right_padded_batch()

    with torch.no_grad():
        pooled = g3.BidirectionalMeanWrapper(model, window=_TINY_WINDOW)(input_ids, attention_mask)

    assert pooled.shape == (input_ids.shape[0], _TINY_HIDDEN)


def test_the_wrapper_masks_with_the_window_it_was_given() -> None:
    """The window is an argument of the graph's construction, not a constant of the module."""
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch(
        seq_len=_SEQ_LEN_OUTSIDE_WINDOW, lengths=(_SEQ_LEN_OUTSIDE_WINDOW,)
    )

    with torch.no_grad():
        narrow = g3.BidirectionalMeanWrapper(model, window=_TINY_WINDOW)(input_ids, attention_mask)
        wide = g3.BidirectionalMeanWrapper(model, window=_SEQ_LEN_OUTSIDE_WINDOW)(
            input_ids, attention_mask
        )

    assert not torch.allclose(narrow, wide, atol=1e-4)


def test_the_wrapper_applies_a_declared_projection_after_pooling() -> None:
    """A declared Dense projection must be part of the graph, after the pooling."""
    model = _tiny_model()
    torch.manual_seed(7)
    dense = torch.nn.Sequential(
        torch.nn.Linear(_TINY_HIDDEN, _DENSE_WIDE, bias=False),
        torch.nn.Linear(_DENSE_WIDE, 8, bias=False),
    ).eval()
    input_ids, attention_mask = _right_padded_batch()

    with torch.no_grad():
        pooled = g3.BidirectionalMeanWrapper(model, window=_TINY_WINDOW)(input_ids, attention_mask)
        projected = g3.BidirectionalMeanWrapper(model, window=_TINY_WINDOW, dense=dense)(
            input_ids, attention_mask
        )

    assert projected.shape == (input_ids.shape[0], 8)
    assert torch.allclose(projected, dense(pooled))


def test_the_wrapper_does_not_normalize_its_output() -> None:
    """The graph always returns the raw vector; normalization is the server's choice."""
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch()

    with torch.no_grad():
        pooled = g3.BidirectionalMeanWrapper(model, window=_TINY_WINDOW)(input_ids, attention_mask)

    norms = torch.linalg.norm(pooled, dim=1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_the_wrapper_returns_one_row_per_input(batch_size: int) -> None:
    """A batch of B rows must come back as B embeddings, each computed on its own."""
    model = _tiny_model()
    lengths = (12, 7, 3)[:batch_size]
    input_ids, attention_mask = _right_padded_batch(seq_len=12, lengths=lengths)
    wrapper = g3.BidirectionalMeanWrapper(model, window=_TINY_WINDOW)

    with torch.no_grad():
        batched = wrapper(input_ids, attention_mask)
        alone = torch.cat(
            [
                wrapper(input_ids[row : row + 1], attention_mask[row : row + 1])
                for row in range(batch_size)
            ]
        )

    assert batched.shape == (batch_size, _TINY_HIDDEN)
    assert torch.allclose(batched, alone, rtol=0, atol=1e-5)


def test_the_wrapper_keeps_a_row_without_any_real_token_finite() -> None:
    """A fully masked row is never asked for, but it must not poison the batch with NaN."""
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch(seq_len=8, lengths=(8, 0))

    with torch.no_grad():
        pooled = g3.BidirectionalMeanWrapper(model, window=_TINY_WINDOW)(input_ids, attention_mask)

    assert bool(torch.isfinite(pooled).all())


def _patched_wrapper() -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    """Build the wrapper the pipeline traces, and a batch longer than the window."""
    backend = g3.Gemma3Backend()
    model = _tiny_model()
    backend.apply_patches(_loaded(model))
    wrapper = backend.wrap(_loaded(model))
    input_ids, attention_mask = _right_padded_batch(
        seq_len=_SEQ_LEN_OUTSIDE_WINDOW, lengths=(11, _SEQ_LEN_OUTSIDE_WINDOW)
    )
    return wrapper, input_ids, attention_mask


def test_the_traced_wrapper_graph_has_no_dynamic_size_node() -> None:
    """Shape arithmetic in the graph is what the conversion patches exist to remove."""
    wrapper, input_ids, attention_mask = _patched_wrapper()

    traced = conversion.trace_model(wrapper, _example(input_ids, attention_mask))

    assert _DYNAMIC_SIZE_NODE not in str(traced.inlined_graph)


def test_the_traced_wrapper_graph_holds_no_tensor_above_rank_four() -> None:
    """A rank-5 intermediate makes the Neural Engine compiler reject the attention."""
    wrapper, input_ids, attention_mask = _patched_wrapper()

    traced = conversion.trace_model(wrapper, _example(input_ids, attention_mask))

    ranks = _traced_ranks(traced)
    assert ranks  # the graph really was inspected
    assert max(ranks) <= _MAX_GRAPH_RANK


def test_the_unpatched_graph_holds_what_the_patches_remove() -> None:
    """Guard for the two graph tests above: without the patches both constructs are there."""
    model = _tiny_model()
    wrapper = g3.Gemma3Backend().wrap(_loaded(model))
    input_ids, attention_mask = _right_padded_batch(
        seq_len=_SEQ_LEN_OUTSIDE_WINDOW, lengths=(11, _SEQ_LEN_OUTSIDE_WINDOW)
    )

    traced = conversion.trace_model(wrapper, _example(input_ids, attention_mask))

    assert _DYNAMIC_SIZE_NODE in str(traced.inlined_graph)
    assert max(_traced_ranks(traced)) > _MAX_GRAPH_RANK


def test_tracing_does_not_change_what_the_wrapper_computes() -> None:
    """A traced graph that baked in the example's padding would still pass its own trace."""
    wrapper, input_ids, attention_mask = _patched_wrapper()
    traced = conversion.trace_model(wrapper, _example(input_ids, attention_mask))
    # A second batch whose rows end at different positions than the traced
    # one, so a mask or a pooling baked in at tracing time would show up.
    other_ids, other_mask = _right_padded_batch(
        seq_len=_SEQ_LEN_OUTSIDE_WINDOW, lengths=(_SEQ_LEN_OUTSIDE_WINDOW, 4), seed=8
    )

    with torch.no_grad():
        expected = wrapper(other_ids, other_mask)
        replayed = traced(other_ids, other_mask)

    assert torch.allclose(replayed, expected, atol=1e-6)


# --- interface attributes, fixtures and kind validation ----------------------


def test_backend_declares_the_interface_attributes() -> None:
    """The backend must name itself (matching its registry key) and its only kind."""
    backend = g3.Gemma3Backend()

    assert backend.name == "Gemma3Text"
    assert backend.supported_kinds == ("embedding",)


def test_the_backend_matches_the_declared_interface_signature() -> None:
    """Every protocol member must exist here with the declared parameters."""
    members = sorted(
        name
        for name, member in vars(base.CompileBackend).items()
        if not name.startswith("_") and callable(member)
    )

    for name in members:
        implemented = getattr(g3.Gemma3Backend, name, None)
        assert implemented is not None, f"Gemma3Backend does not implement {name}()"
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
    assert g3.Gemma3Backend().output_name("embedding") == "embedding"


def test_the_backend_serves_the_shared_sanity_sets() -> None:
    """Nothing about this family calls for fixtures of its own."""
    spec = g3.Gemma3Backend().sanity_spec("embedding")

    assert spec.input_sets == common.SANITY_TEXT_SETS
    assert spec.languages == ("en", "ja", "zh")
    assert spec.relevant_index is None
    assert spec.irrelevant_index is None


def test_the_fixtures_are_non_empty_single_texts() -> None:
    """A fully masked row says nothing about the model, so no fixture may be empty."""
    backend = g3.Gemma3Backend()

    assert isinstance(backend.trace_example("embedding"), str)
    assert backend.trace_example("embedding")
    assert isinstance(backend.padding_input("embedding"), str)
    assert backend.padding_input("embedding")
    assert all(
        isinstance(text, str) and text for text in backend.sanity_spec("embedding").all_inputs
    )


def test_pair_template_of_an_embedding_model_is_none(tmp_path: Path) -> None:
    """An embedding model encodes one text and has no pair to shape."""
    assert g3.Gemma3Backend().pair_template(tmp_path, "embedding") is None


def test_the_reranker_score_space_follows_the_embedding_only_precedent() -> None:
    """A backend without a reranker answers as the other embedding-only backend does."""
    assert g3.Gemma3Backend().reranker_score_space() == base.SCORE_SPACE_PROBABILITY


@pytest.mark.parametrize(
    "method",
    ["trace_example", "sanity_spec", "padding_input", "output_name"],
)
@pytest.mark.parametrize("kind", ["reranker", "classifier", "generation"])
def test_unknown_kind_is_rejected(method: str, kind: str) -> None:
    """Every kind-dispatching method must reject a kind this backend cannot compile."""
    backend = g3.Gemma3Backend()

    with pytest.raises(ValueError, match="kind"):
        getattr(backend, method)(kind)


@pytest.mark.parametrize("kind", ["reranker", "classifier"])
def test_pair_template_rejects_an_unsupported_kind(tmp_path: Path, kind: str) -> None:
    """A kind this backend cannot compile must be refused, not answered with None."""
    with pytest.raises(ValueError, match="kind"):
        g3.Gemma3Backend().pair_template(tmp_path, kind)


@pytest.mark.parametrize("kind", ["reranker", "classifier"])
def test_load_rejects_an_unsupported_kind(tmp_path: Path, kind: str) -> None:
    """load must validate the kind before touching the filesystem."""
    with pytest.raises(ValueError, match="kind"):
        g3.Gemma3Backend().load(tmp_path, kind)


def test_a_reranker_is_refused_with_the_reason(tmp_path: Path) -> None:
    """Asking for a reranker must say what this backend compiles instead."""
    with pytest.raises(ValueError) as excinfo:
        g3.Gemma3Backend().load(_model_dir(tmp_path, pooling_mode_mean_tokens=True), "reranker")

    message = str(excinfo.value)
    assert "reranker" in message
    assert "bidirectional" in message
    assert "embedding" in message


@pytest.mark.parametrize("method", ["wrap", "tokenize"])
@pytest.mark.parametrize("kind", ["reranker", "classifier"])
def test_handle_taking_methods_reject_an_unsupported_kind(method: str, kind: str) -> None:
    """A handle carrying an unsupported kind must be rejected, not silently wrapped."""
    backend = g3.Gemma3Backend()
    loaded = _loaded(_tiny_model(), kind=kind)
    arguments = {"tokenize": (["text"], 8)}.get(method, ())

    with pytest.raises(ValueError, match="kind"):
        getattr(backend, method)(loaded, *arguments)


def test_reference_outputs_rejects_an_unsupported_kind(tmp_path: Path) -> None:
    """The reference path must validate the kind before loading any weights."""
    with pytest.raises(ValueError, match="kind"):
        g3.Gemma3Backend().reference_outputs(tmp_path, "reranker", [("q", "d")], 8)


def test_reference_outputs_rejects_empty_inputs(tmp_path: Path) -> None:
    """No inputs means nothing to compare; that must raise before loading weights."""
    with pytest.raises(ValueError, match="inputs"):
        g3.Gemma3Backend().reference_outputs(tmp_path, "embedding", [], 8)


# --- refusing a directory this backend does not implement --------------------
#
# None of the directories below holds any weight, so an error that names
# the declaration proves it was decided before a checkpoint was looked for.


@pytest.mark.parametrize("declared", ["gemma3", "gemma3n_text", "gemma2", "gemma", "qwen3", None])
def test_load_rejects_a_directory_of_another_model_type(
    tmp_path: Path, declared: str | None
) -> None:
    """A related architecture must be refused before any weight is read."""
    model_dir = _model_dir(
        tmp_path, config=_declared_config(model_type=declared), pooling_mode_mean_tokens=True
    )

    with pytest.raises(ValueError) as excinfo:
        g3.Gemma3Backend().load(model_dir, "embedding")

    message = str(excinfo.value)
    assert "model_type" in message
    assert g3.MODEL_TYPE in message
    assert repr(declared) in message


def test_load_rejects_a_directory_without_a_readable_config(tmp_path: Path) -> None:
    """No configuration means no declared architecture, which is a refusal of its own."""
    with pytest.raises(ValueError, match="model_type"):
        g3.Gemma3Backend().load(tmp_path / "absent", "embedding")


@pytest.mark.parametrize("declared", [False, None, "true", 1, 0])
def test_load_rejects_a_model_that_does_not_attend_both_ways(tmp_path: Path, declared: Any) -> None:
    """A causal Gemma 3 text model is a different model and must be refused as one.

    Only a literal ``true`` counts: anything else is a declaration this
    backend cannot claim to understand.
    """
    config = _declared_config()
    if declared is None:
        del config["use_bidirectional_attention"]
    else:
        config["use_bidirectional_attention"] = declared
    model_dir = _model_dir(tmp_path, config=config, pooling_mode_mean_tokens=True)

    with pytest.raises(ValueError) as excinfo:
        g3.Gemma3Backend().load(model_dir, "embedding")

    message = str(excinfo.value)
    assert "use_bidirectional_attention" in message
    assert "bidirectional" in message


def test_the_architecture_is_refused_before_the_attention_direction(tmp_path: Path) -> None:
    """The more fundamental refusal comes first, so the message names the real problem."""
    model_dir = _model_dir(
        tmp_path,
        config=_declared_config(model_type="gemma3", use_bidirectional_attention=False),
    )

    with pytest.raises(ValueError, match="model_type"):
        g3.Gemma3Backend().load(model_dir, "embedding")


def test_load_reports_an_undeclared_pooling_before_loading_weights(tmp_path: Path) -> None:
    """An embedding model without a pooling declaration must fail fast and clearly."""
    model_dir = _model_dir(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        g3.Gemma3Backend().load(model_dir, "embedding")

    message = str(excinfo.value)
    assert common.POOLING_DIRNAME in message
    assert "pooling_mode_mean_tokens" in message


def test_load_refuses_an_unreproducible_chain_before_reading_any_weight(tmp_path: Path) -> None:
    """A declared module this backend cannot reproduce must be refused from the declaration."""
    model_dir = _model_dir(tmp_path, pooling_mode_mean_tokens=True)
    _write_json(
        model_dir / common.ST_MODULES_FILENAME,
        [
            {"idx": 0, "name": "0", "path": "", "type": common.ST_MODULE_TRANSFORMER},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": common.ST_MODULE_POOLING},
            {"idx": 2, "name": "2", "path": "2_LSTM", "type": "sentence_transformers.models.LSTM"},
        ],
    )

    with pytest.raises(ValueError, match="modules.json"):
        g3.Gemma3Backend().load(model_dir, "embedding")


def test_reference_outputs_refuses_what_load_refuses(tmp_path: Path) -> None:
    """Whichever side of the self-check reaches a directory first, it fails the same way."""
    model_dir = _model_dir(tmp_path, config=_declared_config(use_bidirectional_attention=False))

    with pytest.raises(ValueError, match="use_bidirectional_attention"):
        g3.Gemma3Backend().reference_outputs(model_dir, "embedding", ["text"], 8)


# --- effective maximum sequence length ---------------------------------------


@pytest.mark.parametrize("configured", [512, 2048, 1])
def test_max_seq_len_reports_the_whole_position_budget(tmp_path: Path, configured: int) -> None:
    """Rotary positions reserve no leading slot, so nothing may be subtracted."""
    model_dir = _model_dir(tmp_path, config=_declared_config(max_position_embeddings=configured))

    assert g3.Gemma3Backend().max_seq_len(model_dir) == configured


def test_max_seq_len_is_not_the_attention_window(tmp_path: Path) -> None:
    """The window limits what a sliding layer sees, not how long a sequence may be."""
    model_dir = _model_dir(
        tmp_path, config=_declared_config(max_position_embeddings=2048, sliding_window=512)
    )

    assert g3.Gemma3Backend().max_seq_len(model_dir) == 2048


def test_max_seq_len_of_a_missing_directory_is_none(tmp_path: Path) -> None:
    """No config.json means no known limit, not a crash."""
    assert g3.Gemma3Backend().max_seq_len(tmp_path / "absent") is None


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "gemma3_text"},
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
    assert g3.Gemma3Backend().max_seq_len(_model_dir(tmp_path, config=config)) is None


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"], ids=["corrupt", "not-an-object"])
def test_max_seq_len_of_an_unusable_config_is_none(tmp_path: Path, content: str) -> None:
    """An unparsable config must not turn an optional check into a compile failure."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / g3.CONFIG_FILENAME).write_text(content, encoding="utf-8")

    assert g3.Gemma3Backend().max_seq_len(model_dir) is None


# --- round trip through a synthetic saved model directory --------------------


def _write_tokenizer_files(directory: Path, model_max_length: int = _TINY_POSITIONS) -> None:
    """Save a byte-level toy tokenizer into a synthetic model directory.

    Byte-level vocabulary with no merges: every byte is its own token, so
    the multilingual fixtures tokenize without shipping a real vocab file.
    The template wraps the text in the begin- and end-of-sequence tokens,
    as this family's own tokenizers do, and the padding goes on the right.

    Args:
        directory: Model directory the tokenizer files are written to.
        model_max_length: Length the saved tokenizer declares; only used
            to keep its own warnings quiet, since every encoding here is
            truncated by the caller.
    """
    vocab = {"<pad>": _PAD_ID, "<eos>": _EOS_ID, "<bos>": _BOS_ID, "<unk>": 3}
    first_byte_id = len(vocab)
    for index, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet())):
        vocab[character] = index + first_byte_id
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B <eos>",
        special_tokens=[("<bos>", _BOS_ID), ("<eos>", _EOS_ID)],
    )
    tokenizer.decoder = decoders.ByteLevel()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        padding_side="right",
        model_max_length=model_max_length,
    ).save_pretrained(directory)


def _write_dense_module(
    directory: Path, name: str, in_features: int, out_features: int, seed: int
) -> None:
    """Write one bias-free Dense module with an identity activation.

    Args:
        directory: Model directory the module lives under.
        name: Module directory name, as declared in ``modules.json``.
        in_features: Declared input width.
        out_features: Declared output width.
        seed: Seed of the random weights, so the two stages differ.
    """
    module_dir = directory / name
    _write_json(
        module_dir / common.DENSE_CONFIG_FILENAME,
        {
            common.DENSE_IN_FEATURES_KEY: in_features,
            common.DENSE_OUT_FEATURES_KEY: out_features,
            common.DENSE_BIAS_FLAG_KEY: False,
            common.DENSE_ACTIVATION_KEY: "torch.nn.modules.linear.Identity",
        },
    )
    generator = torch.Generator().manual_seed(seed)
    weight = torch.rand(out_features, in_features, generator=generator) - 0.5
    save_file({common.DENSE_WEIGHT_KEY: weight}, str(module_dir / common.DENSE_WEIGHTS_FILENAME))


def _write_model_directory(directory: Path) -> Path:
    """Save a tiny randomly initialised embedding model as a HuggingFace directory.

    The directory declares what a published model of this family does: a
    mean pooling, two bias-free Dense projections and a trailing
    normalization.

    Args:
        directory: Destination directory; created if needed.

    Returns:
        ``directory``, holding weights, config, tokenizer files and the
        sentence-transformers declarations.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _tiny_model().save_pretrained(directory)
    # Saving writes the window the model sees, which loading would derive
    # a narrower one from yet again. A published configuration stores the
    # value the derivation starts from, so that is what is put back.
    config_path = directory / g3.CONFIG_FILENAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["sliding_window"] = _TINY_STORED_WINDOW
    _write_json(config_path, config)
    _write_tokenizer_files(directory)

    _write_json(
        directory / common.POOLING_DIRNAME / common.POOLING_CONFIG_FILENAME,
        {"word_embedding_dimension": _TINY_HIDDEN, "pooling_mode_mean_tokens": True},
    )
    _write_dense_module(directory, "2_Dense", _TINY_HIDDEN, _DENSE_WIDE, seed=3)
    _write_dense_module(directory, "3_Dense", _DENSE_WIDE, _DENSE_OUT, seed=4)
    _write_json(
        directory / common.ST_MODULES_FILENAME,
        [
            {"idx": 0, "name": "0", "path": "", "type": common.ST_MODULE_TRANSFORMER},
            {"idx": 1, "name": "1", "path": "1_Pooling", "type": common.ST_MODULE_POOLING},
            {"idx": 2, "name": "2", "path": "2_Dense", "type": common.ST_MODULE_DENSE},
            {"idx": 3, "name": "3", "path": "3_Dense", "type": common.ST_MODULE_DENSE},
            {"idx": 4, "name": "4", "path": "4_Normalize", "type": common.ST_MODULE_NORMALIZE},
        ],
    )
    return directory


@pytest.fixture(scope="module")
def embedding_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic Gemma 3 embedding model directory with two Dense projections."""
    return _write_model_directory(tmp_path_factory.mktemp("gemma3-embedding"))


def test_load_returns_a_conforming_handle(embedding_dir: Path) -> None:
    """load must hand back an eval/FP32/tuple-output model plus its tokenizer and config."""
    loaded = g3.Gemma3Backend().load(embedding_dir, "embedding")

    assert loaded.kind == "embedding"
    assert loaded.attn == "eager"
    assert loaded.pooling == common.POOLING_MEAN
    assert loaded.model_dir == embedding_dir
    assert isinstance(loaded.model, Gemma3TextModel)
    assert loaded.model.training is False
    assert next(loaded.model.parameters()).dtype == torch.float32
    assert loaded.config.return_dict is False
    assert loaded.config.use_cache is False
    assert loaded.config is loaded.model.config
    assert loaded.config._attn_implementation == "eager"
    assert loaded.tokenizer("x")["input_ids"]


def test_load_carries_the_declared_projections_and_their_record(embedding_dir: Path) -> None:
    """Both Dense stages must be resolved, in order, and described for the metadata."""
    loaded = g3.Gemma3Backend().load(embedding_dir, "embedding")

    assert isinstance(loaded.dense, torch.nn.Module)
    assert loaded.dense_config == (
        {"in": _TINY_HIDDEN, "out": _DENSE_WIDE, "bias": False, "activation": "identity"},
        {"in": _DENSE_WIDE, "out": _DENSE_OUT, "bias": False, "activation": "identity"},
    )


def test_the_window_is_the_one_the_loaded_configuration_holds(embedding_dir: Path) -> None:
    """The configuration already derived the window; deriving it again would halve it twice."""
    backend = g3.Gemma3Backend()
    loaded = backend.load(embedding_dir, "embedding")
    stored = json.loads((embedding_dir / g3.CONFIG_FILENAME).read_text(encoding="utf-8"))

    assert stored["sliding_window"] == _TINY_STORED_WINDOW
    assert loaded.config.sliding_window == _TINY_WINDOW
    assert backend.wrap(loaded).window == _TINY_WINDOW


def test_tokenize_returns_fixed_shape_int32_arrays(embedding_dir: Path) -> None:
    """Tokenized Core ML inputs must be (N, S) int32 with only the two graph keys."""
    backend = g3.Gemma3Backend()
    loaded = backend.load(embedding_dir, "embedding")
    inputs = ["a short text", "a considerably longer text than the first one"]

    tokens = backend.tokenize(loaded, inputs, _ROUND_TRIP_SEQ_LEN)

    assert set(tokens) == {"input_ids", "attention_mask"}
    for key in ("input_ids", "attention_mask"):
        assert tokens[key].dtype == np.int32
        assert tokens[key].shape == (len(inputs), _ROUND_TRIP_SEQ_LEN)
    # The first row really is padded, and on the right.
    assert tokens["attention_mask"][0, -1] == 0
    assert tokens["attention_mask"][0, 0] == 1


def test_tokenize_leaves_the_special_tokens_to_the_tokenizer(embedding_dir: Path) -> None:
    """The begin and end markers come from the tokenizer's own template, exactly once."""
    backend = g3.Gemma3Backend()
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
    backend = g3.Gemma3Backend()
    loaded = backend.load(embedding_dir, "embedding")

    tokens = backend.tokenize(loaded, [backend.padding_input("embedding")], _ROUND_TRIP_SEQ_LEN)

    assert tokens["attention_mask"].sum() > 0


def test_tokenize_rejects_empty_inputs(embedding_dir: Path) -> None:
    """An empty batch would produce a zero-row graph input and must raise."""
    backend = g3.Gemma3Backend()
    loaded = backend.load(embedding_dir, "embedding")

    with pytest.raises(ValueError, match="inputs"):
        backend.tokenize(loaded, [], _ROUND_TRIP_SEQ_LEN)


@pytest.mark.parametrize("seq_len", [0, -1])
def test_tokenize_rejects_a_non_positive_sequence_length(embedding_dir: Path, seq_len: int) -> None:
    """A non-positive fixed length is never a valid graph shape."""
    backend = g3.Gemma3Backend()
    loaded = backend.load(embedding_dir, "embedding")

    with pytest.raises(ValueError, match="seq_len"):
        backend.tokenize(loaded, ["abc"], seq_len)


def test_reference_outputs_loads_a_copy_of_its_own_on_the_other_attention_path(
    embedding_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The baseline must not reuse the implementation the conversion rewrites."""
    requested: list[str] = []
    original = g3.Gemma3Backend.load

    def spy(self: Any, model_dir: Path, kind: str, attn: str = "eager") -> base.LoadedModel:
        requested.append(attn)
        return original(self, model_dir, kind, attn=attn)

    monkeypatch.setattr(g3.Gemma3Backend, "load", spy)

    g3.Gemma3Backend().reference_outputs(embedding_dir, "embedding", ["text"], 8)

    assert requested == ["sdpa"]


def test_reference_outputs_refuses_a_pooling_the_wrapper_would_refuse(tmp_path: Path) -> None:
    """The baseline must not quietly pool another way than the traced module can."""
    model_dir = _write_model_directory(tmp_path / "cls")
    _write_json(
        model_dir / common.POOLING_DIRNAME / common.POOLING_CONFIG_FILENAME,
        {"word_embedding_dimension": _TINY_HIDDEN, "pooling_mode_cls_token": True},
    )

    with pytest.raises(ValueError, match="pooling"):
        g3.Gemma3Backend().reference_outputs(model_dir, "embedding", ["text"], 8)


@pytest.fixture(scope="module")
def round_trip(embedding_dir: Path) -> Iterator[dict[str, Any]]:
    """Run the patched wrapper and the FP32 sdpa reference over the sanity fixtures."""
    backend = g3.Gemma3Backend()
    inputs = list(backend.sanity_spec("embedding").all_inputs)
    loaded = backend.load(embedding_dir, "embedding")
    try:
        backend.apply_patches(loaded)
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
        modeling_gemma3.rotate_half = _UPSTREAM_ROTATE_HALF
        modeling_gemma3.repeat_kv = _UPSTREAM_REPEAT_KV
    del loaded, wrapper
    gc.collect()
    yield {
        "inputs": inputs,
        "tokens": tokens,
        "wrapped": wrapped.numpy().reshape(len(inputs), -1),
        "reference": np.asarray(reference, dtype=np.float32).reshape(len(inputs), -1),
    }


def test_the_wrapper_matches_the_fp32_reference_on_padded_rows(embedding_dir: Path) -> None:
    """A batch mixing a short and a long row must agree with the baseline row by row.

    The byte-level tokenizer fills every sanity fixture up to the fixed
    length, so the comparison over those never pads; this one does, with
    one row inside the window and one beyond it.
    """
    backend = g3.Gemma3Backend()
    inputs = ["ab", "a text long enough to fill the whole row"]
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
    assert real_lengths[0] <= _TINY_WINDOW < real_lengths[1] == _ROUND_TRIP_SEQ_LEN
    np.testing.assert_allclose(wrapped, reference, rtol=0, atol=_ROUND_TRIP_TOLERANCE)


def test_the_round_trip_runs_beyond_the_window(round_trip: dict[str, Any]) -> None:
    """Guard for the comparison below: its rows must be longer than the window."""
    real_lengths = round_trip["tokens"]["attention_mask"].sum(axis=1)

    assert int(real_lengths.max()) > _TINY_WINDOW


def test_the_wrapper_matches_the_fp32_reference(round_trip: dict[str, Any]) -> None:
    """The traced module and the baseline must compute the same function.

    The two sides reach it by different routes -- the wrapper builds its
    own two masks over the patched eager attention, the baseline lets the
    framework build them for the sdpa attention -- so a disagreement about
    the masks, the pooling, the projections or the tokenization shows up
    here rather than only against real weights.
    """
    assert np.isfinite(round_trip["wrapped"]).all()
    assert np.isfinite(round_trip["reference"]).all()
    np.testing.assert_allclose(
        round_trip["wrapped"],
        round_trip["reference"],
        rtol=0,
        atol=_ROUND_TRIP_TOLERANCE,
    )


def test_the_reference_distinguishes_the_sanity_fixtures(round_trip: dict[str, Any]) -> None:
    """Guard for the comparison above: identical rows would make it prove nothing."""
    reference = round_trip["reference"]

    assert reference.shape == (len(round_trip["inputs"]), _DENSE_OUT)
    assert reference.dtype == np.float32
    for row in range(1, reference.shape[0]):
        assert not np.allclose(reference[0], reference[row], atol=_ROUND_TRIP_TOLERANCE)


# --- Core ML conversion of the traced wrapper --------------------------------


def test_the_traced_wrapper_converts_and_agrees_with_the_fp32_wrapper(
    embedding_dir: Path,
) -> None:
    """The conversion must produce a program that still computes the same embedding."""
    backend = g3.Gemma3Backend()
    loaded = backend.load(embedding_dir, "embedding")
    backend.apply_patches(loaded)
    wrapper = backend.wrap(loaded)
    inputs = list(backend.sanity_spec("embedding").input_sets[0][1])
    tokens = backend.tokenize(loaded, inputs, _ROUND_TRIP_SEQ_LEN)
    batch_size = len(inputs)
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

    description = mlmodel.get_spec().description
    assert [tensor.name for tensor in description.input] == ["input_ids", "attention_mask"]
    assert [tensor.name for tensor in description.output] == ["embedding"]
    assert embeddings.shape == expected.shape == (batch_size, _DENSE_OUT)
    assert bool(np.isfinite(embeddings).all())
    cosines = np.sum(embeddings * expected, axis=1) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(expected, axis=1)
    )
    assert float(cosines.min()) >= _CONVERSION_COSINE_THRESHOLD
