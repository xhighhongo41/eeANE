"""Tests for the Qwen3 compile backend.

Four layers:

* The two conversion patches, checked against the upstream implementations
  they replace: both must compute the same values, or the compiled model
  would no longer be the model the reference computes.
* The in-graph attention mask and the last-token pooling, checked against
  the framework's own mask path and against hand-written expectations.
* Conformance to the backend interface declared in
  ``eeane.compiler.backends.base``: the kind validation, the fixtures, the
  refusal of a model this backend does not implement, and the round trip
  from ``load`` to the FP32 reference on a synthetic model directory.
* The Core ML conversion of the traced wrapper, on the same tiny synthetic
  model.

Nothing here downloads weights: every model is a small randomly
initialised Qwen3 built from an in-test configuration.
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
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from eeane.compiler import conversion
from eeane.compiler.backends import base, common
from eeane.compiler.backends import qwen3 as q3

# The upstream implementations, captured at import time (that is, before
# any test can have replaced them). They are both the comparison baseline
# for the patch tests and the value the autouse fixture restores.
_UPSTREAM_ROTATE_HALF = modeling_qwen3.rotate_half
_UPSTREAM_REPEAT_KV = modeling_qwen3.repeat_kv

# Geometry of the tiny model used throughout: two grouped query heads per
# key/value head, so ``repeat_kv`` really has to expand something, and an
# even head dimension, which ``rotate_half`` requires.
_TINY_HIDDEN = 64
_TINY_HEADS = 4
_TINY_KV_HEADS = 2
_TINY_HEAD_DIM = 16
_TINY_LAYERS = 2
_TINY_INTERMEDIATE = 128
_TINY_POSITIONS = 128

# Vocabulary of the synthetic model directory: the byte-level tokenizer
# below emits up to ~260 distinct ids (256 bytes plus its special tokens),
# so every id it can produce stays addressable.
_TINY_VOCAB = 300

# Sequence length of the round-trip and conversion tests: short enough to
# leave the sanity fixtures visibly truncated and padded.
_ROUND_TRIP_SEQ_LEN = 16

# Tolerance between the traced wrapper (patched eager, in-graph 4-D mask)
# and the FP32 baseline (untouched sdpa, framework mask). What is left is
# the kernel difference between the two attention implementations.
_ROUND_TRIP_TOLERANCE = 1e-5

# Minimum cosine between the converted FP16 program and the FP32 wrapper.
_CONVERSION_COSINE_THRESHOLD = 0.99

# Graph node the Core ML converter cannot fold into a static slice; it is
# what a Python-level division or index on a traced shape lowers to.
_DYNAMIC_SIZE_NODE = "aten::Int"


@pytest.fixture(autouse=True)
def _restore_transformers_patches() -> Iterator[None]:
    """Undo the process-wide Qwen3 monkeypatches after every test.

    ``patch_rotate_half`` and ``patch_repeat_kv`` rebind names in the
    transformers module, which affects every Qwen3 model in the process,
    so each test starts and ends on the upstream implementations.
    """
    modeling_qwen3.rotate_half = _UPSTREAM_ROTATE_HALF
    modeling_qwen3.repeat_kv = _UPSTREAM_REPEAT_KV
    try:
        yield
    finally:
        modeling_qwen3.rotate_half = _UPSTREAM_ROTATE_HALF
        modeling_qwen3.repeat_kv = _UPSTREAM_REPEAT_KV


def _tiny_config(vocab_size: int = _TINY_VOCAB) -> Qwen3Config:
    """Build the configuration of the tiny model used by these tests."""
    config = Qwen3Config(
        hidden_size=_TINY_HIDDEN,
        num_hidden_layers=_TINY_LAYERS,
        num_attention_heads=_TINY_HEADS,
        num_key_value_heads=_TINY_KV_HEADS,
        head_dim=_TINY_HEAD_DIM,
        intermediate_size=_TINY_INTERMEDIATE,
        vocab_size=vocab_size,
        max_position_embeddings=_TINY_POSITIONS,
        attn_implementation="eager",
    )
    config.use_cache = False
    config.return_dict = False
    return config


def _tiny_model(seed: int = 0, vocab_size: int = _TINY_VOCAB) -> Qwen3Model:
    """Build a small randomly initialised Qwen3 backbone in eval mode."""
    torch.manual_seed(seed)
    return Qwen3Model(_tiny_config(vocab_size)).eval()


def _right_padded_batch(
    seq_len: int = 12, lengths: tuple[int, ...] = (7, 10), seed: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a right-padded batch whose rows have different real lengths."""
    torch.manual_seed(seed)
    input_ids = torch.randint(1, _TINY_VOCAB, (len(lengths), seq_len))
    attention_mask = torch.ones(len(lengths), seq_len, dtype=torch.long)
    for row, length in enumerate(lengths):
        attention_mask[row, length:] = 0
        input_ids[row, length:] = 0
    return input_ids, attention_mask


def _loaded(
    model: Any,
    kind: str = "embedding",
    pooling: str | None = common.POOLING_LASTTOKEN,
    tokenizer: Any = None,
    model_dir: Path = Path("/nonexistent-model-dir"),
) -> base.LoadedModel:
    """Build the handle the backend interface passes between its stages."""
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
    _write_json(
        model_dir / q3.CONFIG_FILENAME,
        config if config is not None else {"model_type": q3.MODEL_TYPE},
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

    q3.patch_rotate_half()

    assert modeling_qwen3.rotate_half is not _UPSTREAM_ROTATE_HALF
    assert torch.equal(modeling_qwen3.rotate_half(x), expected)


def test_patched_repeat_kv_reproduces_the_upstream_head_order() -> None:
    """The rank-4 expansion must place every repeated head where upstream puts it."""
    torch.manual_seed(1)
    hidden_states = torch.randn(2, _TINY_KV_HEADS, 3, _TINY_HEAD_DIM)
    n_rep = _TINY_HEADS // _TINY_KV_HEADS
    expected = _UPSTREAM_REPEAT_KV(hidden_states, n_rep)

    q3.patch_repeat_kv()
    patched = modeling_qwen3.repeat_kv(hidden_states, n_rep)

    assert patched.shape == expected.shape
    assert patched.dim() == 4
    assert torch.equal(patched, expected)


def test_patched_repeat_kv_returns_the_input_when_nothing_is_shared() -> None:
    """``n_rep == 1`` must short-circuit to the very input tensor, as upstream does."""
    torch.manual_seed(2)
    hidden_states = torch.randn(2, 3, 4, 5)

    q3.patch_repeat_kv()

    assert modeling_qwen3.repeat_kv(hidden_states, 1) is hidden_states


def test_apply_patches_rebinds_both_module_functions() -> None:
    """The record must describe rewrites that really happened."""
    backend = q3.Qwen3Backend()

    backend.apply_patches(_loaded(_tiny_model()))

    assert modeling_qwen3.rotate_half is not _UPSTREAM_ROTATE_HALF
    assert modeling_qwen3.repeat_kv is not _UPSTREAM_REPEAT_KV


def test_apply_patches_returns_a_json_serializable_record() -> None:
    """The record is stored verbatim in the artifact metadata, so it must be JSON."""
    backend = q3.Qwen3Backend()

    applied = backend.apply_patches(_loaded(_tiny_model()))

    assert applied == {
        "rotate_half_static": True,
        "repeat_kv_rank4": True,
        "mask_fill_value": q3.MASK_FILL_VALUE,
    }
    assert json.loads(json.dumps(applied)) == applied


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
    backend = q3.Qwen3Backend()

    with pytest.raises(ValueError, match="head dim"):
        backend.apply_patches(_loaded(SimpleNamespace(config=config)))

    assert modeling_qwen3.rotate_half is _UPSTREAM_ROTATE_HALF


def test_apply_patches_reads_the_declared_head_dimension() -> None:
    """The declared head dimension is not always ``hidden_size / heads`` in this family."""
    backend = q3.Qwen3Backend()
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=30, num_attention_heads=2, head_dim=16)
    )

    applied = backend.apply_patches(_loaded(model))

    assert applied["rotate_half_static"] is True


def test_apply_patches_refuses_a_fill_value_the_wrapper_cannot_apply() -> None:
    """The graph's mask fill is fixed by the wrapper; a different one must not be claimed."""
    backend = q3.Qwen3Backend()

    with pytest.raises(ValueError, match="mask fill"):
        backend.apply_patches(_loaded(_tiny_model()), mask_fill_value=-30000.0)


def test_apply_patches_accepts_the_fill_value_the_wrapper_uses() -> None:
    """Asking for exactly the value the graph already uses is a no-op, not an error."""
    backend = q3.Qwen3Backend()

    applied = backend.apply_patches(_loaded(_tiny_model()), mask_fill_value=q3.MASK_FILL_VALUE)

    assert applied["mask_fill_value"] == q3.MASK_FILL_VALUE


def test_the_mask_fill_value_survives_the_precision_the_graph_runs_in() -> None:
    """A fill value that becomes -inf in FP16 would turn a masked softmax row into NaN."""
    as_half = np.float16(q3.MASK_FILL_VALUE)

    assert np.isfinite(as_half)
    assert float(as_half) == q3.MASK_FILL_VALUE
    assert float(np.exp(as_half.astype(np.float32))) == 0.0


# --- the in-graph attention mask ---------------------------------------------


def test_build_causal_padding_mask_matches_the_hand_computed_layout() -> None:
    """Only a causal, unpadded key position may be left attendable."""
    attention_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    fill = -5.0

    mask = q3.build_causal_padding_mask(attention_mask, fill_value=fill)

    expected = torch.tensor(
        [
            [
                [
                    [0.0, fill, fill],
                    [0.0, 0.0, fill],
                    [0.0, 0.0, fill],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    assert mask.shape == (1, 1, 3, 3)
    assert mask.dtype == torch.float32
    assert torch.equal(mask, expected)


def test_build_causal_padding_mask_defaults_to_the_finite_fill_value() -> None:
    """The default must be the finite value, not the float minimum."""
    attention_mask = torch.tensor([[1, 0]], dtype=torch.long)

    mask = q3.build_causal_padding_mask(attention_mask)

    assert torch.equal(
        mask,
        torch.tensor(
            [[[[0.0, q3.MASK_FILL_VALUE], [0.0, q3.MASK_FILL_VALUE]]]], dtype=torch.float32
        ),
    )


def test_build_causal_padding_mask_accepts_a_float_mask() -> None:
    """A float mask must give the same result as the integer one."""
    int_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.long)

    from_int = q3.build_causal_padding_mask(int_mask)
    from_float = q3.build_causal_padding_mask(int_mask.to(torch.float32))

    assert torch.equal(from_int, from_float)


def test_the_four_dimensional_mask_path_matches_the_frameworks_own_mask_path() -> None:
    """Feeding the 4-D mask must reproduce what the 2-D mask path computes.

    Only real (unpadded) positions are compared: a padded row position is
    fully masked, so its hidden state is unconstrained and the two paths
    are free to differ there.
    """
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch(seq_len=8, lengths=(8, 5, 3))

    with torch.no_grad():
        from_2d = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        from_4d = model(
            input_ids=input_ids,
            attention_mask=q3.build_causal_padding_mask(attention_mask),
        )[0]

    real = attention_mask.to(torch.bool)
    assert bool((~real).any())  # the batch really exercises padding
    assert from_4d.shape == from_2d.shape
    assert torch.equal(from_4d[real], from_2d[real])


# --- last-token pooling (eeane.compiler.backends.common) ---------------------


def test_last_token_pool_selects_the_last_real_token_of_a_right_padded_row() -> None:
    """Each row must be read at its own ``sum(mask) - 1`` position."""
    torch.manual_seed(3)
    hidden = torch.randn(2, 5, 4)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.long)

    pooled = common.last_token_pool(hidden, attention_mask)

    assert torch.equal(pooled, torch.stack([hidden[0, 2], hidden[1, 1]]))
    # A wrong implementation reading the final position would pass every
    # assertion above only if the two happened to agree; they must not.
    assert not torch.allclose(pooled, hidden[:, -1, :])


def test_last_token_pool_reads_the_final_position_of_an_unpadded_row() -> None:
    """With no padding at all the last real token is the last position."""
    torch.manual_seed(4)
    hidden = torch.randn(3, 6, 5)
    attention_mask = torch.ones(3, 6, dtype=torch.long)

    pooled = common.last_token_pool(hidden, attention_mask)

    assert torch.equal(pooled, hidden[:, -1, :])


def test_last_token_pool_clamps_a_row_without_any_real_token() -> None:
    """An empty row must read position 0 instead of wrapping around to the end."""
    torch.manual_seed(5)
    hidden = torch.randn(1, 4, 3)
    attention_mask = torch.zeros(1, 4, dtype=torch.long)

    pooled = common.last_token_pool(hidden, attention_mask)

    assert torch.equal(pooled, hidden[:, 0, :])


def test_last_token_pool_accepts_a_float_mask() -> None:
    """The mask may arrive as a float tensor; the index must stay an integer one."""
    torch.manual_seed(6)
    hidden = torch.randn(2, 4, 3)
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.float32)

    pooled = common.last_token_pool(hidden, attention_mask)

    assert torch.equal(pooled, torch.stack([hidden[0, 1], hidden[1, 2]]))


def test_encode_pytorch_pools_the_last_token_when_that_pooling_is_declared() -> None:
    """The FP32 baseline must follow the declared pooling, not fall back to CLS."""
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    class _Model:
        """Backbone stand-in returning a fixed hidden state as a tuple."""

        config = SimpleNamespace(hidden_size=2)

        def __call__(self, **_kwargs: Any) -> tuple[torch.Tensor, ...]:
            return (hidden,)

    class _Tokenizer:
        """Tokenizer stand-in producing one right-padded row."""

        def __call__(self, texts: list[str], **_kwargs: Any) -> dict[str, np.ndarray]:
            return {
                "input_ids": np.array([[7, 8, 0]] * len(texts), dtype=np.int64),
                "attention_mask": np.array([[1, 1, 0]] * len(texts), dtype=np.int64),
            }

    pooled = common.encode_pytorch(
        _Model(), _Tokenizer(), ["text"], 3, pooling=common.POOLING_LASTTOKEN
    )

    np.testing.assert_array_equal(pooled, np.array([[3.0, 4.0]], dtype=np.float32))


# --- the traceable wrapper ---------------------------------------------------


def test_wrap_selects_the_last_token_wrapper() -> None:
    """A handle declaring last-token pooling must be wrapped by that wrapper."""
    backend = q3.Qwen3Backend()

    wrapper = backend.wrap(_loaded(torch.nn.Identity()))

    assert isinstance(wrapper, q3.CausalLastTokenWrapper)
    assert wrapper.training is False


@pytest.mark.parametrize("pooling", [None, "mean", "cls", "max", ""])
def test_wrap_rejects_a_pooling_no_wrapper_implements(pooling: str | None) -> None:
    """A causal stack cannot be pooled like an encoder; that must raise, not fall back."""
    backend = q3.Qwen3Backend()

    with pytest.raises(ValueError, match="pooling"):
        backend.wrap(_loaded(torch.nn.Identity(), "embedding", pooling=pooling))


def test_the_wrapper_pools_what_the_shared_helper_pools() -> None:
    """The wrapper and the FP32 baseline must read the same position of the same state."""
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch()
    wrapper = q3.CausalLastTokenWrapper(model)

    with torch.no_grad():
        hidden = model(
            input_ids=input_ids, attention_mask=q3.build_causal_padding_mask(attention_mask)
        )[0]
        expected = common.last_token_pool(hidden, attention_mask)
        pooled = wrapper(input_ids, attention_mask)

    assert pooled.shape == (input_ids.shape[0], _TINY_HIDDEN)
    assert torch.equal(pooled, expected)


def test_the_wrapper_applies_a_declared_projection_after_pooling() -> None:
    """A declared Dense projection must be part of the graph, after the pooling."""
    model = _tiny_model()
    torch.manual_seed(7)
    dense = torch.nn.Sequential(torch.nn.Linear(_TINY_HIDDEN, 8)).eval()
    input_ids, attention_mask = _right_padded_batch()

    with torch.no_grad():
        pooled = q3.CausalLastTokenWrapper(model)(input_ids, attention_mask)
        projected = q3.CausalLastTokenWrapper(model, dense=dense)(input_ids, attention_mask)

    assert projected.shape == (input_ids.shape[0], 8)
    assert torch.allclose(projected, dense(pooled))


def test_the_wrapper_does_not_normalize_its_output() -> None:
    """The graph always returns the raw vector; normalization is the server's choice."""
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch()

    with torch.no_grad():
        pooled = q3.CausalLastTokenWrapper(model)(input_ids, attention_mask)

    norms = torch.linalg.norm(pooled, dim=1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_the_traced_wrapper_graph_has_no_dynamic_size_node() -> None:
    """Shape arithmetic in the graph is what the conversion patches exist to remove."""
    backend = q3.Qwen3Backend()
    model = _tiny_model()
    backend.apply_patches(_loaded(model))
    wrapper = backend.wrap(_loaded(model))
    input_ids, attention_mask = _right_padded_batch()
    example = {
        "input_ids": input_ids.numpy().astype(np.int32),
        "attention_mask": attention_mask.numpy().astype(np.int32),
    }

    traced = conversion.trace_model(wrapper, example)

    assert _DYNAMIC_SIZE_NODE not in str(traced.inlined_graph)


def test_tracing_does_not_change_what_the_wrapper_computes() -> None:
    """A traced graph that pooled a fixed position would still pass its own trace."""
    backend = q3.Qwen3Backend()
    model = _tiny_model()
    backend.apply_patches(_loaded(model))
    wrapper = backend.wrap(_loaded(model))
    input_ids, attention_mask = _right_padded_batch()
    example = {
        "input_ids": input_ids.numpy().astype(np.int32),
        "attention_mask": attention_mask.numpy().astype(np.int32),
    }
    traced = conversion.trace_model(wrapper, example)
    # A second batch whose rows end at different positions than the traced
    # one, so a pooling index baked in at tracing time would show up.
    other_ids, other_mask = _right_padded_batch(lengths=(4, 12), seed=8)

    with torch.no_grad():
        expected = wrapper(other_ids, other_mask)
        replayed = traced(other_ids, other_mask)

    assert torch.allclose(replayed, expected, atol=1e-6)


# --- interface attributes, fixtures and kind validation ----------------------


def test_backend_declares_the_interface_attributes() -> None:
    """The backend must name itself (matching its registry key) and its only kind."""
    backend = q3.Qwen3Backend()

    assert backend.name == "Qwen3"
    assert backend.supported_kinds == ("embedding",)


def test_the_backend_matches_the_declared_interface_signature() -> None:
    """Every protocol member must exist here with the declared parameters."""
    members = sorted(
        name
        for name, member in vars(base.CompileBackend).items()
        if not name.startswith("_") and callable(member)
    )

    for name in members:
        implemented = getattr(q3.Qwen3Backend, name, None)
        assert implemented is not None, f"Qwen3Backend does not implement {name}()"
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
    assert q3.Qwen3Backend().output_name("embedding") == "embedding"


def test_the_backend_serves_the_shared_sanity_sets() -> None:
    """Nothing about this family calls for fixtures of its own."""
    spec = q3.Qwen3Backend().sanity_spec("embedding")

    assert spec.input_sets == common.SANITY_TEXT_SETS
    assert spec.languages == ("en", "ja", "zh")
    assert spec.relevant_index is None
    assert spec.irrelevant_index is None


def test_the_fixtures_are_non_empty_single_texts() -> None:
    """A fully masked row can make the attention softmax degenerate, so none may be empty."""
    backend = q3.Qwen3Backend()

    assert isinstance(backend.trace_example("embedding"), str)
    assert backend.trace_example("embedding")
    assert isinstance(backend.padding_input("embedding"), str)
    assert backend.padding_input("embedding")
    assert all(
        isinstance(text, str) and text for text in backend.sanity_spec("embedding").all_inputs
    )


@pytest.mark.parametrize(
    "method",
    ["trace_example", "sanity_spec", "padding_input", "output_name"],
)
@pytest.mark.parametrize("kind", ["reranker", "classifier"])
def test_unknown_kind_is_rejected(method: str, kind: str) -> None:
    """Every kind-dispatching method must reject a kind this backend cannot compile."""
    backend = q3.Qwen3Backend()

    with pytest.raises(ValueError, match="kind"):
        getattr(backend, method)(kind)


def test_load_rejects_an_unsupported_kind(tmp_path: Path) -> None:
    """load must validate the kind before touching the filesystem."""
    with pytest.raises(ValueError, match="kind"):
        q3.Qwen3Backend().load(tmp_path, "reranker")


@pytest.mark.parametrize("method", ["wrap", "tokenize"])
def test_handle_taking_methods_reject_an_unsupported_kind(method: str) -> None:
    """A handle carrying an unsupported kind must be rejected, not silently wrapped."""
    backend = q3.Qwen3Backend()
    loaded = _loaded(torch.nn.Identity(), kind="reranker")
    arguments = {"tokenize": (["text"], 8)}.get(method, ())

    with pytest.raises(ValueError, match="kind"):
        getattr(backend, method)(loaded, *arguments)


def test_reference_outputs_rejects_an_unsupported_kind(tmp_path: Path) -> None:
    """The reference path must validate the kind before loading any weights."""
    with pytest.raises(ValueError, match="kind"):
        q3.Qwen3Backend().reference_outputs(tmp_path, "reranker", [("q", "d")], 8)


def test_reference_outputs_rejects_empty_inputs(tmp_path: Path) -> None:
    """No inputs means nothing to compare; that must raise before loading weights."""
    with pytest.raises(ValueError, match="inputs"):
        q3.Qwen3Backend().reference_outputs(tmp_path, "embedding", [], 8)


# --- refusing a directory this backend does not implement --------------------


@pytest.mark.parametrize("declared", ["qwen3_moe", "qwen2", "llama", None])
def test_load_rejects_a_directory_of_another_model_type(
    tmp_path: Path, declared: str | None
) -> None:
    """A closely related decoder must be refused before any weight is read.

    The backend is selected by an architecture-name prefix, which
    ``Qwen3MoeForCausalLM`` also matches; only the declared ``model_type``
    distinguishes the architecture this backend actually implements.
    """
    config: dict[str, Any] = {"architectures": ["Qwen3MoeForCausalLM"]}
    if declared is not None:
        config["model_type"] = declared
    model_dir = _model_dir(tmp_path, config=config, pooling_mode_lasttoken=True)

    with pytest.raises(ValueError) as excinfo:
        q3.Qwen3Backend().load(model_dir, "embedding")

    message = str(excinfo.value)
    assert "model_type" in message
    assert q3.MODEL_TYPE in message


def test_load_reports_an_undeclared_pooling_before_loading_weights(tmp_path: Path) -> None:
    """An embedding model without a pooling declaration must fail fast and clearly."""
    model_dir = _model_dir(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        q3.Qwen3Backend().load(model_dir, "embedding")

    message = str(excinfo.value)
    assert common.POOLING_DIRNAME in message
    assert "pooling_mode_lasttoken" in message


# --- effective maximum sequence length ---------------------------------------


@pytest.mark.parametrize("configured", [512, 32768, 1])
def test_max_seq_len_reports_the_whole_position_budget(tmp_path: Path, configured: int) -> None:
    """Rotary positions reserve no leading slot, so nothing may be subtracted."""
    model_dir = _model_dir(
        tmp_path,
        config={"model_type": q3.MODEL_TYPE, "max_position_embeddings": configured},
    )

    assert q3.Qwen3Backend().max_seq_len(model_dir) == configured


def test_max_seq_len_of_a_missing_directory_is_none(tmp_path: Path) -> None:
    """No config.json means no known limit, not a crash."""
    assert q3.Qwen3Backend().max_seq_len(tmp_path / "absent") is None


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen3"},
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
    assert q3.Qwen3Backend().max_seq_len(_model_dir(tmp_path, config=config)) is None


def test_max_seq_len_of_a_corrupt_config_is_none(tmp_path: Path) -> None:
    """Unparsable JSON must not turn an optional check into a compile failure."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / q3.CONFIG_FILENAME).write_text("{not json", encoding="utf-8")

    assert q3.Qwen3Backend().max_seq_len(model_dir) is None


# --- round trip through a synthetic saved model directory --------------------


def _write_model_directory(directory: Path) -> Path:
    """Save a tiny randomly initialised embedding model as a HuggingFace directory.

    Args:
        directory: Destination directory; created if needed.

    Returns:
        ``directory``, holding weights, config, tokenizer files and a
        pooling declaration selecting last-token pooling.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _tiny_model().save_pretrained(directory)

    # Byte-level vocabulary with no merges: every byte is its own token, so
    # the multilingual fixtures tokenize without shipping a real vocab
    # file. The template appends the end-of-text token only -- a causal
    # model has no leading classification token to prepend.
    vocab = {"<pad>": 0, "<unk>": 1, "<|endoftext|>": 2}
    for index, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet())):
        vocab[character] = index + 3
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <|endoftext|>",
        pair="$A <|endoftext|> $B <|endoftext|>",
        special_tokens=[("<|endoftext|>", 2)],
    )
    tokenizer.decoder = decoders.ByteLevel()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<|endoftext|>",
        model_max_length=_TINY_POSITIONS,
    ).save_pretrained(directory)

    _write_json(
        directory / common.POOLING_DIRNAME / common.POOLING_CONFIG_FILENAME,
        {"word_embedding_dimension": _TINY_HIDDEN, "pooling_mode_lasttoken": True},
    )
    return directory


@pytest.fixture(scope="module")
def embedding_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic Qwen3 embedding model directory declaring last-token pooling."""
    return _write_model_directory(tmp_path_factory.mktemp("qwen3-embedding"))


def test_load_returns_a_conforming_handle(embedding_dir: Path) -> None:
    """load must hand back an eval/FP32/tuple-output model plus its tokenizer and config."""
    loaded = q3.Qwen3Backend().load(embedding_dir, "embedding")

    assert loaded.kind == "embedding"
    assert loaded.attn == "eager"
    assert loaded.pooling == common.POOLING_LASTTOKEN
    assert loaded.model_dir == embedding_dir
    assert loaded.model.training is False
    assert next(loaded.model.parameters()).dtype == torch.float32
    assert loaded.config.return_dict is False
    assert loaded.config is loaded.model.config
    assert loaded.tokenizer("x")["input_ids"]


def test_load_disables_the_key_value_cache(embedding_dir: Path) -> None:
    """A cache left enabled would put its own update into the traced graph."""
    loaded = q3.Qwen3Backend().load(embedding_dir, "embedding")

    assert loaded.config.use_cache is False


def test_tokenize_returns_fixed_shape_int32_arrays(embedding_dir: Path) -> None:
    """Tokenized Core ML inputs must be (N, S) int32 with only the two graph keys."""
    backend = q3.Qwen3Backend()
    loaded = backend.load(embedding_dir, "embedding")
    inputs = ["a short text", "a considerably longer text than the first one"]

    tokens = backend.tokenize(loaded, inputs, _ROUND_TRIP_SEQ_LEN)

    assert set(tokens) == {"input_ids", "attention_mask"}
    for key in ("input_ids", "attention_mask"):
        assert tokens[key].dtype == np.int32
        assert tokens[key].shape == (len(inputs), _ROUND_TRIP_SEQ_LEN)
    assert tokens["attention_mask"].sum() > 0
    # The rows really are padded, and on the right, which is what the
    # last-token pooling assumes.
    assert tokens["attention_mask"][0, -1] == 0
    assert tokens["attention_mask"][0, 0] == 1


def test_the_padding_input_encodes_to_a_non_empty_mask(embedding_dir: Path) -> None:
    """A filler row that masked out everything could make the softmax produce NaN."""
    backend = q3.Qwen3Backend()
    loaded = backend.load(embedding_dir, "embedding")

    tokens = backend.tokenize(loaded, [backend.padding_input("embedding")], _ROUND_TRIP_SEQ_LEN)

    assert tokens["attention_mask"].sum() > 0


def test_tokenize_rejects_empty_inputs(embedding_dir: Path) -> None:
    """An empty batch would produce a zero-row graph input and must raise."""
    backend = q3.Qwen3Backend()
    loaded = backend.load(embedding_dir, "embedding")

    with pytest.raises(ValueError, match="inputs"):
        backend.tokenize(loaded, [], _ROUND_TRIP_SEQ_LEN)


@pytest.mark.parametrize("seq_len", [0, -1])
def test_tokenize_rejects_a_non_positive_sequence_length(embedding_dir: Path, seq_len: int) -> None:
    """A non-positive fixed length is never a valid graph shape."""
    backend = q3.Qwen3Backend()
    loaded = backend.load(embedding_dir, "embedding")

    with pytest.raises(ValueError, match="seq_len"):
        backend.tokenize(loaded, ["abc"], seq_len)


@pytest.fixture(scope="module")
def round_trip(embedding_dir: Path) -> dict[str, Any]:
    """Run the patched wrapper and the FP32 sdpa reference over the sanity fixtures."""
    backend = q3.Qwen3Backend()
    inputs = list(backend.sanity_spec("embedding").all_inputs)
    loaded = backend.load(embedding_dir, "embedding")
    backend.apply_patches(loaded)
    wrapper = backend.wrap(loaded)
    tokens = backend.tokenize(loaded, inputs, _ROUND_TRIP_SEQ_LEN)
    with torch.no_grad():
        wrapped = wrapper(
            torch.from_numpy(tokens["input_ids"]).long(),
            torch.from_numpy(tokens["attention_mask"]).long(),
        )
    reference = backend.reference_outputs(embedding_dir, "embedding", inputs, _ROUND_TRIP_SEQ_LEN)
    del loaded, wrapper
    gc.collect()
    return {
        "inputs": inputs,
        "tokens": tokens,
        "wrapped": wrapped.numpy().reshape(len(inputs), -1),
        "reference": np.asarray(reference, dtype=np.float32).reshape(len(inputs), -1),
    }


def test_the_wrapper_matches_the_fp32_reference(round_trip: dict[str, Any]) -> None:
    """The traced module and the baseline must compute the same function.

    The two sides reach it by different routes -- the wrapper builds its
    own 4-D mask over the patched eager attention, the baseline lets the
    framework build the mask for the untouched sdpa attention -- so a
    disagreement about the mask, the pooling or the tokenization shows up
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

    assert reference.shape == (len(round_trip["inputs"]), _TINY_HIDDEN)
    for row in range(1, reference.shape[0]):
        assert not np.allclose(reference[0], reference[row], atol=_ROUND_TRIP_TOLERANCE)


# --- Core ML conversion of the traced wrapper --------------------------------


def test_the_traced_wrapper_converts_and_agrees_with_the_fp32_wrapper(
    embedding_dir: Path,
) -> None:
    """The conversion must produce a program that still computes the same embedding."""
    backend = q3.Qwen3Backend()
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
    assert embeddings.shape == expected.shape
    assert bool(np.isfinite(embeddings).all())
    cosines = np.sum(embeddings * expected, axis=1) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(expected, axis=1)
    )
    assert float(cosines.min()) >= _CONVERSION_COSINE_THRESHOLD
