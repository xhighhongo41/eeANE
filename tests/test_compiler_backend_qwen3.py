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

from eeane.compiler import conversion, selfcheck
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
    score_token_ids: tuple[int, int] | None = None,
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
        score_token_ids=score_token_ids,
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
    assert backend.supported_kinds == ("embedding", "reranker")


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
@pytest.mark.parametrize("kind", ["classifier", "generation"])
def test_unknown_kind_is_rejected(method: str, kind: str) -> None:
    """Every kind-dispatching method must reject a kind this backend cannot compile."""
    backend = q3.Qwen3Backend()

    with pytest.raises(ValueError, match="kind"):
        getattr(backend, method)(kind)


def test_load_rejects_an_unsupported_kind(tmp_path: Path) -> None:
    """load must validate the kind before touching the filesystem."""
    with pytest.raises(ValueError, match="kind"):
        q3.Qwen3Backend().load(tmp_path, "classifier")


@pytest.mark.parametrize("method", ["wrap", "tokenize"])
def test_handle_taking_methods_reject_an_unsupported_kind(method: str) -> None:
    """A handle carrying an unsupported kind must be rejected, not silently wrapped."""
    backend = q3.Qwen3Backend()
    loaded = _loaded(torch.nn.Identity(), kind="classifier")
    arguments = {"tokenize": (["text"], 8)}.get(method, ())

    with pytest.raises(ValueError, match="kind"):
        getattr(backend, method)(loaded, *arguments)


def test_reference_outputs_rejects_an_unsupported_kind(tmp_path: Path) -> None:
    """The reference path must validate the kind before loading any weights."""
    with pytest.raises(ValueError, match="kind"):
        q3.Qwen3Backend().reference_outputs(tmp_path, "classifier", [("q", "d")], 8)


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
    _write_tokenizer_files(directory)

    _write_json(
        directory / common.POOLING_DIRNAME / common.POOLING_CONFIG_FILENAME,
        {"word_embedding_dimension": _TINY_HIDDEN, "pooling_mode_lasttoken": True},
    )
    return directory


def _write_tokenizer_files(directory: Path, model_max_length: int = _TINY_POSITIONS) -> None:
    """Save a byte-level toy tokenizer into a synthetic model directory.

    Byte-level vocabulary with no merges: every byte is its own token, so
    the multilingual fixtures tokenize without shipping a real vocab file.
    The template appends the end-of-text token only -- a causal model has
    no leading classification token to prepend.

    Args:
        directory: Model directory the tokenizer files are written to.
        model_max_length: Length the saved tokenizer declares; only used
            to keep its own warnings quiet, since every encoding here is
            truncated by the caller.
    """
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
        model_max_length=model_max_length,
    ).save_pretrained(directory)


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


# --- the generative reranker -------------------------------------------------

# Vocabulary ids the synthetic reranker directory declares. Both stay
# inside _TINY_VOCAB, and they are far apart so a swapped pair is visible.
_RERANKER_TRUE_ID = 100
_RERANKER_FALSE_ID = 200

# Geometry of the reranker round trip. The toy tokenizer spells every byte
# out as its own token, so the chat prompt alone fills a couple of hundred
# positions; the bucket has to leave a body long enough for two fixtures
# sharing a query to still differ in their documents.
_RERANKER_POSITIONS = 512
_RERANKER_SEQ_LEN = 448

# Instruction the synthetic reranker directory declares as its default
# prompt. It is deliberately not the wording of any published model: the
# point is that whatever the directory declares is what ends up in the
# template.
_DECLARED_PROMPT = "Decide whether the document answers the question"

# Module types of a generative reranker's chain, spelled in the nested
# namespace so the suffix matching is exercised.
_SCORING_CHAIN = [
    "sentence_transformers.base.modules.transformer.Transformer",
    "sentence_transformers.cross_encoder.modules.logit_score.LogitScore",
]


def _tiny_causal_model(seed: int = 0, tie_word_embeddings: bool = True) -> Any:
    """Build a small randomly initialised Qwen3 causal LM in eval mode.

    Its position budget is the reranker one: a byte-level tokenizer turns
    the chat prompt into a few hundred tokens, so the rows this model is
    exercised on are longer than the embedding fixture's.
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    torch.manual_seed(seed)
    config = _tiny_config()
    config.tie_word_embeddings = tie_word_embeddings
    config.max_position_embeddings = _RERANKER_POSITIONS
    return Qwen3ForCausalLM(config).eval()


def _write_reranker_directory(
    directory: Path,
    *,
    tie_word_embeddings: bool = True,
    logit_score: dict[str, Any] | None = None,
    modules: list[str] | None = _SCORING_CHAIN,
    prompt: str | None = _DECLARED_PROMPT,
) -> Path:
    """Save a tiny generative reranker as a HuggingFace model directory.

    Args:
        directory: Destination directory; created if needed.
        tie_word_embeddings: Whether the checkpoint ties its output
            projection to its input embeddings. Untied checkpoints store
            ``lm_head.weight`` of their own.
        logit_score: Scoring declaration to write, or ``None`` to write
            none at all.
        modules: Module types to declare, or ``None`` to declare no chain.
        prompt: Default prompt to declare, or ``None`` to declare none.

    Returns:
        ``directory``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _tiny_causal_model(tie_word_embeddings=tie_word_embeddings).save_pretrained(directory)
    _write_tokenizer_files(directory, model_max_length=_RERANKER_POSITIONS)
    if logit_score is None:
        logit_score = {
            common.LOGIT_SCORE_TRUE_KEY: _RERANKER_TRUE_ID,
            common.LOGIT_SCORE_FALSE_KEY: _RERANKER_FALSE_ID,
        }
    if logit_score:
        _write_json(
            directory / common.LOGIT_SCORE_DIRNAME / common.LOGIT_SCORE_CONFIG_FILENAME,
            logit_score,
        )
    if modules is not None:
        _write_json(
            directory / common.ST_MODULES_FILENAME,
            [
                {"idx": i, "name": str(i), "path": "", "type": name}
                for i, name in enumerate(modules)
            ],
        )
    if prompt is not None:
        _write_json(
            directory / common.ST_CONFIG_FILENAME,
            {
                common.ST_DEFAULT_PROMPT_NAME_KEY: common.ST_FALLBACK_PROMPT_NAME,
                common.ST_PROMPTS_KEY: {common.ST_FALLBACK_PROMPT_NAME: prompt},
            },
        )
    return directory


@pytest.fixture(scope="module")
def reranker_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic generative reranker directory with a tied output projection."""
    return _write_reranker_directory(tmp_path_factory.mktemp("qwen3-reranker"))


@pytest.fixture(scope="module")
def untied_reranker_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic generative reranker whose output projection is its own matrix."""
    return _write_reranker_directory(
        tmp_path_factory.mktemp("qwen3-reranker-untied"), tie_word_embeddings=False
    )


# --- the published pair template ---------------------------------------------


def test_pair_template_writes_the_declared_instruction_into_the_body(reranker_dir: Path) -> None:
    """The question the model is asked comes from the model, not from this backend."""
    template = q3.Qwen3Backend().pair_template(reranker_dir, "reranker")

    assert isinstance(template, base.PairTemplate)
    assert _DECLARED_PROMPT in template.body_format
    assert q3.INSTRUCTION_PLACEHOLDER not in template.body_format


def test_pair_template_publishes_the_two_fixed_halves(reranker_dir: Path) -> None:
    """The prefix and the suffix are the published chat prompt, verbatim."""
    template = q3.Qwen3Backend().pair_template(reranker_dir, "reranker")

    assert template.prefix == q3.RERANKER_PROMPT_PREFIX
    assert template.suffix == q3.RERANKER_PROMPT_SUFFIX


def test_pair_template_marks_both_texts_exactly_once(reranker_dir: Path) -> None:
    """A missing marker would drop a text; a repeated one would make the body ambiguous."""
    template = q3.Qwen3Backend().pair_template(reranker_dir, "reranker")

    assert template.body_format.count("{query}") == 1
    assert template.body_format.count("{document}") == 1


def test_pair_template_of_an_embedding_model_is_none(embedding_dir: Path) -> None:
    """An embedding model encodes one text and has no pair to shape."""
    assert q3.Qwen3Backend().pair_template(embedding_dir, "embedding") is None


def test_pair_template_refuses_a_reranker_declaring_no_prompt(
    tmp_path: Path,
) -> None:
    """Without the declared instruction the model would be asked a different question."""
    model_dir = _write_reranker_directory(tmp_path / "no-prompt", prompt=None)

    with pytest.raises(ValueError) as excinfo:
        q3.Qwen3Backend().pair_template(model_dir, "reranker")

    assert common.ST_CONFIG_FILENAME in str(excinfo.value)


def test_pair_template_rejects_an_unsupported_kind(tmp_path: Path) -> None:
    """A kind this backend cannot compile must be refused, not answered with None."""
    with pytest.raises(ValueError, match="kind"):
        q3.Qwen3Backend().pair_template(tmp_path, "classifier")


def test_the_template_renders_the_prompt_this_family_publishes(reranker_dir: Path) -> None:
    """Laid out end to end, the three parts must read as the model's own chat prompt."""
    template = q3.Qwen3Backend().pair_template(reranker_dir, "reranker")

    rendered = (
        template.prefix
        + template.body_format.replace("{query}", "Q").replace("{document}", "D")
        + template.suffix
    )

    assert rendered == (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
        'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
        f"<Instruct>: {_DECLARED_PROMPT}\n<Query>: Q\n<Document>: D"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


# --- interface answers for the reranker kind ---------------------------------


def test_the_reranker_score_space_is_the_raw_logit() -> None:
    """A difference of two vocabulary logits spans a range the sigmoid squashes unevenly."""
    assert q3.Qwen3Backend().reranker_score_space() == base.SCORE_SPACE_LOGIT


def test_output_name_of_the_reranker_kind() -> None:
    """The graph emits one raw logit under the name every reranker of this project uses."""
    assert q3.Qwen3Backend().output_name("reranker") == "logits"


def test_the_reranker_serves_the_shared_pair_sets() -> None:
    """The ordering expectation must point at the relevant and the irrelevant pair."""
    spec = q3.Qwen3Backend().sanity_spec("reranker")

    assert spec.input_sets == common.SANITY_PAIR_SETS
    assert spec.relevant_index == common.SANITY_RELEVANT_INDEX
    assert spec.irrelevant_index == common.SANITY_IRRELEVANT_INDEX


def test_the_reranker_fixtures_are_non_empty_pairs() -> None:
    """A fully masked row can make the attention softmax degenerate."""
    backend = q3.Qwen3Backend()

    for fixture in (backend.trace_example("reranker"), backend.padding_input("reranker")):
        assert isinstance(fixture, tuple)
        assert len(fixture) == 2
        assert all(isinstance(part, str) and part for part in fixture)


# --- the scoring weight ------------------------------------------------------


def test_build_score_weight_subtracts_the_two_tied_vocabulary_rows(reranker_dir: Path) -> None:
    """With a tied projection the two rows are two rows of the input embedding matrix."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(reranker_dir, dtype=torch.float32).eval()

    weight = q3.build_score_weight(model, reranker_dir, _RERANKER_TRUE_ID, _RERANKER_FALSE_ID)

    embeddings = model.get_input_embeddings().weight
    expected = (embeddings[_RERANKER_TRUE_ID] - embeddings[_RERANKER_FALSE_ID]).unsqueeze(0)
    assert weight.shape == (1, _TINY_HIDDEN)
    assert weight.dtype == torch.float32
    assert torch.allclose(weight, expected)


def test_build_score_weight_reads_an_untied_output_projection(untied_reranker_dir: Path) -> None:
    """An untied checkpoint keeps its own projection, which is the one to cut rows out of."""
    from safetensors.torch import load_file
    from transformers import AutoModel

    model = AutoModel.from_pretrained(untied_reranker_dir, dtype=torch.float32).eval()
    stored = load_file(str(untied_reranker_dir / "model.safetensors"))
    lm_head = stored[q3.LM_HEAD_WEIGHT_KEY].to(torch.float32)

    weight = q3.build_score_weight(
        model, untied_reranker_dir, _RERANKER_TRUE_ID, _RERANKER_FALSE_ID
    )

    expected = (lm_head[_RERANKER_TRUE_ID] - lm_head[_RERANKER_FALSE_ID]).unsqueeze(0)
    assert torch.allclose(weight, expected)
    # Guard: the input embeddings are a different matrix here, so reading
    # them instead would have produced different numbers.
    embeddings = model.get_input_embeddings().weight
    tied_guess = (embeddings[_RERANKER_TRUE_ID] - embeddings[_RERANKER_FALSE_ID]).unsqueeze(0)
    assert not torch.allclose(weight, tied_guess)


@pytest.mark.parametrize("true_id", [_TINY_VOCAB, _TINY_VOCAB + 5])
def test_build_score_weight_rejects_an_id_outside_the_vocabulary(
    reranker_dir: Path, true_id: int
) -> None:
    """An id past the last row would read whatever the tensor happens to hold next."""
    from transformers import AutoModel

    model = AutoModel.from_pretrained(reranker_dir, dtype=torch.float32).eval()

    with pytest.raises(ValueError):
        q3.build_score_weight(model, reranker_dir, true_id, _RERANKER_FALSE_ID)


def test_build_score_weight_reports_a_projection_it_cannot_reach(tmp_path: Path) -> None:
    """Neither tied embeddings nor a stored projection means nothing to cut rows out of."""
    model = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=False),
        get_input_embeddings=lambda: None,
    )

    with pytest.raises(ValueError):
        q3.build_score_weight(model, tmp_path, _RERANKER_TRUE_ID, _RERANKER_FALSE_ID)


# --- the traceable reranker wrapper ------------------------------------------


def _reranker_batch(reranker_dir: Path) -> dict[str, Any]:
    """Tokenize two sanity pairs and load both views of the synthetic model."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    backend = q3.Qwen3Backend()
    pairs = list(common.SANITY_PAIRS_EN[:2])
    loaded = backend.load(reranker_dir, "reranker")
    tokens = backend.tokenize(loaded, pairs, _RERANKER_SEQ_LEN)
    # The whole language model, output projection included: what the graph
    # cuts two rows out of is what this computes in full.
    causal = Qwen3ForCausalLM.from_pretrained(reranker_dir, dtype=torch.float32).eval()
    causal.config.return_dict = True
    causal.config.use_cache = False
    return {
        "pairs": pairs,
        "tokens": tokens,
        "input_ids": torch.from_numpy(tokens["input_ids"]).long(),
        "attention_mask": torch.from_numpy(tokens["attention_mask"]).long(),
        "loaded": loaded,
        "backbone": loaded.model,
        "causal": causal,
    }


def _auto_tokenizer(model_dir: Path) -> Any:
    """Load the synthetic directory's tokenizer through AutoTokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir)


@pytest.fixture(scope="module")
def reranker_batch(reranker_dir: Path) -> Iterator[dict[str, Any]]:
    """Two tokenized sanity pairs plus both views of the synthetic model."""
    batch = _reranker_batch(reranker_dir)
    yield batch
    del batch
    gc.collect()


def test_the_reranker_wrapper_emits_one_logit_per_row(
    reranker_dir: Path, reranker_batch: dict[str, Any]
) -> None:
    """The graph output is the (B, 1) logit the server already applies a sigmoid to."""
    weight = q3.build_score_weight(
        reranker_batch["backbone"], reranker_dir, _RERANKER_TRUE_ID, _RERANKER_FALSE_ID
    )
    wrapper = q3.GenerativeRerankerWrapper(reranker_batch["backbone"], weight).eval()

    with torch.no_grad():
        scores = wrapper(reranker_batch["input_ids"], reranker_batch["attention_mask"])

    assert scores.shape == (len(reranker_batch["pairs"]), 1)


def test_the_reranker_wrapper_computes_the_difference_of_the_two_vocabulary_logits(
    reranker_dir: Path, reranker_batch: dict[str, Any]
) -> None:
    """The cut-out rows must reproduce what the full output projection produces.

    The wrapper multiplies the pooled state by two rows lifted out of the
    projection; the reference runs the whole projection and subtracts the
    two logits afterwards. Agreement here is what proves the right rows
    were lifted, and in the right order.
    """
    weight = q3.build_score_weight(
        reranker_batch["backbone"], reranker_dir, _RERANKER_TRUE_ID, _RERANKER_FALSE_ID
    )
    wrapper = q3.GenerativeRerankerWrapper(reranker_batch["backbone"], weight).eval()
    attention_mask = reranker_batch["attention_mask"]

    with torch.no_grad():
        scores = wrapper(reranker_batch["input_ids"], attention_mask).reshape(-1)
        logits = reranker_batch["causal"](
            input_ids=reranker_batch["input_ids"], attention_mask=attention_mask
        ).logits

    rows = torch.arange(logits.shape[0])
    last = attention_mask.sum(dim=1) - 1
    expected = logits[rows, last, _RERANKER_TRUE_ID] - logits[rows, last, _RERANKER_FALSE_ID]
    assert torch.allclose(scores, expected, atol=1e-4)
    # Guard: a constant output would satisfy the comparison above only if
    # the two pairs happened to score alike; they must not.
    assert not torch.allclose(scores[0], scores[1], atol=1e-3)


def test_the_emitted_logit_is_the_two_way_softmax_the_model_publishes(
    reranker_dir: Path, reranker_batch: dict[str, Any]
) -> None:
    """sigmoid(true - false) is exactly the "true" side of a two-element softmax.

    This identity is why the graph may emit one number: the server's own
    sigmoid then produces the very probability the published procedure
    (log_softmax over the two logits, then exp) computes.
    """
    attention_mask = reranker_batch["attention_mask"]
    weight = q3.build_score_weight(
        reranker_batch["backbone"], reranker_dir, _RERANKER_TRUE_ID, _RERANKER_FALSE_ID
    )
    wrapper = q3.GenerativeRerankerWrapper(reranker_batch["backbone"], weight).eval()

    with torch.no_grad():
        scores = wrapper(reranker_batch["input_ids"], attention_mask).reshape(-1)
        logits = reranker_batch["causal"](
            input_ids=reranker_batch["input_ids"], attention_mask=attention_mask
        ).logits

    rows = torch.arange(logits.shape[0])
    last = attention_mask.sum(dim=1) - 1
    pair = torch.stack(
        [logits[rows, last, _RERANKER_FALSE_ID], logits[rows, last, _RERANKER_TRUE_ID]], dim=-1
    )
    published = torch.exp(torch.log_softmax(pair, dim=-1))[:, 1]

    assert torch.allclose(torch.sigmoid(scores), published, atol=1e-5)


def test_the_reranker_wrapper_reads_the_last_real_position_of_a_padded_row(
    reranker_dir: Path, reranker_batch: dict[str, Any]
) -> None:
    """A short pair leaves the row padded; the verdict is not at its padded end.

    The graph pools ``sum(mask) - 1``, and a wrapper reading the final
    position instead would agree with the baseline on every row that
    happens to fill the bucket -- which is why this measures a row that
    does not.
    """
    backend = q3.Qwen3Backend()
    tokens = backend.tokenize(reranker_batch["loaded"], [("q", "d")], _RERANKER_SEQ_LEN)
    input_ids = torch.from_numpy(tokens["input_ids"]).long()
    attention_mask = torch.from_numpy(tokens["attention_mask"]).long()
    assert int(attention_mask.sum()) < _RERANKER_SEQ_LEN  # the row really is padded
    weight = q3.build_score_weight(
        reranker_batch["backbone"], reranker_dir, _RERANKER_TRUE_ID, _RERANKER_FALSE_ID
    )
    wrapper = q3.GenerativeRerankerWrapper(reranker_batch["backbone"], weight).eval()

    with torch.no_grad():
        score = wrapper(input_ids, attention_mask).reshape(-1)
        logits = reranker_batch["causal"](input_ids=input_ids, attention_mask=attention_mask).logits

    last = int(attention_mask.sum()) - 1
    expected = logits[0, last, _RERANKER_TRUE_ID] - logits[0, last, _RERANKER_FALSE_ID]
    at_the_padded_end = logits[0, -1, _RERANKER_TRUE_ID] - logits[0, -1, _RERANKER_FALSE_ID]
    assert torch.allclose(score[0], expected, atol=1e-4)
    # Guard: the two positions must really disagree, or reading either
    # would satisfy the assertion above.
    assert not torch.allclose(expected, at_the_padded_end, atol=1e-3)


def test_the_reranker_wrappers_mask_path_matches_the_frameworks_own(
    reranker_dir: Path, reranker_batch: dict[str, Any]
) -> None:
    """The in-graph 4-D mask must compute what the framework's 2-D path computes."""
    backbone = reranker_batch["backbone"]
    weight = q3.build_score_weight(backbone, reranker_dir, _RERANKER_TRUE_ID, _RERANKER_FALSE_ID)
    wrapper = q3.GenerativeRerankerWrapper(backbone, weight).eval()
    input_ids, attention_mask = reranker_batch["input_ids"], reranker_batch["attention_mask"]

    with torch.no_grad():
        from_wrapper = wrapper(input_ids, attention_mask).reshape(-1)
        hidden = backbone(input_ids=input_ids, attention_mask=attention_mask)[0]
        pooled = common.last_token_pool(hidden, attention_mask)
        from_2d = torch.nn.functional.linear(pooled, weight).reshape(-1)

    assert torch.allclose(from_wrapper, from_2d, atol=1e-5)


@pytest.mark.parametrize("shape", [(2, _TINY_HIDDEN), (_TINY_HIDDEN,), (1, 1, _TINY_HIDDEN)])
def test_the_reranker_wrapper_rejects_a_weight_of_another_shape(shape: tuple[int, ...]) -> None:
    """Anything but one row would emit something other than the single logit."""
    with pytest.raises(ValueError, match="score_weight"):
        q3.GenerativeRerankerWrapper(torch.nn.Identity(), torch.zeros(shape))


def test_the_traced_reranker_graph_has_no_dynamic_size_node(
    reranker_dir: Path, reranker_batch: dict[str, Any]
) -> None:
    """Shape arithmetic in the graph is what the conversion patches exist to remove."""
    backend = q3.Qwen3Backend()
    backend.apply_patches(_loaded(reranker_batch["backbone"], kind="reranker"))
    weight = q3.build_score_weight(
        reranker_batch["backbone"], reranker_dir, _RERANKER_TRUE_ID, _RERANKER_FALSE_ID
    )
    wrapper = q3.GenerativeRerankerWrapper(reranker_batch["backbone"], weight).eval()

    traced = conversion.trace_model(wrapper, reranker_batch["tokens"])

    assert _DYNAMIC_SIZE_NODE not in str(traced.inlined_graph)


# --- load / tokenize / reference_outputs for the reranker kind ---------------


def test_load_returns_a_reranker_handle_carrying_the_declared_ids(reranker_dir: Path) -> None:
    """The two declared ids must reach the wrapper through the handle."""
    loaded = q3.Qwen3Backend().load(reranker_dir, "reranker")

    assert loaded.kind == "reranker"
    assert loaded.score_token_ids == (_RERANKER_TRUE_ID, _RERANKER_FALSE_ID)
    # A reranker of this family declares neither: its verdict is read off
    # the backbone's last position, not pooled and projected.
    assert loaded.pooling is None
    assert loaded.dense is None
    assert loaded.config.return_dict is False
    assert loaded.config.use_cache is False


def test_load_of_a_reranker_refuses_a_missing_scoring_declaration(tmp_path: Path) -> None:
    """Which two vocabulary entries carry the verdict cannot be guessed."""
    model_dir = _write_reranker_directory(tmp_path / "no-score", logit_score={})

    with pytest.raises(ValueError) as excinfo:
        q3.Qwen3Backend().load(model_dir, "reranker")

    assert common.LOGIT_SCORE_DIRNAME in str(excinfo.value)


def test_load_of_a_reranker_refuses_a_chain_it_cannot_reproduce(tmp_path: Path) -> None:
    """A directory whose modules say something else must not be compiled as this."""
    model_dir = _write_reranker_directory(
        tmp_path / "wrong-chain",
        modules=[
            "sentence_transformers.models.Transformer",
            "sentence_transformers.models.Pooling",
        ],
    )

    with pytest.raises(ValueError):
        q3.Qwen3Backend().load(model_dir, "reranker")


def test_load_of_a_reranker_refuses_a_directory_declaring_no_prompt(tmp_path: Path) -> None:
    """The instruction is part of the question the model was trained to answer."""
    model_dir = _write_reranker_directory(tmp_path / "load-no-prompt", prompt=None)

    with pytest.raises(ValueError):
        q3.Qwen3Backend().load(model_dir, "reranker")


def test_wrap_of_a_reranker_builds_the_generative_wrapper(reranker_dir: Path) -> None:
    """The reranker kind must be wrapped by the module emitting one logit."""
    backend = q3.Qwen3Backend()
    loaded = backend.load(reranker_dir, "reranker")

    wrapper = backend.wrap(loaded)

    assert isinstance(wrapper, q3.GenerativeRerankerWrapper)
    assert wrapper.training is False
    assert wrapper.score_weight.shape == (1, _TINY_HIDDEN)


def test_tokenize_of_a_reranker_lays_the_pairs_into_the_template(reranker_dir: Path) -> None:
    """The traced graph must be built for the very rows the server will send."""
    backend = q3.Qwen3Backend()
    loaded = backend.load(reranker_dir, "reranker")
    pairs = list(common.SANITY_PAIRS_EN[:2])

    tokens = backend.tokenize(loaded, pairs, _RERANKER_SEQ_LEN)

    expected = common.tokenize_templated_pairs(
        _auto_tokenizer(reranker_dir),
        pairs,
        _RERANKER_SEQ_LEN,
        backend.pair_template(reranker_dir, "reranker"),
    )
    np.testing.assert_array_equal(tokens["input_ids"], expected["input_ids"])
    np.testing.assert_array_equal(tokens["attention_mask"], expected["attention_mask"])
    assert tokens["input_ids"].dtype == np.int32
    # The plain pair encoding of the same pairs is a different row, which
    # is what makes the template worth applying here.
    plain = common.tokenize_pairs(loaded.tokenizer, pairs, _RERANKER_SEQ_LEN)
    assert not np.array_equal(tokens["input_ids"], plain["input_ids"])


def test_reference_outputs_of_a_reranker_matches_the_traced_wrapper(reranker_dir: Path) -> None:
    """The FP32 baseline and the graph must compute the same score by two routes.

    The baseline runs the full vocabulary projection over the untouched
    sdpa attention with the framework's own mask; the wrapper runs two
    lifted rows over the patched eager attention with its own 4-D mask.
    """
    backend = q3.Qwen3Backend()
    pairs = list(common.SANITY_PAIRS_EN)
    loaded = backend.load(reranker_dir, "reranker")
    backend.apply_patches(loaded)
    wrapper = backend.wrap(loaded)
    tokens = backend.tokenize(loaded, pairs, _RERANKER_SEQ_LEN)
    with torch.no_grad():
        wrapped = (
            wrapper(
                torch.from_numpy(tokens["input_ids"]).long(),
                torch.from_numpy(tokens["attention_mask"]).long(),
            )
            .numpy()
            .reshape(-1)
        )
    del loaded, wrapper
    gc.collect()

    reference = backend.reference_outputs(reranker_dir, "reranker", pairs, _RERANKER_SEQ_LEN)

    assert reference.shape == (len(pairs),)
    assert reference.dtype == np.float32
    np.testing.assert_allclose(wrapped, reference, rtol=0, atol=1e-4)
    # Guard: identical scores would make the comparison prove nothing.
    assert len(set(round(float(value), 5) for value in reference)) > 1


def test_reference_outputs_of_a_reranker_rejects_empty_inputs(reranker_dir: Path) -> None:
    """No pairs means nothing to compare; that must raise before loading weights."""
    with pytest.raises(ValueError, match="inputs"):
        q3.Qwen3Backend().reference_outputs(reranker_dir, "reranker", [], _RERANKER_SEQ_LEN)


def test_the_traced_reranker_converts_and_agrees_with_the_fp32_wrapper(
    reranker_dir: Path,
) -> None:
    """The conversion must produce a program still emitting the same relevance logit."""
    backend = q3.Qwen3Backend()
    loaded = backend.load(reranker_dir, "reranker")
    backend.apply_patches(loaded)
    wrapper = backend.wrap(loaded)
    pairs = list(common.SANITY_PAIRS_EN[:2])
    tokens = backend.tokenize(loaded, pairs, _RERANKER_SEQ_LEN)
    batch_size = len(pairs)
    with torch.no_grad():
        expected = (
            wrapper(
                torch.from_numpy(tokens["input_ids"]).long(),
                torch.from_numpy(tokens["attention_mask"]).long(),
            )
            .numpy()
            .astype(np.float32)
            .reshape(-1)
        )

    traced = conversion.trace_model(wrapper, tokens)
    mlmodel = conversion.convert_model(
        traced,
        _RERANKER_SEQ_LEN,
        "fp16",
        "macos13",
        backend.output_name("reranker"),
        batch_size=batch_size,
    )
    prediction = mlmodel.predict(dict(tokens))
    scores = np.asarray(prediction["logits"], dtype=np.float32).reshape(-1)

    description = mlmodel.get_spec().description
    assert [tensor.name for tensor in description.input] == ["input_ids", "attention_mask"]
    assert [tensor.name for tensor in description.output] == ["logits"]
    assert scores.shape == expected.shape
    assert bool(np.isfinite(scores).all())
    # A converted graph is a lower-precision one, so the two are compared
    # on the same measure the self-check applies to this backend.
    assert float(np.abs(scores - expected).max()) <= selfcheck.SANITY_LOGIT_TOLERANCE
