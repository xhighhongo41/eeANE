"""Tests for the sentence-transformers Dense support of the compile backends.

Some sentence-transformers models declare, next to the Transformer and the
Pooling module, one or more ``Dense`` linear projections whose output is
the embedding the model publishes. Skipping them produces a plausible but
wrong vector of the wrong width, so the module chain a model directory
declares is read, validated and reproduced -- both inside the traced graph
and in the FP32 baseline the self-check compares it against.

Three layers, all of which run anywhere (no Core ML, no real weights):

* the reader of ``modules.json`` and of a Dense module's own
  ``config.json``/weights;
* the traceable wrappers and the FP32 baseline, which must apply the same
  projection in the same order;
* the wiring of both into every backend's ``load``/``wrap``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from test_compiler_backend import _write_model_directory
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from eeane.compiler.backends import base, bert, common
from eeane.compiler.backends import modernbert as mb
from eeane.compiler.backends import xlm_roberta as xlmr

# Module type strings as sentence-transformers writes them into
# modules.json. They are spelled out here rather than imported so a test
# failure tells whether the reader or the expected on-disk format moved.
TRANSFORMER_TYPE = "sentence_transformers.models.Transformer"
POOLING_TYPE = "sentence_transformers.models.Pooling"
DENSE_TYPE = "sentence_transformers.models.Dense"
NORMALIZE_TYPE = "sentence_transformers.models.Normalize"

# Activation class paths a Dense config.json can name, likewise verbatim.
IDENTITY_ACTIVATION = "torch.nn.modules.linear.Identity"
TANH_ACTIVATION = "torch.nn.modules.activation.Tanh"

# Directory name a Dense module conventionally lives in.
DENSE_DIRNAME = "2_Dense"

# Widths of the synthetic Dense projections below: small enough to write
# the expected values out by hand, different from each other so a stage
# that is skipped or applied twice changes the shape.
DENSE_IN = 4
DENSE_OUT = 3

# Sequence length of the tokenizer-driven baseline tests.
DENSE_SEQ_LEN = 8


def _write_json(path: Path, payload: object) -> None:
    """Write ``payload`` as JSON, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_modules(model_dir: Path, chain: Sequence[tuple[str, str]]) -> Path:
    """Write a ``modules.json`` declaring ``(type, path)`` entries in order.

    Args:
        model_dir: Model directory to write into.
        chain: Declared modules, in declaration order.

    Returns:
        ``model_dir``.
    """
    _write_json(
        model_dir / "modules.json",
        [
            {"idx": idx, "name": str(idx), "path": path, "type": module_type}
            for idx, (module_type, path) in enumerate(chain)
        ],
    )
    return model_dir


def _dense_weights(
    in_features: int = DENSE_IN, out_features: int = DENSE_OUT
) -> dict[str, torch.Tensor]:
    """Build deterministic Dense weights of the given shape."""
    weight = torch.arange(out_features * in_features, dtype=torch.float32).reshape(
        out_features, in_features
    )
    return {"linear.weight": weight / 10.0, "linear.bias": torch.arange(out_features) / 4.0}


def _write_dense(
    model_dir: Path,
    path: str = DENSE_DIRNAME,
    *,
    in_features: int = DENSE_IN,
    out_features: int = DENSE_OUT,
    bias: bool = True,
    activation: str = IDENTITY_ACTIVATION,
    config: dict[str, Any] | None = None,
    weights: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Write one Dense module directory (config plus safetensors weights).

    Args:
        model_dir: Model directory the module lives under.
        path: Module directory name, as declared in ``modules.json``.
        in_features: Declared input width.
        out_features: Declared output width.
        bias: Whether the declaration claims a bias.
        activation: Declared activation class path.
        config: Complete replacement for the declaration, for the tests
            that write a malformed one.
        weights: Complete replacement for the checkpoint tensors.

    Returns:
        The created module directory.
    """
    dense_dir = model_dir / path
    dense_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        dense_dir / "config.json",
        config
        if config is not None
        else {
            "in_features": in_features,
            "out_features": out_features,
            "bias": bias,
            "activation_function": activation,
        },
    )
    if weights is None:
        weights = _dense_weights(in_features, out_features)
        if not bias:
            weights.pop("linear.bias")
    save_file(weights, str(dense_dir / "model.safetensors"))
    return dense_dir


def _dense_model_dir(tmp_path: Path, **dense: Any) -> Path:
    """Build a model directory declaring Transformer, Pooling and one Dense."""
    model_dir = tmp_path / "model"
    _write_modules(
        model_dir,
        [(TRANSFORMER_TYPE, ""), (POOLING_TYPE, "1_Pooling"), (DENSE_TYPE, DENSE_DIRNAME)],
    )
    _write_dense(model_dir, **dense)
    return model_dir


class _StubBackbone(torch.nn.Module):
    """Deterministic stand-in for a transformer backbone.

    Returns one embedding row per token as a tuple, which is the shape the
    wrappers and the FP32 baseline expect of a model loaded with
    ``config.return_dict = False``.
    """

    def __init__(self, vocab_size: int = 300, hidden_size: int = DENSE_IN) -> None:
        """Build the lookup table the stub's forward returns rows of."""
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        generator = torch.Generator().manual_seed(0)
        table = torch.rand(vocab_size, hidden_size, generator=generator)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        with torch.no_grad():
            self.embedding.weight.copy_(table)
        self.eval()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Return the looked-up rows, ignoring the mask (the pooling uses it)."""
        return (self.embedding(input_ids),)


def _byte_tokenizer() -> PreTrainedTokenizerFast:
    """Build an in-memory byte-level tokenizer usable for any UTF-8 text."""
    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
    for index, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet())):
        vocab[character] = index + 4
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[("<s>", 2), ("</s>", 3)],
    )
    tokenizer.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        model_max_length=64,
    )


def _tiny_inputs(seq_len: int = 6, batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic ``(input_ids, attention_mask)`` tensors with padding."""
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(4, 200, (batch_size, seq_len), generator=generator)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[-1, seq_len // 2 :] = 0  # the last row is half padding
    return input_ids, attention_mask


def _loaded(model: torch.nn.Module, dense: torch.nn.Module | None = None) -> base.LoadedModel:
    """Build the handle a backend hands from ``load`` to ``wrap``."""
    return base.LoadedModel(
        model=model,
        tokenizer=None,
        config=getattr(model, "config", None),
        model_dir=Path("/nonexistent-model-dir"),
        kind="embedding",
        attn="eager",
        pooling="mean",
        dense=dense,
    )


# --- the declared module chain ------------------------------------------------


def test_a_directory_without_modules_json_declares_no_dense(tmp_path: Path) -> None:
    """Most embedding models ship no modules.json at all; they keep converting as before."""
    assert common.read_dense_modules(tmp_path) == ()


def test_a_transformer_pooling_chain_declares_no_dense(tmp_path: Path) -> None:
    """The plain two-module chain projects nothing after the pooling."""
    model_dir = _write_modules(
        tmp_path / "model", [(TRANSFORMER_TYPE, ""), (POOLING_TYPE, "1_Pooling")]
    )

    assert common.read_dense_modules(model_dir) == ()


def test_a_trailing_normalize_without_dense_is_accepted(tmp_path: Path) -> None:
    """Normalization is applied by the server, so a declared Normalize is read past."""
    model_dir = _write_modules(
        tmp_path / "model",
        [(TRANSFORMER_TYPE, ""), (POOLING_TYPE, "1_Pooling"), (NORMALIZE_TYPE, "2_Normalize")],
    )

    assert common.read_dense_modules(model_dir) == ()


def test_a_declared_dense_is_described_by_its_own_config(tmp_path: Path) -> None:
    """The stage carries the widths, the bias flag and the activation it declares."""
    model_dir = _dense_model_dir(tmp_path, bias=False, activation=TANH_ACTIVATION)

    stages = common.read_dense_modules(model_dir)

    assert len(stages) == 1
    stage = stages[0]
    assert stage.path == model_dir / DENSE_DIRNAME
    assert (stage.in_features, stage.out_features) == (DENSE_IN, DENSE_OUT)
    assert stage.bias is False
    assert stage.activation == common.DENSE_ACTIVATION_TANH


def test_a_dense_followed_by_a_normalize_is_accepted(tmp_path: Path) -> None:
    """A Dense plus a parameter-less Normalize is a chain this backend reproduces."""
    model_dir = tmp_path / "model"
    _write_modules(
        model_dir,
        [
            (TRANSFORMER_TYPE, ""),
            (POOLING_TYPE, "1_Pooling"),
            (DENSE_TYPE, DENSE_DIRNAME),
            (NORMALIZE_TYPE, "3_Normalize"),
        ],
    )
    _write_dense(model_dir)

    assert len(common.read_dense_modules(model_dir)) == 1


def test_two_dense_stages_are_described_in_declaration_order(tmp_path: Path) -> None:
    """A chain of projections must stay in the order the model applies them."""
    model_dir = tmp_path / "model"
    _write_modules(
        model_dir,
        [
            (TRANSFORMER_TYPE, ""),
            (POOLING_TYPE, "1_Pooling"),
            (DENSE_TYPE, DENSE_DIRNAME),
            (DENSE_TYPE, "3_Dense"),
        ],
    )
    _write_dense(model_dir, in_features=DENSE_IN, out_features=DENSE_OUT)
    _write_dense(model_dir, "3_Dense", in_features=DENSE_OUT, out_features=DENSE_IN + 3)

    stages = common.read_dense_modules(model_dir)

    assert [(stage.in_features, stage.out_features) for stage in stages] == [
        (DENSE_IN, DENSE_OUT),
        (DENSE_OUT, DENSE_IN + 3),
    ]


def test_two_stages_whose_widths_do_not_chain_are_refused(tmp_path: Path) -> None:
    """A stage takes its predecessor's output, so widths that disagree are unusable."""
    model_dir = tmp_path / "model"
    _write_modules(
        model_dir,
        [
            (TRANSFORMER_TYPE, ""),
            (POOLING_TYPE, "1_Pooling"),
            (DENSE_TYPE, DENSE_DIRNAME),
            (DENSE_TYPE, "3_Dense"),
        ],
    )
    _write_dense(model_dir, in_features=DENSE_IN, out_features=DENSE_OUT)
    _write_dense(model_dir, "3_Dense", in_features=DENSE_OUT + 1, out_features=DENSE_IN)

    with pytest.raises(ValueError, match="in_features"):
        common.read_dense_modules(model_dir)


@pytest.mark.parametrize(
    "chain",
    [
        pytest.param(
            [
                (TRANSFORMER_TYPE, ""),
                (POOLING_TYPE, "1_Pooling"),
                ("sentence_transformers.models.LSTM", "2_LSTM"),
            ],
            id="unknown-type",
        ),
        pytest.param(
            [(TRANSFORMER_TYPE, ""), (DENSE_TYPE, DENSE_DIRNAME), (POOLING_TYPE, "1_Pooling")],
            id="dense-before-pooling",
        ),
        pytest.param(
            [
                (TRANSFORMER_TYPE, ""),
                (POOLING_TYPE, "1_Pooling"),
                (NORMALIZE_TYPE, "2_Normalize"),
                (DENSE_TYPE, DENSE_DIRNAME),
            ],
            id="dense-after-normalize",
        ),
        pytest.param([(TRANSFORMER_TYPE, "")], id="no-pooling"),
        pytest.param(
            [(TRANSFORMER_TYPE, ""), (POOLING_TYPE, "1_Pooling"), (POOLING_TYPE, "2_Pooling")],
            id="two-poolings",
        ),
        pytest.param(
            [(POOLING_TYPE, "1_Pooling"), (TRANSFORMER_TYPE, "")], id="no-leading-transformer"
        ),
        pytest.param(
            [
                (TRANSFORMER_TYPE, ""),
                (POOLING_TYPE, "1_Pooling"),
                (NORMALIZE_TYPE, "2_Normalize"),
                (NORMALIZE_TYPE, "3_Normalize"),
            ],
            id="two-normalizes",
        ),
        pytest.param([], id="empty-chain"),
    ],
)
def test_an_unreproducible_module_chain_is_refused(
    tmp_path: Path, chain: list[tuple[str, str]]
) -> None:
    """A chain this backend cannot reproduce must be refused, never silently trimmed."""
    model_dir = _write_modules(tmp_path / "model", chain)

    with pytest.raises(ValueError) as excinfo:
        common.read_dense_modules(model_dir)

    message = str(excinfo.value)
    assert "modules.json" in message
    # The declared chain is quoted back, so the reason is actionable.
    for module_type, _ in chain:
        assert module_type in message


def test_a_corrupt_modules_json_is_refused(tmp_path: Path) -> None:
    """Unparsable JSON must raise instead of being read as "no Dense"."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "modules.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="modules.json"):
        common.read_dense_modules(model_dir)


@pytest.mark.parametrize(
    "payload",
    [{"modules": []}, "Transformer", [["Transformer"]], [{"type": 3}], [{"name": "0"}]],
    ids=["object", "string", "nested-list", "non-string-type", "no-type"],
)
def test_a_malformed_modules_json_is_refused(tmp_path: Path, payload: object) -> None:
    """Only a list of objects declaring a string type can be read as a module chain."""
    model_dir = tmp_path / "model"
    _write_json(model_dir / "modules.json", payload)

    with pytest.raises(ValueError, match="modules.json"):
        common.read_dense_modules(model_dir)


def test_a_dense_declaring_an_unsupported_activation_is_refused(tmp_path: Path) -> None:
    """An activation this backend does not implement must be named, not ignored."""
    model_dir = _dense_model_dir(tmp_path, activation="torch.nn.modules.activation.GELU")

    with pytest.raises(ValueError) as excinfo:
        common.read_dense_modules(model_dir)

    message = str(excinfo.value)
    assert "GELU" in message
    assert IDENTITY_ACTIVATION in message and TANH_ACTIVATION in message


@pytest.mark.parametrize(
    "config",
    [
        {"out_features": DENSE_OUT, "bias": True, "activation_function": IDENTITY_ACTIVATION},
        {"in_features": DENSE_IN, "bias": True, "activation_function": IDENTITY_ACTIVATION},
        {"in_features": DENSE_IN, "out_features": DENSE_OUT, "bias": True},
        {
            "in_features": DENSE_IN,
            "out_features": 0,
            "bias": True,
            "activation_function": IDENTITY_ACTIVATION,
        },
        {
            "in_features": DENSE_IN,
            "out_features": DENSE_OUT,
            "bias": "yes",
            "activation_function": IDENTITY_ACTIVATION,
        },
        {
            "in_features": DENSE_IN,
            "out_features": "1024",
            "bias": True,
            "activation_function": IDENTITY_ACTIVATION,
        },
    ],
    ids=["no-in", "no-out", "no-activation", "zero-out", "string-bias", "string-out"],
)
def test_an_incomplete_dense_declaration_is_refused(tmp_path: Path, config: dict[str, Any]) -> None:
    """Every field the projection is built from must be declared, and be of its type."""
    model_dir = _dense_model_dir(tmp_path, config=config)

    with pytest.raises(ValueError, match="Dense"):
        common.read_dense_modules(model_dir)


def test_a_missing_dense_declaration_is_refused(tmp_path: Path) -> None:
    """A declared module directory without a config.json cannot be reproduced."""
    model_dir = tmp_path / "model"
    _write_modules(
        model_dir,
        [(TRANSFORMER_TYPE, ""), (POOLING_TYPE, "1_Pooling"), (DENSE_TYPE, DENSE_DIRNAME)],
    )

    with pytest.raises(ValueError, match=DENSE_DIRNAME):
        common.read_dense_modules(model_dir)


@pytest.mark.parametrize("path", ["../outside", "/etc", ""], ids=["parent", "absolute", "empty"])
def test_a_dense_path_outside_the_model_directory_is_refused(tmp_path: Path, path: str) -> None:
    """A module path may only address the model directory it was declared in."""
    model_dir = _write_modules(
        tmp_path / "model",
        [(TRANSFORMER_TYPE, ""), (POOLING_TYPE, "1_Pooling"), (DENSE_TYPE, path)],
    )

    with pytest.raises(ValueError, match="path"):
        common.read_dense_modules(model_dir)


def test_the_stage_record_describes_the_declaration_for_the_metadata(tmp_path: Path) -> None:
    """The recorded description must be JSON-serializable and name every declared field."""
    model_dir = _dense_model_dir(tmp_path, activation=TANH_ACTIVATION)

    record = common.dense_record(common.read_dense_modules(model_dir))

    assert record == ({"in": DENSE_IN, "out": DENSE_OUT, "bias": True, "activation": "tanh"},)
    json.dumps(record)


def test_the_stage_record_of_a_chain_without_dense_is_none() -> None:
    """A model that projects nothing records ``null``, not an empty list."""
    assert common.dense_record(()) is None


# --- building the projection --------------------------------------------------


def test_the_built_projection_reproduces_the_declared_linear_layer(tmp_path: Path) -> None:
    """The module must compute ``weight @ x + bias`` with the checkpoint's own tensors."""
    model_dir = _dense_model_dir(tmp_path)
    weights = _dense_weights()

    dense, record = common.load_dense(model_dir)

    pooled = torch.arange(DENSE_IN, dtype=torch.float32).unsqueeze(0)
    expected = pooled @ weights["linear.weight"].T + weights["linear.bias"]
    assert record == ({"in": DENSE_IN, "out": DENSE_OUT, "bias": True, "activation": "identity"},)
    assert torch.allclose(dense(pooled), expected)


def test_the_built_projection_applies_the_declared_tanh(tmp_path: Path) -> None:
    """A declared Tanh must be applied after the linear layer, not before it."""
    model_dir = _dense_model_dir(tmp_path, activation=TANH_ACTIVATION)
    weights = _dense_weights()

    dense, _ = common.load_dense(model_dir)

    pooled = torch.arange(DENSE_IN, dtype=torch.float32).unsqueeze(0)
    expected = torch.tanh(pooled @ weights["linear.weight"].T + weights["linear.bias"])
    assert torch.allclose(dense(pooled), expected)


def test_a_bias_free_projection_adds_nothing(tmp_path: Path) -> None:
    """``bias: false`` must build a linear layer that really has no bias."""
    model_dir = _dense_model_dir(tmp_path, bias=False)

    dense, record = common.load_dense(model_dir)

    pooled = torch.arange(DENSE_IN, dtype=torch.float32).unsqueeze(0)
    expected = pooled @ _dense_weights()["linear.weight"].T
    assert record == ({"in": DENSE_IN, "out": DENSE_OUT, "bias": False, "activation": "identity"},)
    assert torch.allclose(dense(pooled), expected)


def test_two_stages_are_applied_in_declaration_order(tmp_path: Path) -> None:
    """A two-stage chain must feed the first stage's output into the second."""
    model_dir = tmp_path / "model"
    _write_modules(
        model_dir,
        [
            (TRANSFORMER_TYPE, ""),
            (POOLING_TYPE, "1_Pooling"),
            (DENSE_TYPE, DENSE_DIRNAME),
            (DENSE_TYPE, "3_Dense"),
        ],
    )
    _write_dense(model_dir, activation=TANH_ACTIVATION)
    _write_dense(model_dir, "3_Dense", in_features=DENSE_OUT, out_features=DENSE_IN)

    dense, record = common.load_dense(model_dir)

    first = _dense_weights()
    second = _dense_weights(DENSE_OUT, DENSE_IN)
    pooled = torch.arange(DENSE_IN, dtype=torch.float32).unsqueeze(0)
    hidden = torch.tanh(pooled @ first["linear.weight"].T + first["linear.bias"])
    expected = hidden @ second["linear.weight"].T + second["linear.bias"]
    assert record is not None and len(record) == 2
    assert torch.allclose(dense(pooled), expected)


def test_a_directory_without_dense_builds_no_projection(tmp_path: Path) -> None:
    """A chain without a Dense must produce no module at all, not an empty Sequential."""
    model_dir = _write_modules(
        tmp_path / "model", [(TRANSFORMER_TYPE, ""), (POOLING_TYPE, "1_Pooling")]
    )

    assert common.load_dense(model_dir) == (None, None)


def test_the_built_projection_is_fp32_whatever_the_checkpoint_stores(tmp_path: Path) -> None:
    """Conversion to a lower precision happens in Core ML, never in PyTorch."""
    weights = {key: value.to(torch.float16) for key, value in _dense_weights().items()}
    model_dir = _dense_model_dir(tmp_path, weights=weights)

    dense, _ = common.load_dense(model_dir)

    assert all(parameter.dtype == torch.float32 for parameter in dense.parameters())


def test_the_built_projection_is_in_eval_mode(tmp_path: Path) -> None:
    """Everything that reaches the trace must be in eval mode, as the backends promise."""
    dense, _ = common.load_dense(_dense_model_dir(tmp_path))

    assert dense.training is False


def test_pickle_dense_weights_are_read_when_no_safetensors_are_present(tmp_path: Path) -> None:
    """A ``.bin`` module is only ever reached through the --allow-pickle gate, and loads."""
    model_dir = _dense_model_dir(tmp_path)
    (model_dir / DENSE_DIRNAME / "model.safetensors").unlink()
    torch.save(_dense_weights(), model_dir / DENSE_DIRNAME / "pytorch_model.bin")

    dense, _ = common.load_dense(model_dir)

    pooled = torch.arange(DENSE_IN, dtype=torch.float32).unsqueeze(0)
    weights = _dense_weights()
    expected = pooled @ weights["linear.weight"].T + weights["linear.bias"]
    assert torch.allclose(dense(pooled), expected)


def test_missing_dense_weights_are_reported(tmp_path: Path) -> None:
    """A declared module without any checkpoint must name what it looked for."""
    model_dir = _dense_model_dir(tmp_path)
    (model_dir / DENSE_DIRNAME / "model.safetensors").unlink()

    with pytest.raises(ValueError) as excinfo:
        common.load_dense(model_dir)

    message = str(excinfo.value)
    assert "model.safetensors" in message
    assert DENSE_DIRNAME in message


@pytest.mark.parametrize(
    "weights",
    [
        pytest.param({"linear.weight": torch.zeros(DENSE_OUT + 1, DENSE_IN)}, id="wrong-out"),
        pytest.param({"linear.weight": torch.zeros(DENSE_OUT, DENSE_IN + 1)}, id="wrong-in"),
        pytest.param({"linear.weight": torch.zeros(DENSE_OUT)}, id="not-a-matrix"),
        pytest.param(
            {
                "linear.weight": torch.zeros(DENSE_OUT, DENSE_IN),
                "linear.bias": torch.zeros(DENSE_OUT + 1),
            },
            id="wrong-bias",
        ),
    ],
)
def test_weights_contradicting_the_declaration_are_refused(
    tmp_path: Path, weights: dict[str, torch.Tensor]
) -> None:
    """Declared widths and checkpoint shapes must agree, or the projection is a guess."""
    model_dir = _dense_model_dir(tmp_path, bias="linear.bias" in weights, weights=weights)

    with pytest.raises(ValueError, match="shape"):
        common.load_dense(model_dir)


def test_a_checkpoint_without_the_declared_bias_is_refused(tmp_path: Path) -> None:
    """``bias: true`` with no bias tensor cannot be reproduced."""
    model_dir = _dense_model_dir(
        tmp_path, weights={"linear.weight": torch.zeros(DENSE_OUT, DENSE_IN)}
    )

    with pytest.raises(ValueError, match="linear.bias"):
        common.load_dense(model_dir)


def test_an_unexpected_checkpoint_tensor_is_refused(tmp_path: Path) -> None:
    """A module holding more than the linear layer is not one this backend understands."""
    weights = {**_dense_weights(), "extra.weight": torch.zeros(2, 2)}
    model_dir = _dense_model_dir(tmp_path, weights=weights)

    with pytest.raises(ValueError, match="extra.weight"):
        common.load_dense(model_dir)


# --- the traceable wrappers ---------------------------------------------------


@pytest.mark.parametrize(
    ("wrapper_class", "pool"),
    [
        (common.EmbeddingWrapper, "mean"),
        (common.ClsEmbeddingWrapper, "cls"),
    ],
)
def test_a_wrapper_without_a_dense_computes_exactly_what_it_always_did(
    wrapper_class: type[torch.nn.Module], pool: str
) -> None:
    """The graph of a model without a Dense must stay bit-for-bit the one it was."""
    model = _StubBackbone()
    input_ids, attention_mask = _tiny_inputs()
    hidden = model(input_ids=input_ids, attention_mask=attention_mask)[0]
    expected = common.mean_pool(hidden, attention_mask) if pool == "mean" else hidden[:, 0]

    output = wrapper_class(model)(input_ids, attention_mask)

    assert torch.equal(output, expected)


@pytest.mark.parametrize(
    ("wrapper_class", "pool"),
    [
        (common.EmbeddingWrapper, "mean"),
        (common.ClsEmbeddingWrapper, "cls"),
    ],
)
def test_a_wrapper_projects_the_pooled_vector_through_the_dense(
    wrapper_class: type[torch.nn.Module], pool: str
) -> None:
    """The projection is applied to the pooled vector, after pooling and before nothing."""
    model = _StubBackbone()
    weights = _dense_weights()
    dense = torch.nn.Sequential(torch.nn.Linear(DENSE_IN, DENSE_OUT), torch.nn.Tanh())
    with torch.no_grad():
        dense[0].weight.copy_(weights["linear.weight"])
        dense[0].bias.copy_(weights["linear.bias"])
    input_ids, attention_mask = _tiny_inputs()
    hidden = model(input_ids=input_ids, attention_mask=attention_mask)[0]
    pooled = common.mean_pool(hidden, attention_mask) if pool == "mean" else hidden[:, 0]
    expected = torch.tanh(pooled @ weights["linear.weight"].T + weights["linear.bias"])

    output = wrapper_class(model, dense=dense)(input_ids, attention_mask)

    assert output.shape == (input_ids.shape[0], DENSE_OUT)
    assert torch.allclose(output, expected, atol=1e-6)


# --- the FP32 baseline --------------------------------------------------------


def test_the_baseline_follows_the_same_dense_as_the_wrapper() -> None:
    """Baseline and traced graph must compute the same function, projection included.

    The self-check compares the two against each other, so a projection
    applied on only one side would either hide a real error or invent one.
    """
    model = _StubBackbone()
    tokenizer = _byte_tokenizer()
    texts = ["a short one", "a considerably longer sentence to pad less"]
    dense = torch.nn.Sequential(torch.nn.Linear(DENSE_IN, DENSE_OUT), torch.nn.Tanh()).eval()

    baseline = common.encode_pytorch(
        model, tokenizer, texts, DENSE_SEQ_LEN, pooling="mean", dense=dense
    )

    batch = common.tokenize_batch(tokenizer, texts, DENSE_SEQ_LEN)
    wrapper = common.EmbeddingWrapper(model, dense=dense)
    with torch.no_grad():
        traced = wrapper(
            torch.from_numpy(batch["input_ids"]).long(),
            torch.from_numpy(batch["attention_mask"]).long(),
        )
    assert baseline.shape == (len(texts), DENSE_OUT)
    assert baseline.dtype == np.float32
    np.testing.assert_allclose(baseline, traced.numpy(), atol=1e-6)


def test_the_baseline_without_a_dense_is_the_pooled_vector() -> None:
    """Leaving the projection out must not change the established baseline."""
    model = _StubBackbone()
    tokenizer = _byte_tokenizer()
    texts = ["a short one", "another sentence"]

    baseline = common.encode_pytorch(model, tokenizer, texts, DENSE_SEQ_LEN, pooling="mean")

    batch = common.tokenize_batch(tokenizer, texts, DENSE_SEQ_LEN)
    with torch.no_grad():
        expected = common.EmbeddingWrapper(model)(
            torch.from_numpy(batch["input_ids"]).long(),
            torch.from_numpy(batch["attention_mask"]).long(),
        )
    assert baseline.shape == (len(texts), DENSE_IN)
    np.testing.assert_allclose(baseline, expected.numpy(), atol=1e-6)


# --- wiring into the backends -------------------------------------------------

_BACKENDS = [
    pytest.param(mb.ModernBertBackend(), id="modernbert"),
    pytest.param(bert.BertBackend(), id="bert"),
    pytest.param(xlmr.XlmRobertaBackend(), id="xlm-roberta"),
]


def _unreproducible_model_dir(tmp_path: Path) -> Path:
    """Build a weightless model directory declaring a chain no backend can reproduce."""
    model_dir = tmp_path / "model"
    _write_json(model_dir / "config.json", {"architectures": ["StubModel"]})
    _write_json(model_dir / "1_Pooling" / "config.json", {"pooling_mode_mean_tokens": True})
    _write_modules(
        model_dir,
        [
            (TRANSFORMER_TYPE, ""),
            (POOLING_TYPE, "1_Pooling"),
            ("sentence_transformers.models.LSTM", "2_LSTM"),
        ],
    )
    return model_dir


@pytest.mark.parametrize("backend", _BACKENDS)
def test_load_refuses_an_unreproducible_chain_before_reading_any_weight(
    backend: Any, tmp_path: Path
) -> None:
    """The refusal must come from the declaration, not from a failed forward pass.

    The directory holds no weights at all, so an error naming the module
    chain proves the declaration was read before the checkpoint.
    """
    model_dir = _unreproducible_model_dir(tmp_path)

    with pytest.raises(ValueError, match="modules.json"):
        backend.load(model_dir, "embedding")


@pytest.mark.parametrize("backend", [_BACKENDS[0], _BACKENDS[2]], ids=["modernbert", "xlm-roberta"])
def test_load_of_a_reranker_ignores_the_module_chain(backend: Any, tmp_path: Path) -> None:
    """A reranker scores with its own head; a sentence-transformers chain never applies."""
    model_dir = _unreproducible_model_dir(tmp_path)

    with pytest.raises(Exception) as excinfo:  # noqa: B017 - whatever transformers reports
        backend.load(model_dir, "reranker")

    assert "modules.json" not in str(excinfo.value)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_wrap_hands_the_dense_to_the_pooling_wrapper(backend: Any) -> None:
    """Whatever ``load`` resolved must be what the traced module applies."""
    dense = torch.nn.Sequential(torch.nn.Linear(DENSE_IN, DENSE_OUT)).eval()
    model = _StubBackbone()

    wrapper = backend.wrap(_loaded(model, dense=dense))

    assert wrapper.dense is dense


@pytest.mark.parametrize("backend", _BACKENDS)
def test_wrap_without_a_dense_leaves_the_wrapper_projection_free(backend: Any) -> None:
    """A model that declares no projection must trace exactly as it did before."""
    wrapper = backend.wrap(_loaded(_StubBackbone()))

    assert wrapper.dense is None


# --- a synthetic model directory end to end (no Core ML) ----------------------


@pytest.fixture(scope="module")
def dense_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a tiny ModernBERT directory declaring a Dense projection."""
    model_dir = _write_model_directory(
        tmp_path_factory.mktemp("modernbert-dense"), pooling_flag="pooling_mode_mean_tokens"
    )
    hidden_size = 32
    projected = 48
    _write_modules(
        model_dir,
        [
            (TRANSFORMER_TYPE, ""),
            (POOLING_TYPE, "1_Pooling"),
            (DENSE_TYPE, DENSE_DIRNAME),
            (NORMALIZE_TYPE, "3_Normalize"),
        ],
    )
    generator = torch.Generator().manual_seed(3)
    _write_dense(
        model_dir,
        in_features=hidden_size,
        out_features=projected,
        bias=True,
        weights={
            "linear.weight": torch.rand(projected, hidden_size, generator=generator) - 0.5,
            "linear.bias": torch.rand(projected, generator=generator) - 0.5,
        },
    )
    return model_dir


def test_a_loaded_dense_model_carries_its_projection_and_its_record(
    dense_model_dir: Path,
) -> None:
    """``load`` must resolve the projection and describe it for the metadata."""
    loaded = mb.ModernBertBackend().load(dense_model_dir, "embedding")

    assert loaded.pooling == "mean"
    assert isinstance(loaded.dense, torch.nn.Module)
    assert loaded.dense_config == ({"in": 32, "out": 48, "bias": True, "activation": "identity"},)


def test_the_traced_module_and_the_fp32_baseline_agree_on_a_dense_model(
    dense_model_dir: Path,
) -> None:
    """The projection must reach both sides of the self-check, and widen the output.

    Only the attention implementation differs between the two sides (the
    traced module uses eager, the baseline sdpa), so anything beyond that
    kernel difference means one of them left the projection out.
    """
    backend = mb.ModernBertBackend()
    texts = ["これは短い文です。", "こちらは少しだけ長い日本語の文章です。"]
    seq_len = 16

    loaded = backend.load(dense_model_dir, "embedding")
    wrapper = backend.wrap(loaded)
    batch = backend.tokenize(loaded, texts, seq_len)
    with torch.no_grad():
        traced = wrapper(
            torch.from_numpy(batch["input_ids"]).long(),
            torch.from_numpy(batch["attention_mask"]).long(),
        ).numpy()
    reference = backend.reference_outputs(dense_model_dir, "embedding", texts, seq_len)

    assert traced.shape == (len(texts), 48)
    assert reference.shape == (len(texts), 48)
    assert np.abs(traced - reference).max() < 1e-4
