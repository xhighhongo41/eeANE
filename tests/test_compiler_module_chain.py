"""Tests for the two on-disk dialects of a sentence-transformers declaration.

sentence-transformers 6 writes the same modules a model has always had
under nested namespaces (``...base.modules.transformer.Transformer``
instead of ``...models.Transformer``), and declares its pooling as one
string (``"pooling_mode": "mean"``) instead of a set of boolean flags.
Both dialects describe the same model, so the shared reader must accept
either, while still refusing every chain or pooling it cannot reproduce.

What is covered:

* the role a module type name maps to;
* a directory laid out exactly as a 6-series export (a published
  ``google/embeddinggemma-2`` style layout);
* the module chains that are accepted and refused, in either namespace
  and in a mix of both;
* a Dense module under the nested namespace, whose own files keep the
  layout they always had;
* the pooling declared in either dialect, or in both.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from eeane.compiler.backends import common

# Module type strings as sentence-transformers writes them into
# modules.json, spelled out rather than imported so a failure tells
# whether the reader or the expected on-disk format moved. LEGACY_* is the
# flat namespace of earlier releases, NESTED_* the one of the 6 series.
LEGACY_TRANSFORMER = "sentence_transformers.models.Transformer"
LEGACY_POOLING = "sentence_transformers.models.Pooling"
LEGACY_DENSE = "sentence_transformers.models.Dense"
LEGACY_NORMALIZE = "sentence_transformers.models.Normalize"
NESTED_TRANSFORMER = "sentence_transformers.base.modules.transformer.Transformer"
NESTED_POOLING = "sentence_transformers.sentence_transformer.modules.pooling.Pooling"
NESTED_DENSE = "sentence_transformers.base.modules.dense.Dense"
NESTED_NORMALIZE = "sentence_transformers.base.modules.normalize.Normalize"

# The module a generative reranker ends its chain in; never part of an
# embedding chain.
NESTED_LOGIT_SCORE = "sentence_transformers.cross_encoder.modules.logit_score.LogitScore"

IDENTITY_ACTIVATION = "torch.nn.modules.linear.Identity"
TANH_ACTIVATION = "torch.nn.modules.activation.Tanh"

# Widths of the synthetic Dense projections: equal on purpose, so any
# number of them chains without a width mismatch getting in the way of
# what a test is about.
DENSE_WIDTH = 4
DENSE_DIRNAME = "2_Dense"


def _write_json(path: Path, payload: object) -> None:
    """Write ``payload`` as JSON, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_chain(model_dir: Path, chain: Sequence[tuple[str, str]]) -> Path:
    """Write a ``modules.json`` of ``(type, path)`` entries, as the 6 series does.

    Args:
        model_dir: Model directory to write into.
        chain: Declared modules, in declaration order.

    Returns:
        ``model_dir``.
    """
    _write_json(
        model_dir / "modules.json",
        [{"type": module_type, "path": path} for module_type, path in chain],
    )
    return model_dir


def _write_dense(
    model_dir: Path, path: str = DENSE_DIRNAME, *, activation: str = IDENTITY_ACTIVATION
) -> None:
    """Write one Dense module directory in the layout it has always had."""
    _write_json(
        model_dir / path / "config.json",
        {
            "in_features": DENSE_WIDTH,
            "out_features": DENSE_WIDTH,
            "bias": True,
            "activation_function": activation,
        },
    )
    weight = torch.arange(DENSE_WIDTH * DENSE_WIDTH, dtype=torch.float32).reshape(
        DENSE_WIDTH, DENSE_WIDTH
    )
    save_file(
        {"linear.weight": weight / 10.0, "linear.bias": torch.arange(DENSE_WIDTH) / 4.0},
        str(model_dir / path / "model.safetensors"),
    )


def _write_pooling(model_dir: Path, declaration: object) -> Path:
    """Write ``declaration`` as the model's ``1_Pooling/config.json``."""
    _write_json(model_dir / common.POOLING_DIRNAME / "config.json", declaration)
    return model_dir


def _model_with_chain(tmp_path: Path, chain: Sequence[tuple[str, str]]) -> Path:
    """Build a model directory declaring ``chain``, with a Dense module for each Dense entry."""
    model_dir = _write_chain(tmp_path / "model", chain)
    for module_type, path in chain:
        if module_type.split(".")[-1] == "Dense":
            _write_dense(model_dir, path)
    return model_dir


# --- the role of a module type -------------------------------------------------


@pytest.mark.parametrize(
    ("module_type", "role"),
    [
        (LEGACY_TRANSFORMER, "Transformer"),
        (LEGACY_POOLING, "Pooling"),
        (LEGACY_DENSE, "Dense"),
        (LEGACY_NORMALIZE, "Normalize"),
        (NESTED_TRANSFORMER, "Transformer"),
        (NESTED_POOLING, "Pooling"),
        (NESTED_NORMALIZE, "Normalize"),
        (NESTED_DENSE, "Dense"),
        # The class name is what decides, wherever under the package it lives.
        ("sentence_transformers.Transformer", "Transformer"),
        ("sentence_transformers.sentence_transformer.modules.dense.Dense", "Dense"),
    ],
)
def test_module_role_names_the_role_of_a_sentence_transformers_module(
    module_type: str, role: str
) -> None:
    """A sentence-transformers class named after a role has that role, in any namespace."""
    assert common._module_role(module_type) == role


@pytest.mark.parametrize(
    "module_type",
    [
        "sentence_transformers.models.LSTM",
        "sentence_transformers.models.WordEmbeddings",
        # A role name is matched whole, not as a prefix, suffix or substring.
        "sentence_transformers.models.TransformerX",
        "sentence_transformers.models.MyTransformer",
        "sentence_transformers.models.transformer",
        "sentence_transformers.models.POOLING",
        # Only sentence-transformers' own classes carry a role.
        "other_pkg.models.Transformer",
        "sentence_transformers_extra.models.Transformer",
        "my_sentence_transformers.models.Pooling",
        "Transformer",
        "",
        # A module of the module path is not the class.
        "sentence_transformers.models.Transformer.helpers",
        "sentence_transformers.",
        "sentence_transformers",
        NESTED_LOGIT_SCORE,
    ],
)
def test_module_role_is_none_for_anything_else(module_type: str) -> None:
    """Unknown classes, foreign packages and near misses of a role name have no role."""
    assert common._module_role(module_type) is None


def test_the_legacy_module_names_keep_their_roles() -> None:
    """The exported full names other modules and tests use still resolve to a role."""
    assert common._module_role(common.ST_MODULE_TRANSFORMER) == common.ST_ROLE_TRANSFORMER
    assert common._module_role(common.ST_MODULE_POOLING) == common.ST_ROLE_POOLING
    assert common._module_role(common.ST_MODULE_DENSE) == common.ST_ROLE_DENSE
    assert common._module_role(common.ST_MODULE_NORMALIZE) == common.ST_ROLE_NORMALIZE


# --- a directory laid out as a 6-series export --------------------------------


def _write_series_six_embedding_model(model_dir: Path) -> Path:
    """Write the declaration files of an embedding model saved by sentence-transformers 6.1."""
    _write_json(
        model_dir / "modules.json",
        [
            {
                "type": "sentence_transformers.base.modules.transformer.Transformer",
                "path": "",
            },
            {
                "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling",
                "path": "1_Pooling",
            },
            {
                "type": "sentence_transformers.base.modules.normalize.Normalize",
                "path": "2_Normalize",
            },
        ],
    )
    _write_json(
        model_dir / "1_Pooling" / "config.json",
        {"embedding_dimension": 768, "pooling_mode": "mean", "include_prompt": True},
    )
    return model_dir


def test_a_series_six_export_is_read_as_a_plain_mean_pooled_model(tmp_path: Path) -> None:
    """Transformer, Pooling and Normalize in the nested namespace project nothing further."""
    model_dir = _write_series_six_embedding_model(tmp_path / "model")

    assert common.read_dense_modules(model_dir) == ()
    assert common.read_pooling_mode(model_dir) == common.POOLING_MEAN
    assert common.load_dense(model_dir) == (None, None)


# --- the module chains that are accepted ---------------------------------------


@pytest.mark.parametrize(
    "chain",
    [
        pytest.param(
            [(NESTED_TRANSFORMER, ""), (NESTED_POOLING, "1_Pooling")], id="nested-two-modules"
        ),
        pytest.param(
            [
                (NESTED_TRANSFORMER, ""),
                (NESTED_POOLING, "1_Pooling"),
                (NESTED_NORMALIZE, "2_Normalize"),
            ],
            id="nested-with-normalize",
        ),
        pytest.param(
            [(LEGACY_TRANSFORMER, ""), (NESTED_POOLING, "1_Pooling")], id="legacy-then-nested"
        ),
        pytest.param(
            [
                (NESTED_TRANSFORMER, ""),
                (LEGACY_POOLING, "1_Pooling"),
                (LEGACY_NORMALIZE, "2_Normalize"),
            ],
            id="nested-then-legacy",
        ),
        pytest.param(
            [
                (NESTED_TRANSFORMER, ""),
                (NESTED_POOLING, "1_Pooling"),
                (NESTED_DENSE, DENSE_DIRNAME),
            ],
            id="nested-with-dense",
        ),
        pytest.param(
            [
                (NESTED_TRANSFORMER, ""),
                (NESTED_POOLING, "1_Pooling"),
                (LEGACY_DENSE, DENSE_DIRNAME),
                (NESTED_DENSE, "3_Dense"),
                (NESTED_NORMALIZE, "4_Normalize"),
            ],
            id="mixed-dense-stack-with-normalize",
        ),
    ],
)
def test_a_chain_of_roles_is_accepted_in_either_namespace(
    tmp_path: Path, chain: list[tuple[str, str]]
) -> None:
    """Order and count of the roles decide, not which namespace a type name is spelled in."""
    model_dir = _model_with_chain(tmp_path, chain)

    stages = common.read_dense_modules(model_dir)

    dense_paths = [
        model_dir / path for module_type, path in chain if "Dense" in module_type.split(".")[-1]
    ]
    assert [stage.path for stage in stages] == dense_paths


# --- the module chains that are refused ----------------------------------------


@pytest.mark.parametrize(
    "chain",
    [
        pytest.param([(NESTED_POOLING, "1_Pooling"), (NESTED_TRANSFORMER, "")], id="pooling-first"),
        pytest.param([(NESTED_TRANSFORMER, "")], id="no-pooling"),
        pytest.param(
            [
                (NESTED_TRANSFORMER, ""),
                (NESTED_POOLING, "1_Pooling"),
                (NESTED_POOLING, "2_Pooling"),
            ],
            id="two-poolings",
        ),
        pytest.param(
            [
                (NESTED_TRANSFORMER, ""),
                (NESTED_POOLING, "1_Pooling"),
                (NESTED_NORMALIZE, "2_Normalize"),
                (NESTED_DENSE, DENSE_DIRNAME),
            ],
            id="dense-after-normalize",
        ),
        pytest.param(
            [
                (NESTED_TRANSFORMER, ""),
                (NESTED_POOLING, "1_Pooling"),
                (NESTED_NORMALIZE, "2_Normalize"),
                (LEGACY_NORMALIZE, "3_Normalize"),
            ],
            id="two-normalizes",
        ),
        pytest.param(
            [
                (NESTED_TRANSFORMER, ""),
                (NESTED_POOLING, "1_Pooling"),
                (NESTED_NORMALIZE, "2_Normalize"),
                ("sentence_transformers.base.modules.lstm.LSTM", "3_LSTM"),
            ],
            id="something-after-normalize",
        ),
        pytest.param(
            [
                (NESTED_TRANSFORMER, ""),
                (NESTED_POOLING, "1_Pooling"),
                ("sentence_transformers.base.modules.lstm.LSTM", "2_LSTM"),
            ],
            id="unknown-nested-type",
        ),
        pytest.param(
            [
                (NESTED_TRANSFORMER, ""),
                (NESTED_DENSE, DENSE_DIRNAME),
                (NESTED_POOLING, "1_Pooling"),
            ],
            id="dense-before-pooling",
        ),
        pytest.param(
            [
                (NESTED_TRANSFORMER, ""),
                (NESTED_POOLING, "1_Pooling"),
                ("other_pkg.models.Dense", DENSE_DIRNAME),
            ],
            id="foreign-dense",
        ),
        pytest.param(
            [("other_pkg.models.Transformer", ""), (NESTED_POOLING, "1_Pooling")],
            id="foreign-transformer",
        ),
        pytest.param(
            [
                ("sentence_transformers.models.TransformerX", ""),
                (NESTED_POOLING, "1_Pooling"),
            ],
            id="near-miss-transformer",
        ),
        pytest.param(
            [(NESTED_TRANSFORMER, ""), (NESTED_LOGIT_SCORE, "1_LogitScore")],
            id="reranker-chain",
        ),
    ],
)
def test_a_chain_this_backend_cannot_reproduce_is_refused_with_its_reason(
    tmp_path: Path, chain: list[tuple[str, str]]
) -> None:
    """The nested namespace buys no leniency: every wrong chain is still named and refused."""
    model_dir = _model_with_chain(tmp_path, chain)

    with pytest.raises(ValueError) as excinfo:
        common.read_dense_modules(model_dir)

    message = str(excinfo.value)
    assert "modules.json" in message
    # The declared chain is quoted back, so the reason is actionable.
    for module_type, _ in chain:
        assert module_type in message
    # The requirement is stated in terms of roles, not of one namespace.
    for role in (
        common.ST_ROLE_TRANSFORMER,
        common.ST_ROLE_POOLING,
        common.ST_ROLE_DENSE,
        common.ST_ROLE_NORMALIZE,
    ):
        assert role in message


# --- a Dense module under the nested namespace --------------------------------


@pytest.mark.parametrize("activation", [IDENTITY_ACTIVATION, TANH_ACTIVATION])
def test_a_nested_dense_is_read_exactly_like_the_legacy_one(
    tmp_path: Path, activation: str
) -> None:
    """Only the type name differs; the module's own files are read as they always were."""
    model_dir = tmp_path / "model"
    _write_dense(model_dir, activation=activation)
    legacy_chain = [
        (LEGACY_TRANSFORMER, ""),
        (LEGACY_POOLING, "1_Pooling"),
        (LEGACY_DENSE, DENSE_DIRNAME),
        (LEGACY_NORMALIZE, "3_Normalize"),
    ]
    nested_chain = [
        (NESTED_TRANSFORMER, ""),
        (NESTED_POOLING, "1_Pooling"),
        (NESTED_DENSE, DENSE_DIRNAME),
        (NESTED_NORMALIZE, "3_Normalize"),
    ]

    _write_chain(model_dir, legacy_chain)
    legacy_stages = common.read_dense_modules(model_dir)
    legacy_module, legacy_record = common.load_dense(model_dir)
    _write_chain(model_dir, nested_chain)
    nested_stages = common.read_dense_modules(model_dir)
    nested_module, nested_record = common.load_dense(model_dir)

    assert len(nested_stages) == 1
    assert nested_stages == legacy_stages
    assert nested_record == legacy_record
    assert nested_module is not None and legacy_module is not None
    pooled = torch.linspace(-1.0, 1.0, DENSE_WIDTH).reshape(1, DENSE_WIDTH)
    assert torch.equal(nested_module(pooled), legacy_module(pooled))


def test_a_nested_dense_without_a_readable_config_is_refused_with_its_reason(
    tmp_path: Path,
) -> None:
    """A Dense whose files are not in the known layout fails by what is wrong with them."""
    model_dir = _write_chain(
        tmp_path / "model",
        [(NESTED_TRANSFORMER, ""), (NESTED_POOLING, "1_Pooling"), (NESTED_DENSE, DENSE_DIRNAME)],
    )

    with pytest.raises(ValueError, match="cannot read the Dense declaration") as excinfo:
        common.read_dense_modules(model_dir)
    assert str(model_dir / DENSE_DIRNAME / "config.json") in str(excinfo.value)

    # A declaration in some other layout names the field it is missing.
    _write_json(model_dir / DENSE_DIRNAME / "config.json", {"input_dim": 4, "output_dim": 4})
    with pytest.raises(ValueError, match="in_features"):
        common.read_dense_modules(model_dir)


def test_a_nested_dense_without_weights_is_refused_when_it_is_built(tmp_path: Path) -> None:
    """Missing weights surface from the build step exactly as for the legacy name."""
    model_dir = _write_chain(
        tmp_path / "model",
        [(NESTED_TRANSFORMER, ""), (NESTED_POOLING, "1_Pooling"), (NESTED_DENSE, DENSE_DIRNAME)],
    )
    _write_json(
        model_dir / DENSE_DIRNAME / "config.json",
        {
            "in_features": DENSE_WIDTH,
            "out_features": DENSE_WIDTH,
            "bias": True,
            "activation_function": IDENTITY_ACTIVATION,
        },
    )

    assert len(common.read_dense_modules(model_dir)) == 1
    with pytest.raises(ValueError, match="neither"):
        common.load_dense(model_dir)


# --- pooling declared as one string --------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mean", common.POOLING_MEAN),
        ("cls", common.POOLING_CLS),
        ("lasttoken", common.POOLING_LASTTOKEN),
    ],
)
def test_a_pooling_mode_string_selects_that_mode(tmp_path: Path, name: str, expected: str) -> None:
    """The single-string dialect names the mode directly."""
    model_dir = _write_pooling(
        tmp_path / "model",
        {"embedding_dimension": 768, "pooling_mode": name, "include_prompt": True},
    )

    assert common.read_pooling_mode(model_dir) == expected


def test_the_string_dialect_names_every_mode_the_flag_dialect_does() -> None:
    """Both tables must lead to the same set of modes, or one dialect would lag behind."""
    assert set(common.POOLING_MODE_NAMES.values()) == set(common.POOLING_MODES)
    assert set(common.POOLING_MODE_KEYS.values()) == set(common.POOLING_MODES)


def test_other_keys_of_the_string_dialect_do_not_take_part(tmp_path: Path) -> None:
    """``include_prompt`` and the width are not read, and no flag is needed beside the string."""
    model_dir = _write_pooling(
        tmp_path / "model", {"pooling_mode": "cls", "include_prompt": False, "unrelated": [1]}
    )

    assert common.read_pooling_mode(model_dir) == common.POOLING_CLS


@pytest.mark.parametrize(
    "value",
    [
        "max",
        "weightedmean",
        "mean_sqrt_len_tokens",
        "MEAN",
        "Mean",
        " mean",
        "mean ",
        "last_token",
        "",
        ["mean"],
        {"mean": True},
        1,
        0,
        1.5,
        None,
        True,
        False,
    ],
)
def test_a_pooling_mode_this_backend_cannot_apply_is_refused(tmp_path: Path, value: Any) -> None:
    """Anything but one of the three known names is refused and quoted, never defaulted."""
    model_dir = _write_pooling(tmp_path / "model", {"pooling_mode": value})

    with pytest.raises(ValueError) as excinfo:
        common.read_pooling_mode(model_dir)

    message = str(excinfo.value)
    assert common.POOLING_DIRNAME in message
    assert common.POOLING_MODE_KEY in message
    assert repr(value) in message


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ({"word_embedding_dimension": 768, "pooling_mode_mean_tokens": True}, "mean"),
        ({"pooling_mode_mean_tokens": False, "pooling_mode_cls_token": True}, "cls"),
        ({"pooling_mode_lasttoken": True, "pooling_mode_cls_token": False}, "lasttoken"),
    ],
    ids=["mean", "cls", "lasttoken"],
)
def test_a_flag_dialect_only_declaration_is_still_read(
    tmp_path: Path, declaration: dict[str, Any], expected: str
) -> None:
    """The boolean flags of earlier releases keep selecting their mode."""
    model_dir = _write_pooling(tmp_path / "model", declaration)

    assert common.read_pooling_mode(model_dir) == expected


@pytest.mark.parametrize(
    "declaration",
    [
        {"pooling_mode_mean_tokens": True, "pooling_mode_cls_token": True},
        {"pooling_mode_mean_tokens": False, "pooling_mode_cls_token": False},
        {"pooling_mode_max_tokens": True},
        {"pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": True},
        {"pooling_mode_mean_tokens": "true"},
        {"pooling_mode_mean_tokens": 1},
        {"embedding_dimension": 768, "include_prompt": True},
        {},
    ],
    ids=["both", "neither", "unsupported", "mixed", "string-flag", "int-flag", "no-mode", "empty"],
)
def test_an_ambiguous_flag_declaration_is_still_refused(
    tmp_path: Path, declaration: dict[str, Any]
) -> None:
    """The earlier dialect's refusals are unchanged, and the message names both dialects."""
    model_dir = _write_pooling(tmp_path / "model", declaration)

    with pytest.raises(ValueError, match="pooling") as excinfo:
        common.read_pooling_mode(model_dir)

    message = str(excinfo.value)
    assert common.POOLING_DIRNAME in message
    assert "pooling_mode_mean_tokens" in message
    assert common.POOLING_MODE_KEY in message


@pytest.mark.parametrize(
    ("declaration", "expected"),
    [
        ({"pooling_mode": "mean", "pooling_mode_mean_tokens": True}, "mean"),
        (
            {
                "pooling_mode": "cls",
                "pooling_mode_cls_token": True,
                "pooling_mode_mean_tokens": False,
            },
            "cls",
        ),
        ({"pooling_mode": "lasttoken", "pooling_mode_lasttoken": True}, "lasttoken"),
        # A flag left false beside the string takes no side.
        ({"pooling_mode": "mean", "pooling_mode_cls_token": False}, "mean"),
    ],
    ids=["mean", "cls-with-false-flag", "lasttoken", "false-flag-only"],
)
def test_both_dialects_agreeing_are_accepted(
    tmp_path: Path, declaration: dict[str, Any], expected: str
) -> None:
    """A declaration written in both dialects is fine as long as they say the same mode."""
    model_dir = _write_pooling(tmp_path / "model", declaration)

    assert common.read_pooling_mode(model_dir) == expected


def test_both_dialects_disagreeing_are_refused_citing_both(tmp_path: Path) -> None:
    """Two statements of different modes cannot be reconciled by picking one."""
    model_dir = _write_pooling(
        tmp_path / "model", {"pooling_mode": "mean", "pooling_mode_cls_token": True}
    )

    with pytest.raises(ValueError) as excinfo:
        common.read_pooling_mode(model_dir)

    message = str(excinfo.value)
    assert "pooling_mode_cls_token" in message
    assert "'mean'" in message
    assert common.POOLING_DIRNAME in message


@pytest.mark.parametrize(
    "declaration",
    [
        # More than one flag is ambiguous whatever the string says.
        {
            "pooling_mode": "mean",
            "pooling_mode_mean_tokens": True,
            "pooling_mode_cls_token": True,
        },
        # A flag for a mode this backend cannot apply is not made fine by the string.
        {"pooling_mode": "mean", "pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": True},
        {"pooling_mode": "mean", "pooling_mode_max_tokens": True},
        # An unusable string is not made fine by a valid flag.
        {"pooling_mode": "max", "pooling_mode_mean_tokens": True},
        {"pooling_mode": None, "pooling_mode_mean_tokens": True},
        {"pooling_mode": True, "pooling_mode_mean_tokens": True},
    ],
    ids=["two-flags", "unsupported-flag-too", "unsupported-flag-only", "bad-name", "null", "bool"],
)
def test_one_unusable_statement_refuses_the_whole_declaration(
    tmp_path: Path, declaration: dict[str, Any]
) -> None:
    """Each statement must stand on its own; a valid one never rescues an invalid one."""
    model_dir = _write_pooling(tmp_path / "model", declaration)

    with pytest.raises(ValueError, match="pooling"):
        common.read_pooling_mode(model_dir)


def test_the_pooling_mode_key_is_not_one_of_the_flags() -> None:
    """The string key must not be swept up by the scan for ``pooling_mode_<x>`` flags."""
    assert not common.POOLING_MODE_KEY.startswith(common.POOLING_MODE_PREFIX)
