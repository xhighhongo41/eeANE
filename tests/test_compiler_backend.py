"""Tests for the compile-backend interface and its ModernBERT implementation.

Two layers:

* The interface types of ``eeane.compiler.backends.base`` (the handle and
  the sanity specification every backend hands to the pipeline and the
  self-check), which are pure data and run anywhere.
* ``eeane.compiler.backends.modernbert``: the numerics ported from the
  frozen PoC scripts (masked mean pooling, stable sigmoid, patched-eager
  == unpatched-sdpa forward outputs) plus the conformance of every
  registered backend to the interface that pipeline.py drives.

Tests needing the real 310M models skip themselves when
``models/ruri-v3-310m`` / ``models/ruri-v3-reranker-310m`` are absent.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from transformers.models.modernbert import modeling_modernbert

from eeane.compiler.backends import base
from eeane.compiler.backends import modernbert as mb
from eeane.compiler.backends import xlm_roberta as xlmr

_REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL_DIR = _REPO_ROOT / "models" / "ruri-v3-310m"
RERANKER_MODEL_DIR = _REPO_ROOT / "models" / "ruri-v3-reranker-310m"
_EMBEDDING_AVAILABLE = (EMBEDDING_MODEL_DIR / "config.json").exists()
_RERANKER_AVAILABLE = (RERANKER_MODEL_DIR / "config.json").exists()

# Sequence length used by the real-model smoke tests: short enough to keep
# the FP32 forward passes quick, long enough to exercise padding.
SMOKE_SEQ_LEN = 128

# Tolerance between the patched eager path and the untouched sdpa path.
# The rank-4 rewrite is bit-exact against upstream eager (v0.3実装記録
# §6-6); what is left here is the eager-vs-sdpa kernel difference in FP32.
PATCH_ABS_TOLERANCE = 1e-4


@pytest.fixture(autouse=True, scope="module")
def _restore_transformers_patches() -> Iterator[None]:
    """Undo the global ModernBert monkeypatches after this module's tests."""
    original_rotate_half = modeling_modernbert.rotate_half
    original_forward = modeling_modernbert.ModernBertAttention.forward
    yield
    modeling_modernbert.rotate_half = original_rotate_half
    modeling_modernbert.ModernBertAttention.forward = original_forward


def _tiny_model(seed: int = 0) -> torch.nn.Module:
    """Build a randomly initialised ModernBertModel small enough for a unit test."""
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
    torch.manual_seed(seed)
    model = modeling_modernbert.ModernBertModel(config)
    return model.eval()


def _tiny_inputs(seq_len: int = 16, batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic (input_ids, attention_mask) tensors with padding."""
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(1, 128, (batch_size, seq_len), generator=generator)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[-1, seq_len // 2 :] = 0  # last row is half padding
    return input_ids, attention_mask


def _loaded(
    model: Any,
    kind: str = "embedding",
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
        pooling="mean" if kind == "embedding" else None,
    )


# --- interface types (backends/base.py) --------------------------------------


def test_loaded_model_is_frozen_and_defaults_pooling_to_none() -> None:
    """The handle must be immutable and carry no pooling mode unless one is given."""
    loaded = base.LoadedModel(
        model="model",
        tokenizer="tokenizer",
        config="config",
        model_dir=Path("/models/example"),
        kind="reranker",
        attn="eager",
    )

    assert loaded.pooling is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded.kind = "embedding"  # type: ignore[misc]


def test_sanity_spec_defaults_have_no_ordering_expectation() -> None:
    """Fixtures without expected ordering (embeddings) must default both indices to None."""
    spec = base.SanitySpec(inputs=("a", "b"))

    assert spec.relevant_index is None
    assert spec.irrelevant_index is None


def test_sanity_spec_is_frozen_and_holds_immutable_inputs() -> None:
    """Neither the spec nor its inputs may be modified by a consumer."""
    spec = base.SanitySpec(inputs=("a", "b"), relevant_index=0, irrelevant_index=1)

    assert isinstance(spec.inputs, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.relevant_index = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.inputs[0] = "mutated"  # type: ignore[index]


@pytest.mark.parametrize(
    ("relevant", "irrelevant"),
    [(2, 1), (0, 2), (-1, 0), (0, -1)],
)
def test_sanity_spec_rejects_out_of_range_indices(relevant: int, irrelevant: int) -> None:
    """An index that does not address an input would only fail deep inside the self-check."""
    with pytest.raises(ValueError, match="out of range"):
        base.SanitySpec(inputs=("a", "b"), relevant_index=relevant, irrelevant_index=irrelevant)


# --- pooling / sigmoid (the numerics moved out of poc/common.py) -------------


def test_mean_pool_matches_manual_average() -> None:
    """mean_pool must equal the manual average of unmasked positions only."""
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 4)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])

    result = mb.mean_pool(hidden, attention_mask)

    expected = torch.stack([hidden[0].mean(dim=0), hidden[1, :3].mean(dim=0)])
    assert torch.allclose(result, expected, atol=1e-6)


def test_mean_pool_fully_masked_row_stays_finite() -> None:
    """A fully masked row must not divide by zero (the clamp guards it)."""
    hidden = torch.ones(1, 4, 3)
    attention_mask = torch.zeros(1, 4, dtype=torch.long)

    result = mb.mean_pool(hidden, attention_mask)

    assert torch.isfinite(result).all()


def test_sigmoid_np_matches_torch_for_moderate_values() -> None:
    """sigmoid_np must agree with torch.sigmoid on ordinary logits."""
    logits = np.array([-3.0, -0.5, 0.0, 0.5, 3.0], dtype=np.float32)

    result = mb.sigmoid_np(logits)

    expected = torch.sigmoid(torch.from_numpy(logits)).numpy()
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_sigmoid_np_is_stable_for_large_magnitudes() -> None:
    """Large-magnitude logits must saturate without overflow warnings or NaN."""
    logits = np.array([-1000.0, -80.0, 80.0, 1000.0], dtype=np.float32)

    with np.errstate(over="raise"):
        result = mb.sigmoid_np(logits)

    assert np.isfinite(result).all()
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)
    assert np.all((result >= 0.0) & (result <= 1.0))


# --- backend interface conformance ------------------------------------------

# Callable members the interface declares; the parametrized signature test
# below drives itself from this list, so a member added to (or dropped
# from) the protocol is automatically checked against the implementation.
_PROTOCOL_METHODS = sorted(
    name
    for name, member in vars(base.CompileBackend).items()
    if not name.startswith("_") and callable(member)
)

# Every registered backend implementation, so that a new architecture is
# held to the same interface as the existing ones.
_BACKEND_CLASSES = [mb.ModernBertBackend, xlmr.XlmRobertaBackend]


def _parameters(function: Any) -> list[tuple[str, Any]]:
    """Return a function's parameter names and defaults, ignoring annotations."""
    return [(name, param.default) for name, param in inspect.signature(function).parameters.items()]


def test_the_protocol_declares_the_documented_members() -> None:
    """The interface must stay the set of members the module docstring describes."""
    assert set(_PROTOCOL_METHODS) == {
        "load",
        "apply_patches",
        "wrap",
        "output_name",
        "max_seq_len",
        "trace_example",
        "sanity_spec",
        "padding_input",
        "tokenize",
        "reference_outputs",
    }


@pytest.mark.parametrize("backend_class", _BACKEND_CLASSES, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("method", _PROTOCOL_METHODS)
def test_every_backend_matches_the_declared_signature(backend_class: type, method: str) -> None:
    """Every interface member must exist on each backend with the declared parameters."""
    implemented = getattr(backend_class, method, None)

    assert implemented is not None, f"{backend_class.__name__} does not implement {method}()"
    assert _parameters(implemented) == _parameters(getattr(base.CompileBackend, method))


def test_modernbert_backend_declares_the_interface_attributes() -> None:
    """The backend must name itself and the kinds it can compile."""
    backend = mb.ModernBertBackend()

    assert backend.name == "ModernBert"
    assert backend.supported_kinds == ("embedding", "reranker")


def test_output_name_per_kind() -> None:
    """Embeddings and rerankers must keep the PoC's graph output names."""
    backend = mb.ModernBertBackend()

    assert backend.output_name("embedding") == "embedding"
    assert backend.output_name("reranker") == "logits"


def test_fixtures_have_the_expected_shape_per_kind() -> None:
    """trace/sanity/padding fixtures must be texts for embeddings, pairs for rerankers."""
    backend = mb.ModernBertBackend()

    assert isinstance(backend.trace_example("embedding"), str)
    assert all(isinstance(text, str) for text in backend.sanity_spec("embedding").inputs)
    assert backend.padding_input("embedding") == ""

    trace_pair = backend.trace_example("reranker")
    assert isinstance(trace_pair, tuple) and len(trace_pair) == 2
    assert all(len(pair) == 2 for pair in backend.sanity_spec("reranker").inputs)
    assert backend.padding_input("reranker") == ("", "")


def test_sanity_spec_inputs_are_immutable() -> None:
    """The fixtures a caller receives must not be corruptible module state."""
    backend = mb.ModernBertBackend()

    inputs = backend.sanity_spec("embedding").inputs

    assert isinstance(inputs, tuple)
    assert len(inputs) == len(mb.SANITY_TEXTS)
    with pytest.raises(TypeError):
        inputs[0] = "mutated"  # type: ignore[index]


def test_embedding_sanity_spec_declares_no_ordering() -> None:
    """Embedding fixtures are compared row-wise; there is no expected ordering."""
    spec = mb.ModernBertBackend().sanity_spec("embedding")

    assert spec.inputs
    assert spec.relevant_index is None
    assert spec.irrelevant_index is None


def test_reranker_sanity_spec_points_at_the_relevant_and_irrelevant_pairs() -> None:
    """The reranker ordering check must be driven by the spec, not by module constants."""
    spec = mb.ModernBertBackend().sanity_spec("reranker")

    assert spec.relevant_index == 0
    assert spec.irrelevant_index == 1
    assert spec.inputs[0] == mb.SANITY_PAIRS[0]
    assert spec.inputs[1] == mb.SANITY_PAIRS[1]
    # The relevant pair shares its query with the irrelevant one, so only the
    # document decides the expected ordering.
    assert spec.inputs[0][0] == spec.inputs[1][0]
    assert spec.inputs[0][1] != spec.inputs[1][1]


@pytest.mark.parametrize(
    "method",
    ["trace_example", "sanity_spec", "padding_input", "output_name"],
)
def test_unknown_kind_is_rejected(method: str) -> None:
    """Every kind-dispatching method must reject an unsupported kind."""
    backend = mb.ModernBertBackend()

    with pytest.raises(ValueError, match="kind"):
        getattr(backend, method)("classifier")


def test_load_rejects_unknown_kind(tmp_path: Path) -> None:
    """load must validate the kind before touching the filesystem."""
    backend = mb.ModernBertBackend()

    with pytest.raises(ValueError, match="kind"):
        backend.load(tmp_path, "classifier")


@pytest.mark.parametrize("method", ["wrap", "tokenize"])
def test_handle_taking_methods_reject_an_unsupported_kind(method: str) -> None:
    """A handle carrying an unsupported kind must be rejected, not silently wrapped."""
    backend = mb.ModernBertBackend()
    loaded = _loaded(torch.nn.Identity(), kind="classifier")
    arguments = {"tokenize": (["text"], 8)}.get(method, ())

    with pytest.raises(ValueError, match="kind"):
        getattr(backend, method)(loaded, *arguments)


def test_wrap_selects_the_kind_specific_wrapper() -> None:
    """wrap must return the mean-pooling wrapper vs the raw-logits wrapper."""
    backend = mb.ModernBertBackend()
    model = torch.nn.Identity()

    assert isinstance(backend.wrap(_loaded(model, "embedding")), mb.EmbeddingWrapper)
    assert isinstance(backend.wrap(_loaded(model, "reranker")), mb.RerankerWrapper)


def test_apply_patches_rejects_odd_rope_head_dim() -> None:
    """An odd RoPE head dim breaks the rotate_half rewrite and must raise."""
    backend = mb.ModernBertBackend()
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=12, num_attention_heads=4))

    with pytest.raises(ValueError, match="head dim"):
        backend.apply_patches(_loaded(model))


def test_apply_patches_returns_the_applied_patch_record() -> None:
    """The two mandatory rewrites must be reported, without a mask fill entry."""
    backend = mb.ModernBertBackend()
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=32, num_attention_heads=4))

    applied = backend.apply_patches(_loaded(model))

    assert applied == {"rotate_half_static": True, "eager_attention_rank4": True}


def test_apply_patches_records_the_mask_fill_value_when_given() -> None:
    """A given mask_fill_value must be recorded verbatim in the returned record."""
    backend = mb.ModernBertBackend()
    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=32, num_attention_heads=4, local_attention=8),
        _update_attention_mask=lambda *args, **kwargs: None,
    )

    applied = backend.apply_patches(_loaded(model), mask_fill_value=-30000.0)

    assert applied == {
        "rotate_half_static": True,
        "eager_attention_rank4": True,
        "mask_fill_value": -30000.0,
    }


# --- effective maximum sequence length ---------------------------------------


def _write_config(directory: Path, config: dict[str, Any]) -> Path:
    """Write a minimal ``config.json`` into ``directory`` and return the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def test_max_seq_len_reads_max_position_embeddings(tmp_path: Path) -> None:
    """The configured position budget is the effective maximum for this architecture."""
    model_dir = _write_config(
        tmp_path / "model", {"architectures": ["ModernBertModel"], "max_position_embeddings": 8192}
    )

    assert mb.ModernBertBackend().max_seq_len(model_dir) == 8192


def test_max_seq_len_of_a_missing_directory_is_none(tmp_path: Path) -> None:
    """No config.json means no known limit, not a crash."""
    assert mb.ModernBertBackend().max_seq_len(tmp_path / "absent") is None


@pytest.mark.parametrize(
    "config",
    [
        {"architectures": ["ModernBertModel"]},
        {"max_position_embeddings": None},
        {"max_position_embeddings": "8192"},
        {"max_position_embeddings": 0},
        {"max_position_embeddings": -1},
        {"max_position_embeddings": True},
    ],
)
def test_max_seq_len_ignores_a_missing_or_unusable_value(
    tmp_path: Path, config: dict[str, Any]
) -> None:
    """A missing or nonsensical value must degrade to 'unknown', never to a bogus limit."""
    model_dir = _write_config(tmp_path / "model", config)

    assert mb.ModernBertBackend().max_seq_len(model_dir) is None


def test_max_seq_len_of_a_corrupt_config_is_none(tmp_path: Path) -> None:
    """Unparsable JSON must not turn an optional check into a compile failure."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{not json", encoding="utf-8")

    assert mb.ModernBertBackend().max_seq_len(model_dir) is None


# --- patch behaviour on a tiny randomly initialised model --------------------


def test_patched_eager_matches_unpatched_sdpa_on_a_tiny_model() -> None:
    """The patched eager path must reproduce the untouched sdpa outputs."""
    model = _tiny_model()
    input_ids, attention_mask = _tiny_inputs()

    model.config._attn_implementation = "sdpa"
    with torch.no_grad():
        reference = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    mb.ModernBertBackend().apply_patches(_loaded(model))
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        patched = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    assert torch.isfinite(patched).all()
    assert (patched - reference).abs().max().item() < PATCH_ABS_TOLERANCE


def test_patches_are_idempotent() -> None:
    """Applying the patches twice must not recurse or change the outputs."""
    model = _tiny_model()
    input_ids, attention_mask = _tiny_inputs()
    backend = mb.ModernBertBackend()
    loaded = _loaded(model)

    backend.apply_patches(loaded)
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        once = model(input_ids=input_ids, attention_mask=attention_mask)[0]
    backend.apply_patches(loaded)
    with torch.no_grad():
        twice = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    assert torch.equal(once, twice)


def test_rotate_half_patch_matches_the_upstream_formula() -> None:
    """The static-shape rotate_half must return the same values as the sliced one."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 8)
    expected = torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

    mb.patch_rotate_half()

    assert torch.equal(modeling_modernbert.rotate_half(x), expected)


def test_mask_fill_value_patch_produces_finite_masks() -> None:
    """The mask patch must replace -inf-prone fills with the requested finite value."""
    model = _tiny_model()
    fill_value = -30000.0

    mb.patch_mask_fill_value(model, fill_value)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    global_mask, sliding_window_mask = model._update_attention_mask(attention_mask)

    assert global_mask.shape == (1, 1, 1, 4)
    assert sliding_window_mask.shape == (1, 1, 4, 4)
    assert torch.isfinite(global_mask).all()
    assert torch.isfinite(sliding_window_mask).all()
    # Unmasked positions stay at 0.0, masked ones carry exactly fill_value.
    assert global_mask[0, 0, 0].tolist() == [0.0, 0.0, 0.0, fill_value]
    # local_attention=8 -> a +-4 window, so every position is in-window here.
    assert sliding_window_mask[0, 0, 0].tolist() == [0.0, 0.0, 0.0, fill_value]


def test_mask_fill_value_patch_masks_outside_the_sliding_window() -> None:
    """Positions beyond local_attention//2 must be filled even when unmasked."""
    model = _tiny_model()
    model.config.local_attention = 2  # +-1 window
    fill_value = -30000.0

    mb.patch_mask_fill_value(model, fill_value)
    _, sliding_window_mask = model._update_attention_mask(torch.ones(1, 4, dtype=torch.long))

    assert sliding_window_mask[0, 0, 0].tolist() == [0.0, 0.0, fill_value, fill_value]


# --- real-model smoke tests (local only) ------------------------------------


@pytest.fixture(scope="module")
def embedding_smoke() -> dict[str, Any]:
    """Run the patched embedding wrapper and the FP32 sdpa reference once."""
    if not _EMBEDDING_AVAILABLE:
        pytest.skip("ruri-v3-310m model directory not found")
    return _run_smoke(EMBEDDING_MODEL_DIR, "embedding")


@pytest.fixture(scope="module")
def reranker_smoke() -> dict[str, Any]:
    """Run the patched reranker wrapper and the FP32 sdpa reference once."""
    if not _RERANKER_AVAILABLE:
        pytest.skip("ruri-v3-reranker-310m model directory not found")
    return _run_smoke(RERANKER_MODEL_DIR, "reranker")


def _run_smoke(model_dir: Path, kind: str) -> dict[str, Any]:
    """Compare one patched eager forward pass against the FP32 sdpa reference."""
    backend = mb.ModernBertBackend()
    inputs = list(backend.sanity_spec(kind).inputs[:1])
    loaded = backend.load(model_dir, kind, attn="eager")
    backend.apply_patches(loaded)
    wrapper = backend.wrap(loaded)
    tokens = backend.tokenize(loaded, inputs, SMOKE_SEQ_LEN)
    with torch.no_grad():
        patched = wrapper(
            torch.from_numpy(tokens["input_ids"]).long(),
            torch.from_numpy(tokens["attention_mask"]).long(),
        )
    reference = backend.reference_outputs(model_dir, kind, inputs, SMOKE_SEQ_LEN)
    return {
        "tokens": tokens,
        "patched": patched.numpy().reshape(len(inputs), -1),
        "reference": np.asarray(reference, dtype=np.float32).reshape(len(inputs), -1),
        # Only cheap facts about the handle are kept: retaining the handle
        # itself would pin ~1.2 GB of FP32 weights for the whole module.
        "handle": {
            "kind": loaded.kind,
            "attn": loaded.attn,
            "pooling": loaded.pooling,
            "model_dir": loaded.model_dir,
            "training": loaded.model.training,
            "dtype": next(loaded.model.parameters()).dtype,
            "return_dict": loaded.config.return_dict,
            "config_is_model_config": loaded.config is loaded.model.config,
            "tokenizer_encodes": bool(loaded.tokenizer("x")["input_ids"]),
        },
    }


@pytest.mark.parametrize("smoke_fixture", ["embedding_smoke", "reranker_smoke"])
def test_tokenize_returns_fixed_shape_int32_arrays(
    smoke_fixture: str, request: pytest.FixtureRequest
) -> None:
    """Tokenized Core ML inputs must be (N, S) int32 with a non-empty mask."""
    tokens = request.getfixturevalue(smoke_fixture)["tokens"]

    for key in ("input_ids", "attention_mask"):
        assert tokens[key].dtype == np.int32
        assert tokens[key].shape == (1, SMOKE_SEQ_LEN)
    assert tokens["attention_mask"].sum() > 0
    assert tokens["attention_mask"].sum() < SMOKE_SEQ_LEN  # the row really is padded


@pytest.mark.parametrize(
    ("smoke_fixture", "kind", "pooling"),
    [("embedding_smoke", "embedding", "mean"), ("reranker_smoke", "reranker", None)],
)
def test_load_returns_a_conforming_handle(
    smoke_fixture: str, kind: str, pooling: str | None, request: pytest.FixtureRequest
) -> None:
    """load must hand back an eval/FP32/tuple-output model plus its tokenizer and config."""
    handle = request.getfixturevalue(smoke_fixture)["handle"]

    assert handle["kind"] == kind
    assert handle["pooling"] == pooling
    assert handle["attn"] == "eager"
    assert handle["model_dir"].is_dir()
    assert handle["training"] is False
    assert handle["dtype"] == torch.float32
    assert handle["return_dict"] is False
    assert handle["config_is_model_config"] is True
    assert handle["tokenizer_encodes"] is True


@pytest.mark.parametrize("smoke_fixture", ["embedding_smoke", "reranker_smoke"])
def test_patched_forward_matches_fp32_reference(
    smoke_fixture: str, request: pytest.FixtureRequest
) -> None:
    """Patched eager outputs must stay within tolerance of the FP32 sdpa baseline."""
    smoke = request.getfixturevalue(smoke_fixture)

    assert np.isfinite(smoke["patched"]).all()
    assert np.abs(smoke["patched"] - smoke["reference"]).max() < PATCH_ABS_TOLERANCE
