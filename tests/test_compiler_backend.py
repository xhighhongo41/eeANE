"""Tests for the compile-backend interface and its ModernBERT implementation.

Three layers:

* The interface types of ``eeane.compiler.backends.base`` (the handle and
  the sanity specification every backend hands to the pipeline and the
  self-check), which are pure data and run anywhere.
* The per-language sanity fixtures of
  ``eeane.compiler.backends.common``, checked through every backend that
  serves them, since the self-check's "any set may carry the variant"
  rule only holds while each set really covers its own language.
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
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast
from transformers.models.modernbert import modeling_modernbert

from eeane.compiler.backends import base, bert, common
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
# The rank-4 rewrite is bit-exact against upstream eager; what is left
# here is the eager-vs-sdpa kernel difference in FP32.
PATCH_ABS_TOLERANCE = 1e-4


@pytest.fixture(autouse=True, scope="module")
def _restore_transformers_patches() -> Iterator[None]:
    """Undo the global ModernBert monkeypatches after this module's tests."""
    original_rotate_half = modeling_modernbert.rotate_half
    original_forward = modeling_modernbert.ModernBertAttention.forward
    yield
    modeling_modernbert.rotate_half = original_rotate_half
    modeling_modernbert.ModernBertAttention.forward = original_forward


def _tiny_model(seed: int = 0, vocab_size: int = 128) -> torch.nn.Module:
    """Build a randomly initialised ModernBertModel small enough for a unit test."""
    config = modeling_modernbert.ModernBertConfig(
        vocab_size=vocab_size,
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
    pooling: str | None = "mean",
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
    _write_json(model_dir / mb.CONFIG_FILENAME, config or {"architectures": ["ModernBertModel"]})
    if pooling:
        _write_json(model_dir / mb.POOLING_DIRNAME / "config.json", pooling)
    return model_dir


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
    spec = base.SanitySpec(input_sets=(("en", ("a", "b")),))

    assert spec.relevant_index is None
    assert spec.irrelevant_index is None


def test_sanity_spec_is_frozen_and_holds_immutable_inputs() -> None:
    """Neither the spec nor its sets may be modified by a consumer."""
    spec = base.SanitySpec(
        input_sets=(("en", ("a", "b")), ("ja", ("c", "d"))),
        relevant_index=0,
        irrelevant_index=1,
    )

    assert isinstance(spec.input_sets, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.relevant_index = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        spec.input_sets[0][1][0] = "mutated"  # type: ignore[index]


def test_sanity_spec_reports_its_languages_in_declaration_order() -> None:
    """The evaluation order of the sets is part of the contract, so it must be readable."""
    spec = base.SanitySpec(input_sets=(("en", ("a",)), ("ja", ("b",)), ("zh", ("c",))))

    assert spec.languages == ("en", "ja", "zh")


def test_sanity_spec_all_inputs_concatenates_the_sets_in_order() -> None:
    """The flat view a language-agnostic consumer takes must keep every set's inputs."""
    spec = base.SanitySpec(input_sets=(("en", ("a", "b")), ("ja", ("c",)), ("zh", ("d", "e"))))

    assert spec.all_inputs == ("a", "b", "c", "d", "e")


@pytest.mark.parametrize(
    ("relevant", "irrelevant"),
    [(2, 1), (0, 2), (-1, 0), (0, -1)],
)
def test_sanity_spec_rejects_out_of_range_indices(relevant: int, irrelevant: int) -> None:
    """An index that does not address an input would only fail deep inside the self-check."""
    with pytest.raises(ValueError, match="out of range"):
        base.SanitySpec(
            input_sets=(("en", ("a", "b")),),
            relevant_index=relevant,
            irrelevant_index=irrelevant,
        )


def test_sanity_spec_rejects_an_index_out_of_range_for_a_later_set() -> None:
    """The indices apply to every set, so validating only the first one would miss this."""
    with pytest.raises(ValueError, match="out of range"):
        base.SanitySpec(
            input_sets=(("en", ("a", "b", "c")), ("ja", ("d", "e"))),
            relevant_index=0,
            irrelevant_index=2,
        )


def test_sanity_spec_rejects_a_repeated_language() -> None:
    """Two sets under one language would overwrite each other in the per-set report."""
    with pytest.raises(ValueError, match="ja"):
        base.SanitySpec(input_sets=(("ja", ("a",)), ("ja", ("b",))))


def test_sanity_spec_rejects_declaring_no_set_at_all() -> None:
    """A spec without a single set leaves the self-check nothing to compare."""
    with pytest.raises(ValueError, match="no sanity input set"):
        base.SanitySpec(input_sets=())


def test_sanity_spec_rejects_an_empty_set() -> None:
    """An empty set would contribute no row, yet still be reported as a passing set."""
    with pytest.raises(ValueError, match="zh"):
        base.SanitySpec(input_sets=(("en", ("a",)), ("zh", ())))


def test_sanity_spec_rejects_inputs_that_are_not_a_tuple() -> None:
    """A mutable input sequence would let a consumer change the fixtures under the check."""
    with pytest.raises(TypeError, match="tuple"):
        base.SanitySpec(input_sets=(("en", ["a", "b"]),))  # type: ignore[arg-type]


# --- shared sanity language sets (backends/common.py) ------------------------

# Every (backend, kind) pair that hands the pipeline a sanity
# specification, so the shared expectations below are checked against all
# of them rather than one representative.
_SANITY_SPEC_CASES = [
    (bert.BertBackend(), "embedding"),
    (mb.ModernBertBackend(), "embedding"),
    (mb.ModernBertBackend(), "reranker"),
    (xlmr.XlmRobertaBackend(), "embedding"),
    (xlmr.XlmRobertaBackend(), "reranker"),
]
_SANITY_SPEC_IDS = [f"{backend.name}-{kind}" for backend, kind in _SANITY_SPEC_CASES]

# The same cases split by kind, for the expectations only one kind's
# fixtures can carry.
_EMBEDDING_BACKENDS = [backend for backend, kind in _SANITY_SPEC_CASES if kind == "embedding"]
_RERANKER_BACKENDS = [backend for backend, kind in _SANITY_SPEC_CASES if kind == "reranker"]

# Number of inputs every language set holds: one short, one medium and one
# long fixture, so a single fixed sequence length exercises three different
# amounts of padding.
_INPUTS_PER_SANITY_SET = 3


@pytest.mark.parametrize(("backend", "kind"), _SANITY_SPEC_CASES, ids=_SANITY_SPEC_IDS)
def test_every_sanity_spec_offers_the_three_languages_in_one_order(backend: Any, kind: str) -> None:
    """A model whose vocabulary misses one language must still have two other sets to pass on."""
    assert backend.sanity_spec(kind).languages == ("en", "ja", "zh")


@pytest.mark.parametrize(("backend", "kind"), _SANITY_SPEC_CASES, ids=_SANITY_SPEC_IDS)
def test_every_sanity_set_holds_three_inputs(backend: Any, kind: str) -> None:
    """Every set must exercise the same short/medium/long shape, whatever its language."""
    spec = backend.sanity_spec(kind)

    assert all(len(inputs) == _INPUTS_PER_SANITY_SET for _, inputs in spec.input_sets)
    assert len(spec.all_inputs) == _INPUTS_PER_SANITY_SET * len(spec.input_sets)


@pytest.mark.parametrize("backend", _EMBEDDING_BACKENDS, ids=lambda backend: backend.name)
def test_every_embedding_sanity_set_runs_from_the_shortest_to_the_longest(backend: Any) -> None:
    """The padding an input leaves unused is what the three lengths per set exercise.

    Only the embedding sets: a reranker's three pairs are ordered by the
    role each plays in the expected ordering, not by length.
    """
    for language, texts in backend.sanity_spec("embedding").input_sets:
        lengths = [len(text) for text in texts]
        assert lengths == sorted(lengths), (language, lengths)


@pytest.mark.parametrize(("backend", "kind"), _SANITY_SPEC_CASES, ids=_SANITY_SPEC_IDS)
def test_no_sanity_input_is_empty_or_repeated(backend: Any, kind: str) -> None:
    """A blank or duplicated fixture would spend a prediction on nothing new."""
    inputs = backend.sanity_spec(kind).all_inputs

    assert all(all(part.strip() for part in _parts(item)) for item in inputs)
    assert len(set(inputs)) == len(inputs)


def test_the_english_sanity_sets_are_written_in_ascii() -> None:
    """The English sets are what carries a checkpoint with an English-only vocabulary."""
    english_texts = dict(common.SANITY_TEXT_SETS)["en"]
    english_pairs = dict(common.SANITY_PAIR_SETS)["en"]

    assert all(text.isascii() for text in english_texts), english_texts
    assert all(part.isascii() for pair in english_pairs for part in pair), english_pairs


@pytest.mark.parametrize("language", ["ja", "zh"])
def test_the_non_english_sanity_sets_are_not_ascii(language: str) -> None:
    """A set that lost its own script would stop covering the vocabulary it stands for."""
    texts = dict(common.SANITY_TEXT_SETS)[language]
    pairs = dict(common.SANITY_PAIR_SETS)[language]

    assert not any(text.isascii() for text in texts), texts
    assert not any(pair[1].isascii() for pair in pairs), pairs


@pytest.mark.parametrize("backend", _RERANKER_BACKENDS, ids=lambda backend: backend.name)
def test_every_reranker_pair_set_shares_the_query_of_its_first_two_pairs(backend: Any) -> None:
    """Only the document may decide the expected ordering, in every language."""
    spec = backend.sanity_spec("reranker")
    relevant, irrelevant = spec.relevant_index, spec.irrelevant_index

    for language, pairs in spec.input_sets:
        assert pairs[relevant][0] == pairs[irrelevant][0], language
        assert pairs[relevant][1] != pairs[irrelevant][1], language


def test_the_shared_sets_are_what_the_multilingual_backends_serve() -> None:
    """XLM-RoBERTa and BERT take the shared fixtures; nothing is duplicated per backend."""
    assert xlmr.XlmRobertaBackend().sanity_spec("embedding").input_sets == common.SANITY_TEXT_SETS
    assert xlmr.XlmRobertaBackend().sanity_spec("reranker").input_sets == common.SANITY_PAIR_SETS
    assert bert.BertBackend().sanity_spec("embedding").input_sets == common.SANITY_TEXT_SETS


@pytest.mark.parametrize("kind", ["embedding", "reranker"])
def test_modernbert_overrides_the_japanese_set_only(kind: str) -> None:
    """Its Japanese fixtures are the ones its verified models were measured with."""
    shared = dict(common.SANITY_TEXT_SETS if kind == "embedding" else common.SANITY_PAIR_SETS)
    own = tuple(mb.SANITY_TEXTS if kind == "embedding" else mb.SANITY_PAIRS)

    sets = dict(mb.ModernBertBackend().sanity_spec(kind).input_sets)

    assert sets["ja"] == own
    assert sets["ja"] != shared["ja"]
    assert sets["en"] == shared["en"]
    assert sets["zh"] == shared["zh"]


def test_override_sanity_set_replaces_only_the_named_language() -> None:
    """A backend with its own fixtures for one language must keep the shared rest."""
    sets = (("en", ("a",)), ("ja", ("b",)), ("zh", ("c",)))

    replaced = common.override_sanity_set(sets, "ja", ("own",))

    assert replaced == (("en", ("a",)), ("ja", ("own",)), ("zh", ("c",)))


def test_override_sanity_set_rejects_a_language_the_sets_do_not_hold() -> None:
    """A typo would otherwise silently leave the shared fixtures in place."""
    with pytest.raises(ValueError, match="ko"):
        common.override_sanity_set((("en", ("a",)),), "ko", ("own",))


def _parts(sanity_input: Any) -> tuple[str, ...]:
    """Return the text parts of one sanity input (a text, or a pair's two halves)."""
    return (sanity_input,) if isinstance(sanity_input, str) else tuple(sanity_input)


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
_BACKEND_CLASSES = [bert.BertBackend, mb.ModernBertBackend, xlmr.XlmRobertaBackend]


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
    assert all(isinstance(text, str) for text in backend.sanity_spec("embedding").all_inputs)
    assert backend.padding_input("embedding") == ""

    trace_pair = backend.trace_example("reranker")
    assert isinstance(trace_pair, tuple) and len(trace_pair) == 2
    assert all(len(pair) == 2 for pair in backend.sanity_spec("reranker").all_inputs)
    assert backend.padding_input("reranker") == ("", "")


def test_sanity_spec_inputs_are_immutable() -> None:
    """The fixtures a caller receives must not be corruptible module state."""
    backend = mb.ModernBertBackend()

    japanese = dict(backend.sanity_spec("embedding").input_sets)["ja"]

    assert isinstance(japanese, tuple)
    assert len(japanese) == len(mb.SANITY_TEXTS)
    with pytest.raises(TypeError):
        japanese[0] = "mutated"  # type: ignore[index]


def test_embedding_sanity_spec_declares_no_ordering() -> None:
    """Embedding fixtures are compared row-wise; there is no expected ordering."""
    spec = mb.ModernBertBackend().sanity_spec("embedding")

    assert spec.input_sets
    assert spec.relevant_index is None
    assert spec.irrelevant_index is None


def test_reranker_sanity_spec_points_at_the_relevant_and_irrelevant_pairs() -> None:
    """The reranker ordering check must be driven by the spec, not by module constants."""
    spec = mb.ModernBertBackend().sanity_spec("reranker")
    japanese = dict(spec.input_sets)["ja"]

    assert spec.relevant_index == 0
    assert spec.irrelevant_index == 1
    assert japanese[0] == mb.SANITY_PAIRS[0]
    assert japanese[1] == mb.SANITY_PAIRS[1]


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


def test_wrap_selects_the_wrapper_of_the_detected_pooling() -> None:
    """The pooling recorded on the handle must decide which wrapper is traced."""
    backend = mb.ModernBertBackend()
    model = torch.nn.Identity()

    mean_wrapper = backend.wrap(_loaded(model, "embedding", pooling="mean"))
    cls_wrapper = backend.wrap(_loaded(model, "embedding", pooling="cls"))
    reranker_wrapper = backend.wrap(_loaded(model, "reranker"))

    assert isinstance(mean_wrapper, mb.EmbeddingWrapper)
    assert isinstance(cls_wrapper, mb.ClsEmbeddingWrapper)
    assert isinstance(reranker_wrapper, mb.RerankerWrapper)
    assert not any(wrapper.training for wrapper in (mean_wrapper, cls_wrapper, reranker_wrapper))


@pytest.mark.parametrize("pooling", [None, "max", "lasttoken", ""])
def test_wrap_rejects_a_pooling_no_wrapper_implements(pooling: str | None) -> None:
    """An embedding handle with an unknown pooling must raise, not fall back to mean."""
    backend = mb.ModernBertBackend()

    with pytest.raises(ValueError, match="pooling"):
        backend.wrap(_loaded(torch.nn.Identity(), "embedding", pooling=pooling))


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

    assert mb.read_pooling_mode(model_dir) == expected


def test_read_pooling_mode_without_a_pooling_module_explains_what_is_missing(
    tmp_path: Path,
) -> None:
    """A missing declaration must name the file and the accepted flags, not default."""
    model_dir = _model_dir(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        mb.read_pooling_mode(model_dir)

    message = str(excinfo.value)
    assert mb.POOLING_DIRNAME in message
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
        mb.read_pooling_mode(model_dir)


def test_read_pooling_mode_rejects_a_corrupt_declaration(tmp_path: Path) -> None:
    """Unparsable JSON must surface as a ValueError naming the file."""
    model_dir = _model_dir(tmp_path)
    pooling_config = model_dir / mb.POOLING_DIRNAME / "config.json"
    pooling_config.parent.mkdir(parents=True, exist_ok=True)
    pooling_config.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match=mb.POOLING_DIRNAME):
        mb.read_pooling_mode(model_dir)


def test_read_pooling_mode_rejects_a_non_object_declaration(tmp_path: Path) -> None:
    """A JSON document that is not an object cannot declare a pooling mode."""
    model_dir = _model_dir(tmp_path)
    _write_json(model_dir / mb.POOLING_DIRNAME / "config.json", ["mean"])

    with pytest.raises(ValueError, match="pooling"):
        mb.read_pooling_mode(model_dir)


def test_load_reports_an_undeclared_pooling_before_loading_weights(tmp_path: Path) -> None:
    """An embedding model without a pooling declaration must fail fast and clearly."""
    model_dir = _model_dir(tmp_path)

    with pytest.raises(ValueError, match="pooling"):
        mb.ModernBertBackend().load(model_dir, "embedding")


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


def _run_smoke(model_dir: Path, kind: str, seq_len: int = SMOKE_SEQ_LEN) -> dict[str, Any]:
    """Compare one patched eager forward pass against the FP32 sdpa reference."""
    backend = mb.ModernBertBackend()
    inputs = list(backend.sanity_spec(kind).all_inputs[:1])
    loaded = backend.load(model_dir, kind, attn="eager")
    backend.apply_patches(loaded)
    wrapper = backend.wrap(loaded)
    tokens = backend.tokenize(loaded, inputs, seq_len)
    with torch.no_grad():
        patched = wrapper(
            torch.from_numpy(tokens["input_ids"]).long(),
            torch.from_numpy(tokens["attention_mask"]).long(),
        )
    reference = backend.reference_outputs(model_dir, kind, inputs, seq_len)
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


# --- synthetic pooling round trip (always runs) -------------------------------

# Sequence length for the synthetic round-trip test: well within the tiny
# model's small max_position_embeddings, unlike SMOKE_SEQ_LEN.
_SYNTHETIC_SMOKE_SEQ_LEN = 16


def _write_model_directory(directory: Path, pooling_flag: str) -> Path:
    """Save a tiny randomly initialised embedding model as a HuggingFace directory.

    Args:
        directory: Destination directory; created if needed.
        pooling_flag: sentence-transformers pooling flag to enable.

    Returns:
        ``directory``, holding weights, config, tokenizer files and the
        pooling declaration.
    """
    directory.mkdir(parents=True, exist_ok=True)
    # The byte-level tokenizer below can emit up to ~260 distinct token ids
    # (256 bytes + 4 special tokens), so the vocabulary must be larger than
    # the default _tiny_model() size to keep every id addressable.
    _tiny_model(vocab_size=300).save_pretrained(directory)

    # Byte-level vocabulary with no merges: every byte is its own token, so
    # Japanese fixtures tokenize without shipping a real vocab file.
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
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        model_max_length=64,
    ).save_pretrained(directory)

    _write_json(
        directory / mb.POOLING_DIRNAME / "config.json",
        {"word_embedding_dimension": 32, pooling_flag: True},
    )
    return directory


@pytest.fixture(scope="module")
def cls_embedding_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic ModernBERT embedding model directory declaring CLS pooling."""
    return _write_model_directory(
        tmp_path_factory.mktemp("modernbert-embedding-cls"), pooling_flag="pooling_mode_cls_token"
    )


def test_synthetic_cls_embedding_model_wrap_matches_the_fp32_reference(
    cls_embedding_dir: Path,
) -> None:
    """A directory declaring CLS pooling must wire that pooling through wrap and the baseline.

    Unlike the real-model smoke tests below (gated to a local machine and
    only ever exercising ruri-v3-310m's mean pooling), this runs against a
    tiny synthetic directory and therefore always runs. If either wrap() or
    reference_outputs() silently fell back to mean pooling, the two sides
    would disagree well beyond the patch-vs-reference kernel tolerance.
    """
    smoke = _run_smoke(cls_embedding_dir, "embedding", seq_len=_SYNTHETIC_SMOKE_SEQ_LEN)

    assert smoke["handle"]["pooling"] == "cls"
    assert np.isfinite(smoke["patched"]).all()
    assert np.abs(smoke["patched"] - smoke["reference"]).max() < PATCH_ABS_TOLERANCE


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
    """load must hand back an eval/FP32/tuple-output model plus its tokenizer and config.

    ``pooling`` for the embedding case is not a hard-coded constant: it is
    whatever load() detected from ruri-v3-310m's own
    ``1_Pooling/config.json`` (which declares mean pooling), so this also
    doubles as a real-weight check that the detection is wired correctly.
    """
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
