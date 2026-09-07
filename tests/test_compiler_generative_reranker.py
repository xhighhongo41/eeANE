"""Tests for the compile-side pieces a generative reranker adds.

A generative reranker carries no scoring head: it asks a causal language
model a templated yes/no question and subtracts two vocabulary logits at
the final position. Three architecture-independent pieces make that
compilable, and they are what this module covers:

* the readers for what such a model directory declares -- the two
  vocabulary ids, the two-module chain, and the prompt the model applies
  by default (:mod:`eeane.compiler.backends.common`);
* the compile-side templated pair encoding, checked against the runtime's
  own template path: the two must build byte-identical rows, or the model
  would be converted for a different input than it is served with;
* the frozen-tokenizer gate on that same template
  (:mod:`eeane.compiler.tokenizer_freeze`).

Nothing here downloads weights: the tokenizer is a byte-level toy built
in-test and the FP32 baseline is driven by a stand-in model returning
fixed logits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from eeane import runtime
from eeane.compiler import tokenizer_freeze
from eeane.compiler.backends import common
from eeane.runtime import PairTemplate

# A template short enough to leave room for a body inside the small
# buckets used below, and carrying both markers exactly once.
_TEMPLATE = PairTemplate(
    prefix="<<begin>>\n",
    body_format="Q: {query}\nD: {document}",
    suffix="\n<<answer>>",
)

# Bucket used by the templated-encoding tests. Long enough to hold the
# template plus a few body tokens, short enough that a longer body is
# visibly truncated.
_SEQ_LEN = 48

# Vocabulary ids a synthetic scoring declaration names.
_TRUE_ID = 11
_FALSE_ID = 4


def _write_json(path: Path, payload: object) -> None:
    """Write ``payload`` as JSON, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_logit_score(model_dir: Path, declaration: object) -> Path:
    """Write a scoring declaration into ``model_dir`` and return the directory."""
    _write_json(
        model_dir / common.LOGIT_SCORE_DIRNAME / common.LOGIT_SCORE_CONFIG_FILENAME, declaration
    )
    return model_dir


def _write_modules(model_dir: Path, types: list[str]) -> Path:
    """Write a module declaration naming ``types``, in order."""
    _write_json(
        model_dir / common.ST_MODULES_FILENAME,
        [
            {"idx": index, "name": str(index), "path": "", "type": name}
            for index, name in enumerate(types)
        ],
    )
    return model_dir


def _write_tokenizer(model_dir: Path) -> Path:
    """Save a byte-level toy tokenizer as a HuggingFace tokenizer directory.

    Every byte is its own token and no merge is declared, so any UTF-8
    text encodes deterministically without shipping a vocabulary file.

    Args:
        model_dir: Directory the tokenizer files are written to.

    Returns:
        ``model_dir``.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for index, character in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet())):
        vocab[character] = index + 3
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.post_processor = processors.TemplateProcessing(
        single="$A <eos>",
        pair="$A <eos> $B <eos>",
        special_tokens=[("<eos>", 2)],
    )
    backend.decoder = decoders.ByteLevel()
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
        model_max_length=4096,
    ).save_pretrained(model_dir)
    return model_dir


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Model directory holding the toy tokenizer, built once for the module."""
    return _write_tokenizer(tmp_path_factory.mktemp("toy-tokenizer"))


@pytest.fixture(scope="module")
def frozen_path(tokenizer_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Frozen ``tokenizer.json`` of the toy tokenizer."""
    out_path = tmp_path_factory.mktemp("toy-frozen") / "tokenizer.json"
    tokenizer_freeze.freeze_tokenizer(tokenizer_dir, out_path)
    return out_path


# --- the scoring declaration --------------------------------------------------


def test_read_logit_score_returns_the_declared_ids(tmp_path: Path) -> None:
    """The two ids must be read from the declaration, true first."""
    model_dir = _write_logit_score(
        tmp_path / "model",
        {common.LOGIT_SCORE_TRUE_KEY: _TRUE_ID, common.LOGIT_SCORE_FALSE_KEY: _FALSE_ID},
    )

    assert common.read_logit_score(model_dir) == (_TRUE_ID, _FALSE_ID)


def test_read_logit_score_accepts_zero_as_an_id(tmp_path: Path) -> None:
    """Zero is a valid vocabulary index and must not be mistaken for "unset"."""
    model_dir = _write_logit_score(
        tmp_path / "model",
        {common.LOGIT_SCORE_TRUE_KEY: 0, common.LOGIT_SCORE_FALSE_KEY: 7},
    )

    assert common.read_logit_score(model_dir) == (0, 7)


def test_read_logit_score_without_a_declaration_explains_what_is_missing(tmp_path: Path) -> None:
    """A model with no declaration must be refused with an actionable message."""
    with pytest.raises(ValueError) as excinfo:
        common.read_logit_score(tmp_path / "model")

    message = str(excinfo.value)
    assert common.LOGIT_SCORE_DIRNAME in message
    assert common.LOGIT_SCORE_TRUE_KEY in message
    assert common.LOGIT_SCORE_FALSE_KEY in message


def test_read_logit_score_rejects_a_corrupt_declaration(tmp_path: Path) -> None:
    """Unparsable JSON must raise, not be treated as an absent declaration."""
    path = tmp_path / "model" / common.LOGIT_SCORE_DIRNAME / common.LOGIT_SCORE_CONFIG_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        common.read_logit_score(tmp_path / "model")


def test_read_logit_score_rejects_a_non_object_declaration(tmp_path: Path) -> None:
    """A JSON list carries no keys to read the two ids from."""
    model_dir = _write_logit_score(tmp_path / "model", [1, 2])

    with pytest.raises(ValueError):
        common.read_logit_score(model_dir)


@pytest.mark.parametrize(
    "declaration",
    [
        {common.LOGIT_SCORE_FALSE_KEY: 2},
        {common.LOGIT_SCORE_TRUE_KEY: 9},
        {},
        {common.LOGIT_SCORE_TRUE_KEY: 9, common.LOGIT_SCORE_FALSE_KEY: None},
        {common.LOGIT_SCORE_TRUE_KEY: "9", common.LOGIT_SCORE_FALSE_KEY: 2},
        {common.LOGIT_SCORE_TRUE_KEY: 9.0, common.LOGIT_SCORE_FALSE_KEY: 2},
        {common.LOGIT_SCORE_TRUE_KEY: True, common.LOGIT_SCORE_FALSE_KEY: 2},
        {common.LOGIT_SCORE_TRUE_KEY: 9, common.LOGIT_SCORE_FALSE_KEY: -1},
    ],
    ids=["no-true", "no-false", "empty", "null", "string", "float", "bool", "negative"],
)
def test_read_logit_score_rejects_an_unusable_id(
    tmp_path: Path, declaration: dict[str, Any]
) -> None:
    """Anything but two non-negative integers must be refused, not coerced."""
    model_dir = _write_logit_score(tmp_path / "model", declaration)

    with pytest.raises(ValueError):
        common.read_logit_score(model_dir)


def test_read_logit_score_rejects_two_identical_ids(tmp_path: Path) -> None:
    """Scoring an id against itself is a constant zero, never a relevance signal."""
    model_dir = _write_logit_score(
        tmp_path / "model",
        {common.LOGIT_SCORE_TRUE_KEY: 5, common.LOGIT_SCORE_FALSE_KEY: 5},
    )

    with pytest.raises(ValueError):
        common.read_logit_score(model_dir)


# --- the module chain of a generative reranker --------------------------------


@pytest.mark.parametrize(
    "types",
    [
        ["sentence_transformers.models.Transformer", "sentence_transformers.models.LogitScore"],
        [
            "sentence_transformers.base.modules.transformer.Transformer",
            "sentence_transformers.cross_encoder.modules.logit_score.LogitScore",
        ],
    ],
    ids=["short-namespace", "nested-namespace"],
)
def test_check_scoring_module_chain_accepts_either_namespace(
    tmp_path: Path, types: list[str]
) -> None:
    """The same two roles are published under more than one namespace."""
    model_dir = _write_modules(tmp_path / "model", types)

    common.check_scoring_module_chain(model_dir)  # must not raise


@pytest.mark.parametrize(
    "types",
    [
        [],
        ["sentence_transformers.models.Transformer"],
        ["sentence_transformers.models.LogitScore"],
        ["sentence_transformers.models.LogitScore", "sentence_transformers.models.Transformer"],
        ["sentence_transformers.models.Transformer", "sentence_transformers.models.Pooling"],
        [
            "sentence_transformers.models.Transformer",
            "sentence_transformers.models.Dense",
            "sentence_transformers.models.LogitScore",
        ],
        [
            "sentence_transformers.models.Transformer",
            "sentence_transformers.models.LogitScore",
            "sentence_transformers.models.Normalize",
        ],
        ["my.TransformerLike", "sentence_transformers.models.LogitScore"],
    ],
    ids=[
        "empty",
        "transformer-only",
        "score-only",
        "reversed",
        "pooling-instead",
        "extra-in-between",
        "trailing-extra",
        "suffix-must-match-the-class-name",
    ],
)
def test_check_scoring_module_chain_rejects_any_other_chain(
    tmp_path: Path, types: list[str]
) -> None:
    """A chain this backend does not reproduce must be refused before any weight is read."""
    model_dir = _write_modules(tmp_path / "model", types)

    with pytest.raises(ValueError):
        common.check_scoring_module_chain(model_dir)


def test_check_scoring_module_chain_rejects_a_missing_declaration(tmp_path: Path) -> None:
    """No declaration at all means the model never said it is a reranker."""
    with pytest.raises(ValueError) as excinfo:
        common.check_scoring_module_chain(tmp_path / "model")

    # The error must describe the chain this caller reproduces, not the
    # one an embedding model declares.
    message = str(excinfo.value)
    assert common.ST_LOGIT_SCORE_SUFFIX in message
    assert common.ST_MODULE_POOLING not in message


def test_check_scoring_module_chain_rejects_a_corrupt_declaration(tmp_path: Path) -> None:
    """Unparsable JSON must raise rather than silently accept the chain."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / common.ST_MODULES_FILENAME).write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError):
        common.check_scoring_module_chain(model_dir)


# --- the declared default prompt ----------------------------------------------


def test_read_default_prompt_reads_the_prompt_the_default_name_addresses(tmp_path: Path) -> None:
    """The declared default name selects which of the named prompts applies."""
    model_dir = tmp_path / "model"
    _write_json(
        model_dir / common.ST_CONFIG_FILENAME,
        {
            common.ST_DEFAULT_PROMPT_NAME_KEY: "retrieval",
            common.ST_PROMPTS_KEY: {"retrieval": "Find the passage", "other": "Something else"},
        },
    )

    assert common.read_default_prompt(model_dir) == "Find the passage"


@pytest.mark.parametrize(
    "declaration",
    [
        {common.ST_PROMPTS_KEY: {common.ST_FALLBACK_PROMPT_NAME: "Find the passage"}},
        {
            common.ST_DEFAULT_PROMPT_NAME_KEY: None,
            common.ST_PROMPTS_KEY: {common.ST_FALLBACK_PROMPT_NAME: "Find the passage"},
        },
    ],
    ids=["absent", "null"],
)
def test_read_default_prompt_falls_back_to_the_conventional_name(
    tmp_path: Path, declaration: dict[str, Any]
) -> None:
    """A directory naming no default is still readable through the usual name."""
    model_dir = tmp_path / "model"
    _write_json(model_dir / common.ST_CONFIG_FILENAME, declaration)

    assert common.read_default_prompt(model_dir) == "Find the passage"


@pytest.mark.parametrize(
    "declaration",
    [
        {},
        {common.ST_PROMPTS_KEY: {}},
        {common.ST_PROMPTS_KEY: {"other": "Something else"}},
        {common.ST_DEFAULT_PROMPT_NAME_KEY: "missing", common.ST_PROMPTS_KEY: {"other": "x"}},
        {common.ST_DEFAULT_PROMPT_NAME_KEY: "query", common.ST_PROMPTS_KEY: {"query": 3}},
        {common.ST_DEFAULT_PROMPT_NAME_KEY: 7, common.ST_PROMPTS_KEY: {"query": "x"}},
        {common.ST_PROMPTS_KEY: "not a table"},
        [1, 2, 3],
    ],
    ids=[
        "no-prompts",
        "empty-prompts",
        "no-matching-name",
        "default-name-not-declared",
        "prompt-not-a-string",
        "default-name-not-a-string",
        "prompts-not-a-table",
        "not-an-object",
    ],
)
def test_read_default_prompt_reports_nothing_it_cannot_read(
    tmp_path: Path, declaration: Any
) -> None:
    """An unreadable declaration is an unanswered question here, not an error."""
    model_dir = tmp_path / "model"
    _write_json(model_dir / common.ST_CONFIG_FILENAME, declaration)

    assert common.read_default_prompt(model_dir) is None


def test_read_default_prompt_of_a_missing_file_is_none(tmp_path: Path) -> None:
    """Most model directories carry no such file at all."""
    assert common.read_default_prompt(tmp_path / "model") is None


def test_read_default_prompt_of_a_corrupt_file_is_none(tmp_path: Path) -> None:
    """A corrupt file must not turn into an exception from a best-effort reader."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / common.ST_CONFIG_FILENAME).write_text("{not json", encoding="utf-8")

    assert common.read_default_prompt(model_dir) is None


# --- the compile-side templated pair encoding ---------------------------------

_PAIRS: list[tuple[str, str]] = [
    ("short query", "short document"),
    ("", ""),
    ("a query with {braces} in it", "a document with {query} written inside it"),
    ("long query " * 20, "long document " * 40),
]


def test_templated_pairs_match_the_runtime_row_for_row(
    tokenizer_dir: Path, frozen_path: Path
) -> None:
    """The compiled model and the served request must see the very same tokens."""
    reference = AutoTokenizer.from_pretrained(tokenizer_dir)
    frozen = runtime.load_frozen_tokenizer(frozen_path)
    prepared = runtime.prepare_pair_template(frozen, _TEMPLATE)

    compile_side = common.tokenize_templated_pairs(reference, _PAIRS, _SEQ_LEN, _TEMPLATE)
    serve_side = runtime.tokenize_pairs(frozen, _PAIRS, _SEQ_LEN, prepared)

    np.testing.assert_array_equal(compile_side["input_ids"], serve_side["input_ids"])
    np.testing.assert_array_equal(compile_side["attention_mask"], serve_side["attention_mask"])


def test_templated_pairs_are_fixed_shape_int32_arrays(tokenizer_dir: Path) -> None:
    """Core ML takes exactly two int32 arrays of the bucket's shape."""
    reference = AutoTokenizer.from_pretrained(tokenizer_dir)

    tokens = common.tokenize_templated_pairs(reference, _PAIRS, _SEQ_LEN, _TEMPLATE)

    assert set(tokens) == {"input_ids", "attention_mask"}
    for name in ("input_ids", "attention_mask"):
        assert tokens[name].dtype == np.int32
        assert tokens[name].shape == (len(_PAIRS), _SEQ_LEN)


def test_templated_pairs_keep_the_fixed_parts_and_cut_only_the_body(
    tokenizer_dir: Path,
) -> None:
    """The suffix carries the position the answer is read at; it may never be cut."""
    reference = AutoTokenizer.from_pretrained(tokenizer_dir)
    prefix_ids = reference.encode(_TEMPLATE.prefix, add_special_tokens=False)
    suffix_ids = reference.encode(_TEMPLATE.suffix, add_special_tokens=False)
    overlong = [("q" * 500, "d" * 500)]

    tokens = common.tokenize_templated_pairs(reference, overlong, _SEQ_LEN, _TEMPLATE)

    row = tokens["input_ids"][0].tolist()
    assert row[: len(prefix_ids)] == prefix_ids
    assert row[_SEQ_LEN - len(suffix_ids) :] == suffix_ids
    # Every position is real: the body filled the whole budget.
    assert tokens["attention_mask"][0].tolist() == [1] * _SEQ_LEN


def test_templated_pairs_pad_on_the_right(tokenizer_dir: Path) -> None:
    """Last-token pooling reads ``sum(mask) - 1``, which assumes right padding."""
    reference = AutoTokenizer.from_pretrained(tokenizer_dir)

    tokens = common.tokenize_templated_pairs(reference, [("q", "d")], _SEQ_LEN, _TEMPLATE)

    mask = tokens["attention_mask"][0].tolist()
    assert mask[0] == 1
    assert mask[-1] == 0
    assert mask == sorted(mask, reverse=True)  # no real token after a padded one
    assert int(tokens["input_ids"][0][-1]) == reference.pad_token_id


def test_templated_pairs_refuse_a_bucket_the_template_alone_fills(tokenizer_dir: Path) -> None:
    """With no room left for the pair, the encoding says so instead of dropping it."""
    reference = AutoTokenizer.from_pretrained(tokenizer_dir)
    fixed = len(reference.encode(_TEMPLATE.prefix, add_special_tokens=False)) + len(
        reference.encode(_TEMPLATE.suffix, add_special_tokens=False)
    )

    with pytest.raises(ValueError, match="template"):
        common.tokenize_templated_pairs(reference, [("q", "d")], fixed, _TEMPLATE)


@pytest.mark.parametrize("seq_len", [0, -1])
def test_templated_pairs_reject_a_non_positive_sequence_length(
    tokenizer_dir: Path, seq_len: int
) -> None:
    """A non-positive fixed length is never a valid graph shape."""
    reference = AutoTokenizer.from_pretrained(tokenizer_dir)

    with pytest.raises(ValueError, match="seq_len"):
        common.tokenize_templated_pairs(reference, [("q", "d")], seq_len, _TEMPLATE)


def test_templated_pairs_of_an_empty_batch_keep_the_bucket_shape(tokenizer_dir: Path) -> None:
    """Zero pairs must still produce arrays of the bucket's width."""
    reference = AutoTokenizer.from_pretrained(tokenizer_dir)

    tokens = common.tokenize_templated_pairs(reference, [], _SEQ_LEN, _TEMPLATE)

    assert tokens["input_ids"].shape == (0, _SEQ_LEN)
    assert tokens["attention_mask"].shape == (0, _SEQ_LEN)


def test_templated_pairs_write_a_document_reading_like_a_marker_verbatim(
    tokenizer_dir: Path,
) -> None:
    """A text carrying a marker of its own must be written in, never substituted into."""
    reference = AutoTokenizer.from_pretrained(tokenizer_dir)
    pairs = [("QUERY", "{query}")]

    tokens = common.tokenize_templated_pairs(reference, pairs, _SEQ_LEN, _TEMPLATE)

    expected = _TEMPLATE.prefix + "Q: QUERY\nD: {query}" + _TEMPLATE.suffix
    length = int(tokens["attention_mask"][0].sum())
    assert reference.decode(tokens["input_ids"][0][:length].tolist()) == expected


# --- the FP32 baseline of a generative reranker -------------------------------


class _FixedLogitsModel:
    """Causal-LM stand-in returning a fixed per-position logit table."""

    def __init__(self, logits: torch.Tensor) -> None:
        """Store the ``(1, S, V)`` logits every call returns."""
        self._logits = logits
        self.seen: list[tuple[int, ...]] = []

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[Any, ...]:
        """Record the row it was handed and return the fixed logits."""
        self.seen.append(tuple(int(value) for value in attention_mask[0]))
        return (self._logits,)


def test_score_pytorch_generative_subtracts_the_two_logits_of_the_last_real_position(
    tokenizer_dir: Path,
) -> None:
    """The verdict is read where the model would answer, not at the padded end."""
    reference = AutoTokenizer.from_pretrained(tokenizer_dir)
    pairs = [("q", "d")]
    tokens = common.tokenize_templated_pairs(reference, pairs, _SEQ_LEN, _TEMPLATE)
    length = int(tokens["attention_mask"][0].sum())
    vocab = 16
    logits = torch.zeros(1, _SEQ_LEN, vocab)
    logits[0, length - 1, _TRUE_ID] = 2.5
    logits[0, length - 1, _FALSE_ID] = -1.5
    # A different verdict at the padded position, which must not be read.
    logits[0, _SEQ_LEN - 1, _TRUE_ID] = -9.0
    logits[0, _SEQ_LEN - 1, _FALSE_ID] = 9.0
    model = _FixedLogitsModel(logits)

    scores = common.score_pytorch_generative(
        model, reference, pairs, _SEQ_LEN, _TEMPLATE, _TRUE_ID, _FALSE_ID
    )

    assert scores.shape == (1,)
    assert scores.dtype == np.float32
    assert float(scores[0]) == pytest.approx(4.0)


def test_score_pytorch_generative_scores_every_pair_in_order(tokenizer_dir: Path) -> None:
    """One row per pair, in the order the pairs were given."""
    reference = AutoTokenizer.from_pretrained(tokenizer_dir)
    pairs = [("q1", "d1"), ("q2", "d2"), ("q3", "d3")]
    vocab = 16
    logits = torch.zeros(1, _SEQ_LEN, vocab)
    model = _FixedLogitsModel(logits)

    scores = common.score_pytorch_generative(
        model, reference, pairs, _SEQ_LEN, _TEMPLATE, _TRUE_ID, _FALSE_ID
    )

    assert scores.shape == (3,)
    assert len(model.seen) == 3


# --- the frozen-tokenizer gate on a template ----------------------------------


def test_verify_frozen_tokenizer_passes_on_the_template_path(
    tokenizer_dir: Path, frozen_path: Path
) -> None:
    """The frozen file must reproduce an independently built templated encoding."""
    report = tokenizer_freeze.verify_frozen_tokenizer(
        tokenizer_dir,
        frozen_path,
        ["a text"],
        _PAIRS,
        [_SEQ_LEN],
        pair_template=_TEMPLATE,
    )

    assert report["passed"] is True
    assert report["n_pairs"] == len(_PAIRS)


def test_verify_frozen_tokenizer_reports_what_the_template_reserves(
    tokenizer_dir: Path, frozen_path: Path
) -> None:
    """The pipeline prints how much of each bucket the template takes up."""
    reference = AutoTokenizer.from_pretrained(tokenizer_dir)

    report = tokenizer_freeze.verify_frozen_tokenizer(
        tokenizer_dir, frozen_path, [], _PAIRS, [_SEQ_LEN], pair_template=_TEMPLATE
    )

    assert report["pair_template"] == {
        "prefix_tokens": len(reference.encode(_TEMPLATE.prefix, add_special_tokens=False)),
        "suffix_tokens": len(reference.encode(_TEMPLATE.suffix, add_special_tokens=False)),
    }


def test_verify_frozen_tokenizer_reports_no_template_when_none_is_given(
    tokenizer_dir: Path, frozen_path: Path
) -> None:
    """A model needing no template must produce the report it always did."""
    report = tokenizer_freeze.verify_frozen_tokenizer(
        tokenizer_dir, frozen_path, ["a text"], _PAIRS, [_SEQ_LEN]
    )

    assert report["passed"] is True
    assert report["pair_template"] is None


def test_verify_frozen_tokenizer_detects_a_tampered_frozen_file_on_the_template_path(
    tmp_path: Path, tokenizer_dir: Path, frozen_path: Path
) -> None:
    """The gate must catch a frozen file that no longer pads like the reference."""
    raw = json.loads(frozen_path.read_text(encoding="utf-8"))
    raw["padding"]["pad_id"] = raw["padding"]["pad_id"] + 1
    tampered = tmp_path / "tokenizer.json"
    tampered.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(tokenizer_freeze.TokenizerFreezeError) as excinfo:
        tokenizer_freeze.verify_frozen_tokenizer(
            tokenizer_dir, tampered, [], [("q", "d")], [_SEQ_LEN], pair_template=_TEMPLATE
        )

    assert excinfo.value.report["passed"] is False
    assert "input_ids" in str(excinfo.value)


def test_verify_frozen_tokenizer_detects_a_template_the_frozen_side_was_not_given(
    tokenizer_dir: Path, frozen_path: Path
) -> None:
    """Guard for the gate itself: the plain pair encoding must not pass as a templated one.

    Runs the two sides against each other with the template applied to the
    reference only, which is exactly the mistake the gate exists to catch.
    """
    reference = AutoTokenizer.from_pretrained(tokenizer_dir)
    frozen = runtime.load_frozen_tokenizer(frozen_path)

    plain = runtime.tokenize_pairs(frozen, [("q", "d")], _SEQ_LEN)
    templated = common.tokenize_templated_pairs(reference, [("q", "d")], _SEQ_LEN, _TEMPLATE)

    assert not np.array_equal(plain["input_ids"], templated["input_ids"])


def test_verify_frozen_tokenizer_refuses_a_bucket_the_template_alone_fills(
    tokenizer_dir: Path, frozen_path: Path
) -> None:
    """A bucket too short for the template is reported, not silently truncated."""
    reference = AutoTokenizer.from_pretrained(tokenizer_dir)
    fixed = len(reference.encode(_TEMPLATE.prefix, add_special_tokens=False)) + len(
        reference.encode(_TEMPLATE.suffix, add_special_tokens=False)
    )

    with pytest.raises(ValueError):
        tokenizer_freeze.verify_frozen_tokenizer(
            tokenizer_dir, frozen_path, [], [("q", "d")], [fixed], pair_template=_TEMPLATE
        )


def test_verify_frozen_tokenizer_compares_the_templated_token_counts(
    tokenizer_dir: Path, frozen_path: Path
) -> None:
    """Bucket selection runs on the untruncated templated length, so it is verified too."""
    frozen = runtime.load_frozen_tokenizer(frozen_path)
    prepared = runtime.prepare_pair_template(frozen, _TEMPLATE)
    reference = AutoTokenizer.from_pretrained(tokenizer_dir)
    query, document = "a query", "a document"

    counted = runtime.count_pair_tokens(frozen, query, document, prepared)

    body = _TEMPLATE.body_format.replace("{query}", query).replace("{document}", document)
    expected = (
        len(reference.encode(_TEMPLATE.prefix, add_special_tokens=False))
        + len(reference.encode(body, add_special_tokens=False))
        + len(reference.encode(_TEMPLATE.suffix, add_special_tokens=False))
    )
    assert counted == expected
    # And the gate agrees with that definition on the same inputs.
    report = tokenizer_freeze.verify_frozen_tokenizer(
        tokenizer_dir,
        frozen_path,
        [],
        [(query, document)],
        [_SEQ_LEN],
        pair_template=_TEMPLATE,
    )
    assert report["passed"] is True
